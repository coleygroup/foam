"""server_utils.py

Pyro4-based remote server utilities for running ICEBERG spectrum prediction
as a distributed service. Enables GPU-accelerated batch inference with
automatic load balancing and CUDA memory error recovery.

Example usage:
    # Start server
    python server_utils.py --host myserver.mit.edu --port 5000 \\
        --inten_model_ckpt path/to/inten.ckpt \\
        --gen_model_ckpt path/to/gen.ckpt

    # Client connection
    server = Pyro4.Proxy("PYRO:iceberg_server@myserver.mit.edu:5000")
    specs, workers = server.predict_mol(smiles_list, gpu_workers=4, iceberg_kwargs={...})
"""
import argparse
import logging
import os
import socket
import traceback
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import Pyro4
import torch
from ms_pred import common
from ms_pred.dag_pred import gen_model, inten_model, joint_model
from multiprocess import active_children, current_process

Pyro4.config.SERIALIZERS_ACCEPTED.add("pickle")
Pyro4.config.SERIALIZER = "pickle"


def worker_init(
    inten_ckpt: str, gen_ckpt: str, gpu_id: Union[str, int]
) -> joint_model.JointModel:
    """Initialize an ICEBERG model worker for parallel inference.

    Loads intensity and generation model checkpoints, combines them into a
    JointModel, and moves to the specified device. Sets single-threaded
    execution for better parallelization.

    Args:
        inten_ckpt: Path to the intensity model checkpoint file.
        gen_ckpt: Path to the generation model checkpoint file.
        gpu_id: Device identifier, either "cuda:N" for GPU or "cpu".

    Returns:
        Initialized JointModel ready for inference.
    """
    import torch
    from ms_pred.dag_pred import inten_model, gen_model, joint_model
    # load once
    inten = inten_model.IntenGNN.load_from_checkpoint(inten_ckpt, map_location='cpu')
    gen   = gen_model.FragGNN.load_from_checkpoint(gen_ckpt,   map_location='cpu')
    model = joint_model.JointModel(gen_model_obj=gen, inten_model_obj=inten)
    model.eval(); model.freeze()
    if isinstance(gpu_id, str) and gpu_id.startswith("cuda"):
        # e.g. "cuda:0"
        device = torch.device(gpu_id)
        torch.cuda.set_device(device)   # pin the correct CUDA context
    else:
        # e.g. "cpu"
        device = torch.device("cpu")

    model.to(device)
    torch.set_num_threads(1)
    return model

@Pyro4.expose
class ICEBERGServer:
    """Pyro4-exposed server for distributed ICEBERG spectrum prediction.

    Manages ICEBERG model lifecycle and provides remote prediction interface
    with automatic GPU load balancing and CUDA memory error recovery.

    Attributes:
        inten_model_ckpt: Path to intensity model checkpoint.
        gen_model_ckpt: Path to generation model checkpoint.
        denoise_specs: Whether to apply denoising to predicted spectra.
        iceberg_model: The loaded JointModel instance.
        avail_gpu_num: Number of available GPUs detected.
        new_gpu_workers: Override for GPU worker count (set dynamically on OOM).
    """

    def __init__(
        self,
        inten_model_ckpt: str = None,
        gen_model_ckpt: str = None,
        denoise_specs: bool = False,
    ) -> None:
        """Initialize the ICEBERG server with model checkpoints.

        Args:
            inten_model_ckpt: Path to the intensity prediction model checkpoint.
            gen_model_ckpt: Path to the fragment generation model checkpoint.
            denoise_specs: If True, apply denoising to output spectra.
        """
        self.inten_model_ckpt = inten_model_ckpt
        self.gen_model_ckpt = gen_model_ckpt
        self.denoise_specs = denoise_specs
        inten_model_obj = inten_model.IntenGNN.load_from_checkpoint(inten_model_ckpt, strict=False, map_location='cpu')
        gen_model_obj = gen_model.FragGNN.load_from_checkpoint(gen_model_ckpt, strict=False, map_location='cpu')

        self.iceberg_model = joint_model.JointModel(
            gen_model_obj=gen_model_obj, inten_model_obj=inten_model_obj
        )
        print("ICEBERG model set up")
        self.iceberg_model.eval()
        self.iceberg_model.freeze()
        cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', None)
        if cuda_visible_devices is not None:
            self.avail_gpu_num = len(cuda_visible_devices.split(','))
        else:
            self.avail_gpu_num = torch.cuda.device_count()
        self.new_gpu_workers = None

    @Pyro4.expose
    def get_model(self) -> joint_model.JointModel:
        """Return the loaded ICEBERG JointModel instance."""
        return self.iceberg_model

    @Pyro4.expose
    def get_denoise_specs(self) -> bool:
        """Return whether spectrum denoising is enabled."""
        return self.denoise_specs

    def predict_mol(
        self,
        smiles: List[str],
        gpu_workers: int = 0,
        iceberg_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[Optional[Dict[str, np.ndarray]]], int]:
        """Predict mass spectra for a batch of SMILES strings.

        Runs ICEBERG inference in parallel across available GPU workers,
        with automatic retry on CUDA OOM errors.

        Args:
            smiles: List of SMILES strings to predict spectra for.
            gpu_workers: Number of parallel GPU workers to use.
            iceberg_kwargs: Dictionary containing prediction parameters:
                - collision_eng: List of collision energies
                - adduct: Adduct type(s)
                - precursor_mz: Precursor m/z value
                - instrument: Optional instrument type
                - batch_size: Batch size for inference (default 32)
                - device: Device to run on ("cuda:N" or "cpu")
                - threshold: Intensity threshold for peaks
                - max_nodes: Maximum fragment nodes
                - num_bins: Number of bins for binned output
                - mass_upper_limit: Upper mass limit for binning
                - final_binned: Whether to return binned spectra
                - adduct_shift: Adduct mass shift

        Returns:
            Tuple of (predicted_spectra, updated_gpu_workers) where:
                - predicted_spectra: List of dicts mapping collision energy to spectrum
                - updated_gpu_workers: Possibly reduced worker count after OOM recovery
        """
        try: 
            # Preprocess inputs
            if self.new_gpu_workers is not None:
                gpu_workers = self.new_gpu_workers

            collision_engs = [float(a) for a in iceberg_kwargs["collision_eng"]]
            adducts = iceberg_kwargs["adduct"]
            instrument = iceberg_kwargs.get("instrument", None)
            precursor_mz = iceberg_kwargs["precursor_mz"]

            # Ensure lists
            if isinstance(adducts, str):
                adducts = [adducts]
            if isinstance(instrument, str):
                instrument = [instrument]

            full = {
                "smiles": np.repeat(np.array(smiles), len(collision_engs)).tolist(),
                "collision_eng": (collision_engs * len(smiles)),
                "adducts": (adducts * len(smiles) * len(collision_engs)),
                "precursor_mz": [precursor_mz] * len(smiles) * len(collision_engs),
            }
            if instrument is not None:
                full["instrument"] = instrument * len(smiles) * len(collision_engs)

            batch_size = iceberg_kwargs.get("batch_size", 32)
            batches = [
                {k: v[i:i + batch_size] for k, v in full.items()}
                for i in range(0, len(full["smiles"]), batch_size)
            ]
            
            def batch_worker(batch, kwargs=iceberg_kwargs):
                torch.set_num_threads(1)
                worker_id = current_process()._identity[0] if current_process()._identity else 0
                kwargs["final_binned"] = kwargs.get("final_binned", True)

                gpu_id = kwargs.get("device", "cpu")
                # option1: copy
                # worker_model = self.iceberg_model.copy()
                #worker_model.to(torch.device(gpu_id))
                #
                # option2: try referencing same model
                # self.iceberg_model.to(torch.device(gpu_id))

                #  print("attributes available", self.__dict__, flush=True)
                # ^ suggests no way to get model with self.iceberg_model, so use get

                model = self.get_model()
                model.to(torch.device(gpu_id))

                #self.iceberg_model.to(torch.device(gpu_id))
                
                # model = worker_init(self.inten_model_ckpt, self.gen_model_ckpt, gpu_id)

                try:
                    # self.iceberg_model... 
                    specs = model.predict_mol(
                                            smi=batch["smiles"],
                                            collision_eng=batch["collision_eng"],
                                            adduct=batch["adducts"],
                                            precursor_mz=batch["precursor_mz"],
                                            instrument=batch.get("instrument", None),
                                            threshold=kwargs["threshold"],
                                            device=torch.device(gpu_id),
                                            max_nodes=kwargs["max_nodes"],
                                            binned_out=False,
                                            adduct_shift=kwargs["adduct_shift"]
                                        )
                    # if tensor: convert to numpy
                    specs["spec"] = [spec.cpu().numpy() if torch.is_tensor(spec) else spec for spec in specs["spec"]]
                    if kwargs["final_binned"]:
                        # swap
                        if self.get_denoise_specs():
                            specs["spec"] = [common.bin_spectra([common.denoise_spectrum(s)],
                                                            kwargs["num_bins"], 
                                                            kwargs["mass_upper_limit"],
                                                            )[0] for s in specs["spec"]]
                        else: 
                            specs["spec"] = [common.bin_spectra([s], 
                                                                kwargs["num_bins"], 
                                                                kwargs["mass_upper_limit"],
                                                                )[0] for s in specs["spec"]]
                        torch.cuda.empty_cache()
                        specs_only = dict()
                        specs_only["spec"] = specs["spec"]
                        return specs_only # lower IO
                    
                    else: 
                        def postprocess(pred_spec):
                            if self.get_denoise_specs():
                                pred_spec = common.denoise_spectrum(pred_spec)

                            # then "bin"/collect values across the same exact m/z [TODO: can remove/tweak based on Runzhong's code, 4/10]
                            unique_mz, inv = np.unique(pred_spec[:, 0], return_inverse=True)
                            summed_inten = np.bincount(inv, weights=pred_spec[:, 1])
                            slim_pred_spec = np.column_stack((unique_mz, summed_inten))
                            # finally, normalize to biggest peak.
                            if len(slim_pred_spec):
                                slim_pred_spec[:, 1] = slim_pred_spec[:, 1] / slim_pred_spec[:, 1].max()
                            

                            # Finally: turn back from np.array to list of tuples
                            # slim_pred_spec = [tuple(a) for a in slim_pred_spec]
                            return slim_pred_spec
                        specs["spec"] = [postprocess(spec) for spec in specs["spec"]]
                        torch.cuda.empty_cache()
                        return specs
                    

                except RuntimeError as err:
                    print(err, flush=True)
                    model.to("cpu")
                    torch.cuda.empty_cache()
                    return "cuda" if "out of memory" in str(err).lower() else "error"
                except Exception as e:
                    print(e, flush=True)
                    model.to("cpu")
                    return "error"
                finally:
                    model.to("cpu")
                    torch.cuda.empty_cache()


            # def wrapped_batch_worker(batch):
            #     worker_id = current_process()._identity[0] if current_process()._identity else 0
            #     return batch_worker(batch, iceberg_kwargs, worker_id)

            
            batches_specs = common.chunked_parallel(batches, 
                                                    batch_worker, 
                                                    chunks=4000, max_cpu=gpu_workers,
                                                    # gpu=False,
                                                    # initializer=worker_init,
                                                    # initargs=(self.inten_model_ckpt, 
                                                    #           self.gen_model_ckpt,
                                                    #           iceberg_kwargs["device"]),
                                                    task_name="Remote ICEBERG scoring")
            
            # TODO: wrap error handling in "retries" block with fewer workers. 
            issue_batches = [batch for batch in batches_specs if type(batch) == str]
            if len(issue_batches) > 0:
                print(issue_batches)
                cuda_batches = [batch for i, batch in enumerate(batches) if batches_specs[i] == "cuda"]
            
                # redispatch cuda_batches once and hope that's sufficient to alleviate the memory isue:
                if len(cuda_batches) > 0:
                    print(len(cuda_batches), "batches failed due to CUDA memory issues")
                    # Lower the number of GPU workers
                    gpu_workers = gpu_workers - 2
                    print("Reducing GPU workers from", gpu_workers + 2, "to", gpu_workers)
                
                    cuda_specs = common.chunked_parallel(cuda_batches, batch_worker, chunks=2000, max_cpu=gpu_workers,
                                                        task_name="Rerunning CUDA batches [Remote]")
                    # replace the cuda entries with the new ones
                    for i, batch in enumerate(batches_specs):
                        if batch == "cuda":
                            print("updating batch")
                            batches_specs[i] = cuda_specs.pop(0)

                # rerun failed batches, from either first or second time
                unrun_batches = [input for input, output in zip(batches, batches_specs) if type(output) != dict]
                
                if len(unrun_batches) > 0:
                    raise RuntimeError("Unsuccessful batches: " + str(len(unrun_batches)) + "batches unrun")
                    # print(len(unrun_batches), "batches failed due to CUDA a second time, or other invalid error")
                    # rerun_batches = []

                    # temp = kwargs["device"]
                    # kwargs["device"] = "cpu"
                    # self.iceberg_model.to(kwargs["device"]) # move to CPU
                    # for batch in unrun_batches:
                    #     inputs = list(zip(*batch.values()))
                    #     output = utils.chunked_parallel(inputs, pred_single_tuple, 
                    #                                     chunks=500, max_cpu=self.num_workers, task_name="Rerunning failed batches")

                    #     rerun_batches.append({"spec": output})

                    # self.iceberg_kwargs["device"] = temp
                    # # self.iceberg_model.to(self.iceberg_kwargs["device"]) # move back to whatever device it was on before
                    # # insert back in with same format
                    # batches_specs = [orig if type(orig) == dict else rerun_batches.pop(0) for orig in batches_specs]


            all_specs_raw = [spec for batch in batches_specs for spec in batch["spec"]]
            all_pred_specs = []

            try: 
                for i in range(len(smiles)):
                    start = i * len(iceberg_kwargs["collision_eng"])
                    end = (i + 1) * len(iceberg_kwargs["collision_eng"])
                    if any([s is None for s in all_specs_raw[start:end]]):
                        print(f"{smiles[i]} errored")
                        all_pred_specs.append(None)
                        continue
                    spec_dict = {str(ce_eng): spec for ce_eng, spec in zip(iceberg_kwargs["collision_eng"], 
                                                                            all_specs_raw[start:end])}
                    all_pred_specs.append(spec_dict)

            except Exception as e:
                print(f"Error in processing spectra: {e}", flush=True)

                raise

            return all_pred_specs, gpu_workers
        except Exception:
            print("exception", flush=True)
            traceback.print_exc()      # <-- prints full traceback with line numbers
            raise                     # re‐raise so the client still sees an error

    def eval(self) -> None:
        """Set the model to evaluation mode."""
        self.iceberg_model.eval()

    def freeze(self) -> None:
        """Freeze model parameters to prevent gradient computation."""
        self.iceberg_model.freeze()

    def to(self, device: Union[str, torch.device]) -> None:
        """Move the model to the specified device.

        Args:
            device: Target device ("cpu", "cuda:N", or torch.device).
        """
        self.iceberg_model.to(device)


def main() -> None:
    """Entry point for starting the ICEBERG Pyro4 server.

    Parses command-line arguments for host, port, and model checkpoints,
    then starts a Pyro4 daemon to serve prediction requests.
    """
    parser = argparse.ArgumentParser(description="Start the Pyro4 server.")
    parser.add_argument("--host", type=str, default=socket.gethostname() + ".mit.edu")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--inten_model_ckpt", type=str, required=True)
    parser.add_argument("--gen_model_ckpt", type=str, required=True)

    args = parser.parse_args()
    print("parsed", flush=True)

    daemon = Pyro4.Daemon(host=args.host, 
                        port=args.port)
    print("daemon set up", flush=True)
    iceberg_obj = ICEBERGServer(
        inten_model_ckpt=args.inten_model_ckpt,
        gen_model_ckpt=args.gen_model_ckpt
    )
    print("ICEBERG object set up", flush=True)
    uri = daemon.register(iceberg_obj, objectId="iceberg_server")
    print("Ready. Object uri =", uri, flush=True)
    
    daemon.requestLoop()

if __name__ == "__main__":
    main()