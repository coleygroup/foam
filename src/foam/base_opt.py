""" base.py

Define a series of optimizers that can be used to query the next batch of
information

"""
import logging
from abc import ABC, abstractmethod
from enum import Enum
from itertools import chain
from pathlib import Path
from typing import Union, Tuple, List

import foam.evaluators as evaluators
import foam.oracles as oracles
import numpy as np
import wandb
import yaml
from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula


class OptSignals(Enum):
    INVALID = 0
    CONT = 1
    STOP = 2


class OptimizerBase(ABC):
    def __init__(
        self,
        save_dir: str = "",
        oracle: oracles.MolOracle = None,
        max_calls: int = 100,
        keep_population: int = 10,
        replace_population: bool = False,
        num_workers: int = 10,
        patience: int = 10,
        log_freq: int = 2,
        wandb_mode: str = "online",
        eval_names: List[str] = [
            "DiversityEval",
            "TopScore",
            "BestMol",
            "FormulaDiffEval",
        ],
        **kwargs,
    ):
        """ """
        super().__init__()
        self.oracle = oracle
        self.max_calls = max_calls
        self.num_workers = num_workers
        self.calls_made = 0
        self.c_info = oracle.conditional_info()
        self.replace_population = replace_population
        self.keep_population = keep_population if ~replace_population else None
        self.log_freq = log_freq
        self.save_dir = Path(save_dir)

        self.eval_names = eval_names
        self.evaluators = self._init_evaluators(**kwargs)
        self.wandb_mode = wandb_mode
        if self.wandb_mode == "offline":
            from wandb_osh.hooks import TriggerWandbSyncHook
            trigger_sync = TriggerWandbSyncHook()
            self.trigger_sync = trigger_sync

        self.patience = patience

    def _init_evaluators(self, **kwargs) -> List[evaluators.BaseEvaluator]:
        """_init_evaluators."""
        eval_list = []
        for eval_name in self.eval_names:
            eval_cls = evaluators.get_evaluator_cls(eval_name)
            if eval_cls is None:
                logging.info(f"No eval: {eval_name}")
            eval_obj = eval_cls(oracle=self.oracle, **kwargs)
            eval_list.append(eval_obj)

        self.evaluators = eval_list
        return eval_list

    def sanitize(self, mol_list: List[Chem.Mol]) -> List[Chem.Mol]:
        """sanitize.py"""
        new_mol_list = []
        smiles_set = set()
        for mol in mol_list:
            if mol is not None:
                try:
                    smiles = Chem.MolToSmiles(mol)
                    if smiles is not None and smiles not in smiles_set:
                        smiles_set.add(smiles)
                        new_mol_list.append(mol)
                except ValueError:
                    logging.warning(f"Bad smiles")
        return new_mol_list

    @staticmethod
    def opt_name():
        return "Base"

    @abstractmethod
    def _optimize(self, **kwargs) -> None:
        """
        Optimize a oracle function by suggesting new molecules

        This function should call self.score_mol repeatedly until receiving a stop
        signal at which point it should exit
        """
        pass

    # TODO: simplify score_mol and move superwriting into graph_ga file. 
    def score_mol(
        self, mols: List[Union[Chem.Mol, str]], mol_type: str = "Mol", **kwargs
    ) -> Tuple[List[Chem.Mol], List[float], int]:
        """score_mol.

        Score a single molecule with an optimizer. Handles all logging so that
        the optimizer does not have to worry about it.

        Args:
            mols: List of Smiles or rdkit Objs
            mol_type: Type of molecule
            kwargs

        Return: Tuple[mol_objs, float, int]: mols, scores, signal to continue

        """

        calls_remaining = self.max_calls - self.calls_made
        # If above the limt, do not score
        if calls_remaining <= 0:
            return None, None, OptSignals.STOP

        logging.info("Converting mol list")
        if mol_type == "Mol":
            smis = [Chem.MolToSmiles(mol) for mol in mols]
        elif mol_type == "smiles":
            smis = mols
            mols = [Chem.MolFromSmiles(mol) for mol in mols]
        else:
            raise NotImplementedError()

        if self.replace_population: 
            self.mol_buffer = {}
            new_scores = self.oracle(np.array(mols))
            self.calls_made += len(smis)
            inds = np.arange(self.calls_made - len(smis), self.calls_made)
            for score, smi, mol in zip(new_scores, smis, mols):
                smi = str(smi)
                chem_formulae = CalcMolFormula(mol) if mol is not None else ""
                tani = 0
                if hasattr(self.oracle, 'mol_smiles') and self.oracle.mol_smiles is not None:
                    target_fp = self.oracle._get_fp()[np.newaxis, :]
                    fp = self.oracle.get_morgan_fp(mol)[np.newaxis, :]
                    tani = self.oracle._tanimoto_sim(target_fp, fp).item()

                new_entry = {
                    "score": float(score),
                    "formula": chem_formulae,
                    "inchikey-match": 1 if Chem.MolToInchiKey(mol) == self.oracle.mol_inchikey else 0,
                    # "mces": self.oracle.mces(smi), # TODO: can probably get hung up if the molecule is big!! consider computing in background?
                    "tanimoto": tani,
                    }
                self.mol_buffer[smi] = new_entry
            
            # Log output with some frequency
                
            if np.any(inds % self.log_freq == 0):

                output_stats = self._get_stats_log()
                output_stats["Meta"] = {"Calls Made": self.calls_made}

                out_str = yaml.dump(output_stats)
                logging.info(f"Batch statistics after {self.calls_made} calls:\n {out_str}")

                # Submit to wandb
                if self.wandb_mode != "disable":
                    wandb.log(output_stats)
                    if self.wandb_mode == "offline":
                        self.trigger_sync()
            
            return None, new_scores, OptSignals.CONT
            
        # Filter down to only mols not yet scored
        all_scores = np.array(
            [self.mol_buffer.get(i, {}).get("score", None) for i in smis]
        )
        already_sampled = np.array([i is not None for i in all_scores])
        mols = np.array(mols)

        not_sampled = ~already_sampled

        already_sampled_mols = mols[already_sampled]
        already_sampled_scores = all_scores[already_sampled]

        # Subset molecules not yet scored down to only the amount remaining
        not_sampled = (np.cumsum(not_sampled) <= calls_remaining) * not_sampled
        not_sampled_mols = mols[not_sampled]
        not_sampled_smis = np.array(smis)[not_sampled]

        # Compute scores
        new_scores = self.oracle(not_sampled_mols)

        # Increment number of calls
        self.calls_made += len(not_sampled_smis)
        inds = np.arange(self.calls_made - len(not_sampled_smis), self.calls_made)
        
        # Update buffer and score with all outputs
        for call_num, score, smi, mol in zip(
            inds, new_scores, not_sampled_smis, not_sampled_mols
        ):
            smi = str(smi)
            chem_formulae = CalcMolFormula(mol) if mol is not None else ""
            # Add mol into our mols list and cutoff if below score thresh
            tani = 0
            if hasattr(self.oracle, 'mol_smiles') and self.oracle.mol_smiles is not None:
                target_fp = self.oracle._get_fp()[np.newaxis, :]
                fp = self.oracle.get_morgan_fp(mol)[np.newaxis, :]
                tani = self.oracle._tanimoto_sim(target_fp, fp).item()
            new_entry = {
                "score": float(score),
                "sample_num": int(call_num),
                "formula": chem_formulae,
                "inchikey-match": 1 if hasattr(self.oracle, 'mol_inchikey') and Chem.MolToInchiKey(mol) == self.oracle.mol_inchikey else 0,
                # "mces": self.oracle.mces(smi), # TODO: can probably get hung up if the molecule is big!! consider computing in background?
                "tanimoto": tani,
            
            }

            if self.min_score is None:
                self.mol_buffer[smi] = new_entry
                self.min_score = score
            elif score > self.min_score:
                # Pop one item from our dictionary
                self.mol_buffer[smi] = new_entry
                sorted_dict = sorted(self.mol_buffer.items(), key=lambda x: x[1]["score"], reverse=True)

                if self.keep_population is not None:
                    sorted_dict = sorted_dict[: self.keep_population]

                self.mol_buffer = dict(sorted_dict)
            elif (
                self.keep_population is None
                or len(self.mol_buffer) < self.keep_population
            ):
                self.mol_buffer[smi] = new_entry
            else:
                pass

        # Log output with some frequency
        best_score = np.max([i["score"] for i in self.mol_buffer.values()])
        if np.any(inds % self.log_freq == 0):

            output_stats = self._get_stats_log()
            # TODO: consider making separate evaluator for SA variation/range
            
            output_stats["Meta"] = {"Calls Made": self.calls_made}
            out_str = yaml.dump(output_stats)
            logging.info(f"Batch statistics after {self.calls_made} calls:\n {out_str}")

            # Submit to wandb
            if self.wandb_mode != "disable":
                # log differently
                wandb.log(output_stats)
                if self.wandb_mode == "offline":
                    self.trigger_sync()

        # Add new scores with old scores
        # ret_mols = already_sampled_mols.tolist() + not_sampled_mols.tolist()
        # ret_scores = [float(i) for i in chain(already_sampled_scores, new_scores)]
        # TODO: Should be multi-objective handle-able
        ret_mols, ret_scores = np.array([None] * mols.shape[0]), np.array([None] * mols.shape[0])
        ret_mols[already_sampled], ret_mols[not_sampled] = already_sampled_mols, not_sampled_mols
        ret_scores[already_sampled], ret_scores[not_sampled] = already_sampled_scores, new_scores
        ret_mols = [_ for _ in ret_mols.tolist() if _ is not None]
        ret_scores = [j for j in ret_scores.tolist() if j is not None]

        return ret_mols, ret_scores, OptSignals.CONT

    def optimize(self, **kwargs):
        """optimize.

        Wrapper for the _optimize function

        """
        self.calls_made = 0
        self.min_score = None

        # Reset a mol buffer
        self.mol_buffer = {}

        # Optimization procedure
        self._optimize(**kwargs)

        # 2. Calculate all metrics
        output_stats = self._get_stats_log()
        out_str = yaml.dump(output_stats)
        logging.info(f"Batch statistics:\n {out_str}")

        # Dump results
        output_dump = {"mol_buffer": self.mol_buffer, "output_stats": output_stats}

        # Dump all molecules
        dumped_results = yaml.dump(output_dump)
        with open(self.save_dir / "output_mols.yaml", "w") as fp:
            fp.write(dumped_results)

        wandb.finish()
        return output_dump

    def _get_stats_log(self):
        """_get_stats_log."""
        output_stats = {}
        if len(self.mol_buffer) == 0:
            return output_stats
        for eval_ in self.evaluators:
            new_stats = eval_.eval_batch(self.mol_buffer)
            new_stats = {eval_.__class__.eval_name(): new_stats}
            output_stats.update(new_stats)
        return output_stats

    def get_seed_smiles(self, *args, **kwargs):
        """get_seed_smiles.

        Ask oracle to provide seed smile options

        """
        return self.oracle.get_seed_smiles(*args, **kwargs)
