""" base_parsing.py

Optimizer parsing


"""
import argparse
from datetime import datetime
from pathlib import Path

import foam.evaluators as evaluators
import foam.oracles as oracles


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_baseparser() -> argparse.ArgumentParser:
    """get base parser"""
    default_dir = lambda: datetime.now().strftime("%Y_%m_%d") + "_default_dir"

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--config", type=str, help="Path to the config file", default=None
    )
    base_args = parser.add_argument_group("Base Args")
    base_args.add_argument("--debug", default=False, action="store_true")
    base_args.add_argument("--seed", default=None, action="store", type=int)
    base_args.add_argument(
        "--wandb-mode",
        default="online",
        action="store",
        choices=["offline", "online", "disable"],
    )
    base_args.add_argument(
        "--wandb-project",
        action="store",
        help="Name of wandb project if --wandb is enabled",
    )
    base_args.add_argument(
        "--tags",
        default="",
        nargs="+",
        action="store",
        help="Comma separated list of tags to add to wandb",
    )
    base_args.add_argument(
        "--num-workers", default=1, action="store", type=int, help="Number of processes"
    )
    base_args.add_argument(
        "--save-dir",
        default=Path("results") / default_dir(),
        action="store",
        help="Directory to save all outputs",
    )

    base_args.add_argument(
        "--use-multi-node", 
        action="store",
        type=str2bool,
        nargs='?',
        const=True,
        default=False,
        help="Use multi-node running",
    )

    base_args.add_argument(
        "--server-uri",
        action="store",
        help="Server URI for multi-node running",
    )
    return parser


def add_oracle_args(parser: argparse.ArgumentParser):
    """Add oracle args

    Args:
        parser (argparse.ArgumentParser): Parser to add sub args to
    """
    all_oracles = list(oracles.oracle_registry.keys())
    parser.add_argument(
        "--oracle-name",
        choices=all_oracles,
        help="Name of oracle",
        action="store",
        default="Tani_pindolol",
    ) # don't change default behavior here, but just overwrite if spec-lib-dir and spec-lib-label are both provided 

    parser.add_argument("--seed-lib-dir",
                        action="store",
                        help="Folder containing chem formula seed files",
                        type=str,
                        default="",
                        )

    # Path to spectrum library
    parser.add_argument("--spec-lib-dir",
                        action="store",
                        help="Folder containing spectrum library files",
                        type=str,
                        default="",
                        )
    parser.add_argument("--spec-lib-label",
                        action="store",
                        help="Path to spectrum library labels",
                        type=str,
                        default="",
                        )
    
    parser.add_argument("--spec-id",
                        action="store",
                        help="Spectrum identifier to use for oracle, if not pre-designated oracle",
                        type=str,
                        default=None,
                        )

    parser.add_argument("--merge-by-precursor-mz-inchi", 
                        action="store_true", 
                        help="If entries in spec library are deaggregated, collect based on matched inchi + precursor m/z"
                        )


    parser.add_argument("--max-seed-sim",
                        action="store",
                        help="Max sim of seed to target",
                        type=float,
                        default=0.5
                        )

    # ICEBERG oracle arguments
    parser.add_argument("--threshold",
                        action="store",
                        help="Discard peaks with probability less than threshold",
                        type=float,
                        default=0.0)
    
    parser.add_argument("--device",
                        action="store",
                        help="Computing device",
                        choices=["cpu", "gpu", "cuda:0", "cuda:1", "cuda:2", "cuda:3", "cuda:visible"],
                        type=str,
                        default="cpu")
    
    parser.add_argument("--max-nodes",
                        action="store",
                        help="Maximum number of nodes predicted",
                        type=int,
                        default=100)
    
    parser.add_argument("--nce", 
                        action="store_true",
                        help="Denote whether input spectra uses NCE or HCD notation",
                        default=False)
    
    parser.add_argument("--limited-evs",
                        default=None,
                        nargs="+",
                        action="store",
                        type=int,
                        help="If specified, limit comparison to spectra with these collision energies",
                    )
    
    parser.add_argument("--use-clustered-evs",
                        default=False,
                        action="store_true",
                        help="Use clustered collision energies")
    
    parser.add_argument("--ignore-precursor-peak",
                        default=False,
                        action="store_true")

    parser.add_argument("--denoise-specs",
                        default=False,
                        action="store_true")
    
    parser.add_argument("--oracle-type",
                        default="Cos_",
                        action="store")
    
    parser.add_argument("--gpu-workers",
                        action="store",
                        help="Number of ICEBERG models running in parallel on GPU",
                        type=int,
                        default=4)
    
    parser.add_argument("--batch-size",
                        action="store",
                        help="number of entries per batch to process on GPU",
                        type=int,
                        default=16)
    
    parser.add_argument("--inten-model-ckpt", 
                        default=None,
                        action="store",
                        help="Path to intensity model checkpoint")
    
    parser.add_argument("--gen-model-ckpt",
                        default=None,
                        action="store",
                        help="Path to generator model checkpoint")
    
    parser.add_argument("--use-iceberg-spectra",
                        default=False,
                        action="store_true",
                        help="As a self-check, use ICEBERG spectra instead of experimental for oracle")
    
    parser.add_argument("--full-seed-file",
                        default="data/pubchem/pubchem_formula_map.p",
                        action="store",
                        help="Path to full PubChem mapping")
    
    parser.add_argument("--multiobj",
                        default=False,
                        action="store_true",
                        help="Use multi-objective optimization")


    return parser


def add_oracle_hyper_args(parser: argparse.ArgumentParser):
    """Add oracle args for hyperparam opt

    Args:
        parser (argparse.ArgumentParser): Parser to add sub args to
    """
    all_oracles = list(oracles.oracle_registry.keys())
    parser.add_argument(
        "--oracle-names",
        choices=all_oracles,
        help="Names of oracle",
        nargs="+",
        action="store",
        default=["PindololTani"],
    )

    parser.add_argument("--seed-lib-dir",
                        action="store",
                        help="Folder containing chem formula seed files or txt file with seeds",
                        type=str,
                        default="data/isomer_candidates/",
                        )
    parser.add_argument("--max-seed-sim",
                        action="store",
                        help="Max sim of seed to target",
                        type=float,
                        default=0.5
                        )

    return parser


def add_eval_args(parser: argparse.ArgumentParser):
    """Add evaluation args

    Args:
        parser (argparse.ArgumentParser): Parser to add sub args to
    """
    all_evals = list(evaluators.evaluator_registry.keys())
    parser.add_argument(
        "--eval-names",
        choices=all_evals,
        help="Names of evaluators",
        action="store",
        nargs="+",
        default=[
            "DiversityEval",
            "TopScore",
            "TopIsoScore",
            "BestMol",
            "BestIsoMol",
            "FormulaDiffEval",
        ],
    )
    parser.add_argument("--top-k",
                        action="store",
                        nargs="+",
                        help="Top k to do evaluations",
                        type=int,
                        default=[1, 10, 100],
                        )
    return parser


def add_base_opt_args(parser):
    parser.add_argument(
        "--max-calls",
        help="Maximum number of oracle calls allowed",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--keep-population",
        help=("Number of molecules to track during optimization"),
        type=int,
        default=None,
    )
    parser.add_argument(
        "--log-freq", default=10, type=int, help="Frequency of calls to log results"
    )
    parser.add_argument(
        "--patience",
        default=8,
        type=int,
        help="Steps to wait before early stopping",
    )
    parser.add_argument(
        "--criteria",
        default="cosine",
        type=str,
        help="Criteria to use for similarity",
    )

def add_hyperopt_args(parser: argparse.ArgumentParser):

    """add_hopt_args."""
    # Tune args
    parser.add_argument("--cpus-per-trial", default=1, type=int)
    parser.add_argument("--gpus-per-trial", default=1, type=int)
    parser.add_argument("--num-h-samples", default=50, type=int, help="Num of trials")
    parser.add_argument("--num-seeds", default=3, type=int, help="Num runs per arg config")
    parser.add_argument("--grace-period", default=60*15, type=int)
    parser.add_argument("--max-concurrent", default=10, type=int)
    parser.add_argument("--tune-checkpoint", default=None)

    # Overwrite default savedir
    time_name = datetime.now().strftime("%Y_%m_%d_%H")
    save_default = f"results/{time_name}_hyperopt/"
    parser.set_defaults(save_dir=save_default)
    parser.add_argument(
        "--opt-eval-name",
        help="Name of optimizer criteria to opt",
        action="store",
        type=str,
        default="TopIsoScore@10",
    )
    return parser


def opt_baseparser():
    """parse train args"""
    base = get_baseparser()

    data_group = base.add_argument_group("Oracle Args")
    add_oracle_args(data_group)

    eval_group = base.add_argument_group("Eval Args")
    add_eval_args(eval_group)

    model_group = base.add_argument_group("Base Opt Args")
    add_base_opt_args(model_group)
    return base


def opt_baseparser_hyperopt():
    """parse hyperopt args"""
    base = get_baseparser()

    data_group = base.add_argument_group("Oracle Args")
    add_oracle_hyper_args(data_group)

    eval_group = base.add_argument_group("Eval Args")
    add_eval_args(eval_group)

    model_group = base.add_argument_group("Base Opt Args")
    add_base_opt_args(model_group)

    hyperopt_group = base.add_argument_group("HyperOpt Args")
    add_hyperopt_args(hyperopt_group)
    return base
