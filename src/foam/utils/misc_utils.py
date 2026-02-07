""" utils.py """
import copy
import logging
import os
import sys
from pathlib import Path

import pytorch_lightning as pl
import wandb
import yaml

try:
    from pytorch_lightning.loggers import LightningLoggerBase # cuda11
except ImportError:
    from pytorch_lightning.loggers import Logger # cuda12
    LightningLoggerBase = Logger

try:
    from pytorch_lightning.loggers.base import rank_zero_experiment # cuda11
except ImportError:
    from pytorch_lightning.loggers.logger import rank_zero_experiment # cuda12

from pytorch_lightning.utilities import rank_zero_only


def setup_run(save_dir: str, wandb_mode: str = "disable",
              debug: bool = False, seed: int = None,
              criteria: str = None, tags: list = None,
              **kwargs):
    """setup_run.

    Set seed, define logger, dump args, & update kwargs for debug

    Args:
        save_dir (str): Save dir
        wandb_mode (str): Wandb mode
        debug (bool): If true, debug
        seed (int): Seed
        kwargs
    """

    # Define default root dir
    setup_logger(save_dir, debug=debug)

    dump_dict = {}
    dump_dict.update(kwargs)
    dump_dict.update({"wandb_mode": wandb_mode,
                      "debug": debug,
                      "seed": seed
                      })

    # Dump args
    logging.info(yaml.dump(dump_dict, indent=2))

    # Dump args to file
    with open(Path(save_dir) / "args.yaml", "w") as fp:
        fp.write(yaml.dump(dump_dict, indent=2))

    # Magic to make ray / hyperopt work on slurm
    # https://github.com/ray-project/ray/issues/10995
    os.environ["SLURM_JOB_NAME"] = "bash"
    wandb_mode = wandb_mode

    # Setup wandb
    if wandb_mode == "offline":
        os.environ["WANDB_MODE"] = "offline"

    settings=wandb.Settings(
        x_stats_sampling_interval=2.0  # sample every 2 seconds instead of 15
    )
    if wandb_mode == "online" or wandb_mode == "offline":
        if kwargs.get('spec_lib_dir') and kwargs.get('spec_lib_dir') == '/home/mrunali/data/':
            tags = [kwargs['spec_id'], criteria] + tags
            wandb.init(project="isomer-opt-deploy", name=kwargs['spec_id'], tags=tags, settings=settings)
        if kwargs.get('spec_id'):
            tags = [kwargs['spec_id'], criteria] + tags
            if kwargs.get('wandb_project'):
                wandb.init(project=kwargs['wandb_project'], name=kwargs['spec_id'], tags=tags, settings=settings)
            else:
                wandb.init(project="isomer-opt-benchmark-update", name=kwargs['spec_id'], tags=tags, settings=settings)
        if kwargs.get('oracle_name'):
            wandb.init(project="isomer-opt", name=kwargs['oracle_name'], settings=settings)
        else:
            wandb.init(project="isomer_opt", settings=settings)

        wandb.config = dump_dict
    elif wandb_mode == "disable":
        pass
    else:
        raise ValueError()
    # Seed everything
    try:
        pl.utilities.seed.seed_everything(seed)
    except AttributeError:
        pl.seed_everything(seed)


def setup_logger(save_dir, log_name="output.log", debug=False):
    """Create output directory"""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    log_file = save_dir / log_name

    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(level)

    file_handler = logging.FileHandler(log_file)

    file_handler.setLevel(level)

    # Define basic logger
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[
            stream_handler,
            file_handler,
        ],
    )

    # configure logging at the root level of lightning
    # logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

    # configure logging on module level, redirect to file
    logger = logging.getLogger("pytorch_lightning.core")
    logger.addHandler(logging.FileHandler(log_file))


class ConsoleLogger(LightningLoggerBase):
    """Custom console logger class"""

    def __init__(self):
        super().__init__()

    @property
    @rank_zero_experiment
    def name(self):
        pass

    @property
    @rank_zero_experiment
    def experiment(self):
        pass

    @property
    @rank_zero_experiment
    def version(self):
        pass

    @rank_zero_only
    def log_hyperparams(self, params):
        ## No need to log hparams
        pass

    @rank_zero_only
    def log_metrics(self, metrics, step):

        metrics = copy.deepcopy(metrics)

        epoch_num = "??"
        if "epoch" in metrics:
            epoch_num = metrics.pop("epoch")

        for k, v in metrics.items():
            logging.info(f"Epoch {epoch_num}, step {step}-- {k} : {v}")

    @rank_zero_only
    def finalize(self, status):
        pass
