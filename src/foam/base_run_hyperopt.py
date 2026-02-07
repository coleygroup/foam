""" base_run_opt.py
"""
import argparse
import copy
import logging
import os
from pathlib import Path
from typing import Callable

import foam.oracles as oracles
import foam.utils as utils
import numpy as np
import pytorch_lightning as pl
import ray
import yaml
from ray import tune
from ray.air import session
from ray.air.config import RunConfig
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search import Repeater
from ray.tune.search.optuna import OptunaSearch

# from ray.tune.schedulers.async_hyperband import ASHAScheduler
# from ray.tune.integration.pytorch_lightning import TuneReportCallback

def score_fn(config: dict, base_args: dict, orig_dir: str, opt_cls: Callable):
    """score_fn.

    Score function to optimize against.

    Args:
        config (dict): config
        base_args (dict): base args
        orig_dir (str): Original directory
        opt_cls (class obj):

    """
    os.chdir(orig_dir)
    kwargs_copy = copy.deepcopy(base_args)
    kwargs_copy.update(config)
    logging.info("Starting score fn")
    kwargs_copy["save_dir"] = tune.get_trial_dir()

    repeat_param = "__trial_index__"
    if repeat_param in config:
        seed = config[repeat_param] + 1
    else:
        seed = 42

    # Ensure seed has a real value and reset seed before we split
    pl.utilities.seed.seed_everything(seed)
    kwargs_copy["num_workers"] = kwargs_copy["cpus_per_trial"]

    per_oracle_res = {}
    for oracle_name in kwargs_copy.get("oracle_names"):
        kwargs_copy["oracle_name"] = oracle_name
        oracle = oracles.get_oracle(**kwargs_copy)
        optimizer = opt_cls(oracle=oracle, **kwargs_copy)
        outputs = optimizer.optimize()
        opt_criteria = kwargs_copy.get("opt_eval_name")
        output_stats = outputs.get("output_stats", {})
        all_metrics = {k: v
                       for i, j in output_stats.items()
                       for k, v in j.items()}
        per_oracle_res[oracle_name] = all_metrics

    top_scores = [v.get(opt_criteria) for k, v in per_oracle_res.items()]
    top_score = np.mean(top_scores)
    out_str = yaml.dump(per_oracle_res)
    logging.info(f"Finished trial with output:\n{out_str}")

    trial_out = {"score": top_score}
    session.report(trial_out)
    return trial_out


def run_hyperopt(opt_cls, args: argparse.Namespace,
                 suggest_fn: Callable, default_params: list):
    """run_hyperopt.

    Args:
        opt_cls: Class definition
        args: Namespace parsed args
        suggest_fn: Function used to suggest new params/configs
        default_params: List of initial configs
    """
    kwargs = args.__dict__
    opt_name = opt_cls.opt_name()
    kwargs['opt_name'] = opt_name
    utils.setup_run(**kwargs)
    ray.init()

    # Fix base_args based upon tune args
    kwargs['gpu'] = args.gpus_per_trial > 0
    if kwargs['debug']:
        kwargs['num_h_samples'] = 10
    save_dir = kwargs['save_dir']
    pl.utilities.seed.seed_everything(kwargs.get("seed"))

    # Define score function
    trainable = tune.with_parameters(
        score_fn,
        base_args=kwargs,
        orig_dir=Path().resolve(),
        opt_cls=opt_cls
    )
    metric = "score"
    # Include cpus and gpus per trial
    trainable = tune.with_resources(trainable,
                                    {"cpu": kwargs['cpus_per_trial'],
                                     "gpu": kwargs['gpus_per_trial']})
    search_algo = OptunaSearch(metric=metric, mode="max",
                               points_to_evaluate=default_params,
                               space=suggest_fn,)
    search_algo = ConcurrencyLimiter(search_algo,
                                     max_concurrent=args.max_concurrent)
    search_algo = Repeater(search_algo, repeat=kwargs.get("num_seeds"))
    tuner = tune.Tuner(
        trainable,
        # param_space=param_space,
        tune_config=tune.TuneConfig(
            mode="max",
            metric=metric,
            search_alg=search_algo,
            num_samples=kwargs.get("num_h_samples")*kwargs.get("num_seeds"),),
        run_config=RunConfig(name=None, local_dir=args.save_dir)
    )

    if kwargs.get('tune_checkpoint') is not None:
        ckpt = str(Path(kwargs['tune_checkpoint']).resolve())
        tuner = tuner.restore(path=ckpt)

    results = tuner.fit()
    best_trial = results.get_best_result()
    output = {"score": best_trial.metrics[metric],
              "config": best_trial.config}
    out_str = yaml.dump(output, indent=2)
    logging.info(out_str)
    with open(Path(save_dir) / "best_trial.yaml", "w") as f:
        f.write(out_str)

    # Output full res table
    results.get_dataframe().to_csv(Path(save_dir) / "full_res_tbl.tsv",
                                   sep="\t",
                                   index=None)
