""" base_run_opt.py
"""
import argparse

import foam.oracles as oracles
import foam.utils as utils


def run_optimization(opt_cls,  args: argparse.Namespace):
    """ run_optmization.

    Args:
        opt_cls: Optimizer class
        args: argparse inputs

    """
    kwargs = args.__dict__
    opt_name = opt_cls.opt_name()
    kwargs['opt_name'] = opt_name
    utils.setup_run(**kwargs)

    oracle = oracles.get_oracle(**kwargs)
    optimizer = opt_cls(oracle=oracle, **kwargs)
    optimizer.optimize()
