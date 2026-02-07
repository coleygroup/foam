""" run_hyperopt_ga.py
"""

from typing import List, Dict

import foam.base_parsing as base_parsing
import foam.base_run_hyperopt as base_run_hyperopt
import foam.opt_graph_ga_fc.graph_ga_fc as graph_ga_fc


def get_args():
    base_opt_parser = base_parsing.opt_baseparser_hyperopt()
    parser = base_opt_parser.add_argument_group("Optimizer Args")
    parser.add_argument(
        "--population-size", type=int, help="Size of population", default=100
    )
    parser.add_argument(
        "--offspring-size", type=int, help="Number of offspring", default=100
    )
    parser.add_argument(
        "--mutation-rate", type=float, help="Mutation rate", default=0.07
    )
    # For formula penalized
    parser.add_argument(
        "--formula-diff-weight",
        help="Weight to subtract formula similarity penalty",
        type=float,
        default=0.1,
    )
    return base_opt_parser.parse_args()


def get_param_space(trial):
    """ get_param_space. 

    Use Optuna to define this dynamically

    """
    trial.suggest_int("population_size", 50,200, step=50)
    trial.suggest_int("offspring_size", 50, 200, step=50)
    trial.suggest_float("mutation_rate", 0.01, 1.0)


def get_initial_points() -> List[Dict]:
    """ get_intiial_points.

    Create dictionaries defining initial configurations to test

    """
    init_base = {
        "population_size": 50,
        "offspring_size": 50,
        "mutation_rate": 0.07
    }
    return [init_base]


def run_hyperopt():
    """ run_hyperopt. """
    args = get_args()
    opt_cls = graph_ga_fc.GraphGAFCOptimizer
    base_run_hyperopt.run_hyperopt(opt_cls=opt_cls, args=args,
                                   suggest_fn=get_param_space,
                                   default_params=get_initial_points())


if __name__=="__main__":
    run_hyperopt()
