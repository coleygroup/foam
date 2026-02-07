""" run_opt.py
"""
import foam.base_parsing as base_parsing
import foam.base_run_opt as base_run_opt
import foam.opt_graph_ga_fc.graph_ga_fc as graph_ga_fc
import warnings
warnings.filterwarnings('ignore')
import sys
import yaml

def get_args():
    base_opt_parser = base_parsing.opt_baseparser()
    parser = base_opt_parser.add_argument_group("Optimizer Args")
    parser.add_argument(
        "--population-size", type=int, help="Size of population", default=100
    )
    parser.add_argument(
        "--offspring-size", type=int, help="Number of offspring", default=100
    )
    parser.add_argument(
        "--starting-seed-size", type=int, help="Number of seeds to start with. if -1 (or negative), use all seeds available.", default=200
    )
    parser.add_argument(
        "--num-islands", type=int, help="Number of islands", default=10
    )
    parser.add_argument(
        "--mutation-rate", type=float, help="Mutation rate", default=0.07
    )
    parser.add_argument(
        "--iceberg-param", type=float, help="ICEBERG score weight, use with sa-param", default=0.8
    )
    parser.add_argument(
        "--sa-param", type=float, help="SA score weight, use with iceberg-param", default=0.2
    )
    parser.add_argument(
        "--truncate", nargs="?", const=0.4, default=False, help="Truncate the population"
    )
    parser.add_argument(
        "--selection-sorting-type", default="cand_crowding", action="store", help="Type of sorting to apply during selection"
    )
    parser.add_argument(
        "--mutate-parents", default=False, action="store_true", help="Whether to mutate parents as additional offspring"
    )
    parser.add_argument(
        "--parent-tiebreak", default="cand_crowding", action="store", help="Type of tiebreaking to apply during binary tournament selection"
    )
    parser.add_argument(
        "--pubchem-seeds", default=False, action="store", help="Whether to use PubChem seeds in addition to the starting seeds"
    )
    parser.add_argument(
        "--extra-seeds", default=None, type=str, help="Path to additional set of seeds provided if desired, in addition to an existing retrieval library already specified. Should be a JSON (key: List[Inchis]); only supports MassSpecGym IDs."
    )

    args = override_config_with_cli(base_opt_parser)
    return args

def override_config_with_cli(parser):
    default_args, _ = parser.parse_known_args([])
    provided_args, unknown = parser.parse_known_args()
    if any(['use-multi-node' in f for f in sys.argv]):
        turn_off_rpc = True
    else:
        turn_off_rpc = False

    if provided_args.config:
        with open(provided_args.config, 'r') as f:
            config_dict = yaml.safe_load(f)

        valid_args = set(action.dest for action in parser._actions)

        for key, value in config_dict.items():
            config_attr_name = key.replace("-", "_")
            
            if config_attr_name not in valid_args:
                raise ValueError(f"Invalid argument '{key}' in config file.")

        parser.set_defaults(**{k.replace("-", "_"): v for k, v in config_dict.items()})
    args = parser.parse_args()
    return args


def run_opt():
    args = get_args()
    opt_cls = graph_ga_fc.GraphGAFCOptimizer
    base_run_opt.run_optimization(opt_cls=opt_cls, args=args)

if __name__=="__main__":
    run_opt()
