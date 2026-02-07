"""FOAM: Formula-constrained Optimization for Annotating Metabolites

A toolkit for molecular structure elucidation using genetic algorithms
and forward spectral simulation.

Main Components:
    - OptimizerBase: Abstract base class for optimization algorithms
    - GraphGAOptimizer: Formula-constrained graph genetic algorithm
    - MolOracle: Scoring interface for molecular candidates
    - BaseEvaluator: Evaluation metrics for optimization runs

Example:
    >>> from foam import GraphGAOptimizer, oracle_registry
    >>> oracle = oracle_registry["TaniOracle"](...)
    >>> optimizer = GraphGAOptimizer(oracle=oracle, max_calls=1000)
    >>> optimizer.optimize()
"""

from foam.base_opt import OptimizerBase, OptSignals
from foam.oracles import MolOracle, Oracle, oracle_registry
from foam.evaluators import BaseEvaluator, evaluator_registry

# Lazy imports for heavier modules
def get_graph_ga_optimizer():
    """Lazily import GraphGAOptimizer to avoid heavy dependencies on import."""
    from foam.opt_graph_ga_fc.graph_ga_fc import GraphGAOptimizer
    return GraphGAOptimizer

__all__ = [
    # Core classes
    "OptimizerBase",
    "OptSignals",
    "MolOracle",
    "Oracle",
    "BaseEvaluator",
    # Registries
    "oracle_registry",
    "evaluator_registry",
    # Lazy loaders
    "get_graph_ga_optimizer",
]

__version__ = "0.1.0"
