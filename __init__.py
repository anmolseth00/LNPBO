"""LNPBO: Bayesian Optimization for Lipid Nanoparticle Formulation Design.

A Python toolkit for data-driven optimization of lipid nanoparticle (LNP)
formulations using Bayesian optimization and tree-ensemble surrogates.

Main entry points:
    Optimizer: High-level API for running LNP optimization campaigns.
    Dataset: Load and encode LNP formulation datasets from LNPDB.

Low-level GP-BO functions (for advanced users):
    fit_gp: Fit a Gaussian process surrogate model.
    predict: Generate predictions from a fitted GP.
    score_acquisition: Evaluate acquisition function values.
    select_batch: Select a batch of candidates for evaluation.

Example:
    >>> from LNPBO import Optimizer
    >>> opt = Optimizer.from_study("39060305")
    >>> results = opt.run()

See Also:
    Collins & Seth et al. (2026). "Benchmarking Optimization Strategies
    for Lipid Nanoparticle Design: 38 Strategy Configurations Across 26 Studies."
"""

__version__ = "0.1.0"

__all__ = [
    "Dataset",
    "Optimizer",
    "fit_gp",
    "predict",
    "score_acquisition",
    "select_batch",
    "select_batch_mixed",
]


def __getattr__(name):
    if name == "Optimizer":
        from .optimization.optimizer import Optimizer

        return Optimizer
    if name == "Dataset":
        from .data.dataset import Dataset

        return Dataset
    if name in ("fit_gp", "predict", "score_acquisition", "select_batch", "select_batch_mixed"):
        from .optimization import gp_bo

        return getattr(gp_bo, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
