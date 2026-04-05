"""Benchmarks subpackage for LNPBO.

Provides the within-study benchmark runner, the Optimizer-based acquisition
loop, statistical utilities (bootstrap CI, Wilcoxon test, BH-FDR correction,
effect sizes), and analysis scripts for evaluating Bayesian optimization
strategies on LNPDB studies.
"""

_LAZY_IMPORTS = {
    # stats
    "bootstrap_ci": ".stats",
    "paired_wilcoxon": ".stats",
    "benjamini_hochberg": ".stats",
    "cohens_d_paired": ".stats",
    "rank_biserial": ".stats",
    "post_hoc_power": ".stats",
    "simple_regret": ".stats",
    "cumulative_regret": ".stats",
    "acceleration_factor": ".stats",
    "enhancement_factor": ".stats",
    "format_result": ".stats",
    # runner
    "prepare_benchmark_data": ".runner",
    "compute_metrics": ".runner",
    "STRATEGY_CONFIGS": ".runner",
    "ALL_STRATEGIES": ".runner",
    "init_history": ".runner",
    "update_history": ".runner",
    # benchmark
    "filter_study_df": ".benchmark",
}

__all__ = sorted(_LAZY_IMPORTS)


def __getattr__(name):
    if name in _LAZY_IMPORTS:
        import importlib

        mod = importlib.import_module(_LAZY_IMPORTS[name], __name__)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
