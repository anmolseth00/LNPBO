"""Optimization subpackage: GP-BO pipeline, acquisition functions, and kernels."""

_LAZY_IMPORTS = {
    "Optimizer": ".optimizer",
    "fit_gp": ".gp_bo",
    "predict": ".gp_bo",
    "score_acquisition": ".gp_bo",
    "select_batch": ".gp_bo",
    "select_batch_mixed": ".gp_bo",
    "TanimotoKernel": ".kernels",
    "AitchisonKernel": ".kernels",
    "CompositionalProductKernel": ".kernels",
    "RandomForestKernel": ".rf_kernel",
}

__all__ = sorted(_LAZY_IMPORTS)


def __getattr__(name):
    if name in _LAZY_IMPORTS:
        import importlib

        mod = importlib.import_module(_LAZY_IMPORTS[name], __name__)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
