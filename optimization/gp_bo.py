"""GPyTorch + BoTorch GP Bayesian Optimization with direct discrete pool scoring.

Replaces the sklearn GP + continuous optimization + nearest-neighbor pipeline.
Key speedups:
  - GPU/MPS-accelerated kernel computations via GPyTorch
  - Direct pool scoring (no continuous optimization + projection + NN matching)
  - condition_on_observations for KB/RKB (rank-1 update, not O(N^3) refit)
  - LOVE-accelerated variance via fast_pred_var
  - Optional SVGP for large N (O(NM^2) instead of O(N^3))

This module is the public surface. The implementation is decomposed into
cohesive private modules (imported back here so the public names are unchanged):
  - ``_gp_device``: compute-device selection + NumPy->Tensor conversion.
  - ``_gp_fit``: GP-fit backends (DKL/RF/robust/multitask/exact/sparse) and
    the DKL feature-extractor / input-transform classes.
  - ``_gp_batch``: batch strategies (KB/RKB, LP, TS, q-LogEI, GIBBON) and the
    mixed discrete-continuous selector ``select_batch_mixed``.

torch/botorch/gpytorch are optional dependencies; this module is loaded lazily
via ``__getattr__`` in ``optimization/__init__.py`` only when those names are
requested, so the heavy imports here run only when GP-BO is actually used.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import gpytorch
import numpy as np
import torch
from botorch.models import SingleTaskGP, SingleTaskVariationalGP
from botorch.models.model import Model
from scipy.stats import norm

from ._gp_batch import (
    _batch_gibbon,
    _batch_kb,
    _batch_lp,
    _batch_qlogei,
    _batch_ts,
    select_batch_mixed,
)
from ._gp_device import _to_tensor, get_device
from ._gp_fit import (
    KERNEL_TYPES,
    _DKLInputTransform,
    fit_gp,
    fit_multitask_gp,
)
from .acquisition import _log_h_stable

# Re-exported names kept importable from this module for backwards-stable
# import paths (tests + optimizer.py import several of these from here, and
# tests monkeypatch ``gp_bo.fit_gp`` — which the batch module resolves through
# this module's namespace at call time, so the patch stays effective).
__all__ = [
    "KERNEL_TYPES",
    "_DKLInputTransform",
    "_to_tensor",
    "fit_gp",
    "fit_multitask_gp",
    "get_device",
    "predict",
    "score_acquisition",
    "select_batch",
    "select_batch_mixed",
]

if TYPE_CHECKING:

    # Type alias: covers SingleTaskGP, SingleTaskVariationalGP, and
    # BatchedMultiOutputGPyTorchModel (returned by condition_on_observations).
    GPModel = SingleTaskGP | SingleTaskVariationalGP | Model


def predict(
    model: GPModel,
    X: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Posterior mean and std over points X.

    Uses LOVE (Lanczos Variance Estimates) for fast variance computation.

    Parameters
    ----------
    model : Fitted GP model (exact or variational).
    X : (M, D) test points.

    Returns
    -------
    mean : (M,) posterior mean.
    std : (M,) posterior standard deviation.
    """
    device = next(model.parameters()).device
    X_t = _to_tensor(X, device)

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        posterior = model.posterior(X_t)
        mean = posterior.mean.squeeze(-1).cpu().numpy()  # type: ignore[union-attr]
        std = posterior.variance.squeeze(-1).sqrt().cpu().numpy()  # type: ignore[union-attr]

    return mean, std


def score_acquisition(
    model: GPModel,
    X_pool: np.ndarray,
    acq_type: str,
    y_best: float,
    kappa: float = 5.0,
    xi: float = 0.01,
) -> np.ndarray:
    """Score pool points under an acquisition function.

    Parameters
    ----------
    model : Fitted GP model.
    X_pool : (M, D) candidate pool.
    acq_type : One of "UCB", "EI", "LogEI".
    y_best : Best observed value (incumbent).
    kappa : UCB exploration parameter (default: 5.0).
    xi : EI/LogEI improvement jitter.

    Returns
    -------
    scores : (M,) acquisition values (higher = more desirable).
    """
    mean, std = predict(model, X_pool)
    std = np.maximum(std, 1e-10)

    if acq_type == "UCB":
        return mean + kappa * std
    elif acq_type == "EI":
        z = (mean - y_best - xi) / std
        return std * (z * norm.cdf(z) + norm.pdf(z))
    elif acq_type == "LogEI":
        z = (mean - y_best - xi) / std
        return np.log(std) + _log_h_stable(z)
    else:
        raise ValueError(f"Unknown acq_type '{acq_type}'. Choose from: UCB, EI, LogEI")


def _estimate_lipschitz(model: GPModel) -> float:
    """Estimate Lipschitz constant L from kernel hyperparameters.

    For Matern/RBF: L = sqrt(outputscale) / min(lengthscale), a conservative
    upper bound on ||d mu / dx||.

    For kernels without lengthscale (e.g. Tanimoto): returns 1.0 as a
    reasonable default. LP penalization still works but the exclusion
    radius is not calibrated to the kernel geometry. For Tanimoto
    kernels, prefer TS-Batch over LP for more reliable batch diversity.
    """
    kernel = getattr(model, "covar_module", None)
    if kernel is None:
        # SingleTaskVariationalGP wraps the kernel
        inner = getattr(model, "model", None)
        kernel = getattr(inner, "covar_module", None) if inner is not None else None
    if kernel is None:
        return 1.0

    base = getattr(kernel, "base_kernel", None)
    if base is not None and getattr(base, "lengthscale", None) is not None:
        outputscale = kernel.outputscale.detach().cpu().item()  # type: ignore[union-attr]
        lengthscale = base.lengthscale.detach().cpu().numpy().ravel()  # type: ignore[union-attr]
        return float(np.sqrt(outputscale) / np.min(lengthscale))

    ls = getattr(kernel, "lengthscale", None)
    if ls is not None:
        lengthscale = ls.detach().cpu().numpy().ravel()
        return float(1.0 / np.min(lengthscale))

    return 1.0


def select_batch(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_pool: np.ndarray,
    pool_indices: np.ndarray | list,
    batch_size: int,
    acq_type: str,
    batch_strategy: str,
    kappa: float = 5.0,
    xi: float = 0.01,
    seed: int = 42,
    use_sparse: bool = False,
    n_inducing: int = 512,
    kernel_type: str = "matern",
    kernel_kwargs: dict | None = None,
    train_Yvar: np.ndarray | None = None,
) -> list[int]:
    """Select a batch of points from a discrete candidate pool.

    Parameters
    ----------
    X_train : (N, D) training features.
    y_train : (N,) training targets.
    X_pool : (M, D) candidate pool features.
    pool_indices : Original dataframe indices for pool rows.
    batch_size : Number of points to select.
    acq_type : Acquisition function ("UCB", "EI", "LogEI").
    batch_strategy : Batch construction method:
        "kb"    - Kriging Believer (Ginsbourger et al., 2010)
        "rkb"   - Randomized Kriging Believer (Sugiura et al., 2026)
        "lp"    - Local Penalization (Gonzalez et al., 2016)
        "ts"    - Thompson Sampling (Kandasamy et al., 2018)
        "qlogei" - Joint q-LogEI (Ament et al., 2023, via BoTorch)
        "gibbon" - GIBBON: information-theoretic + DPP diversity (Moss et al., 2021)
    kappa : UCB exploration parameter.
    xi : EI/LogEI improvement jitter.
    seed : Random seed.
    use_sparse : Use variational GP for large training sets.
    n_inducing : Number of inducing points for sparse GP.
    kernel_type : Kernel function ("matern", "tanimoto", "compositional", etc.).
    kernel_kwargs : dict, optional
        Extra arguments for kernel construction (e.g., index lists for
        ``kernel_type="compositional"``).

    Returns
    -------
    List of selected pool_indices (length batch_size or fewer if pool exhausted).
    """
    pool_indices = np.asarray(pool_indices)
    batch_size = min(batch_size, len(X_pool))

    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)

    model = fit_gp(
        X_train,
        y_train,
        use_sparse=use_sparse,
        n_inducing=n_inducing,
        _rng=rng,
        kernel_type=kernel_type,
        kernel_kwargs=kernel_kwargs,
        train_Yvar=train_Yvar,
        seed=seed,
    )
    y_best = float(y_train.max())

    strategy = batch_strategy.lower()
    device = next(model.parameters()).device
    # Robust Cholesky jitter for the entire acquisition step. Ill-conditioned
    # kernels (e.g. Tanimoto / DKL on near-duplicate fingerprint rows) yield a
    # posterior covariance that fails gpytorch's default 1e-3 double-precision
    # jitter cap; raising the cap here keeps the posterior positive-definite for
    # every batch strategy (kb/rkb/lp/ts/qlogei/gibbon) instead of crashing and
    # truncating the run. The GP is float64, so double_value is the binding one.
    with (
        gpytorch.settings.cholesky_jitter(float_value=1e-2, double_value=1e-2),
        gpytorch.settings.cholesky_max_tries(6),
    ):
        if strategy == "qlogei":
            return _batch_qlogei(model, X_train, X_pool, pool_indices, batch_size, device=device)
        elif strategy == "gibbon":
            return _batch_gibbon(model, X_pool, pool_indices, batch_size, device=device)
        elif strategy == "ts":
            return _batch_ts(model, X_pool, pool_indices, batch_size, seed)
        elif strategy == "kb":
            return _batch_kb(
                model,
                X_pool,
                pool_indices,
                batch_size,
                acq_type,
                y_best,
                kappa,
                xi,
                randomize=False,
            )
        elif strategy == "rkb":
            return _batch_kb(
                model,
                X_pool,
                pool_indices,
                batch_size,
                acq_type,
                y_best,
                kappa,
                xi,
                randomize=True,
            )
        elif strategy == "lp":
            return _batch_lp(
                model,
                X_pool,
                pool_indices,
                batch_size,
                acq_type,
                y_best,
                kappa,
                xi,
            )
        else:
            raise ValueError(f"Unknown batch_strategy '{batch_strategy}'. Choose from: kb, rkb, lp, ts, qlogei, gibbon")
