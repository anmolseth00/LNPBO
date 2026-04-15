"""Exact and baseline online recalibration methods for Bayesian optimization.

This module keeps the exact Deshpande et al. (2024) recalibration path separate
from the older cumulative split-conformal baseline.

Exact path
----------
- Build a recalibration dataset from cross-validated held-out forecasts
  (Algorithm 4).
- Learn a pointwise monotone recalibrator ``R_t`` via the paper's online
  linearized quantile-pinball objective (Eq. 11).
- Compose the base probabilistic model ``M_t`` with ``R_t`` to obtain a
  calibrated quantile/CDF interface for BO acquisitions.

Baseline path
-------------
- Maintain accumulated absolute residuals and use the finite-sample conformal
  order statistic as a UCB width. This is a cumulative split-conformal
  baseline, not the paper's recalibration algorithm.

References
----------
Deshpande, S., Marx, C., & Kuleshov, V. (2024).
"Online Calibrated and Conformal Prediction Improves Bayesian Optimization."
AISTATS 2024. PMLR 238.

Vovk, V., Gammerman, A., & Shafer, G. (2005).
"Algorithmic Learning in a Random World." Springer.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol

import numpy as np
from scipy import integrate
from scipy.stats import norm
from sklearn.preprocessing import MinMaxScaler


class ProbabilisticModelAdapter(Protocol):
    """Interface required by the exact recalibration algorithms."""

    def quantile(self, X: np.ndarray, p: float) -> np.ndarray: ...

    def cdf(self, X: np.ndarray, y: np.ndarray) -> np.ndarray: ...

    def inverse_quantile_level(self, X: np.ndarray, y: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class RecalibrationDataset:
    """Held-out forecast outcomes expressed as inverse quantile levels."""

    inverse_quantile_levels: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "inverse_quantile_levels",
            np.clip(np.asarray(self.inverse_quantile_levels, dtype=float).ravel(), 0.0, 1.0),
        )


def create_leave_one_out_splits(n_samples: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Default ``CreateSplits`` implementation used in the paper experiments."""
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(n_samples):
        train_idx = np.array([j for j in range(n_samples) if j != i], dtype=int)
        test_idx = np.array([i], dtype=int)
        splits.append((train_idx, test_idx))
    return splits


def build_recalibration_dataset(
    X: np.ndarray,
    y: np.ndarray,
    *,
    model_factory,
    splits: list[tuple[np.ndarray, np.ndarray]] | None = None,
) -> RecalibrationDataset:
    """Construct Algorithm 4's recalibration set from held-out forecasts."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    if len(X) == 0:
        return RecalibrationDataset(np.empty((0,), dtype=float))
    if len(X) == 1:
        # No held-out training split is possible. The exact path falls back to
        # the identity recalibrator until enough data is available.
        return RecalibrationDataset(np.empty((0,), dtype=float))

    if splits is None:
        splits = create_leave_one_out_splits(len(X))

    inverse_levels = []
    for train_idx, test_idx in splits:
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        model = model_factory()
        model.fit(X[train_idx], y[train_idx])
        inverse_levels.extend(model.inverse_quantile_level(X[test_idx], y[test_idx]).tolist())

    return RecalibrationDataset(np.asarray(inverse_levels, dtype=float))


class ExactOnlineRecalibrator:
    """Exact pointwise solver for Eq. 11 from Deshpande et al. (2024)."""

    def __init__(self, eta: float = 0.05) -> None:
        self.eta = float(eta)
        if self.eta <= 0:
            raise ValueError("eta must be positive.")
        self.dataset_ = RecalibrationDataset(np.empty((0,), dtype=float))

    def fit(self, dataset: RecalibrationDataset | np.ndarray) -> "ExactOnlineRecalibrator":
        if not isinstance(dataset, RecalibrationDataset):
            dataset = RecalibrationDataset(np.asarray(dataset, dtype=float))
        self.dataset_ = dataset
        self._solve_uncached.cache_clear()
        return self

    def _step(self, q: float, u: float, p: float) -> float:
        grad = float(u <= q) - float(p)
        return float(np.clip(q - self.eta * grad, 0.0, 1.0))

    @lru_cache(maxsize=2048)
    def _solve_uncached(self, p_rounded: float) -> float:
        p = float(np.clip(p_rounded, 0.0, 1.0))
        q = 0.0
        for u in self.dataset_.inverse_quantile_levels:
            q = self._step(q, float(u), p)
        return float(q)

    def recalibrate(self, p: float) -> float:
        return self._solve_uncached(round(float(p), 12))

    def __call__(self, p: float) -> float:
        return self.recalibrate(p)

    def inverse(self, q: float, *, tol: float = 1e-6, max_iter: int = 80) -> float:
        """Invert the monotone recalibrator via bisection."""
        target = float(np.clip(q, 0.0, 1.0))
        lo, hi = 0.0, 1.0
        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            val = self.recalibrate(mid)
            if val <= target:
                lo = mid
            else:
                hi = mid
            if hi - lo <= tol:
                break
        return float(lo)


class CalibratedProbabilisticModel:
    """Compose a base probabilistic model ``M`` with recalibrator ``R``."""

    def __init__(self, base_model: ProbabilisticModelAdapter, recalibrator: ExactOnlineRecalibrator) -> None:
        self.base_model = base_model
        self.recalibrator = recalibrator

    def quantile(self, X: np.ndarray, p: float) -> np.ndarray:
        return self.base_model.quantile(X, self.recalibrator.recalibrate(p))

    def cdf(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        raw_levels = self.base_model.inverse_quantile_level(X, y)
        return np.asarray([self.recalibrator.inverse(level) for level in raw_levels], dtype=float)

    def inverse_quantile_level(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.cdf(X, y)

    def probability_of_improvement(self, X: np.ndarray, incumbent: float) -> np.ndarray:
        return 1.0 - self.cdf(X, np.full(len(np.asarray(X)), incumbent, dtype=float))

    def expected_improvement(self, X: np.ndarray, incumbent: float) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        incumbent = float(incumbent)
        eis = np.zeros(len(X), dtype=float)
        for i, x in enumerate(X):
            x_single = x[None, :]
            p0 = float(np.clip(self.cdf(x_single, np.array([incumbent], dtype=float))[0], 0.0, 1.0))
            if p0 >= 1.0 - 1e-12:
                eis[i] = 0.0
                continue

            def integrand(p: float) -> float:
                q = float(self.quantile(x_single, p)[0])
                return max(q - incumbent, 0.0)

            eis[i] = float(integrate.quad(integrand, p0, 1.0, limit=64)[0])
        return eis


@dataclass
class GaussianXGBQuantileModel:
    """XGBoost mean predictor with a Gaussian predictive distribution."""

    n_estimators: int = 200
    random_state: int = 0
    n_jobs: int = 1
    min_sigma: float = 1e-6

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianXGBQuantileModel":
        from xgboost import XGBRegressor

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()

        self.scaler_ = MinMaxScaler()
        X_scaled = self.scaler_.fit_transform(X)
        self.mean_model_ = XGBRegressor(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbosity=0,
        )
        self.mean_model_.fit(X_scaled, y)

        mu_train = self.mean_model_.predict(X_scaled)
        residuals = y - mu_train
        sigma = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else float(np.std(residuals))
        self.sigma_ = max(sigma, self.min_sigma)
        return self

    def _predict_mean(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return np.asarray(self.mean_model_.predict(self.scaler_.transform(X)), dtype=float).ravel()

    def predict_mean_std(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mu = self._predict_mean(X)
        sigma = np.full(mu.shape, self.sigma_, dtype=float)
        return mu, sigma

    def cdf(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        mu, sigma = self.predict_mean_std(X)
        z = (np.asarray(y, dtype=float).ravel() - mu) / sigma
        return norm.cdf(z)

    def inverse_quantile_level(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.cdf(X, y)

    def quantile(self, X: np.ndarray, p: float) -> np.ndarray:
        p = float(np.clip(p, 1e-9, 1.0 - 1e-9))
        mu, sigma = self.predict_mean_std(X)
        return mu + sigma * norm.ppf(p)


class CumulativeSplitConformalUCBBaseline:
    """Residual-accumulation baseline kept separate from the exact method."""

    def __init__(self, alpha: float = 0.1):
        self.alpha = float(alpha)
        self.residuals: list[float] = []

    def update(self, y_pred: np.ndarray, y_true: np.ndarray) -> None:
        y_pred = np.asarray(y_pred).ravel()
        y_true = np.asarray(y_true).ravel()
        self.residuals.extend(np.abs(y_true - y_pred).tolist())

    def get_quantile(self) -> float:
        n = len(self.residuals)
        if n == 0:
            return float("inf")
        rank = int(np.ceil((n + 1) * (1 - self.alpha)))
        if rank > n:
            return float("inf")
        sorted_residuals = sorted(self.residuals)
        return float(sorted_residuals[rank - 1])

    def get_coverage(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        if len(self.residuals) == 0:
            return float("nan")
        q = self.get_quantile()
        errors = np.abs(np.asarray(y_true) - np.asarray(y_pred))
        return float(np.mean(errors <= q))


def update_cumulative_split_conformal_batch(
    calibrator: CumulativeSplitConformalUCBBaseline,
    batch_preds: np.ndarray,
    batch_true: np.ndarray,
) -> tuple[float, float]:
    """Measure baseline coverage against the pre-update quantile, then ingest."""
    q_before = calibrator.get_quantile()
    if np.isinf(q_before):
        coverage = float("nan")
    else:
        errors = np.abs(np.asarray(batch_true) - np.asarray(batch_preds))
        coverage = float(np.mean(errors <= q_before))
    calibrator.update(batch_preds, batch_true)
    return coverage, float(calibrator.get_quantile())
