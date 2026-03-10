"""Online conformal recalibration for Bayesian Optimization (simplified variant).

At each BO round, after evaluating the batch, the calibrator accumulates
new residuals and recomputes the conformal quantile. This maintains valid
coverage throughout the campaign, even as the surrogate drifts into new
regions of chemical space.

The implementation uses standard split conformal quantile accumulation:
absolute residuals are collected across rounds and the finite-sample-valid
order statistic is taken as the conformal quantile. This is a simplified
approach inspired by — but not implementing — the full online quantile
recalibration of Deshpande et al. (2024), which uses online gradient
descent on pinball loss to adaptively track miscoverage.

The fixed-calibration baseline (current MAPIE split conformal) uses a
calibration set drawn once at initialization. As BO moves away from the
seed distribution, the fixed quantile becomes stale and coverage degrades.

References
----------
Vovk, V., Gammerman, A., & Shafer, G. (2005).
"Algorithmic Learning in a Random World." Springer.
(Primary reference for conformal quantile computation via order statistics.)

Deshpande, S., Marx, C., & Kuleshov, V.
"Online Calibrated and Conformal Prediction Improves Bayesian Optimization."
AISTATS 2024. arXiv:2112.04620.
(Motivation for online recalibration in BO; full method not implemented here.)

Kandasamy, K., Krishnamurthy, A., Schneider, J., & Poczos, B.
"Parallelised Bayesian Optimisation via Thompson Sampling."
AISTATS 2018. (cited for TS batch context)
"""

import numpy as np


class OnlineConformalCalibrator:
    """Maintains a growing set of residuals for online conformal recalibration.

    At each BO round:
      1. The surrogate predicts mu for the batch candidates before evaluation.
      2. After oracle evaluation, call ``update(y_pred, y_true)`` to record
         new absolute residuals.
      3. Call ``get_quantile()`` to obtain the recalibrated conformal quantile
         for the next round's UCB scoring.

    The quantile tracks the (1-alpha) level of the residual distribution,
    growing more accurate as more rounds are observed.

    Parameters
    ----------
    alpha : float
        Miscoverage rate. Default 0.1 gives 90% target coverage.
    """

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.residuals: list[float] = []

    def update(self, y_pred: np.ndarray, y_true: np.ndarray) -> None:
        """Add new residuals from the latest batch evaluation.

        Parameters
        ----------
        y_pred : array of shape (batch_size,)
            Surrogate predictions for the batch points (before evaluation).
        y_true : array of shape (batch_size,)
            Oracle-observed values for the batch points.
        """
        y_pred = np.asarray(y_pred).ravel()
        y_true = np.asarray(y_true).ravel()
        self.residuals.extend(np.abs(y_true - y_pred).tolist())

    def get_quantile(self) -> float:
        """Compute conformal quantile from all accumulated residuals.

        Uses the finite-sample-valid quantile: the ceil((n+1)(1-alpha))-th
        smallest residual (exact order statistic, no interpolation).

        Returns
        -------
        float
            The conformal quantile. Multiply by kappa for UCB scoring:
            ``score = mu + kappa * quantile``
        """
        n = len(self.residuals)
        if n == 0:
            return float("inf")
        rank = int(np.ceil((n + 1) * (1 - self.alpha)))
        if rank > n:
            return float("inf")
        sorted_residuals = sorted(self.residuals)
        return float(sorted_residuals[rank - 1])

    def get_coverage(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Compute empirical coverage at the current quantile level.

        Parameters
        ----------
        y_pred : array of shape (n,)
        y_true : array of shape (n,)

        Returns
        -------
        float
            Fraction of points where |y_true - y_pred| <= quantile.
        """
        if len(self.residuals) == 0:
            return 1.0
        q = self.get_quantile()
        errors = np.abs(np.asarray(y_true) - np.asarray(y_pred))
        return float(np.mean(errors <= q))

    @property
    def n_residuals(self) -> int:
        return len(self.residuals)

    def reset(self) -> None:
        self.residuals = []
