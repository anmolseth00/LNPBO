from __future__ import annotations
from bayes_opt import acquisition
from bayes_opt.target_space import TargetSpace
from copy import deepcopy
import numpy as np
from scipy.stats import norm
from scipy.special import erfcx


class KrigingBeliever(acquisition.AcquisitionFunction):
    """Batch acquisition via the Kriging Believer heuristic.

    Sequentially selects batch points by hallucinating each selected point
    as having a target equal to the GP posterior mean, then refitting the GP
    on the augmented dataset before selecting the next point.

    Reference
    ---------
    Ginsbourger, D., Le Riche, R., & Carraro, L.
    "Kriging Is Well-Suited to Parallelize Optimization."
    Computational Intelligence in Expensive Optimization Problems,
    Springer, 2010, pp. 131-162.
    """

    def __init__(self, base_acquisition: acquisition.AcquisitionFunction, random_state=None, atol=1e-5, rtol=1e-8) -> None:
        super().__init__(random_state)
        self.base_acquisition = base_acquisition
        self.dummies = []
        self.atol = atol
        self.rtol = rtol

    def base_acq(self, *args, **kwargs):
        return self.base_acquisition.base_acq(*args, **kwargs)

    def clear_dummies(self):
        self.dummies = []

    def _remove_expired_dummies(self, target_space: TargetSpace) -> None:
        dummies = []
        for dummy in self.dummies:
            close = np.isclose(dummy, target_space.params, rtol=self.rtol, atol=self.atol)
            if not close.all(axis=1).any():
                dummies.append(dummy)
        self.dummies = dummies

    def _create_dummy_target_space(self, gp, target_space: TargetSpace, fit_gp: bool=True) -> TargetSpace:
        # Check if any dummies have been evaluated and remove them
        self._remove_expired_dummies(target_space)
        if fit_gp:
            self._fit_gp(gp, target_space)
        # Create a copy of the target space
        dummy_target_space = deepcopy(target_space)

        if self.dummies:
            dummy_targets = gp.predict(np.array(self.dummies).reshape((len(self.dummies), -1)))
            if dummy_target_space.constraint is not None:
                dummy_constraints = target_space.constraint.approx(np.array(self.dummies).reshape((len(self.dummies), -1)))
            for idx, dummy in enumerate(self.dummies):
                if dummy_target_space.constraint is not None:
                    dummy_target_space.register(dummy, dummy_targets[idx].squeeze(), dummy_constraints[idx].squeeze())
                else:
                    dummy_target_space.register(dummy, dummy_targets[idx].squeeze())
        return dummy_target_space

    def suggest(self, gp, target_space: TargetSpace, n_random=10_000, n_smart=10, fit_gp:bool=True, random_state=None) -> np.ndarray:
        if len(target_space) == 0:
            raise ValueError("Cannot suggest a point without previous samples. Use target_space.random_sample() to generate a point.")

        # fit GP only if necessary
        # GP needs to be fitted to predict dummy targets
        dummy_target_space = self._create_dummy_target_space(gp, target_space, fit_gp=fit_gp)

        # Create a copy of the GP
        dummy_gp = deepcopy(gp)
        # Always fit dummy GP!
        x_max = self.base_acquisition.suggest(
            dummy_gp,
            dummy_target_space,
            n_random=n_random,
            n_smart=n_smart,
            fit_gp=True,
            random_state=random_state
        )
        self.dummies.append(x_max)

        return x_max

    def get_acquisition_params(self):
        return self.base_acquisition.get_acquisition_params()

    def set_acquisition_params(self, **params):
        self.base_acquisition.set_acquisition_params(**params)


# ---------------------------------------------------------------------------
# Numerically stable log_h for LogEI
# ---------------------------------------------------------------------------

_LOG_2PI_HALF = 0.5 * np.log(2.0 * np.pi)
_LOG_PI2_HALF = 0.5 * np.log(np.pi / 2.0)
_INV_SQRT_EPS = 1.0 / np.sqrt(np.finfo(float).eps)


def _log_h_stable(z):
    """Compute log(z * Phi(z) + phi(z)) with three-case numerical stability.

    Implements Equation 9 from Ament et al. (2023):
      Case 1 (z > -1):      direct log(z*Phi(z) + phi(z))
      Case 2 (-1/sqrt(eps) < z <= -1): log1mexp-based stable form
      Case 3 (z <= -1/sqrt(eps)):       asymptotic expansion

    Reference: Ament et al., "Unexpected Improvements to Expected Improvement
    for Bayesian Optimization", NeurIPS 2023, Eq. 9 (arXiv:2310.20708).
    """
    z = np.asarray(z, dtype=float)
    result = np.empty_like(z)

    # Case 1: z > -1 (direct computation is numerically fine)
    mask1 = z > -1.0
    if np.any(mask1):
        z1 = z[mask1]
        h = z1 * norm.cdf(z1) + norm.pdf(z1)
        result[mask1] = np.log(np.maximum(h, np.finfo(float).tiny))

    # Case 3: z <= -1/sqrt(eps) (asymptotic, h ≈ phi(z)/(-z))
    mask3 = z <= -_INV_SQRT_EPS
    if np.any(mask3):
        z3 = z[mask3]
        result[mask3] = -0.5 * z3 ** 2 - _LOG_2PI_HALF - 2.0 * np.log(-z3)

    # Case 2: -1/sqrt(eps) < z <= -1 (intermediate, use erfcx + log1mexp)
    mask2 = ~mask1 & ~mask3
    if np.any(mask2):
        z2 = z[mask2]
        abs_z = np.abs(z2)
        # erfcx(x) = exp(x^2) * erfc(x), so erfcx(-z/sqrt(2)) = exp(z^2/2) * erfc(-z/sqrt(2))
        # Using erfcx directly avoids overflow from exp(z^2/2) for large |z|
        erfcx_val = erfcx(-z2 / np.sqrt(2.0))
        inner = np.log(np.maximum(erfcx_val * abs_z, np.finfo(float).tiny)) + _LOG_PI2_HALF
        # log1mexp(a) = log(1 - exp(a)) for a < 0; here inner should be < 0
        # Clamp inner to be negative for stability
        inner = np.minimum(inner, -np.finfo(float).eps)
        log1mexp_val = np.where(
            inner > -0.6931,  # -log(2)
            np.log(-np.expm1(inner)),
            np.log1p(-np.exp(inner)),
        )
        result[mask2] = -0.5 * z2 ** 2 - _LOG_2PI_HALF + log1mexp_val

    return result


class LogExpectedImprovement(acquisition.AcquisitionFunction):
    """Expected Improvement computed in log-space for numerical stability.

    Standard EI suffers from numerical underflow in regions far from the
    incumbent, creating flat zero gradients that stall acquisition function
    optimization. LogEI stays finite everywhere and produces a smoother
    landscape for the optimizer.

    EI(x) = sigma(x) * h(z),  where h(z) = z*Phi(z) + phi(z)
    LogEI(x) = log(sigma(x)) + log_h(z)

    log_h uses a three-case piecewise computation for full numerical
    stability across all z values (Eq. 9 of the reference).

    Reference
    ---------
    Ament, S., Daulton, S., Eriksson, D., Balandat, M., & Bakshy, E.
    "Unexpected Improvements to Expected Improvement for Bayesian
    Optimization." NeurIPS 2023. arXiv:2310.20708.
    """

    def __init__(self, xi=0.01, random_state=None):
        super().__init__(random_state)
        self.xi = xi
        self.y_max = None

    def base_acq(self, mean, std):
        if self.y_max is None:
            raise ValueError("y_max is not set. Call suggest() or set y_max manually.")
        z = (mean - self.y_max - self.xi) / std
        return np.log(std) + _log_h_stable(z)

    def suggest(self, gp, target_space, n_random=10_000, n_smart=10, fit_gp=True, random_state=None):
        if len(target_space) == 0:
            raise ValueError("Cannot suggest a point without previous samples.")
        self.y_max = target_space._target_max()
        return super().suggest(
            gp, target_space, n_random=n_random, n_smart=n_smart,
            fit_gp=fit_gp, random_state=random_state,
        )

    def get_acquisition_params(self):
        return {"xi": self.xi}

    def set_acquisition_params(self, **params):
        self.xi = params.get("xi", self.xi)
