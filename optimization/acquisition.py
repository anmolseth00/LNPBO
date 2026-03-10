import warnings
from copy import deepcopy

import numpy as np
from bayes_opt import acquisition
from bayes_opt.target_space import TargetSpace
from scipy.special import erfcx
from scipy.stats import norm


def _sample_f(gp, X, n_samples=1, random_state=None):
    """Draw samples from the GP posterior over f (noise-free).

    ``gp.sample_y`` draws from the posterior predictive y = f + noise.
    For Thompson sampling and Randomized KB, we need samples from f
    (the latent function) without observation noise.

    Reference: Rasmussen & Williams (2006), Algorithm 2.1.
    """
    rng = np.random.RandomState(random_state)
    mu, cov = gp.predict(X, return_cov=True)
    # Jitter for numerical stability
    cov += np.eye(len(X)) * 1e-10
    try:
        L = np.linalg.cholesky(cov)
        z = rng.standard_normal((len(X), n_samples))
        samples = mu[:, np.newaxis] + L @ z
    except np.linalg.LinAlgError:
        # Fallback: diagonal approximation
        std = np.sqrt(np.maximum(np.diag(cov), 0.0))
        samples = mu[:, np.newaxis] + std[:, np.newaxis] * rng.standard_normal((len(X), n_samples))
    return samples


class KrigingBeliever(acquisition.AcquisitionFunction):
    """Batch acquisition via the Kriging Believer heuristic.

    Sequentially selects batch points by hallucinating each selected point
    as having a target equal to the GP posterior mean, then refitting the GP
    on the augmented dataset before selecting the next point.

    When ``randomize=True``, dummy targets are drawn from the GP posterior
    over f (noise-free) instead of the posterior mean. This adds diversity
    to the batch and avoids corrupting the GP with deterministic fictitious
    data, especially at larger batch sizes.

    References
    ----------
    Ginsbourger, D., Le Riche, R., & Carraro, L.
    "Kriging Is Well-Suited to Parallelize Optimization."
    Computational Intelligence in Expensive Optimization Problems,
    Springer, 2010, pp. 131-162.

    Sugiura, S., Takeuchi, I., & Takeno, S. "Randomized Kriging Believer
    for Parallel Bayesian Optimization with Regret Bounds."
    arXiv:2603.01470, March 2026.
    """

    def __init__(
        self, base_acquisition: acquisition.AcquisitionFunction, random_state=None, atol=1e-5, rtol=1e-8,
        randomize: bool = False,
    ) -> None:
        super().__init__(random_state)
        self.base_acquisition = base_acquisition
        self.dummies = []
        self.atol = atol
        self.rtol = rtol
        self.randomize = randomize
        self._sample_counter = 0

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

    def _create_dummy_target_space(self, gp, target_space: TargetSpace, fit_gp: bool = True) -> TargetSpace:
        # Check if any dummies have been evaluated and remove them
        self._remove_expired_dummies(target_space)
        if fit_gp:
            self._fit_gp(gp, target_space)
        # Create a copy of the target space
        dummy_target_space = deepcopy(target_space)

        if self.dummies:
            X_dummies = np.array(self.dummies).reshape((len(self.dummies), -1))
            if self.randomize:
                dummy_targets = _sample_f(
                    gp, X_dummies, n_samples=1, random_state=self._sample_counter,
                ).ravel()
                self._sample_counter += 1
            else:
                dummy_targets = gp.predict(X_dummies)
            if dummy_target_space.constraint is not None:
                dummy_constraints = target_space.constraint.approx(  # type: ignore[union-attr]
                    np.array(self.dummies).reshape((len(self.dummies), -1))
                )
            for idx, dummy in enumerate(self.dummies):
                if dummy_target_space.constraint is not None:
                    dummy_target_space.register(dummy, dummy_targets[idx].squeeze(), dummy_constraints[idx].squeeze())
                else:
                    dummy_target_space.register(dummy, dummy_targets[idx].squeeze())
        return dummy_target_space

    def suggest(
        self, gp, target_space: TargetSpace, n_random=10_000, n_smart=10, fit_gp: bool = True, random_state=None
    ) -> np.ndarray:
        if len(target_space) == 0:
            raise ValueError(
                "Cannot suggest a point without previous samples. Use target_space.random_sample() to generate a point."
            )

        # fit GP only if necessary
        # GP needs to be fitted to predict dummy targets
        dummy_target_space = self._create_dummy_target_space(gp, target_space, fit_gp=fit_gp)

        # Create a copy of the GP
        dummy_gp = deepcopy(gp)
        # Always fit dummy GP!
        x_max = self.base_acquisition.suggest(
            dummy_gp, dummy_target_space, n_random=n_random, n_smart=n_smart, fit_gp=True, random_state=random_state
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
        result[mask3] = -0.5 * z3**2 - _LOG_2PI_HALF - 2.0 * np.log(-z3)

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
        result[mask2] = -0.5 * z2**2 - _LOG_2PI_HALF + log1mexp_val

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
            gp,
            target_space,
            n_random=n_random,
            n_smart=n_smart,
            fit_gp=fit_gp,
            random_state=random_state,
        )

    def get_acquisition_params(self):
        return {"xi": self.xi}

    def set_acquisition_params(self, **params):
        self.xi = params.get("xi", self.xi)


class LocalPenalization(acquisition.AcquisitionFunction):
    """Batch acquisition via local penalization (Gonzalez et al., 2016).

    Instead of hallucinating observations (Kriging Believer), LP penalizes
    the acquisition function near already-selected batch points, avoiding
    corruption of the GP posterior with fictitious data.

    For each pending point x_j, a soft exclusion zone suppresses nearby
    acquisition values (Proposition 1 / Appendix A of the paper):

        phi_j(x) = Phi(z_j),  z_j = (L * ||x - x_j|| - r_j) / sigma(x_j)

    where L is the Lipschitz constant of the GP mean, r_j = y_max - mu(x_j)
    is the exclusion radius, and sigma(x_j) is the GP posterior std at x_j.

    The penalized acquisition is:
        acq_LP(x) = acq(x) * prod_j phi_j(x)

    For log-space acquisitions (LogEI), the product becomes a sum of
    log-penalties: log_acq(x) + sum_j log(phi_j(x)).

    Implementation notes vs. paper:
    - Lipschitz constant L is estimated from kernel hyperparameters
      (sqrt(variance) / min(lengthscale)) rather than the GP-LCA gradient-norm
      maximization in Sec. 3.2. scikit-learn's GaussianProcessRegressor does
      not expose predictive_gradients(). The kernel heuristic is a conservative
      upper bound on the true Lipschitz constant.
    - The softplus transform g(z) = log(1 + exp(z)) from the paper is not
      applied for non-log acquisitions (UCB, EI). This is only relevant when
      the base acquisition returns exact zeros; LogEI avoids this issue.

    Reference
    ---------
    Gonzalez, J., Dai, Z., Hennig, P., & Lawrence, N.D.
    "Batch Bayesian Optimization via Local Penalization."
    AISTATS 2016. arXiv:1505.08052.
    """

    def __init__(self, base_acquisition: acquisition.AcquisitionFunction, lipschitz_scale=1.0, random_state=None):
        super().__init__(random_state)
        self.base_acquisition = base_acquisition
        self.lipschitz_scale = lipschitz_scale
        self.pending = []
        self._is_log_space = isinstance(base_acquisition, LogExpectedImprovement)

    def base_acq(self, mean, std):
        return self.base_acquisition.base_acq(mean, std)

    def clear_pending(self):
        self.pending = []

    def suggest(self, gp, target_space, n_random=10_000, n_smart=10, fit_gp=True, random_state=None):
        if len(target_space) == 0:
            raise ValueError("Cannot suggest a point without previous samples.")

        if fit_gp:
            self._fit_gp(gp, target_space)

        # Set y_max for improvement-based acquisitions
        if hasattr(self.base_acquisition, "y_max"):
            self.base_acquisition.y_max = target_space._target_max()  # type: ignore[assignment]

        self.i += 1
        acq = self._get_acq(gp=gp, constraint=target_space.constraint)
        x_max = self._acq_min(
            acq,
            target_space,
            n_random=n_random,
            n_smart=n_smart,
            random_state=np.random.RandomState(random_state) if isinstance(random_state, int) else random_state,  # type: ignore[arg-type]
        )
        self.pending.append(x_max.copy())
        return x_max

    def _get_acq(self, gp, constraint=None):
        dim = gp.X_train_.shape[1]
        pending = list(self.pending)
        is_log = self._is_log_space

        # Pre-compute Lipschitz constant and per-pending-point parameters
        L = self._estimate_lipschitz(gp)
        y_max = float(gp.y_train_.max())
        pending_params = []
        if pending:
            pending_arr = np.array(pending)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mu_pending, sigma_pending = gp.predict(
                    pending_arr.reshape(-1, dim),
                    return_std=True,
                )
            for j in range(len(pending)):
                r_j = max(y_max - mu_pending[j], 1e-8)
                s_j = max(sigma_pending[j], 1e-8)
                pending_params.append((pending[j], r_j, s_j))

        def acq(x):
            x = x.reshape(-1, dim)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mean, std = gp.predict(x, return_std=True)

            acq_val = self.base_acq(mean, std)

            for xp, r_j, s_j in pending_params:
                dist = np.sqrt(np.sum((x - xp.reshape(1, -1)) ** 2, axis=1))
                z_j = (L * dist - r_j) / s_j
                penalty = norm.cdf(z_j)
                if is_log:
                    acq_val = acq_val + np.log(np.maximum(penalty, 1e-10))
                else:
                    acq_val = acq_val * penalty

            result = -1 * acq_val
            if constraint is not None:
                p_constraints = constraint.predict(x)
                result = result * p_constraints
            return result

        return acq

    def _estimate_lipschitz(self, gp):
        """Estimate Lipschitz constant from GP kernel hyperparameters.

        Uses L = lipschitz_scale * sqrt(variance) / min(lengthscale) as
        a conservative upper bound on ||d mu/dx||. The paper's GP-LCA method
        (Gonzalez et al. 2016, Sec. 3.2) maximizes ||d mu/dx|| numerically,
        but this requires predictive_gradients() which scikit-learn does not
        expose. The kernel-based bound is always >= the true Lipschitz constant.
        """
        try:
            params = gp.kernel_.get_params()
            lengthscales = None
            variance = 1.0
            for key, val in params.items():
                if "length_scale" in key and not key.endswith("_bounds"):
                    lengthscales = np.atleast_1d(val)
                if "constant_value" in key and not key.endswith("_bounds"):
                    variance = float(val)
            if lengthscales is not None:
                return self.lipschitz_scale * np.sqrt(variance) / np.min(lengthscales)
        except Exception:
            pass
        return self.lipschitz_scale

    def get_acquisition_params(self):
        return {"lipschitz_scale": self.lipschitz_scale}

    def set_acquisition_params(self, **params):
        self.lipschitz_scale = params.get("lipschitz_scale", self.lipschitz_scale)


class ThompsonSamplingBatch(acquisition.AcquisitionFunction):
    """Batch acquisition via Thompson Sampling on the GP posterior.

    For each batch point, draws a sample from the GP posterior using
    ``gp.sample_y()``, optimizes it via random + local search, and adds
    the maximizer to the batch. Each sample uses a different random state,
    so the batch naturally explores diverse regions of input space.

    Unlike Kriging Believer, TS batch does not corrupt the GP with
    fictitious data. Unlike Local Penalization, it requires no Lipschitz
    estimation. The only tunable is the number of random restarts.

    References
    ----------
    Kandasamy, K., Krishnamurthy, A., Schneider, J., & Poczos, B.
    "Parallelised Bayesian Optimisation via Thompson Sampling."
    AISTATS 2018.
    """

    def __init__(self, random_state=None):
        super().__init__(random_state)
        self._sample_counter = 0
        self.pending = []

    def base_acq(self, mean, std):
        return mean

    def clear_pending(self):
        self.pending = []

    def suggest(
        self, gp, target_space: TargetSpace, n_random=10_000, n_smart=10, fit_gp: bool = True, random_state=None
    ) -> np.ndarray:
        if len(target_space) == 0:
            raise ValueError(
                "Cannot suggest a point without previous samples. "
                "Use target_space.random_sample() to generate a point."
            )

        if fit_gp:
            self._fit_gp(gp, target_space)

        sample_seed = self._sample_counter
        self._sample_counter += 1

        dim = gp.X_train_.shape[1]

        def ts_acq(x):
            x = np.atleast_2d(x).reshape(-1, dim)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                samples = _sample_f(gp, x, n_samples=1, random_state=sample_seed)
            return -samples.ravel()

        self.i += 1
        x_max = self._acq_min(
            ts_acq,
            target_space,
            n_random=n_random,
            n_smart=n_smart,
            random_state=np.random.RandomState(random_state) if isinstance(random_state, int) else random_state,
        )
        self.pending.append(x_max.copy())
        return x_max

    def get_acquisition_params(self):
        return {}

    def set_acquisition_params(self, **params):
        pass
