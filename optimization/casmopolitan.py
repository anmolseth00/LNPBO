"""CASMOPOLITAN mixed-variable Bayesian optimization for LNP formulations.

Implements the core ingredients of Wan et al. (2021) for mixed categorical +
continuous BO on a finite candidate pool:

1. Exponentiated categorical kernel (Eq. 1) with learnable per-dimension
   lengthscales.
2. Mixed categorical/continuous kernel (Eq. 4) with a learned additive-vs-
   product tradeoff.
3. TuRBO-style trust regions over the continuous coordinates together with
   Hamming trust regions over the categorical coordinates.
4. Restart when the trust region collapses, choosing the next center by a
   global GP-UCB rule over the candidate pool (Eq. 3).
5. Kriging Believer batching for ``b > 1`` on the discrete pool adaptation.

The LNP search space is mixed by construction: IL identity is categorical,
while molar ratios and molecular encodings are continuous. The implementation
here keeps that structure explicit instead of flattening everything into a
single Euclidean space.

Reference
---------
Wan, X., Nguyen, V., Ha, H., Ru, B., Lu, C., & Osborne, M.A. (2021).
    "Think Global and Act Local: Bayesian Optimisation over High-Dimensional
    Categorical and Mixed Search Spaces." ICML 2021. arXiv:2102.07188.
"""

import json
import logging
import time
import warnings
from pathlib import Path

logger = logging.getLogger("lnpbo")

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    Kernel,
    Matern,
    WhiteKernel,
)
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Exact CASMOPOLITAN kernels
# ---------------------------------------------------------------------------


class ExponentiatedCategoricalKernel(Kernel):
    """Exponentiated categorical kernel from Wan et al. (2021), Eq. (1).

    k_h(h, h') = exp((1 / d_h) * sum_i ell_i * 1[h_i = h'_i])

    ``ell_i`` are positive lengthscales. With ARD enabled, each categorical
    coordinate gets its own learned relevance parameter.
    """

    def __init__(self, n_cat_dims=1, lengthscales=None):
        self.n_cat_dims = n_cat_dims
        if lengthscales is None:
            lengthscales = tuple(np.ones(n_cat_dims, dtype=float).tolist())
        if np.asarray(lengthscales, dtype=float).shape != (n_cat_dims,):
            raise ValueError(f"lengthscales must have shape ({n_cat_dims},)")
        self.lengthscales = lengthscales

    @property
    def hyperparameter_lengthscales(self):
        """Sklearn Hyperparameter descriptor for ARD categorical lengthscales."""
        from sklearn.gaussian_process.kernels import Hyperparameter

        return Hyperparameter("lengthscales", "numeric", (1e-5, 1e3), self.n_cat_dims)

    def __call__(self, X, Y=None, eval_gradient=False):
        X = np.atleast_2d(X)
        if Y is None:
            Y = X
        elif eval_gradient:
            raise ValueError("Gradient can only be evaluated when Y is None.")

        X_cat = X[:, : self.n_cat_dims]
        Y_cat = Y[:, : self.n_cat_dims]
        match = (X_cat[:, np.newaxis, :] == Y_cat[np.newaxis, :, :]).astype(float)
        ls = np.asarray(self.lengthscales, dtype=float)
        weighted_match = match * ls[np.newaxis, np.newaxis, :]
        exponent = weighted_match.sum(axis=2) / max(self.n_cat_dims, 1)
        K = np.exp(exponent)

        if eval_gradient:
            dK = K[:, :, np.newaxis] * (
                match * ls[np.newaxis, np.newaxis, :] / max(self.n_cat_dims, 1)
            )
            return K, dK
        return K

    def diag(self, X):
        X = np.atleast_2d(X)
        exponent = np.sum(np.asarray(self.lengthscales, dtype=float)) / max(self.n_cat_dims, 1)
        return np.full(X.shape[0], np.exp(exponent))

    def is_stationary(self):
        return False

    def get_params(self, deep=True):
        return {"n_cat_dims": self.n_cat_dims, "lengthscales": self.lengthscales}

    def __repr__(self):
        return f"ExponentiatedCategoricalKernel(n_cat={self.n_cat_dims}, lengthscales={self.lengthscales})"

    def clone_with_theta(self, theta):
        clone = ExponentiatedCategoricalKernel(n_cat_dims=self.n_cat_dims, lengthscales=self.lengthscales)
        clone.theta = theta
        return clone

    @property
    def theta(self):
        return np.log(np.asarray(self.lengthscales, dtype=float))

    @theta.setter
    def theta(self, value):
        self.lengthscales = tuple(np.exp(np.asarray(value, dtype=float)).tolist())

    @property
    def bounds(self):
        return np.log(np.tile(np.array([[1e-5, 1e3]], dtype=float), (self.n_cat_dims, 1)))

    @property
    def n_dims(self):
        return self.n_cat_dims


# ---------------------------------------------------------------------------
# Mixed CASMOPOLITAN kernel (Eq. 4)
# ---------------------------------------------------------------------------


class MixedCasmopolitanKernel(Kernel):
    """Mixed kernel from Wan et al. (2021), Eq. (4).

    k(z, z') = lambda * (k_x(x, x') * k_h(h, h'))
             + (1 - lambda) * (k_h(h, h') + k_x(x, x'))

    where ``k_h`` is the exponentiated categorical kernel (Eq. 1) and
    ``k_x`` is a Matérn 5/2 kernel over the continuous coordinates.
    """

    def __init__(self, n_cat_dims, n_cont_dims, mix_weight=0.5, cat_lengthscales=None):
        self.n_cat_dims = n_cat_dims
        self.n_cont_dims = n_cont_dims
        if cat_lengthscales is None:
            cat_lengthscales = tuple(np.ones(n_cat_dims, dtype=float).tolist())
        if np.asarray(cat_lengthscales, dtype=float).shape != (n_cat_dims,):
            raise ValueError(f"cat_lengthscales must have shape ({n_cat_dims},)")
        self.cat_lengthscales = cat_lengthscales
        self.cat_kernel = ExponentiatedCategoricalKernel(n_cat_dims=n_cat_dims, lengthscales=self.cat_lengthscales)
        self.cont_kernel = ConstantKernel(1.0) * Matern(
            length_scale=np.ones(n_cont_dims),
            nu=2.5,
        )
        mix_weight_value = float(mix_weight)
        if not (0.0 < mix_weight_value < 1.0):
            raise ValueError("mix_weight must lie strictly between 0 and 1.")
        self.mix_weight = mix_weight

    def __call__(self, X, Y=None, eval_gradient=False):
        X = np.atleast_2d(X)
        X_cat = X[:, : self.n_cat_dims]
        X_cont = X[:, self.n_cat_dims :]

        if Y is None:
            Y_cat = None
            Y_cont = None
        else:
            Y = np.atleast_2d(Y)
            Y_cat = Y[:, : self.n_cat_dims]
            Y_cont = Y[:, self.n_cat_dims :]

        if not eval_gradient:
            K_cat = self.cat_kernel(X_cat, Y_cat)
            K_cont = self.cont_kernel(X_cont, Y_cont)
            return self.mix_weight * (K_cat * K_cont) + (1.0 - self.mix_weight) * (K_cat + K_cont)

        K_cat, dK_cat = self.cat_kernel(X_cat, Y_cat, eval_gradient=True)
        K_cont, dK_cont = self.cont_kernel(X_cont, Y_cont, eval_gradient=True)
        K_prod = K_cat * K_cont
        K_sum = K_cat + K_cont
        K = self.mix_weight * K_prod + (1.0 - self.mix_weight) * K_sum

        n_cat_params = dK_cat.shape[2]
        n_cont_params = dK_cont.shape[2]
        dK = np.zeros((K.shape[0], K.shape[1], 1 + n_cat_params + n_cont_params))

        lam = self.mix_weight
        dlam_dlogit = lam * (1.0 - lam)
        dK[:, :, 0] = (K_prod - K_sum) * dlam_dlogit

        cat_weight = (lam * K_cont + (1.0 - lam))[:, :, np.newaxis]
        dK[:, :, 1 : 1 + n_cat_params] = cat_weight * dK_cat

        cont_weight = (lam * K_cat + (1.0 - lam))[:, :, np.newaxis]
        dK[:, :, 1 + n_cat_params :] = cont_weight * dK_cont
        return K, dK

    def diag(self, X):
        X = np.atleast_2d(X)
        d_cat = self.cat_kernel.diag(X[:, : self.n_cat_dims])
        d_cont = self.cont_kernel.diag(X[:, self.n_cat_dims :])
        return self.mix_weight * (d_cat * d_cont) + (1.0 - self.mix_weight) * (d_cat + d_cont)

    def is_stationary(self):
        return False

    def get_params(self, deep=True):
        params = {
            "n_cat_dims": self.n_cat_dims,
            "n_cont_dims": self.n_cont_dims,
            "mix_weight": self.mix_weight,
            "cat_lengthscales": self.cat_lengthscales,
        }
        if deep:
            cat_params = self.cat_kernel.get_params(deep=True)
            params.update({f"cat_kernel__{k}": v for k, v in cat_params.items()})
            cont_params = self.cont_kernel.get_params(deep=True)
            params.update({f"cont_kernel__{k}": v for k, v in cont_params.items()})
        return params

    def __repr__(self):
        return f"MixedCasmopolitanKernel(n_cat={self.n_cat_dims}, n_cont={self.n_cont_dims}, mix={self.mix_weight:.3f})"

    @property
    def theta(self):
        mix_logit = np.log(self.mix_weight / (1.0 - self.mix_weight))
        return np.concatenate([[mix_logit], self.cat_kernel.theta, self.cont_kernel.theta])

    @theta.setter
    def theta(self, value):
        value = np.asarray(value, dtype=float)
        self.mix_weight = 1.0 / (1.0 + np.exp(-value[0]))
        n_cat = self.cat_kernel.n_dims
        self.cat_kernel.theta = value[1 : 1 + n_cat]
        self.cat_lengthscales = self.cat_kernel.lengthscales
        self.cont_kernel.theta = value[1 + n_cat :]

    @property
    def bounds(self):
        mix_bounds = np.array([[-6.0, 6.0]])
        return np.vstack([mix_bounds, self.cat_kernel.bounds, self.cont_kernel.bounds])

    @property
    def n_dims(self):
        return 1 + self.cat_kernel.n_dims + self.cont_kernel.n_dims


# ---------------------------------------------------------------------------
# Additive + product kernel (categorical + continuous + interaction)
# ---------------------------------------------------------------------------


class AdditiveProductKernel(Kernel):
    """Additive decomposition of categorical and continuous kernels with
    a product interaction term.

    k((c,x), (c',x')) = alpha * k_cat(c,c')
                       + beta  * k_cont(x,x')
                       + gamma * k_cat(c,c') * k_cont(x,x')

    where alpha, beta, gamma > 0 are learnable weights stored in log-space.
    The additive structure allows the GP to capture:
      - Pure categorical effects (alpha term): molecular identity alone
      - Pure continuous effects (beta term): molar ratios alone
      - Interaction effects (gamma term): identity x ratio synergy

    The GP marginal likelihood handles overall scale, so no sum-to-one
    constraint is imposed on the weights.

    Motivation:
      - Duvenaud, D. et al. (2011). "Additive Gaussian Processes." NeurIPS.
        Additive decompositions let the GP discover which input groups
        contribute independently vs. through interactions.
      - Wan, X. et al. (2021). "Think Global and Act Local: Bayesian
        Optimisation over High-Dimensional Categorical and Mixed Search
        Spaces." ICML 2021. arXiv:2102.07188.

    The intuition is that molecular identity and molar ratios may have
    both independent AND interaction effects on LNP efficacy.

    Parameters
    ----------
    n_cat_dims : int
        Number of categorical dimensions.
    n_cont_dims : int
        Number of continuous dimensions.
    lambd : float
        Initial lambda for the categorical overlap kernel.
    alpha : float
        Initial weight for the categorical-only term.
    beta : float
        Initial weight for the continuous-only term.
    gamma : float
        Initial weight for the interaction (product) term.
    """

    def __init__(self, n_cat_dims, n_cont_dims, lambd=0.5, alpha=1.0, beta=1.0, gamma=1.0):
        """Initialize the additive-product kernel.

        Args:
            n_cat_dims: Number of categorical dimensions.
            n_cont_dims: Number of continuous dimensions.
            lambd: Initial lambda for the categorical overlap kernel.
            alpha: Initial weight for the categorical-only term.
            beta: Initial weight for the continuous-only term.
            gamma: Initial weight for the interaction (product) term.
        """
        self.n_cat_dims = n_cat_dims
        self.n_cont_dims = n_cont_dims
        approx_lengthscale = np.full(n_cat_dims, -np.log(np.clip(lambd, 1e-5, 1.0 - 1e-5)))
        self.cat_kernel = ExponentiatedCategoricalKernel(n_cat_dims=n_cat_dims, lengthscales=approx_lengthscale)
        self.cont_kernel = ConstantKernel(1.0) * Matern(
            length_scale=np.ones(n_cont_dims),
            nu=2.5,
        )
        self.log_alpha = np.log(alpha)
        self.log_beta = np.log(beta)
        self.log_gamma = np.log(gamma)

    @property
    def _alpha(self):
        """Categorical-only weight (exp of log_alpha)."""
        return np.exp(self.log_alpha)

    @property
    def _beta(self):
        """Continuous-only weight (exp of log_beta)."""
        return np.exp(self.log_beta)

    @property
    def _gamma(self):
        """Interaction weight (exp of log_gamma)."""
        return np.exp(self.log_gamma)

    def __call__(self, X, Y=None, eval_gradient=False):
        """Compute the additive-product kernel.

        K = alpha * K_cat + beta * K_cont + gamma * K_cat * K_cont

        Gradients are computed w.r.t. log-space parameters:
        [log_alpha, log_beta, log_gamma, *cont_kernel.theta].

        Args:
            X: Array of shape (N, n_cat + n_cont).
            Y: Array of shape (M, n_cat + n_cont), or None for K(X, X).
            eval_gradient: If True, also return dK/d(log_theta).

        Returns:
            K: Kernel matrix of shape (N, M).
            dK: Gradient array of shape (N, M, n_params) if eval_gradient.
        """
        X = np.atleast_2d(X)
        X_cat = X[:, : self.n_cat_dims]
        X_cont = X[:, self.n_cat_dims :]

        if Y is None:
            Y_cat = None
            Y_cont = None
        else:
            Y = np.atleast_2d(Y)
            Y_cat = Y[:, : self.n_cat_dims]
            Y_cont = Y[:, self.n_cat_dims :]

        K_cat = self.cat_kernel(X_cat, Y_cat)

        if not eval_gradient:
            K_cont = self.cont_kernel(X_cont, Y_cont)
            return self._alpha * K_cat + self._beta * K_cont + self._gamma * K_cat * K_cont

        # With gradients: need dK/d(theta) for all learnable parameters.
        # theta layout: [log_alpha, log_beta, log_gamma, *cont_kernel.theta]
        # sklearn convention: gradients are dK/d(log_param), which for
        # exp-parameterized weights equals weight * dK/d(weight).
        K_cont, dK_cont = self.cont_kernel(X_cont, Y_cont, eval_gradient=True)

        K = self._alpha * K_cat + self._beta * K_cont + self._gamma * K_cat * K_cont

        n_cont_params = dK_cont.shape[2]
        n_params = 3 + n_cont_params
        n_x = K.shape[0]
        n_y = K.shape[1]
        dK = np.zeros((n_x, n_y, n_params))

        # d/d(log_alpha): alpha * K_cat  (chain rule: d(e^t * f)/dt = e^t * f)
        dK[:, :, 0] = self._alpha * K_cat
        # d/d(log_beta): beta * K_cont
        dK[:, :, 1] = self._beta * K_cont
        # d/d(log_gamma): gamma * K_cat * K_cont
        dK[:, :, 2] = self._gamma * K_cat * K_cont

        # d/d(theta_cont_i): beta * dK_cont_i + gamma * K_cat * dK_cont_i
        #                   = (beta + gamma * K_cat) * dK_cont_i
        weight = (self._beta + self._gamma * K_cat)[:, :, np.newaxis]
        dK[:, :, 3:] = weight * dK_cont

        return K, dK

    def diag(self, X):
        """Compute the diagonal of the additive-product kernel.

        Args:
            X: Array of shape (N, n_cat + n_cont).

        Returns:
            Diagonal array of shape (N,).
        """
        X = np.atleast_2d(X)
        d_cat = self.cat_kernel.diag(X[:, : self.n_cat_dims])
        d_cont = self.cont_kernel.diag(X[:, self.n_cat_dims :])
        return self._alpha * d_cat + self._beta * d_cont + self._gamma * d_cat * d_cont

    def is_stationary(self):
        """Return False (additive-product kernel is not stationary)."""
        return False

    def get_params(self, deep=True):
        """Get kernel parameters for sklearn compatibility.

        Args:
            deep: If True, include sub-kernel parameters.

        Returns:
            Dict of parameter name -> value.
        """
        params = {
            "n_cat_dims": self.n_cat_dims,
            "n_cont_dims": self.n_cont_dims,
            "lambd": float(np.exp(-np.mean(self.cat_kernel.lengthscales))),
            "alpha": self._alpha,
            "beta": self._beta,
            "gamma": self._gamma,
        }
        if deep:
            cont_params = self.cont_kernel.get_params(deep=True)
            params.update({f"cont_kernel__{k}": v for k, v in cont_params.items()})
        return params

    def __repr__(self):
        return (
            f"AdditiveProductKernel(n_cat={self.n_cat_dims}, "
            f"n_cont={self.n_cont_dims}, "
            f"alpha={self._alpha:.3f}, beta={self._beta:.3f}, "
            f"gamma={self._gamma:.3f}, lambd={float(np.exp(-np.mean(self.cat_kernel.lengthscales))):.3f})"
        )

    @property
    def theta(self):
        """Log-space hyperparameters: [log_alpha, log_beta, log_gamma, *cont_theta]."""
        return np.concatenate(
            [
                [self.log_alpha, self.log_beta, self.log_gamma],
                self.cont_kernel.theta,
            ]
        )

    @theta.setter
    def theta(self, value):
        """Set all hyperparameters from log-space vector."""
        self.log_alpha = value[0]
        self.log_beta = value[1]
        self.log_gamma = value[2]
        self.cont_kernel.theta = value[3:]

    @property
    def bounds(self):
        """Log-space bounds: [[-5, 5]] x 3 for weights + cont kernel bounds."""
        # Weights: allow range [e^-5, e^5] ~ [0.007, 148]
        weight_bounds = np.array([[-5.0, 5.0]] * 3)
        return np.vstack([weight_bounds, self.cont_kernel.bounds])

    @property
    def n_dims(self):
        """Number of hyperparameters (3 weights + continuous kernel dims)."""
        return 3 + self.cont_kernel.n_dims


# ---------------------------------------------------------------------------
# Trust region
# ---------------------------------------------------------------------------


class TrustRegion:
    """Trust region for mixed categorical + continuous search spaces.

    Manages a local search region around the current best point, with
    separate mechanisms for categorical and continuous dimensions:
    - Continuous: hypercube of radius ``length`` around center (clipped to bounds)
    - Categorical: allow up to ``n_cat_perturb`` category changes from center

    The trust region expands on improvement and shrinks on failure, following
    the TuRBO-style length adaptation (Eriksson et al., NeurIPS 2019).

    Reference: Wan et al. (2021), Sec. 3.2. arXiv:2102.07188.
    """

    def __init__(
        self,
        center_cat: np.ndarray,
        center_cont: np.ndarray,
        length: float,
        n_cat_dims: int,
        n_cont_dims: int,
        cont_bounds: np.ndarray | None = None,
        length_min: float = 0.01,
        length_max: float = 2.0,
        success_tol: int = 3,
        failure_tol: int = 5,
    ):
        """Initialize a trust region for mixed categorical + continuous spaces.

        Args:
            center_cat: Categorical center values, shape (n_cat_dims,).
            center_cont: Continuous center values, shape (n_cont_dims,).
            length: Initial trust region half-length for continuous dims.
            n_cat_dims: Number of categorical dimensions.
            n_cont_dims: Number of continuous dimensions.
            cont_bounds: Global bounds of shape (n_cont_dims, 2) for
                clipping the trust region. None = unbounded.
            length_min: Minimum trust region half-length.
            length_max: Maximum trust region half-length.
            success_tol: Number of consecutive improvements before expanding.
            failure_tol: Number of consecutive failures before shrinking.

        Reference:
            Wan et al. (2021), Sec. 3.2. arXiv:2102.07188.
            Eriksson et al. (2019). "Scalable Global Optimization via
            Local Bayesian Optimization." NeurIPS (TuRBO).
        """
        self.center_cat = center_cat.copy()
        self.center_cont = center_cont.copy()
        self.length = length
        self.n_cat_dims = n_cat_dims
        self.n_cont_dims = n_cont_dims
        self.cont_bounds = cont_bounds
        self.length_min = length_min
        self.length_max = length_max
        self.success_tol = success_tol
        self.failure_tol = failure_tol
        self.n_cat_perturb = max(1, int(0.2 * n_cat_dims))
        self._successes = 0
        self._failures = 0

    def get_cont_bounds(self) -> np.ndarray:
        """Get continuous bounds clipped to the trust region."""
        lb = self.center_cont - self.length
        ub = self.center_cont + self.length
        if self.cont_bounds is not None:
            lb = np.maximum(lb, self.cont_bounds[:, 0])
            ub = np.minimum(ub, self.cont_bounds[:, 1])
        return np.column_stack([lb, ub])

    def update(self, improved: bool) -> bool:
        """Update trust region based on whether the batch improved.

        Returns ``True`` when the trust region has collapsed to its minimum
        length and should be restarted according to Wan et al. (2021).
        """
        if improved:
            self._successes += 1
            self._failures = 0
            if self._successes >= self.success_tol:
                self.length = min(self.length * 2.0, self.length_max)
                self._successes = 0
            return False

        self._failures += 1
        self._successes = 0
        if self._failures < self.failure_tol:
            return False

        self.length = max(self.length / 2.0, self.length_min)
        self._failures = 0
        return bool(self.length <= self.length_min + 1e-12)

    def set_center(self, center_cat: np.ndarray, center_cont: np.ndarray) -> None:
        """Update trust region center to new best point."""
        self.center_cat = center_cat.copy()
        self.center_cont = center_cont.copy()

    def contains_cat(self, cat_vals: np.ndarray) -> bool:
        """Check if a categorical config is within the trust region."""
        n_diff = np.sum(cat_vals != self.center_cat)
        return n_diff <= self.n_cat_perturb

    def __repr__(self):
        return (
            f"TrustRegion(length={self.length:.4f}, "
            f"n_cat_perturb={self.n_cat_perturb}, "
            f"successes={self._successes}, failures={self._failures})"
        )


# ---------------------------------------------------------------------------
# Mixed acquisition optimization
# ---------------------------------------------------------------------------


def _ucb_acquisition(mu: np.ndarray, sigma: np.ndarray, kappa: float = 5.0) -> np.ndarray:
    """Compute Upper Confidence Bound acquisition values.

    UCB(x) = mu(x) + kappa * sigma(x).

    Args:
        mu: Posterior mean array of shape (N,).
        sigma: Posterior std array of shape (N,).
        kappa: Exploration weight (default 5.0).

    Returns:
        UCB values of shape (N,).
    """
    return mu + kappa * sigma


def _ei_acquisition(mu: np.ndarray, sigma: np.ndarray, y_best: float) -> np.ndarray:
    """Compute Expected Improvement acquisition values.

    EI(x) = (mu(x) - y_best) * Phi(z) + sigma(x) * phi(z),
    where z = (mu(x) - y_best) / sigma(x).

    Args:
        mu: Posterior mean array of shape (N,).
        sigma: Posterior std array of shape (N,).
        y_best: Best observed value (incumbent).

    Returns:
        EI values of shape (N,). Zero where sigma < 1e-10.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (mu - y_best) / sigma
        ei = (mu - y_best) * norm.cdf(z) + sigma * norm.pdf(z)
        ei = np.where(sigma > 1e-10, ei, 0.0)
    return ei


def optimize_mixed_acquisition(
    gp: GaussianProcessRegressor,
    unique_cats: np.ndarray,
    trust_region: TrustRegion,
    n_cat_dims: int,
    acq_func: str = "ucb",
    kappa: float = 5.0,
    n_cont_restarts: int = 10,
    n_cat_samples: int = 50,
    rng: np.random.RandomState | None = None,
) -> np.ndarray:
    """Optimize acquisition over mixed categorical + continuous space.

    Strategy (Wan et al. 2021, Sec. 3.3):
    1. For each categorical configuration (enumerate if small, sample otherwise):
       a. Fix categorical dims
       b. Optimize continuous dims via L-BFGS-B within trust region bounds
    2. Return the point with the highest acquisition value

    Parameters
    ----------
    gp : fitted GaussianProcessRegressor
    unique_cats : array of shape (n_unique, n_cat_dims)
        All unique categorical configurations in the pool.
    trust_region : TrustRegion
    n_cat_dims : int
    acq_func : str, "ucb" or "ei"
    kappa : float, UCB exploration weight
    n_cont_restarts : int, number of random restarts for L-BFGS-B
    n_cat_samples : int, max categorical configs to evaluate
    rng : RandomState

    Returns
    -------
    x_best : array of shape (n_cat_dims + n_cont_dims,)
    """
    if rng is None:
        rng = np.random.RandomState()

    tr_cont_bounds = trust_region.get_cont_bounds()

    # Filter categorical configs to those within trust region
    if unique_cats.shape[0] <= n_cat_samples:
        cat_candidates = unique_cats
    else:
        # Sample within trust region: prefer configs close to center
        in_tr = np.array([trust_region.contains_cat(c) for c in unique_cats])
        in_tr_cats = unique_cats[in_tr]
        if len(in_tr_cats) > n_cat_samples:
            idx = rng.choice(len(in_tr_cats), size=n_cat_samples, replace=False)
            cat_candidates = in_tr_cats[idx]
        elif len(in_tr_cats) > 0:
            cat_candidates = in_tr_cats
        else:
            # Trust region is too small; fall back to random sample
            idx = rng.choice(len(unique_cats), size=min(n_cat_samples, len(unique_cats)), replace=False)
            cat_candidates = unique_cats[idx]

    # Compute y_best for EI
    y_best = float(gp.y_train_.max()) if acq_func == "ei" else 0.0

    best_acq_val = -np.inf
    best_x = None

    for cat in cat_candidates:
        # Optimize continuous dims for this categorical config
        _cat = cat  # bind loop variable for closure

        def neg_acq(cont_x, _cat=_cat):
            """Negated acquisition for L-BFGS-B minimization."""
            x = np.concatenate([_cat, cont_x]).reshape(1, -1)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mu, sigma = gp.predict(x, return_std=True)
            if acq_func == "ucb":
                return -_ucb_acquisition(mu, sigma, kappa)[0]
            else:
                return -_ei_acquisition(mu, sigma, y_best)[0]

        # Multiple random restarts
        for _ in range(n_cont_restarts):
            x0 = rng.uniform(tr_cont_bounds[:, 0], tr_cont_bounds[:, 1])
            try:
                result = minimize(
                    neg_acq,
                    x0,
                    bounds=list(zip(tr_cont_bounds[:, 0], tr_cont_bounds[:, 1])),
                    method="L-BFGS-B",
                )
                if -result.fun > best_acq_val:
                    best_acq_val = -result.fun
                    best_x = np.concatenate([cat, result.x])
            except (ValueError, RuntimeError, np.linalg.LinAlgError):
                continue

        # Also evaluate at the trust region center's continuous values
        x_center = np.concatenate([cat, trust_region.center_cont]).reshape(1, -1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mu_c, sigma_c = gp.predict(x_center, return_std=True)
        if acq_func == "ucb":
            acq_c = _ucb_acquisition(mu_c, sigma_c, kappa)[0]
        else:
            acq_c = _ei_acquisition(mu_c, sigma_c, y_best)[0]
        if acq_c > best_acq_val:
            best_acq_val = acq_c
            best_x = x_center.ravel()

    if best_x is None:
        # Fallback: return trust region center
        best_x = np.concatenate([trust_region.center_cat, trust_region.center_cont])

    return best_x


def select_batch_casmopolitan(
    gp: GaussianProcessRegressor,
    unique_cats: np.ndarray,
    trust_region: TrustRegion,
    n_cat_dims: int,
    batch_size: int = 12,
    acq_func: str = "ucb",
    kappa: float = 5.0,
    n_cont_restarts: int = 10,
    n_cat_samples: int = 50,
    rng: np.random.RandomState | None = None,
) -> np.ndarray:
    """Select a batch of points using Kriging Believer + CASMOPOLITAN acquisition.

    For each point in the batch:
    1. Optimize mixed acquisition function
    2. Hallucinate the selected point with the GP posterior mean (KB)
    3. Add the hallucinated point to the GP training set
    4. Repeat

    Parameters
    ----------
    gp : fitted GaussianProcessRegressor
    unique_cats : array of shape (n_unique, n_cat_dims)
    trust_region : TrustRegion
    n_cat_dims : int
    batch_size : int
    acq_func : str
    kappa : float
    n_cont_restarts : int
    n_cat_samples : int
    rng : RandomState

    Returns
    -------
    batch : array of shape (batch_size, n_cat_dims + n_cont_dims)
    """
    if rng is None:
        rng = np.random.RandomState()

    batch = []
    current_gp = gp

    for _b in range(batch_size):
        x_new = optimize_mixed_acquisition(
            current_gp,
            unique_cats,
            trust_region,
            n_cat_dims,
            acq_func=acq_func,
            kappa=kappa,
            n_cont_restarts=n_cont_restarts,
            n_cat_samples=n_cat_samples,
            rng=rng,
        )
        batch.append(x_new)

        # Kriging Believer: hallucinate with GP posterior mean
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y_halluc = current_gp.predict(x_new.reshape(1, -1))[0]

        # Create augmented GP (refit with hallucinated point)
        X_aug = np.vstack([current_gp.X_train_, x_new.reshape(1, -1)])
        y_aug = np.concatenate([current_gp.y_train_, [y_halluc]])

        new_gp = GaussianProcessRegressor(
            kernel=current_gp.kernel_,
            alpha=current_gp.alpha,
            n_restarts_optimizer=0,
            random_state=rng.randint(10000),
            optimizer=None,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            new_gp.fit(X_aug, y_aug)
        current_gp = new_gp

    return np.array(batch)


# ---------------------------------------------------------------------------
# Discrete-pool helpers for CASMOPOLITAN
# ---------------------------------------------------------------------------


def _assemble_mixed_blocks(cat_block: np.ndarray, cont_block: np.ndarray) -> np.ndarray:
    """Concatenate categorical and continuous blocks, handling empty sides."""
    if cat_block.size and cont_block.size:
        return np.column_stack([cat_block, cont_block])
    if cat_block.size:
        return cat_block.copy()
    return cont_block.copy()


def _append_restart_observation(
    archive_X_raw: np.ndarray | None,
    archive_y: np.ndarray | None,
    incumbent_raw: np.ndarray,
    incumbent_y: float,
    X_train_raw: np.ndarray,
    y_train: np.ndarray,
    rng: np.random.RandomState,
) -> tuple[np.ndarray, np.ndarray]:
    """Append a restart center observation following Wan et al. (2021), Eq. (3).

    The paper restarts from local maxima found in previous trust regions and
    falls back to a random data point when the local maximum duplicates an
    earlier restart center. We store the archive in raw mixed coordinates so it
    can be re-embedded under the current round's continuous scaling.
    """
    archive_X = (
        np.empty((0, X_train_raw.shape[1]), dtype=float)
        if archive_X_raw is None
        else np.asarray(archive_X_raw, dtype=float).reshape(-1, X_train_raw.shape[1])
    )
    archive_targets = np.empty((0,), dtype=float) if archive_y is None else np.asarray(archive_y, dtype=float).ravel()

    candidate_x = np.asarray(incumbent_raw, dtype=float).ravel()
    candidate_y = float(incumbent_y)
    duplicate = len(archive_X) > 0 and np.any(np.all(np.isclose(archive_X, candidate_x, atol=1e-12), axis=1))
    if duplicate:
        order = rng.permutation(len(X_train_raw))
        for idx in order:
            alt_x = np.asarray(X_train_raw[idx], dtype=float).ravel()
            if len(archive_X) == 0 or not np.any(np.all(np.isclose(archive_X, alt_x, atol=1e-12), axis=1)):
                candidate_x = alt_x
                candidate_y = float(y_train[idx])
                break

    archive_X = np.vstack([archive_X, candidate_x.reshape(1, -1)])
    archive_targets = np.concatenate([archive_targets, [candidate_y]])
    return archive_X, archive_targets


def _build_casmopolitan_gp(
    X_train_mixed: np.ndarray,
    y_train: np.ndarray,
    n_cat: int,
    n_cont: int,
    random_seed: int,
) -> GaussianProcessRegressor:
    """Fit the exact Wan et al. mixed kernel with learned tradeoff."""
    kernel = MixedCasmopolitanKernel(n_cat_dims=n_cat, n_cont_dims=n_cont, mix_weight=0.5)
    kernel = kernel + WhiteKernel(noise_level=0.1)

    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-6,
        n_restarts_optimizer=5,
        random_state=random_seed,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gp.fit(X_train_mixed, y_train)
    return gp


def _restart_center_from_archive(
    archive_X_raw: np.ndarray,
    archive_y: np.ndarray,
    X_pool_cat: np.ndarray,
    X_pool_cont: np.ndarray,
    cont_scaler: StandardScaler,
    cont_feature_indices: list[int],
    cat_feature_indices: list[int],
    random_seed: int,
    restart_kappa: float,
) -> np.ndarray:
    """Choose the next trust-region center by GP-UCB over archived restart maxima."""
    if len(archive_X_raw) == 0:
        raise ValueError("Restart archive must be non-empty.")

    n_cat = len(cat_feature_indices)
    n_cont = len(cont_feature_indices)
    archive_cat = archive_X_raw[:, cat_feature_indices] if n_cat else np.zeros((len(archive_X_raw), 0))
    if n_cont:
        archive_cont = cont_scaler.transform(archive_X_raw[:, cont_feature_indices])
    else:
        archive_cont = np.zeros((len(archive_X_raw), 0))
    archive_mixed = _assemble_mixed_blocks(archive_cat, archive_cont)
    pool_mixed = _assemble_mixed_blocks(X_pool_cat, X_pool_cont)

    restart_gp = _build_casmopolitan_gp(archive_mixed, archive_y, n_cat=n_cat, n_cont=n_cont, random_seed=random_seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mu, sigma = restart_gp.predict(pool_mixed, return_std=True)
    scores = _ucb_acquisition(mu, sigma, kappa=restart_kappa)
    return pool_mixed[int(np.argmax(scores))]


def _fit_pool_casmopolitan_gp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_pool: np.ndarray,
    cont_feature_indices: list[int],
    cat_feature_indices: list[int],
    random_seed: int,
    trust_length: float,
    trust_region: TrustRegion | None = None,
    restart_from_archive: bool = False,
    restart_X_raw: np.ndarray | None = None,
    restart_y: np.ndarray | None = None,
    restart_kappa: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, GaussianProcessRegressor, TrustRegion]:
    """Fit the mixed GP on a discrete pool and initialize/update the trust region."""
    n_cat = len(cat_feature_indices)
    n_cont = len(cont_feature_indices)

    if n_cont:
        cont_scaler = StandardScaler()
        X_train_cont = cont_scaler.fit_transform(X_train[:, cont_feature_indices])
        X_pool_cont = cont_scaler.transform(X_pool[:, cont_feature_indices])
        # Keep the incumbent inside the clipped region even when the residual
        # pool no longer spans its coordinate. Otherwise lb/ub can invert and
        # the optimizer falls back to snapping the center onto an out-of-TR row.
        cont_all = np.vstack([X_train_cont, X_pool_cont])
        cont_bounds = np.column_stack([cont_all.min(axis=0), cont_all.max(axis=0)])
    else:
        X_train_cont = np.zeros((len(X_train), 0))
        X_pool_cont = np.zeros((len(X_pool), 0))
        cont_bounds = np.zeros((0, 2))

    X_train_cat = X_train[:, cat_feature_indices] if n_cat else np.zeros((len(X_train), 0))
    X_pool_cat = X_pool[:, cat_feature_indices] if n_cat else np.zeros((len(X_pool), 0))
    X_train_mixed = _assemble_mixed_blocks(X_train_cat, X_train_cont)
    X_pool_mixed = _assemble_mixed_blocks(X_pool_cat, X_pool_cont)

    gp = _build_casmopolitan_gp(X_train_mixed, y_train, n_cat=n_cat, n_cont=n_cont, random_seed=random_seed)

    best_train_idx = int(np.argmax(y_train))
    best_cat = X_train_mixed[best_train_idx, :n_cat]
    best_cont = X_train_mixed[best_train_idx, n_cat:]

    if restart_from_archive:
        if restart_X_raw is None or restart_y is None or len(restart_X_raw) == 0:
            raise ValueError("restart_from_archive=True requires a non-empty restart archive.")
        restart_center = _restart_center_from_archive(
            np.asarray(restart_X_raw, dtype=float),
            np.asarray(restart_y, dtype=float),
            X_pool_cat,
            X_pool_cont,
            cont_scaler,
            cont_feature_indices,
            cat_feature_indices,
            random_seed=random_seed,
            restart_kappa=restart_kappa,
        )
        best_cat = restart_center[:n_cat]
        best_cont = restart_center[n_cat:]

    if trust_region is None or restart_from_archive:
        trust_region = TrustRegion(
            center_cat=best_cat,
            center_cont=best_cont,
            length=trust_length,
            n_cat_dims=n_cat,
            n_cont_dims=n_cont,
            cont_bounds=cont_bounds,
        )
    else:
        trust_region.cont_bounds = cont_bounds
        trust_region.set_center(best_cat, best_cont)

    return X_train_mixed, X_pool_mixed, gp, trust_region


def _trust_region_pool_mask(
    X_pool_mixed: np.ndarray,
    trust_region: TrustRegion,
    n_cat_dims: int,
) -> np.ndarray:
    """Return a boolean mask for pool rows that are inside the current trust region."""
    in_tr = np.ones(len(X_pool_mixed), dtype=bool)

    if n_cat_dims:
        cat_block = X_pool_mixed[:, :n_cat_dims]
        cat_mismatch = np.sum(cat_block != trust_region.center_cat, axis=1)
        in_tr &= cat_mismatch <= trust_region.n_cat_perturb

    if trust_region.n_cont_dims:
        tr_cont_bounds = trust_region.get_cont_bounds()
        cont_pool_vals = X_pool_mixed[:, n_cat_dims:]
        in_tr &= np.all(
            (cont_pool_vals >= tr_cont_bounds[:, 0]) & (cont_pool_vals <= tr_cont_bounds[:, 1]),
            axis=1,
        )

    return in_tr


def _apply_trust_region_penalty(
    scores: np.ndarray,
    X_pool_mixed: np.ndarray,
    trust_region: TrustRegion,
    n_cat_dims: int,
) -> np.ndarray:
    """Penalize candidates outside the current trust region.

    When the pool contains in-region candidates, treat the trust region as a
    hard feasibility filter and mask everything else. If no pool point falls
    inside the current region, fall back to an additive distance penalty so we
    still return a ranking while making farther violations strictly less
    attractive regardless of the sign of ``scores``.
    """
    in_tr = _trust_region_pool_mask(X_pool_mixed, trust_region, n_cat_dims)
    violation = np.zeros(len(X_pool_mixed), dtype=float)

    if n_cat_dims:
        cat_block = X_pool_mixed[:, :n_cat_dims]
        cat_mismatch = np.sum(cat_block != trust_region.center_cat, axis=1)
        violation += np.clip(cat_mismatch - trust_region.n_cat_perturb, a_min=0.0, a_max=None)

    if trust_region.n_cont_dims:
        tr_cont_bounds = trust_region.get_cont_bounds()
        cont_pool_vals = X_pool_mixed[:, n_cat_dims:]
        lower_gap = np.clip(tr_cont_bounds[:, 0] - cont_pool_vals, a_min=0.0, a_max=None)
        upper_gap = np.clip(cont_pool_vals - tr_cont_bounds[:, 1], a_min=0.0, a_max=None)
        cont_violation = np.linalg.norm(lower_gap + upper_gap, axis=1)
        violation += cont_violation

    penalized = scores.copy()
    if in_tr.any():
        penalized[~in_tr] = -np.inf
        return penalized

    penalty_scale = max(float(np.max(np.abs(scores))), float(np.ptp(scores)), 1.0)
    penalized -= penalty_scale * (1.0 + violation)
    return penalized


def _map_candidates_to_pool(
    candidates: np.ndarray,
    X_pool_mixed: np.ndarray,
    n_cat_dims: int,
    trust_region: TrustRegion | None = None,
) -> np.ndarray:
    """Map optimized mixed candidates back to the nearest available pool rows."""
    candidates = np.atleast_2d(candidates)
    available_mask = np.ones(len(X_pool_mixed), dtype=bool)
    selected = []

    for candidate in candidates:
        available_idx = np.where(available_mask)[0]
        pool_avail = X_pool_mixed[available_mask]

        if trust_region is not None:
            in_tr_avail = _trust_region_pool_mask(pool_avail, trust_region, n_cat_dims)
            if in_tr_avail.any():
                pool_avail = pool_avail[in_tr_avail]
                available_idx = available_idx[in_tr_avail]

        if n_cat_dims:
            cat_match = (pool_avail[:, :n_cat_dims] == candidate[:n_cat_dims]).all(axis=1)
            if cat_match.any():
                pool_avail = pool_avail[cat_match]
                available_idx = available_idx[cat_match]

        if pool_avail.shape[1] > n_cat_dims:
            dists = np.sum((pool_avail[:, n_cat_dims:] - candidate[n_cat_dims:]) ** 2, axis=1)
        else:
            dists = np.zeros(len(pool_avail))

        chosen = int(available_idx[int(np.argmin(dists))])
        selected.append(chosen)
        available_mask[chosen] = False

    return np.array(selected, dtype=int)


def score_pool_casmopolitan(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_pool: np.ndarray,
    il_cat_train: np.ndarray,
    il_cat_pool: np.ndarray,
    cont_feature_indices: list[int],
    cat_feature_indices: list[int],
    batch_size: int = 12,
    kappa: float = 5.0,
    random_seed: int = 42,
    trust_length: float = 0.5,
    acq_func: str = "ucb",
) -> tuple[np.ndarray, np.ndarray]:
    """Score a discrete pool using CASMOPOLITAN mixed-variable GP.

    This is the discrete-pool variant: instead of continuous optimization,
    we fit the mixed kernel GP, compute acquisition values for all pool
    candidates, and select the top-K. The trust region filters candidates.

    For LNP formulations:
    - Categorical dims: integer-encoded IL identity
    - Continuous dims: ILR-transformed molar ratios (or raw ratios, scaled)

    Parameters
    ----------
    X_train : array of shape (n_train, n_features)
        Training features (categorical + continuous, pre-assembled).
    y_train : array of shape (n_train,)
    X_pool : array of shape (n_pool, n_features)
    il_cat_train : array of shape (n_train,)
        Integer-encoded IL category for each training point.
    il_cat_pool : array of shape (n_pool,)
        Integer-encoded IL category for each pool point.
    cont_feature_indices : list of int
        Column indices in X_train/X_pool for continuous features.
    cat_feature_indices : list of int
        Column indices in X_train/X_pool for categorical features.
    batch_size : int
    kappa : float
    random_seed : int
    trust_length : float
        Initial trust region length for continuous dims.
    acq_func : str, "ucb" or "ei"

    Returns
    -------
    top_indices : array of shape (batch_size,)
    scores : array of shape (n_pool,)
    """
    n_cat = len(cat_feature_indices)
    _, X_pool_mixed, gp, trust_region = _fit_pool_casmopolitan_gp(
        X_train,
        y_train,
        X_pool,
        cont_feature_indices,
        cat_feature_indices,
        random_seed,
        trust_length,
    )

    # Score all pool candidates
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mu, sigma = gp.predict(X_pool_mixed, return_std=True)

    y_best = float(y_train.max())
    if acq_func == "ucb":
        scores = _ucb_acquisition(mu, sigma, kappa)
    else:
        scores = _ei_acquisition(mu, sigma, y_best)

    penalized_scores = _apply_trust_region_penalty(scores, X_pool_mixed, trust_region, n_cat)
    top_indices = np.argsort(penalized_scores)[-batch_size:][::-1]
    return top_indices, penalized_scores


def select_pool_batch_casmopolitan(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_pool: np.ndarray,
    il_cat_train: np.ndarray,
    il_cat_pool: np.ndarray,
    cont_feature_indices: list[int],
    cat_feature_indices: list[int],
    batch_size: int = 12,
    kappa: float = 5.0,
    random_seed: int = 42,
    trust_length: float = 0.5,
    acq_func: str = "ucb",
    trust_region: TrustRegion | None = None,
    restart_from_archive: bool = False,
    restart_X_raw: np.ndarray | None = None,
    restart_y: np.ndarray | None = None,
    restart_kappa: float = 2.0,
) -> tuple[np.ndarray, TrustRegion]:
    """Select a discrete-pool batch using the CASMOPOLITAN trust-region KB loop.

    This finite-pool adaptation keeps the paper's mixed kernel, trust-region,
    restart, and Kriging Believer mechanisms, then maps each optimized mixed
    candidate back onto an available pool formulation.
    """
    del il_cat_train, il_cat_pool  # categories are already encoded inside X_train / X_pool

    n_cat = len(cat_feature_indices)
    _, X_pool_mixed, gp, trust_region = _fit_pool_casmopolitan_gp(
        X_train,
        y_train,
        X_pool,
        cont_feature_indices,
        cat_feature_indices,
        random_seed,
        trust_length,
        trust_region=trust_region,
        restart_from_archive=restart_from_archive,
        restart_X_raw=restart_X_raw,
        restart_y=restart_y,
        restart_kappa=restart_kappa,
    )

    unique_cats = np.unique(X_pool_mixed[:, :n_cat], axis=0) if n_cat else np.zeros((1, 0))
    candidates = select_batch_casmopolitan(
        gp,
        unique_cats,
        trust_region,
        n_cat_dims=n_cat,
        batch_size=batch_size,
        acq_func=acq_func,
        kappa=kappa,
        rng=np.random.RandomState(random_seed),
    )
    selected = _map_candidates_to_pool(candidates, X_pool_mixed, n_cat, trust_region=trust_region)
    return selected, trust_region


# ---------------------------------------------------------------------------
# Full CASMOPOLITAN BO loop (for benchmark integration)
# ---------------------------------------------------------------------------


def run_casmopolitan_strategy(
    encoded_df,
    feature_cols,
    seed_idx,
    oracle_idx,
    batch_size=12,
    n_rounds=15,
    seed=42,
    kappa=5.0,
    normalize="copula",
    trust_length_init=0.5,
    acq_func="ucb",
    use_ilr=True,
    max_train_for_gp=2000,
    top_k_values=None,
):
    """Run CASMOPOLITAN mixed-variable BO loop as a benchmark strategy.

    Integrates with the benchmark runner infrastructure. At each round:
    1. Encode IL identity as integer categories
    2. Optionally ILR-transform molar ratios
    3. Fit the Wan et al. mixed kernel (Eq. 4)
    4. Select a sequential KB batch inside the trust region
    5. Restart via GP-UCB when the trust region collapses

    Parameters
    ----------
    encoded_df : pd.DataFrame
        Full encoded dataset with feature columns and Experiment_value.
    feature_cols : list of str
        All feature column names.
    seed_idx : list of int
        Initial seed indices.
    oracle_idx : list of int
        Oracle pool indices.
    batch_size : int
    n_rounds : int
    seed : int
    kappa : float
    normalize : str, "copula", "zscore", or "none"
    trust_length_init : float
    acq_func : str, "ucb" or "ei"
    use_ilr : bool
        If True, apply ILR transform to molar ratio features.
    max_train_for_gp : int
        Deprecated compatibility argument. The exact CASMOPOLITAN path uses the
        full training set and the paper's mixed kernel.

    Returns
    -------
    history : dict
        Compatible with benchmarks.runner.compute_metrics().
    """
    from LNPBO.benchmarks.runner import init_history, update_history
    from LNPBO.data.compositional import ilr_transform
    from LNPBO.optimization._normalize import copula_transform

    del max_train_for_gp

    # Identify feature structure
    ratio_cols = [c for c in feature_cols if c.endswith("_molratio")]
    mass_ratio_cols = [c for c in feature_cols if c == "IL_to_nucleicacid_massratio"]
    enc_cols = [c for c in feature_cols if c not in ratio_cols and c not in mass_ratio_cols]

    # IL name is the categorical variable (integer-encoded)
    il_names = encoded_df["IL_name"].values
    unique_il_names = np.unique(il_names)
    il_name_to_int = {name: i for i, name in enumerate(unique_il_names)}
    il_cat_all = np.array([il_name_to_int[n] for n in il_names])

    # Continuous features: molecular encodings + ratios (+ optional ILR)
    ratio_indices = [feature_cols.index(c) for c in ratio_cols]
    mass_ratio_indices = [feature_cols.index(c) for c in mass_ratio_cols]
    enc_indices = [feature_cols.index(c) for c in enc_cols]

    training_idx = list(seed_idx)
    pool_idx = list(oracle_idx)
    history = init_history(encoded_df, training_idx, top_k_values=top_k_values)

    # Trust region and restart archive
    trust_region = None
    round_start_best = None
    restart_X_raw = None
    restart_y = None

    for r in range(n_rounds):
        if len(pool_idx) < batch_size:
            break

        # Extract features
        X_all = encoded_df[feature_cols].values
        y_all = encoded_df["Experiment_value"].values

        X_train_raw = X_all[training_idx]
        y_train = y_all[training_idx].copy()
        X_pool_raw = X_all[pool_idx]

        # Normalize targets
        if normalize == "copula":
            y_train = copula_transform(y_train)
        elif normalize == "zscore":
            mu_y, sigma_y = y_train.mean(), y_train.std()
            if sigma_y > 0:
                y_train = (y_train - mu_y) / sigma_y

        # Build mixed feature matrix: [il_cat_int | continuous_features]
        # Continuous = molecular PCs + ILR(ratios) + mass_ratio
        cont_train_parts = [X_train_raw[:, enc_indices]]
        cont_pool_parts = [X_pool_raw[:, enc_indices]]

        if ratio_cols:
            if use_ilr:
                ratios_train = X_train_raw[:, ratio_indices]
                ratios_pool = X_pool_raw[:, ratio_indices]
                ilr_train = ilr_transform(ratios_train)
                ilr_pool = ilr_transform(ratios_pool)
                cont_train_parts.append(ilr_train)
                cont_pool_parts.append(ilr_pool)
            else:
                cont_train_parts.append(X_train_raw[:, ratio_indices])
                cont_pool_parts.append(X_pool_raw[:, ratio_indices])

        if mass_ratio_indices:
            cont_train_parts.append(X_train_raw[:, mass_ratio_indices])
            cont_pool_parts.append(X_pool_raw[:, mass_ratio_indices])

        cont_train = np.hstack(cont_train_parts)
        cont_pool = np.hstack(cont_pool_parts)
        n_cont = cont_train.shape[1]

        cat_train = il_cat_all[training_idx].reshape(-1, 1).astype(float)
        cat_pool = il_cat_all[pool_idx].reshape(-1, 1).astype(float)
        X_train_aug = np.column_stack([cat_train, cont_train])
        X_pool_aug = np.column_stack([cat_pool, cont_pool])
        cont_indices = list(range(1, X_train_aug.shape[1]))

        current_best = float(np.max(y_train))
        restart_from_archive = False
        if trust_region is not None and round_start_best is not None:
            improved = current_best > round_start_best
            restart_from_archive = trust_region.update(improved)
            if restart_from_archive:
                incumbent_idx = int(np.argmax(y_train))
                restart_X_raw, restart_y = _append_restart_observation(
                    restart_X_raw,
                    restart_y,
                    X_train_aug[incumbent_idx],
                    current_best,
                    X_train_aug,
                    y_train,
                    np.random.RandomState(seed + r),
                )

        selected_pool_idx, trust_region = select_pool_batch_casmopolitan(
            X_train_aug,
            y_train,
            X_pool_aug,
            il_cat_train=cat_train.ravel(),
            il_cat_pool=cat_pool.ravel(),
            cont_feature_indices=cont_indices,
            cat_feature_indices=[0],
            batch_size=batch_size,
            kappa=kappa,
            random_seed=seed + r,
            trust_length=trust_length_init,
            acq_func=acq_func,
            trust_region=trust_region,
            restart_from_archive=restart_from_archive,
            restart_X_raw=restart_X_raw,
            restart_y=restart_y,
            restart_kappa=kappa,
        )
        batch_idx = [pool_idx[i] for i in selected_pool_idx]
        round_start_best = current_best

        batch_vals = encoded_df.loc[batch_idx, "Experiment_value"].values

        # Update pool and training
        batch_set = set(batch_idx)
        pool_idx = [i for i in pool_idx if i not in batch_set]
        training_idx.extend(batch_idx)
        update_history(history, encoded_df, training_idx, batch_idx, r, top_k_values=top_k_values)

        batch_best = float(batch_vals.max())
        cum_best = history["best_so_far"][-1]
        logger.info(
            "  Round %d: batch_best=%.3f, cum_best=%.3f, TR_length=%.4f, n_pool=%d",
            r + 1, batch_best, cum_best, trust_region.length, len(pool_idx),
        )

    return history


# ---------------------------------------------------------------------------
# Standalone benchmark entry point
# ---------------------------------------------------------------------------


def main():
    """Run CASMOPOLITAN benchmark as a standalone script.

    Usage: python -m optimization.casmopolitan
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="CASMOPOLITAN mixed-variable BO benchmark for LNP formulations",
    )
    parser.add_argument("--seeds", type=str, default="42,123,456,789,2024", help="Comma-separated random seeds")
    parser.add_argument("--n-seed", type=int, default=500, help="Initial seed pool size")
    parser.add_argument("--batch-size", type=int, default=12, help="Batch size per round")
    parser.add_argument("--n-rounds", type=int, default=15, help="Number of BO rounds")
    parser.add_argument("--kappa", type=float, default=5.0, help="UCB exploration weight")
    parser.add_argument("--acq-func", type=str, default="ucb", choices=["ucb", "ei"], help="Acquisition function")
    parser.add_argument(
        "--normalize", type=str, default="copula", choices=["copula", "zscore", "none"], help="Target normalization"
    )
    parser.add_argument(
        "--feature-type", type=str, default="lantern_il_only", help="Feature type for molecular encoding"
    )
    parser.add_argument("--trust-length", type=float, default=0.5, help="Initial trust region length")
    parser.add_argument("--no-ilr", action="store_true", help="Disable ILR transform on ratios")
    parser.add_argument("--max-train-gp", type=int, default=2000, help="Max training size for GP (subsample if larger)")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    from LNPBO.benchmarks.runner import (
        _run_random,
        compute_metrics,
        prepare_benchmark_data,
    )

    seeds = [int(s) for s in args.seeds.split(",")]

    results_dir = Path(__file__).resolve().parent.parent / "benchmark_results"
    results_dir.mkdir(exist_ok=True)
    output_path = args.output or str(results_dir / "casmopolitan.json")

    logger.info("=" * 70)
    logger.info("CASMOPOLITAN Mixed-Variable BO Benchmark")
    logger.info("=" * 70)
    logger.info("Seeds: %s", seeds)
    logger.info("n_seed=%d, batch_size=%d, n_rounds=%d", args.n_seed, args.batch_size, args.n_rounds)
    logger.info("kappa=%s, acq_func=%s, normalize=%s", args.kappa, args.acq_func, args.normalize)
    logger.info("feature_type=%s, trust_length=%s", args.feature_type, args.trust_length)
    logger.info("use_ilr=%s, max_train_gp=%d", not args.no_ilr, args.max_train_gp)
    logger.info("")

    all_seed_results = {
        "casmopolitan": [],
        "random": [],
        "discrete_xgb_greedy": [],
    }

    for s in seeds:
        logger.info("\n" + "=" * 70)
        logger.info("Seed: %d", s)
        logger.info("=" * 70)

        data = prepare_benchmark_data(
            n_seed=args.n_seed,
            random_seed=s,
            feature_type=args.feature_type,
        )
        encoded, encoded_df, feature_cols, seed_idx, oracle_idx, top_k_values = data

        # --- CASMOPOLITAN ---
        logger.info("\n--- CASMOPOLITAN ---")
        t0 = time.time()
        history_cas = run_casmopolitan_strategy(
            encoded_df,
            feature_cols,
            seed_idx,
            oracle_idx,
            batch_size=args.batch_size,
            n_rounds=args.n_rounds,
            seed=s,
            kappa=args.kappa,
            normalize=args.normalize,
            trust_length_init=args.trust_length,
            acq_func=args.acq_func,
            use_ilr=not args.no_ilr,
            max_train_for_gp=args.max_train_gp,
        )
        elapsed_cas = time.time() - t0
        metrics_cas = compute_metrics(history_cas, top_k_values, len(encoded_df))
        all_seed_results["casmopolitan"].append(
            {
                "seed": s,
                "metrics": metrics_cas,
                "elapsed": elapsed_cas,
            }
        )
        logger.info("  CASMOPOLITAN Time: %.1fs", elapsed_cas)
        logger.info("  Top-K recall: %s", {k: f'{v:.1%}' for k, v in metrics_cas['top_k_recall'].items()})

        # --- Random baseline ---
        logger.info("\n--- Random ---")
        t0 = time.time()
        history_rand = _run_random(encoded_df, seed_idx, oracle_idx, args.batch_size, args.n_rounds, s)
        elapsed_rand = time.time() - t0
        metrics_rand = compute_metrics(history_rand, top_k_values, len(encoded_df))
        all_seed_results["random"].append(
            {
                "seed": s,
                "metrics": metrics_rand,
                "elapsed": elapsed_rand,
            }
        )
        logger.info("  Random Time: %.1fs", elapsed_rand)
        logger.info("  Top-K recall: %s", {k: f'{v:.1%}' for k, v in metrics_rand['top_k_recall'].items()})

        # --- XGB greedy baseline ---
        logger.info("\n--- XGB Greedy ---")
        t0 = time.time()
        from LNPBO.benchmarks._optimizer_runner import OptimizerRunner
        from LNPBO.optimization.optimizer import Optimizer

        xgb_opt = Optimizer(
            surrogate_type="xgb",
            batch_strategy="greedy",
            random_seed=s,
            kappa=args.kappa,
            normalize=args.normalize,
            batch_size=args.batch_size,
        )
        xgb_runner = OptimizerRunner(xgb_opt)
        history_xgb = xgb_runner.run(
            encoded_df,
            feature_cols,
            seed_idx,
            oracle_idx,
            n_rounds=args.n_rounds,
            batch_size=args.batch_size,
            encoded_dataset=encoded,
        )
        elapsed_xgb = time.time() - t0
        metrics_xgb = compute_metrics(history_xgb, top_k_values, len(encoded_df))
        all_seed_results["discrete_xgb_greedy"].append(
            {
                "seed": s,
                "metrics": metrics_xgb,
                "elapsed": elapsed_xgb,
            }
        )
        logger.info("  XGB Greedy Time: %.1fs", elapsed_xgb)
        logger.info("  Top-K recall: %s", {k: f'{v:.1%}' for k, v in metrics_xgb['top_k_recall'].items()})

    # --- Aggregate results ---
    logger.info("\n" + "=" * 70)
    logger.info("AGGREGATE RESULTS")
    logger.info("=" * 70)

    summary = {}
    for strategy_name, seed_results in all_seed_results.items():
        recalls = {k: [] for k in [10, 50, 100]}
        for sr in seed_results:
            for k in recalls:
                recalls[k].append(sr["metrics"]["top_k_recall"].get(k, 0))

        summary[strategy_name] = {}
        for k in recalls:
            vals = np.array(recalls[k])
            summary[strategy_name][f"top_{k}_mean"] = float(vals.mean())
            summary[strategy_name][f"top_{k}_std"] = float(vals.std())
            summary[strategy_name][f"top_{k}_values"] = [float(v) for v in vals]

        elapsed_vals = [sr["elapsed"] for sr in seed_results]
        summary[strategy_name]["mean_elapsed"] = float(np.mean(elapsed_vals))

    for strategy_name, s in summary.items():
        logger.info("\n%s:", strategy_name)
        for k in [10, 50, 100]:
            mean = s[f"top_{k}_mean"]
            std = s[f"top_{k}_std"]
            logger.info("  Top-%d: %.1f%% +/- %.1f%%", k, mean * 100, std * 100)
        logger.info("  Mean time: %.1fs", s['mean_elapsed'])

    # Save
    output = {
        "config": {
            "seeds": seeds,
            "n_seed": args.n_seed,
            "batch_size": args.batch_size,
            "n_rounds": args.n_rounds,
            "kappa": args.kappa,
            "acq_func": args.acq_func,
            "normalize": args.normalize,
            "feature_type": args.feature_type,
            "trust_length": args.trust_length,
            "use_ilr": not args.no_ilr,
            "max_train_gp": args.max_train_gp,
        },
        "summary": summary,
        "per_seed": {
            strategy: [{"seed": sr["seed"], "metrics": sr["metrics"], "elapsed": sr["elapsed"]} for sr in seed_results]
            for strategy, seed_results in all_seed_results.items()
        },
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Results saved to %s", output_path)


if __name__ == "__main__":
    main()
