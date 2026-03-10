"""CASMOPOLITAN-style mixed-variable Bayesian optimization for LNP formulations.

Implements trust-region-based BO for mixed categorical + continuous search
spaces, following the CASMOPOLITAN framework. The key insight is that LNP
formulation design has fundamentally mixed structure: discrete IL identity
(categorical) and continuous molar ratios (on a simplex). Standard BO treats
encoded molecular features as continuous, losing the discrete structure.

Architecture:
    1. CategoricalOverlapKernel: k(x,y) = 1 if x==y, lambda otherwise
    2. MixedProductKernel: categorical overlap x Matern-ARD on continuous dims
    3. TrustRegion: local search around current best with separate categorical
       and continuous trust region management
    4. Mixed acquisition optimization: enumerate/sample categorical configs,
       L-BFGS-B on continuous dims within trust region bounds

Reference
---------
Wan, X., Nguyen, V., Ha, H., Ru, B., Lu, C., & Osborne, M.A. (2021).
    "Think Global and Act Local: Bayesian Optimisation over High-Dimensional
    Categorical and Mixed Search Spaces." ICML 2021. arXiv:2102.07188.
"""

import json
import time
import warnings
from pathlib import Path

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
# Categorical overlap kernel
# ---------------------------------------------------------------------------


class CategoricalOverlapKernel(Kernel):
    """Overlap (Hamming) kernel for categorical variables.

    k(x, y) = prod_j (1 if x_j == y_j else lambda_j)

    where lambda_j in [0, 1] is a learnable similarity parameter for
    dimension j, controlling how similar different categories are. When
    lambda=0, different categories are completely dissimilar (standard
    Hamming). When lambda=1, the kernel ignores that dimension.

    Reference: Wan et al. (2021), Sec. 3.1, Eq. 2. arXiv:2102.07188.
    """

    def __init__(self, n_cat_dims=1, lambd=0.5):
        self.n_cat_dims = n_cat_dims
        self.lambd = lambd

    @property
    def hyperparameter_lambd(self):
        from sklearn.gaussian_process.kernels import Hyperparameter
        return Hyperparameter("lambd", "numeric", (1e-5, 1.0 - 1e-5))

    def __call__(self, X, Y=None, eval_gradient=False):
        X = np.atleast_2d(X)
        if Y is None:
            Y = X

        n_x, n_y = X.shape[0], Y.shape[0]
        K = np.ones((n_x, n_y))

        for j in range(min(self.n_cat_dims, X.shape[1])):
            match = (X[:, j:j+1] == Y[:, j:j+1].T).astype(float)
            K *= (match + (1.0 - match) * self.lambd)

        if eval_gradient:
            dK = np.zeros((n_x, n_y, 1))
            for j in range(min(self.n_cat_dims, X.shape[1])):
                match = (X[:, j:j+1] == Y[:, j:j+1].T).astype(float)
                # d/d(lambd) of (match + (1-match)*lambd) = (1-match)
                factor = (1.0 - match)
                # Product rule: derivative w.r.t. lambd of product over dims
                other_dims = np.ones((n_x, n_y))
                for k in range(min(self.n_cat_dims, X.shape[1])):
                    if k != j:
                        m = (X[:, k:k+1] == Y[:, k:k+1].T).astype(float)
                        other_dims *= (m + (1.0 - m) * self.lambd)
                dK[:, :, 0] += factor * other_dims
            return K, dK * self.lambd
        return K

    def diag(self, X):
        return np.ones(X.shape[0])

    def is_stationary(self):
        return False

    def get_params(self, deep=True):
        return {"n_cat_dims": self.n_cat_dims, "lambd": self.lambd}

    def __repr__(self):
        return f"CategoricalOverlapKernel(n_cat={self.n_cat_dims}, lambd={self.lambd:.3f})"

    def clone_with_theta(self, theta):
        clone = CategoricalOverlapKernel(
            n_cat_dims=self.n_cat_dims,
            lambd=self.lambd,
        )
        clone.theta = theta
        return clone

    @property
    def theta(self):
        return np.log(np.array([self.lambd]))

    @theta.setter
    def theta(self, value):
        self.lambd = np.exp(value[0])

    @property
    def bounds(self):
        return np.log(np.array([[1e-5, 1.0 - 1e-5]]))

    @property
    def n_dims(self):
        return 1


# ---------------------------------------------------------------------------
# Mixed product kernel (categorical x continuous)
# ---------------------------------------------------------------------------


class MixedProductKernel(Kernel):
    """Product of categorical overlap kernel on IL identity x Matern-ARD
    on continuous ratio dimensions.

    k((c, x), (c', x')) = k_cat(c, c') * k_cont(x, x')

    where k_cat is the overlap kernel on categorical dims and k_cont is
    Matern-5/2 with ARD lengthscales on continuous dims.

    The product structure encodes the prior that molecular identity and
    formulation ratios contribute independently to efficacy, with the
    relative importance determined by the kernel hyperparameters.

    Reference: Wan et al. (2021), Sec. 3.1. arXiv:2102.07188.
    """

    def __init__(self, n_cat_dims, n_cont_dims, lambd=0.5):
        self.n_cat_dims = n_cat_dims
        self.n_cont_dims = n_cont_dims
        self.cat_kernel = CategoricalOverlapKernel(n_cat_dims=n_cat_dims, lambd=lambd)
        self.cont_kernel = ConstantKernel(1.0) * Matern(
            length_scale=np.ones(n_cont_dims),
            nu=2.5,
        )

    def __call__(self, X, Y=None, eval_gradient=False):
        X = np.atleast_2d(X)
        X_cat = X[:, :self.n_cat_dims]
        X_cont = X[:, self.n_cat_dims:]

        if Y is None:
            Y_cat = None
            Y_cont = None
        else:
            Y = np.atleast_2d(Y)
            Y_cat = Y[:, :self.n_cat_dims]
            Y_cont = Y[:, self.n_cat_dims:]

        K_cat = self.cat_kernel(X_cat, Y_cat)
        K_cont = self.cont_kernel(X_cont, Y_cont)

        if eval_gradient:
            # For sklearn GP optimization, we don't pass gradients through
            # the categorical kernel (its lambda is not optimized by sklearn).
            # Only the continuous kernel gradients matter.
            K_cont_full, dK_cont = self.cont_kernel(X_cont, Y_cont, eval_gradient=True)
            K = K_cat * K_cont_full
            # Chain rule: d(K_cat * K_cont)/d(theta_cont) = K_cat * dK_cont
            dK = K_cat[:, :, np.newaxis] * dK_cont
            return K, dK
        return K_cat * K_cont

    def diag(self, X):
        X = np.atleast_2d(X)
        return self.cat_kernel.diag(X[:, :self.n_cat_dims]) * \
               self.cont_kernel.diag(X[:, self.n_cat_dims:])

    def is_stationary(self):
        return False

    def get_params(self, deep=True):
        params = {
            "n_cat_dims": self.n_cat_dims,
            "n_cont_dims": self.n_cont_dims,
            "lambd": self.cat_kernel.lambd,
        }
        if deep:
            cont_params = self.cont_kernel.get_params(deep=True)
            params.update({f"cont_kernel__{k}": v for k, v in cont_params.items()})
        return params

    def __repr__(self):
        return (
            f"MixedProductKernel(n_cat={self.n_cat_dims}, n_cont={self.n_cont_dims}, "
            f"lambd={self.cat_kernel.lambd:.3f})"
        )

    @property
    def theta(self):
        return self.cont_kernel.theta

    @theta.setter
    def theta(self, value):
        self.cont_kernel.theta = value

    @property
    def bounds(self):
        return self.cont_kernel.bounds

    @property
    def n_dims(self):
        return self.cont_kernel.n_dims


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

    def update(self, improved: bool) -> None:
        """Update trust region based on whether the batch improved."""
        if improved:
            self._successes += 1
            self._failures = 0
            if self._successes >= self.success_tol:
                self.length = min(self.length * 2.0, self.length_max)
                self._successes = 0
        else:
            self._failures += 1
            self._successes = 0
            if self._failures >= self.failure_tol:
                self.length = max(self.length / 2.0, self.length_min)
                self._failures = 0

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
    """Upper Confidence Bound: mu + kappa * sigma."""
    return mu + kappa * sigma


def _ei_acquisition(mu: np.ndarray, sigma: np.ndarray, y_best: float) -> np.ndarray:
    """Expected Improvement."""
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
    n_cont_dims = trust_region.n_cont_dims

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
        def neg_acq(cont_x):
            x = np.concatenate([cat, cont_x]).reshape(1, -1)
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
            except Exception:
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

    for b in range(batch_size):
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
# Discrete pool scoring for CASMOPOLITAN
# ---------------------------------------------------------------------------


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
    rng = np.random.RandomState(random_seed)

    n_cat = len(cat_feature_indices)
    n_cont = len(cont_feature_indices)

    # Scale continuous features
    cont_scaler = StandardScaler()
    X_train_cont = cont_scaler.fit_transform(X_train[:, cont_feature_indices])
    X_pool_cont = cont_scaler.transform(X_pool[:, cont_feature_indices])

    # Assemble mixed features: [cat_dims | cont_dims]
    X_train_mixed = np.column_stack([X_train[:, cat_feature_indices], X_train_cont])
    X_pool_mixed = np.column_stack([X_pool[:, cat_feature_indices], X_pool_cont])

    # Build mixed kernel
    kernel = MixedProductKernel(n_cat_dims=n_cat, n_cont_dims=n_cont, lambd=0.5)
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

    # Score all pool candidates
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mu, sigma = gp.predict(X_pool_mixed, return_std=True)

    y_best = float(y_train.max())
    if acq_func == "ucb":
        scores = _ucb_acquisition(mu, sigma, kappa)
    else:
        scores = _ei_acquisition(mu, sigma, y_best)

    top_indices = np.argsort(scores)[-batch_size:][::-1]
    return top_indices, scores


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
):
    """Run CASMOPOLITAN mixed-variable BO loop as a benchmark strategy.

    Integrates with the benchmark runner infrastructure. At each round:
    1. Encode IL identity as integer categories
    2. Optionally ILR-transform molar ratios
    3. Fit GP with mixed product kernel
    4. Score pool candidates with acquisition function
    5. Select top-K batch, update training set

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
        Maximum training set size for GP (subsample if larger, since
        sklearn GP is O(n^3)).

    Returns
    -------
    history : dict
        Compatible with benchmarks.runner.compute_metrics().
    """
    from LNPBO.benchmarks.runner import copula_transform, init_history, update_history
    from LNPBO.data.compositional import ilr_transform

    rng = np.random.RandomState(seed)

    # Identify feature structure
    ratio_cols = [c for c in feature_cols if c.endswith("_molratio")]
    mass_ratio_cols = [c for c in feature_cols if c == "IL_to_nucleicacid_massratio"]
    enc_cols = [c for c in feature_cols if c not in ratio_cols and c not in mass_ratio_cols]

    # IL name is the categorical variable (integer-encoded)
    il_names = encoded_df["IL_name"].values
    unique_il_names = np.unique(il_names)
    il_name_to_int = {name: i for i, name in enumerate(unique_il_names)}
    il_cat_all = np.array([il_name_to_int[n] for n in il_names])
    n_unique_il = len(unique_il_names)

    # Continuous features: molecular encodings + ratios (+ optional ILR)
    ratio_indices = [feature_cols.index(c) for c in ratio_cols]
    mass_ratio_indices = [feature_cols.index(c) for c in mass_ratio_cols]
    enc_indices = [feature_cols.index(c) for c in enc_cols]

    training_idx = list(seed_idx)
    pool_idx = list(oracle_idx)
    history = init_history(encoded_df, training_idx)

    # Trust region (initialized after first round)
    trust_region = None

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

        # Scale continuous features
        cont_scaler = StandardScaler()
        cont_train_s = cont_scaler.fit_transform(cont_train)
        cont_pool_s = cont_scaler.transform(cont_pool)

        # Integer-encoded IL categories
        cat_train = il_cat_all[training_idx].reshape(-1, 1).astype(float)
        cat_pool = il_cat_all[pool_idx].reshape(-1, 1).astype(float)
        n_cat = 1

        # Assemble: [cat | cont]
        X_train_mixed = np.hstack([cat_train, cont_train_s])
        X_pool_mixed = np.hstack([cat_pool, cont_pool_s])

        # Subsample training if too large for GP
        if len(X_train_mixed) > max_train_for_gp:
            sub_idx = rng.choice(len(X_train_mixed), size=max_train_for_gp, replace=False)
            X_train_gp = X_train_mixed[sub_idx]
            y_train_gp = y_train[sub_idx]
        else:
            X_train_gp = X_train_mixed
            y_train_gp = y_train

        # Build and fit GP with mixed kernel
        kernel = MixedProductKernel(n_cat_dims=n_cat, n_cont_dims=n_cont, lambd=0.5)
        kernel = kernel + WhiteKernel(noise_level=0.1)

        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            n_restarts_optimizer=5,
            random_state=seed + r,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gp.fit(X_train_gp, y_train_gp)

        # Initialize or update trust region
        best_train_idx = np.argmax(y_train_gp)
        best_cat = X_train_gp[best_train_idx, :n_cat]
        best_cont = X_train_gp[best_train_idx, n_cat:]

        if trust_region is None:
            cont_bounds = np.column_stack([
                cont_pool_s.min(axis=0),
                cont_pool_s.max(axis=0),
            ])
            trust_region = TrustRegion(
                center_cat=best_cat,
                center_cont=best_cont,
                length=trust_length_init,
                n_cat_dims=n_cat,
                n_cont_dims=n_cont,
                cont_bounds=cont_bounds,
            )
        else:
            trust_region.set_center(best_cat, best_cont)

        # Score pool with trust region filtering
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mu_pool, sigma_pool = gp.predict(X_pool_mixed, return_std=True)

        y_best = float(y_train_gp.max())
        if acq_func == "ucb":
            scores = _ucb_acquisition(mu_pool, sigma_pool, kappa)
        else:
            scores = _ei_acquisition(mu_pool, sigma_pool, y_best)

        # Apply trust region penalty: discount candidates outside trust region
        # Vectorized for efficiency over large pools
        tr_cont_bounds = trust_region.get_cont_bounds()

        # Categorical: penalize candidates with different IL than center
        cat_match = (X_pool_mixed[:, :n_cat] == trust_region.center_cat).all(axis=1)
        scores = np.where(cat_match, scores, scores * 0.1)

        # Continuous: penalize candidates outside trust region bounds
        cont_pool_vals = X_pool_mixed[:, n_cat:]
        below_lb = np.any(cont_pool_vals < tr_cont_bounds[:, 0], axis=1)
        above_ub = np.any(cont_pool_vals > tr_cont_bounds[:, 1], axis=1)
        out_of_tr = below_lb | above_ub
        scores = np.where(out_of_tr, scores * 0.5, scores)

        # Select top-K
        top_k_pool = np.argsort(scores)[-batch_size:][::-1]
        batch_idx = [pool_idx[i] for i in top_k_pool]

        # Update trust region: did we improve?
        batch_vals = encoded_df.loc[batch_idx, "Experiment_value"].values
        prev_best = history["best_so_far"][-1]
        improved = float(batch_vals.max()) > prev_best
        trust_region.update(improved)

        # Update pool and training
        batch_set = set(batch_idx)
        pool_idx = [i for i in pool_idx if i not in batch_set]
        training_idx.extend(batch_idx)
        update_history(history, encoded_df, training_idx, batch_idx, r)

        batch_best = float(batch_vals.max())
        cum_best = history["best_so_far"][-1]
        print(
            f"  Round {r+1}: batch_best={batch_best:.3f}, cum_best={cum_best:.3f}, "
            f"TR_length={trust_region.length:.4f}, n_pool={len(pool_idx)}",
            flush=True,
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
    parser.add_argument("--seeds", type=str, default="42,123,456,789,2024",
                        help="Comma-separated random seeds")
    parser.add_argument("--n-seed", type=int, default=500,
                        help="Initial seed pool size")
    parser.add_argument("--batch-size", type=int, default=12,
                        help="Batch size per round")
    parser.add_argument("--n-rounds", type=int, default=15,
                        help="Number of BO rounds")
    parser.add_argument("--kappa", type=float, default=5.0,
                        help="UCB exploration weight")
    parser.add_argument("--acq-func", type=str, default="ucb",
                        choices=["ucb", "ei"],
                        help="Acquisition function")
    parser.add_argument("--normalize", type=str, default="copula",
                        choices=["copula", "zscore", "none"],
                        help="Target normalization")
    parser.add_argument("--feature-type", type=str, default="lantern_il_only",
                        help="Feature type for molecular encoding")
    parser.add_argument("--trust-length", type=float, default=0.5,
                        help="Initial trust region length")
    parser.add_argument("--no-ilr", action="store_true",
                        help="Disable ILR transform on ratios")
    parser.add_argument("--max-train-gp", type=int, default=2000,
                        help="Max training size for GP (subsample if larger)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path")
    args = parser.parse_args()

    from LNPBO.benchmarks.runner import (
        _run_random,
        compute_metrics,
        init_history,
        prepare_benchmark_data,
    )

    seeds = [int(s) for s in args.seeds.split(",")]

    results_dir = Path(__file__).resolve().parent.parent / "benchmark_results"
    results_dir.mkdir(exist_ok=True)
    output_path = args.output or str(results_dir / "casmopolitan.json")

    print("=" * 70)
    print("CASMOPOLITAN Mixed-Variable BO Benchmark")
    print("=" * 70)
    print(f"Seeds: {seeds}")
    print(f"n_seed={args.n_seed}, batch_size={args.batch_size}, n_rounds={args.n_rounds}")
    print(f"kappa={args.kappa}, acq_func={args.acq_func}, normalize={args.normalize}")
    print(f"feature_type={args.feature_type}, trust_length={args.trust_length}")
    print(f"use_ilr={not args.no_ilr}, max_train_gp={args.max_train_gp}")
    print()

    all_seed_results = {
        "casmopolitan": [],
        "random": [],
        "discrete_xgb_greedy": [],
    }

    for s in seeds:
        print(f"\n{'='*70}")
        print(f"Seed: {s}")
        print(f"{'='*70}")

        data = prepare_benchmark_data(
            n_seed=args.n_seed,
            random_seed=s,
            feature_type=args.feature_type,
        )
        encoded, encoded_df, feature_cols, seed_idx, oracle_idx, top_k_values = data

        # --- CASMOPOLITAN ---
        print("\n--- CASMOPOLITAN ---")
        t0 = time.time()
        history_cas = run_casmopolitan_strategy(
            encoded_df, feature_cols, seed_idx, oracle_idx,
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
        all_seed_results["casmopolitan"].append({
            "seed": s, "metrics": metrics_cas, "elapsed": elapsed_cas,
        })
        print(f"  CASMOPOLITAN Time: {elapsed_cas:.1f}s")
        print(f"  Top-K recall: { {k: f'{v:.1%}' for k, v in metrics_cas['top_k_recall'].items()} }")

        # --- Random baseline ---
        print("\n--- Random ---")
        t0 = time.time()
        history_rand = _run_random(encoded_df, seed_idx, oracle_idx, args.batch_size, args.n_rounds, s)
        elapsed_rand = time.time() - t0
        metrics_rand = compute_metrics(history_rand, top_k_values, len(encoded_df))
        all_seed_results["random"].append({
            "seed": s, "metrics": metrics_rand, "elapsed": elapsed_rand,
        })
        print(f"  Random Time: {elapsed_rand:.1f}s")
        print(f"  Top-K recall: { {k: f'{v:.1%}' for k, v in metrics_rand['top_k_recall'].items()} }")

        # --- XGB greedy baseline ---
        print("\n--- XGB Greedy ---")
        t0 = time.time()
        from LNPBO.benchmarks._discrete_common import run_discrete_strategy
        history_xgb = run_discrete_strategy(
            encoded_df, feature_cols, seed_idx, oracle_idx,
            surrogate="xgb", batch_size=args.batch_size,
            n_rounds=args.n_rounds, seed=s, kappa=args.kappa,
            normalize=args.normalize, encoded_dataset=encoded,
        )
        elapsed_xgb = time.time() - t0
        metrics_xgb = compute_metrics(history_xgb, top_k_values, len(encoded_df))
        all_seed_results["discrete_xgb_greedy"].append({
            "seed": s, "metrics": metrics_xgb, "elapsed": elapsed_xgb,
        })
        print(f"  XGB Greedy Time: {elapsed_xgb:.1f}s")
        print(f"  Top-K recall: { {k: f'{v:.1%}' for k, v in metrics_xgb['top_k_recall'].items()} }")

    # --- Aggregate results ---
    print(f"\n{'='*70}")
    print("AGGREGATE RESULTS")
    print(f"{'='*70}")

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
        print(f"\n{strategy_name}:")
        for k in [10, 50, 100]:
            mean = s[f"top_{k}_mean"]
            std = s[f"top_{k}_std"]
            print(f"  Top-{k}: {mean:.1%} +/- {std:.1%}")
        print(f"  Mean time: {s['mean_elapsed']:.1f}s")

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
            strategy: [
                {"seed": sr["seed"], "metrics": sr["metrics"], "elapsed": sr["elapsed"]}
                for sr in seed_results
            ]
            for strategy, seed_results in all_seed_results.items()
        },
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
