"""CASMOPOLITAN trust-region and pool-selection utilities."""

from __future__ import annotations

import warnings

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel
from sklearn.preprocessing import StandardScaler

from ._casmopolitan_kernels import MixedCasmopolitanKernel


class TrustRegion:
    """Trust region for mixed categorical + continuous search spaces."""

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
        lb = self.center_cont - self.length
        ub = self.center_cont + self.length
        if self.cont_bounds is not None:
            lb = np.maximum(lb, self.cont_bounds[:, 0])
            ub = np.minimum(ub, self.cont_bounds[:, 1])
        return np.column_stack([lb, ub])

    def update(self, improved: bool) -> bool:
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
        self.center_cat = center_cat.copy()
        self.center_cont = center_cont.copy()

    def contains_cat(self, cat_vals: np.ndarray) -> bool:
        return np.sum(cat_vals != self.center_cat) <= self.n_cat_perturb

    def __repr__(self):
        return (
            f"TrustRegion(length={self.length:.4f}, "
            f"n_cat_perturb={self.n_cat_perturb}, "
            f"successes={self._successes}, failures={self._failures})"
        )


def _ucb_acquisition(mu: np.ndarray, sigma: np.ndarray, kappa: float = 5.0) -> np.ndarray:
    return mu + kappa * sigma


def _ei_acquisition(mu: np.ndarray, sigma: np.ndarray, y_best: float) -> np.ndarray:
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
    """Optimize acquisition over mixed categorical + continuous space."""
    if rng is None:
        rng = np.random.RandomState()

    tr_cont_bounds = trust_region.get_cont_bounds()

    if unique_cats.shape[0] <= n_cat_samples:
        cat_candidates = unique_cats
    else:
        in_tr = np.array([trust_region.contains_cat(c) for c in unique_cats])
        in_tr_cats = unique_cats[in_tr]
        if len(in_tr_cats) > n_cat_samples:
            idx = rng.choice(len(in_tr_cats), size=n_cat_samples, replace=False)
            cat_candidates = in_tr_cats[idx]
        elif len(in_tr_cats) > 0:
            cat_candidates = in_tr_cats
        else:
            idx = rng.choice(len(unique_cats), size=min(n_cat_samples, len(unique_cats)), replace=False)
            cat_candidates = unique_cats[idx]

    y_best = float(gp.y_train_.max()) if acq_func == "ei" else 0.0
    best_acq_val = -np.inf
    best_x = None

    for cat in cat_candidates:
        def neg_acq(cont_x, _cat=cat):
            x = np.concatenate([_cat, cont_x]).reshape(1, -1)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mu, sigma = gp.predict(x, return_std=True)
            if acq_func == "ucb":
                return -_ucb_acquisition(mu, sigma, kappa)[0]
            return -_ei_acquisition(mu, sigma, y_best)[0]

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

        x_center = np.concatenate([cat, trust_region.center_cont]).reshape(1, -1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mu_c, sigma_c = gp.predict(x_center, return_std=True)
        acq_c = _ucb_acquisition(mu_c, sigma_c, kappa)[0] if acq_func == "ucb" else _ei_acquisition(mu_c, sigma_c, y_best)[0]
        if acq_c > best_acq_val:
            best_acq_val = acq_c
            best_x = x_center.ravel()

    if best_x is None:
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
    """Select a batch of points using Kriging Believer + CASMOPOLITAN acquisition."""
    if rng is None:
        rng = np.random.RandomState()

    batch = []
    current_gp = gp

    for _ in range(batch_size):
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

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y_halluc = current_gp.predict(x_new.reshape(1, -1))[0]

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


def _assemble_mixed_blocks(cat_block: np.ndarray, cont_block: np.ndarray) -> np.ndarray:
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
    if len(archive_X_raw) == 0:
        raise ValueError("Restart archive must be non-empty.")

    n_cat = len(cat_feature_indices)
    n_cont = len(cont_feature_indices)
    archive_cat = archive_X_raw[:, cat_feature_indices] if n_cat else np.zeros((len(archive_X_raw), 0))
    archive_cont = cont_scaler.transform(archive_X_raw[:, cont_feature_indices]) if n_cont else np.zeros((len(archive_X_raw), 0))
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
    n_cat = len(cat_feature_indices)
    n_cont = len(cont_feature_indices)

    if n_cont:
        cont_scaler = StandardScaler()
        X_train_cont = cont_scaler.fit_transform(X_train[:, cont_feature_indices])
        X_pool_cont = cont_scaler.transform(X_pool[:, cont_feature_indices])
        cont_all = np.vstack([X_train_cont, X_pool_cont])
        cont_bounds = np.column_stack([cont_all.min(axis=0), cont_all.max(axis=0)])
    else:
        cont_scaler = StandardScaler()
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

        dists = np.sum((pool_avail[:, n_cat_dims:] - candidate[n_cat_dims:]) ** 2, axis=1) if pool_avail.shape[1] > n_cat_dims else np.zeros(len(pool_avail))
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
    del il_cat_train, il_cat_pool
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

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mu, sigma = gp.predict(X_pool_mixed, return_std=True)

    scores = _ucb_acquisition(mu, sigma, kappa) if acq_func == "ucb" else _ei_acquisition(mu, sigma, float(y_train.max()))
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
    del il_cat_train, il_cat_pool

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
