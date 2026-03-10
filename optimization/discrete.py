"""Discrete pool scoring for BO surrogates (XGBoost, RF, NGBoost, CQR, TabPFN, conformal).

XGB-UCB uses MAPIE conformal prediction for uncertainty quantification:
    Barber, R.F. et al. (2021). "Predictive Inference with the Jackknife+."
    Ann. Statist. 49(1), 486-507. doi:10.1214/20-AOS1965.

NGBoost (Natural Gradient Boosting) provides native distributional UQ:
    Duan, T. et al. (2020). "NGBoost: Natural Gradient Boosting for
    Probabilistic Prediction." ICML 2020. arXiv:1910.03225.

Conformalized Quantile Regression (CQR) produces adaptive-width intervals:
    Romano, Y. et al. (2019). "Conformalized Quantile Regression."
    NeurIPS 2019. arXiv:1905.03222.

TabPFN zero-shot tabular foundation model:
    Hollmann, N. et al. (2025). "Accurate Predictions on Small Data with a
    Tabular Foundation Model." Nature.
"""

import warnings

import numpy as np
from sklearn.preprocessing import MinMaxScaler


def score_candidate_pool(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_pool: np.ndarray,
    surrogate: str = "xgb",
    batch_size: int = 12,
    kappa: float = 5.0,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Score a discrete candidate pool and return top-K indices.

    Fits a surrogate model on training data, predicts scores for all pool
    candidates, and returns the top batch_size indices ranked by score.

    Parameters
    ----------
    X_train : array of shape (n_train, n_features)
    y_train : array of shape (n_train,)
    X_pool : array of shape (n_pool, n_features)
    surrogate : str
        "xgb" (greedy mean), "xgb_ucb" (XGB + MAPIE conformal UCB),
        "rf_ucb" (mean + kappa*std), "rf_ts" (Thompson sampling),
        "gp_ucb" (GP mean + kappa*sigma), "ngboost" (NGBoost distributional UCB),
        "xgb_cqr" (CQR adaptive-width conformal UCB),
        "tabpfn" (TabPFN zero-shot, Hollmann et al. 2025).
    batch_size : int
        Number of top candidates to return.
    kappa : float
        Exploration weight for UCB-based surrogates.
    random_seed : int

    Returns
    -------
    top_indices : array of shape (batch_size,)
        Indices into X_pool of the top-scoring candidates.
    scores : array of shape (n_pool,)
        Full score array for all pool candidates.
    """
    scaler = MinMaxScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_pool_s = scaler.transform(X_pool)

    if surrogate == "xgb":
        from xgboost import XGBRegressor

        model = XGBRegressor(n_estimators=200, random_state=random_seed, n_jobs=-1, verbosity=0)
        model.fit(X_train_s, y_train)
        scores = model.predict(X_pool_s)

    elif surrogate == "xgb_ucb":
        from mapie.regression import CrossConformalRegressor
        from xgboost import XGBRegressor

        base = XGBRegressor(n_estimators=200, random_state=random_seed, n_jobs=-1, verbosity=0)
        # confidence_level=0.68 gives ~1-sigma equivalent intervals
        # method="plus" uses CV+ for tighter bounds (Barber et al., AoS 2021)
        n_cv = min(5, len(X_train_s))
        mapie = CrossConformalRegressor(
            base, method="plus", cv=n_cv, confidence_level=0.68,
            random_state=random_seed, n_jobs=-1,
        )
        mapie.fit_conformalize(X_train_s, y_train)
        y_pred, y_intervals = mapie.predict_interval(X_pool_s)
        # y_intervals shape: (n_samples, 2, 1) for single confidence level
        half_width = (y_intervals[:, 1, 0] - y_intervals[:, 0, 0]) / 2
        scores = y_pred + kappa * half_width

    elif surrogate == "rf_ucb":
        from sklearn.ensemble import RandomForestRegressor

        rf = RandomForestRegressor(n_estimators=200, random_state=random_seed, n_jobs=-1)
        rf.fit(X_train_s, y_train)
        tree_preds = np.array([t.predict(X_pool_s) for t in rf.estimators_])
        mu = tree_preds.mean(axis=0)
        sigma = tree_preds.std(axis=0)
        scores = mu + kappa * sigma

    elif surrogate == "rf_ts":
        from sklearn.ensemble import RandomForestRegressor

        rf = RandomForestRegressor(n_estimators=200, random_state=random_seed, n_jobs=-1)
        rf.fit(X_train_s, y_train)
        # Draw a different tree for each batch slot to ensure diversity.
        # Kandasamy et al. (AISTATS 2018): each batch point should use
        # an independent posterior sample.
        rng = np.random.RandomState(random_seed)
        n_trees = len(rf.estimators_)
        remaining = list(range(len(X_pool_s)))
        batch = []
        scores = np.full(len(X_pool_s), np.nan)
        for _ in range(min(batch_size, len(remaining))):
            tree_idx = rng.randint(0, n_trees)
            rem_arr = np.array(remaining)
            tree_scores = rf.estimators_[tree_idx].predict(X_pool_s[rem_arr])
            best_local = np.argmax(tree_scores)
            best_idx = remaining[best_local]
            batch.append(best_idx)
            scores[best_idx] = tree_scores[best_local]
            remaining.pop(best_local)
        top_indices = np.array(batch, dtype=int)
        return top_indices, scores

    elif surrogate == "gp_ucb":
        from sklearn.gaussian_process import GaussianProcessRegressor

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gp = GaussianProcessRegressor(alpha=1e-6, n_restarts_optimizer=10, random_state=random_seed)
            gp.fit(X_train_s, y_train)
            mu, sigma = gp.predict(X_pool_s, return_std=True)
        scores = mu + kappa * sigma

    elif surrogate == "ngboost":
        from ngboost import NGBRegressor
        from ngboost.distns import Normal

        model = NGBRegressor(
            Dist=Normal, n_estimators=200,
            random_state=random_seed, verbose=False,
        )
        model.fit(X_train_s, y_train)
        dists = model.pred_dist(X_pool_s)
        scores = dists.mean() + kappa * dists.scale

    elif surrogate == "xgb_cqr":
        from mapie.regression import ConformalizedQuantileRegressor
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.model_selection import train_test_split

        X_fit, X_cal, y_fit, y_cal = train_test_split(
            X_train_s, y_train, test_size=0.2, random_state=random_seed,
        )
        base = HistGradientBoostingRegressor(
            loss="quantile", max_iter=200, random_state=random_seed,
        )
        cqr = ConformalizedQuantileRegressor(base, confidence_level=0.68)
        cqr.fit(X_fit, y_fit)
        cqr.conformalize(X_cal, y_cal)
        y_pred, y_intervals = cqr.predict_interval(X_pool_s)
        half_width = (y_intervals[:, 1, 0] - y_intervals[:, 0, 0]) / 2
        scores = y_pred + kappa * half_width

    elif surrogate == "deep_ensemble":
        # Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017).
        # "Simple and Scalable Predictive Uncertainty Estimation using
        # Deep Ensembles." NeurIPS 2017. arXiv:1612.01474.
        from LNPBO.models.deep_ensemble import DeepEnsemble

        ensemble = DeepEnsemble(
            input_dim=X_train_s.shape[1],
            n_models=5,
            epochs=100,
            lr=1e-3,
        )
        ensemble.fit(X_train_s, y_train, bootstrap=True, seed=random_seed)
        mu, sigma = ensemble.predict(X_pool_s)
        scores = mu + kappa * sigma

    elif surrogate == "tabpfn":
        # TabPFN: zero-shot tabular foundation model surrogate.
        # Hollmann, N. et al. (2025). "Accurate Predictions on Small Data
        # with a Tabular Foundation Model." Nature.
        from tabpfn import TabPFNRegressor

        MAX_TABPFN_TRAIN = 3000
        X_fit, y_fit = X_train_s, y_train
        if len(X_train_s) > MAX_TABPFN_TRAIN:
            rng = np.random.RandomState(random_seed)
            n_bins = min(10, len(y_train) // 5)
            bins = np.digitize(
                y_train,
                np.percentile(y_train, np.linspace(0, 100, n_bins + 1)[1:-1]),
            )
            selected = []
            unique_bins = np.unique(bins)
            per_bin = max(1, MAX_TABPFN_TRAIN // len(unique_bins))
            for b in unique_bins:
                idx_in_bin = np.where(bins == b)[0]
                take = min(per_bin, len(idx_in_bin))
                selected.extend(rng.choice(idx_in_bin, size=take, replace=False).tolist())
            remaining = MAX_TABPFN_TRAIN - len(selected)
            if remaining > 0:
                leftover = np.setdiff1d(np.arange(len(y_train)), selected)
                if len(leftover) > 0:
                    selected.extend(
                        rng.choice(leftover, size=min(remaining, len(leftover)), replace=False).tolist()
                    )
            sub_idx = np.array(selected[:MAX_TABPFN_TRAIN])
            X_fit, y_fit = X_train_s[sub_idx], y_train[sub_idx]

        model = TabPFNRegressor()
        model.fit(X_fit, y_fit)

        sigma = None
        try:
            mu, sigma = model.predict(X_pool_s, return_std=True)
        except TypeError:
            mu = model.predict(X_pool_s)

        if sigma is not None and len(sigma) == len(mu):
            scores = mu + kappa * sigma
        else:
            scores = mu

    else:
        raise ValueError(
            f"Unknown surrogate: {surrogate!r}. "
            "Use 'xgb', 'xgb_ucb', 'ngboost', 'xgb_cqr', "
            "'rf_ucb', 'rf_ts', 'gp_ucb', 'deep_ensemble', or 'tabpfn'."
        )

    top_indices = np.argsort(scores)[-batch_size:][::-1]
    return top_indices, scores


def score_candidate_pool_ts_batch(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_pool: np.ndarray,
    surrogate: str = "rf_ucb",
    batch_size: int = 12,
    kappa: float = 5.0,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Thompson Sampling batch: draw from predictive distribution, pick top-1, repeat.

    For each batch slot, draws independent samples from the surrogate's
    predictive distribution N(mu, sigma^2) for all remaining pool candidates,
    selects the candidate with the highest sample, removes it from the pool,
    and repeats. Each batch slot uses a different random draw, providing
    natural diversity without hyperparameters.

    Works with any surrogate that provides (mu, sigma): rf_ucb, xgb_ucb, gp_ucb.

    Parameters
    ----------
    X_train : array of shape (n_train, n_features)
    y_train : array of shape (n_train,)
    X_pool : array of shape (n_pool, n_features)
    surrogate : str
        Must be one of "rf_ucb", "xgb_ucb", "gp_ucb".
    batch_size : int
    kappa : float
        Unused (kept for API compatibility).
    random_seed : int

    Returns
    -------
    top_indices : array of shape (batch_size,)
        Indices into X_pool of the selected batch.
    scores : array of shape (n_pool,)
        Final Thompson sample scores (from the last batch slot draw).

    References
    ----------
    Kandasamy, K., Krishnamurthy, A., Schneider, J., & Poczos, B.
    "Parallelised Bayesian Optimisation via Thompson Sampling."
    AISTATS 2018.
    """
    scaler = MinMaxScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_pool_s = scaler.transform(X_pool)

    if surrogate == "rf_ucb":
        from sklearn.ensemble import RandomForestRegressor

        rf = RandomForestRegressor(n_estimators=200, random_state=random_seed, n_jobs=-1)
        rf.fit(X_train_s, y_train)
        tree_preds = np.array([t.predict(X_pool_s) for t in rf.estimators_])
        mu = tree_preds.mean(axis=0)
        sigma = tree_preds.std(axis=0)

    elif surrogate == "xgb_ucb":
        # Use a bootstrap ensemble of XGBoost models to get proper
        # (mu, sigma) for Thompson sampling.  Conformal interval
        # half-widths are not standard deviations and should not be
        # used as the sigma parameter in a Gaussian draw.
        from xgboost import XGBRegressor

        rng_boot = np.random.RandomState(random_seed)
        n_boot = 20
        preds = np.zeros((n_boot, len(X_pool_s)))
        for b in range(n_boot):
            idx = rng_boot.choice(len(X_train_s), size=len(X_train_s), replace=True)
            m = XGBRegressor(n_estimators=200, random_state=random_seed + b, n_jobs=-1, verbosity=0)
            m.fit(X_train_s[idx], y_train[idx])
            preds[b] = m.predict(X_pool_s)
        mu = preds.mean(axis=0)
        sigma = preds.std(axis=0)

    elif surrogate == "gp_ucb":
        from sklearn.gaussian_process import GaussianProcessRegressor

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gp = GaussianProcessRegressor(alpha=1e-6, n_restarts_optimizer=10, random_state=random_seed)
            gp.fit(X_train_s, y_train)
            mu, sigma = gp.predict(X_pool_s, return_std=True)

    else:
        raise ValueError(
            f"Unknown surrogate for TS batch: {surrogate!r}. "
            "Use 'rf_ucb', 'xgb_ucb', or 'gp_ucb'."
        )

    sigma = np.maximum(sigma, 1e-8)

    rng = np.random.RandomState(random_seed)
    remaining = list(range(len(X_pool_s)))
    batch = []
    scores = np.full(len(X_pool_s), np.nan)

    for _ in range(min(batch_size, len(remaining))):
        rem_arr = np.array(remaining)
        samples = rng.normal(mu[rem_arr], sigma[rem_arr])
        best_local = np.argmax(samples)
        best_idx = remaining[best_local]
        batch.append(best_idx)
        scores[best_idx] = samples[best_local]
        remaining.pop(best_local)

    return np.array(batch, dtype=int), scores
