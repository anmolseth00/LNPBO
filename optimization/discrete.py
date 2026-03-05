from __future__ import annotations

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
        "gp_ucb" (GP mean + kappa*sigma).
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
        rng = np.random.RandomState(random_seed)
        tree_idx = rng.randint(0, len(rf.estimators_))
        scores = rf.estimators_[tree_idx].predict(X_pool_s)

    elif surrogate == "gp_ucb":
        from sklearn.gaussian_process import GaussianProcessRegressor

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gp = GaussianProcessRegressor(alpha=1e-6, n_restarts_optimizer=10, random_state=random_seed)
            gp.fit(X_train_s, y_train)
            mu, sigma = gp.predict(X_pool_s, return_std=True)
        scores = mu + kappa * sigma

    else:
        raise ValueError(f"Unknown surrogate: {surrogate!r}. Use 'xgb', 'xgb_ucb', 'rf_ucb', 'rf_ts', or 'gp_ucb'.")

    top_indices = np.argsort(scores)[-batch_size:][::-1]
    return top_indices, scores
