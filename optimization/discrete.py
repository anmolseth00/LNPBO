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
        "xgb" (greedy mean), "rf_ucb" (mean + kappa*std),
        "rf_ts" (Thompson sampling), "gp_ucb" (GP mean + kappa*sigma).
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
        raise ValueError(f"Unknown surrogate: {surrogate!r}. Use 'xgb', 'rf_ucb', 'rf_ts', or 'gp_ucb'.")

    top_indices = np.argsort(scores)[-batch_size:][::-1]
    return top_indices, scores
