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


def _scale_features(X_train, X_pool):
    """MinMaxScale train and pool features."""
    scaler = MinMaxScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_pool)


def score_candidate_pool(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_pool: np.ndarray,
    surrogate: str = "xgb",
    batch_size: int = 12,
    kappa: float = 5.0,
    random_seed: int = 42,
    surrogate_kwargs: dict | None = None,
    group_ids: np.ndarray | None = None,
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
        "ridge" (BayesianRidge mean + kappa*std, MacKay 1992),
        "tabpfn" (TabPFN zero-shot, Hollmann et al. 2025).
    batch_size : int
        Number of top candidates to return.
    kappa : float
        Exploration weight for UCB-based surrogates.
    random_seed : int
    surrogate_kwargs : dict or None
        Extra keyword arguments forwarded to the surrogate model constructor
        (e.g., ``{"n_estimators": 500, "max_depth": 10}``). Overrides defaults.

    Returns
    -------
    top_indices : array of shape (batch_size,)
        Indices into X_pool of the top-scoring candidates.
    scores : array of shape (n_pool,)
        Full score array for all pool candidates.
    """
    if surrogate_kwargs is None:
        surrogate_kwargs = {}
    # n_jobs is only meaningful for RF and XGBoost; pop it here to avoid
    # TypeError when surrogate_kwargs leaks into NGBoost/DeepEnsemble/Ridge/GP.
    n_jobs = surrogate_kwargs.pop("n_jobs", -1)
    X_train_s, X_pool_s = _scale_features(X_train, X_pool)

    if surrogate == "xgb":
        from xgboost import XGBRegressor

        xgb_kw = {"n_estimators": 200, "random_state": random_seed, "n_jobs": n_jobs, "verbosity": 0}
        xgb_kw.update(surrogate_kwargs)
        model = XGBRegressor(**xgb_kw)
        model.fit(X_train_s, y_train)
        scores = model.predict(X_pool_s)

    elif surrogate == "xgb_ucb":
        from mapie.regression import CrossConformalRegressor
        from xgboost import XGBRegressor

        xgb_kw = {"n_estimators": 200, "random_state": random_seed, "n_jobs": n_jobs, "verbosity": 0}
        xgb_kw.update(surrogate_kwargs)
        base = XGBRegressor(**xgb_kw)
        # confidence_level=0.68 gives ~1-sigma equivalent intervals
        # method="plus" uses CV+ for tighter bounds (Barber et al., AoS 2021)
        n_cv = min(5, len(X_train_s))
        mapie = CrossConformalRegressor(
            base,
            method="plus",
            cv=n_cv,
            confidence_level=0.68,
            random_state=random_seed,
            n_jobs=-1,
        )
        mapie.fit_conformalize(X_train_s, y_train)
        y_pred, y_intervals = mapie.predict_interval(X_pool_s)
        # y_intervals shape: (n_samples, 2, 1) for single confidence level
        half_width = (y_intervals[:, 1, 0] - y_intervals[:, 0, 0]) / 2
        scores = y_pred + kappa * half_width

    elif surrogate == "rf_ucb":
        from sklearn.ensemble import RandomForestRegressor

        rf_kw = {"n_estimators": 200, "random_state": random_seed, "n_jobs": n_jobs}
        rf_kw.update(surrogate_kwargs)
        rf = RandomForestRegressor(**rf_kw)
        rf.fit(X_train_s, y_train)
        tree_preds = np.array([t.predict(X_pool_s) for t in rf.estimators_])
        mu = tree_preds.mean(axis=0)
        sigma = tree_preds.std(axis=0)
        scores = mu + kappa * sigma

    elif surrogate == "rf_ts":
        from sklearn.ensemble import RandomForestRegressor

        rf_kw = {"n_estimators": 200, "random_state": random_seed, "n_jobs": n_jobs}
        rf_kw.update(surrogate_kwargs)
        rf = RandomForestRegressor(**rf_kw)
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

        gp_kw = {"alpha": 1e-6, "n_restarts_optimizer": 10, "random_state": random_seed}
        gp_kw.update(surrogate_kwargs)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gp = GaussianProcessRegressor(**gp_kw)
            gp.fit(X_train_s, y_train)
            mu, sigma = gp.predict(X_pool_s, return_std=True)
        scores = mu + kappa * sigma

    elif surrogate == "ngboost":
        from ngboost import NGBRegressor
        from ngboost.distns import Normal

        ngb_kw = {"Dist": Normal, "n_estimators": 200, "random_state": random_seed, "verbose": False}
        ngb_kw.update(surrogate_kwargs)
        model = NGBRegressor(**ngb_kw)
        model.fit(X_train_s, y_train)
        dists = model.pred_dist(X_pool_s)
        scores = dists.mean() + kappa * dists.scale

    elif surrogate == "xgb_cqr":
        from mapie.regression import ConformalizedQuantileRegressor
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.model_selection import train_test_split

        X_fit, X_cal, y_fit, y_cal = train_test_split(
            X_train_s,
            y_train,
            test_size=0.2,
            random_state=random_seed,
        )
        cqr_kw = {"loss": "quantile", "max_iter": 200, "random_state": random_seed}
        cqr_kw.update(surrogate_kwargs)
        base = HistGradientBoostingRegressor(**cqr_kw)
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

        de_kw = {"input_dim": X_train_s.shape[1], "n_models": 5, "epochs": 100, "lr": 1e-3}
        de_kw.update(surrogate_kwargs)
        ensemble = DeepEnsemble(**de_kw)
        ensemble.fit(X_train_s, y_train, bootstrap=True, seed=random_seed)
        mu, sigma = ensemble.predict(X_pool_s)
        scores = mu + kappa * sigma

    elif surrogate == "ridge":
        # BayesianRidge: linear model with built-in uncertainty estimates.
        # Provides calibrated mean and std from the posterior over weights.
        # MacKay, D.J.C. (1992). "Bayesian Interpolation." Neural Computation.
        # Tipping, M.E. (2001). "Sparse Bayesian Learning and the Relevance
        # Vector Machine." JMLR.
        from sklearn.linear_model import BayesianRidge

        ridge_kw = {}
        ridge_kw.update(surrogate_kwargs)
        model = BayesianRidge(**ridge_kw)
        model.fit(X_train_s, y_train)
        mu, sigma = model.predict(X_pool_s, return_std=True)
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
                    selected.extend(rng.choice(leftover, size=min(remaining, len(leftover)), replace=False).tolist())
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
            warnings.warn("TabPFN did not return std — using mean-only scoring (exploration disabled)", stacklevel=2)
            scores = mu

    elif surrogate == "sngp":
        # SNGP: spectral-normalized MLP + RFF GP head → (mu, sigma).
        # Liu et al. (2023), "A Simple Approach to Improve Single-Model Deep
        # Uncertainty via Distance-Awareness." JMLR 24(42).
        import torch

        from LNPBO.models.sngp import train_sngp

        X_t = torch.tensor(X_train_s, dtype=torch.float32)
        y_t = torch.tensor(y_train, dtype=torch.float32)
        sngp_kw = {"input_dim": X_train_s.shape[1], "epochs": 100, "lr": 1e-3}
        sngp_kw.update(surrogate_kwargs)
        model = train_sngp(X_t, y_t, **sngp_kw)
        mu, sigma = model.predict_with_uncertainty(
            torch.tensor(X_pool_s, dtype=torch.float32)
        )
        scores = mu + kappa * np.maximum(sigma, 1e-8)

    elif surrogate == "laplace":
        # MLP + post-hoc Laplace approximation → (mu, var).
        # Daxberger et al. (2021), "Laplace Redux." NeurIPS 2021.
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        from LNPBO.models.laplace import SurrogateMLP, build_laplace, train_mlp

        X_t = torch.tensor(X_train_s, dtype=torch.float32)
        y_t = torch.tensor(y_train, dtype=torch.float32)
        model = SurrogateMLP(X_train_s.shape[1])
        mlp_kw = {"epochs": 100, "lr": 1e-3}
        mlp_kw.update({k: v for k, v in surrogate_kwargs.items()
                       if k in ("epochs", "lr", "batch_size")})
        train_mlp(model, X_t, y_t, **mlp_kw)
        model.eval()

        la = build_laplace(model)
        train_loader = DataLoader(TensorDataset(X_t, y_t), batch_size=256)
        la.fit(train_loader)
        la.optimize_prior_precision()
        mu, var = la.predict(torch.tensor(X_pool_s, dtype=torch.float32))
        sigma = np.sqrt(np.maximum(var, 0.0))
        scores = mu + kappa * np.maximum(sigma, 1e-8)

    elif surrogate == "bradley_terry":
        # Bradley-Terry pairwise preference model → utility scores.
        # Bradley & Terry (1952), Biometrika 39(3/4).
        import torch

        from LNPBO.models.bradley_terry import train_bt_model

        model = train_bt_model(
            X_train_s, y_train, group_ids=group_ids,
            epochs=20, seed=random_seed,
        )
        model.eval()
        with torch.no_grad():
            scores = model(torch.tensor(X_pool_s, dtype=torch.float32)).cpu().numpy()

    elif surrogate == "groupdro":
        # GroupDRO: distributionally robust MLP across groups/studies.
        # Sagawa et al. (2020), "Distributionally Robust Neural Networks." ICLR.
        import torch

        from LNPBO.models.groupdro import train_groupdro

        gids = group_ids if group_ids is not None else np.zeros(len(y_train), dtype=int)
        gdro_kw = {"eta": 0.01, "epochs": 200, "lr": 1e-3}
        gdro_kw.update(surrogate_kwargs)
        model = train_groupdro(X_train_s, y_train, gids, **gdro_kw)
        model.eval()
        with torch.no_grad():
            scores = model(torch.tensor(X_pool_s, dtype=torch.float32)).cpu().numpy()

    elif surrogate == "vrex":
        # V-REx: risk extrapolation across groups/studies.
        # Krueger et al. (2021), "Out-of-Distribution Generalization via
        # Risk Extrapolation." ICML.
        import torch

        from LNPBO.models.vrex import train_vrex

        gids = group_ids if group_ids is not None else np.zeros(len(y_train), dtype=int)
        vrex_kw = {"lambda_rex": 1.0, "epochs": 200, "lr": 1e-3}
        vrex_kw.update(surrogate_kwargs)
        model = train_vrex(X_train_s, y_train, gids, **vrex_kw)
        model.eval()
        with torch.no_grad():
            scores = model(torch.tensor(X_pool_s, dtype=torch.float32)).cpu().numpy()

    else:
        raise ValueError(
            f"Unknown surrogate: {surrogate!r}. "
            "Use 'xgb', 'xgb_ucb', 'ngboost', 'xgb_cqr', 'rf_ucb', 'rf_ts', "
            "'gp_ucb', 'deep_ensemble', 'ridge', 'tabpfn', 'sngp', 'laplace', "
            "'bradley_terry', 'groupdro', or 'vrex'."
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
    surrogate_kwargs: dict | None = None,
    group_ids: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Thompson Sampling batch: draw from predictive distribution, pick top-1, repeat.

    For each batch slot, draws independent samples from the surrogate's
    predictive distribution N(mu, sigma^2) for all remaining pool candidates,
    selects the candidate with the highest sample, removes it from the pool,
    and repeats. Each batch slot uses a different random draw, providing
    natural diversity without hyperparameters.

    Works with any surrogate that provides (mu, sigma): rf_ucb, xgb_ucb,
    gp_ucb, ridge.

    Parameters
    ----------
    X_train : array of shape (n_train, n_features)
    y_train : array of shape (n_train,)
    X_pool : array of shape (n_pool, n_features)
    surrogate : str
        Must be one of "rf_ucb", "xgb_ucb", "gp_ucb", "ridge".
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
    if surrogate_kwargs is None:
        surrogate_kwargs = {}
    n_jobs = surrogate_kwargs.pop("n_jobs", 1)
    X_train_s, X_pool_s = _scale_features(X_train, X_pool)

    if surrogate == "rf_ucb":
        from sklearn.ensemble import RandomForestRegressor

        rf_kw = {"n_estimators": 200, "random_state": random_seed, "n_jobs": n_jobs}
        rf_kw.update(surrogate_kwargs)
        rf = RandomForestRegressor(**rf_kw)
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
        xgb_kw = {"n_estimators": 200, "n_jobs": n_jobs, "verbosity": 0}
        xgb_kw.update(surrogate_kwargs)
        for b in range(n_boot):
            idx = rng_boot.choice(len(X_train_s), size=len(X_train_s), replace=True)
            m = XGBRegressor(random_state=random_seed + b, **xgb_kw)
            m.fit(X_train_s[idx], y_train[idx])
            preds[b] = m.predict(X_pool_s)
        mu = preds.mean(axis=0)
        sigma = preds.std(axis=0)

    elif surrogate == "gp_ucb":
        from sklearn.gaussian_process import GaussianProcessRegressor

        gp_kw = {"alpha": 1e-6, "n_restarts_optimizer": 10, "random_state": random_seed}
        gp_kw.update(surrogate_kwargs)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gp = GaussianProcessRegressor(**gp_kw)
            gp.fit(X_train_s, y_train)
            mu, sigma = gp.predict(X_pool_s, return_std=True)

    elif surrogate == "ridge":
        from sklearn.linear_model import BayesianRidge

        ridge_kw = {}
        ridge_kw.update(surrogate_kwargs)
        model = BayesianRidge(**ridge_kw)
        model.fit(X_train_s, y_train)
        mu, sigma = model.predict(X_pool_s, return_std=True)

    else:
        raise ValueError(
            f"Unknown surrogate for TS batch: {surrogate!r}. "
            "Use 'rf_ucb', 'xgb_ucb', 'gp_ucb', or 'ridge'."
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
