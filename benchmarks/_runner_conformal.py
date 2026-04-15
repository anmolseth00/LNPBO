"""Online conformal benchmark loops."""

from __future__ import annotations

import time

import numpy as np

from LNPBO.benchmarks._runner_history import init_history, update_history
from LNPBO.benchmarks._runner_logging import _log_round_complete, _log_round_start


def run_discrete_online_conformal_strategy(
    encoded_df,
    feature_cols,
    seed_idx,
    oracle_idx,
    batch_size,
    n_rounds,
    seed,
    kappa=5.0,
    alpha=0.1,
    normalize="copula",
    encoded_dataset=None,
    top_k_values=None,
):
    """Run exact online quantile recalibration BO from Deshpande et al."""
    from scipy.stats import norm

    from LNPBO.optimization._normalize import copula_transform
    from LNPBO.optimization.online_conformal import (
        CalibratedProbabilisticModel,
        ExactOnlineRecalibrator,
        GaussianXGBQuantileModel,
        build_recalibration_dataset,
    )

    training_idx = list(seed_idx)
    pool_idx = list(oracle_idx)
    history = init_history(encoded_df, training_idx, top_k_values=top_k_values)
    history["coverage"] = []
    history["conformal_quantile"] = []

    p_ucb = float(norm.cdf(kappa))

    for r in range(n_rounds):
        if len(pool_idx) < batch_size:
            break

        _log_round_start(r, n_rounds, len(pool_idx), len(training_idx))
        round_t0 = time.time()

        if encoded_dataset is not None and getattr(encoded_dataset, "raw_fingerprints", None):
            encoded_dataset.refit_pls(training_idx, external_df=encoded_df)

        X_train = encoded_df.loc[training_idx, feature_cols].values
        y_train_raw = encoded_df.loc[training_idx, "Experiment_value"].values.copy()

        if normalize == "copula":
            y_train = copula_transform(y_train_raw)
        elif normalize == "zscore":
            mu_t, sigma_t = y_train_raw.mean(), y_train_raw.std()
            y_train = (y_train_raw - mu_t) / sigma_t if sigma_t > 0 else y_train_raw.copy()
        else:
            y_train = y_train_raw.copy()

        X_pool = encoded_df.loc[pool_idx, feature_cols].values

        base_model = GaussianXGBQuantileModel(
            n_estimators=200,
            random_state=seed + r,
            n_jobs=1,
        ).fit(X_train, y_train)
        recal_dataset = build_recalibration_dataset(
            X_train,
            y_train,
            model_factory=lambda: GaussianXGBQuantileModel(
                n_estimators=200,
                random_state=seed + r,
                n_jobs=1,
            ),
        )
        recalibrator = ExactOnlineRecalibrator(eta=alpha).fit(recal_dataset)
        calibrated_model = CalibratedProbabilisticModel(base_model, recalibrator)
        q_level = recalibrator.recalibrate(p_ucb)
        scores = calibrated_model.quantile(X_pool, p_ucb)

        top_indices = np.argsort(scores)[-batch_size:][::-1]
        batch_idx = [pool_idx[i] for i in top_indices]

        batch_true_raw = encoded_df.loc[batch_idx, "Experiment_value"].values
        if normalize == "copula":
            batch_true = copula_transform(y_train_raw, x_new=batch_true_raw)
        elif normalize == "zscore":
            mu_t, sigma_t = y_train_raw.mean(), y_train_raw.std()
            batch_true = (batch_true_raw - mu_t) / sigma_t if sigma_t > 0 else batch_true_raw.copy()
        else:
            batch_true = batch_true_raw.copy()

        batch_quantiles = calibrated_model.quantile(
            encoded_df.loc[batch_idx, feature_cols].values,
            p_ucb,
        )
        coverage = float(np.mean(batch_true <= batch_quantiles))
        history["coverage"].append(float(coverage))
        history["conformal_quantile"].append(float(q_level))

        batch_set = set(batch_idx)
        pool_idx = [i for i in pool_idx if i not in batch_set]
        training_idx.extend(batch_idx)
        update_history(history, encoded_df, training_idx, batch_idx, r, top_k_values=top_k_values)

        _log_round_complete(
            r,
            encoded_df.loc[batch_idx, "Experiment_value"].max(),
            history["best_so_far"][-1],
            time.time() - round_t0,
            coverage=f"{coverage:.2f}",
            q=f"{q_level:.4f}",
            n_cal=len(recal_dataset.inverse_quantile_levels),
        )

    return history


def run_discrete_cumulative_split_conformal_ucb_baseline(
    encoded_df,
    feature_cols,
    seed_idx,
    oracle_idx,
    batch_size,
    n_rounds,
    seed,
    kappa=5.0,
    alpha=0.1,
    normalize="copula",
    encoded_dataset=None,
    top_k_values=None,
):
    """Run the legacy residual-accumulation split-conformal UCB baseline."""
    from sklearn.preprocessing import MinMaxScaler
    from xgboost import XGBRegressor

    from LNPBO.optimization._normalize import copula_transform
    from LNPBO.optimization.online_conformal import (
        CumulativeSplitConformalUCBBaseline,
        update_cumulative_split_conformal_batch,
    )

    training_idx = list(seed_idx)
    pool_idx = list(oracle_idx)
    history = init_history(encoded_df, training_idx, top_k_values=top_k_values)
    history["coverage"] = []
    history["conformal_quantile"] = []

    calibrator = CumulativeSplitConformalUCBBaseline(alpha=alpha)

    for r in range(n_rounds):
        if len(pool_idx) < batch_size:
            break

        _log_round_start(r, n_rounds, len(pool_idx), len(training_idx))
        round_t0 = time.time()

        if encoded_dataset is not None and getattr(encoded_dataset, "raw_fingerprints", None):
            encoded_dataset.refit_pls(training_idx, external_df=encoded_df)

        X_train = encoded_df.loc[training_idx, feature_cols].values
        y_train_raw = encoded_df.loc[training_idx, "Experiment_value"].values.copy()

        if normalize == "copula":
            y_train = copula_transform(y_train_raw)
        elif normalize == "zscore":
            mu_t, sigma_t = y_train_raw.mean(), y_train_raw.std()
            y_train = (y_train_raw - mu_t) / sigma_t if sigma_t > 0 else y_train_raw.copy()
        else:
            y_train = y_train_raw.copy()

        X_pool = encoded_df.loc[pool_idx, feature_cols].values

        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_pool_scaled = scaler.transform(X_pool)
        model = XGBRegressor(n_estimators=200, random_state=seed + r, n_jobs=1, verbosity=0)
        model.fit(X_train_scaled, y_train)

        y_pred_pool = model.predict(X_pool_scaled)
        q = calibrator.get_quantile()
        scores = y_pred_pool + kappa * q

        top_indices = np.argsort(scores)[-batch_size:][::-1]
        batch_idx = [pool_idx[i] for i in top_indices]
        batch_pred = y_pred_pool[top_indices]

        batch_true_raw = encoded_df.loc[batch_idx, "Experiment_value"].values
        if normalize == "copula":
            batch_true = copula_transform(y_train_raw, x_new=batch_true_raw)
        elif normalize == "zscore":
            mu_t, sigma_t = y_train_raw.mean(), y_train_raw.std()
            batch_true = (batch_true_raw - mu_t) / sigma_t if sigma_t > 0 else batch_true_raw.copy()
        else:
            batch_true = batch_true_raw.copy()

        coverage, q_after = update_cumulative_split_conformal_batch(calibrator, batch_pred, batch_true)
        history["coverage"].append(float(coverage))
        history["conformal_quantile"].append(float(q_after))

        batch_set = set(batch_idx)
        pool_idx = [i for i in pool_idx if i not in batch_set]
        training_idx.extend(batch_idx)
        update_history(history, encoded_df, training_idx, batch_idx, r, top_k_values=top_k_values)

        _log_round_complete(
            r,
            encoded_df.loc[batch_idx, "Experiment_value"].max(),
            history["best_so_far"][-1],
            time.time() - round_t0,
            coverage=f"{coverage:.2f}" if not np.isnan(coverage) else "nan",
            q=f"{q_after:.4f}" if np.isfinite(q_after) else "inf",
        )

    return history
