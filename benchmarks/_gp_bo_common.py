"""GPyTorch-based GP BO benchmark integration.

Replaces _gp_common.py by using direct pool scoring via select_batch(),
eliminating FormulationSpace, continuous optimization, simplex projection,
and nearest-neighbor matching.
"""

import numpy as np

from LNPBO.optimization._normalize import normalize_targets

from .runner import init_history, update_history

ACQ_TYPE_MAP = {
    "UCB":       ("UCB", "kb"),
    "EI":        ("EI", "kb"),
    "LogEI":     ("LogEI", "kb"),
    "RKB_LogEI": ("LogEI", "rkb"),
    "RKB_UCB":   ("UCB", "rkb"),
    "RKB_EI":    ("EI", "rkb"),
    "LP_UCB":    ("UCB", "lp"),
    "LP_EI":     ("EI", "lp"),
    "LP_LogEI":  ("LogEI", "lp"),
    "TS_Batch":  ("UCB", "ts"),     # TS ignores acq_type; UCB is placeholder
    "GIBBON":    ("UCB", "gibbon"),  # info-theoretic; acq_type ignored
    "JES":       ("UCB", "jes"),    # info-theoretic; acq_type ignored
}


def run_gp_strategy(
    encoded_dataset,
    encoded_df,
    feature_cols,
    seed_idx,
    oracle_idx,
    acq_type,
    batch_size,
    n_rounds,
    seed,
    kappa=5.0,
    xi=0.01,
    normalize="copula",
):
    from LNPBO.optimization.gp_bo import select_batch

    if acq_type not in ACQ_TYPE_MAP:
        raise ValueError(
            f"Unknown acq_type {acq_type!r}. "
            f"Valid options: {list(ACQ_TYPE_MAP)}"
        )
    base_acq, batch_strategy = ACQ_TYPE_MAP[acq_type]

    training_idx = list(seed_idx)
    pool_idx = list(oracle_idx)
    history = init_history(encoded_df, training_idx)

    for r in range(n_rounds):
        if len(pool_idx) < batch_size:
            break

        # Prospective PLS refit (avoids target leakage)
        if getattr(encoded_dataset, "raw_fingerprints", None):
            encoded_dataset.refit_pls(training_idx, external_df=encoded_df)

        # Training data
        train_df = encoded_df.loc[training_idx].copy()
        normalize_targets(train_df, normalize)

        X_train = train_df[feature_cols].values.astype(np.float64)
        y_train = train_df["Experiment_value"].values.astype(np.float64)

        # Pool data
        X_pool = encoded_df.loc[pool_idx, feature_cols].values.astype(np.float64)
        pool_indices = np.array(pool_idx)

        # Drop rows with NaN/inf features
        valid_train = np.isfinite(X_train).all(axis=1)
        X_train = X_train[valid_train]
        y_train = y_train[valid_train]

        valid_pool = np.isfinite(X_pool).all(axis=1)
        X_pool = X_pool[valid_pool]
        pool_indices = pool_indices[valid_pool]

        if len(X_train) == 0 or len(X_pool) < batch_size:
            break

        # Select batch via direct pool scoring
        use_sparse = len(X_train) > 1000
        try:
            selected = select_batch(
                X_train, y_train, X_pool, pool_indices,
                batch_size=batch_size,
                acq_type=base_acq,
                batch_strategy=batch_strategy,
                kappa=kappa, xi=xi, seed=seed + r,
                use_sparse=use_sparse,
            )
        except Exception as e:
            print(f"  Round {r+1}: BO failed ({e})", flush=True)
            import traceback
            traceback.print_exc()
            break

        # Update indices
        match_set = set(selected)
        pool_idx = [i for i in pool_idx if i not in match_set]
        training_idx.extend(selected)
        update_history(history, encoded_df, training_idx, selected, r)

        batch_best = encoded_df.loc[selected, "Experiment_value"].max()
        cum_best = history["best_so_far"][-1]
        print(
            f"  Round {r+1}: batch_best={batch_best:.3f}, cum_best={cum_best:.3f},"
            f" n_new={len(selected)}",
            flush=True,
        )

    return history
