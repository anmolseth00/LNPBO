"""Shared discrete candidate pool strategy loop."""


from LNPBO.optimization.discrete import score_candidate_pool

from .runner import copula_transform, init_history, update_history


def run_discrete_strategy(
    encoded_df,
    feature_cols,
    seed_idx,
    oracle_idx,
    surrogate,
    batch_size,
    n_rounds,
    seed,
    kappa=5.0,
    normalize="copula",
    encoded_dataset=None,
):
    training_idx = list(seed_idx)
    pool_idx = list(oracle_idx)
    history = init_history(encoded_df, training_idx)

    for r in range(n_rounds):
        if len(pool_idx) < batch_size:
            break

        # Prospective PLS: re-fit using only training targets to avoid leakage.
        if encoded_dataset is not None and getattr(encoded_dataset, "raw_fingerprints", None):
            encoded_dataset.refit_pls(training_idx, external_df=encoded_df)

        X_train = encoded_df.loc[training_idx, feature_cols].values
        y_train = encoded_df.loc[training_idx, "Experiment_value"].values

        # Normalize training targets (affects UCB/TS scale balance)
        if normalize == "copula":
            y_train = copula_transform(y_train)
        elif normalize == "zscore":
            mu, sigma = y_train.mean(), y_train.std()
            if sigma > 0:
                y_train = (y_train - mu) / sigma

        X_pool = encoded_df.loc[pool_idx, feature_cols].values

        top_k, _ = score_candidate_pool(
            X_train, y_train, X_pool,
            surrogate=surrogate,
            batch_size=batch_size,
            kappa=kappa,
            random_seed=seed + r,
        )

        batch_idx = [pool_idx[i] for i in top_k]

        batch_set = set(batch_idx)
        pool_idx = [i for i in pool_idx if i not in batch_set]
        training_idx.extend(batch_idx)
        update_history(history, encoded_df, training_idx, batch_idx, r)

        batch_best = encoded_df.loc[batch_idx, "Experiment_value"].max()
        cum_best = history["best_so_far"][-1]
        print(
            f"  Round {r+1}: batch_best={batch_best:.3f}, cum_best={cum_best:.3f}, n_new={len(batch_idx)}",
            flush=True,
        )

    return history
