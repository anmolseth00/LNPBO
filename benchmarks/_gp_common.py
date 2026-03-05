"""Shared GP-based strategy loop."""

import os
import sys
import warnings

from .runner import LNPDBOracle, init_history, normalize_targets, update_history


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
    from LNPBO.data.dataset import Dataset
    from LNPBO.optimization.bayesopt import perform_bayesian_optimization
    from LNPBO.space.formulation import FormulationSpace

    training_idx = list(seed_idx)
    pool_idx = list(oracle_idx)
    oracle = LNPDBOracle(encoded_df, feature_cols)
    history = init_history(encoded_df, training_idx)

    for r in range(n_rounds):
        if len(pool_idx) < batch_size:
            break

        train_df = encoded_df.loc[training_idx].copy().reset_index(drop=True)
        train_df["Formulation_ID"] = range(1, len(train_df) + 1)
        train_df["Round"] = 0

        normalize_targets(train_df, normalize)

        dataset = Dataset(
            train_df,
            source="lnpdb",
            name="benchmark_train",
            metadata=encoded_dataset.metadata,
            encoders=encoded_dataset.encoders,
            fitted_transformers=encoded_dataset.fitted_transformers,
        )

        space = FormulationSpace.from_dataset(dataset)

        try:
            _devnull = open(os.devnull, "w")
            _saved_stdout = sys.stdout
            sys.stdout = _devnull
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    suggestions = perform_bayesian_optimization(
                        data=dataset.df,
                        formulation_space=space,
                        round_number=r,
                        acq_type=acq_type,
                        BATCH_SIZE=batch_size,
                        RANDOM_STATE_SEED=seed,
                        KAPPA=kappa,
                        XI=xi,
                        verbose=0,
                        save_gp=False,
                    )
            finally:
                sys.stdout = _saved_stdout
                _devnull.close()
        except Exception as e:
            print(f"  Round {r}: BO failed ({e})", flush=True)
            import traceback
            traceback.print_exc()
            break

        suggestion_features = suggestions[feature_cols].values
        matched_idx = oracle.lookup(suggestion_features, pool_idx)

        unique_matched = list(dict.fromkeys(matched_idx))[:batch_size]

        match_set = set(unique_matched)
        pool_idx = [i for i in pool_idx if i not in match_set]
        training_idx.extend(unique_matched)
        update_history(history, encoded_df, training_idx, unique_matched, r)

        batch_best = encoded_df.loc[unique_matched, "Experiment_value"].max()
        cum_best = history["best_so_far"][-1]
        print(f"  Round {r+1}: batch_best={batch_best:.3f}, cum_best={cum_best:.3f}, n_new={len(unique_matched)}", flush=True)

    return history
