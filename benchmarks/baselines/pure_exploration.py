#!/usr/bin/env python3
"""Pure Exploration Baseline.

Separates exploitation from exploration. Fits a Random Forest with UQ,
then selects the batch_size candidates with the highest uncertainty
(sigma only, ignoring mean). This tests whether selecting the most
uncertain candidates finds good formulations.

Also runs RF-UCB (mu + kappa*sigma) for comparison, so the exploration
vs exploitation contribution can be decomposed.

Usage:
    python -m benchmarks.baselines.pure_exploration
    python -m benchmarks.baselines.pure_exploration --subset 500
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

from LNPBO.benchmarks.runner import (
    compute_metrics,
    init_history,
    prepare_benchmark_data,
    update_history,
)
from LNPBO.optimization._normalize import copula_transform

from ..constants import SEEDS


def run_pure_exploration(
    encoded_df,
    feature_cols,
    seed_idx,
    oracle_idx,
    top_k_values,
    batch_size,
    n_rounds,
    seed,
    normalize="copula",
):
    """RF surrogate selecting top-batch_size by sigma (uncertainty only)."""
    training_idx = list(seed_idx)
    pool_idx = list(oracle_idx)
    history = init_history(encoded_df, training_idx)

    for r in range(n_rounds):
        if len(pool_idx) < batch_size:
            break

        X_train = encoded_df.loc[training_idx, feature_cols].values
        y_train = encoded_df.loc[training_idx, "Experiment_value"].values

        if normalize == "copula":
            y_train = copula_transform(y_train)
        elif normalize == "zscore":
            mu, sigma = y_train.mean(), y_train.std()
            if sigma > 0:
                y_train = (y_train - mu) / sigma

        scaler = MinMaxScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_pool = encoded_df.loc[pool_idx, feature_cols].values
        X_pool_s = scaler.transform(X_pool)

        rf = RandomForestRegressor(
            n_estimators=200,
            random_state=seed + r,
            n_jobs=-1,
        )
        rf.fit(X_train_s, y_train)
        tree_preds = np.array([t.predict(X_pool_s) for t in rf.estimators_])
        sigma = tree_preds.std(axis=0)

        # Select by sigma only (pure exploration)
        top_k = np.argsort(sigma)[-batch_size:][::-1]
        batch_idx = [pool_idx[i] for i in top_k]

        batch_set = set(batch_idx)
        pool_idx = [i for i in pool_idx if i not in batch_set]
        training_idx.extend(batch_idx)
        update_history(history, encoded_df, training_idx, batch_idx, r)

        batch_best = encoded_df.loc[batch_idx, "Experiment_value"].max()
        cum_best = history["best_so_far"][-1]
        print(
            f"  Round {r + 1}: batch_best={batch_best:.3f}, cum_best={cum_best:.3f}",
            flush=True,
        )

    return history


def main():
    parser = argparse.ArgumentParser(description="Pure Exploration Baseline")
    parser.add_argument("--n-seed", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument("--subset", type=int, default=None)
    parser.add_argument("--feature-type", type=str, default="lantern_il_only")
    parser.add_argument("--reduction", type=str, default="pca")
    parser.add_argument("--normalize", type=str, default="copula", choices=["none", "zscore", "copula"])
    args = parser.parse_args()

    results_dir = Path(__file__).resolve().parent.parent.parent / "benchmark_results"
    results_dir.mkdir(exist_ok=True)

    print(f"{'=' * 70}")
    print("Pure Exploration Baseline (RF sigma-only)")
    print(f"n_seed={args.n_seed}, batch={args.batch_size}, rounds={args.rounds}")
    print(f"{'=' * 70}")

    seed_metrics = []
    seed_details = {}

    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")
        t0 = time.time()

        _, encoded_df, feature_cols, seed_idx, oracle_idx, top_k_values = prepare_benchmark_data(
            n_seed=args.n_seed,
            random_seed=seed,
            subset=args.subset,
            reduction=args.reduction,
            feature_type=args.feature_type,
        )

        history = run_pure_exploration(
            encoded_df,
            feature_cols,
            seed_idx,
            oracle_idx,
            top_k_values,
            args.batch_size,
            args.rounds,
            seed,
            normalize=args.normalize,
        )
        metrics = compute_metrics(history, top_k_values, len(encoded_df))
        elapsed = time.time() - t0

        seed_metrics.append(metrics)
        seed_details[seed] = {
            "best_so_far": history["best_so_far"],
            "round_best": history["round_best"],
            "n_evaluated": history["n_evaluated"],
            "metrics": metrics,
            "elapsed": elapsed,
        }

        print(f"  Top-K recall: {{{', '.join(f'{k}: {v:.1%}' for k, v in metrics['top_k_recall'].items())}}}")
        print(f"  Time: {elapsed:.1f}s")

    recall_arrays = {}
    for k in [10, 50, 100]:
        vals = [m["top_k_recall"][k] for m in seed_metrics]
        recall_arrays[k] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "values": vals,
        }

    auc_vals = [m["auc"] for m in seed_metrics]

    result = {
        "config": {
            "n_seed": args.n_seed,
            "batch_size": args.batch_size,
            "n_rounds": args.rounds,
            "seeds": SEEDS,
            "feature_type": args.feature_type,
            "reduction": args.reduction,
            "normalize": args.normalize,
            "subset": args.subset,
        },
        "recall": recall_arrays,
        "auc": {"mean": float(np.mean(auc_vals)), "std": float(np.std(auc_vals))},
        "seed_results": seed_details,
    }

    print("\n--- Summary ---")
    for k in [10, 50, 100]:
        r = recall_arrays[k]
        print(f"  Top-{k}: {r['mean']:.1%} +/- {r['std']:.1%}")

    output_path = results_dir / "baseline_pure_exploration.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
