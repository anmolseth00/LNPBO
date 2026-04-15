#!/usr/bin/env python3
"""Diversity Selection Baseline.

Chemistry-specific baseline using greedy maximin diversity selection.
At each round, selects batch_size candidates that maximize the minimum
pairwise distance to already-evaluated formulations in feature space.
This is a "cover the space" strategy -- pure exploration with no
exploitation.

Maximin design criterion:
    Johnson, M.E., Moore, L.M. & Ylvisaker, D. (1990). "Minimax and
    maximin distance designs." J. Statist. Plann. Inference 26, 131-148.

Usage:
    python -m benchmarks.baselines.diversity_selection
    python -m benchmarks.baselines.diversity_selection --subset 500
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import pairwise_distances

from LNPBO.benchmarks.runner import (
    compute_metrics,
    init_history,
    prepare_benchmark_data,
    update_history,
)
from LNPBO.runtime_paths import benchmark_results_root, package_root_from

from ..constants import SEEDS

_PACKAGE_ROOT = package_root_from(__file__, levels_up=3)


def run_diversity_selection(
    encoded_df,
    feature_cols,
    seed_idx,
    oracle_idx,
    top_k_values,
    batch_size,
    n_rounds,
):
    """Greedy maximin diversity selection at each round."""
    training_idx = list(seed_idx)
    pool_idx = list(oracle_idx)
    history = init_history(encoded_df, training_idx)

    X_all = encoded_df[feature_cols].values

    for r in range(n_rounds):
        if len(pool_idx) < batch_size:
            break

        X_train = X_all[training_idx]
        X_pool = X_all[pool_idx]

        # Greedy maximin: iteratively pick pool candidate with largest
        # minimum distance to all already-selected + training points
        selected_in_round = []
        remaining_pool = list(range(len(pool_idx)))

        # Precompute distances from pool to training set
        dist_to_train = pairwise_distances(X_pool, X_train, metric="euclidean")
        min_dist = dist_to_train.min(axis=1)  # shape: (n_pool,)

        for b in range(batch_size):
            if not remaining_pool:
                break

            # Among remaining pool candidates, pick the one with largest
            # minimum distance to training + already-selected-this-round
            remaining_arr = np.array(remaining_pool)
            best_local = remaining_arr[np.argmax(min_dist[remaining_arr])]
            selected_in_round.append(best_local)
            remaining_pool.remove(best_local)

            # Update min_dist: compute distance from remaining to newly selected
            if remaining_pool and b < batch_size - 1:
                new_point = X_pool[best_local].reshape(1, -1)
                dist_to_new = pairwise_distances(
                    X_pool[remaining_pool],
                    new_point,
                    metric="euclidean",
                ).ravel()
                # Update min_dist for remaining candidates
                for i, pool_local_idx in enumerate(remaining_pool):
                    min_dist[pool_local_idx] = min(
                        min_dist[pool_local_idx],
                        dist_to_new[i],
                    )

        batch_idx = [pool_idx[i] for i in selected_in_round]
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
    parser = argparse.ArgumentParser(description="Diversity Selection Baseline")
    parser.add_argument("--n-seed", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument("--subset", type=int, default=None)
    parser.add_argument("--feature-type", type=str, default="lantern_il_only")
    parser.add_argument("--reduction", type=str, default="pca")
    args = parser.parse_args()

    results_dir = benchmark_results_root(_PACKAGE_ROOT)
    results_dir.mkdir(exist_ok=True)

    print(f"{'=' * 70}")
    print("Diversity Selection Baseline (Greedy Maximin)")
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

        history = run_diversity_selection(
            encoded_df,
            feature_cols,
            seed_idx,
            oracle_idx,
            top_k_values,
            args.batch_size,
            args.rounds,
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

    output_path = results_dir / "baseline_diversity.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
