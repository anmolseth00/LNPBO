#!/usr/bin/env python3
"""Oracle Ceiling Baseline.

Greedy selection using true oracle values as scores. At each round, pick
the batch_size candidates from the pool with the highest true
Experiment_value. This is the best any surrogate could achieve with
perfect predictions and establishes the theoretical maximum recall at
any given budget.

Usage:
    python -m benchmarks.baselines.oracle_ceiling
    python -m benchmarks.baselines.oracle_ceiling --batch-size 6 --rounds 30
    python -m benchmarks.baselines.oracle_ceiling --subset 500
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from LNPBO.benchmarks.runner import (
    compute_metrics,
    init_history,
    prepare_benchmark_data,
    update_history,
)
from LNPBO.runtime_paths import benchmark_results_root, package_root_from

from ..constants import SEEDS

_PACKAGE_ROOT = package_root_from(__file__, levels_up=3)

DEFAULT_CONFIGS = [
    {"batch_size": 12, "n_rounds": 15, "label": "b12_15r"},
    {"batch_size": 6, "n_rounds": 30, "label": "b6_30r"},
]


def run_oracle_ceiling(df, seed_idx, oracle_idx, top_k_values, batch_size, n_rounds):
    """Run oracle ceiling: pick top-batch_size by true value each round."""
    training_idx = list(seed_idx)
    pool_idx = list(oracle_idx)
    history = init_history(df, training_idx)

    for r in range(n_rounds):
        if len(pool_idx) < batch_size:
            break

        pool_values = df.loc[pool_idx, "Experiment_value"].values
        top_k = np.argsort(pool_values)[-batch_size:][::-1]
        batch_idx = [pool_idx[i] for i in top_k]

        batch_set = set(batch_idx)
        pool_idx = [i for i in pool_idx if i not in batch_set]
        training_idx.extend(batch_idx)
        update_history(history, df, training_idx, batch_idx, r)

        batch_best = df.loc[batch_idx, "Experiment_value"].max()
        cum_best = history["best_so_far"][-1]
        print(
            f"  Round {r + 1}: batch_best={batch_best:.3f}, cum_best={cum_best:.3f}",
            flush=True,
        )

    return history


def main():
    parser = argparse.ArgumentParser(description="Oracle Ceiling Baseline")
    parser.add_argument("--n-seed", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size (runs single config)")
    parser.add_argument("--rounds", type=int, default=None, help="Override rounds (runs single config)")
    parser.add_argument("--subset", type=int, default=None)
    parser.add_argument("--feature-type", type=str, default="lantern_il_only")
    parser.add_argument("--reduction", type=str, default="pca")
    args = parser.parse_args()

    if args.batch_size is not None and args.rounds is not None:
        configs = [
            {"batch_size": args.batch_size, "n_rounds": args.rounds, "label": f"b{args.batch_size}_{args.rounds}r"}
        ]
    else:
        configs = DEFAULT_CONFIGS

    results_dir = benchmark_results_root(_PACKAGE_ROOT)
    results_dir.mkdir(exist_ok=True)

    all_results = {}

    for cfg in configs:
        label = cfg["label"]
        batch_size = cfg["batch_size"]
        n_rounds = cfg["n_rounds"]
        print(f"\n{'=' * 70}")
        print(f"Oracle Ceiling: {label} (batch={batch_size}, rounds={n_rounds})")
        print(f"{'=' * 70}")

        seed_metrics = []
        seed_histories = {}

        for seed in SEEDS:
            print(f"\n--- Seed {seed} ---")
            t0 = time.time()

            _, encoded_df, _feature_cols, seed_idx, oracle_idx, top_k_values = prepare_benchmark_data(
                n_seed=args.n_seed,
                random_seed=seed,
                subset=args.subset,
                reduction=args.reduction,
                feature_type=args.feature_type,
            )

            history = run_oracle_ceiling(
                encoded_df,
                seed_idx,
                oracle_idx,
                top_k_values,
                batch_size,
                n_rounds,
            )
            metrics = compute_metrics(history, top_k_values, len(encoded_df))
            elapsed = time.time() - t0

            seed_metrics.append(metrics)
            seed_histories[seed] = {
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

        all_results[label] = {
            "config": {
                "batch_size": batch_size,
                "n_rounds": n_rounds,
                "n_seed": args.n_seed,
                "seeds": SEEDS,
                "feature_type": args.feature_type,
                "reduction": args.reduction,
                "subset": args.subset,
            },
            "recall": recall_arrays,
            "auc": {"mean": float(np.mean(auc_vals)), "std": float(np.std(auc_vals))},
            "seed_results": seed_histories,
        }

        print(f"\n--- {label} Summary ---")
        for k in [10, 50, 100]:
            r = recall_arrays[k]
            print(f"  Top-{k}: {r['mean']:.1%} +/- {r['std']:.1%}")

    output_path = results_dir / "baseline_oracle_ceiling.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
