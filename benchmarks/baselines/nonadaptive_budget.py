#!/usr/bin/env python3
"""Non-Adaptive Same-Budget Baseline.

Tests whether adaptive BO is better than simply evaluating more
formulations up front. Draws n_seed + n_rounds * batch_size formulations
uniformly at random (or via Sobol quasi-random sequences) and checks
recall, with no surrogate and no refit loop.

Sobol quasi-random sequences provide low-discrepancy sampling:
    Sobol, I.M. (1967). "On the distribution of points in a cube and the
    approximate evaluation of integrals." USSR Comput. Math. Math. Phys.
    7(4), 86-112.

Usage:
    python -m benchmarks.baselines.nonadaptive_budget
    python -m benchmarks.baselines.nonadaptive_budget --subset 500
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from LNPBO.benchmarks.runner import prepare_benchmark_data
from LNPBO.runtime_paths import benchmark_results_root, package_root_from

from ..constants import SEEDS

_PACKAGE_ROOT = package_root_from(__file__, levels_up=3)


def compute_recall(selected_indices, top_k_values):
    """Compute Top-K recall for a set of selected indices."""
    selected = set(selected_indices)
    recall = {}
    for k, top_set in top_k_values.items():
        found = len(selected & top_set)
        recall[k] = found / len(top_set)
    return recall


def run_random_budget(df, total_budget, random_seed, top_k_values):
    """Draw total_budget formulations uniformly at random."""
    rng = np.random.RandomState(random_seed)
    all_idx = np.arange(len(df))
    budget = min(total_budget, len(df))
    selected = rng.choice(all_idx, size=budget, replace=False)
    recall = compute_recall(selected, top_k_values)
    best_val = float(df.loc[selected, "Experiment_value"].max())
    return recall, best_val, list(int(i) for i in selected)


def run_sobol_budget(df, feature_cols, total_budget, random_seed, top_k_values):
    """Select formulations via Sobol quasi-random sampling in feature space.

    Generates Sobol points in the feature hypercube and selects the
    nearest pool formulations to each Sobol point.
    """
    from scipy.stats.qmc import Sobol

    X = df[feature_cols].values
    n_features = X.shape[1]
    budget = min(total_budget, len(df))

    sampler = Sobol(d=n_features, scramble=True, seed=random_seed)
    # Generate more Sobol points than budget, then nearest-neighbor match
    n_sobol = budget * 2
    sobol_points = sampler.random(n_sobol)

    # Scale Sobol points to feature ranges
    feat_min = X.min(axis=0)
    feat_max = X.max(axis=0)
    feat_range = feat_max - feat_min
    feat_range[feat_range == 0] = 1.0
    sobol_scaled = sobol_points * feat_range + feat_min

    # Greedy nearest-neighbor assignment (no duplicates)
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nn.fit(X)
    _, nn_idx = nn.kneighbors(sobol_scaled)
    nn_idx = nn_idx.ravel()

    selected = []
    seen = set()
    for idx in nn_idx:
        if idx not in seen:
            selected.append(idx)
            seen.add(idx)
        if len(selected) >= budget:
            break

    # If not enough unique matches, fill with random
    if len(selected) < budget:
        remaining = [i for i in range(len(df)) if i not in seen]
        rng = np.random.RandomState(random_seed)
        extra = rng.choice(remaining, size=budget - len(selected), replace=False)
        selected.extend(extra)

    recall = compute_recall(selected, top_k_values)
    best_val = float(df.loc[selected, "Experiment_value"].max())
    return recall, best_val, list(int(i) for i in selected)


def main():
    parser = argparse.ArgumentParser(description="Non-Adaptive Same-Budget Baseline")
    parser.add_argument("--n-seed", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument("--subset", type=int, default=None)
    parser.add_argument("--feature-type", type=str, default="lantern_il_only")
    parser.add_argument("--reduction", type=str, default="pca")
    args = parser.parse_args()

    total_budget = args.n_seed + args.rounds * args.batch_size

    results_dir = benchmark_results_root(_PACKAGE_ROOT)
    results_dir.mkdir(exist_ok=True)

    print(f"{'=' * 70}")
    print("Non-Adaptive Same-Budget Baseline")
    print(f"Total budget: {total_budget} ({args.n_seed} seed + {args.rounds}x{args.batch_size} adaptive)")
    print(f"{'=' * 70}")

    methods = ["random", "sobol"]
    all_results = {}

    for method in methods:
        print(f"\n--- Method: {method} ---")
        seed_metrics = []
        seed_details = {}

        for seed in SEEDS:
            print(f"  Seed {seed}...", end=" ")
            t0 = time.time()

            _, encoded_df, feature_cols, _, _, top_k_values = prepare_benchmark_data(
                n_seed=args.n_seed,
                random_seed=seed,
                subset=args.subset,
                reduction=args.reduction,
                feature_type=args.feature_type,
            )

            if method == "random":
                recall, best_val, selected = run_random_budget(
                    encoded_df,
                    total_budget,
                    seed,
                    top_k_values,
                )
            else:
                recall, best_val, selected = run_sobol_budget(
                    encoded_df,
                    feature_cols,
                    total_budget,
                    seed,
                    top_k_values,
                )

            elapsed = time.time() - t0
            metrics = {"recall": recall, "best_value": best_val}
            seed_metrics.append(metrics)
            seed_details[seed] = {
                "recall": recall,
                "best_value": best_val,
                "elapsed": elapsed,
                "n_selected": len(selected),
            }
            print(f"Top-50={recall[50]:.1%}, best={best_val:.3f} ({elapsed:.1f}s)")

        recall_arrays = {}
        for k in [10, 50, 100]:
            vals = [m["recall"][k] for m in seed_metrics]
            recall_arrays[k] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "values": vals,
            }

        best_vals = [m["best_value"] for m in seed_metrics]

        all_results[method] = {
            "config": {
                "total_budget": total_budget,
                "n_seed": args.n_seed,
                "batch_size": args.batch_size,
                "n_rounds": args.rounds,
                "seeds": SEEDS,
                "feature_type": args.feature_type,
                "reduction": args.reduction,
                "subset": args.subset,
            },
            "recall": recall_arrays,
            "best_value": {
                "mean": float(np.mean(best_vals)),
                "std": float(np.std(best_vals)),
            },
            "seed_results": seed_details,
        }

        print(f"\n  {method} Summary:")
        for k in [10, 50, 100]:
            r = recall_arrays[k]
            print(f"    Top-{k}: {r['mean']:.1%} +/- {r['std']:.1%}")

    output_path = results_dir / "baseline_nonadaptive.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
