#!/usr/bin/env python3
"""COMET baseline: compute benchmark metrics from predictions.

This script takes COMET prediction files (from comet_infer.py) and the
exported study data (from comet_wrapper.py --export) and computes top-k%
recall metrics comparable to the within-study benchmark.

This can run in the LNPBO venv OR standalone (only needs numpy).

Usage:
    python -m benchmarks.baselines.comet_baseline
    python -m benchmarks.baselines.comet_baseline --predictions-dir /path/to/predictions
    python -m benchmarks.baselines.comet_baseline --aggregate-only
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from LNPBO.runtime_paths import benchmark_results_root, package_root_from

_PACKAGE_ROOT = package_root_from(__file__, levels_up=3)
RESULTS_DIR = benchmark_results_root(_PACKAGE_ROOT) / "baselines" / "comet"
EXPORT_DIR = RESULTS_DIR / "exported_data"
PRED_DIR = RESULTS_DIR / "predictions"

BATCH_SIZE = 12
MAX_ROUNDS = 15


def compute_comet_metrics(exported_path, pred_path, batch_size=BATCH_SIZE, n_rounds=MAX_ROUNDS):
    """Compute top-k% recall from COMET predictions and exported data.

    Parameters
    ----------
    exported_path : str or Path
        Path to exported study JSON (from comet_wrapper.py --export).
    pred_path : str or Path
        Path to prediction JSON (from comet_infer.py).
    batch_size : int
        Batch size per round.
    n_rounds : int
        Number of rounds.

    Returns
    -------
    dict
        Metrics including top_k_recall and per-round tracking.
    """
    with open(exported_path) as f:
        data = json.load(f)
    with open(pred_path) as f:
        pred_data = json.load(f)

    seed_entries = data["formulations"]["seed"]
    pool_entries = data["formulations"]["pool"]

    # Combine all formulations to compute top-k% thresholds
    all_entries = seed_entries + pool_entries
    all_labels = [e["label"] for e in all_entries]
    all_idx = [e["idx"] for e in all_entries]

    # Build score lookup from predictions
    score_lookup = {p["idx"]: p["score"] for p in pred_data["predictions"]}

    # Compute top-k% sets across ALL formulations (seed + pool)
    n_total = len(all_entries)
    sorted_by_label = sorted(zip(all_labels, all_idx), reverse=True)

    top_k_sets = {}
    for pct in [5, 10, 20]:
        k = max(1, int(np.ceil(n_total * pct / 100)))
        top_k_sets[pct] = set(idx for _, idx in sorted_by_label[:k])

    # Seed set (already "discovered" formulations)
    seed_idx_set = set(e["idx"] for e in seed_entries)

    # Score and rank pool entries
    pool_scores = []
    for entry in pool_entries:
        score = score_lookup.get(entry["idx"], 0.0)
        pool_scores.append((entry["idx"], score))

    # Sort by score descending (higher predicted score = better)
    pool_scores.sort(key=lambda x: -x[1])

    # Simulate round-by-round selection
    total_budget = n_rounds * batch_size
    discovered = set(seed_idx_set)

    # Track metrics per round
    best_so_far = []
    n_evaluated = []
    top_k_recall_per_round = {pct: [] for pct in [5, 10, 20]}

    # Initial state (seed only)
    seed_labels = {e["idx"]: e["label"] for e in seed_entries}
    best_label = max(seed_labels.values()) if seed_labels else float("-inf")
    best_so_far.append(float(best_label))
    n_evaluated.append(len(seed_idx_set))

    for pct in [5, 10, 20]:
        recall = len(discovered & top_k_sets[pct]) / len(top_k_sets[pct])
        top_k_recall_per_round[pct].append(recall)

    pool_label_lookup = {e["idx"]: e["label"] for e in pool_entries}

    n_selected = 0
    for r in range(n_rounds):
        start = r * batch_size
        end = min((r + 1) * batch_size, len(pool_scores), total_budget)
        if start >= end:
            break

        batch = pool_scores[start:end]
        for idx, _ in batch:
            discovered.add(idx)
            label = pool_label_lookup.get(idx, 0.0)
            best_label = max(best_label, label)
        n_selected += len(batch)

        best_so_far.append(float(best_label))
        n_evaluated.append(len(seed_idx_set) + n_selected)

        for pct in [5, 10, 20]:
            recall = len(discovered & top_k_sets[pct]) / len(top_k_sets[pct])
            top_k_recall_per_round[pct].append(recall)

    # Final metrics (after all rounds)
    final_recall = {}
    for pct in [5, 10, 20]:
        final_recall[str(pct)] = top_k_recall_per_round[pct][-1]

    return {
        "top_k_recall": final_recall,
        "top_k_recall_per_round": {str(k): v for k, v in top_k_recall_per_round.items()},
        "best_so_far": best_so_far,
        "n_evaluated": n_evaluated,
        "n_pool": len(pool_entries),
        "n_seed": len(seed_entries),
        "n_total": n_total,
        "n_predictions": len(pred_data["predictions"]),
    }


def main():
    parser = argparse.ArgumentParser(description="Compute COMET baseline metrics")
    parser.add_argument(
        "--predictions-dir",
        type=str,
        default=str(PRED_DIR),
        help="Directory with prediction JSONs from comet_infer.py",
    )
    parser.add_argument(
        "--exported-dir",
        type=str,
        default=str(EXPORT_DIR),
        help="Directory with exported study JSONs from comet_wrapper.py --export",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(RESULTS_DIR),
        help="Directory for output metric JSONs",
    )
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    pred_dir = Path(args.predictions_dir)
    export_dir = Path(args.exported_dir)
    output_dir = Path(args.output_dir)

    if not pred_dir.exists():
        print(f"Predictions directory not found: {pred_dir}")
        return
    if not export_dir.exists():
        print(f"Exported data directory not found: {export_dir}")
        return

    # Find all prediction files
    pred_files = sorted(pred_dir.glob("*_zero_shot.json"))
    if not pred_files and not args.aggregate_only:
        print(f"No prediction files found in {pred_dir}")
        return

    # Process each prediction file
    results_by_study = {}

    if not args.aggregate_only:
        print(f"\n{'=' * 70}")
        print("COMET BASELINE - Computing Metrics")
        print(f"{'=' * 70}")
        print(f"Predictions: {pred_dir}")
        print(f"Exported data: {export_dir}")
        print(f"Output: {output_dir}")
        print(f"Files to process: {len(pred_files)}")

        for pred_path in pred_files:
            # Parse study_id and seed from filename: {study_id}_s{seed}_zero_shot.json
            stem = pred_path.stem.replace("_zero_shot", "")
            parts = stem.rsplit("_s", 1)
            if len(parts) != 2:
                print(f"  Skipping {pred_path.name}: cannot parse study_id/seed")
                continue
            study_id = parts[0]
            seed = int(parts[1])

            # Find corresponding exported file
            export_path = export_dir / f"{study_id}_s{seed}.json"
            if not export_path.exists():
                print(f"  Skipping {pred_path.name}: no exported data at {export_path}")
                continue

            # Check if already done
            out_path = output_dir / study_id / f"comet_zero_shot_s{seed}.json"
            if args.resume and out_path.exists():
                continue

            try:
                metrics = compute_comet_metrics(export_path, pred_path)

                recall_str = ", ".join(f"Top-{k}%={metrics['top_k_recall'].get(str(k), 0):.1%}" for k in [5, 10, 20])
                print(f"  {study_id}/s{seed}: {recall_str}")

                # Save result
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out = {
                    "baseline": "comet_zero_shot",
                    "study_id": study_id,
                    "seed": seed,
                    "result": {
                        "metrics": {
                            "top_k_recall": metrics["top_k_recall"],
                        },
                        "best_so_far": metrics["best_so_far"],
                        "n_evaluated": metrics["n_evaluated"],
                    },
                    "timestamp": datetime.now().isoformat(),
                }
                with open(out_path, "w") as f:
                    json.dump(out, f, indent=2, default=str)

                if study_id not in results_by_study:
                    results_by_study[study_id] = []
                results_by_study[study_id].append(metrics["top_k_recall"]["5"])

            except Exception as e:
                print(f"  FAILED {study_id}/s{seed}: {e}")
                import traceback

                traceback.print_exc()

    # Aggregation
    print(f"\n{'=' * 70}")
    print("AGGREGATION")
    print(f"{'=' * 70}")

    # Collect all results
    all_study_means = []
    all_study_data = []

    for study_dir in sorted(output_dir.iterdir()):
        if not study_dir.is_dir():
            continue
        study_id = study_dir.name
        if study_id in ("exported_data", "predictions"):
            continue

        vals = []
        for result_file in sorted(study_dir.glob("comet_zero_shot_s*.json")):
            with open(result_file) as f:
                data = json.load(f)
            recall = data["result"]["metrics"]["top_k_recall"]
            vals.append(recall.get("5", 0.0))

        if vals:
            m = float(np.mean(vals))
            all_study_means.append(m)
            all_study_data.append(
                {
                    "study_id": study_id,
                    "mean_top5": m,
                    "n_seeds": len(vals),
                    "per_seed": vals,
                }
            )

    if all_study_means:
        grand_mean = float(np.mean(all_study_means))

        # Bootstrap CI
        rng = np.random.RandomState(42)
        n_boot = 10000
        boot_means = []
        arr = np.array(all_study_means)
        for _ in range(n_boot):
            boot_sample = rng.choice(arr, size=len(arr), replace=True)
            boot_means.append(float(np.mean(boot_sample)))
        ci_lo = float(np.percentile(boot_means, 2.5))
        ci_hi = float(np.percentile(boot_means, 97.5))

        print(f"\n{'Baseline':<30} {'Mean Top-5%':>12} {'95% CI':>18} {'N studies':>10}")
        print("-" * 75)
        print(f"{'COMET Zero-Shot':<30} {grand_mean:>11.1%} [{ci_lo:.1%}, {ci_hi:.1%}] {len(all_study_means):>10}")

        # Also show per-study breakdown
        print("\nPer-study breakdown:")
        print(f"  {'Study':<15} {'Mean Top-5%':>12} {'N seeds':>10}")
        print(f"  {'-' * 40}")
        for sd in sorted(all_study_data, key=lambda x: -x["mean_top5"]):
            print(f"  {sd['study_id']:<15} {sd['mean_top5']:>11.1%} {sd['n_seeds']:>10}")

        # Save summary
        summary = {
            "comet_zero_shot": {
                "display": "COMET Zero-Shot",
                "grand_mean_top5": grand_mean,
                "ci_95": [ci_lo, ci_hi],
                "n_studies": len(all_study_means),
                "studies": all_study_data,
            }
        }
        summary_path = output_dir / "comet_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\nSummary saved to {summary_path}")
    else:
        print("No results found to aggregate.")


if __name__ == "__main__":
    main()
