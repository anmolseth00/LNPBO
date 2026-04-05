#!/usr/bin/env python3
"""Hyperparameter sensitivity analysis for tree-based surrogates.

Addresses the reviewer concern that the tree ensemble advantage in the
within-study benchmark might be due to better default hyperparameters
rather than genuinely better surrogates. Tests RF-TS, XGB-UCB, and
NGBoost-UCB across a grid of key hyperparameters on 5 diverse studies.

Hyperparameter grids:
  - RF:      n_estimators=[50, 100, 200, 500], max_depth=[5, 10, None]
  - XGBoost: n_estimators=[50, 100, 200, 500], max_depth=[3, 6, 10]
  - NGBoost: n_estimators=[50, 100, 200, 500]

Design:
  - 5 diverse studies (same as noise sensitivity)
  - 3 seeds (42, 123, 456)
  - Batch 12, up to 15 rounds, LANTERN encoding, PCA 5, copula normalization
  - Default hyperparameters (n_estimators=200) included in the grid for comparison

Usage:
    python -m benchmarks.hyperparam_sensitivity --dry-run
    python -m benchmarks.hyperparam_sensitivity
    python -m benchmarks.hyperparam_sensitivity --resume
    python -m benchmarks.hyperparam_sensitivity --studies 39060305,37985700
"""

import argparse
import json
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

from .benchmark import (
    characterize_studies,
    ensure_top_k_pct,
    get_study_id,
    prepare_study_data,
)
from .runner import (
    STRATEGY_DISPLAY,
    compute_metrics,
    strategy_to_optimizer_kwargs,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEEDS = [42, 123, 456]

DEFAULT_PMIDS = [
    "39060305",  # HeLa(1200) + RAW264.7(1200) sub-studies
    "37985700",  # A549(1801) sub-study
    "36997680",  # A549(720) sub-study
    "38740955",  # HeLa(560) sub-study
    "35879315",  # N=1080, ratio-only (single MT)
]

# Strategy key -> (surrogate name, use_ts_batch flag)
STRATEGY_MAP = {
    "discrete_rf_ts": ("rf_ts", False),
    "discrete_xgb_ucb": ("xgb_ucb", False),
    "discrete_ngboost_ucb": ("ngboost", False),
}

# Hyperparameter grids per surrogate
HYPERPARAM_GRIDS = {
    "discrete_rf_ts": [
        {"n_estimators": n, "max_depth": d}
        for n in [50, 100, 200, 500]
        for d in [5, 10, None]
    ],
    "discrete_xgb_ucb": [
        {"n_estimators": n, "max_depth": d}
        for n in [50, 100, 200, 500]
        for d in [3, 6, 10]
    ],
    "discrete_ngboost_ucb": [
        {"n_estimators": n}
        for n in [50, 100, 200, 500]
    ],
}

RESULTS_DIR = Path(__file__).resolve().parent.parent / "benchmark_results" / "analysis" / "within_study" / "sensitivity"
PER_RUN_DIR = RESULTS_DIR / "hyperparam_runs"


def _hp_tag(hp_dict):
    """Create a short filename-safe tag from a hyperparameter dict."""
    parts = []
    for k in sorted(hp_dict.keys()):
        v = hp_dict[k]
        parts.append(f"{k}={v}")
    return "_".join(parts)


def _result_path(study_id, strategy, hp_dict, seed):
    """Path for a single per-run result file."""
    tag = _hp_tag(hp_dict)
    return PER_RUN_DIR / study_id / f"{strategy}_{tag}_s{seed}.json"


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------


def run_hyperparam_seed(strategy, hp_dict, random_seed, study_info, pca_data):
    """Run a single strategy/hyperparameter/seed combination."""
    from LNPBO.optimization.optimizer import Optimizer

    from ._optimizer_runner import OptimizerRunner

    batch_size = study_info["batch_size"]
    n_rounds = study_info["n_rounds"]
    kappa = 5.0

    s_dataset, s_df, s_fcols, s_seed, s_oracle, s_topk = pca_data

    opt_kwargs = strategy_to_optimizer_kwargs(strategy)
    optimizer = Optimizer(
        random_seed=random_seed,
        kappa=kappa,
        normalize="copula",
        batch_size=batch_size,
        surrogate_kwargs=hp_dict,
        **opt_kwargs,
    )

    runner = OptimizerRunner(optimizer)

    t0 = time.time()
    history = runner.run(
        s_df,
        s_fcols,
        s_seed,
        s_oracle,
        n_rounds=n_rounds,
        batch_size=batch_size,
        encoded_dataset=s_dataset,
        top_k_values=s_topk,
    )

    elapsed = time.time() - t0
    metrics = compute_metrics(history, s_topk, len(s_df))
    metrics["top_k_recall"] = {str(k): v for k, v in metrics["top_k_recall"].items()}

    return {
        "metrics": metrics,
        "elapsed": elapsed,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter sensitivity analysis for tree-based surrogates",
    )
    parser.add_argument(
        "--studies",
        type=str,
        default=None,
        help="Comma-separated PMIDs (default: 5 diverse studies)",
    )
    parser.add_argument(
        "--strategies",
        type=str,
        default=None,
        help="Comma-separated strategies (default: discrete_rf_ts,discrete_xgb_ucb,discrete_ngboost_ucb)",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated random seeds (default: 42,123,456)",
    )
    parser.add_argument("--dry-run", action="store_true", help="List runs without executing")
    parser.add_argument("--resume", action="store_true", help="Skip runs with existing result files")
    args = parser.parse_args()

    study_pmids = [s.strip() for s in args.studies.split(",")] if args.studies else DEFAULT_PMIDS
    strategies = [s.strip() for s in args.strategies.split(",")] if args.strategies else list(STRATEGY_MAP.keys())
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else list(SEEDS)

    for s in strategies:
        if s not in STRATEGY_MAP:
            parser.error(f"Unknown strategy: {s}. Valid: {list(STRATEGY_MAP.keys())}")

    # Load data
    from LNPBO.data.lnpdb_bridge import load_lnpdb_full

    print("Loading LNPDB...")
    dataset = load_lnpdb_full()
    df = dataset.df
    print(f"  {len(df):,} formulations from {df['Publication_PMID'].nunique()} studies")

    all_study_infos = characterize_studies(df)
    ensure_top_k_pct(all_study_infos)

    # Build lookup (PMID -> list of sub-studies for multi-MT PMIDs)
    study_lookup = {}
    pmid_to_studies = {}
    for si in all_study_infos:
        sid = get_study_id(si)
        study_lookup[sid] = si
        pmid_str = str(int(float(si["pmid"])))
        pmid_to_studies.setdefault(pmid_str, []).append(si)

    study_infos = []
    for pmid_str in study_pmids:
        if pmid_str in study_lookup:
            study_infos.append(study_lookup[pmid_str])
        elif pmid_str in pmid_to_studies:
            study_infos.extend(pmid_to_studies[pmid_str])
        else:
            print(f"  WARNING: PMID {pmid_str} not found in qualifying studies, skipping")

    if not study_infos:
        print("No qualifying studies found.")
        return

    # Enumerate all runs
    all_runs = []
    for si in study_infos:
        sid = get_study_id(si)
        for strategy in strategies:
            for hp_dict in HYPERPARAM_GRIDS[strategy]:
                for seed in seeds:
                    all_runs.append((sid, si, strategy, hp_dict, seed))

    total_runs = len(all_runs)

    # Count per strategy
    per_strat = defaultdict(int)
    for _, _, strat, _, _ in all_runs:
        per_strat[strat] += 1

    print(f"\n{'=' * 70}")
    print("HYPERPARAMETER SENSITIVITY ANALYSIS")
    print(f"{'=' * 70}")
    print(f"Studies: {len(study_infos)} ({', '.join(get_study_id(si) for si in study_infos)})")
    print(f"Strategies: {len(strategies)} ({', '.join(strategies)})")
    print(f"Seeds: {seeds}")
    print(f"Total runs: {total_runs}")
    for strat, count in sorted(per_strat.items()):
        n_hp = len(HYPERPARAM_GRIDS[strat])
        display = STRATEGY_DISPLAY.get(strat, strat)
        print(f"  {display}: {n_hp} HP configs x {len(study_infos)} studies x {len(seeds)} seeds = {count}")

    print()
    for si in study_infos:
        sid = get_study_id(si)
        print(
            f"  {sid}: N={si['n_formulations']}, ILs={si['n_unique_il']}, "
            f"type={si['study_type']}, n_seed={si['n_seed']}, rounds={si['n_rounds']}"
        )

    if args.dry_run:
        print("\nDRY RUN -- would execute:")
        count = 0
        for sid, _, strategy, hp_dict, seed in all_runs:
            count += 1
            display = STRATEGY_DISPLAY.get(strategy, strategy)
            print(f"  [{count}/{total_runs}] {sid} / {display} / {_hp_tag(hp_dict)} / seed={seed}")
        print(f"\nTotal: {count} runs")
        return

    # -----------------------------------------------------------------------
    # Execute runs
    # -----------------------------------------------------------------------

    PER_RUN_DIR.mkdir(parents=True, exist_ok=True)

    completed = 0
    skipped = 0
    # Cache prepared data per (study_id, seed) to avoid redundant encoding
    data_cache = {}

    for sid, si, strategy, hp_dict, seed in all_runs:
        completed += 1
        rpath = _result_path(sid, strategy, hp_dict, seed)

        if args.resume and rpath.exists():
            skipped += 1
            continue

        # Prepare data (cached per study+seed)
        cache_key = (sid, seed)
        if cache_key not in data_cache:
            print(f"\n  Preparing data for {sid} seed={seed}...")
            try:
                data_cache[cache_key] = prepare_study_data(df, si, seed)
            except Exception as e:
                print(f"  FAILED to prepare data: {e}")
                data_cache[cache_key] = None

        pca_data = data_cache[cache_key]
        if pca_data is None:
            continue

        display = STRATEGY_DISPLAY.get(strategy, strategy)
        tag = _hp_tag(hp_dict)
        print(
            f"\n  [{completed}/{total_runs}] {display} | {tag} | {sid} | seed={seed}",
            flush=True,
        )

        (PER_RUN_DIR / sid).mkdir(parents=True, exist_ok=True)

        try:
            result = run_hyperparam_seed(strategy, hp_dict, seed, si, pca_data)

            recall_5 = result["metrics"]["top_k_recall"].get("5", 0)
            print(f"    Top-5% recall={recall_5:.3f}, time={result['elapsed']:.1f}s")

            run_data = {
                "study_id": sid,
                "strategy": strategy,
                "hyperparams": hp_dict,
                "hp_tag": tag,
                "seed": seed,
                "metrics": result["metrics"],
                "elapsed": result["elapsed"],
            }
            with open(rpath, "w") as f:
                json.dump(run_data, f, indent=2, default=str)

        except Exception as e:
            print(f"    FAILED: {e}")
            import traceback

            traceback.print_exc()

    if skipped:
        print(f"\n  Skipped {skipped} existing runs (--resume)")

    # -----------------------------------------------------------------------
    # Aggregation
    # -----------------------------------------------------------------------

    print(f"\n{'=' * 70}")
    print("AGGREGATING FROM PER-RUN FILES")
    print(f"{'=' * 70}\n")

    results = []
    for run_file in PER_RUN_DIR.glob("*/*.json"):
        try:
            with open(run_file) as f:
                results.append(json.load(f))
        except Exception:
            pass

    print(f"Loaded {len(results)} per-run result files")

    # Group by (strategy, hp_tag) -> list of recalls
    grouped = defaultdict(list)
    per_study_grouped = defaultdict(lambda: defaultdict(list))
    for r in results:
        key = (r["strategy"], r["hp_tag"])
        recall_5 = r["metrics"]["top_k_recall"].get("5", 0)
        grouped[key].append(recall_5)
        per_study_grouped[r["study_id"]][key].append(recall_5)

    # Print per-strategy tables
    for strategy in strategies:
        display = STRATEGY_DISPLAY.get(strategy, strategy)
        grid = HYPERPARAM_GRIDS[strategy]

        print(f"\n{'=' * 60}")
        print(f"{display} -- Hyperparameter Sensitivity")
        print(f"{'=' * 60}")

        # Build table rows
        rows = []
        for hp_dict in grid:
            tag = _hp_tag(hp_dict)
            key = (strategy, tag)
            vals = grouped.get(key, [])
            mean_r = np.mean(vals) if vals else float("nan")
            std_r = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
            is_default = _is_default(strategy, hp_dict)
            rows.append((hp_dict, tag, mean_r, std_r, len(vals), is_default))

        # Sort by mean recall descending
        rows.sort(key=lambda x: -x[2] if not np.isnan(x[2]) else -999)

        print(f"  {'Config':<40} {'Mean':>8} {'Std':>8} {'N':>5} {'Default':>8}")
        print(f"  {'-' * 40} {'-' * 8} {'-' * 8} {'-' * 5} {'-' * 8}")
        for _hp_dict, tag, mean_r, std_r, n, is_default in rows:
            marker = "  *" if is_default else ""
            print(f"  {tag:<40} {mean_r:>8.3f} {std_r:>8.3f} {n:>5}{marker}")

    # Per-study tables
    print(f"\n{'=' * 70}")
    print("PER-STUDY BREAKDOWN")
    print(f"{'=' * 70}")

    for sid in sorted(per_study_grouped.keys()):
        print(f"\n  Study {sid}:")
        for strategy in strategies:
            display = STRATEGY_DISPLAY.get(strategy, strategy)
            grid = HYPERPARAM_GRIDS[strategy]
            rows = []
            for hp_dict in grid:
                tag = _hp_tag(hp_dict)
                key = (strategy, tag)
                vals = per_study_grouped[sid].get(key, [])
                mean_r = np.mean(vals) if vals else float("nan")
                rows.append((tag, mean_r, _is_default(strategy, hp_dict)))
            rows.sort(key=lambda x: -x[1] if not np.isnan(x[1]) else -999)
            best_tag, best_mean, _ = rows[0] if rows else ("", 0, False)
            default_mean = next((m for t, m, d in rows if d), float("nan"))
            diff = best_mean - default_mean if not (np.isnan(best_mean) or np.isnan(default_mean)) else 0
            print(f"    {display}: best={best_mean:.3f} ({best_tag}), default={default_mean:.3f}, gap={diff:+.3f}")

    # -----------------------------------------------------------------------
    # Robustness summary
    # -----------------------------------------------------------------------

    print(f"\n{'=' * 70}")
    print("ROBUSTNESS SUMMARY")
    print(f"{'=' * 70}\n")

    for strategy in strategies:
        display = STRATEGY_DISPLAY.get(strategy, strategy)
        grid = HYPERPARAM_GRIDS[strategy]

        all_means = []
        default_mean = None
        for hp_dict in grid:
            tag = _hp_tag(hp_dict)
            key = (strategy, tag)
            vals = grouped.get(key, [])
            if vals:
                m = np.mean(vals)
                all_means.append(m)
                if _is_default(strategy, hp_dict):
                    default_mean = m

        if all_means:
            range_val = max(all_means) - min(all_means)
            best_val = max(all_means)
            worst_val = min(all_means)
            print(f"  {display}:")
            print(f"    Range of mean recall: {worst_val:.3f} - {best_val:.3f} (spread={range_val:.3f})")
            if default_mean is not None:
                rank = sorted(all_means, reverse=True).index(default_mean) + 1
                print(f"    Default (n_est=200) recall: {default_mean:.3f} (rank {rank}/{len(all_means)})")
            print()

    # -----------------------------------------------------------------------
    # Save JSON
    # -----------------------------------------------------------------------

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "hyperparam_sensitivity.json"

    output = {
        "config": {
            "studies": [get_study_id(si) for si in study_infos],
            "strategies": strategies,
            "seeds": seeds,
            "grids": {s: HYPERPARAM_GRIDS[s] for s in strategies},
            "timestamp": datetime.now().isoformat(),
        },
        "aggregate": {},
        "per_study": {},
    }

    for strategy in strategies:
        output["aggregate"][strategy] = {}
        for hp_dict in HYPERPARAM_GRIDS[strategy]:
            tag = _hp_tag(hp_dict)
            key = (strategy, tag)
            vals = grouped.get(key, [])
            output["aggregate"][strategy][tag] = {
                "hyperparams": hp_dict,
                "mean_top5_recall": float(np.mean(vals)) if vals else None,
                "std_top5_recall": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "n_runs": len(vals),
                "is_default": _is_default(strategy, hp_dict),
            }

    for sid in sorted(per_study_grouped.keys()):
        output["per_study"][sid] = {}
        for strategy in strategies:
            output["per_study"][sid][strategy] = {}
            for hp_dict in HYPERPARAM_GRIDS[strategy]:
                tag = _hp_tag(hp_dict)
                key = (strategy, tag)
                vals = per_study_grouped[sid].get(key, [])
                output["per_study"][sid][strategy][tag] = {
                    "hyperparams": hp_dict,
                    "mean_top5_recall": float(np.mean(vals)) if vals else None,
                    "n_runs": len(vals),
                }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


def _is_default(strategy, hp_dict):
    """Check if a hyperparameter config matches the production defaults."""
    if strategy == "discrete_rf_ts":
        return hp_dict.get("n_estimators") == 200 and hp_dict.get("max_depth") is None
    elif strategy == "discrete_xgb_ucb":
        return hp_dict.get("n_estimators") == 200 and hp_dict.get("max_depth") == 6
    elif strategy == "discrete_ngboost_ucb":
        return hp_dict.get("n_estimators") == 200
    return False


if __name__ == "__main__":
    main()
