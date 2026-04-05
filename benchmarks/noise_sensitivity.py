#!/usr/bin/env python3
"""Noise sensitivity analysis for within-study benchmark strategies.

Real LNP screening has substantial measurement noise (replicate CVs of 20-50%).
The main within-study benchmark uses a noiseless oracle (exact LNPDB values).
This script tests whether strategy rankings are robust when Gaussian noise is
injected into the oracle responses.

Design:
  - Seed pool data is kept clean (simulating reliable initial measurements).
  - When the BO selects a batch and the oracle reveals values, N(0, sigma)
    noise is added to each revealed Experiment_value.
  - The surrogate trains on noisy values; true top-5% is always defined by
    the clean (original) values.

Noise levels:
  - sigma=0.0  baseline (no noise, sanity check)
  - sigma=0.1  mild (~10% of z-score range)
  - sigma=0.3  moderate (~30% CV equivalent)
  - sigma=0.5  severe (~50% CV equivalent)

Usage:
    python -m benchmarks.noise_sensitivity --dry-run
    python -m benchmarks.noise_sensitivity
    python -m benchmarks.noise_sensitivity --studies 39060305,37985700
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
    STRATEGY_CONFIGS,
    STRATEGY_DISPLAY,
    _run_random,
    compute_metrics,
    init_history,
    strategy_to_optimizer_kwargs,
    update_history,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NOISE_LEVELS = [0.0, 0.1, 0.3, 0.5]

SEEDS = [42, 123, 456]

# 5 diverse PMIDs; multi-MT PMIDs resolve to cell-line sub-studies via characterize_studies
DEFAULT_PMIDS = [
    "39060305",  # HeLa(1200) + RAW264.7(1200) sub-studies
    "37985700",  # A549(1801) sub-study
    "36997680",  # A549(720) sub-study
    "38740955",  # HeLa(560) sub-study
    "35879315",  # N=1080, ratio-only (single MT)
]

DEFAULT_STRATEGIES = [
    "random",
    "discrete_xgb_ucb",
    "discrete_rf_ts",
    "discrete_ngboost_ucb",
    "lnpbo_logei",
]

RESULTS_DIR = Path(__file__).resolve().parent.parent / "benchmark_results" / "analysis" / "within_study" / "sensitivity"
PER_RUN_DIR = RESULTS_DIR / "noise_runs"


def _result_path(study_id, strategy, sigma, seed):
    """Path for a single per-run result file."""
    return PER_RUN_DIR / f"{study_id}" / f"{strategy}_sigma{sigma}_s{seed}.json"


# ---------------------------------------------------------------------------
# Noisy oracle wrappers
# ---------------------------------------------------------------------------


def _inject_noise(encoded_df, oracle_idx, sigma, rng):
    """Return a copy of encoded_df with noise added to oracle Experiment_value.

    Seed indices are untouched. Only rows in oracle_idx get N(0, sigma) noise.
    The returned df is a copy; the original is not modified.
    """
    noisy_df = encoded_df.copy()
    if sigma > 0.0:
        noise = rng.normal(0.0, sigma, size=len(oracle_idx))
        noisy_df.loc[oracle_idx, "Experiment_value"] = encoded_df.loc[oracle_idx, "Experiment_value"].values + noise
    return noisy_df


def _run_noisy_random(clean_df, noisy_df, seed_idx, oracle_idx, batch_size, n_rounds, seed, top_k_values):
    """Random baseline on the noisy df (noise doesn't affect random selection,
    but we track metrics against the clean top-k)."""
    return _run_random(clean_df, seed_idx, oracle_idx, batch_size, n_rounds, seed, top_k_values=top_k_values)


def _run_noisy_optimizer(
    optimizer,
    clean_df,
    noisy_df,
    feature_cols,
    seed_idx,
    oracle_idx,
    batch_size,
    n_rounds,
    encoded_dataset,
    top_k_values,
):
    """Multi-round acquisition loop where the surrogate trains on noisy values
    but metrics are computed against clean (original) values.

    Delegates batch selection to ``Optimizer.suggest_indices()``, passing the
    noisy df so the surrogate sees corrupted targets. History and recall are
    always tracked against the clean df to preserve the noise-sensitivity
    invariant.
    """
    optimizer.batch_size = batch_size

    training_idx = list(seed_idx)
    pool_idx = list(oracle_idx)
    history = init_history(clean_df, training_idx, top_k_values=top_k_values)

    for r in range(n_rounds):
        if len(pool_idx) < batch_size:
            break

        try:
            batch_idx = optimizer.suggest_indices(
                noisy_df,
                feature_cols,
                training_idx,
                pool_idx,
                round_num=r,
                encoded_dataset=encoded_dataset,
            )
        except Exception as e:
            print(f"  Round {r + 1}: suggest_indices failed ({e})", flush=True)
            import traceback

            traceback.print_exc()
            break

        if not batch_idx:
            break

        batch_set = set(batch_idx)
        pool_idx = [i for i in pool_idx if i not in batch_set]
        training_idx.extend(batch_idx)
        update_history(history, clean_df, training_idx, batch_idx, r, top_k_values=top_k_values)

    return history


# ---------------------------------------------------------------------------
# Single run dispatcher
# ---------------------------------------------------------------------------


def run_noisy_seed(strategy, random_seed, study_info, sigma, pca_data, gp_data=None):
    """Run a single strategy/seed/noise-level combination."""
    from LNPBO.optimization.optimizer import Optimizer

    batch_size = study_info["batch_size"]
    n_rounds = study_info["n_rounds"]
    normalize = "copula"
    kappa = 5.0
    xi = 0.01

    config = STRATEGY_CONFIGS[strategy]
    is_gp = config["type"] == "gp"

    if is_gp:
        if gp_data is None:
            return None
        s_dataset, s_df, s_fcols, s_seed, s_oracle, s_topk = gp_data
    else:
        s_dataset, s_df, s_fcols, s_seed, s_oracle, s_topk = pca_data

    # Create noisy copy: seed data stays clean, oracle data gets noise
    rng = np.random.RandomState(random_seed * 1000 + int(sigma * 100))
    noisy_df = _inject_noise(s_df, s_oracle, sigma, rng)

    t0 = time.time()

    if config["type"] == "random":
        history = _run_noisy_random(
            s_df,
            noisy_df,
            s_seed,
            s_oracle,
            batch_size,
            n_rounds,
            random_seed,
            s_topk,
        )
    else:
        opt_kwargs = strategy_to_optimizer_kwargs(strategy)
        optimizer = Optimizer(
            random_seed=random_seed,
            kappa=kappa,
            xi=xi,
            normalize=normalize,
            batch_size=batch_size,
            **opt_kwargs,
        )
        history = _run_noisy_optimizer(
            optimizer,
            s_df,
            noisy_df,
            s_fcols,
            s_seed,
            s_oracle,
            batch_size=batch_size,
            n_rounds=n_rounds,
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
# Analysis: rank correlation
# ---------------------------------------------------------------------------


def spearman_rank_correlation(ranks_a, ranks_b):
    """Compute Spearman rank correlation between two rank vectors."""
    n = len(ranks_a)
    if n < 2:
        return float("nan")
    d = np.array(ranks_a) - np.array(ranks_b)
    return 1.0 - 6.0 * np.sum(d**2) / (n * (n**2 - 1))


def compute_rankings(strategy_means):
    """Rank strategies by mean top-5% recall (1 = best). Returns dict."""
    sorted_strats = sorted(strategy_means.keys(), key=lambda s: strategy_means[s], reverse=True)
    return {s: rank + 1 for rank, s in enumerate(sorted_strats)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Noise sensitivity analysis for within-study benchmark strategies",
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
        help="Comma-separated strategies (default: random,xgb_ucb,rf_ts,ngboost_ucb,lnpbo_logei)",
    )
    parser.add_argument(
        "--noise-levels",
        type=str,
        default=None,
        help="Comma-separated noise sigmas (default: 0.0,0.1,0.3,0.5)",
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

    # Parse arguments
    study_pmids = [s.strip() for s in args.studies.split(",")] if args.studies else DEFAULT_PMIDS
    strategies = [s.strip() for s in args.strategies.split(",")] if args.strategies else list(DEFAULT_STRATEGIES)
    noise_levels = [float(x) for x in args.noise_levels.split(",")] if args.noise_levels else list(NOISE_LEVELS)
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else list(SEEDS)

    # Validate strategies
    for s in strategies:
        if s not in STRATEGY_CONFIGS:
            parser.error(f"Unknown strategy: {s}")

    # Load data
    from LNPBO.data.lnpdb_bridge import load_lnpdb_full

    print("Loading LNPDB...")
    dataset = load_lnpdb_full()
    df = dataset.df
    print(f"  {len(df):,} formulations from {df['Publication_PMID'].nunique()} studies")

    # Characterize and filter studies
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

    # Check if any GP strategies need separate data prep
    gp_strategies = [s for s in strategies if STRATEGY_CONFIGS[s].get("type") == "gp"]

    # Enumerate runs
    total_runs = len(study_infos) * len(strategies) * len(noise_levels) * len(seeds)

    print(f"\n{'=' * 70}")
    print("NOISE SENSITIVITY ANALYSIS")
    print(f"{'=' * 70}")
    print(f"Studies: {len(study_infos)} ({', '.join(get_study_id(si) for si in study_infos)})")
    print(f"Strategies: {len(strategies)} ({', '.join(strategies)})")
    print(f"Noise levels (sigma): {noise_levels}")
    print(f"Seeds: {seeds}")
    print(f"Total runs: {total_runs}")
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
        for si in study_infos:
            sid = get_study_id(si)
            for sigma in noise_levels:
                for strategy in strategies:
                    for seed in seeds:
                        count += 1
                        print(f"  {sid} / sigma={sigma} / {strategy} / seed={seed}")
        print(f"\nTotal: {count} runs")
        return

    # -----------------------------------------------------------------------
    # Run all combinations (per-run saving with resume support)
    # -----------------------------------------------------------------------

    PER_RUN_DIR.mkdir(parents=True, exist_ok=True)

    completed = 0
    skipped = 0
    for si in study_infos:
        sid = get_study_id(si)

        print(f"\n{'=' * 60}")
        print(f"Study {sid}: N={si['n_formulations']}, type={si['study_type']}")
        print(f"{'=' * 60}")

        (PER_RUN_DIR / sid).mkdir(parents=True, exist_ok=True)

        # Prepare data once per (study, seed) pair
        for seed in seeds:
            # Check if ALL runs for this (study, seed) already exist
            if args.resume:
                all_done = all(
                    _result_path(sid, s, sigma, seed).exists()
                    for sigma in noise_levels
                    for s in strategies
                )
                if all_done:
                    n_skip = len(noise_levels) * len(strategies)
                    skipped += n_skip
                    completed += n_skip
                    print(f"\n  seed={seed}: all {n_skip} runs exist, skipping data prep")
                    continue

            print(f"\n  Preparing data for seed={seed}...")
            try:
                pca_data = prepare_study_data(df, si, seed)
            except Exception as e:
                print(f"  FAILED to prepare PCA data: {e}")
                continue

            gp_data = None
            if gp_strategies:
                gp_feature_type = "lantern" if si["study_type"] != "ratio_only" else si["feature_type"]
                gp_si = {**si, "feature_type": gp_feature_type}
                try:
                    gp_data = prepare_study_data(df, gp_si, seed)
                except Exception as e:
                    print(f"  GP data prep failed: {e}")

            for sigma in noise_levels:
                for strategy in strategies:
                    completed += 1
                    rpath = _result_path(sid, strategy, sigma, seed)

                    if args.resume and rpath.exists():
                        skipped += 1
                        continue

                    display = STRATEGY_DISPLAY.get(strategy, strategy)
                    print(
                        f"\n  [{completed}/{total_runs}] {display} | sigma={sigma} | seed={seed}",
                        flush=True,
                    )

                    try:
                        result = run_noisy_seed(
                            strategy,
                            seed,
                            si,
                            sigma,
                            pca_data=pca_data,
                            gp_data=gp_data,
                        )
                        if result is None:
                            print("    SKIPPED (no compatible data)")
                            continue

                        recall_5 = result["metrics"]["top_k_recall"].get("5", 0)
                        print(f"    Top-5% recall={recall_5:.3f}, time={result['elapsed']:.1f}s")

                        # Save per-run result immediately
                        run_data = {
                            "study_id": sid,
                            "strategy": strategy,
                            "sigma": sigma,
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
    # Aggregation from per-run files
    # -----------------------------------------------------------------------

    print(f"\n{'=' * 70}")
    print("AGGREGATING FROM PER-RUN FILES")
    print(f"{'=' * 70}\n")

    # Reload all per-run results from disk (includes any prior runs)
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for run_file in PER_RUN_DIR.glob("*/*.json"):
        try:
            with open(run_file) as f:
                rd = json.load(f)
            results[rd["study_id"]][rd["sigma"]][rd["strategy"]].append(rd)
        except Exception:
            pass

    n_loaded = sum(
        len(v) for sid in results for sigma in results[sid] for v in results[sid][sigma].values()
    )
    print(f"Loaded {n_loaded} per-run result files")

    # Aggregate: strategy x noise_level -> mean top-5% recall across studies and seeds
    agg = defaultdict(lambda: defaultdict(list))

    for sid in results:
        for sigma in results[sid]:
            for strategy in results[sid][sigma]:
                for r in results[sid][sigma][strategy]:
                    recall_5 = r["metrics"]["top_k_recall"].get("5", 0)
                    agg[strategy][sigma].append(recall_5)

    # Print table
    sigma_strs = [f"sigma={s}" for s in noise_levels]
    header = f"{'Strategy':<30} " + " ".join(f"{s:>12}" for s in sigma_strs)
    print(header)
    print("-" * len(header))

    strategy_means_by_sigma = {}  # {sigma: {strategy: mean_recall}}
    for sigma in noise_levels:
        strategy_means_by_sigma[sigma] = {}

    for strategy in strategies:
        display = STRATEGY_DISPLAY.get(strategy, strategy)
        parts = [f"{display:<30}"]
        for sigma in noise_levels:
            vals = agg[strategy].get(sigma, [])
            if vals:
                mean_val = np.mean(vals)
                strategy_means_by_sigma[sigma][strategy] = mean_val
                parts.append(f"{mean_val:>11.3f}")
            else:
                parts.append(f"{'N/A':>12}")
        print(" ".join(parts))

    # Compute rank correlations
    print(f"\n{'=' * 70}")
    print("RANK STABILITY (Spearman correlation of strategy rankings)")
    print(f"{'=' * 70}\n")

    baseline_sigma = noise_levels[0]
    baseline_rankings = compute_rankings(strategy_means_by_sigma.get(baseline_sigma, {}))

    if baseline_rankings:
        print(f"Baseline rankings (sigma={baseline_sigma}):")
        for s, rank in sorted(baseline_rankings.items(), key=lambda x: x[1]):
            display = STRATEGY_DISPLAY.get(s, s)
            print(f"  {rank}. {display} (recall={strategy_means_by_sigma[baseline_sigma][s]:.3f})")

        print()
        rank_correlations = {}
        for sigma in noise_levels:
            if sigma == baseline_sigma:
                rank_correlations[sigma] = 1.0
                continue
            sigma_rankings = compute_rankings(strategy_means_by_sigma.get(sigma, {}))
            if not sigma_rankings:
                continue
            # Align rankings
            common = sorted(set(baseline_rankings) & set(sigma_rankings))
            if len(common) < 2:
                rank_correlations[sigma] = float("nan")
                continue
            ranks_base = [baseline_rankings[s] for s in common]
            ranks_sigma = [sigma_rankings[s] for s in common]
            rho = spearman_rank_correlation(ranks_base, ranks_sigma)
            rank_correlations[sigma] = rho

        print("Spearman rho vs baseline (sigma=0.0):")
        for sigma in noise_levels:
            rho = rank_correlations.get(sigma, float("nan"))
            print(f"  sigma={sigma}: rho={rho:.3f}")

    # -----------------------------------------------------------------------
    # Per-study breakdown
    # -----------------------------------------------------------------------

    print(f"\n{'=' * 70}")
    print("PER-STUDY BREAKDOWN (Top-5% Recall)")
    print(f"{'=' * 70}")

    per_study_data = {}
    for sid in sorted(results.keys()):
        print(f"\n  Study {sid}:")
        per_study_data[sid] = {}
        for sigma in noise_levels:
            per_study_data[sid][str(sigma)] = {}
            for strategy in strategies:
                seed_results = results[sid].get(sigma, {}).get(strategy, [])
                if seed_results:
                    recalls = [r["metrics"]["top_k_recall"].get("5", 0) for r in seed_results]
                    mean_r = np.mean(recalls)
                    per_study_data[sid][str(sigma)][strategy] = {
                        "mean": float(mean_r),
                        "std": float(np.std(recalls, ddof=1)) if len(recalls) > 1 else 0.0,
                        "n_seeds": len(recalls),
                    }
                    display = STRATEGY_DISPLAY.get(strategy, strategy)
                    se = np.std(recalls, ddof=1) if len(recalls) > 1 else 0
                    print(f"    sigma={sigma}, {display}: {mean_r:.3f} +/- {se:.3f}")

    # -----------------------------------------------------------------------
    # Degradation summary
    # -----------------------------------------------------------------------

    print(f"\n{'=' * 70}")
    print("DEGRADATION SUMMARY (absolute drop from sigma=0.0)")
    print(f"{'=' * 70}\n")

    baseline = strategy_means_by_sigma.get(0.0, {})
    for sigma in noise_levels:
        if sigma == 0.0:
            continue
        print(f"  sigma={sigma}:")
        current = strategy_means_by_sigma.get(sigma, {})
        for strategy in strategies:
            base_val = baseline.get(strategy)
            curr_val = current.get(strategy)
            if base_val is not None and curr_val is not None:
                drop = base_val - curr_val
                pct_drop = (drop / base_val * 100) if base_val > 0 else 0
                display = STRATEGY_DISPLAY.get(strategy, strategy)
                print(f"    {display:<30}: {curr_val:.3f} (drop={drop:+.3f}, {pct_drop:+.1f}%)")
        print()

    # -----------------------------------------------------------------------
    # Save JSON
    # -----------------------------------------------------------------------

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "noise_sensitivity.json"

    output = {
        "config": {
            "studies": [get_study_id(si) for si in study_infos],
            "strategies": strategies,
            "noise_levels": noise_levels,
            "seeds": seeds,
            "timestamp": datetime.now().isoformat(),
        },
        "aggregate": {},
        "per_study": per_study_data,
        "rank_correlations": {},
    }

    for strategy in strategies:
        output["aggregate"][strategy] = {}
        for sigma in noise_levels:
            vals = agg[strategy].get(sigma, [])
            output["aggregate"][strategy][str(sigma)] = {
                "mean_top5_recall": float(np.mean(vals)) if vals else None,
                "std_top5_recall": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "n_runs": len(vals),
            }

    if baseline_rankings:
        for sigma in noise_levels:
            rho = rank_correlations.get(sigma, float("nan"))
            output["rank_correlations"][str(sigma)] = float(rho) if not np.isnan(rho) else None

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
