#!/usr/bin/env python3
"""Within-Study Benchmark.

Runs all strategies within individual LNPDB studies to evaluate optimization
performance under realistic conditions — same lab, same assay, same readout.

Cross-study pooling conflates batch effects with real structure-activity
relationships. Within-study evaluation isolates genuine optimization signal.

For each qualifying study:
  - n_seed is set dynamically (seed_fraction * study_size, min 30)
  - batch_size and n_rounds are kept consistent where pool permits
  - Top-k recall is computed at study-relative percentiles (top 5%, 10%, 20%)
  - Feature type is adapted: studies with 1 unique IL use ratios_only

Usage:
    # Run all strategies on all qualifying studies:
    python -m benchmarks.benchmark

    # Run specific strategies:
    python -m benchmarks.benchmark --strategies random,discrete_xgb_ucb

    # Run on specific studies:
    python -m benchmarks.benchmark --pmids 39060305,37985700

    # Resume (skip existing per-seed results):
    python -m benchmarks.benchmark --resume

    # Aggregate only (no new runs):
    python -m benchmarks.benchmark --aggregate-only

    # Dry run:
    python -m benchmarks.benchmark --dry-run
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from .runner import (
    STRATEGY_CONFIGS,
    STRATEGY_DISPLAY,
    TANIMOTO_STRATEGIES,
    _run_random,
    compute_metrics,
    prepare_benchmark_data,
)
from .stats import bootstrap_ci

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEEDS = [42, 123, 456, 789, 2024]

MIN_STUDY_SIZE = 200
SEED_FRACTION = 0.25
BATCH_SIZE = 12
MAX_ROUNDS = 15
MIN_SEED = 30

# Strategies to benchmark (same as comprehensive, minus TabPFN)
DISCRETE_STRATEGIES = [
    "random",
    "discrete_xgb_greedy",
    "discrete_xgb_ucb",
    "discrete_xgb_cqr",
    "discrete_ngboost_ucb",
    "discrete_rf_ucb",
    "discrete_rf_ts",
    "discrete_deep_ensemble",
    "discrete_gp_ucb",
]

TS_BATCH_STRATEGIES = [
    "discrete_rf_ts_batch",
    "discrete_xgb_ucb_ts_batch",
]

ONLINE_CONFORMAL_STRATEGIES = [
    "discrete_xgb_online_conformal",
]

CASMOPOLITAN_STRATEGIES = [
    "casmopolitan_ucb",
    "casmopolitan_ei",
]

GP_STRATEGIES = [
    "lnpbo_ucb",
    "lnpbo_ei",
    "lnpbo_logei",
    "lnpbo_rkb_logei",
    "lnpbo_lp_ei",
    "lnpbo_lp_logei",
    "lnpbo_pls_logei",
    "lnpbo_ts_batch",
    "lnpbo_pls_lp_logei",
    "lnpbo_gibbon",
    "lnpbo_jes",
    "lnpbo_tanimoto_ts",
    "lnpbo_tanimoto_logei",
]

ALL_WITHIN_STUDY_STRATEGIES = (
    DISCRETE_STRATEGIES
    + TS_BATCH_STRATEGIES
    + ONLINE_CONFORMAL_STRATEGIES
    + CASMOPOLITAN_STRATEGIES
    + GP_STRATEGIES
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "benchmark_results" / "within_study"


# ---------------------------------------------------------------------------
# Study characterization
# ---------------------------------------------------------------------------


def characterize_studies(df, min_size=MIN_STUDY_SIZE, seed_fraction=SEED_FRACTION):
    """Identify and characterize qualifying studies.

    Returns a list of dicts with study metadata.
    """
    study_counts = df.groupby("Publication_PMID").size().sort_values(ascending=False)
    studies = []

    for pmid, n in study_counts.items():
        if n < min_size:
            continue
        sub = df[df["Publication_PMID"] == pmid]
        n_unique_il = sub["IL_SMILES"].nunique()
        n_unique_hl = sub["HL_name"].nunique()
        n_unique_chl = sub["CHL_name"].nunique()
        n_unique_peg = sub["PEG_name"].nunique()
        il_ratio_std = sub["IL_molratio"].std() if "IL_molratio" in sub.columns else 0.0

        # Classify study type
        if n_unique_il <= 1:
            study_type = "ratio_only"
        elif il_ratio_std < 1.0:
            study_type = "il_diverse_fixed_ratios"
        else:
            study_type = "il_diverse_variable_ratios"

        # Dynamic n_seed
        n_seed = max(MIN_SEED, int(seed_fraction * n))

        # Adjust rounds: cap acquisitions at 50% of oracle pool so metrics
        # are meaningful (exploring >50% trivially finds top-k)
        oracle_size = n - n_seed
        max_acquisitions = int(0.5 * oracle_size)
        max_rounds_feasible = max(1, max_acquisitions // BATCH_SIZE)
        n_rounds = min(MAX_ROUNDS, max_rounds_feasible)

        # Feature type: no molecular features if only 1 IL
        if n_unique_il <= 1:
            feature_type = "ratios_only"
        else:
            feature_type = "lantern_il_only"

        # Top-k percentiles (relative to study size)
        top_k_abs = {}
        for pct in [5, 10, 20]:
            k = max(1, int(n * pct / 100))
            top_k_abs[pct] = k

        studies.append({
            "pmid": pmid,
            "n_formulations": int(n),
            "n_unique_il": int(n_unique_il),
            "n_unique_hl": int(n_unique_hl),
            "n_unique_chl": int(n_unique_chl),
            "n_unique_peg": int(n_unique_peg),
            "il_ratio_std": float(il_ratio_std),
            "study_type": study_type,
            "n_seed": n_seed,
            "n_rounds": n_rounds,
            "batch_size": BATCH_SIZE,
            "feature_type": feature_type,
            "top_k_pct": top_k_abs,
        })

    return studies


# ---------------------------------------------------------------------------
# Per-seed runner (mirrors comprehensive_benchmark.run_single_seed)
# ---------------------------------------------------------------------------




def run_single_seed(
    strategy, random_seed, study_info, pca_data=None,
):
    """Run a single strategy for a single seed within a study."""
    batch_size = study_info["batch_size"]
    n_rounds = study_info["n_rounds"]
    normalize = "copula"
    kappa = 5.0
    xi = 0.01

    if pca_data is None:
        raise ValueError("pca_data must be provided")

    s_dataset, s_df, s_fcols, s_seed, s_oracle, s_topk = pca_data
    config = STRATEGY_CONFIGS[strategy]
    t0 = time.time()

    if config["type"] == "random":
        history = _run_random(s_df, s_seed, s_oracle, batch_size, n_rounds, random_seed)

    elif config["type"] == "discrete":
        from ._discrete_common import run_discrete_strategy
        history = run_discrete_strategy(
            s_df, s_fcols, s_seed, s_oracle,
            surrogate=config["surrogate"], batch_size=batch_size,
            n_rounds=n_rounds, seed=random_seed, kappa=kappa,
            normalize=normalize, encoded_dataset=s_dataset,
        )

    elif config["type"] == "discrete_ts_batch":
        from ._discrete_common import run_discrete_ts_batch_strategy
        history = run_discrete_ts_batch_strategy(
            s_df, s_fcols, s_seed, s_oracle,
            surrogate=config["surrogate"], batch_size=batch_size,
            n_rounds=n_rounds, seed=random_seed, kappa=kappa,
            normalize=normalize, encoded_dataset=s_dataset,
        )

    elif config["type"] == "discrete_online_conformal":
        from ._discrete_common import run_discrete_online_conformal_strategy
        history = run_discrete_online_conformal_strategy(
            s_df, s_fcols, s_seed, s_oracle,
            batch_size=batch_size,
            n_rounds=n_rounds, seed=random_seed, kappa=kappa,
            normalize=normalize, encoded_dataset=s_dataset,
        )

    elif config["type"] == "casmopolitan":
        from LNPBO.optimization.casmopolitan import run_casmopolitan_strategy
        history = run_casmopolitan_strategy(
            s_df, s_fcols, s_seed, s_oracle,
            batch_size=batch_size,
            n_rounds=n_rounds, seed=random_seed, kappa=kappa,
            normalize=normalize,
            acq_func=config.get("acq_func", "ucb"),
        )

    elif config["type"] == "gp":
        from ._gp_bo_common import run_gp_strategy
        history = run_gp_strategy(
            s_dataset, s_df, s_fcols, s_seed, s_oracle,
            acq_type=config["acq_type"], batch_size=batch_size,
            n_rounds=n_rounds, seed=random_seed, kappa=kappa,
            xi=xi, normalize=normalize,
        )

    else:
        raise ValueError(f"Unknown strategy type: {config['type']!r}")

    elapsed = time.time() - t0
    metrics = compute_metrics(history, s_topk, len(s_df))
    metrics["top_k_recall"] = {str(k): v for k, v in metrics["top_k_recall"].items()}

    result = {
        "metrics": metrics,
        "elapsed": elapsed,
        "best_so_far": history["best_so_far"],
        "round_best": history["round_best"],
        "n_evaluated": history["n_evaluated"],
    }
    if "coverage" in history:
        result["coverage"] = history["coverage"]

    return result


# ---------------------------------------------------------------------------
# Result I/O
# ---------------------------------------------------------------------------


def _per_seed_path(pmid, strategy, seed):
    pmid_str = str(int(float(pmid)))
    return RESULTS_DIR / pmid_str / f"{strategy}_s{seed}.json"


def save_seed_result(pmid, strategy, seed, result, study_info):
    path = _per_seed_path(pmid, strategy, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "pmid": pmid,
        "strategy": strategy,
        "seed": seed,
        "study_info": {k: v for k, v in study_info.items() if k != "top_k_pct"},
        "top_k_pct": study_info["top_k_pct"],
        "result": result,
        "timestamp": datetime.now().isoformat(),
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_seed_result(pmid, strategy, seed):
    path = _per_seed_path(pmid, strategy, seed)
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    return data.get("result")


# ---------------------------------------------------------------------------
# Data preparation (per study)
# ---------------------------------------------------------------------------


def prepare_study_data(df, study_info, random_seed):
    """Filter dataset to a single study and prepare benchmark data."""
    pmid = study_info["pmid"]
    study_df = df[df["Publication_PMID"] == pmid].copy().reset_index(drop=True)
    study_df["Formulation_ID"] = range(1, len(study_df) + 1)

    n_seed = study_info["n_seed"]
    feature_type = study_info["feature_type"]

    # Compute top-k sets at study-relative percentiles
    top_k_values = {}
    for pct, k in study_info["top_k_pct"].items():
        top_k_values[pct] = set(study_df.nlargest(k, "Experiment_value").index)

    pca_data = prepare_benchmark_data(
        n_seed=n_seed,
        random_seed=random_seed,
        reduction="pca" if feature_type != "ratios_only" else "none",
        feature_type=feature_type,
        data_df=study_df,
    )

    # Replace the top_k_values with our percentile-based ones
    encoded, encoded_df, feature_cols, seed_idx, oracle_idx, _ = pca_data

    # Recompute top-k on the encoded (valid-rows) df
    top_k_values_clean = {}
    for pct, k in study_info["top_k_pct"].items():
        actual_k = min(k, len(encoded_df))
        top_k_values_clean[pct] = set(encoded_df.nlargest(actual_k, "Experiment_value").index)

    return encoded, encoded_df, feature_cols, seed_idx, oracle_idx, top_k_values_clean


def prepare_study_data_raw(df, study_info, random_seed):
    """Like prepare_study_data but with reduction='none' for raw fingerprints.

    Used by Tanimoto kernel strategies that need the full-dimensional
    count Morgan fingerprint space (2048-d) rather than PCA-reduced features.
    """
    pmid = study_info["pmid"]
    study_df = df[df["Publication_PMID"] == pmid].copy().reset_index(drop=True)
    study_df["Formulation_ID"] = range(1, len(study_df) + 1)

    n_seed = study_info["n_seed"]
    feature_type = study_info["feature_type"]

    top_k_values = {}
    for pct, k in study_info["top_k_pct"].items():
        top_k_values[pct] = set(study_df.nlargest(k, "Experiment_value").index)

    raw_data = prepare_benchmark_data(
        n_seed=n_seed,
        random_seed=random_seed,
        reduction="none",
        feature_type=feature_type,
        data_df=study_df,
    )

    encoded, encoded_df, feature_cols, seed_idx, oracle_idx, _ = raw_data

    top_k_values_clean = {}
    for pct, k in study_info["top_k_pct"].items():
        actual_k = min(k, len(encoded_df))
        top_k_values_clean[pct] = set(encoded_df.nlargest(actual_k, "Experiment_value").index)

    return encoded, encoded_df, feature_cols, seed_idx, oracle_idx, top_k_values_clean


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate_study_strategy(pmid, strategy, seed_results, random_seed_results=None):
    """Aggregate per-seed results for one strategy within one study."""
    seeds_available = sorted(seed_results.keys())
    n_seeds = len(seeds_available)
    if n_seeds == 0:
        return None

    def _get_recall(seed_result, pct_key):
        recall_dict = seed_result["metrics"]["top_k_recall"]
        return recall_dict.get(str(pct_key), recall_dict.get(pct_key, 0.0))

    summary = {
        "strategy": strategy,
        "display_name": STRATEGY_DISPLAY.get(strategy, strategy),
        "pmid": pmid,
        "n_seeds": n_seeds,
    }

    for pct in [5, 10, 20]:
        vals = np.array([_get_recall(seed_results[s], pct) for s in seeds_available])
        ci_lo, ci_hi = bootstrap_ci(vals) if len(vals) >= 3 else (float("nan"), float("nan"))
        summary[f"top_{pct}pct_recall"] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)) if n_seeds > 1 else 0.0,
            "ci_low": ci_lo,
            "ci_high": ci_hi,
            "per_seed": {int(s): float(v) for s, v in zip(seeds_available, vals)},
        }

    auc_vals = np.array([seed_results[s]["metrics"]["auc"] for s in seeds_available])
    summary["auc"] = {
        "mean": float(np.mean(auc_vals)),
        "std": float(np.std(auc_vals, ddof=1)) if n_seeds > 1 else 0.0,
    }

    elapsed_vals = np.array([seed_results[s]["elapsed"] for s in seeds_available])
    summary["elapsed_seconds"] = {"mean": float(np.mean(elapsed_vals))}

    return summary


# ---------------------------------------------------------------------------
# Markdown output
# ---------------------------------------------------------------------------


def generate_within_study_markdown(all_summaries, study_infos):
    """Generate markdown summary across all studies."""
    lines = []
    lines.append("# Within-Study Benchmark Results")
    lines.append("")
    lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"**Seed fraction:** {SEED_FRACTION}")
    lines.append(f"**Batch size:** {BATCH_SIZE}")
    lines.append(f"**Max rounds:** {MAX_ROUNDS}")
    lines.append(f"**Min study size:** {MIN_STUDY_SIZE}")
    lines.append(f"**Seeds:** {SEEDS}")
    lines.append("**Normalize:** copula")
    lines.append("")

    # Study overview table
    lines.append("## Study Overview")
    lines.append("")
    lines.append("| PMID | N | ILs | HLs | Type | n_seed | Rounds | Features |")
    lines.append("|------|---|-----|-----|------|--------|--------|----------|")
    for si in study_infos:
        lines.append(
            f"| {int(float(si['pmid']))} | {si['n_formulations']} | "
            f"{si['n_unique_il']} | {si['n_unique_hl']} | "
            f"{si['study_type']} | {si['n_seed']} | {si['n_rounds']} | "
            f"{si['feature_type']} |"
        )
    lines.append("")

    # Per-study results
    for si in study_infos:
        pmid = si["pmid"]
        pmid_str = str(int(float(pmid)))
        study_summaries = all_summaries.get(pmid_str, {})
        if not study_summaries:
            continue

        lines.append(f"## PMID {pmid_str}")
        lines.append(f"N={si['n_formulations']}, {si['n_unique_il']} ILs, "
                      f"type={si['study_type']}, n_seed={si['n_seed']}, "
                      f"rounds={si['n_rounds']}, features={si['feature_type']}")
        lines.append("")

        ranked = sorted(
            study_summaries.items(),
            key=lambda x: x[1].get("top_10pct_recall", {}).get("mean", 0),
            reverse=True,
        )

        lines.append("| Rank | Strategy | Top-5% | Top-10% | Top-20% | AUC | Time (s) |")
        lines.append("|------|----------|--------|---------|---------|-----|----------|")

        for rank, (strat, s) in enumerate(ranked, 1):
            t5 = s.get("top_5pct_recall", {})
            t10 = s.get("top_10pct_recall", {})
            t20 = s.get("top_20pct_recall", {})
            auc = s.get("auc", {})
            elapsed = s.get("elapsed_seconds", {})

            t5_str = f"{t5.get('mean', 0):.1%}" if t5 else "N/A"
            t10_str = f"{t10.get('mean', 0):.1%} ±{t10.get('std', 0):.1%}" if t10 else "N/A"
            t20_str = f"{t20.get('mean', 0):.1%}" if t20 else "N/A"
            auc_str = f"{auc.get('mean', 0):.2f}" if auc else "N/A"
            time_str = f"{elapsed.get('mean', 0):.1f}" if elapsed else "N/A"

            lines.append(
                f"| {rank} | {s.get('display_name', strat)} | "
                f"{t5_str} | {t10_str} | {t20_str} | {auc_str} | {time_str} |"
            )

        lines.append("")

    # Cross-study summary: mean of means
    lines.append("## Cross-Study Summary (mean of per-study means)")
    lines.append("")

    strategy_means = {}
    for _pmid_str, study_summaries in all_summaries.items():
        for strat, s in study_summaries.items():
            if strat not in strategy_means:
                strategy_means[strat] = {
                    "top_5": [], "top_10": [], "top_20": [],
                    "display": s.get("display_name", strat),
                }
            t5 = s.get("top_5pct_recall", {}).get("mean", 0)
            t10 = s.get("top_10pct_recall", {}).get("mean", 0)
            t20 = s.get("top_20pct_recall", {}).get("mean", 0)
            strategy_means[strat]["top_5"].append(t5)
            strategy_means[strat]["top_10"].append(t10)
            strategy_means[strat]["top_20"].append(t20)

    ranked_overall = sorted(
        strategy_means.items(),
        key=lambda x: np.mean(x[1]["top_10"]) if x[1]["top_10"] else 0,
        reverse=True,
    )

    lines.append("| Rank | Strategy | Mean Top-10% | Std | N studies |")
    lines.append("|------|----------|-------------|-----|----------|")
    for rank, (_strat, d) in enumerate(ranked_overall, 1):
        mean_t10 = np.mean(d["top_10"]) if d["top_10"] else 0
        std_t10 = np.std(d["top_10"], ddof=1) if len(d["top_10"]) > 1 else 0
        lines.append(
            f"| {rank} | {d['display']} | {mean_t10:.1%} | {std_t10:.1%} | {len(d['top_10'])} |"
        )

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Within-Study Benchmark Suite",
    )
    parser.add_argument(
        "--strategies", type=str, default=None,
        help="Comma-separated list of strategies (default: all non-GP)",
    )
    parser.add_argument(
        "--pmids", type=str, default=None,
        help="Comma-separated list of PMIDs to benchmark (default: all qualifying)",
    )
    parser.add_argument(
        "--min-study-size", type=int, default=MIN_STUDY_SIZE,
        help=f"Minimum study size (default: {MIN_STUDY_SIZE})",
    )
    parser.add_argument(
        "--seed-fraction", type=float, default=SEED_FRACTION,
        help=f"Fraction of study used as seed (default: {SEED_FRACTION})",
    )
    parser.add_argument(
        "--seeds", type=str, default=None,
        help="Comma-separated seeds (default: 42,123,456,789,2024)",
    )
    parser.add_argument("--resume", action="store_true", help="Skip existing results")
    parser.add_argument("--aggregate-only", action="store_true", help="Only aggregate")
    parser.add_argument("--dry-run", action="store_true", help="List runs without executing")
    args = parser.parse_args()

    seed_fraction = args.seed_fraction
    min_study_size = args.min_study_size

    # Load full dataset
    from LNPBO.data.lnpdb_bridge import load_lnpdb_full
    print("Loading LNPDB...")
    dataset = load_lnpdb_full()
    df = dataset.df
    print(f"  {len(df):,} formulations from {df['Publication_PMID'].nunique()} studies")

    # Characterize studies
    study_infos = characterize_studies(df, min_size=min_study_size, seed_fraction=seed_fraction)
    print(f"\n{len(study_infos)} qualifying studies (>= {args.min_study_size} formulations)")

    # Filter PMIDs if specified
    if args.pmids:
        target_pmids = set(float(p.strip()) for p in args.pmids.split(","))
        study_infos = [si for si in study_infos if si["pmid"] in target_pmids]
        print(f"  Filtered to {len(study_infos)} requested studies")

    for si in study_infos:
        print(f"  PMID {int(float(si['pmid']))}: N={si['n_formulations']}, "
              f"ILs={si['n_unique_il']}, type={si['study_type']}, "
              f"n_seed={si['n_seed']}, rounds={si['n_rounds']}, "
              f"features={si['feature_type']}")

    # Parse strategies
    seeds = SEEDS
    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(",")]

    if args.strategies:
        strategies = [s.strip() for s in args.strategies.split(",")]
    else:
        strategies = list(ALL_WITHIN_STUDY_STRATEGIES)

    # Check CASMOPOLITAN availability
    available_strategies = []
    for s in strategies:
        if s in ("casmopolitan_ucb", "casmopolitan_ei"):
            try:
                import botorch  # noqa: F401
                available_strategies.append(s)
            except ImportError:
                print(f"Skipping {s} (botorch not installed)")
        elif s == "discrete_tabpfn":
            try:
                import tabpfn  # noqa: F401
                available_strategies.append(s)
            except ImportError:
                print(f"Skipping {s} (tabpfn not installed)")
        else:
            available_strategies.append(s)
    strategies = available_strategies

    # Determine runs needed
    runs_needed = []
    for si in study_infos:
        pmid = si["pmid"]
        for strategy in strategies:
            for seed in seeds:
                if args.resume or args.aggregate_only:
                    existing = load_seed_result(pmid, strategy, seed)
                    if existing is not None:
                        continue
                if args.aggregate_only:
                    continue
                runs_needed.append((si, strategy, seed))

    total_runs = len(runs_needed)

    print(f"\n{'=' * 70}")
    print("WITHIN-STUDY BENCHMARK")
    print(f"{'=' * 70}")
    print(f"Studies: {len(study_infos)}")
    print(f"Strategies: {len(strategies)}")
    print(f"Seeds: {seeds}")
    print(f"Total runs needed: {total_runs}")
    if args.resume:
        print("Mode: RESUME (skipping existing results)")
    if args.aggregate_only:
        print("Mode: AGGREGATE ONLY")
    print()

    if args.dry_run:
        print("DRY RUN -- would execute:")
        for si, strategy, seed in runs_needed:
            print(f"  PMID {int(float(si['pmid']))} / {strategy} / seed={seed}")
        print(f"\nTotal: {total_runs} runs")
        return

    # -----------------------------------------------------------------------
    # Run strategies
    # -----------------------------------------------------------------------

    if total_runs > 0:
        print(f"\nRunning {total_runs} study-strategy-seed combinations...\n")

        # Group by (study, seed) to share data loading
        runs_by_study_seed = {}
        for si, strategy, seed in runs_needed:
            key = (si["pmid"], seed)
            runs_by_study_seed.setdefault(key, []).append((si, strategy))

        completed = 0
        for (pmid, seed), run_list in runs_by_study_seed.items():
            si = run_list[0][0]
            pmid_str = str(int(float(pmid)))

            print(f"\n{'=' * 50}")
            print(f"PMID {pmid_str} | seed={seed} | "
                  f"N={si['n_formulations']} | n_seed={si['n_seed']} | "
                  f"type={si['study_type']}")
            print(f"{'=' * 50}")

            try:
                pca_data = prepare_study_data(df, si, seed)
            except Exception as e:
                print(f"  FAILED to prepare data: {e}")
                import traceback
                traceback.print_exc()
                continue

            # GP strategies need full 4-component encoding; load separately
            gp_strats = [s for _, s in run_list if STRATEGY_CONFIGS[s]["type"] == "gp"]
            gp_data = None
            if gp_strats:
                gp_si = {**si, "feature_type": "lantern"}
                try:
                    gp_data = prepare_study_data(df, gp_si, seed)
                except Exception as e:
                    print(f"  GP data prep failed (expected for single-component studies): {e}")

            # Tanimoto strategies need raw count_mfp fingerprints (no PCA)
            tanimoto_strats = [s for _, s in run_list if s in TANIMOTO_STRATEGIES]
            tanimoto_data = None
            if tanimoto_strats:
                tanimoto_si = {**si, "feature_type": "count_mfp"}
                try:
                    tanimoto_data = prepare_study_data_raw(df, tanimoto_si, seed)
                except Exception as e:
                    print(f"  Tanimoto data prep failed: {e}")

            for _, strategy in run_list:
                completed += 1
                print(f"\n[{completed}/{total_runs}] {strategy}")
                print("-" * 40)

                is_gp = STRATEGY_CONFIGS[strategy]["type"] == "gp"
                is_tanimoto = strategy in TANIMOTO_STRATEGIES
                if is_tanimoto:
                    data_for_run = tanimoto_data
                elif is_gp:
                    data_for_run = gp_data
                else:
                    data_for_run = pca_data

                if data_for_run is None:
                    print("  SKIPPED: no compatible data for this strategy/study")
                    continue

                try:
                    result = run_single_seed(strategy, seed, si, pca_data=data_for_run)

                    recall_str = ", ".join(
                        f"Top-{k}%={result['metrics']['top_k_recall'].get(str(k), 0):.1%}"
                        for k in [5, 10, 20]
                    )
                    print(f"  Done in {result['elapsed']:.1f}s | {recall_str}")

                    save_seed_result(pmid, strategy, seed, result, si)

                except Exception as e:
                    print(f"  FAILED: {e}")
                    import traceback
                    traceback.print_exc()

    # -----------------------------------------------------------------------
    # Aggregation
    # -----------------------------------------------------------------------

    print(f"\n{'=' * 70}")
    print("AGGREGATION")
    print(f"{'=' * 70}\n")

    all_summaries = {}

    for si in study_infos:
        pmid = si["pmid"]
        pmid_str = str(int(float(pmid)))
        study_summaries = {}

        for strategy in strategies:
            seed_results = {}
            for seed in seeds:
                r = load_seed_result(pmid, strategy, seed)
                if r is not None:
                    seed_results[seed] = r

            if not seed_results:
                continue

            summary = aggregate_study_strategy(pmid_str, strategy, seed_results)
            if summary:
                study_summaries[strategy] = summary

        if study_summaries:
            all_summaries[pmid_str] = study_summaries

    if not all_summaries:
        print("No results to aggregate.")
        return

    # Generate markdown
    md = generate_within_study_markdown(all_summaries, study_infos)

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = RESULTS_DIR / "within_study_summary.md"
    with open(summary_path, "w") as f:
        f.write(md)
    print(f"Summary saved to {summary_path}")

    # Save JSON
    json_path = RESULTS_DIR / "within_study_summary.json"
    with open(json_path, "w") as f:
        json.dump({
            "study_infos": [{k: v for k, v in si.items()} for si in study_infos],
            "summaries": all_summaries,
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2, default=str)
    print(f"JSON saved to {json_path}")

    # Print cross-study summary
    print("\n" + md.split("## Cross-Study Summary")[1] if "## Cross-Study Summary" in md else "")


if __name__ == "__main__":
    main()
