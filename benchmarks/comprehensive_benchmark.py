#!/usr/bin/env python3
"""Benchmark.

Runs all implemented surrogates and strategies through the benchmark
harness with unified settings, computes statistical comparisons, and
produces a JSON + markdown summary table.

Settings: n_seed=500, batch_size=12, 15 rounds, 5 seeds.
Feature type: lantern_il_only with PCA reduction (default).

Usage:
    # Run all strategies:
    python -m benchmarks.comprehensive_benchmark

    # Run specific strategies:
    python -m benchmarks.comprehensive_benchmark --strategies random,discrete_xgb_greedy,discrete_xgb_ucb

    # Resume / accumulate (loads existing results, runs missing strategies):
    python -m benchmarks.comprehensive_benchmark --resume

    # Aggregate only (no new runs, just re-compute stats from existing per-seed JSONs):
    python -m benchmarks.comprehensive_benchmark --aggregate-only

    # Dry run (list strategies without executing):
    python -m benchmarks.comprehensive_benchmark --dry-run
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from .runner import (
    PLS_STRATEGIES,
    STRATEGY_CONFIGS,
    STRATEGY_DISPLAY,
    _run_random,
    compute_metrics,
    prepare_benchmark_data,
)
from .stats import bootstrap_ci, format_result, paired_wilcoxon

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEEDS = [42, 123, 456, 789, 2024]

DEFAULT_SETTINGS = {
    "n_seed": 500,
    "batch_size": 12,
    "n_rounds": 15,
    "feature_type": "lantern_il_only",
    "reduction": "pca",
    "normalize": "copula",
    "kappa": 5.0,
    "xi": 0.01,
    "n_pcs": None,
    "context_features": False,
    "fp_radius": None,
    "fp_bits": None,
}

# Strategies to benchmark, grouped by type for clarity.
# Each entry maps to a key in runner.STRATEGY_CONFIGS.
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

GP_STRATEGIES = [
    "lnpbo_logei",
    "lnpbo_rkb_logei",
    "lnpbo_lp_logei",
]

# Optional strategies that require extra packages.
# These are attempted with try/except at import time.
OPTIONAL_STRATEGIES = [
    "discrete_tabpfn",      # requires tabpfn
    "casmopolitan_ucb",     # requires botorch
    "casmopolitan_ei",      # requires botorch
]

ALL_BENCHMARK_STRATEGIES = (
    DISCRETE_STRATEGIES
    + TS_BATCH_STRATEGIES
    + ONLINE_CONFORMAL_STRATEGIES
    + GP_STRATEGIES
    + OPTIONAL_STRATEGIES
)

REFERENCE_STRATEGY = "discrete_xgb_greedy"

RESULTS_DIR = Path(__file__).resolve().parent.parent / "benchmark_results"


# ---------------------------------------------------------------------------
# Per-seed runner
# ---------------------------------------------------------------------------




def _check_optional_available(strategy):
    """Check if an optional strategy's dependencies are installed."""
    if strategy == "discrete_tabpfn":
        try:
            import tabpfn  # noqa: F401
            return True
        except ImportError:
            return False
    if strategy in ("casmopolitan_ucb", "casmopolitan_ei"):
        try:
            import botorch  # noqa: F401
            return True
        except ImportError:
            return False
    return True


def run_single_seed(
    strategy,
    random_seed,
    settings,
    pca_data=None,
    pls_data=None,
):
    """Run a single strategy for a single seed and return metrics dict.

    Parameters
    ----------
    strategy : str
        Strategy name (must be a key in STRATEGY_CONFIGS).
    random_seed : int
        Random seed for this run.
    settings : dict
        Benchmark settings (n_seed, batch_size, n_rounds, etc.).
    pca_data : tuple, optional
        Pre-loaded PCA data from prepare_benchmark_data. If None, will be loaded.
    pls_data : tuple, optional
        Pre-loaded PLS data. Only needed for PLS strategies.

    Returns
    -------
    result : dict
        Contains "metrics", "elapsed", "best_so_far", "round_best",
        "n_evaluated", and optionally "coverage"/"conformal_quantile".
    """
    batch_size = settings["batch_size"]
    n_rounds = settings["n_rounds"]
    kappa = settings["kappa"]
    xi = settings["xi"]
    normalize = settings["normalize"]

    # Load data if not provided
    if pca_data is None:
        pca_data = prepare_benchmark_data(
            n_seed=settings["n_seed"],
            random_seed=random_seed,
            reduction=settings["reduction"],
            feature_type=settings["feature_type"],
            n_pcs=settings["n_pcs"],
            context_features=settings["context_features"],
            fp_radius=settings["fp_radius"],
            fp_bits=settings["fp_bits"],
        )

    if strategy in PLS_STRATEGIES and pls_data is None:
        pls_data = prepare_benchmark_data(
            n_seed=settings["n_seed"],
            random_seed=random_seed,
            reduction="pls",
            feature_type=settings["feature_type"],
            n_pcs=settings["n_pcs"],
            context_features=settings["context_features"],
        )

    if strategy in PLS_STRATEGIES and pls_data is not None:
        s_dataset, s_df, s_fcols, s_seed, s_oracle, s_topk = pls_data
    else:
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
        from ._gp_common import run_gp_strategy
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
    # Normalize top_k_recall keys to strings for JSON round-trip consistency
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
    if "conformal_quantile" in history:
        result["conformal_quantile"] = history["conformal_quantile"]

    return result


# ---------------------------------------------------------------------------
# Result I/O
# ---------------------------------------------------------------------------


def _per_seed_path(strategy, seed):
    return RESULTS_DIR / "comprehensive" / f"{strategy}_s{seed}.json"


def save_seed_result(strategy, seed, result, settings):
    path = _per_seed_path(strategy, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "strategy": strategy,
        "seed": seed,
        "settings": settings,
        "result": result,
        "timestamp": datetime.now().isoformat(),
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_seed_result(strategy, seed):
    path = _per_seed_path(strategy, seed)
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    return data.get("result")


def load_all_seed_results(strategy, seeds):
    results = {}
    for seed in seeds:
        r = load_seed_result(strategy, seed)
        if r is not None:
            results[seed] = r
    return results


# ---------------------------------------------------------------------------
# Aggregation and statistics
# ---------------------------------------------------------------------------


def aggregate_strategy(strategy, seed_results, random_seed_results=None):
    """Aggregate per-seed results into summary statistics.

    Parameters
    ----------
    strategy : str
        Strategy name.
    seed_results : dict[int, dict]
        Mapping from seed -> result dict (with "metrics" key).
    random_seed_results : dict[int, dict], optional
        Random baseline results for acceleration factor computation.

    Returns
    -------
    summary : dict
        Aggregated statistics including mean, std, bootstrap CI,
        per-seed values, and acceleration factor.
    """
    seeds_available = sorted(seed_results.keys())
    n_seeds = len(seeds_available)

    # Collect per-seed metric arrays
    top10_vals = np.array([seed_results[s]["metrics"]["top_k_recall"]["10"] for s in seeds_available])
    top50_vals = np.array([seed_results[s]["metrics"]["top_k_recall"]["50"] for s in seeds_available])
    top100_vals = np.array([seed_results[s]["metrics"]["top_k_recall"]["100"] for s in seeds_available])
    auc_vals = np.array([seed_results[s]["metrics"]["auc"] for s in seeds_available])
    final_best_vals = np.array([seed_results[s]["metrics"]["final_best"] for s in seeds_available])
    elapsed_vals = np.array([seed_results[s]["elapsed"] for s in seeds_available])

    def _stats(vals):
        ci_lo, ci_hi = bootstrap_ci(vals) if len(vals) >= 3 else (float("nan"), float("nan"))
        return {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "ci_low": ci_lo,
            "ci_high": ci_hi,
            "per_seed": {int(s): float(v) for s, v in zip(seeds_available, vals)},
        }

    summary = {
        "strategy": strategy,
        "display_name": STRATEGY_DISPLAY.get(strategy, strategy),
        "n_seeds": n_seeds,
        "seeds": seeds_available,
        "top_10_recall": _stats(top10_vals),
        "top_50_recall": _stats(top50_vals),
        "top_100_recall": _stats(top100_vals),
        "auc": _stats(auc_vals),
        "final_best": _stats(final_best_vals),
        "elapsed_seconds": _stats(elapsed_vals),
    }

    # Acceleration factor vs random (using Top-50 recall curves)
    if random_seed_results is not None and strategy != "random":
        af_vals = []
        for s in seeds_available:
            if s not in random_seed_results:
                continue
            rand_r = random_seed_results[s]
            bo_r = seed_results[s]
            # Build recall curve: (n_evaluated, cumulative top-50 recall)
            # We approximate by computing recall at each round's n_evaluated
            rand_ne = rand_r.get("n_evaluated", [])
            bo_ne = bo_r.get("n_evaluated", [])
            # Use final top-50 recall as proxy target
            bo_top50 = bo_r["metrics"]["top_k_recall"]["50"]
            if bo_top50 > 0:
                rand_top50 = rand_r["metrics"]["top_k_recall"]["50"]
                # Simple AF: if random achieves less, BO is better
                # Full curve-based AF requires per-round recall tracking
                # which we don't store; use budget ratio as approximation
                if rand_top50 > 0:
                    # Approximate: BO reaches bo_top50 in its budget;
                    # random would need (bo_top50/rand_top50) * random_budget
                    # to reach the same recall (linear extrapolation)
                    bo_budget = bo_ne[-1] if bo_ne else 680
                    rand_budget = rand_ne[-1] if rand_ne else 680
                    af = (bo_top50 / rand_top50) * (rand_budget / bo_budget)
                    af_vals.append(af)
                else:
                    af_vals.append(float("inf"))
        if af_vals:
            finite_af = [v for v in af_vals if np.isfinite(v)]
            summary["acceleration_factor"] = {
                "mean": float(np.mean(finite_af)) if finite_af else float("inf"),
                "per_seed": af_vals,
            }

    return summary


def compute_pairwise_tests(summaries, reference=REFERENCE_STRATEGY):
    """Compute paired Wilcoxon tests for each strategy vs the reference.

    Parameters
    ----------
    summaries : dict[str, dict]
        Strategy name -> aggregated summary dict.
    reference : str
        Reference strategy to compare against.

    Returns
    -------
    p_values : dict[str, float]
        Strategy name -> p-value (two-sided Wilcoxon vs reference, on Top-50).
    """
    if reference not in summaries:
        return {}

    ref_per_seed = summaries[reference]["top_50_recall"]["per_seed"]
    ref_seeds = sorted(ref_per_seed.keys())

    p_values = {}
    for strat, summary in summaries.items():
        if strat == reference:
            p_values[strat] = 1.0
            continue
        strat_per_seed = summary["top_50_recall"]["per_seed"]
        common_seeds = sorted(set(ref_seeds) & set(strat_per_seed.keys()))
        if len(common_seeds) < 3:
            p_values[strat] = float("nan")
            continue
        a = [ref_per_seed[s] for s in common_seeds]
        b = [strat_per_seed[s] for s in common_seeds]
        p_values[strat] = paired_wilcoxon(a, b)

    return p_values


# ---------------------------------------------------------------------------
# Markdown output
# ---------------------------------------------------------------------------


def generate_markdown_table(summaries, p_values, settings):
    """Generate a markdown summary table ranked by Top-50 recall.

    Parameters
    ----------
    summaries : dict[str, dict]
        Strategy name -> aggregated summary.
    p_values : dict[str, float]
        Strategy name -> Wilcoxon p-value vs reference.
    settings : dict
        Benchmark settings for the header.

    Returns
    -------
    md : str
        Markdown-formatted summary.
    """
    lines = []
    lines.append("# Comprehensive Benchmark Results")
    lines.append("")
    lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"**Settings:** n_seed={settings['n_seed']}, "
                 f"batch_size={settings['batch_size']}, "
                 f"rounds={settings['n_rounds']}, "
                 f"seeds={SEEDS}")
    lines.append(f"**Features:** {settings['feature_type']} + {settings['reduction']}")
    lines.append(f"**Normalize:** {settings['normalize']}")
    lines.append(f"**Reference:** {REFERENCE_STRATEGY}")
    lines.append("")

    # Sort by Top-50 mean descending
    ranked = sorted(
        summaries.items(),
        key=lambda x: x[1]["top_50_recall"]["mean"],
        reverse=True,
    )

    lines.append("## Rankings by Top-50 Recall")
    lines.append("")
    lines.append(
        "| Rank | Strategy | Top-50 (mean +/- std) | 95% CI | "
        "Top-10 | Top-100 | AUC | p vs XGB | Time (s) |"
    )
    lines.append(
        "|------|----------|-----------------------|--------|"
        "--------|---------|-----|----------|----------|"
    )

    for rank, (strat, s) in enumerate(ranked, 1):
        t50 = s["top_50_recall"]
        t10 = s["top_10_recall"]
        t100 = s["top_100_recall"]
        auc = s["auc"]
        elapsed = s["elapsed_seconds"]
        p = p_values.get(strat, float("nan"))

        t50_str = format_result(t50["mean"], t50["std"])
        ci_str = f"[{t50['ci_low']:.1%}, {t50['ci_high']:.1%}]" if not np.isnan(t50["ci_low"]) else "N/A"
        t10_str = f"{t10['mean']:.1%}"
        t100_str = f"{t100['mean']:.1%}"
        auc_str = f"{auc['mean']:.2f}"
        p_str = f"{p:.3f}" if not np.isnan(p) else "N/A"
        time_str = f"{elapsed['mean']:.1f}"
        display = s["display_name"]

        lines.append(
            f"| {rank} | {display} | {t50_str} | {ci_str} | "
            f"{t10_str} | {t100_str} | {auc_str} | {p_str} | {time_str} |"
        )

    lines.append("")

    # Acceleration factors
    lines.append("## Acceleration Factors vs Random")
    lines.append("")
    lines.append("| Strategy | AF (mean) |")
    lines.append("|----------|-----------|")
    for _strat, s in ranked:
        if "acceleration_factor" in s:
            af = s["acceleration_factor"]["mean"]
            af_str = f"{af:.2f}x" if np.isfinite(af) else "inf"
        else:
            af_str = "N/A"
        lines.append(f"| {s['display_name']} | {af_str} |")

    lines.append("")

    # Per-seed detail for Top-50
    lines.append("## Per-Seed Top-50 Recall")
    lines.append("")
    seed_header = "| Strategy | " + " | ".join(f"s{s}" for s in SEEDS) + " |"
    seed_sep = "|----------|" + "|".join("------" for _ in SEEDS) + "|"
    lines.append(seed_header)
    lines.append(seed_sep)
    for _strat, s in ranked:
        per_seed = s["top_50_recall"]["per_seed"]
        vals = " | ".join(
            f"{per_seed.get(seed, float('nan')):.1%}" if seed in per_seed else "N/A"
            for seed in SEEDS
        )
        lines.append(f"| {s['display_name']} | {vals} |")

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive Benchmark Suite: "
                    "runs all surrogates with unified settings.",
    )
    parser.add_argument(
        "--strategies",
        type=str,
        default=None,
        help="Comma-separated strategies to run (default: all available). "
             f"Options: {','.join(ALL_BENCHMARK_STRATEGIES)}",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help=f"Comma-separated seeds (default: {','.join(str(s) for s in SEEDS)})",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip strategies/seeds that already have saved results.",
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Do not run any strategies; only aggregate existing per-seed results.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List strategies and seeds that would be run, without executing.",
    )
    parser.add_argument(
        "--n-seed",
        type=int,
        default=DEFAULT_SETTINGS["n_seed"],
        help=f"Initial seed pool size (default: {DEFAULT_SETTINGS['n_seed']})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_SETTINGS["batch_size"],
        help=f"Batch size per round (default: {DEFAULT_SETTINGS['batch_size']})",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=DEFAULT_SETTINGS["n_rounds"],
        help=f"Number of rounds (default: {DEFAULT_SETTINGS['n_rounds']})",
    )
    parser.add_argument(
        "--feature-type",
        type=str,
        default=DEFAULT_SETTINGS["feature_type"],
        help=f"Feature type (default: {DEFAULT_SETTINGS['feature_type']})",
    )
    parser.add_argument(
        "--reduction",
        type=str,
        default=DEFAULT_SETTINGS["reduction"],
        choices=["pca", "pls", "none"],
        help=f"Reduction method (default: {DEFAULT_SETTINGS['reduction']})",
    )
    parser.add_argument(
        "--normalize",
        type=str,
        default=DEFAULT_SETTINGS["normalize"],
        choices=["none", "zscore", "copula"],
        help=f"Target normalization (default: {DEFAULT_SETTINGS['normalize']})",
    )
    parser.add_argument(
        "--kappa",
        type=float,
        default=DEFAULT_SETTINGS["kappa"],
        help=f"UCB kappa (default: {DEFAULT_SETTINGS['kappa']})",
    )
    parser.add_argument(
        "--xi",
        type=float,
        default=DEFAULT_SETTINGS["xi"],
        help=f"EI/LogEI xi (default: {DEFAULT_SETTINGS['xi']})",
    )
    args = parser.parse_args()

    # Build settings dict
    settings = {
        "n_seed": args.n_seed,
        "batch_size": args.batch_size,
        "n_rounds": args.rounds,
        "feature_type": args.feature_type,
        "reduction": args.reduction,
        "normalize": args.normalize,
        "kappa": args.kappa,
        "xi": args.xi,
        "n_pcs": DEFAULT_SETTINGS["n_pcs"],
        "context_features": DEFAULT_SETTINGS["context_features"],
        "fp_radius": DEFAULT_SETTINGS["fp_radius"],
        "fp_bits": DEFAULT_SETTINGS["fp_bits"],
    }

    # Parse seeds
    seeds = SEEDS
    if args.seeds is not None:
        seeds = [int(s.strip()) for s in args.seeds.split(",")]

    # Parse strategies
    if args.strategies is not None:
        strategies = [s.strip() for s in args.strategies.split(",")]
        for s in strategies:
            if s not in STRATEGY_CONFIGS:
                parser.error(
                    f"Unknown strategy: {s}. "
                    f"Choose from: {list(STRATEGY_CONFIGS.keys())}"
                )
    else:
        # Default: all benchmark strategies, filtering out unavailable optional ones
        strategies = []
        for s in ALL_BENCHMARK_STRATEGIES:
            if s in OPTIONAL_STRATEGIES:
                if _check_optional_available(s):
                    strategies.append(s)
                else:
                    print(f"Skipping {s} (dependencies not installed)")
            else:
                strategies.append(s)

    # Determine what needs to run
    runs_needed = []
    for strategy in strategies:
        for seed in seeds:
            if args.resume or args.aggregate_only:
                existing = load_seed_result(strategy, seed)
                if existing is not None:
                    continue
            if args.aggregate_only:
                continue
            runs_needed.append((strategy, seed))

    total_runs = len(runs_needed)

    print("=" * 70)
    print("BENCHMARK")
    print("=" * 70)
    print(f"Strategies: {strategies}")
    print(f"Seeds: {seeds}")
    print(f"Settings: {json.dumps(settings, indent=2)}")
    print(f"Total runs needed: {total_runs}")
    if args.resume:
        print("Mode: RESUME (skipping existing results)")
    if args.aggregate_only:
        print("Mode: AGGREGATE ONLY (no new runs)")
    print()

    if args.dry_run:
        print("DRY RUN -- would execute the following:")
        for strategy, seed in runs_needed:
            print(f"  {strategy} seed={seed}")
        print(f"\nTotal: {total_runs} runs")
        return

    # -----------------------------------------------------------------------
    # Run strategies
    # -----------------------------------------------------------------------

    if total_runs > 0:
        print(f"\nRunning {total_runs} strategy-seed combinations...\n")

        # Group runs by seed to share data loading
        runs_by_seed = {}
        for strategy, seed in runs_needed:
            runs_by_seed.setdefault(seed, []).append(strategy)

        completed = 0
        for seed in seeds:
            strats_for_seed = runs_by_seed.get(seed, [])
            if not strats_for_seed:
                continue

            # Load data once per seed
            print(f"\n--- Loading data for seed={seed} ---")
            pca_data = prepare_benchmark_data(
                n_seed=settings["n_seed"],
                random_seed=seed,
                reduction=settings["reduction"],
                feature_type=settings["feature_type"],
                n_pcs=settings["n_pcs"],
                context_features=settings["context_features"],
                fp_radius=settings["fp_radius"],
                fp_bits=settings["fp_bits"],
            )

            # Load PLS data only if needed
            pls_data = None
            if any(s in PLS_STRATEGIES for s in strats_for_seed):
                pls_data = prepare_benchmark_data(
                    n_seed=settings["n_seed"],
                    random_seed=seed,
                    reduction="pls",
                    feature_type=settings["feature_type"],
                    n_pcs=settings["n_pcs"],
                    context_features=settings["context_features"],
                )

            for strategy in strats_for_seed:
                completed += 1
                print(f"\n[{completed}/{total_runs}] {strategy} seed={seed}")
                print("-" * 50)

                try:
                    result = run_single_seed(
                        strategy, seed, settings,
                        pca_data=pca_data, pls_data=pls_data,
                    )
                    save_seed_result(strategy, seed, result, settings)

                    m = result["metrics"]
                    print(
                        f"  Done in {result['elapsed']:.1f}s | "
                        f"Top-10={m['top_k_recall']['10']:.1%} "
                        f"Top-50={m['top_k_recall']['50']:.1%} "
                        f"Top-100={m['top_k_recall']['100']:.1%} "
                        f"AUC={m['auc']:.2f}"
                    )
                except Exception as e:
                    print(f"  FAILED: {e}")
                    import traceback
                    traceback.print_exc()

    # -----------------------------------------------------------------------
    # Aggregate results
    # -----------------------------------------------------------------------

    print(f"\n{'=' * 70}")
    print("AGGREGATING RESULTS")
    print(f"{'=' * 70}\n")

    # Load all per-seed results
    all_seed_results = {}
    for strategy in strategies:
        seed_results = load_all_seed_results(strategy, seeds)
        if seed_results:
            all_seed_results[strategy] = seed_results
            print(f"  {strategy}: {len(seed_results)} seeds loaded")
        else:
            print(f"  {strategy}: NO results found")

    if not all_seed_results:
        print("\nNo results to aggregate. Run strategies first.")
        return

    # Get random results for acceleration factor
    random_results = all_seed_results.get("random")

    # Compute summaries
    summaries = {}
    for strategy, seed_results in all_seed_results.items():
        summaries[strategy] = aggregate_strategy(
            strategy, seed_results, random_seed_results=random_results,
        )

    # Compute pairwise tests
    p_values = compute_pairwise_tests(summaries, reference=REFERENCE_STRATEGY)

    # Print summary table
    print(f"\n{'=' * 70}")
    print("SUMMARY TABLE (ranked by Top-50 recall)")
    print(f"{'=' * 70}\n")

    ranked = sorted(
        summaries.items(),
        key=lambda x: x[1]["top_50_recall"]["mean"],
        reverse=True,
    )

    header = (
        f"{'Rank':<5} {'Strategy':<32} {'Top-50 mean':>12} {'std':>8} "
        f"{'Top-10':>8} {'Top-100':>8} {'p-val':>8} {'Time':>8}"
    )
    print(header)
    print("-" * len(header))
    for rank, (strat, s) in enumerate(ranked, 1):
        t50 = s["top_50_recall"]
        t10 = s["top_10_recall"]
        t100 = s["top_100_recall"]
        elapsed = s["elapsed_seconds"]
        p = p_values.get(strat, float("nan"))
        p_str = f"{p:.3f}" if not np.isnan(p) else "N/A"
        print(
            f"{rank:<5} {s['display_name']:<32} {t50['mean']:>11.1%} "
            f"{t50['std']:>7.1%} {t10['mean']:>7.1%} {t100['mean']:>7.1%} "
            f"{p_str:>8} {elapsed['mean']:>7.1f}s"
        )

    # -----------------------------------------------------------------------
    # Save comprehensive JSON
    # -----------------------------------------------------------------------

    RESULTS_DIR.mkdir(exist_ok=True)
    json_path = RESULTS_DIR / "comprehensive_results.json"
    comprehensive = {
        "settings": settings,
        "seeds": seeds,
        "timestamp": datetime.now().isoformat(),
        "reference_strategy": REFERENCE_STRATEGY,
        "summaries": summaries,
        "p_values_vs_reference": p_values,
    }
    with open(json_path, "w") as f:
        json.dump(comprehensive, f, indent=2, default=str)
    print(f"\nComprehensive JSON saved to {json_path}")

    # -----------------------------------------------------------------------
    # Save markdown summary
    # -----------------------------------------------------------------------

    md_path = RESULTS_DIR / "comprehensive_summary.md"
    md_content = generate_markdown_table(summaries, p_values, settings)
    with open(md_path, "w") as f:
        f.write(md_content)
    print(f"Markdown summary saved to {md_path}")


if __name__ == "__main__":
    main()
