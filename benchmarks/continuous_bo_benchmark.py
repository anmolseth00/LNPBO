#!/usr/bin/env python3
"""
Continuous BO vs Discrete Pool Scoring: End-to-End Benchmark
=============================================================
Tests whether GP-based continuous BO (acquisition function optimization
with simplex projection and batch generation) can compete with discrete
pool scoring (XGBoost/NGBoost greedy or UCB on the candidate pool).

Strategies compared:
  Continuous (GP-based):
    - lnpbo_logei: GP + Kriging Believer + LogEI
    - lnpbo_rkb_logei: GP + Randomized KB + LogEI
    - lnpbo_lp_logei: GP + Local Penalization + LogEI

  Discrete (pool scoring):
    - discrete_xgb_greedy: XGBoost greedy (current best)
    - discrete_xgb_ucb: XGBoost with conformal UCB (MAPIE)
    - discrete_ngboost_ucb: NGBoost with UCB
    - random: random baseline

Settings: n_seed=500, batch_size=12, 15 rounds, 5 seeds.
Feature type: lantern_il_only with PCA reduction.

Usage:
    python -m benchmarks.continuous_bo_benchmark
    python -m benchmarks.continuous_bo_benchmark --seeds 42,123
    python -m benchmarks.continuous_bo_benchmark --skip-continuous
    python -m benchmarks.continuous_bo_benchmark --skip-discrete

References
----------
Ament, S., Daulton, S., Eriksson, D., Balandat, M., & Bakshy, E.
    "Unexpected Improvements to Expected Improvement for Bayesian
    Optimization." NeurIPS 2023, arXiv:2310.20708.

Gonzalez, J., Dai, Z., Hennig, P., & Lawrence, N.
    "Batch Bayesian Optimization via Local Penalization." AISTATS 2016,
    arXiv:1505.08052.

Ginsbourger, D., Le Riche, R., & Carraro, L.
    "Kriging Is Well-Suited to Parallelize Optimization." Springer 2010.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .runner import (
    STRATEGY_CONFIGS,
    STRATEGY_DISPLAY,
    _run_random,
    compute_metrics,
    init_history,
    prepare_benchmark_data,
)

CONTINUOUS_STRATEGIES = ["lnpbo_logei", "lnpbo_rkb_logei", "lnpbo_lp_logei"]
DISCRETE_STRATEGIES = ["discrete_xgb_greedy", "discrete_xgb_ucb", "discrete_ngboost_ucb"]
ALL_BENCHMARK_STRATEGIES = ["random"] + CONTINUOUS_STRATEGIES + DISCRETE_STRATEGIES

DEFAULT_SEEDS = [42, 123, 456, 789, 2024]

BENCHMARK_CONFIG = {
    "n_seed": 500,
    "batch_size": 12,
    "n_rounds": 15,
    "feature_type": "lantern_il_only",
    "reduction": "pca",
    "kappa": 5.0,
    "xi": 0.01,
    "normalize": "copula",
}


def _run_single_seed(
    strategy,
    seed,
    encoded_dataset,
    encoded_df,
    feature_cols,
    seed_idx,
    oracle_idx,
    top_k_values,
    config,
):
    """Run a single strategy for a single seed, returning history and metrics."""
    cfg = STRATEGY_CONFIGS[strategy]
    t0 = time.time()

    if cfg["type"] == "random":
        history = _run_random(
            encoded_df, seed_idx, oracle_idx,
            config["batch_size"], config["n_rounds"], seed,
        )
    elif cfg["type"] == "discrete":
        from ._discrete_common import run_discrete_strategy

        history = run_discrete_strategy(
            encoded_df, feature_cols, seed_idx, oracle_idx,
            surrogate=cfg["surrogate"],
            batch_size=config["batch_size"],
            n_rounds=config["n_rounds"],
            seed=seed,
            kappa=config["kappa"],
            normalize=config["normalize"],
            encoded_dataset=encoded_dataset,
        )
    elif cfg["type"] == "gp":
        from ._gp_common import run_gp_strategy

        history = run_gp_strategy(
            encoded_dataset, encoded_df, feature_cols, seed_idx, oracle_idx,
            acq_type=cfg["acq_type"],
            batch_size=config["batch_size"],
            n_rounds=config["n_rounds"],
            seed=seed,
            kappa=config["kappa"],
            xi=config["xi"],
            normalize=config["normalize"],
        )
    else:
        raise ValueError(f"Unknown strategy type: {cfg['type']!r}")

    elapsed = time.time() - t0
    metrics = compute_metrics(history, top_k_values, len(encoded_df))

    return {
        "history": history,
        "metrics": metrics,
        "elapsed": elapsed,
    }


def run_benchmark(seeds=None, skip_continuous=False, skip_discrete=False):
    """Run the full continuous vs discrete benchmark across all seeds."""
    if seeds is None:
        seeds = DEFAULT_SEEDS

    strategies = ["random"]
    if not skip_continuous:
        strategies += CONTINUOUS_STRATEGIES
    if not skip_discrete:
        strategies += DISCRETE_STRATEGIES

    config = BENCHMARK_CONFIG.copy()

    print("=" * 70)
    print("CONTINUOUS BO vs DISCRETE POOL SCORING BENCHMARK")
    print("=" * 70)
    print(f"Strategies: {strategies}")
    print(f"Seeds: {seeds}")
    print(f"Config: n_seed={config['n_seed']}, batch_size={config['batch_size']}, "
          f"n_rounds={config['n_rounds']}")
    print(f"Features: {config['feature_type']}, reduction={config['reduction']}")
    print()

    # Collect per-seed results: {strategy: {seed: result}}
    all_seed_results = {s: {} for s in strategies}

    for seed in seeds:
        print(f"\n{'=' * 70}")
        print(f"SEED: {seed}")
        print(f"{'=' * 70}")

        # Prepare data for this seed
        data = prepare_benchmark_data(
            n_seed=config["n_seed"],
            random_seed=seed,
            reduction=config["reduction"],
            feature_type=config["feature_type"],
        )
        encoded_dataset, encoded_df, feature_cols, seed_idx, oracle_idx, top_k_values = data

        for strategy in strategies:
            print(f"\n--- {strategy} (seed={seed}) ---")
            result = _run_single_seed(
                strategy, seed,
                encoded_dataset, encoded_df, feature_cols,
                seed_idx, oracle_idx, top_k_values,
                config,
            )

            m = result["metrics"]
            r = m["top_k_recall"]
            print(
                f"  Final best: {m['final_best']:.4f}, AUC: {m['auc']:.4f}, "
                f"Top-10: {r.get(10, 0):.1%}, Top-50: {r.get(50, 0):.1%}, "
                f"Top-100: {r.get(100, 0):.1%}, Time: {result['elapsed']:.1f}s"
            )

            all_seed_results[strategy][seed] = result

    return all_seed_results, strategies, config


def aggregate_results(all_seed_results, strategies):
    """Compute mean +/- std across seeds for each strategy."""
    summary = {}
    for strategy in strategies:
        seed_results = all_seed_results[strategy]
        if not seed_results:
            continue

        metrics_list = [r["metrics"] for r in seed_results.values()]
        elapsed_list = [r["elapsed"] for r in seed_results.values()]

        final_bests = [m["final_best"] for m in metrics_list]
        aucs = [m["auc"] for m in metrics_list]

        recall_by_k = {}
        for k in [10, 50, 100]:
            vals = [m["top_k_recall"].get(k, 0.0) for m in metrics_list]
            recall_by_k[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

        summary[strategy] = {
            "final_best": {"mean": float(np.mean(final_bests)), "std": float(np.std(final_bests))},
            "auc": {"mean": float(np.mean(aucs)), "std": float(np.std(aucs))},
            "top_k_recall": recall_by_k,
            "elapsed": {"mean": float(np.mean(elapsed_list)), "std": float(np.std(elapsed_list))},
            "n_seeds": len(seed_results),
        }

    return summary


def generate_markdown_summary(summary, strategies, config):
    """Generate a markdown summary table comparing continuous vs discrete."""
    lines = []
    lines.append("# Continuous BO vs Discrete Pool Scoring")
    lines.append("")
    lines.append("## Configuration")
    lines.append(f"- n_seed: {config['n_seed']}")
    lines.append(f"- batch_size: {config['batch_size']}")
    lines.append(f"- n_rounds: {config['n_rounds']}")
    lines.append(f"- total evaluations: {config['n_seed'] + config['batch_size'] * config['n_rounds']}")
    lines.append(f"- feature_type: {config['feature_type']}")
    lines.append(f"- reduction: {config['reduction']}")
    lines.append(f"- normalize: {config['normalize']}")
    lines.append(f"- seeds: {summary[strategies[0]]['n_seeds']}")
    lines.append("")
    lines.append("## Results (mean +/- std across seeds)")
    lines.append("")
    lines.append("| Strategy | Type | Top-10 | Top-50 | Top-100 | AUC | Time (s) |")
    lines.append("|----------|------|--------|--------|---------|-----|----------|")

    for strategy in strategies:
        if strategy not in summary:
            continue
        s = summary[strategy]
        display = STRATEGY_DISPLAY.get(strategy, strategy)
        if strategy == "random":
            stype = "Baseline"
        elif strategy in CONTINUOUS_STRATEGIES:
            stype = "Continuous"
        else:
            stype = "Discrete"

        r10 = s["top_k_recall"][10]
        r50 = s["top_k_recall"][50]
        r100 = s["top_k_recall"][100]
        auc = s["auc"]
        elapsed = s["elapsed"]

        lines.append(
            f"| {display} | {stype} | "
            f"{r10['mean']:.1%} +/-{r10['std']:.1%} | "
            f"{r50['mean']:.1%} +/-{r50['std']:.1%} | "
            f"{r100['mean']:.1%} +/-{r100['std']:.1%} | "
            f"{auc['mean']:.3f} +/-{auc['std']:.3f} | "
            f"{elapsed['mean']:.0f} +/-{elapsed['std']:.0f} |"
        )

    lines.append("")

    # Determine winner
    continuous_strats = [s for s in CONTINUOUS_STRATEGIES if s in summary]
    discrete_strats = [s for s in DISCRETE_STRATEGIES if s in summary]

    if continuous_strats and discrete_strats:
        best_cont = max(continuous_strats, key=lambda s: summary[s]["top_k_recall"][50]["mean"])
        best_disc = max(discrete_strats, key=lambda s: summary[s]["top_k_recall"][50]["mean"])

        cont_r50 = summary[best_cont]["top_k_recall"][50]["mean"]
        disc_r50 = summary[best_disc]["top_k_recall"][50]["mean"]

        lines.append("## Key Finding")
        lines.append("")
        if cont_r50 > disc_r50:
            lines.append(
                f"**Continuous BO wins.** Best continuous ({STRATEGY_DISPLAY.get(best_cont, best_cont)}: "
                f"{cont_r50:.1%}) beats best discrete ({STRATEGY_DISPLAY.get(best_disc, best_disc)}: "
                f"{disc_r50:.1%}) on Top-50 recall."
            )
        elif disc_r50 > cont_r50:
            lines.append(
                f"**Discrete pool scoring wins.** Best discrete ({STRATEGY_DISPLAY.get(best_disc, best_disc)}: "
                f"{disc_r50:.1%}) beats best continuous ({STRATEGY_DISPLAY.get(best_cont, best_cont)}: "
                f"{cont_r50:.1%}) on Top-50 recall."
            )
        else:
            lines.append("**Tie.** Continuous and discrete achieve identical Top-50 recall.")
        lines.append("")

    return "\n".join(lines)


def save_results(all_seed_results, summary, strategies, config, output_dir):
    """Save full results and summary to JSON."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build serializable results
    serializable = {
        "config": config,
        "seeds": sorted(
            next(iter(all_seed_results.values())).keys()
        ) if all_seed_results else [],
        "summary": summary,
        "per_seed": {},
    }

    for strategy in strategies:
        serializable["per_seed"][strategy] = {}
        for seed, result in all_seed_results.get(strategy, {}).items():
            entry = {
                "metrics": result["metrics"],
                "elapsed": result["elapsed"],
                "best_so_far": result["history"]["best_so_far"],
                "round_best": result["history"]["round_best"],
                "n_evaluated": result["history"]["n_evaluated"],
            }
            serializable["per_seed"][strategy][str(seed)] = entry

    json_path = output_dir / "continuous_bo_e2e.json"
    with open(json_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {json_path}")

    return json_path


def main():
    parser = argparse.ArgumentParser(
        description="Continuous BO vs Discrete Pool Scoring Benchmark",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=",".join(str(s) for s in DEFAULT_SEEDS),
        help=f"Comma-separated random seeds (default: {','.join(str(s) for s in DEFAULT_SEEDS)})",
    )
    parser.add_argument(
        "--skip-continuous",
        action="store_true",
        help="Skip continuous GP strategies (run discrete only)",
    )
    parser.add_argument(
        "--skip-discrete",
        action="store_true",
        help="Skip discrete strategies (run continuous only)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: benchmark_results/)",
    )
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = str(Path(__file__).resolve().parent.parent / "benchmark_results")

    all_seed_results, strategies, config = run_benchmark(
        seeds=seeds,
        skip_continuous=args.skip_continuous,
        skip_discrete=args.skip_discrete,
    )

    summary = aggregate_results(all_seed_results, strategies)

    md_summary = generate_markdown_summary(summary, strategies, config)
    print(f"\n{'=' * 70}")
    print(md_summary)

    save_results(all_seed_results, summary, strategies, config, output_dir)


if __name__ == "__main__":
    main()
