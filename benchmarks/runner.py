#!/usr/bin/env python3
"""Benchmark runner. Simulated closed-loop evaluation using LNPDB as oracle.

Usage:
    python -m benchmarks.runner --strategies random,discrete_xgb_greedy --rounds 5
    python -m benchmarks.runner --strategies all --rounds 10 --n-seeds 500
    python -m benchmarks.runner --strategies discrete_xgb_greedy --feature-type lantern --reduction pls
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from LNPBO.benchmarks._runner_config import (
    ACQ_TYPE_MAP,
    AITCHISON_STRATEGIES,
    ALL_STRATEGIES,
    COMPOSITIONAL_STRATEGIES,
    MIXED_STRATEGIES,
    PLS_STRATEGIES,
    STRATEGY_COLORS,
    STRATEGY_CONFIGS,
    STRATEGY_DISPLAY,
    TANIMOTO_STRATEGIES,
    classify_feature_columns,
    strategy_to_optimizer_kwargs,
)
from LNPBO.benchmarks._runner_conformal import (
    run_discrete_cumulative_split_conformal_ucb_baseline,
    run_discrete_online_conformal_strategy,
)
from LNPBO.benchmarks._runner_data import LNPDBOracle, prepare_benchmark_data, select_warmup_seed
from LNPBO.benchmarks._runner_history import _run_random, compute_metrics, init_history, update_history
from LNPBO.benchmarks._runner_logging import _log_round_complete, _log_round_start, _ts
from LNPBO.runtime_paths import benchmark_results_root, package_root_from

__all__ = [
    "_log_round_complete",
    "_log_round_start",
    "_run_random",
    "_ts",
    "ACQ_TYPE_MAP",
    "AITCHISON_STRATEGIES",
    "ALL_STRATEGIES",
    "COMPOSITIONAL_STRATEGIES",
    "LNPDBOracle",
    "MIXED_STRATEGIES",
    "PLS_STRATEGIES",
    "STRATEGY_COLORS",
    "STRATEGY_CONFIGS",
    "STRATEGY_DISPLAY",
    "TANIMOTO_STRATEGIES",
    "classify_feature_columns",
    "compute_metrics",
    "init_history",
    "prepare_benchmark_data",
    "run_discrete_cumulative_split_conformal_ucb_baseline",
    "run_discrete_online_conformal_strategy",
    "select_warmup_seed",
    "strategy_to_optimizer_kwargs",
    "update_history",
]


def plot_results(all_results, output_path="benchmark_output.png"):
    """Generate a two-panel benchmark summary figure."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.pad": 4,
            "ytick.major.pad": 4,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1 = axes[0]
    for name, result in all_results.items():
        bsf = result["history"]["best_so_far"]
        n_eval = result["history"]["n_evaluated"]
        label = STRATEGY_DISPLAY.get(name, name)
        color = STRATEGY_COLORS.get(name)
        style = "--" if name == "random" else "-"
        ax1.plot(n_eval, bsf, style, label=label, color=color, linewidth=1.5, markersize=0)
    ax1.set_xlabel("Formulations evaluated")
    ax1.set_ylabel("Best value found")
    ax1.legend(fontsize=7.5, frameon=True, fancybox=False, edgecolor="#cccccc", loc="lower right")
    ax1.grid(True, alpha=0.15, linewidth=0.5)
    ax1.text(-0.12, 1.05, "A", transform=ax1.transAxes, fontsize=14, fontweight="bold", va="top")

    ax2 = axes[1]
    k_values = sorted(next(iter(all_results.values()))["metrics"]["top_k_recall"].keys())
    x = np.arange(len(k_values))
    n_strats = len(all_results)
    width = 0.8 / n_strats
    for i, (name, result) in enumerate(all_results.items()):
        recalls = [result["metrics"]["top_k_recall"][k] for k in k_values]
        label = STRATEGY_DISPLAY.get(name, name)
        color = STRATEGY_COLORS.get(name)
        ax2.bar(x + i * width, recalls, width, label=label, color=color, edgecolor="white", linewidth=0.3)
    ax2.set_xlabel("K (top-K formulations)")
    ax2.set_ylabel("Recall")
    ax2.set_xticks(x + width * (n_strats - 1) / 2)
    ax2.set_xticklabels([str(k) for k in k_values])
    ax2.set_ylim(0, 1.0)
    ax2.legend(fontsize=7.5, frameon=True, fancybox=False, edgecolor="#cccccc")
    ax2.grid(True, alpha=0.15, linewidth=0.5, axis="y")
    ax2.text(-0.12, 1.05, "B", transform=ax2.transAxes, fontsize=14, fontweight="bold", va="top")

    fig.tight_layout(w_pad=3)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {output_path}")


def main():
    """CLI entry point for the standalone benchmark runner."""
    parser = argparse.ArgumentParser(
        description="LNPBO Benchmark: Simulated closed-loop BO evaluation",
    )
    parser.add_argument(
        "--strategies",
        type=str,
        default="random,discrete_xgb_greedy",
        help=f"Comma-separated strategies (or 'all'). Options: {','.join(ALL_STRATEGIES)}",
    )
    parser.add_argument("--rounds", type=int, default=15, help="Number of rounds (default: 15)")
    parser.add_argument("--batch-size", type=int, default=12, help="Batch size per round (default: 12)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--n-seeds", type=int, default=200, help="Size of initial seed pool (default: 200)")
    parser.add_argument("--subset", type=int, default=None, help="Use a subset of LNPDB (for fast testing)")
    parser.add_argument("--kappa", type=float, default=5.0, help="UCB kappa (default: 5.0)")
    parser.add_argument("--xi", type=float, default=0.01, help="EI/LogEI xi (default: 0.01)")
    parser.add_argument(
        "--normalize",
        type=str,
        default="copula",
        choices=["none", "zscore", "copula"],
        help="Target normalization for GP (default: copula)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output prefix (default: benchmark_results/<strategies>)",
    )
    parser.add_argument("--no-plot", action="store_true", help="Skip plotting")
    parser.add_argument(
        "--feature-type",
        type=str,
        default="mfp",
        choices=[
            "mfp",
            "mordred",
            "unimol",
            "count_mfp",
            "rdkit",
            "chemeleon",
            "lion",
            "raw_mfp",
            "raw_unimol",
            "raw_count_mfp",
            "raw_rdkit",
            "raw_chemeleon",
            "concat",
            "raw_concat",
            "lantern",
            "raw_lantern",
            "lantern_unimol",
            "raw_lantern_unimol",
            "lantern_mordred",
            "raw_lantern_mordred",
            "lantern_il_only",
            "lantern_il_hl",
            "lantern_il_noratios",
            "lion_il_only",
            "mordred_il_only",
            "unimol_il_only",
            "mfp_il_only",
            "count_mfp_il_only",
            "chemeleon_il_only",
            "chemeleon_helper_only",
            "agile",
            "agile_il_only",
            "ratios_only",
        ],
        help="Feature type (default: mfp).",
    )
    parser.add_argument("--n-pcs", type=int, default=None, help="Override PCA/PLS components per role")
    parser.add_argument(
        "--reduction",
        type=str,
        default="pca",
        choices=["pca", "pls", "none"],
        help="Dimensionality reduction method (default: pca)",
    )
    parser.add_argument(
        "--context-features",
        action="store_true",
        help="Include one-hot experimental context (cell type, target, RoA, etc.)",
    )
    parser.add_argument(
        "--fp-radius",
        type=int,
        default=None,
        help="Morgan FP radius (default: 3 for mfp, 3 for count_mfp)",
    )
    parser.add_argument(
        "--fp-bits",
        type=int,
        default=None,
        help="Morgan FP bit size (default: 1024 for mfp, 2048 for count_mfp)",
    )
    args = parser.parse_args()

    if args.strategies == "all":
        strategies = ALL_STRATEGIES
    else:
        strategies = [s.strip() for s in args.strategies.split(",")]
        for strategy in strategies:
            if strategy not in STRATEGY_CONFIGS:
                parser.error(f"Unknown strategy: {strategy}. Choose from: {ALL_STRATEGIES}")

    package_root = package_root_from(__file__, levels_up=2)
    results_dir = benchmark_results_root(package_root)
    results_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = str(results_dir / f"{'_'.join(strategies[:3])}") if args.output is None else args.output

    print("=" * 70)
    print("LNPBO BENCHMARK")
    print("=" * 70)
    print(f"Strategies: {strategies}")
    print(f"Rounds: {args.rounds}, Batch size: {args.batch_size}")
    print(f"Seed pool: {args.n_seeds}, Random seed: {args.seed}")
    print(f"Target normalization: {args.normalize}")
    print(f"Context features: {args.context_features}")
    print()

    pca_data = prepare_benchmark_data(
        n_seed=args.n_seeds,
        random_seed=args.seed,
        subset=args.subset,
        reduction=args.reduction,
        feature_type=args.feature_type,
        n_pcs=args.n_pcs,
        context_features=args.context_features,
        fp_radius=args.fp_radius,
        fp_bits=args.fp_bits,
    )
    pls_data = None
    if any(strategy in PLS_STRATEGIES for strategy in strategies):
        pls_data = prepare_benchmark_data(
            n_seed=args.n_seeds,
            random_seed=args.seed,
            subset=args.subset,
            reduction="pls",
            feature_type=args.feature_type,
            n_pcs=args.n_pcs,
            context_features=args.context_features,
        )

    tanimoto_data = None
    if any(strategy in TANIMOTO_STRATEGIES for strategy in strategies):
        tanimoto_data = prepare_benchmark_data(
            n_seed=args.n_seeds,
            random_seed=args.seed,
            subset=args.subset,
            reduction="none",
            feature_type="count_mfp",
            context_features=args.context_features,
        )

    aitchison_data = None
    if any(strategy in AITCHISON_STRATEGIES for strategy in strategies):
        aitchison_data = prepare_benchmark_data(
            n_seed=args.n_seeds,
            random_seed=args.seed,
            subset=args.subset,
            reduction="none",
            feature_type="ratios_only",
            context_features=args.context_features,
        )

    compositional_data = None
    compositional_kernel_kwargs = None
    if any(strategy in COMPOSITIONAL_STRATEGIES or strategy in MIXED_STRATEGIES for strategy in strategies):
        compositional_data = prepare_benchmark_data(
            n_seed=args.n_seeds,
            random_seed=args.seed,
            subset=args.subset,
            reduction="pca",
            feature_type="lantern",
            context_features=args.context_features,
        )
        _, _comp_df, comp_fcols, _, _, _ = compositional_data
        compositional_kernel_kwargs = classify_feature_columns(comp_fcols)

    all_results = {}
    for strategy in strategies:
        print(f"\n{'=' * 70}")
        print(f"Running: {strategy}")
        print(f"{'=' * 70}")
        t0 = time.time()

        if (strategy in COMPOSITIONAL_STRATEGIES or strategy in MIXED_STRATEGIES) and compositional_data is not None:
            s_dataset, s_df, s_fcols, s_seed, s_oracle, s_topk = compositional_data
        elif strategy in AITCHISON_STRATEGIES and aitchison_data is not None:
            s_dataset, s_df, s_fcols, s_seed, s_oracle, s_topk = aitchison_data
        elif strategy in TANIMOTO_STRATEGIES and tanimoto_data is not None:
            s_dataset, s_df, s_fcols, s_seed, s_oracle, s_topk = tanimoto_data
        elif strategy in PLS_STRATEGIES and pls_data is not None:
            s_dataset, s_df, s_fcols, s_seed, s_oracle, s_topk = pls_data
        else:
            s_dataset, s_df, s_fcols, s_seed, s_oracle, s_topk = pca_data

        config = STRATEGY_CONFIGS[strategy]
        if config["type"] == "random":
            history = _run_random(s_df, s_seed, s_oracle, args.batch_size, args.rounds, args.seed)
        elif config["type"] == "discrete_online_conformal_exact":
            history = run_discrete_online_conformal_strategy(
                s_df,
                s_fcols,
                s_seed,
                s_oracle,
                batch_size=args.batch_size,
                n_rounds=args.rounds,
                seed=args.seed,
                kappa=args.kappa,
                normalize=args.normalize,
                encoded_dataset=s_dataset,
            )
        elif config["type"] == "discrete_online_conformal_baseline":
            history = run_discrete_cumulative_split_conformal_ucb_baseline(
                s_df,
                s_fcols,
                s_seed,
                s_oracle,
                batch_size=args.batch_size,
                n_rounds=args.rounds,
                seed=args.seed,
                kappa=args.kappa,
                normalize=args.normalize,
                encoded_dataset=s_dataset,
            )
        else:
            from LNPBO.optimization.optimizer import Optimizer

            from ._optimizer_runner import OptimizerRunner

            gp_kernel_kwargs = compositional_kernel_kwargs if strategy in COMPOSITIONAL_STRATEGIES or strategy in MIXED_STRATEGIES else None
            opt_kwargs = strategy_to_optimizer_kwargs(strategy, kernel_kwargs=gp_kernel_kwargs)
            optimizer = Optimizer(
                random_seed=args.seed,
                kappa=args.kappa,
                xi=args.xi,
                normalize=args.normalize,
                batch_size=args.batch_size,
                **opt_kwargs,
            )
            runner = OptimizerRunner(optimizer)
            history = runner.run(
                s_df,
                s_fcols,
                s_seed,
                s_oracle,
                n_rounds=args.rounds,
                batch_size=args.batch_size,
                encoded_dataset=s_dataset,
                top_k_values=s_topk,
            )

        elapsed = time.time() - t0
        metrics = compute_metrics(history, s_topk, len(s_df))
        all_results[strategy] = {
            "history": history,
            "metrics": metrics,
            "elapsed": elapsed,
        }

        print(f"  Time: {elapsed:.1f}s")
        print(f"  Final best: {metrics['final_best']:.4f}")
        print(f"  AUC: {metrics['auc']:.4f}")
        print(f"  Top-K recall: { {k: f'{v:.1%}' for k, v in metrics['top_k_recall'].items()} }")

    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    header = f"{'Strategy':<24} {'Final Best':>12} {'AUC':>10} {'Top-10':>8} {'Top-50':>8} {'Top-100':>8} {'Time':>8}"
    print(header)
    print("-" * len(header))
    for name, result in all_results.items():
        metrics = result["metrics"]
        recall = metrics["top_k_recall"]
        print(
            f"{name:<24} {metrics['final_best']:>12.4f} {metrics['auc']:>10.4f} "
            f"{recall.get(10, 0):>7.1%} {recall.get(50, 0):>7.1%} {recall.get(100, 0):>7.1%} "
            f"{result['elapsed']:>7.1f}s"
        )

    json_path = f"{output_prefix}.json"
    serializable = {
        "config": {
            "strategies": strategies,
            "rounds": args.rounds,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "n_seeds": args.n_seeds,
            "subset": args.subset,
            "kappa": args.kappa,
            "xi": args.xi,
            "normalize": args.normalize,
            "feature_type": args.feature_type,
            "n_pcs": args.n_pcs,
            "reduction": args.reduction,
            "context_features": args.context_features,
            "fp_radius": args.fp_radius,
            "fp_bits": args.fp_bits,
        },
        "results": {},
    }
    for name, result in all_results.items():
        entry = {
            "metrics": result["metrics"],
            "elapsed": result["elapsed"],
            "best_so_far": result["history"]["best_so_far"],
            "round_best": result["history"]["round_best"],
            "n_evaluated": result["history"]["n_evaluated"],
        }
        if "coverage" in result["history"]:
            entry["coverage"] = result["history"]["coverage"]
        if "conformal_quantile" in result["history"]:
            entry["conformal_quantile"] = result["history"]["conformal_quantile"]
        serializable["results"][name] = entry
    with open(json_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {json_path}")

    if not args.no_plot:
        plot_results(all_results, output_path=f"{output_prefix}.png")


if __name__ == "__main__":
    main()
