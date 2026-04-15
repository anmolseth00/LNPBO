#!/usr/bin/env python3
"""Unified analysis of ablation experiments.

Loads all ablation results, computes bootstrap CIs, paired Wilcoxon tests,
and generates publication-ready figures.

Usage:
    python -m experiments.analysis.analyze_ablations --experiment encoding
    python -m experiments.analysis.analyze_ablations --experiment all
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from LNPBO.runtime_paths import (
    benchmark_results_root,
    import_from_layout,
    package_root_from,
    workspace_root,
)

_PACKAGE_ROOT = package_root_from(__file__, levels_up=3)
_stats = import_from_layout(
    __package__,
    source_name="benchmarks.stats",
    installed_name="LNPBO.benchmarks.stats",
)
benjamini_hochberg = _stats.benjamini_hochberg
bootstrap_ci = _stats.bootstrap_ci
cohens_d_paired = _stats.cohens_d_paired
paired_wilcoxon = _stats.paired_wilcoxon

RESULTS_BASE = benchmark_results_root(_PACKAGE_ROOT) / "ablations"
FIG_DIR = workspace_root(_PACKAGE_ROOT) / "experiments" / "analysis" / "figures"


def load_experiment_results(experiment_name):
    """Load all per-seed result JSONs for an experiment into a tidy DataFrame."""
    results_dir = RESULTS_BASE / experiment_name
    if not results_dir.exists():
        print(f"No results found at {results_dir}")
        return pd.DataFrame()

    rows = []
    for path in sorted(results_dir.rglob("*.json")):
        with open(path) as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                continue

        result = data.get("result", {})
        condition = data.get("condition", {})
        metrics = result.get("metrics", {})
        recall = metrics.get("top_k_recall", {})

        row = {
            "experiment": data.get("experiment", experiment_name),
            "study_id": data.get("study_id"),
            "pmid": data.get("pmid"),
            "strategy": data.get("strategy"),
            "seed": data.get("seed"),
            "condition_label": condition.get("label", ""),
            "feature_type": condition.get("feature_type"),
            "batch_size": condition.get("batch_size"),
            "n_rounds": condition.get("n_rounds"),
            "n_pcs": condition.get("n_pcs"),
            "reduction": condition.get("reduction"),
            "seed_fraction": condition.get("seed_fraction"),
            "warmup": json.dumps(condition.get("warmup")) if condition.get("warmup") else None,
            "top_5_recall": recall.get("5", 0.0),
            "top_10_recall": recall.get("10", 0.0),
            "top_20_recall": recall.get("20", 0.0),
            "auc": metrics.get("auc"),
            "final_best": metrics.get("final_best"),
            "elapsed": result.get("elapsed"),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} results for experiment '{experiment_name}'")
    return df


def summarize_by_condition(df, group_col="condition_label", metric="top_5_recall"):
    """Compute per-condition summary with bootstrap CIs."""
    summaries = []
    for cond, grp in df.groupby(group_col):
        for strategy, sgrp in grp.groupby("strategy"):
            # Average across seeds per study first, then average across studies
            study_means = sgrp.groupby("study_id")[metric].mean()
            vals = study_means.values
            if len(vals) == 0:
                continue
            ci_lo, ci_hi = bootstrap_ci(vals) if len(vals) >= 3 else (np.nan, np.nan)
            summaries.append(
                {
                    group_col: cond,
                    "strategy": strategy,
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                    "ci_lo": ci_lo,
                    "ci_hi": ci_hi,
                    "n_studies": len(vals),
                    "n_results": len(sgrp),
                }
            )
    return pd.DataFrame(summaries)


def plot_bar_with_ci(summary_df, group_col, metric_label, title, out_path):
    """Bar plot with bootstrap CI error bars."""
    conditions = summary_df[group_col].unique()
    strategies = sorted(summary_df["strategy"].unique())
    n_cond = len(conditions)
    n_strat = len(strategies)

    fig, ax = plt.subplots(figsize=(max(6, n_cond * 1.5), 4.5))
    width = 0.8 / n_strat
    x = np.arange(n_cond)

    colors = plt.cm.Set2(np.linspace(0, 1, n_strat))

    for i, strategy in enumerate(strategies):
        means, errs_lo, errs_hi = [], [], []
        for cond in conditions:
            row = summary_df[(summary_df[group_col] == cond) & (summary_df["strategy"] == strategy)]
            if len(row) == 0:
                means.append(0)
                errs_lo.append(0)
                errs_hi.append(0)
            else:
                m = row["mean"].values[0]
                means.append(m)
                errs_lo.append(m - row["ci_lo"].values[0] if not np.isnan(row["ci_lo"].values[0]) else 0)
                errs_hi.append(row["ci_hi"].values[0] - m if not np.isnan(row["ci_hi"].values[0]) else 0)

        ax.bar(
            x + i * width,
            means,
            width,
            label=strategy,
            color=colors[i],
            edgecolor="white",
            linewidth=0.3,
            yerr=[errs_lo, errs_hi],
            capsize=2,
            error_kw={"linewidth": 0.8},
        )

    ax.set_xticks(x + width * (n_strat - 1) / 2)
    ax.set_xticklabels(conditions, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(metric_label)
    ax.set_title(title)
    ax.legend(fontsize=7, frameon=True, fancybox=False, edgecolor="#cccccc", ncol=min(3, n_strat))
    ax.grid(True, alpha=0.15, linewidth=0.5, axis="y")
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def pairwise_significance_heatmap(df, group_col, metric="top_5_recall", out_path=None):
    """Compute pairwise Wilcoxon p-values between conditions with BH-FDR correction."""
    conditions = sorted(df[group_col].unique())
    n = len(conditions)
    raw_pvals = np.ones((n, n))

    # Collect all raw p-values for BH correction
    pair_indices = []
    pair_p_raw = []
    for i in range(n):
        for j in range(i + 1, n):
            ci = df[df[group_col] == conditions[i]]
            cj = df[df[group_col] == conditions[j]]

            # Per-study means (averaged across seeds and strategies)
            mi = ci.groupby("study_id")[metric].mean()
            mj = cj.groupby("study_id")[metric].mean()

            common = mi.index.intersection(mj.index)
            if len(common) < 3:
                pair_indices.append((i, j))
                pair_p_raw.append(1.0)
                continue

            p = paired_wilcoxon(mi.loc[common].values, mj.loc[common].values)
            pair_indices.append((i, j))
            pair_p_raw.append(p)
            raw_pvals[i, j] = p
            raw_pvals[j, i] = p

    # Apply BH-FDR correction
    pvals_bh = np.ones((n, n))
    if pair_p_raw:
        p_arr = np.array(pair_p_raw)
        p_adjusted, _ = benjamini_hochberg(p_arr)
        for k, (i, j) in enumerate(pair_indices):
            pvals_bh[i, j] = p_adjusted[k]
            pvals_bh[j, i] = p_adjusted[k]

    if out_path:
        fig, ax = plt.subplots(figsize=(max(5, n * 0.8), max(4, n * 0.7)))
        im = ax.imshow(-np.log10(pvals_bh + 1e-300), cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(conditions, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(conditions, fontsize=7)
        ax.set_title("Pairwise significance (-log10 p_BH)")
        plt.colorbar(im, ax=ax, shrink=0.8)

        for i in range(n):
            for j in range(n):
                if i != j:
                    p = pvals_bh[i, j]
                    txt = f"{p:.3f}" if p >= 0.001 else f"{p:.1e}"
                    sig = "*" if p < 0.05 else ""
                    color = "white" if -np.log10(p + 1e-300) > 2 else "black"
                    ax.text(j, i, txt + sig, ha="center", va="center", fontsize=6, color=color)

        fig.tight_layout()
        fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
        fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out_path}")

    return pvals_bh, conditions


def variance_decomposition(df, metric="top_5_recall"):
    """Type-II SS variance decomposition: Study, Strategy, Condition, Seed.

    Computes proper sums of squares as:
        SS_factor = n_per_level * sum((group_mean - grand_mean)^2)
    """
    values = df[metric].values
    grand_mean = np.mean(values)
    ss_total = np.sum((values - grand_mean) ** 2)
    N = len(values)

    factors = ["study_id", "strategy", "condition_label", "seed"]
    factors = [f for f in factors if f in df.columns and df[f].nunique() > 1]

    components = {}
    df_degrees = {}
    for factor in factors:
        group_means = df.groupby(factor)[metric].mean()
        n_levels = len(group_means)
        # SS = sum over groups: n_i * (group_mean_i - grand_mean)^2
        ss = 0.0
        for level, mean_val in group_means.items():
            n_i = len(df[df[factor] == level])
            ss += n_i * (mean_val - grand_mean) ** 2
        components[factor] = ss
        df_degrees[factor] = n_levels - 1

    ss_explained = sum(components.values())
    ss_residual = max(0, ss_total - ss_explained)
    components["residual"] = ss_residual
    df_residual = max(1, N - 1 - sum(df_degrees.values()))
    df_degrees["residual"] = df_residual

    pct = {k: v / ss_total * 100 if ss_total > 0 else 0 for k, v in components.items()}

    return components, pct


def effect_sizes_vs_baseline(df, group_col, metric="top_5_recall", baseline=None):
    """Compute Cohen's d for each condition vs baseline, with BH-FDR."""
    conditions = sorted(df[group_col].unique())
    if baseline is None:
        # Use condition with "random" in name, or the first one
        baseline_cands = [c for c in conditions if "random" in str(c).lower()]
        if baseline_cands:
            baseline = baseline_cands[0]
        else:
            baseline = conditions[0]

    base_df = df[df[group_col] == baseline]
    base_study = base_df.groupby("study_id")[metric].mean()

    rows = []
    raw_pvals = []
    for cond in conditions:
        if cond == baseline:
            continue
        cond_df = df[df[group_col] == cond]
        cond_study = cond_df.groupby("study_id")[metric].mean()

        common = base_study.index.intersection(cond_study.index)
        if len(common) < 3:
            continue

        x = cond_study.loc[common].values
        y = base_study.loc[common].values
        d, ci_lo, ci_hi, interp = cohens_d_paired(x, y)
        p = paired_wilcoxon(x, y)
        raw_pvals.append((cond, p))
        rows.append(
            {
                "condition": cond,
                "d": d,
                "d_ci_lo": ci_lo,
                "d_ci_hi": ci_hi,
                "interpretation": interp,
                "p_raw": p,
                "n": len(common),
            }
        )

    # BH-FDR
    if raw_pvals:
        p_arr = np.array([p for _, p in raw_pvals])
        p_adj, _ = benjamini_hochberg(p_arr)
        p_bh_map = dict(zip([c for c, _ in raw_pvals], p_adj))
    else:
        p_bh_map = {}

    for row in rows:
        row["p_bh"] = p_bh_map.get(row["condition"], 1.0)

    return rows, baseline


def analyze_experiment(experiment_name):
    """Full analysis pipeline for one experiment."""
    df = load_experiment_results(experiment_name)
    if df.empty:
        return

    exp_fig_dir = FIG_DIR / experiment_name
    exp_fig_dir.mkdir(parents=True, exist_ok=True)

    group_col = "condition_label"
    metric = "top_5_recall"

    # Summary table
    summary = summarize_by_condition(df, group_col, metric)
    print(f"\n{'=' * 60}")
    print(f"  {experiment_name.upper()} - Top-5% Recall by condition")
    print(f"{'=' * 60}")
    for _, row in summary.sort_values("mean", ascending=False).iterrows():
        ci = f"[{row['ci_lo']:.1%}, {row['ci_hi']:.1%}]" if not np.isnan(row["ci_lo"]) else ""
        print(
            f"  {row[group_col]:>20s} / {row['strategy']:>25s}: "
            f"{row['mean']:.1%} +/- {row['std']:.1%} {ci} (n={row['n_studies']})"
        )

    # Effect sizes vs baseline condition
    effect_rows, baseline = effect_sizes_vs_baseline(df, group_col, metric)
    if effect_rows:
        print(f"\n  Effect sizes vs baseline ('{baseline}'), BH-FDR corrected:")
        print(f"    {'Condition':>20s}  {'d':>6}  {'95% CI':>17}  {'Interp':>10}  {'p_BH':>8}  {'Sig':>3}")
        print("    " + "-" * 75)
        for row in sorted(effect_rows, key=lambda r: -r["d"]):
            sig = "***" if row["p_bh"] < 0.001 else "**" if row["p_bh"] < 0.01 else "*" if row["p_bh"] < 0.05 else ""
            print(
                f"    {row['condition']:>20s}  {row['d']:>6.2f}  "
                f"[{row['d_ci_lo']:>6.2f}, {row['d_ci_hi']:>6.2f}]  "
                f"{row['interpretation']:>10}  {row['p_bh']:>8.4f}  {sig:>3}"
            )

    # Bar plot
    plot_bar_with_ci(
        summary,
        group_col,
        "Top-5% Recall",
        f"{experiment_name.title()} Ablation: Top-5% Recall",
        exp_fig_dir / f"{experiment_name}_top5_recall",
    )

    # Also do top-10%
    summary_10 = summarize_by_condition(df, group_col, "top_10_recall")
    plot_bar_with_ci(
        summary_10,
        group_col,
        "Top-10% Recall",
        f"{experiment_name.title()} Ablation: Top-10% Recall",
        exp_fig_dir / f"{experiment_name}_top10_recall",
    )

    # Pairwise significance (BH-FDR corrected)
    _pvals, _conds = pairwise_significance_heatmap(
        df,
        group_col,
        metric,
        out_path=exp_fig_dir / f"{experiment_name}_pairwise_significance",
    )

    # Variance decomposition
    _components, pct = variance_decomposition(df, metric)
    print(f"\n  Variance decomposition ({metric}):")
    for k, v in sorted(pct.items(), key=lambda x: -x[1]):
        print(f"    {k:>20s}: {v:.1f}%")

    # Strategy x Condition interaction (descriptive)
    if df["strategy"].nunique() > 1 and df[group_col].nunique() > 1:
        print("\n  Strategy x Condition interaction (study-level means):")
        pivot = (
            df.groupby(["strategy", group_col])
            .apply(lambda g: g.groupby("study_id")[metric].mean().mean())
            .unstack(fill_value=np.nan)
        )
        if pivot.shape[0] > 0 and pivot.shape[1] > 1:
            # Check if rankings change across conditions
            rankings_by_cond = {}
            for col in pivot.columns:
                rankings_by_cond[col] = list(pivot[col].sort_values(ascending=False).index)
            top1_stable = len(set(r[0] for r in rankings_by_cond.values())) == 1
            print(f"    Top strategy stable across conditions: {top1_stable}")
            for cond, ranking in rankings_by_cond.items():
                print(f"    {cond}: {' > '.join(ranking[:3])}")

    # Save summary JSON
    out = {
        "experiment": experiment_name,
        "n_results": len(df),
        "n_studies": df["study_id"].nunique(),
        "n_strategies": df["strategy"].nunique(),
        "n_conditions": df[group_col].nunique(),
        "variance_decomposition_pct": pct,
        "effect_sizes": effect_rows if effect_rows else [],
        "summary": summary.to_dict(orient="records"),
    }
    json_path = exp_fig_dir / f"{experiment_name}_summary.json"
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  Summary saved to {json_path}")


def cross_experiment_synthesis():
    """Compare sensitivity across all experiments."""
    experiments = ["encoding", "batch_size", "budget", "warmup", "pca", "kappa", "kernel"]
    all_variance = {}

    for exp in experiments:
        df = load_experiment_results(exp)
        if df.empty:
            continue
        _, pct = variance_decomposition(df, "top_5_recall")
        all_variance[exp] = pct

    if not all_variance:
        print("No experiment results available for synthesis.")
        return

    print(f"\n{'=' * 60}")
    print("CROSS-EXPERIMENT SENSITIVITY RANKING")
    print(f"{'=' * 60}")

    # For each experiment, what % of variance does the ablated factor explain?
    factor_map = {
        "encoding": "condition_label",
        "batch_size": "condition_label",
        "budget": "condition_label",
        "warmup": "condition_label",
        "pca": "condition_label",
    }

    ranking = []
    for exp, pct in all_variance.items():
        factor = factor_map.get(exp, "condition_label")
        factor_pct = pct.get(factor, 0.0)
        ranking.append((exp, factor_pct))
        print(f"  {exp:>15s}: condition explains {factor_pct:.1f}% of variance")

    ranking.sort(key=lambda x: -x[1])
    print(f"\n  Sensitivity ranking: {' > '.join(r[0] for r in ranking)}")


def main():
    parser = argparse.ArgumentParser(description="Analyze ablation results")
    parser.add_argument("--experiment", type=str, default="all", help="Experiment name or 'all'")
    args = parser.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    if args.experiment == "all":
        for exp in ["encoding", "batch_size", "budget", "warmup", "pca", "kappa", "kernel"]:
            analyze_experiment(exp)
        cross_experiment_synthesis()
    else:
        analyze_experiment(args.experiment)


if __name__ == "__main__":
    main()
