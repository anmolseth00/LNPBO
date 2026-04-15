#!/usr/bin/env python3
"""Study characteristic correlation analysis.

Answers: "What study properties predict how well BO works?"

For each study, computes the BO advantage (best BO family recall minus random
recall) and correlates it against study metadata: pool size, IL diversity,
study type, and ratio variability. Outputs a correlation table, a JSON file,
and a 2x2 scatter plot figure.

Usage:
    .venv/bin/python -m benchmarks.analyze_study_predictors
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as sp_stats

from LNPBO.runtime_paths import benchmark_results_root, package_root_from, paper_root

_PACKAGE_ROOT = package_root_from(__file__, levels_up=2)
_RESULTS_ROOT = benchmark_results_root(_PACKAGE_ROOT)
_PAPER_ROOT = paper_root(_PACKAGE_ROOT)
sys.path.insert(0, str(_PAPER_ROOT))

from paper.figure_style import (
    DOUBLE_COL,
    STUDY_TYPE_COLORS,
    light_ygrid,
    panel_label,
    save_figure,
    setup_style,
)

RESULTS_DIR = _RESULTS_ROOT / "within_study"
STUDIES_JSON = _PACKAGE_ROOT / "experiments" / "data_integrity" / "studies_with_ids.json"
FIG_DIR = _PAPER_ROOT / "figures"
OUTPUT_JSON = _RESULTS_ROOT / "study_predictor_correlations.json"

from .constants import SEEDS
from .strategy_registry import STRATEGY_FAMILY

BO_FAMILIES = ["NGBoost", "RF", "XGBoost", "CASMOPolitan", "Deep Ensemble", "GP (sklearn)", "LNPBO"]

STUDY_TYPE_DISPLAY = {
    "il_diverse_fixed_ratios": "Fixed ratio",
    "il_diverse_variable_ratios": "Variable ratio",
    "ratio_only": "Ratio-only",
}

STUDY_TYPE_MARKERS = {
    "il_diverse_fixed_ratios": "o",
    "il_diverse_variable_ratios": "s",
    "ratio_only": "D",
}

STUDY_TYPE_PLOT_COLORS = {
    "il_diverse_fixed_ratios": STUDY_TYPE_COLORS["IL-diverse"],
    "il_diverse_variable_ratios": "#228833",
    "ratio_only": STUDY_TYPE_COLORS["ratio-only"],
}


def load_all_results():
    from .result_loading import load_benchmark_results
    return load_benchmark_results(RESULTS_DIR)


def build_tables(results):
    study_info = {}
    result_map = {}
    strategies = set()
    for r in results:
        pmid = r.get("study_id", str(int(r["pmid"])))
        strategy = r["strategy"]
        seed = r["seed"]
        study_info[pmid] = r["study_info"]
        result_map[(pmid, strategy, seed)] = r
        strategies.add(strategy)
    pmids = sorted(study_info.keys())
    strategies = sorted(strategies)
    return study_info, result_map, pmids, strategies


def get_top5_recall(result_map, pmid, strategy, seed):
    key = (pmid, strategy, seed)
    if key not in result_map:
        return None
    try:
        return result_map[key]["result"]["metrics"]["top_k_recall"]["5"]
    except (KeyError, TypeError):
        return None


def load_study_metadata():
    """Load study metadata from studies_with_ids.json, keyed by study_id."""
    data = json.loads(STUDIES_JSON.read_text())
    meta = {}
    for entry in data:
        sid = entry.get("study_id") or str(entry.get("pmid"))
        meta[sid] = entry
    return meta


def compute_study_bo_advantage(study_info, result_map, pmids, strategies):
    """For each study, compute mean random recall, best BO family recall, and advantage."""
    rows = []

    for pmid in pmids:
        si = study_info[pmid]

        # Random baseline: mean across seeds
        rand_vals = [get_top5_recall(result_map, pmid, "random", s) for s in SEEDS]
        rand_vals = [v for v in rand_vals if v is not None]
        if not rand_vals:
            continue
        mean_random = np.mean(rand_vals)

        # Per-family: mean across all strategies in family and all seeds
        family_means = {}
        for fam in BO_FAMILIES:
            fam_vals = []
            for strategy in strategies:
                if STRATEGY_FAMILY.get(strategy) != fam:
                    continue
                for seed in SEEDS:
                    v = get_top5_recall(result_map, pmid, strategy, seed)
                    if v is not None:
                        fam_vals.append(v)
            if fam_vals:
                family_means[fam] = np.mean(fam_vals)

        if not family_means:
            continue

        best_fam = max(family_means, key=family_means.get)
        best_bo = family_means[best_fam]
        bo_advantage = best_bo - mean_random

        # Mean across all BO families (not just the best)
        all_bo_mean = np.mean(list(family_means.values()))
        mean_bo_advantage = all_bo_mean - mean_random

        rows.append(
            {
                "study_id": str(pmid),
                "pmid": pmid,
                "n_formulations": si["n_formulations"],
                "n_unique_il": si["n_unique_il"],
                "study_type": si["study_type"],
                "il_ratio_std": si.get("il_ratio_std", 0.0),
                "mean_random": float(mean_random),
                "best_bo_family": best_fam,
                "best_bo_recall": float(best_bo),
                "bo_advantage": float(bo_advantage),
                "mean_bo_recall": float(all_bo_mean),
                "mean_bo_advantage": float(mean_bo_advantage),
                "family_means": {k: float(v) for k, v in family_means.items()},
            }
        )

    return rows


def compute_correlations(study_rows):
    """Compute Spearman rank correlations between study properties and BO advantage."""
    bo_adv = np.array([r["bo_advantage"] for r in study_rows])
    mean_bo_adv = np.array([r["mean_bo_advantage"] for r in study_rows])

    predictors = {
        "n_formulations": np.array([r["n_formulations"] for r in study_rows]),
        "n_unique_il": np.array([r["n_unique_il"] for r in study_rows]),
        "il_ratio_std": np.array([r["il_ratio_std"] for r in study_rows]),
        "log_n_formulations": np.log10(np.array([r["n_formulations"] for r in study_rows])),
    }

    corr_results = []
    for name, x in predictors.items():
        # vs best-family BO advantage
        rho, p = sp_stats.spearmanr(x, bo_adv)
        # vs mean BO advantage
        rho_mean, p_mean = sp_stats.spearmanr(x, mean_bo_adv)
        corr_results.append(
            {
                "predictor": name,
                "rho_best_bo": float(rho),
                "p_best_bo": float(p),
                "rho_mean_bo": float(rho_mean),
                "p_mean_bo": float(p_mean),
                "n": len(x),
            }
        )

    return corr_results


def print_correlation_table(corr_results, study_rows):
    print("\n" + "=" * 90)
    print("  STUDY CHARACTERISTIC CORRELATIONS WITH BO ADVANTAGE")
    print("=" * 90)

    print(f"\n  N = {len(study_rows)} studies\n")

    print(
        f"  {'Predictor':<22}  {'rho (best)':>10}  {'p (best)':>10}  {'rho (mean)':>10}  {'p (mean)':>10}  {'Sig':>5}"
    )
    print("  " + "-" * 75)

    for c in corr_results:
        sig = ""
        p = c["p_best_bo"]
        if p < 0.001:
            sig = "***"
        elif p < 0.01:
            sig = " **"
        elif p < 0.05:
            sig = "  *"
        print(
            f"  {c['predictor']:<22}  {c['rho_best_bo']:>10.3f}  {c['p_best_bo']:>10.4f}  "
            f"{c['rho_mean_bo']:>10.3f}  {c['p_mean_bo']:>10.4f}  {sig:>5}"
        )

    # Study type breakdown
    print("\n  BO Advantage by Study Type:")
    print(f"  {'Study Type':<30}  {'N':>3}  {'Mean Adv':>9}  {'Std':>7}  {'Mean Random':>11}")
    print("  " + "-" * 65)

    type_groups = defaultdict(list)
    for r in study_rows:
        type_groups[r["study_type"]].append(r)

    for stype in ["il_diverse_fixed_ratios", "il_diverse_variable_ratios", "ratio_only"]:
        group = type_groups.get(stype, [])
        if not group:
            continue
        advs = [r["bo_advantage"] for r in group]
        rands = [r["mean_random"] for r in group]
        label = STUDY_TYPE_DISPLAY.get(stype, stype)
        print(
            f"  {label:<30}  {len(group):>3}  {np.mean(advs):>9.3f}  "
            f"{np.std(advs, ddof=1):>7.3f}  {np.mean(rands):>11.3f}"
        )

    # Kruskal-Wallis test for study type effect
    groups_for_test = []
    for stype in ["il_diverse_fixed_ratios", "il_diverse_variable_ratios", "ratio_only"]:
        group = type_groups.get(stype, [])
        if len(group) >= 2:
            groups_for_test.append([r["bo_advantage"] for r in group])

    if len(groups_for_test) >= 2:
        H, p_kw = sp_stats.kruskal(*groups_for_test)
        print(f"\n  Kruskal-Wallis test (study type effect): H={H:.2f}, p={p_kw:.4f}")


def make_figure(study_rows, corr_results):
    setup_style()

    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL, DOUBLE_COL * 0.75))

    bo_adv = np.array([r["bo_advantage"] for r in study_rows])
    study_types = [r["study_type"] for r in study_rows]

    # Helper to add points colored by study type
    def scatter_by_type(ax, x, y, xlabel, rho, p):
        for stype in ["il_diverse_fixed_ratios", "il_diverse_variable_ratios", "ratio_only"]:
            mask = np.array([st == stype for st in study_types])
            if not mask.any():
                continue
            ax.scatter(
                x[mask],
                y[mask],
                c=STUDY_TYPE_PLOT_COLORS[stype],
                marker=STUDY_TYPE_MARKERS[stype],
                s=25,
                alpha=0.8,
                edgecolors="white",
                linewidths=0.3,
                label=STUDY_TYPE_DISPLAY[stype],
                zorder=3,
            )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("BO advantage (recall)")
        # Add correlation annotation
        sig_str = ""
        if p < 0.001:
            sig_str = "***"
        elif p < 0.01:
            sig_str = "**"
        elif p < 0.05:
            sig_str = "*"
        ax.text(
            0.97,
            0.05,
            f"$\\rho$ = {rho:.2f}{sig_str}\n(p = {p:.3f})",
            transform=ax.transAxes,
            fontsize=6.5,
            ha="right",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.9),
        )
        light_ygrid(ax)

    # Find the correlation results by name
    corr_map = {c["predictor"]: c for c in corr_results}

    # Panel (a): n_formulations vs BO advantage
    ax = axes[0, 0]
    x = np.array([r["n_formulations"] for r in study_rows])
    c = corr_map["n_formulations"]
    scatter_by_type(ax, x, bo_adv, "Study size ($n$ formulations)", c["rho_best_bo"], c["p_best_bo"])
    panel_label(ax, "a")

    # Panel (b): n_unique_il vs BO advantage
    ax = axes[0, 1]
    x = np.array([r["n_unique_il"] for r in study_rows])
    c = corr_map["n_unique_il"]
    scatter_by_type(ax, x, bo_adv, "Number of unique ILs", c["rho_best_bo"], c["p_best_bo"])
    panel_label(ax, "b")

    # Panel (c): il_ratio_std vs BO advantage
    ax = axes[1, 0]
    x = np.array([r["il_ratio_std"] for r in study_rows])
    c = corr_map["il_ratio_std"]
    scatter_by_type(ax, x, bo_adv, "Ratio variability (IL ratio std)", c["rho_best_bo"], c["p_best_bo"])
    panel_label(ax, "c")

    # Panel (d): BO advantage by study type (strip/box plot)
    ax = axes[1, 1]
    type_order = ["il_diverse_fixed_ratios", "il_diverse_variable_ratios", "ratio_only"]
    type_labels_short = ["Fixed\nratio", "Variable\nratio", "Ratio-\nonly"]
    positions = []
    for i, stype in enumerate(type_order):
        mask = np.array([st == stype for st in study_types])
        vals = bo_adv[mask]
        if len(vals) == 0:
            continue
        positions.append(i)
        color = STUDY_TYPE_PLOT_COLORS[stype]
        ax.boxplot(
            [vals],
            positions=[i],
            widths=0.5,
            patch_artist=True,
            boxprops=dict(facecolor=color, alpha=0.25, edgecolor=color, linewidth=0.8),
            medianprops=dict(color=color, linewidth=1.2),
            whiskerprops=dict(color=color, linewidth=0.8),
            capprops=dict(color=color, linewidth=0.8),
            flierprops=dict(marker="o", markerfacecolor=color, markersize=3, alpha=0.5),
            showfliers=False,
        )
        # Overlay individual points with jitter
        jitter = np.random.default_rng(42).uniform(-0.12, 0.12, len(vals))
        ax.scatter(
            np.full_like(vals, i) + jitter,
            vals,
            c=color,
            s=20,
            alpha=0.7,
            edgecolors="white",
            linewidths=0.3,
            marker=STUDY_TYPE_MARKERS[stype],
            zorder=3,
        )

    ax.set_xticks(range(len(type_order)))
    ax.set_xticklabels(type_labels_short)
    ax.set_ylabel("BO advantage (recall)")
    ax.set_xlabel("Study type")
    light_ygrid(ax)
    panel_label(ax, "d")

    # Add horizontal line at y=0 to all panels
    for ax in axes.flat:
        ax.axhline(0, color="#999999", linewidth=0.5, linestyle="--", zorder=1)

    # Single legend for panels a-c
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=3,
            bbox_to_anchor=(0.5, 1.02),
            frameon=False,
            fontsize=6.5,
        )

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIG_DIR / "fig_study_predictors.pdf"
    save_figure(fig, out_path)
    plt.close(fig)
    return out_path


def main():
    print("Loading within-study benchmark results...")
    results = load_all_results()
    study_info, result_map, pmids, strategies = build_tables(results)
    print(f"  Loaded {len(results)} results across {len(pmids)} studies, {len(strategies)} strategies")

    # Load richer study metadata from studies_with_ids.json
    study_meta = load_study_metadata()

    # Enrich study_info with metadata fields not in the result files
    for pmid in pmids:
        sid = str(pmid)
        if sid in study_meta:
            meta = study_meta[sid]
            for key in ["n_unique_hl", "n_unique_chl", "n_unique_peg", "il_ratio_std", "study_type"]:
                if key in meta and key not in study_info[pmid]:
                    study_info[pmid][key] = meta[key]

    # Compute BO advantage per study
    study_rows = compute_study_bo_advantage(study_info, result_map, pmids, strategies)
    print(f"  Computed BO advantage for {len(study_rows)} studies")

    # Compute Spearman correlations
    corr_results = compute_correlations(study_rows)

    # Print correlation table
    print_correlation_table(corr_results, study_rows)

    # Save JSON output
    output = {
        "description": "Spearman rank correlations between study properties and BO advantage",
        "n_studies": len(study_rows),
        "correlations": corr_results,
        "per_study": study_rows,
    }
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(output, indent=2))
    print(f"\n  -> {OUTPUT_JSON}")

    # Generate figure
    fig_path = make_figure(study_rows, corr_results)
    print(f"\nDone. Figure saved to {fig_path}")


if __name__ == "__main__":
    main()
