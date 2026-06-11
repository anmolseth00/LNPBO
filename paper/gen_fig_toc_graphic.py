#!/usr/bin/env python3
"""Generate the Table of Contents (TOC) graphic for the LNPBO paper.

Produces a horizontal bar chart of surrogate-family Top-5% Recall,
sized to ACS JCIM TOC dimensions (3.25 x 1.75 in).

Data pipeline mirrors make_all_figures.py: load per-study per-strategy
per-seed results, aggregate to family-level study means, then grand mean.

Usage:
    python paper/gen_fig_toc_graphic.py
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

from paper.figure_style import (
    FAMILY_ORDER,
    save_figure,
    setup_style,
)

RESULTS_DIR = REPO / "benchmark_results" / "within_study"
FIG_DIR = HERE / "figures"

SEEDS = [42, 123, 456, 789, 2024]

# Strategy -> family mapping (same as make_all_figures.py)
STRATEGY_FAMILY = {
    "random": "Random",
    "lnpbo_ucb": "GP (BoTorch)",
    "lnpbo_ei": "GP (BoTorch)",
    "lnpbo_logei": "GP (BoTorch)",
    "lnpbo_lp_ei": "GP (BoTorch)",
    "lnpbo_lp_logei": "GP (BoTorch)",
    "lnpbo_pls_logei": "GP (BoTorch)",
    "lnpbo_pls_lp_logei": "GP (BoTorch)",
    "lnpbo_rkb_logei": "GP (BoTorch)",
    "lnpbo_ts_batch": "GP (BoTorch)",
    "lnpbo_gibbon": "GP (BoTorch)",
    "lnpbo_tanimoto_ts": "GP (BoTorch)",
    "lnpbo_tanimoto_logei": "GP (BoTorch)",
    "lnpbo_aitchison_ts": "GP (BoTorch)",
    "lnpbo_aitchison_logei": "GP (BoTorch)",
    "lnpbo_dkl_ts": "GP (BoTorch)",
    "lnpbo_dkl_logei": "GP (BoTorch)",
    "lnpbo_rf_kernel_ts": "GP (BoTorch)",
    "lnpbo_rf_kernel_logei": "GP (BoTorch)",
    "lnpbo_compositional_ts": "GP (BoTorch)",
    "lnpbo_compositional_logei": "GP (BoTorch)",
    "casmopolitan_ei": "CASMOPolitan",
    "casmopolitan_ucb": "CASMOPolitan",
    "discrete_rf_ucb": "RF",
    "discrete_rf_ts": "RF",
    "discrete_rf_ts_batch": "RF",
    "discrete_xgb_ucb": "XGBoost",
    "discrete_xgb_greedy": "XGBoost",
    "discrete_xgb_cqr": "XGBoost",
    "discrete_xgb_online_conformal": "XGBoost",
    "discrete_xgb_ucb_ts_batch": "XGBoost",
    "discrete_ngboost_ucb": "NGBoost",
    "discrete_deep_ensemble": "Deep Ensemble",
    "discrete_gp_ucb": "GP (sklearn)",
}

# Display names for TOC (shorter than full family names)
TOC_DISPLAY = {
    "Random": "Random",
    "GP (BoTorch)": "GP-BO",
    "GP (sklearn)": "GP (sklearn)",
    "XGBoost": "XGBoost",
    "CASMOPolitan": "CASMOPolitan",
    "RF": "RF",
    "NGBoost": "NGBoost",
    "Deep Ensemble": "Deep Ensemble",
}

# Blue palette: darker for higher-ranked families, lighter for lower
# Ordered from lightest (Random) to darkest (best)
BAR_COLORS = {
    "Random": "#b0bec5",       # light grey-blue
    "GP (BoTorch)": "#90a4ae",
    "GP (sklearn)": "#90a4ae",
    "Deep Ensemble": "#78909c",
    "XGBoost": "#546e7a",
    "CASMOPolitan": "#455a64",
    "RF": "#37474f",
    "NGBoost": "#263238",
}


# ── Data Loading ─────────────────────────────────────────────────────────


def load_all_results():
    """Load all within-study benchmark JSON results."""
    results = []
    for pmid_dir in sorted(RESULTS_DIR.iterdir()):
        if not pmid_dir.is_dir() or not pmid_dir.name[0].isdigit():
            continue
        for f in sorted(pmid_dir.glob("*.json")):
            try:
                results.append(json.loads(f.read_text()))
            except (json.JSONDecodeError, KeyError):
                continue
    return results


# PLS strategies are bit-identical duplicates of their non-PLS twins; exclude
# them from aggregates to avoid double-weighting (see make_all_figures.py).
EXCLUDED_STRATEGIES = frozenset({"lnpbo_pls_logei", "lnpbo_pls_lp_logei"})


def build_tables(results):
    """Build lookup tables from raw results."""
    study_info, result_map = {}, {}
    strategies = set()
    for r in results:
        if r["strategy"] in EXCLUDED_STRATEGIES:
            continue
        study_id = r.get("study_id", str(int(r["pmid"])))
        study_info[study_id] = r["study_info"]
        result_map[(study_id, r["strategy"], r["seed"])] = r
        strategies.add(r["strategy"])
    return study_info, result_map, sorted(study_info.keys()), sorted(strategies)


def get_top5(result_map, pmid, strategy, seed):
    """Extract top-5% recall from a result entry."""
    key = (pmid, strategy, seed)
    if key not in result_map:
        return None
    try:
        return result_map[key]["result"]["metrics"]["top_k_recall"]["5"]
    except (KeyError, TypeError):
        return None


def family_study_means(result_map, pmids, strategies):
    """Compute family-level per-study mean top-5% recall."""
    raw = defaultdict(lambda: defaultdict(list))
    for s in strategies:
        fam = STRATEGY_FAMILY.get(s, "Other")
        for pmid in pmids:
            for seed in SEEDS:
                v = get_top5(result_map, pmid, s, seed)
                if v is not None:
                    raw[fam][pmid].append(v)
    return {
        (fam, pmid): np.mean(vals)
        for fam in raw
        for pmid, vals in raw[fam].items()
    }


# ── Figure ───────────────────────────────────────────────────────────────


def main():
    setup_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading within-study benchmark results...")
    results = load_all_results()
    _study_info, result_map, pmids, strategies = build_tables(results)
    print(f"  {len(results)} results, {len(pmids)} studies, {len(strategies)} strategies")

    # Compute family-level means
    fam_means = family_study_means(result_map, pmids, strategies)

    data = {}
    for fam in FAMILY_ORDER:
        vals = [fam_means[(fam, p)] for p in pmids if (fam, p) in fam_means]
        if vals:
            data[fam] = np.mean(vals)

    # Sort ascending (worst at bottom, best at top for horizontal bar)
    ordered = sorted(data.keys(), key=lambda f: data[f])
    n_families = len(ordered)
    n_strategies = len([s for s in strategies if s != "random"])
    n_studies = len(pmids)

    random_recall = data.get("Random", 0.532)

    # ── Build figure ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(3.25, 1.75))

    y = np.arange(n_families)
    bar_height = 0.65

    for i, fam in enumerate(ordered):
        m = data[fam]
        color = BAR_COLORS.get(fam, "#546e7a")
        ax.barh(i, m, height=bar_height, color=color, edgecolor="none", zorder=3)

        # Value label inside the bar (white text, near right edge)
        ax.text(
            m - 0.012, i, f"{m:.3f}",
            va="center", ha="right",
            fontsize=7, fontweight="bold", color="white", zorder=4,
        )

    # Multiplier annotations for top-2 families (improvement over Random)
    top2 = ordered[-2:]  # last two are the best
    for fam in top2:
        i = ordered.index(fam)
        m = data[fam]
        mult = m / random_recall
        ax.text(
            m + 0.012, i, f"{mult:.2f}x",
            va="center", ha="left",
            fontsize=6.5, fontweight="bold", color="#333333", zorder=4,
        )

    # Random baseline dashed line
    ax.axvline(
        random_recall, color="#9e9e9e", ls="--", lw=0.8, alpha=0.7, zorder=2,
    )

    # Y-axis labels
    ax.set_yticks(y)
    ax.set_yticklabels(
        [TOC_DISPLAY.get(f, f) for f in ordered],
        fontsize=7, fontweight="bold",
    )

    # X-axis
    ax.set_xlabel("Top-5% Recall", fontsize=7)
    ax.set_xlim(0.42, 0.80)
    ax.tick_params(axis="x", labelsize=6)

    # Remove spines for clean look
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)  # no y-tick marks

    # Subtitle annotation: right-aligned below x-axis label
    ax.annotate(
        f"{n_strategies} strategies \u00b7 {n_studies} studies \u00b7 {len(SEEDS)} seeds",
        xy=(1.0, 0), xycoords="axes fraction",
        xytext=(0, -26), textcoords="offset points",
        ha="right", va="top",
        fontsize=5.5, fontstyle="italic", color="#757575",
    )

    fig.tight_layout(pad=0.3)
    fig.subplots_adjust(bottom=0.22)  # room for subtitle below x-label

    out_path = FIG_DIR / "fig_toc_graphic.pdf"
    save_figure(fig, out_path)
    plt.close(fig)

    # Print summary
    print(f"\nFamily-level Top-5% Recall (n={n_studies} studies):")
    for fam in reversed(ordered):
        m = data[fam]
        mult = m / random_recall
        label = TOC_DISPLAY.get(fam, fam)
        print(f"  {label:<16s}  {m:.3f}  ({mult:.2f}x Random)")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
