#!/usr/bin/env python3
"""Generate mixed discrete-continuous BO figure for the LNPBO paper SI.

Panel A: Paired bar chart comparing Mixed LogEI vs Compositional LogEI
         top-5% recall per study, colored by study type.
Panel B: Scatter of elapsed wall-clock time (log scale) vs number of unique
         ionizable lipids, showing the combinatorial scaling wall of the
         mixed-variable formulation.

Target: ACS JCIM double-column width (7 in).
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

from paper.figure_style import (
    DOUBLE_COL,
    light_ygrid,
    panel_label,
    save_figure,
    setup_style,
)

RESULTS_DIR = REPO / "benchmark_results" / "within_study"
FIG_DIR = HERE / "figures"

SEEDS = [42, 123, 456, 789, 2024]

# Studies that ran the mixed-variable optimizer (all 23 canonical studies
# were attempted; 37985700 was skipped because it has no mixed results).
CANONICAL_STUDIES = [
    18438401,
    24969323,
    26729861,
    31570898,
    33542471,
    35879315,
    36997680,
    37127709,
    37480759,
    37661193,
    37985700,
    37990414,
    38060560,
    38082180,
    38424061,
    38437545,
    38740955,
    39060305,
    39099464,
    39354147,
    39497197,
    39742515,
    40060499,
]

# Study type classification
VARIABLE_RATIO = {37985700, 36997680, 40060499, 37127709, 39099464}
RATIO_ONLY = {35879315, 38082180}
# Everything else is fixed-ratio

STUDY_TYPE_COLORS = {
    "fixed-ratio": "#BBBBBB",   # grey
    "variable-ratio": "#4477AA", # blue
    "ratio-only": "#228833",     # green
}

STUDY_TYPE_LABELS = {
    "fixed-ratio": "Fixed-ratio",
    "variable-ratio": "Variable-ratio",
    "ratio-only": "Ratio-only",
}

# Two studies flagged as infeasible for the mixed optimizer due to scaling
INFEASIBLE_STUDIES = {36997680, 40060499}


def _study_type(study_id):
    sid = int(study_id)
    if sid in RATIO_ONLY:
        return "ratio-only"
    if sid in VARIABLE_RATIO:
        return "variable-ratio"
    return "fixed-ratio"


def load_metric(study_id, strategy, seed):
    """Load top-5% recall and elapsed time for a single run."""
    fname = f"{strategy}_s{seed}.json"
    fpath = RESULTS_DIR / str(study_id) / fname
    if not fpath.exists():
        return None, None
    try:
        data = json.loads(fpath.read_text())
        recall = data["result"]["metrics"]["top_k_recall"]["5"]
        elapsed = data["result"]["elapsed"]
        n_unique_il = data["study_info"]["n_unique_il"]
        n_formulations = data["study_info"]["n_formulations"]
        return {
            "recall": recall,
            "elapsed": elapsed,
            "n_unique_il": n_unique_il,
            "n_formulations": n_formulations,
        }, None
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        return None, str(exc)


def collect_data():
    """Gather per-study mean recall and elapsed time for mixed and compositional."""
    rows = []
    for study_id in CANONICAL_STUDIES:
        stype = _study_type(study_id)

        # Mixed LogEI
        mixed_recalls, mixed_elapsed, mixed_n_il, mixed_n_form = [], [], None, None
        for seed in SEEDS:
            info, _ = load_metric(study_id, "lnpbo_mixed_logei", seed)
            if info is not None:
                mixed_recalls.append(info["recall"])
                mixed_elapsed.append(info["elapsed"])
                mixed_n_il = info["n_unique_il"]
                mixed_n_form = info["n_formulations"]

        # Compositional LogEI
        comp_recalls, comp_elapsed = [], []
        for seed in SEEDS:
            info, _ = load_metric(study_id, "lnpbo_compositional_logei", seed)
            if info is not None:
                comp_recalls.append(info["recall"])
                comp_elapsed.append(info["elapsed"])

        if not mixed_recalls and not comp_recalls:
            continue

        rows.append({
            "study_id": study_id,
            "study_type": stype,
            "n_unique_il": mixed_n_il or 0,
            "n_formulations": mixed_n_form or 0,
            "mixed_recall_mean": float(np.mean(mixed_recalls)) if mixed_recalls else None,
            "mixed_recall_seeds": mixed_recalls,
            "mixed_elapsed_mean": float(np.mean(mixed_elapsed)) if mixed_elapsed else None,
            "comp_recall_mean": float(np.mean(comp_recalls)) if comp_recalls else None,
            "comp_recall_seeds": comp_recalls,
            "comp_elapsed_mean": float(np.mean(comp_elapsed)) if comp_elapsed else None,
        })

    return rows


def main():
    setup_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    rows = collect_data()
    # Only keep studies that have BOTH mixed and compositional results
    paired_rows = [r for r in rows if r["mixed_recall_mean"] is not None and r["comp_recall_mean"] is not None]

    # Sort by compositional recall for consistent ordering
    paired_rows.sort(key=lambda r: r["comp_recall_mean"], reverse=True)

    # Print summary table
    print(f"{'Study':>10s}  {'Type':<16s}  {'#IL':>5s}  {'Mixed':>7s}  {'Comp':>7s}  {'Delta':>7s}  {'Mixed t(s)':>11s}  {'Comp t(s)':>11s}  {'Slowdown':>9s}")  # noqa: E501
    print("-" * 105)
    for r in paired_rows:
        delta = r["mixed_recall_mean"] - r["comp_recall_mean"]
        slowdown = r["mixed_elapsed_mean"] / r["comp_elapsed_mean"] if r["comp_elapsed_mean"] else float("nan")
        print(
            f"{r['study_id']:>10d}  {r['study_type']:<16s}  {r['n_unique_il']:>5d}  "
            f"{r['mixed_recall_mean']:.4f}  {r['comp_recall_mean']:.4f}  {delta:+.4f}  "
            f"{r['mixed_elapsed_mean']:>11.1f}  {r['comp_elapsed_mean']:>11.1f}  {slowdown:>8.1f}x"
        )

    # Aggregate stats
    mixed_vals = [r["mixed_recall_mean"] for r in paired_rows]
    comp_vals = [r["comp_recall_mean"] for r in paired_rows]
    deltas = [m - c for m, c in zip(mixed_vals, comp_vals)]
    print(f"\nMean Mixed recall:         {np.mean(mixed_vals):.4f}")
    print(f"Mean Compositional recall: {np.mean(comp_vals):.4f}")
    print(f"Mean Delta:                {np.mean(deltas):+.4f}")
    print(f"Median Delta:              {np.median(deltas):+.4f}")
    print(f"Paired studies:            {len(paired_rows)}")

    # ── Figure ────────────────────────────────────────────────────────────
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(DOUBLE_COL, 3.0),
                                      gridspec_kw={"width_ratios": [2.2, 1]})

    # ── Panel A: Paired bar chart ─────────────────────────────────────────
    n = len(paired_rows)
    x = np.arange(n)
    bar_w = 0.36

    for i, r in enumerate(paired_rows):
        color = STUDY_TYPE_COLORS[r["study_type"]]
        # Compositional bar (left)
        ax_a.bar(x[i] - bar_w / 2, r["comp_recall_mean"], bar_w,
                 color=color, edgecolor="white", linewidth=0.4, alpha=0.55, zorder=3)
        # Mixed bar (right)
        ax_a.bar(x[i] + bar_w / 2, r["mixed_recall_mean"], bar_w,
                 color=color, edgecolor="white", linewidth=0.4, zorder=3)

    # X labels: study IDs (abbreviated)
    xlabels = [str(r["study_id"]) for r in paired_rows]
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(xlabels, rotation=70, ha="right", fontsize=5)
    ax_a.set_ylabel("Top-5% Recall")
    ax_a.set_xlabel("Study (PMID)")

    light_ygrid(ax_a)

    # Legend: study types + bar meaning
    from matplotlib.patches import Patch
    legend_handles = []
    for stype in ["fixed-ratio", "variable-ratio", "ratio-only"]:
        legend_handles.append(
            Patch(facecolor=STUDY_TYPE_COLORS[stype], edgecolor="white",
                  label=STUDY_TYPE_LABELS[stype])
        )
    legend_handles.append(
        Patch(facecolor="#999999", edgecolor="white", alpha=0.55,
              label="Compositional (left)")
    )
    legend_handles.append(
        Patch(facecolor="#999999", edgecolor="white",
              label="Mixed (right)")
    )
    ax_a.legend(handles=legend_handles, loc="upper right", fontsize=5.5,
                frameon=True, framealpha=0.9, edgecolor="#cccccc",
                handlelength=1.2, handleheight=0.8)

    # Y limits
    all_vals = mixed_vals + comp_vals
    ax_a.set_ylim(0, min(1.05, max(all_vals) + 0.08))

    panel_label(ax_a, "a")

    # ── Panel B: Scaling scatter ──────────────────────────────────────────
    # Collect all studies with mixed results (including those without comp)
    mixed_rows = [r for r in rows if r["mixed_elapsed_mean"] is not None]

    for r in mixed_rows:
        stype = r["study_type"]
        color = STUDY_TYPE_COLORS[stype]
        n_il = r["n_unique_il"]
        t = r["mixed_elapsed_mean"]

        is_infeasible = r["study_id"] in INFEASIBLE_STUDIES
        marker = "X" if is_infeasible else "o"
        ms = 5 if is_infeasible else 4
        edge = "#333333" if is_infeasible else "white"
        ew = 0.6 if is_infeasible else 0.3

        ax_b.scatter(n_il, t, c=color, marker=marker, s=ms**2,
                     edgecolors=edge, linewidths=ew, zorder=3)

    # Also plot compositional elapsed times for contrast
    comp_rows = [r for r in rows if r["comp_elapsed_mean"] is not None]
    for r in comp_rows:
        color = STUDY_TYPE_COLORS[r["study_type"]]
        n_il = r["n_unique_il"]
        t = r["comp_elapsed_mean"]
        ax_b.scatter(n_il, t, c=color, marker="^", s=12,
                     edgecolors="white", linewidths=0.3, alpha=0.5, zorder=2)

    ax_b.set_yscale("log")
    ax_b.set_xlabel("Unique ILs")
    ax_b.set_ylabel("Elapsed Time (s)")
    light_ygrid(ax_b)

    # Legend for panel B
    from matplotlib.lines import Line2D
    b_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#888888",
               markeredgecolor="white", markersize=4.5, label="Mixed"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#888888",
               markeredgecolor="white", markeredgewidth=0.3, markersize=4.5,
               alpha=0.5, label="Compositional"),
        Line2D([0], [0], marker="X", color="w", markerfacecolor="#888888",
               markeredgecolor="#333333", markeredgewidth=0.6, markersize=5,
               label="Infeasible study"),
    ]
    ax_b.legend(handles=b_handles, loc="upper left", fontsize=5.5,
                frameon=True, framealpha=0.9, edgecolor="#cccccc",
                handlelength=1.2, handleheight=0.8)

    panel_label(ax_b, "b")

    fig.tight_layout()
    out_pdf = FIG_DIR / "fig_mixed_bo.pdf"
    out_png = FIG_DIR / "fig_mixed_bo.png"
    save_figure(fig, out_pdf)
    save_figure(fig, out_png)
    plt.close(fig)
    print(f"\nSaved to {out_pdf}")


if __name__ == "__main__":
    main()
