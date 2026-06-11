#!/usr/bin/env python3
"""Generate kernel comparison figure (grouped bar chart) for the LNPBO paper.

Reads within-study benchmark results for all 6 GP kernels x 2 acquisition
functions (TS, LogEI) across 27 studies and 5 seeds. Shows top-5% recall
with bootstrap 95% CIs.
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
    DOUBLE_COL,
    bootstrap_ci,
    light_ygrid,
    save_figure,
    setup_style,
)

RESULTS_DIR = REPO / "benchmark_results" / "within_study"
FIG_DIR = HERE / "figures"

SEEDS = [42, 123, 456, 789, 2024]

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

# Strategy -> (kernel display name, acquisition function)
STRATEGY_KERNEL_ACQ = {
    "lnpbo_ts_batch": ("Matern-5/2", "TS"),
    "lnpbo_logei": ("Matern-5/2", "LogEI"),
    "lnpbo_tanimoto_ts": ("Tanimoto", "TS"),
    "lnpbo_tanimoto_logei": ("Tanimoto", "LogEI"),
    "lnpbo_aitchison_ts": ("Aitchison", "TS"),
    "lnpbo_aitchison_logei": ("Aitchison", "LogEI"),
    "lnpbo_compositional_ts": ("Compositional\nProduct", "TS"),
    "lnpbo_compositional_logei": ("Compositional\nProduct", "LogEI"),
    "lnpbo_dkl_ts": ("DKL", "TS"),
    "lnpbo_dkl_logei": ("DKL", "LogEI"),
    "lnpbo_rf_kernel_ts": ("RF Proximity", "TS"),
    "lnpbo_rf_kernel_logei": ("RF Proximity", "LogEI"),
}

KERNEL_COLORS = {
    "Matern-5/2": "#4477AA",
    "Tanimoto": "#EE6677",
    "Aitchison": "#228833",
    "Compositional\nProduct": "#CCBB44",
    "DKL": "#66CCEE",
    "RF Proximity": "#AA3377",
}


def load_top5(study_id, strategy, seed):
    fname = f"{strategy}_s{seed}.json"
    fpath = RESULTS_DIR / str(study_id) / fname
    if not fpath.exists():
        return None
    try:
        data = json.loads(fpath.read_text())
        return data["result"]["metrics"]["top_k_recall"]["5"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def main():
    setup_style()

    # Collect per-study mean recall for each (kernel, acq) pair.
    # For each study: average across seeds first, then treat study means
    # as independent observations for bootstrap CI.
    #
    # Also collect random baseline.
    kernel_acq_study_means = defaultdict(dict)  # (kernel, acq) -> {study: mean}
    random_study_means = {}

    for study_id in CANONICAL_STUDIES:
        # Random baseline
        rvals = []
        for seed in SEEDS:
            v = load_top5(study_id, "random", seed)
            if v is not None:
                rvals.append(v)
        if rvals:
            random_study_means[study_id] = np.mean(rvals)

        # Kernel strategies
        for strategy, (kernel, acq) in STRATEGY_KERNEL_ACQ.items():
            vals = []
            for seed in SEEDS:
                v = load_top5(study_id, strategy, seed)
                if v is not None:
                    vals.append(v)
            if vals:
                kernel_acq_study_means[(kernel, acq)][study_id] = np.mean(vals)

    # Compute overall mean and bootstrap CI for each (kernel, acq)
    kernel_acq_stats = {}  # (kernel, acq) -> (mean, ci_lo, ci_hi, n_studies)
    for (kernel, acq), study_dict in kernel_acq_study_means.items():
        study_vals = list(study_dict.values())
        n = len(study_vals)
        if n == 0:
            continue
        m = float(np.mean(study_vals))
        if n >= 3:
            ci_lo, ci_hi = bootstrap_ci(study_vals, n_boot=10000)
        else:
            ci_lo, ci_hi = m, m
        kernel_acq_stats[(kernel, acq)] = (m, ci_lo, ci_hi, n)

    # Random baseline
    random_vals = list(random_study_means.values())
    random_mean = float(np.mean(random_vals)) if random_vals else None

    # Get unique kernels and sort by best TS recall descending
    kernels = sorted(
        set(k for k, _ in kernel_acq_stats),
        key=lambda k: kernel_acq_stats.get((k, "TS"), (0,))[0],
        reverse=True,
    )

    # Print summary
    print(f"{'Kernel':<25s} {'Acq':>5s}  {'Mean':>6s}  {'95% CI':>15s}  {'N':>3s}")
    print("-" * 62)
    for kernel in kernels:
        for acq in ["TS", "LogEI"]:
            key = (kernel, acq)
            if key in kernel_acq_stats:
                m, lo, hi, n = kernel_acq_stats[key]
                print(f"{kernel:<25s} {acq:>5s}  {m:.3f}  [{lo:.3f}, {hi:.3f}]  {n:>3d}")
    if random_mean is not None:
        print(f"{'Random':<25s} {'':>5s}  {random_mean:.3f}")

    # Build the grouped bar chart
    n_kernels = len(kernels)
    bar_width = 0.32
    gap = 0.06  # gap between TS and LogEI bars within a group
    x = np.arange(n_kernels)

    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 3.2))

    from matplotlib.colors import to_rgba

    def _lighten(hex_color, factor=0.4):
        rgba = to_rgba(hex_color)
        return (*tuple(min(1.0, c + factor * (1.0 - c)) for c in rgba[:3]), rgba[3])

    for i, kernel in enumerate(kernels):
        color = KERNEL_COLORS.get(kernel, "#888888")
        ts_color = color
        logei_color = _lighten(color, 0.4)

        for _j, (acq, acq_color, offset) in enumerate(
            [
                ("TS", ts_color, -(bar_width + gap) / 2),
                ("LogEI", logei_color, (bar_width + gap) / 2),
            ]
        ):
            key = (kernel, acq)
            if key not in kernel_acq_stats:
                continue
            m, ci_lo, ci_hi, n_studies = kernel_acq_stats[key]
            err_lo = m - ci_lo
            err_hi = ci_hi - m

            ax.bar(
                x[i] + offset,
                m,
                bar_width,
                color=acq_color,
                edgecolor="white",
                linewidth=0.5,
                zorder=3,
                yerr=[[err_lo], [err_hi]],
                capsize=2.5,
                error_kw=dict(lw=0.7, capthick=0.7, color="#333333", zorder=4),
            )

            # Annotate with mean value above error bar
            ax.text(
                x[i] + offset,
                ci_hi + 0.008,
                f"{m:.3f}",
                ha="center",
                va="bottom",
                fontsize=5.5,
                color="#333333",
                zorder=5,
            )

    # Add "preliminary (N studies)" below x-tick for incomplete kernels
    n_full = len(CANONICAL_STUDIES)
    for i, kernel in enumerate(kernels):
        for acq in ["TS", "LogEI"]:
            key = (kernel, acq)
            if key in kernel_acq_stats:
                _, _, _, n_studies = kernel_acq_stats[key]
                if n_studies < n_full:
                    ax.annotate(
                        f"preliminary ({n_studies} studies)",
                        xy=(x[i], 0),
                        xycoords=("data", "axes fraction"),
                        xytext=(0, -28),
                        textcoords="offset points",
                        ha="center",
                        va="top",
                        fontsize=5,
                        fontstyle="italic",
                        color="#888888",
                    )
                    break

    # Random baseline
    if random_mean is not None:
        ax.axhline(
            random_mean,
            color="#000000",
            ls="--",
            lw=0.8,
            alpha=0.6,
            zorder=2,
        )

    # Legend: use a representative kernel pair (Matern blue) for dark/light swatches
    from matplotlib.patches import Patch

    ref_color = KERNEL_COLORS["Matern-5/2"]
    ref_light = _lighten(ref_color, 0.4)
    legend_elements = [
        Patch(facecolor=ref_color, edgecolor="white", label="TS (darker)"),
        Patch(facecolor=ref_light, edgecolor="white", label="LogEI (lighter)"),
        plt.Line2D(
            [0],
            [0],
            color="#000000",
            ls="--",
            lw=0.8,
            alpha=0.6,
            label=f"Random ({random_mean:.3f})" if random_mean else "Random",
        ),
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper right",
        frameon=True,
        framealpha=0.9,
        edgecolor="#cccccc",
        fontsize=6.5,
    )

    # Labels and formatting
    ax.set_xticks(x)
    ax.set_xticklabels(kernels, fontsize=7)
    ax.set_ylabel("Mean Top-5% Recall")
    ax.set_xlabel("GP Kernel")
    light_ygrid(ax)

    # Y-axis limits: leave room for annotations above bars
    all_hi = max(s[2] for s in kernel_acq_stats.values())
    has_prelim = any(s[3] < len(CANONICAL_STUDIES) for s in kernel_acq_stats.values())
    y_lo = min(random_mean - 0.04, 0.48) if random_mean else 0.45
    y_hi = all_hi + (0.10 if has_prelim else 0.06)
    ax.set_ylim(y_lo, y_hi)

    fig.tight_layout()
    # Extra bottom margin for preliminary annotation below x-labels
    if has_prelim:
        fig.subplots_adjust(bottom=0.20)
    save_figure(fig, FIG_DIR / "fig_kernel_comparison.pdf")
    plt.close(fig)
    print(f"\nSaved to {FIG_DIR / 'fig_kernel_comparison.pdf'}")


if __name__ == "__main__":
    main()
