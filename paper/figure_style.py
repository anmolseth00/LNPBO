"""Unified publication figure style for LNPBO paper.

Provides consistent colors, typography, and layout for all paper figures.
Targets ACS JCIM: single-column 3.25in, double-column 7in.
"""

import matplotlib.pyplot as plt

from LNPBO.benchmarks.stats import bootstrap_ci  # noqa: F401 -- re-exported

# ── Column widths (inches) ────────────────────────────────────────────────
SINGLE_COL = 3.25
DOUBLE_COL = 7.0
FULL_PAGE_H = 9.0

# ── Color palette ─────────────────────────────────────────────────────────
# Tol bright (colorblind-safe) for strategy families
FAMILY_COLORS = {
    "NGBoost": "#EE6677",  # red
    "RF": "#228833",  # green
    "CASMOPolitan": "#4477AA",  # blue
    "XGBoost": "#CCBB44",  # yellow
    "Deep Ensemble": "#66CCEE",  # cyan
    "GP (sklearn)": "#AA3377",  # purple
    "GP (BoTorch)": "#BBBBBB",  # grey
    "Random": "#000000",  # black
}

FAMILY_ORDER = [
    "NGBoost",
    "RF",
    "CASMOPolitan",
    "XGBoost",
    "Deep Ensemble",
    "GP (sklearn)",
    "GP (BoTorch)",
    "Random",
]

# Encoding palette (sequential from muted blue to warm)
ENCODING_COLORS = {
    "lantern": "#4477AA",
    "mordred": "#66CCEE",
    "chemeleon": "#228833",
    "lion": "#CCBB44",
    "mfp": "#EE6677",
    "count_mfp": "#AA3377",
    "unimol": "#BBBBBB",
    "agile": "#FF8C00",
}

ENCODING_DISPLAY = {
    "lantern": "LANTERN",
    "mfp": "MFP",
    "count_mfp": "Count MFP",
    "mordred": "Mordred",
    "unimol": "Uni-Mol",
    "lion": "LiON",
    "chemeleon": "CheMeleon",
    "agile": "AGILE",
}

# Study type palette
STUDY_TYPE_COLORS = {
    "IL-diverse": "#4477AA",
    "ratio-only": "#EE6677",
    "sub-study": "#228833",
}

# Cross-study transfer
TRANSFER_COLORS = {
    "random": "#000000",
    "cross_study": "#4477AA",
    "within_study": "#EE6677",
}

# ── Typography ────────────────────────────────────────────────────────────


def setup_style():
    """Apply clean, minimal publication style globally."""
    plt.rcParams.update(
        {
            # Font
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 7,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
            "legend.title_fontsize": 7,
            # Axes
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.5,
            "axes.labelpad": 3,
            "axes.titlepad": 6,
            "axes.grid": False,
            "axes.axisbelow": True,
            # Ticks
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "xtick.minor.size": 1.5,
            "ytick.minor.size": 1.5,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.pad": 2,
            "ytick.major.pad": 2,
            # Lines
            "lines.linewidth": 1.0,
            "lines.markersize": 4,
            # Figure
            "figure.dpi": 150,
            "figure.constrained_layout.use": False,
            # Export
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
            "pdf.fonttype": 42,  # TrueType
            "ps.fonttype": 42,
            # Grid (off by default, enable per-axis)
            "grid.linewidth": 0.3,
            "grid.alpha": 0.3,
            "grid.color": "#cccccc",
        }
    )


def panel_label(ax, label, x=-0.12, y=1.08):
    """Add bold panel label (a, b, c, ...) to axis."""
    ax.text(x, y, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top", ha="left")


def light_ygrid(ax):
    """Add subtle horizontal gridlines."""
    ax.yaxis.grid(True, linewidth=0.3, color="#dddddd", zorder=0)
    ax.set_axisbelow(True)


def despine(ax):
    """Remove top and right spines (already default, but explicit)."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_figure(fig, path, **kwargs):
    """Save figure as PDF (and optionally PNG)."""
    fig.savefig(path, **kwargs)
    print(f"  -> {path}")
