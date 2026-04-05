#!/usr/bin/env python3
"""Phase 1: Data landscape and representation skew analysis.

Produces publication figures characterizing LNPDB:
- Cargo type distribution
- Model system distribution (in vitro vs in vivo targets)
- Component diversity (IL >> HL, CHL, PEG)
- IL diversity histogram (long tail of 12k+ unique ILs)
- Per-study context homogeneity summary
- Compositional skew: fixed-ratio dominance

Usage:
    python -m experiments.data_integrity.data_landscape
"""

import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

OUT_DIR = Path(__file__).resolve().parent
FIG_DIR = OUT_DIR / "figures"


def load_data():
    from LNPBO.data.lnpdb_bridge import load_lnpdb_full

    print("Loading LNPDB...")
    dataset = load_lnpdb_full()
    df = dataset.df
    print(f"  {len(df):,} formulations, {df['Publication_PMID'].nunique()} studies")
    return df


def cargo_distribution(df):
    """Bar chart of cargo types."""
    cargo = df["Cargo_type"].fillna("Unknown").value_counts()
    fig, ax = plt.subplots(figsize=(5, 3.5))
    colors = {"mRNA": "#2ca02c", "siRNA": "#1f77b4", "pDNA": "#ff7f0e"}
    bar_colors = [colors.get(c, "#999999") for c in cargo.index]
    ax.barh(cargo.index, cargo.values, color=bar_colors, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Number of formulations")
    ax.set_title("Cargo type distribution in LNPDB")
    for i, (_ct, n) in enumerate(cargo.items()):
        pct = n / len(df) * 100
        ax.text(n + 50, i, f"{n:,} ({pct:.0f}%)", va="center", fontsize=8)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "cargo_distribution.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "cargo_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Cargo distribution: {dict(cargo)}")
    return dict(cargo)


def model_system_distribution(df):
    """Bar chart of model systems (in vitro, liver, lung, etc.)."""
    targets = df["Model_target"].fillna("Unknown").value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(targets.index[:15], targets.values[:15], color="#1f77b4", edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Number of formulations")
    ax.set_title("Model target distribution in LNPDB")
    for i, (_tgt, n) in enumerate(targets.items()):
        if i >= 15:
            break
        pct = n / len(df) * 100
        ax.text(n + 30, i, f"{n:,} ({pct:.0f}%)", va="center", fontsize=7)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "model_target_distribution.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "model_target_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Top model targets: {dict(targets.head(5))}")
    return dict(targets)


def component_diversity(df):
    """Compare diversity across lipid component roles."""
    diversity = {
        "IL": df["IL_SMILES"].nunique(),
        "HL": df["HL_name"].nunique(),
        "CHL": df["CHL_name"].nunique(),
        "PEG": df["PEG_name"].nunique(),
    }

    fig, ax = plt.subplots(figsize=(5, 3))
    roles = list(diversity.keys())
    counts = list(diversity.values())
    bars = ax.bar(roles, counts, color=["#d62728", "#2ca02c", "#ff7f0e", "#1f77b4"], edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Number of unique structures")
    ax.set_title("Component diversity in LNPDB")
    ax.set_yscale("log")
    for bar, c in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.1, f"{c:,}", ha="center", va="bottom", fontsize=9
        )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "component_diversity.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "component_diversity.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Component diversity: {diversity}")
    return diversity


def il_usage_histogram(df):
    """Histogram of IL usage frequency (long-tail)."""
    il_counts = df["IL_SMILES"].value_counts()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))

    # Left: histogram of how many formulations each IL appears in
    ax1.hist(il_counts.values, bins=50, color="#d62728", edgecolor="white", linewidth=0.3)
    ax1.set_xlabel("Formulations per IL")
    ax1.set_ylabel("Number of ILs")
    ax1.set_title("IL usage frequency distribution")
    ax1.set_yscale("log")

    # Right: cumulative coverage
    sorted_counts = np.sort(il_counts.values)[::-1]
    cumulative = np.cumsum(sorted_counts) / sorted_counts.sum()
    ax2.plot(np.arange(1, len(cumulative) + 1), cumulative, color="#d62728", linewidth=1.5)
    ax2.set_xlabel("Number of unique ILs (ranked)")
    ax2.set_ylabel("Cumulative fraction of formulations")
    ax2.set_title("IL coverage curve")
    ax2.axhline(0.8, ls="--", color="#999999", linewidth=0.8)
    n_80 = np.searchsorted(cumulative, 0.8) + 1
    ax2.axvline(n_80, ls="--", color="#999999", linewidth=0.8)
    ax2.text(n_80 + 50, 0.5, f"{n_80} ILs cover 80%", fontsize=8)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "il_usage_histogram.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "il_usage_histogram.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(
        f"  IL usage: {len(il_counts)} unique, median count={il_counts.median():.0f}, "
        f"max={il_counts.max()}, {n_80} ILs cover 80%"
    )
    return {"n_unique": len(il_counts), "median_count": float(il_counts.median()), "n_80pct_coverage": int(n_80)}


def study_type_breakdown(df):
    """Classify and visualize study types."""
    study_counts = df.groupby("Publication_PMID").size()
    qualifying = study_counts[study_counts >= 200]

    types = Counter()
    for pmid in qualifying.index:
        sub = df[df["Publication_PMID"] == pmid]
        n_il = sub["IL_SMILES"].nunique()
        il_std = sub["IL_molratio"].std() if "IL_molratio" in sub.columns else 0.0
        if n_il <= 1:
            types["Ratio-only"] += 1
        elif il_std < 1.0:
            types["IL-diverse, fixed ratios"] += 1
        else:
            types["IL-diverse, variable ratios"] += 1

    fig, ax = plt.subplots(figsize=(5, 3))
    labels = list(types.keys())
    counts = list(types.values())
    ax.bar(labels, counts, color=["#ff7f0e", "#2ca02c", "#1f77b4"], edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Number of studies")
    ax.set_title(f"Study types (N >= 200, {sum(counts)} studies)")
    for i, c in enumerate(counts):
        ax.text(i, c + 0.3, str(c), ha="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "study_types.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "study_types.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Study types: {dict(types)}")
    return dict(types)


def per_study_context_table(df):
    """Create a table of context variable homogeneity per study."""
    study_counts = df.groupby("Publication_PMID").size()
    qualifying = study_counts[study_counts >= 200].index

    ctx_cols = ["Model_target", "Experiment_method", "Cargo_type", "Route_of_administration"]
    rows = []
    for pmid in qualifying:
        sub = df[df["Publication_PMID"] == pmid]
        row = {"pmid": str(int(float(pmid))), "n": len(sub)}
        for col in ctx_cols:
            if col in sub.columns:
                row[f"{col}_nunique"] = int(sub[col].nunique())
                row[f"{col}_values"] = sub[col].fillna("NA").unique().tolist()
            else:
                row[f"{col}_nunique"] = 0
        is_homogeneous = all(row.get(f"{c}_nunique", 0) <= 1 for c in ctx_cols)
        row["is_homogeneous"] = is_homogeneous
        rows.append(row)

    table_df = pd.DataFrame(rows).sort_values("n", ascending=False)
    n_homo = table_df["is_homogeneous"].sum()
    n_total = len(table_df)
    print(f"  Context homogeneity: {n_homo}/{n_total} studies are fully homogeneous")
    return rows


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()

    print("\nGenerating data landscape figures...\n")

    report = {}
    report["cargo"] = cargo_distribution(df)
    report["model_targets"] = model_system_distribution(df)
    report["component_diversity"] = component_diversity(df)
    report["il_usage"] = il_usage_histogram(df)
    report["study_types"] = study_type_breakdown(df)
    report["context_homogeneity"] = per_study_context_table(df)
    report["summary"] = {
        "n_formulations": len(df),
        "n_studies": int(df["Publication_PMID"].nunique()),
        "n_qualifying_studies": sum(1 for _, n in df.groupby("Publication_PMID").size().items() if n >= 200),
        "n_unique_il": int(df["IL_SMILES"].nunique()),
        "n_unique_hl": int(df["HL_name"].nunique()),
        "n_unique_chl": int(df["CHL_name"].nunique()),
        "n_unique_peg": int(df["PEG_name"].nunique()),
    }

    json_path = OUT_DIR / "data_landscape.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport saved to {json_path}")
    print(f"Figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
