#!/usr/bin/env python3
"""Study-level metadata audit for LNPDB."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from diagnostics.utils import load_lnpdb_clean, summarize_study_assay_types


def main() -> int:
    df = load_lnpdb_clean(drop_duplicates=False)

    study_rows = []
    for study_id, sdf in df.groupby("study_id"):
        assay_type, n_assay = summarize_study_assay_types(sdf)
        row = {
            "study_id": study_id,
            "n_formulations": len(sdf),
            "Experiment_value_mean": sdf["Experiment_value"].mean(),
            "Experiment_value_std": sdf["Experiment_value"].std(),
            "Experiment_value_min": sdf["Experiment_value"].min(),
            "Experiment_value_max": sdf["Experiment_value"].max(),
            "assay_type": assay_type,
            "n_assay_types": n_assay,
            "n_unique_ILs": sdf["IL_name"].nunique(dropna=True),
            "n_unique_HLs": sdf["HL_name"].nunique(dropna=True),
        }
        study_rows.append(row)

    meta = pd.DataFrame(study_rows).sort_values("n_formulations", ascending=False)

    out_path = Path("data") / "study_metadata.csv"
    out_path.parent.mkdir(exist_ok=True)
    meta.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")

    print("\nTop-10 largest studies:")
    print(meta.head(10).to_string(index=False))

    # Plots
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(meta["n_formulations"], bins=30, color="#4C72B0", edgecolor="white")
        ax.set_xlabel("Study size (n formulations)")
        ax.set_ylabel("Count")
        ax.set_title("Study Size Distribution")
        fig.tight_layout()
        hist_path = Path("diagnostics") / "study_size_hist.png"
        fig.savefig(hist_path, dpi=200)
        print(f"Saved {hist_path}")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 4))
        counts = meta["assay_type"].value_counts()
        ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%", startangle=90)
        ax.set_title("Assay Type Composition")
        fig.tight_layout()
        pie_path = Path("diagnostics") / "assay_type_pie.png"
        fig.savefig(pie_path, dpi=200)
        print(f"Saved {pie_path}")
        plt.close(fig)
    except Exception as exc:
        print(f"Plotting skipped: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
