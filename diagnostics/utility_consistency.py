#!/usr/bin/env python3
"""Cross-study IL rank consistency using observed Experiment_values.

For each pair of studies sharing >= 3 ionizable lipids, compute Spearman
correlation on the mean observed Experiment_value per IL. This is a direct
empirical test of whether ILs that perform well in one study also perform
well in another — the real scientific question.
"""


import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, ttest_1samp

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from diagnostics.utils import load_lnpdb_clean


def main() -> int:
    df = load_lnpdb_clean(drop_duplicates=False)
    df = df.dropna(subset=["IL_SMILES", "Experiment_value"]).reset_index(drop=True)

    study_sizes = df.groupby("study_id").size()
    keep_ids = study_sizes[study_sizes >= 5].index
    df = df[df["study_id"].isin(keep_ids)].reset_index(drop=True)

    study_ids = df["study_id"].astype(str).values

    # Compute mean observed Experiment_value per IL per study
    study_il_means = {}
    for sid in np.unique(study_ids):
        sdf = df[study_ids == sid]
        il_mean = sdf.groupby("IL_SMILES")["Experiment_value"].mean()
        study_il_means[sid] = il_mean

    # For each pair of studies sharing ILs, compute rank correlation
    study_list = sorted(study_il_means.keys())
    pair_results = []

    for i in range(len(study_list)):
        s1 = study_list[i]
        il_mean1 = study_il_means[s1]

        for j in range(i + 1, len(study_list)):
            s2 = study_list[j]
            il_mean2 = study_il_means[s2]

            shared = sorted(set(il_mean1.index) & set(il_mean2.index))
            if len(shared) < 3:
                continue

            v1 = np.array([il_mean1[s] for s in shared])
            v2 = np.array([il_mean2[s] for s in shared])
            rho, p = spearmanr(v1, v2)
            if np.isfinite(rho):
                pair_results.append({
                    "study_1": s1,
                    "study_2": s2,
                    "n_shared": len(shared),
                    "spearman_rho": float(rho),
                    "p_value": float(p),
                })

    rhos = np.array([p["spearman_rho"] for p in pair_results])
    if len(rhos) > 0:
        mean_rho = float(np.mean(rhos))
        _t_stat, p_mean = ttest_1samp(rhos, 0.0)
    else:
        mean_rho = float("nan")
        p_mean = float("nan")

    report = {
        "n_study_pairs": len(pair_results),
        "mean_spearman_rho": mean_rho,
        "median_spearman_rho": float(np.median(rhos)) if len(rhos) > 0 else float("nan"),
        "p_value_mean_ne_zero": float(p_mean),
        "pair_details": pair_results,
    }

    out_path = Path("diagnostics") / "utility_consistency.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "pair_details"}, indent=2))
    print(f"Saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
