#!/usr/bin/env python3
"""Anchor IL rank consistency across studies."""


import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, spearmanr, ttest_1samp

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from diagnostics.utils import load_lnpdb_clean, summarize_study_assay_types


def main() -> int:
    df = load_lnpdb_clean(drop_duplicates=False)

    # Compute per-study IL percentile ranks
    study_ranks = {}
    study_types = {}

    for study_id, sdf in df.groupby("study_id"):
        assay_type, _ = summarize_study_assay_types(sdf)
        study_types[study_id] = assay_type

        il_vals = sdf.groupby("IL_SMILES")["Experiment_value"].mean()
        ranks = il_vals.rank(pct=True, method="average")
        study_ranks[study_id] = ranks

    # Anchor ILs
    il_study_counts = df.groupby("IL_SMILES")["study_id"].nunique()
    n_anchor = int((il_study_counts >= 2).sum())
    n_anchor_3 = int((il_study_counts >= 3).sum())

    pair_results = []
    study_ids = list(study_ranks.keys())
    for i in range(len(study_ids)):
        s1 = study_ids[i]
        r1 = study_ranks[s1]
        for j in range(i + 1, len(study_ids)):
            s2 = study_ids[j]
            r2 = study_ranks[s2]
            shared = r1.index.intersection(r2.index)
            if len(shared) < 3:
                continue
            v1 = r1.loc[shared].values
            v2 = r2.loc[shared].values
            rho, _ = spearmanr(v1, v2)
            tau, _ = kendalltau(v1, v2)
            if not np.isfinite(rho) or not np.isfinite(tau):
                continue
            t1 = study_types.get(s1, "unknown")
            t2 = study_types.get(s2, "unknown")
            pair_key = "__".join(sorted([t1, t2]))
            pair_results.append(
                {
                    "study_1": s1,
                    "study_2": s2,
                    "n_shared": len(shared),
                    "spearman_rho": float(rho),
                    "kendall_tau": float(tau),
                    "assay_pair": pair_key,
                }
            )

    rhos = np.array([p["spearman_rho"] for p in pair_results])
    if len(rhos) > 0:
        mean_rho = float(np.mean(rhos))
        med_rho = float(np.median(rhos))
        _t_stat, p_mean = ttest_1samp(rhos, 0.0)
    else:
        mean_rho = med_rho = p_mean = float("nan")

    by_pair = {}
    for p in pair_results:
        by_pair.setdefault(p["assay_pair"], []).append(p["spearman_rho"])

    pair_summary = {
        k: {
            "mean_rho": float(np.mean(v)),
            "median_rho": float(np.median(v)),
            "n_pairs": len(v),
        }
        for k, v in by_pair.items()
    }

    report = {
        "n_anchor_ILs": n_anchor,
        "n_anchor_ILs_ge_3": n_anchor_3,
        "n_study_pairs": len(pair_results),
        "mean_spearman_rho": mean_rho,
        "median_spearman_rho": med_rho,
        "mean_rho_p_value": float(p_mean),
        "pairwise_summary": pair_summary,
    }

    out_path = Path("diagnostics") / "anchor_analysis.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"Saved {out_path}")

    # Save distribution arrays for plotting
    dist_path = Path("diagnostics") / "anchor_pairwise_metrics.json"
    dist_path.write_text(json.dumps(pair_results, indent=2))
    print(f"Saved {dist_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
