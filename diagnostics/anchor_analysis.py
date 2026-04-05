#!/usr/bin/env python3
"""Anchor IL rank consistency across studies.

Computes pairwise Spearman and Kendall-tau correlations of IL percentile
ranks between every study pair that shares at least 3 anchor ILs, then
summarizes the distribution of correlation coefficients overall and
stratified by assay-type pair.
"""

import json
import logging
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, spearmanr, ttest_1samp

from LNPBO.benchmarks.stats import benjamini_hochberg
from LNPBO.data.study_utils import load_lnpdb_clean, summarize_study_assay_types

logger = logging.getLogger("lnpbo")


def main() -> int:
    """Run the anchor-IL rank-consistency analysis and write results to disk.

    Outputs:
        ``diagnostics/anchor_analysis.json`` -- summary statistics.
        ``diagnostics/anchor_pairwise_metrics.json`` -- per-pair results.

    Returns:
        Exit code (0 on success).
    """
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
            rho, p_rho = spearmanr(v1, v2)
            tau, p_tau = kendalltau(v1, v2)
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
                    "spearman_p": float(p_rho),
                    "kendall_tau": float(tau),
                    "kendall_p": float(p_tau),
                    "assay_pair": pair_key,
                }
            )

    rhos = np.array([p["spearman_rho"] for p in pair_results])

    # BH-FDR correction across all pairwise Spearman p-values
    if pair_results:
        raw_ps = np.array([p["spearman_p"] for p in pair_results])
        adj_ps, _ = benjamini_hochberg(raw_ps)
        for i, pr in enumerate(pair_results):
            pr["p_adjusted"] = float(adj_ps[i])

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
    logger.info(json.dumps(report, indent=2))
    logger.info("Saved %s", out_path)

    # Save distribution arrays for plotting
    dist_path = Path("diagnostics") / "anchor_pairwise_metrics.json"
    dist_path.write_text(json.dumps(pair_results, indent=2))
    logger.info("Saved %s", dist_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
