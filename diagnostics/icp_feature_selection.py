#!/usr/bin/env python3
"""Invariant Causal Prediction (ICP) feature selection.

Identifies which LANTERN IL-only PCs have a causal (invariant) relationship
with efficacy across studies.

Reference: Peters et al. 2016, "Causal inference using invariant prediction",
           Journal of the Royal Statistical Society, Series B.

Since we have 10 PCs (5 count_mfp + 5 rdkit), exact ICP is feasible (2^10 = 1024).
For each subset S: fit regression Y ~ X_S on each study, test whether residuals
are distributionally equal across studies (Levene test for equal variance).

Multiple-testing correction (2026-03-07):
  Testing 1024 subsets at alpha=0.05 without correction inflates false invariance
  claims. In ICP, a subset is "invariant" when we FAIL to reject (p >= alpha).
  Multiple testing doesn't inflate p-values upward, but with many tests some
  subsets will appear invariant by chance. We apply two corrections:

  1. Bonferroni: raise the rejection threshold to alpha/n_tests. A subset is
     invariant only if both p-values exceed alpha/n_tests. This is conservative.
  2. Benjamini-Hochberg (BH) FDR: adjust p-values upward; a subset is invariant
     only if both adjusted p-values still exceed alpha.

  Both corrections are reported alongside uncorrected results.

  Reference: Benjamini & Hochberg 1995, "Controlling the false discovery rate",
             Journal of the Royal Statistical Society, Series B.
"""

import json
import logging
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import f_oneway, levene
from sklearn.linear_model import LinearRegression

from LNPBO.benchmarks.stats import benjamini_hochberg as _bh_full
from LNPBO.data.study_utils import encode_lantern_il, lantern_il_feature_cols, load_lnpdb_clean

logger = logging.getLogger("lnpbo")


def benjamini_hochberg(pvals):
    """BH-adjusted p-values (wrapper returning only adjusted values)."""
    return _bh_full(pvals)[0]


def compute_invariant_sets(all_results, feat_cols, key_suffix, invariant_key):
    """Compute accepted sets and causal features for a given invariant key."""
    accepted = []
    for r in all_results:
        if r[invariant_key]:
            idx = [feat_cols.index(f) for f in r["features"]]
            accepted.append(set(idx))
    if accepted:
        causal = accepted[0]
        for s in accepted[1:]:
            causal = causal & s
        return [feat_cols[i] for i in sorted(causal)]
    return []


def test_invariance(X, y, study_ids, feature_subset, alpha=0.05):
    """Test whether feature_subset gives invariant residuals across studies."""
    if len(feature_subset) == 0:
        residuals = y.copy()
    else:
        lr = LinearRegression()
        lr.fit(X[:, feature_subset], y)
        residuals = y - lr.predict(X[:, feature_subset])

    unique_studies = np.unique(study_ids)
    study_residuals = [residuals[study_ids == s] for s in unique_studies if (study_ids == s).sum() >= 5]

    if len(study_residuals) < 3:
        return False, 1.0, 1.0

    # Levene test for equal variance across studies
    try:
        _, p_levene = levene(*study_residuals)
    except (ValueError, FloatingPointError):
        p_levene = 0.0

    # F-test for equal means across studies
    try:
        _, p_means = f_oneway(*study_residuals)
    except (ValueError, FloatingPointError):
        p_means = 0.0

    # Both must not be rejected for invariance
    invariant = (p_levene >= alpha) and (p_means >= alpha)
    return invariant, float(p_levene), float(p_means)


def main() -> int:
    df = load_lnpdb_clean(drop_duplicates=False)
    df = df.dropna(subset=["IL_SMILES", "Experiment_value"]).reset_index(drop=True)

    # Filter to studies with >= 10 formulations
    study_sizes = df.groupby("study_id").size()
    keep = study_sizes[study_sizes >= 10].index
    df = df[df["study_id"].isin(keep)].reset_index(drop=True)

    encoded, _ = encode_lantern_il(df, reduction="pca")
    feat_cols = lantern_il_feature_cols(encoded)

    X = encoded[feat_cols].values
    y = encoded["Experiment_value"].values
    study_ids = df["study_id"].astype(str).values

    n_features = len(feat_cols)
    logger.info("Features (%d): %s", n_features, feat_cols)
    logger.info("Studies: %d", len(np.unique(study_ids)))
    logger.info("Rows: %d", len(X))
    logger.info("Total subsets to test: %d", 2**n_features)

    # Test all subsets (2^10 = 1024)
    all_results = []

    for size in range(n_features + 1):
        for subset in combinations(range(n_features), size):
            subset_list = list(subset)
            invariant, p_levene, p_means = test_invariance(X, y, study_ids, subset_list, alpha=0.05)
            feat_names = [feat_cols[i] for i in subset_list]
            result = {
                "features": feat_names,
                "size": size,
                "invariant": invariant,
                "p_levene": p_levene,
                "p_means": p_means,
            }
            all_results.append(result)

    # --- Multiple-testing correction ---
    n_tests = len(all_results)
    alpha = 0.05
    bonferroni_alpha = alpha / n_tests

    all_p_levene = np.array([r["p_levene"] for r in all_results])
    all_p_means = np.array([r["p_means"] for r in all_results])

    p_levene_bh = benjamini_hochberg(all_p_levene)
    p_means_bh = benjamini_hochberg(all_p_means)

    for i, r in enumerate(all_results):
        r["p_levene_bh"] = float(p_levene_bh[i])
        r["p_means_bh"] = float(p_means_bh[i])
        r["invariant_bh"] = bool(p_levene_bh[i] >= alpha and p_means_bh[i] >= alpha)
        r["invariant_bonferroni"] = bool(r["p_levene"] >= bonferroni_alpha and r["p_means"] >= bonferroni_alpha)

    # Intersection of all accepted sets = invariant causal features
    causal_features = compute_invariant_sets(all_results, feat_cols, "", "invariant")
    causal_features_bh = compute_invariant_sets(all_results, feat_cols, "_bh", "invariant_bh")
    causal_features_bonf = compute_invariant_sets(all_results, feat_cols, "_bonferroni", "invariant_bonferroni")

    # Summary by correction method
    def summarize(key):
        n_inv = sum(1 for r in all_results if r[key])
        by_size = {}
        for r in all_results:
            s = r["size"]
            by_size.setdefault(s, {"total": 0, "invariant": 0})
            by_size[s]["total"] += 1
            if r[key]:
                by_size[s]["invariant"] += 1
        return n_inv, by_size

    n_invariant, invariant_by_size = summarize("invariant")
    n_invariant_bh, invariant_by_size_bh = summarize("invariant_bh")
    n_invariant_bonf, invariant_by_size_bonf = summarize("invariant_bonferroni")

    report = {
        "n_features": n_features,
        "n_subsets_tested": n_tests,
        "alpha": alpha,
        "bonferroni_alpha": bonferroni_alpha,
        "uncorrected": {
            "n_invariant": n_invariant,
            "causal_features": causal_features,
            "invariant_by_size": invariant_by_size,
        },
        "bh_corrected": {
            "n_invariant": n_invariant_bh,
            "causal_features": causal_features_bh,
            "invariant_by_size": invariant_by_size_bh,
        },
        "bonferroni_corrected": {
            "n_invariant": n_invariant_bonf,
            "causal_features": causal_features_bonf,
            "invariant_by_size": invariant_by_size_bonf,
        },
        "accepted_examples_uncorrected": [r for r in all_results if r["invariant"]][:20],
    }

    logger.info("=" * 60)
    logger.info("Multiple-testing correction (n_tests=%d, alpha=%s)", n_tests, alpha)
    logger.info("Bonferroni alpha: %.2e", bonferroni_alpha)
    logger.info("=" * 60)

    for label, n_inv, causal, by_size in [
        ("Uncorrected", n_invariant, causal_features, invariant_by_size),
        ("BH-corrected (FDR)", n_invariant_bh, causal_features_bh, invariant_by_size_bh),
        ("Bonferroni-corrected", n_invariant_bonf, causal_features_bonf, invariant_by_size_bonf),
    ]:
        logger.info("--- %s ---", label)
        logger.info("Invariant subsets: %d / %d", n_inv, n_tests)
        logger.info("Causal features (intersection): %s", causal)
        logger.info("Invariant by size:")
        for size in sorted(by_size):
            info = by_size[size]
            logger.info("  size=%d: %d / %d", size, info["invariant"], info["total"])

    out_path = Path("diagnostics") / "icp_results.json"
    out_path.write_text(json.dumps(report, indent=2))
    logger.info("Saved %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
