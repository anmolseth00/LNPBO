#!/usr/bin/env python3
"""Invariant Causal Prediction (ICP) feature selection.

Identifies which LANTERN IL-only PCs have a causal (invariant) relationship
with efficacy across studies.

Reference: Peters et al. 2016, "Causal inference using invariant prediction",
           Journal of the Royal Statistical Society, Series B.

Since we have 10 PCs (5 count_mfp + 5 rdkit), exact ICP is feasible (2^10 = 1024).
For each subset S: fit regression Y ~ X_S on each study, test whether residuals
are distributionally equal across studies (Levene test for equal variance).
"""

from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import levene, f_oneway
from sklearn.linear_model import LinearRegression

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from diagnostics.utils import load_lnpdb_clean, encode_lantern_il, lantern_il_feature_cols


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
    except Exception:
        p_levene = 0.0

    # F-test for equal means across studies
    try:
        _, p_means = f_oneway(*study_residuals)
    except Exception:
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
    print(f"Features ({n_features}): {feat_cols}")
    print(f"Studies: {len(np.unique(study_ids))}")
    print(f"Rows: {len(X)}")
    print(f"Total subsets to test: {2**n_features}")

    # Test all subsets (2^10 = 1024)
    accepted_sets = []
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
            if invariant:
                accepted_sets.append(set(subset_list))

    # Intersection of all accepted sets = invariant causal features
    if accepted_sets:
        causal_set = accepted_sets[0]
        for s in accepted_sets[1:]:
            causal_set = causal_set & s
        causal_features = [feat_cols[i] for i in sorted(causal_set)]
    else:
        causal_features = []

    # Summary
    n_invariant = sum(1 for r in all_results if r["invariant"])
    invariant_by_size = {}
    for r in all_results:
        s = r["size"]
        invariant_by_size.setdefault(s, {"total": 0, "invariant": 0})
        invariant_by_size[s]["total"] += 1
        if r["invariant"]:
            invariant_by_size[s]["invariant"] += 1

    report = {
        "n_features": n_features,
        "n_subsets_tested": len(all_results),
        "n_invariant": n_invariant,
        "causal_features": causal_features,
        "invariant_by_size": invariant_by_size,
        "accepted_examples": [r for r in all_results if r["invariant"]][:20],
    }

    print(f"\nInvariant subsets: {n_invariant} / {len(all_results)}")
    print(f"Causal features (intersection): {causal_features}")
    print(f"\nInvariant by size:")
    for size in sorted(invariant_by_size):
        info = invariant_by_size[size]
        print(f"  size={size}: {info['invariant']} / {info['total']}")

    out_path = Path("diagnostics") / "icp_results.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"Saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
