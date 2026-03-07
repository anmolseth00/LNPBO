#!/usr/bin/env python3
"""XGBoost feature importance analysis for LANTERN benchmark."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from LNPBO.benchmarks.runner import prepare_benchmark_data
from xgboost import XGBRegressor
import numpy as np


def main():
    encoded, encoded_df, feature_cols, seed_idx, oracle_idx, top_k_values = \
        prepare_benchmark_data(n_seed=500, random_seed=42, reduction="pca", feature_type="lantern")

    X = encoded_df[feature_cols].values
    y = encoded_df["Experiment_value"].values

    model = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
    model.fit(X[seed_idx], y[seed_idx])

    importances = model.feature_importances_
    feat_imp = sorted(zip(feature_cols, importances), key=lambda x: -x[1])
    total = sum(importances)

    print("\nXGBoost Feature Importance (gain-based, LANTERN seed=42)")
    print("=" * 60)
    cumulative = 0
    for feat, imp in feat_imp:
        cumulative += imp
        pct = imp / total * 100
        cum_pct = cumulative / total * 100
        print(f"  {feat:<35} {pct:>5.1f}%  (cum: {cum_pct:>5.1f}%)")

    print("\nBy role:")
    for role in ["IL", "HL", "CHL", "PEG"]:
        role_imp = sum(imp for feat, imp in feat_imp if feat.startswith(role + "_"))
        print(f"  {role:<5} {role_imp / total * 100:>5.1f}%")

    print("\nBy feature type:")
    for ftype in ["count_mfp", "rdkit", "molratio", "massratio"]:
        type_imp = sum(imp for feat, imp in feat_imp if ftype in feat)
        print(f"  {ftype:<15} {type_imp / total * 100:>5.1f}%")

    # Multi-seed stability
    print("\n\nMulti-seed importance stability (5 seeds):")
    print("=" * 60)
    all_importances = {}
    for seed in [42, 123, 456, 789, 2024]:
        _, df_s, _, si, _, _ = prepare_benchmark_data(
            n_seed=500, random_seed=seed, reduction="pca", feature_type="lantern"
        )
        m = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=seed)
        m.fit(df_s[feature_cols].values[si], df_s["Experiment_value"].values[si])
        for feat, imp in zip(feature_cols, m.feature_importances_):
            all_importances.setdefault(feat, []).append(imp)

    print(f"  {'Feature':<35} {'Mean':>6} {'Std':>6}")
    for feat in [f for f, _ in feat_imp]:
        vals = np.array(all_importances[feat])
        t = sum(np.array(list(all_importances.values())).sum(axis=0)) / len(seeds := [42, 123, 456, 789, 2024])
        print(f"  {feat:<35} {vals.mean()/t*100:>5.1f}% {vals.std()/t*100:>5.1f}%")


if __name__ == "__main__":
    main()
