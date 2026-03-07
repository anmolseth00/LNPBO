#!/usr/bin/env python3
"""SHAP and permutation importance analysis for LANTERN benchmark.

Uses SHAP TreeExplainer (exact for tree models) and sklearn permutation
importance to avoid XGBoost gain-based bias toward correlated features.
"""

import numpy as np
from xgboost import XGBRegressor

from LNPBO.benchmarks.runner import prepare_benchmark_data


def main():
    import shap
    from sklearn.inspection import permutation_importance

    seeds = [42, 123, 456, 789, 2024]
    all_shap = {}
    all_perm = {}

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        _, encoded_df, feature_cols, seed_idx, oracle_idx, _ = \
            prepare_benchmark_data(n_seed=500, random_seed=seed, reduction="pca", feature_type="lantern")

        X = encoded_df[feature_cols].values
        y = encoded_df["Experiment_value"].values

        train_X, train_y = X[seed_idx], y[seed_idx]
        test_X, test_y = X[oracle_idx[:500]], y[oracle_idx[:500]]

        model = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=seed)
        model.fit(train_X, train_y)

        # SHAP TreeExplainer (exact for XGBoost)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(train_X)
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        total_shap = mean_abs_shap.sum()
        for feat, val in zip(feature_cols, mean_abs_shap):
            all_shap.setdefault(feat, []).append(val / total_shap)

        # Permutation importance on held-out data
        perm = permutation_importance(model, test_X, test_y, n_repeats=10, random_state=seed)
        total_perm = max(perm.importances_mean.sum(), 1e-10)
        for feat, val in zip(feature_cols, perm.importances_mean):
            all_perm.setdefault(feat, []).append(max(val, 0) / total_perm)

    # Report
    for method_name, all_imp in [("SHAP (TreeExplainer)", all_shap), ("Permutation Importance", all_perm)]:
        print(f"\n\n{'='*70}")
        print(f"{method_name} — 5-seed mean")
        print(f"{'='*70}")
        ranked = sorted(all_imp.items(), key=lambda x: -np.mean(x[1]))
        for feat, vals in ranked:
            m = np.mean(vals) * 100
            s = np.std(vals) * 100
            if m >= 0.5:
                print(f"  {feat:<35} {m:>5.1f} ±{s:>4.1f}%")

        print("\n  By role:")
        for role in ["IL", "HL", "CHL", "PEG"]:
            role_imp = sum(np.mean(v) for f, v in all_imp.items() if f.startswith(role + "_"))
            print(f"    {role:<5} {role_imp*100:>5.1f}%")

        print("\n  By feature type:")
        for ftype in ["count_mfp", "rdkit", "molratio", "massratio"]:
            type_imp = sum(np.mean(v) for f, v in all_imp.items() if ftype in f)
            print(f"    {ftype:<15} {type_imp*100:>5.1f}%")


if __name__ == "__main__":
    main()
