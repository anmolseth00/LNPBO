#!/usr/bin/env python3
"""Context-conditioned XGB surrogate: does experimental context improve cross-study R²?

Compares two feature sets:
  1. Molecular only: LANTERN IL-only PCs (10 features)
  2. Molecular + context: LANTERN IL-only PCs + one-hot experimental context

Evaluation:
  - Cross-study R² (80/20 study-level split, stratified by assay type)
  - Within-study R² (per-study 80/20 formulation split for studies with >=20 rows)
  - Feature importance (XGBoost gain-based) on the context-conditioned model
"""


import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.context import encode_context
from diagnostics.utils import (
    encode_lantern_il,
    lantern_il_feature_cols,
    load_lnpdb_clean,
    summarize_study_assay_types,
)


def _study_split(df, seed=42):
    """80/20 study-level split, stratified by assay type."""
    rng = np.random.RandomState(seed)
    study_to_type = {}
    for sid, sdf in df.groupby("study_id"):
        assay_type, _ = summarize_study_assay_types(sdf)
        study_to_type[sid] = assay_type

    train_ids = set()
    test_ids = set()
    for assay_type in sorted(set(study_to_type.values())):
        ids = [sid for sid, at in study_to_type.items() if at == assay_type]
        rng.shuffle(ids)
        cut = max(1, int(0.8 * len(ids))) if len(ids) > 1 else len(ids)
        train_ids.update(ids[:cut])
        test_ids.update(ids[cut:])

    return train_ids, test_ids


def _train_xgb(X_train, y_train, seed=42):
    model = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def _within_study_r2(df, feat_cols, min_n=20, seed=42):
    """Per-study 80/20 formulation split, return mean within-study R²."""
    rng = np.random.RandomState(seed)
    r2s = []
    for _sid, sdf in df.groupby("study_id"):
        if len(sdf) < min_n:
            continue
        idx = sdf.index.tolist()
        rng.shuffle(idx)
        cut = int(0.8 * len(idx))
        train_idx = idx[:cut]
        test_idx = idx[cut:]
        if len(test_idx) < 4:
            continue

        X_tr = df.loc[train_idx, feat_cols].values
        y_tr = df.loc[train_idx, "Experiment_value"].values
        X_te = df.loc[test_idx, feat_cols].values
        y_te = df.loc[test_idx, "Experiment_value"].values

        model = _train_xgb(X_tr, y_tr, seed=seed)
        pred = model.predict(X_te)
        r2 = r2_score(y_te, pred)
        r2s.append(r2)

    return float(np.mean(r2s)) if r2s else float("nan"), r2s


def main() -> int:
    print("Loading data...")
    df = load_lnpdb_clean(drop_duplicates=False)
    df = df.dropna(subset=["IL_SMILES", "Experiment_value"]).reset_index(drop=True)

    # Encode LANTERN IL-only PCs
    print("Encoding LANTERN IL-only features...")
    encoded, _ = encode_lantern_il(df, reduction="pca")
    mol_cols = lantern_il_feature_cols(encoded)
    print(f"  Molecular features: {len(mol_cols)} columns")

    # Encode context features
    print("Encoding context features...")
    encoded, ctx_cols, ctx_levels = encode_context(encoded)
    print(f"  Context features: {len(ctx_cols)} columns")
    for col, lvs in sorted(ctx_levels.items()):
        print(f"    {col}: {len(lvs)} levels")

    # Drop rows with NaN in any feature column
    all_feat_cols = mol_cols + ctx_cols
    before = len(encoded)
    encoded = encoded.dropna(subset=[*all_feat_cols, "Experiment_value"])
    after = len(encoded)
    if before != after:
        print(f"  Dropped {before - after} rows with NaN features")

    # Ensure study_id is carried through
    if "study_id" not in encoded.columns:
        encoded["study_id"] = df.loc[encoded.index, "study_id"].values

    seeds = [42, 123, 456, 789, 2024]
    feature_sets = {
        "molecular_only": mol_cols,
        "molecular_plus_context": all_feat_cols,
    }

    results = {}

    # --- Cross-study evaluation ---
    print("\n" + "=" * 70)
    print("CROSS-STUDY R² (study-level 80/20 split)")
    print("=" * 70)

    for fs_name, feat_cols in feature_sets.items():
        print(f"\n--- {fs_name} ({len(feat_cols)} features) ---")
        cross_r2s = []
        within_r2s_all = []

        for seed in seeds:
            train_ids, test_ids = _study_split(encoded, seed=seed)
            train_mask = encoded["study_id"].isin(train_ids)
            test_mask = encoded["study_id"].isin(test_ids)

            X_train = encoded.loc[train_mask, feat_cols].values
            y_train = encoded.loc[train_mask, "Experiment_value"].values
            X_test = encoded.loc[test_mask, feat_cols].values
            y_test = encoded.loc[test_mask, "Experiment_value"].values

            if len(X_test) < 50:
                print(f"  seed={seed}: too few test rows ({len(X_test)}), skipping")
                continue

            model = _train_xgb(X_train, y_train, seed=seed)
            pred = model.predict(X_test)
            r2 = float(r2_score(y_test, pred))
            cross_r2s.append(r2)

            # Within-study R² on training studies
            train_df = encoded.loc[train_mask].copy()
            ws_mean, ws_list = _within_study_r2(train_df, feat_cols, min_n=20, seed=seed)
            within_r2s_all.append(ws_mean)

            print(
                f"  seed={seed}: cross_R²={r2:.4f}  "
                f"within_R²={ws_mean:.4f} ({len(ws_list)} studies)"
            )

        cross_mean = float(np.mean(cross_r2s))
        cross_std = float(np.std(cross_r2s))
        within_mean = float(np.mean(within_r2s_all))
        within_std = float(np.std(within_r2s_all))

        print(f"\n  Cross-study R²: {cross_mean:.4f} +/- {cross_std:.4f}")
        print(f"  Within-study R²: {within_mean:.4f} +/- {within_std:.4f}")

        results[fs_name] = {
            "cross_study_r2_mean": cross_mean,
            "cross_study_r2_std": cross_std,
            "cross_study_r2_per_seed": dict(zip([str(s) for s in seeds], cross_r2s)),
            "within_study_r2_mean": within_mean,
            "within_study_r2_std": within_std,
            "within_study_r2_per_seed": dict(zip([str(s) for s in seeds], within_r2s_all)),
            "n_features": len(feat_cols),
        }

    # --- Feature importance on context-conditioned model ---
    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE (molecular + context, seed=42)")
    print("=" * 70)

    train_ids, test_ids = _study_split(encoded, seed=42)
    train_mask = encoded["study_id"].isin(train_ids)

    X_train = encoded.loc[train_mask, all_feat_cols].values
    y_train = encoded.loc[train_mask, "Experiment_value"].values
    model = _train_xgb(X_train, y_train, seed=42)

    importances = model.feature_importances_
    feat_imp = sorted(
        zip(all_feat_cols, importances.tolist()),
        key=lambda x: x[1],
        reverse=True,
    )

    # Group by category
    mol_imp = sum(v for k, v in feat_imp if k in mol_cols)
    ctx_imp = sum(v for k, v in feat_imp if k in ctx_cols)
    total_imp = mol_imp + ctx_imp

    print(f"\n  Molecular importance: {mol_imp / total_imp:.1%}")
    print(f"  Context importance:   {ctx_imp / total_imp:.1%}")

    # Top context features
    ctx_ranked = [(k, v) for k, v in feat_imp if k in ctx_cols]
    print("\n  Top context features:")
    for name, imp in ctx_ranked[:15]:
        print(f"    {name}: {imp / total_imp:.2%}")

    # Per context column group
    print("\n  Context importance by column group:")
    col_group_imp = {}
    for name, imp in feat_imp:
        if name in ctx_cols:
            prefix = name.split("__")[0].replace("ctx_", "")
            col_group_imp[prefix] = col_group_imp.get(prefix, 0.0) + imp
    for group, imp in sorted(col_group_imp.items(), key=lambda x: x[1], reverse=True):
        print(f"    {group}: {imp / total_imp:.2%}")

    results["feature_importance"] = {
        "molecular_fraction": float(mol_imp / total_imp),
        "context_fraction": float(ctx_imp / total_imp),
        "top_context_features": {k: float(v / total_imp) for k, v in ctx_ranked[:15]},
        "context_group_importance": {k: float(v / total_imp) for k, v in col_group_imp.items()},
    }

    # --- Summary ---
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    mol_cross = results["molecular_only"]["cross_study_r2_mean"]
    ctx_cross = results["molecular_plus_context"]["cross_study_r2_mean"]
    mol_within = results["molecular_only"]["within_study_r2_mean"]
    ctx_within = results["molecular_plus_context"]["within_study_r2_mean"]

    print(f"\n  Cross-study R² (mol only):    {mol_cross:.4f}")
    print(f"  Cross-study R² (mol+ctx):     {ctx_cross:.4f}")
    print(f"  Delta cross-study:            {ctx_cross - mol_cross:+.4f}")
    print(f"\n  Within-study R² (mol only):   {mol_within:.4f}")
    print(f"  Within-study R² (mol+ctx):    {ctx_within:.4f}")
    print(f"  Delta within-study:           {ctx_within - mol_within:+.4f}")

    ctx_above_zero = ctx_cross > 0
    print(f"\n  Context-conditioned cross-study R² > 0? {ctx_above_zero}")

    # Save results
    out_path = Path("models") / "context_conditioned_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
