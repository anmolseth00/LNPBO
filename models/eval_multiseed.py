#!/usr/bin/env python3
"""Multi-seed evaluation: XGBoost, Random Forest, RF+XGB ensemble.

Evaluates on both scaffold split and study-level split (stratified by assay type).
"""

import json
import time
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from LNPBO.diagnostics.utils import add_assay_type, add_study_id, study_split
from LNPBO.models.data import (
    TABULAR_CONTINUOUS_COLS,
    compute_morgan_fingerprints,
    encode_categoricals,
    learn_categorical_levels,
    load_lnpdb_dataframe,
    scaffold_split,
)


def evaluate(y_true, y_pred):
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


SEEDS = [42, 123, 456, 789, 2024]

print("Loading data...")
df = load_lnpdb_dataframe(None, components=["IL"])
df = df.dropna(subset=["IL_SMILES", "Experiment_value"]).reset_index(drop=True)
df = add_study_id(df)
df = add_assay_type(df)
cont_cols = [c for c in TABULAR_CONTINUOUS_COLS if c in df.columns]
il_smiles = df["IL_SMILES"].tolist()


def _build_features(df_split, il_smiles_list, cat_cols):
    fps = compute_morgan_fingerprints(il_smiles_list)
    parts = [fps]
    if cont_cols:
        parts.append(df_split[cont_cols].fillna(0).values.astype(np.float32))
    if cat_cols:
        parts.append(df_split[cat_cols].fillna(0).values.astype(np.float32))
    return np.concatenate(parts, axis=1)


def _run_split(X_train, y_train, X_val, y_val, X_test, y_test, seed):
    split_results = {}

    t0 = time.time()
    rf = RandomForestRegressor(n_estimators=500, max_depth=None, min_samples_leaf=2, n_jobs=-1, random_state=seed)
    rf.fit(X_train, y_train)
    rf_time = time.time() - t0
    pred_rf = rf.predict(X_test)
    rf_m = evaluate(y_test, pred_rf)
    split_results["rf"] = rf_m
    print(f"    RF:  RMSE={rf_m['rmse']:.4f}  MAE={rf_m['mae']:.4f}  R2={rf_m['r2']:.4f}  ({rf_time:.1f}s)")

    t0 = time.time()
    xgb_model = xgb.XGBRegressor(
        n_estimators=2000, max_depth=8, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8,
        random_state=seed, n_jobs=-1, tree_method="hist",
        early_stopping_rounds=100,
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    xgb_time = time.time() - t0
    pred_xgb = xgb_model.predict(X_test)
    xgb_m = evaluate(y_test, pred_xgb)
    split_results["xgb"] = xgb_m
    print(f"    XGB: RMSE={xgb_m['rmse']:.4f}  MAE={xgb_m['mae']:.4f}  R2={xgb_m['r2']:.4f}  ({xgb_time:.1f}s)")

    pred_ens = np.mean([pred_rf, pred_xgb], axis=0)
    ens_m = evaluate(y_test, pred_ens)
    split_results["rf_xgb"] = ens_m
    print(f"    R+X: RMSE={ens_m['rmse']:.4f}  MAE={ens_m['mae']:.4f}  R2={ens_m['r2']:.4f}")

    return split_results


results = {str(seed): {} for seed in SEEDS}

for seed in SEEDS:
    print(f"\n{'='*60}")
    print(f"SEED = {seed}")
    print(f"{'='*60}")

    # --- Scaffold split ---
    print("  Scaffold split:")
    train_idx, val_idx, test_idx = scaffold_split(il_smiles, sizes=(0.8, 0.1, 0.1), seed=seed)

    cat_levels = learn_categorical_levels(df.iloc[train_idx])
    df_enc = df.copy()
    df_enc, cat_cols = encode_categoricals(df_enc, cat_levels)

    X_train = _build_features(df_enc.iloc[train_idx], df_enc.iloc[train_idx]["IL_SMILES"].tolist(), cat_cols)
    X_val = _build_features(df_enc.iloc[val_idx], df_enc.iloc[val_idx]["IL_SMILES"].tolist(), cat_cols)
    X_test = _build_features(df_enc.iloc[test_idx], df_enc.iloc[test_idx]["IL_SMILES"].tolist(), cat_cols)
    y_train = df_enc.iloc[train_idx]["Experiment_value"].values.astype(np.float32)
    y_val = df_enc.iloc[val_idx]["Experiment_value"].values.astype(np.float32)
    y_test = df_enc.iloc[test_idx]["Experiment_value"].values.astype(np.float32)
    print(f"    train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}, features={X_train.shape[1]}")
    results[str(seed)]["scaffold_split"] = _run_split(X_train, y_train, X_val, y_val, X_test, y_test, seed)

    # --- Study-level split ---
    print("  Study-level split:")
    train_ids, test_ids = study_split(df, seed=seed)
    train_mask = df["study_id"].isin(train_ids).values
    test_mask = df["study_id"].isin(test_ids).values

    cat_levels_s = learn_categorical_levels(df[train_mask])
    df_enc_s = df.copy()
    df_enc_s, cat_cols_s = encode_categoricals(df_enc_s, cat_levels_s)

    X_all = _build_features(df_enc_s, df_enc_s["IL_SMILES"].tolist(), cat_cols_s)
    y_all = df_enc_s["Experiment_value"].values.astype(np.float32)

    X_train_s, y_train_s = X_all[train_mask], y_all[train_mask]
    X_test_s, y_test_s = X_all[test_mask], y_all[test_mask]

    # Hold out 10% of training data for XGB early stopping
    rng = np.random.RandomState(seed)
    n_train = len(X_train_s)
    val_size = max(1, int(0.1 * n_train))
    perm = rng.permutation(n_train)
    val_idx_s = perm[:val_size]
    train_idx_s = perm[val_size:]
    X_val_s = X_train_s[val_idx_s]
    y_val_s = y_train_s[val_idx_s]
    X_train_s_final = X_train_s[train_idx_s]
    y_train_s_final = y_train_s[train_idx_s]

    n_test = int(test_mask.sum())
    print(f"    train={len(train_idx_s)}, val={len(val_idx_s)}, test={n_test}, features={X_train_s.shape[1]}")
    results[str(seed)]["study_split"] = _run_split(
        X_train_s_final, y_train_s_final, X_val_s, y_val_s, X_test_s, y_test_s, seed,
    )

# Summary
print(f"\n{'='*60}")
print("MULTI-SEED SUMMARY (mean +/- std across 5 seeds)")
print(f"{'='*60}")
for split_type in ["scaffold_split", "study_split"]:
    print(f"\n--- {split_type} ---")
    for model_name in ["rf", "xgb", "rf_xgb"]:
        r2s = [results[str(s)][split_type][model_name]["r2"] for s in SEEDS]
        rmses = [results[str(s)][split_type][model_name]["rmse"] for s in SEEDS]
        label = {"rf": "Random Forest", "xgb": "XGBoost", "rf_xgb": "RF+XGB Ensemble"}[model_name]
        r2_mean, r2_std = np.mean(r2s), np.std(r2s)
        rmse_mean, rmse_std = np.mean(rmses), np.std(rmses)
        print(f"  {label}: R2={r2_mean:.4f} +/- {r2_std:.4f}  RMSE={rmse_mean:.4f} +/- {rmse_std:.4f}")

# Save
save_path = Path(__file__).resolve().parent / "runs" / "multiseed_results.json"
save_path.parent.mkdir(parents=True, exist_ok=True)
with open(save_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {save_path}")
