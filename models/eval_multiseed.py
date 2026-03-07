#!/usr/bin/env python3
"""Multi-seed evaluation: XGBoost, Random Forest, RF+XGB ensemble."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data import (
    TABULAR_CONTINUOUS_COLS,
    encode_categoricals,
    learn_categorical_levels,
    load_lnpdb_dataframe,
    scaffold_split,
)


def compute_morgan_fingerprints(smiles_list, radius=3, n_bits=2048):
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fps = np.zeros((len(smiles_list), n_bits), dtype=np.float32)
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is not None:
            fps[i] = np.array(gen.GetFingerprint(mol), dtype=np.float32)
    return fps


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
cont_cols = [c for c in TABULAR_CONTINUOUS_COLS if c in df.columns]
il_smiles = df["IL_SMILES"].tolist()

results = {seed: {} for seed in SEEDS}

for seed in SEEDS:
    print(f"\n{'='*60}")
    print(f"SEED = {seed}")
    print(f"{'='*60}")

    train_idx, val_idx, test_idx = scaffold_split(il_smiles, sizes=(0.8, 0.1, 0.1), seed=seed)

    cat_levels = learn_categorical_levels(df.iloc[train_idx])
    df_enc = df.copy()
    df_enc, cat_cols = encode_categoricals(df_enc, cat_levels)

    df_train = df_enc.iloc[train_idx]
    df_val = df_enc.iloc[val_idx]
    df_test = df_enc.iloc[test_idx]

    fp_train = compute_morgan_fingerprints(df_train["IL_SMILES"].tolist())
    fp_val = compute_morgan_fingerprints(df_val["IL_SMILES"].tolist())
    fp_test = compute_morgan_fingerprints(df_test["IL_SMILES"].tolist())

    parts_train = [fp_train]
    parts_val = [fp_val]
    parts_test = [fp_test]

    if cont_cols:
        parts_train.append(df_train[cont_cols].fillna(0).values.astype(np.float32))
        parts_val.append(df_val[cont_cols].fillna(0).values.astype(np.float32))
        parts_test.append(df_test[cont_cols].fillna(0).values.astype(np.float32))
    if cat_cols:
        parts_train.append(df_train[cat_cols].fillna(0).values.astype(np.float32))
        parts_val.append(df_val[cat_cols].fillna(0).values.astype(np.float32))
        parts_test.append(df_test[cat_cols].fillna(0).values.astype(np.float32))

    X_train = np.concatenate(parts_train, axis=1)
    X_val = np.concatenate(parts_val, axis=1)
    X_test = np.concatenate(parts_test, axis=1)
    y_train = df_train["Experiment_value"].values.astype(np.float32)
    y_val = df_val["Experiment_value"].values.astype(np.float32)
    y_test = df_test["Experiment_value"].values.astype(np.float32)

    print(f"  train={len(df_train)}, val={len(df_val)}, test={len(df_test)}, features={X_train.shape[1]}")

    # Random Forest
    t0 = time.time()
    rf = RandomForestRegressor(n_estimators=500, max_depth=None, min_samples_leaf=2, n_jobs=-1, random_state=seed)
    rf.fit(X_train, y_train)
    rf_time = time.time() - t0
    pred_rf = rf.predict(X_test)
    rf_m = evaluate(y_test, pred_rf)
    results[seed]["rf"] = rf_m
    print(f"  RF:  RMSE={rf_m['rmse']:.4f}  MAE={rf_m['mae']:.4f}  R2={rf_m['r2']:.4f}  ({rf_time:.1f}s)")

    # XGBoost
    import xgboost as xgb
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
    results[seed]["xgb"] = xgb_m
    print(f"  XGB: RMSE={xgb_m['rmse']:.4f}  MAE={xgb_m['mae']:.4f}  R2={xgb_m['r2']:.4f}  ({xgb_time:.1f}s)")

    # RF+XGB ensemble (mean)
    pred_ens = np.mean([pred_rf, pred_xgb], axis=0)
    ens_m = evaluate(y_test, pred_ens)
    results[seed]["rf_xgb"] = ens_m
    print(f"  R+X: RMSE={ens_m['rmse']:.4f}  MAE={ens_m['mae']:.4f}  R2={ens_m['r2']:.4f}")

# Summary
print(f"\n{'='*60}")
print("MULTI-SEED SUMMARY (mean +/- std across 5 seeds)")
print(f"{'='*60}")
for model_name in ["rf", "xgb", "rf_xgb"]:
    rmses = [results[s][model_name]["rmse"] for s in SEEDS]
    maes = [results[s][model_name]["mae"] for s in SEEDS]
    r2s = [results[s][model_name]["r2"] for s in SEEDS]
    label = {"rf": "Random Forest", "xgb": "XGBoost", "rf_xgb": "RF+XGB Ensemble"}[model_name]
    print(f"\n{label}:")
    print(f"  RMSE: {np.mean(rmses):.4f} +/- {np.std(rmses):.4f}  (range: {min(rmses):.4f} - {max(rmses):.4f})")
    print(f"  MAE:  {np.mean(maes):.4f} +/- {np.std(maes):.4f}  (range: {min(maes):.4f} - {max(maes):.4f})")
    print(f"  R2:   {np.mean(r2s):.4f} +/- {np.std(r2s):.4f}  (range: {min(r2s):.4f} - {max(r2s):.4f})")

# Save
save_path = Path(__file__).resolve().parent / "runs" / "multiseed_results.json"
save_path.parent.mkdir(parents=True, exist_ok=True)
with open(save_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {save_path}")
