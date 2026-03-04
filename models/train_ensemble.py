#!/usr/bin/env python3
"""Ensemble baseline: Random Forest + Extra Trees + XGBoost for LNPDB.

Trains three tree-based regressors and averages their predictions.
Uses Morgan fingerprints (2048-bit, radius 3) + continuous + categorical features.

Usage:
    python models/train_ensemble.py
    python models/train_ensemble.py --save-dir models/runs/ensemble
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data import (
    SMILES_COLS,
    TABULAR_CATEGORICAL_COLS,
    TABULAR_CONTINUOUS_COLS,
    encode_categoricals,
    learn_categorical_levels,
    load_lnpdb_dataframe,
    scaffold_split,
)


def compute_morgan_fingerprints(
    smiles_list: list[str],
    radius: int = 3,
    n_bits: int = 2048,
) -> np.ndarray:
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fps = np.zeros((len(smiles_list), n_bits), dtype=np.float32)
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is not None:
            fps[i] = np.array(gen.GetFingerprint(mol), dtype=np.float32)
    return fps


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RF+ET+XGB ensemble on LNPDB")

    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--components", nargs="+", default=["IL"],
                        choices=["IL", "HL", "CHL", "PEG"])
    parser.add_argument("--split", type=str, default="scaffold",
                        choices=["scaffold", "random"])
    parser.add_argument("--split-seed", type=int, default=42)

    parser.add_argument("--fp-bits", type=int, default=2048)
    parser.add_argument("--fp-radius", type=int, default=3)
    parser.add_argument("--no-categorical", action="store_true")
    parser.add_argument("--no-fingerprint", action="store_true")

    # RF / ET
    parser.add_argument("--rf-n-estimators", type=int, default=500)
    parser.add_argument("--et-n-estimators", type=int, default=500)
    # XGBoost
    parser.add_argument("--xgb-n-estimators", type=int, default=1000)
    parser.add_argument("--xgb-max-depth", type=int, default=8)
    parser.add_argument("--xgb-learning-rate", type=float, default=0.05)
    parser.add_argument("--xgb-subsample", type=float, default=0.8)
    parser.add_argument("--xgb-colsample-bytree", type=float, default=0.8)
    parser.add_argument("--xgb-early-stopping", type=int, default=50)

    parser.add_argument("--save-dir", type=str, default="models/runs/ensemble")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main():
    args = parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df = load_lnpdb_dataframe(args.data_path, components=args.components)
    print(f"Loaded {len(df)} rows")

    df = df.dropna(subset=["IL_SMILES", "Experiment_value"]).reset_index(drop=True)
    print(f"After filtering: {len(df)} rows")

    cont_cols = [c for c in TABULAR_CONTINUOUS_COLS if c in df.columns]
    print(f"Continuous features ({len(cont_cols)}): {cont_cols}")

    # Split
    print(f"Splitting ({args.split})...")
    if args.split == "scaffold":
        il_smiles = df["IL_SMILES"].tolist()
        train_idx, val_idx, test_idx = scaffold_split(
            il_smiles, sizes=(0.8, 0.1, 0.1), seed=args.split_seed
        )
    else:
        n = len(df)
        rng = np.random.RandomState(args.split_seed)
        indices = rng.permutation(n).tolist()
        n_train = int(n * 0.8)
        n_val = int(n * 0.1)
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]

    # Categoricals (learn from train only)
    cat_cols: list[str] = []
    if not args.no_categorical:
        cat_levels = learn_categorical_levels(df.iloc[train_idx])
        df, cat_cols = encode_categoricals(df, cat_levels)
        print(f"Categorical features ({len(cat_cols)} one-hot from "
              f"{len(cat_levels)} columns): {list(cat_levels.keys())}")

    df_train = df.iloc[train_idx]
    df_val = df.iloc[val_idx]
    df_test = df.iloc[test_idx]
    print(f"Split: train={len(df_train)}, val={len(df_val)}, test={len(df_test)}")

    # Build feature matrices
    feature_parts_train, feature_parts_val, feature_parts_test = [], [], []
    feature_names: list[str] = []

    if not args.no_fingerprint:
        for comp in args.components:
            smiles_col = SMILES_COLS[comp]
            print(f"Computing Morgan FP ({comp}, radius={args.fp_radius}, bits={args.fp_bits})...")
            t0 = time.time()
            fp_train = compute_morgan_fingerprints(
                df_train[smiles_col].tolist(), args.fp_radius, args.fp_bits
            )
            fp_val = compute_morgan_fingerprints(
                df_val[smiles_col].tolist(), args.fp_radius, args.fp_bits
            )
            fp_test = compute_morgan_fingerprints(
                df_test[smiles_col].tolist(), args.fp_radius, args.fp_bits
            )
            feature_parts_train.append(fp_train)
            feature_parts_val.append(fp_val)
            feature_parts_test.append(fp_test)
            feature_names.extend([f"{comp}_fp_{i}" for i in range(args.fp_bits)])
            print(f"  Done in {time.time() - t0:.1f}s")

    if cont_cols:
        feature_parts_train.append(df_train[cont_cols].fillna(0).values.astype(np.float32))
        feature_parts_val.append(df_val[cont_cols].fillna(0).values.astype(np.float32))
        feature_parts_test.append(df_test[cont_cols].fillna(0).values.astype(np.float32))
        feature_names.extend(cont_cols)

    if cat_cols:
        feature_parts_train.append(df_train[cat_cols].fillna(0).values.astype(np.float32))
        feature_parts_val.append(df_val[cat_cols].fillna(0).values.astype(np.float32))
        feature_parts_test.append(df_test[cat_cols].fillna(0).values.astype(np.float32))
        feature_names.extend(cat_cols)

    X_train = np.concatenate(feature_parts_train, axis=1)
    X_val = np.concatenate(feature_parts_val, axis=1)
    X_test = np.concatenate(feature_parts_test, axis=1)
    y_train = df_train["Experiment_value"].values.astype(np.float32)
    y_val = df_val["Experiment_value"].values.astype(np.float32)
    y_test = df_test["Experiment_value"].values.astype(np.float32)

    print(f"\nFeature matrix: {X_train.shape[1]} features")
    print(f"  Fingerprint: {sum(1 for n in feature_names if '_fp_' in n)}")
    print(f"  Continuous: {len(cont_cols)}")
    print(f"  Categorical: {len(cat_cols)}")

    # ---- Train individual models ----
    models = {}
    predictions = {}
    individual_metrics = {}

    # 1. Random Forest
    print(f"\n[1/3] Training Random Forest (n_estimators={args.rf_n_estimators}, max_depth=None)...")
    t0 = time.time()
    rf = RandomForestRegressor(
        n_estimators=args.rf_n_estimators,
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=args.seed,
    )
    rf.fit(X_train, y_train)
    rf_time = time.time() - t0
    print(f"  Done in {rf_time:.1f}s")

    pred_rf_test = rf.predict(X_test)
    pred_rf_val = rf.predict(X_val)
    m = evaluate(y_test, pred_rf_test)
    individual_metrics["random_forest"] = {**m, "train_time_s": rf_time}
    models["random_forest"] = rf
    predictions["random_forest"] = {"test": pred_rf_test, "val": pred_rf_val}
    print(f"  Test RMSE={m['rmse']:.4f}  MAE={m['mae']:.4f}  R2={m['r2']:.4f}")

    # 2. Extra Trees
    print(f"\n[2/3] Training Extra Trees (n_estimators={args.et_n_estimators}, max_depth=None)...")
    t0 = time.time()
    et = ExtraTreesRegressor(
        n_estimators=args.et_n_estimators,
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=args.seed,
    )
    et.fit(X_train, y_train)
    et_time = time.time() - t0
    print(f"  Done in {et_time:.1f}s")

    pred_et_test = et.predict(X_test)
    pred_et_val = et.predict(X_val)
    m = evaluate(y_test, pred_et_test)
    individual_metrics["extra_trees"] = {**m, "train_time_s": et_time}
    models["extra_trees"] = et
    predictions["extra_trees"] = {"test": pred_et_test, "val": pred_et_val}
    print(f"  Test RMSE={m['rmse']:.4f}  MAE={m['mae']:.4f}  R2={m['r2']:.4f}")

    # 3. XGBoost
    import xgboost as xgb

    print(f"\n[3/3] Training XGBoost (n_estimators={args.xgb_n_estimators}, "
          f"max_depth={args.xgb_max_depth}, lr={args.xgb_learning_rate})...")
    t0 = time.time()
    xgb_model = xgb.XGBRegressor(
        n_estimators=args.xgb_n_estimators,
        max_depth=args.xgb_max_depth,
        learning_rate=args.xgb_learning_rate,
        subsample=args.xgb_subsample,
        colsample_bytree=args.xgb_colsample_bytree,
        random_state=args.seed,
        n_jobs=-1,
        tree_method="hist",
        early_stopping_rounds=args.xgb_early_stopping,
    )
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )
    xgb_time = time.time() - t0
    print(f"  Done in {xgb_time:.1f}s")

    pred_xgb_test = xgb_model.predict(X_test)
    pred_xgb_val = xgb_model.predict(X_val)
    m = evaluate(y_test, pred_xgb_test)
    individual_metrics["xgboost"] = {
        **m,
        "train_time_s": xgb_time,
        "best_iteration": int(xgb_model.best_iteration),
    }
    models["xgboost"] = xgb_model
    predictions["xgboost"] = {"test": pred_xgb_test, "val": pred_xgb_val}
    print(f"  Test RMSE={m['rmse']:.4f}  MAE={m['mae']:.4f}  R2={m['r2']:.4f}")

    # ---- Ensemble (mean of all 3) ----
    print(f"\n{'='*60}")
    print("ENSEMBLE (mean of RF + ET + XGB)")
    print(f"{'='*60}")

    pred_ens_test = np.mean([pred_rf_test, pred_et_test, pred_xgb_test], axis=0)
    pred_ens_val = np.mean([pred_rf_val, pred_et_val, pred_xgb_val], axis=0)
    ens_metrics = evaluate(y_test, pred_ens_test)
    total_time = rf_time + et_time + xgb_time

    print(f"  Test RMSE:  {ens_metrics['rmse']:.4f}")
    print(f"  Test MAE:   {ens_metrics['mae']:.4f}")
    print(f"  Test R2:    {ens_metrics['r2']:.4f}")
    print(f"  Total train time: {total_time:.1f}s")

    # Summary table
    print(f"\n{'='*60}")
    print(f"{'Model':<20s} {'RMSE':>8s} {'MAE':>8s} {'R2':>8s}")
    print(f"{'-'*20} {'-'*8} {'-'*8} {'-'*8}")
    for name, met in individual_metrics.items():
        print(f"{name:<20s} {met['rmse']:8.4f} {met['mae']:8.4f} {met['r2']:8.4f}")
    print(f"{'-'*20} {'-'*8} {'-'*8} {'-'*8}")
    print(f"{'ENSEMBLE':<20s} {ens_metrics['rmse']:8.4f} {ens_metrics['mae']:8.4f} {ens_metrics['r2']:8.4f}")
    print(f"{'='*60}")

    # ---- Feature importance from Random Forest ----
    importance = rf.feature_importances_
    top_idx = np.argsort(importance)[::-1][:20]
    print("\nTop 20 features by RF importance:")
    for i, idx in enumerate(top_idx):
        print(f"  {i+1:2d}. {feature_names[idx]:40s} {importance[idx]:.6f}")

    # ---- Save everything ----
    all_metrics = {
        "ensemble": {
            **ens_metrics,
            "total_train_time_s": total_time,
        },
        "individual": individual_metrics,
        "n_train": len(df_train),
        "n_val": len(df_val),
        "n_test": len(df_test),
        "n_features": X_train.shape[1],
        "args": vars(args),
    }
    with open(save_dir / "test_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    # Save XGBoost model (native format)
    xgb_model.save_model(str(save_dir / "xgb_model.json"))

    # Save RF and ET models (joblib)
    import joblib
    joblib.dump(rf, save_dir / "rf_model.joblib")
    joblib.dump(et, save_dir / "et_model.joblib")

    # Save feature importance
    imp_df = pd.DataFrame({
        "feature": feature_names,
        "rf_importance": rf.feature_importances_,
        "et_importance": et.feature_importances_,
        "xgb_importance": xgb_model.feature_importances_,
    })
    imp_df["mean_importance"] = imp_df[["rf_importance", "et_importance", "xgb_importance"]].mean(axis=1)
    imp_df = imp_df.sort_values("mean_importance", ascending=False)
    imp_df.to_csv(save_dir / "feature_importance.csv", index=False)

    # Plots
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        # Row 1: Pred vs Actual for each model + ensemble
        model_preds = [
            ("Random Forest", pred_rf_test),
            ("Extra Trees", pred_et_test),
            ("XGBoost", pred_xgb_test),
        ]
        for ax, (name, preds) in zip(axes[0, :], model_preds):
            m = evaluate(y_test, preds)
            ax.scatter(y_test, preds, alpha=0.3, s=8, edgecolors="none")
            lims = [min(y_test.min(), preds.min()), max(y_test.max(), preds.max())]
            ax.plot(lims, lims, "k--", alpha=0.5, linewidth=1)
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            ax.set_title(f"{name}: RMSE={m['rmse']:.3f}, R2={m['r2']:.3f}")
            ax.grid(True, alpha=0.3)

        # Row 2, left: Ensemble pred vs actual
        ax = axes[1, 0]
        ax.scatter(y_test, pred_ens_test, alpha=0.3, s=8, edgecolors="none", color="tab:purple")
        lims = [min(y_test.min(), pred_ens_test.min()), max(y_test.max(), pred_ens_test.max())]
        ax.plot(lims, lims, "k--", alpha=0.5, linewidth=1)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title(f"Ensemble: RMSE={ens_metrics['rmse']:.3f}, R2={ens_metrics['r2']:.3f}")
        ax.grid(True, alpha=0.3)

        # Row 2, middle: Residuals
        ax = axes[1, 1]
        residuals = pred_ens_test - y_test
        ax.scatter(y_test, residuals, alpha=0.3, s=8, edgecolors="none", color="tab:purple")
        ax.axhline(y=0, color="k", linestyle="--", alpha=0.5, linewidth=1)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Residual")
        ax.set_title(f"Ensemble Residuals (MAE={ens_metrics['mae']:.3f})")
        ax.grid(True, alpha=0.3)

        # Row 2, right: Feature importance (top 20, RF)
        ax = axes[1, 2]
        top_names = [feature_names[i] for i in top_idx]
        top_imp = importance[top_idx]
        ax.barh(range(len(top_names)), top_imp[::-1])
        ax.set_yticks(range(len(top_names)))
        ax.set_yticklabels(top_names[::-1], fontsize=7)
        ax.set_xlabel("Importance (RF)")
        ax.set_title("Top 20 Features (Random Forest)")

        plt.tight_layout()
        plt.savefig(save_dir / "results.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Plots saved to {save_dir / 'results.png'}")
    except ImportError:
        print("matplotlib not available, skipping plots")

    print(f"\nAll outputs saved to {save_dir}/")


if __name__ == "__main__":
    main()
