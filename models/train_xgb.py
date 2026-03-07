#!/usr/bin/env python3
"""XGBoost tabular baseline for LNPDB property prediction.

Uses Morgan fingerprint bits + continuous features + one-hot categoricals.
No MPNN — pure tabular approach for comparison.

Usage:
    python models/train_xgb.py
    python models/train_xgb.py --fp-bits 2048 --fp-radius 3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data import (
    SMILES_COLS,
    TABULAR_CONTINUOUS_COLS,
    encode_categoricals,
    learn_categorical_levels,
    load_lnpdb_dataframe,
    scaffold_split,
)


def compute_morgan_fingerprints(
    smiles_list: list[str],
    radius: int = 2,
    n_bits: int = 1024,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train XGBoost baseline on LNPDB")

    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--components", nargs="+", default=["IL"],
                        choices=["IL", "HL", "CHL", "PEG"])
    parser.add_argument("--split", type=str, default="scaffold",
                        choices=["scaffold", "random"])
    parser.add_argument("--split-seed", type=int, default=42)

    parser.add_argument("--fp-bits", type=int, default=1024)
    parser.add_argument("--fp-radius", type=int, default=2)
    parser.add_argument("--no-categorical", action="store_true")
    parser.add_argument("--no-fingerprint", action="store_true")

    parser.add_argument("--n-estimators", type=int, default=1000)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--early-stopping", type=int, default=50)

    parser.add_argument("--save-dir", type=str, default="models/runs/xgb_baseline")
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

    # Continuous features
    cont_cols = [c for c in TABULAR_CONTINUOUS_COLS if c in df.columns]
    print(f"Continuous features ({len(cont_cols)}): {cont_cols}")

    # Split first (need train indices for learning categorical levels)
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

    # Categorical features (learn levels from training split only)
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
    feature_parts_train = []
    feature_parts_val = []
    feature_parts_test = []
    feature_names: list[str] = []

    # Morgan fingerprints per component
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

    # Continuous features
    if cont_cols:
        feature_parts_train.append(df_train[cont_cols].fillna(0).values.astype(np.float32))
        feature_parts_val.append(df_val[cont_cols].fillna(0).values.astype(np.float32))
        feature_parts_test.append(df_test[cont_cols].fillna(0).values.astype(np.float32))
        feature_names.extend(cont_cols)

    # Categorical features (already one-hot)
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

    # Train XGBoost
    import xgboost as xgb

    print(f"\nTraining XGBoost (n_estimators={args.n_estimators}, "
          f"max_depth={args.max_depth}, lr={args.learning_rate})...")

    model = xgb.XGBRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        random_state=args.seed,
        n_jobs=-1,
        tree_method="hist",
        early_stopping_rounds=args.early_stopping,
    )

    t0 = time.time()
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )
    train_time = time.time() - t0
    print(f"Training done in {train_time:.1f}s")

    # Evaluate
    y_pred_test = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_test)))
    mae = float(mean_absolute_error(y_test, y_pred_test))
    r2 = float(r2_score(y_test, y_pred_test))

    print(f"\n{'='*60}")
    print("TEST RESULTS:")
    print(f"  RMSE:  {rmse:.4f}")
    print(f"  MAE:   {mae:.4f}")
    print(f"  R2:    {r2:.4f}")
    print(f"  Best iteration: {model.best_iteration}")
    print(f"{'='*60}")

    # Feature importance (top 20)
    importance = model.feature_importances_
    top_idx = np.argsort(importance)[::-1][:20]
    print("\nTop 20 features by importance:")
    for i, idx in enumerate(top_idx):
        print(f"  {i+1:2d}. {feature_names[idx]:40s} {importance[idx]:.4f}")

    # Save metrics
    metrics = {
        "test_rmse": rmse,
        "test_mae": mae,
        "test_r2": r2,
        "best_iteration": int(model.best_iteration),
        "n_train": len(df_train),
        "n_val": len(df_val),
        "n_test": len(df_test),
        "n_features": X_train.shape[1],
        "train_time_s": train_time,
        "args": vars(args),
    }
    with open(save_dir / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Save model
    model.save_model(str(save_dir / "model.json"))

    # Save plots
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        _fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

        # Pred vs Actual
        ax = axes[0]
        ax.scatter(y_test, y_pred_test, alpha=0.3, s=8, edgecolors="none")
        lims = [min(y_test.min(), y_pred_test.min()), max(y_test.max(), y_pred_test.max())]
        ax.plot(lims, lims, "k--", alpha=0.5, linewidth=1)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title(f"Test: RMSE={rmse:.3f}, R2={r2:.3f}")
        ax.grid(True, alpha=0.3)

        # Residuals
        ax = axes[1]
        residuals = y_pred_test - y_test
        ax.scatter(y_test, residuals, alpha=0.3, s=8, edgecolors="none")
        ax.axhline(y=0, color="k", linestyle="--", alpha=0.5, linewidth=1)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Residual")
        ax.set_title(f"Residuals (MAE={mae:.3f})")
        ax.grid(True, alpha=0.3)

        # Feature importance (top 20)
        ax = axes[2]
        top_names = [feature_names[i] for i in top_idx]
        top_imp = importance[top_idx]
        ax.barh(range(len(top_names)), top_imp[::-1])
        ax.set_yticks(range(len(top_names)))
        ax.set_yticklabels(top_names[::-1], fontsize=7)
        ax.set_xlabel("Importance")
        ax.set_title("Top 20 Features")

        plt.tight_layout()
        plt.savefig(save_dir / "results.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Plots saved to {save_dir / 'results.png'}")
    except ImportError:
        print("matplotlib not available, skipping plots")

    print(f"\nAll outputs saved to {save_dir}/")


if __name__ == "__main__":
    main()
