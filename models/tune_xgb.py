#!/usr/bin/env python3
"""Optuna hyperparameter optimization for XGBoost on LNPDB.

Uses the same data pipeline as train_xgb.py: Morgan FP + continuous + categoricals,
scaffold split (seed=42), IL-only.

Usage:
    python models/tune_xgb.py
    python models/tune_xgb.py --n-trials 200
"""


import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import optuna
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data import (
    TABULAR_CONTINUOUS_COLS,
    encode_categoricals,
    learn_categorical_levels,
    load_lnpdb_dataframe,
    scaffold_split,
)

FP_BITS = 2048
FP_RADIUS = 3
SPLIT_SEED = 42
SAVE_DIR = Path(__file__).resolve().parent / "runs" / "xgb_tuned"


def compute_morgan_fingerprints(
    smiles_list: list[str],
    radius: int = FP_RADIUS,
    n_bits: int = FP_BITS,
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


def build_features():
    """Load data, split, featurize. Returns X/y arrays and feature names."""
    print("Loading data...")
    df = load_lnpdb_dataframe(components=["IL"])
    df = df.dropna(subset=["IL_SMILES", "Experiment_value"]).reset_index(drop=True)
    print(f"Rows after filtering: {len(df)}")

    cont_cols = [c for c in TABULAR_CONTINUOUS_COLS if c in df.columns]

    il_smiles = df["IL_SMILES"].tolist()
    train_idx, val_idx, test_idx = scaffold_split(
        il_smiles, sizes=(0.8, 0.1, 0.1), seed=SPLIT_SEED
    )

    cat_levels = learn_categorical_levels(df.iloc[train_idx])
    df, cat_cols = encode_categoricals(df, cat_levels)

    df_train = df.iloc[train_idx]
    df_val = df.iloc[val_idx]
    df_test = df.iloc[test_idx]

    print(f"Split: train={len(df_train)}, val={len(df_val)}, test={len(df_test)}")

    feature_names: list[str] = []

    print(f"Computing Morgan FP (radius={FP_RADIUS}, bits={FP_BITS})...")
    t0 = time.time()
    fp_train = compute_morgan_fingerprints(df_train["IL_SMILES"].tolist())
    fp_val = compute_morgan_fingerprints(df_val["IL_SMILES"].tolist())
    fp_test = compute_morgan_fingerprints(df_test["IL_SMILES"].tolist())
    print(f"  Done in {time.time() - t0:.1f}s")
    feature_names.extend([f"IL_fp_{i}" for i in range(FP_BITS)])

    parts_train = [fp_train]
    parts_val = [fp_val]
    parts_test = [fp_test]

    if cont_cols:
        parts_train.append(df_train[cont_cols].fillna(0).values.astype(np.float32))
        parts_val.append(df_val[cont_cols].fillna(0).values.astype(np.float32))
        parts_test.append(df_test[cont_cols].fillna(0).values.astype(np.float32))
        feature_names.extend(cont_cols)

    if cat_cols:
        parts_train.append(df_train[cat_cols].fillna(0).values.astype(np.float32))
        parts_val.append(df_val[cat_cols].fillna(0).values.astype(np.float32))
        parts_test.append(df_test[cat_cols].fillna(0).values.astype(np.float32))
        feature_names.extend(cat_cols)

    X_train = np.concatenate(parts_train, axis=1)
    X_val = np.concatenate(parts_val, axis=1)
    X_test = np.concatenate(parts_test, axis=1)
    y_train = df_train["Experiment_value"].values.astype(np.float32)
    y_val = df_val["Experiment_value"].values.astype(np.float32)
    y_test = df_test["Experiment_value"].values.astype(np.float32)

    print(f"Feature matrix: {X_train.shape[1]} features")

    return X_train, X_val, X_test, y_train, y_val, y_test, feature_names


def objective(trial: optuna.Trial, X_train, y_train, X_val, y_val) -> float:
    params = {
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 500, 5000),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }

    model = xgb.XGBRegressor(
        **params,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
        early_stopping_rounds=100,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    y_pred_val = model.predict(X_val)
    rmse = float(np.sqrt(mean_squared_error(y_val, y_pred_val)))

    # Store best iteration for later use
    trial.set_user_attr("best_iteration", model.best_iteration)

    return rmse


def main():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    X_train, X_val, X_test, y_train, y_val, y_test, feature_names = build_features()

    n_trials = 100
    if len(sys.argv) > 1:
        for i, arg in enumerate(sys.argv[1:]):
            if arg == "--n-trials" and i + 2 < len(sys.argv):
                n_trials = int(sys.argv[i + 2])

    print(f"\nStarting Optuna optimization ({n_trials} trials)...")
    print("Baseline: RMSE=0.812, R2=0.349")
    print("=" * 60)

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        study_name="xgb_lnpdb",
    )

    def _objective(trial):
        return objective(trial, X_train, y_train, X_val, y_val)

    def _callback(study, trial):
        if trial.number % 10 == 0 or trial.value == study.best_value:
            print(f"Trial {trial.number:3d}: val_rmse={trial.value:.4f} "
                  f"(best={study.best_value:.4f})")

    t0 = time.time()
    study.optimize(_objective, n_trials=n_trials, callbacks=[_callback])
    opt_time = time.time() - t0

    print(f"\nOptimization finished in {opt_time:.1f}s")
    print(f"Best trial: #{study.best_trial.number}")
    print(f"Best val RMSE: {study.best_value:.4f}")
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    # Retrain best model on train, evaluate on test
    print("\nRetraining best model for test evaluation...")
    best_params = study.best_params.copy()

    best_model = xgb.XGBRegressor(
        **best_params,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
        early_stopping_rounds=100,
    )
    best_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    y_pred_test = best_model.predict(X_test)
    test_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_test)))
    test_mae = float(mean_absolute_error(y_test, y_pred_test))
    test_r2 = float(r2_score(y_test, y_pred_test))

    print(f"\n{'=' * 60}")
    print("TEST RESULTS (best model):")
    print(f"  RMSE:  {test_rmse:.4f}")
    print(f"  MAE:   {test_mae:.4f}")
    print(f"  R2:    {test_r2:.4f}")
    print(f"  Best iteration: {best_model.best_iteration}")
    print(f"{'=' * 60}")
    print("\nBaseline comparison:")
    print(f"  Old RMSE: 0.8122  ->  New RMSE: {test_rmse:.4f}  "
          f"({'better' if test_rmse < 0.8122 else 'worse'})")
    print(f"  Old R2:   0.3493  ->  New R2:   {test_r2:.4f}  "
          f"({'better' if test_r2 > 0.3493 else 'worse'})")

    # Save everything
    metrics = {
        "test_rmse": test_rmse,
        "test_mae": test_mae,
        "test_r2": test_r2,
        "best_iteration": int(best_model.best_iteration),
        "best_val_rmse": float(study.best_value),
        "best_trial_number": study.best_trial.number,
        "n_trials": n_trials,
        "optimization_time_s": opt_time,
        "n_train": len(y_train),
        "n_val": len(y_val),
        "n_test": len(y_test),
        "n_features": X_train.shape[1],
        "best_params": best_params,
        "fp_bits": FP_BITS,
        "fp_radius": FP_RADIUS,
        "split_seed": SPLIT_SEED,
    }
    with open(SAVE_DIR / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    best_model.save_model(str(SAVE_DIR / "model.json"))

    # Save trial history
    trial_data = []
    for t in study.trials:
        trial_data.append({
            "number": t.number,
            "value": t.value,
            "params": t.params,
            "best_iteration": t.user_attrs.get("best_iteration"),
        })
    with open(SAVE_DIR / "trial_history.json", "w") as f:
        json.dump(trial_data, f, indent=2)

    # Plots
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        _fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

        # Pred vs Actual
        ax = axes[0]
        ax.scatter(y_test, y_pred_test, alpha=0.3, s=8, edgecolors="none")
        lims = [min(y_test.min(), y_pred_test.min()),
                max(y_test.max(), y_pred_test.max())]
        ax.plot(lims, lims, "k--", alpha=0.5, linewidth=1)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title(f"Test: RMSE={test_rmse:.3f}, R2={test_r2:.3f}")
        ax.grid(True, alpha=0.3)

        # Residuals
        ax = axes[1]
        residuals = y_pred_test - y_test
        ax.scatter(y_test, residuals, alpha=0.3, s=8, edgecolors="none")
        ax.axhline(y=0, color="k", linestyle="--", alpha=0.5, linewidth=1)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Residual")
        ax.set_title(f"Residuals (MAE={test_mae:.3f})")
        ax.grid(True, alpha=0.3)

        # Optimization history
        ax = axes[2]
        vals = [t.value for t in study.trials]
        best_so_far = [min(vals[:i+1]) for i in range(len(vals))]
        ax.plot(vals, alpha=0.3, label="Trial RMSE", linewidth=0.8)
        ax.plot(best_so_far, color="red", linewidth=1.5, label="Best so far")
        ax.axhline(y=0.8122, color="gray", linestyle="--", alpha=0.5,
                    linewidth=1, label="Baseline (0.812)")
        ax.set_xlabel("Trial")
        ax.set_ylabel("Validation RMSE")
        ax.set_title("Optimization History")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(SAVE_DIR / "results.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Plots saved to {SAVE_DIR / 'results.png'}")
    except ImportError:
        print("matplotlib not available, skipping plots")

    # Feature importance (top 20)
    importance = best_model.feature_importances_
    top_idx = np.argsort(importance)[::-1][:20]
    print("\nTop 20 features by importance:")
    for i, idx in enumerate(top_idx):
        print(f"  {i+1:2d}. {feature_names[idx]:40s} {importance[idx]:.4f}")

    print(f"\nAll outputs saved to {SAVE_DIR}/")


if __name__ == "__main__":
    main()
