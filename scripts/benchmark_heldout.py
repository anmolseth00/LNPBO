"""Benchmark XGBoost + LANTERN on the 4 LNPDB held-out studies.

Matches the evaluation protocol from Collins et al., Nat Comms 2026:
train on all_data_all.csv, test on heldout_data_all.csv, report Spearman r.

Usage:
    python -m scripts.benchmark_heldout [--with-aux] [--reduction pca]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

from LNPBO.data.compute_pcs import compute_pcs

HELDOUT_ROOT = Path("data/LNPDB_repo/data/LNPDB_for_LiON/heldout")
STUDIES = ["BL_2023", "LM_2019", "SL_2020", "ZC_2023"]

# LiON baselines: mean Spearman r across 5 CV folds on held-out studies
# Computed from heldout_data_results.csv in LNPDB_for_LiON/heldout/
LION_BASELINES = {
    "BL_2023": 0.0711,
    "LM_2019": 0.0950,
    "SL_2020": 0.6331,
    "ZC_2023": 0.0786,
}

# AGILE baselines: Pearson r from Collins' LNPDB_AGILE_training.ipynb
# (mean across 5 CV folds; AGILE code is nondeterministic)
# Note: paper Fig 3b reports Spearman r; notebook computes Pearson r
AGILE_BASELINES = {
    "BL_2023": -0.0215,
    "LM_2019": -0.0158,
    "SL_2020": -0.2242,
    "ZC_2023": 0.2167,
}


def encode_lantern(
    train_smiles: list[str],
    test_smiles: list[str],
    train_y: np.ndarray,
    n_pcs_count_mfp: int = 5,
    n_pcs_rdkit: int = 5,
    reduction: str = "pca",
) -> tuple[np.ndarray, np.ndarray]:
    """Encode SMILES with LANTERN features (count MFP + RDKit, reduced)."""
    train_blocks, test_blocks = [], []

    if n_pcs_count_mfp > 0:
        pcs_tr, reducer, scaler, _ = compute_pcs(
            train_smiles,
            feature_type="count_mfp",
            experiment_values=train_y.tolist(),
            n_components=n_pcs_count_mfp,
            reduction=reduction,
        )
        pcs_te, _, _, _ = compute_pcs(
            test_smiles,
            feature_type="count_mfp",
            n_components=n_pcs_count_mfp,
            reduction=reduction,
            fitted_reducer=reducer,
            fitted_scaler=scaler,
        )
        train_blocks.append(pcs_tr)
        test_blocks.append(pcs_te)

    if n_pcs_rdkit > 0:
        pcs_tr, reducer, scaler, _ = compute_pcs(
            train_smiles,
            feature_type="rdkit",
            experiment_values=train_y.tolist(),
            n_components=n_pcs_rdkit,
            reduction=reduction,
        )
        pcs_te, _, _, _ = compute_pcs(
            test_smiles,
            feature_type="rdkit",
            n_components=n_pcs_rdkit,
            reduction=reduction,
            fitted_reducer=reducer,
            fitted_scaler=scaler,
        )
        train_blocks.append(pcs_tr)
        test_blocks.append(pcs_te)

    return np.hstack(train_blocks), np.hstack(test_blocks)


def load_aux_features(study: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load pre-encoded auxiliary features (ratios, dose, one-hot names, etc.)."""
    base = HELDOUT_ROOT / f"heldout_{study}"
    train_aux = pd.read_csv(base / "all_data_extra_x.csv")
    test_aux = pd.read_csv(base / "heldout_data_extra_x.csv")

    # Align columns: test may lack some one-hot columns present in train
    all_cols = sorted(set(train_aux.columns) | set(test_aux.columns))
    for col in all_cols:
        if col not in train_aux.columns:
            train_aux[col] = 0.0
        if col not in test_aux.columns:
            test_aux[col] = 0.0
    train_aux = train_aux[all_cols]
    test_aux = test_aux[all_cols]

    # Fill NaN with 0 for one-hot features
    train_aux = train_aux.fillna(0.0)
    test_aux = test_aux.fillna(0.0)

    return train_aux, test_aux


def _train_and_predict(X_train, train_y, X_test, seed=42):
    model = XGBRegressor(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.5,
        random_state=seed,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X_train, train_y)
    return model.predict(X_test)


def run_benchmark(
    with_aux: bool = False,
    reduction: str = "pca",
    n_pcs_count_mfp: int = 5,
    n_pcs_rdkit: int = 5,
    cv_folds: int = 1,
):
    results = []

    for study in STUDIES:
        base = HELDOUT_ROOT / f"heldout_{study}"
        train_df = pd.read_csv(base / "all_data_all.csv")
        test_df = pd.read_csv(base / "heldout_data_all.csv")

        train_smiles = train_df["IL_SMILES"].tolist()
        test_smiles = test_df["IL_SMILES"].tolist()
        train_y = train_df["Experiment_value"].to_numpy(dtype=float)
        test_y = test_df["Experiment_value"].to_numpy(dtype=float)

        print(f"\n{'='*60}")
        print(f"Study: {study} | Train: {len(train_y)} | Test: {len(test_y)}")
        print(f"{'='*60}")

        # LANTERN features (always fit on full training set)
        X_train, X_test = encode_lantern(
            train_smiles, test_smiles, train_y,
            n_pcs_count_mfp=n_pcs_count_mfp,
            n_pcs_rdkit=n_pcs_rdkit,
            reduction=reduction,
        )
        print(f"LANTERN features: {X_train.shape[1]} dims")

        if with_aux:
            train_aux, test_aux = load_aux_features(study)
            X_train = np.hstack([X_train, train_aux.values])
            X_test = np.hstack([X_test, test_aux.values])
            print(f"+ Auxiliary features: {train_aux.shape[1]} dims")
            print(f"Total features: {X_train.shape[1]} dims")

        if cv_folds <= 1:
            # Single model on all training data
            preds = _train_and_predict(X_train, train_y, X_test)
            r, _ = spearmanr(test_y, preds)
            r2 = r2_score(test_y, preds)
            r_std = 0.0
        else:
            # K-fold CV matching paper's protocol: train on 80% of training data,
            # predict on held-out study, average across folds
            from sklearn.model_selection import KFold
            kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
            fold_rs = []
            fold_r2s = []
            for fold_idx, (tr_idx, _) in enumerate(kf.split(X_train)):
                preds = _train_and_predict(
                    X_train[tr_idx], train_y[tr_idx], X_test, seed=fold_idx
                )
                fold_r, _ = spearmanr(test_y, preds)
                fold_rs.append(fold_r)
                fold_r2s.append(r2_score(test_y, preds))
            r = float(np.mean(fold_rs))
            r_std = float(np.std(fold_rs))
            r2 = float(np.mean(fold_r2s))
            print(f"  CV folds: {[f'{x:.4f}' for x in fold_rs]}")

        lion_r = LION_BASELINES[study]
        agile_r = AGILE_BASELINES[study]

        r_str = f"{r:.4f}" if r_std == 0 else f"{r:.4f} ± {r_std:.4f}"
        print(f"\nSpearman r:  {r_str}  (LiON: {lion_r:.4f}, AGILE: {agile_r:.3f})")
        print(f"R²:          {r2:.4f}")
        beat_lion = "YES" if r > lion_r else "no"
        beat_agile = "YES" if r > agile_r else "no"
        print(f"Beat LiON?   {beat_lion}")
        print(f"Beat AGILE?  {beat_agile}")

        results.append({
            "study": study,
            "n_train": len(train_y),
            "n_test": len(test_y),
            "spearman_r": round(r, 4),
            "spearman_r_std": round(r_std, 4),
            "r2": round(r2, 4),
            "lion_r": lion_r,
            "agile_r": agile_r,
            "beat_lion": r > lion_r,
            "beat_agile": r > agile_r,
        })

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    print(f"\nMean Spearman r:  {df['spearman_r'].mean():.4f}")
    print(f"Mean LiON r:      {df['lion_r'].mean():.4f}")
    print(f"Mean AGILE r:     {df['agile_r'].mean():.4f}")
    print(f"Beat LiON:        {df['beat_lion'].sum()}/4 studies")
    print(f"Beat AGILE:       {df['beat_agile'].sum()}/4 studies")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark on LNPDB held-out studies", suggest_on_error=True)
    parser.add_argument("--with-aux", action="store_true", help="Include LiON auxiliary features")
    parser.add_argument("--reduction", default="pca", choices=["pca", "pls", "none"])
    parser.add_argument("--n-pcs-count-mfp", type=int, default=5)
    parser.add_argument("--n-pcs-rdkit", type=int, default=5)
    parser.add_argument("--cv-folds", type=int, default=1,
                        help="Number of CV folds (1=single model, 5=match paper protocol)")
    args = parser.parse_args()

    run_benchmark(
        with_aux=args.with_aux,
        reduction=args.reduction,
        n_pcs_count_mfp=args.n_pcs_count_mfp,
        n_pcs_rdkit=args.n_pcs_rdkit,
        cv_folds=args.cv_folds,
    )
