#!/usr/bin/env python3
"""Permutation decomposition: chemistry vs study identity."""


import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from diagnostics.utils import encode_lantern_il, lantern_il_feature_cols, load_lnpdb_clean, summarize_study_assay_types
from models.splits import scaffold_split


def _study_split(df, seed=42):
    rng = np.random.RandomState(seed)

    # Stratify by assay_type
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

    train_idx = df.index[df["study_id"].isin(train_ids)].tolist()
    test_idx = df.index[df["study_id"].isin(test_ids)].tolist()
    return train_idx, test_idx


def _fit_xgb(X_train, y_train, seed=42):
    model = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=seed,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X_train, y_train)
    return model


def _shuffle_il_smiles(df, seed=42):
    rng = np.random.RandomState(seed)
    df = df.copy()
    for _, idx in df.groupby("study_id").indices.items():
        smi = df.loc[idx, "IL_SMILES"].astype(str).to_numpy()
        rng.shuffle(smi)
        df.loc[idx, "IL_SMILES"] = smi
    return df


def evaluate_split(df, split_name, train_idx, test_idx, seed=42):
    results = {}

    # Model 1: chemistry (LANTERN IL-only)
    train_enc, test_enc, _ = encode_lantern_il(df, train_idx=train_idx, test_idx=test_idx, reduction="pca")
    feat_cols = lantern_il_feature_cols(train_enc)

    X_train = train_enc[feat_cols].values
    y_train = train_enc["Experiment_value"].values
    X_test = test_enc[feat_cols].values
    y_test = test_enc["Experiment_value"].values

    model = _fit_xgb(X_train, y_train, seed=seed)
    preds = model.predict(X_test)
    results["chemistry"] = float(r2_score(y_test, preds))

    # Model 2: study-id-only
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_train_sid = enc.fit_transform(df.loc[train_idx, ["study_id"]])
    X_test_sid = enc.transform(df.loc[test_idx, ["study_id"]])
    model_sid = _fit_xgb(X_train_sid, y_train, seed=seed)
    preds_sid = model_sid.predict(X_test_sid)
    results["study_only"] = float(r2_score(y_test, preds_sid))

    # Model 3: shuffled IL SMILES within each study
    df_shuffled = _shuffle_il_smiles(df, seed=seed)
    train_s, test_s, _ = encode_lantern_il(df_shuffled, train_idx=train_idx, test_idx=test_idx, reduction="pca")
    X_train_s = train_s[feat_cols].values
    y_train_s = train_s["Experiment_value"].values
    X_test_s = test_s[feat_cols].values
    y_test_s = test_s["Experiment_value"].values

    model_s = _fit_xgb(X_train_s, y_train_s, seed=seed)
    preds_s = model_s.predict(X_test_s)
    results["shuffled_il"] = float(r2_score(y_test_s, preds_s))

    return {"split": split_name, **results}


def main() -> int:
    df = load_lnpdb_clean(drop_duplicates=False)
    df = df.dropna(subset=["IL_SMILES", "Experiment_value"]).reset_index(drop=True)

    # Formulation-level scaffold split
    il_smiles = df["IL_SMILES"].tolist()
    train_idx, val_idx, test_idx = scaffold_split(il_smiles, sizes=(0.8, 0.1, 0.1), seed=42)
    train_idx = sorted(set(train_idx + val_idx))

    results = []
    results.append(evaluate_split(df, "formulation_scaffold", train_idx, test_idx, seed=42))

    # Study-level split
    train_idx_s, test_idx_s = _study_split(df, seed=42)
    results.append(evaluate_split(df, "study_level", train_idx_s, test_idx_s, seed=42))

    out_path = Path("diagnostics") / "permutation_decomposition.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"Saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
