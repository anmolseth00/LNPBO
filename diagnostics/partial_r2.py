#!/usr/bin/env python3
"""Partial R^2 decomposition: study vs chemistry."""


import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from diagnostics.utils import encode_lantern_il, lantern_il_feature_cols, load_lnpdb_clean, summarize_study_assay_types


def _study_split(df, seed=42):
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

    train_idx = df.index[df["study_id"].isin(train_ids)].tolist()
    test_idx = df.index[df["study_id"].isin(test_ids)].tolist()
    return train_idx, test_idx


def main() -> int:
    df = load_lnpdb_clean(drop_duplicates=False)
    df = df.dropna(subset=["IL_SMILES", "Experiment_value"]).reset_index(drop=True)

    train_idx, test_idx = _study_split(df, seed=42)

    train_enc, test_enc, _ = encode_lantern_il(df, train_idx=train_idx, test_idx=test_idx, reduction="pca")
    feat_cols = lantern_il_feature_cols(train_enc)

    y_train = train_enc["Experiment_value"].values
    y_test = test_enc["Experiment_value"].values

    # M0: intercept only
    y_pred_m0 = np.full_like(y_test, y_train.mean())
    r2_m0 = float(r2_score(y_test, y_pred_m0))

    # M1: study fixed effects
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_train_sid = enc.fit_transform(train_enc[["study_id"]])
    X_test_sid = enc.transform(test_enc[["study_id"]])
    lr1 = LinearRegression()
    lr1.fit(X_train_sid, y_train)
    y_pred_m1 = lr1.predict(X_test_sid)
    r2_m1 = float(r2_score(y_test, y_pred_m1))

    # M2: study FE + chemistry
    X_train = np.hstack([X_train_sid, train_enc[feat_cols].values])
    X_test = np.hstack([X_test_sid, test_enc[feat_cols].values])
    lr2 = LinearRegression()
    lr2.fit(X_train, y_train)
    y_pred_m2 = lr2.predict(X_test)
    r2_m2 = float(r2_score(y_test, y_pred_m2))

    r2_study = (r2_m1 - r2_m0) / max(1 - r2_m0, 1e-12)
    r2_chem = (r2_m2 - r2_m1) / max(1 - r2_m1, 1e-12)

    report = {
        "r2_m0": r2_m0,
        "r2_m1": r2_m1,
        "r2_m2": r2_m2,
        "r2_study": r2_study,
        "r2_chemistry": r2_chem,
        "ratio_study_to_chem": float(r2_study / r2_chem) if r2_chem != 0 else float("inf"),
    }

    out_path = Path("diagnostics") / "partial_r2.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"Saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
