#!/usr/bin/env python3
"""Cross-study utility consistency test using Bradley-Terry utilities."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr, ttest_1samp

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from diagnostics.utils import encode_lantern_il, lantern_il_feature_cols, load_lnpdb_clean


def main() -> int:
    df = load_lnpdb_clean(drop_duplicates=False)
    df = df.dropna(subset=["IL_SMILES", "Experiment_value"]).reset_index(drop=True)

    study_sizes = df.groupby("study_id").size()
    keep_ids = study_sizes[study_sizes >= 5].index
    df = df[df["study_id"].isin(keep_ids)].reset_index(drop=True)

    encoded, _ = encode_lantern_il(df, reduction="pca")
    feat_cols = lantern_il_feature_cols(encoded)

    X = encoded[feat_cols].values.astype(np.float32)
    y = encoded["Experiment_value"].values.astype(np.float32)
    study_ids = df["study_id"].astype(str).values

    # Train BT on all data
    from models.bradley_terry import train_bt_model
    train_mask = np.ones(len(X), dtype=bool)
    model = train_bt_model(X, y, study_ids, train_mask, epochs=50)

    # Get utilities
    with torch.no_grad():
        u = model(torch.tensor(X)).cpu().numpy()

    # For each pair of studies sharing ILs, compute rank correlation
    study_list = np.unique(study_ids)
    pair_results = []

    for i in range(len(study_list)):
        s1 = study_list[i]
        idx1 = np.where(study_ids == s1)[0]
        il1 = df.iloc[idx1]["IL_SMILES"].values
        # Mean utility per IL in study 1
        il_u1 = {}
        for j, smi in enumerate(il1):
            il_u1.setdefault(smi, []).append(u[idx1[j]])
        il_u1 = {k: np.mean(v) for k, v in il_u1.items()}

        for j in range(i + 1, len(study_list)):
            s2 = study_list[j]
            idx2 = np.where(study_ids == s2)[0]
            il2 = df.iloc[idx2]["IL_SMILES"].values
            il_u2 = {}
            for k, smi in enumerate(il2):
                il_u2.setdefault(smi, []).append(u[idx2[k]])
            il_u2 = {k: np.mean(v) for k, v in il_u2.items()}

            shared = set(il_u1) & set(il_u2)
            if len(shared) < 3:
                continue

            v1 = np.array([il_u1[s] for s in shared])
            v2 = np.array([il_u2[s] for s in shared])
            rho, p = spearmanr(v1, v2)
            if np.isfinite(rho):
                pair_results.append({
                    "study_1": s1,
                    "study_2": s2,
                    "n_shared": len(shared),
                    "spearman_rho": float(rho),
                    "p_value": float(p),
                })

    rhos = np.array([p["spearman_rho"] for p in pair_results])
    if len(rhos) > 0:
        mean_rho = float(np.mean(rhos))
        _t_stat, p_mean = ttest_1samp(rhos, 0.0)
    else:
        mean_rho = float("nan")
        p_mean = float("nan")

    report = {
        "n_study_pairs": len(pair_results),
        "mean_spearman_rho": mean_rho,
        "median_spearman_rho": float(np.median(rhos)) if len(rhos) > 0 else float("nan"),
        "p_value_mean_ne_zero": float(p_mean),
        "pair_details": pair_results,
    }

    out_path = Path("diagnostics") / "utility_consistency.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "pair_details"}, indent=2))
    print(f"Saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
