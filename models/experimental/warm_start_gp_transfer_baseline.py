#!/usr/bin/env python3
"""Warm-started GP transfer baseline for few-shot BO.

This module is an approximate transfer-learning baseline. It does not implement
the exact FSBO method from Wistuba & Grabocka (2021); it only warm-starts
standard GP hyperparameters from source-task data and then performs greedy
posterior-mean search on the target task.
"""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path

import numpy as np
import torch

from LNPBO.data.study_utils import (
    build_study_type_map,
    encode_lantern_il,
    lantern_il_feature_cols,
    load_lnpdb_clean,
    study_split,
)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger("lnpbo")


def _normalize_X(X, bounds=None):
    """Min-max normalize features to [0, 1]. Returns normalized X and bounds."""
    X = np.asarray(X, dtype=np.float64)
    if bounds is None:
        x_min = X.min(axis=0)
        x_max = X.max(axis=0)
        rng = x_max - x_min
        rng[rng < 1e-8] = 1.0
        bounds = (x_min, rng)
    x_min, rng = bounds
    return (X - x_min) / rng, bounds


def _prepare_meta_train_subset(
    X_train,
    y_train,
    study_ids_train,
    train_study_ids,
    max_meta_n=2000,
    seed=42,
    bounds=None,
):
    """Stratify the meta-training subset and normalize it on the deployment scale."""
    rng = np.random.RandomState(seed)

    study_indices = {}
    for sid in sorted(train_study_ids):
        mask = study_ids_train == sid
        study_indices[sid] = np.where(mask)[0]
    total_train = sum(len(v) for v in study_indices.values())

    if total_train <= max_meta_n:
        sub_idx = np.arange(len(X_train))
    else:
        sub_idx = []
        for sid in sorted(train_study_ids):
            sidx = study_indices[sid]
            n_draw = max(5, int(max_meta_n * len(sidx) / total_train))
            n_draw = min(n_draw, len(sidx))
            sub_idx.extend(rng.choice(sidx, size=n_draw, replace=False))
        sub_idx = np.array(sub_idx)

    X_sub = X_train[sub_idx]
    y_sub = y_train[sub_idx]
    X_sub_norm, bounds_used = _normalize_X(X_sub, bounds=bounds)
    return X_sub_norm, y_sub, bounds_used, sub_idx, total_train, study_indices


def meta_train_gp(X_train, y_train, study_ids_train, train_study_ids, max_meta_n=2000, seed=42, bounds=None):
    """Fit a GP on a stratified meta-training subset."""
    import botorch.fit
    import botorch.models
    import gpytorch

    X_sub_norm, y_sub, bounds, sub_idx, total_train, study_indices = _prepare_meta_train_subset(
        X_train,
        y_train,
        study_ids_train,
        train_study_ids,
        max_meta_n=max_meta_n,
        seed=seed,
        bounds=bounds,
    )

    logger.info("  Meta-training baseline GP on %d / %d points (from %d studies)", len(sub_idx), total_train, len(study_indices))

    X_t = torch.tensor(X_sub_norm, dtype=torch.float64)
    y_t = torch.tensor(y_sub, dtype=torch.float64)
    model = botorch.models.SingleTaskGP(X_t, y_t.unsqueeze(-1))
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
    botorch.fit.fit_gpytorch_mll(mll)

    model.eval()
    model.likelihood.eval()
    meta_params = {
        "lengthscales": model.covar_module.lengthscale.detach().clone(),
        "noise": model.likelihood.noise.detach().clone(),
        "mean_constant": model.mean_module.constant.detach().clone(),
    }
    if hasattr(model.covar_module, "outputscale"):
        meta_params["outputscale"] = model.covar_module.outputscale.detach().clone()
    return meta_params, bounds


def _build_warm_gp(X_obs, y_obs, meta_params):
    import botorch.fit
    import botorch.models
    import gpytorch

    X_t = torch.tensor(X_obs, dtype=torch.float64)
    y_t = torch.tensor(y_obs, dtype=torch.float64).unsqueeze(-1)
    model = botorch.models.SingleTaskGP(X_t, y_t)
    model.covar_module.lengthscale = meta_params["lengthscales"].clone()
    model.likelihood.noise = meta_params["noise"].clone()
    model.mean_module.constant = meta_params["mean_constant"].clone()
    if "outputscale" in meta_params and hasattr(model.covar_module, "outputscale"):
        model.covar_module.outputscale = meta_params["outputscale"].clone()
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
    botorch.fit.fit_gpytorch_mll(mll)
    model.eval()
    model.likelihood.eval()
    return model


def _build_cold_gp(X_obs, y_obs):
    import botorch.fit
    import botorch.models
    import gpytorch

    X_t = torch.tensor(X_obs, dtype=torch.float64)
    y_t = torch.tensor(y_obs, dtype=torch.float64).unsqueeze(-1)
    model = botorch.models.SingleTaskGP(X_t, y_t)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
    botorch.fit.fit_gpytorch_mll(mll)
    model.eval()
    model.likelihood.eval()
    return model


def _run_bo_loop(X, y, study_ids, test_study_ids, k_shots, build_model_fn, n_bo_rounds=10, batch_size=6):
    results_by_k = {}
    for k in k_shots:
        study_results = []
        for sid in sorted(test_study_ids):
            idx = np.where(study_ids == sid)[0]
            if len(idx) < k + 10:
                continue
            rng = np.random.RandomState(42)
            perm = rng.permutation(len(idx))
            support_idx = list(idx[perm[:k]])
            pool_idx = list(idx[perm[k:]])
            observed_idx = list(support_idx)
            for _ in range(n_bo_rounds):
                if len(pool_idx) < batch_size:
                    break
                model = build_model_fn(X[observed_idx], y[observed_idx])
                X_pool = torch.tensor(X[pool_idx], dtype=torch.float64)
                with torch.no_grad():
                    means = model.posterior(X_pool).mean.squeeze(-1).numpy()
                top_batch = np.argsort(means)[-batch_size:]
                selected = [pool_idx[i] for i in top_batch]
                observed_idx.extend(selected)
                for s in sorted(selected, reverse=True):
                    pool_idx.remove(s)

            all_vals = y[idx]
            observed_vals = y[observed_idx]
            study_top10 = set(idx[np.argsort(all_vals)[-10:]])
            study_top50 = set(idx[np.argsort(all_vals)[-min(50, len(idx)):]])
            observed_set = set(observed_idx)
            study_results.append(
                {
                    "study_id": str(sid),
                    "n_in_study": len(idx),
                    "k_shots": k,
                    "n_observed": len(observed_idx),
                    "top10_recall": float(len(observed_set & study_top10) / len(study_top10)),
                    "top50_recall": float(len(observed_set & study_top50) / len(study_top50)) if study_top50 else 0.0,
                    "best_found": float(observed_vals.max()),
                    "study_max": float(all_vals.max()),
                }
            )

        if study_results:
            results_by_k[k] = {
                "mean_top10_recall": float(np.mean([r["top10_recall"] for r in study_results])),
                "mean_top50_recall": float(np.mean([r["top50_recall"] for r in study_results])),
                "n_studies_evaluated": len(study_results),
                "per_study": study_results,
            }
    return results_by_k


def random_baseline(X, y, study_ids, test_study_ids, k_shots, n_bo_rounds=10, batch_size=6):
    del X
    results_by_k = {}
    for k in k_shots:
        study_results = []
        for sid in sorted(test_study_ids):
            idx = np.where(study_ids == sid)[0]
            if len(idx) < k + 10:
                continue
            rng = np.random.RandomState(42)
            perm = rng.permutation(len(idx))
            n_total = min(k + n_bo_rounds * batch_size, len(idx))
            observed_idx = list(idx[perm[:n_total]])
            all_vals = y[idx]
            study_top10 = set(idx[np.argsort(all_vals)[-10:]])
            study_top50 = set(idx[np.argsort(all_vals)[-min(50, len(idx)):]])
            observed_set = set(observed_idx)
            study_results.append(
                {
                    "top10_recall": float(len(observed_set & study_top10) / len(study_top10)),
                    "top50_recall": float(len(observed_set & study_top50) / len(study_top50)) if study_top50 else 0.0,
                }
            )
        if study_results:
            results_by_k[k] = {
                "mean_top10_recall": float(np.mean([r["top10_recall"] for r in study_results])),
                "mean_top50_recall": float(np.mean([r["top50_recall"] for r in study_results])),
                "n_studies_evaluated": len(study_results),
            }
    return results_by_k


def main() -> int:
    df = load_lnpdb_clean(drop_duplicates=False)
    df = df.dropna(subset=["IL_SMILES", "Experiment_value"]).reset_index(drop=True)
    study_sizes = df.groupby("study_id").size()
    keep_ids = study_sizes[study_sizes >= 25].index
    df = df[df["study_id"].isin(keep_ids)].reset_index(drop=True)
    study_ids = df["study_id"].astype(str).values
    study_to_type = build_study_type_map(df)
    train_ids, test_ids = study_split(np.unique(study_ids), study_to_type, seed=42)
    train_idx = [i for i, sid in enumerate(study_ids) if sid in train_ids]
    test_idx = [i for i, sid in enumerate(study_ids) if sid in test_ids]

    train_encoded, test_encoded, _fitted = encode_lantern_il(df, train_idx=train_idx, test_idx=test_idx, reduction="pca")
    feat_cols = lantern_il_feature_cols(train_encoded)
    import pandas as pd

    train_encoded.index = train_idx
    test_encoded.index = test_idx
    encoded = pd.concat([train_encoded, test_encoded]).sort_index()
    X_raw = encoded[feat_cols].values.astype(np.float64)
    y = encoded["Experiment_value"].values.astype(np.float64)
    train_mask = np.isin(study_ids, list(train_ids))
    _, norm_bounds = _normalize_X(X_raw[train_mask])
    X, _ = _normalize_X(X_raw, bounds=norm_bounds)
    k_shots = [5, 10, 20]

    meta_params, _ = meta_train_gp(
        X_raw[train_mask],
        y[train_mask],
        study_ids[train_mask],
        train_ids,
        max_meta_n=2000,
        bounds=norm_bounds,
    )
    warm_results = _run_bo_loop(X, y, study_ids, test_ids, k_shots, lambda Xo, yo: _build_warm_gp(Xo, yo, meta_params))
    cold_results = _run_bo_loop(X, y, study_ids, test_ids, k_shots, _build_cold_gp)
    report = {
        "baseline_name": "warm_start_gp_transfer_baseline",
        "warm_start_gp_transfer_baseline": {k: {kk: vv for kk, vv in v.items() if kk != "per_study"} for k, v in warm_results.items()},
        "cold_start_gp": {k: {kk: vv for kk, vv in v.items() if kk != "per_study"} for k, v in cold_results.items()},
    }
    out_path = Path("models") / "warm_start_gp_transfer_baseline_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    logger.info("Saved %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
