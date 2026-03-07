#!/usr/bin/env python3
"""Phase 5.2: Simplified FSBO surrogate for few-shot BO.

Implements a simplified version of Few-Shot Bayesian Optimization
(Wistuba & Grabocka, ICLR 2021, arXiv:2101.07667) using warm-started
GP hyperparameters from meta-training data.

Architecture:
  1. Meta-train: fit a GP on a stratified subsample of training study
     data to learn prior kernel hyperparameters (lengthscales,
     outputscale, noise). Subsampling is required because exact GP
     inference is O(n^3).
  2. For each held-out study: initialize GP with meta-learned
     hyperparameters, condition on k observations, run greedy BO
     (pick top-batch_size by posterior mean each round, refit GP).

Uses the same study split as MAML (Phase 5.1):
  - Filter to studies with >= 25 formulations
  - 80/20 study-level split, stratified by assay type

Reference: Wistuba, M. & Grabocka, J. "Few-Shot Bayesian Optimization
with Deep Kernel Surrogates", ICLR 2021.
"""


import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from diagnostics.utils import (
    encode_lantern_il,
    lantern_il_feature_cols,
    load_lnpdb_clean,
    summarize_study_assay_types,
)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def _study_split(study_ids, study_to_type, seed=42):
    rng = np.random.RandomState(seed)
    train_ids, test_ids = set(), set()
    for assay_type in sorted(set(study_to_type.values())):
        ids = [sid for sid, at in study_to_type.items() if at == assay_type]
        rng.shuffle(ids)
        cut = max(1, int(0.8 * len(ids))) if len(ids) > 1 else len(ids)
        train_ids.update(ids[:cut])
        test_ids.update(ids[cut:])
    return train_ids, test_ids


def _normalize_X(X, bounds=None):
    """Min-max normalize features to [0, 1]. Returns normalized X and bounds."""
    if bounds is None:
        x_min = X.min(axis=0)
        x_max = X.max(axis=0)
        rng = x_max - x_min
        rng[rng < 1e-8] = 1.0
        bounds = (x_min, rng)
    x_min, rng = bounds
    return (X - x_min) / rng, bounds


def meta_train_gp(X_train, y_train, study_ids_train, train_study_ids,
                  max_meta_n=2000, seed=42):
    """Fit a GP on a stratified subsample of meta-training data.

    Stratified sampling: draw up to max_meta_n points total, sampling
    proportionally from each training study so that all studies are
    represented. GP is O(n^3), so we cap at max_meta_n.

    Uses botorch.fit.fit_gpytorch_mll for constrained optimization
    (respects positivity constraints on kernel hyperparameters).

    Returns a dict of meta-learned hyperparameters and the normalization
    bounds used.
    """
    import botorch.fit
    import botorch.models
    import gpytorch

    rng = np.random.RandomState(seed)

    # Stratified subsample across training studies
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

    print(f"  Meta-training on {len(sub_idx)} / {total_train} points "
          f"(from {len(study_indices)} studies)")

    X_sub = X_train[sub_idx]
    y_sub = y_train[sub_idx]

    X_sub_norm, bounds = _normalize_X(X_sub)

    X_t = torch.tensor(X_sub_norm, dtype=torch.float64)
    y_t = torch.tensor(y_sub, dtype=torch.float64)

    model = botorch.models.SingleTaskGP(X_t, y_t.unsqueeze(-1))
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)

    print("  Fitting meta-GP via L-BFGS-B ...")
    botorch.fit.fit_gpytorch_mll(mll)

    model.eval()
    model.likelihood.eval()

    meta_params = {
        "lengthscales": model.covar_module.lengthscale.detach().clone(),
        "noise": model.likelihood.noise.detach().clone(),
        "mean_constant": model.mean_module.constant.detach().clone(),
    }
    # ScaleKernel may or may not wrap the RBF depending on botorch version
    if hasattr(model.covar_module, "outputscale"):
        meta_params["outputscale"] = model.covar_module.outputscale.detach().clone()

    ls = meta_params["lengthscales"].squeeze().numpy()
    print(f"  Meta-learned lengthscales: mean={ls.mean():.3f}, "
          f"min={ls.min():.3f}, max={ls.max():.3f}")
    if "outputscale" in meta_params:
        print(f"  Meta-learned outputscale: {meta_params['outputscale'].item():.4f}")
    print(f"  Meta-learned noise: {meta_params['noise'].item():.6f}")
    print(f"  Meta-learned mean: {meta_params['mean_constant'].item():.4f}")

    return meta_params, bounds


def _build_warm_gp(X_obs, y_obs, meta_params):
    """Build a SingleTaskGP initialized with meta-learned hyperparameters."""
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
    """Build a SingleTaskGP with default (uninformed) hyperparameters."""
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


def _run_bo_loop(X, y, study_ids, test_study_ids, k_shots,
                 build_model_fn, n_bo_rounds=10, batch_size=6, label=""):
    """Generic BO evaluation loop. build_model_fn(X_obs, y_obs) -> model."""
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
                    posterior = model.posterior(X_pool)
                    means = posterior.mean.squeeze(-1).numpy()

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

            r10 = len(observed_set & study_top10) / len(study_top10)
            r50 = len(observed_set & study_top50) / len(study_top50) if study_top50 else 0

            study_results.append({
                "study_id": str(sid),
                "n_in_study": len(idx),
                "k_shots": k,
                "n_observed": len(observed_idx),
                "top10_recall": float(r10),
                "top50_recall": float(r50),
                "best_found": float(observed_vals.max()),
                "study_max": float(all_vals.max()),
            })

        if study_results:
            results_by_k[k] = {
                "mean_top10_recall": float(np.mean([r["top10_recall"] for r in study_results])),
                "mean_top50_recall": float(np.mean([r["top50_recall"] for r in study_results])),
                "n_studies_evaluated": len(study_results),
                "per_study": study_results,
            }
            print(f"  k={k}: top10={results_by_k[k]['mean_top10_recall']:.1%} "
                  f"top50={results_by_k[k]['mean_top50_recall']:.1%} "
                  f"({len(study_results)} studies)")

    return results_by_k


def random_baseline(X, y, study_ids, test_study_ids, k_shots,
                    n_bo_rounds=10, batch_size=6):
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

            r10 = len(observed_set & study_top10) / len(study_top10)
            r50 = len(observed_set & study_top50) / len(study_top50) if study_top50 else 0
            study_results.append({"top10_recall": float(r10), "top50_recall": float(r50)})

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
    print(f"Using {len(df)} rows from {df['study_id'].nunique()} studies (>=25 per study)")

    study_ids = df["study_id"].astype(str).values

    study_to_type = {}
    for sid, sdf in df.groupby("study_id"):
        assay_type, _ = summarize_study_assay_types(sdf)
        study_to_type[str(sid)] = assay_type

    train_ids, test_ids = _study_split(np.unique(study_ids), study_to_type, seed=42)

    train_idx = [i for i, sid in enumerate(study_ids) if sid in train_ids]
    test_idx = [i for i, sid in enumerate(study_ids) if sid in test_ids]

    train_encoded, test_encoded, _fitted = encode_lantern_il(
        df, train_idx=train_idx, test_idx=test_idx, reduction="pca",
    )
    feat_cols = lantern_il_feature_cols(train_encoded)

    import pandas as pd
    encoded = pd.concat([train_encoded, test_encoded]).sort_index()

    X_raw = encoded[feat_cols].values.astype(np.float64)
    y = encoded["Experiment_value"].values.astype(np.float64)
    print(f"Train studies: {len(train_ids)}, Test studies: {len(test_ids)}")

    k_shots = [5, 10, 20]

    # Normalize features to [0, 1] using training bounds
    train_mask = np.isin(study_ids, list(train_ids))
    _, norm_bounds = _normalize_X(X_raw[train_mask])
    X, _ = _normalize_X(X_raw, bounds=norm_bounds)

    # Meta-train: fit GP on stratified subsample to get prior hyperparameters
    print("\n=== FSBO Meta-Training (GP hyperparameter extraction) ===")
    meta_params, _ = meta_train_gp(
        X[train_mask], y[train_mask], study_ids[train_mask],
        train_ids, max_meta_n=2000,
    )

    # FSBO few-shot evaluation (warm-started GP)
    print("\n=== FSBO Few-Shot BO (warm-started GP) ===")
    fsbo_results = _run_bo_loop(
        X, y, study_ids, test_ids, k_shots,
        build_model_fn=lambda Xo, yo: _build_warm_gp(Xo, yo, meta_params),
        label="FSBO",
    )

    # Cold-start GP baseline (no meta-learning)
    print("\n=== Cold-Start GP Baseline ===")
    cold_results = _run_bo_loop(
        X, y, study_ids, test_ids, k_shots,
        build_model_fn=_build_cold_gp,
        label="Cold-Start",
    )

    # Random baseline
    print("\n=== Random Baseline ===")
    random_results = random_baseline(X, y, study_ids, test_ids, k_shots)
    for k, r in random_results.items():
        print(f"  k={k}: top10={r['mean_top10_recall']:.1%} top50={r['mean_top50_recall']:.1%}")

    # Load MAML results for comparison if available
    maml_path = Path("models") / "maml_results.json"
    maml_summary = None
    if maml_path.exists():
        with open(maml_path) as f:
            maml_data = json.load(f)
        maml_summary = maml_data.get("maml", {})

    report = {
        "n_train_studies": len(train_ids),
        "n_test_studies": len(test_ids),
        "n_rows": len(X),
        "n_features": X.shape[1],
        "meta_params": {
            "lengthscales_mean": float(meta_params["lengthscales"].mean()),
            "lengthscales_std": float(meta_params["lengthscales"].std()),
            "noise": float(meta_params["noise"].item()),
            "mean_constant": float(meta_params["mean_constant"].item()),
            **({"outputscale": float(meta_params["outputscale"].item())} if "outputscale" in meta_params else {}),
        },
        "fsbo": {k: {kk: vv for kk, vv in v.items() if kk != "per_study"} for k, v in fsbo_results.items()},
        "cold_start_gp": {k: {kk: vv for kk, vv in v.items() if kk != "per_study"} for k, v in cold_results.items()},
        "random": random_results,
        "fsbo_per_study": {k: v.get("per_study", []) for k, v in fsbo_results.items()},
        "cold_start_per_study": {k: v.get("per_study", []) for k, v in cold_results.items()},
    }
    if maml_summary:
        report["maml_reference"] = maml_summary

    out_path = Path("models") / "fsbo_results.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nSaved {out_path}")

    # Summary table
    print("\n=== Summary ===")
    print(f"{'Method':<20} {'k':>3} {'Top-10':>8} {'Top-50':>8}")
    print("-" * 45)
    for k in k_shots:
        methods = [
            ("Random", random_results),
            ("Cold-Start GP", cold_results),
            ("FSBO (warm GP)", fsbo_results),
        ]
        if maml_summary:
            methods.append(("MAML (ref)", maml_summary))
        for name, res in methods:
            if k in res or str(k) in res:
                r = res.get(k) or res.get(str(k))
                t10 = r.get("mean_top10_recall", 0)
                t50 = r.get("mean_top50_recall", 0)
                print(f"{name:<20} {k:>3} {t10:>7.1%} {t50:>7.1%}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
