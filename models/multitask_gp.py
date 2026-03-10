#!/usr/bin/env python3
"""Multi-Task GP treating each study as a task, using BoTorch's MultiTaskGP with
low-rank ICM (Intrinsic Coregionalization Model).

Multi-task GP models between-study correlations via a PSD task covariance matrix B:
    k((x, s_i), (x', s_j)) = B_ij * k_base(x, x')

This lets the GP learn that some studies are more informative for predicting others,
addressing LNPDB's ICC=0.006 and negative cross-study R^2.

Note: MultiTaskGP is ExactGP-based, so O(n^3). Training data is subsampled to
N_MAX_TRAIN via stratified sampling across studies to keep all tasks represented.

Citation:
    Bonilla, E.V., Chai, K.M.A., & Williams, C.K.I. (2007).
    "Multi-task Gaussian Process Prediction." NIPS 2007.

    Alvarez, M.A., Rosasco, L., & Lawrence, N.D. (2012).
    "Kernels for Vector-Valued Functions: A Review."
    Foundations and Trends in Machine Learning, 4(3), 195-266.
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

from LNPBO.diagnostics.utils import (
    build_study_type_map,
    load_lnpdb_clean,
    prepare_study_data,
    study_split,
)


SEEDS = [42, 123, 456, 789, 2024]
N_MAX_TRAIN = 1000


def _prepare_multitask_data(seed: int = 42, min_study_n: int = 5):
    """Load and prepare data for multi-task GP.

    Returns feature matrix, targets, study integer indices, task-to-study mapping,
    train/test masks, and assay-type mapping.
    """
    X, y, study_ids, train_mask_default, test_mask_default = prepare_study_data(
        min_n=min_study_n, reduction="pca"
    )

    df = load_lnpdb_clean(drop_duplicates=False)
    df = df.dropna(subset=["IL_SMILES", "Experiment_value"]).reset_index(drop=True)
    study_sizes = df.groupby("study_id").size()
    keep_ids = study_sizes[study_sizes >= min_study_n].index
    df = df[df["study_id"].isin(keep_ids)].reset_index(drop=True)
    study_to_type = build_study_type_map(df)

    unique_studies = sorted(set(study_ids))
    study_to_int = {s: i for i, s in enumerate(unique_studies)}
    task_indices = np.array([study_to_int[s] for s in study_ids], dtype=np.int64)

    # Re-split with given seed
    all_study_ids = df["study_id"].unique()
    train_studies, test_studies = study_split(
        all_study_ids, study_to_type, seed=seed
    )
    train_mask = np.array([s in train_studies for s in study_ids])
    test_mask = np.array([s in test_studies for s in study_ids])

    return (
        X, y, task_indices, study_to_int, train_mask, test_mask,
        study_to_type, unique_studies,
    )


def _subsample_train(train_indices, task_indices, n_max, seed):
    """Stratified subsample of training data preserving all tasks."""
    rng = np.random.RandomState(seed)
    train_tasks = task_indices[train_indices]
    unique_tasks = sorted(set(train_tasks))

    if len(train_indices) <= n_max:
        return train_indices

    # Allocate proportionally, min 1 per task
    per_task = {}
    for t in unique_tasks:
        per_task[t] = np.where(train_tasks == t)[0]

    n_tasks = len(unique_tasks)
    per_task_budget = max(1, n_max // n_tasks)
    selected = []
    remaining_budget = n_max

    for t in unique_tasks:
        task_idx = per_task[t]
        n_take = min(len(task_idx), per_task_budget, remaining_budget)
        if n_take > 0:
            chosen = rng.choice(task_idx, size=n_take, replace=False)
            selected.extend(train_indices[chosen])
            remaining_budget -= n_take
        if remaining_budget <= 0:
            break

    # If budget remains, fill from remaining indices
    if remaining_budget > 0:
        selected_set = set(selected)
        remaining = [i for i in train_indices if i not in selected_set]
        if remaining:
            extra = rng.choice(remaining, size=min(remaining_budget, len(remaining)), replace=False)
            selected.extend(extra)

    return np.array(sorted(selected))


def _run_multitask_gp(data, seed: int = 42, rank: int = 3):
    """Fit MultiTaskGP with low-rank ICM and evaluate on held-out studies."""
    import gpytorch
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import MultiTaskGP
    from gpytorch.mlls import ExactMarginalLogLikelihood

    X, y, task_indices, study_to_int, train_mask, test_mask, study_to_type, unique_studies = data

    train_indices = np.where(train_mask)[0]
    test_indices = np.where(test_mask)[0]

    # Subsample training data for ExactGP tractability
    train_indices = _subsample_train(train_indices, task_indices, N_MAX_TRAIN, seed)
    n_train = len(train_indices)
    n_test = test_mask.sum()
    n_tasks = len(study_to_int)
    d = X.shape[1]

    # Standardize features
    scaler = StandardScaler().fit(X[train_indices])
    X_s = scaler.transform(X).astype(np.float64)

    # Standardize targets using training set
    y_mean = y[train_indices].mean()
    y_std = y[train_indices].std()
    if y_std < 1e-8:
        y_std = 1.0
    y_s = ((y - y_mean) / y_std).astype(np.float64)

    # Build train_X: (n_train, d+1) with task index as last column
    X_train = np.column_stack([X_s[train_indices], task_indices[train_indices].reshape(-1, 1)])
    y_train = y_s[train_indices].reshape(-1, 1)

    train_X = torch.tensor(X_train, dtype=torch.float64)
    train_Y = torch.tensor(y_train, dtype=torch.float64)

    n_train_tasks = len(set(task_indices[train_indices]))
    effective_rank = min(rank, n_train_tasks)

    # Set noise floor to stabilize Cholesky
    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        noise_constraint=gpytorch.constraints.GreaterThan(1e-4),
    )
    likelihood.noise = 0.5

    # Register ALL tasks (including held-out test studies) so the model
    # can predict for unseen-during-training task indices via the learned
    # task covariance structure.
    all_task_ids = sorted(set(task_indices))

    torch.manual_seed(seed)
    model = MultiTaskGP(
        train_X=train_X,
        train_Y=train_Y,
        task_feature=-1,
        rank=effective_rank,
        likelihood=likelihood,
        outcome_transform=None,
        all_tasks=all_task_ids,
    )

    mll = ExactMarginalLogLikelihood(model.likelihood, model)

    # Try botorch fit first; fall back to manual Adam if it fails
    try:
        fit_gpytorch_mll(mll)
    except Exception:
        model.train()
        likelihood.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
        for _ in range(100):
            optimizer.zero_grad()
            output = model(*model.train_inputs)
            loss = -mll(output, model.train_targets)
            loss.backward()
            optimizer.step()

    model.eval()
    model.likelihood.eval()

    X_test = np.column_stack([X_s[test_indices], task_indices[test_indices].reshape(-1, 1)])
    test_X = torch.tensor(X_test, dtype=torch.float64)

    # Predict in batches to avoid memory issues with large test sets
    batch_sz = 500
    mu_parts, sigma_parts = [], []
    with torch.no_grad():
        for start in range(0, test_X.size(0), batch_sz):
            end = min(start + batch_sz, test_X.size(0))
            posterior = model.posterior(test_X[start:end])
            mu_parts.append(posterior.mean.squeeze(-1).numpy())
            sigma_parts.append(posterior.variance.squeeze(-1).sqrt().numpy())
    mu_test = np.concatenate(mu_parts)
    sigma_test = np.concatenate(sigma_parts)

    y_test = y_s[test_indices]
    r2 = float(r2_score(y_test, mu_test))

    # Per-study R^2 on test set
    per_study = {}
    test_task_ids = task_indices[test_indices]
    for tidx in sorted(set(test_task_ids)):
        mask_s = test_task_ids == tidx
        if mask_s.sum() < 2:
            continue
        y_s_study = y_test[mask_s]
        mu_s_study = mu_test[mask_s]
        r2_s = float(r2_score(y_s_study, mu_s_study)) if np.std(y_s_study) > 1e-8 else float("nan")
        sid = unique_studies[tidx]
        assay = study_to_type.get(str(sid), "unknown")
        per_study[str(sid)] = {
            "r2": r2_s,
            "n": int(mask_s.sum()),
            "assay_type": assay,
        }

    # Extract task covariance matrix B via BoTorch's MultiTaskGP API
    task_kernel = model.task_covar_module
    B = task_kernel._eval_covar_matrix().detach().numpy()
    eigvals = np.linalg.eigvalsh(B)

    task_assay_map = {}
    for sid, tidx in study_to_int.items():
        task_assay_map[tidx] = study_to_type.get(str(sid), "unknown")

    return {
        "r2": r2,
        "n_train": n_train,
        "n_test": int(n_test),
        "n_tasks": n_tasks,
        "n_train_tasks": n_train_tasks,
        "n_features": d,
        "rank": effective_rank,
        "per_study": per_study,
        "B_eigenvalues": eigvals.tolist(),
        "B_shape": list(B.shape),
        "task_assay_map": {str(k): v for k, v in task_assay_map.items()},
    }


def _run_single_task_gp_baseline(data, seed: int = 42):
    """Single-task sparse GP baseline for comparison (study-level split)."""
    from LNPBO.models.gp_surrogate import _predict, _train_sparse_gp

    X, y, task_indices, study_to_int, train_mask, test_mask, study_to_type, unique_studies = data

    scaler = StandardScaler().fit(X[train_mask])
    X_s = scaler.transform(X).astype(np.float32)

    y_mean = y[train_mask].mean()
    y_std = y[train_mask].std()
    if y_std < 1e-8:
        y_std = 1.0
    y_s = ((y - y_mean) / y_std).astype(np.float32)

    train_x = torch.tensor(X_s[train_mask], dtype=torch.float32)
    train_y = torch.tensor(y_s[train_mask], dtype=torch.float32)
    test_x = torch.tensor(X_s[test_mask], dtype=torch.float32)

    model, likelihood = _train_sparse_gp(
        train_x, train_y,
        noise_init=1.0, fix_noise=False,
        kernel_name="rbf", epochs=20, batch_size=1024,
    )

    mu, sigma = _predict(model, likelihood, test_x)
    mu = mu.numpy()

    y_test = y_s[test_mask]
    r2 = float(r2_score(y_test, mu))

    return {"r2": r2, "n_train": int(train_mask.sum()), "n_test": int(test_mask.sum())}


def _run_random_intercept(data, seed: int = 42):
    """Study-level random intercept model: per-study mean offset on top of GP.

    Simpler alternative to full multi-task GP: fits a single-task sparse GP,
    then adds a per-study random intercept estimated from training residuals.
    At test time, studies seen in training use their estimated intercept;
    unseen studies use intercept=0.
    """
    from LNPBO.models.gp_surrogate import _predict, _train_sparse_gp

    X, y, task_indices, study_to_int, train_mask, test_mask, study_to_type, unique_studies = data

    scaler = StandardScaler().fit(X[train_mask])
    X_s = scaler.transform(X).astype(np.float32)

    y_mean = y[train_mask].mean()
    y_std = y[train_mask].std()
    if y_std < 1e-8:
        y_std = 1.0
    y_s = ((y - y_mean) / y_std).astype(np.float32)

    train_x = torch.tensor(X_s[train_mask], dtype=torch.float32)
    train_y = torch.tensor(y_s[train_mask], dtype=torch.float32)
    test_x = torch.tensor(X_s[test_mask], dtype=torch.float32)

    model, likelihood = _train_sparse_gp(
        train_x, train_y,
        noise_init=1.0, fix_noise=False,
        kernel_name="rbf", epochs=20, batch_size=1024,
    )

    mu_train, _ = _predict(model, likelihood, train_x)
    mu_train = mu_train.numpy()
    residuals = y_s[train_mask] - mu_train

    train_tasks = task_indices[train_mask]
    study_intercepts = {}
    for tidx in sorted(set(train_tasks)):
        mask_t = train_tasks == tidx
        study_intercepts[tidx] = float(residuals[mask_t].mean())

    mu_test, sigma_test = _predict(model, likelihood, test_x)
    mu_test = mu_test.numpy()

    test_tasks = task_indices[test_mask]
    for i, tidx in enumerate(test_tasks):
        mu_test[i] += study_intercepts.get(tidx, 0.0)

    y_test = y_s[test_mask]
    r2 = float(r2_score(y_test, mu_test))

    test_study_set = set(test_tasks)
    n_seen = sum(1 for t in test_study_set if t in study_intercepts)
    n_unseen = len(test_study_set) - n_seen

    return {
        "r2": r2,
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "n_seen_studies": n_seen,
        "n_unseen_studies": n_unseen,
        "mean_abs_intercept": float(np.mean([abs(v) for v in study_intercepts.values()])),
    }


def main() -> int:
    results = {
        "description": (
            "Multi-task GP with ICM kernel treating each study as a task. "
            "Compared against single-task GP and random intercept baselines. "
            "Study-level 80/20 split stratified by assay type."
        ),
        "citation": (
            "Bonilla et al. (2007) 'Multi-task Gaussian Process Prediction', NIPS 2007. "
            "Alvarez et al. (2012) 'Kernels for Vector-Valued Functions: A Review'."
        ),
        "note": (
            "MultiTaskGP is ExactGP (O(n^3)); training subsampled to "
            f"n_max_train={N_MAX_TRAIN} via stratified sampling across studies. "
            "Single-task GP and random intercept use SVGP (no size limit)."
        ),
        "seeds": SEEDS,
        "multitask_gp": {},
        "single_task_gp": {},
        "random_intercept": {},
    }

    mt_r2s = []
    st_r2s = []
    ri_r2s = []

    for seed in SEEDS:
        print(f"\n{'='*60}")
        print(f"Seed {seed}")
        print(f"{'='*60}")

        # Load data once per seed
        data = _prepare_multitask_data(seed=seed)
        n_total = len(data[0])
        n_train = data[4].sum()
        n_test = data[5].sum()
        print(f"  Data: {n_total} total, {n_train} train, {n_test} test")

        # Multi-task GP (rank=3)
        print("  Fitting MultiTaskGP (rank=3)...")
        t0 = time.time()
        try:
            mt_result = _run_multitask_gp(data, seed=seed, rank=3)
            mt_time = time.time() - t0
            mt_result["time_s"] = round(mt_time, 1)
            results["multitask_gp"][str(seed)] = mt_result
            mt_r2s.append(mt_result["r2"])
            nt = mt_result.get("n_train", "?")
            print(f"    R^2 = {mt_result['r2']:.4f} (n_train={nt}, {mt_time:.1f}s)")
        except Exception as e:
            print(f"    FAILED: {e}")
            import traceback
            traceback.print_exc()
            results["multitask_gp"][str(seed)] = {"error": str(e)}

        # Single-task GP baseline
        print("  Fitting single-task sparse GP baseline...")
        t0 = time.time()
        try:
            st_result = _run_single_task_gp_baseline(data, seed=seed)
            st_time = time.time() - t0
            st_result["time_s"] = round(st_time, 1)
            results["single_task_gp"][str(seed)] = st_result
            st_r2s.append(st_result["r2"])
            print(f"    R^2 = {st_result['r2']:.4f} ({st_time:.1f}s)")
        except Exception as e:
            print(f"    FAILED: {e}")
            results["single_task_gp"][str(seed)] = {"error": str(e)}

        # Random intercept
        print("  Fitting random intercept model...")
        t0 = time.time()
        try:
            ri_result = _run_random_intercept(data, seed=seed)
            ri_time = time.time() - t0
            ri_result["time_s"] = round(ri_time, 1)
            results["random_intercept"][str(seed)] = ri_result
            ri_r2s.append(ri_result["r2"])
            print(f"    R^2 = {ri_result['r2']:.4f} ({ri_time:.1f}s)")
        except Exception as e:
            print(f"    FAILED: {e}")
            results["random_intercept"][str(seed)] = {"error": str(e)}

    # Summary
    summary = {}
    if mt_r2s:
        summary["multitask_gp"] = {
            "r2_mean": round(float(np.mean(mt_r2s)), 4),
            "r2_std": round(float(np.std(mt_r2s)), 4),
            "r2_values": [round(v, 4) for v in mt_r2s],
        }
    if st_r2s:
        summary["single_task_gp"] = {
            "r2_mean": round(float(np.mean(st_r2s)), 4),
            "r2_std": round(float(np.std(st_r2s)), 4),
            "r2_values": [round(v, 4) for v in st_r2s],
        }
    if ri_r2s:
        summary["random_intercept"] = {
            "r2_mean": round(float(np.mean(ri_r2s)), 4),
            "r2_std": round(float(np.std(ri_r2s)), 4),
            "r2_values": [round(v, 4) for v in ri_r2s],
        }
    results["summary"] = summary

    # B matrix clustering analysis (from first successful seed)
    for seed_str, mt_res in results["multitask_gp"].items():
        if "B_eigenvalues" in mt_res:
            eigvals = mt_res["B_eigenvalues"]
            total = sum(eigvals)
            if total > 0:
                explained = [round(v / total, 4) for v in sorted(eigvals, reverse=True)]
                cumulative = []
                s = 0
                for e in explained:
                    s += e
                    cumulative.append(round(s, 4))
                results["B_matrix_analysis"] = {
                    "seed": seed_str,
                    "eigenvalues": [round(v, 6) for v in sorted(eigvals, reverse=True)],
                    "explained_variance_ratio": explained,
                    "cumulative_variance": cumulative,
                    "n_components_90pct": next(
                        (i + 1 for i, c in enumerate(cumulative) if c >= 0.9), len(cumulative)
                    ),
                }
            break

    print(f"\n{'='*60}")
    print("SUMMARY (study-level split R^2)")
    print(f"{'='*60}")
    for name, vals in [("MultiTask GP", mt_r2s), ("Single-Task GP", st_r2s), ("Random Intercept", ri_r2s)]:
        if vals:
            print(f"  {name:20s}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")

    out_path = Path("models") / "multitask_gp_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
