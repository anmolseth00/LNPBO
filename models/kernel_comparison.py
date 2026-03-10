"""Kernel comparison for GP surrogate: RBF vs Matern-5/2 vs Matern-3/2.

Benchmarks GP surrogate with different kernel functions on both scaffold
split and study-level split, reporting R^2, 90% coverage, and mean
interval width.

References
----------
Rasmussen, C.E. & Williams, C.K.I. (2006). "Gaussian Processes for Machine
    Learning." MIT Press, Ch. 4.2 (Matern class of covariance functions).

Matern kernels relax the infinite differentiability assumption of the RBF
kernel. Matern-5/2 (nu=2.5) produces twice-differentiable sample paths,
while Matern-3/2 (nu=1.5) produces once-differentiable paths. For LNP
structure-activity data where different IL scaffolds create discontinuous
jumps, less smooth kernels may better capture the response surface.
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import norm
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

from LNPBO.diagnostics.utils import (
    encode_lantern_il,
    lantern_il_feature_cols,
    load_lnpdb_clean,
    study_split,
)
from LNPBO.models.gp_surrogate import _predict, _train_sparse_gp
from LNPBO.models.splits import scaffold_split


def _evaluate_gp(mu: np.ndarray, sigma: np.ndarray, y_true: np.ndarray, alpha: float = 0.1) -> dict:
    """Compute R^2, coverage, and interval width for GP predictions."""
    r2 = float(r2_score(y_true, mu))

    z = norm.ppf(1 - alpha / 2)
    lower = mu - z * sigma
    upper = mu + z * sigma
    covered = (y_true >= lower) & (y_true <= upper)
    coverage = float(covered.mean())
    mean_width = float((upper - lower).mean())

    return {
        "r2": r2,
        "coverage_90": coverage,
        "mean_interval_width": mean_width,
    }


def _run_gp_benchmark(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    kernel_name: str,
    epochs: int = 30,
    batch_size: int = 1024,
    per_study_groups: dict | None = None,
) -> dict:
    """Train sparse GP with given kernel and return evaluation metrics."""
    x_scaler = StandardScaler().fit(X_train)
    X_train_s = x_scaler.transform(X_train)
    X_test_s = x_scaler.transform(X_test)

    y_mean, y_std = y_train.mean(), max(y_train.std(), 1e-6)
    y_train_s = (y_train - y_mean) / y_std
    y_test_s = (y_test - y_mean) / y_std

    train_x = torch.tensor(X_train_s, dtype=torch.float32)
    train_y = torch.tensor(y_train_s, dtype=torch.float32)
    test_x = torch.tensor(X_test_s, dtype=torch.float32)

    model, likelihood = _train_sparse_gp(
        train_x, train_y, noise_init=1.0, fix_noise=False,
        kernel_name=kernel_name, epochs=epochs, batch_size=batch_size,
    )

    mu, sigma = _predict(model, likelihood, test_x)
    mu_np = mu.numpy()
    sigma_np = sigma.numpy()

    metrics = _evaluate_gp(mu_np, sigma_np, y_test_s)
    metrics["n_train"] = len(X_train)
    metrics["n_test"] = len(X_test)
    metrics["n_features"] = X_train.shape[1]

    # Per-study coverage if groups provided
    if per_study_groups is not None:
        per_study = {}
        for sid, mask in per_study_groups.items():
            if mask.sum() < 5:
                continue
            study_metrics = _evaluate_gp(mu_np[mask], sigma_np[mask], y_test_s[mask])
            study_metrics["n"] = int(mask.sum())
            per_study[str(sid)] = study_metrics
        metrics["per_study"] = per_study

    return metrics


def main() -> int:
    t0 = time.time()

    df = load_lnpdb_clean(drop_duplicates=False)
    df = df.dropna(subset=["IL_SMILES", "Experiment_value"]).reset_index(drop=True)

    kernels = ["rbf", "matern52", "matern32"]
    results = {"scaffold_split": {}, "study_split": {}}

    # --- Scaffold split ---
    print("=== Scaffold Split ===")
    il_smiles = df["IL_SMILES"].tolist()
    train_idx, val_idx, test_idx = scaffold_split(il_smiles, sizes=(0.8, 0.1, 0.1), seed=42)
    train_idx = sorted(set(train_idx + val_idx))

    train_enc, test_enc, _ = encode_lantern_il(df, train_idx=train_idx, test_idx=test_idx, reduction="pca")
    feat_cols = lantern_il_feature_cols(train_enc)

    X_train = train_enc[feat_cols].values
    X_test = test_enc[feat_cols].values
    y_train = train_enc["Experiment_value"].values
    y_test = test_enc["Experiment_value"].values

    # Per-study masks for test set
    test_study_groups = {}
    if "study_id" in test_enc.columns:
        for sid, sdf in test_enc.groupby("study_id"):
            mask = np.zeros(len(test_enc), dtype=bool)
            mask[sdf.index - test_enc.index[0]] = True
            test_study_groups[sid] = mask

    for kernel in kernels:
        print(f"  Training {kernel}...")
        metrics = _run_gp_benchmark(
            X_train, y_train, X_test, y_test, kernel,
            per_study_groups=test_study_groups if test_study_groups else None,
        )
        results["scaffold_split"][kernel] = metrics
        print(f"    R^2={metrics['r2']:.4f}, coverage={metrics['coverage_90']:.3f}, "
              f"width={metrics['mean_interval_width']:.3f}")

    # --- Study-level split ---
    print("\n=== Study-Level Split ===")
    train_study_ids, test_study_ids = study_split(df, seed=42)
    s_train_mask = df["study_id"].isin(train_study_ids)
    s_test_mask = df["study_id"].isin(test_study_ids)
    s_train_idx = df.index[s_train_mask].tolist()
    s_test_idx = df.index[s_test_mask].tolist()

    if len(s_test_idx) > 100:
        s_train_enc, s_test_enc, _ = encode_lantern_il(
            df, train_idx=s_train_idx, test_idx=s_test_idx, reduction="pca",
        )
        s_feat_cols = lantern_il_feature_cols(s_train_enc)

        s_X_train = s_train_enc[s_feat_cols].values
        s_X_test = s_test_enc[s_feat_cols].values
        s_y_train = s_train_enc["Experiment_value"].values
        s_y_test = s_test_enc["Experiment_value"].values

        s_test_study_groups = {}
        if "study_id" in s_test_enc.columns:
            for sid, sdf in s_test_enc.groupby("study_id"):
                mask = np.zeros(len(s_test_enc), dtype=bool)
                mask[sdf.index - s_test_enc.index[0]] = True
                s_test_study_groups[sid] = mask

        for kernel in kernels:
            print(f"  Training {kernel}...")
            metrics = _run_gp_benchmark(
                s_X_train, s_y_train, s_X_test, s_y_test, kernel,
                per_study_groups=s_test_study_groups if s_test_study_groups else None,
            )
            results["study_split"][kernel] = metrics
            print(f"    R^2={metrics['r2']:.4f}, coverage={metrics['coverage_90']:.3f}, "
                  f"width={metrics['mean_interval_width']:.3f}")
    else:
        print("  Insufficient test data for study-level split")

    # --- Summary table ---
    print("\n=== Summary ===")
    print(f"{'Split':<15} {'Kernel':<12} {'R^2':>8} {'Cov90':>8} {'Width':>8}")
    print("-" * 55)
    for split_name, split_results in results.items():
        for kernel, m in split_results.items():
            if isinstance(m, dict) and "r2" in m:
                print(f"{split_name:<15} {kernel:<12} {m['r2']:>8.4f} "
                      f"{m['coverage_90']:>8.3f} {m['mean_interval_width']:>8.3f}")

    elapsed = time.time() - t0
    results["elapsed_seconds"] = round(elapsed, 1)

    out_path = Path("models") / "kernel_comparison_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
