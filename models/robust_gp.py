#!/usr/bin/env python3
"""Robust GP via BoTorch's RobustRelevancePursuitSingleTaskGP.

Tests BoTorch's relevance pursuit model which automatically identifies and
downweights unreliable observations via Bayesian model selection, without
requiring manual outlier removal.

The model uses a SparseOutlierGaussianLikelihood that adds per-observation
noise variances for detected outliers, with the support (set of outliers)
determined by the Relevance Pursuit algorithm during fitting.

Note: RobustRelevancePursuitSingleTaskGP requires ExactGP, which is O(n^3).
For the full ~19k dataset, we subsample to n_max_train observations for
training and evaluate on a held-out test set. The outlier detection still
operates on the training set and we report which observations are flagged.

Citation:
    Ament, S. et al. (2024). "Robust Gaussian Processes via Relevance Pursuit."
    arXiv:2410.24222.

    BoTorch documentation: "Robust Regression via Relevance Pursuit" tutorial.
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

from LNPBO.diagnostics.utils import (
    encode_lantern_il,
    lantern_il_feature_cols,
    load_lnpdb_clean,
)
from LNPBO.data.lnpdb_bridge import load_lnpdb_full
from LNPBO.models.splits import scaffold_split


SEEDS = [42, 123, 456, 789, 2024]
N_MAX_TRAIN = 1000


def _load_full_data(drop_unnormalized: bool):
    """Load LNPDB data with or without unnormalized rows."""
    dataset = load_lnpdb_full(
        drop_unnormalized=drop_unnormalized,
        drop_duplicates=False,
    )
    df = dataset.df
    # Add study_id and assay_type
    from LNPBO.diagnostics.utils import add_assay_type, add_study_id
    df = add_study_id(df)
    df = add_assay_type(df)
    df = df.dropna(subset=["IL_SMILES", "Experiment_value"]).reset_index(drop=True)
    return df


def _prepare_data(df, seed, n_max_train=N_MAX_TRAIN):
    """Encode features and split data."""
    il_smiles = df["IL_SMILES"].tolist()
    train_idx, val_idx, test_idx = scaffold_split(il_smiles, sizes=(0.8, 0.1, 0.1), seed=seed)
    train_idx = sorted(set(train_idx + val_idx))

    # Subsample training set if too large for ExactGP
    rng = np.random.RandomState(seed)
    if len(train_idx) > n_max_train:
        train_idx = sorted(rng.choice(train_idx, size=n_max_train, replace=False).tolist())

    train_enc, test_enc, _ = encode_lantern_il(
        df, train_idx=train_idx, test_idx=test_idx, reduction="pca"
    )
    feat_cols = lantern_il_feature_cols(train_enc)

    X_train = train_enc[feat_cols].values.astype(np.float64)
    X_test = test_enc[feat_cols].values.astype(np.float64)
    y_train = train_enc["Experiment_value"].values.astype(np.float64)
    y_test = test_enc["Experiment_value"].values.astype(np.float64)

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    return X_train_s, X_test_s, y_train, y_test, train_enc, test_enc


def _run_robust_gp(X_train, y_train, X_test, y_test, seed=42):
    """Fit RobustRelevancePursuitSingleTaskGP and evaluate."""
    from botorch.fit import fit_gpytorch_mll
    from botorch.models.robust_relevance_pursuit_model import (
        RobustRelevancePursuitSingleTaskGP,
    )
    from gpytorch.mlls import ExactMarginalLogLikelihood

    train_X = torch.tensor(X_train, dtype=torch.float64)
    train_Y = torch.tensor(y_train, dtype=torch.float64).unsqueeze(-1)
    test_X = torch.tensor(X_test, dtype=torch.float64)

    torch.manual_seed(seed)
    model = RobustRelevancePursuitSingleTaskGP(
        train_X=train_X,
        train_Y=train_Y,
        cache_model_trace=True,
    )

    mll = ExactMarginalLogLikelihood(model.likelihood, model)

    t0 = time.time()
    # Use fewer sparsity levels and a timeout to keep runtime manageable
    fit_gpytorch_mll(
        mll,
        fractions_of_outliers=[0.0, 0.05, 0.1, 0.2, 0.5],
        timeout_sec=300.0,
    )
    fit_time = time.time() - t0

    model.eval()

    with torch.no_grad():
        posterior = model.posterior(test_X)
        mu_test = posterior.mean.detach().squeeze(-1).cpu().numpy()
        sigma_test = posterior.variance.detach().squeeze(-1).sqrt().cpu().numpy()

    r2 = float(r2_score(y_test, mu_test))

    # Extract outlier information from the sparse noise module
    sparse_noise = model.likelihood.noise_covar
    outlier_indices = sorted(sparse_noise.support)
    n_outliers = len(outlier_indices)

    # Get rho values for outliers (rho is a property, may require grad)
    with torch.no_grad():
        rho_values = sparse_noise.rho.detach().cpu().numpy().flatten()
    outlier_rhos = {str(idx): float(rho_values[idx]) for idx in outlier_indices}

    # BMC results
    bmc_support_sizes = None
    bmc_probs = None
    if model.bmc_support_sizes is not None:
        bmc_support_sizes = model.bmc_support_sizes.detach().cpu().numpy().tolist()
        bmc_probs = model.bmc_probabilities.detach().cpu().numpy().tolist()

    return {
        "r2": r2,
        "n_train": len(y_train),
        "n_test": len(y_test),
        "fit_time_s": round(fit_time, 1),
        "n_outliers_detected": n_outliers,
        "outlier_fraction": round(n_outliers / len(y_train), 4),
        "outlier_indices": outlier_indices[:100],  # truncate for JSON
        "outlier_rhos_top20": dict(sorted(outlier_rhos.items(), key=lambda x: -x[1])[:20]),
        "bmc_support_sizes": bmc_support_sizes,
        "bmc_probabilities": bmc_probs,
    }


def _run_standard_gp(X_train, y_train, X_test, y_test, seed=42):
    """Fit standard SingleTaskGP for comparison."""
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import SingleTaskGP
    from gpytorch.mlls import ExactMarginalLogLikelihood

    train_X = torch.tensor(X_train, dtype=torch.float64)
    train_Y = torch.tensor(y_train, dtype=torch.float64).unsqueeze(-1)
    test_X = torch.tensor(X_test, dtype=torch.float64)

    torch.manual_seed(seed)
    model = SingleTaskGP(train_X=train_X, train_Y=train_Y)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)

    t0 = time.time()
    fit_gpytorch_mll(mll)
    fit_time = time.time() - t0

    model.eval()

    with torch.no_grad():
        posterior = model.posterior(test_X)
        mu_test = posterior.mean.detach().squeeze(-1).cpu().numpy()

    r2 = float(r2_score(y_test, mu_test))

    return {
        "r2": r2,
        "n_train": len(y_train),
        "n_test": len(y_test),
        "fit_time_s": round(fit_time, 1),
    }


def _run_student_t_gp(X_train, y_train, X_test, y_test, seed=42):
    """Fit GP with Student-T likelihood via variational inference (SVGP).

    Student-T likelihood has heavier tails than Gaussian, providing natural
    robustness to outliers. Since ExactGP requires Gaussian likelihood,
    we use variational inference with inducing points.

    Citation:
        Jylanki, P., Vanhatalo, J., & Vehtari, A. (2011).
        "Robust Gaussian Process Regression with a Student-t Likelihood."
        JMLR 12, 3227-3257.
    """
    import gpytorch
    from gpytorch.likelihoods import StudentTLikelihood
    from torch.utils.data import DataLoader, TensorDataset

    train_x = torch.tensor(X_train, dtype=torch.float32)
    train_y = torch.tensor(y_train, dtype=torch.float32)
    test_x = torch.tensor(X_test, dtype=torch.float32)

    n_inducing = min(512, len(X_train))
    rng_torch = torch.Generator().manual_seed(seed)
    perm = torch.randperm(train_x.size(0), generator=rng_torch)
    inducing_points = train_x[perm[:n_inducing]].clone()

    class SVGPModel(gpytorch.models.ApproximateGP):
        def __init__(self, inducing_points):
            variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
                inducing_points.size(0)
            )
            variational_strategy = gpytorch.variational.VariationalStrategy(
                self, inducing_points, variational_distribution,
                learn_inducing_locations=True,
            )
            super().__init__(variational_strategy)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel()
            )

        def forward(self, x):
            mean = self.mean_module(x)
            covar = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean, covar)

    torch.manual_seed(seed)
    likelihood = StudentTLikelihood()
    model = SVGPModel(inducing_points)

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(likelihood.parameters()), lr=0.01
    )
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_x.size(0))

    loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=min(1024, len(train_x)),
        shuffle=True,
    )

    t0 = time.time()
    n_epochs = 50
    for epoch in range(n_epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            output = model(xb)
            loss = -mll(output, yb)
            loss.backward()
            optimizer.step()
    fit_time = time.time() - t0

    model.eval()
    likelihood.eval()

    with torch.no_grad():
        f_pred = model(test_x)
        pred = likelihood(f_pred)
        mu_test = pred.mean.numpy()
        sigma_test = pred.stddev.numpy()

    r2 = float(r2_score(y_test, mu_test))

    learned_df = float(likelihood.deg_free.item())

    return {
        "r2": r2,
        "n_train": len(y_train),
        "n_test": len(y_test),
        "fit_time_s": round(fit_time, 1),
        "learned_deg_free": round(learned_df, 2),
        "n_inducing": n_inducing,
        "n_epochs": n_epochs,
    }


def _check_outlier_overlap(outlier_indices_in_train, train_enc, known_threshold=10.0):
    """Check overlap between detected outliers and known unnormalized rows."""
    train_exp_vals = train_enc["Experiment_value"].values
    known_outlier_mask = np.abs(train_exp_vals) > known_threshold
    known_outlier_indices = set(np.where(known_outlier_mask)[0])

    detected = set(outlier_indices_in_train)

    if not known_outlier_indices:
        return {
            "n_known_outliers_in_train": 0,
            "n_detected": len(detected),
            "overlap": 0,
            "note": "No known outliers in training set (data was loaded with drop_unnormalized=False but none ended up in train split)",
        }

    overlap = detected & known_outlier_indices
    precision = len(overlap) / len(detected) if detected else 0.0
    recall = len(overlap) / len(known_outlier_indices) if known_outlier_indices else 0.0

    return {
        "n_known_outliers_in_train": len(known_outlier_indices),
        "n_detected": len(detected),
        "n_overlap": len(overlap),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "known_outlier_values": sorted([float(train_exp_vals[i]) for i in known_outlier_indices])[:20],
    }


def main() -> int:
    results = {
        "description": (
            "Robust GP via BoTorch's RobustRelevancePursuitSingleTaskGP. "
            "Compared against standard GP and Student-T likelihood GP. "
            "Tests on full dataset (with unnormalized rows) and clean dataset."
        ),
        "citation": (
            "Ament et al. (2024) 'Robust Gaussian Processes via Relevance Pursuit', "
            "arXiv:2410.24222. "
            "Jylanki et al. (2011) 'Robust GP Regression with a Student-t Likelihood', "
            "JMLR 12."
        ),
        "botorch_version": None,
        "note": (
            f"ExactGP is O(n^3); training subsampled to n_max_train={N_MAX_TRAIN} "
            "per seed. Student-T GP uses SVGP with 512 inducing points (no size limit)."
        ),
        "seeds": SEEDS,
    }

    import botorch
    results["botorch_version"] = botorch.__version__

    # Run on two dataset variants: full (with outliers) and clean
    for data_label, drop_unnormalized in [("full_data", False), ("clean_data", True)]:
        print(f"\n{'#'*60}")
        print(f"Dataset: {data_label} (drop_unnormalized={drop_unnormalized})")
        print(f"{'#'*60}")

        df = _load_full_data(drop_unnormalized=drop_unnormalized)
        print(f"  Loaded {len(df)} rows")

        data_results = {
            "n_rows": len(df),
            "robust_gp": {},
            "standard_gp": {},
            "student_t_gp": {},
        }

        robust_r2s, standard_r2s, student_r2s = [], [], []

        for seed in SEEDS:
            print(f"\n  Seed {seed}")

            X_train, X_test, y_train, y_test, train_enc, test_enc = _prepare_data(
                df, seed=seed, n_max_train=N_MAX_TRAIN
            )

            # Robust GP (RobustRelevancePursuitSingleTaskGP)
            print(f"    Robust GP (n_train={len(y_train)})...")
            try:
                robust_result = _run_robust_gp(X_train, y_train, X_test, y_test, seed=seed)
                print(f"      R^2={robust_result['r2']:.4f}, "
                      f"outliers={robust_result['n_outliers_detected']} "
                      f"({robust_result['outlier_fraction']:.1%}), "
                      f"time={robust_result['fit_time_s']}s")

                # Check overlap with known outliers (only meaningful for full data)
                if not drop_unnormalized:
                    overlap = _check_outlier_overlap(
                        robust_result["outlier_indices"], train_enc
                    )
                    robust_result["known_outlier_overlap"] = overlap
                    if "n_overlap" in overlap:
                        print(f"      Known outlier overlap: {overlap['n_overlap']}/{overlap['n_known_outliers_in_train']} "
                              f"(recall={overlap.get('recall', 'N/A')})")

                data_results["robust_gp"][str(seed)] = robust_result
                robust_r2s.append(robust_result["r2"])
            except Exception as e:
                print(f"      FAILED: {e}")
                import traceback
                traceback.print_exc()
                data_results["robust_gp"][str(seed)] = {"error": str(e)}

            # Standard GP
            print(f"    Standard GP (n_train={len(y_train)})...")
            try:
                std_result = _run_standard_gp(X_train, y_train, X_test, y_test, seed=seed)
                print(f"      R^2={std_result['r2']:.4f}, time={std_result['fit_time_s']}s")
                data_results["standard_gp"][str(seed)] = std_result
                standard_r2s.append(std_result["r2"])
            except Exception as e:
                print(f"      FAILED: {e}")
                data_results["standard_gp"][str(seed)] = {"error": str(e)}

            # Student-T GP (can handle full training set via SVGP)
            # Use all training data (not subsampled) since SVGP scales
            X_train_full, X_test_full, y_train_full, y_test_full, _, _ = _prepare_data(
                df, seed=seed, n_max_train=len(df),
            )
            print(f"    Student-T GP (n_train={len(y_train_full)})...")
            try:
                st_result = _run_student_t_gp(
                    X_train_full, y_train_full, X_test_full, y_test_full, seed=seed
                )
                print(f"      R^2={st_result['r2']:.4f}, "
                      f"df={st_result['learned_deg_free']:.1f}, "
                      f"time={st_result['fit_time_s']}s")
                data_results["student_t_gp"][str(seed)] = st_result
                student_r2s.append(st_result["r2"])
            except Exception as e:
                print(f"      FAILED: {e}")
                import traceback
                traceback.print_exc()
                data_results["student_t_gp"][str(seed)] = {"error": str(e)}

        # Summary for this data variant
        summary = {}
        for name, vals in [("robust_gp", robust_r2s), ("standard_gp", standard_r2s), ("student_t_gp", student_r2s)]:
            if vals:
                summary[name] = {
                    "r2_mean": round(float(np.mean(vals)), 4),
                    "r2_std": round(float(np.std(vals)), 4),
                    "r2_values": [round(v, 4) for v in vals],
                }
        data_results["summary"] = summary
        results[data_label] = data_results

        print(f"\n  Summary ({data_label}):")
        for name, vals in [("Robust GP", robust_r2s), ("Standard GP", standard_r2s), ("Student-T GP", student_r2s)]:
            if vals:
                print(f"    {name:15s}: R^2 = {np.mean(vals):.4f} +/- {np.std(vals):.4f}")

    out_path = Path("models") / "robust_gp_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
