#!/usr/bin/env python3
"""Tanimoto kernel GP on raw count Morgan fingerprints (no PCA reduction).

The Tanimoto kernel (also called the Jaccard kernel for binary vectors) is
the natural similarity metric for molecular fingerprints:

    k(x, y) = <x, y> / (||x||^2 + ||y||^2 - <x, y>)

Unlike RBF on PCA-reduced fingerprints (which retains ~7% of variance at 5
components), the Tanimoto kernel operates in the full 2048-dimensional
fingerprint space, preserving the molecular similarity structure.

This module fits a Sparse Variational GP (SVGP) with 512 inducing points
using the Tanimoto kernel on raw count Morgan fingerprints, and compares R^2
against the RBF-on-PCA baseline from gp_surrogate.py.

References:
    Ralaivola, L. et al. (2005). "Graph Kernels for Chemical Informatics."
        Neural Networks 18(8), 1093-1110.
    Griffiths, R.-R. et al. (2023). "GAUCHE: A Library for Gaussian Processes
        in Chemistry." NeurIPS 2023.
    Tripp, A. et al. (2023). "Tanimoto Random Features for Scalable Molecular
        Machine Learning." NeurIPS 2023.
"""

import json
import time
from pathlib import Path

import gpytorch
import numpy as np
import torch
from scipy.stats import norm
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from LNPBO.data.generate_morgan_fingerprints import single_morgan_fingerprints
from LNPBO.diagnostics.utils import load_lnpdb_clean, study_split
from LNPBO.models.splits import scaffold_split


class TanimotoKernel(gpytorch.kernels.Kernel):
    """Tanimoto (Jaccard) kernel for molecular fingerprints.

    For non-negative feature vectors x, y:
        k(x, y) = <x, y> / (||x||^2 + ||y||^2 - <x, y>)

    This generalizes the binary Tanimoto coefficient to count-based
    fingerprints. The kernel is positive semi-definite on the non-negative
    orthant (Ralaivola et al. 2005).

    References:
        Ralaivola, L. et al. (2005). "Graph Kernels for Chemical Informatics."
            Neural Networks 18(8), 1093-1110.
        Griffiths, R.-R. et al. (2023). "GAUCHE: A Library for Gaussian
            Processes in Chemistry." NeurIPS 2023.
    """

    has_lengthscale = False

    def forward(self, x1, x2, diag=False, **kwargs):
        # Compute in float64 for numerical stability; many inducing point
        # pairs share identical fingerprints, producing K~1 and a nearly
        # singular Gram matrix.
        x1_ = x1.to(dtype=torch.float64)
        x2_ = x2.to(dtype=torch.float64)

        x1x2 = x1_ @ x2_.transpose(-1, -2)
        x1_sq = (x1_ ** 2).sum(dim=-1, keepdim=True)
        x2_sq = (x2_ ** 2).sum(dim=-1, keepdim=True)
        denom = x1_sq + x2_sq.transpose(-1, -2) - x1x2
        K = x1x2 / denom.clamp(min=1e-8)

        # Cast back to input dtype
        K = K.to(dtype=x1.dtype)

        if diag:
            return K.diag()
        return K


class TanimotoSVGP(gpytorch.models.ApproximateGP):
    """Sparse Variational GP with Tanimoto kernel."""

    def __init__(self, inducing_points):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(TanimotoKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class RBFSVGP(gpytorch.models.ApproximateGP):
    """Sparse Variational GP with RBF kernel (baseline for comparison)."""

    def __init__(self, inducing_points):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def _get_raw_il_fingerprints(df, n_bits=2048, radius=3):
    """Get raw count Morgan fingerprints for unique IL SMILES, then map back."""
    unique_smiles = df["IL_SMILES"].dropna().unique().tolist()
    fp_map = {}
    for smi in unique_smiles:
        fp = single_morgan_fingerprints(smi, radius=radius, n_bits=n_bits, count=True)
        fp_map[smi] = fp

    fps = np.array([
        fp_map.get(smi, np.zeros(n_bits))
        for smi in df["IL_SMILES"].values
    ], dtype=np.float32)
    return fps


def _select_diverse_inducing(train_x, n_inducing, seed=42):
    """Select inducing points from unique fingerprints to avoid singularity.

    When many training points share the same IL fingerprint, random selection
    of inducing points can produce a near-singular Gram matrix. This function
    first deduplicates, then selects from the unique set.
    """
    unique_x, inverse = torch.unique(train_x, dim=0, return_inverse=True)
    m = min(n_inducing, unique_x.size(0))
    rng = torch.Generator().manual_seed(seed)
    perm = torch.randperm(unique_x.size(0), generator=rng)
    return unique_x[perm[:m]].clone(), m


def _train_svgp(model, likelihood, train_x, train_y, epochs=50, batch_size=1024, lr=0.01):
    """Train a sparse variational GP."""
    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_x.size(0))

    loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=batch_size,
        shuffle=True,
    )

    with gpytorch.settings.cholesky_jitter(float_value=1e-4, double_value=1e-6), \
         gpytorch.settings.cholesky_max_tries(6):
        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0
            for xb, yb in loader:
                optimizer.zero_grad()
                output = model(xb)
                loss = -mll(output, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch + 1}/{epochs}, loss={epoch_loss / n_batches:.4f}")


def _predict(model, likelihood, test_x, batch_size=2048):
    """Predict with a sparse variational GP, handling large test sets."""
    model.eval()
    likelihood.eval()
    all_mu = []
    all_sigma = []
    with torch.no_grad(), \
         gpytorch.settings.cholesky_jitter(float_value=1e-4, double_value=1e-6), \
         gpytorch.settings.cholesky_max_tries(6):
        for i in range(0, test_x.size(0), batch_size):
            xb = test_x[i:i + batch_size]
            pred = likelihood(model(xb))
            all_mu.append(pred.mean.cpu())
            all_sigma.append(pred.stddev.cpu())
    return torch.cat(all_mu).numpy(), torch.cat(all_sigma).numpy()


def _ece(y_true, mu, sigma, alpha=0.1):
    z = norm.ppf(1 - alpha / 2)
    lower = mu - z * sigma
    upper = mu + z * sigma
    covered = (y_true >= lower) & (y_true <= upper)
    coverage = float(covered.mean())
    ece = abs(coverage - (1 - alpha))
    return coverage, ece


def _run_split(
    X_train, y_train, X_test, y_test,
    split_name, n_inducing=512, epochs=50, batch_size=1024, seed=42,
):
    """Run Tanimoto SVGP and RBF-on-PCA SVGP on a given split."""
    results = {}

    # Standardize targets
    y_mean = y_train.mean()
    y_std = y_train.std() if y_train.std() > 0 else 1.0
    y_train_s = (y_train - y_mean) / y_std
    y_test_s = (y_test - y_mean) / y_std

    # --- Tanimoto on raw fingerprints (no scaling needed for Tanimoto) ---
    train_x = torch.tensor(X_train, dtype=torch.float32)
    train_y = torch.tensor(y_train_s, dtype=torch.float32)
    test_x = torch.tensor(X_test, dtype=torch.float32)

    inducing_points, m = _select_diverse_inducing(train_x, n_inducing, seed=seed)

    tani_model = TanimotoSVGP(inducing_points)
    tani_likelihood = gpytorch.likelihoods.GaussianLikelihood()

    print(f"\n  [{split_name}] Training Tanimoto SVGP ({m} inducing, {epochs} epochs)...")
    t0 = time.time()
    _train_svgp(tani_model, tani_likelihood, train_x, train_y, epochs=epochs, batch_size=batch_size)
    tani_time = time.time() - t0

    mu, sigma = _predict(tani_model, tani_likelihood, test_x)
    tani_r2 = float(r2_score(y_test_s, mu))
    tani_cov, tani_ece = _ece(y_test_s, mu, sigma)
    tani_width = float(2 * norm.ppf(0.95) * np.mean(sigma))

    results["tanimoto_raw_fp"] = {
        "r2": tani_r2,
        "coverage_90": tani_cov,
        "ece_90": tani_ece,
        "mean_interval_width": tani_width,
        "n_inducing": m,
        "n_features": int(X_train.shape[1]),
        "train_time_s": round(tani_time, 1),
    }
    print(f"  [{split_name}] Tanimoto R^2={tani_r2:.4f}, coverage={tani_cov:.3f}, time={tani_time:.1f}s")

    # --- RBF on PCA-reduced fingerprints (baseline) ---
    from sklearn.decomposition import PCA

    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    pca = PCA(n_components=min(10, X_train_scaled.shape[1], X_train_scaled.shape[0]), random_state=seed)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    var_explained = float(sum(pca.explained_variance_ratio_))

    train_x_pca = torch.tensor(X_train_pca, dtype=torch.float32)
    test_x_pca = torch.tensor(X_test_pca, dtype=torch.float32)

    inducing_pca, m_pca = _select_diverse_inducing(train_x_pca, n_inducing, seed=seed)

    rbf_model = RBFSVGP(inducing_pca)
    rbf_likelihood = gpytorch.likelihoods.GaussianLikelihood()

    print(f"  [{split_name}] Training RBF-on-PCA SVGP ({pca.n_components_} PCs, {var_explained:.1%} variance)...")
    t0 = time.time()
    _train_svgp(rbf_model, rbf_likelihood, train_x_pca, train_y, epochs=epochs, batch_size=batch_size)
    rbf_time = time.time() - t0

    mu_rbf, sigma_rbf = _predict(rbf_model, rbf_likelihood, test_x_pca)
    rbf_r2 = float(r2_score(y_test_s, mu_rbf))
    rbf_cov, rbf_ece = _ece(y_test_s, mu_rbf, sigma_rbf)
    rbf_width = float(2 * norm.ppf(0.95) * np.mean(sigma_rbf))

    results["rbf_pca"] = {
        "r2": rbf_r2,
        "coverage_90": rbf_cov,
        "ece_90": rbf_ece,
        "mean_interval_width": rbf_width,
        "n_pca_components": int(pca.n_components_),
        "pca_variance_explained": round(var_explained, 4),
        "n_inducing": m_pca,
        "train_time_s": round(rbf_time, 1),
    }
    print(f"  [{split_name}] RBF-PCA R^2={rbf_r2:.4f}, coverage={rbf_cov:.3f}, time={rbf_time:.1f}s")

    return results


def main() -> int:
    SEEDS = [42, 123, 456, 789, 2024]
    N_BITS = 2048
    RADIUS = 3
    N_INDUCING = 512
    EPOCHS = 50
    BATCH_SIZE = 1024

    print("Loading LNPDB clean data...")
    df = load_lnpdb_clean(drop_duplicates=False)
    df = df.dropna(subset=["IL_SMILES", "Experiment_value"]).reset_index(drop=True)

    print(f"Computing raw count Morgan fingerprints ({N_BITS}-bit, radius={RADIUS})...")
    X_fps = _get_raw_il_fingerprints(df, n_bits=N_BITS, radius=RADIUS)
    y = df["Experiment_value"].values.astype(np.float32)
    il_smiles = df["IL_SMILES"].tolist()

    all_results = {"config": {
        "n_bits": N_BITS,
        "radius": RADIUS,
        "n_inducing": N_INDUCING,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "seeds": SEEDS,
    }}

    # Run for each seed
    for seed in SEEDS:
        print(f"\n{'='*60}")
        print(f"Seed {seed}")
        print(f"{'='*60}")

        seed_results = {}

        # --- Scaffold split ---
        train_idx, val_idx, test_idx = scaffold_split(il_smiles, sizes=(0.8, 0.1, 0.1), seed=seed)
        train_idx = sorted(set(train_idx + val_idx))

        X_train = X_fps[train_idx]
        X_test = X_fps[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        seed_results["scaffold_split"] = _run_split(
            X_train, y_train, X_test, y_test,
            split_name="scaffold",
            n_inducing=N_INDUCING, epochs=EPOCHS, batch_size=BATCH_SIZE, seed=seed,
        )

        # --- Study-level split ---
        train_ids, test_ids = study_split(df, seed=seed)
        train_mask = df["study_id"].isin(train_ids).values
        test_mask = df["study_id"].isin(test_ids).values

        X_train_study = X_fps[train_mask]
        X_test_study = X_fps[test_mask]
        y_train_study = y[train_mask]
        y_test_study = y[test_mask]

        seed_results["study_split"] = _run_split(
            X_train_study, y_train_study, X_test_study, y_test_study,
            split_name="study",
            n_inducing=N_INDUCING, epochs=EPOCHS, batch_size=BATCH_SIZE, seed=seed,
        )

        all_results[f"seed_{seed}"] = seed_results

    # Aggregate across seeds
    for split_type in ["scaffold_split", "study_split"]:
        for kernel_type in ["tanimoto_raw_fp", "rbf_pca"]:
            r2s = [
                all_results[f"seed_{s}"][split_type][kernel_type]["r2"]
                for s in SEEDS
            ]
            all_results.setdefault("summary", {}).setdefault(split_type, {})[kernel_type] = {
                "r2_mean": round(float(np.mean(r2s)), 4),
                "r2_std": round(float(np.std(r2s)), 4),
                "r2_values": [round(v, 4) for v in r2s],
            }

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for split_type in ["scaffold_split", "study_split"]:
        print(f"\n{split_type}:")
        for kernel_type in ["tanimoto_raw_fp", "rbf_pca"]:
            s = all_results["summary"][split_type][kernel_type]
            print(f"  {kernel_type}: R^2 = {s['r2_mean']:.4f} +/- {s['r2_std']:.4f}")

    out_path = Path("models") / "tanimoto_gp_results.json"
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
