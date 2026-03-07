#!/usr/bin/env python3
"""GP surrogate with study-level random effect (marginalized as noise)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from scipy.stats import norm
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from diagnostics.compute_icc import fit_reml_random_intercept
from diagnostics.utils import encode_lantern_il, lantern_il_feature_cols, load_lnpdb_clean
from models.splits import scaffold_split


class ExactGPModel(torch.nn.Module):
    def __init__(self, train_x, train_y, likelihood, kernel):
        super().__init__()
        import gpytorch

        class _Model(gpytorch.models.ExactGP):
            def __init__(self, train_x, train_y, likelihood, kernel):
                super().__init__(train_x, train_y, likelihood)
                self.mean_module = gpytorch.means.ConstantMean()
                self.covar_module = gpytorch.kernels.ScaleKernel(kernel)

            def forward(self, x):
                mean_x = self.mean_module(x)
                covar_x = self.covar_module(x)
                return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

        self.model = _Model(train_x, train_y, likelihood, kernel)


def _train_gp(train_x, train_y, noise_init, fix_noise=False, kernel_name="rbf"):
    import gpytorch

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    likelihood.noise = noise_init
    if fix_noise:
        likelihood.raw_noise.requires_grad_(False)

    if kernel_name == "rbf":
        kernel = gpytorch.kernels.RBFKernel()
    elif kernel_name == "matern":
        kernel = gpytorch.kernels.MaternKernel(nu=2.5)
    else:
        raise ValueError(f"Unknown kernel: {kernel_name}")

    model = ExactGPModel(train_x, train_y, likelihood, kernel).model

    model.train()
    likelihood.train()

    optimizer = torch.optim.LBFGS(model.parameters(), lr=0.1, max_iter=100)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    def closure():
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        return loss

    optimizer.step(closure)
    return model, likelihood


class SparseGPModel(torch.nn.Module):
    def __init__(self, inducing_points, kernel):
        super().__init__()
        import gpytorch

        class _Model(gpytorch.models.ApproximateGP):
            def __init__(self, inducing_points, kernel):
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
                self.covar_module = gpytorch.kernels.ScaleKernel(kernel)

            def forward(self, x):
                mean_x = self.mean_module(x)
                covar_x = self.covar_module(x)
                return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

        self.model = _Model(inducing_points, kernel)


def _train_sparse_gp(train_x, train_y, noise_init, fix_noise=False, kernel_name="rbf", epochs=20, batch_size=1024):
    import gpytorch
    from torch.utils.data import DataLoader, TensorDataset

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    likelihood.noise = noise_init
    if fix_noise:
        likelihood.raw_noise.requires_grad_(False)

    if kernel_name == "rbf":
        kernel = gpytorch.kernels.RBFKernel()
    elif kernel_name == "matern":
        kernel = gpytorch.kernels.MaternKernel(nu=2.5)
    else:
        raise ValueError(f"Unknown kernel: {kernel_name}")

    m = min(512, train_x.size(0))
    rng = torch.Generator().manual_seed(42)
    perm = torch.randperm(train_x.size(0), generator=rng)
    inducing_points = train_x[perm[:m]].clone()
    model = SparseGPModel(inducing_points, kernel).model

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_x.size(0))

    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True)
    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            output = model(xb)
            loss = -mll(output, yb)
            loss.backward()
            optimizer.step()

    return model, likelihood


def _predict(model, likelihood, test_x):
    model.eval()
    likelihood.eval()
    with torch.no_grad():
        pred = likelihood(model(test_x))
    return pred.mean, pred.stddev


def _ece(y_true, mu, sigma, alpha=0.1):
    z = norm.ppf(1 - alpha / 2)
    lower = mu - z * sigma
    upper = mu + z * sigma
    covered = (y_true >= lower) & (y_true <= upper)
    coverage = covered.mean()
    ece = abs(coverage - (1 - alpha))
    return float(coverage), float(ece)


def main() -> int:
    df = load_lnpdb_clean(drop_duplicates=False)
    df = df.dropna(subset=["IL_SMILES", "Experiment_value"]).reset_index(drop=True)

    il_smiles = df["IL_SMILES"].tolist()
    train_idx, val_idx, test_idx = scaffold_split(il_smiles, sizes=(0.8, 0.1, 0.1), seed=42)
    train_idx = sorted(set(train_idx + val_idx))

    train_enc, test_enc, _ = encode_lantern_il(df, train_idx=train_idx, test_idx=test_idx, reduction="pca")
    feat_cols = lantern_il_feature_cols(train_enc)

    X_train = train_enc[feat_cols].values
    X_test = test_enc[feat_cols].values
    y_train = train_enc["Experiment_value"].values
    y_test = test_enc["Experiment_value"].values

    # Standardize
    x_scaler = StandardScaler().fit(X_train)
    X_train_s = x_scaler.transform(X_train)
    X_test_s = x_scaler.transform(X_test)

    y_mean = y_train.mean()
    y_std = y_train.std() if y_train.std() > 0 else 1.0
    y_train_s = (y_train - y_mean) / y_std
    y_test_s = (y_test - y_mean) / y_std

    # Estimate random effect variance via REML on training data
    sigma_a2, sigma_e2, _ = fit_reml_random_intercept(y_train_s, train_enc["study_id"].to_numpy())
    noise_init = sigma_a2 + sigma_e2

    train_x = torch.tensor(X_train_s, dtype=torch.float32)
    train_y = torch.tensor(y_train_s, dtype=torch.float32)
    test_x = torch.tensor(X_test_s, dtype=torch.float32)

    results = {}
    use_sparse = train_x.size(0) > 3000
    for label, fix_noise in [("fixed_noise", True), ("learned_noise", False)]:
        if use_sparse:
            model, likelihood = _train_sparse_gp(
                train_x,
                train_y,
                noise_init,
                fix_noise=fix_noise,
                kernel_name="rbf",
                epochs=20,
                batch_size=1024,
            )
        else:
            model, likelihood = _train_gp(train_x, train_y, noise_init, fix_noise=fix_noise, kernel_name="rbf")
        mu, sigma = _predict(model, likelihood, test_x)
        mu = mu.numpy()
        sigma = sigma.numpy()

        r2 = r2_score(y_test_s, mu)
        coverage, ece = _ece(y_test_s, mu, sigma, alpha=0.1)

        per_study = {}
        for study_id, sdf in test_enc.groupby("study_id"):
            idx = sdf.index
            study_mask = test_enc.index.isin(idx)
            y_s = y_test_s[study_mask]
            mu_s = mu[study_mask]
            sigma_s = sigma[study_mask]
            cov_s, ece_s = _ece(y_s, mu_s, sigma_s, alpha=0.1)
            per_study[str(study_id)] = {
                "coverage_90": cov_s,
                "ece_90": ece_s,
                "n": len(y_s),
            }

        results[label] = {
            "r2": float(r2),
            "coverage_90": coverage,
            "ece_90": ece,
            "sigma_a2": float(sigma_a2),
            "sigma_e2": float(sigma_e2),
            "noise_init": float(noise_init),
            "sparse_gp": bool(use_sparse),
            "per_study": per_study,
        }

    out_path = Path("models") / "gp_surrogate_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"Saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
