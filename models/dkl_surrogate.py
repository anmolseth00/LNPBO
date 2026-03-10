#!/usr/bin/env python3
"""Deep Kernel Learning surrogate with warm-start and end-to-end variants.

DKL maps inputs through a neural network, then applies a GP kernel on the
learned representation: k(x, y) = k_base(g_theta(x), g_theta(y)).

Two training modes:
  1. Warm-start (safe): Pretrain MLP on MSE, freeze weights, fit GP
     hyperparameters only. Avoids Ober & Rasmussen (UAI 2021) overcorrelation.
  2. End-to-end (risky): Joint optimization of NN + GP via marginal likelihood.

References:
    Wilson, A.G. et al. (2016). "Deep Kernel Learning." AISTATS 2016.
        arXiv:1511.02222.
    Ober, S.W., Rasmussen, C.E. & van der Wilk, M. (2021). "The Promises and
        Pitfalls of Deep Kernel Learning." UAI 2021. arXiv:2102.12108.
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

from LNPBO.diagnostics.utils import (
    encode_lantern_il,
    lantern_il_feature_cols,
    load_lnpdb_clean,
    prepare_study_data,
)
from LNPBO.models.splits import scaffold_split
from LNPBO.models.surrogate_mlp import SurrogateMLP


class FeatureExtractor(torch.nn.Module):
    """MLP feature extractor (in -> 256 -> 128 -> out_dim)."""

    def __init__(self, in_dim, out_dim=32):
        super().__init__()
        self.fc1 = torch.nn.Linear(in_dim, 256)
        self.fc2 = torch.nn.Linear(256, 128)
        self.fc3 = torch.nn.Linear(128, out_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class DKLExactGP(gpytorch.models.ExactGP):
    """Exact GP operating in a learned feature space (DKL).

    The feature extractor transforms raw inputs before the GP sees them.
    train_x passed to __init__ must already be in the feature space.

    Reference: Wilson et al. (2016), "Deep Kernel Learning", AISTATS 2016.
    """

    def __init__(self, train_features, train_y, likelihood, out_dim=32):
        super().__init__(train_features, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=out_dim)
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class DKLWrapper(torch.nn.Module):
    """Wraps feature extractor + ExactGP for training and prediction.

    Manages the feature extraction step so the GP always receives data
    in the learned representation space.
    """

    def __init__(self, feature_extractor, gp_model, likelihood):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.gp_model = gp_model
        self.likelihood = likelihood

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.gp_model(features)

    def predict(self, x):
        self.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            features = self.feature_extractor(x)
            pred = self.likelihood(self.gp_model(features))
        return pred.mean.cpu().numpy(), pred.stddev.cpu().numpy()


class SparseDKLModel(gpytorch.models.ApproximateGP):
    """Variational (sparse) GP for DKL.

    Inducing points are in the feature space. The forward method receives
    data already transformed by the feature extractor.
    """

    def __init__(self, inducing_points, out_dim=32):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=out_dim)
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class SparseDKLWrapper(torch.nn.Module):
    """Wraps feature extractor + SparseDKLModel for training and prediction."""

    def __init__(self, feature_extractor, gp_model, likelihood):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.gp_model = gp_model
        self.likelihood = likelihood

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.gp_model(features)

    def predict(self, x):
        self.gp_model.eval()
        self.likelihood.eval()
        self.feature_extractor.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            features = self.feature_extractor(x)
            pred = self.likelihood(self.gp_model(features))
        return pred.mean.cpu().numpy(), pred.stddev.cpu().numpy()


def pretrain_feature_extractor(X_train, y_train, in_dim, out_dim=32, epochs=100, lr=1e-3):
    """Pretrain MLP as a regression model, then extract the feature layers."""
    model = torch.nn.Sequential(
        torch.nn.Linear(in_dim, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, out_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(out_dim, 1),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    dataset = TensorDataset(X_train, y_train.unsqueeze(-1))
    loader = DataLoader(dataset, batch_size=256, shuffle=True)

    model.train()
    for epoch in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(model(xb), yb)
            loss.backward()
            optimizer.step()

    fe = FeatureExtractor(in_dim, out_dim)
    fe.fc1.weight.data.copy_(model[0].weight.data)
    fe.fc1.bias.data.copy_(model[0].bias.data)
    fe.fc2.weight.data.copy_(model[2].weight.data)
    fe.fc2.bias.data.copy_(model[2].bias.data)
    fe.fc3.weight.data.copy_(model[4].weight.data)
    fe.fc3.bias.data.copy_(model[4].bias.data)
    return fe


def train_dkl_warmstart(X_train, y_train, out_dim=32, pretrain_epochs=100,
                        gp_epochs=50, lr=0.01):
    """Warm-start DKL: pretrain MLP, freeze, then fit GP only.

    This avoids the Ober & Rasmussen (UAI 2021) pitfall where joint training
    causes the feature extractor to collapse representations, making the GP
    overconfident.
    """
    in_dim = X_train.shape[1]
    fe = pretrain_feature_extractor(X_train, y_train, in_dim, out_dim, pretrain_epochs)
    for p in fe.parameters():
        p.requires_grad_(False)

    use_sparse = X_train.size(0) > 3000

    with torch.no_grad():
        train_features = fe(X_train)

    if use_sparse:
        m = min(512, train_features.size(0))
        rng = torch.Generator().manual_seed(42)
        perm = torch.randperm(train_features.size(0), generator=rng)
        inducing_points = train_features[perm[:m]].clone()

        gp_model = SparseDKLModel(inducing_points, out_dim)
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        wrapper = SparseDKLWrapper(fe, gp_model, likelihood)

        gp_model.train()
        likelihood.train()

        gp_params = list(gp_model.parameters()) + list(likelihood.parameters())
        optimizer = torch.optim.Adam(gp_params, lr=lr)
        mll = gpytorch.mlls.VariationalELBO(likelihood, gp_model, num_data=X_train.size(0))

        loader = DataLoader(TensorDataset(X_train, y_train), batch_size=1024, shuffle=True)
        for _ in range(gp_epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                with torch.no_grad():
                    feat_b = fe(xb)
                with gpytorch.settings.cholesky_jitter(1e-4):
                    output = gp_model(feat_b)
                    loss = -mll(output, yb)
                loss.backward()
                optimizer.step()
    else:
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        gp_model = DKLExactGP(train_features, y_train, likelihood, out_dim)
        wrapper = DKLWrapper(fe, gp_model, likelihood)

        gp_model.train()
        likelihood.train()

        gp_params = list(gp_model.parameters()) + list(likelihood.parameters())
        optimizer = torch.optim.Adam(gp_params, lr=lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp_model)

        for _ in range(gp_epochs):
            optimizer.zero_grad()
            output = gp_model(train_features)
            loss = -mll(output, y_train)
            loss.backward()
            optimizer.step()

    return wrapper, use_sparse


def train_dkl_e2e(X_train, y_train, out_dim=32, epochs=100, lr=1e-3):
    """End-to-end DKL: joint optimization of NN + GP via marginal likelihood.

    Warning: prone to overfitting per Ober & Rasmussen (UAI 2021).
    """
    in_dim = X_train.shape[1]
    fe = FeatureExtractor(in_dim, out_dim)
    use_sparse = X_train.size(0) > 3000

    if use_sparse:
        with torch.no_grad():
            init_features = fe(X_train)
        m = min(512, init_features.size(0))
        rng = torch.Generator().manual_seed(42)
        perm = torch.randperm(init_features.size(0), generator=rng)
        inducing_points = init_features[perm[:m]].clone()

        gp_model = SparseDKLModel(inducing_points, out_dim)
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        wrapper = SparseDKLWrapper(fe, gp_model, likelihood)

        gp_model.train()
        likelihood.train()
        fe.train()

        all_params = list(fe.parameters()) + list(gp_model.parameters()) + list(likelihood.parameters())
        optimizer = torch.optim.Adam(all_params, lr=lr)
        mll = gpytorch.mlls.VariationalELBO(likelihood, gp_model, num_data=X_train.size(0))

        loader = DataLoader(TensorDataset(X_train, y_train), batch_size=1024, shuffle=True)
        for _ in range(epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                feat_b = fe(xb)
                with gpytorch.settings.cholesky_jitter(1e-4):
                    output = gp_model(feat_b)
                    loss = -mll(output, yb)
                loss.backward()
                optimizer.step()
    else:
        with torch.no_grad():
            init_features = fe(X_train)
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        gp_model = DKLExactGP(init_features, y_train, likelihood, out_dim)
        wrapper = DKLWrapper(fe, gp_model, likelihood)

        gp_model.train()
        likelihood.train()
        fe.train()

        all_params = list(fe.parameters()) + list(gp_model.parameters()) + list(likelihood.parameters())
        optimizer = torch.optim.Adam(all_params, lr=lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp_model)

        for _ in range(epochs):
            optimizer.zero_grad()
            features = fe(X_train)
            gp_model.set_train_data(features, y_train, strict=False)
            output = gp_model(features)
            loss = -mll(output, y_train)
            loss.backward()
            optimizer.step()

    return wrapper, use_sparse


def compute_calibration(y_true, mu, sigma, alpha=0.1):
    z = norm.ppf(1 - alpha / 2)
    lower = mu - z * sigma
    upper = mu + z * sigma
    covered = (y_true >= lower) & (y_true <= upper)
    coverage = float(covered.mean())
    avg_width = float((upper - lower).mean())
    return coverage, avg_width


def eval_dkl(wrapper, X_test_t, y_test, y_test_s, y_mean, y_std):
    mu_s, sigma_s = wrapper.predict(X_test_t)
    mu = mu_s * y_std + y_mean
    sigma = sigma_s * y_std
    r2 = float(r2_score(y_test, mu))
    r2_s = float(r2_score(y_test_s, mu_s))
    cov90, width90 = compute_calibration(y_test, mu, sigma, alpha=0.1)
    cov68, width68 = compute_calibration(y_test, mu, sigma, alpha=0.32)
    return {
        "r2": r2,
        "r2_standardized": r2_s,
        "coverage_90": cov90,
        "coverage_68": cov68,
        "avg_width_90": width90,
        "avg_width_68": width68,
    }


def run_scaffold_split():
    df = load_lnpdb_clean(drop_duplicates=False)
    df = df.dropna(subset=["IL_SMILES", "Experiment_value"]).reset_index(drop=True)

    il_smiles = df["IL_SMILES"].tolist()
    train_idx, val_idx, test_idx = scaffold_split(il_smiles, sizes=(0.8, 0.1, 0.1), seed=42)
    train_idx = sorted(set(train_idx + val_idx))

    train_enc, test_enc, _ = encode_lantern_il(df, train_idx=train_idx, test_idx=test_idx, reduction="pca")
    feat_cols = lantern_il_feature_cols(train_enc)

    X_train = train_enc[feat_cols].values.astype(np.float32)
    X_test = test_enc[feat_cols].values.astype(np.float32)
    y_train = train_enc["Experiment_value"].values.astype(np.float32)
    y_test = test_enc["Experiment_value"].values.astype(np.float32)

    x_scaler = StandardScaler().fit(X_train)
    X_train_s = x_scaler.transform(X_train).astype(np.float32)
    X_test_s = x_scaler.transform(X_test).astype(np.float32)

    y_mean, y_std = float(y_train.mean()), float(y_train.std())
    if y_std == 0:
        y_std = 1.0
    y_train_s = ((y_train - y_mean) / y_std).astype(np.float32)
    y_test_s = ((y_test - y_mean) / y_std).astype(np.float32)

    X_train_t = torch.tensor(X_train_s)
    X_test_t = torch.tensor(X_test_s)
    y_train_t = torch.tensor(y_train_s)

    results = {}

    t0 = time.time()
    wrapper_ws, _ = train_dkl_warmstart(X_train_t, y_train_t, out_dim=32,
                                        pretrain_epochs=100, gp_epochs=50)
    t_ws = time.time() - t0
    res_ws = eval_dkl(wrapper_ws, X_test_t, y_test, y_test_s, y_mean, y_std)
    res_ws["train_time_s"] = round(t_ws, 1)
    results["warmstart"] = res_ws
    print(f"Scaffold | Warm-start DKL: R2={res_ws['r2']:.3f}, "
          f"cov90={res_ws['coverage_90']:.3f}, time={t_ws:.1f}s")

    t0 = time.time()
    wrapper_e2e, _ = train_dkl_e2e(X_train_t, y_train_t, out_dim=32, epochs=100)
    t_e2e = time.time() - t0
    res_e2e = eval_dkl(wrapper_e2e, X_test_t, y_test, y_test_s, y_mean, y_std)
    res_e2e["train_time_s"] = round(t_e2e, 1)
    results["e2e"] = res_e2e
    print(f"Scaffold | End-to-end DKL: R2={res_e2e['r2']:.3f}, "
          f"cov90={res_e2e['coverage_90']:.3f}, time={t_e2e:.1f}s")

    return results


def run_study_split():
    X, y, study_ids, train_mask, test_mask = prepare_study_data(min_n=5)

    x_scaler = StandardScaler().fit(X[train_mask])
    X_s = x_scaler.transform(X).astype(np.float32)

    y_mean, y_std = float(y[train_mask].mean()), float(y[train_mask].std())
    if y_std == 0:
        y_std = 1.0
    y_s = ((y - y_mean) / y_std).astype(np.float32)

    X_train_t = torch.tensor(X_s[train_mask])
    X_test_t = torch.tensor(X_s[test_mask])
    y_train_t = torch.tensor(y_s[train_mask])

    results = {}

    t0 = time.time()
    wrapper_ws, _ = train_dkl_warmstart(X_train_t, y_train_t, out_dim=32,
                                        pretrain_epochs=100, gp_epochs=50)
    t_ws = time.time() - t0
    res_ws = eval_dkl(wrapper_ws, X_test_t, y[test_mask], y_s[test_mask], y_mean, y_std)
    res_ws["train_time_s"] = round(t_ws, 1)
    results["warmstart"] = res_ws
    print(f"Study    | Warm-start DKL: R2={res_ws['r2']:.3f}, "
          f"cov90={res_ws['coverage_90']:.3f}, time={t_ws:.1f}s")

    t0 = time.time()
    wrapper_e2e, _ = train_dkl_e2e(X_train_t, y_train_t, out_dim=32, epochs=100)
    t_e2e = time.time() - t0
    res_e2e = eval_dkl(wrapper_e2e, X_test_t, y[test_mask], y_s[test_mask], y_mean, y_std)
    res_e2e["train_time_s"] = round(t_e2e, 1)
    results["e2e"] = res_e2e
    print(f"Study    | End-to-end DKL: R2={res_e2e['r2']:.3f}, "
          f"cov90={res_e2e['coverage_90']:.3f}, time={t_e2e:.1f}s")

    return results


def main() -> int:
    print("=== Deep Kernel Learning (DKL) Surrogate ===")
    print()

    print("--- Scaffold Split ---")
    scaffold_results = run_scaffold_split()

    print()
    print("--- Study-Level Split ---")
    study_results = run_study_split()

    results = {
        "scaffold_split": scaffold_results,
        "study_split": study_results,
    }

    out_path = Path("models") / "dkl_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {out_path}")
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
