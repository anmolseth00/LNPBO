#!/usr/bin/env python3
"""Spectral-Normalized Neural GP (SNGP) surrogate.

SNGP modifies a standard MLP with:
  (a) spectral normalization on hidden layers to enforce distance-awareness
  (b) a Random Fourier Feature GP output layer (Rahimi & Recht 2007)

Distance-awareness ensures that novel IL scaffolds far from training data
receive high predictive uncertainty, which is critical for BO exploration.

References:
    Liu, J.Z. et al. (2023). "A Simple Approach to Improve Single-Model Deep
        Uncertainty via Distance-Awareness." JMLR 24(42), 1-63.
    Rahimi, A. & Recht, B. (2007). "Random Features for Large-Scale Kernel
        Machines." NeurIPS 2007.
"""

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import norm
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from torch.nn.utils import spectral_norm
from torch.utils.data import DataLoader, TensorDataset

from LNPBO.diagnostics.utils import (
    encode_lantern_il,
    lantern_il_feature_cols,
    load_lnpdb_clean,
    prepare_study_data,
)
from LNPBO.models.splits import scaffold_split


class RandomFourierFeatures(nn.Module):
    """Random Fourier Features for RBF kernel approximation.

    Reference: Rahimi & Recht (2007), "Random Features for Large-Scale
    Kernel Machines", NeurIPS 2007.
    """

    def __init__(self, in_dim, n_features=1024, lengthscale=1.0):
        super().__init__()
        self.register_buffer(
            "W", torch.randn(in_dim, n_features) / lengthscale
        )
        self.register_buffer(
            "b", torch.rand(n_features) * 2 * math.pi
        )
        self.n_features = n_features

    def forward(self, x):
        z = x @ self.W + self.b
        return math.sqrt(2.0 / self.n_features) * torch.cos(z)


class SNGP(nn.Module):
    """Spectral-Normalized Neural GP.

    Architecture:
      1. Spectral-normalized hidden layers preserve input distance in
         representation space (Lipschitz constraint).
      2. Random Fourier Feature layer approximates an RBF kernel.
      3. Laplace approximation on the output layer gives posterior variance.

    Reference: Liu et al. (2023), JMLR 24(42).
    """

    def __init__(self, input_dim, hidden_dims=(256, 128), n_random_features=1024,
                 lengthscale=1.0, ridge_penalty=1.0):
        super().__init__()
        self.n_random_features = n_random_features
        self.ridge_penalty = ridge_penalty

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(spectral_norm(nn.Linear(prev_dim, h_dim)))
            layers.append(nn.ReLU())
            prev_dim = h_dim
        self.backbone = nn.Sequential(*layers)

        self.rff = RandomFourierFeatures(hidden_dims[-1], n_random_features, lengthscale)

        self.output_layer = nn.Linear(n_random_features, 1, bias=True)

        self.register_buffer(
            "precision", torch.eye(n_random_features) * ridge_penalty
        )
        self.register_buffer("feature_sum", torch.zeros(n_random_features, n_random_features))
        self.register_buffer("n_train", torch.tensor(0, dtype=torch.long))

    def forward(self, x):
        h = self.backbone(x)
        phi = self.rff(h)
        return self.output_layer(phi).squeeze(-1)

    def reset_precision(self):
        D = self.n_random_features
        self.precision.copy_(torch.eye(D, device=self.precision.device) * self.ridge_penalty)
        self.n_train.zero_()

    def update_precision(self, x):
        """Update the precision matrix with a batch of training features.

        After training, call this on the full training set (or in batches)
        to build the Laplace posterior covariance.
        """
        with torch.no_grad():
            h = self.backbone(x)
            phi = self.rff(h)
            self.precision.add_(phi.T @ phi)
            self.n_train.add_(phi.size(0))

    def predict_with_uncertainty(self, x):
        """Predict mean and variance using Laplace approximation on the GP layer.

        Returns (mu, sigma) where sigma is the predictive standard deviation.
        """
        self.eval()
        with torch.no_grad():
            h = self.backbone(x)
            phi = self.rff(h)
            mu = self.output_layer(phi).squeeze(-1)

            try:
                L = torch.linalg.cholesky(self.precision)
                phi_solve = torch.linalg.solve_triangular(L, phi.T, upper=False)
                var = (phi_solve ** 2).sum(dim=0)
            except torch.linalg.LinAlgError:
                cov = torch.linalg.solve(self.precision, phi.T)
                var = (phi * cov.T).sum(dim=-1)

        return mu.cpu().numpy(), var.sqrt().cpu().numpy()


def train_sngp(X_train, y_train, input_dim, hidden_dims=(256, 128),
               n_random_features=1024, epochs=100, lr=1e-3, batch_size=256,
               ridge_penalty=1.0, lengthscale=1.0):
    model = SNGP(input_dim, hidden_dims, n_random_features, lengthscale, ridge_penalty)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = nn.functional.mse_loss(pred, yb)
            loss.backward()
            optimizer.step()

    model.reset_precision()
    precision_loader = DataLoader(TensorDataset(X_train,), batch_size=1024, shuffle=False)
    for (xb,) in precision_loader:
        model.update_precision(xb)

    return model


def compute_calibration(y_true, mu, sigma, alpha=0.1):
    z = norm.ppf(1 - alpha / 2)
    lower = mu - z * sigma
    upper = mu + z * sigma
    covered = (y_true >= lower) & (y_true <= upper)
    coverage = float(covered.mean())
    avg_width = float((upper - lower).mean())
    return coverage, avg_width


def eval_sngp(model, X_test_t, y_test, y_test_s, y_mean, y_std):
    mu_s, sigma_s = model.predict_with_uncertainty(X_test_t)
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


def scaffold_uncertainty_analysis(model, df, train_idx, test_idx, X_test_s, y_test_s):
    """Compare uncertainty for seen vs novel Murcko scaffolds in the test set."""
    from rdkit import Chem
    from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles

    def get_scaffold(smi):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return ""
        try:
            return MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        except Exception:
            return ""

    train_smiles = df.iloc[train_idx]["IL_SMILES"].tolist()
    test_smiles = df.iloc[test_idx]["IL_SMILES"].tolist()

    train_scaffolds = set(get_scaffold(s) for s in train_smiles)

    test_scaffold_list = [get_scaffold(s) for s in test_smiles]
    seen_mask = np.array([s in train_scaffolds for s in test_scaffold_list])
    novel_mask = ~seen_mask

    X_test_t = torch.tensor(X_test_s, dtype=torch.float32)
    _, sigma = model.predict_with_uncertainty(X_test_t)

    result = {
        "n_seen": int(seen_mask.sum()),
        "n_novel": int(novel_mask.sum()),
        "n_unique_train_scaffolds": len(train_scaffolds),
    }
    if seen_mask.sum() > 0:
        result["mean_sigma_seen"] = float(sigma[seen_mask].mean())
        result["median_sigma_seen"] = float(np.median(sigma[seen_mask]))
    if novel_mask.sum() > 0:
        result["mean_sigma_novel"] = float(sigma[novel_mask].mean())
        result["median_sigma_novel"] = float(np.median(sigma[novel_mask]))
    if seen_mask.sum() > 0 and novel_mask.sum() > 0:
        result["novel_to_seen_ratio"] = float(sigma[novel_mask].mean() / sigma[seen_mask].mean())

    return result


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

    in_dim = X_train_s.shape[1]
    results = {}

    for n_rff in [512, 1024]:
        label = f"rff{n_rff}"
        t0 = time.time()
        model = train_sngp(X_train_t, y_train_t, in_dim,
                           n_random_features=n_rff, epochs=100, ridge_penalty=1.0)
        t_train = time.time() - t0

        res = eval_sngp(model, X_test_t, y_test, y_test_s, y_mean, y_std)
        res["train_time_s"] = round(t_train, 1)
        res["n_random_features"] = n_rff

        scaffold_uq = scaffold_uncertainty_analysis(model, df, train_idx, test_idx, X_test_s, y_test_s)
        res["scaffold_uncertainty"] = scaffold_uq

        results[label] = res
        print(f"Scaffold | SNGP {label}: R2={res['r2']:.3f}, "
              f"cov90={res['coverage_90']:.3f}, "
              f"novel/seen sigma={scaffold_uq.get('novel_to_seen_ratio', 'N/A')}, "
              f"time={t_train:.1f}s")

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

    in_dim = X_s.shape[1]
    results = {}

    for n_rff in [512, 1024]:
        label = f"rff{n_rff}"
        t0 = time.time()
        model = train_sngp(X_train_t, y_train_t, in_dim,
                           n_random_features=n_rff, epochs=100, ridge_penalty=1.0)
        t_train = time.time() - t0

        res = eval_sngp(model, X_test_t, y[test_mask], y_s[test_mask], y_mean, y_std)
        res["train_time_s"] = round(t_train, 1)
        res["n_random_features"] = n_rff
        results[label] = res
        print(f"Study    | SNGP {label}: R2={res['r2']:.3f}, "
              f"cov90={res['coverage_90']:.3f}, time={t_train:.1f}s")

    return results


def main() -> int:
    print("=== Spectral-Normalized Neural GP (SNGP) Surrogate ===")
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

    out_path = Path("models") / "sngp_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {out_path}")
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
