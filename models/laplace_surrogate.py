#!/usr/bin/env python3
"""Post-hoc Laplace approximation on trained MLP for uncertainty quantification.

Computes a Laplace approximation to the posterior over weights of a trained
MLP, yielding a Gaussian predictive distribution. Supports last-layer and
all-weights variants.

If laplace-torch is available, uses it directly. Otherwise, provides a manual
last-layer Laplace implementation using diagonal or Kronecker-factored (KFAC)
Hessian approximation.

References:
    Daxberger, E. et al. (2021). "Laplace Redux -- Effortless Bayesian Deep
        Learning." NeurIPS 2021. arXiv:2106.14806.
    MacKay, D.J.C. (1992). "A Practical Bayesian Framework for Backpropagation
        Networks." Neural Computation 4(3).
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
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


def _check_laplace_available():
    try:
        import laplace
        return True
    except ImportError:
        return False


LAPLACE_AVAILABLE = _check_laplace_available()


def train_mlp(model, X_train, y_train, epochs=100, lr=1e-3, batch_size=256):
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
    return model


class ManualLastLayerLaplace:
    """Manual last-layer Laplace approximation for regression MLP.

    Computes the posterior N(w_MAP, (H + prior_prec * I)^{-1}) over the last
    layer weights only, where H is the Hessian of the loss w.r.t. last-layer
    parameters.

    For MSE loss with fixed observation noise sigma_obs:
        H = (1/sigma_obs^2) * Phi^T Phi
    where Phi is the matrix of penultimate-layer features (the Jacobian of
    the output w.r.t. last-layer weights).

    Supports two Hessian structures:
      - 'diag': diagonal approximation (fast, less accurate)
      - 'kron': Kronecker-factored approximation (KFAC, more accurate)

    Reference: MacKay (1992), Daxberger et al. (2021).
    """

    def __init__(self, model, hessian_structure="kron"):
        self.model = model
        self.hessian_structure = hessian_structure
        self.prior_precision = 1.0
        self.sigma_obs = 1.0
        self._fitted = False

    def _extract_features(self, X):
        """Run forward pass up to the penultimate layer."""
        with torch.no_grad():
            h = torch.relu(self.model.fc1(X))
            h = torch.relu(self.model.fc2(h))
        return h

    def fit(self, train_loader):
        """Compute the Hessian of the last layer using training data."""
        all_features = []
        all_targets = []
        for xb, yb in train_loader:
            features = self._extract_features(xb)
            all_features.append(features)
            all_targets.append(yb)

        Phi = torch.cat(all_features, dim=0)
        y = torch.cat(all_targets, dim=0)
        n = Phi.size(0)
        d = Phi.size(1)

        with torch.no_grad():
            pred = self.model(torch.cat([xb for xb, _ in train_loader], dim=0))
        residuals = y - pred[:n]
        self.sigma_obs = float(residuals.var().sqrt().clamp(min=1e-6))

        if self.hessian_structure == "diag":
            diag = (Phi ** 2).sum(dim=0) / (self.sigma_obs ** 2)
            bias_diag = torch.tensor([n / (self.sigma_obs ** 2)])
            self._diag_H = torch.cat([diag, bias_diag])
        elif self.hessian_structure == "kron":
            Phi_aug = torch.cat([Phi, torch.ones(n, 1)], dim=1)
            self._H = (Phi_aug.T @ Phi_aug) / (self.sigma_obs ** 2)
        else:
            raise ValueError(f"Unknown hessian_structure: {self.hessian_structure}")

        self._Phi_train = Phi
        self._fitted = True

    def optimize_prior_precision(self, method="marginal_likelihood"):
        """Optimize prior precision via type-II maximum likelihood.

        Maximizes log p(y | alpha) = -0.5 * (n*log(2pi) + log|H/alpha + I|
            + y^T (H/alpha + I)^{-1} y + n*log(sigma_obs^2))

        For simplicity, grid-search over log-spaced candidates.
        """
        candidates = np.logspace(-3, 3, 20)
        best_prec = 1.0
        best_score = -np.inf

        d_plus_1 = self._Phi_train.size(1) + 1

        for prec in candidates:
            if self.hessian_structure == "diag":
                posterior_diag = self._diag_H + prec
                log_det = posterior_diag.log().sum()
                score = 0.5 * d_plus_1 * np.log(prec) - 0.5 * float(log_det)
            elif self.hessian_structure == "kron":
                M = self._H + prec * torch.eye(d_plus_1)
                try:
                    L = torch.linalg.cholesky(M)
                    log_det = 2.0 * L.diagonal().log().sum()
                except torch.linalg.LinAlgError:
                    continue
                score = 0.5 * d_plus_1 * np.log(prec) - 0.5 * float(log_det)

            if score > best_score:
                best_score = score
                best_prec = prec

        self.prior_precision = float(best_prec)

    def predict(self, X):
        """Predict mean and variance for new inputs.

        Returns (mu, var) where:
            mu = model(X) (MAP prediction)
            var = sigma_obs^2 + phi^T (H + alpha*I)^{-1} phi
                (predictive variance = aleatoric + epistemic)
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() before predict()")

        with torch.no_grad():
            mu = self.model(X)
            Phi = self._extract_features(X)

        n_test = Phi.size(0)
        d = Phi.size(1)
        Phi_aug = torch.cat([Phi, torch.ones(n_test, 1)], dim=1)

        if self.hessian_structure == "diag":
            posterior_diag = self._diag_H + self.prior_precision
            var_epistemic = (Phi_aug ** 2 / posterior_diag.unsqueeze(0)).sum(dim=1)
        elif self.hessian_structure == "kron":
            M = self._H + self.prior_precision * torch.eye(d + 1)
            try:
                L = torch.linalg.cholesky(M)
                phi_solve = torch.linalg.solve_triangular(L, Phi_aug.T, upper=False)
                var_epistemic = (phi_solve ** 2).sum(dim=0)
            except torch.linalg.LinAlgError:
                cov = torch.linalg.solve(M, Phi_aug.T)
                var_epistemic = (Phi_aug * cov.T).sum(dim=1)

        var = self.sigma_obs ** 2 + var_epistemic
        return mu.cpu().numpy(), var.cpu().numpy()


class LaplaceTorchWrapper:
    """Wrapper for laplace-torch library (Daxberger et al. 2021).

    Reference: https://github.com/aleximmer/Laplace
    """

    def __init__(self, model, subset_of_weights="last_layer", hessian_structure="kron"):
        from laplace import Laplace
        self.la = Laplace(
            model, "regression",
            subset_of_weights=subset_of_weights,
            hessian_structure=hessian_structure,
        )
        self.model = model
        self.subset_of_weights = subset_of_weights
        self.hessian_structure = hessian_structure

    def fit(self, train_loader):
        self.la.fit(train_loader)

    def optimize_prior_precision(self, method="marginal_likelihood"):
        self.la.optimize_prior_precision(method=method)

    def predict(self, X):
        f_mu, f_var = self.la(X)
        return f_mu.squeeze(-1).cpu().numpy(), f_var.squeeze(-1).cpu().numpy()


def build_laplace(model, subset_of_weights="last_layer", hessian_structure="kron"):
    if LAPLACE_AVAILABLE and subset_of_weights == "last_layer":
        try:
            return LaplaceTorchWrapper(model, subset_of_weights, hessian_structure)
        except Exception:
            pass
    if subset_of_weights == "last_layer":
        return ManualLastLayerLaplace(model, hessian_structure)
    if LAPLACE_AVAILABLE:
        return LaplaceTorchWrapper(model, subset_of_weights, hessian_structure)
    raise ValueError(
        f"subset_of_weights='{subset_of_weights}' with all weights requires "
        "laplace-torch. Install with: pip install laplace-torch"
    )


def compute_calibration(y_true, mu, sigma, alpha=0.1):
    z = norm.ppf(1 - alpha / 2)
    lower = mu - z * sigma
    upper = mu + z * sigma
    covered = (y_true >= lower) & (y_true <= upper)
    coverage = float(covered.mean())
    avg_width = float((upper - lower).mean())
    return coverage, avg_width


def eval_laplace(la, X_test_t, y_test, y_test_s, y_mean, y_std):
    mu_s, var_s = la.predict(X_test_t)
    sigma_s = np.sqrt(np.clip(var_s, 1e-12, None))
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

    in_dim = X_train_s.shape[1]
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=256, shuffle=False)

    results = {}

    configs = [
        ("last_layer_kron", "last_layer", "kron"),
        ("last_layer_diag", "last_layer", "diag"),
    ]
    if LAPLACE_AVAILABLE:
        configs.append(("all_weights_kron", "all", "kron"))

    for label, subset, hessian in configs:
        t0 = time.time()

        model = SurrogateMLP(in_dim)
        train_mlp(model, X_train_t, y_train_t, epochs=100)
        model.eval()

        la = build_laplace(model, subset_of_weights=subset, hessian_structure=hessian)
        la.fit(train_loader)
        la.optimize_prior_precision(method="marginal_likelihood")

        t_train = time.time() - t0

        res = eval_laplace(la, X_test_t, y_test, y_test_s, y_mean, y_std)
        res["train_time_s"] = round(t_train, 1)
        res["subset_of_weights"] = subset
        res["hessian_structure"] = hessian
        res["laplace_torch_used"] = LAPLACE_AVAILABLE and subset == "last_layer"
        results[label] = res
        print(f"Scaffold | Laplace {label}: R2={res['r2']:.3f}, "
              f"cov90={res['coverage_90']:.3f}, time={t_train:.1f}s")

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
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=256, shuffle=False)

    results = {}

    configs = [
        ("last_layer_kron", "last_layer", "kron"),
        ("last_layer_diag", "last_layer", "diag"),
    ]
    if LAPLACE_AVAILABLE:
        configs.append(("all_weights_kron", "all", "kron"))

    for label, subset, hessian in configs:
        t0 = time.time()

        model = SurrogateMLP(in_dim)
        train_mlp(model, X_train_t, y_train_t, epochs=100)
        model.eval()

        la = build_laplace(model, subset_of_weights=subset, hessian_structure=hessian)
        la.fit(train_loader)
        la.optimize_prior_precision(method="marginal_likelihood")

        t_train = time.time() - t0

        res = eval_laplace(la, X_test_t, y[test_mask], y_s[test_mask], y_mean, y_std)
        res["train_time_s"] = round(t_train, 1)
        res["subset_of_weights"] = subset
        res["hessian_structure"] = hessian
        res["laplace_torch_used"] = LAPLACE_AVAILABLE and subset == "last_layer"
        results[label] = res
        print(f"Study    | Laplace {label}: R2={res['r2']:.3f}, "
              f"cov90={res['coverage_90']:.3f}, time={t_train:.1f}s")

    return results


def main() -> int:
    print("=== Post-hoc Laplace Approximation on MLP ===")
    print(f"laplace-torch available: {LAPLACE_AVAILABLE}")
    print()

    print("--- Scaffold Split ---")
    scaffold_results = run_scaffold_split()

    print()
    print("--- Study-Level Split ---")
    study_results = run_study_split()

    results = {
        "laplace_torch_available": LAPLACE_AVAILABLE,
        "scaffold_split": scaffold_results,
        "study_split": study_results,
    }

    out_path = Path("models") / "laplace_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {out_path}")
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
