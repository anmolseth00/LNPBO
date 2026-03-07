#!/usr/bin/env python3
"""Split conformal prediction for multiple surrogates."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from diagnostics.utils import load_lnpdb_clean, encode_lantern_il, lantern_il_feature_cols, summarize_study_assay_types


def _study_split(df, seed=42):
    rng = np.random.RandomState(seed)
    study_to_type = {}
    for sid, sdf in df.groupby("study_id"):
        assay_type, _ = summarize_study_assay_types(sdf)
        study_to_type[sid] = assay_type

    train_ids = set()
    test_ids = set()
    for assay_type in sorted(set(study_to_type.values())):
        ids = [sid for sid, at in study_to_type.items() if at == assay_type]
        rng.shuffle(ids)
        cut = max(1, int(0.8 * len(ids))) if len(ids) > 1 else len(ids)
        train_ids.update(ids[:cut])
        test_ids.update(ids[cut:])

    train_mask = df["study_id"].isin(train_ids)
    test_mask = df["study_id"].isin(test_ids)
    return train_mask.values, test_mask.values


def _split_calibration(mask, seed=42):
    rng = np.random.RandomState(seed)
    idx = np.where(mask)[0]
    rng.shuffle(idx)
    n_cal = max(1, int(0.2 * len(idx)))
    cal_idx = idx[:n_cal]
    train_idx = idx[n_cal:]
    return train_idx, cal_idx


def _quantile(residuals, alpha):
    n = len(residuals)
    # Vovk et al.: quantile level for (1-alpha) coverage guarantee
    q = np.ceil((n + 1) * (1 - alpha)) / (n + 1)
    try:
        return float(np.quantile(residuals, q, method="higher"))
    except TypeError:
        return float(np.quantile(residuals, q, interpolation="higher"))


def fit_xgb(X_train, y_train):
    model = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42, n_jobs=-1, verbosity=0)
    model.fit(X_train, y_train)
    return model


def fit_sparse_gp(X_train, y_train, epochs=20, batch_size=1024):
    import gpytorch
    from torch.utils.data import DataLoader, TensorDataset

    X = torch.tensor(X_train, dtype=torch.float32)
    y = torch.tensor(y_train, dtype=torch.float32)

    m = min(256, X.size(0))
    rng = torch.Generator().manual_seed(42)
    perm = torch.randperm(X.size(0), generator=rng)
    inducing_points = X[perm[:m]].clone()

    class SparseGP(gpytorch.models.ApproximateGP):
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

    model = SparseGP(inducing_points)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=X.size(0))

    loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True)
    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            with gpytorch.settings.cholesky_jitter(1e-4):
                output = model(xb)
                loss = -mll(output, yb)
            loss.backward()
            optimizer.step()

    return model, likelihood


def predict_sparse_gp(model, likelihood, X):
    model.eval()
    likelihood.eval()
    with torch.no_grad():
        pred = likelihood(model(torch.tensor(X, dtype=torch.float32)))
    return pred.mean.squeeze(-1).cpu().numpy()


def fit_vrex_mlp(X, y, study_ids, train_idx, epochs=100, lr=1e-3, lambda_rex=1.0):
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)

    model = torch.nn.Sequential(
        torch.nn.Linear(X.shape[1], 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 1),
    )
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    train_studies = np.unique(study_ids[train_idx])

    for _ in range(epochs):
        opt.zero_grad()
        losses = []
        for sid in train_studies:
            idx = train_idx[study_ids[train_idx] == sid]
            if len(idx) == 0:
                continue
            pred = model(X_t[idx]).squeeze(-1)
            loss = torch.mean((pred - y_t[idx]) ** 2)
            losses.append(loss)
        if not losses:
            break
        losses_t = torch.stack(losses)
        loss = losses_t.mean() + lambda_rex * losses_t.var(unbiased=False)
        loss.backward()
        opt.step()
    return model


def predict_mlp(model, X):
    with torch.no_grad():
        return model(torch.tensor(X, dtype=torch.float32)).squeeze(-1).cpu().numpy()


def main() -> int:
    df = load_lnpdb_clean(drop_duplicates=False)
    df = df.dropna(subset=["IL_SMILES", "Experiment_value"]).reset_index(drop=True)

    encoded, _ = encode_lantern_il(df, reduction="pca")
    feat_cols = lantern_il_feature_cols(encoded)

    X = encoded[feat_cols].values
    y = encoded["Experiment_value"].values
    study_ids = df["study_id"].astype(str).values

    train_mask, test_mask = _study_split(df, seed=42)
    train_idx, cal_idx = _split_calibration(train_mask, seed=42)

    scaler = StandardScaler().fit(X[train_idx])
    X = scaler.transform(X)

    X_train, y_train = X[train_idx], y[train_idx]
    X_cal, y_cal = X[cal_idx], y[cal_idx]
    X_test, y_test = X[test_mask], y[test_mask]
    study_test = study_ids[test_mask]

    results = {}
    alpha = 0.1

    per_study = {}

    # XGB
    xgb = fit_xgb(X_train, y_train)
    pred_cal = xgb.predict(X_cal)
    q = _quantile(np.abs(y_cal - pred_cal), alpha)
    pred_test = xgb.predict(X_test)
    cov = np.mean((y_test >= pred_test - q) & (y_test <= pred_test + q))
    results["xgb"] = {
        "coverage_90": float(cov),
        "interval_width": float(2 * q),
        "r2_test": float(r2_score(y_test, pred_test)),
    }
    for sid in np.unique(study_test):
        idx = study_test == sid
        per_study.setdefault(str(sid), {})["coverage_90_xgb"] = float(
            np.mean((y_test[idx] >= pred_test[idx] - q) & (y_test[idx] <= pred_test[idx] + q))
        )

    # Sparse GP
    gp, gp_likelihood = fit_sparse_gp(X_train, y_train, epochs=20, batch_size=1024)
    pred_cal = predict_sparse_gp(gp, gp_likelihood, X_cal)
    q = _quantile(np.abs(y_cal - pred_cal), alpha)
    pred_test = predict_sparse_gp(gp, gp_likelihood, X_test)
    cov = np.mean((y_test >= pred_test - q) & (y_test <= pred_test + q))
    results["gp"] = {
        "coverage_90": float(cov),
        "interval_width": float(2 * q),
        "r2_test": float(r2_score(y_test, pred_test)),
    }
    for sid in np.unique(study_test):
        idx = study_test == sid
        per_study.setdefault(str(sid), {})["coverage_90_gp"] = float(
            np.mean((y_test[idx] >= pred_test[idx] - q) & (y_test[idx] <= pred_test[idx] + q))
        )

    # V-REx MLP
    vrex = fit_vrex_mlp(X, y, study_ids, train_idx, epochs=50, lambda_rex=1.0)
    pred_cal = predict_mlp(vrex, X_cal)
    q = _quantile(np.abs(y_cal - pred_cal), alpha)
    pred_test = predict_mlp(vrex, X_test)
    cov = np.mean((y_test >= pred_test - q) & (y_test <= pred_test + q))
    results["vrex"] = {
        "coverage_90": float(cov),
        "interval_width": float(2 * q),
        "r2_test": float(r2_score(y_test, pred_test)),
    }
    for sid in np.unique(study_test):
        idx = study_test == sid
        per_study.setdefault(str(sid), {})["coverage_90_vrex"] = float(
            np.mean((y_test[idx] >= pred_test[idx] - q) & (y_test[idx] <= pred_test[idx] + q))
        )
        per_study[str(sid)]["n"] = int(idx.sum())

    results["per_study"] = per_study

    out_path = Path("models") / "conformal_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"Saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
