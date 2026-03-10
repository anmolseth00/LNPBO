#!/usr/bin/env python3
"""Deep Ensemble surrogate for uncertainty quantification.

Trains M independent neural networks with different random initializations
and optional bootstrap sampling. Uncertainty is estimated from ensemble
disagreement (standard deviation of member predictions).

Reference:
    Lakshminarayanan, B., Pritzel, A. & Blundell, C. (2017).
    "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles."
    NeurIPS 2017. arXiv:1612.01474.
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
    prepare_study_data,
    study_split,
)
from LNPBO.models.splits import scaffold_split
from LNPBO.models.surrogate_mlp import SurrogateMLP


class DeepEnsemble:
    """Deep ensemble of independently trained MLPs.

    Reference:
        Lakshminarayanan, B., Pritzel, A. & Blundell, C. (2017).
        "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles."
        NeurIPS 2017. arXiv:1612.01474.

    Parameters
    ----------
    input_dim : int
        Number of input features.
    n_models : int
        Number of ensemble members (M).
    hidden_dims : tuple[int, ...]
        Hidden layer sizes (used by SurrogateMLP: fixed at 256, 128).
    epochs : int
        Training epochs per member.
    lr : float
        Learning rate for Adam optimizer.
    """

    def __init__(
        self,
        input_dim: int,
        n_models: int = 5,
        hidden_dims: tuple[int, ...] = (256, 128),
        epochs: int = 100,
        lr: float = 1e-3,
    ):
        self.input_dim = input_dim
        self.n_models = n_models
        self.hidden_dims = hidden_dims
        self.epochs = epochs
        self.lr = lr
        self.models: list[SurrogateMLP] = []

    def fit(self, X: np.ndarray, y: np.ndarray, bootstrap: bool = True, seed: int = 42):
        """Train each ensemble member independently.

        Each member gets a different random initialization (via distinct seed).
        If bootstrap=True, each member also trains on a different bootstrap
        sample of the data.
        """
        self.models = []
        rng = np.random.RandomState(seed)
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)

        for m in range(self.n_models):
            member_seed = rng.randint(0, 2**31)
            torch.manual_seed(member_seed)
            model = SurrogateMLP(self.input_dim)
            opt = torch.optim.Adam(model.parameters(), lr=self.lr)

            if bootstrap:
                boot_rng = np.random.RandomState(member_seed)
                idx = boot_rng.choice(len(X), size=len(X), replace=True)
                X_b = X_t[idx]
                y_b = y_t[idx]
            else:
                X_b = X_t
                y_b = y_t

            model.train()
            for _ in range(self.epochs):
                opt.zero_grad()
                pred = model(X_b)
                loss = torch.nn.functional.mse_loss(pred, y_b)
                loss.backward()
                opt.step()

            model.eval()
            self.models.append(model)

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return mean and standard deviation across ensemble members."""
        X_t = torch.tensor(X, dtype=torch.float32)
        preds = []
        with torch.no_grad():
            for model in self.models:
                preds.append(model(X_t).cpu().numpy())
        preds = np.array(preds)  # (n_models, n_samples)
        mu = preds.mean(axis=0)
        sigma = preds.std(axis=0)
        return mu, sigma


def _coverage(y_true, mu, sigma, z):
    """Fraction of true values within mu +/- z*sigma."""
    return float(np.mean((y_true >= mu - z * sigma) & (y_true <= mu + z * sigma)))


def _evaluate_ensemble(
    X_train, y_train, X_test, y_test,
    n_models, bootstrap, epochs, lr, seed,
    study_test=None,
):
    """Train a DeepEnsemble and compute evaluation metrics."""
    ensemble = DeepEnsemble(
        input_dim=X_train.shape[1],
        n_models=n_models,
        epochs=epochs,
        lr=lr,
    )
    t0 = time.time()
    ensemble.fit(X_train, y_train, bootstrap=bootstrap, seed=seed)
    train_time = time.time() - t0

    mu, sigma = ensemble.predict(X_test)
    r2 = float(r2_score(y_test, mu))

    # Coverage at 68% (1-sigma) and 90% (~1.645-sigma)
    cov_68 = _coverage(y_test, mu, sigma, 1.0)
    cov_90 = _coverage(y_test, mu, sigma, 1.645)

    result = {
        "r2": r2,
        "coverage_68": cov_68,
        "coverage_90": cov_90,
        "mean_sigma": float(np.mean(sigma)),
        "median_sigma": float(np.median(sigma)),
        "train_time_s": round(train_time, 2),
        "n_models": n_models,
        "bootstrap": bootstrap,
        "epochs": epochs,
    }

    # Per-study coverage (study-level split only)
    if study_test is not None:
        per_study = {}
        for sid in np.unique(study_test):
            idx = study_test == sid
            n = int(idx.sum())
            if n == 0:
                continue
            per_study[str(sid)] = {
                "n": n,
                "r2": float(r2_score(y_test[idx], mu[idx])) if n > 1 else None,
                "coverage_68": _coverage(y_test[idx], mu[idx], sigma[idx], 1.0),
                "coverage_90": _coverage(y_test[idx], mu[idx], sigma[idx], 1.645),
                "mean_sigma": float(np.mean(sigma[idx])),
            }
        result["per_study"] = per_study

    return result


def main() -> int:
    seeds = [42, 123, 456, 789, 2024]
    m_values = [3, 5, 10]
    bootstrap_options = [True, False]
    epochs = 100
    lr = 1e-3

    # --- Load data ---
    df = load_lnpdb_clean(drop_duplicates=False)
    df = df.dropna(subset=["IL_SMILES", "Experiment_value"]).reset_index(drop=True)

    study_ids = df["study_id"].astype(str).values
    smiles = df["IL_SMILES"].tolist()

    results = {"scaffold_split": {}, "study_split": {}}

    # ==========================================
    # Scaffold split
    # ==========================================
    print("=" * 60)
    print("SCAFFOLD SPLIT")
    print("=" * 60)

    for seed in seeds:
        train_idx, val_idx, test_idx = scaffold_split(smiles, sizes=(0.8, 0.1, 0.1), seed=seed)
        train_idx = train_idx + val_idx  # merge train+val for simplicity

        train_enc, test_enc, _ = encode_lantern_il(df, train_idx=train_idx, test_idx=test_idx, reduction="pca")
        feat_cols = lantern_il_feature_cols(train_enc)
        train_enc.index = train_idx
        test_enc.index = test_idx

        scaler = StandardScaler().fit(train_enc[feat_cols].values)
        X_train = scaler.transform(train_enc[feat_cols].values)
        y_train = train_enc["Experiment_value"].values
        X_test = scaler.transform(test_enc[feat_cols].values)
        y_test = test_enc["Experiment_value"].values

        for n_models in m_values:
            for bootstrap in bootstrap_options:
                key = f"M{n_models}_boot{bootstrap}_seed{seed}"
                print(f"  Scaffold | {key}...", flush=True)
                res = _evaluate_ensemble(
                    X_train, y_train, X_test, y_test,
                    n_models=n_models, bootstrap=bootstrap,
                    epochs=epochs, lr=lr, seed=seed,
                )
                results["scaffold_split"][key] = res
                print(
                    f"    R2={res['r2']:.3f}  "
                    f"cov68={res['coverage_68']:.3f}  "
                    f"cov90={res['coverage_90']:.3f}  "
                    f"sigma={res['mean_sigma']:.4f}  "
                    f"time={res['train_time_s']:.1f}s"
                )

    # ==========================================
    # Study-level split
    # ==========================================
    print("\n" + "=" * 60)
    print("STUDY-LEVEL SPLIT")
    print("=" * 60)

    train_ids, test_ids = study_split(df, seed=42)
    train_mask = df["study_id"].isin(train_ids).values
    test_mask = df["study_id"].isin(test_ids).values
    study_test = study_ids[test_mask]

    s_train_idx = np.flatnonzero(train_mask).tolist()
    s_test_idx = np.flatnonzero(test_mask).tolist()
    train_enc_s, test_enc_s, _ = encode_lantern_il(df, train_idx=s_train_idx, test_idx=s_test_idx, reduction="pca")
    feat_cols_s = lantern_il_feature_cols(train_enc_s)
    train_enc_s.index = s_train_idx
    test_enc_s.index = s_test_idx

    scaler_study = StandardScaler().fit(train_enc_s[feat_cols_s].values)
    X_train_study = scaler_study.transform(train_enc_s[feat_cols_s].values)
    y_train_study = train_enc_s["Experiment_value"].values
    X_test_study = scaler_study.transform(test_enc_s[feat_cols_s].values)
    y_test_study = test_enc_s["Experiment_value"].values

    for n_models in m_values:
        for bootstrap in bootstrap_options:
            key = f"M{n_models}_boot{bootstrap}"
            print(f"  Study | {key}...", flush=True)
            res = _evaluate_ensemble(
                X_train_study, y_train_study, X_test_study, y_test_study,
                n_models=n_models, bootstrap=bootstrap,
                epochs=epochs, lr=lr, seed=42,
                study_test=study_test,
            )
            results["study_split"][key] = res
            print(
                f"    R2={res['r2']:.3f}  "
                f"cov68={res['coverage_68']:.3f}  "
                f"cov90={res['coverage_90']:.3f}  "
                f"sigma={res['mean_sigma']:.4f}  "
                f"time={res['train_time_s']:.1f}s"
            )

    # ==========================================
    # Summary across seeds (scaffold split)
    # ==========================================
    print("\n" + "=" * 60)
    print("SUMMARY (scaffold split, mean +/- std across seeds)")
    print("=" * 60)

    summary = {}
    for n_models in m_values:
        for bootstrap in bootstrap_options:
            config_key = f"M{n_models}_boot{bootstrap}"
            r2s, c68s, c90s = [], [], []
            for seed in seeds:
                key = f"M{n_models}_boot{bootstrap}_seed{seed}"
                r = results["scaffold_split"][key]
                r2s.append(r["r2"])
                c68s.append(r["coverage_68"])
                c90s.append(r["coverage_90"])
            summary[config_key] = {
                "r2_mean": float(np.mean(r2s)),
                "r2_std": float(np.std(r2s)),
                "coverage_68_mean": float(np.mean(c68s)),
                "coverage_68_std": float(np.std(c68s)),
                "coverage_90_mean": float(np.mean(c90s)),
                "coverage_90_std": float(np.std(c90s)),
            }
            print(
                f"  {config_key}: "
                f"R2={np.mean(r2s):.3f}+/-{np.std(r2s):.3f}  "
                f"cov68={np.mean(c68s):.3f}+/-{np.std(c68s):.3f}  "
                f"cov90={np.mean(c90s):.3f}+/-{np.std(c90s):.3f}"
            )

    results["scaffold_summary"] = summary

    out_path = Path("models") / "deep_ensemble_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
