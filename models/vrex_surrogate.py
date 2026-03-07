#!/usr/bin/env python3
"""V-REx surrogate on LANTERN IL-only PCs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import r2_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from diagnostics.utils import encode_lantern_il, lantern_il_feature_cols, load_lnpdb_clean, summarize_study_assay_types


def _study_split(study_ids, study_to_type, seed=42):
    rng = np.random.RandomState(seed)
    train_ids = set()
    test_ids = set()
    for assay_type in sorted(set(study_to_type.values())):
        ids = [sid for sid, at in study_to_type.items() if at == assay_type]
        rng.shuffle(ids)
        cut = max(1, int(0.8 * len(ids))) if len(ids) > 1 else len(ids)
        train_ids.update(ids[:cut])
        test_ids.update(ids[cut:])
    return train_ids, test_ids


def prepare_data(min_n=5):
    df = load_lnpdb_clean(drop_duplicates=False)
    df = df.dropna(subset=["IL_SMILES", "Experiment_value"]).reset_index(drop=True)

    # Filter studies with enough data
    study_sizes = df.groupby("study_id").size()
    keep_ids = study_sizes[study_sizes >= min_n].index
    df = df[df["study_id"].isin(keep_ids)].reset_index(drop=True)

    # Encode LANTERN IL-only
    encoded, _ = encode_lantern_il(df, reduction="pca")
    feat_cols = lantern_il_feature_cols(encoded)

    # Study types
    study_to_type = {}
    for sid, sdf in df.groupby("study_id"):
        assay_type, _ = summarize_study_assay_types(sdf)
        study_to_type[sid] = assay_type

    train_ids, test_ids = _study_split(df["study_id"].unique(), study_to_type, seed=42)

    train_mask = df["study_id"].isin(train_ids)
    test_mask = df["study_id"].isin(test_ids)

    X = encoded[feat_cols].values.astype(np.float32)
    y = encoded["Experiment_value"].values.astype(np.float32)
    study_ids = df["study_id"].astype(str).values

    return X, y, study_ids, train_mask.values, test_mask.values


class MLP(torch.nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_vrex(X, y, study_ids, train_mask, lambda_rex, epochs=200, lr=1e-3):
    device = torch.device("cpu")
    X_t = torch.tensor(X, device=device)
    y_t = torch.tensor(y, device=device)

    model = MLP(X.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_idx = np.where(train_mask)[0]
    train_studies = np.unique(study_ids[train_idx])

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()

        losses = []
        for sid in train_studies:
            idx = train_idx[study_ids[train_idx] == sid]
            if len(idx) == 0:
                continue
            pred = model(X_t[idx])
            loss = torch.mean((pred - y_t[idx]) ** 2)
            losses.append(loss)

        if not losses:
            break

        losses_t = torch.stack(losses)
        loss = losses_t.mean() + lambda_rex * losses_t.var(unbiased=False)
        loss.backward()
        optimizer.step()

    return model


def eval_model(model, X, y, mask):
    device = torch.device("cpu")
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X, device=device)).cpu().numpy()
    return float(r2_score(y[mask], preds[mask]))


def main() -> int:
    X, y, study_ids, train_mask, test_mask = prepare_data(min_n=5)

    lambdas = [0.0, 0.1, 1.0, 10.0, 100.0]
    results = {}
    for lam in lambdas:
        model = train_vrex(X, y, study_ids, train_mask, lambda_rex=lam)
        r2_test = eval_model(model, X, y, test_mask)
        r2_train = eval_model(model, X, y, train_mask)
        results[str(lam)] = {
            "r2_train": r2_train,
            "r2_test": r2_test,
        }
        print(f"lambda={lam}: r2_train={r2_train:.3f} r2_test={r2_test:.3f}")

    out_path = Path("models") / "vrex_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
