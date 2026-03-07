#!/usr/bin/env python3
"""GroupDRO surrogate on LANTERN IL-only PCs."""

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

    study_sizes = df.groupby("study_id").size()
    keep_ids = study_sizes[study_sizes >= min_n].index
    df = df[df["study_id"].isin(keep_ids)].reset_index(drop=True)

    encoded, _ = encode_lantern_il(df, reduction="pca")
    feat_cols = lantern_il_feature_cols(encoded)

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


def train_groupdro(X, y, study_ids, train_mask, eta=0.01, epochs=200, lr=1e-3):
    device = torch.device("cpu")
    X_t = torch.tensor(X, device=device)
    y_t = torch.tensor(y, device=device)

    model = MLP(X.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_idx = np.where(train_mask)[0]
    train_studies = np.unique(study_ids[train_idx])
    q = np.ones(len(train_studies)) / len(train_studies)

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()

        losses = []
        for sid in train_studies:
            idx = train_idx[study_ids[train_idx] == sid]
            pred = model(X_t[idx])
            loss = torch.mean((pred - y_t[idx]) ** 2)
            losses.append(loss)

        losses_t = torch.stack(losses)
        # Update q using detached losses (Sagawa et al. 2020, Algorithm 1):
        # q is updated via exponentiated gradient on the dual variable,
        # separate from the model parameter gradient step.
        with torch.no_grad():
            q = q * np.exp(eta * losses_t.cpu().numpy())
            q = q / q.sum()
        q_t = torch.tensor(q, dtype=losses_t.dtype, device=losses_t.device)

        loss = torch.sum(q_t.detach() * losses_t)
        loss.backward()
        optimizer.step()

    return model, train_studies, q


def eval_model(model, X, y, mask):
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X)).cpu().numpy()
    return float(r2_score(y[mask], preds[mask])), preds


def worst_k_study_r2(y_true, preds, study_ids, mask, k=5):
    r2s = []
    for sid in np.unique(study_ids[mask]):
        idx = mask & (study_ids == sid)
        if idx.sum() < 2:
            continue
        r2s.append(r2_score(y_true[idx], preds[idx]))
    if not r2s:
        return float("nan")
    r2s = sorted(r2s)
    return float(np.mean(r2s[: min(k, len(r2s))]))


def main() -> int:
    X, y, study_ids, train_mask, test_mask = prepare_data(min_n=5)

    etas = [0.001, 0.01, 0.1]
    results = {}
    for eta in etas:
        model, train_studies, q = train_groupdro(X, y, study_ids, train_mask, eta=eta)
        r2_test, preds = eval_model(model, X, y, test_mask)
        r2_train, _ = eval_model(model, X, y, train_mask)
        worst5 = worst_k_study_r2(y, preds, study_ids, test_mask, k=5)
        results[str(eta)] = {
            "r2_train": r2_train,
            "r2_test": r2_test,
            "worst5_r2": worst5,
            "top_weighted_studies": {
                sid: float(q[i]) for i, sid in enumerate(train_studies)
            },
        }
        print(f"eta={eta}: r2_test={r2_test:.3f} worst5={worst5:.3f}")

    out_path = Path("models") / "groupdro_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
