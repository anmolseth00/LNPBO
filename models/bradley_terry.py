#!/usr/bin/env python3
"""Bradley-Terry pairwise preference model using an MLP utility.

Reference:
    Bradley, R.A. & Terry, M.E. (1952). "Rank Analysis of Incomplete Block
    Designs: I. The Method of Paired Comparisons." Biometrika, 39(3/4), 324-345.
"""


import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from LNPBO.diagnostics.utils import prepare_study_data
from LNPBO.models.surrogate_mlp import SurrogateMLP


def _sample_pairs(indices, y, max_pairs, rng):
    n = len(indices)
    if n < 2:
        return [], []
    total_pairs = n * (n - 1) // 2
    n_pairs = min(max_pairs, total_pairs)
    pairs = []
    seen = set()
    max_attempts = n_pairs * 20
    attempts = 0
    while len(pairs) < n_pairs and attempts < max_attempts:
        attempts += 1
        i, j = rng.choice(indices, size=2, replace=False)
        if y[i] == y[j]:
            continue
        key = (min(i, j), max(i, j))
        if key in seen:
            continue
        seen.add(key)
        # Randomly assign order to avoid degenerate all-positive labels
        if rng.random() < 0.5:
            # winner first -> label 1
            if y[i] > y[j]:
                pairs.append((i, j, 1))
            else:
                pairs.append((j, i, 1))
        else:
            # loser first -> label 0
            if y[i] > y[j]:
                pairs.append((j, i, 0))
            else:
                pairs.append((i, j, 0))
    X1 = [(i, j) for i, j, _ in pairs]
    y1 = [lbl for _, _, lbl in pairs]
    return X1, y1


def build_pair_dataset(X, y, study_ids, mask, max_pairs=500, seed=42):
    rng = np.random.RandomState(seed)
    pairs = []
    labels = []
    for sid in np.unique(study_ids[mask]):
        idx = np.where(mask & (study_ids == sid))[0]
        pair_idx, pair_labels = _sample_pairs(idx, y, max_pairs, rng)
        pairs.extend(pair_idx)
        labels.extend(pair_labels)
    return np.array(pairs), np.array(labels)


def train_bt_model(X, y, study_ids, train_mask, epochs=20, batch_size=1024):
    pairs, labels = build_pair_dataset(X, y, study_ids, train_mask, max_pairs=500)
    device = torch.device("cpu")
    model = SurrogateMLP(X.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    X_t = torch.tensor(X, device=device)
    labels_t = torch.tensor(labels, dtype=torch.float32, device=device)

    print(f"Training on {len(pairs)} pairs across studies")
    for _ in range(epochs):
        perm = np.random.permutation(len(pairs))
        for start in range(0, len(pairs), batch_size):
            batch = perm[start:start + batch_size]
            i_idx = pairs[batch, 0]
            j_idx = pairs[batch, 1]
            u_i = model(X_t[i_idx])
            u_j = model(X_t[j_idx])
            logits = u_i - u_j
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels_t[batch])
            opt.zero_grad()
            loss.backward()
            opt.step()

    return model


def evaluate_pairs(model, X, y, study_ids, test_mask):
    pairs, labels = build_pair_dataset(X, y, study_ids, test_mask, max_pairs=500)
    if len(pairs) == 0:
        return float("nan")
    with torch.no_grad():
        u = model(torch.tensor(X)).cpu().numpy()
    logits = u[pairs[:, 0]] - u[pairs[:, 1]]
    preds = (logits > 0).astype(float)
    acc = float((preds == labels).mean())
    return acc


def evaluate_rank_correlation(model, X, y, study_ids, test_mask):
    with torch.no_grad():
        u = model(torch.tensor(X)).cpu().numpy()
    rhos = []
    for sid in np.unique(study_ids[test_mask]):
        idx = np.where(test_mask & (study_ids == sid))[0]
        if len(idx) < 3:
            continue
        ranks = pd.Series(y[idx]).rank(pct=True).values
        rho, _ = spearmanr(u[idx], ranks)
        if np.isfinite(rho):
            rhos.append(rho)
    return float(np.mean(rhos)) if rhos else float("nan")


def main() -> int:
    X, y, study_ids, train_mask, test_mask = prepare_study_data(min_n=5)
    model = train_bt_model(X, y, study_ids, train_mask, epochs=50)

    acc = evaluate_pairs(model, X, y, study_ids, test_mask)
    rho = evaluate_rank_correlation(model, X, y, study_ids, test_mask)

    # Per-study top-K recall (within-study ranking, then averaged)
    with torch.no_grad():
        u = model(torch.tensor(X)).cpu().numpy()
    per_study_recall = {"top10": [], "top50": [], "top100": []}
    for sid in np.unique(study_ids[test_mask]):
        idx = np.where(test_mask & (study_ids == sid))[0]
        if len(idx) < 10:
            continue
        u_study = u[idx]
        y_study = y[idx]
        for k_name, k in [("top10", min(10, len(idx))), ("top50", min(50, len(idx))), ("top100", min(100, len(idx)))]:
            pred_top = set(idx[np.argsort(u_study)[-k:]])
            true_top = set(idx[np.argsort(y_study)[-k:]])
            per_study_recall[k_name].append(len(pred_top & true_top) / len(true_top))

    results = {
        "pairwise_accuracy": acc,
        "rank_spearman_mean": rho,
        "top10_recall_mean": float(np.mean(per_study_recall["top10"])) if per_study_recall["top10"] else float("nan"),
        "top50_recall_mean": float(np.mean(per_study_recall["top50"])) if per_study_recall["top50"] else float("nan"),
        "top100_recall_mean": float(np.mean(per_study_recall["top100"])) if per_study_recall["top100"] else float("nan"),
        "n_studies_evaluated": len(per_study_recall["top10"]),
    }

    out_path = Path("models") / "bradley_terry_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"Saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
