#!/usr/bin/env python3
"""V-REx surrogate on LANTERN IL-only PCs.

Reference:
    Krueger, D. et al. (2021). "Out-of-Distribution Generalization via Risk
    Extrapolation (REx)." ICML 2021. arXiv:2003.00688.
"""


import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import r2_score

from LNPBO.diagnostics.utils import prepare_study_data
from LNPBO.models.surrogate_mlp import SurrogateMLP


def train_vrex(X, y, study_ids, train_mask, lambda_rex, epochs=200, lr=1e-3):
    device = torch.device("cpu")
    X_t = torch.tensor(X, device=device)
    y_t = torch.tensor(y, device=device)

    model = SurrogateMLP(X.shape[1]).to(device)
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
    X, y, study_ids, train_mask, test_mask = prepare_study_data(min_n=5)

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
