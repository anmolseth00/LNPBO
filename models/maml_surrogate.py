#!/usr/bin/env python3
"""Phase 5.1: MAML surrogate for few-shot BO.

Implements Model-Agnostic Meta-Learning (Finn et al. 2017, ICML)
without the learn2learn dependency — uses torch.autograd.grad directly.

Each study is a task. Meta-train on 80% of studies, meta-test on 20%.
Evaluate few-shot BO: given k observations from a held-out study,
fine-tune and run greedy BO.

Reference: Finn et al. "Model-Agnostic Meta-Learning for Fast Adaptation
of Deep Networks", ICML 2017.
"""

from __future__ import annotations

import copy
import json
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from diagnostics.utils import load_lnpdb_clean, encode_lantern_il, lantern_il_feature_cols, summarize_study_assay_types


class MLP(torch.nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(in_dim, 256)
        self.fc2 = torch.nn.Linear(256, 128)
        self.fc3 = torch.nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x).squeeze(-1)

    def functional_forward(self, x, params):
        x = F.relu(F.linear(x, params["fc1.weight"], params["fc1.bias"]))
        x = F.relu(F.linear(x, params["fc2.weight"], params["fc2.bias"]))
        return F.linear(x, params["fc3.weight"], params["fc3.bias"]).squeeze(-1)


def _study_split(study_ids, study_to_type, seed=42):
    rng = np.random.RandomState(seed)
    train_ids, test_ids = set(), set()
    for assay_type in sorted(set(study_to_type.values())):
        ids = [sid for sid, at in study_to_type.items() if at == assay_type]
        rng.shuffle(ids)
        cut = max(1, int(0.8 * len(ids))) if len(ids) > 1 else len(ids)
        train_ids.update(ids[:cut])
        test_ids.update(ids[cut:])
    return train_ids, test_ids


def maml_train(model, X, y, study_ids, train_study_ids,
               inner_lr=0.01, outer_lr=1e-3, inner_steps=3,
               n_episodes=5000, support_size=10, query_size=10, seed=42):
    rng = np.random.RandomState(seed)
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=outer_lr)

    train_ids_list = sorted(train_study_ids)
    study_indices = {sid: np.where(study_ids == sid)[0] for sid in train_ids_list}
    # Only use studies with enough samples
    valid_studies = [sid for sid in train_ids_list if len(study_indices[sid]) >= support_size + query_size]
    if not valid_studies:
        print("No studies with enough samples for MAML episodes")
        return model

    losses = []
    for ep in range(n_episodes):
        # Sample a task (study)
        sid = valid_studies[rng.randint(len(valid_studies))]
        idx = study_indices[sid]
        perm = rng.permutation(len(idx))
        sup_idx = idx[perm[:support_size]]
        qry_idx = idx[perm[support_size:support_size + query_size]]

        # Clone parameters for inner loop
        params = OrderedDict(model.named_parameters())
        fast_params = OrderedDict({k: v.clone() for k, v in params.items()})

        # Inner loop: adapt on support set
        for _ in range(inner_steps):
            pred = model.functional_forward(X_t[sup_idx], fast_params)
            loss = F.mse_loss(pred, y_t[sup_idx])
            grads = torch.autograd.grad(loss, fast_params.values(), create_graph=True)
            fast_params = OrderedDict({
                k: v - inner_lr * g for (k, v), g in zip(fast_params.items(), grads)
            })

        # Outer loop: evaluate on query set with adapted params
        qry_pred = model.functional_forward(X_t[qry_idx], fast_params)
        meta_loss = F.mse_loss(qry_pred, y_t[qry_idx])

        opt.zero_grad()
        meta_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()

        losses.append(meta_loss.item())
        if (ep + 1) % 1000 == 0:
            print(f"  Episode {ep+1}: meta_loss={np.mean(losses[-100:]):.4f}")

    return model


def few_shot_evaluate(model, X, y, study_ids, test_study_ids,
                      k_shots, inner_lr=0.01, inner_steps=10, n_bo_rounds=10, batch_size=6):
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)

    results_by_k = {}
    for k in k_shots:
        study_results = []
        for sid in sorted(test_study_ids):
            idx = np.where(study_ids == sid)[0]
            if len(idx) < k + 10:
                continue

            rng = np.random.RandomState(42)
            perm = rng.permutation(len(idx))
            support_idx = idx[perm[:k]]
            pool_idx = list(idx[perm[k:]])

            # Fine-tune a copy of the meta-learned model
            adapted = copy.deepcopy(model)
            opt = torch.optim.SGD(adapted.parameters(), lr=inner_lr)
            for _ in range(inner_steps):
                pred = adapted(X_t[support_idx])
                loss = F.mse_loss(pred, y_t[support_idx])
                opt.zero_grad()
                loss.backward()
                opt.step()

            # Greedy BO: pick top-batch by predicted utility
            observed_idx = list(support_idx)
            for _ in range(n_bo_rounds):
                if len(pool_idx) < batch_size:
                    break
                with torch.no_grad():
                    scores = adapted(X_t[pool_idx]).cpu().numpy()
                top_batch = np.argsort(scores)[-batch_size:]
                selected = [pool_idx[i] for i in top_batch]
                observed_idx.extend(selected)
                for s in sorted(selected, reverse=True):
                    pool_idx.remove(s)

                # Re-fine-tune on observed data
                opt = torch.optim.SGD(adapted.parameters(), lr=inner_lr * 0.5)
                for _ in range(5):
                    pred = adapted(X_t[observed_idx])
                    loss = F.mse_loss(pred, y_t[observed_idx])
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

            # Evaluate: what fraction of study's top-k did we find?
            all_vals = y[idx]
            observed_vals = y[observed_idx]
            best_found = observed_vals.max()

            study_top10 = set(idx[np.argsort(all_vals)[-10:]])
            study_top50 = set(idx[np.argsort(all_vals)[-min(50, len(idx)):]])
            observed_set = set(observed_idx)

            r10 = len(observed_set & study_top10) / len(study_top10)
            r50 = len(observed_set & study_top50) / len(study_top50) if study_top50 else 0

            study_results.append({
                "study_id": str(sid),
                "n_in_study": len(idx),
                "k_shots": k,
                "n_observed": len(observed_idx),
                "top10_recall": float(r10),
                "top50_recall": float(r50),
                "best_found": float(best_found),
                "study_max": float(all_vals.max()),
            })

        if study_results:
            results_by_k[k] = {
                "mean_top10_recall": float(np.mean([r["top10_recall"] for r in study_results])),
                "mean_top50_recall": float(np.mean([r["top50_recall"] for r in study_results])),
                "n_studies_evaluated": len(study_results),
                "per_study": study_results,
            }
            print(f"  k={k}: top10={results_by_k[k]['mean_top10_recall']:.1%} "
                  f"top50={results_by_k[k]['mean_top50_recall']:.1%} "
                  f"({len(study_results)} studies)")

    return results_by_k


def random_baseline(X, y, study_ids, test_study_ids, k_shots, n_bo_rounds=10, batch_size=6):
    """Random selection baseline for comparison."""
    results_by_k = {}
    for k in k_shots:
        study_results = []
        for sid in sorted(test_study_ids):
            idx = np.where(study_ids == sid)[0]
            if len(idx) < k + 10:
                continue
            rng = np.random.RandomState(42)
            perm = rng.permutation(len(idx))
            n_total = min(k + n_bo_rounds * batch_size, len(idx))
            observed_idx = list(idx[perm[:n_total]])

            all_vals = y[idx]
            observed_vals = y[observed_idx]
            study_top10 = set(idx[np.argsort(all_vals)[-10:]])
            study_top50 = set(idx[np.argsort(all_vals)[-min(50, len(idx)):]])
            observed_set = set(observed_idx)

            r10 = len(observed_set & study_top10) / len(study_top10)
            r50 = len(observed_set & study_top50) / len(study_top50) if study_top50 else 0
            study_results.append({"top10_recall": float(r10), "top50_recall": float(r50)})

        if study_results:
            results_by_k[k] = {
                "mean_top10_recall": float(np.mean([r["top10_recall"] for r in study_results])),
                "mean_top50_recall": float(np.mean([r["top50_recall"] for r in study_results])),
                "n_studies_evaluated": len(study_results),
            }
    return results_by_k


def erm_baseline(X, y, study_ids, train_ids, test_ids, k_shots, inner_lr=0.01, inner_steps=10, n_bo_rounds=10, batch_size=6):
    """ERM-initialized model fine-tuned on support set."""
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)

    # Train ERM model on all training studies
    train_idx = np.concatenate([np.where(study_ids == sid)[0] for sid in train_ids])
    model = MLP(X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(100):
        pred = model(X_t[train_idx])
        loss = F.mse_loss(pred, y_t[train_idx])
        opt.zero_grad()
        loss.backward()
        opt.step()

    return few_shot_evaluate(model, X, y, study_ids, test_ids, k_shots,
                             inner_lr=inner_lr, inner_steps=inner_steps,
                             n_bo_rounds=n_bo_rounds, batch_size=batch_size)


def main() -> int:
    df = load_lnpdb_clean(drop_duplicates=False)
    df = df.dropna(subset=["IL_SMILES", "Experiment_value"]).reset_index(drop=True)

    # Filter to studies with >= 25 formulations (need enough for support + query + pool)
    study_sizes = df.groupby("study_id").size()
    keep_ids = study_sizes[study_sizes >= 25].index
    df = df[df["study_id"].isin(keep_ids)].reset_index(drop=True)
    print(f"Using {len(df)} rows from {df['study_id'].nunique()} studies (>=25 per study)")

    encoded, _ = encode_lantern_il(df, reduction="pca")
    feat_cols = lantern_il_feature_cols(encoded)

    X = encoded[feat_cols].values.astype(np.float32)
    y = encoded["Experiment_value"].values.astype(np.float32)
    study_ids = df["study_id"].astype(str).values

    study_to_type = {}
    for sid, sdf in df.groupby("study_id"):
        assay_type, _ = summarize_study_assay_types(sdf)
        study_to_type[str(sid)] = assay_type

    train_ids, test_ids = _study_split(np.unique(study_ids), study_to_type, seed=42)
    print(f"Train studies: {len(train_ids)}, Test studies: {len(test_ids)}")

    k_shots = [5, 10, 20]

    # MAML training
    print("\n=== MAML Training ===")
    model = MLP(X.shape[1])
    model = maml_train(model, X, y, study_ids, train_ids,
                       inner_lr=0.01, outer_lr=1e-3, inner_steps=3,
                       n_episodes=5000, support_size=10, query_size=10)

    # MAML few-shot evaluation
    print("\n=== MAML Few-Shot BO ===")
    maml_results = few_shot_evaluate(model, X, y, study_ids, test_ids, k_shots)

    # Random baseline
    print("\n=== Random Baseline ===")
    random_results = random_baseline(X, y, study_ids, test_ids, k_shots)
    for k, r in random_results.items():
        print(f"  k={k}: top10={r['mean_top10_recall']:.1%} top50={r['mean_top50_recall']:.1%}")

    # ERM baseline
    print("\n=== ERM Fine-Tuned Baseline ===")
    erm_results = erm_baseline(X, y, study_ids, train_ids, test_ids, k_shots)

    report = {
        "n_train_studies": len(train_ids),
        "n_test_studies": len(test_ids),
        "n_rows": len(X),
        "maml": {k: {kk: vv for kk, vv in v.items() if kk != "per_study"} for k, v in maml_results.items()},
        "erm_finetuned": erm_results,
        "random": random_results,
        "maml_per_study": {k: v.get("per_study", []) for k, v in maml_results.items()},
    }

    out_path = Path("models") / "maml_results.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nSaved {out_path}")

    # Summary table
    print("\n=== Summary ===")
    print(f"{'Method':<20} {'k':>3} {'Top-10':>8} {'Top-50':>8}")
    print("-" * 45)
    for k in k_shots:
        for name, res in [("Random", random_results), ("ERM+FT", erm_results), ("MAML", maml_results)]:
            if k in res:
                r = res[k]
                t10 = r.get("mean_top10_recall", 0)
                t50 = r.get("mean_top50_recall", 0)
                print(f"{name:<20} {k:>3} {t10:>7.1%} {t50:>7.1%}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
