#!/usr/bin/env python3
"""GP-based discrete BO benchmark using BoTorch acquisitions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from botorch.acquisition import qExpectedImprovement, qUpperConfidenceBound
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT.parent))

from LNPBO.benchmarks.runner import compute_metrics, init_history, prepare_benchmark_data, update_history


def _fit_gp(X_train, y_train):
    X = torch.tensor(X_train, dtype=torch.double)
    Y = torch.tensor(y_train.reshape(-1, 1), dtype=torch.double)
    model = SingleTaskGP(
        X,
        Y,
        input_transform=Normalize(X.shape[-1]),
        outcome_transform=Standardize(m=1),
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll, options={"maxiter": 20})
    return model


def _score_acq(model, X_pool, acq_type="qei", beta=2.0, best_f=None):
    X = torch.tensor(X_pool, dtype=torch.double)
    if acq_type == "qei":
        if best_f is None:
            raise ValueError("best_f required for qEI")
        acq = qExpectedImprovement(model, best_f=best_f)
    elif acq_type == "qucb":
        acq = qUpperConfidenceBound(model, beta=beta)
    else:
        raise ValueError(f"Unknown acq_type: {acq_type}")
    with torch.no_grad():
        vals = acq(X.unsqueeze(1)).squeeze(-1).cpu().numpy()
    return vals


def run_gp_discrete(
    encoded_df, feature_cols, seed_idx, oracle_idx, batch_size, n_rounds,
    acq_type, seed, beta=2.0, encoded_dataset=None,
):
    training_idx = list(seed_idx)
    pool_idx = list(oracle_idx)
    history = init_history(encoded_df, training_idx)

    for r in range(n_rounds):
        if len(pool_idx) < batch_size:
            break

        if encoded_dataset is not None and getattr(encoded_dataset, "raw_fingerprints", None):
            encoded_dataset.refit_pls(training_idx, external_df=encoded_df)

        X_train = encoded_df.loc[training_idx, feature_cols].values
        y_train = encoded_df.loc[training_idx, "Experiment_value"].values
        X_pool = encoded_df.loc[pool_idx, feature_cols].values

        model = _fit_gp(X_train, y_train)
        best_f = y_train.max()
        scores = _score_acq(model, X_pool, acq_type=acq_type, beta=beta, best_f=best_f)

        top_indices = np.argsort(scores)[-batch_size:][::-1]
        batch_idx = [pool_idx[i] for i in top_indices]

        pool_set = set(batch_idx)
        pool_idx = [i for i in pool_idx if i not in pool_set]
        training_idx.extend(batch_idx)
        update_history(history, encoded_df, training_idx, batch_idx, r)

        batch_best = encoded_df.loc[batch_idx, "Experiment_value"].max()
        cum_best = history["best_so_far"][-1]
        print(
            f"  Round {r+1}: batch_best={batch_best:.3f}, cum_best={cum_best:.3f}, n_new={len(batch_idx)}",
            flush=True,
        )

    return history


def main():
    parser = argparse.ArgumentParser(description="GP-based BO benchmark (discrete pool)")
    parser.add_argument("--acq", type=str, default="qei", choices=["qei", "qucb"], help="Acquisition")
    parser.add_argument("--beta", type=float, default=2.0, help="Beta for qUCB")
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--n-seeds", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--feature-type", type=str, default="lantern_il_only")
    args = parser.parse_args()

    seeds = [42, 123, 456, 789, 2024]
    print("Loading/encoding dataset once for all seeds...")
    encoded, encoded_df, feature_cols, _, _, top_k_values = prepare_benchmark_data(
        n_seed=args.n_seeds,
        random_seed=seeds[0],
        reduction="pca",
        feature_type=args.feature_type,
    )
    all_idx = np.arange(len(encoded_df))

    results = []
    for seed in seeds:
        print(f"\n=== Seed {seed} ===")
        rng = np.random.RandomState(seed)
        rng.shuffle(all_idx)
        seed_idx = sorted(all_idx[: args.n_seeds])
        oracle_idx = sorted(all_idx[args.n_seeds :])

        history = run_gp_discrete(
            encoded_df,
            feature_cols,
            seed_idx,
            oracle_idx,
            batch_size=args.batch_size,
            n_rounds=args.rounds,
            acq_type=args.acq,
            seed=seed,
            beta=args.beta,
            encoded_dataset=encoded,
        )
        metrics = compute_metrics(history, top_k_values, len(encoded_df))
        results.append(metrics)
        print(
            f"  final_best={metrics['final_best']:.3f} "
            f"top10={metrics['top_k_recall'][10]:.1%} top50={metrics['top_k_recall'][50]:.1%}"
        )

    summary = {
        "top10_mean": float(np.mean([m["top_k_recall"][10] for m in results])),
        "top50_mean": float(np.mean([m["top_k_recall"][50] for m in results])),
        "top100_mean": float(np.mean([m["top_k_recall"][100] for m in results])),
        "top10_std": float(np.std([m["top_k_recall"][10] for m in results])),
        "top50_std": float(np.std([m["top_k_recall"][50] for m in results])),
        "top100_std": float(np.std([m["top_k_recall"][100] for m in results])),
    }

    out = {
        "acq": args.acq,
        "beta": args.beta,
        "results": results,
        "summary": summary,
    }

    out_path = PROJECT_ROOT / "benchmark_results" / f"gp_{args.acq}_benchmark.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
