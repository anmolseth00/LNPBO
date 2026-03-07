#!/usr/bin/env python3
"""Study-level holdout benchmark: train BO on 80% of studies, evaluate on 20%."""


import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks._discrete_common import run_discrete_strategy
from benchmarks.runner import compute_metrics, prepare_benchmark_data
from diagnostics.utils import load_lnpdb_clean, summarize_study_assay_types


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

    return train_ids, test_ids


def main() -> int:
    df = load_lnpdb_clean(drop_duplicates=False)

    seeds = [42, 123, 456, 789, 2024]
    configs = ["lantern_il_only"]

    all_results = {}
    for config in configs:
        print(f"\n{'='*60}")
        print(f"Config: {config}")
        print(f"{'='*60}")

        per_seed = []
        for seed in seeds:
            # Study-level split
            train_ids, test_ids = _study_split(df, seed=seed)
            # Use train studies as seed pool, test studies as oracle
            train_idx_in_full = df.index[df["study_id"].isin(train_ids)].tolist()
            test_idx_in_full = df.index[df["study_id"].isin(test_ids)].tolist()

            if len(test_idx_in_full) < 100:
                print(f"  seed={seed}: too few test rows ({len(test_idx_in_full)}), skipping")
                continue

            encoded, encoded_df, feature_cols, seed_idx, oracle_idx, top_k_values = prepare_benchmark_data(
                n_seed=min(500, len(train_idx_in_full)),
                random_seed=seed,
                reduction="pca",
                feature_type=config,
            )

            # Override: use study-split indices
            # seed_idx = random subset of train studies
            rng = np.random.RandomState(seed)
            n_seed = min(500, len(train_idx_in_full))
            seed_idx = sorted(rng.choice(train_idx_in_full, size=n_seed, replace=False).tolist())

            # oracle = test studies only
            oracle_idx = sorted(test_idx_in_full)
            # Filter to valid indices in encoded_df
            valid = set(encoded_df.index)
            seed_idx = [i for i in seed_idx if i in valid]
            oracle_idx = [i for i in oracle_idx if i in valid]

            if len(oracle_idx) < 50:
                print(f"  seed={seed}: too few oracle rows ({len(oracle_idx)}), skipping")
                continue

            # Recompute top-k from test pool (as index sets)
            test_series = encoded_df.loc[oracle_idx, "Experiment_value"]
            top_k_values = {
                k: set(test_series.nlargest(k).index)
                for k in [10, 50, 100]
                if k <= len(test_series)
            }

            history = run_discrete_strategy(
                encoded_df,
                feature_cols,
                seed_idx,
                oracle_idx,
                surrogate="xgb",
                batch_size=12,
                n_rounds=15,
                seed=seed,
                kappa=5.0,
                normalize="copula",
                encoded_dataset=encoded,
            )
            metrics = compute_metrics(history, top_k_values, len(encoded_df))
            per_seed.append(metrics)
            print(
                f"  seed={seed} final_best={metrics['final_best']:.3f} "
                f"top10={metrics['top_k_recall'].get(10, 0):.1%} "
                f"top50={metrics['top_k_recall'].get(50, 0):.1%}"
            )

        if per_seed:
            summary = {}
            for k in [10, 50, 100]:
                vals = [m["top_k_recall"].get(k, 0) for m in per_seed]
                if vals:
                    summary[f"top{k}_mean"] = float(np.mean(vals))
                    summary[f"top{k}_std"] = float(np.std(vals))

            all_results[config] = {
                "eval_split": "study_level",
                "summary": summary,
                "per_seed": per_seed,
            }

    out_path = Path("diagnostics") / "study_split_benchmark.json"
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
