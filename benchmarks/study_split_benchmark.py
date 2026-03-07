#!/usr/bin/env python3
"""Study-level holdout benchmark: train BO on 80% of studies, evaluate on 20%."""


import json
from pathlib import Path

import numpy as np

from LNPBO.benchmarks._discrete_common import run_discrete_strategy
from LNPBO.benchmarks.runner import compute_metrics, prepare_benchmark_data
from LNPBO.diagnostics.utils import build_study_type_map, load_lnpdb_clean, study_split


def main() -> int:
    df = load_lnpdb_clean(drop_duplicates=False)

    seeds = [42, 123, 456, 789, 2024]
    configs = ["lantern_il_only"]

    # Build study-type map once (deterministic, doesn't depend on seed)
    study_to_type = build_study_type_map(df)
    all_study_ids = df["study_id"].unique()

    all_results = {}
    for config in configs:
        print(f"\n{'='*60}")
        print(f"Config: {config}")
        print(f"{'='*60}")

        per_seed = []
        for seed in seeds:
            # Study-level split
            train_ids, test_ids = study_split(all_study_ids, study_to_type, seed=seed)
            encoded, encoded_df, feature_cols, seed_idx, oracle_idx, top_k_values = prepare_benchmark_data(
                n_seed=500,
                random_seed=seed,
                reduction="pca",
                feature_type=config,
                data_df=df,
            )

            # Partition encoded_df by study_id (indices are 0..N-1 after reset)
            train_mask = encoded_df["study_id"].isin(train_ids)
            test_mask = encoded_df["study_id"].isin(test_ids)
            train_rows = encoded_df.index[train_mask].tolist()
            test_rows = encoded_df.index[test_mask].tolist()

            if len(test_rows) < 50:
                print(f"  seed={seed}: too few test rows ({len(test_rows)}), skipping")
                continue

            rng = np.random.RandomState(seed)
            n_seed = min(500, len(train_rows))
            seed_idx = sorted(rng.choice(train_rows, size=n_seed, replace=False).tolist())
            oracle_idx = sorted(test_rows)

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
