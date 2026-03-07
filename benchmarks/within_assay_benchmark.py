#!/usr/bin/env python3
"""Phase 4.2: Within-assay-type study-level holdout benchmark.

For each assay type with enough studies, do study-level 80/20 split
and run BO within that stratum only.

Key question: does within-assay-type performance >> cross-assay-type?
"""


import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks._discrete_common import run_discrete_strategy
from benchmarks.runner import compute_metrics, prepare_benchmark_data
from diagnostics.utils import load_lnpdb_clean


def _study_split_within_stratum(df, assay_type, seed=42):
    rng = np.random.RandomState(seed)
    sdf = df[df["assay_type"] == assay_type]
    study_ids = sdf["study_id"].unique().tolist()
    rng.shuffle(study_ids)
    cut = max(1, int(0.8 * len(study_ids)))
    train_ids = set(study_ids[:cut])
    test_ids = set(study_ids[cut:])
    return train_ids, test_ids


def main() -> int:
    df = load_lnpdb_clean(drop_duplicates=False)

    # Assign assay types
    from diagnostics.utils import _infer_assay_type_row
    df["assay_type"] = df.apply(_infer_assay_type_row, axis=1)

    seeds = [42, 123, 456, 789, 2024]
    assay_types = df["assay_type"].value_counts()

    all_results = {}
    for assay_type, count in assay_types.items():
        n_studies = df[df["assay_type"] == assay_type]["study_id"].nunique()
        if n_studies < 5:
            print(f"Skipping {assay_type}: only {n_studies} studies (need >=5)")
            continue
        if count < 200:
            print(f"Skipping {assay_type}: only {count} rows (need >=200)")
            continue

        print(f"\n{'='*60}")
        print(f"Assay: {assay_type} ({count} rows, {n_studies} studies)")
        print(f"{'='*60}")

        per_seed = []
        for seed in seeds:
            train_ids, test_ids = _study_split_within_stratum(df, assay_type, seed=seed)
            if len(test_ids) < 2:
                print(f"  seed={seed}: only {len(test_ids)} test studies, skipping")
                continue

            sdf = df[df["assay_type"] == assay_type].copy()

            encoded, encoded_df, feature_cols, seed_idx, oracle_idx, top_k_values = prepare_benchmark_data(
                n_seed=min(500, int(0.8 * len(sdf))),
                random_seed=seed,
                reduction="pca",
                feature_type="lantern_il_only",
                data_df=sdf,
            )

            # Override with study-level split
            rng = np.random.RandomState(seed)
            train_rows = encoded_df.index[encoded_df.index.isin(
                sdf[sdf["study_id"].isin(train_ids)].index
            )].tolist()
            test_rows = encoded_df.index[encoded_df.index.isin(
                sdf[sdf["study_id"].isin(test_ids)].index
            )].tolist()

            if len(train_rows) < 50 or len(test_rows) < 20:
                print(f"  seed={seed}: insufficient rows (train={len(train_rows)}, test={len(test_rows)})")
                continue

            n_seed = min(500, len(train_rows))
            seed_idx = sorted(rng.choice(train_rows, size=n_seed, replace=False).tolist())
            oracle_idx = sorted(test_rows)

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

            t10 = metrics["top_k_recall"].get(10, 0)
            t50 = metrics["top_k_recall"].get(50, 0)
            print(f"  seed={seed} final_best={metrics['final_best']:.3f} top10={t10:.1%} top50={t50:.1%}")

        if per_seed:
            summary = {}
            for k in [10, 50, 100]:
                vals = [m["top_k_recall"].get(k, 0) for m in per_seed]
                if vals:
                    summary[f"top{k}_mean"] = float(np.mean(vals))
                    summary[f"top{k}_std"] = float(np.std(vals))

            all_results[assay_type] = {
                "eval_split": "within_assay_study_level",
                "n_studies": n_studies,
                "n_rows": int(count),
                "summary": summary,
                "per_seed": per_seed,
            }

    out_path = Path("benchmarks") / "within_assay_benchmark.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
