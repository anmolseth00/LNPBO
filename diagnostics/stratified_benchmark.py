#!/usr/bin/env python3
"""Run LNPBO benchmark within each assay_type stratum."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks._discrete_common import run_discrete_strategy
from benchmarks.runner import compute_metrics, prepare_benchmark_data
from diagnostics.utils import load_lnpdb_clean

ASSAY_TYPES = [
    "in_vitro_single_formulation",
    "in_vitro_barcode_screen",
    "in_vivo_liver",
    "in_vivo_other",
]


def main() -> int:
    df = load_lnpdb_clean(drop_duplicates=False)

    seeds = [42, 123, 456, 789, 2024]
    results = {}

    for assay_type in ASSAY_TYPES:
        sdf = df[df["assay_type"] == assay_type].copy()
        if len(sdf) < 800 or sdf["study_id"].nunique() < 3:
            print(f"Skipping {assay_type}: insufficient data ({len(sdf)} rows)")
            continue

        print(f"\n=== {assay_type} ({len(sdf)} rows, {sdf['study_id'].nunique()} studies) ===")
        per_seed = []
        for seed in seeds:
            encoded, encoded_df, feature_cols, seed_idx, oracle_idx, top_k_values = prepare_benchmark_data(
                n_seed=500,
                random_seed=seed,
                subset=None,
                reduction="pca",
                feature_type="lantern_il_only",
                data_df=sdf,
            )
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
                f"top10={metrics['top_k_recall'][10]:.1%} top50={metrics['top_k_recall'][50]:.1%}"
            )

        summary = {
            "n_seeds": len(per_seed),
            "top10_mean": float(np.mean([m["top_k_recall"][10] for m in per_seed])),
            "top50_mean": float(np.mean([m["top_k_recall"][50] for m in per_seed])),
            "top100_mean": float(np.mean([m["top_k_recall"][100] for m in per_seed])),
            "top10_std": float(np.std([m["top_k_recall"][10] for m in per_seed])),
            "top50_std": float(np.std([m["top_k_recall"][50] for m in per_seed])),
            "top100_std": float(np.std([m["top_k_recall"][100] for m in per_seed])),
        }

        results[assay_type] = {
            "summary": summary,
            "per_seed": per_seed,
        }

    out_path = Path("diagnostics") / "stratified_benchmark.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
