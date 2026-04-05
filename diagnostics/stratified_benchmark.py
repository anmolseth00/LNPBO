#!/usr/bin/env python3
"""Run LNPBO benchmark within each assay_type stratum."""

import json
import logging
from pathlib import Path

import numpy as np

from LNPBO.benchmarks._optimizer_runner import OptimizerRunner
from LNPBO.benchmarks.runner import compute_metrics, prepare_benchmark_data
from LNPBO.data.context import ASSAY_TYPES
from LNPBO.data.study_utils import load_lnpdb_clean
from LNPBO.optimization.optimizer import Optimizer

logger = logging.getLogger("lnpbo")


def main() -> int:
    df = load_lnpdb_clean(drop_duplicates=False)

    seeds = [42, 123, 456, 789, 2024]
    results = {}

    for assay_type in ASSAY_TYPES:
        sdf = df[df["assay_type"] == assay_type].copy()
        if len(sdf) < 800 or sdf["study_id"].nunique() < 3:
            logger.info("Skipping %s: insufficient data (%d rows)", assay_type, len(sdf))
            continue

        logger.info("=== %s (%d rows, %d studies) ===", assay_type, len(sdf), sdf["study_id"].nunique())
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
            optimizer = Optimizer(
                surrogate_type="xgb",
                batch_strategy="greedy",
                random_seed=seed,
                kappa=5.0,
                normalize="copula",
                batch_size=12,
            )
            runner = OptimizerRunner(optimizer)
            history = runner.run(
                encoded_df,
                feature_cols,
                seed_idx,
                oracle_idx,
                n_rounds=15,
                batch_size=12,
                encoded_dataset=encoded,
            )
            metrics = compute_metrics(history, top_k_values, len(encoded_df))
            per_seed.append(metrics)
            logger.info(
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
    logger.info("Saved %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
