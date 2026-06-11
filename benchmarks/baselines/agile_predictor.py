#!/usr/bin/env python3
"""AGILE Embedding Predict-and-Rank Baseline.

Uses pre-computed AGILE (GNN) embeddings as features for predict-and-rank.
This tests whether AGILE's learned molecular representations outperform
LANTERN (count MFP + RDKit) for single-shot prediction.

Two modes:
  - "embedding_xgb": AGILE embeddings → XGBoost predict-and-rank
  - "embedding_rf":  AGILE embeddings → RF predict-and-rank

Reference: Xu, Y., Ma, S., Cui, H. et al. (2024). "AGILE platform: a deep
learning powered approach to accelerate LNP development for mRNA delivery."
Nature Communications 15:6305. DOI 10.1038/s41467-024-50619-z.

Usage:
    python -m benchmarks.baselines.agile_predictor
    python -m benchmarks.baselines.agile_predictor --surrogates xgb,rf
    python -m benchmarks.baselines.agile_predictor --resume
    python -m benchmarks.baselines.agile_predictor --aggregate-only
"""


from LNPBO.benchmarks.baselines.predict_and_rank import run_pr_cli
from LNPBO.runtime_paths import benchmark_results_root, package_root_from

_PACKAGE_ROOT = package_root_from(__file__, levels_up=3)
RESULTS_DIR = benchmark_results_root(_PACKAGE_ROOT) / "baselines" / "agile_predictor"

SURROGATES = {
    "xgb": "XGBoost",
    "rf": "Random Forest",
}


def main():
    run_pr_cli(
        results_dir=RESULTS_DIR,
        surrogates_map=SURROGATES,
        default_surrogates="xgb,rf",
        banner="AGILE EMBEDDING PREDICT-AND-RANK BASELINE",
        baseline_name="agile_predictor",
        file_prefix="agile",
        display_prefix="AGILE+",
        feature_type_override="agile_il_only",
        extra_json_fields={"feature_type": "agile_il_only"},
    )


if __name__ == "__main__":
    main()
