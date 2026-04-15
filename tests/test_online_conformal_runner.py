"""Runner-level regression tests for online conformal BO strategies."""

import numpy as np
import pandas as pd

from LNPBO.benchmarks.runner import (
    STRATEGY_CONFIGS,
    run_discrete_cumulative_split_conformal_ucb_baseline,
    run_discrete_online_conformal_strategy,
)


def test_strategy_registry_keeps_exact_and_baseline_paths_separate() -> None:
    assert STRATEGY_CONFIGS["discrete_xgb_online_conformal"]["type"] == "discrete_online_conformal_exact"
    assert (
        STRATEGY_CONFIGS["discrete_xgb_cumulative_split_conformal_ucb_baseline"]["type"]
        == "discrete_online_conformal_baseline"
    )


def test_online_conformal_runner_rebuilds_exact_recalibrator_each_round() -> None:
    df = pd.DataFrame(
        {
            "feat": np.linspace(0.0, 1.0, 8),
            "Experiment_value": np.array([0.1, 0.15, 0.2, 0.35, 0.4, 0.55, 0.6, 0.8]),
        }
    )

    history = run_discrete_online_conformal_strategy(
        encoded_df=df,
        feature_cols=["feat"],
        seed_idx=[0, 1, 2, 3],
        oracle_idx=[4, 5, 6, 7],
        batch_size=2,
        n_rounds=2,
        seed=0,
        kappa=1.0,
        normalize="none",
    )

    assert len(history["coverage"]) == 2
    assert len(history["conformal_quantile"]) == 2
    assert all(0.0 <= q <= 1.0 for q in history["conformal_quantile"])
    assert all(0.0 <= c <= 1.0 for c in history["coverage"])


def test_baseline_runner_keeps_legacy_quantile_tracking() -> None:
    df = pd.DataFrame(
        {
            "feat": np.linspace(0.0, 1.0, 8),
            "Experiment_value": np.array([0.1, 0.12, 0.2, 0.21, 0.5, 0.55, 0.6, 0.8]),
        }
    )

    history = run_discrete_cumulative_split_conformal_ucb_baseline(
        encoded_df=df,
        feature_cols=["feat"],
        seed_idx=[0, 1, 2, 3],
        oracle_idx=[4, 5, 6, 7],
        batch_size=2,
        n_rounds=2,
        seed=0,
        kappa=1.0,
        normalize="none",
    )

    assert len(history["coverage"]) == 2
    assert len(history["conformal_quantile"]) == 2
