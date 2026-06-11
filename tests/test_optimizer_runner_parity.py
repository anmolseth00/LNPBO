"""Tests for OptimizerRunner and Optimizer.suggest_indices().

Verifies that Optimizer + OptimizerRunner produces correct history dicts
for representative strategies (discrete and GP).

Run: .venv/bin/python -m pytest tests/test_optimizer_runner_parity.py -v
"""

import numpy as np
import pandas as pd
import pytest

from benchmarks._optimizer_runner import OptimizerRunner
from benchmarks.runner import strategy_to_optimizer_kwargs
from LNPBO.optimization.optimizer import Optimizer


@pytest.fixture
def synthetic_data():
    """Create synthetic benchmark data with reproducible seed."""
    rng = np.random.RandomState(42)
    n = 200
    n_features = 5
    lipid_names = [f"Lipid_{i}" for i in range(8)]

    df = pd.DataFrame(
        {
            "Formulation_ID": np.arange(1, n + 1),
            "IL_name": rng.choice(lipid_names, n),
            "HL_name": "HL_A",
            "CHL_name": "CHL_A",
            "PEG_name": "PEG_A",
            "IL_molratio": rng.uniform(30, 60, n),
            "HL_molratio": rng.uniform(20, 40, n),
            "CHL_molratio": rng.uniform(5, 15, n),
            "PEG_molratio": rng.uniform(1, 3, n),
            "Experiment_value": rng.randn(n) * 2 + 3,
        }
    )
    for i in range(n_features):
        df[f"IL_count_mfp_pc{i + 1}"] = rng.randn(n)
        df[f"IL_rdkit_pc{i + 1}"] = rng.randn(n)

    feature_cols = [f"IL_count_mfp_pc{i + 1}" for i in range(n_features)]
    feature_cols += [f"IL_rdkit_pc{i + 1}" for i in range(n_features)]
    feature_cols += ["IL_molratio", "HL_molratio", "CHL_molratio", "PEG_molratio"]

    all_idx = np.arange(n)
    rng2 = np.random.RandomState(42)
    rng2.shuffle(all_idx)
    seed_idx = sorted(all_idx[:50].tolist())
    oracle_idx = sorted(all_idx[50:].tolist())

    top_k_values = {
        10: set(df.nlargest(10, "Experiment_value").index),
        50: set(df.nlargest(50, "Experiment_value").index),
    }

    return df, feature_cols, seed_idx, oracle_idx, top_k_values


class TestDiscreteRFTS:
    """discrete_rf_ts via OptimizerRunner."""

    def test_history_structure(self, synthetic_data):
        df, feature_cols, seed_idx, oracle_idx, top_k_values = synthetic_data

        opt = Optimizer(
            surrogate_type="rf_ts", batch_strategy="greedy",
            random_seed=42, kappa=5.0, normalize="copula", batch_size=12,
        )
        history = OptimizerRunner(opt).run(
            df, feature_cols, seed_idx, oracle_idx,
            n_rounds=3, batch_size=12, top_k_values=top_k_values,
        )

        assert "best_so_far" in history
        assert "round_best" in history
        assert "n_evaluated" in history
        assert "per_round_recall" in history
        assert "all_evaluated" in history
        # best_so_far includes the seed entry + 3 rounds = 4
        assert len(history["best_so_far"]) == 4
        assert len(history["round_best"]) == 3

    def test_determinism(self, synthetic_data):
        df, feature_cols, seed_idx, oracle_idx, top_k_values = synthetic_data

        results = []
        for _ in range(2):
            opt = Optimizer(
                surrogate_type="rf_ts", batch_strategy="greedy",
                random_seed=42, kappa=5.0, normalize="copula", batch_size=12,
            )
            history = OptimizerRunner(opt).run(
                df, feature_cols, seed_idx, oracle_idx,
                n_rounds=3, batch_size=12, top_k_values=top_k_values,
            )
            results.append(history)

        assert results[0]["best_so_far"] == results[1]["best_so_far"]
        assert results[0]["round_best"] == results[1]["round_best"]
        assert results[0]["n_evaluated"] == results[1]["n_evaluated"]


class TestGPLogEI:
    """lnpbo_logei via OptimizerRunner."""

    def test_history_structure(self, synthetic_data):
        df, feature_cols, seed_idx, oracle_idx, top_k_values = synthetic_data

        opt = Optimizer(
            surrogate_type="gp", gp_engine="botorch",
            acquisition_type="LogEI", batch_strategy="kb",
            kernel_type="matern",
            random_seed=42, kappa=5.0, xi=0.01,
            normalize="copula", batch_size=12,
        )
        history = OptimizerRunner(opt).run(
            df, feature_cols, seed_idx, oracle_idx,
            n_rounds=2, batch_size=12, top_k_values=top_k_values,
        )

        assert "best_so_far" in history
        assert "round_best" in history
        assert "n_evaluated" in history
        # best_so_far includes the seed entry + 2 rounds = 3
        assert len(history["best_so_far"]) == 3


class TestStrategyMapping:
    """Verify strategy_to_optimizer_kwargs covers all non-special strategies."""

    def test_all_strategies_mapped(self):
        from benchmarks.runner import STRATEGY_CONFIGS

        for name, config in STRATEGY_CONFIGS.items():
            if config["type"] in (
                "random",
                "discrete_online_conformal",
                "discrete_online_conformal_exact",
                "discrete_online_conformal_baseline",
            ):
                with pytest.raises(ValueError):
                    strategy_to_optimizer_kwargs(name)
            else:
                kwargs = strategy_to_optimizer_kwargs(name)
                assert "surrogate_type" in kwargs

    def test_optimizer_validates_all_mappings(self):
        from benchmarks.runner import STRATEGY_CONFIGS

        for name, config in STRATEGY_CONFIGS.items():
            if config["type"] in (
                "random",
                "discrete_online_conformal",
                "discrete_online_conformal_exact",
                "discrete_online_conformal_baseline",
            ):
                continue
            kwargs = strategy_to_optimizer_kwargs(name)
            opt = Optimizer(
                random_seed=42, kappa=5.0, xi=0.01,
                normalize="copula", batch_size=12,
                **kwargs,
            )
            assert opt._family in ("gp_botorch", "gp_mixed", "discrete", "casmopolitan")


class TestHistoryDictStructure:
    """Verify the history dict from OptimizerRunner has expected keys."""

    def test_history_keys(self, synthetic_data):
        df, feature_cols, seed_idx, oracle_idx, top_k_values = synthetic_data

        opt = Optimizer(
            surrogate_type="rf_ts", batch_strategy="greedy",
            random_seed=42, kappa=5.0, normalize="copula", batch_size=12,
        )
        history = OptimizerRunner(opt).run(
            df, feature_cols, seed_idx, oracle_idx,
            n_rounds=2, batch_size=12, top_k_values=top_k_values,
        )

        expected_keys = {
            "best_so_far", "round_best", "n_evaluated",
            "all_evaluated", "per_round_recall",
        }
        assert expected_keys.issubset(set(history.keys()))


class TestSuggestIndicesAPI:
    """Test Optimizer.suggest_indices() directly."""

    def test_returns_list_of_pool_indices(self, synthetic_data):
        df, feature_cols, seed_idx, oracle_idx, _ = synthetic_data

        opt = Optimizer(
            surrogate_type="rf_ts", batch_strategy="greedy",
            random_seed=42, kappa=5.0, normalize="copula", batch_size=12,
        )
        result = opt.suggest_indices(
            df, feature_cols, seed_idx, oracle_idx, round_num=0,
        )

        assert isinstance(result, list)
        assert len(result) == 12
        assert all(idx in oracle_idx for idx in result)

    def test_suggest_requires_space(self):
        opt = Optimizer(
            surrogate_type="gp", random_seed=42,
            kappa=5.0, normalize="copula", batch_size=12,
        )
        with pytest.raises(ValueError, match="space is required"):
            opt.suggest()

    def test_gp_returns_pool_indices(self, synthetic_data):
        df, feature_cols, seed_idx, oracle_idx, _ = synthetic_data

        opt = Optimizer(
            surrogate_type="gp", gp_engine="botorch",
            acquisition_type="LogEI", batch_strategy="kb",
            kernel_type="matern",
            random_seed=42, kappa=5.0, xi=0.01,
            normalize="copula", batch_size=12,
        )
        result = opt.suggest_indices(
            df, feature_cols, seed_idx, oracle_idx, round_num=0,
        )

        assert isinstance(result, list)
        assert len(result) == 12
        # GP path returns actual DataFrame indices from pool_idx
        pool_set = set(oracle_idx)
        assert all(idx in pool_set for idx in result)

    def test_multitask_gp_does_not_crash(self, synthetic_data):
        """Regression: the multitask_gp path referenced an out-of-scope
        ``train_Yvar`` and raised NameError on every call (commit a924806)."""
        df, feature_cols, seed_idx, oracle_idx, _ = synthetic_data
        df = df.copy()
        # multitask_gp requires a study_id column to build task indices.
        df["study_id"] = np.random.RandomState(0).choice(["A", "B", "C"], len(df))

        opt = Optimizer(
            surrogate_type="multitask_gp", gp_engine="botorch",
            acquisition_type="LogEI",
            random_seed=42, kappa=5.0, xi=0.01,
            normalize="copula", batch_size=8,
        )
        result = opt.suggest_indices(
            df, feature_cols, seed_idx, oracle_idx, round_num=0,
        )
        assert isinstance(result, list)
        assert len(result) == 8
        assert all(idx in set(oracle_idx) for idx in result)

    def test_nan_rows_keep_study_labels_aligned(self, synthetic_data):
        """Regression: NaN/inf row filtering dropped feature rows but not the
        auxiliary study-label/IL-name arrays, silently misaligning them."""
        df, feature_cols, seed_idx, oracle_idx, _ = synthetic_data
        df = df.copy()
        df["study_id"] = np.random.RandomState(0).choice(["A", "B", "C"], len(df))
        # Inject a NaN feature into the first training row and first pool row so
        # both train and pool filtering drop a row.
        df.loc[seed_idx[0], feature_cols[0]] = np.nan
        df.loc[oracle_idx[0], feature_cols[0]] = np.nan

        opt = Optimizer(
            surrogate_type="multitask_gp", gp_engine="botorch",
            acquisition_type="LogEI",
            random_seed=42, kappa=5.0, xi=0.01,
            normalize="copula", batch_size=8,
        )
        # Before the fix this raised a length-mismatch error (or silently
        # misaligned labels); it must now run cleanly and exclude the dropped pool row.
        result = opt.suggest_indices(
            df, feature_cols, seed_idx, oracle_idx, round_num=0,
        )
        assert len(result) == 8
        assert oracle_idx[0] not in result  # dropped NaN pool row not selectable
