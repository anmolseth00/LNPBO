"""Tests for the unified Optimizer API (optimization/optimizer.py).

Covers validation logic, registry completeness, shared helpers,
normalization integration, and a smoke test through suggest().

Run: .venv/bin/python -m pytest tests/test_optimizer.py -v
"""

import numpy as np
import pandas as pd
import pytest

from LNPBO.optimization._normalize import copula_transform, normalize_values
from LNPBO.optimization.optimizer import (
    _FAMILY_CAPS,
    ACQUISITION_TYPES,
    ALL_BATCH_STRATEGIES,
    BATCH_STRATEGIES,
    DISCRETE_BATCH_STRATEGIES,
    ENC_PREFIXES,
    SURROGATE_TYPES,
    Optimizer,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_optimizer(**overrides):
    """Create an Optimizer with only validation fields set (no real data)."""
    o = Optimizer.__new__(Optimizer)
    defaults = {
        "surrogate_type": "gp",
        "acquisition_type": "UCB",
        "batch_strategy": "kb",
        "normalize": "copula",
    }
    defaults.update(overrides)
    for k, v in defaults.items():
        setattr(o, k, v)
    return o


def _make_synthetic_df(n=30, n_features=5, n_lipids=3, seed=42):
    """Create a synthetic DataFrame that looks like an encoded dataset."""
    rng = np.random.RandomState(seed)
    lipid_names = [f"Lipid_{i}" for i in range(n_lipids)]

    df = pd.DataFrame({
        "Formulation_ID": np.arange(1, n + 1),
        "Round": 1,
        "IL_name": rng.choice(lipid_names, n),
        "HL_name": "HL_A",
        "CHL_name": "CHL_A",
        "PEG_name": "PEG_A",
        "IL_molratio": rng.uniform(30, 60, n),
        "HL_molratio": rng.uniform(20, 40, n),
        "CHL_molratio": rng.uniform(5, 15, n),
        "PEG_molratio": rng.uniform(1, 3, n),
        "IL_to_nucleicacid_massratio": 50.0,
        "Experiment_value": rng.randn(n),
    })
    # Add fake PC columns
    for i in range(n_features):
        df[f"IL_count_mfp_pc{i+1}"] = rng.randn(n)
        df[f"IL_rdkit_pc{i+1}"] = rng.randn(n)
    return df


# ---------------------------------------------------------------------------
# 1. Registry Tests
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_all_surrogates_have_valid_family(self):
        for name, family in SURROGATE_TYPES.items():
            assert family in _FAMILY_CAPS, f"{name!r} maps to unknown family {family!r}"

    def test_all_families_have_capabilities(self):
        for _family, caps in _FAMILY_CAPS.items():
            assert "needs_pool" in caps
            assert "supports_acq" in caps
            assert "supports_batch" in caps

    def test_surrogate_count(self):
        assert len(SURROGATE_TYPES) == 12

    def test_acquisition_types(self):
        assert {"UCB", "EI", "LogEI"} == ACQUISITION_TYPES

    def test_batch_strategies_disjoint_except_ts(self):
        overlap = BATCH_STRATEGIES & DISCRETE_BATCH_STRATEGIES
        assert overlap == {"ts"}, f"Unexpected overlap: {overlap}"

    def test_all_batch_strategies_is_union(self):
        assert ALL_BATCH_STRATEGIES == BATCH_STRATEGIES | DISCRETE_BATCH_STRATEGIES

    def test_enc_prefixes_derived(self):
        # ENC_PREFIXES should end with "pc" (the PCA column suffix)
        for p in ENC_PREFIXES:
            assert p.endswith("pc"), f"Prefix {p!r} doesn't end with 'pc'"


# ---------------------------------------------------------------------------
# 2. Validation Tests
# ---------------------------------------------------------------------------

class TestValidation:
    def test_valid_gp_config(self):
        for acq in ACQUISITION_TYPES:
            for bs in BATCH_STRATEGIES:
                o = _make_optimizer(surrogate_type="gp", acquisition_type=acq, batch_strategy=bs)
                o._validate_config()  # should not raise

    def test_valid_discrete_configs(self):
        discrete = [k for k, v in SURROGATE_TYPES.items() if v == "discrete"]
        for surr in discrete:
            for bs in ("greedy", "ts", "kb"):
                o = _make_optimizer(surrogate_type=surr, batch_strategy=bs)
                o._validate_config()

    def test_valid_casmopolitan_configs(self):
        for acq in ACQUISITION_TYPES:
            for bs in ("kb", "greedy"):
                o = _make_optimizer(
                    surrogate_type="casmopolitan", acquisition_type=acq, batch_strategy=bs,
                )
                o._validate_config()

    def test_invalid_surrogate_type(self):
        o = _make_optimizer(surrogate_type="magic_model")
        with pytest.raises(ValueError, match="Unknown surrogate_type"):
            o._validate_config()

    def test_invalid_acquisition_type(self):
        o = _make_optimizer(acquisition_type="PI")
        with pytest.raises(ValueError, match="Unknown acquisition_type"):
            o._validate_config()

    def test_invalid_normalize(self):
        o = _make_optimizer(normalize="log")
        with pytest.raises(ValueError, match="Unknown normalize"):
            o._validate_config()

    def test_discrete_rejects_ei(self):
        o = _make_optimizer(surrogate_type="xgb_ucb", acquisition_type="EI")
        with pytest.raises(ValueError, match="does not support"):
            o._validate_config()

    def test_discrete_rejects_lp_batch(self):
        o = _make_optimizer(surrogate_type="rf_ucb", batch_strategy="lp")
        with pytest.raises(ValueError, match="only supported with"):
            o._validate_config()

    def test_gp_rejects_greedy_batch(self):
        o = _make_optimizer(surrogate_type="gp", batch_strategy="greedy")
        with pytest.raises(ValueError, match="Unknown batch_strategy"):
            o._validate_config()

    def test_casmopolitan_rejects_ts_batch(self):
        o = _make_optimizer(surrogate_type="casmopolitan", batch_strategy="ts")
        with pytest.raises(ValueError, match="not supported"):
            o._validate_config()


# ---------------------------------------------------------------------------
# 3. Shared Helper Tests
# ---------------------------------------------------------------------------

class TestGetFeatureCols:
    def test_finds_pc_columns(self):
        df = _make_synthetic_df()
        o = _make_optimizer()
        cols = o._get_feature_cols(df)
        pc_cols = [c for c in cols if "_pc" in c]
        assert len(pc_cols) == 10  # 5 count_mfp + 5 rdkit for IL

    def test_includes_variable_ratios(self):
        df = _make_synthetic_df()
        o = _make_optimizer()
        cols = o._get_feature_cols(df)
        ratio_cols = [c for c in cols if c.endswith("_molratio")]
        assert len(ratio_cols) == 4  # all 4 roles vary

    def test_excludes_constant_ratio(self):
        df = _make_synthetic_df()
        df["IL_molratio"] = 50.0  # make constant
        o = _make_optimizer()
        cols = o._get_feature_cols(df)
        assert "IL_molratio" not in cols

    def test_includes_mass_ratio_when_variable(self):
        df = _make_synthetic_df()
        df["IL_to_nucleicacid_massratio"] = np.random.uniform(30, 70, len(df))
        o = _make_optimizer()
        cols = o._get_feature_cols(df)
        assert "IL_to_nucleicacid_massratio" in cols

    def test_excludes_constant_mass_ratio(self):
        df = _make_synthetic_df()
        o = _make_optimizer()
        cols = o._get_feature_cols(df)
        assert "IL_to_nucleicacid_massratio" not in cols


class TestOrderColumns:
    def test_formulation_id_first(self):
        df = _make_synthetic_df()
        ordered = Optimizer._order_columns(df)
        assert ordered.columns[0] == "Formulation_ID"
        assert ordered.columns[1] == "Round"

    def test_experiment_value_before_encodings(self):
        df = _make_synthetic_df()
        ordered = Optimizer._order_columns(df)
        cols = list(ordered.columns)
        ev_idx = cols.index("Experiment_value")
        pc_indices = [cols.index(c) for c in cols if "_pc" in c]
        assert all(ev_idx < i for i in pc_indices)

    def test_preserves_all_columns(self):
        df = _make_synthetic_df()
        ordered = Optimizer._order_columns(df)
        assert set(ordered.columns) == set(df.columns)


# ---------------------------------------------------------------------------
# 4. Normalization Tests
# ---------------------------------------------------------------------------

class TestNormalization:
    def test_copula_ranks_to_normal(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = copula_transform(y)
        # Should be symmetric around 0
        assert abs(result.mean()) < 0.01
        # Monotonic
        assert all(result[i] < result[i+1] for i in range(len(result)-1))

    def test_copula_with_x_new(self):
        y_ref = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        x_new = np.array([2.5, 4.5])
        result = copula_transform(y_ref, x_new=x_new)
        assert len(result) == 2
        assert result[0] < result[1]  # 4.5 should rank higher

    def test_normalize_values_copula(self):
        y = np.random.randn(50)
        result = normalize_values(y, "copula")
        assert len(result) == len(y)
        assert np.all(np.isfinite(result))

    def test_normalize_values_zscore(self):
        y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        result = normalize_values(y, "zscore")
        assert abs(result.mean()) < 1e-10
        assert abs(result.std() - 1.0) < 1e-10

    def test_normalize_values_none(self):
        y = np.array([1.0, 2.0, 3.0])
        result = normalize_values(y, "none")
        np.testing.assert_array_equal(result, y)

    def test_zscore_constant_returns_original(self):
        y = np.array([5.0, 5.0, 5.0])
        result = normalize_values(y, "zscore")
        np.testing.assert_array_equal(result, y)


# ---------------------------------------------------------------------------
# 5. PreparePool Tests
# ---------------------------------------------------------------------------

class TestPreparePool:
    def _make_optimizer_with_pool(self, n_train=30, n_pool=100, seed=42):
        train_df = _make_synthetic_df(n=n_train, seed=seed)
        pool_df = _make_synthetic_df(n=n_pool, seed=seed + 1)
        # Ensure pool IDs don't overlap with train
        pool_df["Formulation_ID"] = np.arange(1000, 1000 + n_pool)

        # Mock dataset
        class MockDataset:
            def __init__(self, df):
                self.df = df
            def max_round(self):
                return int(self.df["Round"].max())

        dataset = MockDataset(train_df)

        o = Optimizer.__new__(Optimizer)
        o.surrogate_type = "xgb_ucb"
        o.candidate_pool = pool_df
        o.context_features = False

        feature_cols = o._get_feature_cols(train_df)
        return o, dataset, feature_cols

    def test_prepare_pool_returns_train_and_pool(self):
        o, dataset, fcols = self._make_optimizer_with_pool()
        train_df, pool_df, feature_cols = o._prepare_pool(dataset, fcols)
        assert len(train_df) > 0
        assert len(pool_df) > 0
        assert len(feature_cols) > 0

    def test_prepare_pool_excludes_evaluated(self):
        o, dataset, fcols = self._make_optimizer_with_pool()
        # Put some pool IDs into training set
        o.candidate_pool = o.candidate_pool.copy()
        o.candidate_pool.iloc[0, o.candidate_pool.columns.get_loc("Formulation_ID")] = 1  # overlap
        _train_df, pool_df, _feature_cols = o._prepare_pool(dataset, fcols)
        assert 1 not in pool_df["Formulation_ID"].values

    def test_prepare_pool_drops_nan_features(self):
        o, dataset, fcols = self._make_optimizer_with_pool()
        # Inject NaN into one training row
        dataset.df.iloc[0, dataset.df.columns.get_loc("IL_count_mfp_pc1")] = np.nan
        train_df, _pool_df, _feature_cols = o._prepare_pool(dataset, fcols)
        assert len(train_df) == len(dataset.df) - 1

    def test_prepare_pool_drops_inf_features(self):
        o, dataset, fcols = self._make_optimizer_with_pool()
        dataset.df.iloc[0, dataset.df.columns.get_loc("IL_count_mfp_pc1")] = np.inf
        train_df, _pool_df, _feature_cols = o._prepare_pool(dataset, fcols)
        assert len(train_df) == len(dataset.df) - 1

    def test_prepare_pool_raises_without_pool(self):
        o, dataset, fcols = self._make_optimizer_with_pool()
        o.candidate_pool = None
        with pytest.raises(ValueError, match="candidate_pool is required"):
            o._prepare_pool(dataset, fcols)

    def test_prepare_pool_deduplicates(self):
        o, dataset, fcols = self._make_optimizer_with_pool()
        # Duplicate a pool row
        dup = o.candidate_pool.iloc[[0]].copy()
        dup["Formulation_ID"] = 9999
        o.candidate_pool = pd.concat([o.candidate_pool, dup], ignore_index=True)
        _train_df, pool_df, _feature_cols = o._prepare_pool(dataset, fcols)
        # Pool should not contain both 1000 and 9999 if they're identical compositions
        ids = pool_df["Formulation_ID"].values
        assert not (1000 in ids and 9999 in ids)

    def test_prepare_pool_does_not_mutate_dataset(self):
        o, dataset, fcols = self._make_optimizer_with_pool()
        o.context_features = False
        original_cols = list(dataset.df.columns)
        o._prepare_pool(dataset, fcols)
        assert list(dataset.df.columns) == original_cols


# ---------------------------------------------------------------------------
# 6. Suggest Dispatch Tests
# ---------------------------------------------------------------------------

class TestSuggestDispatch:
    def test_gp_dispatches_to_botorch(self):
        assert SURROGATE_TYPES["gp"] == "gp_botorch"

    def test_sklearn_dispatches_to_sklearn(self):
        assert SURROGATE_TYPES["gp_sklearn"] == "gp_sklearn"

    def test_xgb_dispatches_to_discrete(self):
        for surr in ("xgb", "xgb_ucb", "rf_ucb", "rf_ts", "ngboost",
                      "xgb_cqr", "deep_ensemble", "tabpfn", "gp_ucb"):
            assert SURROGATE_TYPES[surr] == "discrete", f"{surr} should be discrete"

    def test_casmopolitan_dispatches_to_casmopolitan(self):
        assert SURROGATE_TYPES["casmopolitan"] == "casmopolitan"
