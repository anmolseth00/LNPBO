"""Tests for round-level partial checkpoint + resume.

Covers:
- ``_serialize_history`` / ``_deserialize_history`` roundtrip
- ``_estimate_remaining_eta`` (per-strategy ETA)
- ``save_partial_seed_result`` / ``load_partial_seed_result`` /
  ``clear_partial_seed_result`` on disk, including corrupt-file recovery
- ``OptimizerRunner.run(resume_state=...)`` byte-parity vs uninterrupted runs
  on both a deterministic synthetic optimizer and real RF-TS / GP-LogEI
  strategies.

Run: uv run python -m pytest tests/test_partial_resume.py -v
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import benchmarks.benchmark as bm
from benchmarks._optimizer_runner import OptimizerRunner
from benchmarks.benchmark import (
    _deserialize_history,
    _estimate_remaining_eta,
    _partial_seed_path,
    _serialize_history,
    clear_partial_seed_result,
    load_partial_seed_result,
    save_partial_seed_result,
)
from LNPBO.optimization.optimizer import Optimizer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_results_dir(monkeypatch):
    """Redirect RESULTS_DIR to a temp dir so tests don't touch real results."""
    with tempfile.TemporaryDirectory() as d:
        monkeypatch.setattr(bm, "RESULTS_DIR", Path(d))
        yield Path(d)


@pytest.fixture
def synthetic_data():
    rng = np.random.RandomState(42)
    n = 200
    df = pd.DataFrame(
        {
            "Formulation_ID": np.arange(1, n + 1),
            "IL_name": rng.choice([f"L_{i}" for i in range(8)], n),
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
    for i in range(5):
        df[f"IL_count_mfp_pc{i + 1}"] = rng.randn(n)
        df[f"IL_rdkit_pc{i + 1}"] = rng.randn(n)
    feature_cols = (
        [f"IL_count_mfp_pc{i + 1}" for i in range(5)]
        + [f"IL_rdkit_pc{i + 1}" for i in range(5)]
        + ["IL_molratio", "HL_molratio", "CHL_molratio", "PEG_molratio"]
    )
    all_idx = np.arange(n)
    rng2 = np.random.RandomState(42)
    rng2.shuffle(all_idx)
    seed_idx = sorted(all_idx[:50].tolist())
    oracle_idx = sorted(all_idx[50:].tolist())
    top_k_values = {
        10: set(df.nlargest(10, "Experiment_value").index),
        20: set(df.nlargest(20, "Experiment_value").index),
    }
    return df, feature_cols, seed_idx, oracle_idx, top_k_values


# ---------------------------------------------------------------------------
# History serialization
# ---------------------------------------------------------------------------


class TestHistoryRoundtrip:
    def test_minimal(self):
        h = {
            "best_so_far": [0.5, 0.7, 0.9],
            "round_best": [0.6, 0.85],
            "n_evaluated": [10, 22, 34],
            "all_evaluated": {1, 2, 3, 10, 20, 30},
        }
        r = _deserialize_history(json.loads(json.dumps(_serialize_history(h), default=str)))
        assert r["best_so_far"] == h["best_so_far"]
        assert r["round_best"] == h["round_best"]
        assert r["n_evaluated"] == h["n_evaluated"]
        assert r["all_evaluated"] == h["all_evaluated"]

    def test_per_round_recall_keys_survive_json(self):
        h = {
            "best_so_far": [0.5],
            "round_best": [],
            "n_evaluated": [10],
            "all_evaluated": set(),
            "per_round_recall": {5: [0.1], 10: [0.2], 20: [0.3]},
        }
        r = _deserialize_history(json.loads(json.dumps(_serialize_history(h))))
        assert r["per_round_recall"] == h["per_round_recall"]
        assert all(isinstance(k, int) for k in r["per_round_recall"])

    def test_numpy_ints_in_all_evaluated(self):
        h = {
            "best_so_far": [0.5],
            "round_best": [],
            "n_evaluated": [10],
            "all_evaluated": {np.int64(1), np.int64(2), np.int64(3)},
        }
        r = _deserialize_history(json.loads(json.dumps(_serialize_history(h), default=str)))
        assert r["all_evaluated"] == {1, 2, 3}


# ---------------------------------------------------------------------------
# ETA estimator
# ---------------------------------------------------------------------------


class TestETAEstimator:
    def test_empty_history_returns_none(self):
        assert _estimate_remaining_eta([("s1", "rf", 42)], []) is None

    def test_all_observed_uses_eta_label(self):
        completed = [
            {"strategy": "rf", "elapsed": 10.0},
            {"strategy": "rf", "elapsed": 20.0},
            {"strategy": "xgb", "elapsed": 60.0},
        ]
        remaining = [("s1", "rf", 42), ("s1", "xgb", 42)]
        label, secs = _estimate_remaining_eta(remaining, completed)
        assert label == "ETA"
        assert secs == pytest.approx(15.0 + 60.0)

    def test_unseen_strategy_falls_back_to_median(self):
        completed = [
            {"strategy": "rf", "elapsed": 10.0},
            {"strategy": "rf", "elapsed": 20.0},
            {"strategy": "xgb", "elapsed": 60.0},
        ]
        remaining = [("s1", "rf", 42), ("s1", "unseen", 42)]
        label, secs = _estimate_remaining_eta(remaining, completed)
        # rf mean = 15; overall median of [10, 20, 60] = 20
        assert label == "rough ETA"
        assert secs == pytest.approx(15.0 + 20.0)

    def test_mixed_gp_dominated_queue(self):
        """Reproduces the flat-average ETA failure mode from prod runs.

        Pre-fix: 1 slow GP run + 1 fast TS run averaged together and projected
        over a queue dominated by TS runs produced a wildly inflated ETA.
        Per-strategy ETA stays sensible.
        """
        completed = [
            {"strategy": "lnpbo_mixed_logei", "elapsed": 366552.0},
            {"strategy": "lnpbo_mixed_ts", "elapsed": 67.0},
        ]
        remaining = [("s1", "lnpbo_mixed_logei", 42)] + [
            ("s1", "lnpbo_mixed_ts", s) for s in [42, 123, 456, 789, 2024, 999, 111]
        ]
        label, secs = _estimate_remaining_eta(remaining, completed)
        assert label == "ETA"
        assert secs == pytest.approx(366552.0 + 7 * 67.0)


# ---------------------------------------------------------------------------
# Disk persistence
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_state():
    return {
        "completed_rounds": 3,
        "training_idx": [1, 2, 3, 4, 5],
        "pool_idx": [6, 7, 8, 9],
        "history": {
            "best_so_far": [0.5, 0.7, 0.8, 0.9],
            "round_best": [0.6, 0.75, 0.85],
            "n_evaluated": [5, 6, 7, 8],
            "all_evaluated": {1, 2, 3, 4, 5},
            "per_round_recall": {5: [0.1, 0.2, 0.3, 0.4], 10: [0.05, 0.1, 0.15, 0.2]},
        },
    }


@pytest.fixture
def sample_study_info():
    return {"pmid": "12345", "study_id": "TEST_S1", "top_k_pct": {5: 1, 10: 2}, "n_rounds": 5}


class TestDiskPersistence:
    def test_save_load_roundtrip(self, tmp_results_dir, sample_state, sample_study_info):
        save_partial_seed_result("12345", "lnpbo_logei", 42, sample_state, sample_study_info)
        loaded = load_partial_seed_result("12345", "lnpbo_logei", 42, study_id="TEST_S1")
        assert loaded["completed_rounds"] == 3
        assert loaded["training_idx"] == [1, 2, 3, 4, 5]
        assert loaded["pool_idx"] == [6, 7, 8, 9]
        assert loaded["history"]["all_evaluated"] == {1, 2, 3, 4, 5}
        assert loaded["history"]["per_round_recall"] == sample_state["history"]["per_round_recall"]

    def test_clear_removes_file(self, tmp_results_dir, sample_state, sample_study_info):
        save_partial_seed_result("12345", "lnpbo_logei", 42, sample_state, sample_study_info)
        path = _partial_seed_path("12345", "lnpbo_logei", 42, study_id="TEST_S1")
        assert path.exists()
        clear_partial_seed_result("12345", "lnpbo_logei", 42, study_id="TEST_S1")
        assert not path.exists()

    def test_clear_nonexistent_is_silent(self, tmp_results_dir):
        clear_partial_seed_result("99999", "lnpbo_logei", 42, study_id="MISSING")  # no raise

    def test_load_nonexistent_returns_none(self, tmp_results_dir):
        assert load_partial_seed_result("99999", "lnpbo_logei", 42, study_id="MISSING") is None

    def test_corrupt_json_returns_none_and_deletes(
        self, tmp_results_dir, sample_state, sample_study_info, capsys
    ):
        save_partial_seed_result("12345", "lnpbo_logei", 42, sample_state, sample_study_info)
        path = _partial_seed_path("12345", "lnpbo_logei", 42, study_id="TEST_S1")
        path.write_text("{not valid json{{")
        result = load_partial_seed_result("12345", "lnpbo_logei", 42, study_id="TEST_S1")
        assert result is None
        assert not path.exists()
        captured = capsys.readouterr()
        assert "unreadable" in captured.out

    def test_truncated_json_returns_none_and_deletes(
        self, tmp_results_dir, sample_state, sample_study_info
    ):
        save_partial_seed_result("12345", "lnpbo_logei", 42, sample_state, sample_study_info)
        path = _partial_seed_path("12345", "lnpbo_logei", 42, study_id="TEST_S1")
        path.write_text('{"pmid": "12345", "partial": {')
        assert load_partial_seed_result("12345", "lnpbo_logei", 42, study_id="TEST_S1") is None
        assert not path.exists()

    def test_missing_required_keys_returns_none_and_deletes(
        self, tmp_results_dir, sample_state, sample_study_info, capsys
    ):
        save_partial_seed_result("12345", "lnpbo_logei", 42, sample_state, sample_study_info)
        path = _partial_seed_path("12345", "lnpbo_logei", 42, study_id="TEST_S1")
        path.write_text(json.dumps({"pmid": "12345", "partial": {"completed_rounds": 3}}))
        assert load_partial_seed_result("12345", "lnpbo_logei", 42, study_id="TEST_S1") is None
        assert not path.exists()
        captured = capsys.readouterr()
        assert "missing required keys" in captured.out

    def test_atomic_write_leaves_no_tmp(
        self, tmp_results_dir, sample_state, sample_study_info
    ):
        save_partial_seed_result("12345", "lnpbo_logei", 42, sample_state, sample_study_info)
        path = _partial_seed_path("12345", "lnpbo_logei", 42, study_id="TEST_S1")
        tmp = path.with_suffix(path.suffix + ".tmp")
        assert not tmp.exists()
        assert path.exists()

    def test_stray_tmp_does_not_affect_canonical_file(
        self, tmp_results_dir, sample_state, sample_study_info
    ):
        """A SIGKILL during the tmp write must leave the prior canonical
        checkpoint intact and loadable."""
        save_partial_seed_result("12345", "lnpbo_logei", 42, sample_state, sample_study_info)
        path = _partial_seed_path("12345", "lnpbo_logei", 42, study_id="TEST_S1")
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text("{junk")
        loaded = load_partial_seed_result("12345", "lnpbo_logei", 42, study_id="TEST_S1")
        assert loaded["completed_rounds"] == 3


# ---------------------------------------------------------------------------
# Config-drift detection on load
# ---------------------------------------------------------------------------


def _study_info_with(base, **overrides):
    out = dict(base)
    out.update(overrides)
    return out


class TestConfigDriftDetection:
    """``load_partial_seed_result(study_info=...)`` must reject checkpoints
    whose saved study_info disagrees with the current one on any field that
    affects the trajectory or history schema."""

    @pytest.fixture
    def saved_si(self):
        return {
            "pmid": "12345",
            "study_id": "TEST_S1",
            "n_rounds": 15,
            "batch_size": 12,
            "n_seed": 50,
            "feature_type": "lantern",
            "top_k_pct": {5: 5, 10: 10, 20: 20},
            "lnp_ids": ["L1", "L2", "L3"],
        }

    def _save(self, tmp_results_dir, sample_state, si):
        save_partial_seed_result("12345", "lnpbo_logei", 42, sample_state, si)

    def test_matching_study_info_loads(self, tmp_results_dir, sample_state, saved_si):
        self._save(tmp_results_dir, sample_state, saved_si)
        loaded = load_partial_seed_result(
            "12345", "lnpbo_logei", 42, study_id="TEST_S1", study_info=saved_si
        )
        assert loaded is not None and loaded["completed_rounds"] == 3

    def test_batch_size_drift_discards(self, tmp_results_dir, sample_state, saved_si, capsys):
        self._save(tmp_results_dir, sample_state, saved_si)
        new_si = _study_info_with(saved_si, batch_size=8)
        result = load_partial_seed_result(
            "12345", "lnpbo_logei", 42, study_id="TEST_S1", study_info=new_si
        )
        assert result is None
        assert "batch_size" in capsys.readouterr().out

    def test_n_seed_drift_discards(self, tmp_results_dir, sample_state, saved_si, capsys):
        self._save(tmp_results_dir, sample_state, saved_si)
        new_si = _study_info_with(saved_si, n_seed=100)
        assert load_partial_seed_result(
            "12345", "lnpbo_logei", 42, study_id="TEST_S1", study_info=new_si
        ) is None
        assert "n_seed" in capsys.readouterr().out

    def test_feature_type_drift_discards(self, tmp_results_dir, sample_state, saved_si):
        self._save(tmp_results_dir, sample_state, saved_si)
        new_si = _study_info_with(saved_si, feature_type="unimol")
        assert load_partial_seed_result(
            "12345", "lnpbo_logei", 42, study_id="TEST_S1", study_info=new_si
        ) is None

    def test_top_k_pct_key_drift_discards(self, tmp_results_dir, sample_state, saved_si):
        """Changing top_k_pct keys would KeyError in update_history."""
        self._save(tmp_results_dir, sample_state, saved_si)
        new_si = _study_info_with(saved_si, top_k_pct={5: 5, 10: 10, 25: 25})
        assert load_partial_seed_result(
            "12345", "lnpbo_logei", 42, study_id="TEST_S1", study_info=new_si
        ) is None

    def test_lnp_ids_drift_discards(self, tmp_results_dir, sample_state, saved_si):
        self._save(tmp_results_dir, sample_state, saved_si)
        new_si = _study_info_with(saved_si, lnp_ids=["L1", "L2", "L99"])
        assert load_partial_seed_result(
            "12345", "lnpbo_logei", 42, study_id="TEST_S1", study_info=new_si
        ) is None

    def test_lnp_ids_order_does_not_matter(self, tmp_results_dir, sample_state, saved_si):
        self._save(tmp_results_dir, sample_state, saved_si)
        new_si = _study_info_with(saved_si, lnp_ids=["L3", "L1", "L2"])  # same set, reordered
        assert load_partial_seed_result(
            "12345", "lnpbo_logei", 42, study_id="TEST_S1", study_info=new_si
        ) is not None

    def test_n_rounds_extension_is_allowed(self, tmp_results_dir, sample_state, saved_si):
        """Increasing n_rounds from a stored checkpoint is safe — the runner
        just runs more rounds. Decreasing it is unsafe."""
        self._save(tmp_results_dir, sample_state, saved_si)
        new_si = _study_info_with(saved_si, n_rounds=20)
        assert load_partial_seed_result(
            "12345", "lnpbo_logei", 42, study_id="TEST_S1", study_info=new_si
        ) is not None

    def test_n_rounds_truncation_discards(self, tmp_results_dir, sample_state, saved_si):
        """If new n_rounds < saved n_rounds, the checkpoint may report more
        completed rounds than the user now wants — discard."""
        self._save(tmp_results_dir, sample_state, saved_si)
        new_si = _study_info_with(saved_si, n_rounds=2)  # < completed_rounds (3)
        assert load_partial_seed_result(
            "12345", "lnpbo_logei", 42, study_id="TEST_S1", study_info=new_si
        ) is None

    def test_no_study_info_skips_fingerprint_check(self, tmp_results_dir, sample_state, saved_si):
        """Calling load_partial_seed_result without study_info preserves the
        old behavior (pure shape check) — useful for ad-hoc tools."""
        self._save(tmp_results_dir, sample_state, saved_si)
        loaded = load_partial_seed_result("12345", "lnpbo_logei", 42, study_id="TEST_S1")
        assert loaded is not None


# ---------------------------------------------------------------------------
# Casmopolitan resume refusal
# ---------------------------------------------------------------------------


class TestCasmopolitanResumeRefused:
    """An optimizer that maintains cross-round state not captured by the
    checkpoint must refuse to resume rather than silently lose state."""

    def test_optimizer_flags_casmopolitan(self):
        opt = Optimizer(surrogate_type="casmopolitan", acquisition_type="UCB",
                        random_seed=42, batch_size=4)
        assert opt.has_unrepresented_runtime_state is True

    def test_optimizer_flags_gp_as_safe(self):
        opt = Optimizer(surrogate_type="gp", gp_engine="botorch",
                        acquisition_type="LogEI", batch_strategy="kb",
                        kernel_type="matern", random_seed=42, batch_size=4)
        assert opt.has_unrepresented_runtime_state is False

    def test_optimizer_flags_rf_ts_as_safe(self):
        opt = Optimizer(surrogate_type="rf_ts", batch_strategy="greedy",
                        random_seed=42, batch_size=8)
        assert opt.has_unrepresented_runtime_state is False

    def test_runner_refuses_resume_for_casmopolitan(self, synthetic_data):
        df, fcols, seed_idx, oracle_idx, topk = synthetic_data
        opt = Optimizer(surrogate_type="casmopolitan", acquisition_type="UCB",
                        random_seed=42, batch_size=4)
        fake_resume = {
            "completed_rounds": 1,
            "training_idx": list(seed_idx),
            "pool_idx": list(oracle_idx),
            "history": {
                "best_so_far": [0.0, 0.0],
                "round_best": [0.0],
                "n_evaluated": [len(seed_idx), len(seed_idx) + 4],
                "all_evaluated": set(seed_idx),
            },
        }
        with pytest.raises(RuntimeError, match="cross-round runtime state"):
            OptimizerRunner(opt).run(
                df, fcols, seed_idx, oracle_idx, 3, 4,
                top_k_values=topk, resume_state=fake_resume,
            )

    def test_runner_allows_fresh_run_for_casmopolitan(self, synthetic_data):
        """Refusal only fires on resume — fresh runs still work."""
        df, fcols, seed_idx, oracle_idx, topk = synthetic_data
        opt = Optimizer(surrogate_type="casmopolitan", acquisition_type="UCB",
                        random_seed=42, batch_size=4)
        hist = OptimizerRunner(opt).run(
            df, fcols, seed_idx, oracle_idx, 1, 4, top_k_values=topk
        )
        assert "best_so_far" in hist


# ---------------------------------------------------------------------------
# Resume parity (synthetic + real strategies)
# ---------------------------------------------------------------------------


class _InterruptingCallback:
    def __init__(self, k):
        self.k = k
        self.state = None
        self.count = 0

    def __call__(self, state):
        self.state = {
            "completed_rounds": state["completed_rounds"],
            "training_idx": list(state["training_idx"]),
            "pool_idx": list(state["pool_idx"]),
            "history": {
                **state["history"],
                "all_evaluated": set(state["history"]["all_evaluated"]),
            },
        }
        self.count += 1
        if self.count >= self.k:
            raise RuntimeError("INTERRUPT")


def _roundtrip_state(state):
    s = {
        "completed_rounds": int(state["completed_rounds"]),
        "training_idx": [int(i) for i in state["training_idx"]],
        "pool_idx": [int(i) for i in state["pool_idx"]],
        "history": _serialize_history(state["history"]),
    }
    r = json.loads(json.dumps(s, default=str))
    return {
        "completed_rounds": int(r["completed_rounds"]),
        "training_idx": [int(i) for i in r["training_idx"]],
        "pool_idx": [int(i) for i in r["pool_idx"]],
        "history": _deserialize_history(r["history"]),
    }


def _assert_history_parity(h1, h2):
    assert list(h1["best_so_far"]) == list(h2["best_so_far"])
    assert list(h1["round_best"]) == list(h2["round_best"])
    assert list(h1["n_evaluated"]) == list(h2["n_evaluated"])
    assert set(h1["all_evaluated"]) == set(h2["all_evaluated"])
    if "per_round_recall" in h1:
        assert h1["per_round_recall"] == h2["per_round_recall"]


class _DeterministicOptimizer:
    """Picks the batch_size highest-y items in pool — no RNG."""

    def suggest_indices(self, df, feature_cols, training_idx, pool_idx, round_num,
                        encoded_dataset=None, batch_size=8):
        ranked = df.loc[pool_idx, "Experiment_value"].sort_values(ascending=False)
        return list(ranked.index[:batch_size])


class TestResumeParity:
    def test_synthetic_deterministic_optimizer(self, synthetic_data):
        df, fcols, seed_idx, oracle_idx, topk = synthetic_data
        n_rounds, bs = 5, 8

        hist_full = OptimizerRunner(_DeterministicOptimizer()).run(
            df, fcols, seed_idx, oracle_idx, n_rounds, bs, top_k_values=topk
        )

        cb = _InterruptingCallback(2)
        with pytest.raises(RuntimeError, match="INTERRUPT"):
            OptimizerRunner(_DeterministicOptimizer()).run(
                df, fcols, seed_idx, oracle_idx, n_rounds, bs,
                top_k_values=topk, checkpoint_callback=cb,
            )
        resume = _roundtrip_state(cb.state)
        hist_res = OptimizerRunner(_DeterministicOptimizer()).run(
            df, fcols, seed_idx, oracle_idx, n_rounds, bs,
            top_k_values=topk, resume_state=resume,
        )
        _assert_history_parity(hist_full, hist_res)

    def test_real_rf_ts(self, synthetic_data):
        df, fcols, seed_idx, oracle_idx, topk = synthetic_data
        kw = dict(surrogate_type="rf_ts", batch_strategy="greedy",
                  random_seed=42, kappa=5.0, normalize="copula", batch_size=8)
        n_rounds, bs = 4, 8

        hist_full = OptimizerRunner(Optimizer(**kw)).run(
            df, fcols, seed_idx, oracle_idx, n_rounds, bs, top_k_values=topk
        )

        cb = _InterruptingCallback(2)
        with pytest.raises(RuntimeError, match="INTERRUPT"):
            OptimizerRunner(Optimizer(**kw)).run(
                df, fcols, seed_idx, oracle_idx, n_rounds, bs,
                top_k_values=topk, checkpoint_callback=cb,
            )
        resume = _roundtrip_state(cb.state)
        hist_res = OptimizerRunner(Optimizer(**kw)).run(
            df, fcols, seed_idx, oracle_idx, n_rounds, bs,
            top_k_values=topk, resume_state=resume,
        )
        _assert_history_parity(hist_full, hist_res)

    def test_real_gp_logei(self, synthetic_data):
        df, fcols, seed_idx, oracle_idx, topk = synthetic_data
        kw = dict(surrogate_type="gp", gp_engine="botorch",
                  acquisition_type="LogEI", batch_strategy="kb",
                  kernel_type="matern", random_seed=42, kappa=5.0, xi=0.01,
                  normalize="copula", batch_size=4)
        n_rounds, bs = 3, 4

        hist_full = OptimizerRunner(Optimizer(**kw)).run(
            df, fcols, seed_idx, oracle_idx, n_rounds, bs, top_k_values=topk
        )

        cb = _InterruptingCallback(1)
        with pytest.raises(RuntimeError, match="INTERRUPT"):
            OptimizerRunner(Optimizer(**kw)).run(
                df, fcols, seed_idx, oracle_idx, n_rounds, bs,
                top_k_values=topk, checkpoint_callback=cb,
            )
        resume = _roundtrip_state(cb.state)
        hist_res = OptimizerRunner(Optimizer(**kw)).run(
            df, fcols, seed_idx, oracle_idx, n_rounds, bs,
            top_k_values=topk, resume_state=resume,
        )
        _assert_history_parity(hist_full, hist_res)

    def test_resume_at_final_round_is_noop(self, synthetic_data):
        """If completed_rounds==n_rounds, the loop must not run again."""
        df, fcols, seed_idx, oracle_idx, topk = synthetic_data
        n_rounds, bs = 3, 8

        hist_full = OptimizerRunner(_DeterministicOptimizer()).run(
            df, fcols, seed_idx, oracle_idx, n_rounds, bs, top_k_values=topk
        )
        resume = _roundtrip_state(
            {
                "completed_rounds": n_rounds,
                "training_idx": list(seed_idx) + [],
                "pool_idx": list(oracle_idx),
                "history": hist_full,
            }
        )
        hist_res = OptimizerRunner(_DeterministicOptimizer()).run(
            df, fcols, seed_idx, oracle_idx, n_rounds, bs,
            top_k_values=topk, resume_state=resume,
        )
        # No new rounds were added — the loaded history is returned as-is.
        assert list(hist_res["round_best"]) == list(hist_full["round_best"])
