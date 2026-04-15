"""Regression tests for acquisition functions."""

import numpy as np

from LNPBO.optimization import acquisition as acquisition_mod
from LNPBO.optimization.acquisition import ThompsonSamplingBatch


class _DummySpace:
    def __init__(self, candidates: np.ndarray):
        self._candidates = candidates

    def __len__(self) -> int:
        return 1

    def random_sample(self, n_random: int, random_state=None) -> np.ndarray:
        assert n_random == len(self._candidates)
        return self._candidates.copy()


def test_thompson_sampling_batch_draws_joint_sample_once(monkeypatch):
    candidates = np.array([[0.1], [0.3], [0.9]], dtype=float)
    calls = {}

    def fake_sample_f(gp, X, n_samples=1, random_state=None):
        calls["count"] = calls.get("count", 0) + 1
        calls["X"] = X.copy()
        return np.array([[0.2], [1.1], [0.5]], dtype=float)

    monkeypatch.setattr(acquisition_mod, "_sample_f", fake_sample_f)

    acq = ThompsonSamplingBatch()
    x_best = acq.suggest(object(), _DummySpace(candidates), n_random=3, n_smart=0, fit_gp=False, random_state=0)

    assert calls["count"] == 1
    assert np.allclose(calls["X"], candidates)
    assert np.allclose(x_best, candidates[1])
