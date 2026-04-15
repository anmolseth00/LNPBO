"""Exactness tests for the Deshpande et al. online recalibration path."""

from __future__ import annotations

import numpy as np

from LNPBO.optimization.online_conformal import (
    CalibratedProbabilisticModel,
    ExactOnlineRecalibrator,
    RecalibrationDataset,
    build_recalibration_dataset,
)


class _ToyProbabilisticModel:
    def __init__(self, shift: float = 0.0, scale: float = 1.0):
        self.shift = float(shift)
        self.scale = float(scale)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_ToyProbabilisticModel":
        del X
        self.offset_ = float(np.mean(y)) + self.shift
        return self

    def quantile(self, X: np.ndarray, p: float) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return np.full(len(X), self.offset_ + self.scale * float(p), dtype=float)

    def cdf(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        del X
        return np.clip((np.asarray(y, dtype=float).ravel() - self.offset_) / self.scale, 0.0, 1.0)

    def inverse_quantile_level(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.cdf(X, y)


def test_build_recalibration_dataset_uses_exact_leave_one_out_construction() -> None:
    X = np.array([[0.0], [1.0], [2.0]], dtype=float)
    y = np.array([0.1, 0.4, 0.9], dtype=float)

    dataset = build_recalibration_dataset(
        X,
        y,
        model_factory=lambda: _ToyProbabilisticModel(shift=0.0, scale=1.0),
    )

    expected = np.array(
        [
            0.1 - np.mean([0.4, 0.9]),
            0.4 - np.mean([0.1, 0.9]),
            0.9 - np.mean([0.1, 0.4]),
        ],
        dtype=float,
    )
    expected = np.clip(expected, 0.0, 1.0)
    assert np.allclose(dataset.inverse_quantile_levels, expected)


def test_exact_recalibrator_matches_eq11_objective_solution() -> None:
    dataset = RecalibrationDataset(np.array([0.15, 0.2, 0.8], dtype=float))
    recalibrator = ExactOnlineRecalibrator(eta=0.2).fit(dataset)
    p = 0.75

    q_hist = []
    q = 0.0
    grads = []
    for u in dataset.inverse_quantile_levels:
        grad = float(u <= q) - p
        grads.append(grad)
        q_hist.append(q)
        q = float(np.clip(q - 0.2 * grad, 0.0, 1.0))

    objective_grid = np.linspace(0.0, 1.0, 2001)
    objective = objective_grid**2 / (2 * 0.2)
    for q_s, grad in zip(q_hist, grads):
        objective += (objective_grid - q_s) * grad
    q_star = float(objective_grid[np.argmin(objective)])

    assert abs(recalibrator.recalibrate(p) - q_star) < 1e-3


def test_exact_recalibrator_is_monotone_and_avoids_quantile_crossing() -> None:
    recalibrator = ExactOnlineRecalibrator(eta=0.1).fit(
        RecalibrationDataset(np.array([0.05, 0.1, 0.2, 0.6, 0.8], dtype=float))
    )
    p_grid = np.linspace(0.05, 0.95, 19)
    q_grid = np.array([recalibrator.recalibrate(p) for p in p_grid])

    assert np.all(np.diff(q_grid) >= -1e-9)

    model = _ToyProbabilisticModel(scale=2.0).fit(np.array([[0.0], [1.0]]), np.array([0.2, 0.3]))
    calibrated = CalibratedProbabilisticModel(model, recalibrator)
    quantiles = np.array([calibrated.quantile(np.array([[0.5]]), p)[0] for p in p_grid])
    assert np.all(np.diff(quantiles) >= -1e-9)


def test_exact_recalibration_improves_coverage_on_miscalibrated_levels() -> None:
    rng = np.random.RandomState(0)
    u_cal = rng.beta(0.4, 1.0, size=512)
    u_test = rng.beta(0.4, 1.0, size=512)
    recalibrator = ExactOnlineRecalibrator(eta=0.05).fit(RecalibrationDataset(u_cal))

    p = 0.9
    raw_coverage = float(np.mean(u_test <= p))
    calibrated_coverage = float(np.mean(u_test <= recalibrator.recalibrate(p)))

    assert abs(calibrated_coverage - p) < abs(raw_coverage - p)


def test_calibrated_acquisitions_move_in_expected_direction_for_overconfident_model() -> None:
    recalibrator = ExactOnlineRecalibrator(eta=0.1).fit(
        RecalibrationDataset(np.array([0.02, 0.03, 0.05, 0.08, 0.1], dtype=float))
    )
    base_model = _ToyProbabilisticModel(scale=1.0).fit(np.array([[0.0], [1.0]]), np.array([0.0, 0.2]))
    calibrated = CalibratedProbabilisticModel(base_model, recalibrator)

    X = np.array([[0.5]], dtype=float)
    p_ucb = 0.9
    incumbent = 0.6

    raw_ucb = base_model.quantile(X, p_ucb)[0]
    calibrated_ucb = calibrated.quantile(X, p_ucb)[0]
    raw_pi = 1.0 - base_model.cdf(X, np.array([incumbent]))[0]
    calibrated_pi = calibrated.probability_of_improvement(X, incumbent)[0]
    raw_grid = np.linspace(float(base_model.cdf(X, np.array([incumbent]))[0]), 1.0, 256)
    raw_ei = float(
        np.trapezoid(
            np.array(
                [max(float(base_model.quantile(X, float(p))[0]) - incumbent, 0.0) for p in raw_grid],
                dtype=float,
            ),
            raw_grid,
        )
    )
    calibrated_ei = calibrated.expected_improvement(X, incumbent)[0]

    assert calibrated_ucb < raw_ucb
    assert calibrated_pi < raw_pi
    assert calibrated_ei < raw_ei
