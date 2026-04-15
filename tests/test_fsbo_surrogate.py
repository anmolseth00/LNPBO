"""Regression tests for the warm-start GP transfer baseline helpers."""

import numpy as np

from LNPBO.models.experimental.warm_start_gp_transfer_baseline import _normalize_X, _prepare_meta_train_subset


def test_prepare_meta_train_subset_uses_provided_bounds():
    X_train = np.array(
        [
            [2.0, 4.0],
            [4.0, 8.0],
            [6.0, 12.0],
            [8.0, 16.0],
        ],
        dtype=float,
    )
    y_train = np.array([0.1, 0.2, 0.3, 0.4], dtype=float)
    study_ids = np.array(["A", "A", "B", "B"])
    bounds = (np.array([0.0, 0.0]), np.array([10.0, 20.0]))

    X_sub_norm, y_sub, bounds_used, sub_idx, *_ = _prepare_meta_train_subset(
        X_train,
        y_train,
        study_ids,
        train_study_ids=np.array(["A", "B"]),
        max_meta_n=10,
        seed=0,
        bounds=bounds,
    )

    assert np.allclose(X_sub_norm, X_train[sub_idx] / bounds[1])
    assert np.allclose(y_sub, y_train[sub_idx])
    assert np.allclose(bounds_used[0], bounds[0])
    assert np.allclose(bounds_used[1], bounds[1])


def test_prepare_meta_train_subset_matches_deployment_scale_from_raw_features():
    X_raw = np.array(
        [
            [2.0, 4.0],
            [4.0, 8.0],
            [6.0, 12.0],
            [8.0, 16.0],
        ],
        dtype=float,
    )
    y_train = np.array([0.1, 0.2, 0.3, 0.4], dtype=float)
    study_ids = np.array(["A", "A", "B", "B"])

    _, deployment_bounds = _normalize_X(X_raw)
    X_norm, _ = _normalize_X(X_raw, bounds=deployment_bounds)

    X_sub_norm, _, _, sub_idx, *_ = _prepare_meta_train_subset(
        X_raw,
        y_train,
        study_ids,
        train_study_ids=np.array(["A", "B"]),
        max_meta_n=10,
        seed=0,
        bounds=deployment_bounds,
    )

    assert np.allclose(X_sub_norm, X_norm[sub_idx])
