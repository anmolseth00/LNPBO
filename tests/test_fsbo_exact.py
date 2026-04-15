"""Exactness tests for the deep-kernel FSBO implementation."""

from __future__ import annotations

import tempfile

import numpy as np
import torch
from gpytorch.mlls import ExactMarginalLogLikelihood

from LNPBO.models.experimental.fsbo import (
    build_meta_initialized_model,
    evolutionary_warm_start,
    meta_train_fsbo,
    optimize_gp_model,
    predict_mean_std,
    run_fsbo_bo_loop,
    sample_task_scaling_bounds,
    augment_task_labels,
)
from LNPBO.models.experimental.warm_start_gp_transfer_baseline import _build_warm_gp, meta_train_gp


def _make_multitask_sine_data():
    x = np.linspace(0.0, 1.0, 18, dtype=np.float64)[:, None]
    source_specs = [
        ("source_a", 0.9, -0.2),
        ("source_b", 1.3, 0.1),
        ("source_c", 0.6, 0.4),
    ]
    target_spec = ("target", 1.1, 0.25)

    X_parts = []
    y_parts = []
    study_ids = []
    for study_id, scale, offset in [*source_specs, target_spec]:
        y = scale * np.sin(2.0 * np.pi * x[:, 0]) + offset
        X_parts.append(x.copy())
        y_parts.append(y.astype(np.float64))
        study_ids.extend([study_id] * len(x))

    return (
        np.vstack(X_parts),
        np.concatenate(y_parts),
        np.asarray(study_ids),
        np.asarray([spec[0] for spec in source_specs]),
        np.asarray([target_spec[0]]),
    )


def _negative_log_marginal_likelihood(model, likelihood, X, y) -> float:
    train_x = torch.tensor(X, dtype=torch.float64)
    train_y = torch.tensor(y, dtype=torch.float64)
    model.set_train_data(train_x, train_y, strict=False)
    model.train()
    likelihood.train()
    mll = ExactMarginalLogLikelihood(likelihood, model)
    loss = -mll(model(train_x), train_y)
    model.eval()
    likelihood.eval()
    return float(loss.item())


def test_task_augmentation_samples_ordered_bounds_and_applies_affine_transform() -> None:
    rng = np.random.RandomState(0)
    lower, upper = sample_task_scaling_bounds(-2.0, 3.0, rng)
    y = np.array([-1.0, 0.5, 2.0], dtype=np.float64)
    augmented = augment_task_labels(y, lower, upper)

    assert -2.0 <= lower < upper <= 3.0
    assert np.allclose(augmented, (y - lower) / (upper - lower))


def test_evolutionary_warm_start_optimizes_set_objective() -> None:
    loss_matrix = np.array(
        [
            [0.90, 0.20, 0.10, 0.70],
            [0.80, 0.10, 0.60, 0.20],
        ],
        dtype=np.float64,
    )
    chosen = evolutionary_warm_start(
        loss_matrix,
        set_size=2,
        population_size=16,
        generations=48,
        elite_size=4,
        seed=0,
    )
    assert set(chosen.tolist()) == {1, 2}


def test_meta_train_fsbo_returns_deep_kernel_state_and_predictive_posterior() -> None:
    X, y, study_ids, train_ids, _ = _make_multitask_sine_data()
    train_mask = np.isin(study_ids, train_ids)

    meta_state = meta_train_fsbo(
        X[train_mask],
        y[train_mask],
        study_ids[train_mask],
        train_ids,
        hidden_dims=(8, 8),
        n_iterations=8,
        batch_size=6,
        seed=0,
    )

    source_idx = np.where(study_ids == train_ids[0])[0]
    model, likelihood = build_meta_initialized_model(X[source_idx], y[source_idx], meta_state=meta_state)
    mean, std = predict_mean_std(model, likelihood, X[source_idx])

    assert meta_state.hidden_dims == (8, 8)
    assert meta_state.base_kernel == "rbf"
    assert len(meta_state.meta_losses) == 8
    assert mean.shape == (len(source_idx),)
    assert std.shape == (len(source_idx),)
    assert np.all(std > 0.0)


def test_target_finetuning_reduces_exact_gp_negative_log_marginal_likelihood() -> None:
    X, y, study_ids, train_ids, test_ids = _make_multitask_sine_data()
    train_mask = np.isin(study_ids, train_ids)
    target_idx = np.where(study_ids == test_ids[0])[0][:6]

    meta_state = meta_train_fsbo(
        X[train_mask],
        y[train_mask],
        study_ids[train_mask],
        train_ids,
        hidden_dims=(8, 8),
        n_iterations=10,
        batch_size=6,
        seed=1,
    )

    model, likelihood = build_meta_initialized_model(X[target_idx], y[target_idx], meta_state=meta_state)
    before = _negative_log_marginal_likelihood(model, likelihood, X[target_idx], y[target_idx])
    optimize_gp_model(model, likelihood, X[target_idx], y[target_idx], n_steps=12, lr=1e-2)
    after = _negative_log_marginal_likelihood(model, likelihood, X[target_idx], y[target_idx])

    assert after < before


def test_exact_path_is_behaviorally_distinct_from_warm_start_gp_baseline() -> None:
    X, y, study_ids, train_ids, test_ids = _make_multitask_sine_data()
    train_mask = np.isin(study_ids, train_ids)
    target_idx = np.where(study_ids == test_ids[0])[0][:6]

    exact_state = meta_train_fsbo(
        X[train_mask],
        y[train_mask],
        study_ids[train_mask],
        train_ids,
        hidden_dims=(8, 8),
        n_iterations=8,
        batch_size=6,
        seed=2,
    )
    exact_model, exact_likelihood = build_meta_initialized_model(X[target_idx], y[target_idx], meta_state=exact_state)
    exact_mean, _ = predict_mean_std(exact_model, exact_likelihood, X[target_idx])

    warm_params, _ = meta_train_gp(
        X[train_mask],
        y[train_mask],
        study_ids[train_mask],
        train_ids,
        max_meta_n=32,
        seed=2,
    )
    warm_model = _build_warm_gp(X[target_idx], y[target_idx], warm_params)
    with torch.no_grad():
        warm_mean = warm_model.posterior(torch.tensor(X[target_idx], dtype=torch.float64)).mean.squeeze(-1).numpy()

    assert not np.allclose(exact_mean, warm_mean)


def test_run_bo_loop_supports_fsbo_warm_start_and_ei_selection() -> None:
    X, y, study_ids, train_ids, test_ids = _make_multitask_sine_data()
    train_mask = np.isin(study_ids, train_ids)
    meta_state = meta_train_fsbo(
        X[train_mask],
        y[train_mask],
        study_ids[train_mask],
        train_ids,
        hidden_dims=(8, 8),
        n_iterations=8,
        batch_size=6,
        seed=3,
    )

    results = run_fsbo_bo_loop(
        X,
        y,
        study_ids,
        test_ids,
        [3],
        build_model_fn=lambda Xo, yo: build_meta_initialized_model(Xo, yo, meta_state=meta_state),
        acquisition_label="FSBO-EI",
        source_study_ids=train_ids,
        meta_state=meta_state,
        warm_start=True,
        n_bo_rounds=2,
        batch_size=1,
        target_finetune_steps=4,
        target_finetune_lr=1e-2,
        seed=3,
    )

    assert 3 in results
    assert results[3]["n_studies_evaluated"] == 1
    assert len(results[3]["per_study"]) == 1
    assert results[3]["per_study"][0]["n_observed"] == 5


def test_meta_state_serialization_round_trips_prediction_path() -> None:
    X, y, study_ids, train_ids, _ = _make_multitask_sine_data()
    train_mask = np.isin(study_ids, train_ids)
    source_idx = np.where(study_ids == train_ids[0])[0]

    meta_state = meta_train_fsbo(
        X[train_mask],
        y[train_mask],
        study_ids[train_mask],
        train_ids,
        hidden_dims=(8, 8),
        n_iterations=8,
        batch_size=6,
        seed=4,
    )
    model_a, likelihood_a = build_meta_initialized_model(X[source_idx], y[source_idx], meta_state=meta_state)
    mean_a, std_a = predict_mean_std(model_a, likelihood_a, X[source_idx])

    with tempfile.TemporaryDirectory() as tmpdir:
        path = f"{tmpdir}/meta_state.pt"
        torch.save(meta_state, path)
        loaded = torch.load(path, weights_only=False)

    model_b, likelihood_b = build_meta_initialized_model(X[source_idx], y[source_idx], meta_state=loaded)
    mean_b, std_b = predict_mean_std(model_b, likelihood_b, X[source_idx])

    assert np.allclose(mean_a, mean_b)
    assert np.allclose(std_a, std_b)
