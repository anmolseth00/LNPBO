"""Regression tests for CASMOPOLITAN trust-region behavior."""

import numpy as np
import pandas as pd

from LNPBO.benchmarks._optimizer_runner import OptimizerRunner
from LNPBO.optimization.casmopolitan import (
    ExponentiatedCategoricalKernel,
    MixedCasmopolitanKernel,
    TrustRegion,
    _apply_trust_region_penalty,
    _fit_pool_casmopolitan_gp,
    score_pool_casmopolitan,
)
from LNPBO.optimization.optimizer import Optimizer


def test_apply_trust_region_penalty_never_improves_out_of_tr_scores():
    trust_region = TrustRegion(
        center_cat=np.array([0.0, 0.0]),
        center_cont=np.array([0.0]),
        length=0.1,
        n_cat_dims=2,
        n_cont_dims=1,
        cont_bounds=np.array([[-1.0, 1.0]]),
    )
    scores = np.array([-1.0, -2.0, 3.0], dtype=float)
    X_pool_mixed = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 0.5],
        ],
        dtype=float,
    )

    penalized = _apply_trust_region_penalty(scores, X_pool_mixed, trust_region, n_cat_dims=2)

    assert penalized[0] == scores[0]
    assert penalized[1] < scores[1]
    assert penalized[2] < scores[2]


def test_score_pool_casmopolitan_respects_trust_length():
    X_train = np.array([[0.0, 0.0], [0.0, 0.1], [1.0, 1.0]], dtype=float)
    y_train = np.array([1.0, 0.8, 0.0], dtype=float)
    X_pool = np.array([[0.0, 0.05], [0.0, 0.8], [1.0, 1.0]], dtype=float)

    _, small_scores = score_pool_casmopolitan(
        X_train,
        y_train,
        X_pool,
        il_cat_train=X_train[:, 0].astype(int),
        il_cat_pool=X_pool[:, 0].astype(int),
        cont_feature_indices=[1],
        cat_feature_indices=[0],
        batch_size=3,
        trust_length=0.05,
        acq_func="ucb",
        random_seed=0,
    )
    _, large_scores = score_pool_casmopolitan(
        X_train,
        y_train,
        X_pool,
        il_cat_train=X_train[:, 0].astype(int),
        il_cat_pool=X_pool[:, 0].astype(int),
        cont_feature_indices=[1],
        cat_feature_indices=[0],
        batch_size=3,
        trust_length=10.0,
        acq_func="ucb",
        random_seed=0,
    )

    assert small_scores[1] < large_scores[1]


def test_optimizer_runner_resets_casmopolitan_state_between_runs():
    df = pd.DataFrame(
        {
            "IL_name": ["A", "A", "B", "B", "C", "C"],
            "feat": [0.0, 0.05, 0.9, 0.95, 0.4, 0.45],
            "Experiment_value": [0.2, 0.25, 0.8, 0.85, 0.3, 0.35],
        }
    )

    def run_once(opt: Optimizer) -> set[int]:
        history = OptimizerRunner(opt).run(
            df,
            ["feat"],
            seed_idx=[0, 2],
            oracle_idx=[1, 3, 4, 5],
            n_rounds=1,
            batch_size=1,
        )
        return {int(i) for i in history["all_evaluated"]}

    fresh = Optimizer(
        surrogate_type="casmopolitan",
        acquisition_type="UCB",
        batch_strategy="kb",
        normalize="none",
        random_seed=0,
    )
    stale = Optimizer(
        surrogate_type="casmopolitan",
        acquisition_type="UCB",
        batch_strategy="kb",
        normalize="none",
        random_seed=0,
    )
    stale._casmopolitan_trust_region = TrustRegion(
        center_cat=np.array([99.0]),
        center_cont=np.array([99.0]),
        length=0.01,
        n_cat_dims=1,
        n_cont_dims=1,
        cont_bounds=np.array([[0.0, 1.0]]),
    )
    stale._casmopolitan_round_start_best = 100.0

    assert run_once(stale) == run_once(fresh)


def test_optimizer_casmopolitan_prefers_in_tr_pool_rows_when_available():
    df = pd.DataFrame(
        {
            "IL_name": ["IL1", "IL0", "IL1", "IL0"],
            "feat": [10.0, 0.0, 5.0, 9.9],
            "Experiment_value": [10.0, 0.0, 0.5, 0.4],
        }
    )

    opt = Optimizer(
        surrogate_type="casmopolitan",
        acquisition_type="UCB",
        batch_strategy="kb",
        normalize="none",
        random_seed=0,
    )
    history = OptimizerRunner(opt).run(
        df,
        ["feat"],
        seed_idx=[0, 1],
        oracle_idx=[2, 3],
        n_rounds=1,
        batch_size=1,
    )

    selected = sorted(int(i) for i in history["all_evaluated"] - {0, 1})
    assert selected == [3]


def test_fit_pool_casmopolitan_keeps_tr_bounds_valid_when_incumbent_exceeds_pool():
    X_train = np.array([[1.0, 10.0], [0.0, 0.0]], dtype=float)
    y_train = np.array([10.0, 0.0], dtype=float)
    X_pool = np.array([[1.0, 5.0], [0.0, 9.9]], dtype=float)

    _, _, _, trust_region = _fit_pool_casmopolitan_gp(
        X_train,
        y_train,
        X_pool,
        cont_feature_indices=[1],
        cat_feature_indices=[0],
        random_seed=0,
        trust_length=0.1,
    )

    bounds = trust_region.get_cont_bounds()
    assert np.all(bounds[:, 0] <= bounds[:, 1])


def test_exponentiated_categorical_kernel_learns_per_dimension_relevance():
    kernel = ExponentiatedCategoricalKernel(n_cat_dims=2, lengthscales=(4.0, 0.1))
    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=float)
    K = kernel(X)

    assert K[0, 2] > K[0, 1]


def test_trust_region_signals_restart_at_min_length():
    trust_region = TrustRegion(
        center_cat=np.array([0.0]),
        center_cont=np.array([0.0]),
        length=0.02,
        n_cat_dims=1,
        n_cont_dims=1,
        cont_bounds=np.array([[-1.0, 1.0]]),
        length_min=0.01,
        failure_tol=1,
    )

    assert trust_region.update(improved=False) is True
    assert trust_region.length == trust_region.length_min


def test_mixed_casmopolitan_kernel_exposes_mix_and_categorical_hyperparameters():
    kernel = MixedCasmopolitanKernel(n_cat_dims=2, n_cont_dims=1)
    assert kernel.n_dims == 1 + 2 + kernel.cont_kernel.n_dims
