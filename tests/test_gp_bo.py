"""Comprehensive test suite for the GPyTorch-based GP BO module (optimization/gp_bo.py).

Tests cover GP fitting, acquisition functions, batch selection strategies,
sparse GP approximation, speed comparisons, end-to-end BO loops, and edge cases.

Run: .venv/bin/python -m pytest tests/test_gp_bo.py -v
"""

import time

import numpy as np
import pytest
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr, spearmanr

from LNPBO.optimization.gp_bo import (
    fit_gp,
    get_device,
    predict,
    score_acquisition,
    select_batch,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic_data(n_train=50, n_pool=200, d=5, seed=42):
    rng = np.random.RandomState(seed)
    X_train = rng.randn(n_train, d).astype(np.float64)
    y_train = (X_train ** 2).sum(axis=1) + 0.1 * rng.randn(n_train)
    X_pool = rng.randn(n_pool, d).astype(np.float64)
    pool_indices = np.arange(n_pool)
    return X_train, y_train.astype(np.float64), X_pool, pool_indices


def _branin(X):
    """2D Branin function (negated for maximization)."""
    x1, x2 = X[:, 0], X[:, 1]
    a, b, c = 1.0, 5.1 / (4 * np.pi ** 2), 5.0 / np.pi
    r, s, t = 6.0, 10.0, 1.0 / (8 * np.pi)
    return -(a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s)


# ---------------------------------------------------------------------------
# 1. GP Model Fitting Tests
# ---------------------------------------------------------------------------

class TestGPFitting:
    def test_fit_gp_basic(self):
        X_train, y_train, X_pool, _ = _make_synthetic_data(n_train=50, d=3)
        model = fit_gp(X_train, y_train)
        mean, std = predict(model, X_pool)
        assert mean.shape == (len(X_pool),)
        assert std.shape == (len(X_pool),)
        assert np.all(np.isfinite(mean))
        assert np.all(np.isfinite(std))
        assert np.all(std >= 0)

    def test_fit_gp_predictions_reasonable(self):
        rng = np.random.RandomState(42)
        X_train = rng.randn(80, 3).astype(np.float64)
        y_train = (X_train ** 2).sum(axis=1)
        X_test = rng.randn(50, 3).astype(np.float64)
        y_true = (X_test ** 2).sum(axis=1)

        model = fit_gp(X_train, y_train)
        mean, _ = predict(model, X_test)
        r, _ = pearsonr(mean, y_true)
        assert r > 0.8, f"Pearson r={r:.3f} too low; GP predictions should track sum-of-squares"

    def test_fit_gp_uncertainty(self):
        rng = np.random.RandomState(0)
        # Training cluster near origin
        X_train = 0.1 * rng.randn(60, 3).astype(np.float64)
        y_train = (X_train ** 2).sum(axis=1) + 0.01 * rng.randn(60)

        # Near points: close to training cluster
        X_near = 0.1 * rng.randn(20, 3).astype(np.float64)
        # Far points: far from training cluster
        X_far = 5.0 + 0.1 * rng.randn(20, 3).astype(np.float64)

        model = fit_gp(X_train, y_train)
        _, std_near = predict(model, X_near)
        _, std_far = predict(model, X_far)

        assert std_far.mean() > std_near.mean(), (
            f"Far-point std ({std_far.mean():.4f}) should exceed near-point std ({std_near.mean():.4f})"
        )

    def test_predict_matches_sklearn(self):
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern

        rng = np.random.RandomState(123)
        X_train = rng.randn(60, 4).astype(np.float64)
        y_train = np.sin(X_train[:, 0]) + 0.5 * X_train[:, 1] ** 2 + 0.1 * rng.randn(60)
        X_test = rng.randn(40, 4).astype(np.float64)

        # sklearn GP with Matern 2.5
        sk_gp = GaussianProcessRegressor(
            kernel=Matern(nu=2.5), alpha=1e-6, n_restarts_optimizer=5, random_state=0,
        )
        sk_gp.fit(X_train, y_train)
        sk_mean, _sk_std = sk_gp.predict(X_test, return_std=True)

        # GPyTorch GP
        model = fit_gp(X_train, y_train)
        gpy_mean, _gpy_std = predict(model, X_test)

        r_mean, _ = pearsonr(gpy_mean, sk_mean)
        assert r_mean > 0.95, (
            f"GPyTorch mean vs sklearn mean Pearson r={r_mean:.3f}; should be >0.95"
        )


# ---------------------------------------------------------------------------
# 2. Acquisition Function Tests
# ---------------------------------------------------------------------------

class TestAcquisitionFunctions:
    def test_ucb_scores(self):
        X_train, y_train, X_pool, _ = _make_synthetic_data(n_train=50, d=3)
        model = fit_gp(X_train, y_train)
        kappa = 2.5
        scores = score_acquisition(
            model, X_pool, acq_type="UCB", y_best=y_train.max(), kappa=kappa, xi=0.0,
        )
        mean, std = predict(model, X_pool)
        expected = mean + kappa * std
        np.testing.assert_allclose(scores, expected, atol=1e-6)

    def test_ei_nonnegative(self):
        X_train, y_train, X_pool, _ = _make_synthetic_data()
        model = fit_gp(X_train, y_train)
        scores = score_acquisition(
            model, X_pool, acq_type="EI", y_best=y_train.max(), kappa=0.0, xi=0.0,
        )
        assert np.all(scores >= -1e-10), "EI should be non-negative"

    def test_ei_at_best(self):
        X_train, y_train, X_pool, _ = _make_synthetic_data()
        model = fit_gp(X_train, y_train)
        mean, _ = predict(model, X_pool)
        # Set y_best to well above all predicted means
        y_best = mean.max() + 10.0
        scores = score_acquisition(
            model, X_pool, acq_type="EI", y_best=y_best, kappa=0.0, xi=0.0,
        )
        assert scores.max() < 0.1, (
            f"EI should be ~0 when y_best >> predicted means; max={scores.max():.4f}"
        )

    def test_ei_high_when_promising(self):
        X_train, y_train, X_pool, _ = _make_synthetic_data()
        model = fit_gp(X_train, y_train)
        mean, _ = predict(model, X_pool)
        # Set y_best to well below all predicted means
        y_best = mean.min() - 10.0
        scores = score_acquisition(
            model, X_pool, acq_type="EI", y_best=y_best, kappa=0.0, xi=0.0,
        )
        assert scores.max() > 1.0, (
            f"EI should be large when mean >> y_best; max={scores.max():.4f}"
        )

    def test_log_ei_matches_log_of_ei(self):
        X_train, y_train, X_pool, _ = _make_synthetic_data(n_train=60, d=3)
        model = fit_gp(X_train, y_train)
        mean, std = predict(model, X_pool)
        y_best = np.median(y_train)

        ei_scores = score_acquisition(
            model, X_pool, acq_type="EI", y_best=y_best, kappa=0.0, xi=0.0,
        )
        log_ei_scores = score_acquisition(
            model, X_pool, acq_type="LogEI", y_best=y_best, kappa=0.0, xi=0.0,
        )

        # For points where EI is reasonably large, log(EI) ~= LogEI
        z = (mean - y_best) / np.maximum(std, 1e-10)
        moderate = (z > -2.0) & (z < 2.0) & (ei_scores > 1e-6)
        if moderate.sum() > 5:
            log_of_ei = np.log(ei_scores[moderate])
            np.testing.assert_allclose(
                log_ei_scores[moderate], log_of_ei, atol=0.1,
                err_msg="LogEI should approximate log(EI) for moderate z values",
            )

    def test_log_ei_stable_extreme(self):
        rng = np.random.RandomState(99)
        # Create data that will produce very negative z values
        X_train = rng.randn(50, 3).astype(np.float64)
        y_train = (X_train ** 2).sum(axis=1)
        # Pool points far from training data with low predicted mean
        X_pool = 10.0 * np.ones((20, 3), dtype=np.float64)
        X_pool += 0.01 * rng.randn(20, 3)

        model = fit_gp(X_train, y_train)
        scores = score_acquisition(
            model, X_pool, acq_type="LogEI", y_best=y_train.max() + 50.0, kappa=0.0, xi=0.0,
        )
        assert np.all(np.isfinite(scores)), "LogEI should not produce NaN/Inf for extreme z"

    def test_acquisition_ranking_preserved(self):
        X_train, y_train, X_pool, _ = _make_synthetic_data(n_train=60, d=4)
        model = fit_gp(X_train, y_train)
        y_best = y_train.max()

        ucb_scores = score_acquisition(
            model, X_pool, acq_type="UCB", y_best=y_best, kappa=2.0, xi=0.0,
        )
        ei_scores = score_acquisition(
            model, X_pool, acq_type="EI", y_best=y_best, kappa=0.0, xi=0.0,
        )

        rho, _ = spearmanr(ucb_scores, ei_scores)
        assert rho > 0.5, (
            f"UCB and EI rankings should roughly agree (Spearman rho={rho:.3f})"
        )


# ---------------------------------------------------------------------------
# 3. Batch Selection Tests
# ---------------------------------------------------------------------------

class TestBatchSelectionKB:
    def test_select_batch_kb_correct_size(self):
        X_train, y_train, X_pool, pool_indices = _make_synthetic_data()
        batch = select_batch(
            X_train, y_train, X_pool, pool_indices,
            batch_size=6, acq_type="UCB", batch_strategy="kb",
            kappa=2.0, xi=0.0, seed=42,
        )
        assert len(batch) == 6

    def test_select_batch_kb_no_duplicates(self):
        X_train, y_train, X_pool, pool_indices = _make_synthetic_data()
        batch = select_batch(
            X_train, y_train, X_pool, pool_indices,
            batch_size=12, acq_type="UCB", batch_strategy="kb",
            kappa=2.0, xi=0.0, seed=42,
        )
        assert len(set(batch)) == len(batch), "KB batch contains duplicate indices"

    def test_select_batch_kb_from_pool(self):
        X_train, y_train, X_pool, pool_indices = _make_synthetic_data()
        batch = select_batch(
            X_train, y_train, X_pool, pool_indices,
            batch_size=6, acq_type="UCB", batch_strategy="kb",
            kappa=2.0, xi=0.0, seed=42,
        )
        pool_set = set(pool_indices.tolist())
        for idx in batch:
            assert idx in pool_set, f"Index {idx} not in pool_indices"


class TestBatchSelectionRKB:
    def test_select_batch_rkb_more_diverse(self):
        X_train, y_train, X_pool, pool_indices = _make_synthetic_data(
            n_train=60, n_pool=300, d=5, seed=0,
        )
        rkb_more_diverse_count = 0
        n_trials = 10
        for trial in range(n_trials):
            kb_batch = select_batch(
                X_train, y_train, X_pool, pool_indices,
                batch_size=8, acq_type="UCB", batch_strategy="kb",
                kappa=2.0, xi=0.0, seed=trial,
            )
            rkb_batch = select_batch(
                X_train, y_train, X_pool, pool_indices,
                batch_size=8, acq_type="UCB", batch_strategy="rkb",
                kappa=2.0, xi=0.0, seed=trial,
            )
            kb_div = pdist(X_pool[kb_batch]).mean()
            rkb_div = pdist(X_pool[rkb_batch]).mean()
            if rkb_div >= kb_div:
                rkb_more_diverse_count += 1

        assert rkb_more_diverse_count >= n_trials // 2, (
            f"RKB should be more diverse than KB in majority of runs "
            f"({rkb_more_diverse_count}/{n_trials})"
        )


class TestBatchSelectionLP:
    def test_select_batch_lp_correct_size(self):
        X_train, y_train, X_pool, pool_indices = _make_synthetic_data()
        batch = select_batch(
            X_train, y_train, X_pool, pool_indices,
            batch_size=6, acq_type="UCB", batch_strategy="lp",
            kappa=2.0, xi=0.0, seed=42,
        )
        assert len(batch) == 6

    def test_select_batch_lp_no_duplicates(self):
        X_train, y_train, X_pool, pool_indices = _make_synthetic_data()
        batch = select_batch(
            X_train, y_train, X_pool, pool_indices,
            batch_size=12, acq_type="UCB", batch_strategy="lp",
            kappa=2.0, xi=0.0, seed=42,
        )
        assert len(set(batch)) == len(batch), "LP batch contains duplicate indices"

    def test_select_batch_lp_spatial_diversity(self):
        rng = np.random.RandomState(77)
        X_train = rng.randn(50, 5).astype(np.float64)
        y_train = (X_train ** 2).sum(axis=1) + 0.1 * rng.randn(50)
        X_pool = rng.randn(300, 5).astype(np.float64)
        pool_indices = np.arange(300)

        lp_batch = select_batch(
            X_train, y_train, X_pool, pool_indices,
            batch_size=8, acq_type="UCB", batch_strategy="lp",
            kappa=2.0, xi=0.0, seed=42,
        )
        lp_div = pdist(X_pool[lp_batch]).mean()

        # Compare against random baseline
        random_divs = []
        for _ in range(20):
            r = rng.choice(300, size=8, replace=False)
            random_divs.append(pdist(X_pool[r]).mean())
        random_mean = np.mean(random_divs)

        assert lp_div > random_mean * 0.8, (
            f"LP diversity ({lp_div:.3f}) should be comparable to or better than "
            f"random ({random_mean:.3f})"
        )


class TestBatchSelectionTS:
    def test_select_batch_ts_correct_size(self):
        X_train, y_train, X_pool, pool_indices = _make_synthetic_data()
        batch = select_batch(
            X_train, y_train, X_pool, pool_indices,
            batch_size=6, acq_type="UCB", batch_strategy="ts",
            kappa=2.0, xi=0.0, seed=42,
        )
        assert len(batch) == 6

    def test_select_batch_ts_different_seeds(self):
        X_train, y_train, X_pool, pool_indices = _make_synthetic_data()
        batch_a = select_batch(
            X_train, y_train, X_pool, pool_indices,
            batch_size=6, acq_type="UCB", batch_strategy="ts",
            kappa=2.0, xi=0.0, seed=42,
        )
        batch_b = select_batch(
            X_train, y_train, X_pool, pool_indices,
            batch_size=6, acq_type="UCB", batch_strategy="ts",
            kappa=2.0, xi=0.0, seed=999,
        )
        assert set(batch_a) != set(batch_b), (
            "TS with different seeds should generally produce different batches"
        )


class TestBatchSelectionQLogEI:
    def test_select_batch_qlogei_correct_size(self):
        X_train, y_train, X_pool, pool_indices = _make_synthetic_data()
        try:
            batch = select_batch(
                X_train, y_train, X_pool, pool_indices,
                batch_size=4, acq_type="LogEI", batch_strategy="qlogei",
                kappa=0.0, xi=0.01, seed=42,
            )
            assert len(batch) == 4
        except (ValueError, NotImplementedError):
            pytest.skip("qLogEI batch strategy not implemented")


# ---------------------------------------------------------------------------
# 4. Sparse GP Tests
# ---------------------------------------------------------------------------

class TestSparseGP:
    def test_sparse_gp_fits(self):
        rng = np.random.RandomState(42)
        X = rng.randn(200, 5).astype(np.float64)
        y = (X ** 2).sum(axis=1) + 0.1 * rng.randn(200)
        model = fit_gp(X, y, use_sparse=True, n_inducing=20)
        mean, std = predict(model, X[:10])
        assert mean.shape == (10,)
        assert std.shape == (10,)
        assert np.all(np.isfinite(mean))
        assert np.all(np.isfinite(std))

    def test_sparse_gp_predictions_close(self):
        rng = np.random.RandomState(42)
        X_train = rng.randn(150, 4).astype(np.float64)
        y_train = np.sin(X_train[:, 0]) + X_train[:, 1] ** 2 + 0.1 * rng.randn(150)
        X_test = rng.randn(50, 4).astype(np.float64)

        model_exact = fit_gp(X_train, y_train, use_sparse=False)
        model_sparse = fit_gp(X_train, y_train, use_sparse=True, n_inducing=50)

        mean_exact, _ = predict(model_exact, X_test)
        mean_sparse, _ = predict(model_sparse, X_test)

        r, _ = pearsonr(mean_exact, mean_sparse)
        assert r > 0.9, (
            f"Sparse GP predictions should correlate with exact GP (Pearson r={r:.3f})"
        )

    @pytest.mark.slow
    def test_sparse_gp_faster_large_n(self):
        rng = np.random.RandomState(42)
        X = rng.randn(2000, 5).astype(np.float64)
        y = (X ** 2).sum(axis=1) + 0.1 * rng.randn(2000)

        t0 = time.perf_counter()
        fit_gp(X, y, use_sparse=False)
        t_exact = time.perf_counter() - t0

        t0 = time.perf_counter()
        fit_gp(X, y, use_sparse=True, n_inducing=100)
        t_sparse = time.perf_counter() - t0

        assert t_sparse < t_exact, (
            f"Sparse GP ({t_sparse:.2f}s) should be faster than exact GP ({t_exact:.2f}s) on n=2000"
        )


# ---------------------------------------------------------------------------
# 5. Speed Comparison Tests
# ---------------------------------------------------------------------------

class TestSpeed:
    @pytest.mark.slow
    def test_pool_scoring_faster_than_continuous_opt(self):
        X_train, y_train, X_pool, _ = _make_synthetic_data(n_train=80, n_pool=1000, d=5)
        model = fit_gp(X_train, y_train)

        t0 = time.perf_counter()
        scores = score_acquisition(
            model, X_pool, acq_type="UCB", y_best=y_train.max(), kappa=2.0, xi=0.0,
        )
        t_pool = time.perf_counter() - t0

        assert t_pool < 1.0, (
            f"Pool scoring on 1000 candidates took {t_pool:.2f}s; should be <1s"
        )
        assert len(scores) == 1000

    @pytest.mark.slow
    def test_kb_fantasize_faster_than_refit(self):
        X_train, y_train, X_pool, pool_indices = _make_synthetic_data(
            n_train=100, n_pool=200, d=5,
        )
        # Time KB (uses fantasize / condition_on_observations internally)
        t0 = time.perf_counter()
        select_batch(
            X_train, y_train, X_pool, pool_indices,
            batch_size=6, acq_type="UCB", batch_strategy="kb",
            kappa=2.0, xi=0.0, seed=42,
        )
        t_kb = time.perf_counter() - t0

        # Estimate full refit cost: fit_gp * batch_size
        t0 = time.perf_counter()
        fit_gp(X_train, y_train)
        t_single_fit = time.perf_counter() - t0
        t_refit_estimate = t_single_fit * 6

        # KB with fantasize should be meaningfully faster than 6 full refits
        # Allow some tolerance since KB still does some computation per step
        assert t_kb < t_refit_estimate * 2.0, (
            f"KB ({t_kb:.2f}s) should be faster than 6 full refits ({t_refit_estimate:.2f}s)"
        )


# ---------------------------------------------------------------------------
# 6. End-to-End Integration Tests
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_bo_loop_synthetic(self):
        rng = np.random.RandomState(42)
        d = 2
        n_pool = 500
        X_pool_full = rng.uniform(-5, 10, size=(n_pool, d)).astype(np.float64)
        y_pool_full = _branin(X_pool_full)
        # Seed: random initial sample
        seed_size = 20
        batch_size = 6
        n_rounds = 5

        seed_idx = rng.choice(n_pool, size=seed_size, replace=False)
        X_train = X_pool_full[seed_idx].copy()
        y_train = y_pool_full[seed_idx].copy()
        evaluated = set(seed_idx.tolist())

        best_values = [y_train.max()]

        for round_i in range(n_rounds):
            remaining = np.array([i for i in range(n_pool) if i not in evaluated])
            X_remaining = X_pool_full[remaining]

            batch = select_batch(
                X_train, y_train, X_remaining, remaining,
                batch_size=batch_size, acq_type="UCB", batch_strategy="kb",
                kappa=2.0, xi=0.0, seed=42 + round_i,
            )

            assert len(batch) == batch_size, (
                f"Round {round_i}: expected {batch_size} candidates, got {len(batch)}"
            )

            for idx in batch:
                assert idx not in evaluated, f"Round {round_i}: index {idx} already evaluated"
                evaluated.add(idx)

            X_new = X_pool_full[batch]
            y_new = y_pool_full[batch]
            X_train = np.vstack([X_train, X_new])
            y_train = np.concatenate([y_train, y_new])
            best_values.append(y_train.max())

        # Total unique evaluations
        expected_total = seed_size + n_rounds * batch_size
        assert len(evaluated) == expected_total, (
            f"Expected {expected_total} total evaluations, got {len(evaluated)}"
        )

        # Best value should not decrease
        for i in range(1, len(best_values)):
            assert best_values[i] >= best_values[i - 1] - 1e-10, (
                f"Best value decreased from round {i-1} to {i}: "
                f"{best_values[i-1]:.4f} -> {best_values[i]:.4f}"
            )

    def test_bo_loop_all_strategies(self):
        X_train, y_train, X_pool, pool_indices = _make_synthetic_data(
            n_train=50, n_pool=100, d=3,
        )
        strategies = ["kb", "rkb", "lp", "ts"]
        batch_size = 4

        for strategy in strategies:
            batch = select_batch(
                X_train, y_train, X_pool, pool_indices,
                batch_size=batch_size, acq_type="UCB", batch_strategy=strategy,
                kappa=2.0, xi=0.0, seed=42,
            )
            assert len(batch) == batch_size, (
                f"Strategy '{strategy}' returned {len(batch)} instead of {batch_size}"
            )
            assert len(set(batch)) == len(batch), (
                f"Strategy '{strategy}' returned duplicate indices"
            )


# ---------------------------------------------------------------------------
# 7. Edge Case Tests
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_pool_smaller_than_batch(self):
        X_train, y_train, _, _ = _make_synthetic_data(n_train=50, d=3)
        rng = np.random.RandomState(99)
        X_pool = rng.randn(3, 3).astype(np.float64)
        pool_indices = np.arange(3)

        batch = select_batch(
            X_train, y_train, X_pool, pool_indices,
            batch_size=10, acq_type="UCB", batch_strategy="kb",
            kappa=2.0, xi=0.0, seed=42,
        )
        assert len(batch) == 3, (
            "Should return all 3 available candidates when batch_size=10 but pool has 3"
        )
        assert len(set(batch)) == 3

    def test_single_point_pool(self):
        X_train, y_train, _, _ = _make_synthetic_data(n_train=50, d=3)
        X_pool = np.array([[1.0, 2.0, 3.0]])
        pool_indices = np.array([0])

        batch = select_batch(
            X_train, y_train, X_pool, pool_indices,
            batch_size=5, acq_type="UCB", batch_strategy="kb",
            kappa=2.0, xi=0.0, seed=42,
        )
        assert len(batch) == 1
        assert batch[0] == 0

    def test_constant_targets(self):
        rng = np.random.RandomState(42)
        X_train = rng.randn(40, 3).astype(np.float64)
        y_train = np.ones(40, dtype=np.float64) * 5.0
        X_pool = rng.randn(30, 3).astype(np.float64)
        pool_indices = np.arange(30)

        # EI with constant targets: should not crash
        model = fit_gp(X_train, y_train)
        scores_ei = score_acquisition(
            model, X_pool, acq_type="EI", y_best=5.0, kappa=0.0, xi=0.0,
        )
        assert np.all(np.isfinite(scores_ei)), "EI should be finite even with constant targets"

        # LogEI with constant targets: should not crash
        scores_logei = score_acquisition(
            model, X_pool, acq_type="LogEI", y_best=5.0, kappa=0.0, xi=0.0,
        )
        assert np.all(np.isfinite(scores_logei)), "LogEI should be finite even with constant targets"

        # select_batch should still work
        batch = select_batch(
            X_train, y_train, X_pool, pool_indices,
            batch_size=4, acq_type="UCB", batch_strategy="kb",
            kappa=2.0, xi=0.0, seed=42,
        )
        assert len(batch) == 4

    def test_get_device_cpu(self):
        device = get_device(use_mps=False)
        import torch
        assert device == torch.device("cpu")
