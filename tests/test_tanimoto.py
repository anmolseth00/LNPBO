"""Tests for the Tanimoto kernel and its integration with the GP BO pipeline.

Covers:
  1. TanimotoKernel unit tests (symmetry, PSD, Jaccard equivalence, edge cases)
  2. Integration tests with fit_gp / select_batch using kernel_type="tanimoto"

Run: .venv/bin/python -m pytest tests/test_tanimoto.py -v
"""

import numpy as np
import pytest
import torch

from LNPBO.optimization.kernels import TanimotoKernel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_binary_fp_data(n_train=30, n_pool=50, d=100, sparsity=0.7, seed=42):
    """Generate sparse binary fingerprint-like data with some structure."""
    rng = np.random.RandomState(seed)
    X_train = (rng.random((n_train, d)) > sparsity).astype(np.float64)
    y_train = X_train[:, :5].sum(axis=1) + rng.normal(0, 0.1, n_train)
    X_pool = (rng.random((n_pool, d)) > sparsity).astype(np.float64)
    pool_indices = np.arange(n_pool)
    return X_train, y_train.astype(np.float64), X_pool, pool_indices


# ---------------------------------------------------------------------------
# 1. TanimotoKernel Unit Tests
# ---------------------------------------------------------------------------

class TestTanimotoKernel:
    def test_self_similarity_is_one(self):
        """k(x, x) should be 1.0 for any non-zero vector."""
        kernel = TanimotoKernel()
        rng = np.random.RandomState(42)
        X = torch.tensor(rng.random((10, 50)), dtype=torch.float64)

        K = kernel(X, X).evaluate()
        diag = K.diag()
        torch.testing.assert_close(
            diag, torch.ones(10, dtype=torch.float64), atol=1e-6, rtol=0,
        )

    def test_orthogonal_vectors_zero(self):
        """k(x, y) = 0 when x and y have disjoint support (no overlap)."""
        kernel = TanimotoKernel()
        x = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
        y = torch.tensor([[0.0, 1.0, 0.0, 1.0]])

        k_val = kernel(x, y).evaluate().item()
        assert abs(k_val) < 1e-6, f"Expected ~0 for orthogonal vectors, got {k_val}"

    def test_identical_vectors_one(self):
        """k(x, y) = 1 when x == y (non-zero)."""
        kernel = TanimotoKernel()
        x = torch.tensor([[1.0, 0.0, 1.0, 1.0, 0.0]])

        k_val = kernel(x, x).evaluate().item()
        assert abs(k_val - 1.0) < 1e-6, f"Expected 1.0 for identical vectors, got {k_val}"

    def test_binary_matches_jaccard(self):
        """On binary {0,1} vectors, Tanimoto should equal |intersection|/|union|."""
        kernel = TanimotoKernel()
        x = torch.tensor([[1, 0, 1, 1, 0, 1, 0, 0]], dtype=torch.float64)
        y = torch.tensor([[1, 1, 0, 1, 0, 0, 1, 0]], dtype=torch.float64)

        # Manual Jaccard: intersection = {0,3} -> 2, union = {0,1,2,3,5,6} -> 6
        intersection = (x * y).sum().item()
        union = ((x + y) > 0).float().sum().item()
        expected = intersection / union

        k_val = kernel(x, y).evaluate().item()
        assert abs(k_val - expected) < 1e-6, (
            f"Expected Jaccard={expected:.4f}, got Tanimoto={k_val:.4f}"
        )

    def test_symmetric(self):
        """k(x, y) == k(y, x)."""
        kernel = TanimotoKernel()
        rng = np.random.RandomState(7)
        X = torch.tensor(rng.random((8, 20)), dtype=torch.float64)
        Y = torch.tensor(rng.random((5, 20)), dtype=torch.float64)

        K_xy = kernel(X, Y).evaluate()
        K_yx = kernel(Y, X).evaluate()
        torch.testing.assert_close(K_xy, K_yx.T, atol=1e-10, rtol=0)

    def test_positive_definite(self):
        """The kernel matrix should be positive semi-definite (eigenvalues >= 0)."""
        kernel = TanimotoKernel()
        rng = np.random.RandomState(123)
        X = torch.tensor(rng.random((20, 30)), dtype=torch.float64)

        K = kernel(X, X).evaluate()
        eigenvalues = torch.linalg.eigvalsh(K)
        assert (eigenvalues >= -1e-6).all(), (
            f"Kernel matrix not PSD; min eigenvalue = {eigenvalues.min().item():.6e}"
        )

    def test_batch_dimensions(self):
        """Should handle batch dims correctly (important for BoTorch fantasization)."""
        kernel = TanimotoKernel()
        # (batch=3, n=5, d=10)
        X1 = torch.rand(3, 5, 10, dtype=torch.float64)
        X2 = torch.rand(3, 4, 10, dtype=torch.float64)

        K = kernel(X1, X2).evaluate()
        assert K.shape == (3, 5, 4), f"Expected shape (3, 5, 4), got {K.shape}"

    def test_diag_mode(self):
        """diag=True should return vector of self-similarities (all ones for non-zero)."""
        kernel = TanimotoKernel()
        rng = np.random.RandomState(0)
        X = torch.tensor(rng.random((15, 25)), dtype=torch.float64)

        diag = kernel(X, X, diag=True)
        # diag mode is a lazy tensor; evaluate or compare
        if hasattr(diag, "evaluate"):
            diag = diag.evaluate()
        torch.testing.assert_close(
            diag, torch.ones(15, dtype=torch.float64), atol=1e-6, rtol=0,
        )

    def test_diag_mode_batch(self):
        """diag=True with batch dimensions."""
        kernel = TanimotoKernel()
        X = torch.rand(2, 7, 10, dtype=torch.float64)

        diag = kernel(X, X, diag=True)
        if hasattr(diag, "evaluate"):
            diag = diag.evaluate()
        assert diag.shape == (2, 7), f"Expected shape (2, 7), got {diag.shape}"
        torch.testing.assert_close(
            diag, torch.ones(2, 7, dtype=torch.float64), atol=1e-6, rtol=0,
        )

    def test_count_vectors(self):
        """Should work on non-negative count vectors (not just binary)."""
        kernel = TanimotoKernel()
        x = torch.tensor([[3.0, 0.0, 2.0, 1.0]])
        y = torch.tensor([[1.0, 2.0, 0.0, 1.0]])

        # Manual: dot=3*1+0*2+2*0+1*1=4, ||x||^2=14, ||y||^2=6, denom=14+6-4=16
        expected = 4.0 / 16.0
        k_val = kernel(x, y).evaluate().item()
        assert abs(k_val - expected) < 1e-6, (
            f"Expected {expected:.4f} for count vectors, got {k_val:.4f}"
        )

    def test_zero_vector_handled(self):
        """Zero vectors should not cause NaN/inf (clamped denominator)."""
        kernel = TanimotoKernel()
        x = torch.tensor([[0.0, 0.0, 0.0]])
        y = torch.tensor([[1.0, 0.0, 1.0]])

        k_val = kernel(x, y).evaluate()
        assert torch.isfinite(k_val).all(), f"Got non-finite value for zero vector: {k_val}"
        assert k_val.item() >= 0, f"Expected non-negative, got {k_val.item()}"

    def test_zero_zero_handled(self):
        """k(0, 0) should not produce NaN/inf."""
        kernel = TanimotoKernel()
        x = torch.tensor([[0.0, 0.0, 0.0]])

        k_val = kernel(x, x).evaluate()
        assert torch.isfinite(k_val).all(), f"Got non-finite for k(0,0): {k_val}"

    def test_values_in_unit_interval(self):
        """All kernel values should be in [0, 1] for non-negative inputs."""
        kernel = TanimotoKernel()
        rng = np.random.RandomState(99)
        X = torch.tensor(np.abs(rng.randn(20, 15)), dtype=torch.float64)

        K = kernel(X, X).evaluate()
        assert (K >= -1e-6).all(), f"Found negative kernel value: {K.min().item()}"
        assert (K <= 1.0 + 1e-6).all(), f"Found kernel value > 1: {K.max().item()}"

    def test_is_not_stationary(self):
        assert TanimotoKernel.is_stationary is False

    def test_has_no_lengthscale(self):
        assert TanimotoKernel.has_lengthscale is False


# ---------------------------------------------------------------------------
# 2. Integration Tests with GP Pipeline
# ---------------------------------------------------------------------------

class TestTanimotoGP:
    def test_fit_gp_tanimoto(self):
        """fit_gp() with kernel_type='tanimoto' should return a working model."""
        from LNPBO.optimization.gp_bo import fit_gp, predict

        X_train, y_train, X_pool, _ = _make_binary_fp_data()
        model = fit_gp(X_train, y_train, kernel_type="tanimoto")

        mean, std = predict(model, X_pool)
        assert mean.shape == (len(X_pool),)
        assert std.shape == (len(X_pool),)
        assert np.all(np.isfinite(mean)), "Predictions contain NaN/inf"
        assert np.all(np.isfinite(std)), "Uncertainties contain NaN/inf"
        assert np.all(std >= 0), "Negative uncertainty"

    def test_tanimoto_predictions_reasonable(self):
        """Model predictions should be finite and track the target structure."""
        from LNPBO.optimization.gp_bo import fit_gp, predict
        from scipy.stats import pearsonr

        rng = np.random.RandomState(42)
        d = 50
        X_train = (rng.random((60, d)) > 0.7).astype(np.float64)
        y_train = X_train[:, :5].sum(axis=1) + rng.normal(0, 0.1, 60)
        X_test = (rng.random((40, d)) > 0.7).astype(np.float64)
        y_true = X_test[:, :5].sum(axis=1)

        model = fit_gp(X_train, y_train, kernel_type="tanimoto")
        mean, std = predict(model, X_test)

        assert np.all(np.isfinite(mean))
        assert np.all(np.isfinite(std))

        r, _ = pearsonr(mean, y_true)
        assert r > 0.3, (
            f"Pearson r={r:.3f} too low; Tanimoto GP should capture "
            "at least some structure in fingerprint-based targets"
        )

    def test_select_batch_tanimoto_ts(self):
        """select_batch() with kernel_type='tanimoto' and batch_strategy='ts'."""
        from LNPBO.optimization.gp_bo import select_batch

        X_train, y_train, X_pool, pool_indices = _make_binary_fp_data()
        batch = select_batch(
            X_train, y_train, X_pool, pool_indices,
            batch_size=6, acq_type="UCB", batch_strategy="ts",
            kappa=2.0, xi=0.0, seed=42,
            kernel_type="tanimoto",
        )
        assert len(batch) == 6
        assert len(set(batch)) == 6, "Batch contains duplicates"
        pool_set = set(pool_indices.tolist())
        for idx in batch:
            assert idx in pool_set, f"Index {idx} not in pool"

    def test_select_batch_tanimoto_kb(self):
        """select_batch() with kernel_type='tanimoto' and batch_strategy='kb'."""
        from LNPBO.optimization.gp_bo import select_batch

        X_train, y_train, X_pool, pool_indices = _make_binary_fp_data()
        batch = select_batch(
            X_train, y_train, X_pool, pool_indices,
            batch_size=6, acq_type="UCB", batch_strategy="kb",
            kappa=2.0, xi=0.0, seed=42,
            kernel_type="tanimoto",
        )
        assert len(batch) == 6
        assert len(set(batch)) == 6, "Batch contains duplicates"
        pool_set = set(pool_indices.tolist())
        for idx in batch:
            assert idx in pool_set, f"Index {idx} not in pool"

    def test_tanimoto_uncertainty_higher_far(self):
        """Uncertainty should be higher for fingerprints far from training data."""
        from LNPBO.optimization.gp_bo import fit_gp, predict

        rng = np.random.RandomState(0)
        d = 80
        # Training: first 40 bits active, last 40 off
        X_train = np.zeros((40, d), dtype=np.float64)
        X_train[:, :40] = (rng.random((40, 40)) > 0.5).astype(np.float64)
        y_train = X_train[:, :5].sum(axis=1) + rng.normal(0, 0.05, 40)

        # Near: similar pattern (first 40 bits active)
        X_near = np.zeros((20, d), dtype=np.float64)
        X_near[:, :40] = (rng.random((20, 40)) > 0.5).astype(np.float64)

        # Far: opposite pattern (last 40 bits active, first 40 off)
        X_far = np.zeros((20, d), dtype=np.float64)
        X_far[:, 40:] = (rng.random((20, 40)) > 0.5).astype(np.float64)

        model = fit_gp(X_train, y_train, kernel_type="tanimoto")
        _, std_near = predict(model, X_near)
        _, std_far = predict(model, X_far)

        assert std_far.mean() > std_near.mean(), (
            f"Far-point std ({std_far.mean():.4f}) should exceed "
            f"near-point std ({std_near.mean():.4f})"
        )

    def test_tanimoto_acquisition_scores_finite(self):
        """Acquisition function scores should be finite with Tanimoto kernel."""
        from LNPBO.optimization.gp_bo import fit_gp, score_acquisition

        X_train, y_train, X_pool, _ = _make_binary_fp_data()
        model = fit_gp(X_train, y_train, kernel_type="tanimoto")

        for acq_type in ["UCB", "EI", "LogEI"]:
            scores = score_acquisition(
                model, X_pool, acq_type=acq_type,
                y_best=y_train.max(), kappa=2.0, xi=0.01,
            )
            assert np.all(np.isfinite(scores)), (
                f"{acq_type} scores contain NaN/inf with Tanimoto kernel"
            )

    def test_tanimoto_lp_batch(self):
        """LP batch strategy should work with Tanimoto kernel."""
        from LNPBO.optimization.gp_bo import select_batch

        X_train, y_train, X_pool, pool_indices = _make_binary_fp_data()
        batch = select_batch(
            X_train, y_train, X_pool, pool_indices,
            batch_size=4, acq_type="UCB", batch_strategy="lp",
            kappa=2.0, xi=0.0, seed=42,
            kernel_type="tanimoto",
        )
        assert len(batch) == 4
        assert len(set(batch)) == 4

    def test_tanimoto_bo_loop(self):
        """End-to-end BO loop with Tanimoto kernel on binary fingerprints."""
        from LNPBO.optimization.gp_bo import select_batch

        rng = np.random.RandomState(42)
        d = 80
        n_pool = 200

        X_all = (rng.random((n_pool, d)) > 0.7).astype(np.float64)
        y_all = X_all[:, :5].sum(axis=1) + rng.normal(0, 0.1, n_pool)

        seed_size = 15
        batch_size = 4
        n_rounds = 3

        seed_idx = rng.choice(n_pool, size=seed_size, replace=False)
        X_train = X_all[seed_idx].copy()
        y_train = y_all[seed_idx].copy()
        evaluated = set(seed_idx.tolist())

        for round_i in range(n_rounds):
            remaining = np.array([i for i in range(n_pool) if i not in evaluated])
            X_remaining = X_all[remaining]

            batch = select_batch(
                X_train, y_train, X_remaining, remaining,
                batch_size=batch_size, acq_type="UCB", batch_strategy="kb",
                kappa=2.0, xi=0.0, seed=42 + round_i,
                kernel_type="tanimoto",
            )

            assert len(batch) == batch_size, (
                f"Round {round_i}: expected {batch_size}, got {len(batch)}"
            )
            for idx in batch:
                assert idx not in evaluated, f"Round {round_i}: {idx} already evaluated"
                evaluated.add(idx)

            X_train = np.vstack([X_train, X_all[batch]])
            y_train = np.concatenate([y_train, y_all[batch]])

        expected_total = seed_size + n_rounds * batch_size
        assert len(evaluated) == expected_total
