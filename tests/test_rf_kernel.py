"""Smoke tests for the Random Forest proximity kernel GP.

Verifies:
1. RF kernel produces a valid PSD kernel matrix
2. GP with RF kernel fits and produces posterior mean/variance
3. Thompson Sampling batch acquisition returns valid candidates
4. Kernel diagonal is always 1.0 (self-proximity)
5. Kernel values are in [0, 1]
"""

import numpy as np
import pytest
import torch

from LNPBO.optimization.gp_bo import fit_gp, predict, select_batch
from LNPBO.optimization.rf_kernel import RandomForestKernel


@pytest.fixture
def synthetic_data():
    rng = np.random.RandomState(42)
    n_train, n_pool, d = 50, 200, 5
    X_train = rng.randn(n_train, d)
    y_train = np.sin(X_train[:, 0]) + 0.5 * X_train[:, 1] ** 2 + 0.1 * rng.randn(n_train)
    X_pool = rng.randn(n_pool, d)
    return X_train, y_train, X_pool


class TestRandomForestKernel:
    def test_kernel_matrix_shape(self, synthetic_data):
        X_train, y_train, _X_pool = synthetic_data
        X_t = torch.tensor(X_train, dtype=torch.float64)
        Y_t = torch.tensor(y_train, dtype=torch.float64)
        kernel = RandomForestKernel(X_t, Y_t, n_estimators=50)

        K = kernel(X_t, X_t).evaluate()
        assert K.shape == (50, 50)

    def test_kernel_psd(self, synthetic_data):
        X_train, y_train, _ = synthetic_data
        X_t = torch.tensor(X_train, dtype=torch.float64)
        Y_t = torch.tensor(y_train, dtype=torch.float64)
        kernel = RandomForestKernel(X_t, Y_t, n_estimators=100)

        K = kernel(X_t, X_t).evaluate()
        eigvals = torch.linalg.eigvalsh(K)
        assert (eigvals >= -1e-6).all(), f"Kernel not PSD: min eigenvalue = {eigvals.min():.6e}"

    def test_kernel_diagonal_is_one(self, synthetic_data):
        X_train, y_train, _ = synthetic_data
        X_t = torch.tensor(X_train, dtype=torch.float64)
        Y_t = torch.tensor(y_train, dtype=torch.float64)
        kernel = RandomForestKernel(X_t, Y_t, n_estimators=50)

        diag = kernel(X_t, X_t, diag=True)
        assert torch.allclose(diag, torch.ones_like(diag))

    def test_kernel_values_in_unit_interval(self, synthetic_data):
        X_train, y_train, X_pool = synthetic_data
        X_t = torch.tensor(X_train, dtype=torch.float64)
        Y_t = torch.tensor(y_train, dtype=torch.float64)
        X_p = torch.tensor(X_pool[:20], dtype=torch.float64)
        kernel = RandomForestKernel(X_t, Y_t, n_estimators=50)

        K = kernel(X_t, X_p).evaluate()
        assert (K >= 0).all()
        assert (K <= 1).all()

    def test_kernel_symmetry(self, synthetic_data):
        X_train, y_train, _ = synthetic_data
        X_t = torch.tensor(X_train, dtype=torch.float64)
        Y_t = torch.tensor(y_train, dtype=torch.float64)
        kernel = RandomForestKernel(X_t, Y_t, n_estimators=50)

        K = kernel(X_t, X_t).evaluate()
        assert torch.allclose(K, K.T, atol=1e-10)


class TestRFKernelGP:
    def test_fit_gp_rf_kernel(self, synthetic_data):
        X_train, y_train, X_pool = synthetic_data
        model = fit_gp(X_train, y_train, kernel_type="rf")
        mean, std = predict(model, X_pool)

        assert mean.shape == (200,)
        assert std.shape == (200,)
        assert np.all(np.isfinite(mean))
        assert np.all(std > 0)

    def test_ts_batch_acquisition(self, synthetic_data):
        X_train, y_train, X_pool = synthetic_data
        pool_indices = np.arange(len(X_pool))
        batch_size = 5

        selected = select_batch(
            X_train,
            y_train,
            X_pool,
            pool_indices,
            batch_size=batch_size,
            acq_type="UCB",
            batch_strategy="ts",
            kernel_type="rf",
            seed=42,
        )

        assert len(selected) == batch_size
        assert len(set(selected)) == batch_size
        assert all(0 <= idx < len(X_pool) for idx in selected)

    def test_kb_logei_batch_acquisition(self, synthetic_data):
        X_train, y_train, X_pool = synthetic_data
        pool_indices = np.arange(len(X_pool))
        batch_size = 5

        selected = select_batch(
            X_train,
            y_train,
            X_pool,
            pool_indices,
            batch_size=batch_size,
            acq_type="LogEI",
            batch_strategy="kb",
            kernel_type="rf",
            seed=42,
        )

        assert len(selected) == batch_size
        assert len(set(selected)) == batch_size
        assert all(0 <= idx < len(X_pool) for idx in selected)

    def test_posterior_variance_nonzero_for_unseen(self, synthetic_data):
        X_train, y_train, X_pool = synthetic_data
        model = fit_gp(X_train, y_train, kernel_type="rf")
        _, std = predict(model, X_pool)
        assert np.mean(std) > 0.01, "Posterior std should be meaningfully nonzero for unseen points"

    def test_reproducibility(self, synthetic_data):
        X_train, y_train, X_pool = synthetic_data
        pool_indices = np.arange(len(X_pool))

        sel1 = select_batch(
            X_train,
            y_train,
            X_pool,
            pool_indices,
            batch_size=5,
            acq_type="UCB",
            batch_strategy="ts",
            kernel_type="rf",
            seed=123,
        )
        sel2 = select_batch(
            X_train,
            y_train,
            X_pool,
            pool_indices,
            batch_size=5,
            acq_type="UCB",
            batch_strategy="ts",
            kernel_type="rf",
            seed=123,
        )
        assert sel1 == sel2
