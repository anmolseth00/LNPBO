"""Shared test helpers and fixtures."""

import numpy as np
import pytest


@pytest.fixture()
def synthetic_gp_data():
    """Standard synthetic GP test data: (X_train, y_train, X_pool, pool_indices)."""
    rng = np.random.RandomState(42)
    d = 5
    X_train = rng.randn(50, d).astype(np.float64)
    y_train = (X_train**2).sum(axis=1) + 0.1 * rng.randn(50)
    X_pool = rng.randn(200, d).astype(np.float64)
    pool_indices = np.arange(200)
    return X_train, y_train.astype(np.float64), X_pool, pool_indices


def assert_psd(K, name="kernel", tol=1e-6):
    """Assert a kernel matrix is positive semi-definite."""
    import torch

    eigs = torch.linalg.eigvalsh(K)
    assert (eigs >= -tol).all(), f"{name} not PSD: min eigenvalue = {eigs.min():.6e}"


def assert_batch_from_pool(batch, pool_indices):
    """Assert all batch indices come from the pool."""
    pool_set = set(int(i) for i in pool_indices)
    for idx in batch:
        assert idx in pool_set, f"Index {idx} not in pool_indices"
