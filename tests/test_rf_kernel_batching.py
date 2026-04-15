"""Regression tests for batched RandomForestKernel behavior."""

import torch

from LNPBO.optimization.rf_kernel import RandomForestKernel


def test_rf_kernel_supports_batched_inputs() -> None:
    train_x = torch.randn(8, 3, dtype=torch.float64)
    train_y = torch.randn(8, 1, dtype=torch.float64)
    kernel = RandomForestKernel(train_x, train_y, n_estimators=10)

    x = torch.randn(2, 4, 3, dtype=torch.float64)
    K = kernel(x, x).to_dense()

    assert K.shape == (2, 4, 4)
    assert torch.isfinite(K).all()


def test_rf_kernel_batched_diag_shape() -> None:
    train_x = torch.randn(8, 3, dtype=torch.float64)
    train_y = torch.randn(8, 1, dtype=torch.float64)
    kernel = RandomForestKernel(train_x, train_y, n_estimators=10)

    x = torch.randn(2, 4, 3, dtype=torch.float64)
    diag = kernel(x, x, diag=True)

    assert diag.shape == (2, 4)
    torch.testing.assert_close(diag, torch.ones(2, 4, dtype=torch.float64))


def test_rf_kernel_batched_matches_individual_batches() -> None:
    train_x = torch.randn(8, 3, dtype=torch.float64)
    train_y = torch.randn(8, 1, dtype=torch.float64)
    kernel = RandomForestKernel(train_x, train_y, n_estimators=10)

    x = torch.randn(2, 4, 3, dtype=torch.float64)
    batched = kernel(x, x).to_dense()
    stacked = torch.stack([kernel(x[i], x[i]).to_dense() for i in range(x.shape[0])])

    torch.testing.assert_close(batched, stacked)
