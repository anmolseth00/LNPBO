"""Regression tests for the manual Laplace prior-precision objective."""

import numpy as np
import pytest
import torch

from LNPBO.models.laplace import ManualLastLayerLaplace
from LNPBO.models.surrogate_mlp import SurrogateMLP


def _make_manual_laplace(weight_scale: float) -> ManualLastLayerLaplace:
    laplace = ManualLastLayerLaplace(SurrogateMLP(1), hessian_structure="diag")
    laplace._diag_H = torch.tensor([1.0, 1.0, 1.0])
    laplace._Phi_train = torch.zeros(1, 2)
    laplace._w_map = torch.full((3,), weight_scale)
    return laplace


def test_manual_laplace_prior_precision_depends_on_map_weights() -> None:
    small_weights = _make_manual_laplace(0.0)
    large_weights = _make_manual_laplace(10.0)

    small_weights.optimize_prior_precision()
    large_weights.optimize_prior_precision()

    candidates = np.logspace(-3, 3, 20)

    def objective(prec: float, weight_scale: float) -> float:
        diag_h = np.array([1.0, 1.0, 1.0], dtype=float)
        w_map_sq = 3.0 * (weight_scale ** 2)
        return 0.5 * 3 * np.log(prec) - 0.5 * np.log(diag_h + prec).sum() - 0.5 * prec * w_map_sq

    expected_small = max(candidates, key=lambda prec: objective(prec, 0.0))
    expected_large = max(candidates, key=lambda prec: objective(prec, 10.0))

    assert small_weights.prior_precision == pytest.approx(expected_small)
    assert large_weights.prior_precision == pytest.approx(expected_large)
    assert large_weights.prior_precision < small_weights.prior_precision
