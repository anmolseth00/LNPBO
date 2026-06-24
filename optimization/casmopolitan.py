"""CASMOPOLITAN mixed-variable Bayesian optimization for LNP formulations.

Implements the core ingredients of Wan et al. (2021) for mixed categorical +
continuous BO on a finite candidate pool. This module is the public surface;
the implementation lives in ``_casmopolitan_core`` and ``_casmopolitan_kernels``.
"""

from __future__ import annotations

from ._casmopolitan_core import (
    TrustRegion,
    _append_restart_observation,
    _fit_pool_casmopolitan_gp,
    _map_candidates_to_pool,
    _trust_region_pool_mask,
    _ucb_acquisition,
    optimize_mixed_acquisition,
    select_batch_casmopolitan,
    select_pool_batch_casmopolitan,
)
from ._casmopolitan_kernels import (
    AdditiveProductKernel,
    ExponentiatedCategoricalKernel,
    MixedCasmopolitanKernel,
)

__all__ = [
    "AdditiveProductKernel",
    "ExponentiatedCategoricalKernel",
    "MixedCasmopolitanKernel",
    "TrustRegion",
    "_append_restart_observation",
    "_fit_pool_casmopolitan_gp",
    "_map_candidates_to_pool",
    "_trust_region_pool_mask",
    "_ucb_acquisition",
    "optimize_mixed_acquisition",
    "select_batch_casmopolitan",
    "select_pool_batch_casmopolitan",
]
