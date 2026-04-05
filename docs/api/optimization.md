# Optimization API

The optimization module provides the core Bayesian optimization pipeline, including the high-level `Optimizer` class, the BoTorch/GPyTorch GP engine, custom kernels, and acquisition functions.

---

## Optimizer

The main entry point for LNP formulation optimization. Supports multiple surrogate models, acquisition functions, and batch strategies through a unified API.

::: LNPBO.optimization.optimizer.Optimizer
    options:
      members:
        - __init__
        - suggest

---

## GP Bayesian Optimization (BoTorch)

The `gp_bo` module implements the BoTorch/GPyTorch GP pipeline with direct discrete pool scoring. This replaces the older sklearn GP + continuous optimization + nearest-neighbor pipeline.

### fit_gp

::: LNPBO.optimization.gp_bo.fit_gp

### predict

::: LNPBO.optimization.gp_bo.predict

### score_acquisition

::: LNPBO.optimization.gp_bo.score_acquisition

### select_batch

::: LNPBO.optimization.gp_bo.select_batch

---

## Kernels

Custom GP kernels for molecular fingerprint and compositional spaces.

### TanimotoKernel

::: LNPBO.optimization.kernels.TanimotoKernel
    options:
      members:
        - forward

### AitchisonKernel

::: LNPBO.optimization.kernels.AitchisonKernel
    options:
      members:
        - forward

### CompositionalProductKernel

::: LNPBO.optimization.kernels.CompositionalProductKernel

---

## CASMOPolitan

Mixed-variable Bayesian optimization for LNP formulations with categorical (IL identity) and continuous (molar ratios) dimensions. Implements trust-region-based BO following the CASMOPOLITAN framework.

::: LNPBO.optimization.casmopolitan.score_pool_casmopolitan

---

## Constants

::: LNPBO.optimization.optimizer.SURROGATE_TYPES

::: LNPBO.optimization.optimizer.ACQUISITION_TYPES

::: LNPBO.optimization.optimizer.BATCH_STRATEGIES

::: LNPBO.optimization.optimizer.DISCRETE_BATCH_STRATEGIES
