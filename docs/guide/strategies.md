# Optimization Strategies

LNPBO provides a diverse set of optimization strategies spanning Gaussian Processes, tree-based ensembles, deep ensembles, and mixed-variable methods. Each strategy combines a surrogate model with an acquisition function and batch selection mechanism.

---

## Strategy Overview

| Family | `surrogate_type` | Uncertainty | Batch Strategies |
|--------|------------------|-------------|------------------|
| GP (BoTorch) | `gp` | Posterior variance | KB, RKB, LP, TS, qLogEI, GIBBON |
| GP (sklearn) | `gp_sklearn` | Posterior variance | KB, LP |
| XGBoost Greedy | `xgb` | None | Greedy (top-K) |
| XGBoost UCB | `xgb_ucb` | MAPIE conformal | Greedy, TS |
| XGBoost CQR | `xgb_cqr` | Conformalized quantile | Greedy, TS |
| Random Forest UCB | `rf_ucb` | Tree variance | Greedy, TS |
| Random Forest TS | `rf_ts` | Per-tree draws | Thompson Sampling |
| NGBoost | `ngboost` | Distributional | Greedy, TS |
| Deep Ensemble | `deep_ensemble` | Network disagreement | Greedy, TS |
| TabPFN | `tabpfn` | Zero-shot foundation model | Greedy |
| CASMOPolitan | `casmopolitan` | Mixed-variable GP | Internal KB |
| GP Mixed | `gp_mixed` | Mixed discrete-continuous GP | KB, TS |
| Robust GP | `robust_gp` | Relevance Pursuit | KB, TS |
| Multi-task GP | `multitask_gp` | ICM coregionalization | KB, TS |
| GP UCB (sklearn) | `gp_ucb` | Sklearn GP pool scoring | Greedy |
| Ridge | `ridge` | BayesianRidge | Greedy |
| SNGP | `sngp` | Distance-aware MLP + RFF | Greedy |
| Laplace | `laplace` | Post-hoc Laplace approx | Greedy |
| Bradley-Terry | `bradley_terry` | Pairwise preference | Greedy |
| GroupDRO | `groupdro` | Worst-group robust | Greedy |
| VREx | `vrex` | Variance Risk Extrapolation | Greedy |

---

## Gaussian Process (GP)

GP surrogates model the unknown objective function as a draw from a Gaussian process, providing both predictions and calibrated uncertainty estimates. LNPBO provides two GP backends.

### BoTorch/GPyTorch GP (`surrogate_type="gp"`)

The default and recommended GP backend. Uses BoTorch's `SingleTaskGP` with GPyTorch for GPU/MPS-accelerated kernel computations. Operates by direct discrete pool scoring -- candidates are scored directly without continuous relaxation and nearest-neighbor projection.

```python
optimizer = Optimizer(
    space=space,
    candidate_pool=encoded.df,
    surrogate_type="gp",
    acquisition_type="UCB",
    batch_strategy="kb",
    kappa=5.0,
    batch_size=24,
)
```

Key features:

- **Kernel options:** Matern 5/2 (default), Tanimoto (molecular fingerprints), Aitchison (compositional data), Deep Kernel Learning (DKL), Random Forest kernel, Compositional product kernel
- **LOVE-accelerated variance** for fast uncertainty estimation
- **Condition-on-observations** for efficient KB/RKB batch construction (rank-1 updates instead of refitting)
- **Optional SVGP** for large datasets (O(NM^2) instead of O(N^3))

### sklearn GP (`surrogate_type="gp_sklearn"`)

The legacy GP backend using scikit-learn's `GaussianProcessRegressor`. Uses continuous acquisition optimization with L-BFGS-B, requiring continuous relaxation of discrete variables.

```python
optimizer = Optimizer(
    space=space,
    surrogate_type="gp_sklearn",
    acquisition_type="UCB",
    kappa=5.0,
    batch_size=24,
)
```

!!! note
    The sklearn GP backend does not require a `candidate_pool` as it optimizes acquisition functions over the continuous parameter space directly. This is appropriate for ratio-only optimization where the search space is inherently continuous.

### Acquisition Functions

| Function | Parameter | Description |
|----------|-----------|-------------|
| `UCB` | `kappa` (default 5.0) | Upper Confidence Bound: mu + kappa * sigma. Higher kappa favors exploration. |
| `EI` | `xi` (default 0.01) | Expected Improvement over current best. |
| `LogEI` | `xi` (default 0.01) | Log Expected Improvement -- numerically stable variant of EI. |

### Batch Strategies (GP)

| Strategy | Key | Description |
|----------|-----|-------------|
| Kriging Believer | `kb` | Hallucinate selected points with posterior mean, recompute acquisition. Sequential and deterministic. |
| Randomized KB | `rkb` | Like KB but hallucinate with posterior samples instead of mean. Adds stochastic exploration to the batch. |
| Local Penalization | `lp` | Apply soft Gaussian exclusion zones around selected points to encourage diversity. |
| Thompson Sampling | `ts` | Draw independent posterior samples and select the argmax of each draw. Naturally diverse batches. |
| q-Log Noisy EI | `qlogei` | BoTorch native joint acquisition optimization. Jointly optimizes all batch members. |
| GIBBON | `gibbon` | Information-theoretic batch selection with DPP diversity. Based on Moss et al. (2021). |

---

## XGBoost

XGBoost surrogates train a gradient-boosted tree ensemble on observed data and score candidates via predicted mean or uncertainty-augmented acquisition.

### XGBoost Greedy (`surrogate_type="xgb"`)

Selects the top-K candidates by predicted mean. No uncertainty quantification -- pure exploitation.

```python
optimizer = Optimizer(
    space=space,
    candidate_pool=encoded.df,
    surrogate_type="xgb",
    batch_size=24,
)
```

### XGBoost UCB (`surrogate_type="xgb_ucb"`)

Combines XGBoost predictions with MAPIE conformal prediction intervals for UCB-style acquisition.

```python
optimizer = Optimizer(
    space=space,
    candidate_pool=encoded.df,
    surrogate_type="xgb_ucb",
    kappa=5.0,
    batch_size=24,
)
```

### XGBoost CQR (`surrogate_type="xgb_cqr"`)

XGBoost with Conformalized Quantile Regression for calibrated prediction intervals.

```python
optimizer = Optimizer(
    space=space,
    candidate_pool=encoded.df,
    surrogate_type="xgb_cqr",
    kappa=5.0,
    batch_size=24,
)
```

---

## Random Forest

Random Forest surrogates use tree-level predictions for natural uncertainty quantification.

### RF UCB (`surrogate_type="rf_ucb"`)

UCB acquisition using inter-tree variance as the uncertainty estimate.

```python
optimizer = Optimizer(
    space=space,
    candidate_pool=encoded.df,
    surrogate_type="rf_ucb",
    kappa=5.0,
    batch_size=24,
)
```

### RF Thompson Sampling (`surrogate_type="rf_ts"`)

Each batch member is selected by drawing a random tree's prediction as a Thompson sample.

```python
optimizer = Optimizer(
    space=space,
    candidate_pool=encoded.df,
    surrogate_type="rf_ts",
    batch_size=24,
)
```

---

## NGBoost

NGBoost is a gradient boosting framework that directly outputs distributional predictions (mean and variance), providing natural uncertainty quantification without conformal wrappers.

```python
# Requires: pip install "LNPBO[bench]"
optimizer = Optimizer(
    space=space,
    candidate_pool=encoded.df,
    surrogate_type="ngboost",
    kappa=5.0,
    batch_size=24,
)
```

NGBoost performed best overall in within-study benchmarks (mean top-5% recall = 0.752, 1.38x lift over random).

**Reference:** Duan, T. et al. "NGBoost: Natural Gradient Boosting for Probabilistic Prediction." *ICML 2020*.

---

## Deep Ensemble

An ensemble of 5 neural networks trained with different random initializations. Uncertainty is estimated from the disagreement (variance) across network predictions.

```python
optimizer = Optimizer(
    space=space,
    candidate_pool=encoded.df,
    surrogate_type="deep_ensemble",
    kappa=5.0,
    batch_size=24,
)
```

**Reference:** Lakshminarayanan, B., Pritzel, A., & Blundell, C. "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles." *NeurIPS 2017*.

---

## CASMOPolitan

Mixed-variable Bayesian optimization for search spaces with both categorical (IL identity) and continuous (molar ratios) dimensions. Uses a product kernel combining categorical overlap and Matern-ARD, with trust-region-based local search.

```python
optimizer = Optimizer(
    space=space,
    candidate_pool=encoded.df,
    surrogate_type="casmopolitan",
    acquisition_type="UCB",
    kappa=5.0,
    batch_size=24,
)
```

CASMOPolitan is particularly strong on ratio-only optimization tasks (1.74x lift in benchmarks) where it can leverage the inherent mixed structure of the search space.

**Reference:** Wan, X. et al. "Think Global and Act Local: Bayesian Optimisation over High-Dimensional Categorical and Mixed Search Spaces." *ICML 2021*.

---

## TabPFN

TabPFN is a zero-shot foundation model for tabular data that provides predictions without any training. It processes the entire training set as context and generates predictions in a single forward pass.

```python
# Requires: pip install "LNPBO[bench]"
optimizer = Optimizer(
    space=space,
    candidate_pool=encoded.df,
    surrogate_type="tabpfn",
    batch_size=24,
)
```

**Reference:** Hollmann, N. et al. "TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second." *ICLR 2023*.

---

## Random Baseline

For benchmarking purposes, a random baseline selects candidates uniformly at random from the pool. This is implemented in the benchmark runner rather than the Optimizer class:

```python
# In benchmark runner
from LNPBO.benchmarks.runner import STRATEGY_CONFIGS
# STRATEGY_CONFIGS["random"] = {"type": "random"}
```

---

## Benchmark Results Summary

Based on within-study benchmarks across 26 studies, 38 strategies, 5 seeds (top-5% recall, 15 rounds):

| Family | Mean Recall | Lift vs Random | Win % |
|--------|-------------|----------------|-------|
| NGBoost | 0.752 | 1.38x | 4% |
| Random Forest | 0.740 | 1.36x | 37% |
| CASMOPolitan | 0.731 | 1.34x | 15% |
| XGBoost | 0.731 | 1.34x | 11% |
| GP (sklearn) | 0.699 | 1.28x | 0% |
| Deep Ensemble | 0.698 | 1.28x | 11% |
| GP (BoTorch) | 0.658 | 1.21x | 19% |
| Random | 0.545 | --- | 0% |

All 37 non-random strategies are statistically significant vs random (p < 0.001). No single strategy wins all studies -- NGBoost wins 22%, TabPFN and CASMOPolitan each 19%, RF/Ridge/Deep Ensemble each 11%.

### By Study Type

- **Fixed-ratio IL screening (20 studies):** Tree models dominate (NGBoost 1.40x, GP-BO only 1.17x)
- **Variable-ratio IL screening (5 studies):** Gap narrows (RF 1.33x, GP-BO 1.22x)
- **Ratio-only optimization (2 studies):** CASMOPolitan leads (1.74x), GP-BO competitive (1.58x)
