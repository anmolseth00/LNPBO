# Surrogate Model Reference

The `Optimizer` supports 20+ surrogate models through a unified API. This page
documents every available `surrogate_type`, organized by family.

## Quick reference

| `surrogate_type` | Family | UQ method | Needs `study_id` | Key strength |
|---|---|---|---|---|
| `"gp"` | GP | Posterior | No | Gold standard; exact posterior with kernel flexibility |
| `"gp_mixed"` | GP | Posterior | No | Joint discrete IL + continuous ratio optimization |
| `"robust_gp"` | GP | Posterior | No | Automatic outlier detection and downweighting |
| `"multitask_gp"` | GP | Posterior | Yes | Cross-study transfer via learned task covariance |
| `"casmopolitan"` | GP | Posterior | No | Mixed-variable trust regions (categorical + continuous) |
| `"gp_sklearn"` | GP | Posterior | No | Continuous acquisition optimization (no pool needed) |
| `"xgb"` | Tree | None | No | Fast greedy baseline |
| `"xgb_ucb"` | Tree | Conformal | No | Conformal prediction intervals for exploration |
| `"rf_ucb"` | Tree | Tree variance | No | Per-tree disagreement as uncertainty |
| `"rf_ts"` | Tree | Tree variance | No | Thompson sampling with per-tree draws |
| `"ngboost"` | Tree | Distributional | No | Native probabilistic predictions |
| `"xgb_cqr"` | Tree | CQR | No | Adaptive-width conformal intervals |
| `"ridge"` | Linear | Bayesian | No | Calibrated posterior; fast and interpretable |
| `"deep_ensemble"` | Neural | Ensemble | No | Simple, scalable UQ via disagreement |
| `"sngp"` | Neural | Laplace/RFF | No | Distance-aware; high uncertainty far from data |
| `"laplace"` | Neural | Laplace | No | Post-hoc UQ on trained MLP; no retraining |
| `"tabpfn"` | Neural | Foundation | No | Zero-shot; no training needed |
| `"gp_ucb"` | GP (sklearn) | Posterior | No | Lightweight GP UCB on discrete pool |
| `"bradley_terry"` | Preference | None | No | Learns ranking from pairwise comparisons |
| `"groupdro"` | Robust | None | Yes | Worst-group robustness across studies |
| `"vrex"` | Robust | None | Yes | Penalizes cross-study loss variance |


## GP-based surrogates

### `"gp"` — Gaussian Process (default)

The default and recommended surrogate. Fits a BoTorch `SingleTaskGP` (or
`SingleTaskVariationalGP` for N > 1000) and selects batches via acquisition
function optimization on a discrete candidate pool.

**Kernel types** (set via `kernel_type=`):

- `"matern"` (default) — Matern 5/2 with ARD lengthscales.
- `"tanimoto"` — Tanimoto kernel for molecular fingerprints. Operates in
  the full fingerprint space without dimensionality reduction.
  *Ralaivola et al. (2005); Tripp et al. (2023, NeurIPS).*
- `"aitchison"` — Aitchison kernel for compositional (simplex) data.
- `"dkl"` — Deep Kernel Learning. MLP feature extractor + GP.
  *Wilson et al. (2016, AISTATS). Caveats: Ober & Rasmussen (2021, UAI).*
- `"rf"` — Random Forest proximity kernel. Data-adaptive, PSD.
  *Scornet (2016, IEEE Trans. Info. Theory).*
- `"compositional"` — Product kernel: Tanimoto on fingerprints x Aitchison
  on ratios x Matern on synthesis parameters.
- `"robust"` — Equivalent to `surrogate_type="robust_gp"`.

```python
# Tanimoto kernel on Morgan fingerprints:
opt = Optimizer(space=space, kernel_type="tanimoto", candidate_pool=pool)

# Deep Kernel Learning:
opt = Optimizer(space=space, kernel_type="dkl", candidate_pool=pool)
```

### `"robust_gp"` — Robust GP via Relevance Pursuit

Uses BoTorch's `RobustRelevancePursuitSingleTaskGP` which automatically
identifies and downweights unreliable observations via Bayesian model
selection. No manual outlier removal needed.

*Ament, S. et al. (2024). "Robust Gaussian Processes via Relevance Pursuit."
arXiv:2410.24222.*

```python
opt = Optimizer(space=space, surrogate_type="robust_gp", candidate_pool=pool)
```

### `"multitask_gp"` — Multi-Task GP with ICM

Models between-study correlations via a low-rank Intrinsic Coregionalization
Model (ICM). Learns which studies are informative for predicting others.
Requires `study_id` column in the dataset.

*Bonilla, E.V., Chai, K.M.A., & Williams, C.K.I. (2007). "Multi-task
Gaussian Process Prediction." NIPS 2007.*

```python
opt = Optimizer(space=space, surrogate_type="multitask_gp", candidate_pool=pool)
```

### `"casmopolitan"` — Mixed-Variable GP

CASMOPolitan: trust-region BO for mixed categorical + continuous spaces.
Treats IL identity as categorical and molar ratios as continuous.

*Wan, X. et al. (2021). "Think Global and Act Local: Bayesian Optimisation
over High-Dimensional Categorical and Mixed Search Spaces." ICML.*

### `"gp_mixed"` — Mixed Discrete-Continuous Optimization

Enumerates unique IL fingerprint configurations from the pool, fixes them
as discrete choices, and optimizes molar ratios continuously using BoTorch's
`optimize_acqf_mixed`. Best with `kernel_type="compositional"`.


## Tree/ensemble surrogates

All tree-based surrogates operate on a discrete candidate pool. They fit
quickly and scale well to large pools. Set `batch_strategy="ts"` for
Thompson sampling batches; default is greedy top-K.

### `"xgb"` — XGBoost greedy

Predicted mean only (no exploration). Fast baseline.

### `"xgb_ucb"` — XGBoost + Conformal UCB

Uses MAPIE's `CrossConformalRegressor` (CV+ method) for uncertainty
quantification. Score = mean + kappa * conformal_half_width.

*Barber, R.F. et al. (2021). "Predictive Inference with the Jackknife+."
Ann. Statist. 49(1).*

### `"rf_ucb"` / `"rf_ts"` — Random Forest

`rf_ucb`: mean + kappa * per-tree std. `rf_ts`: per-tree Thompson sampling
(Kandasamy et al., AISTATS 2018).

### `"ngboost"` — Natural Gradient Boosting

Native distributional predictions via the NGBoost library.

*Duan, T. et al. (2020). "NGBoost: Natural Gradient Boosting for
Probabilistic Prediction." ICML.*

### `"xgb_cqr"` — Conformalized Quantile Regression

Adaptive-width prediction intervals via CQR.

*Romano, Y. et al. (2019). "Conformalized Quantile Regression." NeurIPS.*


## Neural UQ surrogates

### `"deep_ensemble"` — Deep Ensemble

Trains 5 independent MLPs with different initializations. Uncertainty from
ensemble disagreement (mean +/- kappa * std across members).

*Lakshminarayanan, B. et al. (2017). "Simple and Scalable Predictive
Uncertainty Estimation using Deep Ensembles." NeurIPS.*

### `"sngp"` — Spectral-Normalized Neural GP

MLP backbone with spectral normalization (enforces Lipschitz constraint for
distance preservation) + Random Fourier Feature output layer + Laplace
approximation. Provides calibrated uncertainty that grows with distance from
training data.

*Liu, J.Z. et al. (2023). "A Simple Approach to Improve Single-Model Deep
Uncertainty via Distance-Awareness." JMLR 24(42).*

```python
opt = Optimizer(
    space=space, surrogate_type="sngp", candidate_pool=pool,
    surrogate_kwargs={"epochs": 200, "n_random_features": 2048},
)
```

### `"laplace"` — MLP + Laplace Approximation

Trains a standard MLP, then computes a post-hoc Laplace approximation over
last-layer weights. Uses `laplace-torch` if available, otherwise a built-in
KFAC implementation.

*Daxberger, E. et al. (2021). "Laplace Redux -- Effortless Bayesian Deep
Learning." NeurIPS.*

### `"tabpfn"` — TabPFN Zero-Shot

Transformer pretrained on synthetic tabular datasets. Outputs predictions
with no training on your data. Best for small N (< 3000).

*Hollmann, N. et al. (2025). "Accurate Predictions on Small Data with a
Tabular Foundation Model." Nature.*


## Preference and domain-robust surrogates

### `"bradley_terry"` — Pairwise Preference Model

Learns a utility function u(x) from pairwise comparisons (derived from
observed values within groups/studies). Candidates are ranked by u(x).
No explicit uncertainty — purely exploitative.

If `study_id` is present, pairs are sampled within studies (natural
comparison groups). Without `study_id`, all observations form one group.

*Bradley, R.A. & Terry, M.E. (1952). "Rank Analysis of Incomplete Block
Designs." Biometrika, 39(3/4).*

### `"groupdro"` — Group Distributionally Robust Optimization

Trains an MLP with GroupDRO: upweights groups (studies) with higher loss
via exponentiated gradient ascent. Encourages robust performance across
diverse experimental conditions rather than optimizing average performance.

Requires `study_id` column. Without it, falls back to standard ERM.

*Sagawa, S. et al. (2020). "Distributionally Robust Neural Networks for
Group Shifts." ICLR.*

```python
opt = Optimizer(
    space=space, surrogate_type="groupdro", candidate_pool=pool,
    surrogate_kwargs={"eta": 0.01, "epochs": 200},
)
```

### `"vrex"` — Variance Risk Extrapolation

Trains an MLP with V-REx: penalizes the variance of per-study losses.
This encourages invariant predictive features that transfer across studies.

Requires `study_id` column. Without it, falls back to standard ERM.

*Krueger, D. et al. (2021). "Out-of-Distribution Generalization via Risk
Extrapolation (REx)." ICML.*

```python
opt = Optimizer(
    space=space, surrogate_type="vrex", candidate_pool=pool,
    surrogate_kwargs={"lambda_rex": 1.0},
)
```


## Experimental (not yet in Optimizer)

These models are available under `models/experimental/` but require
different integration patterns and are not accessible through `Optimizer`:

- **MAML** (`maml_surrogate.py`) — Model-Agnostic Meta-Learning for few-shot
  BO. Meta-trains across studies, fine-tunes for target study with k
  observations. *Finn et al. (2017), ICML.*
- **FSBO** (`fsbo_surrogate.py`) — Warm-started GP hyperparameters from
  meta-training. *Inspired by Wistuba & Grabocka (2021), ICLR.*
- **D-MPNN** (`mpnn.py`) — Directed Message Passing Neural Network operating
  on molecular graphs. *Yang et al. (2019), JCIM.*
- **GPS-MPNN** (`gps_mpnn.py`) — D-MPNN with RWSE positional encodings,
  global self-attention, and cross-component attention.


## Choosing a surrogate

**Start with `"gp"` (default).** It provides exact posterior uncertainty,
supports all batch strategies, and works well across dataset sizes.

**When to switch:**

- **Noisy or outlier-heavy data** → `"robust_gp"`
- **Multi-study datasets** → `"multitask_gp"` (if transferring across studies)
  or `"groupdro"` / `"vrex"` (if robustness to study shift matters)
- **Large pools (> 10k candidates)** → tree-based (`"xgb_ucb"`, `"rf_ucb"`)
  for speed, or `"gp"` with automatic sparse GP (N > 1000)
- **Molecular fingerprint features** → `kernel_type="tanimoto"` with `"gp"`
- **Zero training data / cold start** → `"tabpfn"`
- **Want distance-aware neural UQ** → `"sngp"` or `"laplace"`
- **Compositional ratio optimization** → `"gp_mixed"` with
  `kernel_type="compositional"`
