# LNPBO Development Progress

## 1. Codebase Modernization (from v0)

Refactored the original LNPBO codebase into a clean Python package:

- Added `__init__.py` files, `.gitignore`, `requirements.txt`
- Removed dead code (`preprocessing/scaling.py`)
- Modernized serialization (pickle to joblib)
- Fixed space module (import order, bounds shape, simplex projection)
- Refactored fingerprint generators to return `(scaled, scaler)` tuples
- Added PLS reduction support in `compute_pcs.py`
- Refactored `dataset.py` (reduction param, fitted_transformers, data quality)
- Added `data/lnpdb_bridge.py` for loading full LNPDB (~19,800 formulations)
- Refactored `FormulationSpace` with `_build_parameters`, `update_with_dataset`
- Cleaned up `KrigingBeliever`, added `LogExpectedImprovement`, `LocalPenalization`
- Wired new acquisition functions in `bayesopt.py` and `optimizer.py`
- Added end-to-end `pipeline.py` and CLI module refactoring
- Lint/type fixes with ruff/ty

## 2. Surrogate Model Training

Trained RF, XGBoost, and RF+XGB ensemble on LNPDB using Morgan fingerprints (2048-bit per component) as features. Train/val/test = 80/10/10 split. Scaffold-based splitting attempted but largest scaffold group (55.2%) forced fallback to stratified random split.

### 2a. Single-Component (IL-only) Results (5 seeds)

| Model | RMSE | MAE | R2 |
|-------|------|-----|-----|
| RF | 0.773 +/- 0.031 | 0.549 +/- 0.017 | 0.384 +/- 0.034 |
| XGB (default) | 0.771 +/- 0.031 | 0.555 +/- 0.016 | 0.387 +/- 0.035 |
| RF+XGB Ensemble | 0.764 +/- 0.030 | 0.546 +/- 0.016 | 0.397 +/- 0.033 |

### 2b. XGBoost Optuna-Tuned (5 seeds)

100-trial Optuna search over learning_rate, max_depth, n_estimators, subsample, colsample_bytree, min_child_weight, reg_alpha, reg_lambda.

| Model | RMSE | MAE | R2 |
|-------|------|-----|-----|
| XGB Tuned | 0.764 +/- 0.030 | 0.548 +/- 0.014 | 0.398 +/- 0.031 |
| RF | 0.774 +/- 0.031 | 0.549 +/- 0.017 | 0.382 +/- 0.034 |
| Tuned Ensemble | 0.762 +/- 0.031 | 0.544 +/- 0.016 | 0.400 +/- 0.032 |

### 2c. Multi-Component (IL+HL+CHL+PEG) Results (5 seeds)

| Model | RMSE | MAE | R2 |
|-------|------|-----|-----|
| RF | 0.772 +/- 0.032 | 0.549 +/- 0.017 | 0.385 +/- 0.035 |
| XGB | 0.772 +/- 0.034 | 0.556 +/- 0.017 | 0.385 +/- 0.037 |
| RF+XGB Ensemble | 0.765 +/- 0.033 | 0.546 +/- 0.017 | 0.397 +/- 0.036 |

### 2d. Key Takeaway

Best R2 on LNPDB is ~0.40 regardless of model choice (RF, XGB, ensemble, tuned or default). The ceiling is the feature representation (Morgan FP PCs), not the model architecture. Multi-component encoding provides no improvement over IL-only, consistent with IL being the primary driver of LNP activity.

## 3. Bayesian Optimization Benchmark

Built `benchmark.py`: a simulated closed-loop BO harness using LNPDB as oracle. At each round, optimizer suggests a batch of 12 formulations, oracle returns true Experiment_value.

### 3a. GP-Based Strategies (continuous optimization + NN matching)

Config: 100 seed formulations, 10 rounds, batch=12, copula normalization, subset=5000.

| Strategy | Final Best | AUC | Top-10 | Top-50 | Top-100 |
|----------|-----------|-----|--------|--------|---------|
| Random | 214.69 | 14.14 | 20% | 6% | 7% |
| GP+KB(UCB) | 4.02 | 2.77 | 0% | 4% | 4% |
| GP+KB(EI) | 4.02 | 2.77 | 0% | 4% | 4% |
| GP+KB(LogEI) | 4.02 | 2.77 | 0% | 4% | 4% |
| GP+LP(EI) | 4.02 | 2.77 | 0% | 4% | 4% |
| GP+LP(LogEI) | 4.02 | 2.77 | 0% | 4% | 4% |
| GP+KB(PLS+LogEI) | 2.92 | 2.32 | 0% | 2% | 2% |
| GP+LP(PLS+LogEI) | 2.92 | 2.32 | 0% | 2% | 2% |

All GP-based strategies (UCB, EI, LogEI, LP, KB) produce identical batch selections. The GP posterior with 100 training points in 19D has a single dominant mode; all acquisition functions agree on the same maximizer. PLS performs worse due to target leakage in encoding.

Random sampling massively outperforms all BO strategies because:
1. BO operates in continuous space then NN-matches to oracle pool (quantization noise)
2. GP surrogate is essentially non-predictive (Spearman ~0.03-0.06 with 100 training points on Morgan FP PCs)
3. Random selects directly from the pool, avoiding the NN-matching penalty

### 3b. Discrete Candidate Pool Strategies

To remove the NN-matching confound, implemented discrete strategies that score all remaining pool candidates directly. Smoke test (subset=2000, 3 rounds):

| Strategy | Top-10 | Top-50 | Top-100 |
|----------|--------|--------|---------|
| Random | 10% | 12% | 8% |
| Discrete GP-UCB | 10% | 8% | 8% |
| Discrete RF-UCB | 10% | 8% | 7% |
| Discrete RF-TS | 20% | 12% | 8% |
| Discrete XGB | 10% | 8% | 9% |

RF-TS (Thompson Sampling) shows the strongest signal. All strategies start from the same seed pool outlier (4.46), so final_best is dominated by seed quality. Full-scale discrete benchmark pending.

### 3c. Root Cause

The bottleneck is surrogate model quality, not acquisition function choice. With only 100 training points, Morgan FP-based models (GP, RF, XGB) cannot predict LNP activity. The acquisition function (UCB vs EI vs LogEI) and batch strategy (KB vs LP) are irrelevant when the surrogate provides no useful signal.

## 4. Literature Review Findings

- **LANTERN (2025)**: Morgan FP + MLP achieves R2=0.816 on LNP prediction, outperforming GNNs (R2=0.266). Suggests MLP on fingerprints is a strong baseline.
- **COMET (Nature Nanotechnology)**: Purpose-built for LNP design. Uses Uni-Mol 3D molecular embeddings, Gaussian composition encoding, multi-component attention. Achieves Spearman 0.873 on LANCE dataset. Code at github.com/alvinchangw/COMET.
- Acquisition function improvements (LogEI, LP) provide marginal gains in standard benchmarks but are irrelevant when the surrogate is non-predictive.

## 5. Next Steps

Primary bottleneck: surrogate quality. Priority is improving feature representations.

Options (ordered by expected impact):
1. **Fine-tune COMET on LNPDB** (Option B): Leverage pre-trained Uni-Mol embeddings with COMET architecture, fine-tune on our 19,800 formulations.
2. **Use COMET as feature extractor** (Option A): Extract Uni-Mol embeddings from COMET, use as drop-in replacement for Morgan FP PCs in existing GP/RF/XGB surrogates.
3. Run full-scale discrete benchmark with more rounds and seeds to establish baseline performance metrics.
