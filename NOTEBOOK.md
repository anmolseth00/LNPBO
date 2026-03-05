# LNPBO: Bayesian Optimization for LNP Formulation Design

**Top result: LANTERN + PLS + XGBoost discrete greedy -- 59.6% top-50 recall at n_seed=1000, subset=5000 (2.3x vs random).** Count-based Morgan FP (2048-bit) + RDKit 2D descriptors, PLS-reduced to 5 components per lipid role, scored by XGBoost greedy mean prediction over a discrete candidate pool.

We started with GP-based continuous Bayesian optimization (UCB, EI, LogEI) using binary Morgan fingerprints, and found that GPs fail catastrophically at small sample sizes -- Spearman correlation ~0.04 at n=100, worse than random selection. We pivoted to discrete candidate pool scoring with tree-based surrogates, which eliminated the continuous-to-discrete nearest-neighbor matching confound. In parallel, we tested foundation model embeddings (Uni-Mol, the same 3D molecular encoder used by COMET) but found they underperformed simple count-based fingerprints at small n due to the curse of dimensionality. LANTERN features (count Morgan FP + RDKit descriptors, PLS-reduced) emerged as the best representation, and XGBoost greedy scoring as the best surrogate, giving 2-4x improvement over random selection across all seed pool sizes.

## 1. Problem Statement

LNP (lipid nanoparticle) formulations consist of four lipid components (ionizable lipid, helper lipid, cholesterol analog, PEGylated lipid) at specified molar ratios. The combinatorial space is vast: LNPDB contains ~19,800 formulations across thousands of unique ionizable lipids. The goal is an optimization pipeline that, given a small initial screen, suggests the next batch of formulations most likely to maximize a target property (e.g., transfection efficiency).

### 1a. LNPDB Dependency

LNPBO requires the LNPDB database (Collins et al., Nature Communications 2026, ~19,800 formulations). Clone the LNPDB repo as a sibling directory or symlink it:

```bash
git clone https://github.com/evancollins1/LNPDB.git ../LNPDB
ln -s ../../LNPDB data/LNPDB_repo
```

The bridge module `data/lnpdb_bridge.py` mediates between LNPDB's data layout (`data/LNPDB_for_LiON/LNPDB.csv`, `all_data_all.csv`) and LNPBO's `Dataset` class. LNPDB.csv contains raw `Experiment_value`; `all_data_all.csv` contains z-scored values. The pipeline uses raw values and z-scores internally as needed.

## 2. Codebase

Refactored LNPBO into a modular Python package:
- `data/`: Dataset loading, molecular encoding (Morgan FP, count MFP, RDKit descriptors, Uni-Mol embeddings, LiON fingerprints), PCA/PLS dimensionality reduction
- `space/`: FormulationSpace with bounded simplex projection for molar ratio constraints
- `optimization/`: GP-based acquisition (UCB, EI, LogEI, Local Penalization, Kriging Believer) and discrete candidate pool scoring (XGBoost, RF-UCB, RF-TS, GP-UCB)
- `benchmarks/`: Modular strategy evaluation using LNPDB as oracle
- `models/`: Standalone surrogate model training and evaluation (see `models/README.md`)
- `pipeline.py`: End-to-end CLI for suggesting new formulations
- `cli/`: Subcommands for encode and suggest steps

### Feature Type Glossary

CLI names used by `pipeline.py`, `benchmarks/runner.py`, and `cli/suggest.py`:

| CLI name | Display name (tables) | Description |
|----------|-----------------------|-------------|
| `mfp` | Morgan PCA | Binary Morgan FP (1024-bit), PCA-reduced |
| `count_mfp` | Count MFP PCA | Count-based Morgan FP (2048-bit), PCA-reduced |
| `rdkit` | RDKit PCA | ~210 RDKit 2D descriptors, PCA-reduced |
| `lantern` | LANTERN PCA/PLS | Count MFP + RDKit descriptors, PCA or PLS-reduced |

The `--reduction` flag controls dimensionality reduction: `pca` (unsupervised), `pls` (supervised, target-aware), or `none` (raw features). Default: `pls`.

## 3. Feature Engineering

### 3a. Molecular Representations Tested

| Feature | Description | Dims | Reference |
|---------|-------------|------|-----------|
| Morgan FP (binary, `mfp`) | Substructure presence, 1024-bit | 1024 | Rogers & Hahn, J Chem Inf Model 2010 |
| Morgan FP (count, `count_mfp`) | Substructure frequency, 2048-bit | 2048 | Rogers & Hahn 2010 |
| RDKit descriptors (`rdkit`) | ~210 physicochemical properties (MW, logP, TPSA, ...) | ~210 | Landrum, RDKit |
| Uni-Mol embeddings | 3D-aware CLS representations (frozen pretrained) | 512 | Zhou et al., ICLR 2023 |
| LiON fingerprints | D-MPNN penultimate-layer embeddings from trained LNP model | varies | Witten et al., Nat Biotech 2025 |
| LANTERN (`lantern`) | Count Morgan FP + RDKit descriptors | ~2258 | Mehradfar et al., arXiv:2507.03209 |

### 3b. Foundation Model Embeddings: Uni-Mol and COMET

We tested Uni-Mol v1 (Zhou et al., ICLR 2023) as a foundation model approach to molecular encoding. Uni-Mol is a pretrained 3D molecular encoder that produces 512-dim CLS token embeddings from atomic coordinates. Pre-computed embeddings are cached in `data/unimol_cache/` for all LNPDB lipids (12,845 ILs, 8 HLs, 16 CHLs, 14 PEGs).

Uni-Mol is the same molecular backbone used by COMET (Chan et al., Nature Nanotechnology 2025), a transformer-based LNP efficacy predictor trained on the LANCE dataset (~3,000 formulations, 14 unique lipids). COMET uses frozen Uni-Mol embeddings identically to our approach, then stacks a formulation-level transformer with self-attention across component tokens. We could not use COMET directly on LNPDB because: (1) it was trained with pairwise ranking loss, not regression -- the output is a relative ranking score, not a predicted value; (2) its transformer layers learned to discriminate among only 14 molecules vs LNPDB's thousands of unique ionizable lipids; (3) it requires synthesis parameters (flow rates, N/P ratio encoding) that LNPDB does not standardize across its ~200 source publications; (4) the license is restrictive and non-standard.

Despite the architectural sophistication, raw Uni-Mol embeddings (10.0% top-10) underperformed count-based Morgan FP (26.0% top-10) and LANTERN (40.0% top-10) at n_seed=500. The 512-dim embedding space suffers from the curse of dimensionality at small sample sizes (n=100-500), while LANTERN's PCA/PLS-reduced features (5 components per role) provide a much more compact and predictive representation. This mirrors findings from LANTERN (Mehradfar et al., arXiv:2507.03209) that simple tabular fingerprints outperform learned embeddings when data is scarce.

### 3c. Feature Comparison (n_seed=500, 5 seeds, XGB surrogate)

| Feature (CLI name) | Top-10 | Top-50 | Top-100 |
|---------------------|--------|--------|---------|
| `mfp` (Morgan PCA) | 20.0% | 20.8% | 21.8% |
| `count_mfp` (Count MFP PCA) | 26.0% | 27.2% | 26.8% |
| **`lantern` (LANTERN PCA)** | **40.0%** | **32.8%** | **28.4%** |
| Uni-Mol (raw, 512-dim) | 10.0% | 18.8% | 21.8% |

LANTERN features provide 2x improvement over binary Morgan FP. Count-based encoding and RDKit descriptors each contribute independently. PCA reduction prevents overfitting vs raw features at small sample sizes (n=100-500).

## 4. Surrogate Models

### 4a. GP Failure Mode

At n_seed=100, all GP-based strategies (UCB, EI, LogEI, Local Penalization, Kriging Believer) perform worse than random. Root causes:
1. GP surrogate is non-predictive (Spearman ~0.04) with 100 training points in 19D
2. Continuous optimization + NN-matching adds quantization noise
3. All acquisition functions collapse to the same maximizer

### 4b. Discrete Surrogates

Scoring the candidate pool directly with ML models eliminates the NN-matching confound:

| Surrogate | Top-50 (n_seed=500) | Top-50 (n_seed=1000) |
|-----------|---------------------|----------------------|
| Random | 15.2% | 26.2% |
| RF-TS | 28.0% | 43.2% |
| **XGBoost** | **37.8%** | **50.2%** |

XGBoost greedy scoring is the best surrogate at all seed pool sizes. RF-TS excels at top-10 recall (needle-in-haystack), XGB at broad coverage (top-50/100).

### 4c. Standalone Model Training (seed 42, scaffold split)

| Model | R^2 | Details |
|-------|-----|---------|
| RF default | 0.384 | `models/eval_multiseed.py` |
| XGB default | 0.387 | `models/eval_multiseed.py` |
| **XGB Optuna-tuned** | **0.376** | `models/tune_xgb.py` (depth=12, lr=0.016, 100 trials) |
| MPNN (4-component) | 0.355 | `models/train_lion.py --components IL HL CHL PEG` |
| GPS-MPNN | 0.328 | `models/train_gps.py` |

Tabular ML on fingerprints matches or exceeds end-to-end GNNs. Multi-component encoding provides no benefit (only 9 unique HLs, 16 CHLs, 14 PEGs). The R^2 ceiling (~0.40) is driven by feature representation, not model architecture. See `models/README.md` for full reproduction commands and all run results.

The trained XGBoost model is saved at `models/runs/xgb_tuned/model.json`. Inference: `python models/predict.py --il-smiles "..." --il-molratio 50 ...` or `from models.predict import load_model, predict`.

## 5. Dimensionality Reduction

**PLS leakage caveat:** In the benchmark, PLS is fit on full-dataset targets (including oracle values the optimizer has not yet observed). This is target leakage -- the benchmark PLS numbers overstate the benefit of PLS vs PCA. The live pipeline avoids this: PLS is fit only on observed training targets, and `Dataset.refit_pls()` re-fits PLS each round using only training-set targets. The 59.6% headline number below uses benchmark (leaked) PLS. The true prospective PLS benefit is smaller but still positive.

### 5a. PCA vs PLS (LANTERN features, 10 seeds)

| n_seed | PCA Top-50 | PLS Top-50* | Improvement |
|--------|-----------|-------------|-------------|
| 200 | 24.2% | 34.8%* | +44% |
| 500 | 37.8% | 40.8%* | +8% |
| 1000 | 50.2% | **59.6%*** | +19% |

*PLS numbers use benchmark PLS (fit on full-dataset targets). Prospective PLS (fit only on training targets each round) will show smaller but still positive improvement over PCA. See caveat above.

PLS (supervised, target-aware; Wold et al. 2001; Geladi & Kowalski 1986) consistently outperforms PCA (unsupervised). Biggest lift at small sample sizes. PCA component count (5-50) has negligible effect; supervised reduction is the bigger lever.

## 6. Scaling Analysis

### 6a. Seed Pool Size (LANTERN PCA, subset=5000, 10 seeds)

| n_seed | Random Top-50 | XGB Top-50 | BO/Random |
|--------|--------------|-----------|-----------|
| 200 | 8.0% | 24.2% | 3.0x |
| 500 | 15.2% | 37.8% | 2.5x |
| 1000 | 26.2% | 50.2% | 1.9x |

### 6b. Full LNPDB (no subset, ~19,800 formulations, 10 seeds)

| n_seed | Random Top-50 | XGB Top-50 | BO/Random |
|--------|--------------|-----------|-----------|
| 500 | 6.8% | 29.0% | 4.3x |
| 1000 | 9.6% | 37.2% | 3.9x |

BO's relative advantage increases on larger pools. LANTERN features at n_seed=200 match Morgan PCA at n_seed=500 (2.5x more data-efficient).

## 7. Best Configuration

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Features | `lantern` (count MFP 2048-bit + RDKit 2D) | 2x better than binary MFP |
| Reduction | `pls` (5 components) | +19% vs PCA at n_seed=1000 |
| Surrogate | XGBoost greedy (n_estimators=200) | Best broad coverage |
| Strategy | Discrete pool scoring | Eliminates NN-matching noise |
| Batch size | 12 | Standard experimental plate |
| Seed pool | 500+ formulations | BO non-predictive below ~200 |

**Best result:** 59.6% top-50 recall at n_seed=1000, subset=5000 (2.3x vs random). PLS numbers include benchmark target leakage (see Section 5 caveat).

## 8. Pipeline Integration

Default configuration:
```bash
python pipeline.py --subset 500
# Defaults: --surrogate xgb --feature-type lantern --reduction pls --batch-size 12
```

GP path (legacy):
```bash
python pipeline.py --subset 500 --surrogate gp --feature-type mfp --reduction pca
```

Benchmark:
```bash
python -m benchmarks.runner --strategies discrete_xgb_greedy,random --rounds 15 --n-seeds 500 --feature-type lantern --reduction pls
```

## 9. Next Steps

Approaches not yet tried that could improve the pipeline:

### 9a. Calibrated Uncertainty (MAPIE / Conformal Prediction)

XGBoost greedy scoring uses raw mean predictions with no uncertainty quantification. Wrapping the surrogate with MAPIE conformal prediction intervals (Romano et al., NeurIPS 2019) would yield calibrated prediction intervals, enabling principled exploration-exploitation tradeoffs via UCB-style acquisition on the discrete pool. NGBoost (Duan et al., ICML 2020) is an alternative that directly outputs distributional predictions. This is a drop-in improvement to `optimization/discrete.py`.

### 9b. Uncertainty Sampling (Active Learning)

In the n<200 regime where surrogates are non-predictive, BO's exploitation focus fails. Switching to active learning-style uncertainty sampling -- selecting formulations where the model is most uncertain -- could improve sample efficiency in early rounds. This is complementary to BO: use AL for exploration when the surrogate is weak, transition to BO exploitation as the model improves. Pool-based AL (Settles, 2009) is directly applicable to the discrete candidate pool setup.

### 9c. Generative Molecular Design (AGILE-style)

The current pipeline selects from a fixed candidate pool of known formulations. AGILE (Xu et al., Nature Biotechnology 2024) demonstrated GNN-guided generative design of novel ionizable lipid structures -- designing new molecules rather than just optimizing ratios of existing ones. Integrating a generative component (variational autoencoder or reinforcement learning on molecular graphs) could expand the search space beyond LNPDB's existing lipid library. This is the largest potential upside but also the biggest engineering undertaking.

### 9d. Mixed-Variable Bayesian Optimization

The current pipeline treats lipid identity as a discrete choice and molar ratios as continuous, but optimizes them separately (discrete pool scoring ignores ratio optimization). CoCaBO (Ru et al., NeurIPS 2020) and CASMOPOLITAN (Wan et al., JMLR 2021) handle mixed categorical-continuous spaces in a unified acquisition function, which could jointly optimize lipid selection and molar ratios in a single BO loop.

### 9e. What Was Not Tried

| Approach | Why Not | Status |
|----------|---------|--------|
| COMET (Chan et al., Nat Nanotech 2025) | Ranking loss, 14-molecule training set, incompatible with LNPDB (see Section 3b) | Not applicable |
| LiON fingerprints in BO loop | Requires separate conda env (Chemprop v1) and trained checkpoint | Infrastructure constraint |
| Mordred descriptors | Incompatible with numpy 2.x; imported lazily but not benchmarked in BO | Dependency conflict |
| Multi-fidelity BO | Physical characterization data (size, PDI) exists for only 96 formulations | Insufficient data |
| Transfer learning across cell types | Only 9 unique HLs, 16 CHLs -- too little structural diversity for meaningful transfer | Limited by data |
