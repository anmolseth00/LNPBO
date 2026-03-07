# LNPBO: Bayesian Optimization for LNP Formulation Design

**Top result: LANTERN IL-only + XGBoost discrete greedy -- 24.0% top-10 recall (±4.9%) at n_seed=500.** For Top-50, LANTERN+UniMol batch6 (30 rounds, batch=6) leads at 26.0% ±7.4%. At n_seed=1000, prospective PLS achieves 26.4% Top-50 (3.0x vs random). Ionizable lipid molecular structure accounts for 82% of SHAP importance and 100% of permutation importance — helper lipid encoding adds no marginal predictive value.

## 1. Problem Statement

LNP (lipid nanoparticle) formulations consist of four lipid components (ionizable lipid, helper lipid, cholesterol analog, PEGylated lipid) at specified molar ratios. The combinatorial space is vast: LNPDB contains ~19,800 formulations across thousands of unique ionizable lipids. The goal is an optimization pipeline that, given a small initial screen, suggests the next batch of formulations most likely to maximize a target property (e.g., transfection efficiency).

### 1a. LNPDB Data Quality

LNPDB (Collins et al., Nature Communications 2026) z-scores `Experiment_value` per study (mean=0, std=1). We identified 122 formulations with `|Experiment_value| > 10`, implausible as z-scores, indicating normalization failures in the source data. The most extreme cases are 8 formulations from PMID 38424061 with raw values of 135-215. `load_lnpdb_full(drop_unnormalized=True)` (default) removes these. Clean dataset: 16,787 rows, mean=0.005, std=1.007, max=9.27.

All results in this notebook use the clean dataset unless explicitly noted.

### 1b. LNPDB Dependency

Clone the LNPDB repo as a sibling directory or symlink it:

```bash
git clone https://github.com/evancollins1/LNPDB.git ../LNPDB
ln -s ../../LNPDB data/LNPDB_repo
```

The bridge module `data/lnpdb_bridge.py` mediates between LNPDB's data layout and LNPBO's `Dataset` class.

## 2. Codebase

- `data/`: Dataset loading, molecular encoding (Morgan FP, count MFP, RDKit, mordred, Uni-Mol, CheMeleon, LiON), PCA/PLS reduction
- `space/`: FormulationSpace with bounded simplex projection for molar ratio constraints
- `optimization/`: GP-based acquisition (UCB, EI, LogEI, Local Penalization, Kriging Believer) and discrete pool scoring (XGBoost, RF-UCB, RF-TS, GP-UCB)
- `benchmarks/`: Strategy evaluation using LNPDB as oracle, SHAP analysis, PCA interpretation
- `models/`: Standalone surrogate model training and evaluation (see `models/README.md`)
- `pipeline.py`: End-to-end CLI for suggesting new formulations
- `cli/`: Subcommands for encode, suggest, and propose-ils

### Feature Type Glossary

| CLI name | Description |
|----------|-------------|
| `mfp` | Binary Morgan FP (1024-bit), PCA-reduced |
| `count_mfp` | Count-based Morgan FP (2048-bit), PCA-reduced |
| `rdkit` | ~210 RDKit 2D descriptors, PCA-reduced |
| `mordred` | ~1613 mordred 2D descriptors, PCA-reduced |
| `chemeleon` | CheMeleon pretrained embeddings (2048-dim), PCA-reduced |
| `lantern` | Count MFP + RDKit descriptors, PCA/PLS-reduced (all 4 roles) |
| `lantern_il_only` | LANTERN encoding for IL only (helpers get ratios only) |
| `lantern_unimol` | Count MFP + RDKit + Uni-Mol, PCA-reduced |
| `ratios_only` | Molar ratios and mass ratio only (no molecular encoding) |

The `--reduction` flag controls dimensionality reduction: `pca` (unsupervised), `pls` (supervised, target-aware), or `none` (raw features).

## 3. Feature Engineering

### 3a. Molecular Representations

| Feature | Dims | Reference |
|---------|------|-----------|
| Morgan FP (binary, `mfp`) | 1024 | Rogers & Hahn, J Chem Inf Model 2010 |
| Morgan FP (count, `count_mfp`) | 2048 | Rogers & Hahn 2010 |
| RDKit descriptors (`rdkit`) | ~210 | Landrum, RDKit |
| Mordred descriptors (`mordred`) | ~1613 | Moriwaki et al., J Cheminformatics 2018 |
| CheMeleon embeddings (`chemeleon`) | 2048 | Vasan et al. |
| Uni-Mol embeddings | 512 | Zhou et al., ICLR 2023 |
| LANTERN (`lantern`) | ~2258 | Mehradfar et al., arXiv:2507.03209 |

Uni-Mol is the same molecular backbone used by COMET (Chan et al., Nature Nanotechnology 2025). We could not use COMET directly on LNPDB because: (1) pairwise ranking loss, not regression; (2) trained on only 14 molecules; (3) requires synthesis parameters LNPDB doesn't standardize; (4) restrictive license.

### 3b. Feature Comparison (n_seed=500, 5 seeds, XGB discrete greedy, 15 rounds, batch=12)

| Feature | Top-10 | Top-50 | Top-100 |
|---------|--------|--------|---------|
| Random baseline | 10.0 ±8.9% | 7.6 ±5.0% | 6.6 ±2.6% |
| Ratios only | 8.0 ±7.5% | 4.8 ±2.0% | 5.4 ±1.9% |
| Count MFP PCA | 14.0 ±10.2% | 23.2 ±7.4% | 22.2 ±4.3% |
| Mordred 20PC | 10.0 ±15.5% | 22.8 ±5.3% | 21.6 ±3.5% |
| **LANTERN IL-only** | **24.0 ±4.9%** | 20.4 ±5.3% | 20.6 ±5.4% |
| LANTERN PCA (all roles) | 12.0 ±19.4% | 23.6 ±11.1% | 22.4 ±8.1% |
| LANTERN+UniMol | 8.0 ±11.7% | 21.6 ±6.6% | 20.8 ±4.5% |
| LANTERN batch6 (30r) | 8.0 ±11.7% | 22.8 ±10.5% | 22.8 ±7.4% |
| **LANTERN+UniMol batch6** | 12.0 ±14.7% | **26.0 ±7.4%** | **23.6 ±4.4%** |

Key findings:

- **Ratios alone are worse than random** — lipid identity dominates composition.
- **Count-based MFP > binary MFP** — substructure frequency matters more than presence/absence.
- **LANTERN IL-only is the most robust Top-10 config** (24% ±4.9%, consistent 20-30% every seed).
- **LANTERN+UniMol batch6 leads Top-50** (26.0%) — Uni-Mol adds signal only with enough refit cycles (30 rounds vs 15).
- **Mordred 20PC has the lowest variance** (±5.3% Top-50, ±3.5% Top-100) — most reliable for budget-constrained campaigns.
- Foundation model embeddings (Uni-Mol, CheMeleon) underperform simple count-based fingerprints at batch=12/15 rounds.

### 3c. Top-10 vs Top-50: Different Optimal Configs

LANTERN IL-only wins Top-10 but trails on Top-50. This makes chemical sense: the **top-10 formulations share a single IL scaffold** (12_A\*_T\*b tricyclohexyl series) regardless of helper lipid composition. Extra helper features add noise for these specific molecules. The broader Top-50 spans more diverse IL scaffolds where composition varies more, so additional features provide marginal signal.

Practical implication: use IL-only encoding for finding the very best formulations; use LANTERN+UniMol batch6 for finding a broad set of good ones.

### 3d. Top-10 Bimodal Distribution

In a 20-seed analysis of LANTERN PCA, Top-10 recall shows a phase transition: 55% of seeds find 0 hits, 25% find 80-100%. This is a genuine structural property of LNPDB — 9 of the top 10 formulations share one Murcko scaffold. If the seed set contains any member, XGBoost finds the rest; if not, it cannot.

LANTERN IL-only avoids this collapse: its simpler feature space (10 PCs vs 33 for full LANTERN) yields consistent 20-30% across all seeds instead of the bimodal 0-or-50% pattern.

**Top-50 is the correct primary metric** — it spans multiple scaffolds and is stable across seeds.

## 4. Feature Attribution

### 4a. SHAP TreeExplainer (5-seed mean)

| Feature Group | SHAP % |
|---------------|--------|
| IL molecular (count_mfp PCs) | 44.1% |
| IL molecular (RDKit PCs) | 39.1% |
| Molar ratios | 14.1% |
| IL-to-nucleic-acid mass ratio | 2.8% |
| HL molecular | 3.7% |
| PEG molecular | 0.7% |
| CHL molecular | 0.2% |

**By role:** IL=82.3%, HL=8.2% (mostly ratios), CHL=3.1% (mostly ratios), PEG=6.5% (mostly ratios).

### 4b. Permutation Importance (5-seed mean)

Permutation importance on held-out data gives the most honest measure of marginal predictive value:

- **IL molecular: ~100%** of all importance
- **HL, CHL, PEG molecular: 0%** — zero marginal predictive contribution
- **All molar ratios: 0%** — zero marginal contribution
- **Mass ratio: ~1%**

The SHAP vs permutation discrepancy is expected: SHAP measures how much the model *uses* a feature, permutation measures whether *removing* it degrades prediction. The model uses ratios internally, but they are redundant with IL features for prediction.

### 4c. Importance Method Comparison

| Method | IL | HL | Ratios | Interpretation |
|--------|----|----|--------|---------------|
| XGBoost gain | 65% | 12% | 14.5% | Biased toward correlated features |
| SHAP TreeExplainer | 82.3% | 3.7% | 14.1% | What the model uses |
| Permutation importance | ~100% | 0% | 0% | What actually predicts |

XGBoost gain inflates HL importance from 3.7% (SHAP) to 12% (gain) due to correlation with IL features. Use SHAP or permutation importance for attribution in tree models.

Script: `benchmarks/shap_importance.py`

## 5. PCA Interpretation: What XGBoost Learns

Analysis of count_mfp PCA components (2048-bit, 5 PCs):

- Each PC explains <2% variance — very high intrinsic dimensionality in fingerprint space
- Cumulative 5 PCs explain ~7% of total fingerprint variance
- Each PC captures a different IL chemistry series (SN, A34-T, disulfides, paracyclophanes, 12_A\*_T\*b)
- PC5 separates the dominant 12_A\*_T\*b scaffold — this directly explains the bimodal Top-10 distribution
- Individual PC-transfection correlations are weak (Spearman rho < 0.08); power comes from nonlinear PC interactions
- **XGBoost learns scaffold family identification, not generalizable structure-activity rules**

Script: `benchmarks/pc_interpretation.py`

## 6. Surrogate Models

### 6a. GP Failure Mode

At n_seed=100, all GP-based strategies (UCB, EI, LogEI, Local Penalization, Kriging Believer) perform worse than random:
1. GP surrogate is non-predictive (Spearman ~0.04) with 100 training points in 19D
2. Continuous optimization + NN-matching adds quantization noise
3. All acquisition functions collapse to the same maximizer

### 6b. Discrete Surrogates

Scoring the candidate pool directly eliminates the NN-matching confound. XGBoost greedy scoring is the best surrogate at all seed pool sizes, with RF-TS excelling at Top-10 (needle-in-haystack).

### 6c. Standalone Model Training (seed 42, scaffold split)

| Model | R^2 | Details |
|-------|-----|---------|
| RF default | 0.384 | `models/eval_multiseed.py` |
| XGB default | 0.387 | `models/eval_multiseed.py` |
| **XGB Optuna-tuned** | **0.376** | depth=12, lr=0.016, 100 trials |
| MPNN (4-component) | 0.355 | `models/train_lion.py` |
| GPS-MPNN | 0.328 | `models/train_gps.py` |

Tabular ML on fingerprints matches or exceeds end-to-end GNNs. Multi-component encoding provides no benefit (only 9 unique HLs, 16 CHLs, 14 PEGs). The R^2 ceiling (~0.40) is driven by feature representation, not model architecture.

## 7. Dimensionality Reduction

### 7a. PCA vs PLS

PLS (supervised, target-aware; Wold et al. 2001) can outperform PCA but requires careful implementation. Fitting PLS on full-dataset targets is target leakage; the live pipeline re-fits PLS each round using only observed training targets (`Dataset.refit_pls()`).

**Prospective PLS results (5 seeds, no leakage):**

| n_seed | Top-10 | Top-50 | Top-100 | vs Random |
|--------|--------|--------|---------|-----------|
| 500 | 10.0 ±15.5% | 19.6 ±11.7% | 19.8 ±4.7% | 2.6x |
| 1000 | 18.0 ±17.2% | **26.4 ±9.1%** | **28.4 ±6.5%** | 3.0x |

At n=500, PCA (23.6% Top-50) outperforms PLS (19.6%) — supervised reduction overfits at smaller sample sizes. PLS becomes advantageous at n≥1000. Leaked PLS (fit on full-dataset targets) showed 59.6% Top-50 at n=1000 — dramatically inflated.

### 7b. Mordred PCA Sweep

Mordred's 1613 descriptors require more PCs than Morgan FP's 2048 bits:
- 5 PCs: 14.8% Top-50 (too aggressive)
- 10 PCs: 20.8%
- **20 PCs: 22.8 ±5.3%** (optimal, lowest variance)
- 50 PCs: 21.2%

IL chemical space intrinsic dimensionality is low — ~20 PCs capture the useful variation.

## 8. Ablation Study

| Configuration | Top-50 | Delta | Interpretation |
|---------------|--------|-------|----------------|
| Random | 7.6% | — | Baseline |
| Ratios only | 4.8% | -2.8pp | Ratios alone are worse than random |
| LANTERN IL-only | 20.4% | +15.6pp | **IL structure is the dominant signal** |
| LANTERN PCA (all roles) | 23.6% | +3.2pp | Helper encoding adds modest signal |

- **IL structure: +15.6pp** over ratios-only (dominant, highly significant)
- **Helper lipid encoding: +3.2pp** over IL-only (marginal, within SE at 5 seeds)
- **Molar ratios: no marginal value** (permutation importance = 0%)

## 9. Context Features (SHAP analysis)

LNPDB spans diverse experimental conditions. One-hot encoding of 6 context columns (cell type, target, cargo, measurement method, route of administration) improves R² from 0.813 to 0.878 (scaffold split).

| Feature Group | SHAP % |
|---------------|--------|
| IL fingerprint | 55.5% |
| Molar ratios | 10.9% |
| Experiment method | 9.9% |
| Cargo type | 7.7% |
| HL fingerprint | 5.6% |
| Cell type | 3.0% |
| Mass ratio | 2.8% |
| Target organ | 2.1% |
| PEG + CHL fingerprint | 1.5% |

**Molecular: 76.5% / Context: 23.5%.** Context adds real signal — different cell types and cargo types shift the response distribution. The top context feature (`Experiment_method__diameter`) reflects that diameter/zeta measurements are on fundamentally different scales than transfection luminescence.

Script: `scripts/shap_analysis.py`

## 10. Best Configuration

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Features | `lantern_il_only` | Best Top-10, lowest variance, simplest |
| Reduction | `pca` (5 components) | Unsupervised, no leakage risk |
| Surrogate | XGBoost greedy | Best broad coverage |
| Strategy | Discrete pool scoring | Eliminates NN-matching noise |
| Batch size | 6 or 12 | 6 with 30r for best Top-50; 12 for convenience |
| Seed pool | 500+ | BO non-predictive below ~200 |

**For Top-10 (finding the best):** `lantern_il_only`, PCA, batch=12, 15 rounds → 24.0% ±4.9%

**For Top-50 (finding many good ones):** `lantern_unimol`, PCA, batch=6, 30 rounds → 26.0% ±7.4%

**At n_seed=1000:** `lantern`, PLS, batch=12, 15 rounds → 26.4% ±9.1% (3.0x vs random)

## 11. Pipeline Usage

```bash
# Recommended (IL-only, most robust)
python pipeline.py --feature-type lantern_il_only --reduction pca

# Best Top-50 (needs 30 rounds)
python -m benchmarks.runner --strategies discrete_xgb_greedy --feature-type lantern_unimol --rounds 30 --batch-size 6 --n-seeds 500

# Benchmark comparison
python -m benchmarks.runner --strategies discrete_xgb_greedy,random --rounds 15 --n-seeds 500 --feature-type lantern_il_only
```

## 12. Next Steps

### 12a. Calibrated Uncertainty (Implemented)

XGB-UCB wraps XGBoost with MAPIE conformal prediction (CV+ method, Barber et al., AoS 2021). Performs comparably to greedy at n=1000 — exploration doesn't help when the pool is large enough that greedy exploitation is effective.

### 12b. Generative Molecular Design (Implemented)

AGILE-lite via `propose-ils`: SELFIES mutations around known ILs, LANTERN+PLS encoding, MAPIE uncertainty, LCB ranking (mean - std), MaxMin diversity selection. See `cli/propose_ils.py`.

### 12c. Uncertainty Sampling

In the n<200 regime where surrogates are non-predictive, active learning-style uncertainty sampling could improve early-round sample efficiency. Pool-based AL (Settles, 2009) is directly applicable.

### 12d. Mixed-Variable Bayesian Optimization

CoCaBO (Ru et al., NeurIPS 2020) and CASMOPOLITAN (Wan et al., JMLR 2021) handle mixed categorical-continuous spaces, which could jointly optimize lipid selection and molar ratios in a single BO loop.
