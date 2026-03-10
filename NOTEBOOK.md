# LNPBO: Bayesian Optimization for LNP Formulation Design

**Headline result:** In a within-study benchmark across 23 LNPDB studies, all 22 optimization strategies significantly outperform random screening (p < 0.001, paired Wilcoxon). Tree-based models (NGBoost 1.39x, RF 1.37x) lead overall, but no single method wins everywhere -- CASMOPolitan wins 30% of individual studies, NGBoost 26%, RF 17%. The dominant source of performance variation is which study the data comes from (57%), not which algorithm is used (8%). LNPBO's GP-based optimizer shows its greatest advantage on ratio-only optimization (1.58x vs random), but trails tree models on lipid screening (1.17x vs 1.40x).

---

## 1. Problem Statement

Lipid nanoparticles (LNPs) are the delivery vehicle behind mRNA vaccines and therapeutics. Each formulation combines four components -- an ionizable lipid (IL), helper lipid (HL), cholesterol analog (CHL), and PEGylated lipid (PEG) -- at specified molar ratios. The design space is vast: LNPDB (Collins et al., Nature Communications 2026) catalogs ~19,200 formulations across ~12,000 unique ionizable lipids from 42 studies.

The optimization problem: given a small initial screen (typically 25% of a study's formulations, chosen at random), suggest which formulations to test next to find the best performers as quickly as possible. This is a pool-based Bayesian optimization problem where the oracle is the experimental readout (transfection efficiency, protein expression, etc.), z-scored within each study.

### Data Quality

LNPDB z-scores `Experiment_value` per study (mean=0, std=1). Two issues were corrected:

1. **Unnormalized rows:** 122 formulations with |value| > 10 (implausible as z-scores). Removed via `load_lnpdb_full(drop_unnormalized=True)`. Clean dataset: 16,787 rows.
2. **Z-score source inconsistency:** `LNPDB.csv` mixes raw and z-scored values for ~8,800 rows. The authoritative source is `all_data_all.csv`. Fixed via `load_lnpdb_full(use_zscore_source=True)`. Clean dataset: 19,199 rows, mean=0.005, std=1.007.

### LNPDB Setup

```bash
git clone https://github.com/evancollins1/LNPDB.git ../LNPDB
ln -s ../../LNPDB data/LNPDB_repo
```

The bridge module `data/lnpdb_bridge.py` mediates between LNPDB's data layout and LNPBO's `Dataset` class.

---

## 2. Codebase Overview

### Directory Structure

| Directory | Purpose |
|-----------|---------|
| `data/` | Dataset loading, molecular encoding (Morgan FP, count MFP, RDKit, mordred, Uni-Mol, CheMeleon, LiON), PCA/PLS reduction |
| `space/` | FormulationSpace with bounded simplex projection for molar ratio constraints |
| `optimization/` | GP-based acquisition (UCB, EI, LogEI, LP, KB) and discrete pool scoring (XGBoost, RF, GP, NGBoost, Deep Ensemble, CASMOPolitan) |
| `benchmarks/` | Strategy evaluation using LNPDB as oracle, SHAP analysis, within-study benchmark |
| `models/` | Standalone surrogate model training, transfer/meta-learning experiments |
| `diagnostics/` | Cross-study comparability diagnostics (ICC, ICP, scaffold analysis) |
| `cli/` | Subcommands for encode, suggest, and propose-ils |

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

### Running Benchmarks

```bash
# Within-study benchmark (main result)
python -m benchmarks.within_study_benchmark --resume

# Analyze results
python benchmarks/analyze_within_study.py

# Earlier formulation-level benchmark
python -m benchmarks.runner --strategies discrete_xgb_ucb --feature-type lantern_il_only --reduction pca
```

---

## 3. Within-Study Benchmark (Main Result)

This is the centerpiece experiment. Rather than pooling all formulations and splitting at the formulation level (which leaks within-study structure), each study is benchmarked independently: seed from that study, optimize within that study, evaluate against that study's oracle.

### 3a. Experimental Design

- **23 studies** from LNPDB, split by type:
  - 16 IL-diverse with fixed ratios (pure lipid screening, N=248-2400)
  - 5 IL-diverse with variable ratios (mixed optimization, N=286-1902)
  - 2 ratio-only (continuous optimization, N=1079-1080)
- **Pool-based retrospective benchmark:** 25% seed pool, 15 rounds of batch 12
- **23 strategies x 5 seeds = 2,645 total runs** (all combinations present, no missing data)
- **Metric:** Top-5% recall -- fraction of each study's top 5% formulations found within the evaluation budget
- **Features:** IL-diverse studies use LANTERN PCA (Morgan FP reduced to 5 PCs); ratio-only studies use raw molar ratios

### 3b. Overall Results

All 22 non-random strategies are statistically significant vs random (p < 0.001, paired Wilcoxon signed-rank, one-sided). Grand mean Top-5% recall across all strategies: 0.678.

**Strategy family rankings:**

| Family | Mean Top-5% | Lift vs Random | p (vs random) |
|--------|-------------|----------------|---------------|
| NGBoost | 0.737 | 1.39x | < 0.001 |
| RF | 0.731 | 1.37x | < 0.001 |
| CASMOPolitan | 0.716 | 1.35x | < 0.001 |
| XGBoost | 0.712 | 1.34x | < 0.001 |
| Deep Ensemble | 0.691 | 1.30x | < 0.001 |
| GP (sklearn) | 0.684 | 1.29x | < 0.001 |
| LNPBO (GP-based) | 0.642 | 1.21x | < 0.001 |
| Random | 0.532 | --- | --- |

Top individual strategies: RF-TS (0.772, 1.45x), NGBoost-UCB (0.737, 1.39x), XGB-UCB-TS-Batch (0.724, 1.36x), CASMO-UCB (0.721, 1.36x).

**Variance decomposition:**

| Source | % of Total Variance |
|--------|---------------------|
| Study | 57.1% |
| Strategy | 7.6% |
| Seed | 0.5% |
| Residual | 34.8% |

### 3c. What Works Where

**Lipid screening (fixed ratios, 16 studies).** The task is pure structure-activity learning: given a molecular fingerprint, predict efficacy. Tree models dominate.

| Family | Mean Top-5% | Lift |
|--------|-------------|------|
| NGBoost | 0.756 | 1.40x |
| RF | 0.744 | 1.38x |
| XGBoost | 0.735 | 1.36x |
| CASMOPolitan | 0.733 | 1.36x |
| LNPBO | 0.630 | 1.17x |
| Random | 0.540 | --- |

**Mixed optimization (variable ratios, 5 studies).** IL screening plus ratio tuning. The gap narrows between tree models and GP-based methods.

| Family | Mean Top-5% | Lift |
|--------|-------------|------|
| RF | 0.742 | 1.33x |
| NGBoost | 0.701 | 1.26x |
| XGBoost | 0.698 | 1.25x |
| GP (sklearn) | 0.694 | 1.24x |
| LNPBO | 0.684 | 1.22x |
| Random | 0.559 | --- |

**Ratio-only optimization (2 studies).** When the problem is smooth continuous optimization over mixing ratios, GP-based methods close the gap. CASMOPolitan (which uses a GP with trust-region search) leads, and LNPBO outperforms XGBoost for the first time.

| Family | Mean Top-5% | Lift |
|--------|-------------|------|
| CASMOPolitan | 0.689 | 1.74x |
| Deep Ensemble | 0.672 | 1.69x |
| NGBoost | 0.670 | 1.69x |
| LNPBO | 0.625 | 1.58x |
| RF | 0.600 | 1.51x |
| XGBoost | 0.560 | 1.41x |
| Random | 0.396 | --- |

Note: only 2 ratio-only studies -- treat these numbers with caution.

**Small vs large datasets.** All methods benefit more from optimization on larger datasets (where random sampling covers less of the pool). Tree models maintain their advantage at all sizes, but the lift gap widens with dataset size:

| Size Bin | Random | LNPBO (Lift) | XGBoost (Lift) | RF (Lift) |
|----------|--------|-------------|----------------|-----------|
| Small (<500, 10 studies) | 0.641 | 0.721 (1.12x) | 0.788 (1.23x) | 0.801 (1.25x) |
| Medium (500-1000, 6 studies) | 0.528 | 0.627 (1.19x) | 0.702 (1.33x) | 0.754 (1.43x) |
| Large (>1000, 7 studies) | 0.378 | 0.540 (1.43x) | 0.612 (1.62x) | 0.611 (1.62x) |

**No single winner.** Best family per study across all 23 studies: CASMOPolitan wins 7 (30%), NGBoost wins 6 (26%), RF wins 4 (17%), Deep Ensemble wins 3 (13%), XGBoost wins 1, GP (sklearn) wins 2.

### 3d. Key Takeaways

- **Every optimization strategy we tested finds top formulations significantly faster than random screening.** Even the weakest method (LNPBO) is 1.21x better than random, and the best (RF-TS) is 1.45x better. Directed optimization works.

- **Simple machine learning models consistently outperform more complex Gaussian Process models for identifying promising lipid structures.** Random Forest and NGBoost -- fast, off-the-shelf methods -- beat our custom GP-based optimizer by ~15 percentage points on lipid screening tasks.

- **Our GP-based optimizer shows its strength when optimizing mixing ratios rather than screening lipid libraries.** On ratio-only studies, LNPBO achieves 1.58x lift (outperforming XGBoost at 1.41x), though CASMOPolitan still leads at 1.74x.

- **Which experiment the data comes from matters far more than which algorithm you use.** Study identity explains 57% of performance variation; strategy choice explains only 8%. Investing in better experimental design may yield larger returns than surrogate engineering.

- **No single algorithm wins everywhere.** CASMOPolitan, NGBoost, and RF each win the most studies in different contexts. An adaptive approach that selects the strategy based on the problem type could outperform any fixed choice.

- **The seed random pool already captures most of the value.** All methods start at 87% of the oracle best after the 25% seed pool. The remaining optimization budget adds 5-15 percentage points depending on strategy.

- **Optimization gains scale with dataset size.** On large studies (>1000 formulations), the best methods achieve 1.62x lift vs random, compared to only 1.25x on small studies. This makes sense: larger pools mean random sampling covers less of the space, leaving more room for directed search.

### 3e. Caveats

1. **Cross-study comparability.** Each study's values were z-scored before benchmarking. Top-5% recall is rank-based within each study, making it somewhat comparable, but the underlying assays, cell types, and readouts differ. Aggregate statistics weight all studies equally regardless of clinical relevance or assay quality.

2. **Limited ratio diversity.** 16 of 23 studies use fixed molar ratios, reducing the optimization to pure IL screening. Only 5 studies have variable ratios with diverse ILs, and only 2 are ratio-only. Conclusions about continuous optimization rest on very few studies.

3. **Study size heterogeneity.** Study sizes span an order of magnitude (248 to 2400). The 25% seed pool means smaller studies start more explored. Larger studies have more room for improvement and dominate aggregate statistics.

4. **Seed sensitivity.** Only 5 random seeds per strategy. With 23 studies this yields 115 (study, seed) pairs, which is modest. Confidence intervals may be wider than they appear.

5. **Retrospective, not prospective.** This is a pool-based benchmark with a noise-free oracle. In a real self-driving lab, the candidate pool would need to be enumerated or generated, and synthesis/assay noise would affect results.

6. **Budget constraints.** All strategies use the same budget: 25% seed + 15 rounds of batch 12 (total ~28-34% of pool depending on study size). Rankings might differ under tighter budgets.

7. **Molecular encoding.** IL-diverse studies use LANTERN PCA (5 PCs); ratio-only studies use raw molar ratios. Different molecular representations might change relative strategy rankings.

---

## 4. Convergence Analysis

Convergence trajectories (fraction of oracle best, averaged across all studies and seeds) reveal distinct patterns:

| Round | Random | LNPBO | CASMOPolitan | XGBoost | RF | NGBoost |
|-------|--------|-------|--------------|---------|-------|---------|
| 0 (seed) | 0.873 | 0.873 | 0.873 | 0.873 | 0.873 | 0.873 |
| 5 | 0.908 | 0.911 | 0.921 | 0.914 | 0.916 | 0.919 |
| 10 | 0.940 | 0.930 | 0.935 | 0.927 | 0.930 | 0.949 |
| 15 (final) | 0.947 | 0.965 | 0.947 | 0.936 | 0.940 | 0.953 |

Key observations:

- **All families start at 87% of oracle after the seed pool.** The 25% random seed already covers most of the space.
- **LNPBO shows a distinctive late-surge pattern.** It trails tree models through rounds 1-12, then surges in rounds 13-15 to end up at 96.5% of oracle -- the highest final value of any family. This suggests the GP is slow to build an accurate model but makes high-quality suggestions once it does.
- **Tree models converge faster early.** NGBoost reaches 94.9% by round 10; LNPBO doesn't catch up until round 14.
- **Rounds to 90% of final performance:** CASMOPolitan (1.0), RF (1.1), XGBoost (1.2), NGBoost (1.5), LNPBO (1.6). Tree models and CASMOPolitan make their best gains in the first 1-2 rounds after seeding.

---

## 5. Timing and Compute

**Pareto frontier (performance vs wall-clock time):**

| Strategy | Time (s) | Top-5% Recall | Pareto-optimal? |
|----------|----------|---------------|-----------------|
| Random | 0.0 | 0.532 | Yes |
| GP-UCB (sklearn) | 0.1 | 0.684 | Yes |
| **RF-TS** | **1.4** | **0.772** | **Yes** |
| NGBoost-UCB | 7.8 | 0.737 | No |
| CASMO-UCB | 25.2 | 0.721 | No |
| XGB-CQR | 40.2 | 0.708 | No |

**Family-level timing:**

| Family | Mean (s) | Median (s) |
|--------|----------|------------|
| RF | 1.5 | 1.5 |
| Deep Ensemble | 3.9 | 3.9 |
| NGBoost | 7.8 | 8.4 |
| LNPBO | 8.2 | 5.0 |
| XGBoost | 17.2 | 4.7 |
| CASMOPolitan | 26.1 | 13.4 |

RF-TS is the clear performance-per-dollar winner: the best overall strategy at 1.4 seconds average runtime. NGBoost-UCB offers a good tradeoff at 7.8 seconds. CASMOPolitan is the most expensive at 26 seconds (up to 423s on large studies) and does not compensate with higher aggregate performance.

---

## 6. Earlier Results (Retained for Reference)

The within-study benchmark (Section 3) supersedes these earlier analyses, which used formulation-level splits (mixing formulations from different studies in train and test sets). They are retained for context.

**Formulation-level benchmark.** On the pooled 19,199-formulation dataset (n_seed=500, batch=12, 15 rounds), only 3 strategies achieved p < 0.05 vs random: XGB-UCB and NGBoost-UCB (both 13.2% Top-50) and RF-TS-Batch (11.6%). Variance decomposition: seed 38%, strategy 24%, residual 38%. This benchmark conflated within-study and cross-study signal.

**Cross-study generalization.** ICC = 0.006 globally (z-scoring removes batch effects). ICP found zero invariant causal features across studies. Top-50 recall dropped from 20.4% (formulation split) to 8.4% (study-level split). Conclusion: the model learns within-study scaffold identification, not transferable SAR rules.

**Held-out study benchmark.** On Collins et al.'s 4 held-out studies (5-fold CV, matched protocol), XGB+LANTERN PCA achieved mean Spearman r=0.307, beating LiON D-MPNN (0.219) on all 4 studies. AGILE GNN was essentially random (Pearson r ~ 0). Key: aggressive PCA reduction to 10 features prevents overfitting.

**Model training.** Best standalone model: XGBoost Optuna-tuned R^2=0.376 (scaffold split, seed 42). MPNN 4-component R^2=0.355. Tabular ML on fingerprints matches or exceeds end-to-end GNNs. The R^2 ceiling (~0.40) is driven by feature representation.

**SHAP feature attribution.** IL molecular features account for 82.3% of SHAP importance (count MFP 44.1%, RDKit 39.1%). Molar ratios: 14.1%. HL/CHL/PEG molecular: <5% combined. Permutation importance assigns ~100% to IL -- helper lipid encoding has zero marginal predictive value.

**Meta-learning and transfer.** MAML, FSBO, Bradley-Terry, V-REx, GroupDRO all failed to rescue cross-study transfer. No learned surrogate outperformed random on held-out studies. The structure-activity relationship is fundamentally study-dependent.

---

## 7. GP Implementation Notes

LNPBO implements GP-based Bayesian optimization with BoTorch/GPyTorch backends:

- **Numerical parity verified:** UCB, EI, and LogEI implementations all match BoTorch native acquisitions to atol=1e-6 (test suite: `tests/test_botorch_numerical_parity.py`).
- **Kriging Believer and Local Penalization** batch strategies have no BoTorch equivalent -- these are custom implementations citing Ginsbourger et al. (2010) and Gonzalez et al. (2016) respectively.
- **qLogNoisyExpectedImprovement** is used instead of qLogExpectedImprovement (correct for noisy experimental data).
- **Sparse GP** uses manual Adam optimization with VariationalELBO (not `fit_gpytorch_mll`), which provides more control over convergence in the high-dimensional fingerprint space.
- **LogEI** follows Ament et al., NeurIPS 2023 (arXiv:2310.20708, Eq. 9) for numerically stable log-space expected improvement.
- **Simplex projection** for molar ratio constraints uses Michelot (1986) KKT + Brent root-finding.

---

## 8. Future Work

- **Statistical rigor:** Bootstrap confidence intervals on family rankings; mixed-effects model with study as random effect, strategy as fixed effect.
- **Adaptive strategy selection:** Meta-learner that picks the optimization strategy based on study characteristics (size, ratio variability, IL diversity). No single method wins everywhere — an adaptive approach could outperform any fixed choice.
- **Tighter budget experiments:** Current budget is 28-34% of pool. Rankings may change under a 10% budget.
- **Prospective validation:** All results are retrospective pool-based benchmarks. A real self-driving lab demonstration is needed.
- **Ionizable lipid design:** Generative design of novel ionizable lipids guided by the optimization pipeline.
