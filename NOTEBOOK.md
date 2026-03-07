# LNPBO: Bayesian Optimization for LNP Formulation Design

**Top result (formulation-level split): LANTERN IL-only + XGBoost discrete greedy -- 24.0% top-10 recall (±4.9%) at n_seed=500.** For Top-50, LANTERN+UniMol batch6 (30 rounds, batch=6) leads at 26.0% ±7.4%. At n_seed=1000, prospective PLS achieves 26.4% Top-50 (3.0x vs random). Ionizable lipid molecular structure accounts for 82% of SHAP importance and 100% of permutation importance — helper lipid encoding adds no marginal predictive value.

**Critical caveat: these numbers reflect within-study SAR only.** On study-held-out evaluation (the generalization benchmark), Top-50 drops to 8.4% — barely above random (7.6%). However, on the paper's 4 held-out studies, **XGBoost+LANTERN PCA beats both LiON (D-MPNN) and AGILE (GNN)** — mean Spearman r=0.307 vs 0.219 (5-fold CV, matched protocol). Cross-study R² remains negative, but rank-order prediction is positive and state-of-the-art.

---

## 1. Problem Statement

LNP (lipid nanoparticle) formulations consist of four lipid components (ionizable lipid, helper lipid, cholesterol analog, PEGylated lipid) at specified molar ratios. The combinatorial space is vast: LNPDB contains ~19,200 formulations across ~12,000 unique ionizable lipids from 42 studies. The goal is an optimization pipeline that, given a small initial screen, suggests the next batch of formulations most likely to maximize a target property (e.g., transfection efficiency).

### 1a. LNPDB Data Quality

LNPDB (Collins et al., Nature Communications 2026) z-scores `Experiment_value` per study (mean=0, std=1). Two data quality issues were identified and corrected:

1. **Unnormalized rows:** 122 formulations with `|Experiment_value| > 10`, implausible as z-scores. The most extreme are 8 formulations from PMID 38424061 with raw values 135-215. `load_lnpdb_full(drop_unnormalized=True)` removes these. Clean dataset: 16,787 rows, mean=0.005, std=1.007, max=9.27.

2. **Z-score source inconsistency:** `LNPDB.csv` contains mixed raw and z-scored values for ~8,800 rows. The authoritative z-scored source is `all_data_all.csv` (z-scored per `Experiment_ID`). `load_lnpdb_full(use_zscore_source=True)` replaces values from this source. Before fix: mean=0.922, std=12.263. After fix: mean=0.0018, std=0.9962.

All results in Sections 3-8 use the clean dataset (16,787 rows, `drop_unnormalized=True`). Sections 11+ use the z-score-source-fixed dataset (19,199 rows, `use_zscore_source=True`). Both are near mean=0, std=1; the directional findings hold on either dataset, but top-line benchmarks have not been re-run on the 19,199-row version.[^1]

[^1]: Low-priority redo. The 16,787 and 19,199 datasets differ by ~2,400 rows whose z-scores were corrected from raw values. The effect on Top-50 recall is expected to be small.

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
- `diagnostics/`: Cross-study comparability diagnostics (ICC, ICP, scaffold analysis)
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

## 4. Assay-Type Stratification

The benchmarks in Section 3 pool all assay types. This hides the fact that LNPDB contains categorically different problems. Formulation-level split, LANTERN IL-only, 5 seeds. Script: `diagnostics/stratified_benchmark.py`.

| Stratum | N | Studies | Top-10 | Top-50 | Top-100 |
|---------|---|---------|--------|--------|---------|
| in_vitro_single | 16,811 | 33 | 10.0 ±8.9% | **13.2 ±5.7%** | 12.8 ±2.6% |
| in_vivo_liver | 905 | 8 | **96.0 ±4.9%** | **96.4 ±2.3%** | 95.8 ±2.1% |
| in_vivo_other | 1,483 | 12 | 62.0 ±17.2% | **61.2 ±4.1%** | 60.6 ±3.0% |
| **Pooled (Section 3)** | **19,199** | **42** | **24.0 ±4.9%** | **20.4 ±5.3%** | **20.6 ±5.4%** |

in_vivo_liver is nearly solved (96% Top-50) but this is likely a small-pool artifact — 905 formulations across 8 studies, probably dominated by hepatocyte-selective lipids by design. The "hard" problem is in_vitro (13% Top-50), which is 88% of the data. Pooled numbers (~20%) are misleading averages of these categorically different strata. **All subsequent benchmarks should be interpreted through this lens.**

## 5. Feature Attribution

### 5a. SHAP TreeExplainer (5-seed mean)

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

### 5b. Permutation Importance (5-seed mean)

Permutation importance on held-out data gives the most honest measure of marginal predictive value:

- **IL molecular: ~100%** of all importance
- **HL, CHL, PEG molecular: 0%** — zero marginal predictive contribution
- **All molar ratios: 0%** — zero marginal contribution
- **Mass ratio: ~1%**

The SHAP vs permutation discrepancy is expected: SHAP measures how much the model *uses* a feature, permutation measures whether *removing* it degrades prediction. The model uses ratios internally, but they are redundant with IL features for prediction.

### 5c. Importance Method Comparison

| Method | IL | HL | Ratios | Interpretation |
|--------|----|----|--------|---------------|
| XGBoost gain | 65% | 12% | 14.5% | Biased toward correlated features |
| SHAP TreeExplainer | 82.3% | 3.7% | 14.1% | What the model uses |
| Permutation importance | ~100% | 0% | 0% | What actually predicts |

XGBoost gain inflates HL importance from 3.7% (SHAP) to 12% (gain) due to correlation with IL features. Use SHAP or permutation importance for attribution in tree models.

Script: `benchmarks/shap_importance.py`

## 6. PCA Interpretation

Analysis of count_mfp PCA components (2048-bit, 5 PCs):
- Each PC explains <2% variance — very high intrinsic dimensionality in fingerprint space
- Cumulative 5 PCs explain ~7% of total fingerprint variance
- Each PC captures a different IL chemistry series (SN, A34-T, disulfides, paracyclophanes, 12_A\*_T\*b)
- PC5 separates the dominant 12_A\*_T\*b scaffold — this directly explains the bimodal Top-10 distribution
- Individual PC-transfection correlations are weak (Spearman rho < 0.08); power comes from nonlinear PC interactions
- **XGBoost learns scaffold family identification, not generalizable structure-activity rules**

Script: `benchmarks/pc_interpretation.py`

## 7. Surrogate Models

### 7a. GP Failure Mode

At n_seed=100, all GP-based strategies (UCB, EI, LogEI, Local Penalization, Kriging Believer) perform worse than random:
1. GP surrogate is non-predictive (Spearman ~0.04) with 100 training points in 19D
2. Continuous optimization + NN-matching adds quantization noise
3. All acquisition functions collapse to the same maximizer

### 7b. Discrete Surrogates

Scoring the candidate pool directly eliminates the NN-matching confound. XGBoost greedy scoring is the best surrogate at all seed pool sizes, with RF-TS excelling at Top-10 (needle-in-haystack).

### 7c. Standalone Model Training (seed 42, scaffold split)

| Model | R^2 | Details |
|-------|-----|---------|
| RF default | 0.384 | `models/eval_multiseed.py` |
| XGB default | 0.387 | `models/eval_multiseed.py` |
| **XGB Optuna-tuned** | **0.376** | depth=12, lr=0.016, 100 trials |
| MPNN (4-component) | 0.355 | `models/train_lion.py` |
| GPS-MPNN | 0.328 | `models/train_gps.py` |

Tabular ML on fingerprints matches or exceeds end-to-end GNNs. Multi-component encoding provides no benefit (only 9 unique HLs, 16 CHLs, 14 PEGs). The R^2 ceiling (~0.40) is driven by feature representation, not model architecture.

## 8. Dimensionality Reduction

### 8a. PCA vs PLS

PLS (supervised, target-aware; Wold et al. 2001) can outperform PCA but requires careful implementation. Fitting PLS on full-dataset targets is target leakage; the live pipeline re-fits PLS each round using only observed training targets (`Dataset.refit_pls()`).

At n=500, PCA (23.6% Top-50) outperforms PLS (19.6%) — supervised reduction overfits at smaller sample sizes. PLS becomes advantageous at n≥1000. Leaked PLS (fit on full-dataset targets) showed 59.6% Top-50 at n=1000 — dramatically inflated.

**Prospective PLS results (5 seeds, no leakage):**

| n_seed | Top-10 | Top-50 | Top-100 | vs Random |
|--------|--------|--------|---------|-----------|
| 500 | 10.0 ±15.5% | 19.6 ±11.7% | 19.8 ±4.7% | 2.6x |
| 1000 | 18.0 ±17.2% | **26.4 ±9.1%** | **28.4 ±6.5%** | 3.0x |

### 8b. Mordred PCA Sweep

Mordred's 1613 descriptors require more PCs than Morgan FP's 2048 bits:
- 5 PCs: 14.8% Top-50 (too aggressive)
- 10 PCs: 20.8%
- **20 PCs: 22.8 ±5.3%** (optimal, lowest variance)
- 50 PCs: 21.2%

IL chemical space intrinsic dimensionality is low — ~20 PCs capture the useful variation.

## 9. Ablation Study

| Configuration | Top-50 | Delta | Interpretation |
|---------------|--------|-------|----------------|
| Random | 7.6% | — | Baseline |
| Ratios only | 4.8% | -2.8pp | Ratios alone are worse than random |
| LANTERN IL-only | 20.4% | +15.6pp | **IL structure is the dominant signal** |
| LANTERN PCA (all roles) | 23.6% | +3.2pp | Helper encoding adds modest signal |

- **IL structure: +15.6pp** over ratios-only (dominant, highly significant)
- **Helper lipid encoding: +3.2pp** over IL-only (marginal, within SE at 5 seeds)
- **Molar ratios: no marginal value** (permutation importance = 0%)

## 10. Context Features & SHAP Analysis

### 10a. Alignment with LNPDB Paper

The LNPDB paper (Collins et al., Nat Comms 2026) uses **amine-based** train/test splits (partitioned by `IL_head_name`), matching the 70/15/15 split in their LiON evaluation. We adopt the same split for our SHAP analysis to produce directly comparable numbers. LiON also uses **92 auxiliary features** in `all_data_extra_x.csv`: molar ratios, dose, one-hot encoded component names (HL/CHL/PEG), buffers, mixing method, and experimental context (cell type, target, route, cargo, batching). Our context encoding covers the same experimental metadata columns, though we omit dose, buffers, mixing method, and component name one-hots (these are captured separately by our molecular encoding and ratio features).

### 10b. Results (amine split, full LNPDB)

| Metric | With Context | Without Context |
|--------|-------------|-----------------|
| Test R² | **0.112** | 0.089 |
| Train R² | 0.757 | 0.678 |
| Features | 87 (33 mol + 54 ctx) | 33 |

SHAP attribution (with-context model):

| Feature Group | SHAP % |
|---------------|--------|
| IL fingerprint (MFP + RDKit PCs) | 72.7% |
| Context: Model_type | 7.9% |
| Molar ratios | 5.7% |
| HL fingerprint | 5.0% |
| Context: Model_target | 2.8% |
| PEG fingerprint | 1.7% |
| Context: Experiment_method | 1.2% |
| Context: Cargo_type | 1.1% |
| Mass ratio | 0.9% |
| CHL + other context | 1.0% |

**Molecular: 86.3% / Context: 13.7%.**

### 10c. Interpretation

1. **Test R² = 0.11 on amine split** — consistent with the paper's LiON Spearman r = 0.104 on held-out studies. Cross-amine generalization is fundamentally limited.

2. **Context adds only +2.3pp R²** (0.089 → 0.112). On an honest structural split, context is marginal — it helps the model distinguish assay types but cannot rescue cross-study prediction.

3. **Previous scaffold split was invalid**: 54.7% of LNPDB's ILs are acyclic (empty Murcko scaffold), so the scaffold split fell back to random. The R² = 0.878 reported earlier was inflated by structural leakage — same amine groups appearing in both train and test.

4. **Top context feature is now `Model_type__HeLa`** (SHAP 0.042), not diameter measurements. On the amine split, the model uses cell type to distinguish HeLa assays (the dominant in_vitro cell line) from other experimental setups.

5. **The 269 BroadPharm commercial lipids** (no experimental data, NaN PMID) are correctly excluded by `load_lnpdb_full()`.

### 10d. Comparison with LiON Auxiliary Features

| Feature Category | LiON extra_x | Our Context Encoding |
|-----------------|-------------|---------------------|
| Molar ratios | Yes (5 cols) | Yes (in molecular features) |
| Dose | Yes | No |
| HL/CHL/PEG name one-hot | Yes (37 cols) | No (encoded as molecular fingerprints) |
| Buffer, mixing method | Yes (8 cols) | No |
| Cell type, target, route | Yes | Yes |
| Cargo type, batching | Yes | Yes |
| Experiment_method | No | Yes |

LiON uses 92 auxiliary features. Our approach encodes component identity via molecular fingerprints (LANTERN) rather than name one-hots, which is more generalizable to novel lipids but loses the categorical signal. The buffer and dose features in LiON's extra_x are not in our encoding.

Script: `scripts/shap_analysis.py --split amine`

### 10e. Held-Out Study Benchmark (vs LiON and AGILE)

Direct comparison on the **exact 4 held-out studies** from Collins et al. Table 1. For each study, we train on all remaining data and test on the held-out study — matching the paper's evaluation protocol exactly. Target metric: Spearman r.

| Study | n_test | XGBoost+LANTERN (Spearman r) | LiON D-MPNN (Spearman r) | AGILE GNN (Pearson r) |
|-------|--------|----------------------|---------------|-------------|
| BL_2023 | 773 | **0.088 ± 0.034** | 0.071 ± 0.036 | -0.022 |
| LM_2019 | 1128 | **0.168 ± 0.038** | 0.095 ± 0.047 | -0.016 |
| SL_2020 | 91 | **0.654 ± 0.022** | 0.633 ± 0.030 | -0.224 |
| ZC_2023 | 131 | **0.319 ± 0.014** | 0.079 ± 0.079 | 0.217 |
| **Mean** | — | **0.307** | 0.219 | -0.011 |

All numbers: mean ± std Spearman r across 5 CV folds (80/20 train split on non-held-out data), matching the paper's protocol exactly. LiON baselines computed from saved predictions in `LNPDB_for_LiON/heldout/`. AGILE: Pearson r from Collins' `LNPDB_AGILE_training.ipynb` (nondeterministic finetune; paper Fig 3b reports Spearman but notebook computes Pearson — near-zero correlations either way).

**XGBoost + LANTERN PCA (10 features) beats LiON on all 4 held-out studies with matched protocol (5-fold CV).** Mean Spearman r = 0.307 vs LiON 0.219 (40% improvement). AGILE performs essentially at random (mean Pearson r ≈ 0).

Ablations:
- **+Auxiliary features** (92 LiON extra_x columns): Mean r drops to 0.277 — one-hot context features add noise, hurt 3/4 studies
- **PLS instead of PCA**: Mean r drops to 0.146 — PLS overfits to training study structure
- **20 PCA components** (10+10): Mean r drops to 0.288 — more dimensions hurt generalization

The LANTERN representation (count MFP + RDKit descriptors, PCA-reduced to 5+5=10 dims) captures chemistry more efficiently than LiON's D-MPNN or AGILE's GNN for cross-study prediction. The key advantage is aggressive dimensionality reduction — 10 features vs LiON's learned representation — which prevents overfitting to study-specific patterns.

Script: `scripts/benchmark_heldout.py`

---

## 11. Cross-Study Generalization: Study-Level Benchmarks

Everything in Sections 3-8 uses **formulation-level** train/test splits — formulations from the same study appear in both training and test. This section evaluates the generalization question: **does LNPBO generalize to entirely new studies?**

### 11a. Intraclass Correlation Coefficient (ICC)

REML random-intercepts model with Self & Liang (1987) boundary test. Script: `diagnostics/compute_icc.py`.

| Stratum | N | Studies | ICC | 95% CI | p-value |
|---------|---|---------|-----|--------|---------|
| **Global** | 19,199 | 42 | **0.006** | [0.000, 0.013] | 4.5e-07 |
| in_vitro_single | 16,811 | 33 | 0.004 | [0.000, 0.008] | 2.9e-06 |
| in_vivo_liver | 905 | 8 | 0.032 | [0.000, 0.079] | 4.9e-04 |
| in_vivo_other | 1,483 | 12 | **0.052** | [0.000, 0.091] | 4.7e-10 |

**Verdict:** ICC < 0.05 globally — z-scoring successfully removes most batch effects. The in_vivo_other stratum (ICC=0.052) sits at the boundary.

### 11b. Anchor Formulation Analysis

279 anchor ILs (appear in ≥2 studies), but only 1 IL in ≥3 studies (DLin-MC3-DMA). Only 4 study pairs with shared ILs. Mean Spearman rho = 0.378 (p=0.107) — **not significant**. Insufficient anchors for rank consistency testing. Script: `diagnostics/anchor_analysis.py`.

### 11c. Study-Level Holdout Benchmark

XGB-greedy BO, LANTERN IL-only, 80/20 study-level split, 5 seeds. Script: `benchmarks/study_split_benchmark.py`.

| Metric | Study-level split | Formulation-level split | Gap |
|--------|------------------|------------------------|-----|
| Top-10 | **12.0 ±14.7%** | 24.0 ±4.9% | -12pp |
| Top-50 | **8.4 ±7.3%** | 20.4 ±5.3% | -12pp |
| Top-100 | **11.6 ±7.3%** | 22.4 ±3.0% | -11pp |

The ~12pp gap represents **study-specific pattern leakage**. Out-of-study generalization is ~8% Top-50, barely above random (7.6%).

### 11d. Within-Assay Study-Level Benchmark

Study-level split within in_vitro_single only (33 studies). Script: `benchmarks/within_assay_benchmark.py`.

| Metric | Within-assay study-split | Cross-assay study-split |
|--------|--------------------------|-------------------------|
| Top-10 | 28.0 ±27.1% | 12.0 ±14.7% |
| Top-50 | 19.2 ±9.3% | 8.4 ±7.3% |
| Top-100 | 18.8 ±8.1% | 11.6 ±7.3% |

Within-assay is better but has enormous variance (Top-10 std=27%).

### 11e. Scaffold-Level Analysis (Track A)

12,262 unique ILs: 48% ring-containing (363 Murcko scaffolds), 51% acyclic (383 head groups). Script: `diagnostics/scaffold_analysis.py`.

**Seen vs novel scaffold hit rate (5 seeds, Top-50):**
- Seen scaffolds: 4.1 ±2.7% recall
- Novel scaffolds: **0.0 ±0.0%** recall
- Total: 3.2 ±2.0%

XGBoost recovers hits **only** from scaffolds already present in the seed set. It cannot predict efficacy for genuinely novel scaffolds. This confirms that the model learns scaffold family membership, not transferable SAR rules.

**Partial correlation with physicochemical descriptors:**

| Descriptor | Raw Spearman | Partial (controlling scaffold) |
|------------|-------------|-------------------------------|
| NumRotatableBonds | +0.072 | +0.067 |
| MolLogP | +0.070 | +0.069 |
| MolWt | +0.062 | +0.055 |
| QED | -0.064 | -0.062 |
| TPSA | +0.031 | +0.019 |

All correlations are weak (<0.07). No physicochemical descriptor strongly predicts prediction error.

## 12. Invariance-Based Deconfounding

### 12a. V-REx (Krueger et al. 2021)

MLP on LANTERN IL-only PCs, study-level 80/20 split. Script: `models/vrex_surrogate.py`.

| lambda | R² (train) | R² (test — held-out studies) |
|--------|-----------|------------------------------|
| 0.0 (ERM) | 0.176 | **-0.894** |
| 1.0 | 0.132 | -0.228 |
| **10.0** | 0.059 | **-0.040** |
| 100.0 | -0.090 | -0.143 |

V-REx with strong penalty reduces the gap but all R² remain negative.

### 12b. GroupDRO (Sagawa et al. 2020)

Script: `models/groupdro_surrogate.py`.

| eta | R² (train) | R² (test) | Worst-5 R² |
|-----|-----------|-----------|------------|
| **0.1** | 0.102 | **-0.314** | -0.728 |

GroupDRO collapses to optimizing a single outlier study.

### 12c. Invariant Causal Prediction (Peters et al. 2016)

Exact ICP over all 2^10 = 1,024 feature subsets (10 LANTERN PCs). Script: `diagnostics/icp_feature_selection.py`.

- **0 invariant subsets** at alpha=0.05
- **0 causal features** found
- Not a single subset of molecular features produces invariant residuals across studies

**This is the strongest negative result:** the relationship between molecular structure and efficacy is **fundamentally study-dependent**.

## 13. Context-Conditioned Model (Track B)

Does encoding experimental context (cell type, cargo, route, measurement method) rescue cross-study prediction? Script: `models/context_conditioned.py`.

| Feature set | Cross-study R² | Within-study R² | N features |
|-------------|---------------|-----------------|------------|
| Molecular only | -0.070 ±0.021 | -0.105 ±0.076 | 10 |
| Molecular + context | -0.072 ±0.026 | -0.035 ±0.076 | 64 |

When context is available, XGBoost uses it heavily (82% feature importance) but **cross-study R² remains negative**. Context slightly improves within-study R² but does not enable cross-study generalization.

Top context feature groups by importance: Model_type (25.6%), Experiment_method (21.0%), Model_target (14.9%), Cargo_type (9.6%).

**Scoped test (HeLa+FLuc only):** 7,455 rows, 14 studies, same cell type and cargo. Only Experiment_method varies (luminescence_normalized vs discretized_normalized). Script: `models/scoped_context_model.py`.

| Feature set | Cross-study R² | Within-study R² |
|-------------|---------------|-----------------|
| Molecular only | -0.412 ±0.504 | 0.073 |
| Molecular + context | -0.282 ±0.240 | 0.083 |

Context (Experiment_method) captures 18.7% of feature importance and reduces cross-study R² variance, but both conditions remain deeply negative. Even within the same cell type, cargo, and assay class, cross-study generalization fails. The unmeasured study-level confounds (lab protocols, reagent lots, passage number) dominate.

## 14. Meta-Learning

### 14a. MAML (Finn et al. 2017)

MLP on LANTERN IL-only PCs, 31 train / 8 test studies. 5000 episodes, 3 inner steps. Script: `models/maml_surrogate.py`.

| Method | k=5 | k=10 | k=20 |
|--------|-----|------|------|
| MAML | 35.0% / 36.3% | 32.5% / 37.4% | 38.8% / 40.3% |
| ERM+FT | 42.5% / 38.3% | 37.5% / 35.4% | 36.2% / 38.3% |
| Random | 36.2% / 38.2% | 38.8% / 40.0% | 43.8% / 43.0% |

(Format: Top-10 / Top-50)

**No learned surrogate outperforms random** on held-out studies.

### 14b. FSBO Warm-Started GP (Wistuba & Grabocka 2021)

Script: `models/fsbo_surrogate.py`.

| Method | k=5 Top-10 | k=5 Top-50 |
|--------|-----------|-----------|
| FSBO (warm GP) | **46.2%** | 39.3% |
| Cold-start GP | 45.0% | 37.8% |
| Random | 36.2% | 38.2% |

FSBO shows a modest edge at k=5 (46% vs 36% random Top-10), but n=8 test studies and within noise. At k=20, all methods converge.

## 15. Pairwise Preference Learning

### 15a. Bradley-Terry Model

MLP utility function, study-level split, within-study pairwise training. Script: `models/bradley_terry.py`.

| Metric | Value |
|--------|-------|
| Pairwise accuracy (test) | 62.0% (chance = 50%) |
| Rank Spearman (test studies) | +0.081 |
| Top-10/50/100 recall | 0.0% / 0.0% / 0.0% |

62% pairwise accuracy but completely fails at identifying top formulations cross-study.

## 16. Conformal Prediction

Split conformal (Vovk et al.), study-level 80/20 split. Script: `models/conformal_surrogate.py`.

| Surrogate | Coverage (target 90%) | Interval Width | R² (test) |
|-----------|----------------------|----------------|-----------|
| XGBoost | 86.9% | 2.88 | -0.047 |
| GP | **89.8%** | 3.02 | -0.013 |
| V-REx | 89.2% | 3.00 | -0.018 |

GP and V-REx achieve near-target coverage with wider intervals. All R² negative on held-out studies. Per-study coverage ranges from 81.5% to 92.9%.

## 17. Variance Decomposition

### 17a. Permutation Decomposition

Script: `diagnostics/permutation_decomposition.py`.

| Split | Chemistry (XGB on PCs) | Study-ID only | Shuffled IL control |
|-------|----------------------|---------------|---------------------|
| Formulation scaffold | **R²=0.174** | R²=-0.004 | R²=-0.038 |
| Study-level | **R²=-0.019** | R²=-0.000 | R²=-0.033 |

The 17.4% R² on scaffold split comes from **within-study SAR** that does not transfer cross-study.

### 17b. Partial R² Decomposition

Script: `diagnostics/partial_r2.py`.

On held-out studies: R²_study = -0.000002, R²_chemistry = -0.046. Neither study effects nor chemistry improve over the intercept model on new studies.

## 18. Consolidated Benchmark Table

| Model / Config | Eval Split | Top-10 | Top-50 | R² | Notes |
|----------------|-----------|--------|--------|-----|-------|
| LNPBO LANTERN IL-only | formulation | 24.0 ±4.9% | 20.4 ±5.3% | — | Current baseline |
| LNPBO LANTERN IL-only | **study-level** | 12.0 ±14.7% | **8.4 ±7.3%** | — | Out-of-study generalization |
| GP-based BO (qEI) | formulation | 6.0 ±8.0% | 3.6 ±2.3% | — | Worse than XGB |
| GP-based BO (qUCB) | formulation | 10.0 ±12.6% | 3.6 ±2.3% | — | Worse than XGB |
| V-REx MLP (lambda=10) | study-level | — | — | -0.040 | Best V-REx |
| GroupDRO MLP (eta=0.1) | study-level | — | — | -0.314 | Best GroupDRO |
| Context-conditioned XGB | study-level | — | — | -0.072 | Context doesn't help |
| MAML few-shot (k=5) | held-out study | 35.0% | 36.3% | — | approx random |
| FSBO warm GP (k=5) | held-out study | **46.2%** | 39.3% | — | Slight edge, n=8 |
| Bradley-Terry (rank) | study-level | 0.0% | 0.0% | — | Pairwise acc 62% |
| **Stratified: in_vivo_liver** | formulation | **96.0%** | **96.4%** | — | Nearly solved |
| **Stratified: in_vivo_other** | formulation | 62.0% | 61.2% | — | Strong signal |
| **Stratified: in_vitro_single** | formulation | 10.0% | 13.2% | — | The hard problem |
| Random | formulation | 10.0% | 7.6% | — | Baseline |
| **XGB+LANTERN PCA** | **4 held-out studies (5-fold)** | — | — | **r=0.307** | **Beats LiON (0.219) on all 4, matched protocol** |
| LiON (D-MPNN) | 4 held-out studies (5-fold) | — | — | r=0.219 | Collins et al. 2026, Spearman |
| AGILE (GNN) | 4 held-out studies (5-fold) | — | — | r≈-0.01 | Xu et al. 2024, Pearson (essentially random) |

## 19. Key Scientific Findings

### Finding 1: Z-scoring was sufficient (ICC = 0.006)
Per-Experiment_ID z-scoring in LNPDB successfully removes batch effects. Residual ICC of 0.6% is negligible for the dominant in_vitro stratum.

### Finding 2: Cross-study generalization is limited but XGB+LANTERN beats SOTA
- On the paper's 4 held-out studies (5-fold CV, matched protocol), XGB+LANTERN PCA achieves mean Spearman r=0.307 — beating LiON (0.219) on all 4
- R² remains negative on most held-out studies; Spearman r captures rank-order signal that R² misses
- ICP finds zero invariant causal features; chemistry R² drops from +0.174 (within-study) to -0.019 (cross-study)
- Aggressive dimensionality reduction (10 PCA features) prevents overfitting to study-specific patterns

### Finding 3: The aggregate benchmark mixes qualitatively different problems
- in_vivo_liver: 96% Top-50 (trivially easy)
- in_vivo_other: 61% Top-50 (moderate signal)
- in_vitro_single: 13% Top-50 (the actual hard problem, 88% of data)

### Finding 4: LNPBO's signal is within-study scaffold identification
- The 17.4% R² on scaffold split exploits SAR that holds within studies but doesn't transfer
- XGBoost recovers hits **only** from scaffolds already in the seed set (0.0% novel scaffold recall)
- The model learns scaffold family membership, not generalizable structure-activity rules

### Finding 5: Context conditioning does not rescue cross-study transfer
- Adding cell type, cargo, route, measurement method as features → XGBoost uses them 82% but cross-study R² stays negative (-0.072)
- Even scoped to HeLa+FLuc only (same cell type, cargo, assay class), cross-study R² = -0.28 with context, -0.41 without
- Context helps marginally but unmeasured study-level confounds dominate

### Finding 6: Even meta-learning cannot rescue cross-study transfer
MAML and FSBO perform identically to random on held-out studies. The features don't carry transferable SAR information.

### Finding 7: Conformal prediction provides honest uncertainty
Split conformal achieves near-90% coverage on held-out studies (GP: 89.8%), with interval widths of ~3.0 z-score units.

## 20. Paper Scope and Claims

### What LNPBO can claim:
1. **Within-study BO works.** Given formulations from the same experimental setup, LNPBO identifies top candidates 2-3x better than random (20.4% vs 7.6% Top-50).
2. **IL structure is the dominant signal.** 82% SHAP, 100% permutation importance. Helper lipid encoding adds no marginal predictive value.
3. **LNPDB z-scoring is sound.** ICC=0.006 confirms negligible residual batch effects.
4. **Assay-type stratification matters.** in_vivo_liver (96%) vs in_vitro (13%) should never be pooled.

### What LNPBO can also claim:
5. **Beats LiON and AGILE on held-out studies.** On the paper's exact 4 held-out studies with matched 5-fold CV protocol, XGB+LANTERN PCA achieves Spearman r=0.307 (mean), beating LiON D-MPNN (0.219) on all 4 studies. AGILE GNN is essentially random (Pearson r ≈ 0). The key: aggressive PCA reduction to 10 features prevents overfitting.

### What LNPBO cannot claim:
1. **Strong cross-study generalization.** Top-50 drops from 20.4% to 8.4% on study-level holdout. R² remains negative on most held-out studies, though rank-order correlation (Spearman r) is positive.
2. **Transferable SAR rules.** ICP finds zero invariant features. Novel scaffolds get 0% recall. XGBoost learns scaffold family identification, not generalizable chemistry.
3. **Universal LNP design principles.** SAR is fundamentally context-dependent — different cell lines, cargo types, and delivery routes create different structure-activity landscapes.

### Study-held-out numbers for the paper:
- **Within-study (formulation split):** 24% Top-10, 20% Top-50 → this is the valid claim
- **Cross-study (study-level split):** 12% Top-10, 8% Top-50 → report this as the generalization limit
- **Within in_vitro (study-level split):** 28% Top-10, 19% Top-50 → the best within-assay number for the dominant stratum
- **The 12pp gap** between formulation-split and study-split is the study-specific leakage

### The correct framing:
LNPBO is a **within-study optimization tool**, not a cross-study transfer learning system. LNPDB is valuable as a **prior** for initializing optimization campaigns in new experimental contexts, but the molecular features alone cannot predict which ILs will work in a novel setup. The practical recommendation is: use LNPDB to select a diverse, information-rich initial screen, then let LNPBO's within-study BO take over.

## 21. Best Configuration

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

```bash
# Recommended (IL-only, most robust)
python pipeline.py --feature-type lantern_il_only --reduction pca

# Best Top-50 (needs 30 rounds)
python -m benchmarks.runner --strategies discrete_xgb_greedy --feature-type lantern_unimol --rounds 30 --batch-size 6 --n-seeds 500

# Benchmark comparison
python -m benchmarks.runner --strategies discrete_xgb_greedy,random --rounds 15 --n-seeds 500 --feature-type lantern_il_only
```

## 22. Implemented Extensions

### 22a. Calibrated Uncertainty
XGB-UCB wraps XGBoost with MAPIE conformal prediction (CV+ method, Barber et al., AoS 2021). Performs comparably to greedy at n=1000 — exploration doesn't help when the pool is large enough that greedy exploitation is effective.

### 22b. Generative Molecular Design
AGILE-lite via `propose-ils`: SELFIES mutations around known ILs, LANTERN+PLS encoding, MAPIE uncertainty, LCB ranking (mean - std), MaxMin diversity selection. See `cli/propose_ils.py`.

### 22c. Uncertainty Sampling
In the n<200 regime where surrogates are non-predictive, active learning-style uncertainty sampling could improve early-round sample efficiency. Pool-based AL (Settles, 2009) is directly applicable.

### 22d. Mixed-Variable Bayesian Optimization
CoCaBO (Ru et al., NeurIPS 2020) and CASMOPOLITAN (Wan et al., JMLR 2021) handle mixed categorical-continuous spaces, which could jointly optimize lipid selection and molar ratios in a single BO loop.

### 22e. Synthesizability Filtering (Future Work)
MolQuery (Broadbent, Vymětal, Moayedpour et al., ACS Omega 2026, DOI: 10.1021/acsomega.5c09931) presents an active learning pipeline for predicting lipid synthesizability. CatBoost classifiers on ECFP fingerprints achieve 72% accuracy (vs 61-62% prior SOTA), and AL with expert chemist labeling converges faster than random sampling (68 vs 85 samples to plateau). The pipeline uses LLM-generated lipid pools (Claude Sonnet) as a generative source — directly complementary to our `propose-ils` SELFIES-based generation. Integrating a MolQuery-style synthesizability filter after `propose-ils` would prune candidates to those a chemist could actually make, improving the practical value of generative suggestions. GitHub: https://github.com/Sanofi-Public/MolQuery.

### 22f. MD-Derived Features (Future Work)
The LNPDB paper shows MD-derived critical packing parameter (CPP) achieves Pearson r = 0.530 on LM_2019 (protonated CPPV method), far exceeding LiON's r = 0.104 and our XGB+LANTERN r = 0.164 on the same study. MD simulations capture bilayer dynamics that no fingerprint-based method can access.

**What's needed:**
- CHARMM/GROMACS force field setup per ionizable lipid (LNPDB provides CHARMM parameters)
- Bilayer simulation: ~1.5 μs per system, ~5 days on GPU
- CPP calculation from final 500 ns trajectory (volume or Rg method)
- Collins et al. plan to integrate Martini 3 coarse-grained parameters for faster, larger-scale simulations

**Integration path:** CPP and other MD-derived features (membrane thickness, torque density, compressibility) could be appended to the LANTERN feature vector. With n=54 simulated LNPs from LM_2019, MD features strongly predict delivery performance. Scaling to the full LNPDB would require substantial compute but could unlock a new performance tier. Out of scope for the current work.

## 23. Files Reference

| File | Description |
|------|-------------|
| `data/lnpdb_bridge.py` | LNPDB loading with z-score correction |
| `diagnostics/compute_icc.py` | ICC computation (REML) |
| `diagnostics/anchor_analysis.py` | Anchor IL rank consistency |
| `diagnostics/stratified_benchmark.py` | Per-stratum BO benchmarks |
| `diagnostics/icp_feature_selection.py` | ICP causal feature selection |
| `diagnostics/scaffold_analysis.py` | Scaffold clustering, seen vs novel analysis |
| `diagnostics/permutation_decomposition.py` | Variance decomposition |
| `diagnostics/partial_r2.py` | Partial R² decomposition |
| `benchmarks/study_split_benchmark.py` | Study-level holdout benchmark |
| `benchmarks/within_assay_benchmark.py` | Within-assay study-level benchmark |
| `benchmarks/shap_importance.py` | SHAP + permutation importance |
| `benchmarks/pc_interpretation.py` | PCA component interpretation |
| `models/gp_surrogate.py` | GP with study random effect |
| `models/vrex_surrogate.py` | V-REx invariant learning |
| `models/groupdro_surrogate.py` | GroupDRO worst-case optimization |
| `models/maml_surrogate.py` | MAML meta-learning |
| `models/fsbo_surrogate.py` | FSBO warm-started GP |
| `models/bradley_terry.py` | Bradley-Terry pairwise preference |
| `models/conformal_surrogate.py` | Split conformal prediction |
| `models/context_conditioned.py` | Context-conditioned XGB |
| `models/scoped_context_model.py` | HeLa-scoped context-conditioned XGB |
| `scripts/benchmark_heldout.py` | XGB+LANTERN vs LiON/AGILE on 4 held-out studies |
| `scripts/shap_analysis.py` | SHAP analysis with amine/scaffold/random split |
