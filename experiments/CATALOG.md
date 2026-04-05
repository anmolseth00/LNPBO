# Experiment Catalog

All experiments for "Benchmarking Optimization Strategies for Lipid Nanoparticle Design" (ACS JCIM).

Studies are defined in `experiments/data_integrity/studies.json` (27 entries,
25 non-pooled). Pooled studies (37661193, 38424061) are split by Model_target
into sub-studies (e.g., 37661193_liver, 37661193_spleen, 38424061_in_vitro,
38424061_multiorgan). The 25 non-pooled studies form the analysis set for all
ablation experiments. The within-study benchmark uses all 26 studies.

---

## Hardware Requirements

- **CPU**: 8+ cores recommended for parallel runs (Apple Silicon or x86-64)
- **RAM**: 8 GB minimum, 16 GB recommended (each process loads ~500 MB LNPDB into memory)
- **GPU**: Not required (all strategies run on CPU; BoTorch GP fitting uses CPU)
- **Storage**: ~2 GB for all result JSONs across all experiments

## Per-Strategy Timing Reference

Measured from encoding ablation logs on Apple M-series, study 39060305 (2,400
formulations, 1,800 oracle pool, batch 12, 15 rounds). Smaller studies run
proportionally faster.

| Strategy family | Time per run | Notes |
|-----------------|-------------|-------|
| Random | <0.1 s | No model fitting |
| RF (UCB, TS, TS-Batch) | 2--3 s | Fastest surrogate |
| GP sklearn | 0.3 s | Tiny study only; scales poorly |
| XGB (greedy, UCB, online conformal) | 6--20 s | |
| NGBoost UCB | 15--25 s | |
| Deep Ensemble | 4--23 s | |
| GP BoTorch standard (UCB, EI, LogEI, RKB) | 6--15 s | |
| GP BoTorch LP / TS-Batch | 6--7 s | |
| GP BoTorch compositional | 19--26 s | |
| GP BoTorch DKL | 33--40 s | |
| CASMOPolitan | 8--78 s | Depends on study size |
| GP BoTorch RF-Kernel | 115--505 s | Slow; random forest kernel |
| GP BoTorch GIBBON | ~340 s | Greedy info-gain batching |
| XGB CQR (conformal quantile) | 240--740 s | Many quantile regressions |
| XGB TS-Batch | 100--420 s | Many bootstrap resamples |

## Estimated Total Compute

All times assume serial (single-core) execution unless noted. Multiply by
~0.12--0.15 for 8-core parallel runs (overhead from data loading reduces
perfect scaling).

| Experiment | Runs | Avg/run | Serial | 4 cores |
|------------|------|---------|--------|---------|
| Within-study benchmark | 4,590 | ~51 s | ~65 h | ~16 h |
| Encoding ablation (5-strat) | 5,000 | ~18 s | ~25 h | ~6 h |
| Encoding ablation (full) | 34,000 | ~51 s | ~480 h | ~14 h |
| Batch size ablation | 2,400 | ~10 s | ~7 h | ~2 h |
| PCA dimensionality | 1,500 | ~10 s | ~4 h | ~1 h |
| Budget sensitivity | 1,200 | ~10 s | ~3 h | ~45 min |
| Warmup architecture | 3,198 | ~10 s | ~9 h | ~2 h |
| Kernel ablation | 675 | ~50 s | ~9 h | ~2 h |
| Kappa sensitivity | 2,500 | ~10 s | ~7 h | ~2 h |
| Kappa optimal (full) | 4,590 | ~51 s | ~65 h | ~16 h |
| All baselines (P&R + COMET + AGILE) | ~1,140 | ~5 s | ~1.5 h | ~25 min |
| Cross-study + calibration | -- | -- | ~30 min | ~30 min |
| Noise sensitivity | 300 | ~24 s | ~2 h | ~2 h |
| Hyperparam sensitivity | 420 | ~13 s | ~0.1 h | ~0.1 h |
| **Grand total** | **~61,490** | | **~678 h** | **~65 h** |

## Quick Reproduction (Smoke Test)

Run 3 studies x 5 strategies x 1 seed to verify the pipeline works:

```bash
python -m benchmarks.benchmark --pmids 39060305,37985700,26729861 \
  --strategies random,discrete_xgb_ucb,discrete_rf_ts,discrete_ngboost_ucb,lnpbo_logei \
  --seeds 42
```

Expected: ~15 minutes, 15 result files in `benchmark_results/within_study/`.

---

## 1. Within-Study Benchmark (Main Result)

- **Hypothesis**: Iterative Bayesian optimization outperforms random screening
  for LNP formulation design within individual studies. Tree-based surrogates
  (RF, XGBoost, NGBoost) match or exceed GP-BO on finite discrete pools.
- **Config**: `experiments/data_integrity/studies.json`
  - 25% seed pool, batch size 12, 15 rounds, LANTERN (count MFP + RDKit, PCA to 5 PCs)
  - Feature type adapts per study: `lantern_il_only` for IL-diverse, `ratios_only` for ratio-only
- **Run script**:
  ```
  python -m benchmarks.benchmark --studies-json experiments/data_integrity/studies.json --resume
  ```
- **Results**: `benchmark_results/within_study/{study_id}/`
- **Analysis**: `benchmarks/analyze_within_study.py`
- **Figures** (main text):
  - `fig_family_rankings.pdf` -- Strategy family rankings with bootstrap 95% CIs
  - `fig_critical_difference.pdf` -- Nemenyi post-hoc critical difference diagram
  - `fig_convergence.pdf` -- Top-5% recall by round and study type
  - `fig_study_type_stratified.pdf` -- Family performance stratified by study type
  - `fig_heatmap.pdf` -- Per-study strategy heatmap
  - `fig_win_matrix.pdf` -- Pairwise win matrix between families
  - `fig_variance.pdf` -- Variance decomposition (Study/Strategy/Seed)
- **Status**: 26 studies x 37 strategies x 5 seeds = **4,810 runs (IN PROGRESS)**
    (50 runs excluded: GP-Mixed strategies on 5 sub-studies lacking ratio features)
  - 38 strategies: random, 8 GP-BO (BoTorch), 6 specialized-kernel GP,
    4 compositional GP, 2 mixed-variable GP, 2 CASMOPolitan, 3 RF, 5 XGBoost,
    1 NGBoost, 1 Deep Ensemble, 1 GP (sklearn), 1 GIBBON, 1 PLS-GP,
    1 Ridge, 1 TabPFN
  - 5 seeds: 42, 123, 456, 789, 2024
- **Compute**: ~65 hours serial, ~8 hours on 8 cores. Dominated by slow
  strategies (XGB-CQR ~500 s/run, GIBBON ~340 s/run, RF-Kernel ~300 s/run).
  Fast strategies (RF, random, standard GP) finish in <15 s/run.

---

## 2. Encoding Ablation

- **Hypothesis**: Molecular encoding choice affects BO performance. Learned
  representations (AGILE, Uni-Mol, LiON, CheMeleon) may outperform
  handcrafted fingerprints (Morgan FP, LANTERN, mordred).

### 2a. Encoding Ablation (5-strategy, primary)

- **Config**: `experiments/ablations/encoding/config.json`
  - 8 encodings: LANTERN, MFP, count MFP, mordred, Uni-Mol, CheMeleon, LiON, AGILE
  - 5 strategies: RF-TS, XGB-UCB, NGBoost-UCB, CASMOPolitan-UCB, Random
  - 5 seeds: 42, 123, 456, 789, 2024
  - PCA to 5 PCs, batch 12, 15 rounds
- **Run script**:
  ```
  python -m experiments.run_ablation --config experiments/ablations/encoding/config.json --resume
  ```
- **Results**: `benchmark_results/ablations/encoding/{study_id}/`
- **Analysis**: `experiments/analysis/analyze_ablations.py --experiment encoding`
- **Figures**:
  - `fig_encoding.pdf` -- Main text: encoding comparison bar chart with 95% CIs
  - `fig_encoding_heatmap.pdf` -- Main text: encoding x strategy heatmap
  - `fig_si_encoding_heatmap.pdf` -- SI: detailed encoding heatmap
  - `experiments/analysis/figures/encoding/encoding_top5_recall.pdf` -- Standalone analysis figure
  - `experiments/analysis/figures/encoding/encoding_pairwise_significance.pdf` -- Pairwise tests
- **Status**: 25 studies x 8 encodings x 5 strategies x 5 seeds = **5,000 runs (COMPLETE)**
  - 200 result files per study
- **Compute**: ~25 hours serial, ~3 hours on 8 cores. Uses only fast
  strategies (RF-TS ~3 s, XGB-UCB ~15 s, NGBoost-UCB ~20 s,
  CASMOPolitan-UCB ~50 s, Random <0.1 s); average ~18 s/run.

### 2b. Encoding Ablation (full 34-strategy)

- **Config**: `experiments/ablations/encoding/config_full.json`
  - 8 encodings: LANTERN, MFP, count MFP, mordred, Uni-Mol, CheMeleon, LiON, AGILE
  - 34 strategies: all from within-study benchmark (random, 8 GP-BO, 6 specialized-kernel
    GP, 4 compositional GP, 2 CASMOPolitan, 3 RF, 5 XGBoost, 1 NGBoost,
    1 Deep Ensemble, 1 GP sklearn, 1 GIBBON, 1 PLS-GP)
  - 5 seeds: 42, 123, 456, 789, 2024
  - PCA to 5 PCs, batch 12, 15 rounds
- **Run script**:
  ```
  # Per-encoding (recommended, supports parallel launch):
  python -m experiments.run_ablation --config experiments/ablations/encoding/config_full.json \
      --condition lantern --resume
  # Or via overnight runner (4 concurrent, all encodings):
  ./run_overnight.sh encoding
  ```
- **Results**: `benchmark_results/ablations/encoding/{study_id}/` (same dir as 2a,
  files co-exist with different strategy prefixes)
- **Status**: 25 studies x 8 encodings x 34 strategies x 5 seeds = **34,000 runs (COMPLETE)**
  - 1,360 result files per study
- **Compute**: ~480 hours serial, ~14 hours on 4 cores. Dominated by slow
  strategies (RF-Kernel ~300 s, GIBBON ~340 s, XGB-CQR ~500 s). Fast
  strategies (RF, Random) finish in <3 s/run.

---

## 3. Batch Size Ablation

- **Hypothesis**: Batch size trades off exploration breadth vs. feedback
  frequency. Smaller batches (more rounds) may improve recall by enabling
  more model updates, but increase computational overhead.
- **Config**: `experiments/ablations/batch_size/config.json`
  (also: `experiments/ablations/batch_size/config.json`)
  - Batch sizes: 4, 8, 12, 24
  - Two variants: fixed-budget (total evals = 180, rounds adjusted) and
    fixed-rounds (15 rounds, total evals vary)
  - 4 strategies: RF-TS, XGB-UCB, NGBoost-UCB, Random
  - 3 seeds: 42, 123, 456
- **Run script**:
  ```
  python -m experiments.run_ablation --config experiments/ablations/batch_size/config.json --resume
  ```
- **Results**: `benchmark_results/ablations/batch_size/{study_id}/`
- **Analysis**: `experiments/analysis/analyze_ablations.py --experiment batch_size`
- **Figures**:
  - `fig_batch_sensitivity.pdf` -- Main text: batch size sensitivity panel
  - `fig_si_batch_size.pdf` -- SI: extended batch size results
  - `experiments/analysis/figures/batch_size/batch_size_top5_recall.pdf` -- Standalone analysis figure
  - `experiments/analysis/figures/batch_size/batch_size_pairwise_significance.pdf` -- Pairwise tests
- **Status**: 25 studies x 4 strategies x 8 conditions x 3 seeds = **2,400 runs (COMPLETE)**
  - 96 result files per study
- **Compute**: ~7 hours serial, ~1 hour on 8 cores. Average ~10 s/run
  (RF-TS ~3 s, XGB-UCB ~15 s, NGBoost-UCB ~20 s, Random <0.1 s).

---

## 4. PCA Dimensionality Ablation

- **Hypothesis**: Aggressive PCA reduction (5 PCs) may discard useful
  structural information. Higher dimensionality or raw features could help
  on structurally diverse studies.
- **Config**: `experiments/ablations/pca/config.json`
  (also: `experiments/ablations/pca/config.json`)
  - n_components: 3, 5, 10, 20, raw (no PCA)
  - 4 strategies: RF-TS, XGB-UCB, NGBoost-UCB, Random
  - 3 seeds: 42, 123, 456
  - LANTERN encoding, batch 12, 15 rounds
- **Run script**:
  ```
  python -m experiments.run_ablation --config experiments/ablations/pca/config.json --resume
  ```
- **Results**: `benchmark_results/ablations/pca/{study_id}/`
- **Analysis**: `experiments/analysis/analyze_ablations.py --experiment pca`
- **Figures**:
  - `fig_pca_justification.pdf` -- SI: PCA component justification + BO performance
  - `fig_si_pca_components.pdf` -- SI: cumulative explained variance curves
  - `experiments/analysis/figures/pca/pca_top5_recall.pdf` -- Standalone analysis figure
  - `experiments/analysis/figures/pca/pca_pairwise_significance.pdf` -- Pairwise tests
- **Status**: 25 studies x 4 strategies x 5 conditions x 3 seeds = **1,500 runs (COMPLETE)**
  - 60 result files per study
- **Compute**: ~4 hours serial, ~30 minutes on 8 cores. Same fast strategy
  mix as batch size ablation; average ~10 s/run.

---

## 5. Budget Sensitivity Ablation

- **Hypothesis**: Strategy rankings may shift under tighter or more generous
  experimental budgets. Simpler surrogates (RF) may be more data-efficient
  under tight budgets.
- **Config**: `experiments/ablations/budget/config.json`
  (also: `experiments/ablations/budget/config.json`)
  - 4 budgets: tight (10% seed, 5 rounds), moderate (15%, 10), standard (25%, 15), generous (40%, 15)
  - 4 strategies: RF-TS, XGB-UCB, NGBoost-UCB, Random
  - 3 seeds: 42, 123, 456
  - LANTERN encoding, batch 12
- **Run script**:
  ```
  python -m experiments.run_ablation --config experiments/ablations/budget/config.json --resume
  ```
- **Results**: `benchmark_results/ablations/budget/{study_id}/`
- **Analysis**: `experiments/analysis/analyze_ablations.py --experiment budget`
- **Figures**:
  - `fig_ablation_summary.pdf` -- Main text: multi-panel ablation summary (includes budget)
  - `experiments/analysis/figures/budget/budget_top5_recall.pdf` -- Standalone analysis figure
  - `experiments/analysis/figures/budget/budget_pairwise_significance.pdf` -- Pairwise tests
- **Status**: 25 studies x 4 strategies x 4 conditions x 3 seeds = **1,200 runs (COMPLETE)**
  - 48 result files per study
- **Compute**: ~3 hours serial, ~25 minutes on 8 cores. Average ~10 s/run.
  Tight budget (5 rounds) runs finish faster than standard (15 rounds).

---

## 6. Warmup Architecture Ablation

- **Hypothesis**: A large initial screen (warmup) followed by small BO batches
  may outperform the standard 25% random seed design. Pessimistic warmup
  (bottom 75% selection) tests whether starting from bad data helps BO.
- **Config**: `experiments/ablations/warmup/config.json`
  (also: `experiments/ablations/warmup/config.json`)
  - 11 conditions: baseline (25% random seed) + 10 warmup configurations
    - Warmup sizes: 48, 96, 144 (smaller studies skip large warmups)
    - Selection: random vs. bottom-75%
    - BO batch: 6, 12, 24
  - 4 strategies: RF-TS, XGB-UCB, NGBoost-UCB, Random
  - 3 seeds: 42, 123, 456
- **Run script**:
  ```
  python -m experiments.run_ablation --config experiments/ablations/warmup/config.json --resume
  ```
- **Results**: `benchmark_results/ablations/warmup/{study_id}/`
- **Analysis**: `experiments/analysis/analyze_ablations.py --experiment warmup`
- **Figures**:
  - `fig_ablation_summary.pdf` -- Main text: multi-panel ablation summary (includes warmup)
  - `experiments/analysis/figures/warmup/warmup_top5_recall.pdf` -- Standalone analysis figure
  - `experiments/analysis/figures/warmup/warmup_pairwise_significance.pdf` -- Pairwise tests
- **Status**: 25 studies x variable conditions (some studies skip large warmups) = **3,198 runs (COMPLETE)**
  - 108--132 result files per study (varies by study size)
- **Compute**: ~9 hours serial, ~1 hour on 8 cores. Average ~10 s/run.
  Same fast strategy mix; variable round counts per warmup condition.

---

## 7. Kernel Ablation

- **Hypothesis**: Domain-specific GP kernels (Tanimoto for fingerprints,
  Aitchison for simplex ratios) may outperform the generic Matern-5/2 kernel.
  Deep Kernel Learning (DKL) may capture nonlinear feature interactions.
- **Config**: `experiments/ablations/kernel/config.json` (master config)
  - Per-kernel configs: `config_matern.json`, `config_tanimoto.json`,
    `config_aitchison.json`, `config_dkl.json`
  - 6 kernels via 9 strategies: Matern (TS-Batch, LogEI), Tanimoto (TS, LogEI),
    Aitchison (TS, LogEI), DKL (TS, LogEI), + Random
  - 3 seeds: 42, 123, 456
  - Feature types vary by kernel: LANTERN+PCA for Matern/DKL, raw count MFP
    for Tanimoto, ratios_only for Aitchison
- **Run script**:
  ```
  # Run all kernels (master config):
  python -m experiments.run_ablation --config experiments/ablations/kernel/config.json --resume
  # Or per-kernel:
  python -m experiments.run_ablation --config experiments/ablations/kernel/config_tanimoto.json --resume
  ```
- **Results**: `benchmark_results/ablations/kernel/{study_id}/`
- **Analysis**: `paper/gen_fig_kernel_comparison.py`
- **Figures**:
  - `fig_kernel_comparison.pdf` -- Main text + SI: kernel comparison panel
- **Status**: 25 studies x 9 strategies x 3 seeds = **675 runs (COMPLETE)**
  - 27 result files per study
- **Compute**: ~9 hours serial, ~1 hour on 8 cores. All GP strategies;
  average ~50 s/run. DKL (~35 s) and Aitchison (~10 s) are fast; Tanimoto
  on raw fingerprints may be slower on large studies.

---

## 8. Kappa Sensitivity (UCB Exploration Weight)

- **Hypothesis**: The default UCB exploration parameter kappa = 5.0 may be
  suboptimal for finite discrete pools. Lower kappa (more exploitation) or
  higher kappa (more exploration) could shift performance.
- **Config**: `experiments/ablations/kappa/config.json`
  - 5 kappa values: 0.5, 1.0, 2.0, 5.0, 10.0
  - 4 strategies: NGBoost-UCB, RF-UCB, XGB-UCB, Random
  - 5 seeds: 42, 123, 456, 789, 2024
  - LANTERN encoding, PCA 5, batch 12, 15 rounds
- **Run script**:
  ```
  python -m experiments.run_ablation --config experiments/ablations/kappa/config.json --resume
  ```
- **Results**: `benchmark_results/ablations/kappa/{study_id}/`
- **Analysis**: `paper/make_all_figures.py` (fig_kappa_sensitivity)
- **Figures**:
  - `fig_kappa_sensitivity.pdf` -- SI: kappa sensitivity curves
- **Status**: 25 studies x 4 strategies x 5 kappa values x 5 seeds = **2,500 runs (COMPLETE)**
  - 100 result files per study
- **Compute**: ~7 hours serial, ~1 hour on 8 cores. Average ~10 s/run.
  Uses same fast tree strategies (RF-UCB, XGB-UCB, NGBoost-UCB, Random).

### 8b. Full Benchmark at Optimal Kappa (kappa = 0.5)

- **Hypothesis**: The kappa sensitivity sweep (8a) identified kappa = 0.5 as
  optimal. Rerunning the full 34-strategy benchmark at kappa = 0.5 tests
  whether strategy family rankings change under the optimal exploration weight.
- **Config**: Same as within-study benchmark (Section 1) but with kappa = 0.5
  for all UCB-based strategies. Non-UCB strategies (EI, LogEI, TS, greedy)
  are invariant to kappa by construction.
- **Run script**:
  ```
  python -m benchmarks.benchmark --studies-json experiments/data_integrity/studies.json \
      --kappa 0.5 --output-dir benchmark_results/ablations/kappa_optimal --resume
  ```
- **Results**: `benchmark_results/ablations/kappa_optimal/{study_id}/`
- **Analysis**: `scripts/analyze_kappa_optimal.py`
- **Key findings**:
  - Family rankings are stable (Spearman rho = 0.79, p = 0.036)
  - UCB-based strategies improve: GP-UCB +0.070, Deep Ensemble +0.047, RF-UCB +0.030
  - Non-UCB strategies invariant (|delta| < 0.005)
  - Deep Ensemble rises from 6th to 3rd due to its UCB acquisition
  - Core conclusion unchanged: tree surrogates outperform GP-BO regardless of kappa
  - Tanimoto kernel strategies failed (0 rounds) due to feature type mismatch
    in ablation config (lantern_il_only instead of raw count_mfp)
- **Status**: 26 studies x 34 strategies x 5 seeds = **4,420 runs (NOT STARTED)**
  - 170 result files per study
  - 126 broken runs (Tanimoto: 98, DKL: 28) due to feature config mismatch
- **Compute**: ~65 hours serial, ~8 hours on 8 cores. Same strategy mix as
  within-study benchmark.

---

## 9. External Baselines (P&R, COMET, AGILE)

Non-BO comparison methods. Each is a single-shot prediction baseline.

### 9a. Predict-and-Rank (P&R)

- **Hypothesis**: Single-shot surrogate ranking (train on 25% seed, rank
  oracle pool by predicted score) may be competitive with iterative BO,
  suggesting the optimization loop provides little additional value.
- **Run script**: `python -m benchmarks.baselines.predict_and_rank --resume`
- **Results**: `benchmark_results/baselines/predict_and_rank/{study_id}/`
- **Strategies**: predict_rank_xgb, predict_rank_rf, predict_rank_ngboost
- **Status**: 26 studies x 3 surrogates x 5 seeds = 390 expected; **391 files present (COMPLETE)**
  (extra files from pooled sub-studies)
- **Compute**: ~30 minutes serial. Single-shot ranking (no iterative loop),
  ~5 s per run.
- **Analysis**: `benchmarks/analyze_bo_vs_pr.py`
- **Figures**:
  - `fig_bo_vs_pr_analysis.pdf` -- Main text: BO vs P&R comparison
  - `fig_external_baselines.pdf` -- Main text: all baselines comparison
  - `fig_diversity.pdf` -- Main text: formulation diversity analysis

### 9b. COMET Zero-Shot Transfer

- **Hypothesis**: A pretrained LNP property predictor (COMET, trained on
  LANCE dataset) can rank formulations without any within-study training.
  Expected to underperform within-study surrogates due to domain shift.
- **Run scripts**:
  ```
  python -m benchmarks.baselines.comet_wrapper --export
  # (run COMET inference in COMET venv)
  python -m benchmarks.baselines.comet_baseline
  ```
- **Results**: `benchmark_results/baselines/comet/{study_id}/`
- **Status**: 25 studies x 5 seeds = 125 zero-shot result files + smi2mol mapping files + other = **401 total files (COMPLETE)**
- **Compute**: ~15 minutes total. COMET inference ~30 s per study (zero-shot,
  no iterative loop). Requires separate COMET conda environment.
- **Figures**: Included in `fig_external_baselines.pdf`

### 9c. AGILE Predictor

- **Hypothesis**: AGILE GNN embeddings (frozen, from pretrained model) used
  as features for XGB/RF surrogates may match or exceed LANTERN.
- **Run script**: `python -m benchmarks.baselines.agile_predictor --resume`
- **Results**: `benchmark_results/baselines/agile_predictor/{study_id}/`
- **Strategies**: agile_xgb, agile_rf
- **Status**: 26 studies x 2 surrogates x 5 seeds = 260 expected (IN PROGRESS)
  (some studies lack AGILE embeddings)
- **Compute**: ~20 minutes serial. Uses frozen AGILE embeddings (precomputed
  in `data/agile_embeddings.npz`); XGB/RF fitting ~3 s per run.
- **Figures**: Included in `fig_external_baselines.pdf`

### 9d. AGILE Fine-Tuned MLP

- **Hypothesis**: Fine-tuning an MLP head on frozen AGILE embeddings for
  each study improves prediction over the generic AGILE predictor.
- **Run script**: `python -m benchmarks.baselines.agile_baseline --mode mlp --resume`
- **Results**: `benchmark_results/baselines/agile_finetuned/{study_id}/`
- **Status**: 25 studies x 1 strategy x 5 seeds = 125 expected; **117 files present (COMPLETE)**
  (some studies lack AGILE embeddings)
- **Compute**: ~25 minutes serial. MLP fine-tuning ~60 s per study (trains
  a small MLP head on frozen AGILE embeddings for each seed).
- **Figures**: Included in `fig_external_baselines.pdf`

---

## 10. Cross-Study Transfer

- **Hypothesis**: Cross-study transfer (leave-one-study-out) is expected to
  perform near random because SAR is study-dependent (different assays, cell
  types, readouts). This is a negative result that motivates within-study BO.
- **Run script**: `python -m experiments.cross_study_transfer`
- **Results**: `benchmark_results/cross_study_transfer/`
  - `cross_study_transfer_results.json` -- Warm-start (25% seed + transfer)
  - `cross_study_transfer_results_cold.json` -- Cold-start (zero-shot transfer)
  - `cross_study_transfer_summary.csv` -- Warm-start summary table
  - `cross_study_transfer_summary_cold.csv` -- Cold-start summary table
- **Analysis**: Embedded in `experiments/cross_study_transfer.py`
- **Figures**:
  - `fig_cross_study.pdf` -- Main text: cross-study transfer results
- **Status**: **COMPLETE** (both warm-start and cold-start variants)
- **Compute**: ~20 minutes. Leave-one-study-out XGB fitting (25 train/test
  splits); no iterative BO loop.

---

## 11. Calibration Analysis

- **Hypothesis**: RF provides well-calibrated uncertainty estimates while GP
  surrogates are overconfident. Poor calibration explains why UCB degenerates
  toward greedy selection for some surrogates.
- **Run script**: `python -m experiments.calibration_analysis`
- **Results**: `benchmark_results/calibration_analysis.json`
- **Figures**:
  - `fig_calibration.pdf` -- Main text: reliability diagram
- **Status**: **COMPLETE**
- **Compute**: ~10 minutes. Fits surrogates on within-study benchmark seed
  pools and evaluates predicted interval coverage.

---

## 12. Supplementary Analyses

### 12a. BO vs Predict-and-Rank

- **Analysis**: `benchmarks/analyze_bo_vs_pr.py`
- **Results**: `benchmark_results/bo_vs_pr_analysis.json`
- **Figures**: `fig_bo_vs_pr_analysis.pdf`

### 12b. Formulation Diversity

- **Analysis**: `benchmarks/analyze_diversity.py`
- **Results**: `benchmark_results/diversity_analysis.json`
- **Figures**: `fig_diversity.pdf`

### 12c. Study Predictors

- **Analysis**: `benchmarks/analyze_study_predictors.py`
- **Results**: `benchmark_results/study_predictor_correlations.json`
- **Figures**: `fig_study_predictors.pdf`

### 12d. Data Landscape

- **Run script**: `python -m experiments.data_integrity.data_landscape`
- **Results**: `experiments/data_integrity/data_landscape.json`
- **Figures**:
  - `fig_data_landscape.pdf` -- Main text: LNPDB data landscape panel
  - `experiments/data_integrity/figures/component_diversity.pdf`
  - `experiments/data_integrity/figures/cargo_distribution.pdf`
  - `experiments/data_integrity/figures/il_usage_histogram.pdf`
  - `experiments/data_integrity/figures/model_target_distribution.pdf`
  - `experiments/data_integrity/figures/study_types.pdf`

### 12e. Study Stratification Audit

- **Run script**: `python -m experiments.data_integrity.audit_stratification`
- **Results**: `experiments/data_integrity/stratification_report.json`,
  `experiments/data_integrity/stratification_report.md`
- **Output**: `experiments/data_integrity/studies.json`,
  `experiments/data_integrity/studies_with_ids.json`

### 12f. Gap Analysis

- **Analysis**: `benchmarks/gap_analysis.py`
- **Results**: `benchmark_results/analysis/within_study/gap_analysis/`
  - 7 hypothesis-driven figures: convergence, study size, batch strategy,
    exploitation, variance, study-conditional, composite summary
- **Status**: COMPLETE

### 12g. GIBBON Analysis

- **Analysis**: Exploratory analysis of GIBBON (greedy information-theoretic batching)
- **Results**: `benchmark_results/analysis/within_study/gibbon_analysis/`
  - gibbon_convergence.png, gibbon_early_vs_late.png,
    gibbon_rank_per_study.png, gibbon_vs_tsbatch.png
- **Status**: COMPLETE (exploratory, not in main text)

### 12h. TS-Batch Analysis

- **Analysis**: Exploratory analysis of Thompson sampling batch strategies
- **Results**: `benchmark_results/analysis/within_study/tsbatch_analysis/`
  - convergence_comparison.png, diversity_proxy.png,
    round_best_quality.png, tsbatch_summary.png
- **Status**: COMPLETE (exploratory, not in main text)

---

## 13. Sensitivity Analyses

Robustness checks verifying that conclusions are stable under perturbation.

### 13a. Threshold Sensitivity

- **Hypothesis**: Strategy family rankings are stable across different
  top-k% recall thresholds (5%, 10%, 20%).
- **Run script**: `python -m benchmarks.threshold_sensitivity`
- **Results**: `benchmark_results/analysis/within_study/sensitivity/threshold_sensitivity.json`
- **Key finding**: Rankings are mostly stable (Spearman rho 0.76--0.91).
  CASMOPolitan leads at top-10%/20%; NGBoost at top-5%.
- **Status**: COMPLETE

### 13b. Sub-Study Sensitivity

- **Hypothesis**: Conclusions are robust when excluding the 4 multi-organ
  sub-studies (37661193_liver, 37661193_spleen, 38424061_in_vitro,
  38424061_multiorgan) and their parent pooled studies.
- **Run script**: `python -m benchmarks.substudy_sensitivity`
- **Results**: `benchmark_results/analysis/within_study/sensitivity/substudy_sensitivity.json`
- **Key finding**: Conclusions unchanged (max recall delta 0.019, all
  rank swaps within overlapping CIs). Cluster-robust SEs show ratio
  1.01--1.20 (mild positive intra-cluster correlation).
- **Status**: COMPLETE

### 13c. Noise Sensitivity

- **Hypothesis**: Strategy rankings are robust under measurement noise
  injected into oracle responses.
- **Run script**: `python -m benchmarks.noise_sensitivity --resume`
- **Design**: 5 studies x 5 strategies x 4 noise levels (sigma=0, 0.1,
  0.3, 0.5) x 3 seeds = 300 runs. Noise is N(0, sigma) added to oracle
  z-scores; seed data stays clean.
- **Results**:
  - Per-run: `benchmark_results/analysis/within_study/sensitivity/noise_runs/{study_id}/{strategy}_sigma{sigma}_s{seed}.json`
  - Summary: `benchmark_results/analysis/within_study/sensitivity/noise_sensitivity.json`
- **Status**: **300 runs (COMPLETE)**
- **Compute**: ~2 hours serial. GP strategy (lnpbo_logei) dominates runtime
  (~15 s/run on large studies); tree strategies ~3--20 s/run.

### 13d. Hyperparameter Sensitivity

- **Hypothesis**: The tree-ensemble advantage in the within-study benchmark
  reflects genuinely better surrogates, not favorable default hyperparameters.
  Strategy family rankings should be stable across a grid of key hyperparameters.
- **Run script**: `python -m benchmarks.hyperparam_sensitivity --resume`
- **Design**: 3 surrogates (RF-TS, XGB-UCB, NGBoost-UCB) x HP grids
  (RF: 12 configs, XGB: 12 configs, NGBoost: 4 configs) x 5 studies x 3 seeds
  = 420 runs.
  - RF: n_estimators=[50,100,200,500] x max_depth=[5,10,None]
  - XGB: n_estimators=[50,100,200,500] x max_depth=[3,6,10]
  - NGBoost: n_estimators=[50,100,200,500]
- **Results**:
  - Per-run: `benchmark_results/analysis/within_study/sensitivity/hyperparam_runs/{study_id}/{strategy}_{hp_tag}_s{seed}.json`
  - Summary: `benchmark_results/analysis/within_study/sensitivity/hyperparam_sensitivity.json`
- **Key finding**: Surrogate family ordering is preserved across all HP configs.
  Worst RF (0.647) > best NGBoost (0.628) > best XGB (0.598) > random (0.545).
  RF spread = 0.072, XGB spread = 0.035, NGBoost spread = 0.012. The advantage
  is not an artifact of default hyperparameter selection.
- **Status**: **420 runs (COMPLETE)**
- **Compute**: ~30 minutes serial. Tree strategies only (RF ~3 s, XGB ~15 s,
  NGBoost ~20 s per run).

---

## 14. Small Study Benchmark (N=25-199)

- **Hypothesis**: BO strategies remain effective on small LNPDB studies
  excluded from the main benchmark (MIN_STUDY_SIZE=200). Tests whether
  the performance patterns observed on large studies (N=200-1800)
  generalize to studies with as few as 25 formulations.
- **Design**: 28 studies with N=25-199. Batch size 5 (Evan's proposed
  minimum), 25% seed fraction (min 5), rounds computed dynamically per
  study so BO explores at most 50% of the oracle pool. 4 strategies
  (RF-TS, XGB-UCB, NGBoost-UCB, random) x 5 seeds x 28 studies = 560 runs.
- **Config**: `experiments/ablations/small_study/config.json`
- **Study definitions**: `experiments/ablations/small_study/studies_small.json`
- **Run script**:
  ```
  python -m experiments.run_ablation \
    --config experiments/ablations/small_study/config.json \
    --studies-json experiments/ablations/small_study/studies_small.json \
    --resume
  ```
- **Results**: `benchmark_results/ablations/small_study_benchmark/`
- **Status**: **NOT STARTED**
- **Compute**: ~560 runs x ~5 s avg = ~45 min serial, ~12 min on 4 cores.
  Small studies are fast.

---

## Figure-to-Experiment Mapping

All paper figures are generated by `paper/make_all_figures.py` unless noted.

### Main Text Figures

| Figure | File | Data Source |
|--------|------|-------------|
| TOC Graphic | `fig_toc_graphic.pdf` | Static artwork |
| Strategy Rankings (38) | `fig_strategy_rankings.pdf` | Within-study benchmark |
| Per-Study Heatmap | `fig_heatmap.pdf` | Within-study benchmark |
| Encoding Comparison | `fig_encoding.pdf` | Encoding ablation |
| Convergence Curves | `fig_convergence.pdf` | Within-study benchmark |

### Supporting Information Figures

| Figure | File | Data Source |
|--------|------|-------------|
| Family Rankings | `fig_family_rankings.pdf` | Within-study benchmark |
| Critical Difference | `fig_critical_difference.pdf` | Within-study benchmark |
| Study-Type Stratified | `fig_study_type_stratified.pdf` | Within-study benchmark |
| Pairwise Win Matrix | `fig_win_matrix.pdf` | Within-study benchmark |
| Convergence Z-Score | `fig_convergence_zscore.pdf` | Within-study benchmark |
| Acquisition Breakdown | `fig_acquisition_breakdown.pdf` | Within-study benchmark |
| Data Landscape | `fig_data_landscape.pdf` | Data integrity analysis |
| Variance Decomposition | `fig_variance.pdf` | Within-study benchmark |
| PCA Justification | `fig_pca_justification.pdf` | PCA ablation |
| PCA Components | `fig_si_pca_components.pdf` | PCA ablation |
| Batch Sensitivity | `fig_batch_sensitivity.pdf` | Batch size ablation |
| Batch Size (extended) | `fig_si_batch_size.pdf` | Batch size ablation |
| Kappa Sensitivity | `fig_kappa_sensitivity.pdf` | Kappa ablation |
| Kernel Comparison | `fig_kernel_comparison.pdf` | Kernel ablation |
| Ablation Summary | `fig_ablation_summary.pdf` | Budget + warmup + batch + PCA |
| Encoding Heatmap | `fig_encoding_heatmap.pdf` | Encoding ablation |
| Encoding Heatmap (SI) | `fig_si_encoding_heatmap.pdf` | Encoding ablation |
| Acquisition (SI) | `fig_si_acquisition.pdf` | Within-study benchmark |
| Cross-Study Transfer | `fig_cross_study.pdf` | Cross-study transfer |
| BO vs P&R | `fig_bo_vs_pr_analysis.pdf` | `bo_vs_pr_analysis.json` |
| External Baselines | `fig_external_baselines.pdf` | Within-study + baselines |
| Baselines Grouped | `fig_baselines.pdf` | Within-study + baselines |
| Calibration | `fig_calibration.pdf` | `calibration_analysis.json` |
| Diversity | `fig_diversity.pdf` | `diversity_analysis.json` |
| Study Predictors | `fig_study_predictors.pdf` | `study_predictor_correlations.json` |

All 30 figures present in `paper/figures/`. Regenerate with `python paper/make_all_figures.py`.

---

## Result File Counts Summary

| Experiment | Studies | Conditions | Strategies | Seeds | Total Files | Status |
|------------|---------|------------|------------|-------|-------------|--------|
| Within-study benchmark | 27 | 1 | 34 | 5 | 4,590 | COMPLETE |
| Encoding ablation (5-strat) | 25 | 8 encodings | 5 | 5 | 5,000 | COMPLETE |
| Encoding ablation (full) | 25 | 8 encodings | 34 | 5 | 34,000 | COMPLETE |
| Batch size ablation | 25 | 8 (4 sizes x 2 variants) | 4 | 3 | 2,400 | COMPLETE |
| Budget sensitivity | 25 | 4 budgets | 4 | 3 | 1,200 | COMPLETE |
| PCA dimensionality | 25 | 5 (3/5/10/20/raw) | 4 | 3 | 1,500 | COMPLETE |
| Warmup architecture | 25 | 11 (variable per study) | 4 | 3 | 3,198 | COMPLETE |
| Kappa sensitivity | 25 | 5 kappa values | 4 | 5 | 2,500 | COMPLETE |
| Kappa optimal (full) | 27 | 1 (kappa=0.5) | 34 | 5 | 4,590 | COMPLETE |
| Kernel ablation | 25 | 1 (4 kernels) | 9 | 3 | 675 | COMPLETE |
| Predict-and-Rank | 27 | 1 | 3 | 5 | 407 | COMPLETE |
| COMET zero-shot | 25 | 1 | 1 | 5 | 401 | COMPLETE |
| AGILE predictor | 27 | 1 | 2 | 5 | 252 | COMPLETE |
| AGILE fine-tuned | 25 | 1 | 1 | 5 | 117 | COMPLETE |
| Cross-study transfer | 25 | 2 (warm/cold) | 1 | -- | 4 files | COMPLETE |
| Calibration analysis | -- | -- | -- | -- | 1 file | COMPLETE |
| Noise sensitivity | 5 | 4 sigma levels | 5 | 3 | 300 | COMPLETE |
| Hyperparam sensitivity | 5 | 28 HP configs | 3 | 3 | 420 | COMPLETE |
| **Total** | | | | | **~61,518** | |

---

## Regeneration Commands

```bash
# Run all figures:
cd paper && python make_all_figures.py

# Run specific analysis scripts:
python -m benchmarks.analyze_within_study
python -m benchmarks.analyze_bo_vs_pr
python -m benchmarks.analyze_diversity
python -m benchmarks.analyze_study_predictors
python -m experiments.analysis.analyze_ablations --experiment all
python -m experiments.calibration_analysis
python -m experiments.cross_study_transfer
python -m experiments.data_integrity.data_landscape

# Sensitivity analyses:
python -m benchmarks.threshold_sensitivity
python -m benchmarks.substudy_sensitivity
python -m benchmarks.noise_sensitivity --resume
python -m benchmarks.hyperparam_sensitivity --resume

# Full encoding ablation (34 strategies, long-running):
python -m experiments.run_ablation --config experiments/ablations/encoding/config_full.json --resume

# Overnight runner (encoding full + noise, 4 concurrent, memory-safe):
./run_overnight.sh           # both
./run_overnight.sh encoding  # encoding only
./run_overnight.sh noise     # noise only

# Or use the Makefile:
make figures        # Regenerate all publication figures
make experiments    # Re-run all experiments (WARNING: 50+ hours)
make reproduce      # Both + compile LaTeX
```

---

## Archived Scripts

Exploratory or superseded scripts preserved for reference.

### `benchmarks/_archive/` (11 files)

| File | Original purpose |
|------|-----------------|
| `analyze_encoding_ablation.py` | Early encoding analysis (superseded by `experiments/analysis/analyze_ablations.py`) |
| `analyze_tsbatch.py` | TS-batch deep-dive (superseded by `benchmarks/analyze_within_study.py`) |
| `continuous_bo_benchmark.py` | Continuous BO benchmark (superseded by discrete pool approach) |
| `feature_importance.py` | SHAP feature importance (superseded by `scripts/shap_analysis.py`) |
| `figures.py` | Early figure generation (superseded by `paper/make_all_figures.py`) |
| `gibbon_analysis.py` | GIBBON deep-dive (results in `benchmark_results/analysis/within_study/gibbon_analysis/`) |
| `pc_interpretation.py` | PCA component interpretation |
| `shap_importance.py` | SHAP importance (superseded) |
| `study_split_benchmark.py` | Cross-study split benchmark (superseded by `experiments/cross_study_transfer.py`) |
| `within_assay_benchmark.py` | Within-assay benchmark (merged into main benchmark) |
| `README.md` | Archive readme |

### `experiments/analysis/_archive/` (2 files)

| File | Original purpose |
|------|-----------------|
| `evaluate_encodings.py` | Early encoding evaluation (superseded by `analyze_ablations.py`) |
| `generate_paper_figures.py` | Early figure generation (superseded by `paper/make_all_figures.py`) |
