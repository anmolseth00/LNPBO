# Benchmark Study Design

This page explains the retrospective simulation protocol used to evaluate optimization strategies across LNPDB studies.

## Overview

Every formulation in each LNPDB study has already been tested, so we know the true experimental outcomes. The benchmark hides those outcomes, reveals a random subset as a starting set, and then lets each strategy iteratively choose which formulations to "test" next. After a fixed number of rounds, we measure how many of the study's true top performers the strategy managed to find within its budget.

## Simulation protocol

For each study:

1. **Seed pool (round 0):** 25% of formulations are revealed at random as the initial training set. The model sees both features (molecular structure, molar ratios) and outcomes (`Experiment_value`) for these formulations.

2. **Oracle pool:** The remaining 75% of formulations form the candidate pool. The model can see their features but not their outcomes.

3. **Iterative acquisition (rounds 1-N):** Each round, the strategy selects a batch of 12 formulations from the oracle pool. Their true `Experiment_value` is revealed, they join the training set, and the surrogate model is retrained. This repeats for up to 15 rounds.

4. **Evaluation:** After all rounds, the metric is **top-k recall** — what fraction of the study's true top 5% (or 10%, 20%) formulations did the strategy find within its total budget? Total formulations explored per strategy: ~40-50% of the library.

5. **Replication:** Each (strategy, study) pair is run with 5 random seeds (controlling the initial seed pool). Results are averaged across seeds to reduce variance from lucky/unlucky starting sets.

## What is a "strategy"?

A strategy is the combination of:

- **Surrogate model** — the ML model that predicts formulation performance from features. Options include Random Forest, XGBoost, NGBoost (tree-based), Gaussian Process (BoTorch/GPyTorch), Deep Ensemble, Ridge regression, and others.

- **Acquisition function** — the rule for selecting the next batch. UCB (upper confidence bound) picks formulations the model is both confident and optimistic about. Thompson Sampling samples from the model's posterior uncertainty. EI/LogEI pick formulations with high expected improvement over the current best.

- **Batch strategy** — how to select multiple formulations per round. Kriging Believer hallucinates outcomes for selected points. Local Penalization discourages nearby selections. Thompson Sampling draws independent posterior samples.

The benchmark evaluates 37 non-random strategies plus a random baseline (38 total configurations).

## Handling variable study sizes

Each LNPDB study has a different number of formulations (26 to 1,801 in the main benchmark). The design adapts:

- **Seed pool:** Always 25% of the study's library (minimum 30 formulations), so the fraction is constant across studies.
- **Top-k thresholds:** Always relative to the study — "top 5%" means the best 5% of that study, not a fixed number.
- **Round cap:** Acquisition is capped so BO explores at most 50% of the oracle pool. Smaller studies get fewer rounds.
- **Aggregation:** Final results are averaged across all 26 studies, so no single large study dominates. Each study contributes one data point (its mean across 5 seeds).

## Confidence intervals

Error bars are bootstrap confidence intervals (95%) computed by resampling the 26 study-level means. Each study contributes one value (the strategy's mean top-k recall across 5 seeds for that study). This gives a confidence interval for "how would this strategy perform on a new, unseen LNP study?"

Statistical comparisons use the Wilcoxon signed-rank test (paired by study) with Benjamini-Hochberg FDR correction for multiple comparisons. Effect sizes are reported as Cohen's d.

## Molecular features

Lipid structures (SMILES) are converted to numeric features via molecular encodings. The default encoding is LANTERN: count Morgan fingerprints + RDKit 2D descriptors, compressed to 5 principal components per lipid role via PCA. Molar ratios and the IL:mRNA mass ratio are appended as raw features.

Studies where only one ionizable lipid is used (ratio-only optimization) skip molecular encoding entirely — only molar ratios are used as features.

## Key design choices

| Parameter | Default | Rationale |
|-----------|---------|-----------|
| Seed fraction | 25% | Balances initial information vs exploration budget |
| Batch size | 12 | Near-optimal in ablation studies; matches common lab plate sizes |
| Max rounds | 15 | Capped so BO explores at most 50% of oracle pool |
| Encoding | LANTERN | Best overall in encoding ablation (count MFP + RDKit, PCA to 5 PCs) |
| Normalization | Copula | Gaussian copula transform of targets before surrogate fitting |
| Seeds | 5 (42, 123, 456, 789, 2024) | Enough for stable means; more would increase compute linearly |

All design parameters have ablation studies testing sensitivity. See `experiments/CATALOG.md` for the full list.
