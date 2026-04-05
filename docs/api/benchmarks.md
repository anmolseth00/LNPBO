# Benchmarks API

The benchmarks module provides a statistical evaluation framework for comparing optimization strategies on LNPDB studies. It includes the benchmark runner, strategy configuration registry, and a suite of statistical utilities.

---

## Benchmark Runner

### Strategy Configurations

::: LNPBO.benchmarks.runner.STRATEGY_CONFIGS

::: LNPBO.benchmarks.runner.ALL_STRATEGIES

### prepare_benchmark_data

::: LNPBO.benchmarks.runner.prepare_benchmark_data

### compute_metrics

::: LNPBO.benchmarks.runner.compute_metrics

---

## Within-Study Benchmark

### filter_study_df

::: LNPBO.benchmarks.benchmark.filter_study_df

---

## Statistical Utilities

### bootstrap_ci

::: LNPBO.benchmarks.stats.bootstrap_ci

### paired_wilcoxon

::: LNPBO.benchmarks.stats.paired_wilcoxon

### benjamini_hochberg

::: LNPBO.benchmarks.stats.benjamini_hochberg

### cohens_d_paired

::: LNPBO.benchmarks.stats.cohens_d_paired

### rank_biserial

::: LNPBO.benchmarks.stats.rank_biserial

### post_hoc_power

::: LNPBO.benchmarks.stats.post_hoc_power

---

## Regret and Performance Metrics

### simple_regret

::: LNPBO.benchmarks.stats.simple_regret

### cumulative_regret

::: LNPBO.benchmarks.stats.cumulative_regret

### acceleration_factor

::: LNPBO.benchmarks.stats.acceleration_factor

### enhancement_factor

::: LNPBO.benchmarks.stats.enhancement_factor

### format_result

::: LNPBO.benchmarks.stats.format_result
