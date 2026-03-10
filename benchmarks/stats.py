"""Statistical Infrastructure for Benchmark Analysis.

Provides bootstrap confidence intervals, paired Wilcoxon signed-rank
tests, acceleration factors, enhancement factors, and result formatting
for rigorous comparison of benchmark strategies.

Bootstrap CI:
    Efron, B. & Tibshirani, R.J. (1993). "An Introduction to the
    Bootstrap." Chapman & Hall/CRC.

Wilcoxon signed-rank test:
    Wilcoxon, F. (1945). "Individual comparisons by ranking methods."
    Biometrics Bulletin 1(6), 80-83.

Usage:
    from LNPBO.benchmarks.stats import bootstrap_ci, paired_wilcoxon
"""

import numpy as np
from scipy.stats import wilcoxon


def bootstrap_ci(values, n_boot=10000, alpha=0.05, statistic=np.mean):
    """Compute bootstrap confidence interval for a statistic.

    Parameters
    ----------
    values : array-like
        Observed values (one per seed).
    n_boot : int
        Number of bootstrap resamples.
    alpha : float
        Significance level. Default 0.05 gives 95% CI.
    statistic : callable
        Statistic to compute on each bootstrap sample. Default: np.mean.

    Returns
    -------
    ci_low : float
        Lower bound of the confidence interval.
    ci_high : float
        Upper bound of the confidence interval.

    Reference
    ---------
    Efron, B. & Tibshirani, R.J. (1993). "An Introduction to the
    Bootstrap." Chapman & Hall/CRC, Ch. 13 (percentile intervals).
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    rng = np.random.default_rng(seed=42)
    boot_indices = rng.integers(0, n, size=(n_boot, n))
    boot_stats = np.array([statistic(values[idx]) for idx in boot_indices])
    ci_low = float(np.percentile(boot_stats, 100 * alpha / 2))
    ci_high = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
    return ci_low, ci_high


def paired_wilcoxon(strategy_a_seeds, strategy_b_seeds):
    """Paired Wilcoxon signed-rank test across seeds.

    Tests the null hypothesis that the distribution of differences
    between paired observations is symmetric about zero.

    Parameters
    ----------
    strategy_a_seeds : array-like
        Metric values for strategy A (one per seed).
    strategy_b_seeds : array-like
        Metric values for strategy B (one per seed).

    Returns
    -------
    p_value : float
        Two-sided p-value from the Wilcoxon signed-rank test.
        Returns 1.0 if all differences are zero or if sample size
        is too small for the test.

    Reference
    ---------
    Wilcoxon, F. (1945). "Individual comparisons by ranking methods."
    Biometrics Bulletin 1(6), 80-83.
    """
    a = np.asarray(strategy_a_seeds, dtype=float)
    b = np.asarray(strategy_b_seeds, dtype=float)
    diff = a - b
    if np.all(diff == 0):
        return 1.0
    if len(a) < 6:
        # Wilcoxon requires at least 6 non-zero differences for a
        # meaningful p-value; return 1.0 to indicate non-significance
        try:
            _, p = wilcoxon(a, b, alternative="two-sided")
            return float(p)
        except ValueError:
            return 1.0
    _, p = wilcoxon(a, b, alternative="two-sided")
    return float(p)


def acceleration_factor(random_curve, bo_curve, target_recall):
    """Compute acceleration factor: how many random evaluations are needed
    to reach the same recall that BO reaches, divided by BO's budget.

    AF = n_random_to_reach_target / n_bo_to_reach_target

    Parameters
    ----------
    random_curve : dict or list of (n_evaluated, recall) pairs
        Random strategy's recall as a function of evaluations.
        If dict, keys are n_evaluated, values are recall.
        If list, each element is (n_evaluated, recall).
    bo_curve : dict or list of (n_evaluated, recall) pairs
        BO strategy's recall as a function of evaluations.
    target_recall : float
        The recall target to compare at.

    Returns
    -------
    af : float
        Acceleration factor. > 1 means BO is faster.
        Returns float('inf') if random never reaches target.
        Returns float('nan') if BO never reaches target.
    """
    def _to_arrays(curve):
        if isinstance(curve, dict):
            ns = sorted(curve.keys())
            rs = [curve[n] for n in ns]
        else:
            ns = [x[0] for x in curve]
            rs = [x[1] for x in curve]
        return np.array(ns, dtype=float), np.array(rs, dtype=float)

    n_rand, r_rand = _to_arrays(random_curve)
    n_bo, r_bo = _to_arrays(bo_curve)

    def _budget_to_reach(ns, rs, target):
        above = np.where(rs >= target)[0]
        if len(above) == 0:
            return None
        return float(ns[above[0]])

    n_bo_target = _budget_to_reach(n_bo, r_bo, target_recall)
    if n_bo_target is None:
        return float("nan")

    n_rand_target = _budget_to_reach(n_rand, r_rand, target_recall)
    if n_rand_target is None:
        return float("inf")

    return n_rand_target / n_bo_target


def enhancement_factor(random_best_at_n, bo_best_at_n):
    """Compute enhancement factor: BO's best value / random's best value
    at the same budget N.

    EF = bo_best / random_best

    Parameters
    ----------
    random_best_at_n : float
        Best value found by random at budget N.
    bo_best_at_n : float
        Best value found by BO at budget N.

    Returns
    -------
    ef : float
        Enhancement factor. > 1 means BO found a better formulation.
        Returns float('nan') if random_best is zero.
    """
    if random_best_at_n == 0:
        return float("nan")
    return bo_best_at_n / random_best_at_n


def format_result(mean, std, ci_low=None, ci_high=None):
    """Format a benchmark result as a human-readable string.

    Parameters
    ----------
    mean : float
        Mean value across seeds.
    std : float
        Standard deviation across seeds.
    ci_low : float, optional
        Lower bound of confidence interval.
    ci_high : float, optional
        Upper bound of confidence interval.

    Returns
    -------
    formatted : str
        e.g., "23.6% +/- 11.1% [12.4%, 34.8%]" for recall values,
        or "0.376 +/- 0.042 [0.334, 0.418]" for R^2 values.
    """
    if abs(mean) <= 1.0 and abs(std) <= 1.0 and (ci_low is None or abs(ci_low) <= 1.0):
        # Likely a fraction/recall -- display as percentage
        parts = [f"{mean:.1%} +/- {std:.1%}"]
        if ci_low is not None and ci_high is not None:
            parts.append(f"[{ci_low:.1%}, {ci_high:.1%}]")
    else:
        parts = [f"{mean:.3f} +/- {std:.3f}"]
        if ci_low is not None and ci_high is not None:
            parts.append(f"[{ci_low:.3f}, {ci_high:.3f}]")
    return " ".join(parts)
