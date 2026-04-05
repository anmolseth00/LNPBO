"""Statistical Infrastructure for Benchmark Analysis.

Provides bootstrap confidence intervals, paired Wilcoxon signed-rank
tests, BH-FDR correction, effect sizes, power analysis, regret metrics,
acceleration factors, enhancement factors, and result formatting.

Bootstrap CI:
    Efron, B. & Tibshirani, R.J. (1993). "An Introduction to the
    Bootstrap." Chapman & Hall/CRC.

Wilcoxon signed-rank test:
    Wilcoxon, F. (1945). "Individual comparisons by ranking methods."
    Biometrics Bulletin 1(6), 80-83.

Benjamini-Hochberg FDR:
    Benjamini, Y. & Hochberg, Y. (1995). "Controlling the false discovery
    rate: a practical and powerful approach to multiple testing."
    JRSS-B 57(1), 289-300.

Cohen's d:
    Cohen, J. (1988). "Statistical Power Analysis for the Behavioral
    Sciences." 2nd ed. Lawrence Erlbaum Associates.

Usage:
    from LNPBO.benchmarks.stats import (
        bootstrap_ci, paired_wilcoxon, benjamini_hochberg,
        cohens_d_paired, post_hoc_power, simple_regret,
    )
"""

import numpy as np
from scipy.stats import t as t_dist
from scipy.stats import wilcoxon

from .constants import MIN_N_WILCOXON


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
    if len(a) < MIN_N_WILCOXON:
        # Wilcoxon requires at least MIN_N_WILCOXON non-zero differences
        # for a meaningful p-value; return 1.0 to indicate non-significance
        try:
            _, p = wilcoxon(a, b, alternative="two-sided")
            return float(p)
        except ValueError:
            return 1.0
    _, p = wilcoxon(a, b, alternative="two-sided")
    return float(p)


def benjamini_hochberg(p_values, alpha=0.05):
    """Benjamini-Hochberg FDR correction for multiple comparisons.

    Parameters
    ----------
    p_values : array-like
        Raw p-values from multiple hypothesis tests.
    alpha : float
        Target FDR level. Default 0.05.

    Returns
    -------
    p_adjusted : np.ndarray
        BH-adjusted p-values (same length as input). Compare to alpha
        for significance: reject H0 where p_adjusted < alpha.
    rejected : np.ndarray
        Boolean mask of rejected hypotheses at the given alpha.

    Reference
    ---------
    Benjamini, Y. & Hochberg, Y. (1995). "Controlling the false discovery
    rate." JRSS-B 57(1), 289-300.
    """
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return np.array([]), np.array([], dtype=bool)

    sorted_idx = np.argsort(p)
    sorted_p = p[sorted_idx]
    ranks = np.arange(1, n + 1)

    # Step-up adjustment: p_adj[i] = min(p[i] * n / rank[i], 1.0)
    # then enforce monotonicity from the bottom
    adjusted = sorted_p * n / ranks
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)

    # Map back to original order
    p_adjusted = np.empty(n)
    p_adjusted[sorted_idx] = adjusted
    rejected = p_adjusted < alpha
    return p_adjusted, rejected


def cohens_d_paired(x, y):
    """Cohen's d for paired samples with 95% CI via noncentral t.

    Parameters
    ----------
    x, y : array-like
        Paired observations (same length).

    Returns
    -------
    d : float
        Cohen's d (paired): mean(x - y) / std(x - y).
    ci_lo : float
        Lower bound of 95% CI.
    ci_hi : float
        Upper bound of 95% CI.
    interpretation : str
        One of 'negligible', 'small', 'medium', 'large'.

    Reference
    ---------
    Cohen, J. (1988). Ch. 2: "The t Test for Means."
    Lakens, D. (2013). "Calculating and reporting effect sizes."
    Frontiers in Psychology, 4, 863.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    diff = x - y
    n = len(diff)
    d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else 0.0

    # SE via noncentral t approximation (Hedges & Olkin, 1985)
    se_d = np.sqrt(1 / n + d**2 / (2 * n))
    t_crit = t_dist.ppf(0.975, n - 1)
    ci_lo = d - t_crit * se_d
    ci_hi = d + t_crit * se_d

    if abs(d) < 0.2:
        interpretation = "negligible"
    elif abs(d) < 0.5:
        interpretation = "small"
    elif abs(d) < 0.8:
        interpretation = "medium"
    else:
        interpretation = "large"

    return d, ci_lo, ci_hi, interpretation


def rank_biserial(x, y):
    """Rank-biserial correlation (effect size for Wilcoxon signed-rank test).

    Parameters
    ----------
    x, y : array-like
        Paired observations.

    Returns
    -------
    r : float
        Rank-biserial correlation in [-1, 1]. Positive means x > y.

    Reference
    ---------
    Kerby, D.S. (2014). "The simple difference formula."
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    diff = x - y
    diff = diff[diff != 0]
    n = len(diff)
    if n == 0:
        return 0.0
    ranks = np.argsort(np.argsort(np.abs(diff))) + 1
    r_plus = np.sum(ranks[diff > 0])
    r_minus = np.sum(ranks[diff < 0])
    r = (r_plus - r_minus) / (n * (n + 1) / 2)
    return float(r)


def post_hoc_power(effect_size_d, n, alpha=0.05):
    """Post-hoc power for a paired t-test (approximation).

    Uses the noncentral t distribution to estimate the probability of
    rejecting H0 given the observed effect size and sample size.

    Parameters
    ----------
    effect_size_d : float
        Observed Cohen's d (paired).
    n : int
        Number of paired observations (e.g., number of studies).
    alpha : float
        Significance level.

    Returns
    -------
    power : float
        Estimated power (probability of rejecting H0).

    Reference
    ---------
    Cohen, J. (1988). Ch. 2, Table 2.4.
    """
    if n < 2 or effect_size_d == 0:
        return 0.0
    ncp = effect_size_d * np.sqrt(n)  # noncentrality parameter
    df = n - 1
    t_crit_val = t_dist.ppf(1 - alpha / 2, df)
    # Power = P(|T| > t_crit) under noncentral t(df, ncp)
    from scipy.stats import nct

    power = 1 - nct.cdf(t_crit_val, df, ncp) + nct.cdf(-t_crit_val, df, ncp)
    return float(np.clip(power, 0, 1))


def simple_regret(best_so_far, oracle_best):
    """Compute simple regret at each round.

    Parameters
    ----------
    best_so_far : array-like
        Best value found up to each round (length = n_rounds + 1).
    oracle_best : float
        Global optimum value.

    Returns
    -------
    regret : np.ndarray
        Simple regret per round: oracle_best - best_so_far[r].
    """
    bsf = np.asarray(best_so_far, dtype=float)
    return oracle_best - bsf


def cumulative_regret(round_best, oracle_best):
    """Compute cumulative regret.

    Parameters
    ----------
    round_best : array-like
        Best value in each round's batch (length = n_rounds).
    oracle_best : float
        Global optimum value.

    Returns
    -------
    cum_regret : np.ndarray
        Cumulative sum of per-round regret.
    """
    rb = np.asarray(round_best, dtype=float)
    per_round = oracle_best - rb
    return np.cumsum(per_round)


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
        """Convert a recall curve (dict or list of pairs) to sorted arrays."""
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
        """Return the budget (n_evaluated) at which recall first reaches target."""
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


def win_rate_ci(family_means_per_study, n_bootstrap=10000, ci=0.95, seed=42):
    """Bootstrap confidence intervals for per-family win rates across studies.

    A "win" means a family achieves the highest mean recall in a given study.
    Ties are broken by awarding a fractional win (1/k for k-way ties).

    Parameters
    ----------
    family_means_per_study : dict[str, list[float]]
        Maps family name to a list of per-study mean recall values.
        All lists must have the same length (one entry per study).
    n_bootstrap : int
        Number of bootstrap resamples.
    ci : float
        Confidence level. Default 0.95 gives 95% CI.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    result : dict[str, tuple[float, float, float]]
        Maps family name to (win_rate, ci_lower, ci_upper).

    Reference
    ---------
    Efron, B. & Tibshirani, R.J. (1993). "An Introduction to the
    Bootstrap." Chapman & Hall/CRC, Ch. 13 (percentile intervals).
    """
    families = list(family_means_per_study.keys())
    # Stack into (n_families, n_studies) array
    matrix = np.array([family_means_per_study[f] for f in families], dtype=float)
    n_families, n_studies = matrix.shape

    rng = np.random.default_rng(seed=seed)
    alpha = 1 - ci

    # Precompute wins for each bootstrap resample
    boot_indices = rng.integers(0, n_studies, size=(n_bootstrap, n_studies))
    win_rates_boot = np.zeros((n_bootstrap, n_families))

    for b in range(n_bootstrap):
        resampled = matrix[:, boot_indices[b]]  # (n_families, n_studies)
        maxvals = resampled.max(axis=0)  # (n_studies,)
        is_winner = resampled == maxvals[np.newaxis, :]  # (n_families, n_studies)
        n_winners = is_winner.sum(axis=0)  # (n_studies,) — handle ties
        fractional = is_winner / n_winners[np.newaxis, :]
        win_rates_boot[b] = fractional.mean(axis=1)

    # Point estimate from the original data
    maxvals_orig = matrix.max(axis=0)
    is_winner_orig = matrix == maxvals_orig[np.newaxis, :]
    n_winners_orig = is_winner_orig.sum(axis=0)
    fractional_orig = is_winner_orig / n_winners_orig[np.newaxis, :]
    point_estimates = fractional_orig.mean(axis=1)

    result = {}
    for i, fam in enumerate(families):
        lo = float(np.percentile(win_rates_boot[:, i], 100 * alpha / 2))
        hi = float(np.percentile(win_rates_boot[:, i], 100 * (1 - alpha / 2)))
        result[fam] = (float(point_estimates[i]), lo, hi)

    return result


def hierarchical_bootstrap_ci(study_seed_values, n_bootstrap=10000, ci=0.95, seed=42):
    """Two-level hierarchical bootstrap CI for nested study/seed data.

    Level 1: resample studies with replacement.
    Level 2: for each resampled study, resample seeds with replacement.
    Compute the grand mean of the doubly-resampled data.

    This accounts for the nested variance structure where seeds are
    clustered within studies, avoiding the underestimation of uncertainty
    that occurs when ignoring the clustering.

    Parameters
    ----------
    study_seed_values : list[list[float]]
        Outer list indexes studies; inner list contains seed-level metric
        values for that study.
    n_bootstrap : int
        Number of bootstrap resamples.
    ci : float
        Confidence level. Default 0.95 gives 95% CI.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    mean : float
        Grand mean of the original data.
    ci_lower : float
        Lower bound of the confidence interval.
    ci_upper : float
        Upper bound of the confidence interval.

    Reference
    ---------
    Davison, A.C. & Hinkley, D.V. (1997). "Bootstrap Methods and their
    Application." Cambridge University Press, Ch. 3.5 (hierarchical
    resampling).

    Ren, S., Lai, H., Tong, W., Aminzadeh, M., Hou, X. & Lai, S. (2010).
    "Nonparametric bootstrapping for hierarchical data." Journal of Applied
    Statistics, 37(9), 1487-1498. DOI: 10.1080/02664760903046102
    """
    studies = [np.asarray(s, dtype=float) for s in study_seed_values]
    n_studies = len(studies)

    rng = np.random.default_rng(seed=seed)
    alpha = 1 - ci

    # Point estimate: grand mean (mean of study means)
    study_means = np.array([s.mean() for s in studies])
    grand_mean = float(study_means.mean())

    boot_means = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        # Level 1: resample studies
        study_idx = rng.integers(0, n_studies, size=n_studies)
        resampled_means = np.empty(n_studies)
        for i, si in enumerate(study_idx):
            study = studies[si]
            # Level 2: resample seeds within each study
            seed_idx = rng.integers(0, len(study), size=len(study))
            resampled_means[i] = study[seed_idx].mean()
        boot_means[b] = resampled_means.mean()

    ci_lower = float(np.percentile(boot_means, 100 * alpha / 2))
    ci_upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))

    return grand_mean, ci_lower, ci_upper


def prospective_power(n_studies, effect_sizes=(0.2, 0.5, 0.8), alpha=0.05):
    """Report power for detecting given effect sizes with n paired observations.

    Uses the t-approximation to the Wilcoxon signed-rank test (asymptotic
    relative efficiency = pi/3 ~= 1.047 for normal data, so the t-based
    power is a reasonable approximation).

    Parameters
    ----------
    n_studies : int
        Number of paired observations (studies).
    effect_sizes : tuple of float
        Cohen's d values to evaluate.
    alpha : float
        Significance level.

    Returns
    -------
    power_dict : dict[float, float]
        Maps each effect size to its estimated power.

    Reference
    ---------
    Cohen, J. (1988). Ch. 2, Table 2.4.
    """
    from scipy.stats import nct

    power_dict = {}
    for d in effect_sizes:
        if n_studies < 2 or d == 0:
            power_dict[d] = 0.0
            continue
        ncp = d * np.sqrt(n_studies)
        df = n_studies - 1
        t_crit_val = t_dist.ppf(1 - alpha / 2, df)
        power = 1 - nct.cdf(t_crit_val, df, ncp) + nct.cdf(-t_crit_val, df, ncp)
        power_dict[d] = float(np.clip(power, 0, 1))
    return power_dict


def higgins_heterogeneity(study_effects, study_ses=None):
    """Compute I-squared and tau-squared (DerSimonian-Laird).

    I-squared quantifies the percentage of total variability due to
    true between-study heterogeneity rather than chance.

    Parameters
    ----------
    study_effects : array-like
        Observed effect sizes (e.g., mean recall) per study.
    study_ses : array-like, optional
        Standard errors per study. If None, uses equal weights
        (unweighted analysis).

    Returns
    -------
    I2 : float
        Higgins I-squared (0-100%). <25% low, 25-75% moderate, >75% high.
    tau2 : float
        DerSimonian-Laird estimate of between-study variance.
    Q : float
        Cochran's Q statistic.
    Q_p : float
        P-value for Q (chi-squared test with k-1 df).

    Reference
    ---------
    Higgins, J.P.T. & Thompson, S.G. (2002). "Quantifying heterogeneity
    in a meta-analysis." Statistics in Medicine, 21, 1539-1558.

    DerSimonian, R. & Laird, N. (1986). "Meta-analysis in clinical
    trials." Controlled Clinical Trials, 7, 177-188.
    """
    from scipy.stats import chi2

    effects = np.asarray(study_effects, dtype=float)
    k = len(effects)
    if k < 2:
        return 0.0, 0.0, 0.0, 1.0

    if study_ses is not None:
        ses = np.asarray(study_ses, dtype=float)
        w = 1.0 / (ses**2)
    else:
        w = np.ones(k)

    w_sum = w.sum()
    mu_hat = np.sum(w * effects) / w_sum
    Q = float(np.sum(w * (effects - mu_hat) ** 2))
    df = k - 1

    Q_p = float(1 - chi2.cdf(Q, df)) if df > 0 else 1.0
    I2 = max(0.0, (Q - df) / Q * 100) if Q > 0 else 0.0

    # DerSimonian-Laird tau^2
    c = w_sum - np.sum(w**2) / w_sum
    tau2 = max(0.0, (Q - df) / c) if c > 0 else 0.0

    return float(I2), float(tau2), Q, Q_p
