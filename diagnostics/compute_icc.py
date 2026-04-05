#!/usr/bin/env python3
"""Compute ICC using a random-intercepts REML model.

Reference:
    Searle, S.R., Casella, G. & McCulloch, C.E. (1992). "Variance Components."
    Wiley Series in Probability and Mathematical Statistics.
"""

import json
import logging
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2

from LNPBO.benchmarks.stats import benjamini_hochberg
from LNPBO.data.study_utils import load_lnpdb_clean

logger = logging.getLogger("lnpbo")


def _group_summaries(y: np.ndarray, groups: np.ndarray):
    """Compute per-group sufficient statistics for REML estimation.

    Args:
        y: Response vector of shape ``(n,)``.
        groups: Group labels of shape ``(n,)``.

    Returns:
        List of dicts, each with keys ``"group"``, ``"n"``,
        ``"sum_y"``, and ``"sum_y2"``.
    """
    unique = np.unique(groups)
    stats = []
    for g in unique:
        mask = groups == g
        y_g = y[mask]
        stats.append(
            {
                "group": g,
                "n": len(y_g),
                "sum_y": float(y_g.sum()),
                "sum_y2": float((y_g**2).sum()),
            }
        )
    return stats


def _reml_loglik(params, stats, n_total):
    """Evaluate the REML log-likelihood for a random-intercepts model.

    Args:
        params: Length-2 array ``[log(sigma_e^2), log(sigma_a^2)]``.
        stats: Per-group sufficient statistics from ``_group_summaries``.
        n_total: Total number of observations.

    Returns:
        REML log-likelihood (scalar).
    """
    log_sigma_e2, log_sigma_a2 = params
    sigma_e2 = np.exp(log_sigma_e2)
    sigma_a2 = np.exp(log_sigma_a2)

    logdet = 0.0
    xtvix = 0.0
    weighted_sum = 0.0

    for s in stats:
        n = s["n"]
        denom = sigma_e2 + n * sigma_a2
        logdet += (n - 1) * np.log(sigma_e2) + np.log(denom)
        xtvix += n / denom
        weighted_sum += s["sum_y"] / denom

    beta_hat = weighted_sum / xtvix

    quad = 0.0
    for s in stats:
        n = s["n"]
        denom = sigma_e2 + n * sigma_a2
        sum_y = s["sum_y"]
        sum_y2 = s["sum_y2"]
        sum_r = sum_y - n * beta_hat
        sum_r2 = sum_y2 - 2 * beta_hat * sum_y + n * beta_hat**2
        quad += (sum_r2 / sigma_e2) - (sigma_a2 / (sigma_e2 * denom)) * (sum_r**2)

    ll = -0.5 * (logdet + np.log(xtvix) + quad + (n_total - 1) * np.log(2 * np.pi))
    return ll


def fit_reml_random_intercept(y: np.ndarray, groups: np.ndarray):
    """Fit a random-intercepts model via REML and return variance components.

    Optimizes the REML log-likelihood using L-BFGS-B on the
    log-transformed variance parameters.

    Args:
        y: Response vector of shape ``(n,)``.
        groups: Group labels of shape ``(n,)``.

    Returns:
        Tuple of ``(sigma_a^2, sigma_e^2, log_likelihood)`` where
        ``sigma_a^2`` is the between-group variance and ``sigma_e^2``
        is the within-group (residual) variance.

    Raises:
        RuntimeError: If the L-BFGS-B optimizer fails to converge.
    """
    stats = _group_summaries(y, groups)
    n_total = len(y)

    # Initial guesses
    overall_var = float(np.var(y, ddof=1))
    group_means = np.array([s["sum_y"] / s["n"] for s in stats])
    mean_n = np.mean([s["n"] for s in stats])
    between_var = float(np.var(group_means, ddof=1)) if len(group_means) > 1 else 0.0
    sigma_a2_init = max(between_var - overall_var / max(mean_n, 1.0), 1e-6)
    sigma_e2_init = max(overall_var - sigma_a2_init, 1e-6)

    x0 = np.log([sigma_e2_init, sigma_a2_init])

    def obj(p):
        return -_reml_loglik(p, stats, n_total)

    res = minimize(obj, x0=x0, method="L-BFGS-B")
    if not res.success:
        raise RuntimeError(f"REML optimization failed: {res.message}")

    sigma_e2 = float(np.exp(res.x[0]))
    sigma_a2 = float(np.exp(res.x[1]))
    ll = float(_reml_loglik(res.x, stats, n_total))

    return sigma_a2, sigma_e2, ll


def fit_reml_null(y: np.ndarray):
    """Fit the null model (no random effect) and return its REML log-likelihood.

    Fixes ``sigma_a^2 = 0`` and estimates ``sigma_e^2`` as the sample
    variance with ``n - 1`` denominator.

    Args:
        y: Response vector of shape ``(n,)``.

    Returns:
        Tuple of ``(sigma_e^2, log_likelihood)``.
    """
    # sigma_a2 fixed at 0; REML estimate for sigma_e2 with intercept only
    n = len(y)
    mu = float(np.mean(y))
    sum_sq = float(((y - mu) ** 2).sum())
    sigma_e2 = max(sum_sq / max(n - 1, 1), 1e-12)

    logdet = n * np.log(sigma_e2)
    xtvix = n / sigma_e2
    quad = sum_sq / sigma_e2
    ll = -0.5 * (logdet + np.log(xtvix) + quad + (n - 1) * np.log(2 * np.pi))
    return sigma_e2, ll


def icc_from_variances(sigma_a2: float, sigma_e2: float) -> float:
    """Compute the intraclass correlation coefficient from variance components.

    Args:
        sigma_a2: Between-group variance.
        sigma_e2: Within-group (residual) variance.

    Returns:
        ICC value in ``[0, 1]``.  Returns 0.0 if both variances are zero.
    """
    denom = sigma_a2 + sigma_e2
    return float(sigma_a2 / denom) if denom > 0 else 0.0


def bootstrap_icc(y: np.ndarray, groups: np.ndarray, n_boot: int, seed: int = 42):
    """Compute a bootstrap 95% confidence interval for the ICC.

    Resamples groups (with replacement) and refits the random-intercepts
    model on each bootstrap sample.

    Args:
        y: Response vector of shape ``(n,)``.
        groups: Group labels of shape ``(n,)``.
        n_boot: Number of bootstrap iterations.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of ``(ci_lower, ci_upper)`` at the 2.5th and 97.5th
        percentiles.  Returns ``(nan, nan)`` if all bootstrap fits fail.
    """
    rng = np.random.RandomState(seed)
    unique = np.unique(groups)
    boot = []
    for _ in range(n_boot):
        sample_groups = rng.choice(unique, size=len(unique), replace=True)
        mask = np.isin(groups, sample_groups)
        y_b = y[mask]
        g_b = groups[mask]
        try:
            sigma_a2, sigma_e2, _ = fit_reml_random_intercept(y_b, g_b)
            boot.append(icc_from_variances(sigma_a2, sigma_e2))
        except RuntimeError:
            continue
    if not boot:
        return (float("nan"), float("nan"))
    return (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))


def compute_icc(df, label: str, n_boot: int = 200):
    """Compute the ICC for a dataframe with study-level grouping.

    Fits a REML random-intercepts model, performs a likelihood-ratio test
    against the null (no group effect), and estimates bootstrap confidence
    intervals.

    Args:
        df: DataFrame with columns ``"Experiment_value"`` and
            ``"study_id"``.
        label: Descriptive label for this ICC computation (e.g.
            ``"global"`` or ``"assay:transfection"``).
        n_boot: Number of bootstrap iterations for the CI.

    Returns:
        Dict with keys ``"label"``, ``"n"``, ``"n_studies"``,
        ``"sigma_a2"``, ``"sigma_e2"``, ``"icc"``, ``"icc_ci"``,
        ``"lr_stat"``, and ``"p_value"``.
    """
    y = df["Experiment_value"].to_numpy(dtype=float)
    groups = df["study_id"].to_numpy(dtype=str)

    sigma_a2, sigma_e2, ll_full = fit_reml_random_intercept(y, groups)
    icc = icc_from_variances(sigma_a2, sigma_e2)

    _sigma_e2_null, ll_null = fit_reml_null(y)
    lr_stat = 2 * (ll_full - ll_null)
    # Mixture distribution for boundary (Self & Liang 1987): 0.5*chi2_0 + 0.5*chi2_1
    p_val = 0.5 * (1 - chi2.cdf(lr_stat, df=1))

    ci_low, ci_high = bootstrap_icc(y, groups, n_boot=n_boot)

    return {
        "label": label,
        "n": len(y),
        "n_studies": len(np.unique(groups)),
        "sigma_a2": sigma_a2,
        "sigma_e2": sigma_e2,
        "icc": icc,
        "icc_ci": [ci_low, ci_high],
        "lr_stat": float(lr_stat),
        "p_value": float(p_val),
    }


def main() -> int:
    """Run ICC analysis globally and per assay type, write results to disk.

    Outputs ``diagnostics/icc_results.json`` with global and per-assay-type
    ICC estimates, confidence intervals, and likelihood-ratio test p-values.

    Returns:
        Exit code (0 on success).
    """
    df = load_lnpdb_clean(drop_duplicates=False)

    results = []
    results.append(compute_icc(df, label="global", n_boot=200))

    for assay_type, sdf in df.groupby("assay_type"):
        if len(sdf) < 50 or sdf["study_id"].nunique() < 3:
            continue
        results.append(compute_icc(sdf, label=f"assay:{assay_type}", n_boot=100))

    # BH-FDR correction across all ICC p-values
    if len(results) > 1:
        raw_ps = np.array([r["p_value"] for r in results])
        adj_ps, _ = benjamini_hochberg(raw_ps)
        for i, res in enumerate(results):
            res["p_adjusted"] = float(adj_ps[i])

    out_path = Path("diagnostics") / "icc_results.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))

    for res in results:
        p_adj_str = f" p_adj={res['p_adjusted']:.2e}" if "p_adjusted" in res else ""
        logger.info(
            f"{res['label']}: ICC={res['icc']:.3f} "
            f"CI[{res['icc_ci'][0]:.3f}, {res['icc_ci'][1]:.3f}] "
            f"p={res['p_value']:.2e}{p_adj_str} n={res['n']} studies={res['n_studies']}"
        )

    logger.info("Saved %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
