#!/usr/bin/env python3
"""Compute ICC using a random-intercepts REML model."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from diagnostics.utils import load_lnpdb_clean


def _group_summaries(y: np.ndarray, groups: np.ndarray):
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
                "sum_y2": float((y_g ** 2).sum()),
            }
        )
    return stats


def _reml_loglik(params, stats, n_total):
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
    denom = sigma_a2 + sigma_e2
    return float(sigma_a2 / denom) if denom > 0 else 0.0


def bootstrap_icc(y: np.ndarray, groups: np.ndarray, n_boot: int, seed: int = 42):
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
    y = df["Experiment_value"].to_numpy(dtype=float)
    groups = df["study_id"].to_numpy(dtype=str)

    sigma_a2, sigma_e2, ll_full = fit_reml_random_intercept(y, groups)
    icc = icc_from_variances(sigma_a2, sigma_e2)

    sigma_e2_null, ll_null = fit_reml_null(y)
    lr_stat = 2 * (ll_full - ll_null)
    # Mixture distribution for boundary (Self & Liang 1987): 0.5*chi2_0 + 0.5*chi2_1
    p_val = 0.5 * (1 - chi2.cdf(lr_stat, df=1))

    ci_low, ci_high = bootstrap_icc(y, groups, n_boot=n_boot)

    return {
        "label": label,
        "n": int(len(y)),
        "n_studies": int(len(np.unique(groups))),
        "sigma_a2": sigma_a2,
        "sigma_e2": sigma_e2,
        "icc": icc,
        "icc_ci": [ci_low, ci_high],
        "lr_stat": float(lr_stat),
        "p_value": float(p_val),
    }


def main() -> int:
    df = load_lnpdb_clean(drop_duplicates=False)

    results = []
    results.append(compute_icc(df, label="global", n_boot=200))

    for assay_type, sdf in df.groupby("assay_type"):
        if len(sdf) < 50 or sdf["study_id"].nunique() < 3:
            continue
        results.append(compute_icc(sdf, label=f"assay:{assay_type}", n_boot=100))

    out_path = Path("diagnostics") / "icc_results.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))

    for res in results:
        print(
            f"{res['label']}: ICC={res['icc']:.3f} "
            f"CI[{res['icc_ci'][0]:.3f}, {res['icc_ci'][1]:.3f}] "
            f"p={res['p_value']:.2e} n={res['n']} studies={res['n_studies']}"
        )

    print(f"Saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
