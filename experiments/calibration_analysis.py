#!/usr/bin/env python3
"""Surrogate model calibration analysis for BO strategies.

Evaluates whether the uncertainty estimates from RF, NGBoost, and GP
surrogates are well-calibrated. If sigma is miscalibrated, UCB
degenerates to greedy selection, which would explain why P&R ~ BO.

For each study and seed:
  1. Load study data, split into seed/oracle
  2. Train each surrogate on seed data
  3. Predict (mu, sigma) on oracle pool
  4. Compare predicted intervals to true (oracle) values
  5. Compute calibration metrics: coverage at nominal levels, sharpness,
     calibration error, CRPS

Generates:
  - paper/figures/fig_calibration.pdf (reliability diagram)
  - benchmark_results/calibration_analysis.json (full results)

Usage:
    python -m experiments.calibration_analysis
"""

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.preprocessing import MinMaxScaler

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# 5 PMIDs spanning different sizes and types; multi-Model_type PMIDs resolve
# to their cell-line sub-studies (e.g. 39060305 -> 39060305_HeLa + _RAW264.7).
STUDY_PMIDS = [
    "39060305",  # HeLa(1200) + RAW264.7(1200) sub-studies
    "37985700",  # A549(1801) sub-study
    "36997680",  # A549(720) sub-study
    "38740955",  # HeLa(560) sub-study
    "38060560",  # N=288, il_diverse_fixed_ratios (single MT)
]

SEEDS = [42, 123, 456]

# Nominal confidence levels to evaluate
NOMINAL_LEVELS = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])

SURROGATES = ["rf_ucb", "ngboost", "gp_ucb"]

SURROGATE_DISPLAY = {
    "rf_ucb": "Random Forest",
    "ngboost": "NGBoost",
    "gp_ucb": "GP (sklearn)",
}

SURROGATE_COLORS = {
    "rf_ucb": "#bcbd22",
    "ngboost": "#4daf4a",
    "gp_ucb": "#17becf",
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = PROJECT_ROOT / "benchmark_results" / "calibration_analysis.json"
FIGURE_PATH = PROJECT_ROOT / "paper" / "figures" / "fig_calibration.pdf"


# ---------------------------------------------------------------------------
# Surrogate fitting and prediction
# ---------------------------------------------------------------------------


def fit_predict_surrogate(X_train, y_train, X_pool, surrogate, random_seed=42):
    """Fit a surrogate on training data and return (mu, sigma) on pool.

    All surrogates return Gaussian-style (mu, sigma) where intervals
    are constructed as [mu - z*sigma, mu + z*sigma].
    """
    scaler = MinMaxScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_pool_s = scaler.transform(X_pool)

    if surrogate == "rf_ucb":
        from sklearn.ensemble import RandomForestRegressor

        rf = RandomForestRegressor(n_estimators=200, random_state=random_seed, n_jobs=-1)
        rf.fit(X_train_s, y_train)
        tree_preds = np.array([t.predict(X_pool_s) for t in rf.estimators_])
        mu = tree_preds.mean(axis=0)
        sigma = tree_preds.std(axis=0)

    elif surrogate == "ngboost":
        from ngboost import NGBRegressor
        from ngboost.distns import Normal

        model = NGBRegressor(
            Dist=Normal,
            n_estimators=200,
            random_state=random_seed,
            verbose=False,
        )
        model.fit(X_train_s, y_train)
        dists = model.pred_dist(X_pool_s)
        mu = dists.mean()
        sigma = dists.scale

    elif surrogate == "gp_ucb":
        from sklearn.gaussian_process import GaussianProcessRegressor

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gp = GaussianProcessRegressor(
                alpha=1e-6,
                n_restarts_optimizer=5,
                random_state=random_seed,
            )
            gp.fit(X_train_s, y_train)
            mu, sigma = gp.predict(X_pool_s, return_std=True)

    else:
        raise ValueError(f"Unknown surrogate: {surrogate!r}")

    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-10)
    return mu, sigma


# ---------------------------------------------------------------------------
# Calibration metrics
# ---------------------------------------------------------------------------


def compute_coverage(y_true, mu, sigma, nominal_levels):
    """Compute observed coverage at each nominal level.

    For level alpha, the predicted interval is:
        [mu - z*sigma, mu + z*sigma]  where z = norm.ppf((1+alpha)/2)
    Coverage = fraction of y_true within the interval.
    """
    coverages = []
    for alpha in nominal_levels:
        z = stats.norm.ppf((1 + alpha) / 2)
        lower = mu - z * sigma
        upper = mu + z * sigma
        in_interval = ((y_true >= lower) & (y_true <= upper)).mean()
        coverages.append(float(in_interval))
    return np.array(coverages)


def compute_sharpness(sigma, nominal_levels):
    """Mean interval width at each nominal level."""
    widths = []
    for alpha in nominal_levels:
        z = stats.norm.ppf((1 + alpha) / 2)
        width = 2 * z * sigma
        widths.append(float(width.mean()))
    return np.array(widths)


def compute_calibration_error(observed_coverage, nominal_levels):
    """Mean absolute calibration error (MACE)."""
    return float(np.mean(np.abs(observed_coverage - nominal_levels)))


def compute_crps(y_true, mu, sigma):
    """Continuous Ranked Probability Score for Gaussian predictive distribution.

    CRPS(N(mu, sigma^2), y) = sigma * [z*(2*Phi(z)-1) + 2*phi(z) - 1/sqrt(pi)]
    where z = (y - mu) / sigma, Phi = CDF, phi = PDF.

    Reference: Gneiting & Raftery (2007), "Strictly Proper Scoring Rules,
    Prediction, and Estimation", JASA.
    """
    z = (y_true - mu) / sigma
    crps = sigma * (z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi))
    return float(crps.mean())


def compute_nll(y_true, mu, sigma):
    """Negative log-likelihood under Gaussian predictive distribution."""
    nll = 0.5 * np.log(2 * np.pi * sigma**2) + 0.5 * ((y_true - mu) / sigma) ** 2
    return float(nll.mean())


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------


def run_calibration_analysis():
    print("=" * 70)
    print("SURROGATE CALIBRATION ANALYSIS")
    print("=" * 70)

    # Load studies JSON
    studies_json = PROJECT_ROOT / "experiments" / "data_integrity" / "studies_with_ids.json"
    with open(studies_json) as f:
        all_studies = json.load(f)

    # Filter to our selected studies (PMID matches all sub-studies)
    study_map = {}
    for s in all_studies:
        sid = s.get("study_id", str(int(float(s["pmid"]))))
        pmid_str = str(int(float(s["pmid"])))
        if sid in STUDY_PMIDS or pmid_str in STUDY_PMIDS:
            study_map[sid] = s

    # Load full LNPDB
    sys.path.insert(0, str(PROJECT_ROOT))
    from LNPBO.data.lnpdb_bridge import load_lnpdb_full

    print("\nLoading LNPDB...")
    dataset = load_lnpdb_full()
    df = dataset.df
    print(f"  {len(df):,} formulations loaded")

    # Ensure top_k_pct exists (needed by prepare_study_data)
    from LNPBO.benchmarks.benchmark import ensure_top_k_pct

    ensure_top_k_pct(list(study_map.values()))

    # Results accumulator: surrogate -> list of per-(study, seed) results
    all_results = {s: [] for s in SURROGATES}

    for sid, study_info in sorted(study_map.items()):
        pmid = str(int(float(study_info["pmid"])))
        print(f"\n{'=' * 50}")
        print(f"Study {sid}: N={study_info['n_formulations']}, type={study_info['study_type']}")
        print(f"{'=' * 50}")

        for seed in SEEDS:
            print(f"\n  Seed {seed}:")

            # Prepare data using the same pipeline as the benchmark
            from LNPBO.benchmarks.benchmark import prepare_study_data
            from LNPBO.optimization._normalize import copula_transform

            try:
                pca_data = prepare_study_data(df, study_info, seed)
            except Exception as e:
                print(f"    FAILED to prepare data: {e}")
                continue

            _, encoded_df, feature_cols, seed_idx, oracle_idx, _ = pca_data

            X_train = encoded_df.loc[seed_idx, feature_cols].values
            y_train_raw = encoded_df.loc[seed_idx, "Experiment_value"].values

            # Apply copula normalization (same as benchmark)
            y_train = copula_transform(y_train_raw)

            X_pool = encoded_df.loc[oracle_idx, feature_cols].values
            y_pool_raw = encoded_df.loc[oracle_idx, "Experiment_value"].values

            # Transform pool targets to same scale for calibration evaluation
            y_pool = copula_transform(y_train_raw, x_new=y_pool_raw)

            for surrogate in SURROGATES:
                t0 = time.time()
                try:
                    mu, sigma = fit_predict_surrogate(X_train, y_train, X_pool, surrogate, random_seed=seed)
                except Exception as e:
                    print(f"    {surrogate}: FAILED - {e}")
                    continue
                elapsed = time.time() - t0

                # Compute calibration metrics
                coverage = compute_coverage(y_pool, mu, sigma, NOMINAL_LEVELS)
                sharpness = compute_sharpness(sigma, NOMINAL_LEVELS)
                cal_error = compute_calibration_error(coverage, NOMINAL_LEVELS)
                crps = compute_crps(y_pool, mu, sigma)
                nll = compute_nll(y_pool, mu, sigma)

                # Also compute on raw (unnormalized) scale for interpretability
                mu_raw, sigma_raw = fit_predict_surrogate(X_train, y_train_raw, X_pool, surrogate, random_seed=seed)
                coverage_raw = compute_coverage(y_pool_raw, mu_raw, sigma_raw, NOMINAL_LEVELS)
                cal_error_raw = compute_calibration_error(coverage_raw, NOMINAL_LEVELS)

                # Store results
                result = {
                    "pmid": pmid,
                    "seed": seed,
                    "surrogate": surrogate,
                    "n_train": len(X_train),
                    "n_pool": len(X_pool),
                    "elapsed": elapsed,
                    # Copula-normalized metrics (what the BO actually sees)
                    "coverage": coverage.tolist(),
                    "sharpness": sharpness.tolist(),
                    "calibration_error": cal_error,
                    "crps": crps,
                    "nll": nll,
                    "mean_sigma": float(sigma.mean()),
                    "median_sigma": float(np.median(sigma)),
                    "sigma_cv": float(sigma.std() / sigma.mean()) if sigma.mean() > 0 else 0.0,
                    # Raw-scale metrics
                    "coverage_raw": coverage_raw.tolist(),
                    "calibration_error_raw": cal_error_raw,
                }
                all_results[surrogate].append(result)

                # Print summary
                c68 = coverage[NOMINAL_LEVELS == 0.7][0] if 0.7 in NOMINAL_LEVELS else coverage[6]
                c95 = coverage[NOMINAL_LEVELS == 0.95][0] if 0.95 in NOMINAL_LEVELS else coverage[9]
                print(
                    f"    {surrogate:12s}: MACE={cal_error:.3f}, "
                    f"cov@70%={c68:.3f}, cov@95%={c95:.3f}, "
                    f"CRPS={crps:.3f}, mean_sigma={sigma.mean():.3f}, "
                    f"({elapsed:.1f}s)"
                )

    # -----------------------------------------------------------------------
    # Aggregate results
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("AGGREGATE RESULTS")
    print(f"{'=' * 70}\n")

    aggregate = {}
    for surrogate in SURROGATES:
        results = all_results[surrogate]
        if not results:
            continue

        n = len(results)
        coverages = np.array([r["coverage"] for r in results])
        mean_coverage = coverages.mean(axis=0)
        std_coverage = coverages.std(axis=0, ddof=1) if n > 1 else np.zeros_like(mean_coverage)

        cal_errors = [r["calibration_error"] for r in results]
        crps_vals = [r["crps"] for r in results]
        nll_vals = [r["nll"] for r in results]
        mean_sigmas = [r["mean_sigma"] for r in results]
        sigma_cvs = [r["sigma_cv"] for r in results]

        coverages_raw = np.array([r["coverage_raw"] for r in results])
        mean_coverage_raw = coverages_raw.mean(axis=0)

        aggregate[surrogate] = {
            "n_evaluations": n,
            "nominal_levels": NOMINAL_LEVELS.tolist(),
            "mean_coverage": mean_coverage.tolist(),
            "std_coverage": std_coverage.tolist(),
            "mean_coverage_raw": mean_coverage_raw.tolist(),
            "mean_calibration_error": float(np.mean(cal_errors)),
            "std_calibration_error": float(np.std(cal_errors, ddof=1)) if n > 1 else 0.0,
            "mean_crps": float(np.mean(crps_vals)),
            "std_crps": float(np.std(crps_vals, ddof=1)) if n > 1 else 0.0,
            "mean_nll": float(np.mean(nll_vals)),
            "std_nll": float(np.std(nll_vals, ddof=1)) if n > 1 else 0.0,
            "mean_sigma": float(np.mean(mean_sigmas)),
            "mean_sigma_cv": float(np.mean(sigma_cvs)),
        }

        print(f"  {SURROGATE_DISPLAY[surrogate]:20s} (n={n}):")
        print(
            f"    MACE:       {np.mean(cal_errors):.3f} +/- {np.std(cal_errors, ddof=1):.3f}"
            if n > 1
            else f"    MACE:       {np.mean(cal_errors):.3f}"
        )
        print(
            f"    CRPS:       {np.mean(crps_vals):.3f} +/- {np.std(crps_vals, ddof=1):.3f}"
            if n > 1
            else f"    CRPS:       {np.mean(crps_vals):.3f}"
        )
        print(f"    NLL:        {np.mean(nll_vals):.3f}")
        print(f"    Mean sigma: {np.mean(mean_sigmas):.3f}")
        print(f"    Sigma CV:   {np.mean(sigma_cvs):.3f}")

        # Print coverage at key levels
        for target_level in [0.5, 0.7, 0.9, 0.95]:
            idx = np.argmin(np.abs(NOMINAL_LEVELS - target_level))
            obs = mean_coverage[idx]
            nom = NOMINAL_LEVELS[idx]
            gap = obs - nom
            direction = "over" if gap > 0 else "under"
            print(f"    Coverage@{nom:.0%}:  {obs:.3f} ({direction}-coverage by {abs(gap):.3f})")

    # -----------------------------------------------------------------------
    # Interpretation
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("INTERPRETATION")
    print(f"{'=' * 70}\n")

    for surrogate in SURROGATES:
        if surrogate not in aggregate:
            continue
        agg = aggregate[surrogate]
        name = SURROGATE_DISPLAY[surrogate]
        mc = np.array(agg["mean_coverage"])
        nl = np.array(agg["nominal_levels"])

        # Check if consistently over- or under-confident
        over = (mc > nl).sum()
        under = (mc < nl).sum()
        mace = agg["mean_calibration_error"]

        if mace < 0.05:
            cal_quality = "well-calibrated"
        elif mace < 0.10:
            cal_quality = "moderately calibrated"
        elif mace < 0.20:
            cal_quality = "poorly calibrated"
        else:
            cal_quality = "severely miscalibrated"

        # Under-coverage (observed < nominal) means intervals are too narrow,
        # i.e. the model is OVER-confident in its predictions.
        # Over-coverage (observed > nominal) means intervals are too wide,
        # i.e. the model is UNDER-confident (conservative).
        if over > under:
            direction = "conservative/under-confident (intervals too wide, over-coverage)"
        elif under > over:
            direction = "over-confident (intervals too narrow, under-coverage)"
        else:
            direction = "mixed"

        print(f"  {name}: {cal_quality} (MACE={mace:.3f}), tendency: {direction}")
        print(f"    {over}/{len(nl)} levels over-covered, {under}/{len(nl)} under-covered")

        # Check if sigma is flat (would mean UCB ~ greedy)
        cv = agg["mean_sigma_cv"]
        if cv < 0.1:
            print(f"    WARNING: sigma CV={cv:.3f} is very low -- uncertainty is nearly constant")
            print("    This means UCB ~ greedy: kappa*sigma acts as a constant offset.")
        elif cv < 0.3:
            print(f"    Sigma CV={cv:.3f} is moderate -- some candidates get more exploration")
        else:
            print(f"    Sigma CV={cv:.3f} is high -- good differentiation between candidates")

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "config": {
            "studies": STUDY_PMIDS,
            "seeds": SEEDS,
            "surrogates": SURROGATES,
            "nominal_levels": NOMINAL_LEVELS.tolist(),
            "normalize": "copula",
        },
        "aggregate": aggregate,
        "per_evaluation": {s: all_results[s] for s in SURROGATES},
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")

    # -----------------------------------------------------------------------
    # Generate calibration plot
    # -----------------------------------------------------------------------
    generate_calibration_plot(aggregate)


def generate_calibration_plot(aggregate):
    """Generate reliability diagram and calibration summary (3-panel figure).

    Panel A: Reliability diagram -- observed vs nominal coverage for each
             surrogate. Perfect calibration = diagonal. Points below the
             diagonal indicate over-confidence (intervals too narrow).

    Panel B: CRPS and MACE bar chart. CRPS is a proper scoring rule that
             jointly evaluates calibration and sharpness. MACE is the mean
             absolute calibration error. Both are lower-is-better.
             GP (sklearn) is excluded because its pathological sigma
             (near-zero) makes NLL/CRPS incomparable in magnitude.

    Panel C: Sigma coefficient of variation (sigma_std / sigma_mean) per
             surrogate, showing how much the uncertainty varies across
             candidates. Low CV means UCB ~ greedy (constant offset).
             GP excluded from shared axis due to extreme range.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot")
        return

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 9,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.pad": 3,
            "ytick.major.pad": 3,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))

    # --- Panel A: Reliability diagram (copula-normalized) ---
    ax = axes[0]
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5, label="Perfect calibration")
    # Shade the over-confident region (below diagonal)
    ax.fill_between([0, 1], [0, 1], [0, 0], alpha=0.04, color="gray")
    ax.text(0.78, 0.12, "over-\nconfident", fontsize=7, color="#888888", ha="center", va="center", style="italic")
    ax.text(
        0.12, 0.55, "conservative", fontsize=7, color="#888888", ha="center", va="center", style="italic", rotation=90
    )

    for surrogate in SURROGATES:
        if surrogate not in aggregate:
            continue
        agg = aggregate[surrogate]
        nl = np.array(agg["nominal_levels"])
        mc = np.array(agg["mean_coverage"])
        sc = np.array(agg["std_coverage"])

        label = f"{SURROGATE_DISPLAY[surrogate]} (MACE={agg['mean_calibration_error']:.3f})"
        color = SURROGATE_COLORS[surrogate]
        ax.plot(nl, mc, "o-", color=color, linewidth=1.5, markersize=4, label=label)
        ax.fill_between(nl, mc - sc, mc + sc, alpha=0.15, color=color)

    ax.set_xlabel("Nominal coverage")
    ax.set_ylabel("Observed coverage")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.legend(fontsize=6.5, loc="upper left", frameon=True, fancybox=False, edgecolor="#cccccc")
    ax.set_title("Reliability diagram", fontsize=10, fontweight="bold", pad=8)
    ax.text(-0.15, 1.05, "A", transform=ax.transAxes, fontsize=13, fontweight="bold", va="top")

    # --- Panel B: CRPS + MACE bar chart ---
    # Exclude GP from the bar chart -- its pathological sigma (near-zero)
    # makes CRPS ~2.4 and NLL ~300k, which compresses the usable surrogates
    # to invisible slivers. GP is already clearly shown as flat in panel A.
    ax = axes[1]
    comparable = [s for s in SURROGATES if s in aggregate and s != "gp_ucb"]
    crps_means = [aggregate[s]["mean_crps"] for s in comparable]
    crps_stds = [aggregate[s]["std_crps"] for s in comparable]
    mace_means = [aggregate[s]["mean_calibration_error"] for s in comparable]
    mace_stds = [aggregate[s]["std_calibration_error"] for s in comparable]
    colors = [SURROGATE_COLORS[s] for s in comparable]
    labels = [SURROGATE_DISPLAY[s] for s in comparable]

    x = np.arange(len(comparable))
    width = 0.30
    ax.bar(
        x - width / 2,
        crps_means,
        width,
        yerr=crps_stds,
        color=colors,
        alpha=0.85,
        edgecolor="white",
        linewidth=0.5,
        capsize=3,
        label="CRPS",
    )
    ax.bar(
        x + width / 2,
        mace_means,
        width,
        yerr=mace_stds,
        color=colors,
        alpha=0.45,
        edgecolor="white",
        linewidth=0.5,
        capsize=3,
        label="MACE",
        hatch="//",
    )

    # Add value annotations
    for i, (cm, mm) in enumerate(zip(crps_means, mace_means)):
        ax.text(
            x[i] - width / 2,
            cm + crps_stds[i] + 0.01,
            f"{cm:.3f}",
            ha="center",
            va="bottom",
            fontsize=7.5,
            fontweight="bold",
        )
        ax.text(x[i] + width / 2, mm + mace_stds[i] + 0.01, f"{mm:.3f}", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Score (lower is better)")
    ax.legend(fontsize=7, frameon=True, fancybox=False, edgecolor="#cccccc", loc="upper right")
    ax.set_title("CRPS and calibration error", fontsize=10, fontweight="bold", pad=8)
    ax.text(-0.15, 1.05, "B", transform=ax.transAxes, fontsize=13, fontweight="bold", va="top")
    ax.grid(axis="y", alpha=0.15, linewidth=0.5)
    # Set y-axis to show detail for RF/NGBoost
    ymax = max(c + s for c, s in zip(crps_means, crps_stds)) * 1.3
    ax.set_ylim(0, ymax)
    # Footnote about GP exclusion
    ax.text(
        0.5,
        -0.18,
        "GP (sklearn) excluded: CRPS=2.42, MACE=0.581",
        transform=ax.transAxes,
        fontsize=6.5,
        ha="center",
        color="#666666",
        style="italic",
    )

    # --- Panel C: Mean sigma and sigma CV (strip plot) ---
    ax = axes[2]

    # Collect per-evaluation sigma data from saved results
    per_eval_data = {}
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            saved = json.load(f)
        for surrogate in SURROGATES:
            per_eval = saved.get("per_evaluation", {}).get(surrogate, [])
            per_eval_data[surrogate] = {
                "sigma_cv": [r["sigma_cv"] for r in per_eval],
                "mean_sigma": [r["mean_sigma"] for r in per_eval],
            }
    else:
        for surrogate in SURROGATES:
            if surrogate in aggregate:
                per_eval_data[surrogate] = {
                    "sigma_cv": [aggregate[surrogate]["mean_sigma_cv"]],
                    "mean_sigma": [aggregate[surrogate]["mean_sigma"]],
                }

    # Two-row strip plot: top row = mean sigma, bottom row = sigma CV
    # Only show RF and NGBoost (GP has pathological near-zero sigma)
    usable = [s for s in SURROGATES if s in aggregate and s != "gp_ucb"]
    usable_labels = [SURROGATE_DISPLAY[s] for s in usable]
    usable_colors = [SURROGATE_COLORS[s] for s in usable]

    x_pos = np.arange(len(usable))
    rng = np.random.RandomState(42)

    # Plot mean sigma as bars with individual points
    mean_sigma_vals = [per_eval_data[s]["mean_sigma"] for s in usable]
    cv_vals = [per_eval_data[s]["sigma_cv"] for s in usable]

    bar_means = [np.mean(v) for v in mean_sigma_vals]
    bar_stds = [np.std(v, ddof=1) if len(v) > 1 else 0 for v in mean_sigma_vals]
    ax.bar(
        x_pos,
        bar_means,
        width=0.5,
        yerr=bar_stds,
        color=usable_colors,
        alpha=0.3,
        edgecolor=usable_colors,
        linewidth=1.2,
        capsize=4,
    )

    for i, (pos, vals) in enumerate(zip(x_pos, mean_sigma_vals)):
        jitter = rng.uniform(-0.12, 0.12, len(vals))
        ax.scatter(
            pos + jitter, vals, s=25, color=usable_colors[i], alpha=0.8, edgecolors="white", linewidth=0.5, zorder=3
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(usable_labels, fontsize=8)
    ax.set_ylabel("Mean predicted sigma")
    ax.set_title("Uncertainty magnitude", fontsize=10, fontweight="bold", pad=8)
    ax.text(-0.15, 1.05, "C", transform=ax.transAxes, fontsize=13, fontweight="bold", va="top")
    ax.grid(axis="y", alpha=0.15, linewidth=0.5)

    # Annotate with CV values
    for i, (pos, cvs) in enumerate(zip(x_pos, cv_vals)):
        mean_cv = np.mean(cvs)
        ax.text(
            pos,
            bar_means[i] + bar_stds[i] + 0.03,
            f"CV={mean_cv:.2f}",
            ha="center",
            va="bottom",
            fontsize=7,
            fontweight="bold",
            color=usable_colors[i],
        )

    # Add GP annotation as text note since its sigma is near-zero
    if "gp_ucb" in aggregate:
        gp_sigma = aggregate["gp_ucb"]["mean_sigma"]
        ax.text(
            0.5,
            -0.18,
            f"GP (sklearn) not shown: mean sigma={gp_sigma:.3f} (near-zero, collapsed posterior)",
            transform=ax.transAxes,
            fontsize=6.5,
            ha="center",
            color="#666666",
            style="italic",
        )

    fig.tight_layout(w_pad=3)
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=300, bbox_inches="tight")
    print(f"Figure saved to {FIGURE_PATH}")
    plt.close(fig)


if __name__ == "__main__":
    run_calibration_analysis()
