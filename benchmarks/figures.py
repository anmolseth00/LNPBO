#!/usr/bin/env python3
"""
Statistical analysis of within-study benchmark results.

Unit of analysis: per-study means (averaging over 5 seeds). The 23 studies are
the independent observations. Seed-level results are replicates within study,
NOT independent samples.

Analyses:
  A. Pairwise Wilcoxon with BH-FDR correction
  B. Effect sizes (Cohen's d, Cliff's delta)
  C. Cluster bootstrap CIs and rank distributions
  D. Linear mixed-effects model
  E. Tanimoto kernel assessment
  F. Family-level analysis
  G. Robustness checks
"""

import json
import sys
import warnings
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multicomp import pairwise_tukeyhsd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "benchmark_results" / "within_study"
FIG_DIR = RESULTS_DIR / "analysis"
FIG_DIR.mkdir(exist_ok=True)

SEEDS = [42, 123, 456, 789, 2024]

STRATEGY_SHORT = {
    "random": "Random",
    "lnpbo_ucb": "LNPBO-UCB",
    "lnpbo_ei": "LNPBO-EI",
    "lnpbo_logei": "LNPBO-LogEI",
    "lnpbo_lp_ei": "LNPBO-LP-EI",
    "lnpbo_lp_logei": "LNPBO-LP-LogEI",
    "lnpbo_pls_logei": "LNPBO-PLS-LogEI",
    "lnpbo_pls_lp_logei": "LNPBO-PLS-LP-LogEI",
    "lnpbo_rkb_logei": "LNPBO-RKB-LogEI",
    "lnpbo_ts_batch": "LNPBO-TS-Batch",
    "lnpbo_gibbon": "LNPBO-GIBBON",
    "lnpbo_tanimoto_ts": "LNPBO-Tani-TS",
    "lnpbo_tanimoto_logei": "LNPBO-Tani-LogEI",
    "casmopolitan_ei": "CASMO-EI",
    "casmopolitan_ucb": "CASMO-UCB",
    "discrete_rf_ucb": "RF-UCB",
    "discrete_rf_ts": "RF-TS",
    "discrete_rf_ts_batch": "RF-TS-Batch",
    "discrete_xgb_ucb": "XGB-UCB",
    "discrete_xgb_greedy": "XGB-Greedy",
    "discrete_xgb_cqr": "XGB-CQR",
    "discrete_xgb_online_conformal": "XGB-OnlineConf",
    "discrete_xgb_ucb_ts_batch": "XGB-UCB-TS-Batch",
    "discrete_ngboost_ucb": "NGBoost-UCB",
    "discrete_deep_ensemble": "DeepEnsemble",
    "discrete_gp_ucb": "GP-UCB-sklearn",
}

STRATEGY_FAMILY = {
    "random": "Random",
    "lnpbo_ucb": "LNPBO-Matern",
    "lnpbo_ei": "LNPBO-Matern",
    "lnpbo_logei": "LNPBO-Matern",
    "lnpbo_lp_ei": "LNPBO-Matern",
    "lnpbo_lp_logei": "LNPBO-Matern",
    "lnpbo_pls_logei": "LNPBO-Matern",
    "lnpbo_pls_lp_logei": "LNPBO-Matern",
    "lnpbo_rkb_logei": "LNPBO-Matern",
    "lnpbo_ts_batch": "LNPBO-Matern",
    "lnpbo_gibbon": "LNPBO-Matern",
    "lnpbo_tanimoto_ts": "LNPBO-Tanimoto",
    "lnpbo_tanimoto_logei": "LNPBO-Tanimoto",
    "casmopolitan_ei": "CASMOPolitan",
    "casmopolitan_ucb": "CASMOPolitan",
    "discrete_rf_ucb": "RF",
    "discrete_rf_ts": "RF",
    "discrete_rf_ts_batch": "RF",
    "discrete_xgb_ucb": "XGBoost",
    "discrete_xgb_greedy": "XGBoost",
    "discrete_xgb_cqr": "XGBoost",
    "discrete_xgb_online_conformal": "XGBoost",
    "discrete_xgb_ucb_ts_batch": "XGBoost",
    "discrete_ngboost_ucb": "NGBoost",
    "discrete_deep_ensemble": "Deep Ensemble",
    "discrete_gp_ucb": "GP-sklearn",
}

# Batch strategy extraction for Tanimoto decomposition
BATCH_STRATEGY = {
    "lnpbo_ts_batch": "TS",
    "lnpbo_tanimoto_ts": "TS",
    "lnpbo_logei": "KB",
    "lnpbo_tanimoto_logei": "KB",
}

KERNEL_TYPE = {
    "lnpbo_ts_batch": "Matern",
    "lnpbo_tanimoto_ts": "Tanimoto",
    "lnpbo_logei": "Matern",
    "lnpbo_tanimoto_logei": "Tanimoto",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data():
    """Load all results, return long-form DataFrame and per-study-mean DataFrame."""
    rows = []
    for pmid_dir in sorted(RESULTS_DIR.iterdir()):
        if not pmid_dir.is_dir() or not pmid_dir.name.isdigit():
            continue
        for f in sorted(pmid_dir.glob("*.json")):
            try:
                d = json.loads(f.read_text())
                rows.append({
                    "pmid": int(d["pmid"]),
                    "strategy": d["strategy"],
                    "seed": d["seed"],
                    "recall5": d["result"]["metrics"]["top_k_recall"]["5"],
                    "study_type": d["study_info"]["study_type"],
                    "n_formulations": d["study_info"]["n_formulations"],
                })
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                print(f"WARNING: {f}: {e}", file=sys.stderr)

    df = pd.DataFrame(rows)

    # Check completeness: require 115 runs per strategy
    counts = df.groupby("strategy").size()
    complete = counts[counts == 115].index.tolist()
    incomplete = counts[counts != 115]
    if len(incomplete) > 0:
        print("Excluded strategies (incomplete):")
        for s, c in incomplete.items():
            print(f"  {s}: {c}/115 runs")
        print()

    df = df[df["strategy"].isin(complete)].copy()

    # Per-study means (the correct unit of analysis)
    study_means = df.groupby(["pmid", "strategy", "study_type", "n_formulations"])[
        "recall5"
    ].mean().reset_index()
    study_means.rename(columns={"recall5": "recall5_mean"}, inplace=True)

    return df, study_means, complete


def section(title):
    w = 100
    print()
    print("=" * w)
    print(f"  {title}")
    print("=" * w)
    print()


def subsection(title):
    print()
    print(f"--- {title} ---")
    print()


# ---------------------------------------------------------------------------
# A. Pairwise Wilcoxon with BH-FDR
# ---------------------------------------------------------------------------

def analysis_a_pairwise_wilcoxon(study_means, strategies):
    section("A. PAIRWISE WILCOXON SIGNED-RANK TESTS WITH BH-FDR CORRECTION")

    pmids = sorted(study_means["pmid"].unique())
    n_studies = len(pmids)
    print(f"Number of studies (independent observations): {n_studies}")
    print(f"Number of strategies: {len(strategies)}")

    # Build per-study mean matrix: rows=studies, cols=strategies
    pivot = study_means.pivot_table(
        index="pmid", columns="strategy", values="recall5_mean"
    )

    pairs = list(combinations(strategies, 2))
    n_pairs = len(pairs)
    print(f"Number of pairwise comparisons: {n_pairs}")

    results = []
    for s1, s2 in pairs:
        v1 = pivot[s1].values
        v2 = pivot[s2].values
        diff = v1 - v2
        nonzero = diff[diff != 0]
        if len(nonzero) >= 5:
            stat, pval = stats.wilcoxon(nonzero)
        else:
            stat, pval = np.nan, 1.0
        results.append({"s1": s1, "s2": s2, "stat": stat, "p_raw": pval})

    res_df = pd.DataFrame(results)

    # BH-FDR correction
    sorted_idx = res_df["p_raw"].argsort().values
    n = len(res_df)
    bh_threshold = np.zeros(n)
    res_df["p_bh"] = 1.0

    sorted_pvals = res_df["p_raw"].values[sorted_idx]
    ranks = np.arange(1, n + 1)
    bh_critical = 0.05 * ranks / n
    # Step-up: find largest k where p_(k) <= k/n * alpha
    bh_adjusted = np.minimum.accumulate(
        (sorted_pvals * n / ranks)[::-1]
    )[::-1]
    bh_adjusted = np.clip(bh_adjusted, 0, 1)
    adjusted = np.empty(n)
    adjusted[sorted_idx] = bh_adjusted
    res_df["p_bh"] = adjusted

    n_sig_raw = (res_df["p_raw"] < 0.05).sum()
    n_sig_bh = (res_df["p_bh"] < 0.05).sum()

    print(f"\nSignificant pairs (raw p < 0.05): {n_sig_raw} / {n_pairs}")
    print(f"Significant pairs (BH-adjusted p < 0.05): {n_sig_bh} / {n_pairs}")

    # Build significance matrix for heatmap
    sig_matrix = pd.DataFrame(np.nan, index=strategies, columns=strategies)
    for _, row in res_df.iterrows():
        sig_matrix.loc[row["s1"], row["s2"]] = row["p_bh"]
        sig_matrix.loc[row["s2"], row["s1"]] = row["p_bh"]
    for s in strategies:
        sig_matrix.loc[s, s] = 0.0

    # Statistically equivalent groupings
    subsection("Statistically Equivalent Groups")
    print("Strategies that cannot be distinguished from each other (BH p >= 0.05):")
    print()

    # Sort strategies by mean recall
    mean_recalls = pivot.mean().sort_values(ascending=False)
    sorted_strats = mean_recalls.index.tolist()

    # Greedy grouping: start from top, group with everything not significantly different
    assigned = set()
    groups = []
    for s in sorted_strats:
        if s in assigned:
            continue
        group = [s]
        assigned.add(s)
        for s2 in sorted_strats:
            if s2 in assigned:
                continue
            # Check if s2 is not significantly different from ALL members of group
            all_equiv = True
            for member in group:
                p = sig_matrix.loc[member, s2]
                if pd.notna(p) and p < 0.05:
                    all_equiv = False
                    break
            if all_equiv:
                group.append(s2)
                assigned.add(s2)
        groups.append(group)

    for i, group in enumerate(groups, 1):
        recalls = [mean_recalls[s] for s in group]
        print(f"  Group {i} (n={len(group)}, recall range: {min(recalls):.3f}-{max(recalls):.3f}):")
        for s in group:
            short = STRATEGY_SHORT.get(s, s)
            print(f"    {short:<25} mean={mean_recalls[s]:.4f}")
        print()

    return res_df, sig_matrix, pivot, mean_recalls


# ---------------------------------------------------------------------------
# B. Effect sizes
# ---------------------------------------------------------------------------

def analysis_b_effect_sizes(pivot, strategies):
    section("B. EFFECT SIZES")

    random_vals = pivot["random"].values
    n = len(random_vals)

    subsection("Cohen's d (paired) vs Random with 95% CI")
    print(f"{'Strategy':<25} {'Mean':>6} {'d':>7} {'d_lo':>7} {'d_hi':>7} {'interp':>10}")
    print("-" * 70)

    cohens_results = []
    for s in sorted(strategies):
        if s == "random":
            continue
        vals = pivot[s].values
        diff = vals - random_vals
        d = np.mean(diff) / np.std(diff, ddof=1)

        # CI via noncentral t approximation
        se_d = np.sqrt(1/n + d**2 / (2*n))
        t_crit = stats.t.ppf(0.975, n - 1)
        d_lo = d - t_crit * se_d
        d_hi = d + t_crit * se_d

        if abs(d) < 0.2:
            interp = "negligible"
        elif abs(d) < 0.5:
            interp = "small"
        elif abs(d) < 0.8:
            interp = "medium"
        else:
            interp = "large"

        short = STRATEGY_SHORT.get(s, s)
        print(f"{short:<25} {np.mean(vals):.4f} {d:>7.3f} {d_lo:>7.3f} {d_hi:>7.3f} {interp:>10}")
        cohens_results.append({"strategy": s, "d": d, "d_lo": d_lo, "d_hi": d_hi})

    subsection("Cliff's Delta for Key Comparisons")
    print("(Non-parametric effect size, range [-1, 1])")
    print()

    def cliffs_delta(x, y):
        """Cliff's delta: proportion of x>y minus proportion of x<y."""
        nx, ny = len(x), len(y)
        count = 0
        for xi in x:
            for yi in y:
                if xi > yi:
                    count += 1
                elif xi < yi:
                    count -= 1
        d = count / (nx * ny)
        # Interpretation
        if abs(d) < 0.147:
            interp = "negligible"
        elif abs(d) < 0.33:
            interp = "small"
        elif abs(d) < 0.474:
            interp = "medium"
        else:
            interp = "large"
        return d, interp

    key_comparisons = [
        ("lnpbo_tanimoto_ts", "lnpbo_ts_batch", "Tanimoto-TS vs Matern-TS"),
        ("lnpbo_tanimoto_logei", "lnpbo_logei", "Tanimoto-LogEI vs Matern-LogEI"),
    ]

    # GP family vs tree family
    gp_strats = [s for s in strategies if STRATEGY_FAMILY.get(s, "").startswith("LNPBO")]
    tree_strats = [s for s in strategies if STRATEGY_FAMILY.get(s, "") in ("RF", "XGBoost", "NGBoost")]

    gp_study_means = pivot[gp_strats].mean(axis=1).values
    tree_study_means = pivot[tree_strats].mean(axis=1).values
    key_comparisons.append(("__gp_family__", "__tree_family__", "GP family vs Tree family"))

    print(f"{'Comparison':<40} {'delta':>7} {'interp':>12}")
    print("-" * 65)

    for s1, s2, label in key_comparisons:
        if s1 == "__gp_family__":
            v1, v2 = gp_study_means, tree_study_means
        else:
            v1 = pivot[s1].values
            v2 = pivot[s2].values
        d, interp = cliffs_delta(v1, v2)
        print(f"{label:<40} {d:>7.3f} {interp:>12}")

    return pd.DataFrame(cohens_results)


# ---------------------------------------------------------------------------
# C. Bootstrap confidence intervals
# ---------------------------------------------------------------------------

def analysis_c_bootstrap(pivot, strategies, n_boot=10000):
    section("C. CLUSTER BOOTSTRAP CONFIDENCE INTERVALS")

    rng = np.random.RandomState(42)
    n_studies = len(pivot)
    study_indices = np.arange(n_studies)

    boot_means = {s: np.zeros(n_boot) for s in strategies}
    boot_ranks = {s: np.zeros(n_boot) for s in strategies}

    for b in range(n_boot):
        idx = rng.choice(study_indices, size=n_studies, replace=True)
        means_this = {}
        for s in strategies:
            means_this[s] = pivot[s].values[idx].mean()
            boot_means[s][b] = means_this[s]

        # Rank (1 = best)
        sorted_strats = sorted(means_this.keys(), key=lambda x: means_this[x], reverse=True)
        for rank, s in enumerate(sorted_strats, 1):
            boot_ranks[s][b] = rank

    subsection("Bootstrap 95% CIs for Mean Top-5% Recall")
    print(f"{'Strategy':<25} {'Mean':>7} {'CI_lo':>7} {'CI_hi':>7} {'Width':>7}")
    print("-" * 60)

    boot_summary = []
    mean_order = sorted(strategies, key=lambda s: pivot[s].mean(), reverse=True)
    for s in mean_order:
        observed = pivot[s].mean()
        ci_lo = np.percentile(boot_means[s], 2.5)
        ci_hi = np.percentile(boot_means[s], 97.5)
        short = STRATEGY_SHORT.get(s, s)
        print(f"{short:<25} {observed:>7.4f} {ci_lo:>7.4f} {ci_hi:>7.4f} {ci_hi - ci_lo:>7.4f}")
        boot_summary.append({
            "strategy": s, "short": short, "mean": observed,
            "ci_lo": ci_lo, "ci_hi": ci_hi
        })

    subsection("Bootstrap Rank Distribution")
    print(f"{'Strategy':<25} {'MeanRank':>8} {'P(#1)':>7} {'P(top3)':>7} {'P(top5)':>7} {'P(top10)':>8}")
    print("-" * 75)

    rank_summary = []
    for s in mean_order:
        ranks = boot_ranks[s]
        short = STRATEGY_SHORT.get(s, s)
        p1 = (ranks == 1).mean()
        p3 = (ranks <= 3).mean()
        p5 = (ranks <= 5).mean()
        p10 = (ranks <= 10).mean()
        mean_rank = ranks.mean()
        print(f"{short:<25} {mean_rank:>8.1f} {p1:>7.3f} {p3:>7.3f} {p5:>7.3f} {p10:>8.3f}")
        rank_summary.append({
            "strategy": s, "short": short, "mean_rank": mean_rank,
            "p_rank1": p1, "p_top3": p3, "p_top5": p5
        })

    return pd.DataFrame(boot_summary), pd.DataFrame(rank_summary), boot_means


# ---------------------------------------------------------------------------
# D. Mixed-effects model
# ---------------------------------------------------------------------------

def analysis_d_mixed_model(df_long, strategies):
    section("D. LINEAR MIXED-EFFECTS MODEL")

    df = df_long[df_long["strategy"].isin(strategies)].copy()
    df["pmid_str"] = df["pmid"].astype(str)
    df["seed_str"] = df["seed"].astype(str)
    df["study_seed"] = df["pmid_str"] + "_" + df["seed_str"]

    subsection("Model: recall5 ~ strategy + (1|study) + (1|study:seed)")

    # statsmodels MixedLM: use strategy as fixed effect, study as random intercept
    # Nested random effect study:seed via variance components
    # MixedLM supports one grouping factor natively; for nested, use VC
    md = smf.mixedlm(
        "recall5 ~ C(strategy, Treatment(reference='random'))",
        df,
        groups=df["pmid_str"],
        re_formula="1",
        vc_formula={"seed": "0 + C(seed_str)"},
    )
    try:
        mdf = md.fit(reml=True)
        print(mdf.summary())
    except Exception as e:
        print(f"Mixed model with VC failed: {e}")
        print("Falling back to simpler model without seed VC...")
        md = smf.mixedlm(
            "recall5 ~ C(strategy, Treatment(reference='random'))",
            df,
            groups=df["pmid_str"],
            re_formula="1",
        )
        mdf = md.fit(reml=True)
        print(mdf.summary())

    # Variance components
    subsection("Variance Components")
    var_study = mdf.cov_re.iloc[0, 0]
    var_resid = mdf.scale

    # Extract seed variance if available
    var_seed = 0.0
    if hasattr(mdf, 'vcomp') and len(mdf.vcomp) > 0:
        var_seed = mdf.vcomp[0]

    var_total = var_study + var_seed + var_resid
    print(f"  Var(study):    {var_study:.6f}  ({100*var_study/var_total:.1f}%)")
    print(f"  Var(seed):     {var_seed:.6f}  ({100*var_seed/var_total:.1f}%)")
    print(f"  Var(residual): {var_resid:.6f}  ({100*var_resid/var_total:.1f}%)")
    print(f"  Var(total):    {var_total:.6f}")
    print()
    print("  Note: strategy variance is captured by the fixed effects, not")
    print("  as a random effect. The residual contains strategy x study")
    print("  interaction and within-cell variability.")

    # Estimated marginal means
    subsection("Estimated Marginal Means (EMMs)")

    # The intercept is the random baseline mean; coefficients are strategy effects
    intercept = mdf.fe_params.iloc[0]
    emms = {}
    for s in strategies:
        if s == "random":
            emms[s] = intercept
        else:
            param_name = f"C(strategy, Treatment(reference='random'))[T.{s}]"
            if param_name in mdf.fe_params.index:
                emms[s] = intercept + mdf.fe_params[param_name]
            else:
                emms[s] = np.nan

    emm_sorted = sorted(emms.items(), key=lambda x: x[1], reverse=True)
    print(f"{'Strategy':<25} {'EMM':>8} {'SE':>8} {'Diff vs Random':>14}")
    print("-" * 60)
    for s, emm in emm_sorted:
        short = STRATEGY_SHORT.get(s, s)
        diff = emm - emms["random"]
        # Get SE from model
        if s == "random":
            se = mdf.bse.iloc[0]
        else:
            param_name = f"C(strategy, Treatment(reference='random'))[T.{s}]"
            se = mdf.bse.get(param_name, np.nan)
        print(f"{short:<25} {emm:>8.4f} {se:>8.4f} {diff:>+14.4f}")

    # Pairwise Tukey HSD on per-study means (since MixedLM pairwise is complex)
    subsection("Tukey HSD on Per-Study Means")
    study_means_for_tukey = df.groupby(["pmid", "strategy"])["recall5"].mean().reset_index()
    tukey = pairwise_tukeyhsd(
        study_means_for_tukey["recall5"],
        study_means_for_tukey["strategy"],
        alpha=0.05,
    )

    # Extract structured results from the Tukey object
    n_sig_tukey = int(tukey.reject.sum())
    n_total_tukey = len(tukey.reject)
    print(f"Significant pairwise differences (Tukey HSD, alpha=0.05): {n_sig_tukey}/{n_total_tukey}")

    # Print only significant pairs
    print("\nSignificant pairs (showing top 30 by |meandiff|):")
    print(f"{'Strategy 1':<25} {'Strategy 2':<25} {'MeanDiff':>9} {'p-adj':>8} {'Reject':>6}")
    print("-" * 80)

    sig_indices = np.where(tukey.reject)[0]
    sig_data = []
    for i in sig_indices:
        s1 = tukey.groupsunique[tukey._multicomp.pairindices[0][i]]
        s2 = tukey.groupsunique[tukey._multicomp.pairindices[1][i]]
        meandiff = tukey.meandiffs[i]
        padj = tukey.pvalues[i]
        sig_data.append((s1, s2, meandiff, padj))
    sig_data.sort(key=lambda r: abs(r[2]), reverse=True)
    for s1, s2, meandiff, padj in sig_data[:30]:
        s1_short = STRATEGY_SHORT.get(s1, s1)
        s2_short = STRATEGY_SHORT.get(s2, s2)
        print(f"{s1_short:<25} {s2_short:<25} {meandiff:>+9.4f} {padj:>8.4f} {'yes':>6}")

    return emms, var_study, var_seed, var_resid


# ---------------------------------------------------------------------------
# E. Tanimoto kernel assessment
# ---------------------------------------------------------------------------

def analysis_e_tanimoto(pivot):
    section("E. TANIMOTO KERNEL ASSESSMENT")

    n_studies = len(pivot)

    tests = [
        ("lnpbo_tanimoto_ts", "lnpbo_ts_batch", "Tanimoto-TS vs Matern-TS"),
        ("lnpbo_tanimoto_logei", "lnpbo_logei", "Tanimoto-LogEI vs Matern-LogEI"),
    ]

    for tani_s, matern_s, label in tests:
        subsection(label)
        tani = pivot[tani_s].values
        matern = pivot[matern_s].values
        diff = tani - matern

        print(f"  Tanimoto mean: {tani.mean():.4f}  (std {tani.std(ddof=1):.4f})")
        print(f"  Matern mean:   {matern.mean():.4f}  (std {matern.std(ddof=1):.4f})")
        print(f"  Mean diff:     {diff.mean():+.4f}  (std {diff.std(ddof=1):.4f})")
        print()

        # Paired t-test
        t_stat, t_pval = stats.ttest_rel(tani, matern)
        print(f"  Paired t-test: t={t_stat:.3f}, p={t_pval:.4f}")

        # Wilcoxon signed-rank
        nonzero = diff[diff != 0]
        if len(nonzero) >= 5:
            w_stat, w_pval = stats.wilcoxon(nonzero)
        else:
            w_stat, w_pval = np.nan, 1.0
        print(f"  Wilcoxon signed-rank: W={w_stat:.1f}, p={w_pval:.4f}")

        # Cohen's d (paired)
        d = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else 0
        se_d = np.sqrt(1/n_studies + d**2 / (2*n_studies))
        t_crit = stats.t.ppf(0.975, n_studies - 1)
        print(f"  Cohen's d: {d:.3f}  95% CI [{d - t_crit*se_d:.3f}, {d + t_crit*se_d:.3f}]")

        # Win/loss/tie
        wins = (diff > 0).sum()
        losses = (diff < 0).sum()
        ties = (diff == 0).sum()
        print(f"  Wins/Losses/Ties: {wins}/{losses}/{ties} (out of {n_studies} studies)")

    # Decompose: kernel effect vs batch effect
    subsection("Factorial Decomposition: Kernel x Batch Strategy")
    print("  2x2 design: {Matern, Tanimoto} x {KB-LogEI, TS-Batch}")
    print()

    # Four cells
    cells = {
        ("Matern", "KB"): "lnpbo_logei",
        ("Matern", "TS"): "lnpbo_ts_batch",
        ("Tanimoto", "KB"): "lnpbo_tanimoto_logei",
        ("Tanimoto", "TS"): "lnpbo_tanimoto_ts",
    }

    print(f"  {'':>15} {'KB-LogEI':>12} {'TS-Batch':>12} {'Row Mean':>12}")
    print("  " + "-" * 55)
    for kernel in ["Matern", "Tanimoto"]:
        kb_mean = pivot[cells[(kernel, "KB")]].mean()
        ts_mean = pivot[cells[(kernel, "TS")]].mean()
        row_mean = (kb_mean + ts_mean) / 2
        print(f"  {kernel:>15} {kb_mean:>12.4f} {ts_mean:>12.4f} {row_mean:>12.4f}")
    col_kb = (pivot[cells[("Matern", "KB")]].mean() + pivot[cells[("Tanimoto", "KB")]].mean()) / 2
    col_ts = (pivot[cells[("Matern", "TS")]].mean() + pivot[cells[("Tanimoto", "TS")]].mean()) / 2
    print(f"  {'Col Mean':>15} {col_kb:>12.4f} {col_ts:>12.4f}")

    # Main effects
    matern_mean = (pivot[cells[("Matern", "KB")]].mean() + pivot[cells[("Matern", "TS")]].mean()) / 2
    tani_mean = (pivot[cells[("Tanimoto", "KB")]].mean() + pivot[cells[("Tanimoto", "TS")]].mean()) / 2
    kernel_effect = tani_mean - matern_mean

    kb_mean_all = (pivot[cells[("Matern", "KB")]].mean() + pivot[cells[("Tanimoto", "KB")]].mean()) / 2
    ts_mean_all = (pivot[cells[("Matern", "TS")]].mean() + pivot[cells[("Tanimoto", "TS")]].mean()) / 2
    batch_effect = ts_mean_all - kb_mean_all

    print()
    print(f"  Kernel main effect (Tanimoto - Matern): {kernel_effect:+.4f}")
    print(f"  Batch main effect  (TS - KB):           {batch_effect:+.4f}")

    # Interaction: difference of differences
    diff_matern = pivot[cells[("Matern", "TS")]].values - pivot[cells[("Matern", "KB")]].values
    diff_tani = pivot[cells[("Tanimoto", "TS")]].values - pivot[cells[("Tanimoto", "KB")]].values
    interaction = diff_tani.mean() - diff_matern.mean()
    print(f"  Interaction (Kernel x Batch):            {interaction:+.4f}")

    # Test kernel effect with paired test (average over batch strategies per study)
    matern_per_study = (pivot[cells[("Matern", "KB")]].values + pivot[cells[("Matern", "TS")]].values) / 2
    tani_per_study = (pivot[cells[("Tanimoto", "KB")]].values + pivot[cells[("Tanimoto", "TS")]].values) / 2
    kernel_diff = tani_per_study - matern_per_study
    t_k, p_k = stats.ttest_rel(tani_per_study, matern_per_study)
    w_nz = kernel_diff[kernel_diff != 0]
    if len(w_nz) >= 5:
        _, p_k_w = stats.wilcoxon(w_nz)
    else:
        p_k_w = 1.0
    d_k = kernel_diff.mean() / kernel_diff.std(ddof=1) if kernel_diff.std(ddof=1) > 0 else 0
    print()
    print(f"  Kernel effect test (controlling for batch):")
    print(f"    Paired t: t={t_k:.3f}, p={p_k:.4f}")
    print(f"    Wilcoxon: p={p_k_w:.4f}")
    print(f"    Cohen's d: {d_k:.3f}")

    # Test batch effect
    kb_per_study = (pivot[cells[("Matern", "KB")]].values + pivot[cells[("Tanimoto", "KB")]].values) / 2
    ts_per_study = (pivot[cells[("Matern", "TS")]].values + pivot[cells[("Tanimoto", "TS")]].values) / 2
    batch_diff = ts_per_study - kb_per_study
    t_b, p_b = stats.ttest_rel(ts_per_study, kb_per_study)
    b_nz = batch_diff[batch_diff != 0]
    if len(b_nz) >= 5:
        _, p_b_w = stats.wilcoxon(b_nz)
    else:
        p_b_w = 1.0
    d_b = batch_diff.mean() / batch_diff.std(ddof=1) if batch_diff.std(ddof=1) > 0 else 0
    print()
    print(f"  Batch effect test (controlling for kernel):")
    print(f"    Paired t: t={t_b:.3f}, p={p_b:.4f}")
    print(f"    Wilcoxon: p={p_b_w:.4f}")
    print(f"    Cohen's d: {d_b:.3f}")


# ---------------------------------------------------------------------------
# F. Family-level analysis
# ---------------------------------------------------------------------------

def analysis_f_families(df_long, pivot, strategies):
    section("F. FAMILY-LEVEL ANALYSIS")

    pmids = sorted(pivot.index)
    families_ordered = ["Random", "LNPBO-Matern", "LNPBO-Tanimoto", "RF", "XGBoost",
                        "NGBoost", "CASMOPolitan", "Deep Ensemble", "GP-sklearn"]

    # Compute family-level per-study means
    family_study_means = {}
    for fam in families_ordered:
        fam_strats = [s for s in strategies if STRATEGY_FAMILY.get(s) == fam]
        if not fam_strats:
            continue
        # Average across strategies within family, then we have one value per study
        family_study_means[fam] = pivot[fam_strats].mean(axis=1).values

    subsection("Family-Level Descriptive Statistics")
    print(f"{'Family':<20} {'Mean':>7} {'Std':>7} {'Median':>7} {'N_strats':>8}")
    print("-" * 55)
    for fam in families_ordered:
        if fam not in family_study_means:
            continue
        vals = family_study_means[fam]
        n_strats = sum(1 for s in strategies if STRATEGY_FAMILY.get(s) == fam)
        print(f"{fam:<20} {vals.mean():>7.4f} {vals.std(ddof=1):>7.4f} {np.median(vals):>7.4f} {n_strats:>8}")

    # Bootstrap CIs for families
    subsection("Family Bootstrap 95% CIs (10,000 resamples of studies)")
    rng = np.random.RandomState(42)
    n_boot = 10000
    n_studies = len(pmids)

    print(f"{'Family':<20} {'Mean':>7} {'CI_lo':>7} {'CI_hi':>7} {'Lift vs Random':>14}")
    print("-" * 60)

    random_mean = family_study_means.get("Random", np.zeros(n_studies)).mean()
    family_boot = {}
    for fam in families_ordered:
        if fam not in family_study_means:
            continue
        vals = family_study_means[fam]
        boot = np.zeros(n_boot)
        for b in range(n_boot):
            idx = rng.choice(n_studies, size=n_studies, replace=True)
            boot[b] = vals[idx].mean()
        ci_lo = np.percentile(boot, 2.5)
        ci_hi = np.percentile(boot, 97.5)
        lift = vals.mean() / random_mean if random_mean > 0 and fam != "Random" else np.nan
        lift_str = f"{lift:.2f}x" if fam != "Random" else "---"
        print(f"{fam:<20} {vals.mean():>7.4f} {ci_lo:>7.4f} {ci_hi:>7.4f} {lift_str:>14}")
        family_boot[fam] = boot

    # Pairwise family comparisons with FDR
    subsection("Pairwise Family Comparisons (Wilcoxon, BH-FDR)")
    fam_list = [f for f in families_ordered if f in family_study_means]
    fam_pairs = list(combinations(fam_list, 2))

    fam_results = []
    for f1, f2 in fam_pairs:
        v1 = family_study_means[f1]
        v2 = family_study_means[f2]
        diff = v1 - v2
        nonzero = diff[diff != 0]
        if len(nonzero) >= 5:
            _, pval = stats.wilcoxon(nonzero)
        else:
            pval = 1.0
        fam_results.append({"f1": f1, "f2": f2, "p_raw": pval,
                            "mean_diff": diff.mean()})

    fam_df = pd.DataFrame(fam_results)
    # BH correction
    n_fam = len(fam_df)
    sorted_idx = fam_df["p_raw"].argsort().values
    sorted_pvals = fam_df["p_raw"].values[sorted_idx]
    ranks = np.arange(1, n_fam + 1)
    bh_adjusted = np.minimum.accumulate((sorted_pvals * n_fam / ranks)[::-1])[::-1]
    bh_adjusted = np.clip(bh_adjusted, 0, 1)
    adjusted = np.empty(n_fam)
    adjusted[sorted_idx] = bh_adjusted
    fam_df["p_bh"] = adjusted

    print(f"{'Family 1':<20} {'Family 2':<20} {'MeanDiff':>9} {'p_raw':>8} {'p_BH':>8} {'Sig':>4}")
    print("-" * 75)
    for _, row in fam_df.sort_values("p_bh").iterrows():
        sig = "***" if row["p_bh"] < 0.001 else ("**" if row["p_bh"] < 0.01 else ("*" if row["p_bh"] < 0.05 else ""))
        print(f"{row['f1']:<20} {row['f2']:<20} {row['mean_diff']:>+9.4f} {row['p_raw']:>8.4f} {row['p_bh']:>8.4f} {sig:>4}")

    return family_study_means, family_boot


# ---------------------------------------------------------------------------
# G. Robustness checks
# ---------------------------------------------------------------------------

def analysis_g_robustness(df_long, pivot, strategies):
    section("G. ROBUSTNESS CHECKS")

    pmids = sorted(pivot.index)
    n_studies = len(pmids)

    # G1. Leave-one-study-out
    subsection("G1. Leave-One-Study-Out Stability (Top-5 Strategies)")

    mean_recalls = pivot.mean().sort_values(ascending=False)
    top5 = mean_recalls.index[:5].tolist()

    print(f"Top-5 strategies: {[STRATEGY_SHORT.get(s, s) for s in top5]}")
    print()

    print(f"{'Dropped Study':>15} ", end="")
    for s in top5:
        print(f"{STRATEGY_SHORT.get(s, s):>18}", end="")
    print(f"  {'Rank Changes':>15}")
    print("-" * (15 + 18 * len(top5) + 17))

    rank_changes = 0
    original_order = list(range(len(top5)))

    for pmid in pmids:
        remaining = pivot.drop(pmid)
        loo_means = {s: remaining[s].mean() for s in top5}
        loo_sorted = sorted(loo_means.keys(), key=lambda x: loo_means[x], reverse=True)
        new_order = [top5.index(s) for s in loo_sorted]

        changed = "RERANK" if new_order != original_order else ""
        if changed:
            rank_changes += 1

        print(f"{pmid:>15} ", end="")
        for s in top5:
            print(f"{loo_means[s]:>18.4f}", end="")
        print(f"  {changed:>15}")

    print()
    print(f"  Rankings changed in {rank_changes}/{n_studies} LOO iterations")
    stability = 1 - rank_changes / n_studies
    print(f"  Ranking stability: {stability:.1%}")

    # G2. Seed sensitivity
    subsection("G2. Seed Sensitivity (Kruskal-Wallis per Strategy)")
    print("If p < 0.05, seed choice significantly affects results (red flag).")
    print()

    df = df_long[df_long["strategy"].isin(strategies)].copy()

    print(f"{'Strategy':<25} {'KW_stat':>8} {'p-value':>8} {'Flag':>6}")
    print("-" * 55)
    n_flags = 0
    for s in sorted(strategies):
        groups = [g["recall5"].values for _, g in df[df["strategy"] == s].groupby("seed")]
        if len(groups) >= 2 and all(len(g) > 0 for g in groups):
            h_stat, p_val = stats.kruskal(*groups)
            flag = "FLAG" if p_val < 0.05 else ""
            if flag:
                n_flags += 1
            short = STRATEGY_SHORT.get(s, s)
            print(f"{short:<25} {h_stat:>8.3f} {p_val:>8.4f} {flag:>6}")

    print()
    if n_flags == 0:
        print("  No strategies show significant seed sensitivity. Good.")
    else:
        print(f"  {n_flags} strategies show significant seed sensitivity!")

    # G3. Normality of per-study means
    subsection("G3. Shapiro-Wilk Normality Test on Per-Study Means")
    print("Testing whether the distribution of per-study means is normal for each strategy.")
    print("Non-normality (p < 0.05) would question parametric test assumptions.")
    print()

    print(f"{'Strategy':<25} {'W':>8} {'p-value':>8} {'Normal?':>8}")
    print("-" * 55)
    n_non_normal = 0
    for s in sorted(strategies):
        vals = pivot[s].values
        if len(vals) >= 3:
            w_stat, p_val = stats.shapiro(vals)
            normal = "yes" if p_val >= 0.05 else "NO"
            if p_val < 0.05:
                n_non_normal += 1
            short = STRATEGY_SHORT.get(s, s)
            print(f"{short:<25} {w_stat:>8.4f} {p_val:>8.4f} {normal:>8}")

    print()
    print(f"  {n_non_normal}/{len(strategies)} strategies have non-normal per-study means")
    if n_non_normal > 0:
        print("  This justifies using Wilcoxon (non-parametric) rather than t-tests.")
    else:
        print("  All distributions appear normal; parametric tests are valid.")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def make_figures(pivot, strategies, boot_summary, sig_matrix,
                 var_study, var_seed, var_resid, family_study_means):
    """Create figures."""

    sns.set_style("whitegrid")
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 150,
    })

    mean_order = boot_summary.sort_values("mean", ascending=True)

    # --- Figure 1: Forest plot with bootstrap CIs ---
    fig, ax = plt.subplots(figsize=(8, 10))
    y_pos = np.arange(len(mean_order))

    colors = []
    for _, row in mean_order.iterrows():
        fam = STRATEGY_FAMILY.get(row["strategy"], "Other")
        cmap = {
            "Random": "#999999",
            "LNPBO-Matern": "#1f77b4",
            "LNPBO-Tanimoto": "#17becf",
            "RF": "#2ca02c",
            "XGBoost": "#ff7f0e",
            "NGBoost": "#d62728",
            "CASMOPolitan": "#9467bd",
            "Deep Ensemble": "#8c564b",
            "GP-sklearn": "#e377c2",
        }
        colors.append(cmap.get(fam, "#333333"))

    for i, (_, row) in enumerate(mean_order.iterrows()):
        ax.plot([row["ci_lo"], row["ci_hi"]], [i, i], color=colors[i], linewidth=2, solid_capstyle="round")
        ax.plot(row["mean"], i, "o", color=colors[i], markersize=7, zorder=5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(mean_order["short"].values)
    ax.set_xlabel("Top-5% Recall (Mean with Bootstrap 95% CI)")
    ax.set_title("Strategy Rankings with Bootstrap 95% Confidence Intervals")
    ax.axvline(x=mean_order[mean_order["strategy"] == "random"]["mean"].values[0],
               color="#999999", linestyle="--", linewidth=1, alpha=0.7, label="Random baseline")
    ax.legend(loc="lower right")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig1_strategy_rankings_bootstrap_ci.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig1_strategy_rankings_bootstrap_ci.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {FIG_DIR / 'fig1_strategy_rankings_bootstrap_ci.png'}")

    # --- Figure 2: Pairwise significance heatmap ---
    mean_recalls = pivot.mean().sort_values(ascending=False)
    sorted_strats = mean_recalls.index.tolist()
    short_labels = [STRATEGY_SHORT.get(s, s) for s in sorted_strats]

    sig_sorted = sig_matrix.loc[sorted_strats, sorted_strats].copy()
    # Convert to binary: 1 = significant, 0 = not significant, NaN = self
    binary_mat = (sig_sorted < 0.05).astype(float)
    np.fill_diagonal(binary_mat.values, np.nan)

    fig, ax = plt.subplots(figsize=(14, 12))
    mask = np.eye(len(sorted_strats), dtype=bool)
    sns.heatmap(
        binary_mat.values,
        xticklabels=short_labels,
        yticklabels=short_labels,
        cmap=["#f0f0f0", "#d62728"],
        mask=mask,
        square=True,
        cbar_kws={"label": "Significantly different (BH p<0.05)", "shrink": 0.3},
        linewidths=0.5,
        linecolor="white",
        ax=ax,
    )
    ax.set_title("Pairwise Significance Matrix (BH-FDR corrected, alpha=0.05)")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig2_pairwise_significance_heatmap.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig2_pairwise_significance_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {FIG_DIR / 'fig2_pairwise_significance_heatmap.png'}")

    # --- Figure 3: Variance decomposition ---
    # Strategy variance: compute SS for strategy from per-study means
    all_vals = []
    for s in strategies:
        for v in pivot[s].values:
            all_vals.append(v)
    grand_mean = np.mean(all_vals)
    ss_total = np.sum((np.array(all_vals) - grand_mean) ** 2)

    # Use the mixed model variance components + compute strategy SS from fixed effects
    strat_means_arr = np.array([pivot[s].mean() for s in strategies])
    ss_strategy = len(pivot) * np.sum((strat_means_arr - grand_mean) ** 2)

    study_means_arr = np.array([pivot.loc[p].mean() for p in pivot.index])
    ss_study_direct = len(strategies) * np.sum((study_means_arr - grand_mean) ** 2)

    ss_residual_direct = ss_total - ss_study_direct - ss_strategy

    labels = ["Study", "Strategy", "Residual\n(Study x Strategy\n+ Seed)"]
    sizes = [ss_study_direct / ss_total, ss_strategy / ss_total, ss_residual_direct / ss_total]
    sizes = [max(s, 0) for s in sizes]
    colors_pie = ["#1f77b4", "#ff7f0e", "#d3d3d3"]

    fig, ax = plt.subplots(figsize=(7, 7))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, autopct="%1.1f%%",
        colors=colors_pie, startangle=90,
        textprops={"fontsize": 12},
        pctdistance=0.6,
    )
    for t in autotexts:
        t.set_fontsize(13)
        t.set_fontweight("bold")
    ax.set_title("Variance Decomposition of Top-5% Recall\n(Type III SS, additive model)", fontsize=14)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig3_variance_decomposition.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig3_variance_decomposition.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {FIG_DIR / 'fig3_variance_decomposition.png'}")

    # --- Figure 4: Tanimoto vs Matern paired comparison ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    comparisons = [
        ("lnpbo_tanimoto_ts", "lnpbo_ts_batch", "TS-Batch"),
        ("lnpbo_tanimoto_logei", "lnpbo_logei", "KB-LogEI"),
    ]

    for ax, (tani_s, matern_s, acq_label) in zip(axes, comparisons):
        tani = pivot[tani_s].values
        matern = pivot[matern_s].values

        ax.scatter(matern, tani, s=40, alpha=0.7, color="#1f77b4", edgecolors="white", linewidth=0.5)
        lims = [min(matern.min(), tani.min()) - 0.05, max(matern.max(), tani.max()) + 0.05]
        ax.plot(lims, lims, "k--", alpha=0.5, linewidth=1, label="y = x")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel(f"Matern {acq_label}")
        ax.set_ylabel(f"Tanimoto {acq_label}")
        ax.set_title(f"Tanimoto vs Matern Kernel ({acq_label})")
        ax.set_aspect("equal")
        ax.legend(loc="upper left")

        # Annotate with stats
        diff = tani - matern
        d = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else 0
        nonzero = diff[diff != 0]
        if len(nonzero) >= 5:
            _, pval = stats.wilcoxon(nonzero)
        else:
            pval = 1.0
        wins = (diff > 0).sum()
        losses = (diff < 0).sum()
        ax.text(0.95, 0.05,
                f"d = {d:.3f}\np = {pval:.3f}\nW/L = {wins}/{losses}",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig4_tanimoto_vs_matern.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig4_tanimoto_vs_matern.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {FIG_DIR / 'fig4_tanimoto_vs_matern.png'}")

    # --- Figure 5: Family-level comparison with bootstrap CIs ---
    families_ordered = ["NGBoost", "RF", "CASMOPolitan", "XGBoost", "Deep Ensemble",
                        "GP-sklearn", "LNPBO-Tanimoto", "LNPBO-Matern", "Random"]

    fig, ax = plt.subplots(figsize=(8, 5))
    fam_data = []
    for fam in families_ordered:
        if fam in family_study_means:
            vals = family_study_means[fam]
            # Bootstrap CI
            rng = np.random.RandomState(42)
            boot = np.array([vals[rng.choice(len(vals), size=len(vals), replace=True)].mean()
                             for _ in range(10000)])
            fam_data.append((fam, vals.mean(), np.percentile(boot, 2.5), np.percentile(boot, 97.5)))

    fam_data.sort(key=lambda x: x[1])

    fam_colors = {
        "Random": "#999999", "LNPBO-Matern": "#1f77b4", "LNPBO-Tanimoto": "#17becf",
        "RF": "#2ca02c", "XGBoost": "#ff7f0e", "NGBoost": "#d62728",
        "CASMOPolitan": "#9467bd", "Deep Ensemble": "#8c564b", "GP-sklearn": "#e377c2",
    }

    y_pos = np.arange(len(fam_data))
    for i, (fam, mean, lo, hi) in enumerate(fam_data):
        c = fam_colors.get(fam, "#333333")
        ax.plot([lo, hi], [i, i], color=c, linewidth=3, solid_capstyle="round")
        ax.plot(mean, i, "o", color=c, markersize=9, zorder=5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([f[0] for f in fam_data])
    ax.set_xlabel("Top-5% Recall (Mean with Bootstrap 95% CI)")
    ax.set_title("Strategy Family Rankings")
    random_mean = [f[1] for f in fam_data if f[0] == "Random"]
    if random_mean:
        ax.axvline(random_mean[0], color="#999999", linestyle="--", linewidth=1, alpha=0.7)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig5_family_rankings.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig5_family_rankings.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {FIG_DIR / 'fig5_family_rankings.png'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 100)
    print("  STATISTICAL ANALYSIS OF WITHIN-STUDY BENCHMARK")
    print("=" * 100)
    print()

    df_long, study_means, strategies = load_data()
    print(f"Loaded {len(df_long)} seed-level observations")
    print(f"  {len(study_means['pmid'].unique())} studies, {len(strategies)} complete strategies")
    print(f"  {len(study_means)} per-study mean observations")
    print(f"  Strategies: {len(strategies)}")
    print()

    # Pivot table: rows = studies, columns = strategies, values = per-study mean recall
    pivot = study_means.pivot_table(
        index="pmid", columns="strategy", values="recall5_mean"
    )
    print(f"Pivot table shape: {pivot.shape} (studies x strategies)")

    # A. Pairwise Wilcoxon with BH-FDR
    res_df, sig_matrix, pivot, mean_recalls = analysis_a_pairwise_wilcoxon(study_means, strategies)

    # B. Effect sizes
    cohens_df = analysis_b_effect_sizes(pivot, strategies)

    # C. Bootstrap CIs
    boot_summary, rank_summary, boot_means = analysis_c_bootstrap(pivot, strategies)

    # D. Mixed-effects model
    emms, var_study, var_seed, var_resid = analysis_d_mixed_model(df_long, strategies)

    # E. Tanimoto kernel assessment
    analysis_e_tanimoto(pivot)

    # F. Family-level analysis
    family_study_means, family_boot = analysis_f_families(df_long, pivot, strategies)

    # G. Robustness checks
    analysis_g_robustness(df_long, pivot, strategies)

    # Figures
    section("FIGURES")
    make_figures(pivot, strategies, boot_summary, sig_matrix,
                 var_study, var_seed, var_resid, family_study_means)

    print()
    print("=" * 100)
    print("  ANALYSIS COMPLETE")
    print(f"  Figures saved to: {FIG_DIR}")
    print("=" * 100)


if __name__ == "__main__":
    main()
