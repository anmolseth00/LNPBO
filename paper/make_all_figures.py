#!/usr/bin/env python3
"""Generate all publication figures for the LNPBO paper.

Uses unified style from figure_style.py. Generates figures sized for
ACS JCIM (single column: 3.25in, double column: 7in).

Usage:
    cd paper && python make_all_figures.py
    # or from repo root:
    python paper/make_all_figures.py
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from scipy import stats as sp_stats

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

from paper.figure_style import (
    DOUBLE_COL,
    ENCODING_COLORS,
    ENCODING_DISPLAY,
    FAMILY_COLORS,
    FAMILY_ORDER,
    SINGLE_COL,
    STUDY_TYPE_COLORS,
    bootstrap_ci,
    light_ygrid,
    panel_label,
    save_figure,
    setup_style,
)

RESULTS_DIR = REPO / "benchmark_results" / "within_study"
ABLATION_DIR = REPO / "benchmark_results" / "ablations"
CROSS_STUDY_DIR = REPO / "benchmark_results" / "cross_study_transfer"
STUDIES_JSON = REPO / "experiments" / "data_integrity" / "studies.json"
FIG_DIR = HERE / "figures"

SEEDS = [42, 123, 456, 789, 2024]

STRATEGY_FAMILY = {
    "random": "Random",
    "lnpbo_ucb": "GP (BoTorch)",
    "lnpbo_ei": "GP (BoTorch)",
    "lnpbo_logei": "GP (BoTorch)",
    "lnpbo_lp_ei": "GP (BoTorch)",
    "lnpbo_lp_logei": "GP (BoTorch)",
    "lnpbo_pls_logei": "GP (BoTorch)",
    "lnpbo_pls_lp_logei": "GP (BoTorch)",
    "lnpbo_rkb_logei": "GP (BoTorch)",
    "lnpbo_ts_batch": "GP (BoTorch)",
    "lnpbo_gibbon": "GP (BoTorch)",
    "lnpbo_tanimoto_ts": "GP (BoTorch)",
    "lnpbo_tanimoto_logei": "GP (BoTorch)",
    "lnpbo_aitchison_ts": "GP (BoTorch)",
    "lnpbo_aitchison_logei": "GP (BoTorch)",
    "lnpbo_dkl_ts": "GP (BoTorch)",
    "lnpbo_dkl_logei": "GP (BoTorch)",
    "lnpbo_rf_kernel_ts": "GP (BoTorch)",
    "lnpbo_rf_kernel_logei": "GP (BoTorch)",
    "lnpbo_compositional_ts": "GP (BoTorch)",
    "lnpbo_compositional_logei": "GP (BoTorch)",
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
    "discrete_gp_ucb": "GP (sklearn)",
}

STRATEGY_SHORT = {
    "random": "Random",
    "lnpbo_ucb": "GP-UCB",
    "lnpbo_ei": "GP-EI",
    "lnpbo_logei": "GP-LogEI",
    "lnpbo_lp_ei": "GP-LP-EI",
    "lnpbo_lp_logei": "GP-LP-LogEI",
    "lnpbo_pls_logei": "GP-PLS-LogEI",
    "lnpbo_pls_lp_logei": "GP-PLS-LP-LogEI",
    "lnpbo_rkb_logei": "GP-RKB-LogEI",
    "lnpbo_ts_batch": "GP-TS-Batch",
    "lnpbo_gibbon": "GP-GIBBON",
    "lnpbo_tanimoto_ts": "GP-Tani-TS",
    "lnpbo_tanimoto_logei": "GP-Tani-LogEI",
    "lnpbo_aitchison_ts": "GP-Aitchison-TS",
    "lnpbo_aitchison_logei": "GP-Aitchison-LogEI",
    "lnpbo_dkl_ts": "GP-DKL-TS",
    "lnpbo_dkl_logei": "GP-DKL-LogEI",
    "lnpbo_rf_kernel_ts": "GP-RFKernel-TS",
    "lnpbo_rf_kernel_logei": "GP-RFKernel-LogEI",
    "lnpbo_compositional_ts": "GP-Comp-TS",
    "lnpbo_compositional_logei": "GP-Comp-LogEI",
    "casmopolitan_ei": "CASMO-EI",
    "casmopolitan_ucb": "CASMO-UCB",
    "discrete_rf_ucb": "RF-UCB",
    "discrete_rf_ts": "RF-TS",
    "discrete_rf_ts_batch": "RF-TS-Batch",
    "discrete_xgb_ucb": "XGB-UCB",
    "discrete_xgb_greedy": "XGB-Greedy",
    "discrete_xgb_cqr": "XGB-CQR",
    "discrete_xgb_online_conformal": "XGB-OnlineConf",
    "discrete_xgb_ucb_ts_batch": "XGB-TS-Batch",
    "discrete_ngboost_ucb": "NGBoost-UCB",
    "discrete_deep_ensemble": "DeepEnsemble",
    "discrete_gp_ucb": "GP-sklearn",
}

STUDY_TYPE_LABELS = {
    "il_diverse_fixed_ratios": "IL-diverse, fixed ratios",
    "il_diverse_variable_ratios": "IL-diverse, variable ratios",
    "ratio_only": "Ratio-only",
}
STUDY_TYPE_ORDER = list(STUDY_TYPE_LABELS.keys())


# ═══════════════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════════════


def load_study_metadata():
    with open(STUDIES_JSON) as f:
        studies = json.load(f)
    by_id, by_pmid = {}, {}
    for s in studies:
        by_id[s["study_id"]] = s
        pmid = str(s["pmid"])
        if pmid not in by_pmid:
            by_pmid[pmid] = s
        else:
            existing = by_pmid[pmid]
            if existing.get("suffix") is None:
                pass
            elif s.get("suffix") is None or s["n_formulations"] > existing["n_formulations"]:
                by_pmid[pmid] = s
    return by_id, by_pmid


def load_all_results():
    results = []
    for pmid_dir in sorted(RESULTS_DIR.iterdir()):
        if not pmid_dir.is_dir() or not pmid_dir.name[0].isdigit():
            continue
        for f in sorted(pmid_dir.glob("*.json")):
            try:
                results.append(json.loads(f.read_text()))
            except (json.JSONDecodeError, KeyError):
                continue
    return results


def build_tables(results):
    study_info, result_map = {}, {}
    strategies = set()
    for r in results:
        study_id = r.get("study_id", str(int(r["pmid"])))
        study_info[study_id] = r["study_info"]
        result_map[(study_id, r["strategy"], r["seed"])] = r
        strategies.add(r["strategy"])
    return study_info, result_map, sorted(study_info.keys()), sorted(strategies)


def get_top5(result_map, pmid, strategy, seed):
    key = (pmid, strategy, seed)
    if key not in result_map:
        return None
    try:
        return result_map[key]["result"]["metrics"]["top_k_recall"]["5"]
    except (KeyError, TypeError):
        return None


def family_study_means(result_map, pmids, strategies):
    raw = defaultdict(lambda: defaultdict(list))
    for s in strategies:
        fam = STRATEGY_FAMILY.get(s, "Other")
        for pmid in pmids:
            for seed in SEEDS:
                v = get_top5(result_map, pmid, s, seed)
                if v is not None:
                    raw[fam][pmid].append(v)
    return {(fam, pmid): np.mean(vals) for fam in raw for pmid, vals in raw[fam].items()}


# ═══════════════════════════════════════════════════════════════════════════
# Figure 1: Strategy Family Rankings (Forest Plot)
# ═══════════════════════════════════════════════════════════════════════════


def fig_family_rankings(result_map, pmids, strategies, study_info):
    """Horizontal forest plot: family-level mean top-5% recall with 95% CI."""
    fam_means = family_study_means(result_map, pmids, strategies)

    families = FAMILY_ORDER
    data = {}
    for fam in families:
        vals = [fam_means[(fam, p)] for p in pmids if (fam, p) in fam_means]
        if vals:
            m = np.mean(vals)
            ci = bootstrap_ci(vals)
            data[fam] = (m, ci[0], ci[1], len(vals))

    # Sort by mean (best on top)
    ordered = sorted(data.keys(), key=lambda f: data[f][0])

    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.8))

    y = np.arange(len(ordered))
    for i, fam in enumerate(ordered):
        m, lo, hi, _n = data[fam]
        color = FAMILY_COLORS.get(fam, "#888888")
        ax.plot(m, i, "o", color=color, ms=5, zorder=5)
        ax.plot([lo, hi], [i, i], "-", color=color, lw=1.5, zorder=4, solid_capstyle="round")
        ax.text(hi + 0.008, i, f"{m:.3f}", va="center", fontsize=6, color="#333333")

    ax.set_yticks(y)
    ax.set_yticklabels(ordered, fontsize=6.5)
    ax.set_xlabel("Mean Top-5% Recall")
    ax.set_xlim(0.45, 0.85)
    light_ygrid(ax)

    # Random baseline
    if "Random" in data:
        ax.axvline(data["Random"][0], color="#000000", ls=":", lw=0.5, alpha=0.5)

    save_figure(fig, FIG_DIR / "fig_family_rankings.pdf")
    plt.close(fig)


def fig_strategy_rankings(result_map, pmids, strategies):
    """Forest plot: per-strategy mean top-5% recall with bootstrap 95% CIs.

    This is the main Figure 1 in the paper showing all 34 individual strategies.
    """
    # Compute per-strategy study-level means, then bootstrap across studies
    strat_data = {}
    for s in strategies:
        study_means = []
        for pmid in pmids:
            vals = [get_top5(result_map, pmid, s, seed) for seed in SEEDS]
            vals = [v for v in vals if v is not None]
            if vals:
                study_means.append(np.mean(vals))
        if study_means:
            m = np.mean(study_means)
            ci = bootstrap_ci(study_means)
            strat_data[s] = (m, ci[0], ci[1])

    # Sort by mean (ascending, so best on top of horizontal plot)
    ordered = sorted(strat_data.keys(), key=lambda s: strat_data[s][0])

    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 6.5))
    y = np.arange(len(ordered))

    for i, s in enumerate(ordered):
        m, lo, hi = strat_data[s]
        fam = STRATEGY_FAMILY.get(s, "Other")
        color = FAMILY_COLORS.get(fam, "#888888")
        ax.plot(m, i, "o", color=color, ms=4.5, zorder=5)
        ax.plot([lo, hi], [i, i], "-", color=color, lw=1.2, zorder=4, solid_capstyle="round")
        ax.text(hi + 0.005, i, f"{m:.3f}", va="center", fontsize=5.5, color="#333333")

    short_labels = [STRATEGY_SHORT.get(s, s) for s in ordered]
    ax.set_yticks(y)
    ax.set_yticklabels(short_labels, fontsize=6)
    ax.set_xlabel("Top-5% Recall (mean across studies)")
    ax.set_xlim(0.40, 0.88)
    light_ygrid(ax)

    # Random baseline
    if "random" in strat_data:
        ax.axvline(strat_data["random"][0], color="#999999", ls="--", lw=0.7, alpha=0.6)

    # Legend: one entry per family
    from matplotlib.lines import Line2D

    seen = set()
    handles = []
    for s in ordered:
        fam = STRATEGY_FAMILY.get(s, "Other")
        if fam not in seen:
            seen.add(fam)
            handles.append(
                Line2D([0], [0], marker="o", color="w", markerfacecolor=FAMILY_COLORS.get(fam, "#888"), ms=5, label=fam)
            )
    ax.legend(handles=handles, fontsize=5.5, loc="lower right", framealpha=0.9, ncol=2)

    fig.tight_layout()
    save_figure(fig, FIG_DIR / "fig_strategy_rankings.pdf")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 2: Critical Difference Diagram
# ═══════════════════════════════════════════════════════════════════════════


def fig_critical_difference(result_map, pmids, strategies, study_info):
    """Nemenyi post-hoc CD diagram."""
    fam_means = family_study_means(result_map, pmids, strategies)
    families = [f for f in FAMILY_ORDER if f != "Random"]

    rank_rows = []
    for pmid in pmids:
        vals = []
        ok = True
        for fam in families:
            v = fam_means.get((fam, pmid))
            if v is None:
                ok = False
                break
            vals.append(v)
        if not ok:
            continue
        order = np.argsort(-np.array(vals))
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, len(vals) + 1)
        # Handle ties
        for i in range(len(vals)):
            tied = [j for j in range(len(vals)) if vals[j] == vals[i]]
            if len(tied) > 1:
                avg = np.mean([ranks[j] for j in tied])
                for j in tied:
                    ranks[j] = avg
        rank_rows.append(ranks)

    rank_matrix = np.array(rank_rows)
    n_studies, k = rank_matrix.shape
    mean_ranks = np.mean(rank_matrix, axis=0)

    # Friedman
    chi2 = (12 * n_studies / (k * (k + 1))) * (np.sum(mean_ranks**2) - k * ((k + 1) ** 2) / 4)
    p_friedman = 1 - sp_stats.chi2.cdf(chi2, k - 1)

    # Nemenyi CD
    q_table = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}
    q = q_table.get(k, 2.326 + 0.154 * (k - 2))
    cd = q * np.sqrt(k * (k + 1) / (6 * n_studies))

    sort_idx = np.argsort(mean_ranks)
    s_fams = [families[i] for i in sort_idx]
    s_ranks = mean_ranks[sort_idx]

    # Maximal cliques
    cliques = []
    for s in range(k):
        for e in range(s + 1, k):
            if s_ranks[e] - s_ranks[s] < cd:
                cliques.append((s, e))
    maximal = [(s, e) for s, e in cliques if not any(s2 <= s and e2 >= e and (s2, e2) != (s, e) for s2, e2 in cliques)]

    # Layout
    n_right = (k + 1) // 2
    label_gap = 0.28
    tick_drop = 0.08

    fig_w = DOUBLE_COL
    fig_h = 1.8 + max(n_right, k - n_right) * 0.22 + len(maximal) * 0.14
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    x_pad = 1.8
    ax.set_xlim(1 - x_pad, k + x_pad)
    ax.invert_xaxis()

    axis_y = 0.0
    lowest = axis_y - 0.18 - (max(n_right, k - n_right) - 1) * label_gap
    bar_top = lowest - 0.25
    bar_bot = bar_top - max(len(maximal) - 1, 0) * 0.15
    ax.set_ylim(bar_bot - 0.08, axis_y + 0.48)

    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_yticks([])
    ax.set_xticks([])

    # Rank axis
    ax.plot([1, k], [axis_y, axis_y], "k-", lw=1.2, clip_on=False)
    for tick in range(1, k + 1):
        ax.plot([tick, tick], [axis_y - 0.03, axis_y + 0.03], "k-", lw=0.7, clip_on=False)
        ax.text(tick, axis_y + 0.06, str(tick), ha="center", va="bottom", fontsize=7, clip_on=False)
    ax.text(
        (1 + k) / 2,
        axis_y + 0.22,
        "Average Rank",
        ha="center",
        va="bottom",
        fontsize=8,
        fontweight="bold",
        clip_on=False,
    )

    # Labels
    right_x = 1 - 0.12
    left_x = k + 0.12

    for i, (fam, rank) in enumerate(zip(s_fams, s_ranks)):
        color = FAMILY_COLORS.get(fam, "#333333")
        if i < n_right:
            row = i
            y = axis_y - 0.18 - row * label_gap
            tick_y = axis_y - tick_drop
            ax.plot([rank, rank], [axis_y, tick_y], "-", color=color, lw=0.6, clip_on=False)
            ax.plot([rank, right_x], [tick_y, y], "-", color=color, lw=0.6, clip_on=False)
            ax.plot(rank, tick_y, "o", color=color, ms=3, clip_on=False, zorder=5)
            ax.text(
                right_x - 0.04,
                y,
                f"{fam} ({rank:.2f})",
                ha="right",
                va="center",
                fontsize=7,
                color=color,
                fontweight="bold",
                clip_on=False,
            )
        else:
            row = i - n_right
            y = axis_y - 0.18 - row * label_gap
            tick_y = axis_y - tick_drop
            ax.plot([rank, rank], [axis_y, tick_y], "-", color=color, lw=0.6, clip_on=False)
            ax.plot([rank, left_x], [tick_y, y], "-", color=color, lw=0.6, clip_on=False)
            ax.plot(rank, tick_y, "o", color=color, ms=3, clip_on=False, zorder=5)
            ax.text(
                left_x + 0.04,
                y,
                f"({rank:.2f}) {fam}",
                ha="left",
                va="center",
                fontsize=7,
                color=color,
                fontweight="bold",
                clip_on=False,
            )

    # Clique bars
    for gi, (s, e) in enumerate(maximal):
        bar_y = bar_top - gi * 0.15
        ax.plot(
            [s_ranks[s], s_ranks[e]], [bar_y, bar_y], color="#333333", lw=3.5, solid_capstyle="round", clip_on=False
        )

    # CD bar
    cd_y = axis_y + 0.35
    cd_left, cd_right = k, k - cd
    ax.plot([cd_left, cd_right], [cd_y, cd_y], "k-", lw=2, clip_on=False)
    ax.plot([cd_left, cd_left], [cd_y - 0.03, cd_y + 0.03], "k-", lw=1, clip_on=False)
    ax.plot([cd_right, cd_right], [cd_y - 0.03, cd_y + 0.03], "k-", lw=1, clip_on=False)
    ax.text(
        (cd_left + cd_right) / 2,
        cd_y + 0.04,
        f"CD = {cd:.2f}",
        ha="center",
        va="bottom",
        fontsize=7,
        fontweight="bold",
        clip_on=False,
    )

    fig.text(
        0.5,
        0.01,
        f"Friedman p = {p_friedman:.1e}  |  n = {n_studies} studies  |  Nemenyi CD = {cd:.2f} (\u03b1 = 0.05)",
        ha="center",
        fontsize=6,
        color="#777777",
    )

    save_figure(fig, FIG_DIR / "fig_critical_difference.pdf")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 3: Convergence Curves by Study Type
# ═══════════════════════════════════════════════════════════════════════════


def _pad(traj, n):
    """Pad or truncate a trajectory to exactly *n* elements."""
    traj = np.asarray(traj, dtype=float)
    if len(traj) >= n:
        return traj[:n]
    out = np.empty(n)
    out[: len(traj)] = traj
    out[len(traj) :] = traj[-1]
    return out


def fig_convergence(result_map, pmids, strategies, study_info):
    """Convergence curves: top-5% recall per round, mean across all studies.

    Shows 4 representative strategies (those with per_round_recall data):
    NGBoost-UCB, RF-TS-Batch, GP-TS-Batch, Random.
    Single panel with thicker lines (1.5pt), distinct line styles and colors,
    shaded +/- 1 SE bands, and value annotations at round 15.
    """
    convergence_strategies = {
        "discrete_ngboost_ucb": "NGBoost-UCB",
        "discrete_rf_ts_batch": "RF-TS",
        "lnpbo_ts_batch": "GP-TS",
        "random": "Random",
    }
    fam_colors = {
        "NGBoost-UCB": FAMILY_COLORS["NGBoost"],
        "RF-TS": FAMILY_COLORS["RF"],
        "GP-TS": "#AA3377",  # distinct purple for GP
        "Random": FAMILY_COLORS["Random"],
    }
    fam_linestyles = {
        "NGBoost-UCB": "-",
        "RF-TS": "--",
        "GP-TS": "-.",
        "Random": ":",
    }
    MAX_R = 16  # index 0 = seed pool, indices 1..15 = BO rounds

    fig, ax = plt.subplots(figsize=(SINGLE_COL + 0.5, 3.0))

    final_vals = {}

    for strat, label in convergence_strategies.items():
        # Collect one trajectory per study (averaged across seeds)
        study_trajs = []
        for p in pmids:
            seed_trajs = []
            for seed in SEEDS:
                key = (p, strat, seed)
                if key not in result_map:
                    continue
                prr = result_map[key]["result"].get("metrics", {}).get("per_round_recall", {})
                recall_5 = prr.get("5", [])
                if len(recall_5) >= 2:
                    seed_trajs.append(_pad(np.array(recall_5), MAX_R))
            if seed_trajs:
                study_trajs.append(np.mean(seed_trajs, axis=0))

        if len(study_trajs) < 2:
            continue

        aligned = np.array(study_trajs)
        n_studies = aligned.shape[0]
        mean_c = np.mean(aligned, axis=0)
        se = np.std(aligned, axis=0, ddof=1) / np.sqrt(n_studies)

        x = np.arange(MAX_R)
        color = fam_colors.get(label, "#888888")
        ls = fam_linestyles.get(label, "-")
        lw = 1.0 if label == "Random" else 1.5
        ax.plot(x, mean_c, color=color, lw=lw, ls=ls, label=label, zorder=3)
        ax.fill_between(x, mean_c - se, mean_c + se, color=color, alpha=0.12, zorder=2)

        final_vals[label] = mean_c[-1]

    # Value annotations at round 15
    if final_vals:
        sorted_items = sorted(final_vals.items(), key=lambda kv: kv[1])
        prev_y = -np.inf
        min_gap = 0.025
        for label, val in sorted_items:
            y_pos = max(val, prev_y + min_gap)
            color = fam_colors.get(label, "#888888")
            ax.text(
                MAX_R - 0.3,
                y_pos,
                f"{val:.2f}",
                va="center",
                ha="left",
                fontsize=6,
                color=color,
                fontweight="bold",
                clip_on=False,
            )
            prev_y = y_pos

    ax.set_xlabel("Round")
    ax.set_ylabel("Top-5% Recall")
    ax.set_xlim(0, MAX_R - 1)
    ax.set_ylim(0.2, 1.04)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    light_ygrid(ax)

    n_total = len(pmids)
    ax.set_title(f"Convergence (mean over {n_total} studies, +/- 1 SE)", fontsize=7)

    ax.legend(
        fontsize=6,
        frameon=True,
        framealpha=0.95,
        edgecolor="#dddddd",
        loc="lower right",
        handlelength=2.5,
    )

    fig.tight_layout()
    save_figure(fig, FIG_DIR / "fig_convergence.pdf")
    plt.close(fig)


def fig_convergence_zscore(result_map, pmids, strategies):
    """SI convergence figure: best z-score found vs. round.

    Two panels: (A) studies with exactly 15 rounds, (B) all studies interpolated.
    Replaces the former h1_convergence.pdf from benchmarks/gap_analysis.py.
    """
    # Map families to strategy lists
    fam_strats = defaultdict(list)
    for s in strategies:
        fam = STRATEGY_FAMILY.get(s)
        if fam:
            fam_strats[fam].append(s)

    show_fams = ["GP (BoTorch)", "RF", "XGBoost", "NGBoost", "Random"]
    fam_labels = {
        "GP (BoTorch)": "GP (BoTorch)",
        "RF": "RF",
        "XGBoost": "XGBoost",
        "NGBoost": "NGBoost",
        "Random": "Random",
    }
    MAX_R = 16  # 0=seed + 15 rounds

    # Collect best_so_far curves
    def get_bsf_curves(family, pmid_filter=None, fixed_length=None):
        curves = []
        for s in fam_strats.get(family, []):
            for p in pmid_filter or pmids:
                for seed in SEEDS:
                    key = (p, s, seed)
                    if key not in result_map:
                        continue
                    bsf = result_map[key]["result"].get("best_so_far", [])
                    if fixed_length and len(bsf) != fixed_length:
                        continue
                    if len(bsf) >= 2:
                        curves.append(bsf)
        return curves

    # Identify 15-round studies
    full_round_pmids = []
    for p in pmids:
        for s in strategies:
            key = (p, s, 42)
            if key in result_map:
                bsf = result_map[key]["result"].get("best_so_far", [])
                if len(bsf) == MAX_R:
                    full_round_pmids.append(p)
                break

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 3.0))

    # Panel A: Fixed 15-round studies
    ax = axes[0]
    for fam in show_fams:
        curves = get_bsf_curves(fam, full_round_pmids, fixed_length=MAX_R)
        if not curves:
            continue
        arr = np.array(curves, dtype=float)
        mc = np.mean(arr, axis=0)
        se = np.std(arr, axis=0, ddof=1) / np.sqrt(len(arr))
        rounds = np.arange(MAX_R)
        color = FAMILY_COLORS.get(fam, "#888888")
        lw = 0.8 if fam == "Random" else 1.3
        ax.plot(rounds, mc, "-o", ms=2, label=f"{fam_labels[fam]} (n={len(arr)})", color=color, lw=lw)
        ax.fill_between(rounds, mc - se, mc + se, alpha=0.12, color=color)
    ax.set_xlabel("Round (0 = seed)")
    ax.set_ylabel("Best z-score found")
    ax.set_title("(a) 15-round studies", fontsize=7)
    ax.legend(fontsize=5, loc="lower right")
    light_ygrid(ax)

    # Panel B: All studies, interpolated to [0, 1] fraction
    ax = axes[1]
    grid = np.linspace(0, 1, 50)
    for fam in show_fams:
        all_curves = get_bsf_curves(fam)
        if not all_curves:
            continue
        interped = []
        for bsf in all_curves:
            x_orig = np.linspace(0, 1, len(bsf))
            interped.append(np.interp(grid, x_orig, bsf))
        arr = np.array(interped)
        mc = np.mean(arr, axis=0)
        se = np.std(arr, axis=0, ddof=1) / np.sqrt(len(arr))
        color = FAMILY_COLORS.get(fam, "#888888")
        lw = 0.8 if fam == "Random" else 1.3
        ax.plot(grid, mc, color=color, lw=lw, label=fam_labels[fam])
        ax.fill_between(grid, mc - se, mc + se, alpha=0.12, color=color)
    ax.set_xlabel("Fraction of total rounds")
    ax.set_ylabel("Best z-score found")
    ax.set_title("(b) All studies (interpolated)", fontsize=7)
    ax.legend(fontsize=5, loc="lower right")
    light_ygrid(ax)

    fig.tight_layout()
    save_figure(fig, FIG_DIR / "fig_convergence_zscore.pdf")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 4: Study-Type Stratified Bar Chart
# ═══════════════════════════════════════════════════════════════════════════


def fig_study_type_stratified(result_map, pmids, strategies, study_info):
    fam_means = family_study_means(result_map, pmids, strategies)
    families = FAMILY_ORDER

    pmids_by_type = defaultdict(list)
    for p in pmids:
        pmids_by_type[study_info[p].get("study_type", "unknown")].append(p)

    n_types = len(STUDY_TYPE_ORDER)
    fig, axes = plt.subplots(1, n_types, figsize=(DOUBLE_COL, 2.5), sharey=True)

    for ax_i, stype in enumerate(STUDY_TYPE_ORDER):
        ax = axes[ax_i]
        type_pmids = pmids_by_type.get(stype, [])

        for fi, fam in enumerate(families):
            vals = [fam_means[(fam, p)] for p in type_pmids if (fam, p) in fam_means]
            if not vals:
                continue
            m = np.mean(vals)
            ci = bootstrap_ci(vals)
            color = FAMILY_COLORS.get(fam, "#888888")
            ax.bar(fi, m, width=0.7, color=color, edgecolor="white", linewidth=0.3, zorder=2)
            ax.errorbar(
                fi, m, yerr=[[m - ci[0]], [ci[1] - m]], fmt="none", ecolor="#333333", capsize=2, lw=0.7, zorder=3
            )

        ax.set_title(f"{STUDY_TYPE_LABELS[stype]}\n(n={len(type_pmids)})", fontsize=6.5)
        ax.set_xticks(range(len(families)))
        ax.set_xticklabels([f.replace(" (sklearn)", "") for f in families], rotation=50, ha="right", fontsize=5.5)
        ax.set_ylim(0, 1.05)
        if ax_i == 0:
            ax.set_ylabel("Mean Top-5% Recall")
        light_ygrid(ax)
        panel_label(ax, chr(97 + ax_i))

    fig.tight_layout(w_pad=0.8)
    save_figure(fig, FIG_DIR / "fig_study_type_stratified.pdf")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 5: Per-Study Strategy Heatmap
# ═══════════════════════════════════════════════════════════════════════════


def fig_heatmap(result_map, pmids, strategies, study_info):
    fam_means = family_study_means(result_map, pmids, strategies)
    families = [f for f in FAMILY_ORDER if f != "Random"]

    data, labels, sort_vals = [], [], []
    for p in pmids:
        row = [fam_means.get((f, p), np.nan) for f in families]
        if np.all(np.isnan(row)):
            continue
        data.append(row)
        si = study_info[p]
        tag = {"il_diverse_fixed_ratios": "F", "il_diverse_variable_ratios": "V", "ratio_only": "R"}.get(
            si.get("study_type", ""), "?"
        )
        labels.append(f"{p} [{tag}]")
        sort_vals.append(np.nanmean(row))

    data = np.array(data)
    idx = np.argsort(sort_vals)
    data = data[idx]
    labels = [labels[i] for i in idx]

    ns, nf = data.shape
    fig, ax = plt.subplots(figsize=(DOUBLE_COL * 0.65, max(3, ns * 0.22)))

    cmap = plt.cm.YlOrRd
    im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=0, vmax=1)

    for i in range(ns):
        winner = np.nanargmax(data[i])
        for j in range(nf):
            v = data[i, j]
            if np.isnan(v):
                continue
            color = "white" if v > 0.65 else "#333333"
            weight = "bold" if j == winner else "normal"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5, color=color, fontweight=weight)

    ax.set_xticks(range(nf))
    ax.set_xticklabels(families, rotation=45, ha="right", fontsize=6)
    ax.set_yticks(range(ns))
    ax.set_yticklabels(labels, fontsize=5)
    ax.set_xlabel("Strategy Family")

    cbar = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02, aspect=30)
    cbar.set_label("Top-5% Recall", fontsize=6.5)
    cbar.ax.tick_params(labelsize=5.5)

    fig.tight_layout()
    save_figure(fig, FIG_DIR / "fig_heatmap.pdf")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 6: Pairwise Win Matrix
# ═══════════════════════════════════════════════════════════════════════════


def fig_win_matrix(result_map, pmids, strategies, study_info):
    fam_means = family_study_means(result_map, pmids, strategies)
    families = [f for f in FAMILY_ORDER if f != "Random"]
    nf = len(families)

    win_frac = np.full((nf, nf), np.nan)
    p_values = np.full((nf, nf), 1.0)

    for i in range(nf):
        for j in range(nf):
            if i == j:
                win_frac[i, j] = 0.5
                continue
            vi = [fam_means.get((families[i], p)) for p in pmids]
            vj = [fam_means.get((families[j], p)) for p in pmids]
            pairs = [(a, b) for a, b in zip(vi, vj) if a is not None and b is not None]
            if len(pairs) < 3:
                continue
            a, b = zip(*pairs)
            a, b = np.array(a), np.array(b)
            win_frac[i, j] = np.mean(a > b)
            diffs = a - b
            nonzero = diffs[diffs != 0]
            if len(nonzero) >= 5:
                try:
                    _, p = sp_stats.wilcoxon(nonzero)
                    p_values[i, j] = p
                except ValueError:
                    pass

    fig, ax = plt.subplots(figsize=(SINGLE_COL + 0.8, SINGLE_COL + 0.3))

    # Diverging colormap centered at 0.5
    cmap = plt.cm.RdBu_r
    im = ax.imshow(win_frac, cmap=cmap, vmin=0, vmax=1)

    for i in range(nf):
        for j in range(nf):
            if np.isnan(win_frac[i, j]):
                continue
            v = win_frac[i, j]
            color = "white" if v > 0.75 or v < 0.25 else "#333333"
            txt = f"{v:.2f}"
            if i != j:
                if p_values[i, j] < 0.01:
                    txt += "**"
                elif p_values[i, j] < 0.05:
                    txt += "*"
            ax.text(j, i, txt, ha="center", va="center", fontsize=5.5, color=color)

    ax.set_xticks(range(nf))
    ax.set_xticklabels(families, rotation=45, ha="right", fontsize=6)
    ax.set_yticks(range(nf))
    ax.set_yticklabels(families, fontsize=6)

    cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02, aspect=25)
    cbar.set_label("Win fraction", fontsize=6.5)
    cbar.ax.tick_params(labelsize=5.5)

    fig.text(0.5, 0.01, "* p < 0.05  ** p < 0.01 (Wilcoxon signed-rank)", ha="center", fontsize=5.5, color="#777777")

    fig.tight_layout()
    save_figure(fig, FIG_DIR / "fig_win_matrix.pdf")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 7: Encoding Comparison
# ═══════════════════════════════════════════════════════════════════════════


_ENCODING_STRATEGIES = {
    "random",
    "discrete_rf_ts",
    "discrete_xgb_ucb",
    "discrete_ngboost_ucb",
    "casmopolitan_ucb",
}


def _load_encoding_data(filter_strategies=True):
    """Load encoding ablation results.

    Parameters
    ----------
    filter_strategies : bool
        If True (default), only include the 5 strategies used in the main
        encoding comparison (matches the SI table).  Set False to load all
        34 strategies for the extended SI heatmap.
    """
    enc_dir = ABLATION_DIR / "encoding"
    if not enc_dir.exists():
        return []
    rows = []
    for p in sorted(enc_dir.rglob("*.json")):
        if p.name.startswith("encoding_comparison"):
            continue
        try:
            d = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        strategy = d.get("strategy", "")
        if filter_strategies and strategy not in _ENCODING_STRATEGIES:
            continue
        cond = d.get("condition", {})
        label = cond.get("label", "")
        if not label:
            continue
        metrics = d.get("result", {}).get("metrics", {})
        recall = metrics.get("top_k_recall", {})
        rows.append(
            {
                "study_id": d.get("study_id", str(d.get("pmid", ""))),
                "strategy": strategy,
                "encoding": label,
                "seed": d.get("seed"),
                "top5": recall.get("5", 0.0),
            }
        )
    return rows


def fig_encoding(rows=None):
    """Encoding comparison bar chart with bootstrap CIs."""
    if rows is None:
        rows = _load_encoding_data()
    if not rows:
        print("  [SKIP] No encoding data")
        return

    nested = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in rows:
        nested[r["encoding"]][r["strategy"]][r["study_id"]].append(r["top5"])

    encodings = sorted(nested.keys())
    all_strats = set()
    for enc in encodings:
        all_strats.update(nested[enc].keys())
    strat_list = sorted(all_strats)

    enc_data = {}
    for enc in encodings:
        study_means = []
        all_sids = set()
        for st in strat_list:
            all_sids.update(nested[enc][st].keys())
        for sid in all_sids:
            vals = []
            for st in strat_list:
                sv = nested[enc][st].get(sid, [])
                if sv:
                    vals.append(np.mean(sv))
            if vals:
                study_means.append(np.mean(vals))
        if study_means:
            m = np.mean(study_means)
            ci = bootstrap_ci(study_means)
            enc_data[enc] = (m, ci[0], ci[1], len(study_means))

    if not enc_data:
        print("  [SKIP] Not enough encoding data")
        return

    ordered = sorted(enc_data.keys(), key=lambda e: enc_data[e][0], reverse=True)

    fig, ax = plt.subplots(figsize=(SINGLE_COL + 0.5, 2.5))

    x = np.arange(len(ordered))
    for i, enc in enumerate(ordered):
        m, lo, hi, _n = enc_data[enc]
        color = ENCODING_COLORS.get(enc, "#888888")
        ax.bar(i, m, width=0.65, color=color, edgecolor="white", linewidth=0.3, zorder=2)
        ax.errorbar(i, m, yerr=[[m - lo], [hi - m]], fmt="none", ecolor="#333333", capsize=2.5, lw=0.7, zorder=3)
        ax.text(i, hi + 0.005, f"{m:.3f}", ha="center", va="bottom", fontsize=5.5, color="#333333")

    ax.set_xticks(x)
    ax.set_xticklabels([ENCODING_DISPLAY.get(e, e) for e in ordered], rotation=30, ha="right", fontsize=6.5)
    ax.set_ylabel("Mean Top-5% Recall")
    ymax = max(enc_data[e][2] for e in ordered) + 0.04
    ax.set_ylim(0.4, min(ymax, 1.0))
    light_ygrid(ax)

    fig.tight_layout()
    save_figure(fig, FIG_DIR / "fig_encoding.pdf")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 8: Cross-Study Transfer
# ═══════════════════════════════════════════════════════════════════════════


def fig_cross_study():
    """Two-panel: per-study scatter + summary bars."""
    # Try warm results first
    warm_path = CROSS_STUDY_DIR / "cross_study_transfer_results.json"
    if not warm_path.exists():
        print("  [SKIP] No cross-study transfer data")
        return

    with open(warm_path) as f:
        data = json.load(f)

    warm = data.get("warm", {})
    if not warm:
        print("  [SKIP] No warm-start results")
        return

    seeds = [str(s) for s in data["config"]["seeds"]]

    study_ids, mean_recalls, seed_recalls = [], [], {}
    for sid, sd in warm.items():
        vals = [sd[s]["recall_5"] for s in seeds if s in sd]
        if vals:
            study_ids.append(sid)
            mean_recalls.append(np.mean(vals))
            seed_recalls[sid] = vals

    mean_recalls = np.array(mean_recalls)

    # Classify
    ratio_only = {"38082180", "35879315"}
    sub_kw = ["liver", "spleen", "pooled", "multiorgan", "in_vitro"]
    cats = []
    for sid in study_ids:
        base = sid.split("_")[0]
        if base in ratio_only:
            cats.append("ratio-only")
        elif any(k in sid for k in sub_kw):
            cats.append("sub-study")
        else:
            cats.append("IL-diverse")

    order = np.argsort(-mean_recalls)
    s_ids = [study_ids[i] for i in order]
    s_means = mean_recalls[order]
    s_cats = [cats[i] for i in order]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.8), gridspec_kw={"width_ratios": [2.2, 1]})

    # Left: scatter
    xpos = np.arange(len(s_ids))
    for i, sid in enumerate(s_ids):
        color = STUDY_TYPE_COLORS.get(s_cats[i], "#888888")
        for v in seed_recalls[sid]:
            ax1.scatter(xpos[i], v, c=color, s=6, alpha=0.2, edgecolors="none", zorder=4)

    for cat, color in STUDY_TYPE_COLORS.items():
        mask = [i for i, c in enumerate(s_cats) if c == cat]
        if mask:
            ax1.scatter(
                [xpos[i] for i in mask],
                [s_means[i] for i in mask],
                c=color,
                s=25,
                zorder=5,
                edgecolors="white",
                linewidths=0.3,
                label=cat,
            )

    ax1.axhline(0.05, color="#999999", ls="--", lw=0.5, zorder=2, label="Random (5%)")
    ax1.axhline(0.73, color="#EE6677", ls="--", lw=0.5, zorder=2, label="Within-study BO")

    short_labels = []
    for sid in s_ids:
        if "_" in sid:
            base, suffix = sid.split("_", 1)
            short_labels.append(f"{base}\n({suffix[:8]})")
        else:
            short_labels.append(sid)

    ax1.set_xticks(xpos)
    ax1.set_xticklabels(short_labels, rotation=55, ha="right", fontsize=4.5)
    ax1.set_ylabel("Top-5% Recall")
    ax1.set_xlabel("Target study")
    ax1.set_ylim(-0.02, 0.82)
    ax1.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    light_ygrid(ax1)
    ax1.legend(fontsize=5.5, frameon=True, framealpha=0.95, edgecolor="#dddddd", loc="upper right", handletextpad=0.3)
    panel_label(ax1, "a")

    # Right: summary bars
    grand_mean = np.mean(mean_recalls)
    ci = bootstrap_ci(mean_recalls)

    bar_labels = ["Random", "Cross-study\nwarm-start", "Within-study\nBO"]
    bar_vals = [0.05, grand_mean, 0.73]
    bar_colors = ["#000000", "#4477AA", "#EE6677"]

    bars = ax2.bar(bar_labels, bar_vals, color=bar_colors, width=0.55, edgecolor="white", linewidth=0.3, zorder=3)
    ax2.errorbar(
        [1],
        [grand_mean],
        yerr=[[grand_mean - ci[0]], [ci[1] - grand_mean]],
        fmt="none",
        ecolor="#333333",
        capsize=3,
        lw=0.8,
        zorder=4,
    )

    for i, (bar, val) in enumerate(zip(bars, bar_vals)):
        y_off = (ci[1] - grand_mean + 0.015) if i == 1 else 0.015
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            val + y_off,
            f"{val:.1%}",
            ha="center",
            va="bottom",
            fontsize=6.5,
            fontweight="bold",
        )

    ax2.set_ylabel("Top-5% Recall")
    ax2.set_ylim(0, 0.88)
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    light_ygrid(ax2)
    panel_label(ax2, "b")

    fig.tight_layout(w_pad=2.0)
    save_figure(fig, FIG_DIR / "fig_cross_study.pdf")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 9: Ablation Summary (multi-panel)
# ═══════════════════════════════════════════════════════════════════════════


def _load_ablation(name):
    d = ABLATION_DIR / name
    if not d.exists():
        return []
    rows = []
    for p in sorted(d.rglob("*.json")):
        try:
            data = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        cond = data.get("condition", {})
        metrics = data.get("result", {}).get("metrics", {})
        recall = metrics.get("top_k_recall", {})
        rows.append(
            {
                "study_id": data.get("study_id", ""),
                "strategy": data.get("strategy", ""),
                "label": cond.get("label", ""),
                "seed": data.get("seed"),
                "top5": recall.get("5", 0.0),
            }
        )
    return rows


def _ablation_bar(ax, rows, title, panel_lbl):
    """Generic ablation bar chart on a given axis."""
    if not rows:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, fontsize=7, color="#999999")
        ax.set_title(title, fontsize=7)
        panel_label(ax, panel_lbl)
        return

    nested = defaultdict(lambda: defaultdict(list))
    for r in rows:
        nested[r["label"]][r["study_id"]].append(r["top5"])

    labels_sorted = sorted(nested.keys())
    means, cis_lo, cis_hi = [], [], []
    for lbl in labels_sorted:
        study_means = [np.mean(v) for v in nested[lbl].values()]
        if study_means:
            m = np.mean(study_means)
            ci = bootstrap_ci(study_means) if len(study_means) >= 3 else (m, m)
            means.append(m)
            cis_lo.append(ci[0])
            cis_hi.append(ci[1])
        else:
            means.append(0)
            cis_lo.append(0)
            cis_hi.append(0)

    x = np.arange(len(labels_sorted))
    colors = plt.cm.Set2(np.linspace(0, 0.8, len(labels_sorted)))

    for i in range(len(labels_sorted)):
        ax.bar(i, means[i], width=0.65, color=colors[i], edgecolor="white", linewidth=0.3, zorder=2)
        ax.errorbar(
            i,
            means[i],
            yerr=[[means[i] - cis_lo[i]], [cis_hi[i] - means[i]]],
            fmt="none",
            ecolor="#333333",
            capsize=2,
            lw=0.6,
            zorder=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels_sorted, rotation=35, ha="right", fontsize=5.5)
    ax.set_title(title, fontsize=7)
    light_ygrid(ax)
    panel_label(ax, panel_lbl)


def fig_ablation_summary():
    """Four-panel ablation summary: batch size, budget, warmup, PCA."""
    experiments = [
        ("batch_size", "Batch Size"),
        ("budget", "Budget"),
        ("warmup", "Warmup"),
        ("pca", "PCA Dims"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL, 4.5))
    axes = axes.ravel()

    for i, (name, title) in enumerate(experiments):
        rows = _load_ablation(name)
        _ablation_bar(axes[i], rows, title, chr(97 + i))

    # Common y-label
    fig.text(0.01, 0.5, "Mean Top-5% Recall", va="center", rotation=90, fontsize=7)

    fig.tight_layout(rect=[0.03, 0, 1, 1], h_pad=2.0, w_pad=1.5)
    save_figure(fig, FIG_DIR / "fig_ablation_summary.pdf")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 10: Data Landscape (2-panel)
# ═══════════════════════════════════════════════════════════════════════════


def fig_data_landscape():
    """Two-panel: component diversity + cargo/study type."""
    try:
        from LNPBO.data.lnpdb_bridge import load_lnpdb_full

        dataset = load_lnpdb_full()
        df = dataset.df
    except Exception as e:
        print(f"  [SKIP] Cannot load LNPDB: {e}")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.5))

    # Left: component diversity (log scale)
    diversity = {
        "IL": df["IL_SMILES"].nunique(),
        "HL": df["HL_name"].nunique(),
        "CHL": df["CHL_name"].nunique(),
        "PEG": df["PEG_name"].nunique(),
    }
    roles = list(diversity.keys())
    counts = list(diversity.values())
    bar_colors = ["#EE6677", "#228833", "#CCBB44", "#4477AA"]

    bars = ax1.bar(roles, counts, color=bar_colors, edgecolor="white", linewidth=0.3, width=0.6)
    ax1.set_yscale("log")
    ax1.set_ylabel("Unique structures")
    ax1.set_title("Component diversity", fontsize=7)
    for bar, c in zip(bars, counts):
        ax1.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.3, f"{c:,}", ha="center", va="bottom", fontsize=6
        )
    panel_label(ax1, "a")

    # Right: cargo type distribution (broad categories: mRNA, siRNA, pDNA)
    cargo = df["Cargo"].fillna("Unknown").value_counts()
    cargo_colors = {"mRNA": "#228833", "siRNA": "#4477AA", "pDNA": "#EE6677"}
    c_colors = [cargo_colors.get(c, "#BBBBBB") for c in cargo.index]
    bars = ax2.barh(cargo.index, cargo.values, color=c_colors, edgecolor="white", linewidth=0.3, height=0.6)
    ax2.set_xlabel("Formulations")
    ax2.set_title("Cargo type distribution", fontsize=7)
    for i, (_ct, n) in enumerate(cargo.items()):
        pct = n / len(df) * 100
        ax2.text(n + len(df) * 0.01, i, f"{n:,} ({pct:.0f}%)", va="center", fontsize=5.5)
    ax2.invert_yaxis()
    panel_label(ax2, "b")

    fig.tight_layout(w_pad=2.5)
    save_figure(fig, FIG_DIR / "fig_data_landscape.pdf")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 11: Variance Decomposition
# ═══════════════════════════════════════════════════════════════════════════


def fig_variance(result_map, pmids, strategies, study_info):
    """Horizontal bar chart of variance components."""
    # Build tidy data
    rows = []
    for (pmid, strat, seed), _r in result_map.items():
        v = get_top5(result_map, pmid, strat, seed)
        if v is not None:
            fam = STRATEGY_FAMILY.get(strat, "Other")
            rows.append({"study": pmid, "strategy": strat, "family": fam, "seed": seed, "value": v})

    values = np.array([r["value"] for r in rows])
    grand_mean = np.mean(values)
    ss_total = np.sum((values - grand_mean) ** 2)

    # Balanced crossed ANOVA: Study x Strategy x Seed
    studies = sorted(set(r["study"] for r in rows))
    strats = sorted(set(r["strategy"] for r in rows))
    seeds = sorted(set(r["seed"] for r in rows))
    n_studies = len(studies)
    n_strats = len(strats)
    n_seeds = len(seeds)

    # Main effects
    study_means = {}
    for s in studies:
        vals = [r["value"] for r in rows if r["study"] == s]
        if vals:
            study_means[s] = np.mean(vals)
    n_per_study = n_strats * n_seeds
    ss_study = n_per_study * sum((study_means[s] - grand_mean) ** 2 for s in studies if s in study_means)

    strat_means = {}
    for s in strats:
        vals = [r["value"] for r in rows if r["strategy"] == s]
        if vals:
            strat_means[s] = np.mean(vals)
    n_per_strat = n_studies * n_seeds
    ss_strategy = n_per_strat * sum((strat_means[s] - grand_mean) ** 2 for s in strats if s in strat_means)

    seed_means = {}
    for s in seeds:
        vals = [r["value"] for r in rows if r["seed"] == s]
        if vals:
            seed_means[s] = np.mean(vals)
    n_per_seed = n_studies * n_strats
    ss_seed = n_per_seed * sum((seed_means[s] - grand_mean) ** 2 for s in seeds if s in seed_means)

    # Study x Strategy interaction (correct balanced formula)
    study_strat_means = {}
    for p in studies:
        for s in strats:
            vals = [r["value"] for r in rows if r["study"] == p and r["strategy"] == s]
            if vals:
                study_strat_means[(p, s)] = np.mean(vals)
    ss_interaction = n_seeds * sum(
        (
            study_strat_means.get((p, s), grand_mean)
            - study_means.get(p, grand_mean)
            - strat_means.get(s, grand_mean)
            + grand_mean
        ) ** 2
        for p in studies
        for s in strats
        if (p, s) in study_strat_means
    )

    ss_residual = ss_total - ss_study - ss_strategy - ss_seed - ss_interaction

    components = {
        "study": ss_study,
        "strategy": ss_strategy,
        "seed": ss_seed,
        "study x strategy": max(0, ss_interaction),
        "residual": max(0, ss_residual),
    }

    pct = {k: v / ss_total * 100 for k, v in components.items()}

    # Sort by size
    ordered = sorted(pct.items(), key=lambda x: -x[1])
    labels = [k.replace("_", " ").title() for k, _ in ordered]
    values = [v for _, v in ordered]

    fig, ax = plt.subplots(figsize=(SINGLE_COL + 0.3, 2.0))

    colors = ["#4477AA", "#EE6677", "#228833", "#CCBB44", "#BBBBBB"]
    y = np.arange(len(labels))
    for i, (_lbl, val) in enumerate(zip(labels, values)):
        ax.barh(i, val, color=colors[i % len(colors)], edgecolor="white", linewidth=0.3, height=0.6)
        ax.text(val + 0.5, i, f"{val:.1f}%", va="center", fontsize=6)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=6.5)
    ax.set_xlabel("% of Total Variance")
    ax.set_xlim(0, max(values) * 1.15)
    ax.invert_yaxis()

    fig.tight_layout()
    save_figure(fig, FIG_DIR / "fig_variance.pdf")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Figure A4-A: PCA Component Justification
# ═══════════════════════════════════════════════════════════════════════════


def fig_pca_justification():
    """Bar chart of PCA components vs top-5% recall with explained variance inset."""
    rows = _load_ablation("pca")
    if not rows:
        print("  [SKIP] No PCA ablation data")
        return

    nested = defaultdict(lambda: defaultdict(list))
    for r in rows:
        nested[r["label"]][r["study_id"]].append(r["top5"])

    # Sort numerically: pca3, pca5, pca10, pca20, raw
    def _pca_sort_key(lbl):
        if lbl.startswith("pca"):
            return (0, int(lbl[3:]))
        return (1, 0)  # "raw" goes last

    labels_sorted = sorted(nested.keys(), key=_pca_sort_key)
    means, cis_lo, cis_hi = [], [], []
    for lbl in labels_sorted:
        study_means = [np.mean(v) for v in nested[lbl].values()]
        if study_means:
            m = np.mean(study_means)
            ci = bootstrap_ci(study_means) if len(study_means) >= 3 else (m, m)
            means.append(m)
            cis_lo.append(ci[0])
            cis_hi.append(ci[1])
        else:
            means.append(0)
            cis_lo.append(0)
            cis_hi.append(0)

    # Try to compute explained variance for a second panel
    evr = None
    try:
        from LNPBO.data.compute_pcs import compute_pcs
        from LNPBO.data.lnpdb_bridge import load_lnpdb_full

        db = load_lnpdb_full()
        il_smiles = db.df["IL_SMILES"].dropna().unique().tolist()
        _, reducer, _, _ = compute_pcs(
            il_smiles,
            feature_type="count_mfp",
            n_components=20,
            reduction="pca",
            cache_name="IL",
        )
        evr = reducer.explained_variance_ratio_
    except Exception as e:
        print(f"  [NOTE] Could not compute explained variance: {e}")

    if evr is not None:
        fig, (ax, ax2) = plt.subplots(
            1,
            2,
            figsize=(DOUBLE_COL, 2.5),
            gridspec_kw={"width_ratios": [1.3, 1]},
        )
    else:
        fig, ax = plt.subplots(figsize=(SINGLE_COL + 0.5, 2.8))
        ax2 = None

    x = np.arange(len(labels_sorted))
    base_colors = plt.cm.Set2(np.linspace(0, 0.8, len(labels_sorted)))

    # Highlight the chosen n_pcs=5 condition
    bar_colors = []
    for i, lbl in enumerate(labels_sorted):
        if lbl == "pca5":
            bar_colors.append("#4477AA")
        else:
            bar_colors.append(base_colors[i])

    for i in range(len(labels_sorted)):
        ax.bar(i, means[i], width=0.65, color=bar_colors[i], edgecolor="white", linewidth=0.3, zorder=2)
        ax.errorbar(
            i,
            means[i],
            yerr=[[means[i] - cis_lo[i]], [cis_hi[i] - means[i]]],
            fmt="none",
            ecolor="#333333",
            capsize=2.5,
            lw=0.7,
            zorder=3,
        )
        ax.text(i, cis_hi[i] + 0.005, f"{means[i]:.3f}", ha="center", va="bottom", fontsize=5.5, color="#333333")

    # Pretty x-labels: "pca3" -> "3", "pca5" -> "5", "raw" -> "Raw"
    display_labels = []
    for lbl in labels_sorted:
        if lbl.startswith("pca"):
            display_labels.append(lbl[3:])
        elif lbl == "raw":
            display_labels.append("Raw")
        else:
            display_labels.append(lbl)

    ax.set_xticks(x)
    ax.set_xticklabels(display_labels, fontsize=6.5)
    ax.set_xlabel("Number of PCA Components")
    ax.set_ylabel("Mean Top-5% Recall")
    if means:
        ymax = max(cis_hi) + 0.04
        ax.set_ylim(0.4, min(ymax, 1.0))
    light_ygrid(ax)
    panel_label(ax, "a")

    # Panel (b): explained variance scree plot
    if ax2 is not None:
        cumvar = np.cumsum(evr)
        n_show = min(20, len(evr))
        pcs = np.arange(1, n_show + 1)
        ax2.bar(
            pcs,
            evr[:n_show] * 100,
            color="#BBBBBB",
            edgecolor="white",
            linewidth=0.3,
            width=0.7,
            zorder=2,
            label="Individual",
        )
        ax2.plot(pcs, cumvar[:n_show] * 100, "o-", color="#EE6677", ms=3, lw=1.0, zorder=3, label="Cumulative")
        ax2.axvline(5, color="#4477AA", ls="--", lw=0.7, alpha=0.7, zorder=1)
        ax2.set_xlabel("Principal Component")
        ax2.set_ylabel("Variance Explained (%)")
        ax2.set_xlim(0.5, n_show + 0.5)
        ax2.set_xticks([1, 5, 10, 15, 20])
        # Scale y-axis to cumulative max so individual bars are visible
        ymax_evr = cumvar[n_show - 1] * 100
        ax2.set_ylim(0, ymax_evr * 1.15)
        ax2.legend(fontsize=6, loc="upper left", frameon=False)
        light_ygrid(ax2)

        # Annotate cumulative at n_pcs=5
        cum5 = cumvar[4] * 100
        ax2.annotate(
            f"5 PCs = {cum5:.0f}%",
            xy=(5, cum5),
            xytext=(8, cum5 + ymax_evr * 0.12),
            fontsize=6,
            color="#4477AA",
            fontweight="bold",
            arrowprops=dict(arrowstyle="-", color="#4477AA", lw=0.5),
        )
        panel_label(ax2, "b")

    fig.tight_layout()
    save_figure(fig, FIG_DIR / "fig_pca_justification.pdf")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Figure A4-B: Batch Size Sensitivity
# ═══════════════════════════════════════════════════════════════════════════


def fig_batch_sensitivity():
    """Two-panel bar chart: batch size vs top-5% and top-10% recall."""
    rows_raw = _load_ablation("batch_size")
    if not rows_raw:
        print("  [SKIP] No batch size ablation data")
        return

    # Re-load with top-10% recall as well
    d = ABLATION_DIR / "batch_size"
    rows = []
    for p in sorted(d.rglob("*.json")):
        try:
            data = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        cond = data.get("condition", {})
        metrics = data.get("result", {}).get("metrics", {})
        recall = metrics.get("top_k_recall", {})
        rows.append(
            {
                "study_id": data.get("study_id", ""),
                "label": cond.get("label", ""),
                "seed": data.get("seed"),
                "top5": recall.get("5", 0.0),
                "top10": recall.get("10", 0.0),
            }
        )

    if not rows:
        print("  [SKIP] No batch size data after reload")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.5), sharey=True)

    for ax, metric, title in [(ax1, "top5", "Top-5% Recall"), (ax2, "top10", "Top-10% Recall")]:
        nested = defaultdict(lambda: defaultdict(list))
        for r in rows:
            nested[r["label"]][r["study_id"]].append(r[metric])

        labels_sorted = sorted(nested.keys())
        means, cis_lo, cis_hi = [], [], []
        for lbl in labels_sorted:
            study_means = [np.mean(v) for v in nested[lbl].values()]
            if study_means:
                m = np.mean(study_means)
                ci = bootstrap_ci(study_means) if len(study_means) >= 3 else (m, m)
                means.append(m)
                cis_lo.append(ci[0])
                cis_hi.append(ci[1])
            else:
                means.append(0)
                cis_lo.append(0)
                cis_hi.append(0)

        x = np.arange(len(labels_sorted))
        colors = plt.cm.Set2(np.linspace(0, 0.8, len(labels_sorted)))

        for i in range(len(labels_sorted)):
            ax.bar(i, means[i], width=0.65, color=colors[i], edgecolor="white", linewidth=0.3, zorder=2)
            ax.errorbar(
                i,
                means[i],
                yerr=[[means[i] - cis_lo[i]], [cis_hi[i] - means[i]]],
                fmt="none",
                ecolor="#333333",
                capsize=2.5,
                lw=0.7,
                zorder=3,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(labels_sorted, rotation=35, ha="right", fontsize=5.5)
        ax.set_title(title, fontsize=7)
        light_ygrid(ax)

    ax1.set_ylabel("Mean Recall")
    ax1.set_xlabel("Batch Size")
    ax2.set_xlabel("Batch Size")
    panel_label(ax1, "a")
    panel_label(ax2, "b")

    fig.tight_layout(w_pad=1.5)
    save_figure(fig, FIG_DIR / "fig_batch_sensitivity.pdf")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Figure A4-C2: Kernel Comparison
# ═══════════════════════════════════════════════════════════════════════════

# Mapping from strategy name to kernel type for kernel ablation results
KERNEL_MAP = {
    "lnpbo_logei": "Matern",
    "lnpbo_ts_batch": "Matern",
    "lnpbo_tanimoto_logei": "Tanimoto",
    "lnpbo_tanimoto_ts": "Tanimoto",
    "lnpbo_aitchison_logei": "Aitchison",
    "lnpbo_aitchison_ts": "Aitchison",
    "lnpbo_dkl_logei": "DKL",
    "lnpbo_dkl_ts": "DKL",
}

KERNEL_COLORS = {
    "Matern": "#4477AA",
    "Tanimoto": "#EE6677",
    "Aitchison": "#228833",
    "DKL": "#CCBB44",
}

KERNEL_ORDER = ["Matern", "Tanimoto", "Aitchison", "DKL"]


def fig_kernel_comparison():
    """Forest plot comparing GP kernel types on top-5% recall."""
    kernel_dir = ABLATION_DIR / "kernel"
    if not kernel_dir.exists():
        print("  [SKIP] No kernel ablation data")
        return

    # Load all kernel ablation results
    rows = []
    for p in sorted(kernel_dir.rglob("*.json")):
        try:
            data = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        strategy = data.get("strategy", "")
        recall = data.get("result", {}).get("metrics", {}).get("top_k_recall", {})
        rows.append(
            {
                "study_id": data.get("study_id", ""),
                "strategy": strategy,
                "seed": data.get("seed"),
                "kernel": KERNEL_MAP.get(strategy, ""),
                "top5": recall.get("5", 0.0),
            }
        )

    # Filter to kernel strategies only (exclude random)
    rows = [r for r in rows if r["kernel"]]
    if not rows:
        print("  [SKIP] No kernel comparison data after filtering")
        return

    # Also load random baseline from kernel ablation
    random_rows = []
    for p in sorted(kernel_dir.rglob("random_*.json")):
        try:
            data = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        recall = data.get("result", {}).get("metrics", {}).get("top_k_recall", {})
        random_rows.append(
            {
                "study_id": data.get("study_id", ""),
                "top5": recall.get("5", 0.0),
            }
        )

    random_mean = None
    if random_rows:
        rnd_nested = defaultdict(list)
        for r in random_rows:
            rnd_nested[r["study_id"]].append(r["top5"])
        rnd_study_means = [np.mean(v) for v in rnd_nested.values()]
        if rnd_study_means:
            random_mean = float(np.mean(rnd_study_means))

    # Aggregate: kernel -> study -> [values], averaging across seeds first,
    # then strategies within kernel, then across studies
    nested = defaultdict(lambda: defaultdict(list))
    for r in rows:
        nested[r["kernel"]][r["study_id"]].append(r["top5"])

    kernel_data = {}
    for kernel in KERNEL_ORDER:
        if kernel not in nested:
            continue
        study_means = [np.mean(v) for v in nested[kernel].values()]
        if study_means:
            m = float(np.mean(study_means))
            ci = bootstrap_ci(study_means) if len(study_means) >= 3 else (m, m)
            n_studies = len(study_means)
            kernel_data[kernel] = (m, ci[0], ci[1], n_studies)

    if not kernel_data:
        print("  [SKIP] Not enough kernel data")
        return

    # Sort by mean (best at top for forest plot)
    ordered = sorted(kernel_data.keys(), key=lambda k: kernel_data[k][0])

    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.2))

    n_full = max(kernel_data[k][3] for k in ordered)
    y = np.arange(len(ordered))

    for i, kernel in enumerate(ordered):
        m, lo, hi, n_studies = kernel_data[kernel]
        color = KERNEL_COLORS.get(kernel, "#888888")
        ax.plot(m, i, "o", color=color, ms=6, zorder=5)
        ax.plot([lo, hi], [i, i], "-", color=color, lw=1.8, zorder=4, solid_capstyle="round")
        note = f"{m:.3f}"
        if n_studies < n_full:
            note += f" (n={n_studies})"
        ax.text(hi + 0.008, i, note, va="center", fontsize=6, color="#333333")

    if random_mean is not None:
        ax.axvline(random_mean, color="#000000", ls=":", lw=0.6, alpha=0.5, zorder=1)
        ax.text(random_mean + 0.003, len(ordered) - 0.3, "Random", fontsize=5, ha="left", va="top", color="#666666")

    ax.set_yticks(y)
    ax.set_yticklabels(ordered, fontsize=7)
    ax.set_xlabel("Mean Top-5% Recall")
    light_ygrid(ax)

    # Set sensible x-limits
    all_lo = min(kernel_data[k][1] for k in ordered)
    all_hi = max(kernel_data[k][2] for k in ordered)
    margin = (all_hi - all_lo) * 0.15
    ax.set_xlim(all_lo - margin, all_hi + margin + 0.06)

    fig.tight_layout()
    save_figure(fig, FIG_DIR / "fig_kernel_comparison.pdf")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Figure A4-C3: Kappa (UCB exploration) Sensitivity
# ═══════════════════════════════════════════════════════════════════════════

KAPPA_STRATEGY_DISPLAY = {
    "discrete_xgb_ucb": "XGBoost",
    "discrete_rf_ucb": "RF",
    "discrete_ngboost_ucb": "NGBoost",
}

KAPPA_COLORS = {
    "XGBoost": FAMILY_COLORS["XGBoost"],
    "RF": FAMILY_COLORS["RF"],
    "NGBoost": FAMILY_COLORS["NGBoost"],
}


def fig_kappa_sensitivity():
    """Line plot: kappa vs top-5% recall per surrogate family."""
    kappa_dir = ABLATION_DIR / "kappa"
    if not kappa_dir.exists():
        print("  [SKIP] No kappa ablation data")
        return

    # Load all kappa ablation results
    rows = []
    for p in sorted(kappa_dir.rglob("*.json")):
        try:
            data = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        strategy = data.get("strategy", "")
        if strategy == "random":
            continue
        cond = data.get("condition", {})
        kappa = cond.get("kappa")
        if kappa is None:
            # Parse from label: "kappa_2" -> 2.0
            label = cond.get("label", "")
            if label.startswith("kappa_"):
                try:
                    kappa = float(label.split("_")[1])
                except (IndexError, ValueError):
                    continue
            else:
                continue

        recall = data.get("result", {}).get("metrics", {}).get("top_k_recall", {})
        display = KAPPA_STRATEGY_DISPLAY.get(strategy)
        if display is None:
            continue
        rows.append(
            {
                "study_id": data.get("study_id", ""),
                "strategy": display,
                "seed": data.get("seed"),
                "kappa": kappa,
                "top5": recall.get("5", 0.0),
            }
        )

    if not rows:
        print("  [SKIP] No kappa sensitivity data")
        return

    # Check if we have enough data: at least 2 kappa values with >=2 studies each
    kappa_vals = sorted(set(r["kappa"] for r in rows))
    kappa_study_counts = {}
    for kv in kappa_vals:
        studies = set(r["study_id"] for r in rows if r["kappa"] == kv)
        kappa_study_counts[kv] = len(studies)

    usable = [kv for kv, n in kappa_study_counts.items() if n >= 2]
    if len(usable) < 2:
        print(f"  [SKIP] Not enough kappa data (only {len(usable)} kappa values with >=2 studies)")
        return

    # Aggregate: strategy x kappa -> study means
    nested = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in rows:
        if r["kappa"] in usable:
            nested[r["strategy"]][r["kappa"]][r["study_id"]].append(r["top5"])

    fig, ax = plt.subplots(figsize=(SINGLE_COL + 0.3, 2.5))

    kappas_sorted = sorted(usable)

    for strat in ["NGBoost", "RF", "XGBoost"]:
        if strat not in nested:
            continue
        x_vals, y_vals, y_lo, y_hi = [], [], [], []
        for kv in kappas_sorted:
            if kv not in nested[strat]:
                continue
            study_means = [np.mean(v) for v in nested[strat][kv].values()]
            if len(study_means) < 2:
                continue
            m = float(np.mean(study_means))
            ci = bootstrap_ci(study_means) if len(study_means) >= 3 else (m, m)
            x_vals.append(kv)
            y_vals.append(m)
            y_lo.append(ci[0])
            y_hi.append(ci[1])

        if len(x_vals) < 2:
            continue

        color = KAPPA_COLORS.get(strat, "#888888")
        ax.plot(x_vals, y_vals, "o-", color=color, ms=4, lw=1.2, label=strat, zorder=3)
        ax.fill_between(x_vals, y_lo, y_hi, color=color, alpha=0.12, zorder=2)

    ax.set_xscale("log")
    ax.set_xticks(kappas_sorted)
    ax.set_xticklabels([f"{k:g}" for k in kappas_sorted], fontsize=6.5)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.minorticks_off()
    ax.set_xlabel(r"$\kappa$ (UCB exploration weight)")
    ax.set_ylabel("Mean Top-5% Recall")
    light_ygrid(ax)
    ax.legend(fontsize=6, frameon=True, framealpha=0.9, edgecolor="#dddddd", loc="best", handletextpad=0.3)

    # Note about partial data
    n_studies = max(kappa_study_counts[kv] for kv in usable)
    fig.text(0.5, 0.01, f"n = {n_studies} studies (partial ablation)", ha="center", fontsize=5.5, color="#777777")

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    save_figure(fig, FIG_DIR / "fig_kappa_sensitivity.pdf")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Figure A4-C: Acquisition Function Breakdown (within GP family)
# ═══════════════════════════════════════════════════════════════════════════


def fig_acquisition_breakdown(result_map, pmids, strategies, study_info):
    """Horizontal forest plot comparing GP-BO acquisition/batch strategies."""
    # Strategy key -> (display label, batch group)
    ACQF_META = {
        "lnpbo_ucb": ("KB (UCB)", "KB"),
        "lnpbo_ei": ("KB (EI)", "KB"),
        "lnpbo_logei": ("KB (LogEI)", "KB"),
        "lnpbo_rkb_logei": ("RKB (LogEI)", "RKB"),
        "lnpbo_lp_ei": ("LP (EI)", "LP"),
        "lnpbo_lp_logei": ("LP (LogEI)", "LP"),
        "lnpbo_ts_batch": ("TS-Batch", "TS-Batch"),
        "lnpbo_gibbon": ("GIBBON", "GIBBON"),
    }

    GROUP_ORDER = ["KB", "RKB", "LP", "TS-Batch", "GIBBON"]
    GROUP_COLORS = {
        "KB": "#4477AA",
        "RKB": "#EE6677",
        "LP": "#228833",
        "TS-Batch": "#CCBB44",
        "GIBBON": "#66CCEE",
    }

    acqf_data = {}
    for skey, (label, group) in ACQF_META.items():
        if skey not in strategies:
            continue
        study_means = []
        for pmid in pmids:
            seed_vals = []
            for seed in SEEDS:
                v = get_top5(result_map, pmid, skey, seed)
                if v is not None:
                    seed_vals.append(v)
            if seed_vals:
                study_means.append(np.mean(seed_vals))
        if study_means:
            m = np.mean(study_means)
            ci = bootstrap_ci(study_means) if len(study_means) >= 3 else (m, m)
            acqf_data[skey] = (label, group, m, ci[0], ci[1], len(study_means))

    if not acqf_data:
        print("  [SKIP] Not enough LNPBO data for acquisition breakdown")
        return

    # Random baseline
    random_means = []
    for pmid in pmids:
        seed_vals = []
        for seed in SEEDS:
            v = get_top5(result_map, pmid, "random", seed)
            if v is not None:
                seed_vals.append(v)
        if seed_vals:
            random_means.append(np.mean(seed_vals))
    random_mean = np.mean(random_means) if random_means else None

    # Order: group order, within each group sort by mean descending
    ordered_keys = []
    for grp in GROUP_ORDER:
        grp_keys = [k for k, v in acqf_data.items() if v[1] == grp]
        grp_keys.sort(key=lambda k: acqf_data[k][2], reverse=True)
        ordered_keys.extend(grp_keys)

    # Reverse so best (highest mean) appears at the top of the plot
    ordered_keys = list(reversed(ordered_keys))

    n = len(ordered_keys)
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 0.28 * n + 0.9))

    # Axis limits based on full-coverage strategies
    n_full = max(acqf_data[k][5] for k in ordered_keys)
    full_keys = [k for k in ordered_keys if acqf_data[k][5] == n_full]
    full_lo = min(acqf_data[k][3] for k in full_keys)
    full_hi = max(acqf_data[k][4] for k in full_keys)
    xlo = min(0.50, full_lo - 0.02)
    xhi = full_hi + 0.06
    ax.set_xlim(xlo, xhi)

    y_positions = np.arange(n)
    prev_group = None

    for i, skey in enumerate(ordered_keys):
        label, group, m, lo, hi, n_studies = acqf_data[skey]
        color = GROUP_COLORS.get(group, "#888888")

        # Clip CI bars to plot range; add arrow if truncated
        lo_draw = max(lo, xlo + 0.005)
        hi_draw = min(hi, xhi - 0.015)
        truncated = hi > xhi - 0.015

        ax.plot(m, i, "o", color=color, ms=5, zorder=5)
        ax.plot([lo_draw, hi_draw], [i, i], "-", color=color, lw=1.5, zorder=4, solid_capstyle="round")

        if truncated:
            ax.annotate(
                "",
                xy=(xhi - 0.008, i),
                xytext=(hi_draw, i),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.0, mutation_scale=5),
                zorder=4,
            )

        note = f"{m:.3f}"
        if n_studies < n_full:
            note += f" (n={n_studies})"
        if truncated:
            ax.text(m, i + 0.35, note, va="bottom", ha="center", fontsize=5.5, color="#333333")
        else:
            ax.text(hi_draw + 0.005, i, note, va="center", fontsize=5.5, color="#333333")

        if prev_group is not None and group != prev_group:
            ax.axhline(i - 0.5, color="#cccccc", lw=0.4, zorder=1)
        prev_group = group

    if random_mean is not None:
        ax.axvline(random_mean, color="#000000", ls=":", lw=0.6, alpha=0.5, zorder=1)
        ax.text(random_mean + 0.003, n - 0.3, "Random", fontsize=5, ha="left", va="top", color="#666666")

    ylabels = [acqf_data[skey][0] for skey in ordered_keys]
    ax.set_yticks(y_positions)
    ax.set_yticklabels(ylabels, fontsize=6.5)
    ax.set_xlabel("Mean Top-5% Recall")

    # Batch-group bracket labels on the right margin
    group_spans = {}
    for i, skey in enumerate(ordered_keys):
        grp = acqf_data[skey][1]
        if grp not in group_spans:
            group_spans[grp] = [i, i]
        else:
            group_spans[grp][1] = i

    for grp, (ylo, yhi) in group_spans.items():
        ymid = (ylo + yhi) / 2
        color = GROUP_COLORS.get(grp, "#888888")
        ax.text(
            1.02,
            ymid,
            grp,
            va="center",
            ha="left",
            fontsize=5.5,
            color=color,
            fontweight="bold",
            clip_on=False,
            transform=ax.get_yaxis_transform(),
        )

    light_ygrid(ax)

    fig.tight_layout()
    save_figure(fig, FIG_DIR / "fig_acquisition_breakdown.pdf")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Figure A4-D: Per-Study x Per-Encoding Heatmap
# ═══════════════════════════════════════════════════════════════════════════


def fig_encoding_heatmap(enc_rows=None):
    """Heatmap: rows = studies, columns = encodings, cell = mean top-5% recall."""
    if enc_rows is None:
        enc_rows = _load_encoding_data()
    if not enc_rows:
        print("  [SKIP] No encoding data for heatmap")
        return

    # Aggregate: study x encoding -> mean top-5% (across strategies and seeds)
    nested = defaultdict(lambda: defaultdict(list))
    for r in enc_rows:
        nested[r["study_id"]][r["encoding"]].append(r["top5"])

    study_ids = sorted(nested.keys())
    encodings = sorted({r["encoding"] for r in enc_rows})

    if not study_ids or not encodings:
        print("  [SKIP] Insufficient encoding heatmap data")
        return

    data = np.full((len(study_ids), len(encodings)), np.nan)
    for i, sid in enumerate(study_ids):
        for j, enc in enumerate(encodings):
            vals = nested[sid].get(enc, [])
            if vals:
                data[i, j] = np.mean(vals)

    # Sort rows by mean recall (ascending, so best at bottom)
    row_means = np.nanmean(data, axis=1)
    sort_idx = np.argsort(row_means)
    data = data[sort_idx]
    study_ids = [study_ids[i] for i in sort_idx]

    ns, ne = data.shape
    fig, ax = plt.subplots(figsize=(max(DOUBLE_COL * 0.7, ne * 0.7), max(3, ns * 0.22)))

    cmap = plt.cm.YlOrRd
    im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=0, vmax=1)

    for i in range(ns):
        row_vals = data[i]
        valid_mask = ~np.isnan(row_vals)
        winner = np.nanargmax(row_vals) if np.any(valid_mask) else -1
        for j in range(ne):
            v = data[i, j]
            if np.isnan(v):
                continue
            color = "white" if v > 0.65 else "#333333"
            weight = "bold" if j == winner else "normal"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5, color=color, fontweight=weight)

    display_encs = [ENCODING_DISPLAY.get(e, e) for e in encodings]
    ax.set_xticks(range(ne))
    ax.set_xticklabels(display_encs, rotation=45, ha="right", fontsize=6)
    ax.set_yticks(range(ns))
    ax.set_yticklabels(study_ids, fontsize=5)
    ax.set_ylabel("Study")
    ax.set_xlabel("Encoding")

    cbar = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02, aspect=30)
    cbar.set_label("Top-5% Recall", fontsize=6.5)
    cbar.ax.tick_params(labelsize=5.5)

    fig.tight_layout()
    save_figure(fig, FIG_DIR / "fig_encoding_heatmap.pdf")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 16: External Baselines — BO vs Predict-and-Rank
# ═══════════════════════════════════════════════════════════════════════════

BASELINES_DIR = REPO / "benchmark_results" / "baselines"


def fig_external_baselines(result_map, pmids, strategies, study_info):
    """Forest plot comparing BO strategies vs predict-and-rank baselines.

    Loads predict-and-rank results (XGBoost/RF/NGBoost P&R, AGILE P&R,
    optionally COMET) and compares against within-study BO results.
    """
    # Collect BO family results
    fam_means = family_study_means(result_map, pmids, strategies)
    bo_data = {}
    for fam in FAMILY_ORDER:
        vals = [fam_means[(fam, p)] for p in pmids if (fam, p) in fam_means]
        if vals:
            m = float(np.mean(vals))
            ci = bootstrap_ci(vals)
            bo_data[fam] = (m, ci[0], ci[1])

    # Collect predict-and-rank baselines
    baseline_data = {}
    for baseline_dir_name in [
        "predict_and_rank",
        "agile_predictor",
        "agile_finetuned",
        "comet",
    ]:
        bdir = BASELINES_DIR / baseline_dir_name
        if not bdir.exists():
            continue
        summary_path = bdir / f"{baseline_dir_name}_summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                summary = json.load(f)
            for _key, sdata in summary.items():
                study_means = [s["mean_top5"] for s in sdata["studies"]]
                if study_means:
                    m = float(np.mean(study_means))
                    ci = bootstrap_ci(study_means) if len(study_means) >= 3 else (m, m)
                    baseline_data[sdata["display"]] = (m, ci[0], ci[1])

    if not baseline_data:
        print("  No baseline results found, skipping figure.")
        return

    # Build combined list: baselines first, then BO families
    entries = []
    for name, (m, lo, hi) in baseline_data.items():
        entries.append((name, m, lo, hi, "baseline"))
    for fam in FAMILY_ORDER:
        if fam in bo_data:
            m, lo, hi = bo_data[fam]
            entries.append((fam + " (BO)", m, lo, hi, "bo"))

    # Sort by mean
    entries.sort(key=lambda x: x[1])

    fig, ax = plt.subplots(figsize=(SINGLE_COL, max(2.5, 0.25 * len(entries))))

    y = np.arange(len(entries))
    for i, (name, m, lo, hi, kind) in enumerate(entries):
        if kind == "bo":
            fam_name = name.replace(" (BO)", "")
            color = FAMILY_COLORS.get(fam_name, "#4488CC")
            marker = "o"
        else:
            color = "#CC4444"
            marker = "s"

        ax.plot(m, i, marker, color=color, ms=5, zorder=5)
        ax.plot([lo, hi], [i, i], "-", color=color, lw=1.5, zorder=4, solid_capstyle="round")
        ax.text(hi + 0.008, i, f"{m:.3f}", va="center", fontsize=5.5, color="#333333")

    ax.set_yticks(y)
    ax.set_yticklabels([e[0] for e in entries], fontsize=6)
    ax.set_xlabel("Mean Top-5% Recall")
    light_ygrid(ax)

    # Random baseline line
    if "Random" in bo_data:
        ax.axvline(bo_data["Random"][0], color="#000000", ls=":", lw=0.5, alpha=0.5)

    # Legend
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], marker="s", color="#CC4444", lw=0, ms=5, label="Predict & Rank"),
        Line2D([0], [0], marker="o", color="#4488CC", lw=0, ms=5, label="Iterative BO"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=6, framealpha=0.9)

    fig.tight_layout()
    save_figure(fig, FIG_DIR / "fig_external_baselines.pdf")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Figure: Baselines Comparison (BO vs P&R vs Foundation Models)
# ═══════════════════════════════════════════════════════════════════════════

# Colors for baseline method types
BASELINE_TYPE_COLORS = {
    "transfer": "#AA3377",  # purple — transfer learning / zero-shot
    "predict_rank": "#CC4444",  # red — single-shot predict-and-rank
    "bo": "#4477AA",  # blue — iterative BO
}


def _load_baseline_study_means():
    """Load per-study mean top-5% recall for all baseline methods.

    Returns dict: method_key -> {display, type, study_means: [float]}
    where study_means is a list of per-study mean top-5% recall values.
    """
    baselines = {}

    # 1. COMET zero-shot (transfer learning)
    comet_summary = BASELINES_DIR / "comet" / "comet_summary.json"
    if comet_summary.exists():
        with open(comet_summary) as f:
            data = json.load(f)
        for key, sdata in data.items():
            studies = sdata.get("studies", [])
            study_means = [s["mean_top5"] for s in studies]
            if study_means:
                baselines[key] = {
                    "display": sdata.get("display", key),
                    "type": "transfer",
                    "study_means": study_means,
                }

    # 2. AGILE zero-shot (transfer learning) — from agile_summary.json
    agile_zs_summary = BASELINES_DIR / "agile_predictor" / "agile_summary.json"
    if agile_zs_summary.exists():
        with open(agile_zs_summary) as f:
            data = json.load(f)
        for key, sdata in data.items():
            studies = sdata.get("studies", [])
            study_means = [s["mean_top5"] for s in studies]
            if study_means:
                baselines[f"agile_zs_{key}"] = {
                    "display": "AGILE Zero-Shot",
                    "type": "transfer",
                    "study_means": study_means,
                }

    # 3. Predict-and-rank baselines (XGB/RF/NGBoost) — from pr_summary.json
    pr_summary = BASELINES_DIR / "predict_and_rank" / "pr_summary.json"
    if pr_summary.exists():
        with open(pr_summary) as f:
            data = json.load(f)
        for key, sdata in data.items():
            studies = sdata.get("studies", [])
            study_means = [s["mean_top5"] for s in studies]
            if study_means:
                baselines[key] = {
                    "display": sdata.get("display", key),
                    "type": "predict_rank",
                    "study_means": study_means,
                }

    # 4. AGILE predictor P&R — from agile_predictor_summary.json
    agile_pr_summary = BASELINES_DIR / "agile_predictor" / "agile_predictor_summary.json"
    if agile_pr_summary.exists():
        with open(agile_pr_summary) as f:
            data = json.load(f)
        for key, sdata in data.items():
            studies = sdata.get("studies", [])
            study_means = [s["mean_top5"] for s in studies]
            if study_means:
                baselines[f"agile_pr_{key}"] = {
                    "display": sdata.get("display", key),
                    "type": "predict_rank",
                    "study_means": study_means,
                }

    # 5. AGILE fine-tuned P&R — from agile_finetuned_summary.json
    agile_ft_summary = BASELINES_DIR / "agile_finetuned" / "agile_finetuned_summary.json"
    if agile_ft_summary.exists():
        with open(agile_ft_summary) as f:
            data = json.load(f)
        for key, sdata in data.items():
            studies = sdata.get("studies", [])
            study_means = [s["mean_top5"] for s in studies]
            if study_means:
                baselines[f"agile_ft_{key}"] = {
                    "display": sdata.get("display", key),
                    "type": "predict_rank",
                    "study_means": study_means,
                }

    return baselines


def fig_baselines(result_map, pmids, strategies, study_info):
    """Grouped forest plot: transfer learning vs predict-and-rank vs iterative BO.

    Compares foundation model zero-shot, single-shot predict-and-rank, and
    iterative BO strategies on mean top-5% recall across all studies.
    """
    from matplotlib.lines import Line2D

    # ── Load baseline results ──
    baselines = _load_baseline_study_means()

    # ── Collect BO results (selected families) ──
    fam_means = family_study_means(result_map, pmids, strategies)

    # Representative BO strategies: Random, best per family
    bo_families = ["Random", "XGBoost", "RF", "NGBoost", "GP (BoTorch)", "GP (sklearn)"]
    bo_entries = {}
    for fam in bo_families:
        vals = [fam_means[(fam, p)] for p in pmids if (fam, p) in fam_means]
        if vals:
            bo_entries[fam] = vals

    if not baselines and not bo_entries:
        print("  [SKIP] No data for baselines figure")
        return

    # ── Build entry list grouped by type ──
    # Each entry: (display_name, mean, ci_lo, ci_hi, method_type, n_studies)
    entries = []

    # Group 1: Transfer learning (foundation models)
    for _key, bdata in sorted(baselines.items(), key=lambda x: x[1]["display"]):
        if bdata["type"] != "transfer":
            continue
        sm = bdata["study_means"]
        m = float(np.mean(sm))
        ci = bootstrap_ci(sm) if len(sm) >= 3 else (m, m)
        entries.append((bdata["display"], m, ci[0], ci[1], "transfer", len(sm)))

    # Group 2: Single-shot predict-and-rank
    for _key, bdata in sorted(baselines.items(), key=lambda x: x[1]["display"]):
        if bdata["type"] != "predict_rank":
            continue
        sm = bdata["study_means"]
        m = float(np.mean(sm))
        ci = bootstrap_ci(sm) if len(sm) >= 3 else (m, m)
        entries.append((bdata["display"], m, ci[0], ci[1], "predict_rank", len(sm)))

    # Group 3: Iterative BO
    for fam in bo_families:
        if fam not in bo_entries:
            continue
        vals = bo_entries[fam]
        m = float(np.mean(vals))
        ci = bootstrap_ci(vals) if len(vals) >= 3 else (m, m)
        display = fam if fam == "Random" else f"{fam} (BO)"
        entries.append((display, m, ci[0], ci[1], "bo", len(vals)))

    if not entries:
        print("  [SKIP] No entries for baselines figure")
        return

    # Sort within each group by mean, then layout groups in order
    group_order = ["transfer", "predict_rank", "bo"]
    group_labels = {
        "transfer": "Transfer Learning",
        "predict_rank": "Predict & Rank",
        "bo": "Iterative BO",
    }
    grouped = {g: [] for g in group_order}
    for e in entries:
        grouped[e[4]].append(e)
    for g in group_order:
        grouped[g].sort(key=lambda x: x[1])

    # Flatten with group separators
    flat = []
    for g in group_order:
        if grouped[g]:
            flat.extend(grouped[g])

    n = len(flat)
    fig, ax = plt.subplots(figsize=(SINGLE_COL + 0.8, max(3.5, 0.28 * n + 1.0)))

    y = np.arange(n)
    random_mean = None

    for i, (name, m, lo, hi, mtype, n_studies) in enumerate(flat):
        color = BASELINE_TYPE_COLORS.get(mtype, "#888888")
        marker = {"transfer": "D", "predict_rank": "s", "bo": "o"}.get(mtype, "o")
        ms = {"transfer": 4.5, "predict_rank": 4.5, "bo": 5}.get(mtype, 5)

        ax.plot(m, i, marker, color=color, ms=ms, zorder=5)
        ax.plot([lo, hi], [i, i], "-", color=color, lw=1.5, zorder=4, solid_capstyle="round")
        note = f"{m:.3f}"
        if n_studies < max(e[5] for e in flat):
            note += f" (n={n_studies})"
        ax.text(hi + 0.008, i, note, va="center", fontsize=5.5, color="#333333")

        if name == "Random":
            random_mean = m

    # Random baseline dashed line
    if random_mean is not None:
        ax.axvline(random_mean, color="#000000", ls=":", lw=0.6, alpha=0.5, zorder=1)

    # Group separator lines and bracket labels
    prev_type = None
    group_spans = {}
    for i, (_, _, _, _, mtype, _) in enumerate(flat):
        if mtype not in group_spans:
            group_spans[mtype] = [i, i]
        else:
            group_spans[mtype][1] = i
        if prev_type is not None and mtype != prev_type:
            ax.axhline(i - 0.5, color="#cccccc", lw=0.4, zorder=1)
        prev_type = mtype

    # Right-margin group labels
    for g in group_order:
        if g in group_spans:
            ylo, yhi = group_spans[g]
            ymid = (ylo + yhi) / 2
            color = BASELINE_TYPE_COLORS.get(g, "#888888")
            ax.text(
                1.02,
                ymid,
                group_labels[g],
                va="center",
                ha="left",
                fontsize=5.5,
                color=color,
                fontweight="bold",
                clip_on=False,
                transform=ax.get_yaxis_transform(),
            )

    ax.set_yticks(y)
    ax.set_yticklabels([e[0] for e in flat], fontsize=6)
    ax.set_xlabel("Mean Top-5% Recall")
    light_ygrid(ax)

    # Legend
    legend_elements = [
        Line2D([0], [0], marker="D", color=BASELINE_TYPE_COLORS["transfer"], lw=0, ms=4.5, label="Transfer Learning"),
        Line2D([0], [0], marker="s", color=BASELINE_TYPE_COLORS["predict_rank"], lw=0, ms=4.5, label="Predict & Rank"),
        Line2D([0], [0], marker="o", color=BASELINE_TYPE_COLORS["bo"], lw=0, ms=5, label="Iterative BO"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="lower right",
        fontsize=6,
        frameon=True,
        framealpha=0.95,
        edgecolor="#dddddd",
    )

    fig.tight_layout()
    save_figure(fig, FIG_DIR / "fig_baselines.pdf")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# SI Figure 4A: PCA Component Selection
# ═══════════════════════════════════════════════════════════════════════════


def fig_si_pca_components():
    """Two-panel: (a) n_pcs vs mean top-5% recall, (b) explained variance scree."""
    pca_dir = ABLATION_DIR / "pca"
    if not pca_dir.exists() or not any(pca_dir.iterdir()):
        print("  [WARN] PCA ablation directory missing or empty -- skipping")
        return

    rows = _load_ablation("pca")
    if not rows:
        print("  [WARN] No PCA ablation data -- skipping")
        return

    nested = defaultdict(lambda: defaultdict(list))
    for r in rows:
        nested[r["label"]][r["study_id"]].append(r["top5"])

    def _pca_sort_key(lbl):
        if lbl.startswith("pca"):
            return (0, int(lbl[3:]))
        return (1, 0)

    labels_sorted = sorted(nested.keys(), key=_pca_sort_key)
    means, cis_lo, cis_hi = [], [], []
    for lbl in labels_sorted:
        study_means = [np.mean(v) for v in nested[lbl].values()]
        if study_means:
            m = np.mean(study_means)
            ci = bootstrap_ci(study_means) if len(study_means) >= 3 else (m, m)
            means.append(m)
            cis_lo.append(ci[0])
            cis_hi.append(ci[1])
        else:
            means.append(0)
            cis_lo.append(0)
            cis_hi.append(0)

    # Attempt explained variance computation for panel (b)
    evr = None
    try:
        from LNPBO.data.compute_pcs import compute_pcs
        from LNPBO.data.lnpdb_bridge import load_lnpdb_full

        db = load_lnpdb_full()
        il_smiles = db.df["IL_SMILES"].dropna().unique().tolist()
        _, reducer, _, _ = compute_pcs(
            il_smiles,
            feature_type="count_mfp",
            n_components=20,
            reduction="pca",
            cache_name="IL",
        )
        evr = reducer.explained_variance_ratio_
    except Exception as e:
        print(f"  [NOTE] Could not compute explained variance: {e}")

    if evr is not None:
        fig, (ax, ax2) = plt.subplots(
            1,
            2,
            figsize=(DOUBLE_COL, 2.5),
            gridspec_kw={"width_ratios": [1.3, 1]},
        )
    else:
        fig, ax = plt.subplots(figsize=(SINGLE_COL + 0.5, 2.8))
        ax2 = None

    # Panel (a): bar chart
    x = np.arange(len(labels_sorted))
    bar_colors = []
    base_colors = plt.cm.Set2(np.linspace(0, 0.8, len(labels_sorted)))
    for i, lbl in enumerate(labels_sorted):
        bar_colors.append("#4477AA" if lbl == "pca5" else base_colors[i])

    for i in range(len(labels_sorted)):
        ax.bar(i, means[i], width=0.65, color=bar_colors[i], edgecolor="white", linewidth=0.3, zorder=2)
        ax.errorbar(
            i,
            means[i],
            yerr=[[means[i] - cis_lo[i]], [cis_hi[i] - means[i]]],
            fmt="none",
            ecolor="#333333",
            capsize=2.5,
            lw=0.7,
            zorder=3,
        )
        ax.text(i, cis_hi[i] + 0.005, f"{means[i]:.3f}", ha="center", va="bottom", fontsize=5.5, color="#333333")

    display_labels = []
    for lbl in labels_sorted:
        if lbl.startswith("pca"):
            display_labels.append(lbl[3:])
        elif lbl == "raw":
            display_labels.append("Raw")
        else:
            display_labels.append(lbl)

    ax.set_xticks(x)
    ax.set_xticklabels(display_labels, fontsize=6.5)
    ax.set_xlabel("Number of PCA Components")
    ax.set_ylabel("Mean Top-5% Recall")
    if means:
        ymax = max(cis_hi) + 0.04
        ax.set_ylim(0.4, min(ymax, 1.0))
    light_ygrid(ax)
    panel_label(ax, "a")

    # Panel (b): explained variance scree plot
    if ax2 is not None:
        cumvar = np.cumsum(evr)
        n_show = min(20, len(evr))
        pcs = np.arange(1, n_show + 1)
        ax2.bar(
            pcs,
            evr[:n_show] * 100,
            color="#BBBBBB",
            edgecolor="white",
            linewidth=0.3,
            width=0.7,
            zorder=2,
            label="Individual",
        )
        ax2.plot(pcs, cumvar[:n_show] * 100, "o-", color="#EE6677", ms=3, lw=1.0, zorder=3, label="Cumulative")
        ax2.axvline(5, color="#4477AA", ls="--", lw=0.7, alpha=0.7, zorder=1)
        ax2.set_xlabel("Principal Component")
        ax2.set_ylabel("Variance Explained (%)")
        ax2.set_xlim(0.5, n_show + 0.5)
        ax2.set_xticks([1, 5, 10, 15, 20])
        ymax_evr = cumvar[n_show - 1] * 100
        ax2.set_ylim(0, ymax_evr * 1.15)
        ax2.legend(fontsize=6, loc="upper left", frameon=False)
        light_ygrid(ax2)

        cum5 = cumvar[4] * 100
        ax2.annotate(
            f"5 PCs = {cum5:.0f}%",
            xy=(5, cum5),
            xytext=(8, cum5 + ymax_evr * 0.12),
            fontsize=6,
            color="#4477AA",
            fontweight="bold",
            arrowprops=dict(arrowstyle="-", color="#4477AA", lw=0.5),
        )
        panel_label(ax2, "b")

    fig.tight_layout()
    save_figure(fig, FIG_DIR / "fig_si_pca_components.pdf")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# SI Figure 4B: Batch Size Sensitivity (fixed-budget & fixed-rounds)
# ═══════════════════════════════════════════════════════════════════════════


def fig_si_batch_size():
    """Two-panel bar chart: (left) fixed-budget, (right) fixed-rounds."""
    bs_dir = ABLATION_DIR / "batch_size"
    if not bs_dir.exists() or not any(bs_dir.iterdir()):
        print("  [WARN] Batch size ablation directory missing or empty -- skipping")
        return

    rows = []
    for p in sorted(bs_dir.rglob("*.json")):
        try:
            data = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        cond = data.get("condition", {})
        label = cond.get("label", "")
        if not label:
            continue
        metrics = data.get("result", {}).get("metrics", {})
        recall = metrics.get("top_k_recall", {})
        rows.append(
            {
                "study_id": data.get("study_id", ""),
                "strategy": data.get("strategy", ""),
                "label": label,
                "seed": data.get("seed"),
                "top5": recall.get("5", 0.0),
            }
        )

    if not rows:
        print("  [WARN] No batch size ablation data -- skipping")
        return

    # Parse label: "bs4_fixed_budget" -> batch_size=4, mode="fixed_budget"
    parsed = []
    for r in rows:
        parts = r["label"].split("_", 1)
        if len(parts) != 2 or not parts[0].startswith("bs"):
            continue
        try:
            bs = int(parts[0][2:])
        except ValueError:
            continue
        parsed.append({**r, "batch_size": bs, "mode": parts[1]})

    if not parsed:
        print("  [WARN] Could not parse batch size labels -- skipping")
        return

    batch_sizes = sorted(set(r["batch_size"] for r in parsed))
    modes = ["fixed_budget", "fixed_rounds"]
    mode_titles = {"fixed_budget": "Fixed Budget", "fixed_rounds": "Fixed Rounds"}

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.5), sharey=True)

    global_min_lo = 1.0
    for ax_i, mode in enumerate(modes):
        ax = axes[ax_i]
        mode_rows = [r for r in parsed if r["mode"] == mode]

        nested = defaultdict(lambda: defaultdict(list))
        for r in mode_rows:
            nested[r["batch_size"]][r["study_id"]].append(r["top5"])

        means, cis_lo, cis_hi = [], [], []
        for bs in batch_sizes:
            study_means = [np.mean(v) for v in nested[bs].values()]
            if study_means:
                m = np.mean(study_means)
                ci = bootstrap_ci(study_means) if len(study_means) >= 3 else (m, m)
                means.append(m)
                cis_lo.append(ci[0])
                cis_hi.append(ci[1])
            else:
                means.append(0)
                cis_lo.append(0)
                cis_hi.append(0)

        if cis_lo:
            global_min_lo = min(global_min_lo, min(cis_lo))

        x = np.arange(len(batch_sizes))
        colors = plt.cm.Set2(np.linspace(0, 0.8, len(batch_sizes)))

        for i in range(len(batch_sizes)):
            ax.bar(i, means[i], width=0.65, color=colors[i], edgecolor="white", linewidth=0.3, zorder=2)
            ax.errorbar(
                i,
                means[i],
                yerr=[[means[i] - cis_lo[i]], [cis_hi[i] - means[i]]],
                fmt="none",
                ecolor="#333333",
                capsize=2.5,
                lw=0.7,
                zorder=3,
            )
            ax.text(i, cis_hi[i] + 0.005, f"{means[i]:.3f}", ha="center", va="bottom", fontsize=5.5, color="#333333")

        ax.set_xticks(x)
        ax.set_xticklabels([str(bs) for bs in batch_sizes], fontsize=6.5)
        ax.set_xlabel("Batch Size")
        ax.set_title(mode_titles.get(mode, mode), fontsize=7)
        light_ygrid(ax)
        panel_label(ax, chr(97 + ax_i))

    # Set shared y-axis with a sensible floor (avoid starting at 0)
    y_floor = max(0.0, global_min_lo - 0.08)
    y_floor = round(y_floor * 10) / 10  # snap to nearest 0.1
    axes[0].set_ylim(y_floor, None)
    axes[0].set_ylabel("Mean Top-5% Recall")

    fig.tight_layout(w_pad=1.5)
    save_figure(fig, FIG_DIR / "fig_si_batch_size.pdf")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# SI Figure 4C: Acquisition Function Breakdown (LNPBO strategies only)
# ═══════════════════════════════════════════════════════════════════════════


def fig_si_acquisition(result_map, pmids, strategies):
    """Forest plot of LNPBO acquisition function variants with 95% CI."""
    LNPBO_DISPLAY = {
        "lnpbo_ucb": "UCB",
        "lnpbo_ei": "EI",
        "lnpbo_logei": "LogEI",
        "lnpbo_lp_ei": "LP-EI",
        "lnpbo_lp_logei": "LP-LogEI",
        "lnpbo_ts_batch": "TS-Batch",
        "lnpbo_gibbon": "GIBBON",
        "lnpbo_rkb_logei": "RKB-LogEI",
    }

    acqf_data = {}
    for skey, display in LNPBO_DISPLAY.items():
        if skey not in strategies:
            continue
        study_means = []
        for pmid in pmids:
            seed_vals = []
            for seed in SEEDS:
                v = get_top5(result_map, pmid, skey, seed)
                if v is not None:
                    seed_vals.append(v)
            if seed_vals:
                study_means.append(np.mean(seed_vals))
        if study_means:
            m = np.mean(study_means)
            ci = bootstrap_ci(study_means) if len(study_means) >= 3 else (m, m)
            acqf_data[skey] = (display, m, ci[0], ci[1], len(study_means))

    if not acqf_data:
        print("  [WARN] No LNPBO acquisition data found -- skipping")
        return

    # Random baseline
    random_means = []
    for pmid in pmids:
        seed_vals = []
        for seed in SEEDS:
            v = get_top5(result_map, pmid, "random", seed)
            if v is not None:
                seed_vals.append(v)
        if seed_vals:
            random_means.append(np.mean(seed_vals))
    random_mean = np.mean(random_means) if random_means else None

    # Sort by mean recall (best at top)
    ordered = sorted(acqf_data.keys(), key=lambda k: acqf_data[k][1])
    n = len(ordered)
    n_full = max(acqf_data[k][4] for k in ordered)

    fig, ax = plt.subplots(figsize=(SINGLE_COL, 0.28 * n + 0.9))

    y = np.arange(n)
    palette = plt.cm.tab10(np.linspace(0, 0.9, n))

    for i, skey in enumerate(ordered):
        display, m, lo, hi, n_studies = acqf_data[skey]
        color = palette[i]
        ax.plot(m, i, "o", color=color, ms=5, zorder=5)
        ax.plot([lo, hi], [i, i], "-", color=color, lw=1.5, zorder=4, solid_capstyle="round")
        note = f"{m:.3f}"
        if n_studies < n_full:
            note += f" (n={n_studies})"
        ax.text(hi + 0.006, i, note, va="center", fontsize=5.5, color="#333333")

    if random_mean is not None:
        ax.axvline(random_mean, color="#000000", ls=":", lw=0.6, alpha=0.5, zorder=1)
        ax.text(random_mean + 0.003, n - 0.3, "Random", fontsize=5, ha="left", va="top", color="#666666")

    ylabels = [acqf_data[skey][0] for skey in ordered]
    ax.set_yticks(y)
    ax.set_yticklabels(ylabels, fontsize=6.5)
    ax.set_xlabel("Mean Top-5% Recall")

    all_lo = min(acqf_data[k][2] for k in ordered)
    all_hi = max(acqf_data[k][3] for k in ordered)
    margin = (all_hi - all_lo) * 0.15
    ax.set_xlim(all_lo - margin, all_hi + margin + 0.06)
    light_ygrid(ax)

    fig.tight_layout()
    save_figure(fig, FIG_DIR / "fig_si_acquisition.pdf")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# SI Figure 4D: Encoding Heatmap (per-study x per-encoding)
# ═══════════════════════════════════════════════════════════════════════════


def fig_si_encoding_heatmap():
    """Heatmap: rows = studies (PMIDs), columns = encodings, cells = mean top-5% recall.

    Uses all 34 strategies (unfiltered) for the extended SI figure.
    """
    enc_dir = ABLATION_DIR / "encoding"
    if not enc_dir.exists() or not any(enc_dir.iterdir()):
        print("  [WARN] Encoding ablation directory missing or empty -- skipping")
        return

    enc_rows = _load_encoding_data(filter_strategies=False)
    if not enc_rows:
        print("  [WARN] No encoding data for heatmap -- skipping")
        return

    nested = defaultdict(lambda: defaultdict(list))
    for r in enc_rows:
        nested[r["study_id"]][r["encoding"]].append(r["top5"])

    study_ids = sorted(nested.keys())
    encodings = sorted({r["encoding"] for r in enc_rows})

    if not study_ids or not encodings:
        print("  [WARN] Insufficient encoding heatmap data -- skipping")
        return

    data = np.full((len(study_ids), len(encodings)), np.nan)
    for i, sid in enumerate(study_ids):
        for j, enc in enumerate(encodings):
            vals = nested[sid].get(enc, [])
            if vals:
                data[i, j] = np.mean(vals)

    # Sort rows by mean recall (ascending, so best studies at bottom)
    row_means = np.nanmean(data, axis=1)
    sort_idx = np.argsort(row_means)
    data = data[sort_idx]
    study_ids = [study_ids[i] for i in sort_idx]

    ns, ne = data.shape
    fig, ax = plt.subplots(figsize=(max(DOUBLE_COL * 0.7, ne * 0.7), max(3, ns * 0.22)))

    cmap = plt.cm.YlOrRd
    im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=0, vmax=1)

    for i in range(ns):
        row_vals = data[i]
        valid_mask = ~np.isnan(row_vals)
        winner = np.nanargmax(row_vals) if np.any(valid_mask) else -1
        for j in range(ne):
            v = data[i, j]
            if np.isnan(v):
                continue
            color = "white" if v > 0.65 else "#333333"
            weight = "bold" if j == winner else "normal"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5, color=color, fontweight=weight)

    display_encs = [ENCODING_DISPLAY.get(e, e) for e in encodings]
    ax.set_xticks(range(ne))
    ax.set_xticklabels(display_encs, rotation=45, ha="right", fontsize=6)
    ax.set_yticks(range(ns))
    ax.set_yticklabels(study_ids, fontsize=5)
    ax.set_ylabel("Study (PMID)")
    ax.set_xlabel("Encoding")

    cbar = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02, aspect=30)
    cbar.set_label("Mean Top-5% Recall", fontsize=6.5)
    cbar.ax.tick_params(labelsize=5.5)

    fig.tight_layout()
    save_figure(fig, FIG_DIR / "fig_si_encoding_heatmap.pdf")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# TOC Graphic
# ═══════════════════════════════════════════════════════════════════════════

# Display names for TOC (shorter than full family names)
_TOC_DISPLAY = {
    "Random": "Random",
    "GP (BoTorch)": "GP-BO",
    "GP (sklearn)": "GP (sklearn)",
    "XGBoost": "XGBoost",
    "CASMOPolitan": "CASMOPolitan",
    "RF": "RF",
    "NGBoost": "NGBoost",
    "Deep Ensemble": "Deep Ensemble",
}

# Blue palette: darker for higher-ranked families, lighter for lower
_TOC_BAR_COLORS = {
    "Random": "#b0bec5",
    "GP (BoTorch)": "#90a4ae",
    "GP (sklearn)": "#90a4ae",
    "Deep Ensemble": "#78909c",
    "XGBoost": "#546e7a",
    "CASMOPolitan": "#455a64",
    "RF": "#37474f",
    "NGBoost": "#263238",
}


def fig_toc_graphic(result_map, pmids, strategies):
    """ACS JCIM TOC graphic: horizontal bar chart of family-level top-5% recall."""
    fam_means = family_study_means(result_map, pmids, strategies)

    data = {}
    for fam in FAMILY_ORDER:
        vals = [fam_means[(fam, p)] for p in pmids if (fam, p) in fam_means]
        if vals:
            data[fam] = np.mean(vals)

    ordered = sorted(data.keys(), key=lambda f: data[f])
    n_families = len(ordered)
    n_strategies = len([s for s in strategies if s != "random"])
    n_studies = len(pmids)
    random_recall = data.get("Random", 0.532)

    fig, ax = plt.subplots(figsize=(3.25, 1.75))

    y = np.arange(n_families)
    bar_height = 0.65

    for i, fam in enumerate(ordered):
        m = data[fam]
        color = _TOC_BAR_COLORS.get(fam, "#546e7a")
        ax.barh(i, m, height=bar_height, color=color, edgecolor="none", zorder=3)
        ax.text(
            m - 0.012, i, f"{m:.3f}",
            va="center", ha="right",
            fontsize=7, fontweight="bold", color="white", zorder=4,
        )

    # Multiplier annotations for top-2 families
    for fam in ordered[-2:]:
        i = ordered.index(fam)
        m = data[fam]
        mult = m / random_recall
        ax.text(
            m + 0.012, i, f"{mult:.2f}x",
            va="center", ha="left",
            fontsize=6.5, fontweight="bold", color="#333333", zorder=4,
        )

    ax.axvline(random_recall, color="#9e9e9e", ls="--", lw=0.8, alpha=0.7, zorder=2)

    ax.set_yticks(y)
    ax.set_yticklabels(
        [_TOC_DISPLAY.get(f, f) for f in ordered],
        fontsize=7, fontweight="bold",
    )
    ax.set_xlabel("Top-5% Recall", fontsize=7)
    ax.set_xlim(0.42, 0.80)
    ax.tick_params(axis="x", labelsize=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    ax.annotate(
        f"{n_strategies} strategies \u00b7 {n_studies} studies \u00b7 {len(SEEDS)} seeds",
        xy=(1.0, 0), xycoords="axes fraction",
        xytext=(0, -26), textcoords="offset points",
        ha="right", va="top",
        fontsize=5.5, fontstyle="italic", color="#757575",
    )

    fig.tight_layout(pad=0.3)
    fig.subplots_adjust(bottom=0.22)
    save_figure(fig, FIG_DIR / "fig_toc_graphic.pdf")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════


def main():
    setup_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading within-study benchmark results...")
    results = load_all_results()
    study_info, result_map, pmids, strategies = build_tables(results)
    print(f"  {len(results)} results, {len(pmids)} studies, {len(strategies)} strategies")

    print("\n[0/22] TOC graphic...")
    fig_toc_graphic(result_map, pmids, strategies)

    print("\n[1/22] Strategy rankings (all 34)...")
    fig_strategy_rankings(result_map, pmids, strategies)

    print("\n[2/20] Family rankings...")
    fig_family_rankings(result_map, pmids, strategies, study_info)

    print("\n[3/20] Critical difference diagram...")
    fig_critical_difference(result_map, pmids, strategies, study_info)

    print("\n[4/21] Convergence curves...")
    fig_convergence(result_map, pmids, strategies, study_info)

    print("\n[5/21] Convergence z-score (SI)...")
    fig_convergence_zscore(result_map, pmids, strategies)

    print("\n[6/21] Study-type stratified...")
    fig_study_type_stratified(result_map, pmids, strategies, study_info)

    print("\n[6/20] Per-study heatmap...")
    fig_heatmap(result_map, pmids, strategies, study_info)

    print("\n[8/21] Pairwise win matrix...")
    fig_win_matrix(result_map, pmids, strategies, study_info)

    print("\n[9/21] Encoding comparison...")
    fig_encoding()

    print("\n[10/21] Cross-study transfer...")
    fig_cross_study()

    print("\n[11/21] Ablation summary...")
    fig_ablation_summary()

    print("\n[12/21] Data landscape...")
    fig_data_landscape()

    print("\n[13/21] Variance decomposition...")
    fig_variance(result_map, pmids, strategies, study_info)

    print("\n[14/21] PCA justification...")
    fig_pca_justification()

    print("\n[15/21] Batch sensitivity...")
    fig_batch_sensitivity()

    print("\n[16/21] Kernel comparison...")
    fig_kernel_comparison()

    print("\n[17/21] Kappa sensitivity...")
    fig_kappa_sensitivity()

    print("\n[18/21] Acquisition breakdown...")
    fig_acquisition_breakdown(result_map, pmids, strategies, study_info)

    print("\n[19/21] Encoding heatmap...")
    fig_encoding_heatmap()

    print("\n[20/21] External baselines...")
    fig_external_baselines(result_map, pmids, strategies, study_info)

    print("\n[21/21] Baselines comparison (BO vs P&R vs Foundation Models)...")
    fig_baselines(result_map, pmids, strategies, study_info)

    # ── SI appendix figures ──────────────────────────────────────────────
    print("\n[SI-4A] PCA component selection...")
    fig_si_pca_components()

    print("\n[SI-4B] Batch size sensitivity...")
    fig_si_batch_size()

    print("\n[SI-4C] Acquisition function breakdown...")
    fig_si_acquisition(result_map, pmids, strategies)

    print("\n[SI-4D] Encoding heatmap...")
    fig_si_encoding_heatmap()

    print("\n[SI-4E] Mixed discrete-continuous BO...")
    try:
        from paper.gen_fig_mixed_bo import main as gen_mixed_bo
        gen_mixed_bo()
    except Exception as exc:
        print(f"  [SKIP] Mixed BO figure failed: {exc}")

    print("\nAll figures generated in", FIG_DIR)


if __name__ == "__main__":
    main()
