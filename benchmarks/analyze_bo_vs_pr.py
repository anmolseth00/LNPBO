"""Deep statistical analysis: iterative BO vs single-shot Predict-and-Rank.

Compares within-study BO strategies against P&R and AGILE baselines across
all benchmark studies.  The "best BO" metric uses oracle-best strategy
selection per seed (i.e., the best of ~20 strategies for each seed),
which inflates BO more than P&R (only 3 surrogates).  The "same-surrogate"
tests use matched surrogates for an apples-to-apples comparison.
"""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

from .constants import SEEDS
from .strategy_registry import STRATEGY_FAMILY as _REG_FAMILY

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "paper"))
from figure_style import (
    DOUBLE_COL,
    FAMILY_COLORS,
    STUDY_TYPE_COLORS,
    bootstrap_ci,
    light_ygrid,
    panel_label,
    save_figure,
    setup_style,
)

WITHIN_DIR = ROOT / "benchmark_results" / "within_study"
PR_DIR = ROOT / "benchmark_results" / "baselines" / "predict_and_rank"
AGILE_DIR = ROOT / "benchmark_results" / "baselines" / "agile_predictor"
META_PATH = ROOT / "experiments" / "data_integrity" / "studies_with_ids.json"
OUT_JSON = ROOT / "benchmark_results" / "bo_vs_pr_analysis.json"
OUT_FIG = ROOT / "paper" / "figures" / "fig_bo_vs_pr_analysis.pdf"

# BO strategies to consider (iterative, closed-loop)
BO_STRATEGIES = [
    "discrete_ngboost_ucb",
    "discrete_rf_ucb",
    "discrete_rf_ts",
    "discrete_rf_ts_batch",
    "discrete_xgb_ucb",
    "discrete_xgb_greedy",
    "discrete_xgb_cqr",
    "discrete_xgb_online_conformal",
    "discrete_xgb_ucb_ts_batch",
    "discrete_deep_ensemble",
    "discrete_gp_ucb",
    "casmopolitan_ei",
    "casmopolitan_ucb",
    "lnpbo_ucb",
    "lnpbo_logei",
    "lnpbo_ei",
    "lnpbo_ts_batch",
    "lnpbo_rkb_logei",
    "lnpbo_lp_ei",
    "lnpbo_lp_logei",
    "lnpbo_tanimoto_logei",
    "lnpbo_tanimoto_ts",
    "lnpbo_gibbon",
    "lnpbo_pls_logei",
    "lnpbo_pls_lp_logei",
]

# The 3 "matched" BO strategies that share a surrogate with P&R
MATCHED_BO_STRATEGIES = ["discrete_xgb_ucb", "discrete_rf_ucb", "discrete_ngboost_ucb"]

# P&R surrogates
PR_SURROGATES = ["xgb", "rf", "ngboost"]
AGILE_SURROGATES = ["xgb", "rf"]

# Map P&R surrogates to their matched BO strategy
SURROGATE_BO_MAP = {
    "xgb": ["discrete_xgb_ucb"],
    "rf": ["discrete_rf_ucb"],
    "ngboost": ["discrete_ngboost_ucb"],
}

# Surrogate family mapping for BO strategies — re-labeled for this analysis
SURROGATE_FAMILY = {
    k: ("LNPBO" if v == "LNPBO (GP)" else v)
    for k, v in _REG_FAMILY.items()
}


def load_json(path):
    with open(path) as f:
        return json.load(f)


def find_matching_studies():
    """Find study IDs that exist in both within_study and predict_and_rank."""
    bo_dirs = {
        d
        for d in os.listdir(WITHIN_DIR)
        if (WITHIN_DIR / d).is_dir() and not d.startswith(("analysis", "gap", "gibbon", "tsbatch"))
    }
    pr_dirs = {d for d in os.listdir(PR_DIR) if (PR_DIR / d).is_dir()}
    agile_dirs = {d for d in os.listdir(AGILE_DIR) if (AGILE_DIR / d).is_dir()}
    common = sorted(bo_dirs & pr_dirs)
    agile_common = sorted(bo_dirs & agile_dirs)
    print(f"Studies in within_study: {len(bo_dirs)}")
    print(f"Studies in predict_and_rank: {len(pr_dirs)}")
    print(f"Studies in agile_predictor: {len(agile_dirs)}")
    print(f"Common (BO & P&R): {len(common)}")
    print(f"Common (BO & AGILE): {len(agile_common)}")

    # Studies only in BO
    bo_only = sorted(bo_dirs - pr_dirs)
    if bo_only:
        print(f"BO-only studies (no P&R match): {bo_only}")
    pr_only = sorted(pr_dirs - bo_dirs)
    if pr_only:
        print(f"P&R-only studies (no BO match): {pr_only}")

    return common, agile_common


def load_bo_results(study_id):
    """Load all BO strategy results for a study. Returns {strategy: {seed: data}}."""
    study_dir = WITHIN_DIR / study_id
    results = defaultdict(dict)
    for strat in BO_STRATEGIES:
        for seed in SEEDS:
            fpath = study_dir / f"{strat}_s{seed}.json"
            if fpath.exists():
                results[strat][seed] = load_json(fpath)
    return dict(results)


def load_random_results(study_id):
    """Load random baseline results."""
    results = {}
    for seed in SEEDS:
        fpath = WITHIN_DIR / study_id / f"random_s{seed}.json"
        if fpath.exists():
            results[seed] = load_json(fpath)
    return results


def load_pr_results(study_id):
    """Load P&R results. Returns {surrogate: {seed: data}}."""
    study_dir = PR_DIR / study_id
    results = defaultdict(dict)
    for surr in PR_SURROGATES:
        for seed in SEEDS:
            fpath = study_dir / f"predict_rank_{surr}_s{seed}.json"
            if fpath.exists():
                results[surr][seed] = load_json(fpath)
    return dict(results)


def load_agile_results(study_id):
    """Load AGILE results. Returns {surrogate: {seed: data}}."""
    study_dir = AGILE_DIR / study_id
    results = defaultdict(dict)
    for surr in AGILE_SURROGATES:
        for seed in SEEDS:
            fpath = study_dir / f"agile_{surr}_s{seed}.json"
            if fpath.exists():
                results[surr][seed] = load_json(fpath)
    return dict(results)


def load_study_metadata():
    """Load study metadata from studies_with_ids.json."""
    studies = load_json(META_PATH)
    meta = {}
    for s in studies:
        sid = s["study_id"]
        meta[sid] = s
    return meta


def get_recall5(data):
    """Extract top-5% recall from result JSON."""
    return data["result"]["metrics"]["top_k_recall"]["5"]


def get_best_so_far(data):
    """Extract best_so_far array (length = n_rounds + 1, index 0 = seed pool)."""
    return data["result"]["best_so_far"]


def get_n_evaluated(data):
    """Extract n_evaluated array."""
    return data["result"]["n_evaluated"]


def mean_recall_across_seeds(seed_results):
    """Compute mean top-5% recall across seeds."""
    vals = [get_recall5(d) for d in seed_results.values()]
    return np.mean(vals) if vals else np.nan


def recall_per_seed(seed_results):
    """Return dict of {seed: recall}."""
    return {s: get_recall5(d) for s, d in seed_results.items()}


def main():
    setup_style()

    print("=" * 80)
    print("BO vs Predict-and-Rank: Deep Statistical Analysis")
    print("=" * 80)
    print()

    common_studies, agile_studies = find_matching_studies()
    meta = load_study_metadata()
    print()

    # ── Collect all data ──────────────────────────────────────────────────
    all_data = {}
    for sid in common_studies:
        bo = load_bo_results(sid)
        pr = load_pr_results(sid)
        ag = load_agile_results(sid) if sid in agile_studies else {}
        rnd = load_random_results(sid)
        study_info = None
        # Get study_info from any result file
        for strat_results in bo.values():
            for d in strat_results.values():
                study_info = d.get("study_info", {})
                break
            break
        if study_info is None:
            for surr_results in pr.values():
                for d in surr_results.values():
                    study_info = d.get("study_info", {})
                    break
                break
        all_data[sid] = {
            "bo": bo,
            "pr": pr,
            "agile": ag,
            "random": rnd,
            "study_info": study_info,
            "meta": meta.get(sid, {}),
        }

    # ══════════════════════════════════════════════════════════════════════
    # A. Study-level BO vs P&R comparison
    # ══════════════════════════════════════════════════════════════════════
    print("=" * 80)
    print("A. STUDY-LEVEL BO vs P&R COMPARISON")
    print("=" * 80)
    print()

    study_rows = []
    for sid in common_studies:
        d = all_data[sid]
        info = d["study_info"]
        n_form = info.get("n_formulations", "?")
        stype = info.get("study_type", "?")

        # Best BO: for each seed, take the best strategy, then average
        bo_per_seed = {}
        for seed in SEEDS:
            best_recall = -1
            best_strat = None
            for strat, seed_results in d["bo"].items():
                if seed in seed_results:
                    r = get_recall5(seed_results[seed])
                    if r > best_recall:
                        best_recall = r
                        best_strat = strat
            if best_strat is not None:
                bo_per_seed[seed] = best_recall
        bo_mean = np.mean(list(bo_per_seed.values())) if bo_per_seed else np.nan

        # Best P&R: for each seed, take the best surrogate, then average
        pr_per_seed = {}
        for seed in SEEDS:
            best_recall = -1
            for _surr, seed_results in d["pr"].items():
                if seed in seed_results:
                    r = get_recall5(seed_results[seed])
                    if r > best_recall:
                        best_recall = r
            if best_recall >= 0:
                pr_per_seed[seed] = best_recall
        pr_mean = np.mean(list(pr_per_seed.values())) if pr_per_seed else np.nan

        # Random
        rnd_recalls = [get_recall5(v) for v in d["random"].values()]
        rnd_mean = np.mean(rnd_recalls) if rnd_recalls else np.nan

        diff = bo_mean - pr_mean

        study_rows.append(
            {
                "study_id": sid,
                "n_formulations": n_form,
                "study_type": stype,
                "n_unique_il": info.get("n_unique_il", "?"),
                "bo_mean": bo_mean,
                "pr_mean": pr_mean,
                "random_mean": rnd_mean,
                "diff": diff,
                "bo_per_seed": bo_per_seed,
                "pr_per_seed": pr_per_seed,
            }
        )

    # Print table
    print(f"{'Study':<25s} {'N':>5s} {'Type':<30s} {'BO':>6s} {'P&R':>6s} {'Rand':>6s} {'Diff':>7s} {'BO wins':>7s}")
    print("-" * 105)
    for r in sorted(study_rows, key=lambda x: x["diff"], reverse=True):
        bo_wins = sum(
            1
            for s in SEEDS
            if s in r["bo_per_seed"] and s in r["pr_per_seed"] and r["bo_per_seed"][s] > r["pr_per_seed"][s]
        )
        n_paired = sum(1 for s in SEEDS if s in r["bo_per_seed"] and s in r["pr_per_seed"])
        print(
            f"{r['study_id']:<25s} {r['n_formulations']:>5} {r['study_type']:<30s} "
            f"{r['bo_mean']:>6.3f} {r['pr_mean']:>6.3f} {r['random_mean']:>6.3f} "
            f"{r['diff']:>+7.3f} {bo_wins}/{n_paired}"
        )
    print()

    # Categorize
    bo_big_win = [r for r in study_rows if r["diff"] > 0.05]
    pr_wins = [r for r in study_rows if r["diff"] <= 0]
    pr_close = [r for r in study_rows if 0 < r["diff"] <= 0.05]
    print(f"Studies where BO > P&R by >5%: {len(bo_big_win)}")
    for r in bo_big_win:
        print(f"  {r['study_id']}: +{r['diff']:.3f}")
    print(f"Studies where P&R matches or beats BO: {len(pr_wins)}")
    for r in pr_wins:
        print(f"  {r['study_id']}: {r['diff']:+.3f}")
    print(f"Studies where BO leads by <=5%: {len(pr_close)}")
    print()

    # ══════════════════════════════════════════════════════════════════════
    # A2. Same-surrogate BO vs P&R comparison (apples-to-apples)
    # ══════════════════════════════════════════════════════════════════════
    print("=" * 80)
    print("A2. SAME-SURROGATE BO vs P&R (apples-to-apples)")
    print("=" * 80)
    print()

    same_surr_data = {}
    for surr in PR_SURROGATES:
        print(f"--- {surr.upper()} ---")
        bo_strats = SURROGATE_BO_MAP[surr]
        bo_vals, pr_vals = [], []
        for sid in common_studies:
            d = all_data[sid]
            bo_seed_vals = []
            for seed in SEEDS:
                best = -1
                for strat in bo_strats:
                    if strat in d["bo"] and seed in d["bo"][strat]:
                        r = get_recall5(d["bo"][strat][seed])
                        if r > best:
                            best = r
                if best >= 0:
                    bo_seed_vals.append(best)
            pr_seed_vals = []
            if surr in d["pr"]:
                for seed in SEEDS:
                    if seed in d["pr"][surr]:
                        pr_seed_vals.append(get_recall5(d["pr"][surr][seed]))
            if bo_seed_vals and pr_seed_vals:
                bo_vals.append(np.mean(bo_seed_vals))
                pr_vals.append(np.mean(pr_seed_vals))

        bo_arr = np.array(bo_vals)
        pr_arr = np.array(pr_vals)
        diff = bo_arr - pr_arr
        same_surr_data[surr] = {"bo": bo_arr, "pr": pr_arr, "diff": diff}
        print(f"  N studies: {len(bo_vals)}")
        print(f"  BO mean: {bo_arr.mean():.4f}, P&R mean: {pr_arr.mean():.4f}")
        print(f"  Mean diff (BO - P&R): {diff.mean():+.4f} +/- {diff.std():.4f}")
        print(f"  BO wins: {(diff > 0).sum()}/{len(diff)}, ties: {(diff == 0).sum()}")
        print()

    # Matched-surrogate aggregate: best-of-3 BO (xgb/rf/ngboost UCB) vs best-of-3 P&R
    print("--- MATCHED AGGREGATE (best of xgb/rf/ngboost) ---")
    matched_bo_vals, matched_pr_vals = [], []
    for sid in common_studies:
        d = all_data[sid]
        bo_seed_agg = {}
        pr_seed_agg = {}
        for seed in SEEDS:
            bo_best = -1
            pr_best = -1
            for surr in PR_SURROGATES:
                for strat in SURROGATE_BO_MAP[surr]:
                    if strat in d["bo"] and seed in d["bo"][strat]:
                        r = get_recall5(d["bo"][strat][seed])
                        if r > bo_best:
                            bo_best = r
                if surr in d["pr"] and seed in d["pr"][surr]:
                    r = get_recall5(d["pr"][surr][seed])
                    if r > pr_best:
                        pr_best = r
            if bo_best >= 0:
                bo_seed_agg[seed] = bo_best
            if pr_best >= 0:
                pr_seed_agg[seed] = pr_best
        if bo_seed_agg and pr_seed_agg:
            matched_bo_vals.append(np.mean(list(bo_seed_agg.values())))
            matched_pr_vals.append(np.mean(list(pr_seed_agg.values())))
    matched_bo_arr = np.array(matched_bo_vals)
    matched_pr_arr = np.array(matched_pr_vals)
    matched_diff = matched_bo_arr - matched_pr_arr
    print(f"  N studies: {len(matched_bo_vals)}")
    print(f"  BO mean: {matched_bo_arr.mean():.4f}, P&R mean: {matched_pr_arr.mean():.4f}")
    print(f"  Mean diff: {matched_diff.mean():+.4f} +/- {matched_diff.std():.4f}")
    print(f"  BO wins: {(matched_diff > 0).sum()}/{len(matched_diff)}, ties: {(matched_diff == 0).sum()}")
    if len(matched_diff) >= 6 and (matched_diff != 0).sum() >= 1:
        w_m, p_m = stats.wilcoxon(matched_bo_arr, matched_pr_arr, alternative="two-sided")
        cd_m = matched_diff.mean() / matched_diff.std() if matched_diff.std() > 0 else 0
        print(f"  Wilcoxon: W={w_m:.1f}, p={p_m:.6f}, Cohen's d={cd_m:.3f}")
    print()

    # ══════════════════════════════════════════════════════════════════════
    # A3. AGILE vs P&R vs BO
    # ══════════════════════════════════════════════════════════════════════
    if agile_studies:
        print("=" * 80)
        print("A3. AGILE vs P&R vs BO")
        print("=" * 80)
        print()

        for surr in AGILE_SURROGATES:
            ag_vals, pr_vals_ag, bo_vals_ag = [], [], []
            for sid in agile_studies:
                d = all_data.get(sid)
                if d is None:
                    continue
                ag_seed = []
                if surr in d.get("agile", {}):
                    for seed in SEEDS:
                        if seed in d["agile"][surr]:
                            ag_seed.append(get_recall5(d["agile"][surr][seed]))
                pr_seed = []
                if surr in d.get("pr", {}):
                    for seed in SEEDS:
                        if seed in d["pr"][surr]:
                            pr_seed.append(get_recall5(d["pr"][surr][seed]))
                bo_strats = SURROGATE_BO_MAP.get(surr, [])
                bo_seed = []
                for seed in SEEDS:
                    best = -1
                    for strat in bo_strats:
                        if strat in d.get("bo", {}) and seed in d["bo"][strat]:
                            r = get_recall5(d["bo"][strat][seed])
                            if r > best:
                                best = r
                    if best >= 0:
                        bo_seed.append(best)
                if ag_seed and pr_seed and bo_seed:
                    ag_vals.append(np.mean(ag_seed))
                    pr_vals_ag.append(np.mean(pr_seed))
                    bo_vals_ag.append(np.mean(bo_seed))

            if ag_vals:
                print(f"--- {surr.upper()} (AGILE features) ---")
                print(f"  N studies: {len(ag_vals)}")
                print(f"  BO mean: {np.mean(bo_vals_ag):.4f}")
                print(f"  P&R (LANTERN) mean: {np.mean(pr_vals_ag):.4f}")
                print(f"  AGILE P&R mean: {np.mean(ag_vals):.4f}")
                diff_ag_pr = np.array(ag_vals) - np.array(pr_vals_ag)
                print(f"  AGILE - P&R(LANTERN): {diff_ag_pr.mean():+.4f}")
                print()

    # ══════════════════════════════════════════════════════════════════════
    # B. What characterizes studies where BO helps?
    # ══════════════════════════════════════════════════════════════════════
    print("=" * 80)
    print("B. STUDY CHARACTERISTICS CORRELATED WITH BO ADVANTAGE")
    print("=" * 80)
    print()

    diffs = np.array([r["diff"] for r in study_rows])
    n_forms = np.array([r["n_formulations"] for r in study_rows if isinstance(r["n_formulations"], (int, float))])
    n_ils = np.array([r["n_unique_il"] for r in study_rows if isinstance(r["n_unique_il"], (int, float))])
    diffs_valid = np.array([r["diff"] for r in study_rows if isinstance(r["n_formulations"], (int, float))])

    # Correlation with study size
    if len(n_forms) == len(diffs_valid) and len(n_forms) >= 3:
        r_size, p_size = stats.spearmanr(n_forms, diffs_valid)
        print(f"Spearman corr (BO advantage vs study size): r={r_size:.3f}, p={p_size:.4f}")

    # Correlation with n unique ILs
    diffs_il = np.array([r["diff"] for r in study_rows if isinstance(r["n_unique_il"], (int, float))])
    if len(n_ils) == len(diffs_il) and len(n_ils) >= 3:
        r_il, p_il = stats.spearmanr(n_ils, diffs_il)
        print(f"Spearman corr (BO advantage vs n_unique_il): r={r_il:.3f}, p={p_il:.4f}")

    # By study type
    print()
    print("By study type:")
    type_groups = defaultdict(list)
    for r in study_rows:
        type_groups[r["study_type"]].append(r["diff"])
    for stype, vals in sorted(type_groups.items()):
        print(
            f"  {stype}: N={len(vals)}, mean diff={np.mean(vals):+.4f}, "
            f"median={np.median(vals):+.4f}, BO wins={sum(1 for v in vals if v > 0)}/{len(vals)}"
        )

    # Pool fraction analysis
    print()
    print("Pool fraction vs BO advantage:")
    fracs = []
    for r in study_rows:
        info = all_data[r["study_id"]]["study_info"]
        n_form = info.get("n_formulations", 0)
        n_seed = info.get("n_seed", 0)
        n_rounds = info.get("n_rounds", 0)
        batch = info.get("batch_size", 0)
        if n_form > 0:
            frac = (n_seed + n_rounds * batch) / n_form
            fracs.append(frac)
        else:
            fracs.append(np.nan)
    fracs = np.array(fracs)
    valid = ~np.isnan(fracs) & ~np.isnan(diffs)
    if valid.sum() >= 3:
        r_frac, p_frac = stats.spearmanr(fracs[valid], diffs[valid])
        print(f"  Spearman corr (BO advantage vs pool fraction evaluated): r={r_frac:.3f}, p={p_frac:.4f}")
        print(
            f"  Pool fractions: min={fracs[valid].min():.2f}, median={np.median(fracs[valid]):.2f}, "
            f"max={fracs[valid].max():.2f}"
        )
    print()

    # ══════════════════════════════════════════════════════════════════════
    # C. Convergence analysis
    # ══════════════════════════════════════════════════════════════════════
    print("=" * 80)
    print("C. CONVERGENCE ANALYSIS")
    print("=" * 80)
    print()

    # best_so_far tracks the best observed value cumulatively across rounds.
    # Both BO and P&R results store this in the same format (the harness
    # replays the P&R ranked list in batches).  We normalize per-study to
    # make curves comparable across studies with different value scales.

    # Use matched-surrogate for fair comparison: ngboost BO vs ngboost P&R
    # (ngboost is the top-performing BO family from the within-study benchmark).
    convergence_pairs = [
        ("discrete_ngboost_ucb", "ngboost", "NGBoost BO vs P&R"),
        ("discrete_xgb_ucb", "xgb", "XGBoost BO vs P&R"),
        ("discrete_rf_ucb", "rf", "RF BO vs P&R"),
    ]

    # For the figure, use the aggregate: best-of-3 matched BO vs best-of-3 P&R
    conv_data = {}
    for sid in common_studies:
        d = all_data[sid]
        info = d["study_info"]

        # Collect per-seed best_so_far for best-of-3 matched BO and P&R
        bo_curves, pr_curves, rnd_curves = [], [], []
        for seed in SEEDS:
            # Best-of-3 matched BO: take pointwise max of the 3 strategies
            matched_bo_bsfs = []
            for surr in PR_SURROGATES:
                for strat in SURROGATE_BO_MAP[surr]:
                    if strat in d["bo"] and seed in d["bo"][strat]:
                        matched_bo_bsfs.append(get_best_so_far(d["bo"][strat][seed]))
            if matched_bo_bsfs:
                min_len_bo = min(len(c) for c in matched_bo_bsfs)
                # Pointwise max across strategies
                agg_bo = np.max([c[:min_len_bo] for c in matched_bo_bsfs], axis=0)
                bo_curves.append(agg_bo)

            # Best-of-3 P&R
            matched_pr_bsfs = []
            for surr in PR_SURROGATES:
                if surr in d["pr"] and seed in d["pr"][surr]:
                    matched_pr_bsfs.append(get_best_so_far(d["pr"][surr][seed]))
            if matched_pr_bsfs:
                min_len_pr = min(len(c) for c in matched_pr_bsfs)
                agg_pr = np.max([c[:min_len_pr] for c in matched_pr_bsfs], axis=0)
                pr_curves.append(agg_pr)

            # Random
            if seed in d["random"]:
                rnd_curves.append(get_best_so_far(d["random"][seed]))

        if bo_curves and pr_curves:
            all_curves = bo_curves + pr_curves + rnd_curves
            min_len = min(len(c) for c in all_curves)
            bo_arr = np.array([c[:min_len] for c in bo_curves])
            pr_arr = np.array([c[:min_len] for c in pr_curves])
            rnd_arr = np.array([c[:min_len] for c in rnd_curves]) if rnd_curves else None

            conv_data[sid] = {
                "bo_mean": bo_arr.mean(axis=0).tolist(),
                "pr_mean": pr_arr.mean(axis=0).tolist(),
                "rnd_mean": rnd_arr.mean(axis=0).tolist() if rnd_arr is not None else None,
                "n_rounds": min_len - 1,
            }

    # Normalize each study's curves
    norm_bo_all, norm_pr_all, norm_rnd_all = [], [], []
    max_rounds = 0
    n_flat_excluded = 0
    for _sid, cd in conv_data.items():
        bo = np.array(cd["bo_mean"])
        pr = np.array(cd["pr_mean"])
        rnd = np.array(cd["rnd_mean"]) if cd["rnd_mean"] is not None else None
        # Use random baseline as floor (worst) and overall best as ceiling
        if rnd is not None:
            worst_val = rnd[0]  # random's seed pool value
        else:
            worst_val = min(bo[0], pr[0])
        best_val = max(bo.max(), pr.max())
        if best_val == worst_val:
            # Both methods and random all have the same best_so_far throughout.
            # This means the best was found in the seed pool and never changed.
            # Include as 1.0 (fully converged from round 0).
            n_flat_excluded += 1
            norm_bo_all.append(np.ones_like(bo))
            norm_pr_all.append(np.ones_like(pr))
            if rnd is not None:
                norm_rnd_all.append(np.ones_like(rnd))
            max_rounds = max(max_rounds, len(bo) - 1)
            continue
        norm_bo = (bo - worst_val) / (best_val - worst_val)
        norm_pr = (pr - worst_val) / (best_val - worst_val)
        norm_bo_all.append(norm_bo)
        norm_pr_all.append(norm_pr)
        if rnd is not None:
            norm_rnd = (rnd - worst_val) / (best_val - worst_val)
            norm_rnd_all.append(norm_rnd)
        max_rounds = max(max_rounds, len(bo) - 1)

    def pad_to(arr_list, length):
        padded = []
        for a in arr_list:
            if len(a) < length + 1:
                padded.append(np.pad(a, (0, length + 1 - len(a)), mode="edge"))
            else:
                padded.append(a[: length + 1])
        return np.array(padded)

    if norm_bo_all:
        norm_bo_mat = pad_to(norm_bo_all, max_rounds)
        norm_pr_mat = pad_to(norm_pr_all, max_rounds)

        mean_bo = norm_bo_mat.mean(axis=0)
        mean_pr = norm_pr_mat.mean(axis=0)

        surpass_round = None
        for r in range(len(mean_bo)):
            if mean_bo[r] > mean_pr[r]:
                surpass_round = r
                break

        print(
            f"Normalized convergence (matched best-of-3, {len(norm_bo_all)} studies, "
            f"{n_flat_excluded} with seed-pool convergence):"
        )
        print(f"  Round 0 (seed pool): BO={mean_bo[0]:.3f}, P&R={mean_pr[0]:.3f}")
        for r in [1, 2, 3, 5, 8, 10, 15]:
            if r < len(mean_bo):
                print(
                    f"  Round {r:>2d}: BO={mean_bo[r]:.3f}, P&R={mean_pr[r]:.3f}, diff={mean_bo[r] - mean_pr[r]:+.3f}"
                )
        if surpass_round is not None:
            print(f"  BO first surpasses P&R at round {surpass_round}")
        else:
            print("  BO does not surpass P&R in normalized best_so_far")
        print()

    # Per-surrogate convergence
    for bo_strat, pr_surr, label in convergence_pairs:
        s_bo_all, s_pr_all = [], []
        for sid in common_studies:
            d = all_data[sid]
            bo_c, pr_c = [], []
            for seed in SEEDS:
                if bo_strat in d["bo"] and seed in d["bo"][bo_strat]:
                    bo_c.append(get_best_so_far(d["bo"][bo_strat][seed]))
                if pr_surr in d["pr"] and seed in d["pr"][pr_surr]:
                    pr_c.append(get_best_so_far(d["pr"][pr_surr][seed]))
            if bo_c and pr_c:
                ml = min(min(len(c) for c in bo_c), min(len(c) for c in pr_c))
                bo_mean = np.mean([c[:ml] for c in bo_c], axis=0)
                pr_mean = np.mean([c[:ml] for c in pr_c], axis=0)
                bo_mean.max() - bo_mean.min() + pr_mean.max() - pr_mean.min()
                best = max(bo_mean.max(), pr_mean.max())
                worst = min(bo_mean.min(), pr_mean.min())
                if best > worst:
                    s_bo_all.append((bo_mean - worst) / (best - worst))
                    s_pr_all.append((pr_mean - worst) / (best - worst))
                else:
                    s_bo_all.append(np.ones_like(bo_mean))
                    s_pr_all.append(np.ones_like(pr_mean))
        if s_bo_all:
            ml2 = max(len(c) for c in s_bo_all)
            s_bo_mat = pad_to(s_bo_all, ml2 - 1)
            s_pr_mat = pad_to(s_pr_all, ml2 - 1)
            final_diff = s_bo_mat[:, -1].mean() - s_pr_mat[:, -1].mean()
            sr = None
            for r in range(s_bo_mat.shape[1]):
                if s_bo_mat[:, r].mean() > s_pr_mat[:, r].mean():
                    sr = r
                    break
            print(
                f"  {label} ({len(s_bo_all)} studies): "
                f"final diff={final_diff:+.3f}, "
                f"surpass round={sr if sr is not None else 'never'}"
            )
        print()

    # ══════════════════════════════════════════════════════════════════════
    # D. Pool fraction analysis
    # ══════════════════════════════════════════════════════════════════════
    print("=" * 80)
    print("D. POOL FRACTION ANALYSIS")
    print("=" * 80)
    print()

    print(f"{'Study':<25s} {'N':>5s} {'Seed':>5s} {'Eval':>5s} {'Frac':>6s} {'BO':>6s} {'P&R':>6s} {'Rnd':>6s}")
    print("-" * 85)
    pool_rows = []
    for r in study_rows:
        info = all_data[r["study_id"]]["study_info"]
        n_form = info.get("n_formulations", 0)
        n_seed = info.get("n_seed", 0)
        n_rounds = info.get("n_rounds", 0)
        batch = info.get("batch_size", 0)
        n_eval = n_seed + n_rounds * batch
        frac = n_eval / n_form if n_form > 0 else 0
        pool_rows.append(
            {
                "study_id": r["study_id"],
                "n_form": n_form,
                "n_seed": n_seed,
                "n_eval": n_eval,
                "frac": frac,
                "bo": r["bo_mean"],
                "pr": r["pr_mean"],
                "rnd": r["random_mean"],
                "diff": r["diff"],
            }
        )
    for pr in sorted(pool_rows, key=lambda x: x["frac"]):
        print(
            f"{pr['study_id']:<25s} {pr['n_form']:>5} {pr['n_seed']:>5} "
            f"{pr['n_eval']:>5} {pr['frac']:>6.2f} "
            f"{pr['bo']:>6.3f} {pr['pr']:>6.3f} {pr['rnd']:>6.3f}"
        )
    print()

    # Fraction threshold analysis
    print("BO advantage by pool fraction quartile:")
    frac_arr = np.array([pr["frac"] for pr in pool_rows])
    diff_arr = np.array([pr["diff"] for pr in pool_rows])
    quartiles = np.percentile(frac_arr, [25, 50, 75])
    bins = [0, *list(quartiles), 1.1]
    labels = ["Q1 (lowest frac)", "Q2", "Q3", "Q4 (highest frac)"]
    for i in range(4):
        mask = (frac_arr >= bins[i]) & (frac_arr < bins[i + 1])
        if mask.sum() > 0:
            print(
                f"  {labels[i]}: N={mask.sum()}, frac range=[{frac_arr[mask].min():.2f}, "
                f"{frac_arr[mask].max():.2f}], mean BO-P&R={diff_arr[mask].mean():+.4f}"
            )
    print()

    # ══════════════════════════════════════════════════════════════════════
    # E. Statistical tests
    # ══════════════════════════════════════════════════════════════════════
    print("=" * 80)
    print("E. STATISTICAL TESTS")
    print("=" * 80)
    print()

    # E1. Overall: best BO vs best P&R (paired by study, averaged over seeds)
    bo_means = np.array([r["bo_mean"] for r in study_rows])
    pr_means = np.array([r["pr_mean"] for r in study_rows])
    diff_overall = bo_means - pr_means

    print("E1. Best BO vs Best P&R (paired by study):")
    print(f"  N studies: {len(diff_overall)}")
    print(f"  Mean BO: {bo_means.mean():.4f} +/- {bo_means.std():.4f}")
    print(f"  Mean P&R: {pr_means.mean():.4f} +/- {pr_means.std():.4f}")
    print(f"  Mean diff: {diff_overall.mean():+.4f} +/- {diff_overall.std():.4f}")

    # Paired Wilcoxon signed-rank test
    w_stat, w_p = stats.wilcoxon(bo_means, pr_means, alternative="two-sided")
    print(f"  Wilcoxon signed-rank: W={w_stat:.1f}, p={w_p:.6f}")

    # Paired t-test
    t_stat, t_p = stats.ttest_rel(bo_means, pr_means)
    print(f"  Paired t-test: t={t_stat:.3f}, p={t_p:.6f}")

    # Effect size (Cohen's d for paired data)
    cohens_d = diff_overall.mean() / diff_overall.std()
    print(f"  Cohen's d: {cohens_d:.3f}")

    # Rank-biserial correlation (effect size for Wilcoxon)
    n = len(diff_overall)
    r_rb = 1 - (2 * w_stat) / (n * (n + 1) / 2)
    print(f"  Rank-biserial r: {r_rb:.3f}")
    print()

    # E2. Same-surrogate paired tests
    print("E2. Same-surrogate paired tests:")
    test_results = []
    for surr in PR_SURROGATES:
        bo_strats = SURROGATE_BO_MAP[surr]
        bo_v, pr_v = [], []
        for sid in common_studies:
            d = all_data[sid]
            bo_seed = []
            for seed in SEEDS:
                best = -1
                for strat in bo_strats:
                    if strat in d["bo"] and seed in d["bo"][strat]:
                        r = get_recall5(d["bo"][strat][seed])
                        if r > best:
                            best = r
                if best >= 0:
                    bo_seed.append(best)
            pr_seed = []
            if surr in d["pr"]:
                for seed in SEEDS:
                    if seed in d["pr"][surr]:
                        pr_seed.append(get_recall5(d["pr"][surr][seed]))
            if bo_seed and pr_seed:
                bo_v.append(np.mean(bo_seed))
                pr_v.append(np.mean(pr_seed))
        bo_a = np.array(bo_v)
        pr_a = np.array(pr_v)
        diff_s = bo_a - pr_a
        w, p = stats.wilcoxon(bo_a, pr_a, alternative="two-sided")
        cd = diff_s.mean() / diff_s.std() if diff_s.std() > 0 else 0
        n_s = len(diff_s)
        rb = 1 - (2 * w) / (n_s * (n_s + 1) / 2)
        test_results.append(
            {
                "surrogate": surr,
                "n": n_s,
                "bo_mean": bo_a.mean(),
                "pr_mean": pr_a.mean(),
                "diff_mean": diff_s.mean(),
                "wilcoxon_W": w,
                "wilcoxon_p": p,
                "cohens_d": cd,
                "rank_biserial": rb,
            }
        )
        print(f"  {surr}: N={n_s}, diff={diff_s.mean():+.4f}, W={w:.1f}, p={p:.6f}, Cohen's d={cd:.3f}, rb={rb:.3f}")

    # Bonferroni correction
    n_tests = len(test_results) + 1  # +1 for overall test
    alpha = 0.05
    bonf_alpha = alpha / n_tests
    print(f"\n  Bonferroni-corrected alpha: {bonf_alpha:.4f} ({n_tests} tests)")
    print(f"  Overall test: p={w_p:.6f} -> {'significant' if w_p < bonf_alpha else 'not significant'}")
    for tr in test_results:
        sig = "significant" if tr["wilcoxon_p"] < bonf_alpha else "not significant"
        print(f"  {tr['surrogate']}: p={tr['wilcoxon_p']:.6f} -> {sig}")
    print()

    # E3. BO vs Random
    print("E3. BO vs Random:")
    rnd_means = np.array([r["random_mean"] for r in study_rows])
    diff_rnd = bo_means - rnd_means
    w_rnd, p_rnd = stats.wilcoxon(bo_means, rnd_means, alternative="greater")
    cd_rnd = diff_rnd.mean() / diff_rnd.std() if diff_rnd.std() > 0 else 0
    print(f"  Mean BO: {bo_means.mean():.4f}, Mean Random: {rnd_means.mean():.4f}")
    print(f"  Mean diff: {diff_rnd.mean():+.4f}")
    print(f"  Wilcoxon (one-sided): W={w_rnd:.1f}, p={p_rnd:.6f}")
    print(f"  Cohen's d: {cd_rnd:.3f}")
    print()

    # E4. P&R vs Random
    print("E4. P&R vs Random:")
    diff_pr_rnd = pr_means - rnd_means
    w_pr_rnd, p_pr_rnd = stats.wilcoxon(pr_means, rnd_means, alternative="greater")
    cd_pr_rnd = diff_pr_rnd.mean() / diff_pr_rnd.std() if diff_pr_rnd.std() > 0 else 0
    print(f"  Mean P&R: {pr_means.mean():.4f}, Mean Random: {rnd_means.mean():.4f}")
    print(f"  Mean diff: {diff_pr_rnd.mean():+.4f}")
    print(f"  Wilcoxon (one-sided): W={w_pr_rnd:.1f}, p={p_pr_rnd:.6f}")
    print(f"  Cohen's d: {cd_pr_rnd:.3f}")
    print()

    # E5. Seed-level paired test (more statistical power)
    print("E5. Seed-level paired test (all study x seed pairs):")
    bo_seed_all, pr_seed_all = [], []
    for r in study_rows:
        for seed in SEEDS:
            if seed in r["bo_per_seed"] and seed in r["pr_per_seed"]:
                bo_seed_all.append(r["bo_per_seed"][seed])
                pr_seed_all.append(r["pr_per_seed"][seed])
    bo_seed_arr = np.array(bo_seed_all)
    pr_seed_arr = np.array(pr_seed_all)
    diff_seed = bo_seed_arr - pr_seed_arr
    w_seed, p_seed = stats.wilcoxon(bo_seed_arr, pr_seed_arr, alternative="two-sided")
    cd_seed = diff_seed.mean() / diff_seed.std() if diff_seed.std() > 0 else 0
    print(f"  N pairs: {len(diff_seed)}")
    print(f"  Mean diff: {diff_seed.mean():+.4f}")
    print(f"  Wilcoxon: W={w_seed:.1f}, p={p_seed:.6f}")
    print(f"  Cohen's d: {cd_seed:.3f}")
    print()

    # ══════════════════════════════════════════════════════════════════════
    # Summary statistics
    # ══════════════════════════════════════════════════════════════════════
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()

    bo_lift = bo_means.mean() / rnd_means.mean()
    pr_lift = pr_means.mean() / rnd_means.mean()
    print(f"Mean top-5% recall: BO={bo_means.mean():.4f}, P&R={pr_means.mean():.4f}, Random={rnd_means.mean():.4f}")
    print(f"Lift over random: BO={bo_lift:.2f}x, P&R={pr_lift:.2f}x")
    print(
        f"BO advantage over P&R: {diff_overall.mean():+.4f} "
        f"({diff_overall.mean() / pr_means.mean() * 100:+.1f}% relative)"
    )
    print(f"BO wins {(diff_overall > 0).sum()}/{len(diff_overall)} studies")
    ci_lo, ci_hi = bootstrap_ci(diff_overall)
    print(f"95% bootstrap CI for mean diff: [{ci_lo:+.4f}, {ci_hi:+.4f}]")
    print()

    # ══════════════════════════════════════════════════════════════════════
    # Save JSON
    # ══════════════════════════════════════════════════════════════════════
    output = {
        "description": "BO vs Predict-and-Rank deep statistical analysis",
        "n_studies": len(common_studies),
        "studies": common_studies,
        "seeds": SEEDS,
        "summary": {
            "bo_mean_recall": float(bo_means.mean()),
            "pr_mean_recall": float(pr_means.mean()),
            "random_mean_recall": float(rnd_means.mean()),
            "bo_lift_over_random": float(bo_lift),
            "pr_lift_over_random": float(pr_lift),
            "mean_bo_minus_pr": float(diff_overall.mean()),
            "mean_bo_minus_pr_relative_pct": float(diff_overall.mean() / pr_means.mean() * 100),
            "bo_wins_n_studies": int((diff_overall > 0).sum()),
            "bootstrap_ci_95": [float(ci_lo), float(ci_hi)],
        },
        "overall_test": {
            "wilcoxon_W": float(w_stat),
            "wilcoxon_p": float(w_p),
            "paired_t_stat": float(t_stat),
            "paired_t_p": float(t_p),
            "cohens_d": float(cohens_d),
            "rank_biserial_r": float(r_rb),
        },
        "same_surrogate_tests": [
            {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in tr.items()} for tr in test_results
        ],
        "bonferroni_alpha": float(bonf_alpha),
        "matched_surrogate_aggregate": {
            "n": len(matched_diff),
            "bo_mean": float(matched_bo_arr.mean()),
            "pr_mean": float(matched_pr_arr.mean()),
            "diff_mean": float(matched_diff.mean()),
            "diff_std": float(matched_diff.std()),
            "bo_wins": int((matched_diff > 0).sum()),
        },
        "bo_vs_random": {
            "wilcoxon_W": float(w_rnd),
            "wilcoxon_p": float(p_rnd),
            "cohens_d": float(cd_rnd),
        },
        "pr_vs_random": {
            "wilcoxon_W": float(w_pr_rnd),
            "wilcoxon_p": float(p_pr_rnd),
            "cohens_d": float(cd_pr_rnd),
        },
        "seed_level_test": {
            "n_pairs": len(diff_seed),
            "wilcoxon_W": float(w_seed),
            "wilcoxon_p": float(p_seed),
            "cohens_d": float(cd_seed),
        },
        "correlations": {
            "bo_advantage_vs_study_size": {
                "spearman_r": float(r_size) if "r_size" in dir() else None,
                "p_value": float(p_size) if "p_size" in dir() else None,
            },
            "bo_advantage_vs_n_unique_il": {
                "spearman_r": float(r_il) if "r_il" in dir() else None,
                "p_value": float(p_il) if "p_il" in dir() else None,
            },
            "bo_advantage_vs_pool_fraction": {
                "spearman_r": float(r_frac) if "r_frac" in dir() else None,
                "p_value": float(p_frac) if "p_frac" in dir() else None,
            },
        },
        "by_study_type": {
            stype: {
                "n_studies": len(vals),
                "mean_diff": float(np.mean(vals)),
                "median_diff": float(np.median(vals)),
                "bo_wins": int(sum(1 for v in vals if v > 0)),
            }
            for stype, vals in type_groups.items()
        },
        "per_study": [
            {
                "study_id": r["study_id"],
                "n_formulations": r["n_formulations"],
                "study_type": r["study_type"],
                "n_unique_il": r["n_unique_il"],
                "bo_mean_recall": float(r["bo_mean"]),
                "pr_mean_recall": float(r["pr_mean"]),
                "random_mean_recall": float(r["random_mean"]),
                "bo_minus_pr": float(r["diff"]),
                "pool_fraction": float(fracs[i]) if not np.isnan(fracs[i]) else None,
            }
            for i, r in enumerate(study_rows)
        ],
        "convergence": {sid: cd for sid, cd in conv_data.items()},
    }

    os.makedirs(OUT_JSON.parent, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved analysis JSON to {OUT_JSON}")

    # ══════════════════════════════════════════════════════════════════════
    # Generate figure
    # ══════════════════════════════════════════════════════════════════════
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL, DOUBLE_COL * 0.75))
    fig.subplots_adjust(hspace=0.4, wspace=0.35)

    type_color_map = {
        "il_diverse_fixed_ratios": STUDY_TYPE_COLORS.get("IL-diverse", "#4477AA"),
        "il_diverse_variable_ratios": STUDY_TYPE_COLORS.get("IL-diverse", "#4477AA"),
        "ratio_only": STUDY_TYPE_COLORS.get("ratio-only", "#EE6677"),
    }

    def _scatter_by_type(ax, x_vals, y_vals, study_rows_local):
        for x, y, r in zip(x_vals, y_vals, study_rows_local):
            stype = r["study_type"]
            color = type_color_map.get(stype, "#999999")
            marker = "o" if "fixed" in stype else ("s" if "variable" in stype else "D")
            ax.scatter(x, y, c=color, marker=marker, s=25, edgecolors="white", linewidths=0.3, zorder=3)

    # ── Panel (a): Study-level BO vs P&R scatter ──────────────────────
    ax = axes[0, 0]
    panel_label(ax, "a")
    _scatter_by_type(ax, [r["pr_mean"] for r in study_rows], [r["bo_mean"] for r in study_rows], study_rows)
    lims = [
        min(ax.get_xlim()[0], ax.get_ylim()[0]),
        max(ax.get_xlim()[1], ax.get_ylim()[1]),
    ]
    ax.plot(lims, lims, "--", color="#888888", linewidth=0.5, zorder=1)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Best P&R top-5% recall")
    ax.set_ylabel("Best BO top-5% recall")
    ax.set_title("BO vs P&R per study (oracle-best)")
    light_ygrid(ax)
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=STUDY_TYPE_COLORS.get("IL-diverse", "#4477AA"),
            markersize=4,
            label="fixed ratio",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor=STUDY_TYPE_COLORS.get("IL-diverse", "#4477AA"),
            markersize=4,
            label="variable ratio",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            color="w",
            markerfacecolor=STUDY_TYPE_COLORS.get("ratio-only", "#EE6677"),
            markersize=4,
            label="ratio-only",
        ),
    ]
    ax.legend(handles=legend_elements, loc="lower right", frameon=False)

    # ── Panel (b): BO advantage vs study size ─────────────────────────
    ax = axes[0, 1]
    panel_label(ax, "b")
    _scatter_by_type(ax, [r["n_formulations"] for r in study_rows], [r["diff"] for r in study_rows], study_rows)
    ax.axhline(0, color="#888888", linewidth=0.5, linestyle="--", zorder=1)
    ax.set_xlabel("Study size (n formulations)")
    ax.set_ylabel("BO$-$P&R (top-5% recall)")
    ax.set_title("BO advantage vs study size")
    light_ygrid(ax)
    if "r_size" in dir():
        ax.text(0.05, 0.95, f"$\\rho$={r_size:.2f}, p={p_size:.3f}", transform=ax.transAxes, fontsize=6, va="top")

    # ── Panel (c): Convergence curves (matched best-of-3) ────────────
    ax = axes[1, 0]
    panel_label(ax, "c")
    if norm_bo_all:
        rounds = np.arange(max_rounds + 1)
        mean_bo_curve = norm_bo_mat.mean(axis=0)
        mean_pr_curve = norm_pr_mat.mean(axis=0)
        se_bo = norm_bo_mat.std(axis=0) / np.sqrt(norm_bo_mat.shape[0])
        se_pr = norm_pr_mat.std(axis=0) / np.sqrt(norm_pr_mat.shape[0])

        c_bo = FAMILY_COLORS.get("NGBoost", "#EE6677")
        c_pr = "#CCBB44"
        ax.plot(rounds, mean_bo_curve, color=c_bo, label="BO (best of 3)")
        ax.fill_between(rounds, mean_bo_curve - se_bo, mean_bo_curve + se_bo, color=c_bo, alpha=0.15)
        ax.plot(rounds, mean_pr_curve, color=c_pr, label="P&R (best of 3)")
        ax.fill_between(rounds, mean_pr_curve - se_pr, mean_pr_curve + se_pr, color=c_pr, alpha=0.15)

        if norm_rnd_all:
            norm_rnd_mat = pad_to(norm_rnd_all, max_rounds)
            mean_rnd_curve = norm_rnd_mat.mean(axis=0)
            ax.plot(rounds, mean_rnd_curve, color="#000000", linestyle=":", label="Random", linewidth=0.8)

        # Mark surpass round
        if surpass_round is not None and surpass_round < len(rounds):
            ax.axvline(surpass_round, color="#888888", linewidth=0.4, linestyle="--", zorder=1)
            ax.text(surpass_round + 0.3, 0.15, f"round {surpass_round}", fontsize=5.5, color="#888888")

    ax.set_xlabel("Round")
    ax.set_ylabel("Normalized best-so-far")
    ax.set_title(f"Convergence (N={len(norm_bo_all)} studies)")
    ax.legend(loc="lower right", frameon=False, fontsize=5.5)
    light_ygrid(ax)

    # ── Panel (d): BO advantage vs pool fraction ──────────────────────
    ax = axes[1, 1]
    panel_label(ax, "d")
    _scatter_by_type(ax, fracs.tolist(), [r["diff"] for r in study_rows], study_rows)
    ax.axhline(0, color="#888888", linewidth=0.5, linestyle="--", zorder=1)
    ax.set_xlabel("Fraction of pool evaluated")
    ax.set_ylabel("BO$-$P&R (top-5% recall)")
    ax.set_title("BO advantage vs sampling fraction")
    light_ygrid(ax)
    if "r_frac" in dir():
        ax.text(0.05, 0.95, f"$\\rho$={r_frac:.2f}, p={p_frac:.3f}", transform=ax.transAxes, fontsize=6, va="top")

    os.makedirs(OUT_FIG.parent, exist_ok=True)
    save_figure(fig, OUT_FIG)
    plt.close(fig)
    print(f"Saved figure to {OUT_FIG}")


if __name__ == "__main__":
    main()
