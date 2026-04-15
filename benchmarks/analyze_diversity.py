#!/usr/bin/env python3
"""
Formulation-level diversity analysis: BO vs Predict-and-Rank.

Answers: "Does iterative BO explore different chemical space than
predict-and-rank, or do they find the same hits?"

Approach:
    Individual formulation indices are not stored in the result JSONs.
    Instead we use final_best values as a proxy for hit identity: if two
    strategies converge to the same final_best (within tolerance), they
    likely found the same top formulation.

Analyses:
    1. BO vs P&R hit agreement (same surrogate, same seed, same study)
    2. BO vs P&R recall comparison (top-5% recall)
    3. Inter-family hit diversity within BO (how many distinct final_best
       values do different BO families find?)
    4. Per-study diversity summary

Outputs:
    - Console summary tables
    - benchmark_results/diversity_analysis.json
    - paper/figures/fig_diversity.pdf
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as sp_stats

from LNPBO.runtime_paths import benchmark_results_root, package_root_from, paper_root

from .stats import bootstrap_ci

_PACKAGE_ROOT = package_root_from(__file__, levels_up=2)
_RESULTS_ROOT = benchmark_results_root(_PACKAGE_ROOT)
BO_DIR = _RESULTS_ROOT / "within_study"
PR_DIR = _RESULTS_ROOT / "baselines" / "predict_and_rank"
FIG_DIR = paper_root(_PACKAGE_ROOT) / "figures"
OUT_JSON = _RESULTS_ROOT / "diversity_analysis.json"

from .constants import SEEDS

TOL = 1e-6

# BO strategy -> surrogate name used in P&R
BO_STRATEGY_TO_SURROGATE = {
    "discrete_xgb_ucb": "xgb",
    "discrete_xgb_greedy": "xgb",
    "discrete_xgb_cqr": "xgb",
    "discrete_xgb_online_conformal": "xgb",
    "discrete_xgb_ucb_ts_batch": "xgb",
    "discrete_rf_ucb": "rf",
    "discrete_rf_ts": "rf",
    "discrete_rf_ts_batch": "rf",
    "discrete_ngboost_ucb": "ngboost",
}

SURROGATE_DISPLAY = {
    "xgb": "XGBoost",
    "rf": "RF",
    "ngboost": "NGBoost",
}

from .strategy_registry import STRATEGY_FAMILY

FAMILIES_TO_SHOW = [
    "NGBoost",
    "RF",
    "CASMOPolitan",
    "XGBoost",
    "Deep Ensemble",
    "GP (sklearn)",
    "LNPBO",
    "Random",
]

FAMILY_COLORS = {
    "NGBoost": "#EE6677",
    "RF": "#228833",
    "CASMOPolitan": "#4477AA",
    "XGBoost": "#CCBB44",
    "Deep Ensemble": "#66CCEE",
    "GP (sklearn)": "#AA3377",
    "LNPBO": "#BBBBBB",
    "Random": "#000000",
    "P&R": "#FF8C00",
}


def section(title):
    print(f"\n{title}\n")


def subsection(title):
    print(f"\n{title}")


# ── Data Loading ──────────────────────────────────────────────────────────


def _get_study_id_from_dir(d):
    """Extract study_id from a directory name."""
    return d.name


def load_bo_results():
    """Load all within-study BO results, keyed by (study_id, strategy, seed)."""
    from .result_loading import load_benchmark_results

    records = load_benchmark_results(BO_DIR)
    results = {}
    study_info = {}
    for data in records:
        strategy = data.get("strategy")
        seed = data.get("seed")
        if strategy is None or seed is None:
            continue
        si = data.get("study_info", {})
        sid = si.get("study_id", str(data.get("pmid", "")))
        results[(sid, strategy, seed)] = data
        if sid not in study_info:
            study_info[sid] = si
    return results, study_info


def load_pr_results():
    """Load all predict-and-rank results, keyed by (study_id, surrogate, seed)."""
    results = {}
    for study_dir in sorted(PR_DIR.iterdir()):
        if not study_dir.is_dir():
            continue
        study_id = _get_study_id_from_dir(study_dir)
        for f in sorted(study_dir.glob("predict_rank_*.json")):
            try:
                data = json.loads(f.read_text())
            except (json.JSONDecodeError, KeyError):
                continue
            surrogate = data.get("surrogate")
            seed = data.get("seed")
            if surrogate is None or seed is None:
                continue
            results[(study_id, surrogate, seed)] = data
            # Also key by the study_id stored in the JSON
            json_sid = data.get("study_id", study_id)
            if json_sid != study_id:
                results[(json_sid, surrogate, seed)] = data
    return results


def _find_matching_study_ids(bo_results, pr_results):
    """Find study_ids that exist in both BO and P&R results with matching n_formulations."""
    bo_sids = set()
    for sid, _, _ in bo_results:
        bo_sids.add(sid)
    pr_sids = set()
    for sid, _, _ in pr_results:
        pr_sids.add(sid)

    # Direct matches
    common = bo_sids & pr_sids

    # For studies that don't match directly, try mapping
    # BO "37661193" (325 formulations) != P&R "37661193_spleen" (162)
    # Skip these mismatched ones
    matched = set()
    for sid in common:
        # Verify n_formulations match
        bo_key = None
        pr_key = None
        for k in bo_results:
            if k[0] == sid:
                bo_key = k
                break
        for k in pr_results:
            if k[0] == sid:
                pr_key = k
                break
        if bo_key and pr_key:
            bo_n = bo_results[bo_key].get("study_info", {}).get("n_formulations", -1)
            pr_n = pr_results[pr_key].get("study_info", {}).get("n_formulations", -2)
            if bo_n == pr_n:
                matched.add(sid)
            else:
                # Mismatch -- skip this study_id for BO-vs-P&R comparison
                pass

    return sorted(matched)


def _get_recall(data, pct="5"):
    """Extract top-k% recall from a result dict."""
    tkr = data.get("result", {}).get("metrics", {}).get("top_k_recall", {})
    return tkr.get(pct)


def _get_final_best(data):
    """Extract final_best from a result dict."""
    return data.get("result", {}).get("metrics", {}).get("final_best")


# ── Analysis 1: BO vs P&R Hit Agreement ──────────────────────────────────


def analyze_bo_vs_pr(bo_results, pr_results, matched_sids):
    section("=" * 80)
    section("  FORMULATION-LEVEL DIVERSITY ANALYSIS: BO vs PREDICT-AND-RANK")
    section("=" * 80)

    subsection("A1. BO vs P&R Hit Agreement (same surrogate, same seed, same study)")
    print("  If final_best values match within tolerance, BO and P&R found the same top hit.")
    print()

    # For each surrogate, collect agreement stats
    surrogate_stats = {}
    all_comparisons = []

    for surrogate, surr_display in SURROGATE_DISPLAY.items():
        bo_strategies = [s for s, sur in BO_STRATEGY_TO_SURROGATE.items() if sur == surrogate]
        agreements = []
        recall_bo_vals = []
        recall_pr_vals = []
        details = []

        for sid in matched_sids:
            for seed in SEEDS:
                pr_key = (sid, surrogate, seed)
                if pr_key not in pr_results:
                    continue
                pr_data = pr_results[pr_key]
                pr_fb = _get_final_best(pr_data)
                pr_recall = _get_recall(pr_data)
                if pr_fb is None:
                    continue

                for bo_strat in bo_strategies:
                    bo_key = (sid, bo_strat, seed)
                    if bo_key not in bo_results:
                        continue
                    bo_data = bo_results[bo_key]
                    bo_fb = _get_final_best(bo_data)
                    bo_recall = _get_recall(bo_data)
                    if bo_fb is None:
                        continue

                    same_hit = abs(bo_fb - pr_fb) < TOL
                    agreements.append(same_hit)
                    if bo_recall is not None and pr_recall is not None:
                        recall_bo_vals.append(bo_recall)
                        recall_pr_vals.append(pr_recall)
                    details.append(
                        {
                            "study_id": sid,
                            "seed": seed,
                            "bo_strategy": bo_strat,
                            "surrogate": surrogate,
                            "bo_final_best": bo_fb,
                            "pr_final_best": pr_fb,
                            "same_hit": same_hit,
                            "bo_recall": bo_recall,
                            "pr_recall": pr_recall,
                        }
                    )
                    all_comparisons.append(details[-1])

        if not agreements:
            continue

        agree_rate = np.mean(agreements)
        ci = bootstrap_ci([float(a) for a in agreements])
        n_comp = len(agreements)

        recall_bo_arr = np.array(recall_bo_vals)
        recall_pr_arr = np.array(recall_pr_vals)

        surrogate_stats[surrogate] = {
            "display": surr_display,
            "n_comparisons": n_comp,
            "hit_agreement_rate": float(agree_rate),
            "hit_agreement_ci": [float(ci[0]), float(ci[1])],
            "mean_bo_recall": float(np.mean(recall_bo_arr)) if len(recall_bo_arr) > 0 else None,
            "mean_pr_recall": float(np.mean(recall_pr_arr)) if len(recall_pr_arr) > 0 else None,
        }

    # Print summary table
    print(
        f"  {'Surrogate':<12}  {'N':>5}  {'Hit Agree':>10}  {'95% CI':>16}  "
        f"{'BO Recall':>10}  {'P&R Recall':>10}  {'Delta':>8}"
    )
    print("  " + "-" * 82)

    for surrogate in ["xgb", "rf", "ngboost"]:
        if surrogate not in surrogate_stats:
            continue
        s = surrogate_stats[surrogate]
        delta = ""
        if s["mean_bo_recall"] is not None and s["mean_pr_recall"] is not None:
            d = s["mean_bo_recall"] - s["mean_pr_recall"]
            delta = f"{d:>+.3f}"
        print(
            f"  {s['display']:<12}  {s['n_comparisons']:>5}  {s['hit_agreement_rate']:>10.1%}  "
            f"[{s['hit_agreement_ci'][0]:.1%}, {s['hit_agreement_ci'][1]:.1%}]  "
            f"{s['mean_bo_recall']:>10.3f}  {s['mean_pr_recall']:>10.3f}  {delta:>8}"
        )

    # Overall
    if all_comparisons:
        all_agree = [c["same_hit"] for c in all_comparisons]
        overall_rate = np.mean(all_agree)
        overall_ci = bootstrap_ci([float(a) for a in all_agree])
        print()
        print(
            f"  Overall hit agreement: {overall_rate:.1%} "
            f"[{overall_ci[0]:.1%}, {overall_ci[1]:.1%}] "
            f"(N={len(all_comparisons)})"
        )
        if overall_rate > 0.7:
            print("  Interpretation: BO and P&R converge to the SAME top hits most of the time.")
            print("  The iterative feedback loop does not substantially change WHICH formulation wins.")
        elif overall_rate > 0.4:
            print("  Interpretation: Moderate agreement. BO sometimes finds different top hits.")
        else:
            print("  Interpretation: Low agreement. BO explores different chemical space than P&R.")

    return surrogate_stats, all_comparisons


# ── Analysis 2: BO vs P&R Recall Comparison ──────────────────────────────


def analyze_recall_comparison(bo_results, pr_results, matched_sids):
    subsection("A2. Recall Comparison: BO vs P&R (top-5% recall, same seed + study)")
    print("  Does iterative BO achieve higher recall than one-shot P&R?")
    print()

    recall_pairs = defaultdict(list)  # surrogate -> list of (bo_recall, pr_recall)

    for surrogate in SURROGATE_DISPLAY:
        bo_strategies = [s for s, sur in BO_STRATEGY_TO_SURROGATE.items() if sur == surrogate]
        for sid in matched_sids:
            for seed in SEEDS:
                pr_key = (sid, surrogate, seed)
                if pr_key not in pr_results:
                    continue
                pr_recall = _get_recall(pr_results[pr_key])
                if pr_recall is None:
                    continue

                # Use the best BO strategy for this surrogate family
                best_bo_recall = None
                for bo_strat in bo_strategies:
                    bo_key = (sid, bo_strat, seed)
                    if bo_key not in bo_results:
                        continue
                    r = _get_recall(bo_results[bo_key])
                    if r is not None and (best_bo_recall is None or r > best_bo_recall):
                        best_bo_recall = r
                if best_bo_recall is not None:
                    recall_pairs[surrogate].append((best_bo_recall, pr_recall))

    recall_stats = {}
    print(
        f"  {'Surrogate':<12}  {'N':>5}  {'BO>P&R':>8}  {'BO=P&R':>8}  {'BO<P&R':>8}  "
        f"{'Mean diff':>10}  {'Wilcoxon p':>10}"
    )
    print("  " + "-" * 72)

    for surrogate in ["xgb", "rf", "ngboost"]:
        if surrogate not in recall_pairs or not recall_pairs[surrogate]:
            continue
        pairs = recall_pairs[surrogate]
        bo_arr = np.array([p[0] for p in pairs])
        pr_arr = np.array([p[1] for p in pairs])
        diffs = bo_arr - pr_arr

        n_bo_better = int(np.sum(diffs > TOL))
        n_equal = int(np.sum(np.abs(diffs) <= TOL))
        n_pr_better = int(np.sum(diffs < -TOL))
        mean_diff = float(np.mean(diffs))

        try:
            _, p_val = sp_stats.wilcoxon(bo_arr, pr_arr, alternative="two-sided")
        except ValueError:
            p_val = 1.0

        recall_stats[surrogate] = {
            "n_pairs": len(pairs),
            "n_bo_better": n_bo_better,
            "n_equal": n_equal,
            "n_pr_better": n_pr_better,
            "mean_diff": mean_diff,
            "p_value": float(p_val),
        }

        p_str = f"{p_val:.4f}" if p_val >= 0.0001 else f"{p_val:.2e}"
        print(
            f"  {SURROGATE_DISPLAY[surrogate]:<12}  {len(pairs):>5}  "
            f"{n_bo_better:>8}  {n_equal:>8}  {n_pr_better:>8}  "
            f"{mean_diff:>+10.4f}  {p_str:>10}"
        )

    print()
    total_bo_better = sum(s.get("n_bo_better", 0) for s in recall_stats.values())
    total_equal = sum(s.get("n_equal", 0) for s in recall_stats.values())
    total_pr_better = sum(s.get("n_pr_better", 0) for s in recall_stats.values())
    total_n = total_bo_better + total_equal + total_pr_better
    if total_n > 0:
        print(
            f"  Overall: BO better {total_bo_better}/{total_n} ({total_bo_better / total_n:.0%}), "
            f"equal {total_equal}/{total_n} ({total_equal / total_n:.0%}), "
            f"P&R better {total_pr_better}/{total_n} ({total_pr_better / total_n:.0%})"
        )

    return recall_stats, recall_pairs


# ── Analysis 3: Inter-Family Hit Diversity ───────────────────────────────


def analyze_interfamily_diversity(bo_results, bo_study_info):
    subsection("A3. Inter-Family Hit Diversity (across BO strategy families)")
    print("  For each study+seed, how many DISTINCT top formulations do different families find?")
    print("  Distinct = final_best values that differ by more than 1e-6.")
    print()

    # Get all BO study_ids
    all_sids = sorted(set(sid for sid, _, _ in bo_results))
    all_strategies = sorted(set(strat for _, strat, _ in bo_results))
    families_present = [f for f in FAMILIES_TO_SHOW if f != "Random"]

    # Per study: average number of distinct final_best values across seeds
    study_diversity = {}
    pairwise_agreement = defaultdict(int)
    pairwise_total = defaultdict(int)

    for sid in all_sids:
        seed_diversities = []
        for seed in SEEDS:
            fam_bests = {}
            for fam in families_present:
                fb_vals = []
                for strat in all_strategies:
                    if STRATEGY_FAMILY.get(strat, "Other") != fam:
                        continue
                    key = (sid, strat, seed)
                    if key not in bo_results:
                        continue
                    fb = _get_final_best(bo_results[key])
                    if fb is not None:
                        fb_vals.append(fb)
                if fb_vals:
                    fam_bests[fam] = max(fb_vals)

            if len(fam_bests) < 2:
                continue

            # Count distinct final_best values
            values = sorted(fam_bests.values(), reverse=True)
            unique_vals = [values[0]]
            for v in values[1:]:
                if all(abs(v - u) > TOL for u in unique_vals):
                    unique_vals.append(v)
            seed_diversities.append(len(unique_vals))

            # Pairwise agreement
            fam_list = sorted(fam_bests.keys())
            for i, fa in enumerate(fam_list):
                for j, fb_name in enumerate(fam_list):
                    if i >= j:
                        continue
                    pairwise_total[(fa, fb_name)] += 1
                    if abs(fam_bests[fa] - fam_bests[fb_name]) < TOL:
                        pairwise_agreement[(fa, fb_name)] += 1

        if seed_diversities:
            si = bo_study_info.get(sid, {})
            study_diversity[sid] = {
                "mean_distinct_hits": float(np.mean(seed_diversities)),
                "n_seeds": len(seed_diversities),
                "n_formulations": si.get("n_formulations", 0),
                "study_type": si.get("study_type", "unknown"),
            }

    # Print per-study table
    type_abbrev = {
        "il_diverse_fixed_ratios": "fix",
        "il_diverse_variable_ratios": "var",
        "ratio_only": "ratio",
    }

    print(f"  {'Study ID':>22}  {'N':>5}  {'Type':>6}  {'Mean Distinct':>14}  {'Diversity Ratio':>15}")
    print("  " + "-" * 72)

    div_ratios = []
    for sid in sorted(study_diversity, key=lambda s: study_diversity[s]["n_formulations"]):
        sd = study_diversity[sid]
        n_fams = len(
            [
                f
                for f in families_present
                if any(
                    (sid, s, seed) in bo_results
                    for s in all_strategies
                    if STRATEGY_FAMILY.get(s, "Other") == f
                    for seed in SEEDS
                )
            ]
        )
        if n_fams == 0:
            continue
        div_ratio = sd["mean_distinct_hits"] / n_fams
        div_ratios.append(div_ratio)
        tabbr = type_abbrev.get(sd["study_type"], "?")
        print(
            f"  {sid:>22}  {sd['n_formulations']:>5}  {tabbr:>6}  {sd['mean_distinct_hits']:>14.1f}  {div_ratio:>15.2f}"
        )

    if div_ratios:
        print()
        mean_dr = np.mean(div_ratios)
        ci = bootstrap_ci(div_ratios)
        print(f"  Mean diversity ratio: {mean_dr:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]")
        if mean_dr < 0.5:
            print("  Low diversity: families mostly converge to the same top formulations.")
        elif mean_dr < 0.75:
            print("  Moderate diversity: some families find different top hits.")
        else:
            print("  High diversity: families explore substantially different regions.")

    # Pairwise agreement matrix
    subsection("Pairwise Hit Agreement between BO Families")
    fams_with_data = [f for f in families_present if any(f in k for k in pairwise_total)]
    if fams_with_data:
        short = {f: f[:10] for f in fams_with_data}
        header = f"  {'':>12}"
        for f in fams_with_data:
            header += f"  {short[f]:>10}"
        print(header)
        print("  " + "-" * (14 + 12 * len(fams_with_data)))

        agreement_rates = {}
        for fa in fams_with_data:
            row = f"  {short[fa]:>12}"
            for fb_name in fams_with_data:
                if fa == fb_name:
                    row += f"  {'---':>10}"
                else:
                    key = (fa, fb_name) if (fa, fb_name) in pairwise_total else (fb_name, fa)
                    total = pairwise_total.get(key, 0)
                    agree = pairwise_agreement.get(key, 0)
                    if total > 0:
                        rate = agree / total
                        row += f"  {rate:>10.1%}"
                        agreement_rates[key] = rate
                    else:
                        row += f"  {'N/A':>10}"
            print(row)

        if agreement_rates:
            mean_agree = np.mean(list(agreement_rates.values()))
            print()
            print(f"  Mean pairwise agreement: {mean_agree:.1%}")

    return study_diversity, dict(pairwise_agreement), dict(pairwise_total)


# ── Analysis 4: Winner Diversity ─────────────────────────────────────────


def analyze_winner_diversity(bo_results, bo_study_info):
    subsection("A4. Which Family Wins Most Often? (best final_best per study-seed)")
    print()

    all_sids = sorted(set(sid for sid, _, _ in bo_results))
    all_strategies = sorted(set(strat for _, strat, _ in bo_results))
    families = [f for f in FAMILIES_TO_SHOW if f != "Random"]

    win_counts = defaultdict(int)
    total_races = 0

    for sid in all_sids:
        for seed in SEEDS:
            fam_bests = {}
            for fam in families:
                fb_vals = []
                for strat in all_strategies:
                    if STRATEGY_FAMILY.get(strat, "Other") != fam:
                        continue
                    key = (sid, strat, seed)
                    if key not in bo_results:
                        continue
                    fb = _get_final_best(bo_results[key])
                    if fb is not None:
                        fb_vals.append(fb)
                if fb_vals:
                    fam_bests[fam] = max(fb_vals)

            if not fam_bests:
                continue

            best_val = max(fam_bests.values())
            winners = [f for f, v in fam_bests.items() if abs(v - best_val) < TOL]
            total_races += 1
            for w in winners:
                win_counts[w] += 1

    if total_races == 0:
        print("  No data available.")
        return {}

    print(f"  {'Family':<16}  {'Wins':>6}  {'Win Rate':>8}  {'Total':>6}")
    print("  " + "-" * 42)

    win_rates = {}
    for fam in families:
        if fam not in win_counts:
            continue
        rate = win_counts[fam] / total_races
        win_rates[fam] = rate
        print(f"  {fam:<16}  {win_counts[fam]:>6}  {rate:>8.1%}  {total_races:>6}")

    print()
    n_unique_winners = len(win_counts)
    print(
        f"  {n_unique_winners} of {len(families)} families won at least once across "
        f"{total_races} study-seed combinations."
    )

    # Exclusive wins: study-seeds where only ONE family found the best
    exclusive = defaultdict(int)
    for sid in all_sids:
        for seed in SEEDS:
            fam_bests = {}
            for fam in families:
                fb_vals = []
                for strat in all_strategies:
                    if STRATEGY_FAMILY.get(strat, "Other") != fam:
                        continue
                    key = (sid, strat, seed)
                    if key not in bo_results:
                        continue
                    fb = _get_final_best(bo_results[key])
                    if fb is not None:
                        fb_vals.append(fb)
                if fb_vals:
                    fam_bests[fam] = max(fb_vals)
            if not fam_bests:
                continue
            best_val = max(fam_bests.values())
            winners = [f for f, v in fam_bests.items() if abs(v - best_val) < TOL]
            if len(winners) == 1:
                exclusive[winners[0]] += 1

    if exclusive:
        subsection("Exclusive Wins (only one family found the top hit)")
        print(f"  {'Family':<16}  {'Exclusive Wins':>14}")
        print("  " + "-" * 34)
        for fam in families:
            if fam in exclusive:
                print(f"  {fam:<16}  {exclusive[fam]:>14}")
        total_exclusive = sum(exclusive.values())
        shared = total_races - total_exclusive
        print()
        print(f"  Shared wins (tied): {shared}/{total_races} ({shared / total_races:.0%})")
        print(f"  Exclusive wins:     {total_exclusive}/{total_races} ({total_exclusive / total_races:.0%})")

    return dict(win_rates)


# ── Analysis 5: BO vs P&R per study ─────────────────────────────────────


def analyze_per_study_bo_vs_pr(bo_results, pr_results, matched_sids, bo_study_info):
    subsection("A5. Per-Study BO vs P&R Comparison (best surrogate family)")
    print("  For each study: does the best BO strategy (XGB/RF/NGBoost family)")
    print("  find the same top formulation as its P&R counterpart?")
    print()

    print(
        f"  {'Study ID':>22}  {'N':>5}  {'Type':>6}  {'Surrog':>8}  "
        f"{'BO fb':>8}  {'P&R fb':>8}  {'Same':>6}  {'BO R5%':>7}  {'P&R R5%':>7}"
    )
    print("  " + "-" * 95)

    type_abbrev = {
        "il_diverse_fixed_ratios": "fix",
        "il_diverse_variable_ratios": "var",
        "ratio_only": "ratio",
    }

    per_study = {}
    for sid in matched_sids:
        si = bo_study_info.get(sid, {})
        tabbr = type_abbrev.get(si.get("study_type", ""), "?")
        n_form = si.get("n_formulations", 0)

        for surrogate in ["xgb", "rf", "ngboost"]:
            bo_strategies = [s for s, sur in BO_STRATEGY_TO_SURROGATE.items() if sur == surrogate]
            seed_results = []
            for seed in SEEDS:
                pr_key = (sid, surrogate, seed)
                if pr_key not in pr_results:
                    continue
                pr_data = pr_results[pr_key]
                pr_fb = _get_final_best(pr_data)
                pr_recall = _get_recall(pr_data)
                if pr_fb is None:
                    continue

                best_bo_fb = None
                best_bo_recall = None
                for bo_strat in bo_strategies:
                    bo_key = (sid, bo_strat, seed)
                    if bo_key not in bo_results:
                        continue
                    fb = _get_final_best(bo_results[bo_key])
                    r = _get_recall(bo_results[bo_key])
                    if fb is not None and (best_bo_fb is None or fb > best_bo_fb):
                        best_bo_fb = fb
                        best_bo_recall = r

                if best_bo_fb is not None:
                    seed_results.append(
                        {
                            "same": abs(best_bo_fb - pr_fb) < TOL,
                            "bo_fb": best_bo_fb,
                            "pr_fb": pr_fb,
                            "bo_recall": best_bo_recall,
                            "pr_recall": pr_recall,
                        }
                    )

            if not seed_results:
                continue

            agree_rate = np.mean([r["same"] for r in seed_results])
            mean_bo_fb = np.mean([r["bo_fb"] for r in seed_results])
            mean_pr_fb = np.mean([r["pr_fb"] for r in seed_results])
            mean_bo_r = np.mean([r["bo_recall"] for r in seed_results if r["bo_recall"] is not None])
            mean_pr_r = np.mean([r["pr_recall"] for r in seed_results if r["pr_recall"] is not None])

            same_str = f"{agree_rate:.0%}"
            print(
                f"  {sid:>22}  {n_form:>5}  {tabbr:>6}  {SURROGATE_DISPLAY[surrogate]:>8}  "
                f"{mean_bo_fb:>8.3f}  {mean_pr_fb:>8.3f}  {same_str:>6}  "
                f"{mean_bo_r:>7.3f}  {mean_pr_r:>7.3f}"
            )

            per_study[(sid, surrogate)] = {
                "agree_rate": float(agree_rate),
                "mean_bo_recall": float(mean_bo_r),
                "mean_pr_recall": float(mean_pr_r),
                "n_seeds": len(seed_results),
            }

    return per_study


# ── Figure ────────────────────────────────────────────────────────────────


def make_figure(recall_pairs, surrogate_stats, study_diversity, bo_results, bo_study_info):
    try:
        sys.path.insert(0, str(paper_root(_PACKAGE_ROOT)))
        from paper.figure_style import (
            DOUBLE_COL,
            light_ygrid,
            panel_label,
            save_figure,
            setup_style,
        )
    except ImportError:
        print("  WARNING: Could not import paper.figure_style; skipping figure.", file=sys.stderr)
        return

    setup_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, 2.5))

    # Panel a: BO vs P&R recall scatter
    ax = axes[0]
    surr_markers = {"xgb": "o", "rf": "s", "ngboost": "^"}
    surr_colors = {"xgb": "#CCBB44", "rf": "#228833", "ngboost": "#EE6677"}

    for surrogate in ["xgb", "rf", "ngboost"]:
        if surrogate not in recall_pairs:
            continue
        pairs = recall_pairs[surrogate]
        bo_vals = [p[0] for p in pairs]
        pr_vals = [p[1] for p in pairs]
        ax.scatter(
            pr_vals,
            bo_vals,
            marker=surr_markers[surrogate],
            c=surr_colors[surrogate],
            s=12,
            alpha=0.4,
            label=SURROGATE_DISPLAY[surrogate],
            edgecolors="none",
            rasterized=True,
        )

    lims = [0, 1.05]
    ax.plot(lims, lims, "k--", lw=0.5, alpha=0.5, zorder=0)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("P&R recall (top 5%)")
    ax.set_ylabel("BO recall (top 5%)")
    ax.legend(fontsize=5.5, loc="lower right", frameon=False)
    light_ygrid(ax)
    panel_label(ax, "a")

    # Panel b: Hit agreement by surrogate
    ax = axes[1]
    surrogates_with_data = [s for s in ["xgb", "rf", "ngboost"] if s in surrogate_stats]
    x_pos = np.arange(len(surrogates_with_data))
    rates = [surrogate_stats[s]["hit_agreement_rate"] for s in surrogates_with_data]
    ci_lo = [surrogate_stats[s]["hit_agreement_ci"][0] for s in surrogates_with_data]
    ci_hi = [surrogate_stats[s]["hit_agreement_ci"][1] for s in surrogates_with_data]
    colors = [surr_colors[s] for s in surrogates_with_data]
    yerr_lo = [r - lo for r, lo in zip(rates, ci_lo)]
    yerr_hi = [hi - r for r, hi in zip(rates, ci_hi)]

    ax.bar(x_pos, rates, color=colors, width=0.6, edgecolor="white", linewidth=0.5)
    ax.errorbar(
        x_pos, rates, yerr=[yerr_lo, yerr_hi], fmt="none", ecolor="black", capsize=3, capthick=0.5, linewidth=0.5
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels([SURROGATE_DISPLAY[s] for s in surrogates_with_data])
    ax.set_ylabel("Hit agreement rate")
    ax.set_ylim(0, 1.05)
    light_ygrid(ax)
    panel_label(ax, "b")

    # Panel c: Distribution of distinct hits per study
    ax = axes[2]
    if study_diversity:
        div_vals = [sd["mean_distinct_hits"] for sd in study_diversity.values()]
        ax.hist(
            div_vals, bins=np.arange(0.5, max(div_vals) + 1.5, 1), color="#4477AA", edgecolor="white", linewidth=0.5
        )
        ax.axvline(np.mean(div_vals), color="black", ls="--", lw=0.8, label=f"mean={np.mean(div_vals):.1f}")
        ax.set_xlabel("Distinct top hits per study")
        ax.set_ylabel("Number of studies")
        ax.legend(fontsize=5.5, frameon=False)
    light_ygrid(ax)
    panel_label(ax, "c")

    fig.tight_layout(w_pad=2.0)
    path = FIG_DIR / "fig_diversity.pdf"
    save_figure(fig, path)
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────


def main():
    print("Loading BO results...")
    bo_results, bo_study_info = load_bo_results()
    print(f"  Loaded {len(bo_results)} BO results across {len(set(s for s, _, _ in bo_results))} studies")

    print("Loading P&R results...")
    pr_results = load_pr_results()
    print(f"  Loaded {len(pr_results)} P&R results across {len(set(s for s, _, _ in pr_results))} study-ids")

    matched_sids = _find_matching_study_ids(bo_results, pr_results)
    print(f"  Matched studies (same n_formulations): {len(matched_sids)}")
    if not matched_sids:
        print("  WARNING: No matching studies found. Cannot compare BO vs P&R.")
        print("  Proceeding with BO-only inter-family analysis.")

    # Analysis 1: BO vs P&R hit agreement
    surrogate_stats, all_comparisons = {}, []
    if matched_sids:
        surrogate_stats, all_comparisons = analyze_bo_vs_pr(bo_results, pr_results, matched_sids)

    # Analysis 2: BO vs P&R recall
    recall_stats, recall_pairs = {}, {}
    if matched_sids:
        recall_stats, recall_pairs = analyze_recall_comparison(bo_results, pr_results, matched_sids)

    # Analysis 3: Inter-family diversity
    study_diversity, _pw_agree, _pw_total = analyze_interfamily_diversity(bo_results, bo_study_info)

    # Analysis 4: Winner diversity
    win_rates = analyze_winner_diversity(bo_results, bo_study_info)

    # Analysis 5: Per-study BO vs P&R
    per_study = {}
    if matched_sids:
        per_study = analyze_per_study_bo_vs_pr(bo_results, pr_results, matched_sids, bo_study_info)

    # Save JSON
    output = {
        "surrogate_stats": surrogate_stats,
        "recall_stats": recall_stats,
        "study_diversity": study_diversity,
        "win_rates": win_rates,
        "n_matched_studies": len(matched_sids),
        "matched_study_ids": matched_sids,
        "n_comparisons": len(all_comparisons),
        "per_study_bo_vs_pr": {f"{k[0]}_{k[1]}": v for k, v in per_study.items()},
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(output, indent=2))
    print(f"\n  Saved results to {OUT_JSON}")

    # Figure
    make_figure(recall_pairs, surrogate_stats, study_diversity, bo_results, bo_study_info)

    print()
    print("=" * 80)
    print("  END OF DIVERSITY ANALYSIS")
    print("=" * 80)


if __name__ == "__main__":
    main()
