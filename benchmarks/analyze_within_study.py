#!/usr/bin/env python3
"""
Analysis of within-study Bayesian optimization benchmark results.

Reads JSON result files from benchmark_results/within_study/<PMID>/<strategy>_s<seed>.json
and produces a structured analysis suitable for a research notebook.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

RESULTS_DIR = Path(__file__).resolve().parent.parent / "benchmark_results" / "within_study"

SEEDS = [42, 123, 456, 789, 2024]

STRATEGY_FAMILY = {
    "random": "Random",
    "lnpbo_ucb": "LNPBO",
    "lnpbo_ei": "LNPBO",
    "lnpbo_logei": "LNPBO",
    "lnpbo_lp_ei": "LNPBO",
    "lnpbo_lp_logei": "LNPBO",
    "lnpbo_pls_logei": "LNPBO",
    "lnpbo_pls_lp_logei": "LNPBO",
    "lnpbo_rkb_logei": "LNPBO",
    "lnpbo_ts_batch": "LNPBO",
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
    "lnpbo_gibbon": "LNPBO",
    "lnpbo_jes": "LNPBO",
    "lnpbo_tanimoto_ts": "LNPBO",
    "lnpbo_tanimoto_logei": "LNPBO",
}

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
    "discrete_gp_ucb": "GP-UCB (sklearn)",
    "lnpbo_gibbon": "LNPBO-GIBBON",
    "lnpbo_jes": "LNPBO-JES",
    "lnpbo_tanimoto_ts": "LNPBO-Tani-TS",
    "lnpbo_tanimoto_logei": "LNPBO-Tani-LogEI",
}


def load_all_results():
    results = []
    for pmid_dir in sorted(RESULTS_DIR.iterdir()):
        if not pmid_dir.is_dir():
            continue
        for f in sorted(pmid_dir.glob("*.json")):
            try:
                data = json.loads(f.read_text())
                results.append(data)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"WARNING: failed to load {f}: {e}", file=sys.stderr)
    return results


def extract_strategy_name(filename):
    name = filename.replace(".json", "")
    for seed in SEEDS:
        name = name.replace(f"_s{seed}", "")
    return name


def build_tables(results):
    # study_info keyed by pmid
    study_info = {}
    # (pmid, strategy, seed) -> result dict
    result_map = {}
    strategies = set()

    for r in results:
        pmid = int(r["pmid"])
        strategy = r["strategy"]
        seed = r["seed"]
        study_info[pmid] = r["study_info"]
        result_map[(pmid, strategy, seed)] = r
        strategies.add(strategy)

    pmids = sorted(study_info.keys())
    strategies = sorted(strategies)
    return study_info, result_map, pmids, strategies


def section(title):
    print(f"\n{title}\n")


def subsection(title):
    print(f"\n{title}")


def print_study_landscape(study_info, pmids):
    section("A. STUDY LANDSCAPE")

    by_type = defaultdict(list)
    for pmid in pmids:
        si = study_info[pmid]
        by_type[si["study_type"]].append(pmid)

    type_order = ["il_diverse_fixed_ratios", "il_diverse_variable_ratios", "ratio_only"]
    type_labels = {
        "il_diverse_fixed_ratios": "IL-diverse, fixed ratios",
        "il_diverse_variable_ratios": "IL-diverse, variable ratios",
        "ratio_only": "Ratio-only",
    }

    print(f"{'PMID':>10}  {'N':>5}  {'ILs':>5}  {'HLs':>4}  {'CHLs':>4}  {'PEGs':>4}  "
          f"{'Seed N':>6}  {'Budget':>6}  {'Ratio Std':>9}  {'Feature Type':<20}  {'Study Type'}")
    print("-" * 120)

    for stype in type_order:
        if stype not in by_type:
            continue
        sorted_pmids = sorted(by_type[stype], key=lambda p: study_info[p]["n_formulations"])
        for pmid in sorted_pmids:
            si = study_info[pmid]
            budget = si["n_seed"] + si["n_rounds"] * si["batch_size"]
            ratio_std = si.get("il_ratio_std", 0)
            print(f"{pmid:>10}  {si['n_formulations']:>5}  {si['n_unique_il']:>5}  "
                  f"{si.get('n_unique_hl', '?'):>4}  {si.get('n_unique_chl', '?'):>4}  "
                  f"{si.get('n_unique_peg', '?'):>4}  {si['n_seed']:>6}  {budget:>6}  "
                  f"{ratio_std:>9.1f}  {si.get('feature_type', 'N/A'):<20}  "
                  f"{type_labels.get(stype, stype)}")
        print()

    subsection("Summary by Study Type")
    for stype in type_order:
        if stype not in by_type:
            continue
        plist = by_type[stype]
        ns = [study_info[p]["n_formulations"] for p in plist]
        print(f"  {type_labels[stype]:40s}  count={len(plist):>2}  "
              f"N: {min(ns):>5}-{max(ns):>5}  (median {int(np.median(ns)):>5})")


def get_top5_recall(result_map, pmid, strategy, seed):
    """Get top-5% recall for a (pmid, strategy, seed) triple. Returns None if missing."""
    key = (pmid, strategy, seed)
    if key not in result_map:
        return None
    try:
        return result_map[key]["result"]["metrics"]["top_k_recall"]["5"]
    except (KeyError, TypeError):
        return None


def print_overall_rankings(study_info, result_map, pmids, strategies):
    section("B. OVERALL STRATEGY RANKINGS (Top-5% Recall)")

    # Build paired arrays: for each (pmid, seed), collect values
    random_vals = {}  # (pmid, seed) -> recall
    for pmid in pmids:
        for seed in SEEDS:
            val = get_top5_recall(result_map, pmid, "random", seed)
            if val is not None:
                random_vals[(pmid, seed)] = val

    pairs = sorted(random_vals.keys())

    rows = []
    for strategy in strategies:
        vals = []
        paired_strategy = []
        paired_random = []
        for pmid, seed in pairs:
            v = get_top5_recall(result_map, pmid, strategy, seed)
            if v is not None:
                vals.append(v)
                paired_strategy.append(v)
                paired_random.append(random_vals[(pmid, seed)])

        if len(vals) == 0:
            continue

        mean_val = np.mean(vals)
        std_val = np.std(vals, ddof=1)
        median_val = np.median(vals)
        n = len(vals)

        # Paired Wilcoxon vs random
        if strategy != "random" and len(paired_strategy) >= 5:
            diffs = np.array(paired_strategy) - np.array(paired_random)
            nonzero = diffs[diffs != 0]
            if len(nonzero) >= 5:
                _stat, p_val = stats.wilcoxon(nonzero, alternative="greater")
            else:
                p_val = 1.0
        else:
            p_val = None

        # Mean of random for lift calculation
        mean_random = np.mean(paired_random) if paired_random else 0

        rows.append({
            "strategy": strategy,
            "short": STRATEGY_SHORT.get(strategy, strategy),
            "family": STRATEGY_FAMILY.get(strategy, "Other"),
            "mean": mean_val,
            "std": std_val,
            "median": median_val,
            "n": n,
            "p_val": p_val,
            "mean_random": mean_random,
        })

    rows.sort(key=lambda r: r["mean"], reverse=True)

    subsection("Individual Strategy Rankings")
    print(f"{'Rank':>4}  {'Strategy':<22}  {'Family':<14}  "
          f"{'Mean':>6}  {'Std':>6}  {'Median':>6}  {'N':>5}  "
          f"{'p (vs rand)':>11}  {'Sig':>3}  {'Lift':>6}")
    print("-" * 110)

    for i, r in enumerate(rows, 1):
        p_str = f"{r['p_val']:.4f}" if r["p_val"] is not None else "   ---"
        sig = ""
        if r["p_val"] is not None:
            if r["p_val"] < 0.001:
                sig = "***"
            elif r["p_val"] < 0.01:
                sig = " **"
            elif r["p_val"] < 0.05:
                sig = "  *"
        lift = r["mean"] / r["mean_random"] if r["mean_random"] > 0 else float("inf")
        lift_str = f"{lift:.2f}x" if r["strategy"] != "random" else "  ---"
        print(f"{i:>4}  {r['short']:<22}  {r['family']:<14}  "
              f"{r['mean']:>6.3f}  {r['std']:>6.3f}  {r['median']:>6.3f}  {r['n']:>5}  "
              f"{p_str:>11}  {sig:>3}  {lift_str:>6}")

    # Family-level aggregation
    subsection("Strategy Family Rankings (aggregated)")
    family_vals = defaultdict(list)
    family_paired = defaultdict(lambda: ([], []))
    for strategy in strategies:
        fam = STRATEGY_FAMILY.get(strategy, "Other")
        for pmid, seed in pairs:
            v = get_top5_recall(result_map, pmid, strategy, seed)
            rv = random_vals.get((pmid, seed))
            if v is not None and rv is not None:
                family_vals[fam].append(v)
                family_paired[fam][0].append(v)
                family_paired[fam][1].append(rv)

    fam_rows = []
    for fam in sorted(family_vals.keys()):
        vals = family_vals[fam]
        mean_val = np.mean(vals)
        std_val = np.std(vals, ddof=1)
        n = len(vals)
        s_vals, r_vals = family_paired[fam]
        diffs = np.array(s_vals) - np.array(r_vals)
        nonzero = diffs[diffs != 0]
        if fam != "Random" and len(nonzero) >= 5:
            _, p_val = stats.wilcoxon(nonzero, alternative="greater")
        else:
            p_val = None
        mean_random = np.mean(r_vals) if r_vals else 0
        fam_rows.append((fam, mean_val, std_val, n, p_val, mean_random))

    fam_rows.sort(key=lambda x: x[1], reverse=True)

    print(f"{'Family':<16}  {'Mean':>6}  {'Std':>6}  {'N obs':>6}  "
          f"{'p (vs rand)':>11}  {'Sig':>3}  {'Lift':>6}")
    print("-" * 75)
    for fam, mean_val, std_val, n, p_val, mean_random in fam_rows:
        p_str = f"{p_val:.4f}" if p_val is not None else "   ---"
        sig = ""
        if p_val is not None:
            if p_val < 0.001:
                sig = "***"
            elif p_val < 0.01:
                sig = " **"
            elif p_val < 0.05:
                sig = "  *"
        lift = mean_val / mean_random if mean_random > 0 and fam != "Random" else 0
        lift_str = f"{lift:.2f}x" if fam != "Random" else "  ---"
        print(f"{fam:<16}  {mean_val:>6.3f}  {std_val:>6.3f}  {n:>6}  "
              f"{p_str:>11}  {sig:>3}  {lift_str:>6}")

    return rows


def print_performance_by_study_type(study_info, result_map, pmids, strategies):
    section("C. PERFORMANCE BY STUDY TYPE")

    type_order = ["il_diverse_fixed_ratios", "il_diverse_variable_ratios", "ratio_only"]
    type_labels = {
        "il_diverse_fixed_ratios": "IL-Diverse, Fixed Ratios",
        "il_diverse_variable_ratios": "IL-Diverse, Variable Ratios",
        "ratio_only": "Ratio-Only",
    }

    pmids_by_type = defaultdict(list)
    for pmid in pmids:
        pmids_by_type[study_info[pmid]["study_type"]].append(pmid)

    for stype in type_order:
        type_pmids = pmids_by_type.get(stype, [])
        if not type_pmids:
            continue

        subsection(f"{type_labels[stype]} ({len(type_pmids)} studies)")

        # Compute per-family means across studies in this type
        family_vals = defaultdict(list)
        random_vals_type = []
        for pmid in type_pmids:
            for seed in SEEDS:
                rv = get_top5_recall(result_map, pmid, "random", seed)
                if rv is not None:
                    random_vals_type.append(rv)

        mean_random = np.mean(random_vals_type) if random_vals_type else 0

        # Per-strategy
        strat_means = {}
        for strategy in strategies:
            vals = []
            for pmid in type_pmids:
                for seed in SEEDS:
                    v = get_top5_recall(result_map, pmid, strategy, seed)
                    if v is not None:
                        vals.append(v)
            if vals:
                strat_means[strategy] = (np.mean(vals), np.std(vals, ddof=1), len(vals))

        # Per-family
        for strategy in strategies:
            fam = STRATEGY_FAMILY.get(strategy, "Other")
            if strategy in strat_means:
                for pmid in type_pmids:
                    for seed in SEEDS:
                        v = get_top5_recall(result_map, pmid, strategy, seed)
                        if v is not None:
                            family_vals[fam].append(v)

        # Print strategy table for this type
        sorted_strats = sorted(strat_means.items(), key=lambda x: x[1][0], reverse=True)
        print(f"  {'Strategy':<22}  {'Family':<14}  {'Mean':>6}  {'Std':>6}  {'N':>5}  {'Lift':>6}")
        print("  " + "-" * 80)
        for strategy, (mean_v, std_v, n) in sorted_strats:
            fam = STRATEGY_FAMILY.get(strategy, "Other")
            short = STRATEGY_SHORT.get(strategy, strategy)
            lift = mean_v / mean_random if mean_random > 0 and strategy != "random" else 0
            lift_str = f"{lift:.2f}x" if strategy != "random" else "  ---"
            print(f"  {short:<22}  {fam:<14}  {mean_v:>6.3f}  {std_v:>6.3f}  {n:>5}  {lift_str:>6}")

        # Family summary
        print()
        print("  Family Summary:")
        fam_summary = []
        for fam, vals in family_vals.items():
            fam_summary.append((fam, np.mean(vals), np.std(vals, ddof=1), len(vals)))
        fam_summary.sort(key=lambda x: x[1], reverse=True)
        for fam, mean_v, std_v, _n in fam_summary:
            lift = mean_v / mean_random if mean_random > 0 and fam != "Random" else 0
            lift_str = f"{lift:.2f}x" if fam != "Random" else "  ---"
            print(f"    {fam:<16}  mean={mean_v:.3f}  std={std_v:.3f}  lift={lift_str}")

    # Direct comparison: LNPBO vs best alternative by type
    subsection("Key Question: Where does LNPBO excel?")
    for stype in type_order:
        type_pmids = pmids_by_type.get(stype, [])
        if not type_pmids:
            continue

        lnpbo_vals = []
        random_type = []
        best_alt_family = defaultdict(list)
        for pmid in type_pmids:
            for seed in SEEDS:
                rv = get_top5_recall(result_map, pmid, "random", seed)
                if rv is not None:
                    random_type.append(rv)
                for strategy in strategies:
                    v = get_top5_recall(result_map, pmid, strategy, seed)
                    if v is None:
                        continue
                    fam = STRATEGY_FAMILY.get(strategy, "Other")
                    if fam == "LNPBO":
                        lnpbo_vals.append(v)
                    elif fam not in ("Random",):
                        best_alt_family[fam].append(v)

        mean_random = np.mean(random_type) if random_type else 0
        mean_lnpbo = np.mean(lnpbo_vals) if lnpbo_vals else 0
        best_alt_name = ""
        best_alt_mean = 0
        for fam, vals in best_alt_family.items():
            m = np.mean(vals)
            if m > best_alt_mean:
                best_alt_mean = m
                best_alt_name = fam

        print(f"  {type_labels[stype]:40s}  "
              f"Random={mean_random:.3f}  LNPBO={mean_lnpbo:.3f} ({mean_lnpbo/mean_random:.2f}x)  "
              f"Best Alt={best_alt_name} {best_alt_mean:.3f} ({best_alt_mean/mean_random:.2f}x)")


def print_performance_by_size(study_info, result_map, pmids, strategies):
    section("D. PERFORMANCE BY DATASET SIZE")

    size_bins = {
        "Small (<500)": lambda n: n < 500,
        "Medium (500-1000)": lambda n: 500 <= n <= 1000,
        "Large (>1000)": lambda n: n > 1000,
    }

    for bin_label, predicate in size_bins.items():
        bin_pmids = [p for p in pmids if predicate(study_info[p]["n_formulations"])]
        if not bin_pmids:
            continue

        ns = [study_info[p]["n_formulations"] for p in bin_pmids]
        subsection(f"{bin_label} ({len(bin_pmids)} studies, N={min(ns)}-{max(ns)})")

        random_vals = []
        for pmid in bin_pmids:
            for seed in SEEDS:
                rv = get_top5_recall(result_map, pmid, "random", seed)
                if rv is not None:
                    random_vals.append(rv)
        mean_random = np.mean(random_vals) if random_vals else 0

        # Family-level aggregation
        family_vals = defaultdict(list)
        for strategy in strategies:
            fam = STRATEGY_FAMILY.get(strategy, "Other")
            for pmid in bin_pmids:
                for seed in SEEDS:
                    v = get_top5_recall(result_map, pmid, strategy, seed)
                    if v is not None:
                        family_vals[fam].append(v)

        fam_rows = []
        for fam, vals in family_vals.items():
            fam_rows.append((fam, np.mean(vals), np.std(vals, ddof=1), len(vals)))
        fam_rows.sort(key=lambda x: x[1], reverse=True)

        print(f"  {'Family':<16}  {'Mean':>6}  {'Std':>6}  {'N obs':>6}  {'Lift':>6}")
        print("  " + "-" * 55)
        for fam, mean_v, std_v, n in fam_rows:
            lift = mean_v / mean_random if mean_random > 0 and fam != "Random" else 0
            lift_str = f"{lift:.2f}x" if fam != "Random" else "  ---"
            print(f"  {fam:<16}  {mean_v:>6.3f}  {std_v:>6.3f}  {n:>6}  {lift_str:>6}")

    # Cross-size comparison: GP vs tree models
    subsection("GP (LNPBO) vs Tree Models by Size")
    print(f"  {'Size Bin':<22}  {'Random':>7}  {'LNPBO':>7}  {'XGBoost':>7}  {'RF':>7}  "
          f"{'GP Lift':>7}  {'XGB Lift':>8}  {'RF Lift':>7}")
    print("  " + "-" * 85)

    for bin_label, predicate in size_bins.items():
        bin_pmids = [p for p in pmids if predicate(study_info[p]["n_formulations"])]
        if not bin_pmids:
            continue

        fam_agg = defaultdict(list)
        for strategy in strategies:
            fam = STRATEGY_FAMILY.get(strategy, "Other")
            for pmid in bin_pmids:
                for seed in SEEDS:
                    v = get_top5_recall(result_map, pmid, strategy, seed)
                    if v is not None:
                        fam_agg[fam].append(v)

        def fm(fam, _agg=fam_agg):
            return np.mean(_agg[fam]) if fam in _agg else 0

        mr = fm("Random")
        gp = fm("LNPBO")
        xgb = fm("XGBoost")
        rf = fm("RF")
        print(f"  {bin_label:<22}  {mr:>7.3f}  {gp:>7.3f}  {xgb:>7.3f}  {rf:>7.3f}  "
              f"{gp/mr if mr else 0:>6.2f}x  {xgb/mr if mr else 0:>7.2f}x  "
              f"{rf/mr if mr else 0:>6.2f}x")


def print_convergence_analysis(study_info, result_map, pmids, strategies):
    section("E. CONVERGENCE ANALYSIS")

    subsection("Mean best_so_far by Round (averaged across all studies and seeds)")

    # Collect trajectories per family
    family_trajectories = defaultdict(list)
    for strategy in strategies:
        fam = STRATEGY_FAMILY.get(strategy, "Other")
        for pmid in pmids:
            for seed in SEEDS:
                key = (pmid, strategy, seed)
                if key not in result_map:
                    continue
                bsf = result_map[key]["result"].get("best_so_far", [])
                if len(bsf) > 0:
                    family_trajectories[fam].append(bsf)

    # Determine max rounds (typically 16: seed + 15 rounds)
    max_len = 16

    # Normalize trajectories: since studies have different scales (z-scored within study),
    # we compute the fraction of oracle best achieved at each round
    # Oracle best per (pmid, seed) = max best_so_far across all strategies
    oracle_best = {}
    for pmid in pmids:
        for seed in SEEDS:
            best = -np.inf
            for strategy in strategies:
                key = (pmid, strategy, seed)
                if key in result_map:
                    bsf = result_map[key]["result"].get("best_so_far", [])
                    if bsf:
                        best = max(best, max(bsf))
            if best > -np.inf:
                oracle_best[(pmid, seed)] = best

    # Compute normalized trajectories per family
    family_norm_trajectories = defaultdict(list)
    for strategy in strategies:
        fam = STRATEGY_FAMILY.get(strategy, "Other")
        for pmid in pmids:
            for seed in SEEDS:
                key = (pmid, strategy, seed)
                if key not in result_map:
                    continue
                bsf = result_map[key]["result"].get("best_so_far", [])
                ob = oracle_best.get((pmid, seed))
                if ob and ob > 0 and len(bsf) == max_len:
                    norm = [v / ob for v in bsf]
                    family_norm_trajectories[fam].append(norm)

    # Print round-by-round table
    families_to_show = ["Random", "LNPBO", "CASMOPolitan", "XGBoost", "RF", "NGBoost",
                        "Deep Ensemble", "GP (sklearn)"]

    header = f"  {'Round':>5}"
    for fam in families_to_show:
        header += f"  {fam:>14}"
    print(header)
    print("  " + "-" * (7 + 16 * len(families_to_show)))

    for rnd in range(max_len):
        row = f"  {rnd:>5}"
        for fam in families_to_show:
            trajs = family_norm_trajectories.get(fam, [])
            if trajs:
                vals = [t[rnd] for t in trajs if rnd < len(t)]
                if vals:
                    row += f"  {np.mean(vals):>14.4f}"
                else:
                    row += f"  {'---':>14}"
            else:
                row += f"  {'---':>14}"
        print(row)

    # Round at which 90% of final performance is reached
    subsection("Rounds to reach 90% of final performance (per family)")
    print(f"  {'Family':<16}  {'Round (90% final)':>17}  {'Round (95% final)':>17}")
    print("  " + "-" * 55)

    for fam in families_to_show:
        trajs = family_norm_trajectories.get(fam, [])
        if not trajs:
            continue
        # For each trajectory, find round where it reaches 90% and 95% of its final value
        rounds_90 = []
        rounds_95 = []
        for t in trajs:
            final = t[-1]
            if final <= 0:
                continue
            for r, v in enumerate(t):
                if v >= 0.90 * final:
                    rounds_90.append(r)
                    break
            else:
                rounds_90.append(len(t) - 1)
            for r, v in enumerate(t):
                if v >= 0.95 * final:
                    rounds_95.append(r)
                    break
            else:
                rounds_95.append(len(t) - 1)
        m90 = np.mean(rounds_90)
        m95 = np.mean(rounds_95)
        print(f"  {fam:<16}  {m90:>17.1f}  {m95:>17.1f}")

    # Strategy-level: mean fraction of oracle at each round for top strategies
    subsection("Top strategies: fraction of oracle best by round")
    top_strats = ["random", "lnpbo_logei", "lnpbo_lp_logei", "lnpbo_ucb",
                  "discrete_xgb_ucb", "discrete_rf_ts_batch", "discrete_ngboost_ucb",
                  "casmopolitan_ucb"]

    strat_norm = defaultdict(list)
    for strategy in top_strats:
        for pmid in pmids:
            for seed in SEEDS:
                key = (pmid, strategy, seed)
                if key not in result_map:
                    continue
                bsf = result_map[key]["result"].get("best_so_far", [])
                ob = oracle_best.get((pmid, seed))
                if ob and ob > 0 and len(bsf) == max_len:
                    norm = [v / ob for v in bsf]
                    strat_norm[strategy].append(norm)

    header = f"  {'Round':>5}"
    for s in top_strats:
        short = STRATEGY_SHORT.get(s, s)[:14]
        header += f"  {short:>14}"
    print(header)
    print("  " + "-" * (7 + 16 * len(top_strats)))

    for rnd in range(max_len):
        row = f"  {rnd:>5}"
        for s in top_strats:
            trajs = strat_norm.get(s, [])
            if trajs:
                vals = [t[rnd] for t in trajs if rnd < len(t)]
                if vals:
                    row += f"  {np.mean(vals):>14.4f}"
                else:
                    row += f"  {'---':>14}"
            else:
                row += f"  {'---':>14}"
        print(row)


def print_timing_analysis(result_map, pmids, strategies):
    section("F. TIMING ANALYSIS")

    subsection("Wall-clock time per strategy (seconds, mean across all studies and seeds)")

    strat_times = defaultdict(list)
    for (_pmid, strategy, _seed), r in result_map.items():
        elapsed = r["result"].get("elapsed", 0)
        strat_times[strategy].append(elapsed)

    rows = []
    for strategy in sorted(strat_times.keys()):
        times = strat_times[strategy]
        rows.append((strategy, np.mean(times), np.median(times), np.std(times, ddof=1),
                      np.min(times), np.max(times), len(times)))

    rows.sort(key=lambda x: x[1])

    print(f"  {'Strategy':<22}  {'Mean (s)':>8}  {'Median':>8}  {'Std':>8}  "
          f"{'Min':>8}  {'Max':>8}  {'N':>4}")
    print("  " + "-" * 85)
    for strategy, mean_t, med_t, std_t, min_t, max_t, n in rows:
        short = STRATEGY_SHORT.get(strategy, strategy)
        print(f"  {short:<22}  {mean_t:>8.1f}  {med_t:>8.1f}  {std_t:>8.1f}  "
              f"{min_t:>8.1f}  {max_t:>8.1f}  {n:>4}")

    # Pareto frontier: performance vs compute
    subsection("Pareto Frontier: Top-5% Recall vs Wall-Clock Time")

    strat_perf = {}
    for strategy in strategies:
        vals = []
        for pmid in pmids:
            for seed in SEEDS:
                v = get_top5_recall(result_map, pmid, strategy, seed)
                if v is not None:
                    vals.append(v)
        if vals:
            strat_perf[strategy] = np.mean(vals)

    pareto_data = []
    for strategy in strategies:
        if strategy in strat_perf and strategy in strat_times:
            mean_time = np.mean(strat_times[strategy])
            mean_recall = strat_perf[strategy]
            pareto_data.append((strategy, mean_time, mean_recall))

    pareto_data.sort(key=lambda x: x[1])

    # Find Pareto-optimal strategies (min time for given performance)
    pareto_front = []
    best_so_far = -1
    for strategy, _t, perf in pareto_data:
        if perf > best_so_far:
            pareto_front.append(strategy)
            best_so_far = perf

    print(f"  {'Strategy':<22}  {'Time (s)':>8}  {'Top-5%':>7}  {'Pareto':>6}")
    print("  " + "-" * 55)
    for strategy, t, perf in pareto_data:
        short = STRATEGY_SHORT.get(strategy, strategy)
        is_pareto = "  ***" if strategy in pareto_front else ""
        print(f"  {short:<22}  {t:>8.1f}  {perf:>7.3f}{is_pareto}")

    # Family-level timing
    subsection("Family-level timing")
    family_times = defaultdict(list)
    for strategy in strategies:
        fam = STRATEGY_FAMILY.get(strategy, "Other")
        family_times[fam].extend(strat_times.get(strategy, []))

    print(f"  {'Family':<16}  {'Mean (s)':>8}  {'Median (s)':>10}")
    print("  " + "-" * 40)
    for fam in sorted(family_times.keys(), key=lambda f: np.mean(family_times[f])):
        times = family_times[fam]
        print(f"  {fam:<16}  {np.mean(times):>8.1f}  {np.median(times):>10.1f}")


def print_caveats():
    section("G. CAVEATS AND LIMITATIONS")

    caveats = [
        ("Cross-study comparability",
         "Each study's Experiment_value was z-scored within study before benchmarking. "
         "Top-5% recall is a rank-based metric within each study, making it somewhat "
         "comparable, but the underlying assays, cell types, and readouts differ across "
         "studies. Aggregate statistics weight all studies equally regardless of clinical "
         "relevance or assay quality."),

        ("Limited ratio diversity",
         "16 of 23 studies use fixed molar ratios (il_diverse_fixed_ratios), meaning the "
         "optimization reduces to pure IL structure screening. Only 5 studies have variable "
         "ratios with diverse ILs, and only 2 are ratio-only. Conclusions about continuous "
         "optimization or mixed discrete-continuous settings rest on very few studies."),

        ("Study size heterogeneity",
         "Study sizes span an order of magnitude (248 to 2400 formulations). The seed pool "
         "is 25% of formulations, so smaller studies start with a larger fraction of the "
         "pool already explored. Larger studies inherently have more room for improvement "
         "and dominate aggregate statistics unless weighted."),

        ("Seed sensitivity",
         "Only 5 random seeds are used. With 23 studies, this yields 115 (study, seed) "
         "pairs per strategy. While the Wilcoxon test is paired, the number of independent "
         "datapoints is modest. Confidence intervals may be wider than they appear."),

        ("Within-study benchmark != prospective performance",
         "This is a retrospective pool-based benchmark. The model selects from a finite "
         "candidate pool, not from the full chemical space. In a real self-driving lab, the "
         "candidate pool would need to be enumerated or generated, and synthesis/assay noise "
         "would affect results. The oracle (z-scored Experiment_value) is also noise-free."),

        ("Budget constraints",
         "All strategies use the same budget: 25% seed + 15 rounds of batch 12. This means "
         "total evaluation budget ranges from ~34% (small studies) to ~28% (large studies) "
         "of the full pool. Strategies might rank differently under tighter budgets."),

        ("Molecular encoding",
         "IL-diverse studies use LANTERN PCA features (Morgan FP reduced to 5 PCs). "
         "Ratio-only studies use raw molar ratios. Results are conditional on these encodings; "
         "different molecular representations might change relative strategy rankings."),
    ]

    for i, (title, text) in enumerate(caveats, 1):
        print(f"  {i}. {title}")
        # Word-wrap text at ~90 chars
        words = text.split()
        line = "     "
        for w in words:
            if len(line) + len(w) + 1 > 95:
                print(line)
                line = "     " + w
            else:
                line += " " + w if line.strip() else "     " + w
        if line.strip():
            print(line)
        print()


def print_per_study_heatmap(study_info, result_map, pmids, strategies):
    """Print a per-study breakdown: for each study, show family-level top-5% recall."""
    section("APPENDIX: PER-STUDY TOP-5% RECALL BY STRATEGY FAMILY")

    families_to_show = ["Random", "LNPBO", "CASMOPolitan", "XGBoost", "RF", "NGBoost",
                        "Deep Ensemble", "GP (sklearn)"]

    header = f"  {'PMID':>10}  {'N':>5}  {'Type':>8}"
    for fam in families_to_show:
        header += f"  {fam[:10]:>10}"
    print(header)
    print("  " + "-" * (30 + 12 * len(families_to_show)))

    type_abbrev = {
        "il_diverse_fixed_ratios": "fix",
        "il_diverse_variable_ratios": "var",
        "ratio_only": "ratio",
    }

    for pmid in sorted(pmids, key=lambda p: study_info[p]["n_formulations"]):
        si = study_info[pmid]
        tabbr = type_abbrev.get(si["study_type"], "?")
        row = f"  {pmid:>10}  {si['n_formulations']:>5}  {tabbr:>8}"

        for fam in families_to_show:
            vals = []
            for strategy in strategies:
                if STRATEGY_FAMILY.get(strategy) != fam:
                    continue
                for seed in SEEDS:
                    v = get_top5_recall(result_map, pmid, strategy, seed)
                    if v is not None:
                        vals.append(v)
            if vals:
                row += f"  {np.mean(vals):>10.3f}"
            else:
                row += f"  {'---':>10}"
        print(row)


def print_statistical_deep_dive(study_info, result_map, pmids, strategies):
    """Additional statistical analyses: effect sizes, variance decomposition."""
    section("APPENDIX: STATISTICAL DEEP DIVE")

    subsection("Variance Decomposition (Top-5% Recall)")
    # Three factors: study, strategy, seed
    # Simple additive decomposition via ANOVA-like sums of squares

    all_data = []  # (pmid, strategy, seed, value)
    for pmid in pmids:
        for strategy in strategies:
            for seed in SEEDS:
                v = get_top5_recall(result_map, pmid, strategy, seed)
                if v is not None:
                    all_data.append((pmid, strategy, seed, v))

    if not all_data:
        print("  No data available.")
        return

    values = np.array([d[3] for d in all_data])
    grand_mean = np.mean(values)
    ss_total = np.sum((values - grand_mean) ** 2)

    # Study means
    study_means = {}
    for pmid in pmids:
        vals = [d[3] for d in all_data if d[0] == pmid]
        if vals:
            study_means[pmid] = np.mean(vals)

    # Strategy means
    strat_means = {}
    for strategy in strategies:
        vals = [d[3] for d in all_data if d[1] == strategy]
        if vals:
            strat_means[strategy] = np.mean(vals)

    # Seed means
    seed_means = {}
    for seed in SEEDS:
        vals = [d[3] for d in all_data if d[2] == seed]
        if vals:
            seed_means[seed] = np.mean(vals)

    # SS for each factor
    ss_study = sum(len([d for d in all_data if d[0] == pmid]) *
                   (study_means[pmid] - grand_mean) ** 2
                   for pmid in pmids if pmid in study_means)

    ss_strategy = sum(len([d for d in all_data if d[1] == strategy]) *
                      (strat_means[strategy] - grand_mean) ** 2
                      for strategy in strategies if strategy in strat_means)

    ss_seed = sum(len([d for d in all_data if d[2] == seed]) *
                  (seed_means[seed] - grand_mean) ** 2
                  for seed in SEEDS if seed in seed_means)

    ss_residual = ss_total - ss_study - ss_strategy - ss_seed

    print(f"  Grand mean Top-5% recall: {grand_mean:.4f}")
    print(f"  Total observations: {len(all_data)}")
    print()
    print(f"  {'Source':<16}  {'SS':>10}  {'% of Total':>10}")
    print("  " + "-" * 40)
    for label, ss in [("Study", ss_study), ("Strategy", ss_strategy),
                      ("Seed", ss_seed), ("Residual", ss_residual)]:
        pct = 100 * ss / ss_total if ss_total > 0 else 0
        print(f"  {label:<16}  {ss:>10.4f}  {pct:>9.1f}%")

    # Per-study: best strategy family
    subsection("Best Strategy Family per Study")
    print(f"  {'PMID':>10}  {'N':>5}  {'Type':>8}  {'Best Family':<16}  "
          f"{'Mean Recall':>11}  {'Random':>7}  {'Lift':>6}")
    print("  " + "-" * 75)

    for pmid in sorted(pmids, key=lambda p: study_info[p]["n_formulations"]):
        si = study_info[pmid]
        tabbr = {"il_diverse_fixed_ratios": "fix",
                 "il_diverse_variable_ratios": "var",
                 "ratio_only": "ratio"}.get(si["study_type"], "?")

        random_mean = 0
        fam_means = {}
        for strategy in strategies:
            fam = STRATEGY_FAMILY.get(strategy, "Other")
            vals = []
            for seed in SEEDS:
                v = get_top5_recall(result_map, pmid, strategy, seed)
                if v is not None:
                    vals.append(v)
            if vals:
                if fam not in fam_means:
                    fam_means[fam] = []
                fam_means[fam].extend(vals)
                if strategy == "random":
                    random_mean = np.mean(vals)

        best_fam = ""
        best_mean = 0
        for fam, vals in fam_means.items():
            if fam == "Random":
                continue
            m = np.mean(vals)
            if m > best_mean:
                best_mean = m
                best_fam = fam

        lift = best_mean / random_mean if random_mean > 0 else 0
        print(f"  {pmid:>10}  {si['n_formulations']:>5}  {tabbr:>8}  {best_fam:<16}  "
              f"{best_mean:>11.3f}  {random_mean:>7.3f}  {lift:>5.2f}x")


def main():
    print("=" * 100)
    print("  WITHIN-STUDY BAYESIAN OPTIMIZATION BENCHMARK ANALYSIS")
    print("  " + "=" * 96)
    print(f"  Results directory: {RESULTS_DIR}")
    print()

    results = load_all_results()
    print(f"  Loaded {len(results)} result files")

    study_info, result_map, pmids, strategies = build_tables(results)
    print(f"  Studies: {len(pmids)}")
    print(f"  Strategies: {len(strategies)}")
    print(f"  Seeds: {SEEDS}")
    print(f"  Expected total: {len(pmids)} x {len(strategies)} x {len(SEEDS)} = "
          f"{len(pmids) * len(strategies) * len(SEEDS)}")
    print(f"  Actual loaded: {len(results)}")

    missing = 0
    for pmid in pmids:
        for strategy in strategies:
            for seed in SEEDS:
                if (pmid, strategy, seed) not in result_map:
                    missing += 1
    if missing > 0:
        print(f"  WARNING: {missing} missing (pmid, strategy, seed) combinations")
    else:
        print("  All combinations present (no missing data)")

    print_study_landscape(study_info, pmids)
    print_overall_rankings(study_info, result_map, pmids, strategies)
    print_performance_by_study_type(study_info, result_map, pmids, strategies)
    print_performance_by_size(study_info, result_map, pmids, strategies)
    print_convergence_analysis(study_info, result_map, pmids, strategies)
    print_timing_analysis(result_map, pmids, strategies)
    print_per_study_heatmap(study_info, result_map, pmids, strategies)
    print_statistical_deep_dive(study_info, result_map, pmids, strategies)
    print_caveats()

    print()
    print("=" * 100)
    print("  END OF ANALYSIS")
    print("=" * 100)


if __name__ == "__main__":
    main()
