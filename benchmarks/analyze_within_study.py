#!/usr/bin/env python3
"""Analysis of within-study Bayesian optimization benchmark results.

Reads per-seed JSON result files from
``benchmark_results/within_study/<study_id>/<strategy>_s<seed>.json``
and produces a structured analysis covering:

  A. Study landscape (size, diversity, type classification)
  B. Overall strategy rankings with BH-FDR corrected Wilcoxon tests
  C. Performance by study type (fixed-ratio, variable-ratio, ratio-only)
  D. Performance by dataset size (small/medium/large)
  E. Convergence analysis (normalized best-so-far trajectories)
  E2. Regret analysis (simple regret curves)
  E3. Leave-one-study-out ranking stability
  E4. Study type x strategy family interaction analysis
  F. Timing and Pareto analysis
  F2. AUC metric analysis
  F3. Acceleration factor analysis
  F4. Hit diversity analysis
  G. Per-study heatmap of family-level recall
  H. Caveats and limitations
  I. Statistical deep dive (variance decomposition, pairwise tests)

Usage:
    python -m benchmarks.analyze_within_study
"""

from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats as sp_stats

from .constants import ASSAY_TYPE_LABELS, MIN_N_CORRELATION, MIN_N_WILCOXON, SEEDS
from .stats import (
    acceleration_factor,
    benjamini_hochberg,
    bootstrap_ci,
    cohens_d_paired,
    post_hoc_power,
    rank_biserial,
    simple_regret,
)
from .strategy_registry import STRATEGY_FAMILY as _REG_FAMILY
from .strategy_registry import STRATEGY_SHORT

RESULTS_DIR = Path(__file__).resolve().parent.parent / "benchmark_results" / "within_study"

# This script uses "GP (BoTorch)" labeling for figures (differs from registry's "LNPBO (GP)")
STRATEGY_FAMILY = {
    k: ("GP (BoTorch)" if v == "LNPBO (GP)" else v)
    for k, v in _REG_FAMILY.items()
}


def load_all_results():
    """Load all per-seed benchmark result JSON files from the results directory."""
    from .result_loading import load_benchmark_results
    return load_benchmark_results(RESULTS_DIR)


def extract_strategy_name(filename):
    """Extract the strategy name from a result filename.

    Strips the ``.json`` extension and ``_s<seed>`` suffix to recover the
    bare strategy identifier.

    Args:
        filename: Filename string (e.g., ``"discrete_xgb_ucb_s42.json"``).

    Returns:
        Strategy name string (e.g., ``"discrete_xgb_ucb"``).
    """
    name = filename.replace(".json", "")
    for seed in SEEDS:
        name = name.replace(f"_s{seed}", "")
    return name


def _load_study_metadata():
    """Load study_metadata.csv and return a dict keyed by PMID string."""
    import csv

    meta_path = Path(__file__).resolve().parent.parent / "data" / "study_metadata.csv"
    if not meta_path.exists():
        return {}
    meta_by_pmid = {}
    with meta_path.open() as f:
        for row in csv.DictReader(f):
            pmid = row["study_id"].split(".")[0]
            meta_by_pmid[pmid] = row
    return meta_by_pmid


# Coarsening maps: raw LNPDB values -> manuscript-level categories
_CARGO_COARSE = {
    "FLuc": "reporter_mRNA",
    "RLuc": "reporter_mRNA",
    "GFP": "reporter_mRNA",
    "hEPO": "therapeutic",
    "base_editor": "therapeutic",
    "DNA_barcode": "barcode_library",
    "peptide_barcode": "barcode_library",
}

_MODEL_COARSE = {
    "HeLa": "HeLa",
    "HepG2": "HepG2",
    "A549": "A549",
    "DC2.4": "DC2.4",
    "IGROV1": "other_cell_line",
    "BeWo_b30": "other_cell_line",
    "HEK293T": "other_cell_line",
    "Mouse_B6": "mouse",
    "Mouse_BALBc": "mouse",
    "Mouse_ICR": "mouse",
    "Mouse_CD1": "mouse",
    "Mouse_Ai14": "mouse",
}


def build_tables(results):
    """Organize raw result dicts into lookup tables.

    Builds a study_info dict, a result_map for fast ``(study_id, strategy,
    seed)`` lookup, and sorted lists of unique study IDs and strategy names.
    Enriches study_info with metadata from ``data/study_metadata.csv``
    (assay_type, cargo_type, model_type) so existing result JSONs that
    lack these fields get them at load time. Also adds coarsened versions
    (cargo_class, model_class) for manuscript-level grouping.

    Args:
        results: List of parsed result dicts from ``load_all_results``.

    Returns:
        Tuple of ``(study_info, result_map, pmids, strategies)`` where:
        - ``study_info``: dict mapping study_id to study metadata.
        - ``result_map``: dict mapping ``(study_id, strategy, seed)``
          to the full result dict.
        - ``pmids``: sorted list of unique study_id strings.
        - ``strategies``: sorted list of unique strategy name strings.
    """
    meta_by_pmid = _load_study_metadata()

    # study_info keyed by study_id (str)
    study_info = {}
    # (study_id, strategy, seed) -> result dict
    result_map = {}
    strategies = set()

    for r in results:
        # Use study_id if present, else derive from pmid
        study_id = r.get("study_id", str(int(r["pmid"])))
        strategy = r["strategy"]
        seed = r["seed"]
        si = r["study_info"]

        # Enrich from study_metadata.csv
        # study_id may be "37661193_liver" or "39060305" — extract numeric PMID prefix
        pmid_key = study_id.split("_")[0].split(".")[0]
        if pmid_key in meta_by_pmid:
            m = meta_by_pmid[pmid_key]
            si.setdefault("assay_type", m.get("assay_type", "unknown"))
            si.setdefault("cargo_type", m.get("cargo_type", "unknown"))
            si.setdefault("model_type", m.get("model_type", "unknown"))

        # Coarsened categories for manuscript grouping
        si["cargo_class"] = _CARGO_COARSE.get(si.get("cargo_type", ""), "other")
        si["model_class"] = _MODEL_COARSE.get(si.get("model_type", ""), "other")
        si["in_vivo"] = si.get("assay_type", "").startswith("in_vivo")

        study_info[study_id] = si
        result_map[(study_id, strategy, seed)] = r
        strategies.add(strategy)

    pmids = sorted(study_info.keys())
    strategies = sorted(strategies)
    return study_info, result_map, pmids, strategies


def section(title):
    """Print a top-level section header with surrounding blank lines.

    Args:
        title: Section title string.
    """
    print(f"\n{title}\n")


def subsection(title):
    """Print a subsection header with a leading blank line.

    Args:
        title: Subsection title string.
    """
    print(f"\n{title}")


def print_study_landscape(study_info, pmids):
    """Print Section A: study landscape overview table.

    Lists all qualifying studies with their size, component diversity,
    seed pool size, budget, and study type classification, grouped by
    study type.

    Args:
        study_info: Dict mapping study_id to study metadata.
        pmids: Sorted list of study_id strings.
    """
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

    print(
        f"{'PMID':>10}  {'N':>5}  {'ILs':>5}  {'HLs':>4}  {'CHLs':>4}  {'PEGs':>4}  "
        f"{'Seed N':>6}  {'Budget':>6}  {'Assay Type':<18}  {'Cargo':<10}  {'Cell Type':<12}  {'Study Type'}"
    )
    print("-" * 140)

    for stype in type_order:
        if stype not in by_type:
            continue
        sorted_pmids = sorted(by_type[stype], key=lambda p: study_info[p]["n_formulations"])
        for pmid in sorted_pmids:
            si = study_info[pmid]
            budget = si["n_seed"] + si["n_rounds"] * si["batch_size"]
            assay_label = ASSAY_TYPE_LABELS.get(si.get("assay_type", "unknown"), "Unknown")
            cargo = si.get("cargo_type", "?")
            model = si.get("model_type", "?")
            print(
                f"{pmid:>10}  {si['n_formulations']:>5}  {si['n_unique_il']:>5}  "
                f"{si.get('n_unique_hl', '?'):>4}  {si.get('n_unique_chl', '?'):>4}  "
                f"{si.get('n_unique_peg', '?'):>4}  {si['n_seed']:>6}  {budget:>6}  "
                f"{assay_label:<18}  {cargo:<10}  {model:<12}  "
                f"{type_labels.get(stype, stype)}"
            )
        print()

    subsection("Summary by Study Type")
    for stype in type_order:
        if stype not in by_type:
            continue
        plist = by_type[stype]
        ns = [study_info[p]["n_formulations"] for p in plist]
        print(
            f"  {type_labels[stype]:40s}  count={len(plist):>2}  "
            f"N: {min(ns):>5}-{max(ns):>5}  (median {int(np.median(ns)):>5})"
        )

    subsection("Summary by Assay Type")
    by_assay = defaultdict(list)
    for pmid in pmids:
        at = study_info[pmid].get("assay_type", "unknown")
        by_assay[at].append(pmid)
    for at in sorted(by_assay.keys()):
        plist = by_assay[at]
        ns = [study_info[p]["n_formulations"] for p in plist]
        label = ASSAY_TYPE_LABELS.get(at, at)
        print(
            f"  {label:40s}  count={len(plist):>2}  "
            f"N: {min(ns):>5}-{max(ns):>5}  (median {int(np.median(ns)):>5})"
        )

    subsection("Summary by Cargo Class")
    by_cargo = defaultdict(list)
    for pmid in pmids:
        ct = study_info[pmid].get("cargo_class", "other")
        by_cargo[ct].append(pmid)
    for ct in sorted(by_cargo.keys()):
        plist = by_cargo[ct]
        print(f"  {ct:40s}  count={len(plist):>2}")

    subsection("Summary by Model Class")
    by_model = defaultdict(list)
    for pmid in pmids:
        mc = study_info[pmid].get("model_class", "other")
        by_model[mc].append(pmid)
    for mc in sorted(by_model.keys()):
        plist = by_model[mc]
        print(f"  {mc:40s}  count={len(plist):>2}")


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
    """Print Section B: overall strategy rankings by top-5% recall.

    Computes per-study means (averaged across 5 seeds), then performs
    one-sided paired Wilcoxon signed-rank tests vs random with
    Benjamini-Hochberg FDR correction. Reports Cohen's d, rank-biserial
    correlation, and lift over random. Also produces family-level
    aggregation.

    Args:
        study_info: Dict mapping study_id to study metadata.
        result_map: Dict mapping ``(study_id, strategy, seed)`` to
            result dicts.
        pmids: Sorted list of study_id strings.
        strategies: Sorted list of strategy name strings.

    Returns:
        List of row dicts sorted by mean recall (descending), used by
        downstream analyses.

    References:
        Wilcoxon, F. (1945). "Individual comparisons by ranking methods."
        Benjamini, Y. & Hochberg, Y. (1995). "Controlling the false
        discovery rate."
        Cohen, J. (1988). "Statistical Power Analysis for the Behavioral
        Sciences."
    """
    section("B. OVERALL STRATEGY RANKINGS (Top-5% Recall)")

    print("  Unit of analysis: per-study means (averaged over 5 seeds).")
    print(f"  Independent observations: {len(pmids)} studies.")

    from .stats import prospective_power

    power_dict = prospective_power(len(pmids))
    power_str = ", ".join(f"d={d}: {p:.0%}" for d, p in sorted(power_dict.items()))
    print(f"  Prospective power (alpha=0.05): {power_str}")
    print()

    # Build per-study means: average across seeds first (correct unit of analysis)
    study_random_mean = {}  # pmid -> mean recall across seeds
    for pmid in pmids:
        vals = [get_top5_recall(result_map, pmid, "random", s) for s in SEEDS]
        vals = [v for v in vals if v is not None]
        if vals:
            study_random_mean[pmid] = np.mean(vals)

    study_strategy_mean = {}  # (pmid, strategy) -> mean recall across seeds
    for strategy in strategies:
        for pmid in pmids:
            vals = [get_top5_recall(result_map, pmid, strategy, s) for s in SEEDS]
            vals = [v for v in vals if v is not None]
            if vals:
                study_strategy_mean[(pmid, strategy)] = np.mean(vals)

    # Collect per-strategy statistics using study-level means
    rows = []
    raw_pvals = []
    for strategy in strategies:
        study_vals = []
        study_random_paired = []
        for pmid in pmids:
            sv = study_strategy_mean.get((pmid, strategy))
            rv = study_random_mean.get(pmid)
            if sv is not None and rv is not None:
                study_vals.append(sv)
                study_random_paired.append(rv)

        if len(study_vals) == 0:
            continue

        study_vals = np.array(study_vals)
        study_random_paired = np.array(study_random_paired)
        mean_val = np.mean(study_vals)
        std_val = np.std(study_vals, ddof=1)
        ci_lo, ci_hi = bootstrap_ci(study_vals)
        n = len(study_vals)

        # Paired Wilcoxon vs random at study level
        p_val = None
        d_val = None
        d_interp = None
        r_rb = None
        if strategy != "random" and n >= MIN_N_CORRELATION:
            diffs = study_vals - study_random_paired
            nonzero = diffs[diffs != 0]
            if len(nonzero) >= MIN_N_CORRELATION:
                _stat, p_val = sp_stats.wilcoxon(nonzero, alternative="greater")
            else:
                p_val = 1.0
            d_val, _, _, d_interp = cohens_d_paired(study_vals, study_random_paired)
            r_rb = rank_biserial(study_vals, study_random_paired)

        mean_random = np.mean(study_random_paired) if len(study_random_paired) > 0 else 0

        rows.append(
            {
                "strategy": strategy,
                "short": STRATEGY_SHORT.get(strategy, strategy),
                "family": STRATEGY_FAMILY.get(strategy, "Other"),
                "mean": mean_val,
                "std": std_val,
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
                "n": n,
                "p_val": p_val,
                "d": d_val,
                "d_interp": d_interp,
                "r_rb": r_rb,
                "mean_random": mean_random,
            }
        )
        if p_val is not None:
            raw_pvals.append((strategy, p_val))

    # Apply BH-FDR correction across all strategy-vs-random tests
    if raw_pvals:
        strat_names = [s for s, _ in raw_pvals]
        pvals_arr = np.array([p for _, p in raw_pvals])
        p_adjusted, _rejected = benjamini_hochberg(pvals_arr)
        p_bh_map = dict(zip(strat_names, p_adjusted))
    else:
        p_bh_map = {}

    rows.sort(key=lambda r: r["mean"], reverse=True)

    subsection("Individual Strategy Rankings (study-level pairing, BH-FDR corrected)")
    print(
        f"{'Rank':>4}  {'Strategy':<22}  {'Family':<14}  "
        f"{'Mean':>6}  {'95% CI':>15}  {'N':>3}  "
        f"{'p_raw':>8}  {'p_BH':>8}  {'Sig':>3}  {'d':>6}  {'Lift':>6}"
    )
    print("-" * 120)

    for i, r in enumerate(rows, 1):
        p_raw_str = f"{r['p_val']:.4f}" if r["p_val"] is not None else "   ---"
        p_bh = p_bh_map.get(r["strategy"])
        p_bh_str = f"{p_bh:.4f}" if p_bh is not None else "   ---"
        sig = ""
        if p_bh is not None:
            if p_bh < 0.001:
                sig = "***"
            elif p_bh < 0.01:
                sig = " **"
            elif p_bh < 0.05:
                sig = "  *"
        d_str = f"{r['d']:.2f}" if r["d"] is not None else "  ---"
        lift = r["mean"] / r["mean_random"] if r["mean_random"] > 0 else float("inf")
        lift_str = f"{lift:.2f}x" if r["strategy"] != "random" else "  ---"
        ci_str = f"[{r['ci_lo']:.3f},{r['ci_hi']:.3f}]"
        print(
            f"{i:>4}  {r['short']:<22}  {r['family']:<14}  "
            f"{r['mean']:>6.3f}  {ci_str:>15}  {r['n']:>3}  "
            f"{p_raw_str:>8}  {p_bh_str:>8}  {sig:>3}  {d_str:>6}  {lift_str:>6}"
        )

    # Effect size summary for top strategies
    subsection("Effect Sizes vs Random (Cohen's d, paired, n=studies)")
    print(f"  {'Strategy':<22}  {'d':>6}  {'Interpretation':<14}  {'r_rb':>6}  {'Power':>6}")
    print("  " + "-" * 65)
    for r in rows:
        if r["d"] is not None:
            pwr = post_hoc_power(r["d"], r["n"])
            short = r["short"]
            print(f"  {short:<22}  {r['d']:>6.2f}  {r['d_interp']:<14}  {r['r_rb']:>6.2f}  {pwr:>6.1%}")

    # Family-level aggregation (still at study level)
    subsection("Strategy Family Rankings (study-level aggregation)")
    family_study_vals = defaultdict(lambda: defaultdict(list))
    for strategy in strategies:
        fam = STRATEGY_FAMILY.get(strategy, "Other")
        for pmid in pmids:
            v = study_strategy_mean.get((pmid, strategy))
            if v is not None:
                family_study_vals[fam][pmid].append(v)

    fam_rows = []
    fam_raw_pvals = []
    for fam in sorted(family_study_vals.keys()):
        # Per-study mean across all strategies in family
        study_means = []
        study_random_paired = []
        for pmid in pmids:
            fam_vals = family_study_vals[fam].get(pmid, [])
            rv = study_random_mean.get(pmid)
            if fam_vals and rv is not None:
                study_means.append(np.mean(fam_vals))
                study_random_paired.append(rv)

        if not study_means:
            continue

        study_means = np.array(study_means)
        study_random_paired = np.array(study_random_paired)
        mean_val = np.mean(study_means)
        std_val = np.std(study_means, ddof=1)
        ci_lo, ci_hi = bootstrap_ci(study_means)
        n = len(study_means)

        p_val = None
        d_val = None
        if fam != "Random" and n >= MIN_N_CORRELATION:
            diffs = study_means - study_random_paired
            nonzero = diffs[diffs != 0]
            if len(nonzero) >= MIN_N_CORRELATION:
                _, p_val = sp_stats.wilcoxon(nonzero, alternative="greater")
            else:
                p_val = 1.0
            d_val, _, _, _ = cohens_d_paired(study_means, study_random_paired)
            fam_raw_pvals.append((fam, p_val))

        mean_random = np.mean(study_random_paired) if len(study_random_paired) > 0 else 0
        fam_rows.append((fam, mean_val, std_val, ci_lo, ci_hi, n, p_val, d_val, mean_random))

    # BH-FDR for family-level
    if fam_raw_pvals:
        fam_names = [f for f, _ in fam_raw_pvals]
        fam_pvals = np.array([p for _, p in fam_raw_pvals])
        fam_p_adj, _ = benjamini_hochberg(fam_pvals)
        fam_p_bh_map = dict(zip(fam_names, fam_p_adj))
    else:
        fam_p_bh_map = {}

    fam_rows.sort(key=lambda x: x[1], reverse=True)

    print(f"{'Family':<16}  {'Mean':>6}  {'95% CI':>15}  {'N':>3}  {'p_BH':>8}  {'Sig':>3}  {'d':>6}  {'Lift':>6}")
    print("-" * 85)
    for fam, mean_val, _std_val, ci_lo, ci_hi, n, _p_val, d_val, mean_random in fam_rows:
        p_bh = fam_p_bh_map.get(fam)
        p_bh_str = f"{p_bh:.4f}" if p_bh is not None else "   ---"
        sig = ""
        if p_bh is not None:
            if p_bh < 0.001:
                sig = "***"
            elif p_bh < 0.01:
                sig = " **"
            elif p_bh < 0.05:
                sig = "  *"
        d_str = f"{d_val:.2f}" if d_val is not None else "  ---"
        lift = mean_val / mean_random if mean_random > 0 and fam != "Random" else 0
        lift_str = f"{lift:.2f}x" if fam != "Random" else "  ---"
        ci_str = f"[{ci_lo:.3f},{ci_hi:.3f}]"
        print(f"{fam:<16}  {mean_val:>6.3f}  {ci_str:>15}  {n:>3}  {p_bh_str:>8}  {sig:>3}  {d_str:>6}  {lift_str:>6}")

    return rows


def print_performance_by_study_type(study_info, result_map, pmids, strategies):
    """Print Section C: strategy performance broken down by study type.

    Groups studies into IL-diverse fixed-ratio, IL-diverse variable-ratio,
    and ratio-only categories, then reports per-strategy and per-family
    recall means and lift within each group.

    Args:
        study_info: Dict mapping study_id to study metadata.
        result_map: Dict mapping ``(study_id, strategy, seed)`` to
            result dicts.
        pmids: Sorted list of study_id strings.
        strategies: Sorted list of strategy name strings.
    """
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

        # Study-level random mean
        random_study_means = []
        for pmid in type_pmids:
            vals = [get_top5_recall(result_map, pmid, "random", s) for s in SEEDS]
            vals = [v for v in vals if v is not None]
            if vals:
                random_study_means.append(np.mean(vals))
        mean_random = np.mean(random_study_means) if random_study_means else 0

        # Per-strategy: study-level means
        strat_study_means = {}
        for strategy in strategies:
            study_vals = []
            for pmid in type_pmids:
                vals = [get_top5_recall(result_map, pmid, strategy, s) for s in SEEDS]
                vals = [v for v in vals if v is not None]
                if vals:
                    study_vals.append(np.mean(vals))
            if study_vals:
                strat_study_means[strategy] = np.array(study_vals)

        # Print strategy table for this type
        sorted_strats = sorted(strat_study_means.items(), key=lambda x: np.mean(x[1]), reverse=True)
        print(f"  {'Strategy':<22}  {'Family':<14}  {'Mean':>6}  {'Std':>6}  {'N_stud':>6}  {'Lift':>6}")
        print("  " + "-" * 80)
        for strategy, vals in sorted_strats:
            fam = STRATEGY_FAMILY.get(strategy, "Other")
            short = STRATEGY_SHORT.get(strategy, strategy)
            mean_v = np.mean(vals)
            std_v = np.std(vals, ddof=1) if len(vals) > 1 else 0
            lift = mean_v / mean_random if mean_random > 0 and strategy != "random" else 0
            lift_str = f"{lift:.2f}x" if strategy != "random" else "  ---"
            print(f"  {short:<22}  {fam:<14}  {mean_v:>6.3f}  {std_v:>6.3f}  {len(vals):>6}  {lift_str:>6}")

        # Family summary (study-level)
        print()
        print("  Family Summary (study-level means):")
        family_study = defaultdict(lambda: defaultdict(list))
        for strategy in strategies:
            fam = STRATEGY_FAMILY.get(strategy, "Other")
            for pmid in type_pmids:
                vals = [get_top5_recall(result_map, pmid, strategy, s) for s in SEEDS]
                vals = [v for v in vals if v is not None]
                if vals:
                    family_study[fam][pmid].append(np.mean(vals))

        fam_summary = []
        for fam in sorted(family_study.keys()):
            study_means = [np.mean(vs) for vs in family_study[fam].values()]
            if study_means:
                fam_summary.append(
                    (
                        fam,
                        np.mean(study_means),
                        np.std(study_means, ddof=1) if len(study_means) > 1 else 0,
                        len(study_means),
                    )
                )
        fam_summary.sort(key=lambda x: x[1], reverse=True)
        for fam, mean_v, std_v, n in fam_summary:
            lift = mean_v / mean_random if mean_random > 0 and fam != "Random" else 0
            lift_str = f"{lift:.2f}x" if fam != "Random" else "  ---"
            print(f"    {fam:<16}  mean={mean_v:.3f}  std={std_v:.3f}  n={n} studies  lift={lift_str}")

    # Direct comparison: GP (BoTorch) vs best alternative by type (study-level)
    subsection("Key Question: Where does GP (BoTorch) excel?")
    for stype in type_order:
        type_pmids = pmids_by_type.get(stype, [])
        if not type_pmids:
            continue

        # Study-level family means
        fam_study = defaultdict(list)
        random_means = []
        for pmid in type_pmids:
            rand_vals = [get_top5_recall(result_map, pmid, "random", s) for s in SEEDS]
            rand_vals = [v for v in rand_vals if v is not None]
            if rand_vals:
                random_means.append(np.mean(rand_vals))

            for fam in set(STRATEGY_FAMILY.values()):
                if fam == "Random":
                    continue
                vals = []
                for strategy in strategies:
                    if STRATEGY_FAMILY.get(strategy) != fam:
                        continue
                    seed_vals = [get_top5_recall(result_map, pmid, strategy, s) for s in SEEDS]
                    vals.extend([v for v in seed_vals if v is not None])
                if vals:
                    fam_study[fam].append(np.mean(vals))

        mean_random = np.mean(random_means) if random_means else 0
        mean_lnpbo = np.mean(fam_study.get("GP (BoTorch)", [0]))
        best_alt_name = ""
        best_alt_mean = 0
        for fam, vals in fam_study.items():
            if fam == "GP (BoTorch)":
                continue
            m = np.mean(vals)
            if m > best_alt_mean:
                best_alt_mean = m
                best_alt_name = fam

        print(
            f"  {type_labels[stype]:40s}  "
            f"Random={mean_random:.3f}  GP={mean_lnpbo:.3f} "
            f"({mean_lnpbo / mean_random:.2f}x)  "
            f"Best Alt={best_alt_name} {best_alt_mean:.3f} "
            f"({best_alt_mean / mean_random:.2f}x)"
        )


def _group_studies_by_dimension(study_info, pmids, dimension):
    """Group study IDs by a metadata dimension (with backward compat)."""
    groups = defaultdict(list)
    for pmid in pmids:
        val = study_info[pmid].get(dimension, "unknown")
        groups[val].append(pmid)
    return dict(groups)


def _family_means_for_group(result_map, group_pmids, strategies):
    """Compute per-family study-level mean recall for a group of studies."""
    family_study = defaultdict(lambda: defaultdict(list))
    random_study_means = []
    for pmid in group_pmids:
        rand_vals = [get_top5_recall(result_map, pmid, "random", s) for s in SEEDS]
        rand_vals = [v for v in rand_vals if v is not None]
        if rand_vals:
            random_study_means.append(np.mean(rand_vals))
        for strategy in strategies:
            fam = STRATEGY_FAMILY.get(strategy, "Other")
            vals = [get_top5_recall(result_map, pmid, strategy, s) for s in SEEDS]
            vals = [v for v in vals if v is not None]
            if vals:
                family_study[fam][pmid].append(np.mean(vals))
    mean_random = np.mean(random_study_means) if random_study_means else 0
    fam_summary = []
    for fam in sorted(family_study.keys()):
        study_means = [np.mean(vs) for vs in family_study[fam].values()]
        if study_means:
            fam_summary.append((fam, np.mean(study_means), len(study_means)))
    fam_summary.sort(key=lambda x: x[1], reverse=True)
    return fam_summary, mean_random


def print_performance_by_assay_type(study_info, result_map, pmids, strategies):
    """Print Section C2: strategy performance by assay type.

    Groups studies by assay_type (in_vitro_single, in_vitro_barcode,
    in_vivo_liver, in_vivo_other) and reports per-family recall means,
    lift vs random, and paired Wilcoxon tests where n >= MIN_N_CORRELATION.
    """
    section("C2. PERFORMANCE BY ASSAY TYPE")

    groups = _group_studies_by_dimension(study_info, pmids, "assay_type")

    for assay_type in sorted(groups.keys()):
        group_pmids = groups[assay_type]
        label = ASSAY_TYPE_LABELS.get(assay_type, assay_type)
        subsection(f"{label} ({len(group_pmids)} studies)")

        fam_summary, mean_random = _family_means_for_group(result_map, group_pmids, strategies)
        print(f"  Random baseline: {mean_random:.3f}")
        print(f"  {'Family':<16}  {'Mean':>6}  {'N':>3}  {'Lift':>6}")
        print("  " + "-" * 40)
        for fam, mean_v, n in fam_summary:
            lift = mean_v / mean_random if mean_random > 0 and fam != "Random" else 0
            lift_str = f"{lift:.2f}x" if fam != "Random" else "  ---"
            print(f"  {fam:<16}  {mean_v:>6.3f}  {n:>3}  {lift_str:>6}")

    # Cross-assay-type comparison
    subsection("Cross-Assay-Type Family Lift Comparison")
    all_families = sorted(set(STRATEGY_FAMILY.values()) - {"Random"})
    header = f"  {'Assay Type':<22}" + "".join(f"  {f:<14}" for f in all_families)
    print(header)
    print("  " + "-" * (22 + 16 * len(all_families)))
    for assay_type in sorted(groups.keys()):
        label = ASSAY_TYPE_LABELS.get(assay_type, assay_type)
        fam_summary, mean_random = _family_means_for_group(
            result_map, groups[assay_type], strategies
        )
        fam_map = {f: m for f, m, _ in fam_summary}
        row = f"  {label:<22}"
        for fam in all_families:
            m = fam_map.get(fam)
            if m is not None and mean_random > 0:
                row += f"  {m / mean_random:>13.2f}x"
            else:
                row += f"  {'---':>14}"
        print(row)


def print_performance_by_cargo_type(study_info, result_map, pmids, strategies):
    """Print Section C3: strategy performance by cargo class."""
    section("C3. PERFORMANCE BY CARGO CLASS")

    groups = _group_studies_by_dimension(study_info, pmids, "cargo_class")

    for cargo_class in sorted(groups.keys()):
        group_pmids = groups[cargo_class]
        subsection(f"{cargo_class} ({len(group_pmids)} studies)")

        fam_summary, mean_random = _family_means_for_group(result_map, group_pmids, strategies)
        print(f"  Random baseline: {mean_random:.3f}")
        print(f"  {'Family':<16}  {'Mean':>6}  {'N':>3}  {'Lift':>6}")
        print("  " + "-" * 40)
        for fam, mean_v, n in fam_summary:
            lift = mean_v / mean_random if mean_random > 0 and fam != "Random" else 0
            lift_str = f"{lift:.2f}x" if fam != "Random" else "  ---"
            print(f"  {fam:<16}  {mean_v:>6.3f}  {n:>3}  {lift_str:>6}")


def print_performance_by_model_class(study_info, result_map, pmids, strategies):
    """Print Section C3b: strategy performance by model class."""
    section("C3b. PERFORMANCE BY MODEL CLASS")

    groups = _group_studies_by_dimension(study_info, pmids, "model_class")

    for model_class in sorted(groups.keys()):
        group_pmids = groups[model_class]
        subsection(f"{model_class} ({len(group_pmids)} studies)")

        fam_summary, mean_random = _family_means_for_group(result_map, group_pmids, strategies)
        print(f"  Random baseline: {mean_random:.3f}")
        print(f"  {'Family':<16}  {'Mean':>6}  {'N':>3}  {'Lift':>6}")
        print("  " + "-" * 40)
        for fam, mean_v, n in fam_summary:
            lift = mean_v / mean_random if mean_random > 0 and fam != "Random" else 0
            lift_str = f"{lift:.2f}x" if fam != "Random" else "  ---"
            print(f"  {fam:<16}  {mean_v:>6.3f}  {n:>3}  {lift_str:>6}")


def print_performance_by_vivo_vitro(study_info, result_map, pmids, strategies):
    """Print Section C4: in vivo vs in vitro performance split."""
    section("C4. IN VIVO vs IN VITRO")

    groups = _group_studies_by_dimension(study_info, pmids, "in_vivo")
    label_map = {True: "In Vivo", False: "In Vitro"}

    fam_lifts = {}
    for is_vivo in [False, True]:
        group_pmids = groups.get(is_vivo, [])
        label = label_map[is_vivo]
        if not group_pmids:
            continue
        subsection(f"{label} ({len(group_pmids)} studies)")

        fam_summary, mean_random = _family_means_for_group(result_map, group_pmids, strategies)
        fam_lifts[label] = {}
        print(f"  Random baseline: {mean_random:.3f}")
        print(f"  {'Family':<16}  {'Mean':>6}  {'N':>3}  {'Lift':>6}")
        print("  " + "-" * 40)
        for fam, mean_v, n in fam_summary:
            lift = mean_v / mean_random if mean_random > 0 and fam != "Random" else 0
            fam_lifts[label][fam] = lift
            lift_str = f"{lift:.2f}x" if fam != "Random" else "  ---"
            print(f"  {fam:<16}  {mean_v:>6.3f}  {n:>3}  {lift_str:>6}")

    # Does the tree vs GP gap differ between vivo and vitro?
    if "In Vivo" in fam_lifts and "In Vitro" in fam_lifts:
        subsection("Tree vs GP Gap by Setting")
        tree_fams = {"RF", "XGBoost", "NGBoost"}
        for setting, lifts in fam_lifts.items():
            tree_lifts = [v for f, v in lifts.items() if f in tree_fams and v > 0]
            gp_lift = lifts.get("GP (BoTorch)", 0)
            tree_mean = np.mean(tree_lifts) if tree_lifts else 0
            gap = tree_mean - gp_lift
            print(f"  {setting:<12}  Tree mean lift={tree_mean:.2f}x  GP lift={gp_lift:.2f}x  Gap={gap:+.2f}x")


def print_cross_dimension_analysis(study_info, result_map, pmids, strategies):
    """Print Section C5: cross-dimension interaction analysis."""
    section("C5. CROSS-DIMENSION INTERACTIONS")

    # Compute per-study, per-family mean recall
    family_study_recall = defaultdict(dict)
    for pmid in pmids:
        for fam in set(STRATEGY_FAMILY.values()):
            vals = []
            for strategy in strategies:
                if STRATEGY_FAMILY.get(strategy) != fam:
                    continue
                seed_vals = [get_top5_recall(result_map, pmid, strategy, s) for s in SEEDS]
                vals.extend([v for v in seed_vals if v is not None])
            if vals:
                family_study_recall[fam][pmid] = np.mean(vals)

    # For each dimension pair, compute Max-Min lift spread
    dimensions = {
        "assay_type": "Assay Type",
        "cargo_class": "Cargo Class",
        "model_class": "Model Class",
        "in_vivo": "In Vivo/Vitro",
        "study_type": "Study Type",
    }

    subsection("Lift Spread per Dimension (Max - Min family lift across groups)")
    print(f"  {'Dimension':<20}  {'N groups':>8}  {'Max-Min Spread':>15}")
    print("  " + "-" * 50)

    for dim_key, dim_label in dimensions.items():
        groups = _group_studies_by_dimension(study_info, pmids, dim_key)
        group_lifts = []
        for gname, gpmids in groups.items():
            if len(gpmids) < 2:
                continue
            fam_summary, mean_random = _family_means_for_group(result_map, gpmids, strategies)
            best_lift = max(
                (m / mean_random for f, m, _ in fam_summary if f != "Random" and mean_random > 0),
                default=0,
            )
            group_lifts.append(best_lift)
        if len(group_lifts) >= 2:
            spread = max(group_lifts) - min(group_lifts)
            print(f"  {dim_label:<20}  {len(group_lifts):>8}  {spread:>14.2f}x")
        else:
            print(f"  {dim_label:<20}  {len(group_lifts):>8}  {'N/A':>15}")


def print_performance_by_size(study_info, result_map, pmids, strategies):
    """Print Section D: strategy performance broken down by dataset size.

    Bins studies into small (<500), medium (500-1000), and large (>1000)
    categories and reports family-level recall means and lift within
    each bin.

    Args:
        study_info: Dict mapping study_id to study metadata.
        result_map: Dict mapping ``(study_id, strategy, seed)`` to
            result dicts.
        pmids: Sorted list of study_id strings.
        strategies: Sorted list of strategy name strings.
    """
    section("D. PERFORMANCE BY DATASET SIZE")

    size_bins = [
        ("Small (<500)", lambda n: n < 500),
        ("Medium (500-1000)", lambda n: 500 <= n <= 1000),
        ("Large (>1000)", lambda n: n > 1000),
    ]

    for bin_label, predicate in size_bins:
        bin_pmids = [p for p in pmids if predicate(study_info[p]["n_formulations"])]
        if not bin_pmids:
            continue

        ns = [study_info[p]["n_formulations"] for p in bin_pmids]
        subsection(f"{bin_label} ({len(bin_pmids)} studies, N={min(ns)}-{max(ns)})")

        # Study-level random mean
        random_study = []
        for pmid in bin_pmids:
            vals = [get_top5_recall(result_map, pmid, "random", s) for s in SEEDS]
            vals = [v for v in vals if v is not None]
            if vals:
                random_study.append(np.mean(vals))
        mean_random = np.mean(random_study) if random_study else 0

        # Family-level: study-level means
        family_study = defaultdict(lambda: defaultdict(list))
        for strategy in strategies:
            fam = STRATEGY_FAMILY.get(strategy, "Other")
            for pmid in bin_pmids:
                vals = [get_top5_recall(result_map, pmid, strategy, s) for s in SEEDS]
                vals = [v for v in vals if v is not None]
                if vals:
                    family_study[fam][pmid].append(np.mean(vals))

        fam_rows = []
        for fam in sorted(family_study.keys()):
            study_means = [np.mean(vs) for vs in family_study[fam].values()]
            if study_means:
                fam_rows.append(
                    (
                        fam,
                        np.mean(study_means),
                        np.std(study_means, ddof=1) if len(study_means) > 1 else 0,
                        len(study_means),
                    )
                )
        fam_rows.sort(key=lambda x: x[1], reverse=True)

        print(f"  {'Family':<16}  {'Mean':>6}  {'Std':>6}  {'N_stud':>6}  {'Lift':>6}")
        print("  " + "-" * 55)
        for fam, mean_v, std_v, n in fam_rows:
            lift = mean_v / mean_random if mean_random > 0 and fam != "Random" else 0
            lift_str = f"{lift:.2f}x" if fam != "Random" else "  ---"
            print(f"  {fam:<16}  {mean_v:>6.3f}  {std_v:>6.3f}  {n:>6}  {lift_str:>6}")

    # Cross-size comparison: GP vs tree models (study-level)
    subsection("GP (BoTorch) vs Tree Models by Size (study-level)")
    print(
        f"  {'Size Bin':<22}  {'Random':>7}  {'GP':>7}  {'XGBoost':>7}  {'RF':>7}  "
        f"{'GP Lift':>7}  {'XGB Lift':>8}  {'RF Lift':>7}"
    )
    print("  " + "-" * 85)

    for bin_label, predicate in size_bins:
        bin_pmids = [p for p in pmids if predicate(study_info[p]["n_formulations"])]
        if not bin_pmids:
            continue

        def _fam_mean(fam, _pmids=bin_pmids):
            """Compute study-level mean recall for a family across given PMIDs."""
            study_vals = []
            for pmid in _pmids:
                vals = []
                for strategy in strategies:
                    if STRATEGY_FAMILY.get(strategy) != fam:
                        continue
                    for seed in SEEDS:
                        v = get_top5_recall(result_map, pmid, strategy, seed)
                        if v is not None:
                            vals.append(v)
                if vals:
                    study_vals.append(np.mean(vals))
            return np.mean(study_vals) if study_vals else 0

        mr = _fam_mean("Random")
        gp = _fam_mean("GP (BoTorch)")
        xgb = _fam_mean("XGBoost")
        rf = _fam_mean("RF")
        print(
            f"  {bin_label:<22}  {mr:>7.3f}  {gp:>7.3f}  {xgb:>7.3f}  {rf:>7.3f}  "
            f"{gp / mr if mr else 0:>6.2f}x  {xgb / mr if mr else 0:>7.2f}x  "
            f"{rf / mr if mr else 0:>6.2f}x"
        )


def _compute_oracle_best(result_map, pmids, strategies):
    """Compute oracle best value per ``(study_id, seed)`` pair.

    The oracle best is defined as the maximum ``best_so_far`` value
    across all strategies for a given study and seed, providing a
    normalization reference for convergence and regret analyses.

    Args:
        result_map: Dict mapping ``(study_id, strategy, seed)`` to
            result dicts.
        pmids: List of study_id strings.
        strategies: List of strategy name strings.

    Returns:
        Dict mapping ``(study_id, seed)`` to float oracle best value.
    """
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
    return oracle_best


def print_convergence_analysis(study_info, result_map, pmids, strategies):
    """Print Section E: convergence analysis.

    Reports normalized best-so-far trajectories (as fraction of oracle
    best) per family, with bootstrap CIs at key rounds, and identifies
    the round at which 90% and 95% of final performance is reached.

    Args:
        study_info: Dict mapping study_id to study metadata.
        result_map: Dict mapping ``(study_id, strategy, seed)`` to
            result dicts.
        pmids: Sorted list of study_id strings.
        strategies: Sorted list of strategy name strings.
    """
    section("E. CONVERGENCE ANALYSIS")

    subsection("Mean best_so_far by Round (averaged across all studies and seeds)")

    max_len = 16
    oracle_best = _compute_oracle_best(result_map, pmids, strategies)

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

    families_to_show = [
        "Random",
        "GP (BoTorch)",
        "CASMOPolitan",
        "XGBoost",
        "RF",
        "NGBoost",
        "Deep Ensemble",
        "GP (sklearn)",
    ]

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

    # Convergence with bootstrap CI bands (study-level)
    subsection("Convergence Curves with 95% Bootstrap CI (study-level)")
    print("  Per-round fraction of oracle best, averaged within study first,")
    print("  then bootstrap CI across studies (n=studies).")
    print()

    for fam in families_to_show:
        # Build study-level mean trajectories
        study_mean_trajs = {}
        for pmid in pmids:
            traj_sum = np.zeros(max_len)
            traj_count = 0
            for strategy in strategies:
                if STRATEGY_FAMILY.get(strategy, "Other") != fam:
                    continue
                for seed in SEEDS:
                    key = (pmid, strategy, seed)
                    if key not in result_map:
                        continue
                    bsf = result_map[key]["result"].get("best_so_far", [])
                    ob = oracle_best.get((pmid, seed))
                    if ob and ob > 0 and len(bsf) == max_len:
                        traj_sum += np.array([v / ob for v in bsf])
                        traj_count += 1
            if traj_count > 0:
                study_mean_trajs[pmid] = traj_sum / traj_count

        if len(study_mean_trajs) < 3:
            continue

        study_arr = np.array(list(study_mean_trajs.values()))  # shape: (n_studies, max_len)
        print(f"  {fam} (n={len(study_arr)} studies):")
        print(f"    {'Round':>5}  {'Mean':>7}  {'CI_lo':>7}  {'CI_hi':>7}")
        for rnd in [0, 3, 6, 9, 12, 15]:
            if rnd >= max_len:
                continue
            col = study_arr[:, rnd]
            ci_lo, ci_hi = bootstrap_ci(col)
            print(f"    {rnd:>5}  {np.mean(col):>7.4f}  {ci_lo:>7.4f}  {ci_hi:>7.4f}")
        print()

    # Round at which 90% of final performance is reached
    subsection("Rounds to reach 90% of final performance (per family)")
    print(f"  {'Family':<16}  {'Round (90% final)':>17}  {'Round (95% final)':>17}")
    print("  " + "-" * 55)

    for fam in families_to_show:
        trajs = family_norm_trajectories.get(fam, [])
        if not trajs:
            continue
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
    top_strats = [
        "random",
        "lnpbo_logei",
        "lnpbo_lp_logei",
        "lnpbo_ucb",
        "discrete_xgb_ucb",
        "discrete_rf_ts_batch",
        "discrete_ngboost_ucb",
        "casmopolitan_ucb",
    ]

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


def print_regret_analysis(study_info, result_map, pmids, strategies):
    """Print Section E2: simple regret analysis.

    Computes normalized simple regret (oracle_best - best_so_far) / oracle_best
    per family, averaged across study-level means. Reports per-round regret
    tables and final regret with bootstrap CIs and reduction vs random.

    Args:
        study_info: Dict mapping study_id to study metadata.
        result_map: Dict mapping ``(study_id, strategy, seed)`` to
            result dicts.
        pmids: Sorted list of study_id strings.
        strategies: Sorted list of strategy name strings.
    """
    section("E2. REGRET ANALYSIS")

    max_len = 16
    oracle_best = _compute_oracle_best(result_map, pmids, strategies)

    families_to_show = ["Random", "GP (BoTorch)", "CASMOPolitan", "XGBoost", "RF", "NGBoost"]

    subsection("Simple Regret by Round (study-level means, then averaged)")
    print("  Simple regret = oracle_best - best_so_far (lower is better)")
    print()

    # Build study-level mean regret per family
    family_study_regret = defaultdict(dict)  # fam -> {pmid -> mean_regret_curve}
    for fam in families_to_show:
        for pmid in pmids:
            regret_curves = []
            for strategy in strategies:
                if STRATEGY_FAMILY.get(strategy, "Other") != fam:
                    continue
                for seed in SEEDS:
                    key = (pmid, strategy, seed)
                    if key not in result_map:
                        continue
                    bsf = result_map[key]["result"].get("best_so_far", [])
                    ob = oracle_best.get((pmid, seed))
                    if ob is not None and len(bsf) == max_len:
                        regret = simple_regret(bsf, ob)
                        # Normalize by oracle_best to make cross-study comparable
                        if ob > 0:
                            regret_curves.append(regret / ob)
            if regret_curves:
                family_study_regret[fam][pmid] = np.mean(regret_curves, axis=0)

    header = f"  {'Round':>5}"
    for fam in families_to_show:
        header += f"  {fam[:12]:>12}"
    print(header)
    print("  " + "-" * (7 + 14 * len(families_to_show)))

    for rnd in range(max_len):
        row = f"  {rnd:>5}"
        for fam in families_to_show:
            study_vals = family_study_regret.get(fam, {})
            if study_vals:
                vals = [v[rnd] for v in study_vals.values() if rnd < len(v)]
                if vals:
                    row += f"  {np.mean(vals):>12.4f}"
                else:
                    row += f"  {'---':>12}"
            else:
                row += f"  {'---':>12}"
        print(row)

    # Final regret comparison with CI
    subsection("Final Simple Regret (round 15) with 95% CI")
    print(f"  {'Family':<16}  {'Mean':>8}  {'95% CI':>17}  {'vs Random':>10}")
    print("  " + "-" * 60)
    random_final = None
    for fam in families_to_show:
        study_vals = family_study_regret.get(fam, {})
        if not study_vals:
            continue
        final_vals = np.array([v[-1] for v in study_vals.values()])
        mean_r = np.mean(final_vals)
        if fam == "Random":
            random_final = mean_r
        ci_lo, ci_hi = bootstrap_ci(final_vals)
        reduction = ""
        if random_final is not None and fam != "Random" and random_final > 0:
            reduction = f"{(1 - mean_r / random_final):>8.1%}"
        print(f"  {fam:<16}  {mean_r:>8.4f}  [{ci_lo:.4f}, {ci_hi:.4f}]  {reduction:>10}")


def print_loo_stability(study_info, result_map, pmids, strategies):
    """Print Section E3: leave-one-study-out ranking stability analysis.

    Drops each study in turn and checks whether the top-3 family ranking
    changes, providing a measure of how sensitive aggregate rankings are
    to individual studies.

    Args:
        study_info: Dict mapping study_id to study metadata.
        result_map: Dict mapping ``(study_id, strategy, seed)`` to
            result dicts.
        pmids: Sorted list of study_id strings.
        strategies: Sorted list of strategy name strings.
    """
    section("E3. LEAVE-ONE-STUDY-OUT STABILITY")

    print("  Drops each study in turn and checks if the top-3 family ranking changes.")
    print("  High instability indicates that a few studies drive the aggregate result.")
    print()

    # Full ranking (family level, study-level means)
    def _family_ranking(use_pmids):
        """Compute family ranking by mean recall across the given studies."""
        fam_means = {}
        for fam in set(STRATEGY_FAMILY.values()):
            if fam == "Random":
                continue
            study_vals = []
            for pmid in use_pmids:
                vals = []
                for strategy in strategies:
                    if STRATEGY_FAMILY.get(strategy) != fam:
                        continue
                    seed_vals = [get_top5_recall(result_map, pmid, strategy, s) for s in SEEDS]
                    seed_vals = [v for v in seed_vals if v is not None]
                    vals.extend(seed_vals)
                if vals:
                    study_vals.append(np.mean(vals))
            if study_vals:
                fam_means[fam] = np.mean(study_vals)
        return sorted(fam_means.keys(), key=lambda f: fam_means.get(f, 0), reverse=True)

    full_ranking = _family_ranking(pmids)
    top3_full = full_ranking[:3]
    print(f"  Full ranking (top 3): {', '.join(top3_full)}")
    print()

    changed = 0
    print(f"  {'Dropped PMID':>12}  {'N':>5}  {'Top 3 (LOO)':>45}  {'Changed':>8}")
    print("  " + "-" * 80)
    for drop_pmid in pmids:
        loo_pmids = [p for p in pmids if p != drop_pmid]
        loo_ranking = _family_ranking(loo_pmids)
        top3_loo = loo_ranking[:3]
        diff = "  *" if top3_loo != top3_full else ""
        if diff:
            changed += 1
        si = study_info[drop_pmid]
        print(f"  {drop_pmid:>12}  {si['n_formulations']:>5}  {', '.join(top3_loo):>45}{diff}")

    print()
    print(f"  Rankings changed in {changed}/{len(pmids)} LOO folds ({changed / len(pmids):.0%})")
    stability = "high" if changed <= 3 else "moderate" if changed <= 8 else "low"
    print(f"  Ranking stability: {stability}")


def print_interaction_analysis(study_info, result_map, pmids, strategies):
    """Print Section E4: study type x strategy family interaction analysis.

    Reports lift (vs random) for each strategy family within each study
    type (fixed/variable/ratio-only) and each size bin (small/medium/large)
    to identify whether certain families excel in particular settings.

    Args:
        study_info: Dict mapping study_id to study metadata.
        result_map: Dict mapping ``(study_id, strategy, seed)`` to
            result dicts.
        pmids: Sorted list of study_id strings.
        strategies: Sorted list of strategy name strings.
    """
    section("E4. INTERACTION ANALYSIS")

    subsection("Study Type x Strategy Family Interaction")
    print("  Do different strategy families excel on different study types?")
    print()

    type_labels = {
        "il_diverse_fixed_ratios": "Fixed",
        "il_diverse_variable_ratios": "Variable",
        "ratio_only": "Ratio",
    }
    families_to_test = ["GP (BoTorch)", "CASMOPolitan", "XGBoost", "RF", "NGBoost"]

    # Compute lift (vs random) per family per study type, at study level
    print(f"  {'Family':<16}", end="")
    for stype in ["il_diverse_fixed_ratios", "il_diverse_variable_ratios", "ratio_only"]:
        print(f"  {type_labels[stype]:>10}", end="")
    print(f"  {'Max-Min':>8}")
    print("  " + "-" * 65)

    for fam in families_to_test:
        lifts_by_type = {}
        for stype in ["il_diverse_fixed_ratios", "il_diverse_variable_ratios", "ratio_only"]:
            type_pmids = [p for p in pmids if study_info[p]["study_type"] == stype]
            if not type_pmids:
                continue
            fam_study_means = []
            random_study_means = []
            for pmid in type_pmids:
                fam_vals = []
                rand_vals = []
                for strategy in strategies:
                    if STRATEGY_FAMILY.get(strategy) != fam:
                        continue
                    for seed in SEEDS:
                        v = get_top5_recall(result_map, pmid, strategy, seed)
                        if v is not None:
                            fam_vals.append(v)
                for seed in SEEDS:
                    rv = get_top5_recall(result_map, pmid, "random", seed)
                    if rv is not None:
                        rand_vals.append(rv)
                if fam_vals and rand_vals:
                    fam_study_means.append(np.mean(fam_vals))
                    random_study_means.append(np.mean(rand_vals))

            if fam_study_means and random_study_means:
                mean_fam = np.mean(fam_study_means)
                mean_rand = np.mean(random_study_means)
                lifts_by_type[stype] = mean_fam / mean_rand if mean_rand > 0 else 0

        row = f"  {fam:<16}"
        vals = []
        for stype in ["il_diverse_fixed_ratios", "il_diverse_variable_ratios", "ratio_only"]:
            if stype in lifts_by_type:
                row += f"  {lifts_by_type[stype]:>9.2f}x"
                vals.append(lifts_by_type[stype])
            else:
                row += f"  {'N/A':>10}"
        if len(vals) >= 2:
            row += f"  {max(vals) - min(vals):>7.2f}x"
        print(row)

    # Study size interaction
    subsection("Study Size x Strategy Family Interaction")
    print("  Lift by family across study size bins:")
    print()

    size_bins = [
        ("Small (<500)", lambda n: n < 500),
        ("Medium (500-1000)", lambda n: 500 <= n <= 1000),
        ("Large (>1000)", lambda n: n > 1000),
    ]

    print(f"  {'Family':<16}", end="")
    for label, _ in size_bins:
        print(f"  {label:>18}", end="")
    print()
    print("  " + "-" * 72)

    for fam in families_to_test:
        row = f"  {fam:<16}"
        for _, predicate in size_bins:
            bin_pmids = [p for p in pmids if predicate(study_info[p]["n_formulations"])]
            if not bin_pmids:
                row += f"  {'N/A':>18}"
                continue
            fam_study_means = []
            random_study_means = []
            for pmid in bin_pmids:
                fam_vals = []
                rand_vals = []
                for strategy in strategies:
                    if STRATEGY_FAMILY.get(strategy) != fam:
                        continue
                    for seed in SEEDS:
                        v = get_top5_recall(result_map, pmid, strategy, seed)
                        if v is not None:
                            fam_vals.append(v)
                for seed in SEEDS:
                    rv = get_top5_recall(result_map, pmid, "random", seed)
                    if rv is not None:
                        rand_vals.append(rv)
                if fam_vals and rand_vals:
                    fam_study_means.append(np.mean(fam_vals))
                    random_study_means.append(np.mean(rand_vals))
            if fam_study_means:
                lift = np.mean(fam_study_means) / np.mean(random_study_means)
                row += f"  {lift:>17.2f}x"
            else:
                row += f"  {'N/A':>18}"
        print(row)


def print_timing_analysis(result_map, pmids, strategies):
    """Print Section F: wall-clock timing and Pareto frontier analysis.

    Reports per-strategy timing statistics (mean, median, std, min, max),
    identifies Pareto-optimal strategies (best recall for given compute
    budget), and summarizes family-level timing.

    Args:
        result_map: Dict mapping ``(study_id, strategy, seed)`` to
            result dicts.
        pmids: Sorted list of study_id strings.
        strategies: Sorted list of strategy name strings.
    """
    section("F. TIMING AND PARETO ANALYSIS")

    subsection("Wall-clock time per strategy (seconds, mean across all studies and seeds)")

    strat_times = defaultdict(list)
    for (_pmid, strategy, _seed), r in result_map.items():
        elapsed = r["result"].get("elapsed", 0)
        strat_times[strategy].append(elapsed)

    rows = []
    for strategy in sorted(strat_times.keys()):
        times = strat_times[strategy]
        rows.append(
            (
                strategy,
                np.mean(times),
                np.median(times),
                np.std(times, ddof=1),
                np.min(times),
                np.max(times),
                len(times),
            )
        )

    rows.sort(key=lambda x: x[1])

    print(f"  {'Strategy':<22}  {'Mean (s)':>8}  {'Median':>8}  {'Std':>8}  {'Min':>8}  {'Max':>8}  {'N':>4}")
    print("  " + "-" * 85)
    for strategy, mean_t, med_t, std_t, min_t, max_t, n in rows:
        short = STRATEGY_SHORT.get(strategy, strategy)
        print(f"  {short:<22}  {mean_t:>8.1f}  {med_t:>8.1f}  {std_t:>8.1f}  {min_t:>8.1f}  {max_t:>8.1f}  {n:>4}")

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


def print_auc_analysis(study_info, result_map, pmids, strategies):
    """Print Section F2: AUC metric analysis.

    Computes family-level AUC rankings (normalized by study-level random
    AUC) with bootstrap CIs, then compares the AUC ranking to the
    top-5% recall ranking via Spearman rank correlation.

    Args:
        study_info: Dict mapping study_id to study metadata.
        result_map: Dict mapping ``(study_id, strategy, seed)`` to
            result dicts.
        pmids: Sorted list of study_id strings.
        strategies: Sorted list of strategy name strings.
    """
    section("F2. AUC ANALYSIS")

    print("  AUC = area under the best-so-far curve (higher is better).")
    print("  Computed per (study, strategy, seed), then averaged within study.")
    print()

    # Helper to extract AUC
    def _get_auc(pmid, strategy, seed):
        """Extract AUC metric for a (pmid, strategy, seed) triple."""
        key = (pmid, strategy, seed)
        if key not in result_map:
            return None
        try:
            return result_map[key]["result"]["metrics"]["auc"]
        except (KeyError, TypeError):
            return None

    # Build study-level means for AUC (average across seeds first)
    study_random_auc = {}
    for pmid in pmids:
        vals = [_get_auc(pmid, "random", s) for s in SEEDS]
        vals = [v for v in vals if v is not None]
        if vals:
            study_random_auc[pmid] = np.mean(vals)

    study_strategy_auc = {}
    for strategy in strategies:
        for pmid in pmids:
            vals = [_get_auc(pmid, strategy, s) for s in SEEDS]
            vals = [v for v in vals if v is not None]
            if vals:
                study_strategy_auc[(pmid, strategy)] = np.mean(vals)

    # Family-level aggregation (study-level means)
    family_study_auc = defaultdict(lambda: defaultdict(list))
    for strategy in strategies:
        fam = STRATEGY_FAMILY.get(strategy, "Other")
        for pmid in pmids:
            v = study_strategy_auc.get((pmid, strategy))
            if v is not None:
                family_study_auc[fam][pmid].append(v)

    subsection("Family-Level AUC Rankings (study-level aggregation)")

    # Normalize AUC within each study so cross-study aggregation is meaningful.
    # For each study, compute the random AUC as baseline and normalize:
    # normalized_auc = strategy_auc / random_auc (ratio > 1 means better than random).
    fam_rows = []
    for fam in sorted(family_study_auc.keys()):
        study_ratios = []
        for pmid in pmids:
            fam_vals = family_study_auc[fam].get(pmid, [])
            rand_auc = study_random_auc.get(pmid)
            if fam_vals and rand_auc and rand_auc > 0:
                study_ratios.append(np.mean(fam_vals) / rand_auc)
        if not study_ratios:
            continue
        study_ratios = np.array(study_ratios)
        mean_ratio = np.mean(study_ratios)
        ci_lo, ci_hi = bootstrap_ci(study_ratios)
        fam_rows.append((fam, mean_ratio, ci_lo, ci_hi, len(study_ratios)))

    fam_rows.sort(key=lambda x: x[1], reverse=True)

    print(f"  {'Family':<16}  {'AUC Ratio':>9}  {'95% CI':>17}  {'N_stud':>6}")
    print("  " + "-" * 55)
    for fam, mean_r, ci_lo, ci_hi, n in fam_rows:
        ratio_str = f"{mean_r:.3f}" if fam != "Random" else "1.000"
        print(f"  {fam:<16}  {ratio_str:>9}  [{ci_lo:.3f}, {ci_hi:.3f}]  {n:>6}")

    # Compare AUC ranking to top-5% recall ranking
    subsection("AUC vs Top-5% Recall Family Rankings")
    print("  Do AUC and top-5% recall agree on which families are best?")
    print()

    # Build top-5% recall family ranking for comparison
    recall_fam_means = {}
    for fam in sorted(family_study_auc.keys()):
        study_vals = []
        for pmid in pmids:
            fam_recalls = []
            for strategy in strategies:
                if STRATEGY_FAMILY.get(strategy, "Other") != fam:
                    continue
                for seed in SEEDS:
                    v = get_top5_recall(result_map, pmid, strategy, seed)
                    if v is not None:
                        fam_recalls.append(v)
            if fam_recalls:
                study_vals.append(np.mean(fam_recalls))
        if study_vals:
            recall_fam_means[fam] = np.mean(study_vals)

    auc_ranking = [r[0] for r in fam_rows if r[0] != "Random"]
    recall_ranking = sorted(
        [f for f in recall_fam_means if f != "Random"],
        key=lambda f: recall_fam_means[f],
        reverse=True,
    )

    print(f"  {'Rank':>4}  {'AUC Ranking':<18}  {'Recall Ranking':<18}")
    print("  " + "-" * 45)
    max_rank = max(len(auc_ranking), len(recall_ranking))
    for i in range(max_rank):
        auc_fam = auc_ranking[i] if i < len(auc_ranking) else "---"
        rec_fam = recall_ranking[i] if i < len(recall_ranking) else "---"
        match = "  <--" if auc_fam == rec_fam else ""
        print(f"  {i + 1:>4}  {auc_fam:<18}  {rec_fam:<18}{match}")

    # Rank correlation
    if len(auc_ranking) >= 3 and len(recall_ranking) >= 3:
        common = [f for f in auc_ranking if f in recall_ranking]
        if len(common) >= 3:
            auc_ranks = [auc_ranking.index(f) for f in common]
            rec_ranks = [recall_ranking.index(f) for f in common]
            rho, p_rho = sp_stats.spearmanr(auc_ranks, rec_ranks)
            print(f"\n  Spearman rank correlation: rho={rho:.3f}, p={p_rho:.3f}")
            if rho > 0.8:
                print("  AUC and recall rankings are highly concordant.")
            elif rho > 0.5:
                print("  AUC and recall rankings show moderate agreement.")
            else:
                print("  AUC and recall rankings diverge substantially.")


def print_acceleration_analysis(study_info, result_map, pmids, strategies):
    """Print Section F3: acceleration factor analysis.

    Computes how many random evaluations are needed to reach the same
    normalized best-so-far level as each BO strategy family, at multiple
    target recall thresholds (50%, 75%, 90%). Reports family-level
    acceleration factors with bootstrap CIs at 75%.

    Args:
        study_info: Dict mapping study_id to study metadata.
        result_map: Dict mapping ``(study_id, strategy, seed)`` to
            result dicts.
        pmids: Sorted list of study_id strings.
        strategies: Sorted list of strategy name strings.
    """
    section("F3. ACCELERATION FACTOR ANALYSIS")

    print("  AF = n_random_to_reach_target / n_bo_to_reach_target")
    print("  AF > 1 means BO reaches the target faster than random.")
    print("  Computed per (study, seed), then averaged within study.")
    print()

    max_len = 16
    oracle_best = _compute_oracle_best(result_map, pmids, strategies)

    # For acceleration factor, we need recall-like curves indexed by n_evaluated.
    # Use normalized best_so_far / oracle_best as the "recall" proxy.
    # The n_evaluated array has 15 entries (rounds 1-15), best_so_far has 16 (rounds 0-15).
    # We need to build curves as list of (n_evaluated, recall) pairs.
    # Round 0 uses n_seed evaluations (from study_info).

    target_recalls = [0.5, 0.75, 0.9]

    families_to_show = ["GP (BoTorch)", "CASMOPolitan", "XGBoost", "RF", "NGBoost", "Deep Ensemble", "GP (sklearn)"]

    # Compute per-(study, seed) acceleration factors for each family
    # family -> target -> [per-study mean AF values]
    family_af = {fam: {t: [] for t in target_recalls} for fam in families_to_show}

    for pmid in pmids:
        n_seed = study_info[pmid]["n_seed"]

        for seed in SEEDS:
            ob = oracle_best.get((pmid, seed))
            if ob is None or ob <= 0:
                continue

            # Build random curve for this (pmid, seed)
            rkey = (pmid, "random", seed)
            if rkey not in result_map:
                continue
            r_bsf = result_map[rkey]["result"].get("best_so_far", [])
            r_ne = result_map[rkey]["result"].get("n_evaluated", [])
            if len(r_bsf) != max_len or len(r_ne) != max_len - 1:
                continue
            # Build (n_eval, recall) pairs: round 0 has n_seed evals
            random_curve = [(n_seed, r_bsf[0] / ob)]
            for i, ne in enumerate(r_ne):
                random_curve.append((ne, r_bsf[i + 1] / ob))

            # For each BO family, compute AF
            for fam in families_to_show:
                fam_afs_this_seed = {t: [] for t in target_recalls}
                for strategy in strategies:
                    if STRATEGY_FAMILY.get(strategy, "Other") != fam:
                        continue
                    key = (pmid, strategy, seed)
                    if key not in result_map:
                        continue
                    bsf = result_map[key]["result"].get("best_so_far", [])
                    ne = result_map[key]["result"].get("n_evaluated", [])
                    if len(bsf) != max_len or len(ne) != max_len - 1:
                        continue
                    bo_curve = [(n_seed, bsf[0] / ob)]
                    for i, n in enumerate(ne):
                        bo_curve.append((n, bsf[i + 1] / ob))

                    for target in target_recalls:
                        af = acceleration_factor(random_curve, bo_curve, target)
                        if not np.isnan(af) and not np.isinf(af):
                            fam_afs_this_seed[target].append(af)

                # Average across strategies in this family for this (pmid, seed)
                for target in target_recalls:
                    vals = fam_afs_this_seed[target]
                    if vals:
                        family_af[fam][target].append(np.mean(vals))

    # Report family-level AF with study-level aggregation
    subsection("Family-Level Acceleration Factors")
    header = f"  {'Family':<16}"
    for t in target_recalls:
        header += f"  {'AF@' + str(int(t * 100)) + '%':>10}"
    header += f"  {'N obs':>6}"
    print(header)
    print("  " + "-" * (20 + 12 * len(target_recalls) + 8))

    for fam in families_to_show:
        row = f"  {fam:<16}"
        n_obs = 0
        for target in target_recalls:
            vals = family_af[fam][target]
            n_obs = max(n_obs, len(vals))
            if vals:
                mean_af = np.mean(vals)
                row += f"  {mean_af:>10.2f}"
            else:
                row += f"  {'N/A':>10}"
        row += f"  {n_obs:>6}"
        print(row)

    # Bootstrap CIs for the main target (0.75)
    subsection("Acceleration Factor at 75% Oracle with 95% Bootstrap CI")
    print(f"  {'Family':<16}  {'Mean AF':>8}  {'95% CI':>17}  {'N':>5}")
    print("  " + "-" * 55)
    target_75 = 0.75
    af_rows = []
    for fam in families_to_show:
        vals = family_af[fam][target_75]
        if len(vals) >= 3:
            mean_af = np.mean(vals)
            ci_lo, ci_hi = bootstrap_ci(np.array(vals))
            af_rows.append((fam, mean_af, ci_lo, ci_hi, len(vals)))
    af_rows.sort(key=lambda x: x[1], reverse=True)
    for fam, mean_af, ci_lo, ci_hi, n in af_rows:
        print(f"  {fam:<16}  {mean_af:>8.2f}  [{ci_lo:.2f}, {ci_hi:.2f}]  {n:>5}")

    print()
    print("  Interpretation: AF=2.0 means random needs 2x more evaluations to reach")
    print("  the same normalized best-so-far level as the BO strategy.")


def print_hit_diversity(study_info, result_map, pmids, strategies):
    """Print Section F4: hit diversity analysis.

    Examines whether different strategy families converge to the same
    top formulations or discover different ones. Reports pairwise hit
    agreement rates, per-study diversity ratios, and evaluation
    efficiency (recall per 100 evaluations) by family.

    Args:
        study_info: Dict mapping study_id to study metadata.
        result_map: Dict mapping ``(study_id, strategy, seed)`` to
            result dicts.
        pmids: Sorted list of study_id strings.
        strategies: Sorted list of strategy name strings.
    """
    section("F4. HIT DIVERSITY ANALYSIS")

    print("  Do different strategies find the SAME top formulations or different ones?")
    print("  Approximation: compare final_best values across strategies within a study.")
    print("  If strategies converge to the same final_best, they likely found the same hit.")
    print()

    families_to_show = ["GP (BoTorch)", "CASMOPolitan", "XGBoost", "RF", "NGBoost", "Deep Ensemble", "GP (sklearn)"]

    # For each study, collect final_best per family (averaged across seeds)
    # Then check how often families agree on the best formulation.

    # Strategy 1: For each study+seed, check if all families find the same final_best.
    # Two final_best values are "same" if they're within 1e-6 (floating point).
    agreement_counts = defaultdict(int)  # (fam_a, fam_b) -> count of study-seeds where they agree
    comparison_counts = defaultdict(int)  # (fam_a, fam_b) -> total study-seeds compared
    family_final_bests = defaultdict(list)  # fam -> list of (pmid, seed, final_best)

    for pmid in pmids:
        for seed in SEEDS:
            fam_bests = {}
            for fam in families_to_show:
                fb_vals = []
                for strategy in strategies:
                    if STRATEGY_FAMILY.get(strategy, "Other") != fam:
                        continue
                    key = (pmid, strategy, seed)
                    if key not in result_map:
                        continue
                    fb = result_map[key]["result"]["metrics"].get("final_best")
                    if fb is not None:
                        fb_vals.append(fb)
                if fb_vals:
                    # Use the max final_best across strategies in the family
                    fam_bests[fam] = max(fb_vals)
                    family_final_bests[fam].append((pmid, seed, max(fb_vals)))

            # Pairwise agreement
            fam_list = sorted(fam_bests.keys())
            for i, fa in enumerate(fam_list):
                for j, fb in enumerate(fam_list):
                    if i >= j:
                        continue
                    comparison_counts[(fa, fb)] += 1
                    if abs(fam_bests[fa] - fam_bests[fb]) < 1e-6:
                        agreement_counts[(fa, fb)] += 1

    subsection("Pairwise Hit Agreement (fraction of study-seeds with same final_best)")
    short_names = {f: f[:10] for f in families_to_show}
    header = f"  {'':>12}"
    for fam in families_to_show:
        header += f"  {short_names[fam]:>10}"
    print(header)
    print("  " + "-" * (14 + 12 * len(families_to_show)))

    for fa in families_to_show:
        row = f"  {short_names[fa]:>12}"
        for fb in families_to_show:
            if fa == fb:
                row += f"  {'---':>10}"
            else:
                key = (fa, fb) if (fa, fb) in comparison_counts else (fb, fa)
                total = comparison_counts.get(key, 0)
                agree = agreement_counts.get(key, 0)
                if total > 0:
                    row += f"  {agree / total:>10.1%}"
                else:
                    row += f"  {'N/A':>10}"
        print(row)

    # Per-study diversity: how many distinct final_best values are found across all families?
    subsection("Hit Diversity per Study (unique final_best values across families)")
    print(f"  {'PMID':>10}  {'N':>5}  {'Type':>8}  {'Unique Hits':>11}  {'Total Fams':>10}  {'Diversity':>9}")
    print("  " + "-" * 65)

    type_abbrev = {
        "il_diverse_fixed_ratios": "fix",
        "il_diverse_variable_ratios": "var",
        "ratio_only": "ratio",
    }

    diversity_vals = []
    for pmid in sorted(pmids, key=lambda p: study_info[p]["n_formulations"]):
        si = study_info[pmid]
        tabbr = type_abbrev.get(si["study_type"], "?")

        # Collect unique final_best values across all families and seeds
        all_fb = []
        for seed in SEEDS:
            for fam in families_to_show:
                for strategy in strategies:
                    if STRATEGY_FAMILY.get(strategy, "Other") != fam:
                        continue
                    key = (pmid, strategy, seed)
                    if key not in result_map:
                        continue
                    fb = result_map[key]["result"]["metrics"].get("final_best")
                    if fb is not None:
                        all_fb.append(fb)

        if not all_fb:
            continue

        # Count distinct hits (unique final_best values within tolerance)
        unique_fb = []
        for fb in sorted(set(all_fb), reverse=True):
            is_dup = False
            for ufb in unique_fb:
                if abs(fb - ufb) < 1e-6:
                    is_dup = True
                    break
            if not is_dup:
                unique_fb.append(fb)

        n_fams = sum(
            1
            for fam in families_to_show
            if any(
                (pmid, strategy, seed) in result_map
                for strategy in strategies
                if STRATEGY_FAMILY.get(strategy, "Other") == fam
                for seed in SEEDS
            )
        )
        # Diversity = unique hits / total families present
        diversity = len(unique_fb) / n_fams if n_fams > 0 else 0
        diversity_vals.append(diversity)

        print(
            f"  {pmid:>10}  {si['n_formulations']:>5}  {tabbr:>8}  "
            f"{len(unique_fb):>11}  {n_fams:>10}  {diversity:>9.2f}"
        )

    if diversity_vals:
        print()
        mean_div = np.mean(diversity_vals)
        print(f"  Mean diversity ratio: {mean_div:.2f}")
        if mean_div < 1.5:
            print("  Low diversity: strategies mostly converge to the same top formulations.")
        elif mean_div < 3.0:
            print("  Moderate diversity: some strategies find different top hits.")
        else:
            print("  High diversity: strategies explore substantially different regions.")

    # Evaluation efficiency: n_evaluated at final round vs recall
    subsection("Evaluation Efficiency by Family")
    print("  Mean total evaluations and top-5% recall across all studies.")
    print()

    print(f"  {'Family':<16}  {'Mean N_eval':>11}  {'Mean Recall':>11}  {'Recall/100 evals':>16}")
    print("  " + "-" * 62)

    for fam in families_to_show:
        n_evals = []
        recalls = []
        for strategy in strategies:
            if STRATEGY_FAMILY.get(strategy, "Other") != fam:
                continue
            for pmid in pmids:
                for seed in SEEDS:
                    key = (pmid, strategy, seed)
                    if key not in result_map:
                        continue
                    ne = result_map[key]["result"].get("n_evaluated", [])
                    if ne:
                        n_evals.append(ne[-1])
                    recall = get_top5_recall(result_map, pmid, strategy, seed)
                    if recall is not None:
                        recalls.append(recall)

        if n_evals and recalls:
            mean_ne = np.mean(n_evals)
            mean_rec = np.mean(recalls)
            efficiency = mean_rec / (mean_ne / 100)
            print(f"  {fam:<16}  {mean_ne:>11.1f}  {mean_rec:>11.3f}  {efficiency:>16.4f}")


def print_caveats():
    """Print Section H: caveats and limitations.

    Enumerates methodological caveats relevant to interpreting the
    benchmark results, including cross-study comparability, ratio
    diversity limitations, statistical power, seed sensitivity, and
    the distinction between retrospective benchmarking and prospective
    performance.
    """
    section("H. CAVEATS AND LIMITATIONS")

    caveats = [
        (
            "Cross-study comparability",
            "Each study's Experiment_value was z-scored within study before benchmarking. "
            "Top-5% recall is a rank-based metric within each study, making it somewhat "
            "comparable, but the underlying assays, cell types, and readouts differ across "
            "studies. Aggregate statistics weight all studies equally regardless of clinical "
            "relevance or assay quality.",
        ),
        (
            "Limited ratio diversity",
            "20 of 27 studies use fixed molar ratios (il_diverse_fixed_ratios), meaning the "
            "optimization reduces to pure IL structure screening. Only 5 studies have variable "
            "ratios with diverse ILs, and only 2 are ratio-only. Conclusions about continuous "
            "optimization or mixed discrete-continuous settings rest on very few studies.",
        ),
        (
            "Study size heterogeneity",
            "Study sizes span an order of magnitude (248 to 2400 formulations). The seed pool "
            "is 25% of formulations, so smaller studies start with a larger fraction of the "
            "pool already explored. Larger studies inherently have more room for improvement "
            "and dominate aggregate statistics unless weighted.",
        ),
        (
            "Statistical power",
            "With 27 independent studies, the paired Wilcoxon test has limited power to "
            "detect small effects (Cohen's d < 0.3). Post-hoc power analysis is reported "
            "alongside p-values. For comparisons where power < 0.8, absence of significance "
            "should not be interpreted as absence of effect. The n=27 sample also limits "
            "interaction tests (e.g., study_type x strategy), which are descriptive only.",
        ),
        (
            "Seed sensitivity",
            "Only 5 random seeds are used. With 27 studies, this yields 135 (study, seed) "
            "pairs per strategy, but seeds within a study are not independent (same data, "
            "different initial sample). Statistical tests use study-level means (n=27) to "
            "avoid pseudoreplication.",
        ),
        (
            "Within-study benchmark != prospective performance",
            "This is a retrospective pool-based benchmark. The model selects from a finite "
            "candidate pool, not from the full chemical space. In a real self-driving lab, the "
            "candidate pool would need to be enumerated or generated, and synthesis/assay noise "
            "would affect results. The oracle (z-scored Experiment_value) is also noise-free.",
        ),
        (
            "Budget constraints",
            "All strategies use the same budget: 25% seed + 15 rounds of batch 12. This means "
            "total evaluation budget ranges from ~34% (small studies) to ~28% (large studies) "
            "of the full pool. Strategies might rank differently under tighter budgets.",
        ),
        (
            "Molecular encoding",
            "IL-diverse studies use LANTERN PCA features (Morgan FP reduced to 5 PCs). "
            "Ratio-only studies use raw molar ratios. Results are conditional on these encodings; "
            "different molecular representations might change relative strategy rankings.",
        ),
        (
            "No external baseline comparison",
            "All strategies compared here are implemented in LNPBO. We do not compare against "
            "external BO frameworks (e.g., Ax, SMAC, Dragonfly, BOSS) or published LNP "
            "optimization methods. Relative rankings are valid within this benchmark but do not "
            "establish absolute state-of-the-art claims.",
        ),
        (
            "Multiple testing",
            "All strategy-vs-random comparisons are corrected using Benjamini-Hochberg FDR at "
            "alpha=0.05. Pairwise family comparisons are also BH-corrected. However, the "
            "overall experiment design (choosing which analyses to report) introduces "
            "uncorrected multiplicity. Results should be interpreted as exploratory.",
        ),
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
    """Print Section G: per-study top-5% recall by strategy family.

    Displays a table with one row per study and one column per strategy
    family, showing the mean top-5% recall across all strategies in
    that family and all seeds.

    Args:
        study_info: Dict mapping study_id to study metadata.
        result_map: Dict mapping ``(study_id, strategy, seed)`` to
            result dicts.
        pmids: Sorted list of study_id strings.
        strategies: Sorted list of strategy name strings.
    """
    section("G. PER-STUDY TOP-5% RECALL BY STRATEGY FAMILY")

    families_to_show = [
        "Random",
        "GP (BoTorch)",
        "CASMOPolitan",
        "XGBoost",
        "RF",
        "NGBoost",
        "Deep Ensemble",
        "GP (sklearn)",
    ]

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
    """Print Section I: statistical deep dive.

    Performs a balanced Type-II ANOVA decomposition of top-5% recall
    variance into study, strategy, seed, study x strategy interaction,
    and residual components (with omega-squared effect sizes). Also
    reports BH-FDR corrected pairwise Wilcoxon tests between strategy
    families and identifies the best strategy family per study.

    Args:
        study_info: Dict mapping study_id to study metadata.
        result_map: Dict mapping ``(study_id, strategy, seed)`` to
            result dicts.
        pmids: Sorted list of study_id strings.
        strategies: Sorted list of strategy name strings.
    """
    section("I. STATISTICAL DEEP DIVE")

    subsection("Variance Decomposition (Top-5% Recall)")
    print("  Balanced Type-II ANOVA: study x strategy x seed (crossed design)")
    print()

    # Build balanced data array: (pmid, strategy, seed) -> value
    all_data = []
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
    N = len(all_data)
    n_studies = len(pmids)
    n_strats = len(strategies)
    n_seeds = len(SEEDS)

    # Main effects SS (Type-II: each factor adjusted for all others)
    # For balanced data, Type-I = Type-II = Type-III, but we compute properly.

    # Study effect: n_ij per study (strategies * seeds)
    study_means = {}
    for pmid in pmids:
        vals = [d[3] for d in all_data if d[0] == pmid]
        if vals:
            study_means[pmid] = np.mean(vals)
    n_per_study = n_strats * n_seeds
    ss_study = n_per_study * sum((study_means[p] - grand_mean) ** 2 for p in pmids if p in study_means)

    # Strategy effect
    strat_means = {}
    for strategy in strategies:
        vals = [d[3] for d in all_data if d[1] == strategy]
        if vals:
            strat_means[strategy] = np.mean(vals)
    n_per_strat = n_studies * n_seeds
    ss_strategy = n_per_strat * sum((strat_means[s] - grand_mean) ** 2 for s in strategies if s in strat_means)

    # Seed effect
    seed_means = {}
    for seed in SEEDS:
        vals = [d[3] for d in all_data if d[2] == seed]
        if vals:
            seed_means[seed] = np.mean(vals)
    n_per_seed = n_studies * n_strats
    ss_seed = n_per_seed * sum((seed_means[s] - grand_mean) ** 2 for s in SEEDS if s in seed_means)

    # Study x Strategy interaction
    study_strat_means = {}
    for pmid in pmids:
        for strategy in strategies:
            vals = [d[3] for d in all_data if d[0] == pmid and d[1] == strategy]
            if vals:
                study_strat_means[(pmid, strategy)] = np.mean(vals)
    ss_study_strat = n_seeds * sum(
        (
            study_strat_means.get((p, s), grand_mean)
            - study_means.get(p, grand_mean)
            - strat_means.get(s, grand_mean)
            + grand_mean
        )
        ** 2
        for p in pmids
        for s in strategies
        if (p, s) in study_strat_means
    )

    ss_residual = ss_total - ss_study - ss_strategy - ss_seed - ss_study_strat

    # Degrees of freedom
    df_study = n_studies - 1
    df_strategy = n_strats - 1
    df_seed = n_seeds - 1
    df_study_strat = df_study * df_strategy
    df_residual = (
        N - n_studies * n_strats - n_studies * n_seeds - n_strats * n_seeds + n_studies + n_strats + n_seeds - 1
    )
    # More robustly: df_residual = N - 1 - df_study - df_strategy - df_seed - df_study_strat
    df_residual = max(1, N - 1 - df_study - df_strategy - df_seed - df_study_strat)

    # Mean squares
    ms_study = ss_study / df_study if df_study > 0 else 0
    ms_strategy = ss_strategy / df_strategy if df_strategy > 0 else 0
    ms_seed = ss_seed / df_seed if df_seed > 0 else 0
    ms_study_strat = ss_study_strat / df_study_strat if df_study_strat > 0 else 0
    ms_residual = ss_residual / df_residual if df_residual > 0 else 0

    # F-statistics with appropriate error terms:
    #   Study, Seed: tested against MS_Residual (fixed-effects F-test)
    #   Strategy: tested against MS_Study*Strategy (studies are a random effect,
    #     so the interaction is the proper error term for the strategy main effect)
    #   Study*Strategy: tested against MS_Residual
    f_study = ms_study / ms_residual if ms_residual > 0 else np.inf
    f_strategy = ms_strategy / ms_study_strat if ms_study_strat > 0 else np.inf
    f_seed = ms_seed / ms_residual if ms_residual > 0 else np.inf
    f_study_strat = ms_study_strat / ms_residual if ms_residual > 0 else np.inf

    # P-values from the F distribution
    p_study = sp_stats.f.sf(f_study, df_study, df_residual)
    p_strategy = sp_stats.f.sf(f_strategy, df_strategy, df_study_strat)
    p_seed = sp_stats.f.sf(f_seed, df_seed, df_residual)
    p_study_strat = sp_stats.f.sf(f_study_strat, df_study_strat, df_residual)

    # Omega-squared: omega^2 = (SS_effect - df_effect * MS_error) / (SS_total + MS_error)
    # Uses the appropriate MS_error for each factor (same error terms as the F-tests).
    denom_resid = ss_total + ms_residual
    denom_interaction = ss_total + ms_study_strat
    omega2_study = (ss_study - df_study * ms_residual) / denom_resid if denom_resid > 0 else 0
    omega2_strategy = (ss_strategy - df_strategy * ms_study_strat) / denom_interaction if denom_interaction > 0 else 0
    omega2_seed = (ss_seed - df_seed * ms_residual) / denom_resid if denom_resid > 0 else 0
    omega2_study_strat = (ss_study_strat - df_study_strat * ms_residual) / denom_resid if denom_resid > 0 else 0
    # Clamp negative omega-squared to zero (can happen when effect is negligible)
    omega2_study = max(0.0, omega2_study)
    omega2_strategy = max(0.0, omega2_strategy)
    omega2_seed = max(0.0, omega2_seed)
    omega2_study_strat = max(0.0, omega2_study_strat)

    print(f"  Grand mean Top-5% recall: {grand_mean:.4f}")
    print(f"  Total observations: {N}")
    print(f"  Design: {n_studies} studies x {n_strats} strategies x {n_seeds} seeds")
    print()
    print(f"  {'Source':<20}  {'SS':>10}  {'df':>5}  {'MS':>10}  {'% SS':>7}  {'F':>10}  {'p':>10}  {'omega^2':>8}")
    print("  " + "-" * 89)
    components = [
        ("Study", ss_study, df_study, f_study, p_study, omega2_study, f"/ MS_Resid (df2={df_residual})"),
        (
            "Strategy",
            ss_strategy,
            df_strategy,
            f_strategy,
            p_strategy,
            omega2_strategy,
            f"/ MS_SxSt (df2={df_study_strat})",
        ),
        (
            "Seed",
            ss_seed,
            df_seed,
            f_seed,
            p_seed,
            omega2_seed,
            f"/ MS_Resid (df2={df_residual})",
        ),
        (
            "Study x Strategy",
            ss_study_strat,
            df_study_strat,
            f_study_strat,
            p_study_strat,
            omega2_study_strat,
            f"/ MS_Resid (df2={df_residual})",
        ),
        ("Residual", ss_residual, df_residual, None, None, None, ""),
    ]
    for label, ss, df, f_val, p_val, omega2, _note in components:
        pct = 100 * ss / ss_total if ss_total > 0 else 0
        ms = ss / df if df > 0 else 0
        if f_val is not None:
            p_str = f"{p_val:.2e}" if p_val < 0.001 else f"{p_val:.4f}"
            print(
                f"  {label:<20}  {ss:>10.4f}  {df:>5}  {ms:>10.6f}"
                f"  {pct:>6.1f}%  {f_val:>10.2f}  {p_str:>10}  {omega2:>8.4f}"
            )
        else:
            print(f"  {label:<20}  {ss:>10.4f}  {df:>5}  {ms:>10.6f}  {pct:>6.1f}%")

    print()
    print("  F-test error terms:")
    for label, _ss, _df, _f_val, _p_val, _omega2, note in components:
        if note:
            print(f"    F({label}) = MS_{label.replace(' x ', 'x').replace(' ', '')} {note}")

    print()
    print("  Note: Strategy is tested against the Study x Strategy interaction (mixed-model F-test)")
    print("  because studies are a random effect -- the interaction is the appropriate error term.")
    print()
    print("  Omega-squared (omega^2) is a less biased effect-size estimate than eta-squared;")
    print("  it adjusts for degrees of freedom and can be negative (clamped to 0).")
    print()
    print("  The Residual term contains unmodeled Study x Seed and Strategy x Seed interactions")
    print("  that cannot be separated without additional structure (these are confounded with error).")
    print()
    print("  Seeds within studies are not truly independent (same data).")
    print("  The Study x Strategy interaction captures study-specific strategy")
    print("  preferences. A large interaction term means no universal 'best' strategy.")

    # Nested decomposition: partition SS_study into metadata dimensions
    subsection("Metadata Factor Decomposition (nested within Study)")
    print("  Partitioning SS_Study into between-group and within-group components:")
    print()

    for meta_dim, meta_label in [("assay_type", "Assay Type"), ("cargo_class", "Cargo Class"), ("model_class", "Model Class")]:
        study_to_group = {p: study_info[p].get(meta_dim, "unknown") for p in pmids}
        unique_groups = sorted(set(study_to_group.values()))
        n_groups = len(unique_groups)
        if n_groups < 2:
            print(f"  {meta_label}: only 1 group — skipping.")
            continue

        # Group means (mean of study means within group)
        group_study_means = defaultdict(list)
        for p in pmids:
            if p in study_means:
                group_study_means[study_to_group[p]].append(study_means[p])
        group_means = {g: np.mean(vs) for g, vs in group_study_means.items()}

        ss_between = n_per_study * sum(
            len(group_study_means[g]) * (group_means[g] - grand_mean) ** 2
            for g in unique_groups if g in group_means
        )
        ss_within = ss_study - ss_between
        df_between = n_groups - 1
        df_within = df_study - df_between

        omega2_meta = max(0.0, (ss_between - df_between * ms_residual) / denom_resid) if denom_resid > 0 else 0
        pct_of_study = 100 * ss_between / ss_study if ss_study > 0 else 0

        print(f"  {meta_label}:")
        print(f"    Groups: {n_groups} ({', '.join(f'{g} (n={len(group_study_means[g])})' for g in unique_groups)})")
        print(f"    SS_{meta_label} (between): {ss_between:.4f}  ({pct_of_study:.1f}% of SS_Study)")
        print(f"    SS_Study|{meta_label} (within): {ss_within:.4f}")
        print(f"    omega^2({meta_label}): {omega2_meta:.4f}")
        print()

    # Heterogeneity (I-squared) per strategy family
    from .stats import higgins_heterogeneity

    subsection("Heterogeneity per Strategy Family (I-squared)")
    print("  I^2 < 25%: low, 25-75%: moderate, > 75%: high heterogeneity")
    print()
    print(f"  {'Family':<16}  {'I^2':>7}  {'tau^2':>8}  {'Q':>8}  {'Q_p':>8}  {'Interpretation'}")
    print("  " + "-" * 65)

    for fam in sorted(set(STRATEGY_FAMILY.values())):
        study_effects = []
        for pmid in pmids:
            vals = []
            for strategy in strategies:
                if STRATEGY_FAMILY.get(strategy) != fam:
                    continue
                for seed in SEEDS:
                    v = get_top5_recall(result_map, pmid, strategy, seed)
                    if v is not None:
                        vals.append(v)
            if vals:
                study_effects.append(np.mean(vals))

        if len(study_effects) < 2:
            continue
        I2, tau2, Q, Q_p = higgins_heterogeneity(study_effects)
        if I2 < 25:
            interp = "low"
        elif I2 < 75:
            interp = "moderate"
        else:
            interp = "high"
        print(f"  {fam:<16}  {I2:>6.1f}%  {tau2:>8.5f}  {Q:>8.2f}  {Q_p:>8.4f}  {interp}")

    # LOO influence diagnostics
    subsection("Leave-One-Study-Out Influence Diagnostics")

    # Grand mean and SE across studies (family-averaged recall)
    study_grand = {}
    for pmid in pmids:
        all_vals = []
        for strategy in strategies:
            for seed in SEEDS:
                v = get_top5_recall(result_map, pmid, strategy, seed)
                if v is not None:
                    all_vals.append(v)
        if all_vals:
            study_grand[pmid] = np.mean(all_vals)

    all_study_vals = list(study_grand.values())
    if len(all_study_vals) >= 3:
        grand = np.mean(all_study_vals)
        se_grand = np.std(all_study_vals, ddof=1) / np.sqrt(len(all_study_vals))
        threshold = 2.0 / np.sqrt(len(all_study_vals))

        print(f"  Grand mean: {grand:.4f}, SE: {se_grand:.4f}")
        print(f"  Influence threshold: |delta_mean| / SE > {threshold:.2f}")
        print()
        print(f"  {'PMID':>10}  {'Study Mean':>10}  {'delta_mean':>10}  {'Influence':>10}  {'Flag'}")
        print("  " + "-" * 60)

        flagged = []
        for pmid in sorted(pmids, key=lambda p: abs(study_grand.get(p, grand) - grand), reverse=True):
            sm = study_grand.get(pmid)
            if sm is None:
                continue
            loo_vals = [v for p, v in study_grand.items() if p != pmid]
            loo_mean = np.mean(loo_vals)
            delta = grand - loo_mean
            influence = abs(delta) / se_grand if se_grand > 0 else 0
            flag = " ***" if influence > threshold else ""
            if flag:
                flagged.append(pmid)
            print(f"  {pmid:>10}  {sm:>10.4f}  {delta:>+10.4f}  {influence:>10.2f}{flag}")

        if flagged:
            print(f"\n  {len(flagged)} influential study(ies): {', '.join(flagged)}")
        else:
            print("\n  No studies exceed the influence threshold.")

    # Pairwise significance: top families
    subsection("Pairwise Family Comparisons (Wilcoxon, BH-FDR, study-level)")
    print("  Each cell: p_BH for row vs column (one-sided: row > column)")
    print()

    families_to_test = ["NGBoost", "RF", "CASMOPolitan", "XGBoost", "Deep Ensemble", "GP (sklearn)", "GP (BoTorch)"]

    # Study-level family means
    fam_study_means = {}  # (fam, pmid) -> mean
    for fam in families_to_test:
        for pmid in pmids:
            vals = []
            for strategy in strategies:
                if STRATEGY_FAMILY.get(strategy) != fam:
                    continue
                for seed in SEEDS:
                    v = get_top5_recall(result_map, pmid, strategy, seed)
                    if v is not None:
                        vals.append(v)
            if vals:
                fam_study_means[(fam, pmid)] = np.mean(vals)

    # Collect pairwise p-values for BH correction
    pairwise_tests = []
    for i, fam_a in enumerate(families_to_test):
        for j, fam_b in enumerate(families_to_test):
            if i >= j:
                continue
            a_vals = []
            b_vals = []
            for pmid in pmids:
                va = fam_study_means.get((fam_a, pmid))
                vb = fam_study_means.get((fam_b, pmid))
                if va is not None and vb is not None:
                    a_vals.append(va)
                    b_vals.append(vb)
            if len(a_vals) >= 5:
                a_arr = np.array(a_vals)
                b_arr = np.array(b_vals)
                diff = a_arr - b_arr
                nonzero = diff[diff != 0]
                if len(nonzero) >= 5:
                    _, p = sp_stats.wilcoxon(a_arr, b_arr, alternative="two-sided")
                else:
                    p = 1.0
                pairwise_tests.append((fam_a, fam_b, float(p)))

    # BH correct all pairwise tests
    if pairwise_tests:
        raw_ps = np.array([t[2] for t in pairwise_tests])
        adj_ps, _ = benjamini_hochberg(raw_ps)
        p_bh_pairwise = {}
        for k, (fa, fb, _) in enumerate(pairwise_tests):
            p_bh_pairwise[(fa, fb)] = adj_ps[k]
            p_bh_pairwise[(fb, fa)] = adj_ps[k]
    else:
        p_bh_pairwise = {}

    # Print matrix
    short_names = {f: f[:8] for f in families_to_test}
    header = f"  {'':>10}"
    for fam in families_to_test:
        header += f"  {short_names[fam]:>8}"
    print(header)
    print("  " + "-" * (12 + 10 * len(families_to_test)))

    for fam_a in families_to_test:
        row = f"  {short_names[fam_a]:>10}"
        for fam_b in families_to_test:
            if fam_a == fam_b:
                row += f"  {'---':>8}"
            else:
                p = p_bh_pairwise.get((fam_a, fam_b))
                if p is not None:
                    sig = "*" if p < 0.05 else ""
                    row += f"  {p:>7.3f}{sig}"
                else:
                    row += f"  {'N/A':>8}"
        print(row)

    # Per-study: best strategy family
    subsection("Best Strategy Family per Study")
    print(
        f"  {'PMID':>10}  {'N':>5}  {'Type':>8}  {'Best Family':<16}  {'Mean Recall':>11}  {'Random':>7}  {'Lift':>6}"
    )
    print("  " + "-" * 75)

    for pmid in sorted(pmids, key=lambda p: study_info[p]["n_formulations"]):
        si = study_info[pmid]
        tabbr = {"il_diverse_fixed_ratios": "fix", "il_diverse_variable_ratios": "var", "ratio_only": "ratio"}.get(
            si["study_type"], "?"
        )

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
        print(
            f"  {pmid:>10}  {si['n_formulations']:>5}  {tabbr:>8}  {best_fam:<16}  "
            f"{best_mean:>11.3f}  {random_mean:>7.3f}  {lift:>5.2f}x"
        )


def main():
    """CLI entry point for the within-study benchmark analysis.

    Loads all result files, builds lookup tables, and runs all analysis
    sections (A through I) in sequence, printing formatted output to
    stdout.
    """
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
    print(
        f"  Expected total: {len(pmids)} x {len(strategies)} x {len(SEEDS)} = "
        f"{len(pmids) * len(strategies) * len(SEEDS)}"
    )
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
    print_performance_by_assay_type(study_info, result_map, pmids, strategies)
    print_performance_by_cargo_type(study_info, result_map, pmids, strategies)
    print_performance_by_model_class(study_info, result_map, pmids, strategies)
    print_performance_by_vivo_vitro(study_info, result_map, pmids, strategies)
    print_cross_dimension_analysis(study_info, result_map, pmids, strategies)
    print_performance_by_size(study_info, result_map, pmids, strategies)
    print_convergence_analysis(study_info, result_map, pmids, strategies)
    print_regret_analysis(study_info, result_map, pmids, strategies)
    print_loo_stability(study_info, result_map, pmids, strategies)
    print_interaction_analysis(study_info, result_map, pmids, strategies)
    print_timing_analysis(result_map, pmids, strategies)
    print_auc_analysis(study_info, result_map, pmids, strategies)
    print_acceleration_analysis(study_info, result_map, pmids, strategies)
    print_hit_diversity(study_info, result_map, pmids, strategies)
    print_per_study_heatmap(study_info, result_map, pmids, strategies)
    print_statistical_deep_dive(study_info, result_map, pmids, strategies)
    print_caveats()

    print()
    print("=" * 100)
    print("  END OF ANALYSIS")
    print("=" * 100)


if __name__ == "__main__":
    main()
