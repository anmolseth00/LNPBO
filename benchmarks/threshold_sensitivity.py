#!/usr/bin/env python3
"""
Threshold sensitivity analysis: do strategy family rankings change
across different top-k% recall thresholds (5%, 10%, 20%)?

Loads within-study benchmark results, computes family-level mean recall
at each threshold, ranks families, and computes Spearman rank correlations
between threshold pairs.

Output saved to benchmark_results/analysis/within_study/sensitivity/threshold_sensitivity.json
"""

import json
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr

from benchmarks.analyze_within_study import (
    RESULTS_DIR,
    SEEDS,
    STRATEGY_FAMILY,
    load_all_results,
)

THRESHOLDS = ["5", "10", "20"]

FAMILY_ORDER = [
    "NGBoost",
    "RF",
    "CASMOPolitan",
    "XGBoost",
    "Deep Ensemble",
    "GP (sklearn)",
    "GP (BoTorch)",
    "Random",
]


def get_top_k_recall(result_map, pmid, strategy, seed, k):
    key = (pmid, strategy, seed)
    if key not in result_map:
        return None
    try:
        return result_map[key]["result"]["metrics"]["top_k_recall"][k]
    except (KeyError, TypeError):
        return None


def build_result_map(results):
    study_info = {}
    result_map = {}
    strategies = set()

    for r in results:
        study_id = r.get("study_id", str(r.get("pmid", "")))
        strategy = r["strategy"]
        seed = r["seed"]
        study_info[study_id] = r["study_info"]
        result_map[(study_id, strategy, seed)] = r
        strategies.add(strategy)

    pmids = sorted(study_info.keys())
    strategies = sorted(strategies)
    return study_info, result_map, pmids, strategies


def compute_family_recalls(result_map, pmids, strategies, thresholds):
    """For each threshold, compute per-family mean recall across all (study, strategy, seed) triples."""
    # family -> threshold -> list of recall values
    family_recalls = defaultdict(lambda: defaultdict(list))

    for pmid in pmids:
        for strategy in strategies:
            family = STRATEGY_FAMILY.get(strategy)
            if family is None:
                continue
            for seed in SEEDS:
                for k in thresholds:
                    val = get_top_k_recall(result_map, pmid, strategy, seed, k)
                    if val is not None:
                        family_recalls[family][k].append(val)

    return family_recalls


def main():
    print("Loading results...")
    results = [r for r in load_all_results() if "strategy" in r]
    print(f"  Loaded {len(results)} result files")

    _, result_map, pmids, strategies = build_result_map(results)
    print(f"  {len(pmids)} studies, {len(strategies)} strategies")

    # Check for top-1% recall
    has_top1 = False
    for _key, r in result_map.items():
        try:
            if "1" in r["result"]["metrics"]["top_k_recall"]:
                has_top1 = True
                break
        except (KeyError, TypeError):
            pass

    if has_top1:
        thresholds = ["1", *THRESHOLDS]
        print("  Top-1% recall data found -- including in analysis")
    else:
        thresholds = THRESHOLDS
        print("  NOTE: Top-1% recall is not stored in result files -- skipping")

    family_recalls = compute_family_recalls(result_map, pmids, strategies, thresholds)

    # Compute family means and SEs for each threshold
    # family_stats[family][k] = {"mean": ..., "se": ..., "n": ...}
    family_stats = {}
    for family in FAMILY_ORDER:
        family_stats[family] = {}
        for k in thresholds:
            vals = family_recalls[family][k]
            if vals:
                family_stats[family][k] = {
                    "mean": float(np.mean(vals)),
                    "se": float(np.std(vals, ddof=1) / np.sqrt(len(vals))),
                    "n": len(vals),
                }
            else:
                family_stats[family][k] = {"mean": None, "se": None, "n": 0}

    # Rank families at each threshold (higher recall = rank 1)
    # family_ranks[k] = {family: rank}
    family_ranks = {}
    for k in thresholds:
        means = []
        for family in FAMILY_ORDER:
            m = family_stats[family][k]["mean"]
            if m is not None:
                means.append((family, m))
        means.sort(key=lambda x: x[1], reverse=True)
        family_ranks[k] = {fam: rank + 1 for rank, (fam, _) in enumerate(means)}

    # Print table
    print()
    header = f"{'Family':<18}"
    for k in thresholds:
        header += f"  {'Top-' + k + '%':>10}  {'Rank':>4}"
    print(header)
    print("-" * len(header))

    for family in FAMILY_ORDER:
        row = f"{family:<18}"
        for k in thresholds:
            m = family_stats[family][k]["mean"]
            r = family_ranks[k].get(family, "-")
            if m is not None:
                row += f"  {m:>10.3f}  {r:>4}"
            else:
                row += f"  {'N/A':>10}  {'-':>4}"
        print(row)

    # Spearman rank correlation between threshold pairs
    print()
    print("Spearman rank correlations between family rankings at different thresholds:")
    print()

    threshold_pairs = []
    for i in range(len(thresholds)):
        for j in range(i + 1, len(thresholds)):
            threshold_pairs.append((thresholds[i], thresholds[j]))

    correlations = {}
    for k1, k2 in threshold_pairs:
        ranks1 = []
        ranks2 = []
        for family in FAMILY_ORDER:
            r1 = family_ranks[k1].get(family)
            r2 = family_ranks[k2].get(family)
            if r1 is not None and r2 is not None:
                ranks1.append(r1)
                ranks2.append(r2)
        if len(ranks1) >= 3:
            rho, pval = spearmanr(ranks1, ranks2)
            correlations[f"top{k1}_vs_top{k2}"] = {
                "rho": float(rho),
                "p_value": float(pval),
            }
            print(f"  Top-{k1}% vs Top-{k2}%:  rho = {rho:.3f}  (p = {pval:.3f})")
        else:
            print(f"  Top-{k1}% vs Top-{k2}%:  insufficient data")

    # Lift over Random at each threshold
    print()
    print("Lift over Random at each threshold:")
    print()
    header = f"{'Family':<18}"
    for k in thresholds:
        header += f"  {'Top-' + k + '%':>10}"
    print(header)
    print("-" * len(header))

    random_means = {}
    for k in thresholds:
        m = family_stats["Random"][k]["mean"]
        random_means[k] = m

    lifts = {}
    for family in FAMILY_ORDER:
        row = f"{family:<18}"
        lifts[family] = {}
        for k in thresholds:
            m = family_stats[family][k]["mean"]
            rm = random_means[k]
            if m is not None and rm is not None and rm > 0:
                lift = m / rm
                lifts[family][k] = float(lift)
                row += f"  {lift:>9.2f}x"
            else:
                lifts[family][k] = None
                row += f"  {'N/A':>10}"
        print(row)

    # Summary
    print()
    print("Summary:")
    all_rhos = [v["rho"] for v in correlations.values()]
    if all_rhos:
        min_rho = min(all_rhos)
        max_rho = max(all_rhos)
        if min_rho > 0.9:
            print(f"  Rankings are highly stable across thresholds (rho range: {min_rho:.3f}-{max_rho:.3f})")
        elif min_rho > 0.7:
            print(f"  Rankings are mostly stable across thresholds (rho range: {min_rho:.3f}-{max_rho:.3f})")
        else:
            print(f"  Rankings show meaningful variation across thresholds (rho range: {min_rho:.3f}-{max_rho:.3f})")

    # Save JSON output
    output = {
        "thresholds": thresholds,
        "family_order": FAMILY_ORDER,
        "family_stats": family_stats,
        "family_ranks": {k: {fam: rank for fam, rank in v.items()} for k, v in family_ranks.items()},
        "spearman_correlations": correlations,
        "lifts_over_random": lifts,
        "top1_available": has_top1,
        "n_studies": len(pmids),
        "n_strategies": len(strategies),
        "n_results": len(results),
    }

    out_dir = RESULTS_DIR.parent / "analysis" / "within_study" / "sensitivity"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "threshold_sensitivity.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
