#!/usr/bin/env python3
"""
Sub-study sensitivity analysis.

Tests whether including sub-studies from multi-cell-line papers affects the
main benchmark conclusions. After the Model_type stratification fix, PMIDs
with multiple cell lines are split into separate studies (e.g. 39060305_HeLa,
39060305_RAW264.7). These sub-studies share the same paper but are
independently z-scored and have non-overlapping LNP_IDs.

Sub-studies:
  - 39060305_HeLa, 39060305_RAW264.7  (from PMID 39060305)
  - 38424061_HeLa, 38424061_Mouse_B6  (from PMID 38424061)
  - 37661193_liver, 37661193_spleen    (from PMID 37661193, Model_target split)

Compares the full set (all studies) against a reduced set that drops both the
sub-studies and their parent IDs (if present as separate results).

Additionally computes cluster-robust standard errors treating papers as clusters,
grouping sub-studies under their parent PMID.

Reference for cluster-robust SEs:
    Cameron, A.C. & Miller, D.L. (2015). "A Practitioner's Guide to
    Cluster-Robust Inference." Journal of Human Resources, 50(2), 317-372.
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from LNPBO.runtime_paths import benchmark_results_root, package_root_from

from .stats import bootstrap_ci

_PACKAGE_ROOT = package_root_from(__file__, levels_up=2)
RESULTS_DIR = benchmark_results_root(_PACKAGE_ROOT) / "within_study"
ANALYSIS_DIR = RESULTS_DIR.parent / "analysis" / "within_study" / "sensitivity"

from .constants import SEEDS
from .strategy_registry import STRATEGY_FAMILY as _REG_FAMILY

# Remap "LNPBO (GP)" -> "GP (BoTorch)" for figure labels
STRATEGY_FAMILY = {
    k: ("GP (BoTorch)" if v == "LNPBO (GP)" else v)
    for k, v in _REG_FAMILY.items()
}

# Sub-study IDs from papers with multiple cell lines or organ targets
SUBSTUDY_IDS = {
    # Model_type splits (different cell lines, same paper)
    "39060305_HeLa",
    "39060305_RAW264.7",
    "38424061_HeLa",
    "38424061_Mouse_B6",
    # Model_target splits (different organs, single cell line)
    "37661193_liver",
    "37661193_spleen",
}
PARENT_PMIDS = {"39060305", "38424061", "37661193"}
# Everything to exclude for the reduced set
EXCLUDE_IDS = SUBSTUDY_IDS | PARENT_PMIDS


# Mapping from study_id to parent PMID (cluster label)
# Sub-studies map to parent; standalone studies map to themselves.
def _cluster_label(study_id: str) -> str:
    for parent in PARENT_PMIDS:
        if study_id.startswith(parent):
            return parent
    return study_id


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_all_results():
    results = []
    for pmid_dir in sorted(RESULTS_DIR.iterdir()):
        if not pmid_dir.is_dir():
            continue
        if pmid_dir.name in ("analysis", "gap_analysis", "gibbon_analysis"):
            continue
        for f in sorted(pmid_dir.glob("*.json")):
            try:
                data = json.loads(f.read_text())
                results.append(data)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"WARNING: failed to load {f}: {e}", file=sys.stderr)
    return results


def build_tables(results):
    study_info = {}
    result_map = {}
    strategies = set()

    for r in results:
        study_id = r.get("study_id", str(int(r["pmid"])) if isinstance(r["pmid"], float) else str(r["pmid"]))
        strategy = r["strategy"]
        seed = r["seed"]
        study_info[study_id] = r["study_info"]
        result_map[(study_id, strategy, seed)] = r
        strategies.add(strategy)

    study_ids = sorted(study_info.keys())
    strategies = sorted(strategies)
    return study_info, result_map, study_ids, strategies


def get_top5_recall(result_map, study_id, strategy, seed):
    key = (study_id, strategy, seed)
    if key not in result_map:
        return None
    try:
        return result_map[key]["result"]["metrics"]["top_k_recall"]["5"]
    except (KeyError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Core analysis routines
# ---------------------------------------------------------------------------


def compute_family_stats(result_map, study_ids, strategies):
    """Compute family-level mean top-5% recall with bootstrap 95% CI.

    Returns dict: family -> {mean, ci_lo, ci_hi, std, n_studies, study_means}
    Also returns per-family ranking (list of (rank, family, mean)).
    """
    # Per-study random mean
    study_random = {}
    for sid in study_ids:
        vals = [get_top5_recall(result_map, sid, "random", s) for s in SEEDS]
        vals = [v for v in vals if v is not None]
        if vals:
            study_random[sid] = np.mean(vals)

    # Per-family, per-study mean (average all strategies in family, then seeds)
    family_study_vals = defaultdict(lambda: defaultdict(list))
    for strategy in strategies:
        fam = STRATEGY_FAMILY.get(strategy, "Other")
        for sid in study_ids:
            vals = [get_top5_recall(result_map, sid, strategy, s) for s in SEEDS]
            vals = [v for v in vals if v is not None]
            if vals:
                family_study_vals[fam][sid].append(np.mean(vals))

    family_stats = {}
    for fam in sorted(family_study_vals.keys()):
        study_means = []
        study_random_paired = []
        for sid in study_ids:
            fam_vals = family_study_vals[fam].get(sid, [])
            rv = study_random.get(sid)
            if fam_vals and rv is not None:
                study_means.append(np.mean(fam_vals))
                study_random_paired.append(rv)

        if not study_means:
            continue

        arr = np.array(study_means)
        rand_arr = np.array(study_random_paired)
        mean_val = float(np.mean(arr))
        std_val = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        ci_lo, ci_hi = bootstrap_ci(arr)
        mean_random = float(np.mean(rand_arr)) if len(rand_arr) > 0 else 0.0
        lift = mean_val / mean_random if mean_random > 0 and fam != "Random" else None

        family_stats[fam] = {
            "mean": mean_val,
            "std": std_val,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "n_studies": len(arr),
            "lift": lift,
            "mean_random": mean_random,
            "study_means": arr,
        }

    # Ranking
    ranked = sorted(family_stats.keys(), key=lambda f: family_stats[f]["mean"], reverse=True)
    ranking = [(i + 1, fam, family_stats[fam]["mean"]) for i, fam in enumerate(ranked)]

    return family_stats, ranking


def compute_variance_decomposition(result_map, study_ids, strategies):
    """Compute variance decomposition (SS percentages) for the given study set.

    Returns dict with keys: Study, Strategy, Seed, Interaction, Residual (percentages).
    """
    all_data = []
    for sid in study_ids:
        for strategy in strategies:
            for seed in SEEDS:
                v = get_top5_recall(result_map, sid, strategy, seed)
                if v is not None:
                    all_data.append((sid, strategy, seed, v))

    if not all_data:
        return {}

    values = np.array([d[3] for d in all_data])
    grand_mean = np.mean(values)
    ss_total = np.sum((values - grand_mean) ** 2)
    N = len(all_data)
    n_studies = len(study_ids)
    n_strats = len(strategies)
    n_seeds = len(SEEDS)

    # Study main effect
    study_means = {}
    for sid in study_ids:
        vals = [d[3] for d in all_data if d[0] == sid]
        if vals:
            study_means[sid] = np.mean(vals)
    n_per_study = n_strats * n_seeds
    ss_study = n_per_study * sum((study_means[s] - grand_mean) ** 2 for s in study_ids if s in study_means)

    # Strategy main effect
    strat_means = {}
    for strategy in strategies:
        vals = [d[3] for d in all_data if d[1] == strategy]
        if vals:
            strat_means[strategy] = np.mean(vals)
    n_per_strat = n_studies * n_seeds
    ss_strategy = n_per_strat * sum((strat_means[s] - grand_mean) ** 2 for s in strategies if s in strat_means)

    # Seed main effect
    seed_means = {}
    for seed in SEEDS:
        vals = [d[3] for d in all_data if d[2] == seed]
        if vals:
            seed_means[seed] = np.mean(vals)
    n_per_seed = n_studies * n_strats
    ss_seed = n_per_seed * sum((seed_means[s] - grand_mean) ** 2 for s in SEEDS if s in seed_means)

    # Study x Strategy interaction
    study_strat_means = {}
    for sid in study_ids:
        for strategy in strategies:
            vals = [d[3] for d in all_data if d[0] == sid and d[1] == strategy]
            if vals:
                study_strat_means[(sid, strategy)] = np.mean(vals)
    ss_interaction = n_seeds * sum(
        (
            study_strat_means.get((p, s), grand_mean)
            - study_means.get(p, grand_mean)
            - strat_means.get(s, grand_mean)
            + grand_mean
        )
        ** 2
        for p in study_ids
        for s in strategies
        if (p, s) in study_strat_means
    )

    ss_residual = ss_total - ss_study - ss_strategy - ss_seed - ss_interaction

    result = {
        "Study": 100 * ss_study / ss_total if ss_total > 0 else 0,
        "Strategy": 100 * ss_strategy / ss_total if ss_total > 0 else 0,
        "Seed": 100 * ss_seed / ss_total if ss_total > 0 else 0,
        "Interaction": 100 * ss_interaction / ss_total if ss_total > 0 else 0,
        "Residual": 100 * ss_residual / ss_total if ss_total > 0 else 0,
        "n_observations": N,
        "n_studies": n_studies,
        "n_strategies": n_strats,
        "n_seeds": n_seeds,
    }
    return result


def compute_win_rates(result_map, study_ids, strategies):
    """For each study, find the best non-Random family. Return win counts and rates."""
    wins = defaultdict(int)
    for sid in study_ids:
        fam_means = defaultdict(list)
        for strategy in strategies:
            fam = STRATEGY_FAMILY.get(strategy, "Other")
            if fam == "Random":
                continue
            for seed in SEEDS:
                v = get_top5_recall(result_map, sid, strategy, seed)
                if v is not None:
                    fam_means[fam].append(v)

        best_fam = None
        best_mean = -np.inf
        for fam, vals in fam_means.items():
            m = np.mean(vals)
            if m > best_mean:
                best_mean = m
                best_fam = fam

        if best_fam is not None:
            wins[best_fam] += 1

    n_total = len(study_ids)
    win_rates = {fam: count / n_total for fam, count in wins.items()}
    return dict(wins), win_rates


def compute_cluster_robust_se(result_map, study_ids, strategies):
    """Compute cluster-robust standard errors for family means.

    Clusters are defined by parent PMID: sub-studies are grouped under their
    parent. Standalone studies are each their own cluster.

    Uses the CR0 sandwich estimator:
        V_cluster = (G / (G-1)) * sum_g (e_g' e_g)
    where e_g is the vector of residuals for cluster g, and G is the number
    of clusters.

    Returns dict: family -> {mean, se_naive, se_cluster, n_clusters, clusters_used}
    """
    # Group study_ids into clusters
    clusters = defaultdict(list)
    for sid in study_ids:
        clusters[_cluster_label(sid)].append(sid)

    G = len(clusters)

    # For each family, compute study-level means then cluster-robust SE
    # Per-family per-study means
    family_study_vals = defaultdict(lambda: defaultdict(list))
    for strategy in strategies:
        fam = STRATEGY_FAMILY.get(strategy, "Other")
        for sid in study_ids:
            vals = [get_top5_recall(result_map, sid, strategy, s) for s in SEEDS]
            vals = [v for v in vals if v is not None]
            if vals:
                family_study_vals[fam][sid].append(np.mean(vals))

    result = {}
    for fam in sorted(family_study_vals.keys()):
        # Per-study family mean
        study_means = {}
        for sid in study_ids:
            fam_vals = family_study_vals[fam].get(sid, [])
            if fam_vals:
                study_means[sid] = np.mean(fam_vals)

        if not study_means:
            continue

        all_vals = np.array(list(study_means.values()))
        grand_mean = np.mean(all_vals)
        n = len(all_vals)

        # Naive SE (treating all studies as independent)
        se_naive = float(np.std(all_vals, ddof=1) / np.sqrt(n)) if n > 1 else 0.0

        # Cluster-robust SE
        # e_g = sum of residuals within cluster g
        # V = (G/(G-1)) * (1/n^2) * sum_g(e_g^2)
        if G > 1:
            cluster_resid_sq_sum = 0.0
            for _cluster_id, cluster_sids in clusters.items():
                e_g = sum(study_means[sid] - grand_mean for sid in cluster_sids if sid in study_means)
                cluster_resid_sq_sum += e_g**2

            v_cluster = (G / (G - 1)) * cluster_resid_sq_sum / (n**2)
            se_cluster = float(np.sqrt(v_cluster))
        else:
            se_cluster = se_naive

        result[fam] = {
            "mean": float(grand_mean),
            "se_naive": se_naive,
            "se_cluster": se_cluster,
            "n_studies": n,
            "n_clusters": G,
        }

    return result


# ---------------------------------------------------------------------------
# Printing and comparison
# ---------------------------------------------------------------------------

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


def _rank_map(ranking):
    """ranking: list of (rank, family, mean). Returns family -> rank."""
    return {fam: rank for rank, fam, _mean in ranking}


def print_comparison(
    full_stats,
    full_ranking,
    full_var,
    full_wins,
    full_win_rates,
    red_stats,
    red_ranking,
    red_var,
    red_wins,
    red_win_rates,
    cluster_se,
):
    """Print a side-by-side comparison of full vs reduced results."""

    full_rank_map = _rank_map(full_ranking)
    red_rank_map = _rank_map(red_ranking)

    print()
    print("=" * 110)
    print("  SUB-STUDY SENSITIVITY ANALYSIS")
    print("=" * 110)
    n_full = full_var.get("n_studies", "?")
    n_red = red_var.get("n_studies", "?")
    print(f"  Full set: {n_full} studies (including sub-studies)")
    print(f"  Reduced set: {n_red} studies (excluding sub-studies and parent PMIDs)")
    print(f"  Excluded: {', '.join(sorted(EXCLUDE_IDS))}")
    print()

    # --- Family Rankings Side-by-Side ---
    print("-" * 110)
    print("  FAMILY RANKINGS (Top-5% Recall)")
    print("-" * 110)
    print(
        f"  {'Family':<16}  "
        f"{'Rank':>4} {'Mean':>6} {'95% CI':>15} {'Lift':>6}  |  "
        f"{'Rank':>4} {'Mean':>6} {'95% CI':>15} {'Lift':>6}  "
        f"{'dRank':>5}"
    )
    print(f"  {'':<16}  {'--- Full Set ---':^35}  |  {'--- Reduced Set ---':^35}  {'':>5}")
    print("  " + "-" * 106)

    for fam in FAMILY_ORDER:
        fs = full_stats.get(fam)
        rs = red_stats.get(fam)
        fr = full_rank_map.get(fam, "-")
        rr = red_rank_map.get(fam, "-")

        if fs:
            f_ci = f"[{fs['ci_lo']:.3f},{fs['ci_hi']:.3f}]"
            f_lift = f"{fs['lift']:.2f}x" if fs["lift"] is not None else "  ---"
            f_part = f"{fr:>4} {fs['mean']:>6.3f} {f_ci:>15} {f_lift:>6}"
        else:
            f_part = f"{'':>35}"

        if rs:
            r_ci = f"[{rs['ci_lo']:.3f},{rs['ci_hi']:.3f}]"
            r_lift = f"{rs['lift']:.2f}x" if rs["lift"] is not None else "  ---"
            r_part = f"{rr:>4} {rs['mean']:>6.3f} {r_ci:>15} {r_lift:>6}"
        else:
            r_part = f"{'':>35}"

        if isinstance(fr, int) and isinstance(rr, int):
            delta = rr - fr
            d_str = f"{delta:+d}" if delta != 0 else "0"
        else:
            d_str = "-"

        print(f"  {fam:<16}  {f_part}  |  {r_part}  {d_str:>5}")

    # Check ranking stability
    full_order = [fam for _, fam, _ in full_ranking if fam != "Random"]
    red_order = [fam for _, fam, _ in red_ranking if fam != "Random"]
    rank_changed = full_order != red_order
    top3_changed = full_order[:3] != red_order[:3]

    print()
    print(f"  Full ranking (non-Random):    {' > '.join(full_order)}")
    print(f"  Reduced ranking (non-Random): {' > '.join(red_order)}")
    print(f"  Top-3 ranking changed: {'YES' if top3_changed else 'No'}")
    print(f"  Full ranking changed:  {'YES' if rank_changed else 'No'}")

    # --- Variance Decomposition Side-by-Side ---
    print()
    print("-" * 110)
    print("  VARIANCE DECOMPOSITION (% of SS_total)")
    print("-" * 110)
    print(f"  {'Source':<20}  {'Full Set':>10}  {'Reduced Set':>12}  {'Delta':>8}")
    print("  " + "-" * 55)
    for source in ["Study", "Strategy", "Seed", "Interaction", "Residual"]:
        fv = full_var.get(source, 0)
        rv = red_var.get(source, 0)
        delta = rv - fv
        print(f"  {source:<20}  {fv:>9.1f}%  {rv:>11.1f}%  {delta:>+7.1f}%")

    # --- Win Rates Side-by-Side ---
    print()
    print("-" * 110)
    print("  WIN RATES (fraction of studies where family is best)")
    print("-" * 110)
    all_fams = sorted(
        set(list(full_win_rates.keys()) + list(red_win_rates.keys())),
        key=lambda f: full_win_rates.get(f, 0),
        reverse=True,
    )
    print(f"  {'Family':<16}  {'Wins':>4} {'Rate':>6}  |  {'Wins':>4} {'Rate':>6}  {'dRate':>7}")
    print(f"  {'':<16}  {'Full':^11}  |  {'Reduced':^11}  {'':>7}")
    print("  " + "-" * 56)
    for fam in all_fams:
        fw = full_wins.get(fam, 0)
        fr_rate = full_win_rates.get(fam, 0)
        rw = red_wins.get(fam, 0)
        rr_rate = red_win_rates.get(fam, 0)
        delta = rr_rate - fr_rate
        print(f"  {fam:<16}  {fw:>4} {fr_rate:>5.0%}  |  {rw:>4} {rr_rate:>5.0%}  {delta:>+6.0%}")

    # --- Cluster-Robust SEs ---
    print()
    print("-" * 110)
    print("  CLUSTER-ROBUST STANDARD ERRORS (full set, papers as clusters)")
    print("-" * 110)
    print(
        f"  {'Family':<16}  {'Mean':>6}  {'SE_naive':>9}  {'SE_cluster':>11}  "
        f"{'Ratio':>6}  {'N_stud':>6}  {'N_clust':>7}"
    )
    print("  " + "-" * 70)
    for fam in FAMILY_ORDER:
        cse = cluster_se.get(fam)
        if cse is None:
            continue
        ratio = cse["se_cluster"] / cse["se_naive"] if cse["se_naive"] > 0 else float("nan")
        print(
            f"  {fam:<16}  {cse['mean']:>6.3f}  {cse['se_naive']:>9.4f}  "
            f"{cse['se_cluster']:>11.4f}  {ratio:>6.2f}  "
            f"{cse['n_studies']:>6}  {cse['n_clusters']:>7}"
        )
    print()
    print("  SE_cluster / SE_naive > 1 indicates positive intra-cluster correlation")
    print("  (sub-studies from the same paper tend to have similar outcomes).")
    print("  Ratio near 1.0 means clustering has negligible effect.")

    # --- Conclusion ---
    print()
    print("-" * 110)
    print("  CONCLUSION")
    print("-" * 110)

    # Quantify the max absolute change in family mean
    max_mean_delta = 0.0
    for fam in FAMILY_ORDER:
        fs = full_stats.get(fam)
        rs = red_stats.get(fam)
        if fs and rs:
            d = abs(fs["mean"] - rs["mean"])
            if d > max_mean_delta:
                max_mean_delta = d

    max_var_delta = max(
        abs(full_var.get(s, 0) - red_var.get(s, 0)) for s in ["Study", "Strategy", "Seed", "Interaction", "Residual"]
    )

    print(f"  Max |delta mean|  across families: {max_mean_delta:.4f}")
    print(f"  Max |delta %SS|   across sources:  {max_var_delta:.1f}%")
    print(f"  Top-3 ranking changed: {'YES' if top3_changed else 'No'}")
    if not rank_changed:
        print("  Full ranking unchanged: including sub-studies does not alter conclusions.")
    elif not top3_changed:
        print("  Minor rank swaps outside top-3: main conclusions are robust.")
    else:
        # Check if the swaps involve families whose CIs overlap heavily
        swap_pairs = []
        for i, (ff, rf) in enumerate(zip(full_order, red_order)):
            if ff != rf:
                swap_pairs.append((i + 1, ff, rf))
        if swap_pairs:
            print("  Rank changes:")
            for pos, ff, rf in swap_pairs:
                ff_ci = f"[{full_stats[ff]['ci_lo']:.3f}, {full_stats[ff]['ci_hi']:.3f}]"
                rf_ci = f"[{full_stats[rf]['ci_lo']:.3f}, {full_stats[rf]['ci_hi']:.3f}]"
                print(f"    Position {pos}: {ff} (full) -> {rf} (reduced)")
                print(f"      Full CIs:  {ff} {ff_ci},  {rf} {rf_ci}")
        # Check if all swaps are between families with overlapping CIs
        all_swaps_within_ci = True
        for _, ff, rf in swap_pairs:
            fs_f = full_stats.get(ff)
            fs_r = full_stats.get(rf)
            if fs_f and fs_r:
                overlap = min(fs_f["ci_hi"], fs_r["ci_hi"]) - max(fs_f["ci_lo"], fs_r["ci_lo"])
                if overlap <= 0:
                    all_swaps_within_ci = False
        if all_swaps_within_ci:
            print("  All rank swaps occur between families with overlapping 95% CIs.")
            print("  The differences are not statistically significant; conclusions are robust.")


def build_output_dict(
    full_stats,
    full_ranking,
    full_var,
    full_wins,
    full_win_rates,
    red_stats,
    red_ranking,
    red_var,
    red_wins,
    red_win_rates,
    cluster_se,
):
    """Build JSON-serializable output dict."""

    def _family_stats_serializable(stats):
        out = {}
        for fam, s in stats.items():
            out[fam] = {k: v for k, v in s.items() if k != "study_means"}
        return out

    def _ranking_serializable(ranking):
        return [{"rank": r, "family": f, "mean": round(m, 4)} for r, f, m in ranking]

    def _cluster_se_serializable(cse):
        return {fam: {k: round(v, 6) if isinstance(v, float) else v for k, v in s.items()} for fam, s in cse.items()}

    full_rank_map = _rank_map(full_ranking)
    red_rank_map = _rank_map(red_ranking)
    full_order = [fam for _, fam, _ in full_ranking if fam != "Random"]
    red_order = [fam for _, fam, _ in red_ranking if fam != "Random"]

    return {
        "description": "Sub-study sensitivity analysis: full vs reduced study set",
        "excluded_ids": sorted(EXCLUDE_IDS),
        "full_set": {
            "n_studies": full_var.get("n_studies"),
            "family_stats": _family_stats_serializable(full_stats),
            "family_ranking": _ranking_serializable(full_ranking),
            "variance_decomposition": {k: round(v, 2) if isinstance(v, float) else v for k, v in full_var.items()},
            "win_counts": full_wins,
            "win_rates": {k: round(v, 4) for k, v in full_win_rates.items()},
        },
        "reduced_set": {
            "n_studies": red_var.get("n_studies"),
            "family_stats": _family_stats_serializable(red_stats),
            "family_ranking": _ranking_serializable(red_ranking),
            "variance_decomposition": {k: round(v, 2) if isinstance(v, float) else v for k, v in red_var.items()},
            "win_counts": red_wins,
            "win_rates": {k: round(v, 4) for k, v in red_win_rates.items()},
        },
        "comparison": {
            "top3_ranking_changed": full_order[:3] != red_order[:3],
            "full_ranking_changed": full_order != red_order,
            "max_abs_mean_delta": round(
                max(abs(full_stats[f]["mean"] - red_stats[f]["mean"]) for f in full_stats if f in red_stats), 4
            )
            if full_stats and red_stats
            else None,
            "rank_changes": {
                fam: red_rank_map.get(fam, None) - full_rank_map.get(fam, 0)
                if fam in red_rank_map and fam in full_rank_map
                else None
                for fam in FAMILY_ORDER
            },
        },
        "cluster_robust_se": _cluster_se_serializable(cluster_se),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    results = load_all_results()
    _study_info, result_map, all_study_ids, strategies = build_tables(results)

    print(f"Loaded {len(results)} result files across {len(all_study_ids)} studies.")
    print(f"Strategies: {len(strategies)}")

    # Full set
    full_ids = all_study_ids
    # Reduced set: exclude sub-studies and parent PMIDs
    reduced_ids = [sid for sid in all_study_ids if sid not in EXCLUDE_IDS]

    print(f"Full set: {len(full_ids)} studies")
    print(f"Reduced set: {len(reduced_ids)} studies")
    excluded_found = [sid for sid in all_study_ids if sid in EXCLUDE_IDS]
    print(f"Excluded (found in data): {', '.join(excluded_found)}")

    # Compute for full set
    full_stats, full_ranking = compute_family_stats(result_map, full_ids, strategies)
    full_var = compute_variance_decomposition(result_map, full_ids, strategies)
    full_wins, full_win_rates = compute_win_rates(result_map, full_ids, strategies)

    # Compute for reduced set
    red_stats, red_ranking = compute_family_stats(result_map, reduced_ids, strategies)
    red_var = compute_variance_decomposition(result_map, reduced_ids, strategies)
    red_wins, red_win_rates = compute_win_rates(result_map, reduced_ids, strategies)

    # Cluster-robust SEs on full set
    cluster_se = compute_cluster_robust_se(result_map, full_ids, strategies)

    # Print comparison
    print_comparison(
        full_stats,
        full_ranking,
        full_var,
        full_wins,
        full_win_rates,
        red_stats,
        red_ranking,
        red_var,
        red_wins,
        red_win_rates,
        cluster_se,
    )

    # Save JSON
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    output = build_output_dict(
        full_stats,
        full_ranking,
        full_var,
        full_wins,
        full_win_rates,
        red_stats,
        red_ranking,
        red_var,
        red_wins,
        red_win_rates,
        cluster_se,
    )
    out_path = ANALYSIS_DIR / "substudy_sensitivity.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
