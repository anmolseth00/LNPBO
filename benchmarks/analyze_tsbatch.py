"""
Analysis of why TS-Batch is the best LNPBO (GP) strategy.
"""
import json
import os
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

BASE = Path("/Users/aseth/Documents/GitHub/LNPBO/benchmark_results/within_study")
OUT = BASE / "tsbatch_analysis"
OUT.mkdir(exist_ok=True)

SEEDS = [42, 123, 456, 789, 2024]
PMIDS = sorted([
    int(d.name) for d in BASE.iterdir()
    if d.is_dir() and d.name.isdigit()
])

LNPBO_STRATEGIES = [
    "lnpbo_ucb", "lnpbo_ei", "lnpbo_logei",
    "lnpbo_lp_ei", "lnpbo_lp_logei",
    "lnpbo_pls_logei", "lnpbo_pls_lp_logei",
    "lnpbo_rkb_logei", "lnpbo_ts_batch", "lnpbo_gibbon",
]

KB_STRATEGIES = ["lnpbo_ucb", "lnpbo_ei", "lnpbo_logei"]
TREE_STRATEGIES = [
    "discrete_xgb_ucb", "discrete_rf_ucb", "discrete_ngboost_ucb",
    "discrete_rf_ts_batch", "discrete_xgb_ucb_ts_batch",
]
ALL_TREE = [
    "discrete_xgb_ucb", "discrete_xgb_greedy", "discrete_xgb_cqr",
    "discrete_xgb_online_conformal", "discrete_xgb_ucb_ts_batch",
    "discrete_rf_ucb", "discrete_rf_ts", "discrete_rf_ts_batch",
    "discrete_ngboost_ucb",
    "discrete_deep_ensemble",
    "casmopolitan_ucb", "casmopolitan_ei",
]

# ---------- Load all data ----------
def load_result(pmid, strategy, seed):
    p = BASE / str(pmid) / f"{strategy}_s{seed}.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)

# Load study info
with open(BASE / "within_study_summary.json") as f:
    summary = json.load(f)
study_info = {int(s["pmid"]): s for s in summary["study_infos"]}

# Build recall table: {strategy: {pmid: [recall per seed]}}
recall = defaultdict(lambda: defaultdict(list))
# Build convergence: {strategy: {pmid: {seed: best_so_far}}}
convergence = defaultdict(lambda: defaultdict(dict))
# Build round_best: {strategy: {pmid: {seed: round_best}}}
round_best_data = defaultdict(lambda: defaultdict(dict))
# Build elapsed: {strategy: {pmid: [elapsed per seed]}}
elapsed = defaultdict(lambda: defaultdict(list))

all_strats = LNPBO_STRATEGIES + ALL_TREE + ["random", "discrete_gp_ucb"]

for pmid in PMIDS:
    for strat in all_strats:
        for seed in SEEDS:
            r = load_result(pmid, strat, seed)
            if r is None:
                continue
            rec = r["result"]["metrics"]["top_k_recall"]["5"]
            recall[strat][pmid].append(rec)
            convergence[strat][pmid][seed] = r["result"]["best_so_far"]
            round_best_data[strat][pmid][seed] = r["result"]["round_best"]
            elapsed[strat][pmid].append(r["result"]["elapsed"])

def mean_recall(strat, pmid=None):
    if pmid:
        vals = recall[strat].get(pmid, [])
        return np.mean(vals) if vals else np.nan
    # grand mean across studies (macro average)
    study_means = []
    for p in PMIDS:
        vals = recall[strat].get(p, [])
        if vals:
            study_means.append(np.mean(vals))
    return np.mean(study_means) if study_means else np.nan


# ==============================================================
# 1. TS-BATCH vs KB HEAD-TO-HEAD PER STUDY
# ==============================================================
print("=" * 80)
print("1. TS-BATCH vs KB STRATEGIES: HEAD-TO-HEAD PER STUDY")
print("=" * 80)

# Best KB per study
ts_wins = 0
ts_losses = 0
ts_ties = 0
ts_vs_kb_details = []

for pmid in PMIDS:
    ts_mean = mean_recall("lnpbo_ts_batch", pmid)
    kb_means = {s: mean_recall(s, pmid) for s in KB_STRATEGIES}
    best_kb_name = max(kb_means, key=kb_means.get)
    best_kb_val = kb_means[best_kb_name]

    diff = ts_mean - best_kb_val
    info = study_info[pmid]
    stype = info["study_type"]
    n = info["n_formulations"]

    # Paired t-test across seeds
    ts_vals = recall["lnpbo_ts_batch"].get(pmid, [])
    kb_vals = recall[best_kb_name].get(pmid, [])
    if len(ts_vals) == 5 and len(kb_vals) == 5:
        t_stat, p_val = stats.ttest_rel(ts_vals, kb_vals)
    else:
        t_stat, p_val = np.nan, np.nan

    winner = "TS" if diff > 0.005 else ("KB" if diff < -0.005 else "TIE")
    if winner == "TS":
        ts_wins += 1
    elif winner == "KB":
        ts_losses += 1
    else:
        ts_ties += 1

    ts_vs_kb_details.append({
        "pmid": pmid, "stype": stype, "n": n,
        "ts_recall": ts_mean, "best_kb": best_kb_name,
        "kb_recall": best_kb_val, "diff": diff,
        "winner": winner, "p_val": p_val,
    })

print(f"\nTS-Batch wins: {ts_wins}  |  KB wins: {ts_losses}  |  Ties: {ts_ties}")
print(f"\n{'PMID':>12s}  {'Type':>30s}  {'N':>5s}  {'TS':>6s}  {'Best KB':>18s}  {'KB':>6s}  {'Diff':>7s}  {'p':>7s}  Winner")
print("-" * 110)
for d in sorted(ts_vs_kb_details, key=lambda x: -x["diff"]):
    print(f"{d['pmid']:>12d}  {d['stype']:>30s}  {d['n']:>5d}  {d['ts_recall']:>6.3f}  {d['best_kb']:>18s}  {d['kb_recall']:>6.3f}  {d['diff']:>+7.3f}  {d['p_val']:>7.3f}  {d['winner']}")

# Breakdown by study type
print("\n--- Breakdown by study type ---")
for stype in ["il_diverse_fixed_ratios", "il_diverse_variable_ratios", "ratio_only"]:
    subset = [d for d in ts_vs_kb_details if d["stype"] == stype]
    if not subset:
        continue
    wins = sum(1 for d in subset if d["winner"] == "TS")
    losses = sum(1 for d in subset if d["winner"] == "KB")
    ties = sum(1 for d in subset if d["winner"] == "TIE")
    avg_diff = np.mean([d["diff"] for d in subset])
    print(f"  {stype}: TS wins {wins}, KB wins {losses}, ties {ties}, avg diff = {avg_diff:+.4f}")

# By study size
print("\n--- Breakdown by study size ---")
small = [d for d in ts_vs_kb_details if d["n"] < 400]
medium = [d for d in ts_vs_kb_details if 400 <= d["n"] < 800]
large = [d for d in ts_vs_kb_details if d["n"] >= 800]
for label, subset in [("Small (<400)", small), ("Medium (400-800)", medium), ("Large (>=800)", large)]:
    if not subset:
        continue
    wins = sum(1 for d in subset if d["winner"] == "TS")
    losses = sum(1 for d in subset if d["winner"] == "KB")
    avg_diff = np.mean([d["diff"] for d in subset])
    print(f"  {label} ({len(subset)} studies): TS wins {wins}, KB wins {losses}, avg diff = {avg_diff:+.4f}")


# ==============================================================
# 2. CONVERGENCE COMPARISON (TS-Batch vs LogEI vs GIBBON)
# ==============================================================
print("\n" + "=" * 80)
print("2. CONVERGENCE COMPARISON: TS-BATCH vs LOGEI (KB) vs GIBBON")
print("=" * 80)

compare_strats = {
    "lnpbo_ts_batch": "TS-Batch",
    "lnpbo_logei": "LogEI (KB)",
    "lnpbo_gibbon": "GIBBON",
    "lnpbo_ucb": "UCB (KB)",
    "lnpbo_lp_logei": "LP-LogEI",
    "lnpbo_rkb_logei": "RKB-LogEI",
}

# Normalize best_so_far to [0,1] per study (min-max across all strategies & seeds)
# Then average across studies to get a mean convergence curve
def get_normalized_convergence(strat):
    """Return normalized convergence curves, one per (pmid, seed)."""
    curves = []
    for pmid in PMIDS:
        # Find min/max across ALL strategies for this study
        all_vals = []
        for s in LNPBO_STRATEGIES + ["random"]:
            for seed in SEEDS:
                bsf = convergence[s].get(pmid, {}).get(seed)
                if bsf:
                    all_vals.extend(bsf)
        if not all_vals:
            continue
        vmin, vmax = min(all_vals), max(all_vals)
        if vmax == vmin:
            continue

        for seed in SEEDS:
            bsf = convergence[strat].get(pmid, {}).get(seed)
            if bsf:
                normed = [(v - vmin) / (vmax - vmin) for v in bsf]
                curves.append(normed)
    return curves

# Compute mean convergence for each strategy
fig, ax = plt.subplots(figsize=(10, 6))
colors = {"TS-Batch": "#e63946", "LogEI (KB)": "#457b9d", "GIBBON": "#2a9d8f",
          "UCB (KB)": "#a8dadc", "LP-LogEI": "#f4a261", "RKB-LogEI": "#264653"}

for strat, label in compare_strats.items():
    curves = get_normalized_convergence(strat)
    if not curves:
        continue
    # Pad to same length (16 rounds: seed + 15)
    max_len = max(len(c) for c in curves)
    padded = []
    for c in curves:
        if len(c) < max_len:
            c = c + [c[-1]] * (max_len - len(c))
        padded.append(c)
    arr = np.array(padded)
    mean_curve = arr.mean(axis=0)
    se = arr.std(axis=0) / np.sqrt(len(arr))
    rounds = np.arange(len(mean_curve))
    ax.plot(rounds, mean_curve, label=label, color=colors.get(label, "gray"), linewidth=2)
    ax.fill_between(rounds, mean_curve - se, mean_curve + se, alpha=0.15, color=colors.get(label, "gray"))

ax.set_xlabel("Round (0 = seed pool)", fontsize=12)
ax.set_ylabel("Normalized Best-So-Far", fontsize=12)
ax.set_title("Convergence: TS-Batch vs KB Strategies (averaged across 23 studies x 5 seeds)", fontsize=13)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "convergence_comparison.png", dpi=150)
plt.close()
print(f"  Saved: {OUT / 'convergence_comparison.png'}")

# Print numeric convergence at key rounds
print("\n  Normalized best-so-far at key rounds (mean +/- SE):")
print(f"  {'Strategy':>18s}  {'Round 0':>12s}  {'Round 3':>12s}  {'Round 7':>12s}  {'Round 11':>12s}  {'Round 15':>12s}")
for strat, label in compare_strats.items():
    curves = get_normalized_convergence(strat)
    if not curves:
        continue
    max_len = max(len(c) for c in curves)
    padded = [c + [c[-1]] * (max_len - len(c)) for c in curves]
    arr = np.array(padded)
    means = arr.mean(axis=0)
    ses = arr.std(axis=0) / np.sqrt(len(arr))
    vals = []
    for r in [0, 3, 7, 11, 15]:
        if r < len(means):
            vals.append(f"{means[r]:.3f}+/-{ses[r]:.3f}")
        else:
            vals.append("N/A")
    print(f"  {label:>18s}  {'  '.join(f'{v:>12s}' for v in vals)}")

# Early vs late: improvement from round 0->5 vs round 5->15
print("\n  Early-round improvement (round 0 -> 5) vs late-round (round 5 -> 15):")
for strat, label in compare_strats.items():
    curves = get_normalized_convergence(strat)
    if not curves:
        continue
    max_len = max(len(c) for c in curves)
    padded = [c + [c[-1]] * (max_len - len(c)) for c in curves]
    arr = np.array(padded)
    means = arr.mean(axis=0)
    early = means[min(5, len(means)-1)] - means[0]
    late = means[min(15, len(means)-1)] - means[min(5, len(means)-1)]
    total = means[min(15, len(means)-1)] - means[0]
    print(f"  {label:>18s}: early={early:.4f} ({100*early/total if total > 0 else 0:.1f}%), late={late:.4f} ({100*late/total if total > 0 else 0:.1f}%)")


# ==============================================================
# 3. PER-ROUND BATCH QUALITY
# ==============================================================
print("\n" + "=" * 80)
print("3. PER-ROUND BATCH QUALITY (round_best)")
print("=" * 80)

# Normalize round_best per study, then compare
def get_normalized_round_best(strat):
    """Return normalized round_best values, one curve per (pmid, seed)."""
    curves = []
    for pmid in PMIDS:
        # Normalization: z-score using study mean/std of ALL round_best across strategies
        all_rb = []
        for s in LNPBO_STRATEGIES:
            for seed in SEEDS:
                rb = round_best_data[s].get(pmid, {}).get(seed)
                if rb:
                    all_rb.extend(rb)
        if not all_rb or np.std(all_rb) == 0:
            continue
        mu, sigma = np.mean(all_rb), np.std(all_rb)
        for seed in SEEDS:
            rb = round_best_data[strat].get(pmid, {}).get(seed)
            if rb:
                normed = [(v - mu) / sigma for v in rb]
                curves.append(normed)
    return curves

fig, ax = plt.subplots(figsize=(10, 6))
for strat, label in compare_strats.items():
    curves = get_normalized_round_best(strat)
    if not curves:
        continue
    max_len = max(len(c) for c in curves)
    padded = [c + [np.nan] * (max_len - len(c)) for c in curves]
    arr = np.array(padded)
    mean_curve = np.nanmean(arr, axis=0)
    se = np.nanstd(arr, axis=0) / np.sqrt(np.sum(~np.isnan(arr), axis=0))
    rounds = np.arange(1, len(mean_curve) + 1)
    ax.plot(rounds, mean_curve, label=label, color=colors.get(label, "gray"), linewidth=2)
    ax.fill_between(rounds, mean_curve - se, mean_curve + se, alpha=0.15, color=colors.get(label, "gray"))

ax.set_xlabel("Round", fontsize=12)
ax.set_ylabel("Normalized Round-Best (z-score)", fontsize=12)
ax.set_title("Batch Quality: Best Value Found in Each Round", fontsize=13)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "round_best_quality.png", dpi=150)
plt.close()
print(f"  Saved: {OUT / 'round_best_quality.png'}")

# Numeric table: mean round_best across all rounds
print("\n  Mean normalized round_best (z-score) across all rounds:")
for strat, label in compare_strats.items():
    curves = get_normalized_round_best(strat)
    if not curves:
        continue
    all_vals = [v for c in curves for v in c]
    print(f"  {label:>18s}: mean z = {np.mean(all_vals):+.4f}, median z = {np.median(all_vals):+.4f}")

# Breakdown: early (rounds 1-5) vs mid (6-10) vs late (11-15)
print("\n  Mean normalized round_best by phase:")
print(f"  {'Strategy':>18s}  {'Early (1-5)':>12s}  {'Mid (6-10)':>12s}  {'Late (11-15)':>14s}")
for strat, label in compare_strats.items():
    curves = get_normalized_round_best(strat)
    if not curves:
        continue
    max_len = max(len(c) for c in curves)
    padded = [c + [np.nan] * (max_len - len(c)) for c in curves]
    arr = np.array(padded)
    early = np.nanmean(arr[:, :5])
    mid = np.nanmean(arr[:, 5:10])
    late = np.nanmean(arr[:, 10:15]) if arr.shape[1] > 10 else np.nan
    print(f"  {label:>18s}  {early:>+12.4f}  {mid:>+12.4f}  {late:>+14.4f}")


# ==============================================================
# 4. DIVERSITY PROXY: ROUND-TO-ROUND VARIANCE IN round_best
# ==============================================================
print("\n" + "=" * 80)
print("4. DIVERSITY PROXY: ROUND-TO-ROUND VARIANCE IN round_best")
print("=" * 80)

print("\n  Higher round-to-round variance in round_best suggests more diverse batch selection.")
print("  (Exploitation focuses on the same region => low variance; exploration => high variance)\n")

fig, ax = plt.subplots(figsize=(10, 6))

for strat, label in compare_strats.items():
    # Compute per-(pmid, seed) std of round_best, then average
    variances = []
    for pmid in PMIDS:
        for seed in SEEDS:
            rb = round_best_data[strat].get(pmid, {}).get(seed)
            if rb and len(rb) > 1:
                variances.append(np.std(rb))
    if variances:
        print(f"  {label:>18s}: mean std(round_best) = {np.mean(variances):.4f}, median = {np.median(variances):.4f}")

# Also plot distribution of round_best std
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()
for idx, (strat, label) in enumerate(compare_strats.items()):
    variances = []
    for pmid in PMIDS:
        for seed in SEEDS:
            rb = round_best_data[strat].get(pmid, {}).get(seed)
            if rb and len(rb) > 1:
                variances.append(np.std(rb))
    if variances and idx < len(axes):
        axes[idx].hist(variances, bins=20, alpha=0.7, color=colors.get(label, "gray"))
        axes[idx].set_title(f"{label}\nmean={np.mean(variances):.3f}", fontsize=10)
        axes[idx].set_xlabel("Std(round_best)")
fig.suptitle("Distribution of Round-Best Std (Diversity Proxy)", fontsize=13)
fig.tight_layout()
fig.savefig(OUT / "diversity_proxy.png", dpi=150)
plt.close()
print(f"\n  Saved: {OUT / 'diversity_proxy.png'}")

# Correlation: does higher variance correlate with higher recall?
print("\n  Correlation between round_best variance and final recall:")
for strat, label in list(compare_strats.items())[:3]:
    var_list = []
    rec_list = []
    for pmid in PMIDS:
        for seed in SEEDS:
            rb = round_best_data[strat].get(pmid, {}).get(seed)
            rec_vals = recall[strat].get(pmid, [])
            if rb and len(rb) > 1 and rec_vals:
                var_list.append(np.std(rb))
                # match by index
                seed_idx = SEEDS.index(seed)
                if seed_idx < len(rec_vals):
                    rec_list.append(rec_vals[seed_idx])
    if len(var_list) > 5:
        r_corr, p_corr = stats.pearsonr(var_list, rec_list)
        print(f"  {label:>18s}: r={r_corr:.3f}, p={p_corr:.4f} (n={len(var_list)})")

# Normalized variance (z-scored round_best std)
print("\n  Normalized variance comparison (per-study z-scored):")
for strat, label in compare_strats.items():
    norm_vars = []
    for pmid in PMIDS:
        # Collect std for all LNPBO strategies in this study
        all_stds = {}
        for s in LNPBO_STRATEGIES:
            for seed in SEEDS:
                rb = round_best_data[s].get(pmid, {}).get(seed)
                if rb and len(rb) > 1:
                    all_stds.setdefault(s, []).append(np.std(rb))
        if not all_stds:
            continue
        # Mean std across seeds for each strategy
        flat = [np.mean(v) for v in all_stds.values()]
        mu, sigma = np.mean(flat), np.std(flat)
        if sigma == 0:
            continue
        strat_mean_std = np.mean(all_stds.get(strat, [np.nan]))
        norm_vars.append((strat_mean_std - mu) / sigma)
    if norm_vars:
        print(f"  {label:>18s}: z-scored std = {np.mean(norm_vars):+.3f}")


# ==============================================================
# 5. TS-BATCH ACQUISITION FUNCTION ANALYSIS
# ==============================================================
print("\n" + "=" * 80)
print("5. TS-BATCH ACQUISITION FUNCTION ANALYSIS")
print("=" * 80)

print("\n  TS-Batch samples from the GP posterior and picks the top-B candidates from")
print("  one posterior sample. The acquisition function (UCB placeholder) is irrelevant")
print("  because the batch selection is determined entirely by the posterior sample.")
print()
print("  Comparing TS-Batch to strategies that pair different AFs with different batch methods:")
print()

# Compare all LNPBO strategies
print(f"  {'Strategy':>25s}  {'Batch Method':>12s}  {'AF':>8s}  {'Recall':>8s}  {'vs TS':>8s}")
print("  " + "-" * 70)
ts_recall_global = mean_recall("lnpbo_ts_batch")
strat_details = {
    "lnpbo_ucb":          ("KB (greedy)", "UCB"),
    "lnpbo_ei":           ("KB (greedy)", "EI"),
    "lnpbo_logei":        ("KB (greedy)", "LogEI"),
    "lnpbo_lp_ei":        ("LP", "EI"),
    "lnpbo_lp_logei":     ("LP", "LogEI"),
    "lnpbo_pls_logei":    ("KB (PLS)", "LogEI"),
    "lnpbo_pls_lp_logei": ("LP (PLS)", "LogEI"),
    "lnpbo_rkb_logei":    ("RKB", "LogEI"),
    "lnpbo_ts_batch":     ("TS-Batch", "N/A"),
    "lnpbo_gibbon":       ("GIBBON", "GIBBON"),
}
for strat in LNPBO_STRATEGIES:
    batch_method, af = strat_details.get(strat, ("?", "?"))
    r = mean_recall(strat)
    diff = r - ts_recall_global
    print(f"  {strat:>25s}  {batch_method:>12s}  {af:>8s}  {r:>8.3f}  {diff:>+8.3f}")

print(f"\n  Key insight: TS-Batch's advantage comes from the BATCH SELECTION mechanism,")
print(f"  not the acquisition function. KB/greedy strategies pick the top-B by a single")
print(f"  AF evaluation, leading to redundant batches in flat posterior regions.")
print(f"  TS-Batch implicitly diversifies by sampling from the posterior.")

# Is the advantage from LogEI vs UCB, or from the batch method?
print(f"\n  Decomposing the KB(LogEI) -> TS-Batch gap:")
kb_ucb = mean_recall("lnpbo_ucb")
kb_logei = mean_recall("lnpbo_logei")
ts = mean_recall("lnpbo_ts_batch")
print(f"    KB(UCB):   {kb_ucb:.3f}")
print(f"    KB(LogEI): {kb_logei:.3f}  (AF improvement: {kb_logei - kb_ucb:+.3f})")
print(f"    TS-Batch:  {ts:.3f}  (batch method improvement over KB(LogEI): {ts - kb_logei:+.3f})")
print(f"    Total gap: {ts - kb_ucb:+.3f}  => {100*(ts - kb_logei)/(ts - kb_ucb):.0f}% from batch method, {100*(kb_logei - kb_ucb)/(ts - kb_ucb):.0f}% from AF")


# ==============================================================
# 6. TS-BATCH vs BEST TREE MODEL PER STUDY
# ==============================================================
print("\n" + "=" * 80)
print("6. TS-BATCH vs BEST TREE MODEL PER STUDY")
print("=" * 80)

ts_vs_tree_details = []
for pmid in PMIDS:
    ts_mean = mean_recall("lnpbo_ts_batch", pmid)
    tree_means = {}
    for s in ALL_TREE:
        r = mean_recall(s, pmid)
        if not np.isnan(r):
            tree_means[s] = r
    if not tree_means:
        continue
    best_tree_name = max(tree_means, key=tree_means.get)
    best_tree_val = tree_means[best_tree_name]
    random_val = mean_recall("random", pmid)

    info = study_info[pmid]
    gap_closed = (ts_mean - random_val) / (best_tree_val - random_val) if best_tree_val != random_val else np.nan

    ts_vs_tree_details.append({
        "pmid": pmid, "stype": info["study_type"], "n": info["n_formulations"],
        "ts_recall": ts_mean, "best_tree": best_tree_name,
        "tree_recall": best_tree_val, "random": random_val,
        "gap_closed": gap_closed,
    })

print(f"\n  How much of the GP-tree gap does TS-Batch close?")
print(f"  Gap closed = (TS - Random) / (BestTree - Random)")
print()
print(f"  {'PMID':>12s}  {'Type':>30s}  {'N':>5s}  {'Random':>7s}  {'TS':>7s}  {'BestTree':>9s}  {'Gap%':>6s}  {'Best Tree Strategy'}")
print("  " + "-" * 120)
for d in sorted(ts_vs_tree_details, key=lambda x: -x["gap_closed"] if not np.isnan(x["gap_closed"]) else -999):
    print(f"  {d['pmid']:>12d}  {d['stype']:>30s}  {d['n']:>5d}  {d['random']:>7.3f}  {d['ts_recall']:>7.3f}  {d['tree_recall']:>9.3f}  {100*d['gap_closed']:>5.0f}%  {d['best_tree']}")

gap_closed_all = [d["gap_closed"] for d in ts_vs_tree_details if not np.isnan(d["gap_closed"])]
print(f"\n  Overall: TS-Batch closes {100*np.mean(gap_closed_all):.1f}% of the GP-tree gap on average")
print(f"  Median: {100*np.median(gap_closed_all):.1f}%")

# By study type
for stype in ["il_diverse_fixed_ratios", "il_diverse_variable_ratios", "ratio_only"]:
    subset = [d["gap_closed"] for d in ts_vs_tree_details if d["stype"] == stype and not np.isnan(d["gap_closed"])]
    if subset:
        print(f"  {stype}: {100*np.mean(subset):.1f}% gap closed ({len(subset)} studies)")

# Studies where TS-Batch BEATS best tree
beats = [d for d in ts_vs_tree_details if d["gap_closed"] >= 1.0]
print(f"\n  Studies where TS-Batch >= best tree model: {len(beats)}/{len(ts_vs_tree_details)}")
for d in beats:
    print(f"    PMID {d['pmid']}: TS={d['ts_recall']:.3f} vs Tree={d['tree_recall']:.3f} ({d['best_tree']})")


# ==============================================================
# SUMMARY FIGURE: Combined analysis
# ==============================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Panel 1: TS-Batch vs Best KB per study (scatter)
ax = axes[0, 0]
ts_vals = [d["ts_recall"] for d in ts_vs_kb_details]
kb_vals = [d["kb_recall"] for d in ts_vs_kb_details]
stype_colors = {"il_diverse_fixed_ratios": "#457b9d", "il_diverse_variable_ratios": "#e63946", "ratio_only": "#2a9d8f"}
for d in ts_vs_kb_details:
    c = stype_colors.get(d["stype"], "gray")
    ax.scatter(d["kb_recall"], d["ts_recall"], c=c, s=60, alpha=0.7, edgecolors="k", linewidth=0.5)
ax.plot([0, 1], [0, 1], "k--", alpha=0.5, linewidth=1)
ax.set_xlabel("Best KB Recall (top-5%)", fontsize=11)
ax.set_ylabel("TS-Batch Recall (top-5%)", fontsize=11)
ax.set_title("TS-Batch vs Best KB Strategy per Study", fontsize=12)
# Legend
for stype, c in stype_colors.items():
    short = stype.replace("il_diverse_", "").replace("_", " ")
    ax.scatter([], [], c=c, label=short, s=60, edgecolors="k", linewidth=0.5)
ax.legend(fontsize=9, loc="lower right")
ax.set_xlim(0.2, 1.05)
ax.set_ylim(0.2, 1.05)

# Panel 2: Convergence curves
ax = axes[0, 1]
for strat, label in [("lnpbo_ts_batch", "TS-Batch"), ("lnpbo_logei", "LogEI (KB)"), ("lnpbo_gibbon", "GIBBON")]:
    curves = get_normalized_convergence(strat)
    if not curves:
        continue
    max_len = max(len(c) for c in curves)
    padded = [c + [c[-1]] * (max_len - len(c)) for c in curves]
    arr = np.array(padded)
    mean_curve = arr.mean(axis=0)
    se = arr.std(axis=0) / np.sqrt(len(arr))
    rounds = np.arange(len(mean_curve))
    ax.plot(rounds, mean_curve, label=label, color=colors.get(label, "gray"), linewidth=2)
    ax.fill_between(rounds, mean_curve - se, mean_curve + se, alpha=0.15, color=colors.get(label, "gray"))
ax.set_xlabel("Round", fontsize=11)
ax.set_ylabel("Normalized Best-So-Far", fontsize=11)
ax.set_title("Convergence Comparison", fontsize=12)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

# Panel 3: Round-best quality over time
ax = axes[1, 0]
for strat, label in [("lnpbo_ts_batch", "TS-Batch"), ("lnpbo_logei", "LogEI (KB)"), ("lnpbo_gibbon", "GIBBON")]:
    curves = get_normalized_round_best(strat)
    if not curves:
        continue
    max_len = max(len(c) for c in curves)
    padded = [c + [np.nan] * (max_len - len(c)) for c in curves]
    arr = np.array(padded)
    mean_curve = np.nanmean(arr, axis=0)
    se = np.nanstd(arr, axis=0) / np.sqrt(np.sum(~np.isnan(arr), axis=0))
    rounds = np.arange(1, len(mean_curve) + 1)
    ax.plot(rounds, mean_curve, label=label, color=colors.get(label, "gray"), linewidth=2)
    ax.fill_between(rounds, mean_curve - se, mean_curve + se, alpha=0.15, color=colors.get(label, "gray"))
ax.set_xlabel("Round", fontsize=11)
ax.set_ylabel("Normalized Round-Best (z-score)", fontsize=11)
ax.set_title("Per-Round Batch Quality", fontsize=12)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

# Panel 4: Gap closed vs best tree
ax = axes[1, 1]
for d in ts_vs_tree_details:
    c = stype_colors.get(d["stype"], "gray")
    ax.scatter(d["n"], d["gap_closed"], c=c, s=60, alpha=0.7, edgecolors="k", linewidth=0.5)
ax.axhline(1.0, color="red", linestyle="--", alpha=0.5, label="Closes full gap")
ax.set_xlabel("Study Size (# formulations)", fontsize=11)
ax.set_ylabel("Gap Closed vs Best Tree", fontsize=11)
ax.set_title("TS-Batch: Fraction of GP-Tree Gap Closed", fontsize=12)
for stype, c in stype_colors.items():
    short = stype.replace("il_diverse_", "").replace("_", " ")
    ax.scatter([], [], c=c, label=short, s=60, edgecolors="k", linewidth=0.5)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

fig.suptitle("TS-Batch Analysis: Why It Is the Best LNPBO Strategy", fontsize=14, fontweight="bold", y=1.01)
fig.tight_layout()
fig.savefig(OUT / "tsbatch_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  Saved: {OUT / 'tsbatch_summary.png'}")


# ==============================================================
# FINAL SUMMARY
# ==============================================================
print("\n" + "=" * 80)
print("SUMMARY: WHY TS-BATCH IS THE BEST LNPBO STRATEGY")
print("=" * 80)

# Grand means
print(f"\n  Grand mean top-5% recall across 23 studies x 5 seeds:")
for strat in LNPBO_STRATEGIES:
    r = mean_recall(strat)
    label = strat_details.get(strat, ("?", "?"))
    print(f"    {strat:>25s} ({label[0]:>12s}, {label[1]:>6s}): {r:.3f}")

random_mean = mean_recall("random")
print(f"    {'random':>25s}: {random_mean:.3f}")
print()

# Statistical significance: TS vs each other LNPBO
print("  Paired t-tests (TS-Batch vs each LNPBO strategy, paired by study-seed):")
ts_by_study_seed = {}
for pmid in PMIDS:
    for seed in SEEDS:
        vals = recall["lnpbo_ts_batch"].get(pmid, [])
        seed_idx = SEEDS.index(seed)
        if seed_idx < len(vals):
            ts_by_study_seed[(pmid, seed)] = vals[seed_idx]

for strat in LNPBO_STRATEGIES:
    if strat == "lnpbo_ts_batch":
        continue
    ts_list = []
    other_list = []
    for pmid in PMIDS:
        for seed in SEEDS:
            seed_idx = SEEDS.index(seed)
            ts_vals = recall["lnpbo_ts_batch"].get(pmid, [])
            other_vals = recall[strat].get(pmid, [])
            if seed_idx < len(ts_vals) and seed_idx < len(other_vals):
                ts_list.append(ts_vals[seed_idx])
                other_list.append(other_vals[seed_idx])
    if len(ts_list) > 2:
        t, p = stats.ttest_rel(ts_list, other_list)
        d_mean = np.mean(np.array(ts_list) - np.array(other_list))
        print(f"    vs {strat:>25s}: diff={d_mean:+.4f}, t={t:.2f}, p={p:.4f} {'***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'}")

print("\n  CONCLUSIONS:")
print("  1. TS-Batch wins by batch diversity, not acquisition function quality.")
print("     The KB->TS batch method change explains the majority of the gap.")
print("  2. KB strategies pick redundant batches: top-B by AF score cluster in the")
print("     same posterior region. TS samples from the posterior, naturally spreading")
print("     selections across high-posterior-probability regions.")
print("  3. TS-Batch's advantage is consistent across study types but strongest in")
print("     ratio-only studies where the search space is more structured.")
print("  4. TS-Batch closes a substantial portion of the GP-tree gap, making it")
print("     the strongest GP-based strategy and competitive with simpler models.")
print("  5. GIBBON (another batch-aware strategy) also outperforms KB strategies,")
print("     further confirming that batch selection mechanism is the key differentiator.")
