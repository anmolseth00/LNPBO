"""
GIBBON benchmark analysis for LNPBO within-study results.

Produces:
  1. Convergence curves (mean best_so_far by round)
  2. GIBBON vs TS-Batch head-to-head
  3. GIBBON per-study rank
  4. Batch diversity (skipped — no selected indices in JSON)
  5. Early vs late round improvement
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

# ── Paths ────────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent / "benchmark_results" / "within_study"
OUT = BASE / "gibbon_analysis"
OUT.mkdir(exist_ok=True)

SEEDS = [42, 123, 456, 789, 2024]
PMIDS = sorted(
    [d.name for d in BASE.iterdir() if d.is_dir() and d.name.isdigit()]
)
print(f"Found {len(PMIDS)} studies: {PMIDS}\n")

# ── Load all results ─────────────────────────────────────────────────────────
# data[pmid][strategy][seed] = result dict
data = defaultdict(lambda: defaultdict(dict))
study_info = {}

for pmid in PMIDS:
    pmid_dir = BASE / pmid
    for f in pmid_dir.glob("*.json"):
        d = json.loads(f.read_text())
        strat = d["strategy"]
        seed = d["seed"]
        data[pmid][strat][seed] = d
        if pmid not in study_info:
            study_info[pmid] = d.get("study_info", {})

all_strategies = sorted({s for p in data for s in data[p]})
print(f"Found {len(all_strategies)} strategies: {all_strategies}\n")

# ── Helper: get metric averaged over seeds ───────────────────────────────────
def mean_recall_5(pmid, strategy):
    """Mean top-5% recall across seeds for a (pmid, strategy)."""
    vals = []
    for seed in SEEDS:
        r = data[pmid].get(strategy, {}).get(seed)
        if r is not None:
            vals.append(r["result"]["metrics"]["top_k_recall"]["5"])
    return np.mean(vals) if vals else np.nan


def mean_best_so_far(pmid, strategy):
    """Mean best_so_far curve across seeds. Returns array of variable length."""
    curves = []
    for seed in SEEDS:
        r = data[pmid].get(strategy, {}).get(seed)
        if r is not None:
            curves.append(r["result"]["best_so_far"])
    if not curves:
        return None
    # Pad to same length across seeds (within this study)
    max_len = max(len(c) for c in curves)
    padded = []
    for c in curves:
        if len(c) < max_len:
            c = c + [c[-1]] * (max_len - len(c))
        padded.append(c)
    return np.mean(padded, axis=0)


# Determine the maximum round count across all studies
MAX_ROUNDS_ALL = 0
for pmid in PMIDS:
    for strat in data[pmid]:
        for seed in SEEDS:
            r = data[pmid].get(strat, {}).get(seed)
            if r is not None:
                MAX_ROUNDS_ALL = max(MAX_ROUNDS_ALL, len(r["result"]["best_so_far"]))
print(f"Max best_so_far length across all studies: {MAX_ROUNDS_ALL}")


# =============================================================================
# 1. CONVERGENCE CURVES
# =============================================================================
print("=" * 70)
print("1. CONVERGENCE CURVES — mean best_so_far by round")
print("=" * 70)

# We need to normalize best_so_far across studies (different z-score scales).
# Approach: for each study, normalize to [0, 1] using (x - min) / (max - min)
# where min/max come from ALL strategies in that study at round 0 and 15.

FOCUS_STRATEGIES = [
    "lnpbo_gibbon",
    "random",
    "lnpbo_ts_batch",
    "lnpbo_logei",
    "discrete_ngboost_ucb",
    "discrete_rf_ts",
]
STRAT_LABELS = {
    "lnpbo_gibbon": "GIBBON",
    "random": "Random",
    "lnpbo_ts_batch": "TS-Batch",
    "lnpbo_logei": "LogEI",
    "discrete_ngboost_ucb": "NGBoost-UCB",
    "discrete_rf_ts": "RF-TS",
}
STRAT_COLORS = {
    "lnpbo_gibbon": "#e41a1c",
    "random": "#999999",
    "lnpbo_ts_batch": "#377eb8",
    "lnpbo_logei": "#4daf4a",
    "discrete_ngboost_ucb": "#ff7f00",
    "discrete_rf_ts": "#984ea3",
}

# For normalization, use top-5% recall instead (already normalized 0-1 by design).
# Actually, best_so_far is z-scored per study, so cross-study averaging IS meaningful,
# but recall is more interpretable. Let's compute recall-based convergence.
#
# Problem: we only have final recall, not per-round recall.
# So let's use best_so_far with per-study normalization to [0,1] range.

# Step 1: For each study, find the global min and max best_so_far across all
# strategies and seeds at any round.
study_bounds = {}
for pmid in PMIDS:
    all_vals = []
    for strat in data[pmid]:
        for seed in SEEDS:
            r = data[pmid].get(strat, {}).get(seed)
            if r is not None:
                all_vals.extend(r["result"]["best_so_far"])
    if all_vals:
        study_bounds[pmid] = (min(all_vals), max(all_vals))

# Step 2: Compute normalized convergence curves
# Studies have different numbers of rounds (7-15). We use a masked array approach:
# for each round r, average only over studies that have data at that round.
norm_curves = {s: [] for s in FOCUS_STRATEGIES}

for pmid in PMIDS:
    lo, hi = study_bounds.get(pmid, (0, 1))
    rng = hi - lo if hi != lo else 1.0
    for strat in FOCUS_STRATEGIES:
        curve = mean_best_so_far(pmid, strat)
        if curve is not None:
            normed = (curve - lo) / rng
            # Pad to MAX_ROUNDS_ALL with NaN
            padded = np.full(MAX_ROUNDS_ALL, np.nan)
            padded[:len(normed)] = normed
            norm_curves[strat].append(padded)

# Average across studies using nanmean
fig, ax = plt.subplots(figsize=(10, 6))
rounds = np.arange(MAX_ROUNDS_ALL)

for strat in FOCUS_STRATEGIES:
    if not norm_curves[strat]:
        continue
    arr = np.array(norm_curves[strat])  # (n_studies, MAX_ROUNDS_ALL)
    mean = np.nanmean(arr, axis=0)
    n_valid = np.sum(~np.isnan(arr), axis=0)
    se = np.nanstd(arr, axis=0) / np.sqrt(np.maximum(n_valid, 1))
    label = STRAT_LABELS.get(strat, strat)
    color = STRAT_COLORS.get(strat, None)
    # Only plot where we have at least 5 studies contributing
    mask = n_valid >= 5
    ax.plot(rounds[mask], mean[mask], label=label, color=color, linewidth=2)
    ax.fill_between(rounds[mask], (mean - se)[mask], (mean + se)[mask],
                    alpha=0.15, color=color)
    # Print summary for first and last valid rounds
    first_valid = np.where(mask)[0][0]
    last_valid = np.where(mask)[0][-1]
    print(f"  {label:20s}: round 0={mean[first_valid]:.4f}, "
          f"round {last_valid}={mean[last_valid]:.4f}, "
          f"gain={mean[last_valid]-mean[first_valid]:.4f}, "
          f"n_studies at end={n_valid[last_valid]}")

ax.set_xlabel("Round (0 = seed pool)", fontsize=12)
ax.set_ylabel("Normalized best-so-far (0=worst, 1=best)", fontsize=12)
ax.set_title("Convergence: GIBBON vs key strategies (mean over 23 studies, 5 seeds)", fontsize=13)
ax.legend(fontsize=10, loc="lower right")
ax.set_xlim(0, 15)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "gibbon_convergence.png", dpi=150)
plt.close(fig)
print(f"\n  Saved: {OUT / 'gibbon_convergence.png'}\n")


# =============================================================================
# 2. GIBBON vs TS-BATCH HEAD-TO-HEAD
# =============================================================================
print("=" * 70)
print("2. GIBBON vs TS-BATCH head-to-head (top-5% recall, per study)")
print("=" * 70)

wins, losses, ties = 0, 0, 0
h2h_rows = []

for pmid in PMIDS:
    g_recall = mean_recall_5(pmid, "lnpbo_gibbon")
    t_recall = mean_recall_5(pmid, "lnpbo_ts_batch")
    stype = study_info.get(pmid, {}).get("study_type", "unknown")
    n_form = study_info.get(pmid, {}).get("n_formulations", 0)

    diff = g_recall - t_recall
    if abs(diff) < 0.005:
        outcome = "TIE"
        ties += 1
    elif diff > 0:
        outcome = "GIBBON"
        wins += 1
    else:
        outcome = "TS-Batch"
        losses += 1

    h2h_rows.append((pmid, stype, n_form, g_recall, t_recall, diff, outcome))
    print(f"  PMID {pmid:>10s} ({stype:30s}, n={n_form:5d}): "
          f"GIBBON={g_recall:.3f}  TS-Batch={t_recall:.3f}  diff={diff:+.3f}  -> {outcome}")

print(f"\n  Summary: GIBBON wins={wins}, TS-Batch wins={losses}, ties={ties}")
print(f"  GIBBON win rate: {wins}/{wins+losses+ties} = {wins/(wins+losses+ties):.1%}")

# Breakdown by study type
from collections import Counter
type_wins = defaultdict(lambda: {"GIBBON": 0, "TS-Batch": 0, "TIE": 0})
for pmid, stype, n_form, g, t, diff, outcome in h2h_rows:
    type_wins[stype][outcome] += 1

print("\n  By study type:")
for stype in sorted(type_wins):
    counts = type_wins[stype]
    total = sum(counts.values())
    print(f"    {stype:35s}: GIBBON {counts['GIBBON']}/{total}, "
          f"TS-Batch {counts['TS-Batch']}/{total}, Ties {counts['TIE']}/{total}")

# Scatter plot
fig, ax = plt.subplots(figsize=(8, 8))
for pmid, stype, n_form, g, t, diff, outcome in h2h_rows:
    marker = {"il_diverse_fixed_ratios": "o",
              "il_diverse_variable_ratios": "s",
              "ratio_only": "D"}.get(stype, "^")
    color = {"il_diverse_fixed_ratios": "#377eb8",
             "il_diverse_variable_ratios": "#ff7f00",
             "ratio_only": "#e41a1c"}.get(stype, "gray")
    ax.scatter(t, g, marker=marker, color=color, s=80, edgecolor="k", linewidth=0.5,
               zorder=3)

# Diagonal
lims = [0, 1]
ax.plot(lims, lims, "k--", alpha=0.4, linewidth=1)
ax.set_xlabel("TS-Batch (top-5% recall)", fontsize=12)
ax.set_ylabel("GIBBON (top-5% recall)", fontsize=12)
ax.set_title(f"GIBBON vs TS-Batch per study\n(GIBBON wins {wins}, TS-Batch wins {losses}, ties {ties})",
             fontsize=13)

# Legend for study types
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#377eb8',
           markersize=10, label='IL-diverse fixed-ratio'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='#ff7f00',
           markersize=10, label='IL-diverse variable-ratio'),
    Line2D([0], [0], marker='D', color='w', markerfacecolor='#e41a1c',
           markersize=10, label='Ratio-only'),
]
ax.legend(handles=legend_elements, fontsize=10)
ax.set_xlim(0, 1.02)
ax.set_ylim(0, 1.02)
ax.set_aspect("equal")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "gibbon_vs_tsbatch.png", dpi=150)
plt.close(fig)
print(f"\n  Saved: {OUT / 'gibbon_vs_tsbatch.png'}\n")


# =============================================================================
# 3. GIBBON PER-STUDY RANK
# =============================================================================
print("=" * 70)
print("3. GIBBON per-study rank among all strategies")
print("=" * 70)

gibbon_ranks = []
n_strategies_per_study = []

for pmid in PMIDS:
    # Compute mean recall for every strategy in this study
    strat_recalls = {}
    for strat in data[pmid]:
        r = mean_recall_5(pmid, strat)
        if not np.isnan(r):
            strat_recalls[strat] = r

    # Rank (1 = best)
    ranked = sorted(strat_recalls.items(), key=lambda x: -x[1])
    n_strats = len(ranked)
    n_strategies_per_study.append(n_strats)

    gibbon_rank = None
    for i, (s, v) in enumerate(ranked, 1):
        if s == "lnpbo_gibbon":
            gibbon_rank = i
            break

    stype = study_info.get(pmid, {}).get("study_type", "unknown")
    g_recall = strat_recalls.get("lnpbo_gibbon", np.nan)
    best_strat, best_val = ranked[0] if ranked else ("?", 0)

    tag = ""
    if gibbon_rank is not None and gibbon_rank <= 5:
        tag = " <-- TOP 5"
    elif gibbon_rank is not None and gibbon_rank > n_strats - 5:
        tag = " <-- BOTTOM 5"

    print(f"  PMID {pmid:>10s} ({stype:30s}): rank {gibbon_rank:2d}/{n_strats} "
          f"(recall={g_recall:.3f}, best={best_strat} {best_val:.3f}){tag}")

    if gibbon_rank is not None:
        gibbon_ranks.append(gibbon_rank)

print(f"\n  Mean rank: {np.mean(gibbon_ranks):.1f} / {np.mean(n_strategies_per_study):.0f}")
print(f"  Median rank: {np.median(gibbon_ranks):.1f}")
print(f"  Top-5 count: {sum(1 for r in gibbon_ranks if r <= 5)}")
print(f"  Bottom-5 count: {sum(1 for r, n in zip(gibbon_ranks, n_strategies_per_study) if r > n - 5)}")

# Bar chart of GIBBON rank per study
fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(len(PMIDS))
colors_bar = []
for pmid in PMIDS:
    stype = study_info.get(pmid, {}).get("study_type", "unknown")
    colors_bar.append({"il_diverse_fixed_ratios": "#377eb8",
                       "il_diverse_variable_ratios": "#ff7f00",
                       "ratio_only": "#e41a1c"}.get(stype, "gray"))

ax.bar(x, gibbon_ranks, color=colors_bar, edgecolor="k", linewidth=0.5)
ax.axhline(y=np.mean(gibbon_ranks), color="red", linestyle="--", alpha=0.7,
           label=f"Mean rank = {np.mean(gibbon_ranks):.1f}")
ax.axhline(y=5, color="green", linestyle=":", alpha=0.7, label="Top 5")
ax.set_xticks(x)
ax.set_xticklabels(PMIDS, rotation=60, ha="right", fontsize=8)
ax.set_ylabel("GIBBON rank (1 = best)", fontsize=12)
ax.set_title("GIBBON rank per study (among all strategies)", fontsize=13)
ax.legend(fontsize=10)
ax.set_xlim(-0.6, len(PMIDS) - 0.4)
ax.invert_yaxis()

# Add legend for study types
from matplotlib.patches import Patch
legend_patches = [
    Patch(facecolor='#377eb8', edgecolor='k', label='IL-diverse fixed-ratio'),
    Patch(facecolor='#ff7f00', edgecolor='k', label='IL-diverse variable-ratio'),
    Patch(facecolor='#e41a1c', edgecolor='k', label='Ratio-only'),
]
ax2 = ax.twinx()
ax2.set_yticks([])
ax2.legend(handles=legend_patches, loc="upper right", fontsize=9)

fig.tight_layout()
fig.savefig(OUT / "gibbon_rank_per_study.png", dpi=150)
plt.close(fig)
print(f"\n  Saved: {OUT / 'gibbon_rank_per_study.png'}\n")


# =============================================================================
# 4. BATCH DIVERSITY — SKIPPED
# =============================================================================
print("=" * 70)
print("4. BATCH DIVERSITY — SKIPPED (no selected indices in result JSON)")
print("=" * 70)
print("  The result JSON contains only best_so_far, round_best, n_evaluated.")
print("  Selected indices per round are not stored, so batch diversity")
print("  analysis is not possible from stored results.\n")


# =============================================================================
# 5. EARLY vs LATE ROUND IMPROVEMENT
# =============================================================================
print("=" * 70)
print("5. GIBBON early vs late round improvement (delta best_so_far)")
print("=" * 70)

# For each study+seed, compute per-round delta = best_so_far[r] - best_so_far[r-1]
# Early = rounds 1-5, Late = rounds 10-15 (using 1-indexed BO rounds)
# best_so_far[0] = seed, best_so_far[1] = after round 1, ...

ANALYSIS_STRATEGIES = [
    "lnpbo_gibbon",
    "lnpbo_ts_batch",
    "lnpbo_logei",
    "random",
    "discrete_ngboost_ucb",
    "discrete_rf_ts",
]

early_deltas = {s: [] for s in ANALYSIS_STRATEGIES}
late_deltas = {s: [] for s in ANALYSIS_STRATEGIES}

# For early vs late, only use studies with >= 15 BO rounds (best_so_far len >= 16)
long_pmids = [p for p in PMIDS
              if any(len(data[p].get("lnpbo_gibbon", {}).get(s, {}).get("result", {}).get("best_so_far", []))
                     if isinstance(data[p].get("lnpbo_gibbon", {}).get(s), dict) else False
                     for s in SEEDS)]
# Simpler: just check length per run
for pmid in PMIDS:
    lo, hi = study_bounds.get(pmid, (0, 1))
    rng = hi - lo if hi != lo else 1.0

    for strat in ANALYSIS_STRATEGIES:
        for seed in SEEDS:
            r = data[pmid].get(strat, {}).get(seed)
            if r is None:
                continue
            bsf = np.array(r["result"]["best_so_far"])
            # Normalize
            bsf_n = (bsf - lo) / rng
            deltas = np.diff(bsf_n)

            # Early: rounds 1-5 (deltas indices 0-4)
            early_end = min(5, len(deltas))
            early_deltas[strat].append(deltas[0:early_end].sum())

            # Late: last 5 rounds of deltas (or rounds 11-15 if available)
            if len(deltas) >= 15:
                late_deltas[strat].append(deltas[10:15].sum())
            elif len(deltas) >= 6:
                # Use last 5 rounds
                late_deltas[strat].append(deltas[-5:].sum())

print(f"  {'Strategy':25s} | {'Early (r1-5)':>12s} | {'Late (r11-15)':>13s} | {'Ratio E/L':>10s}")
print("  " + "-" * 70)

for strat in ANALYSIS_STRATEGIES:
    e = np.mean(early_deltas[strat]) if early_deltas[strat] else 0
    l = np.mean(late_deltas[strat]) if late_deltas[strat] else 0
    ratio = e / l if l > 1e-8 else float("inf")
    label = STRAT_LABELS.get(strat, strat)
    print(f"  {label:25s} | {e:12.5f} | {l:13.5f} | {ratio:10.2f}x")

print()

# Also do per-round delta curves
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Panel A: cumulative best_so_far improvement from round 0
ax = axes[0]
for strat in ANALYSIS_STRATEGIES:
    # Collect per-round cumulative gain from seed, NaN-padded
    all_gains = []
    for pmid in PMIDS:
        lo, hi = study_bounds.get(pmid, (0, 1))
        rng = hi - lo if hi != lo else 1.0
        for seed in SEEDS:
            r = data[pmid].get(strat, {}).get(seed)
            if r is None:
                continue
            bsf = np.array(r["result"]["best_so_far"])
            bsf_n = (bsf - lo) / rng
            gain = bsf_n - bsf_n[0]
            padded = np.full(MAX_ROUNDS_ALL, np.nan)
            padded[:len(gain)] = gain
            all_gains.append(padded)
    if all_gains:
        arr = np.array(all_gains)
        mean = np.nanmean(arr, axis=0)
        n_valid = np.sum(~np.isnan(arr), axis=0)
        se = np.nanstd(arr, axis=0) / np.sqrt(np.maximum(n_valid, 1))
        mask = n_valid >= 5
        label = STRAT_LABELS.get(strat, strat)
        color = STRAT_COLORS.get(strat, None)
        ax.plot(np.arange(MAX_ROUNDS_ALL)[mask], mean[mask], label=label, color=color, linewidth=2)
        ax.fill_between(np.arange(MAX_ROUNDS_ALL)[mask], (mean - se)[mask], (mean + se)[mask],
                        alpha=0.12, color=color)

ax.set_xlabel("Round", fontsize=11)
ax.set_ylabel("Normalized gain from seed", fontsize=11)
ax.set_title("Cumulative gain from seed pool", fontsize=12)
ax.legend(fontsize=9, loc="upper left")
ax.grid(True, alpha=0.3)

# Panel B: per-round marginal improvement
ax = axes[1]
max_delta_len = MAX_ROUNDS_ALL - 1
for strat in ANALYSIS_STRATEGIES:
    all_deltas_arr = []
    for pmid in PMIDS:
        lo, hi = study_bounds.get(pmid, (0, 1))
        rng = hi - lo if hi != lo else 1.0
        for seed in SEEDS:
            r = data[pmid].get(strat, {}).get(seed)
            if r is None:
                continue
            bsf = np.array(r["result"]["best_so_far"])
            bsf_n = (bsf - lo) / rng
            d = np.diff(bsf_n)
            padded = np.full(max_delta_len, np.nan)
            padded[:len(d)] = d
            all_deltas_arr.append(padded)
    if all_deltas_arr:
        arr = np.array(all_deltas_arr)
        mean = np.nanmean(arr, axis=0)
        n_valid = np.sum(~np.isnan(arr), axis=0)
        se = np.nanstd(arr, axis=0) / np.sqrt(np.maximum(n_valid, 1))
        mask = n_valid >= 5
        label = STRAT_LABELS.get(strat, strat)
        color = STRAT_COLORS.get(strat, None)
        rr = np.arange(1, max_delta_len + 1)
        ax.plot(rr[mask], mean[mask], label=label, color=color, linewidth=2)
        ax.fill_between(rr[mask], (mean - se)[mask], (mean + se)[mask], alpha=0.12, color=color)

ax.set_xlabel("Round", fontsize=11)
ax.set_ylabel("Marginal normalized improvement", fontsize=11)
ax.set_title("Per-round marginal improvement", fontsize=12)
ax.legend(fontsize=9, loc="upper right")
ax.grid(True, alpha=0.3)
ax.axhline(0, color="k", linewidth=0.5)

fig.suptitle("Early vs Late round improvement", fontsize=14, y=1.01)
fig.tight_layout()
fig.savefig(OUT / "gibbon_early_vs_late.png", dpi=150)
plt.close(fig)
print(f"  Saved: {OUT / 'gibbon_early_vs_late.png'}\n")


# =============================================================================
# BONUS: Overall GIBBON summary table
# =============================================================================
print("=" * 70)
print("BONUS: Overall strategy recall summary (for context)")
print("=" * 70)

strat_recalls_all = defaultdict(list)
for pmid in PMIDS:
    for strat in all_strategies:
        r = mean_recall_5(pmid, strat)
        if not np.isnan(r):
            strat_recalls_all[strat].append(r)

print(f"\n  {'Strategy':35s} | {'Mean Recall':>11s} | {'Std':>6s} | {'N studies':>9s}")
print("  " + "-" * 70)

sorted_strats = sorted(strat_recalls_all.items(), key=lambda x: -np.mean(x[1]))
gibbon_rank_overall = None
for i, (strat, recalls) in enumerate(sorted_strats, 1):
    tag = " <--" if strat == "lnpbo_gibbon" else ""
    print(f"  {i:2d}. {strat:31s} | {np.mean(recalls):11.4f} | {np.std(recalls):.4f} | {len(recalls):9d}{tag}")
    if strat == "lnpbo_gibbon":
        gibbon_rank_overall = i

print(f"\n  GIBBON overall rank: {gibbon_rank_overall}/{len(sorted_strats)}")
print(f"  GIBBON mean recall: {np.mean(strat_recalls_all['lnpbo_gibbon']):.4f}")
print(f"  Random mean recall: {np.mean(strat_recalls_all['random']):.4f}")
gibbon_lift = np.mean(strat_recalls_all['lnpbo_gibbon']) / np.mean(strat_recalls_all['random'])
print(f"  GIBBON lift over random: {gibbon_lift:.2f}x")

print(f"\nAll figures saved to: {OUT}")
print("Done.")
