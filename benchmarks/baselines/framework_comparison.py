#!/usr/bin/env python3
"""Framework comparison: mapping LNPBO strategies to standard BO frameworks.

This module addresses the reviewer question: "Why didn't you compare with
Ax, SMAC, Optuna, etc.?" The answer is that several of our strategies are
algorithmically equivalent to those frameworks' defaults, differing only in
the optimization domain (discrete pool vs. continuous+rounding).

Relationship to standard frameworks
------------------------------------

**Ax / BoTorch (Meta)**
  Ax's default optimizer uses a BoTorch SingleTaskGP with a Matern-5/2 kernel
  and qLogNoisyExpectedImprovement acquisition, optimized over a continuous
  relaxation of the search space, then rounded to the nearest feasible point.
  Our ``lnpbo_logei`` strategy uses the *same* BoTorch SingleTaskGP and
  LogExpectedImprovement acquisition but scores the discrete candidate pool
  directly -- no continuous relaxation or rounding. Ax's Sobol+GP pattern
  (quasi-random warmup followed by GP-guided selection) mirrors our benchmark
  design of a random 25% seed pool followed by surrogate-guided rounds.

**SMAC (AutoML Freiburg)**
  SMAC v3 uses a random forest surrogate with EI acquisition for
  mixed-integer/conditional spaces. Our ``discrete_rf_ucb`` and
  ``discrete_rf_ts`` strategies use the same surrogate class (scikit-learn
  RandomForestRegressor) with UCB or Thompson sampling acquisition over the
  discrete pool. The key difference is that SMAC uses a local search for
  acquisition optimization while we exhaustively score all remaining
  candidates.

**Optuna / HyperOpt (TPE)**
  Optuna's default sampler is the Tree-structured Parzen Estimator (TPE),
  which models P(x|y) directly rather than fitting a regression surrogate.
  This is a fundamentally different approach that we did not implement.
  Including TPE is a valid direction for future work.

**Dragonfly (Cornell)**
  Dragonfly supports multi-fidelity and mixed-variable BO via additive GP
  kernels over categorical/continuous subspaces. Our ``casmopolitan_ucb``
  and ``casmopolitan_ei`` strategies implement the CASMOPolitan algorithm
  (Wan et al., 2021), which similarly handles mixed-variable spaces with
  a trust-region approach and interleaved categorical/continuous optimization.

**TuRBO (Eriksson et al., 2019)**
  Trust-region BO with local GP models. Our CASMOPolitan strategies extend
  TuRBO's trust-region idea to categorical variables.

Summary
-------
For GP-based BO (Ax, BoTorch), RF-based BO (SMAC), and mixed-variable BO
(Dragonfly), our benchmark includes strategies that use the same surrogate
models and acquisition functions as these frameworks. The difference is in
how the acquisition function is optimized: we score a discrete candidate pool
exhaustively rather than optimizing over a continuous relaxation. For the
finite combinatorial spaces in LNP design, discrete-pool scoring is the
natural choice and avoids rounding artifacts.

The one notable gap is TPE (Optuna/HyperOpt), which we did not include.

Usage:
    python -m benchmarks.baselines.framework_comparison
"""

from __future__ import annotations

import json
from pathlib import Path

from LNPBO.runtime_paths import benchmark_results_root, package_root_from

_PACKAGE_ROOT = package_root_from(__file__, levels_up=3)
RESULTS_DIR = benchmark_results_root(_PACKAGE_ROOT) / "within_study"

# --------------------------------------------------------------------------- #
# Mapping: external framework -> closest LNPBO strategy
# --------------------------------------------------------------------------- #

FRAMEWORK_EQUIVALENTS = {
    "Ax (default GP+LogEI)": {
        "lnpbo_strategy": "lnpbo_logei",
        "relationship": "Same BoTorch SingleTaskGP + LogEI backend",
        "difference": "Ax optimizes continuous relaxation then rounds; we score discrete pool directly",
    },
    "Ax (Sobol + GP)": {
        "lnpbo_strategy": "lnpbo_logei",
        "relationship": "Same warmup-then-GP pattern (our 25% random seed pool + GP rounds)",
        "difference": "Sobol vs. uniform random for initial design",
    },
    "SMAC (RF + EI)": {
        "lnpbo_strategy": "discrete_rf_ucb",
        "relationship": "Same RF surrogate; UCB instead of EI acquisition",
        "difference": "SMAC uses local search for acquisition opt; we score full pool",
    },
    "Optuna (TPE)": {
        "lnpbo_strategy": None,
        "relationship": "Not directly compared",
        "difference": "TPE models P(x|y) rather than fitting a regression surrogate",
    },
    "HyperOpt (TPE)": {
        "lnpbo_strategy": None,
        "relationship": "Not directly compared (same TPE as Optuna)",
        "difference": "TPE is a distinct surrogate family we did not implement",
    },
    "Dragonfly (mixed-var)": {
        "lnpbo_strategy": "casmopolitan_ucb",
        "relationship": "CASMOPolitan extends TuRBO trust regions to categorical variables",
        "difference": "Different trust-region update rules; both handle mixed spaces",
    },
    "BoTorch (direct)": {
        "lnpbo_strategy": "lnpbo_logei",
        "relationship": "We use BoTorch directly for all lnpbo_* strategies",
        "difference": "None -- same library, discrete pool instead of continuous opt",
    },
}

# Strategies to report when printing the comparison table.
_STRATEGIES_OF_INTEREST = [
    "random",
    "lnpbo_logei",
    "lnpbo_ucb",
    "lnpbo_ts_batch",
    "discrete_rf_ucb",
    "discrete_rf_ts",
    "casmopolitan_ucb",
    "casmopolitan_ei",
    "discrete_xgb_ucb",
    "discrete_ngboost_ucb",
]

STRATEGY_SHORT = {
    "random": "Random",
    "lnpbo_ucb": "GP-UCB",
    "lnpbo_ei": "GP-EI",
    "lnpbo_logei": "GP-LogEI",
    "lnpbo_lp_ei": "GP-LP-EI",
    "lnpbo_lp_logei": "GP-LP-LogEI",
    "lnpbo_ts_batch": "GP-TS-Batch",
    "casmopolitan_ei": "CASMO-EI",
    "casmopolitan_ucb": "CASMO-UCB",
    "discrete_rf_ucb": "RF-UCB",
    "discrete_rf_ts": "RF-TS",
    "discrete_xgb_ucb": "XGB-UCB",
    "discrete_ngboost_ucb": "NGBoost-UCB",
    "discrete_deep_ensemble": "DeepEnsemble",
    "discrete_gp_ucb": "GP-UCB (sklearn)",
}


def load_strategy_means() -> dict[str, float]:
    """Compute mean Top-5% recall per strategy across all studies from the summary JSON.

    For each strategy, averages the per-study mean recall (i.e., study-level
    aggregation first, then grand mean across studies).
    """
    summary_path = RESULTS_DIR / "within_study_summary.json"
    if not summary_path.exists():
        return {}

    with open(summary_path) as f:
        data = json.load(f)

    summaries = data.get("summaries", {})

    # Collect per-study means for each strategy
    strategy_study_means: dict[str, list[float]] = {}
    for _pmid, study_strategies in summaries.items():
        for strategy_name, info in study_strategies.items():
            recall_info = info.get("top_5pct_recall", {})
            mean_val = recall_info.get("mean")
            if mean_val is not None:
                strategy_study_means.setdefault(strategy_name, []).append(mean_val)

    # Grand mean across studies
    return {s: sum(vals) / len(vals) for s, vals in strategy_study_means.items() if vals}


def print_comparison_table() -> None:
    """Print a formatted table mapping frameworks to LNPBO equivalents with performance."""
    means = load_strategy_means()
    random_mean = means.get("random", 0.0)

    hdr_fw = "Framework"
    hdr_eq = "LNPBO Equivalent"
    hdr_recall = "Top-5% Recall"
    hdr_lift = "Lift vs Random"
    hdr_note = "Key Difference"

    rows = []
    for framework, info in FRAMEWORK_EQUIVALENTS.items():
        strat = info["lnpbo_strategy"]
        if strat and strat in means:
            short = STRATEGY_SHORT.get(strat, strat)
            recall = means[strat]
            lift = recall / random_mean if random_mean > 0 else float("nan")
            rows.append((framework, short, f"{recall:.3f}", f"{lift:.2f}x", info["difference"]))
        else:
            rows.append((framework, "-- (not implemented)", "--", "--", info["difference"]))

    # Column widths
    w_fw = max(len(hdr_fw), max(len(r[0]) for r in rows))
    w_eq = max(len(hdr_eq), max(len(r[1]) for r in rows))
    w_rc = max(len(hdr_recall), max(len(r[2]) for r in rows))
    w_lt = max(len(hdr_lift), max(len(r[3]) for r in rows))
    w_nt = max(len(hdr_note), max(len(r[4]) for r in rows))

    def fmt(a: str, b: str, c: str, d: str, e: str) -> str:
        return f"  {a:<{w_fw}}  {b:<{w_eq}}  {c:>{w_rc}}  {d:>{w_lt}}  {e}"

    sep = fmt("-" * w_fw, "-" * w_eq, "-" * w_rc, "-" * w_lt, "-" * w_nt)

    print("=" * 80)
    print("Framework Comparison: LNPBO vs. Standard BO Libraries")
    print("=" * 80)
    print()
    print("Within-study benchmark: 26 studies, 5 seeds, 25% seed pool, batch 12")
    print(f"Random baseline Top-5% recall: {random_mean:.3f}")
    print()
    print(fmt(hdr_fw, hdr_eq, hdr_recall, hdr_lift, hdr_note))
    print(sep)
    for row in rows:
        print(fmt(*row))
    print()

    # Additional context: best LNPBO strategies for reference
    print("-" * 80)
    print("Selected LNPBO strategy performance (for reference)")
    print("-" * 80)
    print()
    ref_rows = []
    for s in _STRATEGIES_OF_INTEREST:
        if s in means:
            short = STRATEGY_SHORT.get(s, s)
            recall = means[s]
            lift = recall / random_mean if random_mean > 0 else float("nan")
            ref_rows.append((short, f"{recall:.3f}", f"{lift:.2f}x"))

    w_s = max(len("Strategy"), max(len(r[0]) for r in ref_rows)) if ref_rows else 20
    w_r = max(len("Top-5% Recall"), 7)
    w_l = max(len("Lift"), 6)

    print(f"  {'Strategy':<{w_s}}  {'Top-5% Recall':>{w_r}}  {'Lift':>{w_l}}")
    print(f"  {'-' * w_s}  {'-' * w_r}  {'-' * w_l}")
    for short, recall, lift in ref_rows:
        print(f"  {short:<{w_s}}  {recall:>{w_r}}  {lift:>{w_l}}")

    print()
    print("Note: The only major BO paradigm not covered is TPE (Optuna/HyperOpt).")
    print("GP-based (Ax/BoTorch), RF-based (SMAC), and mixed-variable (Dragonfly/CASMOPolitan)")
    print("are all represented by equivalent LNPBO strategies.")


if __name__ == "__main__":
    print_comparison_table()
