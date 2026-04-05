#!/usr/bin/env python3
"""Analyze kappa=0.5 full benchmark vs kappa=5.0 main benchmark.

Computes family-level mean top-5% recall at kappa=0.5 and compares
to the kappa=5.0 main benchmark results.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

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
}

# Strategies where kappa directly controls UCB exploration weight
UCB_STRATEGIES = {
    "lnpbo_ucb", "discrete_rf_ucb", "discrete_xgb_ucb",
    "discrete_ngboost_ucb", "discrete_gp_ucb", "casmopolitan_ucb",
    "discrete_deep_ensemble", "discrete_xgb_cqr",
    "discrete_xgb_online_conformal",
}

# Strategies invariant to kappa (EI, LogEI, TS, greedy)
INVARIANT_STRATEGIES = {
    "random", "lnpbo_ei", "lnpbo_logei", "lnpbo_lp_ei", "lnpbo_lp_logei",
    "lnpbo_pls_logei", "lnpbo_pls_lp_logei", "lnpbo_ts_batch",
    "casmopolitan_ei", "discrete_rf_ts", "discrete_rf_ts_batch",
    "discrete_xgb_greedy", "discrete_xgb_ucb_ts_batch",
    "lnpbo_gibbon", "lnpbo_tanimoto_ts", "lnpbo_tanimoto_logei",
    "lnpbo_aitchison_ts", "lnpbo_aitchison_logei",
    "lnpbo_dkl_ts", "lnpbo_dkl_logei",
    "lnpbo_rf_kernel_ts", "lnpbo_rf_kernel_logei",
    "lnpbo_compositional_ts", "lnpbo_compositional_logei",
    "lnpbo_rkb_logei",
}


def load_results(results_dir):
    results = []
    rd = Path(results_dir)
    for study_dir in sorted(rd.iterdir()):
        if not study_dir.is_dir():
            continue
        for f in sorted(study_dir.glob("*.json")):
            try:
                data = json.loads(f.read_text())
                results.append(data)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"WARNING: {f}: {e}", file=sys.stderr)
    return results


def extract_recall(results, require_rounds=True):
    family_vals = defaultdict(list)
    strategy_vals = defaultdict(list)
    n_skipped = 0
    for r in results:
        strat = r.get("strategy", "")
        if strat not in STRATEGY_FAMILY:
            continue
        recall = r.get("result", {}).get("metrics", {}).get("top_k_recall", {}).get("5")
        if recall is None:
            continue
        # Skip broken runs (0 rounds = strategy failed to execute)
        if require_rounds and strat != "random":
            n_rounds = r.get("result", {}).get("metrics", {}).get("n_rounds", 0)
            if n_rounds == 0:
                n_skipped += 1
                continue
        family_vals[STRATEGY_FAMILY[strat]].append(recall)
        strategy_vals[strat].append(recall)
    if n_skipped > 0:
        print(f"  (skipped {n_skipped} broken runs with 0 rounds)")
    return family_vals, strategy_vals


def main():
    base = Path(__file__).resolve().parent.parent
    kappa_dir = base / "benchmark_results" / "ablations" / "kappa_optimal"
    ws_dir = base / "benchmark_results" / "within_study"

    print("Loading kappa_optimal results...")
    kappa_results = load_results(kappa_dir)
    print(f"  {len(kappa_results)} files loaded")

    print("Loading within-study results...")
    ws_results = load_results(ws_dir)
    print(f"  {len(ws_results)} files loaded")

    kappa_fam, kappa_strat = extract_recall(kappa_results)
    ws_fam, ws_strat = extract_recall(ws_results)

    fam_order = [
        "NGBoost", "RF", "CASMOPolitan", "XGBoost",
        "GP (sklearn)", "GP (BoTorch)", "Deep Ensemble", "Random",
    ]

    print()
    print("=" * 80)
    print("FAMILY-LEVEL COMPARISON: kappa=5.0 (main) vs kappa=0.5")
    print("=" * 80)
    print(f"{'Family':<18} {'k=5.0':>8} {'k=0.5':>8} {'Delta':>8} {'n(5.0)':>8} {'n(0.5)':>8}")
    print("-" * 80)
    for fam in fam_order:
        ws_m = np.mean(ws_fam.get(fam, [0]))
        k_m = np.mean(kappa_fam.get(fam, [0]))
        d = k_m - ws_m
        n_ws = len(ws_fam.get(fam, []))
        n_ko = len(kappa_fam.get(fam, []))
        print(f"{fam:<18} {ws_m:>8.3f} {k_m:>8.3f} {d:>+8.3f} {n_ws:>8} {n_ko:>8}")

    print()
    print("=" * 80)
    print("LIFT OVER RANDOM")
    print("=" * 80)
    ws_rand = np.mean(ws_fam["Random"])
    k_rand = np.mean(kappa_fam["Random"])
    print(f"{'Family':<18} {'Lift k=5.0':>12} {'Lift k=0.5':>12}")
    print("-" * 50)
    for fam in fam_order:
        if fam == "Random":
            continue
        ws_lift = np.mean(ws_fam.get(fam, [0])) / ws_rand
        k_lift = np.mean(kappa_fam.get(fam, [0])) / k_rand
        print(f"{fam:<18} {ws_lift:>11.2f}x {k_lift:>11.2f}x")

    print()
    print("=" * 80)
    print("RANKINGS COMPARISON")
    print("=" * 80)
    ws_ranked = sorted(
        [(fam, np.mean(ws_fam[fam])) for fam in fam_order if fam != "Random"],
        key=lambda x: -x[1],
    )
    k_ranked = sorted(
        [(fam, np.mean(kappa_fam[fam])) for fam in fam_order if fam != "Random"],
        key=lambda x: -x[1],
    )
    print("kappa=5.0:", " > ".join(f"{f} ({m:.3f})" for f, m in ws_ranked))
    print("kappa=0.5:", " > ".join(f"{f} ({m:.3f})" for f, m in k_ranked))

    print()
    print("=" * 80)
    print("UCB STRATEGIES (most affected by kappa)")
    print("=" * 80)
    print(f"{'Strategy':<35} {'k=5.0':>8} {'k=0.5':>8} {'Delta':>8}")
    print("-" * 65)
    ucb_deltas = []
    for strat in sorted(UCB_STRATEGIES):
        if strat in ws_strat and strat in kappa_strat:
            ws_m = np.mean(ws_strat[strat])
            k_m = np.mean(kappa_strat[strat])
            d = k_m - ws_m
            ucb_deltas.append(d)
            print(f"{strat:<35} {ws_m:>8.3f} {k_m:>8.3f} {d:>+8.3f}")
    print(f"{'Mean UCB delta':<35} {'':>8} {'':>8} {np.mean(ucb_deltas):>+8.3f}")

    print()
    print("=" * 80)
    print("NON-UCB STRATEGIES (should be ~invariant)")
    print("=" * 80)
    print(f"{'Strategy':<35} {'k=5.0':>8} {'k=0.5':>8} {'Delta':>8}")
    print("-" * 65)
    inv_deltas = []
    for strat in sorted(INVARIANT_STRATEGIES):
        if strat in ws_strat and strat in kappa_strat:
            ws_m = np.mean(ws_strat[strat])
            k_m = np.mean(kappa_strat[strat])
            d = k_m - ws_m
            inv_deltas.append(d)
            print(f"{strat:<35} {ws_m:>8.3f} {k_m:>8.3f} {d:>+8.3f}")
    print(f"{'Mean invariant delta':<35} {'':>8} {'':>8} {np.mean(inv_deltas):>+8.3f}")

    # Spearman rank correlation between family rankings
    from scipy.stats import spearmanr
    ws_order = [np.mean(ws_fam[f]) for f in fam_order if f != "Random"]
    k_order = [np.mean(kappa_fam[f]) for f in fam_order if f != "Random"]
    rho, p = spearmanr(ws_order, k_order)
    print(f"\nSpearman rank correlation between family rankings: rho={rho:.3f}, p={p:.3f}")

    # Show broken run detail
    print()
    print("=" * 80)
    print("BROKEN RUN ANALYSIS (kappa_optimal)")
    print("=" * 80)
    from collections import Counter
    broken_strats = Counter()
    for r in kappa_results:
        strat = r.get("strategy", "")
        if strat == "random":
            continue
        nr = r.get("result", {}).get("metrics", {}).get("n_rounds", -1)
        if nr == 0:
            broken_strats[strat] += 1
    for s, c in broken_strats.most_common():
        print(f"  {s}: {c} broken")

    # GP (BoTorch) excluding Tanimoto
    print()
    print("=" * 80)
    print("GP (BoTorch) EXCLUDING Tanimoto strategies")
    print("=" * 80)
    TANIMOTO = {"lnpbo_tanimoto_ts", "lnpbo_tanimoto_logei"}
    def _gp_recalls(results):
        out = []
        for r in results:
            strat = r.get("strategy", "")
            if strat not in STRATEGY_FAMILY or STRATEGY_FAMILY[strat] != "GP (BoTorch)":
                continue
            if strat in TANIMOTO:
                continue
            v = r.get("result", {}).get("metrics", {}).get("top_k_recall", {}).get("5")
            if v is not None:
                out.append(v)
        return out

    ws_gp_no_tani = _gp_recalls(ws_results)
    k_gp_no_tani = _gp_recalls(kappa_results)
    print(f"  WS (k=5.0): {np.mean(ws_gp_no_tani):.3f} (n={len(ws_gp_no_tani)})")
    print(f"  KO (k=0.5): {np.mean(k_gp_no_tani):.3f} (n={len(k_gp_no_tani)})")
    print(f"  Delta: {np.mean(k_gp_no_tani) - np.mean(ws_gp_no_tani):+.3f}")

    # GP (BoTorch) UCB-only
    GP_UCB = {"lnpbo_ucb"}
    ws_gp_ucb = [v for r in ws_results
                 for strat in [r.get("strategy", "")]
                 if strat in GP_UCB
                 for v in [r.get("result", {}).get("metrics", {}).get("top_k_recall", {}).get("5")]
                 if v is not None and r.get("result", {}).get("metrics", {}).get("n_rounds", 0) > 0]
    k_gp_ucb = [v for r in kappa_results
                for strat in [r.get("strategy", "")]
                if strat in GP_UCB
                for v in [r.get("result", {}).get("metrics", {}).get("top_k_recall", {}).get("5")]
                if v is not None and r.get("result", {}).get("metrics", {}).get("n_rounds", 0) > 0]
    print("\n  GP-UCB only:")
    print(f"    WS (k=5.0): {np.mean(ws_gp_ucb):.3f} (n={len(ws_gp_ucb)})")
    print(f"    KO (k=0.5): {np.mean(k_gp_ucb):.3f} (n={len(k_gp_ucb)})")
    print(f"    Delta: {np.mean(k_gp_ucb) - np.mean(ws_gp_ucb):+.3f}")


if __name__ == "__main__":
    main()
