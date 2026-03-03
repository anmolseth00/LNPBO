#!/usr/bin/env python3
"""
LNPBO Benchmark: Simulated Closed-Loop Bayesian Optimization
=============================================================
Uses LNPDB as a ground-truth oracle to evaluate BO strategies
without running real experiments.

Workflow:
  1. Load and encode LNPDB (~19,800 formulations)
  2. Split into seed pool (initial training data) and discovery pool (oracle)
  3. At each round: optimizer suggests a batch → oracle returns true values
  4. Track metrics: best-so-far, top-K recall, AUC

Strategies:
  random            – uniform random sampling from discovery pool
  lnpbo_ucb         – GP + Kriging Believer + UCB (current default)
  lnpbo_ei          – GP + Kriging Believer + EI
  lnpbo_logei       – GP + Kriging Believer + LogEI (log-space EI)
  lnpbo_lp_ei       – GP + Local Penalization + EI
  lnpbo_lp_logei    – GP + Local Penalization + LogEI
  lnpbo_pls_logei   – PLS encoding + Kriging Believer + LogEI
  lnpbo_pls_lp_logei – PLS encoding + Local Penalization + LogEI

Usage:
  python benchmark.py --strategies random,lnpbo_ucb --rounds 5 --batch-size 12 --seed 42
  python benchmark.py --strategies all --rounds 10 --seed 42 --n-seeds 500
"""

import sys
import argparse
import time
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT.parent))

ALL_STRATEGIES = [
    "random",
    "lnpbo_ucb",
    "lnpbo_ei",
    "lnpbo_logei",
    "lnpbo_lp_ei",
    "lnpbo_lp_logei",
    "lnpbo_pls_logei",
    "lnpbo_pls_lp_logei",
]

STRATEGY_TO_ACQ = {
    "lnpbo_ucb": "UCB",
    "lnpbo_ei": "EI",
    "lnpbo_logei": "LogEI",
    "lnpbo_lp_ei": "LP_EI",
    "lnpbo_lp_logei": "LP_LogEI",
    "lnpbo_pls_logei": "LogEI",
    "lnpbo_pls_lp_logei": "LP_LogEI",
}

PLS_STRATEGIES = {"lnpbo_pls_logei", "lnpbo_pls_lp_logei"}


# ---------------------------------------------------------------------------
# Oracle: LNPDB as ground-truth lookup
# ---------------------------------------------------------------------------

class LNPDBOracle:
    """Wraps the full encoded LNPDB for oracle lookup."""

    def __init__(self, encoded_df, feature_cols):
        self.df = encoded_df.copy()
        self.feature_cols = feature_cols
        self._nn = None

    def _build_nn(self, pool_indices):
        X = self.df.loc[pool_indices, self.feature_cols].values
        nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
        nn.fit(X)
        return nn, pool_indices

    def lookup(self, suggestion_features, pool_indices):
        """Find nearest oracle formulation and return its index + Experiment_value."""
        nn, idx_list = self._build_nn(pool_indices)
        idx_arr = np.array(idx_list)
        x = np.atleast_2d(suggestion_features)
        _, nn_idx = nn.kneighbors(x)
        matched_idx = idx_arr[nn_idx.ravel()]
        return matched_idx

    def get_value(self, idx):
        return self.df.loc[idx, "Experiment_value"].values


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_benchmark_data(n_seed=500, random_seed=42, subset=None, reduction="pca"):
    """Load LNPDB, encode, split into seed/oracle.

    Note on PLS target leakage: When reduction="pls", the PLS projection is
    fitted on the full dataset's per-lipid average Experiment_values BEFORE the
    seed/oracle split. This means PLS sees oracle targets during encoding,
    giving it an informational advantage over PCA (which only sees X). This is
    acceptable for comparing acquisition strategies under a fixed encoding, but
    overstates PLS's benefit in a real prospective setting where only seed
    targets would be available. A future improvement would re-fit PLS each
    round using only the current training set's targets.
    """
    from LNPBO.data.lnpdb_bridge import load_lnpdb_full
    from LNPBO.data.dataset import Dataset
    from LNPBO.space.formulation import FormulationSpace

    print(f"Loading LNPDB (reduction={reduction})...")
    dataset = load_lnpdb_full()
    df = dataset.df

    if subset and subset < len(df):
        df = df.sample(n=subset, random_state=random_seed).reset_index(drop=True)
        df["Formulation_ID"] = range(1, len(df) + 1)
        dataset = Dataset(df, source="lnpdb", name="LNPDB_benchmark")

    print(f"  {len(df):,} formulations loaded")
    print(f"  Experiment_value range: [{df['Experiment_value'].min():.2f}, {df['Experiment_value'].max():.2f}]")

    # Determine encoding: encode variable components with >1 unique SMILES
    def _should_encode(role, n_pcs):
        smiles_col = f"{role}_SMILES"
        name_col = f"{role}_name"
        if df[name_col].nunique() <= 1:
            return 0
        if smiles_col not in df.columns:
            return 0
        if df[smiles_col].dropna().nunique() <= 1:
            return 0
        return n_pcs

    il_pcs = _should_encode("IL", 5)
    hl_pcs = _should_encode("HL", 3)
    chl_pcs = _should_encode("CHL", 3)
    peg_pcs = _should_encode("PEG", 3)

    print(f"  Encoding PCs: IL={il_pcs}, HL={hl_pcs}, CHL={chl_pcs}, PEG={peg_pcs}")

    encoded = dataset.encode_dataset(
        IL_n_pcs_morgan=il_pcs,
        HL_n_pcs_morgan=hl_pcs,
        CHL_n_pcs_morgan=chl_pcs,
        PEG_n_pcs_morgan=peg_pcs,
        reduction=reduction,
    )

    # Identify feature columns (PCs + molar ratios + mass ratio)
    feature_cols = []
    for role in ["IL", "HL", "CHL", "PEG"]:
        role_cols = [c for c in encoded.df.columns if c.startswith(f"{role}_mfp_pc")]
        feature_cols.extend(sorted(role_cols))
    for role in ["IL", "HL", "CHL", "PEG"]:
        col = f"{role}_molratio"
        if col in encoded.df.columns and encoded.df[col].nunique() > 1:
            feature_cols.append(col)
    if "IL_to_nucleicacid_massratio" in encoded.df.columns:
        if encoded.df["IL_to_nucleicacid_massratio"].nunique() > 1:
            feature_cols.append("IL_to_nucleicacid_massratio")

    print(f"  Feature columns ({len(feature_cols)}): {feature_cols}")

    # Drop rows with NaN in feature columns and deduplicate
    valid_mask = encoded.df[feature_cols].notna().all(axis=1)
    encoded_df = encoded.df[valid_mask].copy()
    # Encoding merge can inflate rows if lipid name maps to multiple SMILES;
    # deduplicate on Formulation_ID to keep original row count
    if "Formulation_ID" in encoded_df.columns:
        encoded_df = encoded_df.drop_duplicates(subset=["Formulation_ID"])
    encoded_df = encoded_df.reset_index(drop=True)
    print(f"  Valid rows after cleanup: {len(encoded_df):,}")

    # Split into seed and oracle
    rng = np.random.RandomState(random_seed)
    all_idx = np.arange(len(encoded_df))
    rng.shuffle(all_idx)

    seed_idx = sorted(all_idx[:n_seed])
    oracle_idx = sorted(all_idx[n_seed:])

    # Identify top-K formulations in the full dataset for recall calculation
    top_k_values = {
        10: set(encoded_df.nlargest(10, "Experiment_value").index),
        50: set(encoded_df.nlargest(50, "Experiment_value").index),
        100: set(encoded_df.nlargest(100, "Experiment_value").index),
    }

    print(f"  Seed pool: {len(seed_idx)} formulations")
    print(f"  Oracle pool: {len(oracle_idx)} formulations")
    print(f"  Top-10 Experiment_value threshold: {encoded_df['Experiment_value'].nlargest(10).min():.3f}")

    return encoded, encoded_df, feature_cols, seed_idx, oracle_idx, top_k_values


# ---------------------------------------------------------------------------
# Strategy runners
# ---------------------------------------------------------------------------

def run_random(oracle_df, seed_idx, oracle_idx, batch_size, n_rounds, seed):
    """Random baseline: uniformly sample from oracle pool."""
    rng = np.random.RandomState(seed)
    training_idx = list(seed_idx)
    pool_idx = list(oracle_idx)
    history = _init_history(oracle_df, training_idx)

    for r in range(n_rounds):
        if len(pool_idx) < batch_size:
            break
        chosen = rng.choice(len(pool_idx), size=batch_size, replace=False)
        batch_idx = [pool_idx[i] for i in sorted(chosen)]
        for i in sorted(chosen, reverse=True):
            pool_idx.pop(i)
        training_idx.extend(batch_idx)
        _update_history(history, oracle_df, training_idx, batch_idx, r)

    return history


def run_bo_strategy(
    encoded_dataset, encoded_df, feature_cols,
    seed_idx, oracle_idx,
    acq_type, batch_size, n_rounds, seed,
    kappa=5.0, xi=0.01,
):
    """Run a BO strategy with oracle lookup."""
    from LNPBO.data.dataset import Dataset
    from LNPBO.space.formulation import FormulationSpace
    from LNPBO.optimization.bayesopt import perform_bayesian_optimization

    training_idx = list(seed_idx)
    pool_idx = list(oracle_idx)
    oracle = LNPDBOracle(encoded_df, feature_cols)
    history = _init_history(encoded_df, training_idx)

    for r in range(n_rounds):
        if len(pool_idx) < batch_size:
            break

        # Build dataset from current training data
        train_df = encoded_df.loc[training_idx].copy().reset_index(drop=True)
        train_df["Formulation_ID"] = range(1, len(train_df) + 1)
        train_df["Round"] = 0

        dataset = Dataset(
            train_df, source="lnpdb", name="benchmark_train",
            metadata=encoded_dataset.metadata,
            encoders=encoded_dataset.encoders,
            fitted_transformers=encoded_dataset.fitted_transformers,
        )

        space = FormulationSpace.from_dataset(dataset)

        try:
            suggestions = perform_bayesian_optimization(
                data=dataset.df,
                formulation_space=space,
                round_number=r,
                acq_type=acq_type,
                BATCH_SIZE=batch_size,
                RANDOM_STATE_SEED=seed,
                KAPPA=kappa,
                XI=xi,
                verbose=0,
                save_gp=False,
            )
        except Exception as e:
            print(f"  Round {r}: BO failed ({e}), skipping")
            break

        # Match suggestions to nearest oracle formulations
        suggestion_features = suggestions[feature_cols].values
        matched_idx = oracle.lookup(suggestion_features, pool_idx)

        # Deduplicate (NN may map multiple suggestions to same oracle point)
        unique_matched = list(dict.fromkeys(matched_idx))[:batch_size]

        # Remove matched from pool, add to training
        for idx in unique_matched:
            if idx in pool_idx:
                pool_idx.remove(idx)
        training_idx.extend(unique_matched)
        _update_history(history, encoded_df, training_idx, unique_matched, r)

    return history


# ---------------------------------------------------------------------------
# History / metrics tracking
# ---------------------------------------------------------------------------

def _init_history(df, seed_idx):
    seed_vals = df.loc[seed_idx, "Experiment_value"]
    return {
        "best_so_far": [float(seed_vals.max())],
        "round_best": [],
        "n_evaluated": [len(seed_idx)],
        "all_evaluated": set(seed_idx),
    }


def _update_history(history, df, training_idx, batch_idx, round_num):
    batch_vals = df.loc[batch_idx, "Experiment_value"]
    all_vals = df.loc[training_idx, "Experiment_value"]
    history["best_so_far"].append(float(all_vals.max()))
    history["round_best"].append(float(batch_vals.max()))
    history["n_evaluated"].append(len(training_idx))
    history["all_evaluated"].update(batch_idx)


def compute_metrics(history, top_k_values, n_total):
    """Compute summary metrics from a history dict."""
    bsf = np.array(history["best_so_far"])
    n_eval = np.array(history["n_evaluated"])
    evaluated = history["all_evaluated"]

    # AUC of best-so-far curve (normalized by number of rounds)
    _trapezoid = np.trapezoid
    auc = float(_trapezoid(bsf, n_eval) / (n_eval[-1] - n_eval[0])) if len(bsf) > 1 else bsf[0]

    # Top-K recall
    recall = {}
    for k, top_set in top_k_values.items():
        found = len(evaluated & top_set)
        recall[k] = found / len(top_set)

    # Acceleration factor: how many random samples needed to find the same best?
    # (approximated as: best_so_far percentile * n_total)
    final_best = bsf[-1]

    return {
        "final_best": float(final_best),
        "auc": auc,
        "top_k_recall": recall,
        "n_rounds": len(bsf) - 1,
        "n_total_evaluated": int(n_eval[-1]),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(all_results, output_path="benchmark_results.png"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Best-so-far traces
    ax1 = axes[0]
    for name, result in all_results.items():
        bsf = result["history"]["best_so_far"]
        n_eval = result["history"]["n_evaluated"]
        ax1.plot(n_eval, bsf, "o-", label=name, markersize=3)
    ax1.set_xlabel("Formulations evaluated")
    ax1.set_ylabel("Best Experiment_value found")
    ax1.set_title("Incumbent Trace")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Top-K recall
    ax2 = axes[1]
    k_values = sorted(next(iter(all_results.values()))["metrics"]["top_k_recall"].keys())
    x = np.arange(len(k_values))
    width = 0.8 / len(all_results)
    for i, (name, result) in enumerate(all_results.items()):
        recalls = [result["metrics"]["top_k_recall"][k] for k in k_values]
        ax2.bar(x + i * width, recalls, width, label=name)
    ax2.set_xlabel("K")
    ax2.set_ylabel("Top-K Recall")
    ax2.set_title("Top-K Recall")
    ax2.set_xticks(x + width * (len(all_results) - 1) / 2)
    ax2.set_xticklabels([str(k) for k in k_values])
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="LNPBO Benchmark: Simulated closed-loop BO evaluation"
    )
    parser.add_argument(
        "--strategies", type=str, default="random,lnpbo_ucb",
        help=f"Comma-separated strategies to run (or 'all'). Options: {','.join(ALL_STRATEGIES)}",
    )
    parser.add_argument("--rounds", type=int, default=5, help="Number of BO rounds (default: 5)")
    parser.add_argument("--batch-size", type=int, default=12, help="Batch size per round (default: 12)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--n-seeds", type=int, default=500, help="Size of initial seed pool (default: 500)")
    parser.add_argument("--subset", type=int, default=None, help="Use a subset of LNPDB (for fast testing)")
    parser.add_argument("--kappa", type=float, default=5.0, help="UCB kappa (default: 5.0)")
    parser.add_argument("--xi", type=float, default=0.01, help="EI/LogEI xi (default: 0.01)")
    parser.add_argument("--output", type=str, default="benchmark_results", help="Output prefix (default: benchmark_results)")
    parser.add_argument("--no-plot", action="store_true", help="Skip plotting")
    args = parser.parse_args()

    # Parse strategies
    if args.strategies == "all":
        strategies = ALL_STRATEGIES
    else:
        strategies = [s.strip() for s in args.strategies.split(",")]
        for s in strategies:
            if s not in ALL_STRATEGIES:
                parser.error(f"Unknown strategy: {s}. Choose from: {ALL_STRATEGIES}")

    print("=" * 70)
    print("LNPBO BENCHMARK")
    print("=" * 70)
    print(f"Strategies: {strategies}")
    print(f"Rounds: {args.rounds}, Batch size: {args.batch_size}")
    print(f"Seed pool: {args.n_seeds}, Random seed: {args.seed}")
    print()

    # Prepare data (PCA encoding for most strategies)
    pca_data = prepare_benchmark_data(
        n_seed=args.n_seeds, random_seed=args.seed, subset=args.subset, reduction="pca",
    )
    encoded_dataset, encoded_df, feature_cols, seed_idx, oracle_idx, top_k_values = pca_data

    # Prepare PLS-encoded data if any PLS strategies requested
    pls_data = None
    if any(s in PLS_STRATEGIES for s in strategies):
        pls_data = prepare_benchmark_data(
            n_seed=args.n_seeds, random_seed=args.seed, subset=args.subset, reduction="pls",
        )

    # Run strategies
    all_results = {}
    for strategy in strategies:
        print(f"\n{'='*70}")
        print(f"Running: {strategy}")
        print(f"{'='*70}")
        t0 = time.time()

        # Select PCA or PLS data for this strategy
        if strategy in PLS_STRATEGIES and pls_data is not None:
            s_dataset, s_df, s_fcols, s_seed, s_oracle, s_topk = pls_data
        else:
            s_dataset, s_df, s_fcols, s_seed, s_oracle, s_topk = pca_data

        if strategy == "random":
            history = run_random(
                s_df, s_seed, s_oracle,
                batch_size=args.batch_size, n_rounds=args.rounds, seed=args.seed,
            )
        else:
            acq_type = STRATEGY_TO_ACQ[strategy]
            history = run_bo_strategy(
                s_dataset, s_df, s_fcols,
                s_seed, s_oracle,
                acq_type=acq_type,
                batch_size=args.batch_size, n_rounds=args.rounds, seed=args.seed,
                kappa=args.kappa, xi=args.xi,
            )

        elapsed = time.time() - t0
        metrics = compute_metrics(history, s_topk, len(s_df))

        all_results[strategy] = {
            "history": history,
            "metrics": metrics,
            "elapsed": elapsed,
        }

        print(f"  Time: {elapsed:.1f}s")
        print(f"  Final best: {metrics['final_best']:.4f}")
        print(f"  AUC: {metrics['auc']:.4f}")
        print(f"  Top-K recall: { {k: f'{v:.1%}' for k, v in metrics['top_k_recall'].items()} }")

    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    header = f"{'Strategy':<24} {'Final Best':>12} {'AUC':>10} {'Top-10':>8} {'Top-50':>8} {'Top-100':>8} {'Time':>8}"
    print(header)
    print("-" * len(header))
    for name, result in all_results.items():
        m = result["metrics"]
        r = m["top_k_recall"]
        print(f"{name:<24} {m['final_best']:>12.4f} {m['auc']:>10.4f} "
              f"{r.get(10, 0):>7.1%} {r.get(50, 0):>7.1%} {r.get(100, 0):>7.1%} "
              f"{result['elapsed']:>7.1f}s")

    # Save results JSON
    json_path = f"{args.output}.json"
    serializable = {}
    for name, result in all_results.items():
        serializable[name] = {
            "metrics": result["metrics"],
            "elapsed": result["elapsed"],
            "best_so_far": result["history"]["best_so_far"],
            "n_evaluated": result["history"]["n_evaluated"],
        }
    with open(json_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {json_path}")

    # Plot
    if not args.no_plot:
        plot_results(all_results, output_path=f"{args.output}.png")


if __name__ == "__main__":
    main()
