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
  random             - uniform random sampling from discovery pool
  lnpbo_ucb          - GP + Kriging Believer + UCB (current default)
  lnpbo_ei           - GP + Kriging Believer + EI
  lnpbo_logei        - GP + Kriging Believer + LogEI (log-space EI)
  lnpbo_lp_ei        - GP + Local Penalization + EI
  lnpbo_lp_logei     - GP + Local Penalization + LogEI
  lnpbo_pls_logei    - PLS encoding + Kriging Believer + LogEI
  lnpbo_pls_lp_logei - PLS encoding + Local Penalization + LogEI

Usage:
  python benchmark.py --strategies random,lnpbo_ucb --rounds 5 --batch-size 12 --seed 42
  python benchmark.py --strategies all --rounds 10 --seed 42 --n-seeds 500
"""

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import norm as _norm
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
    "discrete_gp_ucb",
    "discrete_rf_ucb",
    "discrete_rf_ts",
    "discrete_xgb_greedy",
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

DISCRETE_STRATEGIES = {"discrete_gp_ucb", "discrete_rf_ucb", "discrete_rf_ts", "discrete_xgb_greedy"}


# ---------------------------------------------------------------------------
# Target normalization
# ---------------------------------------------------------------------------


def _copula_transform(values):
    """Rank-based copula transform to standard normal.

    Maps targets through their empirical CDF, then applies the inverse
    normal CDF (probit). This removes outlier effects and gives the GP
    a well-behaved target distribution.
    """
    import pandas as pd

    n = len(values)
    ranks = pd.Series(values).rank(method="average")
    u = (ranks - 0.5) / n
    return _norm.ppf(u)


def _normalize_targets(df, method):
    """Apply target normalization to a dataframe in-place."""
    if method == "none":
        return
    col = "Experiment_value"
    if method == "copula":
        df[col] = _copula_transform(df[col].values)
    elif method == "zscore":
        mu, sigma = df[col].mean(), df[col].std()
        if sigma > 0:
            df[col] = (df[col] - mu) / sigma


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


def prepare_benchmark_data(n_seed=500, random_seed=42, subset=None, reduction="pca", feature_type="mfp"):
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
    from LNPBO.data.dataset import Dataset
    from LNPBO.data.lnpdb_bridge import load_lnpdb_full

    print(f"Loading LNPDB (reduction={reduction}, features={feature_type})...")
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

    # Route to correct encoder parameter based on feature_type
    encode_kwargs = {}
    param_suffix = {"mfp": "morgan", "mordred": "mordred", "unimol": "unimol"}[feature_type]
    for role, n in [("IL", il_pcs), ("HL", hl_pcs), ("CHL", chl_pcs), ("PEG", peg_pcs)]:
        encode_kwargs[f"{role}_n_pcs_{param_suffix}"] = n

    encoded = dataset.encode_dataset(
        **encode_kwargs,
        reduction=reduction,
    )

    # Identify feature columns (PCs + molar ratios + mass ratio)
    feature_cols = []
    pc_prefix = {"mfp": "mfp_pc", "mordred": "mordred_pc", "unimol": "unimol_pc"}[feature_type]
    for role in ["IL", "HL", "CHL", "PEG"]:
        role_cols = [c for c in encoded.df.columns if c.startswith(f"{role}_{pc_prefix}")]
        feature_cols.extend(sorted(role_cols))
    for role in ["IL", "HL", "CHL", "PEG"]:
        col = f"{role}_molratio"
        if col in encoded.df.columns and encoded.df[col].nunique() > 1:
            feature_cols.append(col)
    if "IL_to_nucleicacid_massratio" in encoded.df.columns and encoded.df["IL_to_nucleicacid_massratio"].nunique() > 1:
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
    encoded_dataset,
    encoded_df,
    feature_cols,
    seed_idx,
    oracle_idx,
    acq_type,
    batch_size,
    n_rounds,
    seed,
    kappa=5.0,
    xi=0.01,
    normalize="copula",
):
    """Run a BO strategy with oracle lookup."""
    from LNPBO.data.dataset import Dataset
    from LNPBO.optimization.bayesopt import perform_bayesian_optimization
    from LNPBO.space.formulation import FormulationSpace

    training_idx = list(seed_idx)
    pool_idx = list(oracle_idx)
    oracle = LNPDBOracle(encoded_df, feature_cols)
    history = _init_history(encoded_df, training_idx)

    for r in range(n_rounds):
        if len(pool_idx) < batch_size:
            break

        # Build dataset from current training data (original values)
        train_df = encoded_df.loc[training_idx].copy().reset_index(drop=True)
        train_df["Formulation_ID"] = range(1, len(train_df) + 1)
        train_df["Round"] = 0

        # Normalize targets for the GP (metrics still use original values)
        _normalize_targets(train_df, normalize)

        dataset = Dataset(
            train_df,
            source="lnpdb",
            name="benchmark_train",
            metadata=encoded_dataset.metadata,
            encoders=encoded_dataset.encoders,
            fitted_transformers=encoded_dataset.fitted_transformers,
        )

        space = FormulationSpace.from_dataset(dataset)

        try:
            _devnull = open(os.devnull, "w")
            _saved_stdout = sys.stdout
            sys.stdout = _devnull
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
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
            finally:
                sys.stdout = _saved_stdout
                _devnull.close()
        except Exception as e:
            print(f"  Round {r}: BO failed ({e})", flush=True)
            import traceback
            traceback.print_exc()
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

        batch_best = encoded_df.loc[unique_matched, "Experiment_value"].max()
        cum_best = history["best_so_far"][-1]
        print(f"  Round {r+1}: batch_best={batch_best:.3f}, cum_best={cum_best:.3f}, n_new={len(unique_matched)}", flush=True)

    return history


def run_discrete_strategy(
    encoded_df,
    feature_cols,
    seed_idx,
    oracle_idx,
    strategy,
    batch_size,
    n_rounds,
    seed,
    kappa=5.0,
    normalize="copula",
):
    """Run a discrete candidate pool strategy.

    Instead of optimizing an acquisition function in continuous space and
    NN-matching, directly score all remaining candidates and pick the top batch.
    """
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.preprocessing import MinMaxScaler

    training_idx = list(seed_idx)
    pool_idx = list(oracle_idx)
    history = _init_history(encoded_df, training_idx)

    for r in range(n_rounds):
        if len(pool_idx) < batch_size:
            break

        X_train = encoded_df.loc[training_idx, feature_cols].values
        y_train = encoded_df.loc[training_idx, "Experiment_value"].values
        X_pool = encoded_df.loc[pool_idx, feature_cols].values

        scaler = MinMaxScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_pool_s = scaler.transform(X_pool)

        if strategy == "discrete_gp_ucb":
            y_fit = _copula_transform(y_train) if normalize == "copula" else y_train.copy()
            if normalize == "zscore":
                mu, sigma = y_fit.mean(), y_fit.std()
                if sigma > 0:
                    y_fit = (y_fit - mu) / sigma
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gp = GaussianProcessRegressor(alpha=1e-6, n_restarts_optimizer=10, random_state=seed)
                gp.fit(X_train_s, y_fit)
                mu_pool, sigma_pool = gp.predict(X_pool_s, return_std=True)
            scores = mu_pool + kappa * sigma_pool

        elif strategy == "discrete_rf_ucb":
            from sklearn.ensemble import RandomForestRegressor
            rf = RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=-1)
            rf.fit(X_train_s, y_train)
            tree_preds = np.array([t.predict(X_pool_s) for t in rf.estimators_])
            mu_pool = tree_preds.mean(axis=0)
            sigma_pool = tree_preds.std(axis=0)
            scores = mu_pool + kappa * sigma_pool

        elif strategy == "discrete_rf_ts":
            from sklearn.ensemble import RandomForestRegressor
            rf = RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=-1)
            rf.fit(X_train_s, y_train)
            rng = np.random.RandomState(seed + r)
            tree_idx = rng.randint(0, len(rf.estimators_))
            scores = rf.estimators_[tree_idx].predict(X_pool_s)

        elif strategy == "discrete_xgb_greedy":
            from xgboost import XGBRegressor
            xgb = XGBRegressor(n_estimators=200, random_state=seed, n_jobs=-1, verbosity=0)
            xgb.fit(X_train_s, y_train)
            scores = xgb.predict(X_pool_s)

        else:
            raise ValueError(f"Unknown discrete strategy: {strategy}")

        top_k = np.argsort(scores)[-batch_size:][::-1]
        batch_idx = [pool_idx[i] for i in top_k]

        for idx in batch_idx:
            pool_idx.remove(idx)
        training_idx.extend(batch_idx)
        _update_history(history, encoded_df, training_idx, batch_idx, r)

        batch_best = encoded_df.loc[batch_idx, "Experiment_value"].max()
        cum_best = history["best_so_far"][-1]
        print(f"  Round {r+1}: batch_best={batch_best:.3f}, cum_best={cum_best:.3f}, n_new={len(batch_idx)}", flush=True)

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


STRATEGY_DISPLAY = {
    "random": "Random",
    "lnpbo_ucb": "GP + KB (UCB)",
    "lnpbo_ei": "GP + KB (EI)",
    "lnpbo_logei": "GP + KB (LogEI)",
    "lnpbo_lp_ei": "GP + LP (EI)",
    "lnpbo_lp_logei": "GP + LP (LogEI)",
    "lnpbo_pls_logei": "GP + KB (PLS+LogEI)",
    "lnpbo_pls_lp_logei": "GP + LP (PLS+LogEI)",
    "discrete_gp_ucb": "Discrete GP-UCB",
    "discrete_rf_ucb": "Discrete RF-UCB",
    "discrete_rf_ts": "Discrete RF-TS",
    "discrete_xgb_greedy": "Discrete XGB",
}

STRATEGY_COLORS = {
    "random": "#999999",
    "lnpbo_ucb": "#1f77b4",
    "lnpbo_ei": "#ff7f0e",
    "lnpbo_logei": "#2ca02c",
    "lnpbo_lp_ei": "#d62728",
    "lnpbo_lp_logei": "#9467bd",
    "lnpbo_pls_logei": "#8c564b",
    "lnpbo_pls_lp_logei": "#e377c2",
    "discrete_gp_ucb": "#17becf",
    "discrete_rf_ucb": "#bcbd22",
    "discrete_rf_ts": "#7f7f7f",
    "discrete_xgb_greedy": "#e41a1c",
}


def plot_results(all_results, output_path="benchmark_output.png"):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.pad": 4,
            "ytick.major.pad": 4,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Panel A: Incumbent trace
    ax1 = axes[0]
    for name, result in all_results.items():
        bsf = result["history"]["best_so_far"]
        n_eval = result["history"]["n_evaluated"]
        label = STRATEGY_DISPLAY.get(name, name)
        color = STRATEGY_COLORS.get(name)
        style = "--" if name == "random" else "-"
        ax1.plot(n_eval, bsf, style, label=label, color=color, linewidth=1.5, markersize=0)
    ax1.set_xlabel("Formulations evaluated")
    ax1.set_ylabel("Best value found")
    ax1.legend(fontsize=7.5, frameon=True, fancybox=False, edgecolor="#cccccc", loc="lower right")
    ax1.grid(True, alpha=0.15, linewidth=0.5)
    ax1.text(-0.12, 1.05, "A", transform=ax1.transAxes, fontsize=14, fontweight="bold", va="top")

    # Panel B: Top-K recall grouped bar chart
    ax2 = axes[1]
    k_values = sorted(next(iter(all_results.values()))["metrics"]["top_k_recall"].keys())
    x = np.arange(len(k_values))
    n_strats = len(all_results)
    width = 0.8 / n_strats
    for i, (name, result) in enumerate(all_results.items()):
        recalls = [result["metrics"]["top_k_recall"][k] for k in k_values]
        label = STRATEGY_DISPLAY.get(name, name)
        color = STRATEGY_COLORS.get(name)
        ax2.bar(x + i * width, recalls, width, label=label, color=color, edgecolor="white", linewidth=0.3)
    ax2.set_xlabel("K (top-K formulations)")
    ax2.set_ylabel("Recall")
    ax2.set_xticks(x + width * (n_strats - 1) / 2)
    ax2.set_xticklabels([str(k) for k in k_values])
    ax2.set_ylim(0, 1.0)
    ax2.legend(fontsize=7.5, frameon=True, fancybox=False, edgecolor="#cccccc")
    ax2.grid(True, alpha=0.15, linewidth=0.5, axis="y")
    ax2.text(-0.12, 1.05, "B", transform=ax2.transAxes, fontsize=14, fontweight="bold", va="top")

    fig.tight_layout(w_pad=3)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="LNPBO Benchmark: Simulated closed-loop BO evaluation")
    parser.add_argument(
        "--strategies",
        type=str,
        default="random,lnpbo_ucb",
        help=f"Comma-separated strategies to run (or 'all'). Options: {','.join(ALL_STRATEGIES)}",
    )
    parser.add_argument("--rounds", type=int, default=15, help="Number of BO rounds (default: 15)")
    parser.add_argument("--batch-size", type=int, default=12, help="Batch size per round (default: 12)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--n-seeds", type=int, default=50, help="Size of initial seed pool (default: 50)")
    parser.add_argument("--subset", type=int, default=None, help="Use a subset of LNPDB (for fast testing)")
    parser.add_argument("--kappa", type=float, default=5.0, help="UCB kappa (default: 5.0)")
    parser.add_argument("--xi", type=float, default=0.01, help="EI/LogEI xi (default: 0.01)")
    parser.add_argument(
        "--normalize",
        type=str,
        default="copula",
        choices=["none", "zscore", "copula"],
        help="Target normalization for GP (default: copula)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_output",
        help="Output prefix (default: benchmark_output)",
    )
    parser.add_argument("--no-plot", action="store_true", help="Skip plotting")
    parser.add_argument(
        "--feature-type",
        type=str,
        default="mfp",
        choices=["mfp", "mordred", "unimol"],
        help="Molecular feature type (default: mfp)",
    )
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
    print(f"Target normalization: {args.normalize}")
    print()

    # Prepare data (PCA encoding for most strategies)
    pca_data = prepare_benchmark_data(
        n_seed=args.n_seeds,
        random_seed=args.seed,
        subset=args.subset,
        reduction="pca",
        feature_type=args.feature_type,
    )
    # Prepare PLS-encoded data if any PLS strategies requested
    pls_data = None
    if any(s in PLS_STRATEGIES for s in strategies):
        pls_data = prepare_benchmark_data(
            n_seed=args.n_seeds,
            random_seed=args.seed,
            subset=args.subset,
            reduction="pls",
            feature_type=args.feature_type,
        )

    # Run strategies
    all_results = {}
    for strategy in strategies:
        print(f"\n{'=' * 70}")
        print(f"Running: {strategy}")
        print(f"{'=' * 70}")
        t0 = time.time()

        # Select PCA or PLS data for this strategy
        if strategy in PLS_STRATEGIES and pls_data is not None:
            s_dataset, s_df, s_fcols, s_seed, s_oracle, s_topk = pls_data
        else:
            s_dataset, s_df, s_fcols, s_seed, s_oracle, s_topk = pca_data

        if strategy == "random":
            history = run_random(
                s_df,
                s_seed,
                s_oracle,
                batch_size=args.batch_size,
                n_rounds=args.rounds,
                seed=args.seed,
            )
        elif strategy in DISCRETE_STRATEGIES:
            history = run_discrete_strategy(
                s_df,
                s_fcols,
                s_seed,
                s_oracle,
                strategy=strategy,
                batch_size=args.batch_size,
                n_rounds=args.rounds,
                seed=args.seed,
                kappa=args.kappa,
                normalize=args.normalize,
            )
        else:
            acq_type = STRATEGY_TO_ACQ[strategy]
            history = run_bo_strategy(
                s_dataset,
                s_df,
                s_fcols,
                s_seed,
                s_oracle,
                acq_type=acq_type,
                batch_size=args.batch_size,
                n_rounds=args.rounds,
                seed=args.seed,
                kappa=args.kappa,
                xi=args.xi,
                normalize=args.normalize,
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
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    header = f"{'Strategy':<24} {'Final Best':>12} {'AUC':>10} {'Top-10':>8} {'Top-50':>8} {'Top-100':>8} {'Time':>8}"
    print(header)
    print("-" * len(header))
    for name, result in all_results.items():
        m = result["metrics"]
        r = m["top_k_recall"]
        print(
            f"{name:<24} {m['final_best']:>12.4f} {m['auc']:>10.4f} "
            f"{r.get(10, 0):>7.1%} {r.get(50, 0):>7.1%} {r.get(100, 0):>7.1%} "
            f"{result['elapsed']:>7.1f}s"
        )

    # Save results JSON with full config for reproducibility
    json_path = f"{args.output}.json"
    serializable = {
        "config": {
            "strategies": strategies,
            "rounds": args.rounds,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "n_seeds": args.n_seeds,
            "subset": args.subset,
            "kappa": args.kappa,
            "xi": args.xi,
            "normalize": args.normalize,
        },
        "results": {},
    }
    for name, result in all_results.items():
        serializable["results"][name] = {
            "metrics": result["metrics"],
            "elapsed": result["elapsed"],
            "best_so_far": result["history"]["best_so_far"],
            "round_best": result["history"]["round_best"],
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
