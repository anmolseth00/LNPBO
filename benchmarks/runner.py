#!/usr/bin/env python3
"""
LNPBO Benchmark Runner
======================
Simulated closed-loop evaluation using LNPDB as a ground-truth oracle.

Usage:
    python -m benchmarks.runner --strategies random,discrete_xgb_greedy --rounds 5
    python -m benchmarks.runner --strategies all --rounds 10 --n-seeds 500
    python -m benchmarks.runner --strategies discrete_xgb_greedy --feature-type lantern --reduction pls
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors

from LNPBO.optimization.optimizer import ENC_PREFIXES

# Strategy configs: name -> (type, type-specific params)
# "gp" strategies dispatch to _gp_bo_common.run_gp_strategy (GPyTorch/BoTorch)
# "discrete" strategies dispatch to _discrete_common.run_discrete_strategy with surrogate
# "random" is inlined below
STRATEGY_CONFIGS = {
    "random": {"type": "random"},
    "lnpbo_ucb": {"type": "gp", "acq_type": "UCB"},
    "lnpbo_ei": {"type": "gp", "acq_type": "EI"},
    "lnpbo_logei": {"type": "gp", "acq_type": "LogEI"},
    "lnpbo_rkb_logei": {"type": "gp", "acq_type": "RKB_LogEI"},
    "lnpbo_lp_ei": {"type": "gp", "acq_type": "LP_EI"},
    "lnpbo_lp_logei": {"type": "gp", "acq_type": "LP_LogEI"},
    "lnpbo_pls_logei": {"type": "gp", "acq_type": "LogEI"},
    "lnpbo_ts_batch": {"type": "gp", "acq_type": "TS_Batch"},
    "lnpbo_pls_lp_logei": {"type": "gp", "acq_type": "LP_LogEI"},
    "discrete_gp_ucb": {"type": "discrete", "surrogate": "gp_ucb"},
    "discrete_rf_ucb": {"type": "discrete", "surrogate": "rf_ucb"},
    "discrete_rf_ts": {"type": "discrete", "surrogate": "rf_ts"},
    "discrete_xgb_greedy": {"type": "discrete", "surrogate": "xgb"},
    "discrete_xgb_ucb": {"type": "discrete", "surrogate": "xgb_ucb"},
    "discrete_ngboost_ucb": {"type": "discrete", "surrogate": "ngboost"},
    "discrete_xgb_cqr": {"type": "discrete", "surrogate": "xgb_cqr"},
    "discrete_deep_ensemble": {"type": "discrete", "surrogate": "deep_ensemble"},
    "discrete_tabpfn": {"type": "discrete", "surrogate": "tabpfn"},
    "discrete_rf_ts_batch": {"type": "discrete_ts_batch", "surrogate": "rf_ucb"},
    "discrete_xgb_ucb_ts_batch": {"type": "discrete_ts_batch", "surrogate": "xgb_ucb"},
    "discrete_xgb_online_conformal": {"type": "discrete_online_conformal"},
    "casmopolitan_ucb": {"type": "casmopolitan", "acq_func": "ucb"},
    "casmopolitan_ei": {"type": "casmopolitan", "acq_func": "ei"},
    "lnpbo_gibbon": {"type": "gp", "acq_type": "GIBBON"},
    "lnpbo_jes": {"type": "gp", "acq_type": "JES"},
    "lnpbo_tanimoto_ts": {"type": "gp", "acq_type": "Tanimoto_TS"},
    "lnpbo_tanimoto_logei": {"type": "gp", "acq_type": "Tanimoto_LogEI"},
}

ALL_STRATEGIES = list(STRATEGY_CONFIGS.keys())

PLS_STRATEGIES = {"lnpbo_pls_logei", "lnpbo_pls_lp_logei"}

# Tanimoto strategies require raw count_mfp fingerprints (no PCA reduction)
TANIMOTO_STRATEGIES = {"lnpbo_tanimoto_ts", "lnpbo_tanimoto_logei"}

STRATEGY_DISPLAY = {
    "random": "Random",
    "lnpbo_ucb": "GP + KB (UCB)",
    "lnpbo_ei": "GP + KB (EI)",
    "lnpbo_logei": "GP + KB (LogEI)",
    "lnpbo_rkb_logei": "GP + RKB (LogEI)",
    "lnpbo_lp_ei": "GP + LP (EI)",
    "lnpbo_lp_logei": "GP + LP (LogEI)",
    "lnpbo_pls_logei": "GP + KB (PLS+LogEI)",
    "lnpbo_ts_batch": "GP + TS-Batch",
    "lnpbo_pls_lp_logei": "GP + LP (PLS+LogEI)",
    "discrete_gp_ucb": "Discrete GP-UCB",
    "discrete_rf_ucb": "Discrete RF-UCB",
    "discrete_rf_ts": "Discrete RF-TS",
    "discrete_xgb_greedy": "Discrete XGB",
    "discrete_xgb_ucb": "Discrete XGB-UCB (MAPIE)",
    "discrete_ngboost_ucb": "Discrete NGBoost-UCB",
    "discrete_xgb_cqr": "Discrete XGB-CQR",
    "discrete_deep_ensemble": "Discrete Deep Ensemble UCB",
    "discrete_tabpfn": "Discrete TabPFN-UCB",
    "discrete_rf_ts_batch": "Discrete RF TS-Batch",
    "discrete_xgb_ucb_ts_batch": "Discrete XGB-UCB TS-Batch",
    "discrete_xgb_online_conformal": "Discrete XGB Online Conformal",
    "casmopolitan_ucb": "CASMOPOLITAN (UCB)",
    "casmopolitan_ei": "CASMOPOLITAN (EI)",
    "lnpbo_gibbon": "GP + GIBBON",
    "lnpbo_jes": "GP + JES",
    "lnpbo_tanimoto_ts": "GP-Tanimoto + TS-Batch",
    "lnpbo_tanimoto_logei": "GP-Tanimoto + KB (LogEI)",
}

STRATEGY_COLORS = {
    "random": "#999999",
    "lnpbo_ucb": "#1f77b4",
    "lnpbo_ei": "#ff7f0e",
    "lnpbo_logei": "#2ca02c",
    "lnpbo_rkb_logei": "#ff9896",
    "lnpbo_ts_batch": "#aec7e8",
    "lnpbo_lp_ei": "#d62728",
    "lnpbo_lp_logei": "#9467bd",
    "lnpbo_pls_logei": "#8c564b",
    "lnpbo_pls_lp_logei": "#e377c2",
    "discrete_gp_ucb": "#17becf",
    "discrete_rf_ucb": "#bcbd22",
    "discrete_rf_ts": "#7f7f7f",
    "discrete_xgb_greedy": "#e41a1c",
    "discrete_xgb_ucb": "#ff6600",
    "discrete_ngboost_ucb": "#4daf4a",
    "discrete_xgb_cqr": "#984ea3",
    "discrete_deep_ensemble": "#8b4513",
    "discrete_tabpfn": "#ff1493",
    "discrete_rf_ts_batch": "#556b2f",
    "discrete_xgb_ucb_ts_batch": "#b8860b",
    "discrete_xgb_online_conformal": "#2f4f4f",
    "casmopolitan_ucb": "#00ced1",
    "casmopolitan_ei": "#8a2be2",
    "lnpbo_gibbon": "#20b2aa",
    "lnpbo_jes": "#daa520",
    "lnpbo_tanimoto_ts": "#ff4500",
    "lnpbo_tanimoto_logei": "#6a0dad",
}


# ---------------------------------------------------------------------------
# Oracle
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


def prepare_benchmark_data(
    n_seed=500, random_seed=42, subset=None, reduction="pca", feature_type="mfp",
    n_pcs=None, context_features=False, fp_radius=None, fp_bits=None, data_df=None,
    pca_train_indices=None,
):
    """Load LNPDB, encode, split into seed/oracle.

    Parameters
    ----------
    pca_train_indices : array-like of int, optional
        Row indices (into data_df) used to fit PCA/scaler. When provided,
        PCA is fit on these rows only, then applied to the full dataset.
        Prevents information leakage in study-level holdout benchmarks.
    """
    from LNPBO.data.dataset import Dataset
    from LNPBO.data.lnpdb_bridge import load_lnpdb_full

    is_ratios_only = feature_type == "ratios_only"
    is_raw = feature_type.startswith("raw_")
    is_concat = feature_type in ("concat", "raw_concat")
    is_lantern = feature_type in ("lantern", "raw_lantern")
    is_lantern_unimol = feature_type in ("lantern_unimol", "raw_lantern_unimol")
    is_lantern_mordred = feature_type in ("lantern_mordred", "raw_lantern_mordred")
    is_chemeleon_il_only = feature_type == "chemeleon_il_only"
    is_chemeleon_helper_only = feature_type == "chemeleon_helper_only"
    effective_reduction = "none" if (is_raw or is_ratios_only) else reduction

    print(f"Loading LNPDB (reduction={effective_reduction}, features={feature_type})...")
    if data_df is None:
        dataset = load_lnpdb_full()
        df = dataset.df
    else:
        df = data_df.copy()
        if "Formulation_ID" not in df.columns:
            df["Formulation_ID"] = range(1, len(df) + 1)
        dataset = Dataset(df, source="lnpdb", name="LNPDB_benchmark")

    if subset and subset < len(df):
        df = df.sample(n=subset, random_state=random_seed).reset_index(drop=True)
        df["Formulation_ID"] = range(1, len(df) + 1)
        dataset = Dataset(df, source="lnpdb", name="LNPDB_benchmark")

    print(f"  {len(df):,} formulations loaded")
    print(f"  Experiment_value range: [{df['Experiment_value'].min():.2f}, {df['Experiment_value'].max():.2f}]")

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

    if is_raw:
        default_pcs = 2048
    elif n_pcs is not None:
        default_pcs = n_pcs
    else:
        default_pcs = 5

    il_pcs = _should_encode("IL", default_pcs)
    hl_pcs = _should_encode("HL", default_pcs if (is_raw or n_pcs is not None) else 3)
    chl_pcs = _should_encode("CHL", default_pcs if (is_raw or n_pcs is not None) else 3)
    peg_pcs = _should_encode("PEG", default_pcs if (is_raw or n_pcs is not None) else 3)

    def _role_pcs():
        return [("IL", il_pcs), ("HL", hl_pcs), ("CHL", chl_pcs), ("PEG", peg_pcs)]

    enc = {}
    if is_ratios_only:
        print("  Ratios-only mode: no molecular encoding")
    elif is_concat:
        for role, n in _role_pcs():
            enc[role] = {"mfp": n, "unimol": n}
        print(f"  Concat (MFP+Uni-Mol) dims per role: IL={il_pcs}, HL={hl_pcs}, CHL={chl_pcs}, PEG={peg_pcs}")
    elif is_lantern_mordred:
        for role, n in _role_pcs():
            enc[role] = {"count_mfp": n, "rdkit": n, "mordred": n}
        print(
            f"  LANTERN+Mordred (count_mfp+rdkit+mordred) dims per role:"
            f" IL={il_pcs}, HL={hl_pcs}, CHL={chl_pcs}, PEG={peg_pcs}"
        )
    elif is_lantern_unimol:
        for role, n in _role_pcs():
            enc[role] = {"count_mfp": n, "rdkit": n, "unimol": n}
        print(
            f"  LANTERN+Uni-Mol (count_mfp+rdkit+unimol) dims per role:"
            f" IL={il_pcs}, HL={hl_pcs}, CHL={chl_pcs}, PEG={peg_pcs}"
        )
    elif is_lantern:
        for role, n in _role_pcs():
            enc[role] = {"count_mfp": n, "rdkit": n}
        print(f"  LANTERN (count_mfp+rdkit) dims per role: IL={il_pcs}, HL={hl_pcs}, CHL={chl_pcs}, PEG={peg_pcs}")
    elif feature_type == "lantern_il_only":
        enc["IL"] = {"count_mfp": il_pcs, "rdkit": il_pcs}
        print(f"  LANTERN IL-only: IL={il_pcs} PCs (helpers get ratios only)")
    elif feature_type == "lantern_il_hl":
        enc["IL"] = {"count_mfp": il_pcs, "rdkit": il_pcs}
        enc["HL"] = {"count_mfp": hl_pcs, "rdkit": hl_pcs}
        print(f"  LANTERN IL+HL: IL={il_pcs}, HL={hl_pcs} PCs (CHL/PEG get ratios only)")
    elif feature_type == "lantern_il_noratios":
        enc["IL"] = {"count_mfp": il_pcs, "rdkit": il_pcs}
        print(f"  LANTERN IL no-ratios: IL={il_pcs} PCs (no molar ratios)")
    elif is_chemeleon_il_only:
        enc["IL"] = {"chemeleon": il_pcs}
        print(f"  CheMeleon IL-only: IL={il_pcs} PCs (helpers get ratios only)")
    elif is_chemeleon_helper_only:
        for role, n in [("HL", hl_pcs), ("CHL", chl_pcs), ("PEG", peg_pcs)]:
            enc[role] = {"chemeleon": n}
        print(f"  CheMeleon helper-only: HL={hl_pcs}, CHL={chl_pcs}, PEG={peg_pcs} (IL gets ratios only)")
    else:
        base_type = feature_type.replace("raw_", "")
        enc_key = {
            "mfp": "mfp", "mordred": "mordred", "unimol": "unimol",
            "count_mfp": "count_mfp", "rdkit": "rdkit", "chemeleon": "chemeleon",
        }[base_type]
        for role, n in _role_pcs():
            enc[role] = {enc_key: n}
        print(f"  Encoding dims: IL={il_pcs}, HL={hl_pcs}, CHL={chl_pcs}, PEG={peg_pcs}")

    if is_ratios_only:
        encoded = dataset
    else:
        fp_kw = {}
        if fp_radius is not None:
            fp_kw["fp_radius"] = fp_radius
        if fp_bits is not None:
            fp_kw["fp_bits"] = fp_bits

        if pca_train_indices is not None and effective_reduction != "none":
            train_df = df.iloc[pca_train_indices].copy()
            train_dataset = Dataset(train_df, source="lnpdb", name="LNPDB_pca_fit")
            train_encoded = train_dataset.encode_dataset(
                enc, reduction=effective_reduction, **fp_kw,
            )
            encoded = dataset.encode_dataset(
                enc, reduction=effective_reduction,
                fitted_transformers_in=train_encoded.fitted_transformers,
                **fp_kw,
            )
        else:
            encoded = dataset.encode_dataset(
                enc,
                reduction=effective_reduction,
                **fp_kw,
            )

    feature_cols = []
    enc_prefixes = ENC_PREFIXES
    for role in ["IL", "HL", "CHL", "PEG"]:
        for prefix in enc_prefixes:
            role_cols = [c for c in encoded.df.columns if c.startswith(f"{role}_{prefix}")]
            feature_cols.extend(sorted(role_cols))
    if feature_type != "lantern_il_noratios":
        for role in ["IL", "HL", "CHL", "PEG"]:
            col = f"{role}_molratio"
            if col in encoded.df.columns and encoded.df[col].nunique() > 1:
                feature_cols.append(col)
        mr_col = "IL_to_nucleicacid_massratio"
        if mr_col in encoded.df.columns and encoded.df[mr_col].nunique() > 1:
            feature_cols.append("IL_to_nucleicacid_massratio")

    if context_features:
        from LNPBO.data.context import encode_context

        encoded.df, ctx_cols, _ = encode_context(encoded.df)
        feature_cols.extend(ctx_cols)
        print(f"  Context features ({len(ctx_cols)}): {ctx_cols[:5]}{'...' if len(ctx_cols) > 5 else ''}")

    print(f"  Feature columns ({len(feature_cols)}): {feature_cols}")

    valid_mask = encoded.df[feature_cols].notna().all(axis=1)
    encoded_df = encoded.df[valid_mask].copy()
    if "Formulation_ID" in encoded_df.columns:
        encoded_df = encoded_df.drop_duplicates(subset=["Formulation_ID"])
    encoded_df = encoded_df.reset_index(drop=True)
    # Sync encoded_dataset.df with filtered encoded_df so indices align
    # (needed for refit_pls which uses .loc with indices from encoded_df)
    encoded.df = encoded_df.copy()
    print(f"  Valid rows after cleanup: {len(encoded_df):,}")

    rng = np.random.RandomState(random_seed)
    all_idx = np.arange(len(encoded_df))
    rng.shuffle(all_idx)

    seed_idx = sorted(all_idx[:n_seed])
    oracle_idx = sorted(all_idx[n_seed:])

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
# History / metrics tracking
# ---------------------------------------------------------------------------


def init_history(df, seed_idx):
    seed_vals = df.loc[seed_idx, "Experiment_value"]
    return {
        "best_so_far": [float(seed_vals.max())],
        "round_best": [],
        "n_evaluated": [len(seed_idx)],
        "all_evaluated": set(seed_idx),
    }


def update_history(history, df, training_idx, batch_idx, round_num):
    batch_vals = df.loc[batch_idx, "Experiment_value"]
    all_vals = df.loc[training_idx, "Experiment_value"]
    history["best_so_far"].append(float(all_vals.max()))
    history["round_best"].append(float(batch_vals.max()))
    history["n_evaluated"].append(len(training_idx))
    history["all_evaluated"].update(batch_idx)


def compute_metrics(history, top_k_values, n_total):
    bsf = np.array(history["best_so_far"])
    n_eval = np.array(history["n_evaluated"])
    evaluated = history["all_evaluated"]

    auc = float(np.trapezoid(bsf, n_eval) / (n_eval[-1] - n_eval[0])) if len(bsf) > 1 else bsf[0]

    recall = {}
    for k, top_set in top_k_values.items():
        found = len(evaluated & top_set)
        recall[k] = found / len(top_set)

    final_best = bsf[-1]

    return {
        "final_best": float(final_best),
        "auc": auc,
        "top_k_recall": recall,
        "n_rounds": len(bsf) - 1,
        "n_total_evaluated": int(n_eval[-1]),
    }


def _run_random(df, seed_idx, oracle_idx, batch_size, n_rounds, seed):
    rng = np.random.RandomState(seed)
    training_idx = list(seed_idx)
    pool_idx = list(oracle_idx)
    history = init_history(df, training_idx)
    for r in range(n_rounds):
        if len(pool_idx) < batch_size:
            break
        chosen = rng.choice(len(pool_idx), size=batch_size, replace=False)
        batch_idx = [pool_idx[i] for i in sorted(chosen)]
        pool_set = set(batch_idx)
        pool_idx = [i for i in pool_idx if i not in pool_set]
        training_idx.extend(batch_idx)
        update_history(history, df, training_idx, batch_idx, r)
    return history


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


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
    parser = argparse.ArgumentParser(
        description="LNPBO Benchmark: Simulated closed-loop BO evaluation",
    )
    parser.add_argument(
        "--strategies",
        type=str,
        default="random,discrete_xgb_greedy",
        help=f"Comma-separated strategies (or 'all'). Options: {','.join(ALL_STRATEGIES)}",
    )
    parser.add_argument("--rounds", type=int, default=15, help="Number of rounds (default: 15)")
    parser.add_argument("--batch-size", type=int, default=12, help="Batch size per round (default: 12)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--n-seeds", type=int, default=200, help="Size of initial seed pool (default: 200)")
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
        default=None,
        help="Output prefix (default: benchmark_results/<strategies>)",
    )
    parser.add_argument("--no-plot", action="store_true", help="Skip plotting")
    parser.add_argument(
        "--feature-type",
        type=str,
        default="mfp",
        choices=["mfp", "mordred", "unimol", "count_mfp", "rdkit", "chemeleon",
                 "raw_mfp", "raw_unimol", "raw_count_mfp", "raw_rdkit", "raw_chemeleon",
                 "concat", "raw_concat", "lantern", "raw_lantern",
                 "lantern_unimol", "raw_lantern_unimol",
                 "lantern_mordred", "raw_lantern_mordred",
                 "lantern_il_only", "lantern_il_hl", "lantern_il_noratios",
                 "chemeleon_il_only", "chemeleon_helper_only", "ratios_only"],
        help="Feature type (default: mfp).",
    )
    parser.add_argument("--n-pcs", type=int, default=None, help="Override PCA/PLS components per role")
    parser.add_argument(
        "--reduction",
        type=str,
        default="pca",
        choices=["pca", "pls", "none"],
        help="Dimensionality reduction method (default: pca)",
    )
    parser.add_argument(
        "--context-features",
        action="store_true",
        help="Include one-hot experimental context (cell type, target, RoA, etc.)",
    )
    parser.add_argument(
        "--fp-radius", type=int, default=None,
        help="Morgan FP radius (default: 3 for mfp, 3 for count_mfp)",
    )
    parser.add_argument(
        "--fp-bits", type=int, default=None,
        help="Morgan FP bit size (default: 1024 for mfp, 2048 for count_mfp)",
    )
    args = parser.parse_args()

    # Parse strategies
    if args.strategies == "all":
        strategies = ALL_STRATEGIES
    else:
        strategies = [s.strip() for s in args.strategies.split(",")]
        for s in strategies:
            if s not in STRATEGY_CONFIGS:
                parser.error(f"Unknown strategy: {s}. Choose from: {ALL_STRATEGIES}")

    # Output path
    results_dir = Path(__file__).resolve().parent.parent / "benchmark_results"
    results_dir.mkdir(exist_ok=True)
    if args.output is None:
        output_prefix = str(results_dir / f"{'_'.join(strategies[:3])}")
    else:
        output_prefix = args.output

    print("=" * 70)
    print("LNPBO BENCHMARK")
    print("=" * 70)
    print(f"Strategies: {strategies}")
    print(f"Rounds: {args.rounds}, Batch size: {args.batch_size}")
    print(f"Seed pool: {args.n_seeds}, Random seed: {args.seed}")
    print(f"Target normalization: {args.normalize}")
    print(f"Context features: {args.context_features}")
    print()

    # Prepare data
    pca_data = prepare_benchmark_data(
        n_seed=args.n_seeds,
        random_seed=args.seed,
        subset=args.subset,
        reduction=args.reduction,
        feature_type=args.feature_type,
        n_pcs=args.n_pcs,
        context_features=args.context_features,
        fp_radius=args.fp_radius,
        fp_bits=args.fp_bits,
    )
    pls_data = None
    if any(s in PLS_STRATEGIES for s in strategies):
        pls_data = prepare_benchmark_data(
            n_seed=args.n_seeds,
            random_seed=args.seed,
            subset=args.subset,
            reduction="pls",
            feature_type=args.feature_type,
            n_pcs=args.n_pcs,
            context_features=args.context_features,
        )

    tanimoto_data = None
    if any(s in TANIMOTO_STRATEGIES for s in strategies):
        tanimoto_data = prepare_benchmark_data(
            n_seed=args.n_seeds,
            random_seed=args.seed,
            subset=args.subset,
            reduction="none",
            feature_type="count_mfp",
            context_features=args.context_features,
        )

    # Run strategies
    all_results = {}
    for strategy in strategies:
        print(f"\n{'=' * 70}")
        print(f"Running: {strategy}")
        print(f"{'=' * 70}")
        t0 = time.time()

        if strategy in TANIMOTO_STRATEGIES and tanimoto_data is not None:
            s_dataset, s_df, s_fcols, s_seed, s_oracle, s_topk = tanimoto_data
        elif strategy in PLS_STRATEGIES and pls_data is not None:
            s_dataset, s_df, s_fcols, s_seed, s_oracle, s_topk = pls_data
        else:
            s_dataset, s_df, s_fcols, s_seed, s_oracle, s_topk = pca_data

        config = STRATEGY_CONFIGS[strategy]
        if config["type"] == "random":
            history = _run_random(s_df, s_seed, s_oracle, args.batch_size, args.rounds, args.seed)
        elif config["type"] == "discrete":
            from ._discrete_common import run_discrete_strategy
            history = run_discrete_strategy(
                s_df, s_fcols, s_seed, s_oracle,
                surrogate=config["surrogate"], batch_size=args.batch_size,
                n_rounds=args.rounds, seed=args.seed, kappa=args.kappa,
                normalize=args.normalize, encoded_dataset=s_dataset,
            )
        elif config["type"] == "discrete_ts_batch":
            from ._discrete_common import run_discrete_ts_batch_strategy
            history = run_discrete_ts_batch_strategy(
                s_df, s_fcols, s_seed, s_oracle,
                surrogate=config["surrogate"], batch_size=args.batch_size,
                n_rounds=args.rounds, seed=args.seed, kappa=args.kappa,
                normalize=args.normalize, encoded_dataset=s_dataset,
            )
        elif config["type"] == "discrete_online_conformal":
            from ._discrete_common import run_discrete_online_conformal_strategy
            history = run_discrete_online_conformal_strategy(
                s_df, s_fcols, s_seed, s_oracle,
                batch_size=args.batch_size,
                n_rounds=args.rounds, seed=args.seed, kappa=args.kappa,
                normalize=args.normalize, encoded_dataset=s_dataset,
            )
        elif config["type"] == "casmopolitan":
            from LNPBO.optimization.casmopolitan import run_casmopolitan_strategy
            history = run_casmopolitan_strategy(
                s_df, s_fcols, s_seed, s_oracle,
                batch_size=args.batch_size,
                n_rounds=args.rounds, seed=args.seed, kappa=args.kappa,
                normalize=args.normalize,
                acq_func=config.get("acq_func", "ucb"),
            )
        elif config["type"] == "gp":
            from ._gp_bo_common import run_gp_strategy
            history = run_gp_strategy(
                s_dataset, s_df, s_fcols, s_seed, s_oracle,
                acq_type=config["acq_type"], batch_size=args.batch_size,
                n_rounds=args.rounds, seed=args.seed, kappa=args.kappa,
                xi=args.xi, normalize=args.normalize,
            )
        else:
            raise ValueError(f"Unknown strategy type: {config['type']!r}")

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

    # Save results JSON
    json_path = f"{output_prefix}.json"
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
            "feature_type": args.feature_type,
            "n_pcs": args.n_pcs,
            "reduction": args.reduction,
            "context_features": args.context_features,
            "fp_radius": args.fp_radius,
            "fp_bits": args.fp_bits,
        },
        "results": {},
    }
    for name, result in all_results.items():
        entry = {
            "metrics": result["metrics"],
            "elapsed": result["elapsed"],
            "best_so_far": result["history"]["best_so_far"],
            "round_best": result["history"]["round_best"],
            "n_evaluated": result["history"]["n_evaluated"],
        }
        if "coverage" in result["history"]:
            entry["coverage"] = result["history"]["coverage"]
        if "conformal_quantile" in result["history"]:
            entry["conformal_quantile"] = result["history"]["conformal_quantile"]
        serializable["results"][name] = entry
    with open(json_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {json_path}")

    if not args.no_plot:
        plot_results(all_results, output_path=f"{output_prefix}.png")


if __name__ == "__main__":
    main()
