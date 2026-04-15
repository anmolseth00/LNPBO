#!/usr/bin/env python3
"""Benchmark runner. Simulated closed-loop evaluation using LNPDB as oracle.

Usage:
    python -m benchmarks.runner --strategies random,discrete_xgb_greedy --rounds 5
    python -m benchmarks.runner --strategies all --rounds 10 --n-seeds 500
    python -m benchmarks.runner --strategies discrete_xgb_greedy --feature-type lantern --reduction pls
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors

from LNPBO.optimization.optimizer import ENC_PREFIXES
from LNPBO.runtime_paths import benchmark_results_root, package_root_from


def _ts() -> str:
    """Return a bracketed HH:MM:SS stamp for per-round log lines.

    Shared by all round loops (OptimizerRunner, conformal, random) so
    output pacing looks consistent across strategy types.
    """
    return datetime.now().strftime("[%H:%M:%S]")


def _log_round_start(r: int, n_rounds: int, pool_size: int, training_size: int) -> None:
    """Announce the start of a round before the (potentially long) fit + acq step."""
    print(
        f"  {_ts()} Round {r + 1}/{n_rounds} starting "
        f"(pool={pool_size}, training={training_size})...",
        flush=True,
    )


def _log_round_complete(
    r: int,
    batch_best: float,
    cum_best: float,
    elapsed_s: float,
    **extras: object,
) -> None:
    """Report a completed round with its shared core + strategy-specific fields.

    Core fields (round, batch_best, cum_best, time) are formatted identically
    across strategies. ``extras`` carries strategy-specific fields; callers
    pass ints/counts directly and pre-format floats that need specific
    precision (e.g. ``coverage=f"{coverage:.2f}"``) so the helper doesn't
    second-guess each strategy's output conventions.
    """
    fragments = [f"batch_best={batch_best:.3f}", f"cum_best={cum_best:.3f}"]
    fragments.extend(f"{key}={value}" for key, value in extras.items())
    fragments.append(f"time={elapsed_s:.1f}s")
    print(f"  {_ts()} Round {r + 1}: " + ", ".join(fragments), flush=True)

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
    "discrete_ridge_ucb": {"type": "discrete", "surrogate": "ridge"},
    "discrete_tabpfn": {"type": "discrete", "surrogate": "tabpfn"},
    "discrete_rf_ts_batch": {"type": "discrete_ts_batch", "surrogate": "rf_ucb"},
    "discrete_xgb_ucb_ts_batch": {"type": "discrete_ts_batch", "surrogate": "xgb_ucb"},
    "discrete_xgb_online_conformal": {"type": "discrete_online_conformal_exact", "acquisition": "ucb"},
    "discrete_xgb_cumulative_split_conformal_ucb_baseline": {
        "type": "discrete_online_conformal_baseline",
        "acquisition": "ucb",
    },
    "casmopolitan_ucb": {"type": "casmopolitan", "acq_func": "ucb"},
    "casmopolitan_ei": {"type": "casmopolitan", "acq_func": "ei"},
    "lnpbo_gibbon": {"type": "gp", "acq_type": "GIBBON"},
    "lnpbo_tanimoto_ts": {"type": "gp", "acq_type": "Tanimoto_TS"},
    "lnpbo_tanimoto_logei": {"type": "gp", "acq_type": "Tanimoto_LogEI"},
    "lnpbo_aitchison_ts": {"type": "gp", "acq_type": "Aitchison_TS"},
    "lnpbo_aitchison_logei": {"type": "gp", "acq_type": "Aitchison_LogEI"},
    "lnpbo_dkl_ts": {"type": "gp", "acq_type": "DKL_TS"},
    "lnpbo_dkl_logei": {"type": "gp", "acq_type": "DKL_LogEI"},
    "lnpbo_rf_kernel_ts": {"type": "gp", "acq_type": "RF_Kernel_TS"},
    "lnpbo_rf_kernel_logei": {"type": "gp", "acq_type": "RF_Kernel_LogEI"},
    "lnpbo_compositional_ts": {"type": "gp", "acq_type": "Compositional_TS"},
    "lnpbo_compositional_logei": {"type": "gp", "acq_type": "Compositional_LogEI"},
    "lnpbo_mixed_logei": {"type": "gp_mixed", "acq_type": "Mixed_LogEI"},
    "lnpbo_mixed_ts": {"type": "gp_mixed", "acq_type": "Mixed_TS"},
}

ALL_STRATEGIES = list(STRATEGY_CONFIGS.keys())

PLS_STRATEGIES = {"lnpbo_pls_logei", "lnpbo_pls_lp_logei"}

# Tanimoto strategies require raw count_mfp fingerprints (no PCA reduction)
TANIMOTO_STRATEGIES = {"lnpbo_tanimoto_ts", "lnpbo_tanimoto_logei"}

# Aitchison strategies use ratio features on the simplex (no molecular encoding)
AITCHISON_STRATEGIES = {"lnpbo_aitchison_ts", "lnpbo_aitchison_logei"}

# Compositional product kernel: raw count_mfp (Tanimoto) + ratios (Aitchison) + synth (Matern)
COMPOSITIONAL_STRATEGIES = {"lnpbo_compositional_ts", "lnpbo_compositional_logei"}

# Mixed discrete-continuous strategies: enumerate ILs + optimize ratios continuously
MIXED_STRATEGIES = {"lnpbo_mixed_logei", "lnpbo_mixed_ts"}

# GP acquisition type -> (base_acq, batch_strategy, kernel_type) mapping
ACQ_TYPE_MAP = {
    "UCB": ("UCB", "kb", "matern"),
    "EI": ("EI", "kb", "matern"),
    "LogEI": ("LogEI", "kb", "matern"),
    "RKB_LogEI": ("LogEI", "rkb", "matern"),
    "RKB_UCB": ("UCB", "rkb", "matern"),
    "RKB_EI": ("EI", "rkb", "matern"),
    "LP_UCB": ("UCB", "lp", "matern"),
    "LP_EI": ("EI", "lp", "matern"),
    "LP_LogEI": ("LogEI", "lp", "matern"),
    "TS_Batch": ("UCB", "ts", "matern"),  # TS ignores acq_type; UCB is placeholder
    "GIBBON": ("UCB", "gibbon", "matern"),  # info-theoretic; acq_type ignored
    "Tanimoto_TS": ("UCB", "ts", "tanimoto"),  # Tanimoto kernel + TS batch
    "Tanimoto_LogEI": ("LogEI", "kb", "tanimoto"),  # Tanimoto kernel + KB-LogEI
    "Aitchison_TS": ("UCB", "ts", "aitchison"),  # Aitchison simplex kernel + TS batch
    "Aitchison_LogEI": ("LogEI", "kb", "aitchison"),  # Aitchison simplex kernel + KB-LogEI
    "DKL_TS": ("UCB", "ts", "dkl"),  # Deep Kernel Learning + TS batch
    "DKL_LogEI": ("LogEI", "kb", "dkl"),  # Deep Kernel Learning + KB-LogEI
    "RF_Kernel_TS": ("UCB", "ts", "rf"),  # RF proximity kernel + TS batch
    "RF_Kernel_LogEI": ("LogEI", "kb", "rf"),  # RF proximity kernel + KB-LogEI
    "Compositional_TS": ("UCB", "ts", "compositional"),  # Product kernel + TS batch
    "Compositional_LogEI": ("LogEI", "kb", "compositional"),  # Product kernel + KB-LogEI
    "Mixed_LogEI": ("LogEI", "kb", "compositional"),  # Mixed discrete-continuous + KB-LogEI
    "Mixed_TS": ("UCB", "ts", "compositional"),  # Mixed discrete-continuous + TS batch
}

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
    "discrete_ridge_ucb": "Discrete Ridge UCB",
    "discrete_tabpfn": "Discrete TabPFN-UCB",
    "discrete_rf_ts_batch": "Discrete RF TS-Batch",
    "discrete_xgb_ucb_ts_batch": "Discrete XGB-UCB TS-Batch",
    "discrete_xgb_online_conformal": "Discrete XGB Exact Online Conformal",
    "discrete_xgb_cumulative_split_conformal_ucb_baseline": "Discrete XGB Cumulative Split-Conformal Baseline",
    "casmopolitan_ucb": "CASMOPOLITAN (UCB)",
    "casmopolitan_ei": "CASMOPOLITAN (EI)",
    "lnpbo_gibbon": "GP + GIBBON",
    "lnpbo_tanimoto_ts": "GP-Tanimoto + TS-Batch",
    "lnpbo_tanimoto_logei": "GP-Tanimoto + KB (LogEI)",
    "lnpbo_aitchison_ts": "GP-Aitchison + TS-Batch",
    "lnpbo_aitchison_logei": "GP-Aitchison + KB (LogEI)",
    "lnpbo_dkl_ts": "DKL-GP + TS-Batch",
    "lnpbo_dkl_logei": "DKL-GP + KB (LogEI)",
    "lnpbo_rf_kernel_ts": "GP-RF Kernel + TS-Batch",
    "lnpbo_rf_kernel_logei": "GP-RF Kernel + KB (LogEI)",
    "lnpbo_compositional_ts": "GP-Compositional + TS-Batch",
    "lnpbo_compositional_logei": "GP-Compositional + KB (LogEI)",
    "lnpbo_mixed_logei": "GP-Mixed + KB (LogEI)",
    "lnpbo_mixed_ts": "GP-Mixed + TS-Batch",
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
    "discrete_ridge_ucb": "#708090",
    "discrete_tabpfn": "#ff1493",
    "discrete_rf_ts_batch": "#556b2f",
    "discrete_xgb_ucb_ts_batch": "#b8860b",
    "discrete_xgb_online_conformal": "#2f4f4f",
    "discrete_xgb_cumulative_split_conformal_ucb_baseline": "#556270",
    "casmopolitan_ucb": "#00ced1",
    "casmopolitan_ei": "#8a2be2",
    "lnpbo_gibbon": "#20b2aa",
    "lnpbo_tanimoto_ts": "#ff4500",
    "lnpbo_tanimoto_logei": "#6a0dad",
    "lnpbo_aitchison_ts": "#228b22",
    "lnpbo_aitchison_logei": "#4682b4",
    "lnpbo_dkl_ts": "#c71585",
    "lnpbo_dkl_logei": "#6b8e23",
    "lnpbo_rf_kernel_ts": "#8b0000",
    "lnpbo_rf_kernel_logei": "#2e8b57",
    "lnpbo_compositional_ts": "#b22222",
    "lnpbo_compositional_logei": "#483d8b",
    "lnpbo_mixed_logei": "#dc143c",
    "lnpbo_mixed_ts": "#191970",
}


# ---------------------------------------------------------------------------
# Warmup seed selection
# ---------------------------------------------------------------------------


def select_warmup_seed(df, warmup_size, selection, random_seed):
    """Select a warmup seed pool and oracle split from the full dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Encoded dataset with Experiment_value.
    warmup_size : int
        Number of formulations in the warmup seed pool.
    selection : str
        How to select the warmup pool:
        - ``"random"``: uniform random sample.
        - ``"bottom_75"``: sample from the bottom 75th percentile by
          Experiment_value (mimicking an unlucky first screen).
    random_seed : int
        RNG seed for reproducibility.

    Returns
    -------
    seed_idx : list[int]
        Indices into df for the warmup pool.
    oracle_idx : list[int]
        Remaining indices (the oracle/candidate pool).
    """
    rng = np.random.RandomState(random_seed)
    all_idx = np.arange(len(df))

    if warmup_size >= len(df):
        raise ValueError(f"warmup_size ({warmup_size}) >= dataset size ({len(df)})")

    if selection == "random":
        rng.shuffle(all_idx)
        seed_idx = sorted(all_idx[:warmup_size])
    elif selection == "bottom_75":
        threshold = df["Experiment_value"].quantile(0.75)
        bottom_mask = df["Experiment_value"] <= threshold
        bottom_idx = all_idx[bottom_mask]
        if len(bottom_idx) < warmup_size:
            seed_idx = sorted(bottom_idx)
            remaining = warmup_size - len(seed_idx)
            top_idx = all_idx[~bottom_mask]
            rng.shuffle(top_idx)
            seed_idx = sorted(list(seed_idx) + list(top_idx[:remaining]))
        else:
            rng.shuffle(bottom_idx)
            seed_idx = sorted(bottom_idx[:warmup_size])
    else:
        raise ValueError(f"Unknown warmup selection: {selection!r}")

    seed_set = set(seed_idx)
    oracle_idx = sorted([i for i in all_idx if i not in seed_set])
    return list(seed_idx), oracle_idx


# ---------------------------------------------------------------------------
# Oracle
# ---------------------------------------------------------------------------


class LNPDBOracle:
    """Oracle that wraps the encoded LNPDB for nearest-neighbor lookup.

    Used in the old GP pipeline (continuous optimization + projection)
    to match suggested feature vectors back to real formulations in the
    candidate pool.

    Args:
        encoded_df: Encoded DataFrame with feature columns and
            ``Experiment_value``.
        feature_cols: List of feature column names used for NN matching.
    """

    def __init__(self, encoded_df, feature_cols):
        """Initialize the oracle with an encoded DataFrame and feature columns."""
        self.df = encoded_df.copy()
        self.feature_cols = feature_cols
        self._nn = None

    def _build_nn(self, pool_indices):
        """Build a 1-nearest-neighbor index on the given pool indices.

        Args:
            pool_indices: List of DataFrame indices to include in the
                NN index.

        Returns:
            Tuple of ``(NearestNeighbors, pool_indices)``.
        """
        X = self.df.loc[pool_indices, self.feature_cols].values
        nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
        nn.fit(X)
        return nn, pool_indices

    def lookup(self, suggestion_features, pool_indices):
        """Find the nearest formulations in the pool to the suggestions.

        Args:
            suggestion_features: Array of shape ``(n_suggestions, n_features)``
                or ``(n_features,)`` with proposed feature vectors.
            pool_indices: List of DataFrame indices defining the candidate pool.

        Returns:
            Array of matched DataFrame indices (one per suggestion).
        """
        nn, idx_list = self._build_nn(pool_indices)
        idx_arr = np.array(idx_list)
        x = np.atleast_2d(suggestion_features)
        _, nn_idx = nn.kneighbors(x)
        matched_idx = idx_arr[nn_idx.ravel()]
        return matched_idx

    def get_value(self, idx):
        """Retrieve ``Experiment_value`` for the given indices.

        Args:
            idx: DataFrame index or array of indices.

        Returns:
            Array of float experiment values.
        """
        return self.df.loc[idx, "Experiment_value"].values


# ---------------------------------------------------------------------------
# Feature column classification (for CompositionalProductKernel)
# ---------------------------------------------------------------------------


def classify_feature_columns(feature_cols):
    """Classify feature columns into fingerprint, ratio, and synthesis groups.

    Returns a dict with ``fp_indices``, ``ratio_indices``, and ``synth_indices``
    (lists of int column positions) suitable for passing as ``kernel_kwargs``
    to ``CompositionalProductKernel``.

    Classification rules:
    - Columns matching encoding prefixes (mfp_pc, count_mfp_pc, rdkit_pc, etc.)
      are fingerprint features.
    - Columns ending with ``_molratio`` are compositional ratio features.
    - Everything else (e.g., ``IL_to_nucleicacid_massratio``) is a synthesis
      / process parameter.
    """
    fp_indices = []
    ratio_indices = []
    synth_indices = []

    for i, col in enumerate(feature_cols):
        if col.endswith("_molratio"):
            ratio_indices.append(i)
        elif any(col.startswith(f"{role}_{prefix}") for role in ("IL", "HL", "CHL", "PEG") for prefix in ENC_PREFIXES):
            fp_indices.append(i)
        else:
            synth_indices.append(i)

    return {
        "fp_indices": fp_indices,
        "ratio_indices": ratio_indices,
        "synth_indices": synth_indices,
    }


# ---------------------------------------------------------------------------
# Online conformal strategy (custom loop, not routed through Optimizer)
# ---------------------------------------------------------------------------


def run_discrete_online_conformal_strategy(
    encoded_df,
    feature_cols,
    seed_idx,
    oracle_idx,
    batch_size,
    n_rounds,
    seed,
    kappa=5.0,
    alpha=0.1,
    normalize="copula",
    encoded_dataset=None,
    top_k_values=None,
):
    """Run exact online quantile recalibration BO from Deshpande et al.

    At every BO step:
    1. Fit the probabilistic base model ``M_t`` on ``D_t``.
    2. Construct the recalibration dataset with leave-one-out CV
       (Algorithm 4 / ``CreateSplits``).
    3. Fit the exact pointwise recalibrator ``R_t`` (Eq. 11).
    4. Compose the calibrated model ``M_t ◦ R_t`` and evaluate the chosen
       acquisition on the pool.

    The history stores:
    - ``coverage``: empirical pre-update coverage of the calibrated UCB
      quantile used for acquisition.
    - ``conformal_quantile``: the recalibrated level ``R_t(Phi(kappa))``.

    Args:
        encoded_df: Encoded DataFrame with feature columns and
            ``Experiment_value``.
        feature_cols: List of feature column names.
        seed_idx: List of initial seed pool indices.
        oracle_idx: List of candidate pool indices.
        batch_size: Number of candidates to acquire per round.
        n_rounds: Maximum number of acquisition rounds.
        seed: Integer RNG seed.
        kappa: Gaussian quantile z-score used to define the acquisition
            level ``p_ucb = Phi(kappa)``.
    alpha: Recalibration step size ``eta`` from Eq. 11 / Algorithm 3.
        normalize: Target normalization method.
        encoded_dataset: Optional ``Dataset`` for prospective PLS refit.
        top_k_values: Optional dict for recall tracking.

    Returns:
        History dict with additional keys ``coverage`` and
        ``conformal_quantile``.

    References:
        Deshpande, S., Marx, C., & Kuleshov, V. (2024). "Online
        Calibrated and Conformal Prediction Improves Bayesian
        Optimization." AISTATS. arXiv:2112.04620.
    """
    from scipy.stats import norm

    from LNPBO.optimization._normalize import copula_transform
    from LNPBO.optimization.online_conformal import (
        CalibratedProbabilisticModel,
        ExactOnlineRecalibrator,
        GaussianXGBQuantileModel,
        build_recalibration_dataset,
    )

    training_idx = list(seed_idx)
    pool_idx = list(oracle_idx)
    history = init_history(encoded_df, training_idx, top_k_values=top_k_values)
    history["coverage"] = []
    history["conformal_quantile"] = []

    p_ucb = float(norm.cdf(kappa))

    for r in range(n_rounds):
        if len(pool_idx) < batch_size:
            break

        _log_round_start(r, n_rounds, len(pool_idx), len(training_idx))
        round_t0 = time.time()

        if encoded_dataset is not None and getattr(encoded_dataset, "raw_fingerprints", None):
            encoded_dataset.refit_pls(training_idx, external_df=encoded_df)

        X_train = encoded_df.loc[training_idx, feature_cols].values
        y_train_raw = encoded_df.loc[training_idx, "Experiment_value"].values.copy()

        if normalize == "copula":
            y_train = copula_transform(y_train_raw)
        elif normalize == "zscore":
            mu_t, sigma_t = y_train_raw.mean(), y_train_raw.std()
            y_train = (y_train_raw - mu_t) / sigma_t if sigma_t > 0 else y_train_raw.copy()
        else:
            y_train = y_train_raw.copy()

        X_pool = encoded_df.loc[pool_idx, feature_cols].values

        base_model = GaussianXGBQuantileModel(
            n_estimators=200,
            random_state=seed + r,
            n_jobs=1,
        ).fit(X_train, y_train)
        recal_dataset = build_recalibration_dataset(
            X_train,
            y_train,
            model_factory=lambda: GaussianXGBQuantileModel(
                n_estimators=200,
                random_state=seed + r,
                n_jobs=1,
            ),
        )
        recalibrator = ExactOnlineRecalibrator(eta=alpha).fit(recal_dataset)
        calibrated_model = CalibratedProbabilisticModel(base_model, recalibrator)
        q_level = recalibrator.recalibrate(p_ucb)
        scores = calibrated_model.quantile(X_pool, p_ucb)

        top_indices = np.argsort(scores)[-batch_size:][::-1]
        batch_idx = [pool_idx[i] for i in top_indices]

        batch_true_raw = encoded_df.loc[batch_idx, "Experiment_value"].values
        if normalize == "copula":
            batch_true = copula_transform(y_train_raw, x_new=batch_true_raw)
        elif normalize == "zscore":
            mu_t, sigma_t = y_train_raw.mean(), y_train_raw.std()
            batch_true = (batch_true_raw - mu_t) / sigma_t if sigma_t > 0 else batch_true_raw.copy()
        else:
            batch_true = batch_true_raw.copy()

        batch_quantiles = calibrated_model.quantile(
            encoded_df.loc[batch_idx, feature_cols].values,
            p_ucb,
        )
        coverage = float(np.mean(batch_true <= batch_quantiles))
        history["coverage"].append(float(coverage))
        history["conformal_quantile"].append(float(q_level))

        batch_set = set(batch_idx)
        pool_idx = [i for i in pool_idx if i not in batch_set]
        training_idx.extend(batch_idx)
        update_history(history, encoded_df, training_idx, batch_idx, r, top_k_values=top_k_values)

        batch_best = encoded_df.loc[batch_idx, "Experiment_value"].max()
        cum_best = history["best_so_far"][-1]
        _log_round_complete(
            r,
            batch_best,
            cum_best,
            time.time() - round_t0,
            coverage=f"{coverage:.2f}",
            q=f"{q_level:.4f}",
            n_cal=len(recal_dataset.inverse_quantile_levels),
        )

    return history


def run_discrete_cumulative_split_conformal_ucb_baseline(
    encoded_df,
    feature_cols,
    seed_idx,
    oracle_idx,
    batch_size,
    n_rounds,
    seed,
    kappa=5.0,
    alpha=0.1,
    normalize="copula",
    encoded_dataset=None,
    top_k_values=None,
):
    """Run the legacy residual-accumulation split-conformal UCB baseline."""
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler
    from xgboost import XGBRegressor

    from LNPBO.optimization._normalize import copula_transform
    from LNPBO.optimization.online_conformal import (
        CumulativeSplitConformalUCBBaseline,
        update_cumulative_split_conformal_batch,
    )

    training_idx = list(seed_idx)
    pool_idx = list(oracle_idx)
    history = init_history(encoded_df, training_idx, top_k_values=top_k_values)
    history["coverage"] = []
    history["conformal_quantile"] = []

    calibrator = CumulativeSplitConformalUCBBaseline(alpha=alpha)

    for r in range(n_rounds):
        if len(pool_idx) < batch_size:
            break

        _log_round_start(r, n_rounds, len(pool_idx), len(training_idx))
        round_t0 = time.time()

        if encoded_dataset is not None and getattr(encoded_dataset, "raw_fingerprints", None):
            encoded_dataset.refit_pls(training_idx, external_df=encoded_df)

        X_train = encoded_df.loc[training_idx, feature_cols].values
        y_train_raw = encoded_df.loc[training_idx, "Experiment_value"].values.copy()

        if normalize == "copula":
            y_train = copula_transform(y_train_raw)
        elif normalize == "zscore":
            mu_t, sigma_t = y_train_raw.mean(), y_train_raw.std()
            y_train = (y_train_raw - mu_t) / sigma_t if sigma_t > 0 else y_train_raw.copy()
        else:
            y_train = y_train_raw.copy()

        X_pool = encoded_df.loc[pool_idx, feature_cols].values

        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_pool_scaled = scaler.transform(X_pool)
        model = XGBRegressor(n_estimators=200, random_state=seed + r, n_jobs=1, verbosity=0)
        model.fit(X_train_scaled, y_train)

        y_pred_pool = model.predict(X_pool_scaled)
        q = calibrator.get_quantile()
        scores = y_pred_pool + kappa * q

        top_indices = np.argsort(scores)[-batch_size:][::-1]
        batch_idx = [pool_idx[i] for i in top_indices]
        batch_pred = y_pred_pool[top_indices]

        batch_true_raw = encoded_df.loc[batch_idx, "Experiment_value"].values
        if normalize == "copula":
            batch_true = copula_transform(y_train_raw, x_new=batch_true_raw)
        elif normalize == "zscore":
            mu_t, sigma_t = y_train_raw.mean(), y_train_raw.std()
            batch_true = (batch_true_raw - mu_t) / sigma_t if sigma_t > 0 else batch_true_raw.copy()
        else:
            batch_true = batch_true_raw.copy()

        coverage, q_after = update_cumulative_split_conformal_batch(calibrator, batch_pred, batch_true)
        history["coverage"].append(float(coverage))
        history["conformal_quantile"].append(float(q_after))

        batch_set = set(batch_idx)
        pool_idx = [i for i in pool_idx if i not in batch_set]
        training_idx.extend(batch_idx)
        update_history(history, encoded_df, training_idx, batch_idx, r, top_k_values=top_k_values)

        batch_best = encoded_df.loc[batch_idx, "Experiment_value"].max()
        cum_best = history["best_so_far"][-1]
        _log_round_complete(
            r,
            batch_best,
            cum_best,
            time.time() - round_t0,
            coverage=f"{coverage:.2f}" if not np.isnan(coverage) else "nan",
            q=f"{q_after:.4f}" if np.isfinite(q_after) else "inf",
        )

    return history


# ---------------------------------------------------------------------------
# Strategy -> Optimizer kwargs mapping
# ---------------------------------------------------------------------------


def strategy_to_optimizer_kwargs(strategy_name, kernel_kwargs=None):
    """Map a STRATEGY_CONFIGS entry to Optimizer constructor keyword arguments.

    Returns a dict suitable for ``Optimizer(**kwargs)`` that reproduces the
    same surrogate, acquisition, batch strategy, and kernel configuration
    used by the legacy per-family runners.

    Args:
        strategy_name: Key into ``STRATEGY_CONFIGS``.
        kernel_kwargs: Optional dict for compositional product kernel
            (forwarded as ``kernel_kwargs`` to the Optimizer).

    Returns:
        Dict of Optimizer constructor keyword arguments.

    Raises:
        ValueError: If the strategy name is unknown or unsupported (e.g.
            ``"random"`` or custom online conformal strategies).
    """
    config = STRATEGY_CONFIGS[strategy_name]
    stype = config["type"]

    if stype == "random":
        raise ValueError("random strategy does not use Optimizer")
    if stype in {"discrete_online_conformal_exact", "discrete_online_conformal_baseline"}:
        raise ValueError(f"{strategy_name} has custom logic and does not use Optimizer")

    kwargs = {}

    if stype == "gp":
        acq_key = config["acq_type"]
        base_acq, batch_strategy, kt = ACQ_TYPE_MAP[acq_key]
        kwargs["surrogate_type"] = "gp"
        kwargs["gp_engine"] = "botorch"
        kwargs["acquisition_type"] = base_acq
        kwargs["batch_strategy"] = batch_strategy
        kwargs["kernel_type"] = kt
        if kernel_kwargs is not None:
            kwargs["kernel_kwargs"] = kernel_kwargs

    elif stype == "gp_mixed":
        acq_key = config["acq_type"]
        base_acq, batch_strategy, kt = ACQ_TYPE_MAP[acq_key]
        kwargs["surrogate_type"] = "gp_mixed"
        kwargs["acquisition_type"] = base_acq
        kwargs["batch_strategy"] = batch_strategy
        kwargs["kernel_type"] = kt
        if kernel_kwargs is not None:
            kwargs["kernel_kwargs"] = kernel_kwargs

    elif stype in ("discrete", "discrete_ts_batch"):
        surrogate = config["surrogate"]
        kwargs["surrogate_type"] = surrogate
        if stype == "discrete_ts_batch":
            kwargs["batch_strategy"] = "ts"
        else:
            kwargs["batch_strategy"] = "greedy"

    elif stype == "casmopolitan":
        kwargs["surrogate_type"] = "casmopolitan"
        acq_func = config.get("acq_func", "ucb")
        kwargs["acquisition_type"] = acq_func.upper() if acq_func != "ei" else "EI"

    else:
        raise ValueError(f"Unknown strategy type: {stype!r}")

    return kwargs


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------


def prepare_benchmark_data(
    n_seed=500,
    random_seed=42,
    subset=None,
    reduction="pca",
    feature_type="mfp",
    n_pcs=None,
    context_features=False,
    fp_radius=None,
    fp_bits=None,
    data_df=None,
    pca_train_indices=None,
):
    """Load LNPDB, encode molecular features, and split into seed/oracle pools.

    Supports a wide range of molecular encodings (Morgan FP, count MFP,
    RDKit, Mordred, Uni-Mol, CheMeleon, LiON, AGILE, LANTERN composites)
    with optional PCA or PLS dimensionality reduction. Handles per-role
    encoding (IL, HL, CHL, PEG) with automatic skip for roles with only
    one unique component.

    Args:
        n_seed: Number of formulations in the initial seed pool.
        random_seed: Integer RNG seed for the train/oracle split.
        subset: If set, subsample the dataset to this many rows (for
            fast debugging).
        reduction: Dimensionality reduction method: ``"pca"``, ``"pls"``,
            or ``"none"``.
        feature_type: Molecular encoding type. See CLI ``--feature-type``
            choices for the full list.
        n_pcs: Override number of PCA/PLS components per role (default:
            5 for IL, 3 for helpers; 2048 for raw modes).
        context_features: If True, append one-hot experimental context
            columns (cell type, target, route of administration, etc.).
        fp_radius: Override Morgan fingerprint radius.
        fp_bits: Override Morgan fingerprint bit length.
        data_df: Pre-filtered DataFrame to use instead of loading the
            full LNPDB (used by within-study benchmarks).
        pca_train_indices: Row indices (into ``data_df``) used to fit
            PCA/scaler. When provided, PCA is fit on these rows only,
            then applied to the full dataset. Prevents information
            leakage in study-level holdout benchmarks.

    Returns:
        Tuple of ``(encoded_dataset, encoded_df, feature_cols, seed_idx,
        oracle_idx, top_k_values)`` where ``top_k_values`` maps k (int)
        to sets of DataFrame indices for the top-k formulations.
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
        """Return n_pcs if the role has >1 unique component, else 0."""
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
        """Return list of (role, n_pcs) pairs for all four LNP components."""
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
    elif feature_type == "lion_il_only":
        enc["IL"] = {"lion": il_pcs}
        print(f"  LiON IL-only: IL={il_pcs} PCs (helpers get ratios only)")
    elif feature_type == "mordred_il_only":
        enc["IL"] = {"mordred": il_pcs}
        print(f"  Mordred IL-only: IL={il_pcs} PCs (helpers get ratios only)")
    elif feature_type == "unimol_il_only":
        enc["IL"] = {"unimol": il_pcs}
        print(f"  Uni-Mol IL-only: IL={il_pcs} PCs (helpers get ratios only)")
    elif feature_type == "mfp_il_only":
        enc["IL"] = {"mfp": il_pcs}
        print(f"  Morgan FP IL-only: IL={il_pcs} PCs (helpers get ratios only)")
    elif feature_type == "count_mfp_il_only":
        enc["IL"] = {"count_mfp": il_pcs}
        print(f"  Count MFP IL-only: IL={il_pcs} PCs (helpers get ratios only)")
    elif is_chemeleon_il_only:
        enc["IL"] = {"chemeleon": il_pcs}
        print(f"  CheMeleon IL-only: IL={il_pcs} PCs (helpers get ratios only)")
    elif feature_type == "agile_il_only":
        enc["IL"] = {"agile": il_pcs}
        print(f"  AGILE IL-only: IL={il_pcs} PCs (helpers get ratios only)")
    elif is_chemeleon_helper_only:
        for role, n in [("HL", hl_pcs), ("CHL", chl_pcs), ("PEG", peg_pcs)]:
            enc[role] = {"chemeleon": n}
        print(f"  CheMeleon helper-only: HL={hl_pcs}, CHL={chl_pcs}, PEG={peg_pcs} (IL gets ratios only)")
    else:
        base_type = feature_type.replace("raw_", "")
        enc_key = {
            "mfp": "mfp",
            "mordred": "mordred",
            "unimol": "unimol",
            "count_mfp": "count_mfp",
            "rdkit": "rdkit",
            "chemeleon": "chemeleon",
            "lion": "lion",
            "agile": "agile",
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
                enc,
                reduction=effective_reduction,
                **fp_kw,
            )
            encoded = dataset.encode_dataset(
                enc,
                reduction=effective_reduction,
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


def init_history(df, seed_idx, top_k_values=None):
    """Initialize a history dict from the seed pool.

    Records the initial best value, evaluation count, and (optionally)
    per-round top-k recall from the seed pool alone.

    Args:
        df: Encoded DataFrame with ``Experiment_value``.
        seed_idx: List of indices in ``df`` forming the initial seed pool.
        top_k_values: Optional dict mapping k (int) to sets of top-k
            indices. If provided, recall is tracked per round.

    Returns:
        History dict with keys ``best_so_far``, ``round_best``,
        ``n_evaluated``, ``all_evaluated``, and optionally
        ``per_round_recall``.
    """
    seed_vals = df.loc[seed_idx, "Experiment_value"]
    history = {
        "best_so_far": [float(seed_vals.max())],
        "round_best": [],
        "n_evaluated": [len(seed_idx)],
        "all_evaluated": set(seed_idx),
    }
    if top_k_values is not None:
        history["per_round_recall"] = {}
        for k, top_set in top_k_values.items():
            found = len(set(seed_idx) & top_set)
            history["per_round_recall"][k] = [found / len(top_set)]
    return history


def update_history(history, df, training_idx, batch_idx, round_num, top_k_values=None):
    """Append one round's results to the history dict.

    Args:
        history: History dict from ``init_history``.
        df: Encoded DataFrame with ``Experiment_value``.
        training_idx: Full list of indices evaluated so far (seed + all
            acquired batches).
        batch_idx: List of indices acquired in this round.
        round_num: Zero-indexed round number.
        top_k_values: Optional dict mapping k to top-k index sets for
            recall tracking.
    """
    batch_vals = df.loc[batch_idx, "Experiment_value"]
    all_vals = df.loc[training_idx, "Experiment_value"]
    history["best_so_far"].append(float(all_vals.max()))
    history["round_best"].append(float(batch_vals.max()))
    history["n_evaluated"].append(len(training_idx))
    history["all_evaluated"].update(batch_idx)
    if top_k_values is not None and "per_round_recall" in history:
        for k, top_set in top_k_values.items():
            found = len(history["all_evaluated"] & top_set)
            history["per_round_recall"][k].append(found / len(top_set))


def compute_metrics(history, top_k_values, n_total):
    """Compute final evaluation metrics from a completed benchmark history.

    Args:
        history: History dict with ``best_so_far``, ``n_evaluated``,
            ``all_evaluated``, and optionally ``per_round_recall``.
        top_k_values: Dict mapping k (int) to sets of top-k indices.
        n_total: Total number of formulations in the study.

    Returns:
        Dict with keys ``final_best``, ``auc`` (area under the
        best-so-far curve, normalized by evaluation budget), ``top_k_recall``
        (dict of recall fractions), ``n_rounds``, ``n_total_evaluated``,
        and optionally ``per_round_recall``.
    """
    bsf = np.array(history["best_so_far"])
    n_eval = np.array(history["n_evaluated"])
    evaluated = history["all_evaluated"]

    auc = float(np.trapezoid(bsf, n_eval) / (n_eval[-1] - n_eval[0])) if len(bsf) > 1 else bsf[0]

    recall = {}
    for k, top_set in top_k_values.items():
        found = len(evaluated & top_set)
        recall[k] = found / len(top_set)

    final_best = bsf[-1]

    result = {
        "final_best": float(final_best),
        "auc": auc,
        "top_k_recall": recall,
        "n_rounds": len(bsf) - 1,
        "n_total_evaluated": int(n_eval[-1]),
    }
    if "per_round_recall" in history:
        result["per_round_recall"] = {str(k): v for k, v in history["per_round_recall"].items()}
    return result


def _run_random(df, seed_idx, oracle_idx, batch_size, n_rounds, seed, top_k_values=None):
    """Run the random baseline strategy (uniform random acquisition).

    At each round, selects ``batch_size`` candidates uniformly at random
    from the remaining oracle pool without replacement.

    Args:
        df: Encoded DataFrame with ``Experiment_value``.
        seed_idx: List of initial seed pool indices.
        oracle_idx: List of candidate pool indices.
        batch_size: Number of candidates to acquire per round.
        n_rounds: Maximum number of acquisition rounds.
        seed: Integer RNG seed.
        top_k_values: Optional dict mapping k to top-k index sets for
            recall tracking.

    Returns:
        History dict (same structure as ``init_history``).
    """
    rng = np.random.RandomState(seed)
    training_idx = list(seed_idx)
    pool_idx = list(oracle_idx)
    history = init_history(df, training_idx, top_k_values=top_k_values)
    for r in range(n_rounds):
        if len(pool_idx) < batch_size:
            break
        chosen = rng.choice(len(pool_idx), size=batch_size, replace=False)
        batch_idx = [pool_idx[i] for i in sorted(chosen)]
        pool_set = set(batch_idx)
        pool_idx = [i for i in pool_idx if i not in pool_set]
        training_idx.extend(batch_idx)
        update_history(history, df, training_idx, batch_idx, r, top_k_values=top_k_values)
    return history


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_results(all_results, output_path="benchmark_output.png"):
    """Generate a two-panel benchmark summary figure.

    Panel A: Best-so-far convergence curves for all strategies.
    Panel B: Top-k recall bar chart comparing strategies at each k.

    Args:
        all_results: Dict mapping strategy name to result dict with
            ``history`` and ``metrics`` keys.
        output_path: File path for the saved PNG figure (300 dpi).
    """
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
    """CLI entry point for the standalone benchmark runner.

    Parses command-line arguments, loads and encodes the dataset, runs
    each requested strategy, prints a summary table, saves results to
    JSON, and optionally generates a summary plot.
    """
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
        choices=[
            "mfp",
            "mordred",
            "unimol",
            "count_mfp",
            "rdkit",
            "chemeleon",
            "lion",
            "raw_mfp",
            "raw_unimol",
            "raw_count_mfp",
            "raw_rdkit",
            "raw_chemeleon",
            "concat",
            "raw_concat",
            "lantern",
            "raw_lantern",
            "lantern_unimol",
            "raw_lantern_unimol",
            "lantern_mordred",
            "raw_lantern_mordred",
            "lantern_il_only",
            "lantern_il_hl",
            "lantern_il_noratios",
            "lion_il_only",
            "mordred_il_only",
            "unimol_il_only",
            "mfp_il_only",
            "count_mfp_il_only",
            "chemeleon_il_only",
            "chemeleon_helper_only",
            "agile",
            "agile_il_only",
            "ratios_only",
        ],
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
        "--fp-radius",
        type=int,
        default=None,
        help="Morgan FP radius (default: 3 for mfp, 3 for count_mfp)",
    )
    parser.add_argument(
        "--fp-bits",
        type=int,
        default=None,
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
    package_root = package_root_from(__file__, levels_up=2)
    results_dir = benchmark_results_root(package_root)
    results_dir.mkdir(parents=True, exist_ok=True)
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

    aitchison_data = None
    if any(s in AITCHISON_STRATEGIES for s in strategies):
        aitchison_data = prepare_benchmark_data(
            n_seed=args.n_seeds,
            random_seed=args.seed,
            subset=args.subset,
            reduction="none",
            feature_type="ratios_only",
            context_features=args.context_features,
        )

    compositional_data = None
    compositional_kernel_kwargs = None
    if any(s in COMPOSITIONAL_STRATEGIES or s in MIXED_STRATEGIES for s in strategies):
        compositional_data = prepare_benchmark_data(
            n_seed=args.n_seeds,
            random_seed=args.seed,
            subset=args.subset,
            reduction="pca",
            feature_type="lantern",
            context_features=args.context_features,
        )
        _, _comp_df, comp_fcols, _, _, _ = compositional_data
        compositional_kernel_kwargs = classify_feature_columns(comp_fcols)

    # Run strategies
    all_results = {}
    for strategy in strategies:
        print(f"\n{'=' * 70}")
        print(f"Running: {strategy}")
        print(f"{'=' * 70}")
        t0 = time.time()

        if (strategy in COMPOSITIONAL_STRATEGIES or strategy in MIXED_STRATEGIES) and compositional_data is not None:
            s_dataset, s_df, s_fcols, s_seed, s_oracle, s_topk = compositional_data
        elif strategy in AITCHISON_STRATEGIES and aitchison_data is not None:
            s_dataset, s_df, s_fcols, s_seed, s_oracle, s_topk = aitchison_data
        elif strategy in TANIMOTO_STRATEGIES and tanimoto_data is not None:
            s_dataset, s_df, s_fcols, s_seed, s_oracle, s_topk = tanimoto_data
        elif strategy in PLS_STRATEGIES and pls_data is not None:
            s_dataset, s_df, s_fcols, s_seed, s_oracle, s_topk = pls_data
        else:
            s_dataset, s_df, s_fcols, s_seed, s_oracle, s_topk = pca_data

        config = STRATEGY_CONFIGS[strategy]
        if config["type"] == "random":
            history = _run_random(s_df, s_seed, s_oracle, args.batch_size, args.rounds, args.seed)
        elif config["type"] == "discrete_online_conformal_exact":
            history = run_discrete_online_conformal_strategy(
                s_df,
                s_fcols,
                s_seed,
                s_oracle,
                batch_size=args.batch_size,
                n_rounds=args.rounds,
                seed=args.seed,
                kappa=args.kappa,
                normalize=args.normalize,
                encoded_dataset=s_dataset,
            )
        elif config["type"] == "discrete_online_conformal_baseline":
            history = run_discrete_cumulative_split_conformal_ucb_baseline(
                s_df,
                s_fcols,
                s_seed,
                s_oracle,
                batch_size=args.batch_size,
                n_rounds=args.rounds,
                seed=args.seed,
                kappa=args.kappa,
                normalize=args.normalize,
                encoded_dataset=s_dataset,
            )
        else:
            from LNPBO.optimization.optimizer import Optimizer

            from ._optimizer_runner import OptimizerRunner

            gp_kernel_kwargs = None
            if strategy in COMPOSITIONAL_STRATEGIES or strategy in MIXED_STRATEGIES:
                gp_kernel_kwargs = compositional_kernel_kwargs
            opt_kwargs = strategy_to_optimizer_kwargs(strategy, kernel_kwargs=gp_kernel_kwargs)
            optimizer = Optimizer(
                random_seed=args.seed,
                kappa=args.kappa,
                xi=args.xi,
                normalize=args.normalize,
                batch_size=args.batch_size,
                **opt_kwargs,
            )
            runner = OptimizerRunner(optimizer)
            history = runner.run(
                s_df,
                s_fcols,
                s_seed,
                s_oracle,
                n_rounds=args.rounds,
                batch_size=args.batch_size,
                encoded_dataset=s_dataset,
                top_k_values=s_topk,
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
