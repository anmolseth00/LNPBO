"""Strategy registry and lightweight runner utilities."""

from __future__ import annotations

from LNPBO.optimization.optimizer import ENC_PREFIXES

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
TANIMOTO_STRATEGIES = {"lnpbo_tanimoto_ts", "lnpbo_tanimoto_logei"}
AITCHISON_STRATEGIES = {"lnpbo_aitchison_ts", "lnpbo_aitchison_logei"}
COMPOSITIONAL_STRATEGIES = {"lnpbo_compositional_ts", "lnpbo_compositional_logei"}
MIXED_STRATEGIES = {"lnpbo_mixed_logei", "lnpbo_mixed_ts"}

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
    "TS_Batch": ("UCB", "ts", "matern"),
    "GIBBON": ("UCB", "gibbon", "matern"),
    "Tanimoto_TS": ("UCB", "ts", "tanimoto"),
    "Tanimoto_LogEI": ("LogEI", "kb", "tanimoto"),
    "Aitchison_TS": ("UCB", "ts", "aitchison"),
    "Aitchison_LogEI": ("LogEI", "kb", "aitchison"),
    "DKL_TS": ("UCB", "ts", "dkl"),
    "DKL_LogEI": ("LogEI", "kb", "dkl"),
    "RF_Kernel_TS": ("UCB", "ts", "rf"),
    "RF_Kernel_LogEI": ("LogEI", "kb", "rf"),
    "Compositional_TS": ("UCB", "ts", "compositional"),
    "Compositional_LogEI": ("LogEI", "kb", "compositional"),
    "Mixed_LogEI": ("LogEI", "kb", "compositional"),
    "Mixed_TS": ("UCB", "ts", "compositional"),
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


def classify_feature_columns(feature_cols):
    """Classify feature columns into fingerprint, ratio, and synthesis groups."""
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


def strategy_to_optimizer_kwargs(strategy_name, kernel_kwargs=None):
    """Map a strategy registry entry to ``Optimizer`` keyword arguments."""
    config = STRATEGY_CONFIGS[strategy_name]
    stype = config["type"]

    if stype == "random":
        raise ValueError("random strategy does not use Optimizer")
    if stype in {"discrete_online_conformal_exact", "discrete_online_conformal_baseline"}:
        raise ValueError(f"{strategy_name} has custom logic and does not use Optimizer")

    kwargs = {}
    if stype == "gp":
        base_acq, batch_strategy, kernel_type = ACQ_TYPE_MAP[config["acq_type"]]
        kwargs["surrogate_type"] = "gp"
        kwargs["gp_engine"] = "botorch"
        kwargs["acquisition_type"] = base_acq
        kwargs["batch_strategy"] = batch_strategy
        kwargs["kernel_type"] = kernel_type
        if kernel_kwargs is not None:
            kwargs["kernel_kwargs"] = kernel_kwargs
    elif stype == "gp_mixed":
        base_acq, batch_strategy, kernel_type = ACQ_TYPE_MAP[config["acq_type"]]
        kwargs["surrogate_type"] = "gp_mixed"
        kwargs["acquisition_type"] = base_acq
        kwargs["batch_strategy"] = batch_strategy
        kwargs["kernel_type"] = kernel_type
        if kernel_kwargs is not None:
            kwargs["kernel_kwargs"] = kernel_kwargs
    elif stype in ("discrete", "discrete_ts_batch"):
        kwargs["surrogate_type"] = config["surrogate"]
        kwargs["batch_strategy"] = "ts" if stype == "discrete_ts_batch" else "greedy"
    elif stype == "casmopolitan":
        kwargs["surrogate_type"] = "casmopolitan"
        acq_func = config.get("acq_func", "ucb")
        kwargs["acquisition_type"] = acq_func.upper() if acq_func != "ei" else "EI"
    else:
        raise ValueError(f"Unknown strategy type: {stype!r}")

    return kwargs
