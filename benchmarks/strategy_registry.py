"""Single source of truth for strategy names, families, and display labels.

All analysis scripts should import from here rather than defining their own
copies of these mappings.
"""

# strategy_name -> family label (for grouping in analysis)
STRATEGY_FAMILY = {
    "random": "Random",
    # GP (BoTorch) strategies
    "lnpbo_ucb": "LNPBO (GP)",
    "lnpbo_ei": "LNPBO (GP)",
    "lnpbo_logei": "LNPBO (GP)",
    "lnpbo_lp_ei": "LNPBO (GP)",
    "lnpbo_lp_logei": "LNPBO (GP)",
    "lnpbo_pls_logei": "LNPBO (GP)",
    "lnpbo_pls_lp_logei": "LNPBO (GP)",
    "lnpbo_rkb_logei": "LNPBO (GP)",
    "lnpbo_ts_batch": "LNPBO (GP)",
    "lnpbo_gibbon": "LNPBO (GP)",
    "lnpbo_tanimoto_ts": "LNPBO (GP)",
    "lnpbo_tanimoto_logei": "LNPBO (GP)",
    "lnpbo_aitchison_ts": "LNPBO (GP)",
    "lnpbo_aitchison_logei": "LNPBO (GP)",
    "lnpbo_dkl_ts": "LNPBO (GP)",
    "lnpbo_dkl_logei": "LNPBO (GP)",
    "lnpbo_rf_kernel_ts": "LNPBO (GP)",
    "lnpbo_rf_kernel_logei": "LNPBO (GP)",
    "lnpbo_compositional_ts": "LNPBO (GP)",
    "lnpbo_compositional_logei": "LNPBO (GP)",
    "lnpbo_mixed_logei": "LNPBO (GP)",
    "lnpbo_mixed_ts": "LNPBO (GP)",
    # Discrete surrogates
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
    "discrete_ridge_ucb": "Ridge",
    "discrete_gp_ucb": "GP (sklearn)",
    "discrete_tabpfn": "TabPFN",
    # CASMOPolitan
    "casmopolitan_ei": "CASMOPolitan",
    "casmopolitan_ucb": "CASMOPolitan",
}

# strategy_name -> short display label (for plots/tables)
STRATEGY_SHORT = {
    "random": "Random",
    "lnpbo_ucb": "GP-UCB",
    "lnpbo_ei": "GP-EI",
    "lnpbo_logei": "GP-LogEI",
    "lnpbo_lp_ei": "GP-LP-EI",
    "lnpbo_lp_logei": "GP-LP-LogEI",
    "lnpbo_pls_logei": "GP-PLS-LogEI",
    "lnpbo_pls_lp_logei": "GP-PLS-LP-LogEI",
    "lnpbo_rkb_logei": "GP-RKB-LogEI",
    "lnpbo_ts_batch": "GP-TS-Batch",
    "lnpbo_gibbon": "GP-GIBBON",
    "lnpbo_tanimoto_ts": "GP-Tani-TS",
    "lnpbo_tanimoto_logei": "GP-Tani-LogEI",
    "lnpbo_aitchison_ts": "GP-Aitch-TS",
    "lnpbo_aitchison_logei": "GP-Aitch-LogEI",
    "lnpbo_dkl_ts": "GP-DKL-TS",
    "lnpbo_dkl_logei": "GP-DKL-LogEI",
    "lnpbo_rf_kernel_ts": "GP-RFKernel-TS",
    "lnpbo_rf_kernel_logei": "GP-RFKernel-LogEI",
    "lnpbo_compositional_ts": "GP-Comp-TS",
    "lnpbo_compositional_logei": "GP-Comp-LogEI",
    "lnpbo_mixed_logei": "GP-Mixed-LogEI",
    "lnpbo_mixed_ts": "GP-Mixed-TS",
    "casmopolitan_ei": "CASMO-EI",
    "casmopolitan_ucb": "CASMO-UCB",
    "discrete_rf_ucb": "RF-UCB",
    "discrete_rf_ts": "RF-TS",
    "discrete_rf_ts_batch": "RF-TS-Batch",
    "discrete_xgb_ucb": "XGB-UCB",
    "discrete_xgb_greedy": "XGB-Greedy",
    "discrete_xgb_cqr": "XGB-CQR",
    "discrete_xgb_online_conformal": "XGB-OnlineConf",
    "discrete_xgb_ucb_ts_batch": "XGB-UCB-TS-Batch",
    "discrete_ngboost_ucb": "NGBoost-UCB",
    "discrete_deep_ensemble": "DeepEnsemble",
    "discrete_ridge_ucb": "Ridge-UCB",
    "discrete_gp_ucb": "GP-UCB (sklearn)",
    "discrete_tabpfn": "TabPFN",
}

# family_label -> list of strategies (inverse of STRATEGY_FAMILY)
STRATEGY_FAMILIES = {}
for _strat, _fam in STRATEGY_FAMILY.items():
    STRATEGY_FAMILIES.setdefault(_fam, []).append(_strat)

FAMILY_COLORS = {
    "LNPBO (GP)": "#1f77b4",
    "RF": "#2ca02c",
    "XGBoost": "#d62728",
    "NGBoost": "#ff7f0e",
    "Deep Ensemble": "#9467bd",
    "GP (sklearn)": "#8c564b",
    "Ridge": "#e377c2",
    "CASMOPolitan": "#17becf",
    "TabPFN": "#bcbd22",
    "Random": "#7f7f7f",
}


def strategy_to_family(strategy_name):
    """Map a strategy name to its family label, with fallback."""
    return STRATEGY_FAMILY.get(strategy_name, strategy_name)


def strategy_short_name(strategy_name):
    """Map a strategy name to its short display label."""
    return STRATEGY_SHORT.get(strategy_name, strategy_name)
