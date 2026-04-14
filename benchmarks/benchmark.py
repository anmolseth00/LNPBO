#!/usr/bin/env python3
"""Within-Study Benchmark.

Runs all strategies within individual LNPDB studies to evaluate optimization
performance under realistic conditions — same lab, same assay, same readout.

Cross-study pooling conflates batch effects with real structure-activity
relationships. Within-study evaluation isolates genuine optimization signal.

For each qualifying study:
  - n_seed is set dynamically (seed_fraction * study_size, min 30)
  - batch_size and n_rounds are kept consistent where pool permits
  - Top-k recall is computed at study-relative percentiles (top 5%, 10%, 20%)
  - Feature type is adapted: studies with 1 unique IL use ratios_only

Usage:
    # Run all strategies on all qualifying studies:
    python -m benchmarks.benchmark

    # Run specific strategies:
    python -m benchmarks.benchmark --strategies random,discrete_xgb_ucb

    # Run on specific studies:
    python -m benchmarks.benchmark --pmids 39060305,37985700

    # Resume (skip existing per-seed results):
    python -m benchmarks.benchmark --resume

    # Aggregate only (no new runs):
    python -m benchmarks.benchmark --aggregate-only

    # Dry run:
    python -m benchmarks.benchmark --dry-run
"""

import argparse
import contextlib
import json
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np

# gpytorch spams NumericalWarning each time it retries Cholesky with extra
# jitter. It's a benign auto-stabilization mechanism, not a failure — but on
# large mixed-variable studies it can emit hundreds of lines per iteration.
# Silence it so progress output stays readable. Real fit failures still raise.
with contextlib.suppress(ImportError):
    from gpytorch.utils.warnings import NumericalWarning

    warnings.filterwarnings("ignore", category=NumericalWarning)

from ..data.context import infer_assay_type_row

from .constants import BATCH_SIZE, MAX_ROUNDS, MIN_SEED, MIN_STUDY_SIZE, SEED_FRACTION, SEEDS
from .runner import (
    COMPOSITIONAL_STRATEGIES,
    MIXED_STRATEGIES,
    STRATEGY_CONFIGS,
    STRATEGY_DISPLAY,
    TANIMOTO_STRATEGIES,
    _run_random,
    classify_feature_columns,
    compute_metrics,
    prepare_benchmark_data,
)
from .stats import bootstrap_ci

# Strategies to benchmark (same as comprehensive, minus TabPFN)
DISCRETE_STRATEGIES = [
    "random",
    "discrete_xgb_greedy",
    "discrete_xgb_ucb",
    "discrete_xgb_cqr",
    "discrete_ngboost_ucb",
    "discrete_rf_ucb",
    "discrete_rf_ts",
    "discrete_deep_ensemble",
    "discrete_ridge_ucb",
    "discrete_gp_ucb",
]

TS_BATCH_STRATEGIES = [
    "discrete_rf_ts_batch",
    "discrete_xgb_ucb_ts_batch",
]

ONLINE_CONFORMAL_STRATEGIES = [
    "discrete_xgb_online_conformal",
]

CASMOPOLITAN_STRATEGIES = [
    "casmopolitan_ucb",
    "casmopolitan_ei",
]

GP_STRATEGIES = [
    "lnpbo_ucb",
    "lnpbo_ei",
    "lnpbo_logei",
    "lnpbo_rkb_logei",
    "lnpbo_lp_ei",
    "lnpbo_lp_logei",
    "lnpbo_pls_logei",
    "lnpbo_ts_batch",
    "lnpbo_pls_lp_logei",
    "lnpbo_gibbon",
    "lnpbo_tanimoto_ts",
    "lnpbo_tanimoto_logei",
    "lnpbo_aitchison_ts",
    "lnpbo_aitchison_logei",
    "lnpbo_dkl_ts",
    "lnpbo_dkl_logei",
    "lnpbo_rf_kernel_ts",
    "lnpbo_rf_kernel_logei",
    "lnpbo_compositional_ts",
    "lnpbo_compositional_logei",
    "lnpbo_mixed_logei",
    "lnpbo_mixed_ts",
]

ALL_WITHIN_STUDY_STRATEGIES = (
    DISCRETE_STRATEGIES + TS_BATCH_STRATEGIES + ONLINE_CONFORMAL_STRATEGIES + CASMOPOLITAN_STRATEGIES + GP_STRATEGIES
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "benchmark_results" / "within_study"


# ---------------------------------------------------------------------------
# Study characterization
# ---------------------------------------------------------------------------


def get_study_id(study_info):
    """Extract canonical study ID from a study_info dict.

    Uses the ``study_id`` key if present (e.g., for sub-PMID splits like
    per-target studies), otherwise derives the ID from the PMID by
    converting to int.

    Args:
        study_info: Dict with at least ``pmid`` and optionally ``study_id``.

    Returns:
        Canonical string study identifier.
    """
    return study_info.get("study_id", str(int(float(study_info["pmid"]))))


def characterize_studies(df, min_size=MIN_STUDY_SIZE, seed_fraction=SEED_FRACTION):
    """Identify and characterize qualifying studies from LNPDB.

    Scans all studies (grouped by ``Publication_PMID``), filters by minimum
    size, classifies each study as IL-diverse (fixed or variable ratios) or
    ratio-only, and computes per-study benchmark parameters (seed pool size,
    number of rounds, feature type, top-k percentile thresholds).

    Args:
        df: Full LNPDB DataFrame with ``Publication_PMID``,
            ``IL_SMILES``, ``HL_name``, ``CHL_name``, ``PEG_name``,
            and ``IL_molratio`` columns.
        min_size: Minimum number of formulations for a study to qualify.
        seed_fraction: Fraction of study size to use as the initial seed
            pool (subject to ``MIN_SEED`` floor).

    Returns:
        List of study metadata dicts, each containing ``pmid``,
        ``n_formulations``, ``study_type``, ``n_seed``, ``n_rounds``,
        ``batch_size``, ``feature_type``, ``top_k_pct``, and component
        diversity counts.
    """
    study_counts = df.groupby("Publication_PMID").size().sort_values(ascending=False)
    studies = []

    # Build (pmid, sub_df, mt_name) groups -- split multi-Model_type PMIDs
    # because z-scores are computed per (Experiment_ID, Model_type).
    study_groups = []
    for pmid, _total_n in study_counts.items():
        full_sub = df[df["Publication_PMID"] == pmid]
        n_mt = full_sub["Model_type"].nunique() if "Model_type" in full_sub.columns else 1
        if n_mt > 1:
            for mt, grp in full_sub.groupby("Model_type"):
                study_groups.append((pmid, grp, str(mt)))
        else:
            study_groups.append((pmid, full_sub, None))

    for pmid, sub, mt_name in study_groups:
        n = len(sub)
        if n < min_size:
            continue
        n_unique_il = sub["IL_SMILES"].nunique()
        n_unique_hl = sub["HL_name"].nunique()
        n_unique_chl = sub["CHL_name"].nunique()
        n_unique_peg = sub["PEG_name"].nunique()
        il_ratio_std = sub["IL_molratio"].std() if "IL_molratio" in sub.columns else 0.0

        # Classify study type (IL diversity)
        if n_unique_il <= 1:
            study_type = "ratio_only"
        elif il_ratio_std < 1.0:
            study_type = "il_diverse_fixed_ratios"
        else:
            study_type = "il_diverse_variable_ratios"

        # Component diversity: do HL/CHL/PEG also vary?
        if n_unique_il <= 1 and n_unique_hl <= 1 and n_unique_chl <= 1 and n_unique_peg <= 1:
            component_diversity = "ratio_only"
        elif n_unique_hl <= 1 and n_unique_chl <= 1 and n_unique_peg <= 1:
            component_diversity = "single_component"
        else:
            component_diversity = "multi_component"

        # Metadata dimensions: assay type, cell type, cargo
        assay_type = sub.apply(infer_assay_type_row, axis=1).mode()
        assay_type = assay_type.iloc[0] if len(assay_type) > 0 else "unknown"
        in_vivo = assay_type.startswith("in_vivo")

        model_type = "unknown"
        if "Model_type" in sub.columns:
            mt = sub["Model_type"].dropna()
            if len(mt) > 0:
                model_type = mt.mode().iloc[0]

        cargo_type = "unknown"
        if "Cargo_type" in sub.columns:
            ct = sub["Cargo_type"].dropna()
            if len(ct) > 0:
                cargo_type = ct.mode().iloc[0]

        # Dynamic n_seed
        n_seed = max(MIN_SEED, int(seed_fraction * n))

        # Adjust rounds: cap acquisitions at 50% of oracle pool so metrics
        # are meaningful (exploring >50% trivially finds top-k)
        oracle_size = n - n_seed
        max_acquisitions = int(0.5 * oracle_size)
        max_rounds_feasible = max(1, max_acquisitions // BATCH_SIZE)
        n_rounds = min(MAX_ROUNDS, max_rounds_feasible)

        # Feature type: no molecular features if only 1 IL
        if n_unique_il <= 1:
            feature_type = "ratios_only"
        else:
            feature_type = "lantern_il_only"

        # Top-k percentiles (relative to study size)
        top_k_abs = {}
        for pct in [5, 10, 20]:
            k = max(1, int(n * pct / 100))
            top_k_abs[pct] = k

        pmid_str = str(int(float(pmid)))
        study_id = f"{pmid_str}_{mt_name}" if mt_name else pmid_str

        studies.append(
            {
                "pmid": pmid,
                "study_id": study_id,
                "model_type_filter": mt_name,
                "n_formulations": int(n),
                "n_unique_il": int(n_unique_il),
                "n_unique_hl": int(n_unique_hl),
                "n_unique_chl": int(n_unique_chl),
                "n_unique_peg": int(n_unique_peg),
                "il_ratio_std": float(il_ratio_std),
                "study_type": study_type,
                "component_diversity": component_diversity,
                "assay_type": assay_type,
                "model_type": model_type,
                "cargo_type": cargo_type,
                "in_vivo": in_vivo,
                "n_seed": n_seed,
                "n_rounds": n_rounds,
                "batch_size": BATCH_SIZE,
                "feature_type": feature_type,
                "top_k_pct": top_k_abs,
            }
        )

    return studies


def ensure_top_k_pct(study_infos):
    """Add ``top_k_pct`` to study_info dicts that lack it.

    The studies JSON files produced by ``characterize_studies`` may not
    persist ``top_k_pct``, but ``prepare_study_data`` requires it.
    This function computes it from ``n_formulations`` for the 5th, 10th,
    and 20th percentile thresholds.

    Args:
        study_infos: List of study_info dicts to mutate in place.

    Returns:
        The same list (mutated), for convenience in chaining.
    """
    for si in study_infos:
        if "top_k_pct" not in si:
            n = si["n_formulations"]
            si["top_k_pct"] = {pct: max(1, int(n * pct / 100)) for pct in [5, 10, 20]}
    return study_infos


# ---------------------------------------------------------------------------
# Per-seed runner
# ---------------------------------------------------------------------------


def run_single_seed(
    strategy,
    random_seed,
    study_info,
    pca_data=None,
    kernel_kwargs=None,
):
    """Run a single strategy for a single seed within a study.

    For ``random`` and ``discrete_xgb_online_conformal``, delegates to their
    dedicated runners. All other 32 strategies go through
    ``OptimizerRunner`` backed by ``Optimizer.suggest_indices()``.

    Args:
        strategy: Strategy name key into ``STRATEGY_CONFIGS``.
        random_seed: Integer RNG seed for reproducibility.
        study_info: Study metadata dict (from ``characterize_studies``),
            must include ``batch_size``, ``n_rounds``, and ``top_k_pct``.
        pca_data: Tuple of ``(encoded_dataset, encoded_df, feature_cols,
            seed_idx, oracle_idx, top_k_values)`` from ``prepare_study_data``.
        kernel_kwargs: Optional dict of kernel configuration for
            compositional product kernel GP strategies (keys:
            ``fp_indices``, ``ratio_indices``, ``synth_indices``).

    Returns:
        Dict with keys ``metrics``, ``elapsed``, ``best_so_far``,
        ``round_best``, ``n_evaluated``, and optionally ``coverage``
        (for online conformal strategies).

    Raises:
        ValueError: If ``pca_data`` is None or strategy type is unknown.
    """
    batch_size = study_info["batch_size"]
    n_rounds = study_info["n_rounds"]
    normalize = "copula"
    kappa = 5.0
    xi = 0.01

    if pca_data is None:
        raise ValueError("pca_data must be provided")

    s_dataset, s_df, s_fcols, s_seed, s_oracle, s_topk = pca_data
    config = STRATEGY_CONFIGS[strategy]
    t0 = time.time()

    if config["type"] == "random":
        history = _run_random(s_df, s_seed, s_oracle, batch_size, n_rounds, random_seed, top_k_values=s_topk)

    elif config["type"] == "discrete_online_conformal":
        from .runner import run_discrete_online_conformal_strategy

        history = run_discrete_online_conformal_strategy(
            s_df,
            s_fcols,
            s_seed,
            s_oracle,
            batch_size=batch_size,
            n_rounds=n_rounds,
            seed=random_seed,
            kappa=kappa,
            normalize=normalize,
            encoded_dataset=s_dataset,
            top_k_values=s_topk,
        )

    else:
        # All other 32 strategies go through Optimizer + OptimizerRunner
        from LNPBO.optimization.optimizer import Optimizer

        from ._optimizer_runner import OptimizerRunner
        from .runner import strategy_to_optimizer_kwargs

        opt_kwargs = strategy_to_optimizer_kwargs(strategy, kernel_kwargs=kernel_kwargs)
        optimizer = Optimizer(
            random_seed=random_seed,
            kappa=kappa,
            xi=xi,
            normalize=normalize,
            batch_size=batch_size,
            **opt_kwargs,
        )

        runner = OptimizerRunner(optimizer)
        history = runner.run(
            s_df,
            s_fcols,
            s_seed,
            s_oracle,
            n_rounds=n_rounds,
            batch_size=batch_size,
            encoded_dataset=s_dataset,
            top_k_values=s_topk,
        )

    elapsed = time.time() - t0
    metrics = compute_metrics(history, s_topk, len(s_df))
    metrics["top_k_recall"] = {str(k): v for k, v in metrics["top_k_recall"].items()}

    result = {
        "metrics": metrics,
        "elapsed": elapsed,
        "best_so_far": history["best_so_far"],
        "round_best": history["round_best"],
        "n_evaluated": history["n_evaluated"],
    }
    if "coverage" in history:
        result["coverage"] = history["coverage"]

    return result


# ---------------------------------------------------------------------------
# Result I/O
# ---------------------------------------------------------------------------


def _per_seed_path(pmid, strategy, seed, *, study_id=None):
    """Build the filesystem path for a per-seed result JSON file.

    Args:
        pmid: Publication PMID (string or numeric).
        strategy: Strategy name.
        seed: Random seed integer.
        study_id: Optional study ID override (used for sub-PMID splits).

    Returns:
        pathlib.Path to the result file under ``RESULTS_DIR``.
    """
    folder = study_id if study_id else str(int(float(pmid)))
    return RESULTS_DIR / folder / f"{strategy}_s{seed}.json"


def save_seed_result(pmid, strategy, seed, result, study_info):
    """Persist a single seed's benchmark result to a JSON file.

    Args:
        pmid: Publication PMID.
        strategy: Strategy name.
        seed: Random seed integer.
        result: Result dict from ``run_single_seed``.
        study_info: Study metadata dict (included in the saved JSON).
    """
    path = _per_seed_path(pmid, strategy, seed, study_id=study_info.get("study_id"))
    path.parent.mkdir(parents=True, exist_ok=True)
    study_id = get_study_id(study_info)
    data = {
        "pmid": pmid,
        "study_id": study_id,
        "strategy": strategy,
        "seed": seed,
        "study_info": {k: v for k, v in study_info.items() if k != "top_k_pct"},
        "top_k_pct": study_info["top_k_pct"],
        "result": result,
        "timestamp": datetime.now().isoformat(),
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_seed_result(pmid, strategy, seed, *, study_id=None):
    """Load a previously saved per-seed benchmark result.

    Args:
        pmid: Publication PMID.
        strategy: Strategy name.
        seed: Random seed integer.
        study_id: Optional study ID override for sub-PMID splits.

    Returns:
        The ``result`` dict if the file exists and is valid JSON, otherwise
        None.
    """
    path = _per_seed_path(pmid, strategy, seed, study_id=study_id)
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    return data.get("result")


# ---------------------------------------------------------------------------
# Data preparation (per study)
# ---------------------------------------------------------------------------


def filter_study_df(df, study_info):
    """Filter full LNPDB DataFrame to a single study.

    Supports corrected study definitions with ``lnp_ids`` for sub-PMID
    filtering, ``model_type_filter`` for cell-line splits, and ``suffix``
    for Model_target splits, as well as classic PMID-only filtering.
    """
    pmid = study_info["pmid"]
    lnp_ids = study_info.get("lnp_ids")
    model_type_filter = study_info.get("model_type_filter")
    suffix = study_info.get("suffix")

    if lnp_ids:
        study_df = df[df["LNP_ID"].isin(lnp_ids)].copy()
    elif model_type_filter and "Model_type" in df.columns:
        pmid_df = df[df["Publication_PMID"] == float(pmid)]
        study_df = pmid_df[pmid_df["Model_type"] == model_type_filter].copy()
    elif suffix and suffix != "pooled" and "Model_target" in df.columns:
        pmid_df = df[df["Publication_PMID"] == float(pmid)]
        # Match suffix against Model_target (try exact, then with underscore→space)
        study_df = pmid_df[pmid_df["Model_target"].str.lower() == suffix.lower()].copy()
        if len(study_df) == 0:
            target = suffix.replace("_", " ")
            study_df = pmid_df[pmid_df["Model_target"].str.lower() == target.lower()].copy()
        if len(study_df) == 0:
            study_df = pmid_df.copy()
    else:
        study_df = df[df["Publication_PMID"] == float(pmid)].copy()

    study_df = study_df.reset_index(drop=True)
    study_df["Formulation_ID"] = range(1, len(study_df) + 1)
    return study_df


def prepare_study_data(df, study_info, random_seed, *, raw=False):
    """Filter dataset to a single study and prepare benchmark data.

    Applies ``filter_study_df`` to isolate the study, encodes molecular
    features, splits into seed and oracle pools, and computes study-relative
    top-k percentile index sets.

    Args:
        df: Full LNPDB DataFrame.
        study_info: Study metadata dict (must include ``pmid``, ``n_seed``,
            ``feature_type``, and ``top_k_pct``).
        random_seed: Integer RNG seed for the train/oracle split.
        raw: If True, skip dimensionality reduction (``reduction='none'``).
            Used by Tanimoto kernel strategies that need full-dimensional
            count Morgan fingerprints (2048-d).

    Returns:
        Tuple of ``(encoded_dataset, encoded_df, feature_cols, seed_idx,
        oracle_idx, top_k_values_clean)`` where ``top_k_values_clean``
        maps percentile int keys to sets of DataFrame indices.
    """
    study_df = filter_study_df(df, study_info)

    n_seed = study_info["n_seed"]
    feature_type = study_info["feature_type"]

    if raw:
        reduction = "none"
    else:
        reduction = "pca" if feature_type != "ratios_only" else "none"

    data = prepare_benchmark_data(
        n_seed=n_seed,
        random_seed=random_seed,
        reduction=reduction,
        feature_type=feature_type,
        data_df=study_df,
    )

    encoded, encoded_df, feature_cols, seed_idx, oracle_idx, _ = data

    # Compute top-k on the encoded (valid-rows) df
    top_k_values_clean = {}
    for pct, k in study_info["top_k_pct"].items():
        actual_k = min(k, len(encoded_df))
        top_k_values_clean[pct] = set(encoded_df.nlargest(actual_k, "Experiment_value").index)

    return encoded, encoded_df, feature_cols, seed_idx, oracle_idx, top_k_values_clean


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate_study_strategy(pmid, strategy, seed_results, random_seed_results=None):
    """Aggregate per-seed results for one strategy within one study.

    Computes mean, standard deviation, and bootstrap 95% CI of top-k
    recall across seeds for the 5th, 10th, and 20th percentile thresholds,
    plus AUC and elapsed time statistics.

    Args:
        pmid: Study identifier (PMID string or study_id).
        strategy: Strategy name.
        seed_results: Dict mapping seed (int) to result dicts from
            ``run_single_seed``.
        random_seed_results: Unused; retained for API compatibility.

    Returns:
        Summary dict with keys ``strategy``, ``display_name``, ``pmid``,
        ``n_seeds``, ``top_5pct_recall``, ``top_10pct_recall``,
        ``top_20pct_recall``, ``auc``, and ``elapsed_seconds``.
        Returns None if ``seed_results`` is empty.
    """
    seeds_available = sorted(seed_results.keys())
    n_seeds = len(seeds_available)
    if n_seeds == 0:
        return None

    def _get_recall(seed_result, pct_key):
        """Extract top-k recall for a percentile key, handling str/int keys."""
        recall_dict = seed_result["metrics"]["top_k_recall"]
        return recall_dict.get(str(pct_key), recall_dict.get(pct_key, 0.0))

    summary = {
        "strategy": strategy,
        "display_name": STRATEGY_DISPLAY.get(strategy, strategy),
        "pmid": pmid,
        "n_seeds": n_seeds,
    }

    for pct in [5, 10, 20]:
        vals = np.array([_get_recall(seed_results[s], pct) for s in seeds_available])
        ci_lo, ci_hi = bootstrap_ci(vals) if len(vals) >= 3 else (float("nan"), float("nan"))
        summary[f"top_{pct}pct_recall"] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)) if n_seeds > 1 else 0.0,
            "ci_low": ci_lo,
            "ci_high": ci_hi,
            "per_seed": {int(s): float(v) for s, v in zip(seeds_available, vals)},
        }

    auc_vals = np.array([seed_results[s]["metrics"]["auc"] for s in seeds_available])
    summary["auc"] = {
        "mean": float(np.mean(auc_vals)),
        "std": float(np.std(auc_vals, ddof=1)) if n_seeds > 1 else 0.0,
    }

    elapsed_vals = np.array([seed_results[s]["elapsed"] for s in seeds_available])
    summary["elapsed_seconds"] = {"mean": float(np.mean(elapsed_vals))}

    return summary


# ---------------------------------------------------------------------------
# Markdown output
# ---------------------------------------------------------------------------


def generate_within_study_markdown(all_summaries, study_infos):
    """Generate a markdown report summarizing within-study benchmark results.

    Produces a study overview table, per-study strategy rankings (sorted by
    top-10% recall), and a cross-study summary (mean of per-study means).

    Args:
        all_summaries: Nested dict ``{study_id: {strategy: summary_dict}}``,
            where each summary_dict is the output of
            ``aggregate_study_strategy``.
        study_infos: List of study metadata dicts from
            ``characterize_studies``.

    Returns:
        Markdown-formatted string.
    """
    lines = []
    lines.append("# Within-Study Benchmark Results")
    lines.append("")
    lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"**Seed fraction:** {SEED_FRACTION}")
    lines.append(f"**Batch size:** {BATCH_SIZE}")
    lines.append(f"**Max rounds:** {MAX_ROUNDS}")
    lines.append(f"**Min study size:** {MIN_STUDY_SIZE}")
    lines.append(f"**Seeds:** {SEEDS}")
    lines.append("**Normalize:** copula")
    lines.append("")

    # Study overview table
    lines.append("## Study Overview")
    lines.append("")
    lines.append("| Study | N | ILs | HLs | Type | n_seed | Rounds | Features |")
    lines.append("|-------|---|-----|-----|------|--------|--------|----------|")
    for si in study_infos:
        lines.append(
            f"| {get_study_id(si)} | {si['n_formulations']} | "
            f"{si['n_unique_il']} | {si['n_unique_hl']} | "
            f"{si['study_type']} | {si['n_seed']} | {si['n_rounds']} | "
            f"{si['feature_type']} |"
        )
    lines.append("")

    # Per-study results
    for si in study_infos:
        sid = get_study_id(si)
        study_summaries = all_summaries.get(sid, {})
        if not study_summaries:
            continue

        lines.append(f"## {sid}")
        lines.append(
            f"N={si['n_formulations']}, {si['n_unique_il']} ILs, "
            f"type={si['study_type']}, n_seed={si['n_seed']}, "
            f"rounds={si['n_rounds']}, features={si['feature_type']}"
        )
        lines.append("")

        ranked = sorted(
            study_summaries.items(),
            key=lambda x: x[1].get("top_10pct_recall", {}).get("mean", 0),
            reverse=True,
        )

        lines.append("| Rank | Strategy | Top-5% | Top-10% | Top-20% | AUC | Time (s) |")
        lines.append("|------|----------|--------|---------|---------|-----|----------|")

        for rank, (strat, s) in enumerate(ranked, 1):
            t5 = s.get("top_5pct_recall", {})
            t10 = s.get("top_10pct_recall", {})
            t20 = s.get("top_20pct_recall", {})
            auc = s.get("auc", {})
            elapsed = s.get("elapsed_seconds", {})

            t5_str = f"{t5.get('mean', 0):.1%}" if t5 else "N/A"
            t10_str = f"{t10.get('mean', 0):.1%} ±{t10.get('std', 0):.1%}" if t10 else "N/A"
            t20_str = f"{t20.get('mean', 0):.1%}" if t20 else "N/A"
            auc_str = f"{auc.get('mean', 0):.2f}" if auc else "N/A"
            time_str = f"{elapsed.get('mean', 0):.1f}" if elapsed else "N/A"

            lines.append(
                f"| {rank} | {s.get('display_name', strat)} | "
                f"{t5_str} | {t10_str} | {t20_str} | {auc_str} | {time_str} |"
            )

        lines.append("")

    # Cross-study summary: mean of means
    lines.append("## Cross-Study Summary (mean of per-study means)")
    lines.append("")

    strategy_means = {}
    for _pmid_str, study_summaries in all_summaries.items():
        for strat, s in study_summaries.items():
            if strat not in strategy_means:
                strategy_means[strat] = {
                    "top_5": [],
                    "top_10": [],
                    "top_20": [],
                    "display": s.get("display_name", strat),
                }
            t5 = s.get("top_5pct_recall", {}).get("mean", 0)
            t10 = s.get("top_10pct_recall", {}).get("mean", 0)
            t20 = s.get("top_20pct_recall", {}).get("mean", 0)
            strategy_means[strat]["top_5"].append(t5)
            strategy_means[strat]["top_10"].append(t10)
            strategy_means[strat]["top_20"].append(t20)

    ranked_overall = sorted(
        strategy_means.items(),
        key=lambda x: np.mean(x[1]["top_10"]) if x[1]["top_10"] else 0,
        reverse=True,
    )

    lines.append("| Rank | Strategy | Mean Top-10% | Std | N studies |")
    lines.append("|------|----------|-------------|-----|----------|")
    for rank, (_strat, d) in enumerate(ranked_overall, 1):
        mean_t10 = np.mean(d["top_10"]) if d["top_10"] else 0
        std_t10 = np.std(d["top_10"], ddof=1) if len(d["top_10"]) > 1 else 0
        lines.append(f"| {rank} | {d['display']} | {mean_t10:.1%} | {std_t10:.1%} | {len(d['top_10'])} |")

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    """CLI entry point for the within-study benchmark suite.

    Parses command-line arguments, loads LNPDB, characterizes qualifying
    studies, runs all requested strategy/seed combinations (with optional
    resume support), aggregates results, and writes markdown and JSON
    summaries to the results directory.
    """
    parser = argparse.ArgumentParser(
        description="Within-Study Benchmark Suite",
    )
    parser.add_argument(
        "--strategies",
        type=str,
        default=None,
        help="Comma-separated list of strategies (default: all non-GP)",
    )
    parser.add_argument(
        "--pmids",
        type=str,
        default=None,
        help="Comma-separated list of PMIDs to benchmark (default: all qualifying)",
    )
    parser.add_argument(
        "--min-study-size",
        type=int,
        default=MIN_STUDY_SIZE,
        help=f"Minimum study size (default: {MIN_STUDY_SIZE})",
    )
    parser.add_argument(
        "--seed-fraction",
        type=float,
        default=SEED_FRACTION,
        help=f"Fraction of study used as seed (default: {SEED_FRACTION})",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated seeds (default: 42,123,456,789,2024)",
    )
    parser.add_argument(
        "--feature-type-override",
        type=str,
        default=None,
        help="Override auto-selected feature type for all studies",
    )
    parser.add_argument(
        "--studies-json",
        type=str,
        default=None,
        help="Path to corrected study definitions JSON (from data_integrity audit)",
    )
    parser.add_argument(
        "--batch-size-override",
        type=int,
        default=None,
        help="Override batch size for all studies",
    )
    parser.add_argument(
        "--seed-fraction-override",
        type=float,
        default=None,
        help="Override seed fraction for all studies",
    )
    parser.add_argument(
        "--n-rounds-override",
        type=int,
        default=None,
        help="Override number of rounds for all studies",
    )
    parser.add_argument(
        "--results-dir-override",
        type=str,
        default=None,
        help="Override results directory (default: benchmark_results/within_study)",
    )
    parser.add_argument("--resume", action="store_true", help="Skip existing results")
    parser.add_argument("--aggregate-only", action="store_true", help="Only aggregate")
    parser.add_argument("--dry-run", action="store_true", help="List runs without executing")
    args = parser.parse_args()

    seed_fraction = args.seed_fraction
    min_study_size = args.min_study_size

    # Override results directory if specified
    global RESULTS_DIR
    if args.results_dir_override:
        RESULTS_DIR = Path(args.results_dir_override).resolve()

    # Load full dataset
    from LNPBO.data.lnpdb_bridge import load_lnpdb_full

    print("Loading LNPDB...")
    dataset = load_lnpdb_full()
    df = dataset.df
    print(f"  {len(df):,} formulations from {df['Publication_PMID'].nunique()} studies")

    # Load or characterize studies
    if args.studies_json:
        import json as _json

        with open(args.studies_json) as _f:
            study_infos = _json.load(_f)
        print(f"\nLoaded {len(study_infos)} studies from {args.studies_json}")
    else:
        study_infos = characterize_studies(df, min_size=min_study_size, seed_fraction=seed_fraction)
        print(f"\n{len(study_infos)} qualifying studies (>= {args.min_study_size} formulations)")

    ensure_top_k_pct(study_infos)

    # Apply overrides
    if args.feature_type_override or args.batch_size_override or args.seed_fraction_override or args.n_rounds_override:
        for si in study_infos:
            if args.feature_type_override and (
                si.get("study_type") != "ratio_only" or args.feature_type_override == "ratios_only"
            ):
                si["feature_type"] = args.feature_type_override
            if args.batch_size_override:
                si["batch_size"] = args.batch_size_override
            if args.seed_fraction_override:
                n = si["n_formulations"]
                si["n_seed"] = max(MIN_SEED, int(args.seed_fraction_override * n))
            if args.n_rounds_override:
                si["n_rounds"] = args.n_rounds_override
            # Recompute rounds if batch size or seed changed
            if args.batch_size_override or args.seed_fraction_override:
                oracle_size = si["n_formulations"] - si["n_seed"]
                max_acq = int(0.5 * oracle_size)
                bs = si.get("batch_size", BATCH_SIZE)
                max_rounds = max(1, max_acq // bs)
                if not args.n_rounds_override:
                    si["n_rounds"] = min(MAX_ROUNDS, max_rounds)

    # Filter PMIDs if specified
    if args.pmids:
        target_pmids = set(p.strip() for p in args.pmids.split(","))
        target_pmids_float = set()
        for p in target_pmids:
            with contextlib.suppress(ValueError):
                target_pmids_float.add(float(p))
        study_infos = [
            si
            for si in study_infos
            if si.get("study_id", si["pmid"]) in target_pmids
            or si["pmid"] in target_pmids
            or float(si["pmid"]) in target_pmids_float
        ]
        print(f"  Filtered to {len(study_infos)} requested studies")

    for si in study_infos:
        sid = get_study_id(si)
        print(
            f"  {sid}: N={si['n_formulations']}, "
            f"ILs={si['n_unique_il']}, type={si['study_type']}, "
            f"n_seed={si['n_seed']}, rounds={si['n_rounds']}, "
            f"features={si['feature_type']}"
        )

    # Parse strategies
    seeds = SEEDS
    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(",")]

    if args.strategies:
        strategies = [s.strip() for s in args.strategies.split(",")]
    else:
        strategies = list(ALL_WITHIN_STUDY_STRATEGIES)

    # Check CASMOPOLITAN availability
    available_strategies = []
    for s in strategies:
        if s in ("casmopolitan_ucb", "casmopolitan_ei"):
            try:
                import botorch  # noqa: F401

                available_strategies.append(s)
            except ImportError:
                print(f"Skipping {s} (botorch not installed)")
        elif s == "discrete_tabpfn":
            try:
                import tabpfn  # noqa: F401

                available_strategies.append(s)
            except ImportError:
                print(f"Skipping {s} (tabpfn not installed)")
        else:
            available_strategies.append(s)
    strategies = available_strategies

    # Determine runs needed
    runs_needed = []
    for si in study_infos:
        pmid = si["pmid"]
        for strategy in strategies:
            for seed in seeds:
                if args.resume or args.aggregate_only:
                    existing = load_seed_result(pmid, strategy, seed, study_id=si.get("study_id"))
                    if existing is not None:
                        continue
                if args.aggregate_only:
                    continue
                runs_needed.append((si, strategy, seed))

    total_runs = len(runs_needed)

    print(f"\n{'=' * 70}")
    print("WITHIN-STUDY BENCHMARK")
    print(f"{'=' * 70}")
    print(f"Studies: {len(study_infos)}")
    print(f"Strategies: {len(strategies)}")
    print(f"Seeds: {seeds}")
    print(f"Total runs needed: {total_runs}")
    if args.resume:
        print("Mode: RESUME (skipping existing results)")
    if args.aggregate_only:
        print("Mode: AGGREGATE ONLY")
    print()

    if args.dry_run:
        print("DRY RUN -- would execute:")
        for si, strategy, seed in runs_needed:
            print(f"  PMID {int(float(si['pmid']))} / {strategy} / seed={seed}")
        print(f"\nTotal: {total_runs} runs")
        return

    # -----------------------------------------------------------------------
    # Run strategies
    # -----------------------------------------------------------------------

    if total_runs > 0:
        print(f"\nRunning {total_runs} study-strategy-seed combinations...\n")

        # Group by (study_id, seed) to share data loading
        runs_by_study_seed = {}
        for si, strategy, seed in runs_needed:
            key = (get_study_id(si), seed)
            runs_by_study_seed.setdefault(key, []).append((si, strategy))

        completed = 0
        for (_study_id, seed), run_list in runs_by_study_seed.items():
            si = run_list[0][0]
            pmid = si["pmid"]
            pmid_str = str(int(float(pmid)))

            print(f"\n{'=' * 50}")
            print(
                f"PMID {pmid_str} | seed={seed} | "
                f"N={si['n_formulations']} | n_seed={si['n_seed']} | "
                f"type={si['study_type']}"
            )
            print(f"{'=' * 50}")

            try:
                pca_data = prepare_study_data(df, si, seed)
            except Exception as e:
                print(f"  FAILED to prepare data: {e}")
                import traceback

                traceback.print_exc()
                continue

            # GP strategies need full 4-component encoding; load separately.
            # When a feature_type_override is active (encoding ablation), use the
            # overridden encoding for GP too so the ablation tests all strategies
            # with the specified encoding.
            gp_strats = [s for _, s in run_list if STRATEGY_CONFIGS[s]["type"] in ("gp", "gp_mixed")]
            gp_data = None
            if gp_strats:
                gp_feature_type = si["feature_type"] if args.feature_type_override else "lantern"
                gp_si = {**si, "feature_type": gp_feature_type}
                try:
                    gp_data = prepare_study_data(df, gp_si, seed)
                except Exception as e:
                    print(f"  GP data prep failed (expected for single-component studies): {e}")

            # Tanimoto strategies need raw count_mfp fingerprints (no PCA)
            tanimoto_strats = [s for _, s in run_list if s in TANIMOTO_STRATEGIES]
            tanimoto_data = None
            if tanimoto_strats:
                tanimoto_si = {**si, "feature_type": "count_mfp"}
                try:
                    tanimoto_data = prepare_study_data(df, tanimoto_si, seed, raw=True)
                except Exception as e:
                    print(f"  Tanimoto data prep failed: {e}")

            # Pre-compute kernel_kwargs for compositional and mixed strategies
            comp_kernel_kwargs = None
            comp_strats = [s for _, s in run_list if s in COMPOSITIONAL_STRATEGIES or s in MIXED_STRATEGIES]
            if comp_strats and gp_data is not None:
                _, _, gp_fcols, _, _, _ = gp_data
                comp_kernel_kwargs = classify_feature_columns(gp_fcols)

            for _, strategy in run_list:
                completed += 1
                print(f"\n[{completed}/{total_runs}] {strategy}")
                print("-" * 40)

                is_gp = STRATEGY_CONFIGS[strategy]["type"] in ("gp", "gp_mixed")
                is_tanimoto = strategy in TANIMOTO_STRATEGIES
                if is_tanimoto:
                    data_for_run = tanimoto_data
                elif is_gp:
                    data_for_run = gp_data
                else:
                    data_for_run = pca_data

                if data_for_run is None:
                    print("  SKIPPED: no compatible data for this strategy/study")
                    continue

                needs_kernel_kw = strategy in COMPOSITIONAL_STRATEGIES or strategy in MIXED_STRATEGIES
                kw = comp_kernel_kwargs if needs_kernel_kw else None
                try:
                    result = run_single_seed(strategy, seed, si, pca_data=data_for_run, kernel_kwargs=kw)

                    recall_str = ", ".join(
                        f"Top-{k}%={result['metrics']['top_k_recall'].get(str(k), 0):.1%}" for k in [5, 10, 20]
                    )
                    print(f"  Done in {result['elapsed']:.1f}s | {recall_str}")

                    save_seed_result(pmid, strategy, seed, result, si)

                except Exception as e:
                    print(f"  FAILED: {e}")
                    import traceback

                    traceback.print_exc()

    # -----------------------------------------------------------------------
    # Aggregation
    # -----------------------------------------------------------------------

    print(f"\n{'=' * 70}")
    print("AGGREGATION")
    print(f"{'=' * 70}\n")

    all_summaries = {}

    for si in study_infos:
        pmid = si["pmid"]
        sid = get_study_id(si)
        study_summaries = {}

        for strategy in strategies:
            seed_results = {}
            for seed in seeds:
                r = load_seed_result(pmid, strategy, seed, study_id=si.get("study_id"))
                if r is not None:
                    seed_results[seed] = r

            if not seed_results:
                continue

            summary = aggregate_study_strategy(sid, strategy, seed_results)
            if summary:
                study_summaries[strategy] = summary

        if study_summaries:
            all_summaries[sid] = study_summaries

    if not all_summaries:
        print("No results to aggregate.")
        return

    # Generate markdown
    md = generate_within_study_markdown(all_summaries, study_infos)

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = RESULTS_DIR / "within_study_summary.md"
    with open(summary_path, "w") as f:
        f.write(md)
    print(f"Summary saved to {summary_path}")

    # Save JSON
    json_path = RESULTS_DIR / "within_study_summary.json"
    with open(json_path, "w") as f:
        json.dump(
            {
                "study_infos": [{k: v for k, v in si.items()} for si in study_infos],
                "summaries": all_summaries,
                "timestamp": datetime.now().isoformat(),
            },
            f,
            indent=2,
            default=str,
        )
    print(f"JSON saved to {json_path}")

    # Print cross-study summary
    print("\n" + md.split("## Cross-Study Summary")[1] if "## Cross-Study Summary" in md else "")


if __name__ == "__main__":
    main()
