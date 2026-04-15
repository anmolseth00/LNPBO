#!/usr/bin/env python3
"""Backfill per-round top-5% recall into existing benchmark result JSONs.

Re-runs each (study, strategy, seed) with identical parameters to compute
per_round_recall, then merges the new field into the existing JSON without
overwriting other data. The BO loop is re-executed deterministically (same
random seeds), so selections are identical.

Usage:
    # Backfill all strategies (full re-run):
    python -m benchmarks.backfill_per_round_recall

    # Backfill only convergence figure families (faster):
    python -m benchmarks.backfill_per_round_recall --convergence-only

    # Dry run:
    python -m benchmarks.backfill_per_round_recall --convergence-only --dry-run

    # Specific studies:
    python -m benchmarks.backfill_per_round_recall --pmids 39060305,37985700
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from LNPBO.runtime_paths import benchmark_results_root, package_root_from, resolve_input_path

from .benchmark import (
    characterize_studies,
    ensure_top_k_pct,
    get_study_id,
    prepare_study_data,
)
from .runner import (
    COMPOSITIONAL_STRATEGIES,
    STRATEGY_CONFIGS,
    classify_feature_columns,
)

_PACKAGE_ROOT = package_root_from(__file__, levels_up=2)
RESULTS_DIR = benchmark_results_root(_PACKAGE_ROOT) / "within_study"
STUDIES_JSON = resolve_input_path(_PACKAGE_ROOT, "experiments/data_integrity/studies.json")
from .constants import SEEDS

# Convergence figure families: one representative strategy per family
CONVERGENCE_STRATEGIES = [
    "random",
    "discrete_rf_ts_batch",
    "discrete_ngboost_ucb",
    "lnpbo_ts_batch",
]


def load_lnpdb():
    from LNPBO.data.lnpdb_bridge import load_lnpdb_full

    dataset = load_lnpdb_full()
    return dataset.df


def backfill_study(df, study_info, strategies, seeds, dry_run=False):
    """Re-run strategies for one study and merge per_round_recall into JSONs."""
    sid = get_study_id(study_info)
    study_dir = RESULTS_DIR / str(sid)
    if not study_dir.exists():
        print(f"  Skipping {sid}: no results directory")
        return 0

    batch_size = study_info.get("batch_size", 12)
    n_rounds = study_info.get("n_rounds", 15)

    count = 0
    for seed in seeds:
        # Check which strategies need backfill for this seed
        need_backfill = []
        for strategy in strategies:
            fname = f"{strategy}_s{seed}.json"
            fpath = study_dir / fname
            if not fpath.exists():
                continue
            existing = json.loads(fpath.read_text())
            prr = existing.get("result", {}).get("metrics", {}).get("per_round_recall", {})
            if prr and "5" in prr:
                continue  # Already has per_round_recall with correct percentile keys
            need_backfill.append((strategy, fpath, existing))

        if not need_backfill:
            continue

        # Prepare data for this seed (same as benchmark.py main loop)
        try:
            data = prepare_study_data(df, study_info, seed)
        except Exception as e:
            print(f"  {sid}/s{seed}: data prep failed: {e}")
            continue

        s_dataset, s_df, s_fcols, s_seed_idx, s_oracle_idx, s_topk = data

        # Compute kernel_kwargs for compositional strategies
        comp_kw = None
        if any(s in COMPOSITIONAL_STRATEGIES for s, _, _ in need_backfill):
            comp_kw = classify_feature_columns(s_fcols)

        for strategy, fpath, existing in need_backfill:
            if dry_run:
                print(f"  [DRY] {sid}/{strategy}_s{seed}")
                count += 1
                continue

            config = STRATEGY_CONFIGS[strategy]
            kappa = 5.0
            xi = 0.01
            normalize = "copula"

            try:
                if config["type"] == "random":
                    from benchmarks.runner import _run_random

                    history = _run_random(
                        s_df,
                        s_seed_idx,
                        s_oracle_idx,
                        batch_size,
                        n_rounds,
                        seed,
                        top_k_values=s_topk,
                    )
                elif config["type"] == "discrete_online_conformal_exact":
                    from benchmarks.runner import run_discrete_online_conformal_strategy

                    history = run_discrete_online_conformal_strategy(
                        s_df,
                        s_fcols,
                        s_seed_idx,
                        s_oracle_idx,
                        batch_size=batch_size,
                        n_rounds=n_rounds,
                        seed=seed,
                        kappa=kappa,
                        normalize=normalize,
                        encoded_dataset=s_dataset,
                        top_k_values=s_topk,
                    )
                elif config["type"] == "discrete_online_conformal_baseline":
                    from benchmarks.runner import run_discrete_cumulative_split_conformal_ucb_baseline

                    history = run_discrete_cumulative_split_conformal_ucb_baseline(
                        s_df,
                        s_fcols,
                        s_seed_idx,
                        s_oracle_idx,
                        batch_size=batch_size,
                        n_rounds=n_rounds,
                        seed=seed,
                        kappa=kappa,
                        normalize=normalize,
                        encoded_dataset=s_dataset,
                        top_k_values=s_topk,
                    )
                else:
                    from benchmarks._optimizer_runner import OptimizerRunner
                    from benchmarks.runner import strategy_to_optimizer_kwargs
                    from LNPBO.optimization.optimizer import Optimizer

                    kw = comp_kw if strategy in COMPOSITIONAL_STRATEGIES else None
                    opt_kwargs = strategy_to_optimizer_kwargs(strategy, kernel_kwargs=kw)
                    optimizer = Optimizer(
                        random_seed=seed,
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
                        s_seed_idx,
                        s_oracle_idx,
                        n_rounds=n_rounds,
                        batch_size=batch_size,
                        encoded_dataset=s_dataset,
                        top_k_values=s_topk,
                    )

                # Verify determinism: best_so_far should match
                old_bsf = existing["result"].get("best_so_far", [])
                new_bsf = history["best_so_far"]
                if old_bsf and len(old_bsf) == len(new_bsf) and not np.allclose(old_bsf, new_bsf, atol=1e-6):
                    print(
                        f"  WARNING: {sid}/{strategy}_s{seed}: "
                        f"best_so_far mismatch (max diff={max(abs(a - b) for a, b in zip(old_bsf, new_bsf)):.6f})"
                    )

                # Extract per_round_recall from history and merge into existing JSON
                if "per_round_recall" in history:
                    prr = {str(k): v for k, v in history["per_round_recall"].items()}
                    existing["result"]["metrics"]["per_round_recall"] = prr
                    fpath.write_text(json.dumps(existing, indent=2))
                    count += 1
                    # Print current round-by-round recall for top-5%
                    r5 = prr.get("5", [])
                    if r5:
                        print(f"  {sid}/{strategy}_s{seed}: recall@5=[{r5[0]:.2f}→{r5[-1]:.2f}]")

            except Exception as e:
                print(f"  {sid}/{strategy}_s{seed}: FAILED: {e}")
                import traceback

                traceback.print_exc()

    return count


def main():
    parser = argparse.ArgumentParser(description="Backfill per-round recall")
    parser.add_argument(
        "--convergence-only", action="store_true", help="Only backfill strategies needed for convergence figure"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--pmids", type=str, default=None, help="Comma-separated PMIDs to process")
    args = parser.parse_args()

    strategies = CONVERGENCE_STRATEGIES if args.convergence_only else list(STRATEGY_CONFIGS.keys())

    print("Loading LNPDB...")
    df = load_lnpdb()

    # Use studies.json for correct study definitions (includes per-target splits)
    if STUDIES_JSON.exists():
        study_infos = json.loads(STUDIES_JSON.read_text())
        print(f"Loaded {len(study_infos)} studies from studies.json")
    else:
        study_infos = characterize_studies(df, min_size=200, seed_fraction=0.25)
        print(f"{len(study_infos)} qualifying studies (from characterize_studies)")
    ensure_top_k_pct(study_infos)

    if args.pmids:
        target = set(args.pmids.split(","))
        study_infos = [
            si
            for si in study_infos
            if str(si.get("study_id", si["pmid"])) in target
            or str(si["pmid"]) in target
            or str(int(float(si["pmid"]))) in target
        ]
        print(f"Filtered to {len(study_infos)} studies")

    total = 0
    t0 = time.time()
    for si in study_infos:
        sid = get_study_id(si)
        print(f"\n=== {sid} (N={si['n_formulations']}) ===")
        n = backfill_study(df, si, strategies, SEEDS, dry_run=args.dry_run)
        total += n

    elapsed = time.time() - t0
    print(f"\nDone. Backfilled {total} results in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
