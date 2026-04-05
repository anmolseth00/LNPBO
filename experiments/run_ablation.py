#!/usr/bin/env python3
"""Unified ablation experiment runner.

Loads a JSON config (or takes CLI args) specifying the experiment design,
dispatches to the existing benchmark strategy runners, and saves per-seed
results with full config metadata for reproducibility.

Usage:
    # Run from config file:
    python -m experiments.run_ablation --config experiments/ablations/encoding/config.json

    # Run specific slice:
    python -m experiments.run_ablation --config .../config.json --studies 39060305 --seeds 42

    # Resume (skip existing):
    python -m experiments.run_ablation --config .../config.json --resume

    # Dry run:
    python -m experiments.run_ablation --config .../config.json --dry-run
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from benchmarks.benchmark import filter_study_df
from benchmarks.runner import (
    COMPOSITIONAL_STRATEGIES,
    STRATEGY_CONFIGS,
    _run_random,
    classify_feature_columns,
    compute_metrics,
    prepare_benchmark_data,
    select_warmup_seed,
)

RESULTS_BASE = REPO / "benchmark_results" / "ablations"


def load_config(config_path):
    with open(config_path) as f:
        return json.load(f)


def load_studies(studies_json_path):
    with open(studies_json_path) as f:
        return json.load(f)


def _result_path(results_dir, study_id, strategy, seed, config_label=None):
    label = f"_{config_label}" if config_label else ""
    return results_dir / study_id / f"{strategy}{label}_s{seed}.json"


def _filter_study_df(df, study):
    """Filter full LNPDB DataFrame to a single study.

    Thin wrapper around :func:`benchmarks.benchmark.filter_study_df`.
    The canonical version already adds ``Formulation_ID``; nothing extra
    is needed here.
    """
    return filter_study_df(df, study)


def prepare_study(
    df,
    study,
    feature_type,
    n_pcs,
    reduction,
    random_seed,
    seed_fraction=None,
    n_seed_override=None,
    batch_size=12,
    warmup_config=None,
    min_seed=None,
):
    """Prepare benchmark data for a single study with the given config.

    When ``min_seed`` is provided, it overrides the default floor of 30
    for the seed pool size.  This allows small-study experiments (N < 200)
    to run with fewer seed formulations.
    """
    study_df = _filter_study_df(df, study)
    n = len(study_df)

    seed_floor = min_seed if min_seed is not None else 30

    if n_seed_override:
        n_seed = n_seed_override
    elif seed_fraction:
        n_seed = max(seed_floor, int(seed_fraction * n))
    else:
        n_seed = study.get("n_seed", max(seed_floor, int(0.25 * n)))

    # Top-k percentiles
    top_k_pct = study.get("top_k_pct", {5: max(1, int(n * 0.05)), 10: max(1, int(n * 0.10)), 20: max(1, int(n * 0.20))})

    # Use ratios_only if study has only 1 IL and feature_type is structural
    actual_ft = feature_type
    if study_df["IL_SMILES"].nunique() <= 1 and feature_type != "ratios_only":
        actual_ft = "ratios_only"

    data = prepare_benchmark_data(
        n_seed=n_seed,
        random_seed=random_seed,
        reduction="none" if actual_ft == "ratios_only" else reduction,
        feature_type=actual_ft,
        n_pcs=n_pcs,
        data_df=study_df,
    )

    encoded, encoded_df, feature_cols, seed_idx, oracle_idx, _ = data

    # Apply warmup if configured
    if warmup_config:
        warmup_size = warmup_config["warmup_size"]
        selection = warmup_config.get("selection", "random")
        if warmup_size < len(encoded_df):
            seed_idx, oracle_idx = select_warmup_seed(
                encoded_df,
                warmup_size,
                selection,
                random_seed,
            )

    # Recompute top-k on encoded df
    top_k_values = {}
    for pct_str, k in top_k_pct.items():
        pct = int(pct_str) if isinstance(pct_str, str) else pct_str
        actual_k = min(k, len(encoded_df))
        top_k_values[pct] = set(encoded_df.nlargest(actual_k, "Experiment_value").index)

    return encoded, encoded_df, feature_cols, seed_idx, oracle_idx, top_k_values


def run_single(
    strategy,
    random_seed,
    encoded_dataset,
    encoded_df,
    feature_cols,
    seed_idx,
    oracle_idx,
    top_k_values,
    batch_size,
    n_rounds,
    normalize="copula",
    kappa=5.0,
    xi=0.01,
    kernel_kwargs=None,
):
    """Run a single strategy and return the result dict."""
    config = STRATEGY_CONFIGS[strategy]
    t0 = time.time()

    if config["type"] == "random":
        history = _run_random(encoded_df, seed_idx, oracle_idx, batch_size, n_rounds, random_seed)
    elif config["type"] == "discrete_online_conformal":
        from benchmarks.runner import run_discrete_online_conformal_strategy

        history = run_discrete_online_conformal_strategy(
            encoded_df,
            feature_cols,
            seed_idx,
            oracle_idx,
            batch_size=batch_size,
            n_rounds=n_rounds,
            seed=random_seed,
            kappa=kappa,
            normalize=normalize,
            encoded_dataset=encoded_dataset,
        )
    else:
        from benchmarks._optimizer_runner import OptimizerRunner
        from benchmarks.runner import strategy_to_optimizer_kwargs
        from LNPBO.optimization.optimizer import Optimizer

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
            encoded_df,
            feature_cols,
            seed_idx,
            oracle_idx,
            n_rounds=n_rounds,
            batch_size=batch_size,
            encoded_dataset=encoded_dataset,
            top_k_values=top_k_values,
        )

    elapsed = time.time() - t0
    metrics = compute_metrics(history, top_k_values, len(encoded_df))
    metrics["top_k_recall"] = {str(k): v for k, v in metrics["top_k_recall"].items()}

    return {
        "metrics": metrics,
        "elapsed": elapsed,
        "best_so_far": history["best_so_far"],
        "round_best": history["round_best"],
        "n_evaluated": history["n_evaluated"],
    }


def run_experiment(config, df, studies, args):
    """Run the full ablation experiment defined by config."""
    experiment_name = config["experiment_name"]
    strategies = config["strategies"]
    seeds = config.get("seeds", [42, 123, 456, 789, 2024])
    reduction = config.get("reduction", "pca")
    normalize = config.get("normalize", "copula")

    results_dir = RESULTS_BASE / experiment_name
    if args.results_dir:
        results_dir = Path(args.results_dir)

    # Filter studies
    if args.studies:
        target_ids = set(args.studies.split(","))
        studies = [s for s in studies if s.get("study_id", s["pmid"]) in target_ids]

    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",")]

    # Include all studies (previously excluded pooled-mixed studies)

    # Build run matrix from config "conditions"
    conditions = config.get("conditions", [{}])
    if args.condition:
        conditions = [c for c in conditions if c.get("label") == args.condition]
        if not conditions:
            print(f"No condition with label {args.condition!r} found in config")
            return

    runs = []
    for cond in conditions:
        cond_label = cond.get("label", "")
        feature_type = cond.get("feature_type", config.get("feature_type", "lantern_il_only"))
        batch_size = cond.get("batch_size", config.get("batch_size", 12))
        n_rounds = cond.get("n_rounds", config.get("n_rounds", 15))
        n_pcs = cond.get("n_pcs", config.get("n_pcs", 5))
        seed_fraction = cond.get("seed_fraction", config.get("seed_fraction"))
        warmup_config = cond.get("warmup")

        kappa = cond.get("kappa", config.get("kappa", 5.0))
        seed_mode = config.get("seed_mode", "fraction")
        max_oracle_frac = config.get("max_oracle_fraction", 0.5)
        min_seed_cfg = config.get("min_seed")

        for study in studies:
            sid = study.get("study_id", study["pmid"])
            n = study["n_formulations"]

            # Skip warmup configs where warmup >= 80% of study
            if warmup_config:
                ws = warmup_config.get("warmup_size", 0)
                if ws >= 0.8 * n:
                    continue

            # Dynamic sizing: compute n_rounds from study size and batch
            if seed_mode == "dynamic":
                sf = seed_fraction if seed_fraction else 0.25
                s_floor = min_seed_cfg if min_seed_cfg is not None else 5
                n_seed_dyn = max(s_floor, int(sf * n))
                oracle_size = n - n_seed_dyn
                max_acq = int(max_oracle_frac * oracle_size)
                study_n_rounds = max(1, max_acq // batch_size)
            else:
                study_n_rounds = n_rounds
                n_seed_dyn = None
                s_floor = None

            for strategy in strategies:
                for seed in seeds:
                    runs.append(
                        {
                            "study": study,
                            "study_id": sid,
                            "strategy": strategy,
                            "seed": seed,
                            "feature_type": feature_type,
                            "batch_size": batch_size,
                            "n_rounds": study_n_rounds,
                            "n_pcs": n_pcs,
                            "seed_fraction": seed_fraction,
                            "warmup_config": warmup_config,
                            "cond_label": cond_label,
                            "min_seed": s_floor,
                            "reduction": reduction,
                            "kappa": kappa,
                        }
                    )

    # Filter resumed
    if args.resume:
        new_runs = []
        for r in runs:
            path = _result_path(results_dir, r["study_id"], r["strategy"], r["seed"], r["cond_label"])
            if not path.exists():
                new_runs.append(r)
        skipped = len(runs) - len(new_runs)
        if skipped:
            print(f"Resuming: skipping {skipped} existing results")
        runs = new_runs

    print(f"\n{'=' * 70}")
    print(f"ABLATION: {experiment_name}")
    print(f"{'=' * 70}")
    print(f"Studies: {len(studies)}")
    print(f"Conditions: {len(conditions)}")
    print(f"Strategies: {strategies}")
    print(f"Seeds: {seeds}")
    print(f"Total runs: {len(runs)}")
    print()

    if args.dry_run:
        for r in runs[:20]:
            print(f"  {r['study_id']} / {r['strategy']} / {r['cond_label']} / s{r['seed']}")
        if len(runs) > 20:
            print(f"  ... and {len(runs) - 20} more")
        return

    # Group by (study_id, seed, feature_type, n_pcs, reduction, warmup, seed_fraction, min_seed)
    # to share data loading
    from collections import defaultdict

    groups = defaultdict(list)
    for r in runs:
        wk = json.dumps(r["warmup_config"], sort_keys=True) if r["warmup_config"] else ""
        key = (r["study_id"], r["seed"], r["feature_type"], r["n_pcs"], r["reduction"], r["seed_fraction"], wk, r.get("min_seed"))
        groups[key].append(r)

    completed = 0
    total = len(runs)
    for key, group_runs in groups.items():
        study_id, seed, feature_type, n_pcs, red, sf, wk, ms = key
        study = group_runs[0]["study"]
        warmup_config = group_runs[0]["warmup_config"]

        print(f"\n--- {study_id} / s{seed} / {feature_type} ---")

        try:
            enc, enc_df, fcols, s_idx, o_idx, topk = prepare_study(
                df,
                study,
                feature_type,
                n_pcs,
                red,
                seed,
                seed_fraction=sf,
                warmup_config=warmup_config,
                min_seed=ms,
            )
        except Exception as e:
            print(f"  Data prep failed: {e}")
            for _r in group_runs:
                completed += 1
            continue

        for r in group_runs:
            completed += 1
            strategy = r["strategy"]
            batch_size = r["batch_size"]
            n_rounds = r["n_rounds"]
            cond_label = r["cond_label"]

            # Cap rounds to available pool
            oracle_size = len(o_idx)
            max_acq = int(0.5 * oracle_size)
            feasible_rounds = max(1, max_acq // batch_size)
            actual_rounds = min(n_rounds, feasible_rounds)

            print(f"  [{completed}/{total}] {strategy} / {cond_label} / bs={batch_size} / r={actual_rounds}")

            run_kappa = r.get("kappa", 5.0)

            kw = classify_feature_columns(fcols) if strategy in COMPOSITIONAL_STRATEGIES else None

            try:
                result = run_single(
                    strategy,
                    seed,
                    enc,
                    enc_df,
                    fcols,
                    s_idx,
                    o_idx,
                    topk,
                    batch_size=batch_size,
                    n_rounds=actual_rounds,
                    normalize=normalize,
                    kappa=run_kappa,
                    kernel_kwargs=kw,
                )

                recall_str = ", ".join(
                    f"Top-{k}%={result['metrics']['top_k_recall'].get(str(k), 0):.1%}" for k in [5, 10, 20]
                )
                print(f"    {result['elapsed']:.1f}s | {recall_str}")

                # Save result
                path = _result_path(results_dir, study_id, strategy, seed, cond_label)
                path.parent.mkdir(parents=True, exist_ok=True)
                out = {
                    "experiment": experiment_name,
                    "study_id": study_id,
                    "pmid": study["pmid"],
                    "strategy": strategy,
                    "seed": seed,
                    "condition": {
                        "label": cond_label,
                        "feature_type": feature_type,
                        "batch_size": batch_size,
                        "n_rounds": actual_rounds,
                        "n_pcs": n_pcs,
                        "reduction": red,
                        "seed_fraction": sf,
                        "warmup": warmup_config,
                        "kappa": run_kappa,
                    },
                    "study_info": {k: v for k, v in study.items() if k not in ("lnp_ids", "top_k_pct")},
                    "result": result,
                    "timestamp": datetime.now().isoformat(),
                }
                with open(path, "w") as f:
                    json.dump(out, f, indent=2, default=str)

            except Exception as e:
                print(f"    FAILED: {e}")
                import traceback

                traceback.print_exc()

    print(f"\nDone. {completed} runs completed. Results in {results_dir}")


def main():
    parser = argparse.ArgumentParser(description="Unified ablation experiment runner")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config JSON")
    parser.add_argument(
        "--studies-json", type=str, default=None, help="Path to corrected studies JSON (default: data_integrity output)"
    )
    parser.add_argument("--studies", type=str, default=None, help="Comma-separated study IDs to run (default: all)")
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated random seeds (overrides config)")
    parser.add_argument("--results-dir", type=str, default=None, help="Override results directory")
    parser.add_argument(
        "--condition",
        type=str,
        default=None,
        help="Run only this condition label (for parallel launches)",
    )
    parser.add_argument("--resume", action="store_true", help="Skip existing results")
    parser.add_argument("--dry-run", action="store_true", help="List runs without executing")
    args = parser.parse_args()

    config = load_config(args.config)

    # Load studies
    studies_path = args.studies_json
    if not studies_path:
        default = REPO / "experiments" / "data_integrity" / "studies_with_ids.json"
        if default.exists():
            studies_path = str(default)
        else:
            print("No studies JSON found. Run data_integrity audit first or provide --studies-json")
            sys.exit(1)

    studies = load_studies(studies_path)

    # Load full dataset
    from LNPBO.data.lnpdb_bridge import load_lnpdb_full

    print("Loading LNPDB...")
    dataset = load_lnpdb_full()
    df = dataset.df
    print(f"  {len(df):,} formulations")

    run_experiment(config, df, studies, args)


if __name__ == "__main__":
    main()
