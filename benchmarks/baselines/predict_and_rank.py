#!/usr/bin/env python3
"""Predict-and-rank baseline: single-shot surrogate selection (no BO loop).

For each study, trains a surrogate on the 25% seed data, predicts scores
for the oracle pool, and greedily selects the top-K candidates. This is
the one-shot alternative to iterative BO.

Usage:
    python -m benchmarks.baselines.predict_and_rank
    python -m benchmarks.baselines.predict_and_rank --pmids 37990414
    python -m benchmarks.baselines.predict_and_rank --surrogates xgb,rf
    python -m benchmarks.baselines.predict_and_rank --resume
    python -m benchmarks.baselines.predict_and_rank --aggregate-only
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from LNPBO.benchmarks.benchmark import (
    BATCH_SIZE,
    MAX_ROUNDS,
    MIN_SEED,
    SEEDS,
    characterize_studies,
    ensure_top_k_pct,
    get_study_id,
    prepare_study_data,
)
from LNPBO.benchmarks.runner import compute_metrics, init_history, update_history
from LNPBO.benchmarks.stats import bootstrap_ci

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "benchmark_results" / "baselines" / "predict_and_rank"

SURROGATES = {
    "xgb": "XGBoost",
    "rf": "Random Forest",
    "ngboost": "NGBoost",
}


def _fit_predict(X_train, y_train, X_pool, surrogate, seed):
    """Fit surrogate on training data and predict scores for pool.

    Applies MinMaxScaler to features and copula normalization to targets
    before fitting, matching the preprocessing used by the iterative BO
    benchmark.
    """
    from sklearn.preprocessing import MinMaxScaler

    from LNPBO.optimization._normalize import copula_transform

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_pool = scaler.transform(X_pool)

    y_train = copula_transform(y_train)

    if surrogate == "xgb":
        from xgboost import XGBRegressor

        model = XGBRegressor(
            n_estimators=200,
            random_state=seed,
            n_jobs=-1,
            verbosity=0,
        )
        model.fit(X_train, y_train)
        scores = model.predict(X_pool)

    elif surrogate == "rf":
        from sklearn.ensemble import RandomForestRegressor

        model = RandomForestRegressor(
            n_estimators=200,
            random_state=seed,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        scores = model.predict(X_pool)

    elif surrogate == "ngboost":
        from ngboost import NGBRegressor

        model = NGBRegressor(
            n_estimators=200,
            random_state=seed,
            verbose=False,
        )
        model.fit(X_train, y_train)
        scores = model.predict(X_pool)

    else:
        raise ValueError(f"Unknown surrogate: {surrogate!r}")

    return scores


def run_predict_and_rank(
    df,
    feature_cols,
    seed_idx,
    oracle_idx,
    top_k_values,
    surrogate,
    batch_size,
    n_rounds,
    random_seed,
):
    """Single-shot predict-and-rank: train on seed, greedily select top-K."""
    X_train = df.loc[seed_idx, feature_cols].values
    y_train = df.loc[seed_idx, "Experiment_value"].values
    X_pool = df.loc[oracle_idx, feature_cols].values

    scores = _fit_predict(X_train, y_train, X_pool, surrogate, random_seed)

    # Total budget: same as BO (n_rounds * batch_size candidates from pool)
    total_budget = n_rounds * batch_size

    # Rank pool by predicted score, select top-K greedily
    ranked_pool_positions = np.argsort(-scores)
    n_select = min(total_budget, len(oracle_idx))
    selected_positions = ranked_pool_positions[:n_select]

    # Simulate round-by-round selection (for convergence tracking)
    training_idx = list(seed_idx)
    history = init_history(df, training_idx, top_k_values=top_k_values)

    for r in range(n_rounds):
        start = r * batch_size
        end = min((r + 1) * batch_size, n_select)
        if start >= n_select:
            break

        batch_positions = selected_positions[start:end]
        batch_idx = [oracle_idx[i] for i in batch_positions]
        training_idx.extend(batch_idx)
        update_history(history, df, training_idx, batch_idx, r, top_k_values=top_k_values)

    return history


def _per_seed_path(study_id, surrogate, seed):
    return RESULTS_DIR / study_id / f"predict_rank_{surrogate}_s{seed}.json"


def run_pr_cli(
    *,
    results_dir,
    surrogates_map,
    default_surrogates,
    banner,
    baseline_name,
    file_prefix,
    display_prefix="",
    feature_type_override=None,
    extra_json_fields=None,
):
    """Shared CLI runner for predict-and-rank variants.

    Parameters
    ----------
    results_dir : Path
        Where to write per-seed JSON results.
    surrogates_map : dict
        Mapping surrogate key -> display name (e.g. {"xgb": "XGBoost"}).
    default_surrogates : str
        Comma-separated default surrogates for --surrogates arg.
    banner : str
        Title for the banner printed at startup.
    baseline_name : str
        Value of "baseline" key in output JSON.
    file_prefix : str
        Prefix for per-seed filenames (e.g. "predict_rank" or "agile").
    display_prefix : str
        Prefix for aggregation display name (e.g. "AGILE+").
    feature_type_override : str or None
        If set, override study_info's feature_type for data preparation.
    extra_json_fields : dict or None
        Extra fields to include in output JSON.
    """
    parser = argparse.ArgumentParser(description=banner)
    parser.add_argument(
        "--surrogates",
        type=str,
        default=default_surrogates,
        help=f"Comma-separated surrogates (default: {default_surrogates})",
    )
    parser.add_argument("--pmids", type=str, default=None)
    parser.add_argument("--seeds", type=str, default=None)
    parser.add_argument("--studies-json", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--seed-fraction",
        type=float,
        default=None,
        help="Override seed fraction (default: use study_info n_seed as-is, i.e. 0.25)",
    )
    args = parser.parse_args()

    surrogates = [s.strip() for s in args.surrogates.split(",")]
    seeds = SEEDS
    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",")]

    from LNPBO.data.lnpdb_bridge import load_lnpdb_full

    print("Loading LNPDB...")
    dataset = load_lnpdb_full()
    df = dataset.df
    print(f"  {len(df):,} formulations")

    if args.studies_json:
        with open(args.studies_json) as f:
            study_infos = json.load(f)
    else:
        default_studies = (
            Path(__file__).resolve().parent.parent.parent / "experiments" / "data_integrity" / "studies.json"
        )
        if default_studies.exists():
            with open(default_studies) as f:
                study_infos = json.load(f)
        else:
            study_infos = characterize_studies(df)

    if args.pmids:
        target = set(args.pmids.split(","))
        study_infos = [
            si for si in study_infos if si.get("study_id", str(si["pmid"])) in target or str(si["pmid"]) in target
        ]

    ensure_top_k_pct(study_infos)

    # --- seed-fraction override ---
    if args.seed_fraction is not None:
        frac = args.seed_fraction
        pct_label = f"budget_{int(round(frac * 100))}pct"
        results_dir = results_dir / pct_label
        for si in study_infos:
            n = si["n_formulations"]
            si["n_seed"] = max(MIN_SEED, int(frac * n))
            oracle_size = n - si["n_seed"]
            max_acq = int(0.5 * oracle_size)
            bs = si.get("batch_size", BATCH_SIZE)
            max_rounds = max(1, max_acq // bs)
            si["n_rounds"] = min(MAX_ROUNDS, max_rounds)

    def _seed_path(study_id, surrogate, seed):
        return results_dir / study_id / f"{file_prefix}_{surrogate}_s{seed}.json"

    runs = []
    for si in study_infos:
        for surrogate in surrogates:
            for seed in seeds:
                sid = get_study_id(si)
                if (args.resume or args.aggregate_only) and _seed_path(sid, surrogate, seed).exists():
                    continue
                if args.aggregate_only:
                    continue
                runs.append((si, surrogate, seed))

    print(f"\n{'=' * 70}")
    print(banner)
    print(f"{'=' * 70}")
    print(f"Studies: {len(study_infos)}")
    print(f"Surrogates: {surrogates}")
    print(f"Seeds: {seeds}")
    if args.seed_fraction is not None:
        print(f"Seed fraction: {args.seed_fraction}")
    print(f"Results dir: {results_dir}")
    print(f"Runs: {len(runs)}")

    if args.dry_run:
        for si, surr, seed in runs[:20]:
            sid = get_study_id(si)
            print(f"  {sid} / {surr} / s{seed}")
        if len(runs) > 20:
            print(f"  ... and {len(runs) - 20} more")
        return

    from collections import defaultdict

    groups = defaultdict(list)
    for si, surr, seed in runs:
        key = (get_study_id(si), seed)
        groups[key].append((si, surr))

    completed = 0
    total = len(runs)

    for (sid, seed), group in groups.items():
        si = group[0][0]
        print(f"\n--- {sid} / seed={seed} ---")

        data_si = {**si, "feature_type": feature_type_override} if feature_type_override else si

        try:
            _enc, enc_df, fcols, s_idx, o_idx, topk = prepare_study_data(
                df,
                data_si,
                seed,
            )
        except Exception as e:
            print(f"  Data prep failed: {e}")
            completed += len(group)
            continue

        batch_size = si.get("batch_size", BATCH_SIZE)
        n_rounds = si.get("n_rounds", MAX_ROUNDS)

        for _, surr in group:
            completed += 1
            print(f"  [{completed}/{total}] {file_prefix}_{surr}")

            try:
                t0 = time.time()
                history = run_predict_and_rank(
                    enc_df,
                    fcols,
                    s_idx,
                    o_idx,
                    topk,
                    surrogate=surr,
                    batch_size=batch_size,
                    n_rounds=n_rounds,
                    random_seed=seed,
                )
                elapsed = time.time() - t0
                metrics = compute_metrics(history, topk, len(enc_df))
                metrics["top_k_recall"] = {str(k): v for k, v in metrics["top_k_recall"].items()}

                recall_str = ", ".join(f"Top-{k}%={metrics['top_k_recall'].get(str(k), 0):.1%}" for k in [5, 10, 20])
                print(f"    {elapsed:.1f}s | {recall_str}")

                result = {
                    "metrics": metrics,
                    "elapsed": elapsed,
                    "best_so_far": history["best_so_far"],
                    "round_best": history["round_best"],
                    "n_evaluated": history["n_evaluated"],
                }

                path = _seed_path(sid, surr, seed)
                path.parent.mkdir(parents=True, exist_ok=True)
                out = {
                    "baseline": baseline_name,
                    "surrogate": surr,
                    "study_id": sid,
                    "pmid": si["pmid"],
                    "seed": seed,
                    "study_info": {k: v for k, v in si.items() if k not in ("lnp_ids", "top_k_pct")},
                    "result": result,
                    "timestamp": datetime.now().isoformat(),
                }
                if extra_json_fields:
                    out.update(extra_json_fields)
                with open(path, "w") as f:
                    json.dump(out, f, indent=2, default=str)

            except Exception as e:
                print(f"    FAILED: {e}")
                import traceback

                traceback.print_exc()

    # Aggregation
    print(f"\n{'=' * 70}")
    print("AGGREGATION")
    print(f"{'=' * 70}")

    if not results_dir.exists():
        print("No results to aggregate.")
        return

    all_summaries = {}
    for si in study_infos:
        sid = get_study_id(si)
        study_dir = results_dir / sid
        if not study_dir.exists():
            continue

        for surr in surrogates:
            key = f"{file_prefix}_{surr}"
            vals = []
            for seed in seeds:
                path = _seed_path(sid, surr, seed)
                if path.exists():
                    with open(path) as f:
                        data = json.load(f)
                    recall = data["result"]["metrics"]["top_k_recall"]
                    vals.append(recall.get("5", 0.0))

            if vals:
                m = float(np.mean(vals))
                ci = bootstrap_ci(vals) if len(vals) >= 3 else (m, m)
                if key not in all_summaries:
                    all_summaries[key] = {"display": f"{display_prefix}{surrogates_map[surr]} P&R", "studies": []}
                all_summaries[key]["studies"].append(
                    {
                        "study_id": sid,
                        "mean_top5": m,
                        "ci": ci,
                        "n_seeds": len(vals),
                    }
                )

    if all_summaries:
        print(f"\n{'Baseline':<30} {'Mean Top-5%':>12} {'95% CI':>18} {'N studies':>10}")
        print("-" * 75)
        for _key, data in sorted(all_summaries.items()):
            study_means = [s["mean_top5"] for s in data["studies"]]
            grand = float(np.mean(study_means))
            ci = bootstrap_ci(study_means) if len(study_means) >= 3 else (grand, grand)
            print(f"{data['display']:<30} {grand:>11.1%} [{ci[0]:.1%}, {ci[1]:.1%}] {len(data['studies']):>10}")

        summary_path = results_dir / f"{baseline_name.replace(' ', '_')}_summary.json"
        with open(summary_path, "w") as f:
            json.dump(all_summaries, f, indent=2, default=str)
        print(f"\nSummary saved to {summary_path}")


def main():
    run_pr_cli(
        results_dir=RESULTS_DIR,
        surrogates_map=SURROGATES,
        default_surrogates="xgb,rf,ngboost",
        banner="PREDICT-AND-RANK BASELINE",
        baseline_name="predict_and_rank",
        file_prefix="predict_rank",
    )


if __name__ == "__main__":
    main()
