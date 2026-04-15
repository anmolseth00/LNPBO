"""Cross-study transfer experiment for LNPBO.

Tests whether a surrogate model trained on N-1 studies can identify top
formulations in a held-out study. This is a formal leave-one-out
evaluation of cross-study generalization.

Two variants:
  - Cold start: train on N-1 studies, rank all held-out formulations (zero shot)
  - Warm start: train on N-1 studies + 25% random seed from held-out,
                 rank the remaining 75%

The expected result is a negative one: cross-study transfer performs near
random because SAR is study-dependent (different assays, cell types, readouts).
This complements the within-study benchmark (57% study-level variance in ICC).

Usage:
    python -m experiments.cross_study_transfer
    python -m experiments.cross_study_transfer --dry-run
    python -m experiments.cross_study_transfer --study 39060305
    python -m experiments.cross_study_transfer --variant cold
    python -m experiments.cross_study_transfer --n-seeds 5
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from LNPBO.benchmarks.benchmark import filter_study_df
from LNPBO.data.compute_pcs import compute_pcs
from LNPBO.data.lnpdb_bridge import load_lnpdb_full
from LNPBO.runtime_paths import benchmark_results_root, package_root_from

_PACKAGE_ROOT = package_root_from(__file__, levels_up=2)
_EXPERIMENTS_DIR = Path(__file__).resolve().parent
_STUDIES_JSON = _EXPERIMENTS_DIR / "data_integrity" / "studies_with_ids.json"
_RESULTS_DIR = benchmark_results_root(_PACKAGE_ROOT) / "cross_study_transfer"

SEED_FRACTION = 0.25
N_SEEDS = 5
RANDOM_SEEDS = [42, 123, 456, 789, 2024]


# ---------------------------------------------------------------------------
# LANTERN encoding with separate PCA fitting
# ---------------------------------------------------------------------------


def encode_lantern(
    train_smiles,
    test_smiles,
    train_y,
    n_pcs_count_mfp=5,
    n_pcs_rdkit=5,
):
    """Encode SMILES with LANTERN features (count MFP + RDKit, PCA-reduced).

    PCA is fitted on train_smiles only, then applied to both sets.
    This prevents information leakage from the held-out study.
    """
    train_blocks, test_blocks = [], []

    if n_pcs_count_mfp > 0:
        pcs_tr, reducer, scaler, _ = compute_pcs(
            train_smiles,
            feature_type="count_mfp",
            experiment_values=train_y.tolist(),
            n_components=n_pcs_count_mfp,
            reduction="pca",
        )
        pcs_te, _, _, _ = compute_pcs(
            test_smiles,
            feature_type="count_mfp",
            n_components=n_pcs_count_mfp,
            reduction="pca",
            fitted_reducer=reducer,
            fitted_scaler=scaler,
        )
        train_blocks.append(pcs_tr)
        test_blocks.append(pcs_te)

    if n_pcs_rdkit > 0:
        pcs_tr, reducer, scaler, _ = compute_pcs(
            train_smiles,
            feature_type="rdkit",
            experiment_values=train_y.tolist(),
            n_components=n_pcs_rdkit,
            reduction="pca",
        )
        pcs_te, _, _, _ = compute_pcs(
            test_smiles,
            feature_type="rdkit",
            n_components=n_pcs_rdkit,
            reduction="pca",
            fitted_reducer=reducer,
            fitted_scaler=scaler,
        )
        train_blocks.append(pcs_tr)
        test_blocks.append(pcs_te)

    if not train_blocks:
        return np.empty((len(train_smiles), 0)), np.empty((len(test_smiles), 0))

    return np.hstack(train_blocks), np.hstack(test_blocks)


_RATIO_COLS = [
    "IL_molratio",
    "HL_molratio",
    "CHL_molratio",
    "PEG_molratio",
    "IL_to_nucleicacid_massratio",
]


def _get_variable_ratio_cols(df):
    """Return ratio column names that vary within df."""
    return [col for col in _RATIO_COLS if col in df.columns and df[col].nunique() > 1]


def _build_ratio_features(df, ratio_cols):
    """Extract specified molar ratio columns, filling missing with 0."""
    if not ratio_cols:
        return np.empty((len(df), 0))
    out = np.zeros((len(df), len(ratio_cols)), dtype=float)
    for i, col in enumerate(ratio_cols):
        if col in df.columns:
            out[:, i] = df[col].values.astype(float)
    return out


# ---------------------------------------------------------------------------
# XGBoost training + prediction
# ---------------------------------------------------------------------------


def _train_predict(X_train, y_train, X_test, seed=42):
    model = XGBRegressor(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.5,
        random_state=seed,
        n_jobs=-1,
        verbosity=0,
    )
    # Scale features for numeric stability
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)
    model.fit(X_tr, y_train)
    return model.predict(X_te)


# ---------------------------------------------------------------------------
# Recall computation
# ---------------------------------------------------------------------------


def compute_recall(true_values, predicted_values, percentile):
    """Compute top-percentile recall.

    Given true and predicted values for a pool, compute the fraction of
    truly top-percentile formulations that appear in the model's
    top-percentile predictions.

    Parameters
    ----------
    true_values : array-like
        True z-scored Experiment_value for each formulation.
    predicted_values : array-like
        Model predictions for each formulation.
    percentile : float
        Percentile threshold (e.g. 5 for top-5%).

    Returns
    -------
    float
        Recall in [0, 1].
    """
    n = len(true_values)
    k = max(1, int(n * percentile / 100))

    true_top_idx = set(np.argsort(true_values)[-k:])
    pred_top_idx = set(np.argsort(predicted_values)[-k:])

    overlap = len(true_top_idx & pred_top_idx)
    return overlap / len(true_top_idx)


# ---------------------------------------------------------------------------
# Cold start: train on N-1 studies, predict all of held-out
# ---------------------------------------------------------------------------


def run_cold_start(train_df, test_df, study_info, seed=42):
    """Zero-shot cross-study transfer.

    Train XGBoost on all N-1 training studies, predict on all formulations
    in the held-out study, measure recall.
    """
    is_ratio_only = study_info.get("feature_type") == "ratios_only"

    train_y = train_df["Experiment_value"].values.astype(float)
    test_y = test_df["Experiment_value"].values.astype(float)

    # Determine ratio columns from training data (consistent for train + test)
    ratio_cols = _get_variable_ratio_cols(train_df)

    if is_ratio_only:
        X_train = _build_ratio_features(train_df, ratio_cols)
        X_test = _build_ratio_features(test_df, ratio_cols)
    else:
        train_smiles = train_df["IL_SMILES"].tolist()
        test_smiles = test_df["IL_SMILES"].tolist()

        X_train_mol, X_test_mol = encode_lantern(
            train_smiles,
            test_smiles,
            train_y,
            n_pcs_count_mfp=5,
            n_pcs_rdkit=5,
        )

        # Append ratio features
        train_ratios = _build_ratio_features(train_df, ratio_cols)
        test_ratios = _build_ratio_features(test_df, ratio_cols)

        if train_ratios.shape[1] > 0:
            X_train = np.hstack([X_train_mol, train_ratios])
            X_test = np.hstack([X_test_mol, test_ratios])
        else:
            X_train = X_train_mol
            X_test = X_test_mol

    if X_train.shape[1] == 0:
        return None

    preds = _train_predict(X_train, train_y, X_test, seed=seed)

    recall_5 = compute_recall(test_y, preds, 5)
    recall_10 = compute_recall(test_y, preds, 10)

    return {
        "recall_5": recall_5,
        "recall_10": recall_10,
        "n_train": len(train_df),
        "n_test": len(test_df),
        "n_features": X_train.shape[1],
    }


# ---------------------------------------------------------------------------
# Warm start: train on N-1 + 25% seed from held-out, predict remaining 75%
# ---------------------------------------------------------------------------


def run_warm_start(train_df, test_df, study_info, seed=42, seed_fraction=SEED_FRACTION):
    """Warm-start cross-study transfer.

    Train XGBoost on all N-1 training studies + a random 25% seed from the
    held-out study. Predict on the remaining 75% of the held-out study.
    """
    is_ratio_only = study_info.get("feature_type") == "ratios_only"

    rng = np.random.RandomState(seed)
    n_test = len(test_df)
    n_seed = max(1, int(seed_fraction * n_test))
    all_idx = np.arange(n_test)
    rng.shuffle(all_idx)

    seed_idx = sorted(all_idx[:n_seed])
    eval_idx = sorted(all_idx[n_seed:])

    if len(eval_idx) == 0:
        return None

    # Combine training studies + seed from held-out
    seed_df = test_df.iloc[seed_idx]
    eval_df = test_df.iloc[eval_idx]
    combined_train_df = pd.concat([train_df, seed_df], ignore_index=True)

    train_y = combined_train_df["Experiment_value"].values.astype(float)
    eval_y = eval_df["Experiment_value"].values.astype(float)

    # Determine ratio columns from combined training data
    ratio_cols = _get_variable_ratio_cols(combined_train_df)

    if is_ratio_only:
        X_train = _build_ratio_features(combined_train_df, ratio_cols)
        X_eval = _build_ratio_features(eval_df, ratio_cols)
    else:
        train_smiles = combined_train_df["IL_SMILES"].tolist()
        eval_smiles = eval_df["IL_SMILES"].tolist()

        X_train_mol, X_eval_mol = encode_lantern(
            train_smiles,
            eval_smiles,
            train_y,
            n_pcs_count_mfp=5,
            n_pcs_rdkit=5,
        )

        train_ratios = _build_ratio_features(combined_train_df, ratio_cols)
        eval_ratios = _build_ratio_features(eval_df, ratio_cols)

        if train_ratios.shape[1] > 0:
            X_train = np.hstack([X_train_mol, train_ratios])
            X_eval = np.hstack([X_eval_mol, eval_ratios])
        else:
            X_train = X_train_mol
            X_eval = X_eval_mol

    if X_train.shape[1] == 0:
        return None

    preds = _train_predict(X_train, train_y, X_eval, seed=seed)

    recall_5 = compute_recall(eval_y, preds, 5)
    recall_10 = compute_recall(eval_y, preds, 10)

    return {
        "recall_5": recall_5,
        "recall_10": recall_10,
        "n_train_external": len(train_df),
        "n_seed": n_seed,
        "n_eval": len(eval_df),
        "n_features": X_train.shape[1],
    }


# ---------------------------------------------------------------------------
# Random baseline (expected recall)
# ---------------------------------------------------------------------------


def random_recall(n, percentile):
    """Expected recall for random selection at a given percentile.

    When randomly selecting k = ceil(n * percentile/100) items from n,
    the expected fraction of truly-top-k items found is k/n normalized
    to the top-k set size, which equals k/n = percentile/100.
    """
    return percentile / 100.0


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------


def load_studies():
    with open(_STUDIES_JSON) as f:
        return json.load(f)


def run_experiment(
    variant="both",
    study_filter=None,
    n_seeds=N_SEEDS,
    dry_run=False,
    seed_fraction=SEED_FRACTION,
):
    print("Loading LNPDB...")
    dataset = load_lnpdb_full()
    full_df = dataset.df
    print(f"  {len(full_df):,} formulations loaded")

    all_studies = load_studies()
    print(f"  {len(all_studies)} studies from studies_with_ids.json")

    if study_filter:
        studies = [s for s in all_studies if s["study_id"] in study_filter]
        print(f"  Filtered to {len(studies)} studies: {[s['study_id'] for s in studies]}")
    else:
        studies = list(all_studies)

    seeds = RANDOM_SEEDS[:n_seeds]

    # Compute total runs
    variants = []
    if variant in ("cold", "both"):
        variants.append("cold")
    if variant in ("warm", "both"):
        variants.append("warm")

    total_runs = len(studies) * len(seeds) * len(variants)

    print(f"\n{'=' * 70}")
    print("CROSS-STUDY TRANSFER EXPERIMENT")
    print(f"{'=' * 70}")
    print(f"Studies: {len(studies)}")
    print(f"Variants: {variants}")
    print(f"Seeds: {seeds}")
    print(f"Total runs: {total_runs}")
    print(f"Seed fraction (warm start): {seed_fraction}")

    if dry_run:
        print("\nDRY RUN -- would execute:")
        for si in studies:
            for v in variants:
                for s in seeds:
                    print(f"  {si['study_id']} / {v} / seed={s}")
        print(f"\nTotal: {total_runs} runs")
        return None

    # Pre-filter study DataFrames for ALL studies (needed for training sets)
    print("\nFiltering study DataFrames...")
    study_dfs = {}
    for si in all_studies:
        sid = si["study_id"]
        sub = filter_study_df(full_df, si)
        if len(sub) == 0:
            print(f"  WARNING: {sid} has 0 rows after filtering, skipping")
            continue
        study_dfs[sid] = sub
        marker = " *" if any(s["study_id"] == sid for s in studies) else ""
        print(f"  {sid}: {len(sub)} formulations{marker}")

    # Results storage
    results = {
        "cold": {},
        "warm": {},
    }
    completed = 0
    t_start = time.time()

    for si in studies:
        sid = si["study_id"]
        if sid not in study_dfs:
            continue

        test_df = study_dfs[sid]

        # Build training set: all other studies (from the full set, not just evaluated ones)
        train_parts = []
        for other_si in all_studies:
            other_sid = other_si["study_id"]
            if other_sid == sid:
                continue
            if other_sid in study_dfs:
                train_parts.append(study_dfs[other_sid])

        if not train_parts:
            print(f"\n  SKIP {sid}: no training studies available")
            continue

        train_df = pd.concat(train_parts, ignore_index=True)

        print(f"\n{'=' * 50}")
        print(f"Held-out: {sid} | N_test={len(test_df)} | N_train={len(train_df)} ({len(train_parts)} studies)")
        print(f"Type: {si['study_type']} | ILs: {si['n_unique_il']}")
        print(f"{'=' * 50}")

        if "cold" in variants:
            results["cold"][sid] = {}
            for s in seeds:
                completed += 1
                t0 = time.time()
                print(f"  [{completed}/{total_runs}] cold / seed={s} ... ", end="", flush=True)
                try:
                    r = run_cold_start(train_df, test_df, si, seed=s)
                    if r is None:
                        print("SKIPPED (no features)")
                        continue
                    results["cold"][sid][s] = r
                    print(f"top5={r['recall_5']:.3f} top10={r['recall_10']:.3f} ({time.time() - t0:.1f}s)")
                except Exception as e:
                    print(f"FAILED: {e}")
                    import traceback

                    traceback.print_exc()

        if "warm" in variants:
            results["warm"][sid] = {}
            for s in seeds:
                completed += 1
                t0 = time.time()
                print(f"  [{completed}/{total_runs}] warm / seed={s} ... ", end="", flush=True)
                try:
                    r = run_warm_start(train_df, test_df, si, seed=s, seed_fraction=seed_fraction)
                    if r is None:
                        print("SKIPPED (no features or too small)")
                        continue
                    results["warm"][sid][s] = r
                    print(f"top5={r['recall_5']:.3f} top10={r['recall_10']:.3f} ({time.time() - t0:.1f}s)")
                except Exception as e:
                    print(f"FAILED: {e}")
                    import traceback

                    traceback.print_exc()

    elapsed_total = time.time() - t_start

    # -----------------------------------------------------------------------
    # Aggregate and print summary
    # -----------------------------------------------------------------------

    print(f"\n{'=' * 70}")
    print("RESULTS SUMMARY")
    print(f"{'=' * 70}")
    print(f"Total time: {elapsed_total:.0f}s")

    summary_rows = []

    for si in studies:
        sid = si["study_id"]
        row = {
            "study_id": sid,
            "pmid": si["pmid"],
            "study_type": si["study_type"],
            "n_formulations": si.get("n_formulations", si.get("n_lnp_ids", 0)),
            "n_unique_il": si["n_unique_il"],
        }

        for v in variants:
            seed_results = results.get(v, {}).get(sid, {})
            if seed_results:
                r5_vals = [sr["recall_5"] for sr in seed_results.values()]
                r10_vals = [sr["recall_10"] for sr in seed_results.values()]
                row[f"{v}_recall_5_mean"] = float(np.mean(r5_vals))
                row[f"{v}_recall_5_std"] = float(np.std(r5_vals, ddof=1)) if len(r5_vals) > 1 else 0.0
                row[f"{v}_recall_10_mean"] = float(np.mean(r10_vals))
                row[f"{v}_recall_10_std"] = float(np.std(r10_vals, ddof=1)) if len(r10_vals) > 1 else 0.0
            else:
                for metric in ["recall_5_mean", "recall_5_std", "recall_10_mean", "recall_10_std"]:
                    row[f"{v}_{metric}"] = float("nan")

        # Random baseline
        row["random_recall_5"] = random_recall(row["n_formulations"], 5)
        row["random_recall_10"] = random_recall(row["n_formulations"], 10)

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    # Print cold start table
    if "cold" in variants:
        print("\n--- Cold Start (Zero Shot) ---")
        print(
            f"{'Study':<25} {'N':>6} {'Type':<30} "
            f"{'Top-5%':>8} {'Top-10%':>8} {'Rand-5%':>8} {'Rand-10%':>8} {'Lift-5%':>8}"
        )
        print("-" * 125)
        for _, row in summary_df.iterrows():
            r5 = row.get("cold_recall_5_mean", float("nan"))
            r10 = row.get("cold_recall_10_mean", float("nan"))
            rand5 = row["random_recall_5"]
            rand10 = row["random_recall_10"]
            lift5 = r5 / rand5 if rand5 > 0 and not np.isnan(r5) else float("nan")
            r5_str = f"{r5:.3f}" if not np.isnan(r5) else "N/A"
            r10_str = f"{r10:.3f}" if not np.isnan(r10) else "N/A"
            lift5_str = f"{lift5:.2f}x" if not np.isnan(lift5) else "N/A"
            print(
                f"{row['study_id']:<25} {row['n_formulations']:>6} "
                f"{row['study_type']:<30} "
                f"{r5_str:>8} {r10_str:>8} {rand5:>8.3f} {rand10:>8.3f} {lift5_str:>8}"
            )

        cold_vals = summary_df["cold_recall_5_mean"].dropna()
        if len(cold_vals) > 0:
            print(f"\nMean cold-start top-5% recall: {cold_vals.mean():.3f} (+/- {cold_vals.std():.3f})")
            print(f"Mean cold-start top-5% lift vs random: {cold_vals.mean() / 0.05:.2f}x")

        cold_10 = summary_df["cold_recall_10_mean"].dropna()
        if len(cold_10) > 0:
            print(f"Mean cold-start top-10% recall: {cold_10.mean():.3f} (+/- {cold_10.std():.3f})")

    # Print warm start table
    if "warm" in variants:
        print("\n--- Warm Start (25% Seed) ---")
        print(
            f"{'Study':<25} {'N':>6} {'Type':<30} "
            f"{'Top-5%':>8} {'Top-10%':>8} {'Rand-5%':>8} {'Rand-10%':>8} {'Lift-5%':>8}"
        )
        print("-" * 125)
        for _, row in summary_df.iterrows():
            r5 = row.get("warm_recall_5_mean", float("nan"))
            r10 = row.get("warm_recall_10_mean", float("nan"))
            rand5 = row["random_recall_5"]
            rand10 = row["random_recall_10"]
            lift5 = r5 / rand5 if rand5 > 0 and not np.isnan(r5) else float("nan")
            r5_str = f"{r5:.3f}" if not np.isnan(r5) else "N/A"
            r10_str = f"{r10:.3f}" if not np.isnan(r10) else "N/A"
            lift5_str = f"{lift5:.2f}x" if not np.isnan(lift5) else "N/A"
            print(
                f"{row['study_id']:<25} {row['n_formulations']:>6} "
                f"{row['study_type']:<30} "
                f"{r5_str:>8} {r10_str:>8} {rand5:>8.3f} {rand10:>8.3f} {lift5_str:>8}"
            )

        warm_vals = summary_df["warm_recall_5_mean"].dropna()
        if len(warm_vals) > 0:
            print(f"\nMean warm-start top-5% recall: {warm_vals.mean():.3f} (+/- {warm_vals.std():.3f})")
            print(f"Mean warm-start top-5% lift vs random: {warm_vals.mean() / 0.05:.2f}x")

        warm_10 = summary_df["warm_recall_10_mean"].dropna()
        if len(warm_10) > 0:
            print(f"Mean warm-start top-10% recall: {warm_10.mean():.3f} (+/- {warm_10.std():.3f})")

    # Comparison with within-study BO (from memory: mean within-study
    # top-5% recall across strategies is ~0.53 for random, ~0.74 for best)
    if "cold" in variants and len(cold_vals) > 0:
        print("\n--- Context ---")
        print("Within-study random baseline (from main benchmark): ~0.53 top-5% recall")
        print("Within-study best (NGBoost): ~0.74 top-5% recall")
        print(f"Cross-study cold start: {cold_vals.mean():.3f} top-5% recall")
        if cold_vals.mean() < 0.10:
            print("Conclusion: Cross-study transfer is near-random, confirming that SAR is study-dependent.")

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save per-seed detailed results
    detailed = {
        "config": {
            "variant": variant,
            "n_seeds": n_seeds,
            "seeds": seeds,
            "seed_fraction": seed_fraction,
            "n_studies": len(studies),
            "timestamp": datetime.now().isoformat(),
        },
        "cold": {},
        "warm": {},
    }

    for v in variants:
        for sid, seed_results in results.get(v, {}).items():
            detailed[v][sid] = {int(s): r for s, r in seed_results.items()}

    # Use variant-specific filenames to avoid overwriting
    suffix = f"_{variant}" if variant != "both" else ""
    json_path = _RESULTS_DIR / f"cross_study_transfer_results{suffix}.json"
    with open(json_path, "w") as f:
        json.dump(detailed, f, indent=2)
    print(f"\nDetailed results saved to {json_path}")

    # Save summary CSV
    csv_path = _RESULTS_DIR / f"cross_study_transfer_summary{suffix}.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"Summary CSV saved to {csv_path}")

    return summary_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Cross-study transfer experiment (leave-one-out)",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="both",
        choices=["cold", "warm", "both"],
        help="Which variant(s) to run (default: both)",
    )
    parser.add_argument(
        "--study",
        type=str,
        default=None,
        help="Comma-separated study IDs to run (default: all)",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=N_SEEDS,
        help=f"Number of random seeds (default: {N_SEEDS})",
    )
    parser.add_argument(
        "--seed-fraction",
        type=float,
        default=SEED_FRACTION,
        help=f"Warm start seed fraction (default: {SEED_FRACTION})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List runs without executing",
    )
    args = parser.parse_args()

    study_filter = None
    if args.study:
        study_filter = set(s.strip() for s in args.study.split(","))

    run_experiment(
        variant=args.variant,
        study_filter=study_filter,
        n_seeds=args.n_seeds,
        dry_run=args.dry_run,
        seed_fraction=args.seed_fraction,
    )


if __name__ == "__main__":
    main()
