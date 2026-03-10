#!/usr/bin/env python3
"""TabPFN zero-shot surrogate evaluation for LNPDB.

TabPFN is a transformer pretrained on millions of synthetic tabular datasets.
It outputs full predictive distributions with zero training. For our n=500-680
BO setting, this is within TabPFN's documented sweet spot.

Citation:
    Hollmann, N. et al. (2025). "Accurate Predictions on Small Data with a
    Tabular Foundation Model." Nature.

Usage:
    python -m models.tabpfn_surrogate
"""

import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

from LNPBO.diagnostics.utils import (
    encode_lantern_il,
    lantern_il_feature_cols,
    load_lnpdb_clean,
)

SEEDS = [42, 123, 456, 789, 2024]
MAX_TRAIN_SIZE = 3000
OUTPUT_PATH = Path(__file__).resolve().parent / "tabpfn_results.json"


def _check_tabpfn_available():
    try:
        from tabpfn import TabPFNRegressor  # noqa: F401
        return True
    except ImportError:
        return False


def _fit_predict_tabpfn(X_train, y_train, X_test, seed=42):
    """Fit TabPFN and return (mu, sigma) predictions.

    TabPFN has context length limits. If X_train exceeds MAX_TRAIN_SIZE,
    we subsample stratified by target quantile.
    """
    from tabpfn import TabPFNRegressor

    if len(X_train) > MAX_TRAIN_SIZE:
        rng = np.random.RandomState(seed)
        idx = _stratified_subsample(y_train, MAX_TRAIN_SIZE, rng)
        X_train = X_train[idx]
        y_train = y_train[idx]

    model = TabPFNRegressor()
    model.fit(X_train, y_train)

    # Try to get uncertainty estimates
    sigma = None
    try:
        mu, sigma = model.predict(X_test, return_std=True)
    except TypeError:
        # return_std not supported; try other approaches
        mu = model.predict(X_test)

    if sigma is None:
        # TabPFN may not support return_std; use predict with output_type
        try:
            full_output = model.predict(X_test, output_type="full")
            if hasattr(full_output, "std"):
                sigma = full_output.std
                mu = full_output.mean
            elif isinstance(full_output, dict):
                mu = full_output.get("mean", mu)
                sigma = full_output.get("std", None)
        except (TypeError, AttributeError):
            pass

    return mu, sigma


def _stratified_subsample(y, n, rng):
    """Subsample n indices stratified by target quantile bins."""
    n_bins = min(10, len(y) // 5)
    bins = np.digitize(y, np.percentile(y, np.linspace(0, 100, n_bins + 1)[1:-1]))
    selected = []
    unique_bins = np.unique(bins)
    per_bin = max(1, n // len(unique_bins))
    for b in unique_bins:
        idx_in_bin = np.where(bins == b)[0]
        take = min(per_bin, len(idx_in_bin))
        selected.extend(rng.choice(idx_in_bin, size=take, replace=False).tolist())
    # Fill remainder if needed
    remaining = n - len(selected)
    if remaining > 0:
        all_idx = np.arange(len(y))
        leftover = np.setdiff1d(all_idx, selected)
        if len(leftover) > 0:
            selected.extend(rng.choice(leftover, size=min(remaining, len(leftover)), replace=False).tolist())
    return np.array(selected[:n])


def evaluate_r2(seeds=SEEDS):
    """Evaluate TabPFN R^2 on scaffold-style random splits (5-fold CV)."""
    print("Loading and encoding data...")
    df = load_lnpdb_clean()
    df = df.dropna(subset=["IL_SMILES", "Experiment_value"]).reset_index(drop=True)

    # Identify ratio columns from a probe encoding (no PCA leakage concern for column names)
    probe_enc, _ = encode_lantern_il(df.iloc[:10].copy(), reduction="pca")
    ratio_cols = []
    for role in ["IL", "HL", "CHL", "PEG"]:
        col = f"{role}_molratio"
        if col in probe_enc.columns:
            ratio_cols.append(col)
    mr_col = "IL_to_nucleicacid_massratio"
    if mr_col in probe_enc.columns:
        ratio_cols.append(mr_col)

    n_rows = len(df)
    print(f"  Rows: {n_rows}")

    all_results = []

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        t0 = time.time()
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        fold_r2s = []
        fold_coverages = {"68": [], "90": []}
        fold_widths = {"68": [], "90": []}

        for fold_i, (train_idx, test_idx) in enumerate(kf.split(np.arange(n_rows))):
            train_enc, test_enc, _ = encode_lantern_il(
                df, train_idx=train_idx.tolist(), test_idx=test_idx.tolist(), reduction="pca",
            )
            feat_cols = lantern_il_feature_cols(train_enc)
            train_enc.index = train_idx
            test_enc.index = test_idx

            # Filter ratio_cols to those actually present and non-constant in this fold
            ratio_cols_in_enc = [c for c in ratio_cols if c in train_enc.columns and train_enc[c].nunique() > 1]
            all_feat_cols = feat_cols + ratio_cols_in_enc

            X_train = train_enc[all_feat_cols].values.astype(np.float64)
            y_train = train_enc["Experiment_value"].values.astype(np.float64)
            X_test = test_enc[all_feat_cols].values.astype(np.float64)
            y_test = test_enc["Experiment_value"].values.astype(np.float64)

            # Drop rows with non-finite values
            train_valid = np.isfinite(X_train).all(axis=1) & np.isfinite(y_train)
            test_valid = np.isfinite(X_test).all(axis=1) & np.isfinite(y_test)
            X_train, y_train = X_train[train_valid], y_train[train_valid]
            X_test, y_test = X_test[test_valid], y_test[test_valid]

            if fold_i == 0:
                print(f"  Features: {len(all_feat_cols)} ({len(feat_cols)} molecular PCs + {len(ratio_cols_in_enc)} ratios)")

            mu, sigma = _fit_predict_tabpfn(X_train, y_train, X_test, seed=seed + fold_i)

            r2 = r2_score(y_test, mu)
            fold_r2s.append(r2)
            print(f"  Fold {fold_i+1}: R^2={r2:.4f}, n_train={len(X_train)}, n_test={len(X_test)}", end="")

            if sigma is not None and len(sigma) == len(mu):
                for level, z in [("68", 1.0), ("90", 1.645)]:
                    lo = mu - z * sigma
                    hi = mu + z * sigma
                    coverage = np.mean((y_test >= lo) & (y_test <= hi))
                    width = np.mean(hi - lo)
                    fold_coverages[level].append(coverage)
                    fold_widths[level].append(width)
                print(f", 68%cov={fold_coverages['68'][-1]:.3f}, 90%cov={fold_coverages['90'][-1]:.3f}")
            else:
                print(" (no UQ)")

        elapsed = time.time() - t0
        mean_r2 = np.mean(fold_r2s)
        std_r2 = np.std(fold_r2s)
        print(f"  Mean R^2: {mean_r2:.4f} +/- {std_r2:.4f} ({elapsed:.1f}s)")

        result = {
            "seed": seed,
            "r2_mean": mean_r2,
            "r2_std": std_r2,
            "r2_folds": fold_r2s,
            "elapsed_s": elapsed,
            "n_train_subsample": MAX_TRAIN_SIZE if n_rows > MAX_TRAIN_SIZE * 5 else None,
            "has_uq": bool(fold_coverages["68"]),
        }
        if fold_coverages["68"]:
            result["coverage_68_mean"] = float(np.mean(fold_coverages["68"]))
            result["coverage_90_mean"] = float(np.mean(fold_coverages["90"]))
            result["width_68_mean"] = float(np.mean(fold_widths["68"]))
            result["width_90_mean"] = float(np.mean(fold_widths["90"]))
        all_results.append(result)

    return all_results


def evaluate_bo_recall(seeds=SEEDS, n_seed=500, batch_size=12, n_rounds=15):
    """Simulate BO loop with TabPFN surrogate and compare recall."""
    from LNPBO.benchmarks.runner import (
        compute_metrics,
        copula_transform,
        init_history,
        prepare_benchmark_data,
        update_history,
    )

    print("\nLoading benchmark data...")
    pca_data = prepare_benchmark_data(
        n_seed=n_seed,
        random_seed=seeds[0],
        feature_type="lantern_il_only",
        reduction="pca",
    )
    _, encoded_df, feature_cols, seed_idx, oracle_idx, top_k_values = pca_data

    all_bo_results = []

    for seed in seeds:
        print(f"\n--- BO Seed {seed} ---")
        t0 = time.time()

        rng = np.random.RandomState(seed)
        all_idx = np.arange(len(encoded_df))
        rng.shuffle(all_idx)
        s_seed_idx = sorted(all_idx[:n_seed])
        s_oracle_idx = sorted(all_idx[n_seed:])

        s_top_k_values = {
            10: set(encoded_df.nlargest(10, "Experiment_value").index),
            50: set(encoded_df.nlargest(50, "Experiment_value").index),
            100: set(encoded_df.nlargest(100, "Experiment_value").index),
        }

        training_idx = list(s_seed_idx)
        pool_idx = list(s_oracle_idx)
        history = init_history(encoded_df, training_idx)

        for r in range(n_rounds):
            if len(pool_idx) < batch_size:
                break

            X_train = encoded_df.loc[training_idx, feature_cols].values
            y_train = encoded_df.loc[training_idx, "Experiment_value"].values
            y_train_norm = copula_transform(y_train)

            X_pool = encoded_df.loc[pool_idx, feature_cols].values

            mu, sigma = _fit_predict_tabpfn(X_train, y_train_norm, X_pool, seed=seed + r)

            if sigma is not None and len(sigma) == len(mu):
                scores = mu + 5.0 * sigma
            else:
                scores = mu

            top_k = np.argsort(scores)[-batch_size:][::-1]
            batch_idx = [pool_idx[i] for i in top_k]
            batch_set = set(batch_idx)
            pool_idx = [i for i in pool_idx if i not in batch_set]
            training_idx.extend(batch_idx)
            update_history(history, encoded_df, training_idx, batch_idx, r)

            batch_best = encoded_df.loc[batch_idx, "Experiment_value"].max()
            cum_best = history["best_so_far"][-1]
            print(f"  Round {r+1}: batch_best={batch_best:.3f}, cum_best={cum_best:.3f}")

        elapsed = time.time() - t0
        metrics = compute_metrics(history, s_top_k_values, len(encoded_df))

        print(f"  Top-10={metrics['top_k_recall'][10]:.1%}, "
              f"Top-50={metrics['top_k_recall'][50]:.1%}, "
              f"Top-100={metrics['top_k_recall'][100]:.1%} ({elapsed:.1f}s)")

        all_bo_results.append({
            "seed": seed,
            "top_10_recall": metrics["top_k_recall"][10],
            "top_50_recall": metrics["top_k_recall"][50],
            "top_100_recall": metrics["top_k_recall"][100],
            "final_best": metrics["final_best"],
            "auc": metrics["auc"],
            "elapsed_s": elapsed,
        })

    return all_bo_results


def main():
    if not _check_tabpfn_available():
        print("ERROR: tabpfn is not installed.")
        print("Install with: .venv/bin/pip install tabpfn")
        print("Writing placeholder results with error status.")
        results = {
            "error": "tabpfn not installed",
            "install_cmd": ".venv/bin/pip install tabpfn",
        }
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_PATH, "w") as f:
            json.dump(results, f, indent=2)
        return

    print("=" * 70)
    print("TabPFN Zero-Shot Surrogate Evaluation")
    print("  Hollmann et al. (2025), Nature")
    print("=" * 70)

    r2_results = evaluate_r2()
    bo_results = evaluate_bo_recall()

    r2_means = [r["r2_mean"] for r in r2_results]
    has_uq = r2_results[0].get("has_uq", False)

    summary = {
        "model": "TabPFNRegressor",
        "citation": "Hollmann et al. (2025). Accurate Predictions on Small Data with a Tabular Foundation Model. Nature.",
        "feature_type": "lantern_il_only",
        "reduction": "pca",
        "max_train_subsample": MAX_TRAIN_SIZE,
        "has_uq": has_uq,
        "r2": {
            "mean": float(np.mean(r2_means)),
            "std": float(np.std(r2_means)),
            "per_seed": r2_results,
        },
        "bo_recall": {
            "top_10_mean": float(np.mean([r["top_10_recall"] for r in bo_results])),
            "top_10_std": float(np.std([r["top_10_recall"] for r in bo_results])),
            "top_50_mean": float(np.mean([r["top_50_recall"] for r in bo_results])),
            "top_50_std": float(np.std([r["top_50_recall"] for r in bo_results])),
            "top_100_mean": float(np.mean([r["top_100_recall"] for r in bo_results])),
            "top_100_std": float(np.std([r["top_100_recall"] for r in bo_results])),
            "per_seed": bo_results,
        },
    }

    if has_uq and r2_results[0].get("coverage_68_mean") is not None:
        summary["calibration"] = {
            "coverage_68_mean": float(np.mean([r["coverage_68_mean"] for r in r2_results])),
            "coverage_90_mean": float(np.mean([r["coverage_90_mean"] for r in r2_results])),
            "width_68_mean": float(np.mean([r["width_68_mean"] for r in r2_results])),
            "width_90_mean": float(np.mean([r["width_90_mean"] for r in r2_results])),
        }

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  R^2: {summary['r2']['mean']:.4f} +/- {summary['r2']['std']:.4f}")
    print(f"  Top-10 recall: {summary['bo_recall']['top_10_mean']:.1%} +/- {summary['bo_recall']['top_10_std']:.1%}")
    print(f"  Top-50 recall: {summary['bo_recall']['top_50_mean']:.1%} +/- {summary['bo_recall']['top_50_std']:.1%}")
    print(f"  UQ available: {has_uq}")
    if "calibration" in summary:
        print(f"  68% coverage: {summary['calibration']['coverage_68_mean']:.3f}")
        print(f"  90% coverage: {summary['calibration']['coverage_90_mean']:.3f}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
