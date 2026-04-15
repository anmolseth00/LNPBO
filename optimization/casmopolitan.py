"""CASMOPOLITAN mixed-variable Bayesian optimization for LNP formulations.

Implements the core ingredients of Wan et al. (2021) for mixed categorical +
continuous BO on a finite candidate pool.
"""

from __future__ import annotations

import json
import logging
import time

import numpy as np

from LNPBO.runtime_paths import benchmark_results_root, package_root_from

from ._casmopolitan_core import (
    TrustRegion,
    _append_restart_observation,
    _apply_trust_region_penalty,
    _fit_pool_casmopolitan_gp,
    _map_candidates_to_pool,
    _trust_region_pool_mask,
    _ucb_acquisition,
    optimize_mixed_acquisition,
    score_pool_casmopolitan,
    select_batch_casmopolitan,
    select_pool_batch_casmopolitan,
)
from ._casmopolitan_kernels import (
    AdditiveProductKernel,
    ExponentiatedCategoricalKernel,
    MixedCasmopolitanKernel,
)

logger = logging.getLogger("lnpbo")

__all__ = [
    "AdditiveProductKernel",
    "ExponentiatedCategoricalKernel",
    "MixedCasmopolitanKernel",
    "TrustRegion",
    "_append_restart_observation",
    "_apply_trust_region_penalty",
    "_fit_pool_casmopolitan_gp",
    "_map_candidates_to_pool",
    "_trust_region_pool_mask",
    "_ucb_acquisition",
    "optimize_mixed_acquisition",
    "run_casmopolitan_strategy",
    "score_pool_casmopolitan",
    "select_batch_casmopolitan",
    "select_pool_batch_casmopolitan",
]


def run_casmopolitan_strategy(
    encoded_df,
    feature_cols,
    seed_idx,
    oracle_idx,
    batch_size=12,
    n_rounds=15,
    seed=42,
    kappa=5.0,
    normalize="copula",
    trust_length_init=0.5,
    acq_func="ucb",
    use_ilr=True,
    max_train_for_gp=2000,
    top_k_values=None,
):
    """Run CASMOPOLITAN mixed-variable BO loop as a benchmark strategy."""
    from LNPBO.benchmarks.runner import init_history, update_history
    from LNPBO.data.compositional import ilr_transform
    from LNPBO.optimization._normalize import copula_transform

    del max_train_for_gp

    ratio_cols = [c for c in feature_cols if c.endswith("_molratio")]
    mass_ratio_cols = [c for c in feature_cols if c == "IL_to_nucleicacid_massratio"]
    enc_cols = [c for c in feature_cols if c not in ratio_cols and c not in mass_ratio_cols]

    il_names = encoded_df["IL_name"].values
    unique_il_names = np.unique(il_names)
    il_name_to_int = {name: i for i, name in enumerate(unique_il_names)}
    il_cat_all = np.array([il_name_to_int[n] for n in il_names])

    ratio_indices = [feature_cols.index(c) for c in ratio_cols]
    mass_ratio_indices = [feature_cols.index(c) for c in mass_ratio_cols]
    enc_indices = [feature_cols.index(c) for c in enc_cols]

    training_idx = list(seed_idx)
    pool_idx = list(oracle_idx)
    history = init_history(encoded_df, training_idx, top_k_values=top_k_values)

    trust_region = None
    round_start_best = None
    restart_X_raw = None
    restart_y = None

    for r in range(n_rounds):
        if len(pool_idx) < batch_size:
            break

        X_all = encoded_df[feature_cols].values
        y_all = encoded_df["Experiment_value"].values

        X_train_raw = X_all[training_idx]
        y_train = y_all[training_idx].copy()
        X_pool_raw = X_all[pool_idx]

        if normalize == "copula":
            y_train = copula_transform(y_train)
        elif normalize == "zscore":
            mu_y, sigma_y = y_train.mean(), y_train.std()
            if sigma_y > 0:
                y_train = (y_train - mu_y) / sigma_y

        cont_train_parts = [X_train_raw[:, enc_indices]]
        cont_pool_parts = [X_pool_raw[:, enc_indices]]

        if ratio_cols:
            if use_ilr:
                cont_train_parts.append(ilr_transform(X_train_raw[:, ratio_indices]))
                cont_pool_parts.append(ilr_transform(X_pool_raw[:, ratio_indices]))
            else:
                cont_train_parts.append(X_train_raw[:, ratio_indices])
                cont_pool_parts.append(X_pool_raw[:, ratio_indices])

        if mass_ratio_indices:
            cont_train_parts.append(X_train_raw[:, mass_ratio_indices])
            cont_pool_parts.append(X_pool_raw[:, mass_ratio_indices])

        cont_train = np.hstack(cont_train_parts)
        cont_pool = np.hstack(cont_pool_parts)
        del cont_train, cont_pool

        cat_train = il_cat_all[training_idx].reshape(-1, 1).astype(float)
        cat_pool = il_cat_all[pool_idx].reshape(-1, 1).astype(float)
        X_train_aug = np.column_stack([cat_train, np.hstack(cont_train_parts)])
        X_pool_aug = np.column_stack([cat_pool, np.hstack(cont_pool_parts)])
        cont_indices = list(range(1, X_train_aug.shape[1]))

        current_best = float(np.max(y_train))
        restart_from_archive = False
        if trust_region is not None and round_start_best is not None:
            improved = current_best > round_start_best
            restart_from_archive = trust_region.update(improved)
            if restart_from_archive:
                incumbent_idx = int(np.argmax(y_train))
                restart_X_raw, restart_y = _append_restart_observation(
                    restart_X_raw,
                    restart_y,
                    X_train_aug[incumbent_idx],
                    current_best,
                    X_train_aug,
                    y_train,
                    np.random.RandomState(seed + r),
                )

        selected_pool_idx, trust_region = select_pool_batch_casmopolitan(
            X_train_aug,
            y_train,
            X_pool_aug,
            il_cat_train=cat_train.ravel(),
            il_cat_pool=cat_pool.ravel(),
            cont_feature_indices=cont_indices,
            cat_feature_indices=[0],
            batch_size=batch_size,
            kappa=kappa,
            random_seed=seed + r,
            trust_length=trust_length_init,
            acq_func=acq_func,
            trust_region=trust_region,
            restart_from_archive=restart_from_archive,
            restart_X_raw=restart_X_raw,
            restart_y=restart_y,
            restart_kappa=kappa,
        )
        batch_idx = [pool_idx[i] for i in selected_pool_idx]
        round_start_best = current_best

        batch_set = set(batch_idx)
        pool_idx = [i for i in pool_idx if i not in batch_set]
        training_idx.extend(batch_idx)
        update_history(history, encoded_df, training_idx, batch_idx, r, top_k_values=top_k_values)

        batch_best = float(encoded_df.loc[batch_idx, "Experiment_value"].max())
        cum_best = history["best_so_far"][-1]
        logger.info(
            "  Round %d: batch_best=%.3f, cum_best=%.3f, TR_length=%.4f, n_pool=%d",
            r + 1,
            batch_best,
            cum_best,
            trust_region.length,
            len(pool_idx),
        )

    return history


def main():
    """Run CASMOPOLITAN benchmark as a standalone script."""
    import argparse

    parser = argparse.ArgumentParser(
        description="CASMOPOLITAN mixed-variable BO benchmark for LNP formulations",
    )
    parser.add_argument("--seeds", type=str, default="42,123,456,789,2024", help="Comma-separated random seeds")
    parser.add_argument("--n-seed", type=int, default=500, help="Initial seed pool size")
    parser.add_argument("--batch-size", type=int, default=12, help="Batch size per round")
    parser.add_argument("--n-rounds", type=int, default=15, help="Number of BO rounds")
    parser.add_argument("--kappa", type=float, default=5.0, help="UCB exploration weight")
    parser.add_argument("--acq-func", type=str, default="ucb", choices=["ucb", "ei"], help="Acquisition function")
    parser.add_argument(
        "--normalize", type=str, default="copula", choices=["copula", "zscore", "none"], help="Target normalization"
    )
    parser.add_argument(
        "--feature-type", type=str, default="lantern_il_only", help="Feature type for molecular encoding"
    )
    parser.add_argument("--trust-length", type=float, default=0.5, help="Initial trust region length")
    parser.add_argument("--no-ilr", action="store_true", help="Disable ILR transform on ratios")
    parser.add_argument("--max-train-gp", type=int, default=2000, help="Max training size for GP (subsample if larger)")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    from LNPBO.benchmarks.runner import _run_random, compute_metrics, prepare_benchmark_data

    seeds = [int(s) for s in args.seeds.split(",")]

    package_root = package_root_from(__file__, levels_up=2)
    results_dir = benchmark_results_root(package_root)
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output or str(results_dir / "casmopolitan.json")

    logger.info("=" * 70)
    logger.info("CASMOPOLITAN Mixed-Variable BO Benchmark")
    logger.info("=" * 70)
    logger.info("Seeds: %s", seeds)
    logger.info("n_seed=%d, batch_size=%d, n_rounds=%d", args.n_seed, args.batch_size, args.n_rounds)
    logger.info("kappa=%s, acq_func=%s, normalize=%s", args.kappa, args.acq_func, args.normalize)
    logger.info("feature_type=%s, trust_length=%s", args.feature_type, args.trust_length)
    logger.info("use_ilr=%s, max_train_gp=%d", not args.no_ilr, args.max_train_gp)
    logger.info("")

    all_seed_results = {
        "casmopolitan": [],
        "random": [],
        "discrete_xgb_greedy": [],
    }

    for seed in seeds:
        logger.info("\n" + "=" * 70)
        logger.info("Seed: %d", seed)
        logger.info("=" * 70)

        data = prepare_benchmark_data(
            n_seed=args.n_seed,
            random_seed=seed,
            feature_type=args.feature_type,
        )
        encoded, encoded_df, feature_cols, seed_idx, oracle_idx, top_k_values = data

        logger.info("\n--- CASMOPOLITAN ---")
        t0 = time.time()
        history_cas = run_casmopolitan_strategy(
            encoded_df,
            feature_cols,
            seed_idx,
            oracle_idx,
            batch_size=args.batch_size,
            n_rounds=args.n_rounds,
            seed=seed,
            kappa=args.kappa,
            normalize=args.normalize,
            trust_length_init=args.trust_length,
            acq_func=args.acq_func,
            use_ilr=not args.no_ilr,
            max_train_for_gp=args.max_train_gp,
        )
        elapsed_cas = time.time() - t0
        metrics_cas = compute_metrics(history_cas, top_k_values, len(encoded_df))
        all_seed_results["casmopolitan"].append({"seed": seed, "metrics": metrics_cas, "elapsed": elapsed_cas})
        logger.info("  CASMOPOLITAN Time: %.1fs", elapsed_cas)
        logger.info("  Top-K recall: %s", {k: f"{v:.1%}" for k, v in metrics_cas["top_k_recall"].items()})

        logger.info("\n--- Random ---")
        t0 = time.time()
        history_rand = _run_random(encoded_df, seed_idx, oracle_idx, args.batch_size, args.n_rounds, seed)
        elapsed_rand = time.time() - t0
        metrics_rand = compute_metrics(history_rand, top_k_values, len(encoded_df))
        all_seed_results["random"].append({"seed": seed, "metrics": metrics_rand, "elapsed": elapsed_rand})
        logger.info("  Random Time: %.1fs", elapsed_rand)
        logger.info("  Top-K recall: %s", {k: f"{v:.1%}" for k, v in metrics_rand["top_k_recall"].items()})

        logger.info("\n--- XGB Greedy ---")
        t0 = time.time()
        from LNPBO.benchmarks._optimizer_runner import OptimizerRunner
        from LNPBO.optimization.optimizer import Optimizer

        xgb_opt = Optimizer(
            surrogate_type="xgb",
            batch_strategy="greedy",
            random_seed=seed,
            kappa=args.kappa,
            normalize=args.normalize,
            batch_size=args.batch_size,
        )
        xgb_runner = OptimizerRunner(xgb_opt)
        history_xgb = xgb_runner.run(
            encoded_df,
            feature_cols,
            seed_idx,
            oracle_idx,
            n_rounds=args.n_rounds,
            batch_size=args.batch_size,
            encoded_dataset=encoded,
        )
        elapsed_xgb = time.time() - t0
        metrics_xgb = compute_metrics(history_xgb, top_k_values, len(encoded_df))
        all_seed_results["discrete_xgb_greedy"].append({"seed": seed, "metrics": metrics_xgb, "elapsed": elapsed_xgb})
        logger.info("  XGB Greedy Time: %.1fs", elapsed_xgb)
        logger.info("  Top-K recall: %s", {k: f"{v:.1%}" for k, v in metrics_xgb["top_k_recall"].items()})

    logger.info("\n" + "=" * 70)
    logger.info("AGGREGATE RESULTS")
    logger.info("=" * 70)

    summary = {}
    for strategy_name, seed_results in all_seed_results.items():
        recalls = {k: [] for k in [10, 50, 100]}
        for seed_result in seed_results:
            for k in recalls:
                recalls[k].append(seed_result["metrics"]["top_k_recall"].get(k, 0))

        summary[strategy_name] = {}
        for k in recalls:
            vals = np.array(recalls[k])
            summary[strategy_name][f"top_{k}_mean"] = float(vals.mean())
            summary[strategy_name][f"top_{k}_std"] = float(vals.std())
            summary[strategy_name][f"top_{k}_values"] = [float(v) for v in vals]

        elapsed_vals = [seed_result["elapsed"] for seed_result in seed_results]
        summary[strategy_name]["mean_elapsed"] = float(np.mean(elapsed_vals))

    for strategy_name, strategy_summary in summary.items():
        logger.info("\n%s:", strategy_name)
        for k in [10, 50, 100]:
            logger.info(
                "  Top-%d: %.1f%% +/- %.1f%%",
                k,
                strategy_summary[f"top_{k}_mean"] * 100,
                strategy_summary[f"top_{k}_std"] * 100,
            )
        logger.info("  Mean time: %.1fs", strategy_summary["mean_elapsed"])

    output = {
        "config": {
            "seeds": seeds,
            "n_seed": args.n_seed,
            "batch_size": args.batch_size,
            "n_rounds": args.n_rounds,
            "kappa": args.kappa,
            "acq_func": args.acq_func,
            "normalize": args.normalize,
            "feature_type": args.feature_type,
            "trust_length": args.trust_length,
            "use_ilr": not args.no_ilr,
            "max_train_gp": args.max_train_gp,
        },
        "summary": summary,
        "per_seed": {
            strategy: [{"seed": sr["seed"], "metrics": sr["metrics"], "elapsed": sr["elapsed"]} for sr in seed_results]
            for strategy, seed_results in all_seed_results.items()
        },
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Results saved to %s", output_path)


if __name__ == "__main__":
    main()
