"""Evaluation harness for exact FSBO and related baselines."""

from __future__ import annotations

import json
import logging
import os
from math import comb
from pathlib import Path

import numpy as np

from LNPBO.data.study_utils import (
    build_study_type_map,
    encode_lantern_il,
    lantern_il_feature_cols,
    load_lnpdb_clean,
    study_split,
)

from .adapt import (
    build_cold_gp_model,
    build_meta_initialized_model,
    expected_improvement,
    optimize_gp_model,
    predict_mean_std,
)
from .encoder import normalize_features
from .meta_train import FSBOMetaState, meta_train_fsbo

logger = logging.getLogger("lnpbo")


def stable_study_seed(base_seed: int, study_id: str, extra: int = 0) -> int:
    return int(base_seed + extra + sum(ord(ch) for ch in str(study_id)))


def _score_warm_start_set(loss_matrix: np.ndarray, subset: tuple[int, ...]) -> float:
    return float(np.mean(np.min(loss_matrix[:, subset], axis=1)))


def evolutionary_warm_start(
    loss_matrix: np.ndarray,
    *,
    set_size: int,
    population_size: int = 32,
    generations: int = 64,
    elite_size: int = 8,
    seed: int = 42,
) -> np.ndarray:
    n_tasks, n_candidates = loss_matrix.shape
    del n_tasks
    if set_size <= 0:
        return np.empty((0,), dtype=int)
    if set_size >= n_candidates:
        return np.arange(n_candidates, dtype=int)
    max_population = comb(n_candidates, set_size)
    population_size = min(population_size, max_population)
    elite_size = min(elite_size, population_size)

    rng = np.random.RandomState(seed)
    single_point_loss = np.min(loss_matrix, axis=0)
    weights = np.exp(-single_point_loss)
    if not np.isfinite(weights).all() or weights.sum() <= 0:
        weights = np.ones_like(single_point_loss)
    weights = weights / weights.sum()

    def sample_subset() -> tuple[int, ...]:
        subset = rng.choice(n_candidates, size=set_size, replace=False, p=weights)
        return tuple(sorted(int(i) for i in subset))

    def mutate(parent: tuple[int, ...]) -> tuple[int, ...]:
        child = list(parent)
        drop_pos = int(rng.randint(len(child)))
        remaining = np.array([i for i in range(n_candidates) if i not in child], dtype=int)
        remaining_weights = weights[remaining]
        remaining_weights = remaining_weights / remaining_weights.sum()
        child[drop_pos] = int(rng.choice(remaining, p=remaining_weights))
        return tuple(sorted(child))

    def crossover(parent_a: tuple[int, ...], parent_b: tuple[int, ...]) -> tuple[int, ...]:
        union = sorted(set(parent_a) | set(parent_b))
        if len(union) < set_size:
            union.extend(i for i in range(n_candidates) if i not in union)
        union = np.array(union, dtype=int)
        union_weights = weights[union]
        union_weights = union_weights / union_weights.sum()
        child = rng.choice(union, size=set_size, replace=False, p=union_weights)
        return tuple(sorted(int(i) for i in child))

    population = {sample_subset() for _ in range(population_size)}
    while len(population) < population_size:
        population.add(sample_subset())

    for _ in range(generations):
        ranked = sorted(population, key=lambda subset: _score_warm_start_set(loss_matrix, subset))
        elites = ranked[: max(1, min(elite_size, len(ranked)))]
        next_population = set(elites)
        while len(next_population) < population_size:
            if rng.rand() < 0.5 or len(elites) == 1:
                child = mutate(elites[int(rng.randint(len(elites)))])
            else:
                pa = elites[int(rng.randint(len(elites)))]
                pb = elites[int(rng.randint(len(elites)))]
                child = crossover(pa, pb)
            next_population.add(child)
        population = next_population

    best = min(population, key=lambda subset: _score_warm_start_set(loss_matrix, subset))
    return np.asarray(best, dtype=int)


def compute_source_task_loss_matrix(
    X: np.ndarray,
    y: np.ndarray,
    study_ids: np.ndarray,
    *,
    source_study_ids: np.ndarray,
    candidate_indices: np.ndarray,
    meta_state: FSBOMetaState,
) -> np.ndarray:
    candidate_X = X[candidate_indices]
    task_losses: list[np.ndarray] = []
    for sid in sorted(source_study_ids):
        source_idx = np.where(study_ids == sid)[0]
        if len(source_idx) < 2:
            continue
        model, likelihood = build_meta_initialized_model(X[source_idx], y[source_idx], meta_state=meta_state)
        mean, _ = predict_mean_std(model, likelihood, candidate_X)
        source_y = y[source_idx]
        source_min = float(np.min(source_y))
        source_range = max(float(np.max(source_y) - source_min), 1e-8)
        normalized_reward = np.clip((mean - source_min) / source_range, 0.0, 1.0)
        task_losses.append(1.0 - normalized_reward)
    if not task_losses:
        return np.zeros((1, len(candidate_indices)), dtype=np.float64)
    return np.asarray(task_losses, dtype=np.float64)


def run_fsbo_bo_loop(
    X: np.ndarray,
    y: np.ndarray,
    study_ids: np.ndarray,
    test_study_ids: np.ndarray,
    k_shots: list[int],
    *,
    build_model_fn,
    acquisition_label: str,
    source_study_ids: np.ndarray | None = None,
    meta_state: FSBOMetaState | None = None,
    warm_start: bool = False,
    n_bo_rounds: int = 10,
    batch_size: int = 1,
    target_finetune_steps: int = 25,
    target_finetune_lr: float = 1e-3,
    seed: int = 42,
) -> dict[int, dict[str, object]]:
    study_ids = np.asarray(study_ids)
    task_index_map = {sid: np.where(study_ids == sid)[0] for sid in np.unique(study_ids)}
    warm_start_cache: dict[str, np.ndarray] = {}
    results_by_k: dict[int, dict[str, object]] = {}

    for k in k_shots:
        study_results = []
        for sid in sorted(test_study_ids):
            idx = task_index_map[sid]
            if len(idx) < k + 10:
                continue

            study_seed = stable_study_seed(seed, str(sid), extra=97 * k)
            rng = np.random.RandomState(study_seed)
            if warm_start:
                if meta_state is None or source_study_ids is None:
                    raise ValueError("Warm start requires meta_state and source_study_ids.")
                if sid not in warm_start_cache:
                    warm_start_cache[sid] = compute_source_task_loss_matrix(
                        X, y, study_ids,
                        source_study_ids=source_study_ids,
                        candidate_indices=idx,
                        meta_state=meta_state,
                    )
                support_idx = list(idx[evolutionary_warm_start(warm_start_cache[sid], set_size=k, seed=study_seed)])
            else:
                perm = rng.permutation(len(idx))
                support_idx = list(idx[perm[:k]])

            observed_idx = list(support_idx)
            pool_idx = [int(i) for i in idx if int(i) not in set(observed_idx)]

            for _ in range(n_bo_rounds):
                n_pick = min(batch_size, len(pool_idx))
                if n_pick <= 0:
                    break
                model, likelihood = build_model_fn(X[observed_idx], y[observed_idx])
                if target_finetune_steps > 0:
                    optimize_gp_model(
                        model, likelihood, X[observed_idx], y[observed_idx],
                        n_steps=target_finetune_steps, lr=target_finetune_lr,
                    )

                for _ in range(n_pick):
                    if not pool_idx:
                        break
                    mean, std = predict_mean_std(model, likelihood, X[pool_idx])
                    ei = expected_improvement(mean, std, float(np.max(y[observed_idx])))
                    best_pos = int(np.argmax(ei))
                    chosen = int(pool_idx.pop(best_pos))
                    observed_idx.append(chosen)
                    if pool_idx:
                        model, likelihood = build_model_fn(X[observed_idx], y[observed_idx])
                        if target_finetune_steps > 0:
                            optimize_gp_model(
                                model, likelihood, X[observed_idx], y[observed_idx],
                                n_steps=target_finetune_steps, lr=target_finetune_lr,
                            )

            all_vals = y[idx]
            observed_vals = y[observed_idx]
            study_top10 = set(idx[np.argsort(all_vals)[-10:]])
            study_top50 = set(idx[np.argsort(all_vals)[-min(50, len(idx)):]])
            observed_set = set(observed_idx)
            study_results.append(
                {
                    "study_id": str(sid),
                    "n_in_study": len(idx),
                    "k_shots": k,
                    "n_observed": len(observed_idx),
                    "top10_recall": float(len(observed_set & study_top10) / len(study_top10)),
                    "top50_recall": float(len(observed_set & study_top50) / len(study_top50)) if study_top50 else 0.0,
                    "best_found": float(np.max(observed_vals)),
                    "study_max": float(np.max(all_vals)),
                    "acquisition": acquisition_label,
                }
            )

        if study_results:
            results_by_k[k] = {
                "mean_top10_recall": float(np.mean([r["top10_recall"] for r in study_results])),
                "mean_top50_recall": float(np.mean([r["top50_recall"] for r in study_results])),
                "n_studies_evaluated": len(study_results),
                "per_study": study_results,
            }
    return results_by_k


def random_baseline(
    X: np.ndarray,
    y: np.ndarray,
    study_ids: np.ndarray,
    test_study_ids: np.ndarray,
    k_shots: list[int],
    *,
    n_bo_rounds: int = 10,
    batch_size: int = 1,
    seed: int = 42,
) -> dict[int, dict[str, float]]:
    del X
    study_ids = np.asarray(study_ids)
    task_index_map = {sid: np.where(study_ids == sid)[0] for sid in np.unique(study_ids)}
    results_by_k: dict[int, dict[str, float]] = {}
    for k in k_shots:
        study_results = []
        for sid in sorted(test_study_ids):
            idx = task_index_map[sid]
            if len(idx) < k + 10:
                continue
            rng = np.random.RandomState(stable_study_seed(seed, str(sid), extra=31 * k))
            perm = rng.permutation(len(idx))
            n_total = min(k + n_bo_rounds * batch_size, len(idx))
            observed_idx = list(idx[perm[:n_total]])
            all_vals = y[idx]
            study_top10 = set(idx[np.argsort(all_vals)[-10:]])
            study_top50 = set(idx[np.argsort(all_vals)[-min(50, len(idx)):]])
            observed_set = set(observed_idx)
            study_results.append(
                {
                    "top10_recall": float(len(observed_set & study_top10) / len(study_top10)),
                    "top50_recall": float(len(observed_set & study_top50) / len(study_top50)) if study_top50 else 0.0,
                }
            )
        if study_results:
            results_by_k[k] = {
                "mean_top10_recall": float(np.mean([r["top10_recall"] for r in study_results])),
                "mean_top50_recall": float(np.mean([r["top50_recall"] for r in study_results])),
                "n_studies_evaluated": len(study_results),
            }
    return results_by_k


def main() -> int:
    smoke_mode = os.environ.get("LNPBO_FSBO_SMOKE", "").lower() in {"1", "true", "yes"}

    df = load_lnpdb_clean(drop_duplicates=False)
    df = df.dropna(subset=["IL_SMILES", "Experiment_value"]).reset_index(drop=True)
    study_sizes = df.groupby("study_id").size()
    keep_ids = study_sizes[study_sizes >= 25].index
    df = df[df["study_id"].isin(keep_ids)].reset_index(drop=True)
    logger.info("Using %d rows from %d studies (>=25 per study)", len(df), df["study_id"].nunique())

    study_ids = df["study_id"].astype(str).values
    study_to_type = build_study_type_map(df)
    train_ids, test_ids = study_split(np.unique(study_ids), study_to_type, seed=42)

    if smoke_mode:
        train_ids = set(sorted(train_ids)[:2])
        test_ids = set(sorted(test_ids)[:1])
        keep_ids = train_ids | test_ids
        df = df[df["study_id"].astype(str).isin(keep_ids)].reset_index(drop=True)
        study_ids = df["study_id"].astype(str).values

    train_ids = np.asarray(sorted(train_ids))
    test_ids = np.asarray(sorted(test_ids))
    train_idx = [i for i, sid in enumerate(study_ids) if sid in train_ids]
    test_idx = [i for i, sid in enumerate(study_ids) if sid in test_ids]

    train_encoded, test_encoded, _fitted = encode_lantern_il(df, train_idx=train_idx, test_idx=test_idx, reduction="pca")
    feat_cols = lantern_il_feature_cols(train_encoded)
    import pandas as pd

    train_encoded.index = train_idx
    test_encoded.index = test_idx
    encoded = pd.concat([train_encoded, test_encoded]).sort_index()
    X_raw = encoded[feat_cols].values.astype(np.float64)
    y = encoded["Experiment_value"].values.astype(np.float64)
    train_mask = np.isin(study_ids, list(train_ids))
    _, norm_bounds = normalize_features(X_raw[train_mask])
    X, _ = normalize_features(X_raw, bounds=norm_bounds)
    k_shots = [5] if smoke_mode else [5, 10, 20]

    logger.info("\n=== FSBO Meta-Training ===")
    meta_state = meta_train_fsbo(
        X[train_mask],
        y[train_mask],
        study_ids[train_mask],
        train_ids,
        hidden_dims=(128, 128),
        base_kernel="rbf",
        batch_size=16 if smoke_mode else 50,
        batches_per_task=1,
        n_iterations=4 if smoke_mode else 300,
        lr_kernel=1e-3,
        lr_feature_extractor=1e-3,
        seed=42,
    )

    fsbo_results = run_fsbo_bo_loop(
        X, y, study_ids, test_ids, k_shots,
        build_model_fn=lambda Xo, yo: build_meta_initialized_model(Xo, yo, meta_state=meta_state),
        acquisition_label="FSBO-EI",
        source_study_ids=np.asarray(train_ids),
        meta_state=meta_state,
        warm_start=True,
        n_bo_rounds=1 if smoke_mode else 10,
        batch_size=1,
        target_finetune_steps=2 if smoke_mode else 25,
        target_finetune_lr=1e-3,
        seed=42,
    )
    cold_results = run_fsbo_bo_loop(
        X, y, study_ids, test_ids, k_shots,
        build_model_fn=build_cold_gp_model,
        acquisition_label="ColdGP-EI",
        warm_start=False,
        n_bo_rounds=1 if smoke_mode else 10,
        batch_size=1,
        target_finetune_steps=4 if smoke_mode else 50,
        target_finetune_lr=5e-3,
        seed=42,
    )
    random_results = random_baseline(
        X,
        y,
        study_ids,
        test_ids,
        k_shots,
        n_bo_rounds=1 if smoke_mode else 10,
        batch_size=1,
    )

    report = {
        "n_train_studies": len(train_ids),
        "n_test_studies": len(test_ids),
        "n_rows": len(X),
        "n_features": X.shape[1],
        "fsbo_config": {
            "hidden_dims": list(meta_state.hidden_dims),
            "base_kernel": meta_state.base_kernel,
            "num_mixtures": meta_state.num_mixtures,
            "target_finetune_steps": 25,
            "warm_start": "evolutionary",
            "acquisition": "expected_improvement",
        },
        "meta_training": {
            "global_y_min": meta_state.y_global_bounds[0],
            "global_y_max": meta_state.y_global_bounds[1],
            "n_meta_updates": len(meta_state.meta_losses),
            "initial_nll": float(meta_state.meta_losses[0]) if meta_state.meta_losses else None,
            "final_nll": float(meta_state.meta_losses[-1]) if meta_state.meta_losses else None,
        },
        "fsbo": {k: {kk: vv for kk, vv in v.items() if kk != "per_study"} for k, v in fsbo_results.items()},
        "cold_start_gp": {k: {kk: vv for kk, vv in v.items() if kk != "per_study"} for k, v in cold_results.items()},
        "random": random_results,
        "fsbo_per_study": {k: v.get("per_study", []) for k, v in fsbo_results.items()},
        "cold_start_per_study": {k: v.get("per_study", []) for k, v in cold_results.items()},
    }
    out_path = Path("models") / "fsbo_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    logger.info("Saved %s", out_path)
    return 0
