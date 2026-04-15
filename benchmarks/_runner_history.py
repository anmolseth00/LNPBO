"""History tracking utilities shared across benchmark strategies."""

from __future__ import annotations

import numpy as np


def init_history(df, seed_idx, top_k_values=None):
    """Initialize a history dict from the seed pool."""
    seed_vals = df.loc[seed_idx, "Experiment_value"]
    history = {
        "best_so_far": [float(seed_vals.max())],
        "round_best": [],
        "n_evaluated": [len(seed_idx)],
        "all_evaluated": set(seed_idx),
    }
    if top_k_values is not None:
        history["per_round_recall"] = {}
        for k, top_set in top_k_values.items():
            found = len(set(seed_idx) & top_set)
            history["per_round_recall"][k] = [found / len(top_set)]
    return history


def update_history(history, df, training_idx, batch_idx, round_num, top_k_values=None):
    """Append one round's results to the history dict."""
    del round_num
    batch_vals = df.loc[batch_idx, "Experiment_value"]
    all_vals = df.loc[training_idx, "Experiment_value"]
    history["best_so_far"].append(float(all_vals.max()))
    history["round_best"].append(float(batch_vals.max()))
    history["n_evaluated"].append(len(training_idx))
    history["all_evaluated"].update(batch_idx)
    if top_k_values is not None and "per_round_recall" in history:
        for k, top_set in top_k_values.items():
            found = len(history["all_evaluated"] & top_set)
            history["per_round_recall"][k].append(found / len(top_set))


def compute_metrics(history, top_k_values, n_total):
    """Compute final evaluation metrics from a completed benchmark history."""
    del n_total
    bsf = np.array(history["best_so_far"])
    n_eval = np.array(history["n_evaluated"])
    evaluated = history["all_evaluated"]

    auc = float(np.trapezoid(bsf, n_eval) / (n_eval[-1] - n_eval[0])) if len(bsf) > 1 else bsf[0]

    recall = {}
    for k, top_set in top_k_values.items():
        found = len(evaluated & top_set)
        recall[k] = found / len(top_set)

    result = {
        "final_best": float(bsf[-1]),
        "auc": auc,
        "top_k_recall": recall,
        "n_rounds": len(bsf) - 1,
        "n_total_evaluated": int(n_eval[-1]),
    }
    if "per_round_recall" in history:
        result["per_round_recall"] = {str(k): v for k, v in history["per_round_recall"].items()}
    return result


def _run_random(df, seed_idx, oracle_idx, batch_size, n_rounds, seed, top_k_values=None):
    """Run the uniform-random acquisition baseline."""
    rng = np.random.RandomState(seed)
    training_idx = list(seed_idx)
    pool_idx = list(oracle_idx)
    history = init_history(df, training_idx, top_k_values=top_k_values)
    for r in range(n_rounds):
        if len(pool_idx) < batch_size:
            break
        chosen = rng.choice(len(pool_idx), size=batch_size, replace=False)
        batch_idx = [pool_idx[i] for i in sorted(chosen)]
        pool_set = set(batch_idx)
        pool_idx = [i for i in pool_idx if i not in pool_set]
        training_idx.extend(batch_idx)
        update_history(history, df, training_idx, batch_idx, r, top_k_values=top_k_values)
    return history
