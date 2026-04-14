"""Benchmark loop that delegates to Optimizer.suggest_indices().

Provides ``OptimizerRunner``, a thin wrapper around ``Optimizer`` that
implements the multi-round acquisition loop for all strategies. The
history dict format is compatible with ``compute_metrics()``.
"""

import time
from datetime import datetime

from .runner import init_history, update_history


def _ts() -> str:
    """Return a bracketed HH:MM:SS stamp for per-round log lines."""
    return datetime.now().strftime("[%H:%M:%S]")


class OptimizerRunner:
    """Multi-round acquisition loop driven by an ``Optimizer`` instance.

    Args:
        optimizer: A configured ``Optimizer`` (no ``space`` required).
    """

    def __init__(self, optimizer):
        self.optimizer = optimizer

    def run(
        self,
        df,
        feature_cols,
        seed_idx,
        oracle_idx,
        n_rounds,
        batch_size,
        encoded_dataset=None,
        top_k_values=None,
    ):
        """Execute a multi-round closed-loop acquisition simulation.

        Args:
            df: Encoded DataFrame with ``feature_cols`` and
                ``Experiment_value``.
            feature_cols: List of feature column names.
            seed_idx: List of initial seed pool indices.
            oracle_idx: List of candidate pool indices (the oracle).
            n_rounds: Maximum number of acquisition rounds.
            batch_size: Number of candidates to acquire per round.
            encoded_dataset: Optional ``Dataset`` object for prospective
                PLS refit (avoids target leakage when using PLS reduction).
            top_k_values: Optional dict mapping k to top-k index sets for
                recall tracking.

        Returns:
            History dict compatible with ``compute_metrics()``.
        """
        training_idx = list(seed_idx)
        pool_idx = list(oracle_idx)
        history = init_history(df, training_idx, top_k_values=top_k_values)

        for r in range(n_rounds):
            if len(pool_idx) < batch_size:
                break

            # Breadcrumb before the potentially long GP fit + acqf solve so
            # operators can distinguish "working" from "hung" on big studies.
            print(
                f"  {_ts()} Round {r + 1}/{n_rounds} starting "
                f"(pool={len(pool_idx)}, training={len(training_idx)})...",
                flush=True,
            )
            round_t0 = time.time()

            try:
                batch_idx = self.optimizer.suggest_indices(
                    df,
                    feature_cols,
                    training_idx,
                    pool_idx,
                    round_num=r,
                    encoded_dataset=encoded_dataset,
                    batch_size=batch_size,
                )
            except Exception as e:
                print(f"  {_ts()} Round {r + 1}: suggest_indices failed ({e})", flush=True)
                import traceback

                traceback.print_exc()
                break

            if not batch_idx:
                break

            batch_set = set(batch_idx)
            pool_idx = [i for i in pool_idx if i not in batch_set]
            training_idx.extend(batch_idx)
            update_history(history, df, training_idx, batch_idx, r, top_k_values=top_k_values)

            batch_best = df.loc[batch_idx, "Experiment_value"].max()
            cum_best = history["best_so_far"][-1]
            round_elapsed = time.time() - round_t0
            print(
                f"  {_ts()} Round {r + 1}: batch_best={batch_best:.3f}, "
                f"cum_best={cum_best:.3f}, n_new={len(batch_idx)}, "
                f"time={round_elapsed:.1f}s",
                flush=True,
            )

        return history
