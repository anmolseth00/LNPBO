"""Shared logging helpers for benchmark runner loops."""

from __future__ import annotations

from datetime import datetime


def _ts() -> str:
    """Return a bracketed HH:MM:SS stamp for per-round log lines."""
    return datetime.now().strftime("[%H:%M:%S]")


def _log_round_start(r: int, n_rounds: int, pool_size: int, training_size: int) -> None:
    """Announce the start of a round before the fit + acquisition step."""
    print(
        f"  {_ts()} Round {r + 1}/{n_rounds} starting "
        f"(pool={pool_size}, training={training_size})...",
        flush=True,
    )


def _log_round_complete(
    r: int,
    batch_best: float,
    cum_best: float,
    elapsed_s: float,
    **extras: object,
) -> None:
    """Report a completed round with shared and strategy-specific fields."""
    fragments = [f"batch_best={batch_best:.3f}", f"cum_best={cum_best:.3f}"]
    fragments.extend(f"{key}={value}" for key, value in extras.items())
    fragments.append(f"time={elapsed_s:.1f}s")
    print(f"  {_ts()} Round {r + 1}: " + ", ".join(fragments), flush=True)
