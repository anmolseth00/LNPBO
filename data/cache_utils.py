import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("lnpbo")
from sklearn.preprocessing import StandardScaler


def load_npz_cache(cache_dir: Path, name: str) -> dict[str, np.ndarray]:
    path = cache_dir / f"{name}.npz"
    if path.exists():
        data = np.load(path, allow_pickle=True)
        return dict(data)
    return {}


def save_npz_cache(cache_dir: Path, name: str, cache: dict[str, Any]):
    cache_dir.mkdir(exist_ok=True)
    np.savez_compressed(cache_dir / f"{name}.npz", **cache)


def cached_encode(
    list_of_smiles: list[str],
    compute_fn,
    cache_dir: Path,
    cache_name: str = "default",
    embed_dim: int | None = None,
    scaler=None,
    label: str = "embeddings",
) -> tuple[np.ndarray, StandardScaler]:
    """Shared cache-lookup + compute + scale pattern for molecular encoders.

    Args:
        list_of_smiles: SMILES strings to encode.
        compute_fn: Callable(todo_smiles) -> dict[smiles, np.ndarray].
            Computes embeddings for uncached SMILES.
        cache_dir: Directory for .npz cache files.
        cache_name: Name of the cache file (without extension).
        embed_dim: Embedding dimension (for zero-vector fallback). If None,
            inferred from the first cached entry.
        scaler: Pre-fitted scaler for transform-only mode.
        label: Display name for progress messages.

    Returns:
        (scaled_array, scaler) tuple matching the standard encoder interface.
    """
    unique_smiles = list(dict.fromkeys(list_of_smiles))
    cache = load_npz_cache(cache_dir, cache_name)
    todo = [s for s in unique_smiles if s not in cache]

    if todo:
        logger.info("Computing %s for %d new molecules (cache has %d)...", label, len(todo), len(cache))
        new_entries = compute_fn(todo)
        cache.update(new_entries)
        save_npz_cache(cache_dir, cache_name, cache)
        logger.info("Cached %d %s to %s.npz", len(todo), label, cache_dir / cache_name)
    else:
        logger.debug("%s: all %d molecules found in cache", label, len(unique_smiles))

    if embed_dim is None and cache:
        embed_dim = next(iter(cache.values())).shape[0]
    zeros = np.zeros(embed_dim or 0)
    arr = np.array([cache.get(s, zeros) for s in list_of_smiles])

    if scaler is not None:
        return scaler.transform(arr), scaler
    s = StandardScaler()
    return s.fit_transform(arr), s
