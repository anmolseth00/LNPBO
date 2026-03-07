from __future__ import annotations

from pathlib import Path

import numpy as np


def load_npz_cache(cache_dir: Path, name: str) -> dict[str, np.ndarray]:
    path = cache_dir / f"{name}.npz"
    if path.exists():
        data = np.load(path, allow_pickle=True)
        return dict(data)
    return {}


def save_npz_cache(cache_dir: Path, name: str, cache: dict[str, np.ndarray]):
    cache_dir.mkdir(exist_ok=True)
    np.savez_compressed(cache_dir / f"{name}.npz", **cache)
