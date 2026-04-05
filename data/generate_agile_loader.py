"""Load pre-computed AGILE (GNN) embeddings from NPZ cache.

AGILE embeddings are pre-computed by ``generate_AGILE_embeddings.py``
(which requires the AGILE repo and its conda env). This module provides
a ``agile_embeddings()`` function matching the standard encoder interface.

Reference: Xu et al., "AGILE: A Graph Neural Network Framework for
Ionizable Lipid Design", Nature Communications, 2024.
"""

import logging
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("lnpbo")

EMBED_DIM = 512
_NPZ_PATH = Path(__file__).parent / "agile_embeddings.npz"

_cache: dict[str, np.ndarray] | None = None


def _load_cache() -> dict[str, np.ndarray]:
    global _cache
    if _cache is not None:
        return _cache
    if not _NPZ_PATH.exists():
        raise FileNotFoundError(
            f"AGILE embeddings not found at {_NPZ_PATH}. Run data/generate_AGILE_embeddings.py first."
        )
    data = np.load(_NPZ_PATH, allow_pickle=False)
    _cache = {str(s): data["embeddings"][i] for i, s in enumerate(data["smiles"])}
    logger.info("AGILE: loaded %d embeddings from cache", len(_cache))
    return _cache


def agile_embeddings(
    list_of_smiles: list[str],
    cache_name: str = "default",
    scaler=None,
) -> tuple[np.ndarray, StandardScaler]:
    """Look up AGILE GNN embeddings for a list of SMILES."""
    cache = _load_cache()
    missing = [s for s in list_of_smiles if s not in cache]
    if missing:
        logger.warning("%d SMILES not found in AGILE cache, using zero vectors", len(missing))

    arr = np.array([cache.get(s, np.zeros(EMBED_DIM)) for s in list_of_smiles])

    if scaler is not None:
        return scaler.transform(arr), scaler
    new_scaler = StandardScaler()
    return new_scaler.fit_transform(arr), new_scaler
