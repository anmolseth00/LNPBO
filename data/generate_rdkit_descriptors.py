import contextlib
import logging
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from .cache_utils import load_npz_cache, save_npz_cache  # not using cached_encode: needs keep_mask post-processing

logger = logging.getLogger("lnpbo")

# All 2D RDKit descriptors (excludes 3D-dependent ones)
_DESC_LIST = [(name, func) for name, func in Descriptors.descList if not name.startswith("PMI")]
CACHE_DIR = Path(__file__).parent / "rdkit_cache"


def rdkit_descriptors(
    list_of_smiles: list[str],
    scaler=None,
    keep_mask=None,
    cache_name: str = "default",
) -> tuple[np.ndarray, StandardScaler]:
    """Compute RDKit 2D molecular descriptors for a list of SMILES.

    When scaler and keep_mask are provided, applies the pre-fitted transformer
    instead of fitting a new one (used for encoding new data consistently with
    a training set).
    """
    n_desc = len(_DESC_LIST)

    def _compute(todo):
        result = {}
        for smi in tqdm(todo, desc="RDKit descriptors"):
            mol = Chem.MolFromSmiles(smi)
            vals = np.zeros(n_desc)
            if mol is not None:
                for j, (_, func) in enumerate(_DESC_LIST):
                    with contextlib.suppress(Exception):
                        vals[j] = func(mol)
            result[smi] = vals
        return result

    # Manual caching: can't use cached_encode because of column-filtering step
    unique_smiles = list(dict.fromkeys(list_of_smiles))
    cache = load_npz_cache(CACHE_DIR, cache_name)
    todo = [s for s in unique_smiles if s not in cache]

    if todo:
        logger.info("Computing RDKit descriptors for %d new molecules (cache has %d)...", len(todo), len(cache))
        cache.update(_compute(todo))
        save_npz_cache(CACHE_DIR, cache_name, cache)

    result = np.array([cache.get(s, np.zeros(n_desc)) for s in list_of_smiles])

    # Replace NaN/Inf with column medians
    for j in range(result.shape[1]):
        col = result[:, j]
        mask = ~np.isfinite(col)
        if mask.any():
            median = np.nanmedian(col[~mask]) if (~mask).any() else 0.0
            result[mask, j] = median

    # Filter columns: use provided mask or compute from data
    if keep_mask is not None:
        result = result[:, keep_mask]
    else:
        variances = result.var(axis=0)
        keep_mask = variances > 0
        result = result[:, keep_mask]

    if scaler is not None:
        return scaler.transform(result), scaler

    new_scaler = StandardScaler()
    scaled = new_scaler.fit_transform(result)
    new_scaler.keep_mask_ = keep_mask  # type: ignore[attr-defined]
    return scaled, new_scaler
