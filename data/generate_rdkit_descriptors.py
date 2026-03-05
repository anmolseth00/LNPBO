from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


# All 2D RDKit descriptors (excludes 3D-dependent ones)
_DESC_LIST = [(name, func) for name, func in Descriptors.descList if not name.startswith("PMI")]


def rdkit_descriptors(list_of_smiles: list[str], scaler=None, keep_mask=None) -> tuple[np.ndarray, StandardScaler]:
    """Compute RDKit 2D molecular descriptors for a list of SMILES.

    Returns (scaled_descriptors, scaler) matching the fingerprint interface.
    When scaler and keep_mask are provided, applies the pre-fitted transformer
    instead of fitting a new one (used for encoding new data consistently with
    a training set).
    """
    n_desc = len(_DESC_LIST)
    result = np.zeros((len(list_of_smiles), n_desc))

    for i, smi in enumerate(tqdm(list_of_smiles, desc="RDKit descriptors")):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        for j, (_, func) in enumerate(_DESC_LIST):
            try:
                result[i, j] = func(mol)
            except Exception:
                pass

    # Replace NaN/Inf with column medians
    for j in range(n_desc):
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
        scaled = scaler.transform(result)
        return scaled, scaler

    new_scaler = StandardScaler()
    scaled = new_scaler.fit_transform(result)
    new_scaler.keep_mask_ = keep_mask
    return scaled, new_scaler
