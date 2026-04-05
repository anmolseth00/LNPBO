from pathlib import Path

import numpy as np
from mordred import Calculator, descriptors
from rdkit import Chem
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from .cache_utils import cached_encode

CACHE_DIR = Path(__file__).parent / "mordred_cache"


def mordred_descriptors(
    list_of_smiles: list[str],
    scaler=None,
    cache_name: str = "default",
) -> tuple[np.ndarray, StandardScaler]:
    """Compute mordred 2D descriptors with disk caching.

    References: Moriwaki et al., J. Cheminformatics 10(1):4, 2018.
    """
    calc = Calculator(descriptors, ignore_3D=True)
    n_desc = len(calc.descriptors)

    def _compute(todo):
        result = {}
        for smi in tqdm(todo, desc="Mordred"):
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                vals = np.array([v if isinstance(v, (int, float, np.number)) else 0 for v in calc(mol).fill_missing(0)])
            else:
                vals = np.zeros(n_desc)
            # Sanitize before caching — some mordred descriptors produce inf/nan
            result[smi] = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
        return result

    return cached_encode(
        list_of_smiles, _compute, CACHE_DIR, cache_name,
        embed_dim=n_desc, scaler=scaler, label="mordred descriptors",
    )
