from pathlib import Path

import numpy as np
from mordred import Calculator, descriptors
from rdkit import Chem
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

CACHE_DIR = Path(__file__).parent / "mordred_cache"


from .cache_utils import load_npz_cache, save_npz_cache


def mordred_descriptors(
    list_of_smiles: list[str],
    scaler=None,
    cache_name: str = "default",
) -> tuple[np.ndarray, StandardScaler]:
    """Compute mordred 2D descriptors with disk caching.

    Uses mordredcommunity (mordred-community), the maintained fork of
    mordred that supports numpy 2.x and Python 3.12+.

    References: Moriwaki et al., J. Cheminformatics 10(1):4, 2018.
    """
    unique_smiles = list(dict.fromkeys(list_of_smiles))
    cache = load_npz_cache(CACHE_DIR, cache_name)
    todo = [s for s in unique_smiles if s not in cache]

    if todo:
        print(f"Computing mordred descriptors for {len(todo)} new molecules "
              f"(cache has {len(cache)})...")
        calc = Calculator(descriptors, ignore_3D=True)
        for smi in tqdm(todo, desc="Mordred"):
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                result = calc(mol)
                vals = np.array(
                    [v if isinstance(v, (int, float, np.number)) else 0
                     for v in result.fill_missing(0)]
                )
            else:
                vals = np.zeros(len(calc.descriptors))
            cache[smi] = vals
        save_npz_cache(CACHE_DIR, cache_name, cache)
        print(f"  Cached {len(todo)} descriptors to {CACHE_DIR / cache_name}.npz")
    else:
        print(f"  Mordred: all {len(unique_smiles)} molecules found in cache")

    arr = np.array([cache[s] for s in list_of_smiles])

    # Replace inf/nan with 0 — some mordred descriptors produce non-finite values
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    if scaler is not None:
        return scaler.transform(arr), scaler
    s = StandardScaler()
    return s.fit_transform(arr), s
