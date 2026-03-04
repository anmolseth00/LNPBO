from __future__ import annotations

import logging
import os
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

EMBED_DIM = 512
CACHE_DIR = Path(__file__).parent / "unimol_cache"


def _get_repr_model():
    warnings.filterwarnings("ignore")
    logging.disable(logging.INFO)
    from unimol_tools import UniMolRepr
    return UniMolRepr(data_type="molecule", remove_hs=False, use_cuda=False)


def _load_cache(name: str) -> dict[str, np.ndarray]:
    path = CACHE_DIR / f"{name}.npz"
    if path.exists():
        data = np.load(path, allow_pickle=True)
        return dict(data)
    return {}


def _save_cache(name: str, cache: dict[str, np.ndarray]):
    CACHE_DIR.mkdir(exist_ok=True)
    np.savez_compressed(CACHE_DIR / f"{name}.npz", **cache)


def unimol_embeddings(
    list_of_smiles: list[str],
    batch_size: int = 256,
    cache_name: str = "default",
) -> tuple[np.ndarray, StandardScaler]:
    """Extract Uni-Mol CLS embeddings for a list of SMILES.

    Returns (scaled_embeddings, scaler) matching the interface of
    morgan_fingerprints() and mordred_descriptors().
    """
    unique_smiles = list(dict.fromkeys(list_of_smiles))
    cache = _load_cache(cache_name)
    todo = [s for s in unique_smiles if s not in cache]

    if todo:
        model = _get_repr_model()
        n_batches = (len(todo) + batch_size - 1) // batch_size
        print(f"  Encoding {len(todo)} SMILES with Uni-Mol ({n_batches} batches)...")
        t0 = time.time()

        for i in tqdm(range(0, len(todo), batch_size), desc="Uni-Mol"):
            batch = todo[i : i + batch_size]
            try:
                reprs = model.get_repr(batch)
                for smi, emb in zip(batch, reprs):
                    cache[smi] = np.asarray(emb)
            except Exception as e:
                print(f"  Batch failed: {e}, falling back to individual encoding")
                for smi in batch:
                    try:
                        r = model.get_repr([smi])
                        cache[smi] = np.asarray(r[0])
                    except Exception:
                        cache[smi] = np.zeros(EMBED_DIM)

            if (i // batch_size + 1) % 10 == 0:
                _save_cache(cache_name, cache)

        _save_cache(cache_name, cache)
        print(f"  Done: {len(todo)} SMILES in {time.time() - t0:.0f}s")
    else:
        print(f"  All {len(unique_smiles)} SMILES found in cache ({cache_name})")

    embeddings = np.array([cache.get(s, np.zeros(EMBED_DIM)) for s in list_of_smiles])
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    return embeddings_scaled, scaler


if __name__ == "__main__":
    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser(description="Pre-compute Uni-Mol embeddings for LNPDB lipids")
    parser.add_argument(
        "--data-path",
        default=str(Path(__file__).parent / "LNPDB_repo" / "data" / "LNPDB_for_LiON" / "LNPDB.csv"),
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--roles", nargs="+", default=["IL", "HL", "CHL", "PEG"])
    args = parser.parse_args()

    SMILES_COLS = {"IL": "IL_SMILES", "HL": "HL_SMILES", "CHL": "CHL_SMILES", "PEG": "PEG_SMILES"}

    df = pd.read_csv(args.data_path, low_memory=False)
    print(f"Loaded {len(df)} rows from {args.data_path}")

    for role in args.roles:
        col = SMILES_COLS[role]
        smiles = df[col].dropna()
        smiles = smiles[~smiles.isin(["None", "Unknown", ""])]
        unique = smiles.unique().tolist()
        print(f"\n{'='*60}")
        print(f"{role}: {len(unique)} unique SMILES")
        print(f"{'='*60}")
        unimol_embeddings(unique, batch_size=args.batch_size, cache_name=role)

    print("\nAll embeddings cached in:", CACHE_DIR)
