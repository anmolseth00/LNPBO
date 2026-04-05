import logging
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from .cache_utils import cached_encode

logger = logging.getLogger("lnpbo")

EMBED_DIM = 512
CACHE_DIR = Path(__file__).parent / "unimol_cache"


def _get_repr_model():
    warnings.filterwarnings("ignore")
    logging.disable(logging.INFO)
    from unimol_tools import UniMolRepr

    return UniMolRepr(data_type="molecule", remove_hs=False, use_cuda=False)


def unimol_embeddings(
    list_of_smiles: list[str],
    batch_size: int = 256,
    cache_name: str = "default",
    scaler=None,
) -> tuple[np.ndarray, StandardScaler]:
    """Extract Uni-Mol CLS embeddings for a list of SMILES."""

    def _compute(todo):
        model = _get_repr_model()
        result = {}
        n_failed = 0
        t0 = time.time()
        logger.info("Encoding %d SMILES with Uni-Mol...", len(todo))

        for i in tqdm(range(0, len(todo), batch_size), desc="Uni-Mol"):
            batch = todo[i : i + batch_size]
            try:
                reprs = model.get_repr(batch)
                for smi, emb in zip(batch, reprs):
                    result[smi] = np.asarray(emb)
            except (RuntimeError, ValueError) as e:
                logger.warning("Batch failed: %s, falling back to individual encoding", e)
                for smi in batch:
                    try:
                        r = model.get_repr([smi])
                        result[smi] = np.asarray(r[0])
                    except (RuntimeError, ValueError):
                        result[smi] = np.zeros(EMBED_DIM)
                        n_failed += 1

        elapsed = time.time() - t0
        if n_failed > 0:
            pct = 100 * n_failed / len(todo)
            logger.warning("%d SMILES failed [%.1f%%] — replaced with zero vectors (%.0fs)", n_failed, pct, elapsed)
            if pct > 5:
                warnings.warn(
                    f"Uni-Mol: {pct:.1f}% of SMILES failed encoding — zero vectors may degrade model quality",
                    stacklevel=3,
                )
        return result

    return cached_encode(
        list_of_smiles, _compute, CACHE_DIR, cache_name,
        embed_dim=EMBED_DIM, scaler=scaler, label="Uni-Mol embeddings",
    )


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
    logger.info("Loaded %d rows from %s", len(df), args.data_path)

    for role in args.roles:
        col = SMILES_COLS[role]
        smiles = df[col].dropna()
        smiles = smiles[~smiles.isin(["None", "Unknown", ""])]
        unique = smiles.unique().tolist()
        logger.info("%s: %d unique SMILES", role, len(unique))
        unimol_embeddings(unique, batch_size=args.batch_size, cache_name=role)

    logger.info("All embeddings cached in: %s", CACHE_DIR)
