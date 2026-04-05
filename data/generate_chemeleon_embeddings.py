import logging
import urllib.request
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from .cache_utils import cached_encode

logger = logging.getLogger("lnpbo")

EMBED_DIM = 2048
CACHE_DIR = Path(__file__).parent / "chemeleon_cache"
CKPT_URL = "https://zenodo.org/records/15460715/files/chemeleon_mp.pt"


def _get_chemeleon_model():
    """Load pre-trained CheMeleon message-passing network.

    References: Burns et al., arXiv:2506.15792, June 2025.
    """
    from chemprop.nn.agg import MeanAggregation
    from chemprop.nn.message_passing import BondMessagePassing

    cache_dir = Path.home() / ".chemprop"
    cache_dir.mkdir(exist_ok=True)
    ckpt_path = cache_dir / "chemeleon_mp.pt"

    if not ckpt_path.exists():
        logger.info("Downloading CheMeleon checkpoint...")
        urllib.request.urlretrieve(CKPT_URL, str(ckpt_path))

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hp = ckpt["hyper_parameters"]

    mp = BondMessagePassing(
        d_v=hp["d_v"], d_e=hp["d_e"], d_h=hp["d_h"],
        bias=hp["bias"], depth=hp["depth"], dropout=hp["dropout"],
        activation=hp["activation"], undirected=hp["undirected"],
    )
    mp.load_state_dict(ckpt["state_dict"])
    mp.eval()

    agg = MeanAggregation()
    return mp, agg


def chemeleon_embeddings(
    list_of_smiles: list[str],
    cache_name: str = "default",
    scaler=None,
) -> tuple[np.ndarray, StandardScaler]:
    """Extract CheMeleon D-MPNN embeddings for a list of SMILES."""
    from chemprop.data.collate import BatchMolGraph
    from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
    from rdkit import Chem

    def _compute(todo):
        mp, agg = _get_chemeleon_model()
        featurizer = SimpleMoleculeMolGraphFeaturizer()
        result = {}
        batch_size = 256
        for i in tqdm(range(0, len(todo), batch_size), desc="CheMeleon"):
            batch_smiles = todo[i : i + batch_size]
            mols = [Chem.MolFromSmiles(smi) for smi in batch_smiles]
            valid = [(smi, mol) for smi, mol in zip(batch_smiles, mols) if mol is not None]
            for smi, mol in zip(batch_smiles, mols):
                if mol is None:
                    result[smi] = np.zeros(EMBED_DIM)
            if valid:
                mgs = [featurizer(mol) for _, mol in valid]
                bmg = BatchMolGraph(mgs)
                with torch.no_grad():
                    fps = agg(mp(bmg), bmg.batch).numpy()
                for j, (smi, _) in enumerate(valid):
                    result[smi] = fps[j]
        return result

    return cached_encode(
        list_of_smiles, _compute, CACHE_DIR, cache_name,
        embed_dim=EMBED_DIM, scaler=scaler, label="CheMeleon embeddings",
    )
