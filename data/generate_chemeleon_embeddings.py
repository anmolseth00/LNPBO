import urllib.request
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

EMBED_DIM = 2048
CACHE_DIR = Path(__file__).parent / "chemeleon_cache"
CKPT_URL = "https://zenodo.org/records/15460715/files/chemeleon_mp.pt"


def _get_chemeleon_model():
    """Load pre-trained CheMeleon message-passing network.

    CheMeleon is a D-MPNN pre-trained on ~1M PubChem molecules to predict
    mordred descriptors.  The learned representations capture rich molecular
    structure information in 2048-dim embeddings.

    References: Burns et al., arXiv:2506.15792, 2025.
    """
    from chemprop.nn.agg import MeanAggregation
    from chemprop.nn.message_passing import BondMessagePassing

    cache_dir = Path.home() / ".chemprop"
    cache_dir.mkdir(exist_ok=True)
    ckpt_path = cache_dir / "chemeleon_mp.pt"

    if not ckpt_path.exists():
        print("Downloading CheMeleon checkpoint...")
        urllib.request.urlretrieve(CKPT_URL, str(ckpt_path))

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hp = ckpt["hyper_parameters"]

    mp = BondMessagePassing(
        d_v=hp["d_v"], d_e=hp["d_e"], d_h=hp["d_h"],
        bias=hp["bias"], depth=hp["depth"],
        dropout=hp["dropout"], activation=hp["activation"],
        undirected=hp["undirected"],
    )
    mp.load_state_dict(ckpt["state_dict"])
    mp.eval()

    agg = MeanAggregation()
    return mp, agg


from .cache_utils import load_npz_cache, save_npz_cache


def chemeleon_embeddings(
    list_of_smiles: list[str],
    cache_name: str = "default",
    scaler=None,
) -> tuple[np.ndarray, StandardScaler]:
    """Extract CheMeleon D-MPNN embeddings for a list of SMILES.

    Returns (scaled_embeddings, scaler) matching the interface of
    morgan_fingerprints() and unimol_embeddings().
    """
    from chemprop.data.collate import BatchMolGraph
    from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
    from rdkit import Chem

    unique_smiles = list(dict.fromkeys(list_of_smiles))
    cache = load_npz_cache(CACHE_DIR,cache_name)
    todo = [s for s in unique_smiles if s not in cache]

    if todo:
        print(f"Computing CheMeleon embeddings for {len(todo)} new molecules "
              f"(cache has {len(cache)})...")
        mp, agg = _get_chemeleon_model()
        featurizer = SimpleMoleculeMolGraphFeaturizer()

        batch_size = 256
        for i in tqdm(range(0, len(todo), batch_size), desc="CheMeleon"):
            batch_smiles = todo[i:i + batch_size]
            mols = [Chem.MolFromSmiles(smi) for smi in batch_smiles]
            valid = [(smi, mol) for smi, mol in zip(batch_smiles, mols) if mol is not None]
            failed = [smi for smi, mol in zip(batch_smiles, mols) if mol is None]

            if valid:
                mgs = [featurizer(mol) for _, mol in valid]
                bmg = BatchMolGraph(mgs)
                with torch.no_grad():
                    H = mp(bmg)
                    fps = agg(H, bmg.batch)
                fps_np = fps.numpy()
                for j, (smi, _) in enumerate(valid):
                    cache[smi] = fps_np[j]

            for smi in failed:
                cache[smi] = np.zeros(EMBED_DIM)

        save_npz_cache(CACHE_DIR,cache_name, cache)
        print(f"  Cached {len(todo)} embeddings to {CACHE_DIR / cache_name}.npz")
    else:
        print(f"  CheMeleon: all {len(unique_smiles)} molecules found in cache")

    arr = np.array([cache[s] for s in list_of_smiles])

    if scaler is not None:
        return scaler.transform(arr), scaler
    s = StandardScaler()
    return s.fit_transform(arr), s
