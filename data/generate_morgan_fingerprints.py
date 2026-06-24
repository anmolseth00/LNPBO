import logging

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

logger = logging.getLogger("lnpbo")


def single_morgan_fingerprints(smiles: str, radius: int = 3, n_bits: int = 1024, count: bool = False):
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        if count:
            from rdkit.DataStructs import ConvertToNumpyArray

            arr = np.zeros(n_bits)
            ConvertToNumpyArray(mfpgen.GetCountFingerprint(mol), arr)
            return arr
        return np.array(mfpgen.GetFingerprint(mol))
    else:
        return np.zeros(n_bits)


def morgan_fingerprints(
    list_of_smiles: list[str],
    radius: int = 3,
    n_bits: int = 1024,
    count: bool = False,
    scaler=None,
):
    # rdkit returns None for unparseable SMILES, which single_morgan_*
    # silently maps to a zero vector. Count parse failures, always warn, and
    # raise above a 5% failure fraction.
    n_failed = sum(1 for s in list_of_smiles if Chem.MolFromSmiles(s) is None)
    if n_failed > 0:
        n_total = len(list_of_smiles)
        pct = 100.0 * n_failed / n_total if n_total else 0.0
        logger.warning(
            "morgan_fingerprints: %d/%d SMILES (%.1f%%) failed to parse - zero vectors used",
            n_failed, n_total, pct,
        )
        if pct > 5.0:
            raise ValueError(
                f"morgan_fingerprints: {pct:.1f}% of SMILES ({n_failed}/{n_total}) failed to "
                "parse and would become zero vectors (threshold 5%). Aborting."
            )

    mfps = np.array(
        [single_morgan_fingerprints(smiles, radius, n_bits, count=count) for smiles in tqdm(list_of_smiles)]
    )
    if scaler is not None:
        return scaler.transform(mfps), scaler
    mfps_scaler = StandardScaler()
    mfps_scaled = mfps_scaler.fit_transform(mfps)
    return mfps_scaled, mfps_scaler
