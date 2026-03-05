from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


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


def morgan_fingerprints(list_of_smiles: list[str], radius: int = 3, n_bits: int = 1024, count: bool = False, scaler=None):
    mfps = np.array([single_morgan_fingerprints(smiles, radius, n_bits, count=count) for smiles in tqdm(list_of_smiles)])
    if scaler is not None:
        return scaler.transform(mfps), scaler
    mfps_scaler = StandardScaler()
    mfps_scaled = mfps_scaler.fit_transform(mfps)
    return mfps_scaled, mfps_scaler
