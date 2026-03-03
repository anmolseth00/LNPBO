from __future__ import annotations
from tqdm import tqdm
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.preprocessing import StandardScaler

def single_morgan_fingerprints(smiles:str, radius:int=3, n_bits:int=1024):
    """
    Generate morgan fingerprints for a single molecule using SMILES
    
    :param smiles: molecule SMILES
    :type smiles: str
    :param radius: encode substructures within this number of bonds
    :type radius: int
    :param n_bits: length of fingerprint bit vector
    :type n_bits: int
    """
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None: return np.array(mfpgen.GetFingerprint(mol))
    else: return np.zeros(n_bits)

def morgan_fingerprints(list_of_smiles:list[str], radius:int=3, n_bits:int=1024):
    """
    Generate morgan fingerprints for a list of molecules using SMILES
    
    :param list_of_smiles: list of molecule SMILES
    :type list_of_smiles: list[str]
    :param radius: encode substructures within this number of bonds
    :type radius: int
    :param n_bits: length of fingerprint bit vector
    :type n_bits: int
    """
    mfps = np.array([single_morgan_fingerprints(smiles, radius, n_bits) for smiles in tqdm(list_of_smiles)])
    mfps_scaler = StandardScaler()
    mfps_scaled = mfps_scaler.fit_transform(mfps)
    return mfps_scaled, mfps_scaler