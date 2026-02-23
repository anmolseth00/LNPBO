from __future__ import annotations
from tqdm import tqdm
import numpy as np
from rdkit import Chem
from mordred import Calculator, descriptors
from sklearn.preprocessing import StandardScaler

def single_mordred_descriptors(smiles):
    """
    Generate mordred descriptors for a single molecule using SMILES
    
    :param smiles: molecule SMILES
    :type smiles: str
    """
    calc = Calculator(descriptors, ignore_3D=True)
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        result = calc(mol)
        values = np.array([value if np.issubdtype(type(value), np.number) else 0 for value in result.fill_missing(0)])
        return values
    else: return np.zeros(len(calc.descriptors))

def mordred_descriptors(list_of_smiles:list[str]):
    """
    Generate mordred descriptors for a list of molecules using SMILES
    
    :param list_of_smiles: list of molecule SMILES
    :type list_of_smiles: list[str]
    """
    mordred = np.array([single_mordred_descriptors(smiles) for smiles in tqdm(list_of_smiles)])
    mordred_scaler = StandardScaler()
    mordred_scaled = mordred_scaler.fit_transform(mordred)
    return mordred_scaled