from __future__ import annotations
from tqdm import tqdm
import numpy as np
from rdkit import Chem
from mordred import Calculator, descriptors
from sklearn.preprocessing import StandardScaler

def single_mordred_descriptors(smiles, calc):
    """
    Generate mordred descriptors for a single molecule using SMILES

    :param smiles: molecule SMILES
    :type smiles: str
    :param calc: pre-instantiated mordred Calculator
    :type calc: mordred.Calculator
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        result = calc(mol)
        values = np.array([value if isinstance(value, (int, float, np.number)) else 0 for value in result.fill_missing(0)])
        return values
    else: return np.zeros(len(calc.descriptors))

def mordred_descriptors(list_of_smiles:list[str]):
    """
    Generate mordred descriptors for a list of molecules using SMILES

    :param list_of_smiles: list of molecule SMILES
    :type list_of_smiles: list[str]
    """
    calc = Calculator(descriptors, ignore_3D=True)
    descriptors_array = np.array([single_mordred_descriptors(smiles, calc) for smiles in tqdm(list_of_smiles)])
    descriptors_scaler = StandardScaler()
    descriptors_scaled = descriptors_scaler.fit_transform(descriptors_array)
    return descriptors_scaled, descriptors_scaler