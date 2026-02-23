from __future__ import annotations
import numpy as np
from sklearn.decomposition import PCA
from .generate_morgan_fingerprints import morgan_fingerprints
from .generate_mordred_descriptors import mordred_descriptors
from .generate_LiON_fingerprints import lion_fingerprints

def compute_pcs(list_of_smiles:list[str], feature_type:str, experiment_values:list[float]|None=None, n_components:int=50):
    """
    Docstring for compute_pcs
    
    :param list_of_smiles: list of molecule SMILES
    :type list_of_smiles: list[str]
    :param feature_type: mfp, mordred, lion
    :type feature_type: str
    :param experiment_values: list of averaged experiment values per IL
    :type experiment_values: list[float]
    :param n_components: number of principle components to decompose to
    :type n_components: int
    """
    if feature_type == "mfp": fp_scaled = morgan_fingerprints(list_of_smiles)
    elif feature_type == "mordred": fp_scaled = mordred_descriptors(list_of_smiles)
    elif feature_type == "lion": fp_scaled = lion_fingerprints(list_of_smiles, experiment_values)
    else: raise NameError("Type of feature not found")
    pca = PCA(n_components=n_components)
    pc = pca.fit_transform(fp_scaled)
    map = {smiles: pc for smiles, pc in zip(list_of_smiles, pc)}
    pc_list = np.array([map[smiles] for smiles in list_of_smiles])
    return pc, map, pc_list
