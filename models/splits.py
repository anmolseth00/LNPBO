"""Train/val/test splitting utilities (scaffold-based and random)."""

from __future__ import annotations

import numpy as np


def _scaffold(smiles: str) -> str:
    """Compute Murcko scaffold for a SMILES string."""
    from rdkit import Chem
    from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    try:
        return MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except Exception:
        return ""


def scaffold_split(
    smiles_list: list[str],
    sizes: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> tuple[list[int], list[int], list[int]]:
    """Split indices by Murcko scaffold, falling back to random if degenerate.

    Groups molecules by scaffold, then assigns entire scaffold groups to
    train/val/test to avoid data leakage of structural motifs.
    """
    from collections import defaultdict

    scaffolds: dict[str, list[int]] = defaultdict(list)
    for i, smi in enumerate(smiles_list):
        scaf = _scaffold(smi)
        scaffolds[scaf].append(i)

    scaffold_groups = sorted(scaffolds.values(), key=lambda g: len(g), reverse=True)

    rng = np.random.RandomState(seed)

    largest_frac = len(scaffold_groups[0]) / len(smiles_list) if scaffold_groups else 0
    if largest_frac > 0.5:
        print(f"Warning: largest scaffold group is {largest_frac:.1%} of data. "
              f"Falling back to stratified random split.")
        return _stratified_random_split(len(smiles_list), sizes, seed)

    n = len(smiles_list)
    n_train = int(n * sizes[0])
    n_val = int(n * sizes[1])

    train_idx, val_idx, test_idx = [], [], []

    order = list(range(len(scaffold_groups)))
    rng.shuffle(order)

    for i in order:
        group = scaffold_groups[i]
        if len(train_idx) < n_train:
            train_idx.extend(group)
        elif len(val_idx) < n_val:
            val_idx.extend(group)
        else:
            test_idx.extend(group)

    return train_idx, val_idx, test_idx


def _stratified_random_split(
    n: int,
    sizes: tuple[float, float, float],
    seed: int,
) -> tuple[list[int], list[int], list[int]]:
    """Simple random split."""
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n).tolist()
    n_train = int(n * sizes[0])
    n_val = int(n * sizes[1])
    return indices[:n_train], indices[n_train:n_train + n_val], indices[n_train + n_val:]
