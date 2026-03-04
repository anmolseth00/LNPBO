"""RDKit atom/bond featurization and molecular graph construction for D-MPNN.

Follows the featurization scheme from Yang et al. 2019 (Chemprop).
One-hot encodes atom/bond properties and constructs directed bond graphs
for message passing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from rdkit import Chem


# ---------------------------------------------------------------------------
# Feature vocabularies
# ---------------------------------------------------------------------------

ATOM_FEATURES = {
    "atomic_num": list(range(1, 119)),  # 1-118
    "degree": [0, 1, 2, 3, 4, 5],
    "formal_charge": [-2, -1, 0, 1, 2],
    "chiral_tag": [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER,
    ],
    "num_hs": [0, 1, 2, 3, 4],
    "hybridization": [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ],
}

# +1 for each category (unknown/other bin) + 1 for aromaticity boolean
ATOM_FDIM = sum(len(v) + 1 for v in ATOM_FEATURES.values()) + 1

BOND_FEATURES = {
    "bond_type": [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ],
    "stereo": [
        Chem.rdchem.BondStereo.STEREONONE,
        Chem.rdchem.BondStereo.STEREOANY,
        Chem.rdchem.BondStereo.STEREOZ,
        Chem.rdchem.BondStereo.STEREOE,
        Chem.rdchem.BondStereo.STEREOCIS,
        Chem.rdchem.BondStereo.STEREOTRANS,
    ],
}

# +1 for each category (unknown bin) + 1 conjugated + 1 in_ring
BOND_FDIM = sum(len(v) + 1 for v in BOND_FEATURES.values()) + 2


def _ohe(value, vocab: list) -> list[int]:
    """One-hot encode value against vocab, with an extra 'other' bin."""
    encoding = [0] * (len(vocab) + 1)
    if value in vocab:
        encoding[vocab.index(value)] = 1
    else:
        encoding[-1] = 1  # other
    return encoding


def atom_features(atom: Chem.Atom) -> list[int]:
    features = []
    features += _ohe(atom.GetAtomicNum(), ATOM_FEATURES["atomic_num"])
    features += _ohe(atom.GetTotalDegree(), ATOM_FEATURES["degree"])
    features += _ohe(atom.GetFormalCharge(), ATOM_FEATURES["formal_charge"])
    features += _ohe(atom.GetChiralTag(), ATOM_FEATURES["chiral_tag"])
    features += _ohe(atom.GetTotalNumHs(), ATOM_FEATURES["num_hs"])
    features += _ohe(atom.GetHybridization(), ATOM_FEATURES["hybridization"])
    features += [int(atom.GetIsAromatic())]
    return features


def bond_features(bond: Chem.Bond) -> list[int]:
    features = []
    features += _ohe(bond.GetBondType(), BOND_FEATURES["bond_type"])
    features += [int(bond.GetIsConjugated())]
    features += [int(bond.IsInRing())]
    features += _ohe(bond.GetStereo(), BOND_FEATURES["stereo"])
    return features


# ---------------------------------------------------------------------------
# Molecular graph
# ---------------------------------------------------------------------------

@dataclass
class MolGraph:
    """Graph representation of a single molecule for D-MPNN.

    The directed message passing operates on directed edges (bonds).
    Each undirected bond (i, j) becomes two directed edges: i->j and j->i.

    Attributes:
        n_atoms: number of atoms
        n_bonds: number of directed edges (2x undirected bonds)
        f_atoms: (n_atoms, atom_fdim) atom feature matrix
        f_bonds: (n_bonds, bond_fdim) bond feature matrix
        a2b: (n_atoms, max_degree) mapping atom -> incoming bond indices
        b2a: (n_bonds,) mapping bond -> source atom
        b2revb: (n_bonds,) mapping bond -> its reverse bond index
    """
    n_atoms: int
    n_bonds: int
    f_atoms: np.ndarray
    f_bonds: np.ndarray
    a2b: list[list[int]]  # ragged: atom_idx -> list of incoming bond indices
    b2a: np.ndarray        # bond_idx -> source atom of that directed edge
    b2revb: np.ndarray     # bond_idx -> reverse bond index


def mol_to_graph(smiles: str) -> MolGraph | None:
    """Convert a SMILES string to a MolGraph. Returns None if parsing fails."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    n_atoms = mol.GetNumAtoms()
    n_bonds = 0

    f_atoms_list = []
    f_bonds_list = []
    a2b: list[list[int]] = [[] for _ in range(n_atoms)]
    b2a_list: list[int] = []
    b2revb_list: list[int] = []

    for atom in mol.GetAtoms():
        f_atoms_list.append(atom_features(atom))

    for bond in mol.GetBonds():
        bf = bond_features(bond)
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        # Directed edge i -> j (bond index = n_bonds)
        # Directed edge j -> i (bond index = n_bonds + 1)
        # For edge i->j: "incoming to j", source is i
        # For edge j->i: "incoming to i", source is j
        f_bonds_list.append(bf)
        f_bonds_list.append(bf)

        # Edge i->j is incoming to atom j
        a2b[j].append(n_bonds)
        b2a_list.append(i)
        b2revb_list.append(n_bonds + 1)

        # Edge j->i is incoming to atom i
        a2b[i].append(n_bonds + 1)
        b2a_list.append(j)
        b2revb_list.append(n_bonds)

        n_bonds += 2

    f_atoms = np.array(f_atoms_list, dtype=np.float32) if f_atoms_list else np.zeros((1, ATOM_FDIM), dtype=np.float32)
    f_bonds = np.array(f_bonds_list, dtype=np.float32) if f_bonds_list else np.zeros((1, BOND_FDIM), dtype=np.float32)
    b2a = np.array(b2a_list, dtype=np.int64) if b2a_list else np.zeros(1, dtype=np.int64)
    b2revb = np.array(b2revb_list, dtype=np.int64) if b2revb_list else np.zeros(1, dtype=np.int64)

    return MolGraph(
        n_atoms=n_atoms,
        n_bonds=n_bonds,
        f_atoms=f_atoms,
        f_bonds=f_bonds,
        a2b=a2b,
        b2a=b2a,
        b2revb=b2revb,
    )


# ---------------------------------------------------------------------------
# Batched graph (for DataLoader collation)
# ---------------------------------------------------------------------------

@dataclass
class BatchMolGraph:
    """Batched molecular graphs for efficient GPU processing.

    All graphs in the batch are merged into a single large graph with
    offset indices. A scope array tracks which atoms belong to which molecule.
    """
    f_atoms: torch.Tensor    # (total_atoms, atom_fdim)
    f_bonds: torch.Tensor    # (total_bonds, bond_fdim)
    a2b: torch.Tensor        # (total_atoms, max_num_bonds) padded
    b2a: torch.Tensor        # (total_bonds,)
    b2revb: torch.Tensor     # (total_bonds,)
    a_scope: list[tuple[int, int]]  # [(start_atom, n_atoms), ...] per molecule
    n_mols: int


def batch_mol_graphs(graphs: list[MolGraph]) -> BatchMolGraph:
    """Batch a list of MolGraph objects into a single BatchMolGraph.

    Adds a padding "ghost" atom/bond at index 0 so that padded indices
    in a2b point to a zero vector.
    """
    # Start with padding atom and bond at index 0
    # Infer dims from actual graph data (supports augmented features like RWSE)
    atom_fdim = graphs[0].f_atoms.shape[1] if graphs else ATOM_FDIM
    bond_fdim = graphs[0].f_bonds.shape[1] if graphs else BOND_FDIM

    f_atoms = [np.zeros(atom_fdim, dtype=np.float32)]  # padding atom
    f_bonds = [np.zeros(bond_fdim, dtype=np.float32)]   # padding bond
    a2b_all: list[list[int]] = [[]]  # padding atom has no bonds
    b2a_all: list[int] = [0]          # padding bond -> padding atom
    b2revb_all: list[int] = [0]       # padding bond -> itself

    a_scope: list[tuple[int, int]] = []

    a_offset = 1  # current atom offset (1 because of padding)
    b_offset = 1  # current bond offset

    for g in graphs:
        f_atoms.extend(g.f_atoms.tolist())
        f_bonds.extend(g.f_bonds.tolist())

        for atom_bonds in g.a2b:
            a2b_all.append([b + b_offset for b in atom_bonds])

        for ba in g.b2a:
            b2a_all.append(int(ba) + a_offset)

        for br in g.b2revb:
            b2revb_all.append(int(br) + b_offset)

        a_scope.append((a_offset, g.n_atoms))
        a_offset += g.n_atoms
        b_offset += g.n_bonds

    # Pad a2b to rectangular tensor
    max_num_bonds = max(len(bonds) for bonds in a2b_all) if a2b_all else 1
    a2b_padded = np.zeros((len(a2b_all), max_num_bonds), dtype=np.int64)
    for i, bonds in enumerate(a2b_all):
        for j, b in enumerate(bonds):
            a2b_padded[i, j] = b

    return BatchMolGraph(
        f_atoms=torch.tensor(np.array(f_atoms, dtype=np.float32)),
        f_bonds=torch.tensor(np.array(f_bonds, dtype=np.float32)),
        a2b=torch.tensor(a2b_padded),
        b2a=torch.tensor(np.array(b2a_all, dtype=np.int64)),
        b2revb=torch.tensor(np.array(b2revb_all, dtype=np.int64)),
        a_scope=a_scope,
        n_mols=len(graphs),
    )
