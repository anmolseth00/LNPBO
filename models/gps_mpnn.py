"""GPS-style Molecular Property Prediction Network.

Extends D-MPNN (Yang et al. 2019) with techniques from recent literature:

1. Random Walk Structural Encoding (RWSE) — positional awareness
   (Dwivedi et al. 2022; MoSE, ICLR 2025)
2. Layer-normalized message passing with residual connections
3. Global self-attention after local message passing (GPS, NeurIPS 2022)
4. Attention-weighted graph readout
5. Cross-component attention for multi-component formulations
"""


from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from featurize import ATOM_FDIM, BOND_FDIM, BatchMolGraph, MolGraph, mol_to_graph

# ---------------------------------------------------------------------------
# RWSE computation
# ---------------------------------------------------------------------------

def compute_rwse(smiles: str, k: int = 16) -> np.ndarray:
    """Compute Random Walk Structural Encoding for a molecule.

    RWSE[v, i] = Pr(i-step random walk starting at v returns to v).
    Captures ring membership, branching patterns, and local topology.

    Returns (n_atoms, k) float32 array.
    """
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros((1, k), dtype=np.float32)

    n_atoms = mol.GetNumAtoms()
    if n_atoms == 0:
        return np.zeros((1, k), dtype=np.float32)

    adj = np.zeros((n_atoms, n_atoms), dtype=np.float32)
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        adj[i, j] = 1.0
        adj[j, i] = 1.0

    degree = adj.sum(axis=1, keepdims=True)
    degree = np.maximum(degree, 1.0)
    transition = adj / degree

    rwse = np.zeros((n_atoms, k), dtype=np.float32)
    M_power = np.eye(n_atoms, dtype=np.float32)
    for i in range(k):
        M_power = M_power @ transition
        rwse[:, i] = np.diag(M_power)

    return rwse


def mol_to_graph_with_rwse(smiles: str, rwse_dim: int = 16) -> MolGraph | None:
    """Create a MolGraph with RWSE appended to atom features.

    Returns a MolGraph whose f_atoms has shape (n_atoms, ATOM_FDIM + rwse_dim).
    """
    graph = mol_to_graph(smiles)
    if graph is None:
        return None

    rwse = compute_rwse(smiles, k=rwse_dim)

    if rwse.shape[0] != graph.f_atoms.shape[0]:
        rwse = np.zeros((graph.f_atoms.shape[0], rwse_dim), dtype=np.float32)

    graph.f_atoms = np.concatenate([graph.f_atoms, rwse], axis=1)
    return graph


def make_graph_fn(rwse_dim: int = 16):
    """Return a graph function with RWSE for use with LNPDataset(graph_fn=...)."""
    return partial(mol_to_graph_with_rwse, rwse_dim=rwse_dim)


# ---------------------------------------------------------------------------
# GPS-DMPNN Encoder
# ---------------------------------------------------------------------------

class GPSDMPNNEncoder(nn.Module):
    """GPS-style D-MPNN encoder for a single molecular component.

    Differences from standard D-MPNN:
    - Accepts RWSE-augmented atom features (wider atom_fdim)
    - Layer normalization + residual connections in message passing
    - GELU activation
    - Global self-attention layer after message passing
    - Attention-weighted graph readout
    """

    def __init__(
        self,
        atom_fdim: int = ATOM_FDIM,
        bond_fdim: int = BOND_FDIM,
        hidden_size: int = 256,
        depth: int = 4,
        n_attn_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.depth = depth

        # D-MPNN: initial bond message
        self.W_i = nn.Linear(atom_fdim + bond_fdim, hidden_size)

        # D-MPNN: message update layers with residual
        self.W_h = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(depth - 1)
        ])
        self.msg_norms = nn.ModuleList([
            nn.LayerNorm(hidden_size) for _ in range(depth - 1)
        ])

        # D-MPNN: atom output
        self.W_o = nn.Linear(atom_fdim + hidden_size, hidden_size)
        self.atom_norm = nn.LayerNorm(hidden_size)

        # Global self-attention (single Transformer encoder layer, pre-norm)
        self.global_attn = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=n_attn_heads,
            dim_feedforward=hidden_size * 2,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )

        # Attention-weighted readout
        self.readout_query = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)
        self.readout_key = nn.Linear(hidden_size, hidden_size)

        self.dropout_layer = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, batch: BatchMolGraph) -> torch.Tensor:
        f_atoms = batch.f_atoms
        f_bonds = batch.f_bonds
        a2b = batch.a2b
        b2a = batch.b2a
        b2revb = batch.b2revb
        a_scope = batch.a_scope

        # --- D-MPNN message passing ---
        input_feats = torch.cat([f_atoms[b2a], f_bonds], dim=1)
        message = self.act(self.W_i(input_feats))

        for t in range(self.depth - 1):
            nei_a_message = message[a2b].sum(dim=1)
            nei_message = nei_a_message[b2a] - message[b2revb]

            # Pre-norm residual
            update = self.dropout_layer(self.act(self.W_h[t](self.msg_norms[t](nei_message))))
            message = message + update

        # Atom representations
        a_message = message[a2b].sum(dim=1)
        atom_hiddens = self.act(self.W_o(torch.cat([f_atoms, a_message], dim=1)))
        atom_hiddens = self.atom_norm(atom_hiddens)
        atom_hiddens = self.dropout_layer(atom_hiddens)

        # --- Global self-attention ---
        n_mols = len(a_scope)
        max_atoms = max(size for _, size in a_scope) if a_scope else 0

        if max_atoms == 0:
            return torch.zeros(n_mols, self.hidden_size, device=f_atoms.device)

        padded = torch.zeros(n_mols, max_atoms, self.hidden_size, device=f_atoms.device)
        mask = torch.ones(n_mols, max_atoms, dtype=torch.bool, device=f_atoms.device)

        for i, (start, size) in enumerate(a_scope):
            if size > 0:
                padded[i, :size] = atom_hiddens[start:start + size]
                mask[i, :size] = False

        padded = self.global_attn(padded, src_key_padding_mask=mask)

        # --- Attention-weighted readout ---
        query = self.readout_query.expand(n_mols, -1, -1)
        keys = self.readout_key(padded)

        attn_scores = torch.bmm(query, keys.transpose(1, 2)).squeeze(1)
        attn_scores = attn_scores.masked_fill(mask, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(-1)

        mol_vecs = (padded * attn_weights).sum(dim=1)
        return mol_vecs


# ---------------------------------------------------------------------------
# Cross-component attention
# ---------------------------------------------------------------------------

class CrossComponentAttention(nn.Module):
    """Attention module for multi-component formulations.

    Models inter-component interactions (e.g., IL-HL synergy)
    instead of simple concatenation.
    """

    def __init__(self, hidden_size: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            hidden_size, n_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(hidden_size)

    def forward(self, component_vecs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            component_vecs: (batch, n_components, hidden_size)
        Returns:
            fused: (batch, n_components * hidden_size)
        """
        residual = component_vecs
        x = self.norm(component_vecs)
        x, _ = self.attn(x, x, x)
        x = residual + x

        residual = x
        x = self.ffn(self.ffn_norm(x))
        x = residual + x

        return x.flatten(1)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class MultiComponentGPS(nn.Module):
    """GPS-MPNN model for multi-component LNP property prediction.

    Per-component GPS-DMPNN encoders → cross-component attention →
    tabular feature fusion → FFN → scalar prediction.
    """

    def __init__(
        self,
        component_names: list[str],
        atom_fdim: int = ATOM_FDIM,
        bond_fdim: int = BOND_FDIM,
        hidden_size: int = 256,
        depth: int = 4,
        n_attn_heads: int = 4,
        ffn_hidden_size: int = 256,
        ffn_num_layers: int = 2,
        tabular_dim: int = 0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.component_names = component_names
        self.hidden_size = hidden_size

        self.encoders = nn.ModuleDict({
            name: GPSDMPNNEncoder(
                atom_fdim=atom_fdim,
                bond_fdim=bond_fdim,
                hidden_size=hidden_size,
                depth=depth,
                n_attn_heads=n_attn_heads,
                dropout=dropout,
            )
            for name in component_names
        })

        self.use_cross_attn = len(component_names) > 1
        if self.use_cross_attn:
            self.cross_attn = CrossComponentAttention(
                hidden_size, n_heads=n_attn_heads, dropout=dropout
            )

        ffn_input = hidden_size * len(component_names) + tabular_dim

        layers: list[nn.Module] = []
        in_dim = ffn_input
        for _ in range(ffn_num_layers):
            layers.extend([
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, ffn_hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = ffn_hidden_size

        self.ffn_body = nn.Sequential(*layers)
        self.output_head = nn.Linear(ffn_hidden_size, 1)
        self._fingerprint_dim = ffn_hidden_size

    @property
    def fingerprint_dim(self) -> int:
        return self._fingerprint_dim

    def encode(
        self,
        component_graphs: dict[str, BatchMolGraph],
        tabular: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mol_vecs = []
        for name in self.component_names:
            mol_vecs.append(self.encoders[name](component_graphs[name]))

        if self.use_cross_attn:
            stacked = torch.stack(mol_vecs, dim=1)
            combined = self.cross_attn(stacked)
        else:
            combined = torch.cat(mol_vecs, dim=1)

        if tabular is not None:
            combined = torch.cat([combined, tabular], dim=1)

        return self.ffn_body(combined)

    def forward(
        self,
        component_graphs: dict[str, BatchMolGraph],
        tabular: torch.Tensor | None = None,
    ) -> torch.Tensor:
        fp = self.encode(component_graphs, tabular)
        return self.output_head(fp)
