"""Directed Message Passing Neural Network (D-MPNN) for molecular property prediction.

Architecture follows Yang et al. 2019 (Chemprop):
  1. Message passing on directed bond graph (messages live on directed edges)
  2. Atom hidden states aggregated via mean readout
  3. Feed-forward network (FFN) maps molecular embedding to scalar output

Multi-component support: independent MPNN encoders per lipid role (IL, HL, CHL, PEG),
concatenated with tabular features before the FFN.
"""


import torch
import torch.nn as nn

try:
    from LNPBO.models.featurize import ATOM_FDIM, BOND_FDIM, BatchMolGraph
except ImportError:
    raise ImportError(
        "models.featurize was removed. MPNN is an experimental model "
        "not wired into the Optimizer. See models/experimental/README."
    )


class DMPNNEncoder(nn.Module):
    """Single D-MPNN encoder for one molecular input.

    Message passing on directed edges:
        m_{ij}^{t+1} = sum_{k->i, k!=j} W_m * m_{ki}^t
        h_i = tau(W_a * [x_i, sum_j m_{ji}^T])

    Where messages are initialized from bond features concatenated with
    the source atom features.
    """

    def __init__(
        self,
        hidden_size: int = 300,
        depth: int = 3,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.depth = depth

        # Initial message: project [atom_feat, bond_feat] -> hidden
        self.W_i = nn.Linear(ATOM_FDIM + BOND_FDIM, hidden_size, bias=bias)

        # Message update at each depth
        self.W_h = nn.Linear(hidden_size, hidden_size, bias=bias)

        # Final atom hidden state: project [atom_feat, aggregated_message] -> hidden
        self.W_o = nn.Linear(ATOM_FDIM + hidden_size, hidden_size)

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, batch: BatchMolGraph) -> torch.Tensor:
        """Run message passing and return per-molecule embeddings.

        Args:
            batch: BatchMolGraph with all tensors on the correct device.

        Returns:
            mol_vecs: (n_mols, hidden_size) molecule embeddings
        """
        f_atoms = batch.f_atoms   # (total_atoms, atom_fdim)
        f_bonds = batch.f_bonds   # (total_bonds, bond_fdim)
        a2b = batch.a2b           # (total_atoms, max_num_bonds)
        b2a = batch.b2a           # (total_bonds,)
        b2revb = batch.b2revb     # (total_bonds,)
        a_scope = batch.a_scope

        # Initial messages: for each directed edge (i->j), use [atom_i_feat, bond_ij_feat]
        # b2a[b] gives source atom of directed edge b
        input_feats = torch.cat([f_atoms[b2a], f_bonds], dim=1)  # (total_bonds, atom_fdim + bond_fdim)
        message = self.act(self.W_i(input_feats))  # (total_bonds, hidden)

        for _ in range(self.depth - 1):
            # For each directed edge b = (i->j), aggregate messages from edges
            # incoming to i (except the reverse edge j->i).
            # nei_message[b] = sum of messages on edges incoming to atom b2a[b],
            #                  minus the reverse message.
            # Efficient approach: gather via a2b for each bond's source atom.

            # For each atom, sum of all incoming edge messages
            # a2b[atom] gives bond indices incoming to that atom
            nei_a_message = message[a2b].sum(dim=1)  # (total_atoms, hidden)

            # For each bond b (i->j), the aggregated message at source atom i,
            # minus the reverse edge j->i message
            nei_message = nei_a_message[b2a] - message[b2revb]  # (total_bonds, hidden)

            message = self.act(self.W_h(nei_message))
            message = self.dropout(message)

        # Compute atom hidden states
        # For each atom, sum incoming messages
        a_message = message[a2b].sum(dim=1)  # (total_atoms, hidden)
        atom_hiddens = self.act(self.W_o(torch.cat([f_atoms, a_message], dim=1)))
        atom_hiddens = self.dropout(atom_hiddens)

        # Readout: mean aggregation per molecule
        mol_vecs = []
        for start, size in a_scope:
            if size == 0:
                mol_vecs.append(torch.zeros(self.hidden_size, device=f_atoms.device))
            else:
                mol_vecs.append(atom_hiddens[start:start + size].mean(dim=0))

        return torch.stack(mol_vecs, dim=0)  # (n_mols, hidden_size)


class MultiComponentMPNN(nn.Module):
    """Multi-component D-MPNN with FFN head.

    Encodes each lipid component (IL, HL, CHL, PEG) with an independent
    D-MPNN encoder, concatenates the molecular embeddings with optional
    tabular features, and passes through a feed-forward network.

    The penultimate FFN layer output serves as the learned fingerprint.
    """

    def __init__(
        self,
        component_names: list[str],
        hidden_size: int = 300,
        depth: int = 3,
        ffn_hidden_size: int = 300,
        ffn_num_layers: int = 2,
        tabular_dim: int = 0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.component_names = component_names
        self.hidden_size = hidden_size
        self.tabular_dim = tabular_dim

        # Independent MPNN encoder per component
        self.encoders = nn.ModuleDict({
            name: DMPNNEncoder(
                hidden_size=hidden_size,
                depth=depth,
                dropout=dropout,
            )
            for name in component_names
        })

        # FFN input: all component embeddings + tabular features
        ffn_input_dim = hidden_size * len(component_names) + tabular_dim

        # Build FFN layers
        layers: list[nn.Module] = []
        in_dim = ffn_input_dim
        for _ in range(ffn_num_layers - 1):
            layers.extend([
                nn.Dropout(dropout),
                nn.Linear(in_dim, ffn_hidden_size),
                nn.ReLU(),
            ])
            in_dim = ffn_hidden_size

        # Penultimate -> output
        layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(in_dim, ffn_hidden_size))
        layers.append(nn.ReLU())

        self.ffn_body = nn.Sequential(*layers)

        # Final output head (separate so we can extract fingerprints before it)
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
        """Encode inputs to penultimate-layer fingerprints.

        Args:
            component_graphs: {component_name: BatchMolGraph}
            tabular: (batch_size, tabular_dim) or None

        Returns:
            fingerprints: (batch_size, fingerprint_dim)
        """
        mol_vecs = []
        for name in self.component_names:
            mol_vecs.append(self.encoders[name](component_graphs[name]))

        combined = torch.cat(mol_vecs, dim=1)
        if tabular is not None:
            combined = torch.cat([combined, tabular], dim=1)

        return self.ffn_body(combined)

    def forward(
        self,
        component_graphs: dict[str, BatchMolGraph],
        tabular: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass: inputs -> scalar prediction.

        Args:
            component_graphs: {component_name: BatchMolGraph}
            tabular: (batch_size, tabular_dim) or None

        Returns:
            predictions: (batch_size, 1)
        """
        fp = self.encode(component_graphs, tabular)
        return self.output_head(fp)
