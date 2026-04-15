"""Regression tests for the experimental D-MPNN stack."""

import numpy as np

from LNPBO.models.experimental.featurize import batch_mol_graphs, mol_to_graph
from LNPBO.models.experimental.mpnn import MultiComponentMPNN


def test_batch_mol_graphs_zero_pads_widened_atom_features() -> None:
    base = mol_to_graph("CC")
    widened = mol_to_graph("CO")
    assert base is not None
    assert widened is not None

    widened.f_atoms = np.concatenate(
        [widened.f_atoms, np.ones((widened.f_atoms.shape[0], 4), dtype=np.float32)],
        axis=1,
    )

    batch = batch_mol_graphs([base, widened])

    assert batch.f_atoms.shape[1] == widened.f_atoms.shape[1]
    base_start, base_size = batch.a_scope[0]
    assert np.allclose(batch.f_atoms[base_start : base_start + base_size, -4:].numpy(), 0.0)


def test_multicomponent_mpnn_accepts_widened_atom_features() -> None:
    graph = mol_to_graph("CC")
    assert graph is not None

    graph.f_atoms = np.concatenate(
        [graph.f_atoms, np.zeros((graph.f_atoms.shape[0], 4), dtype=np.float32)],
        axis=1,
    )

    batch = batch_mol_graphs([graph])
    model = MultiComponentMPNN(
        component_names=["IL"],
        atom_fdim=batch.f_atoms.shape[1],
        hidden_size=32,
        depth=2,
        ffn_hidden_size=16,
    )
    output = model({"IL": batch})

    assert tuple(output.shape) == (1, 1)
