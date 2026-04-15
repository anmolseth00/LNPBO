"""Regression tests for the GPS-style experimental MPNN."""

from LNPBO.models.experimental.featurize import batch_mol_graphs
from LNPBO.models.experimental.gps_mpnn import MultiComponentGPS, mol_to_graph_with_rwse


def test_multicomponent_gps_accepts_default_rwse_graphs() -> None:
    graph = mol_to_graph_with_rwse("CC")
    assert graph is not None

    batch = batch_mol_graphs([graph])
    model = MultiComponentGPS(component_names=["IL"])
    output = model({"IL": batch})

    assert tuple(output.shape) == (1, 1)
