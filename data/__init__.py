"""Data subpackage for LNPBO: dataset loading, molecular encoding, and
dimensionality reduction.

This package provides:

- ``Dataset``: core class for loading LNPDB CSV files, encoding lipid
  molecular features (Morgan fingerprints, RDKit descriptors, Mordred
  descriptors, LiON embeddings, Uni-Mol embeddings, AGILE embeddings,
  CheMeleon embeddings), and tracking optimization rounds.
- ``compute_pcs``: PCA/PLS dimensionality reduction for fingerprint matrices.
- Fingerprint/embedding generators for each molecular representation.
"""

__all__ = ["Dataset"]


def __getattr__(name):
    """Lazy-load the Dataset class on first access to avoid heavy imports at
    package import time.

    Args:
        name: Attribute name being accessed.

    Returns:
        The requested attribute.

    Raises:
        AttributeError: If the attribute is not provided by this package.
    """
    if name == "Dataset":
        from .dataset import Dataset

        return Dataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
