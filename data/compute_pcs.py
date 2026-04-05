"""Dimensionality reduction for molecular fingerprints and descriptors.

Supports PCA (unsupervised), PLS (supervised, target-aware), and identity
(no reduction) modes. Used by ``Dataset.encode_dataset()`` to project
high-dimensional fingerprint vectors into compact PC representations
suitable for Gaussian process and tree-based surrogates.
"""

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA

from .generate_morgan_fingerprints import morgan_fingerprints


def compute_pcs(
    list_of_smiles: list[str],
    feature_type: str,
    experiment_values: list[float] | None = None,
    n_components: int = 50,
    reduction: str = "pca",
    cache_name: str = "default",
    fitted_reducer=None,
    fitted_scaler=None,
    fp_radius: int | None = None,
    fp_bits: int | None = None,
):
    """Compute reduced-dimensionality molecular fingerprint representations.

    Generates molecular fingerprints for the given SMILES using the specified
    ``feature_type``, then applies dimensionality reduction (PCA, PLS, or
    none) to produce a compact matrix of principal components.

    Args:
        list_of_smiles: SMILES strings to encode.
        feature_type: Fingerprint/descriptor type. One of ``"mfp"``
            (binary Morgan), ``"count_mfp"`` (count Morgan), ``"rdkit"``
            (RDKit 2D descriptors), ``"mordred"`` (Mordred descriptors),
            ``"lion"`` (LiON D-MPNN embeddings), ``"unimol"`` (Uni-Mol
            embeddings), ``"chemeleon"`` (CheMeleon embeddings), or
            ``"agile"`` (AGILE GNN embeddings).
        experiment_values: Per-SMILES target values, required for PLS
            reduction and LiON encoding. Ignored for PCA/none.
        n_components: Target number of principal components. Automatically
            clamped to ``min(n_samples, n_features)`` when the data is
            too small.
        reduction: Reduction method: ``"pca"`` (default), ``"pls"``
            (partial least squares, supervised), or ``"none"`` (return
            raw scaled fingerprints).
        cache_name: Cache key passed to fingerprint generators for
            on-disk caching (e.g., lipid role name).
        fitted_reducer: Pre-fitted PCA or PLS reducer for transform-only
            mode. When provided, skips fitting and applies the existing
            transform.
        fitted_scaler: Pre-fitted StandardScaler for transform-only mode.
        fp_radius: Override Morgan fingerprint radius (default 3).
        fp_bits: Override Morgan fingerprint bit length (default 1024
            for binary, 2048 for count).

    Returns:
        Tuple of ``(pc_matrix, reducer, fp_scaler, fp_scaled)`` where:
            - ``pc_matrix``: ndarray of shape ``(n_molecules, n_components)``
              with the reduced representation.
            - ``reducer``: fitted PCA/PLS object (or ``None`` if
              ``reduction="none"``).
            - ``fp_scaler``: fitted StandardScaler used to normalize raw
              fingerprints before reduction.
            - ``fp_scaled``: ndarray of scaled fingerprints before reduction,
              useful for PLS re-fitting in prospective loops.

    Raises:
        ValueError: If ``reduction="pls"`` and ``experiment_values`` is
            ``None`` with no ``fitted_reducer``, or if ``feature_type``
            is unrecognized.

    References:
        Wold S. et al., "PLS-regression: a basic tool of chemometrics,"
        Chemometrics and Intelligent Laboratory Systems, 58(2), 109-130, 2001.

        Geladi P. & Kowalski B.R., "Partial least-squares regression: a
        tutorial," Analytica Chimica Acta, 185, 1-17, 1986.
    """
    if feature_type == "lion" and experiment_values is None:
        experiment_values = [0.0] * len(list_of_smiles)
    if reduction == "pls" and experiment_values is None and fitted_reducer is None:
        raise ValueError("experiment_values required for PLS reduction")

    # Extract keep_mask for rdkit descriptors when reusing a fitted scaler
    _keep_mask = getattr(fitted_scaler, "keep_mask_", None) if fitted_scaler is not None else None

    if feature_type == "mfp":
        _radius = fp_radius if fp_radius is not None else 3
        _n_bits = fp_bits if fp_bits is not None else 1024
        fp_scaled, fp_scaler = morgan_fingerprints(
            list_of_smiles,
            radius=_radius,
            n_bits=_n_bits,
            scaler=fitted_scaler,
        )
    elif feature_type == "count_mfp":
        _radius = fp_radius if fp_radius is not None else 3
        _n_bits = fp_bits if fp_bits is not None else 2048
        fp_scaled, fp_scaler = morgan_fingerprints(
            list_of_smiles,
            radius=_radius,
            n_bits=_n_bits,
            count=True,
            scaler=fitted_scaler,
        )
    elif feature_type == "mordred":
        from .generate_mordred_descriptors import mordred_descriptors

        fp_scaled, fp_scaler = mordred_descriptors(list_of_smiles, scaler=fitted_scaler, cache_name=cache_name)
    elif feature_type == "rdkit":
        from .generate_rdkit_descriptors import rdkit_descriptors

        fp_scaled, fp_scaler = rdkit_descriptors(
            list_of_smiles, scaler=fitted_scaler, keep_mask=_keep_mask, cache_name=cache_name
        )
    elif feature_type == "lion":
        from .generate_LiON_fingerprints import lion_fingerprints

        assert experiment_values is not None
        fp_scaled, fp_scaler = lion_fingerprints(list_of_smiles, experiment_values)
    elif feature_type == "unimol":
        from .generate_unimol_embeddings import unimol_embeddings

        fp_scaled, fp_scaler = unimol_embeddings(list_of_smiles, cache_name=cache_name, scaler=fitted_scaler)
    elif feature_type == "chemeleon":
        from .generate_chemeleon_embeddings import chemeleon_embeddings

        fp_scaled, fp_scaler = chemeleon_embeddings(list_of_smiles, cache_name=cache_name, scaler=fitted_scaler)
    elif feature_type == "agile":
        from .generate_agile_loader import agile_embeddings

        fp_scaled, fp_scaler = agile_embeddings(list_of_smiles, cache_name=cache_name, scaler=fitted_scaler)
    else:
        raise ValueError("Type of feature not found")

    if fitted_reducer is not None:
        pc_matrix = fitted_reducer.transform(fp_scaled)
        reducer = fitted_reducer
    else:
        # Clamp n_components to what the data supports
        max_components = min(fp_scaled.shape[0], fp_scaled.shape[1])
        if reduction == "pls":
            # PLS requires n_components < min(n_samples, n_features)
            max_components = max(max_components - 1, 1)
        n_components = min(n_components, max_components)

        if reduction == "none":
            reducer = None
            pc_matrix = fp_scaled
        elif reduction == "pca":
            reducer = PCA(n_components=n_components, random_state=42)
            pc_matrix = reducer.fit_transform(fp_scaled)
        elif reduction == "pls":
            y = np.asarray(experiment_values, dtype=float)
            reducer = PLSRegression(n_components=n_components, scale=False)
            try:
                reducer.fit(fp_scaled, y)
                pc_matrix = reducer.transform(fp_scaled)
            except (ValueError, np.linalg.LinAlgError):
                # PLS can fail with very few samples or degenerate features
                import warnings
                warnings.warn(
                    f"PLS reduction failed, falling back to PCA with {n_components} components. "
                    "This may affect downstream results.",
                    stacklevel=2,
                )
                reducer = PCA(n_components=n_components, random_state=42)
                pc_matrix = reducer.fit_transform(fp_scaled)
        else:
            raise ValueError(f"Unknown reduction method: {reduction!r}. Use 'pca', 'pls', or 'none'.")

    return pc_matrix, reducer, fp_scaler, fp_scaled
