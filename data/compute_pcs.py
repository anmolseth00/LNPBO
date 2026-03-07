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

    Returns (pc_matrix, reducer, fp_scaler, fp_scaled) where fp_scaled is the
    pre-reduction scaled fingerprint matrix (useful for PLS re-fitting).

    References: Wold et al., Chemometrics and Intelligent Laboratory Systems,
    58(2), 2001; Geladi & Kowalski, Analytica Chimica Acta, 185, 1986.
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
            list_of_smiles, radius=_radius, n_bits=_n_bits, scaler=fitted_scaler,
        )
    elif feature_type == "count_mfp":
        _radius = fp_radius if fp_radius is not None else 3
        _n_bits = fp_bits if fp_bits is not None else 2048
        fp_scaled, fp_scaler = morgan_fingerprints(
            list_of_smiles, radius=_radius, n_bits=_n_bits, count=True,
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
                # PLS can fail with very few samples or degenerate features;
                # fall back to PCA silently
                reducer = PCA(n_components=n_components, random_state=42)
                pc_matrix = reducer.fit_transform(fp_scaled)
        else:
            raise ValueError(f"Unknown reduction method: {reduction!r}. Use 'pca', 'pls', or 'none'.")

    return pc_matrix, reducer, fp_scaler, fp_scaled
