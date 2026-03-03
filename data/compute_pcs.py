from __future__ import annotations
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from .generate_morgan_fingerprints import morgan_fingerprints


def compute_pcs(
    list_of_smiles: list[str],
    feature_type: str,
    experiment_values: list[float] | None = None,
    n_components: int = 50,
    reduction: str = "pca",
):
    """Compute reduced-dimensionality representations of molecular fingerprints.

    Parameters
    ----------
    list_of_smiles : list[str]
        SMILES strings for each molecule.
    feature_type : str
        Fingerprint type: "mfp" (Morgan), "mordred", or "lion".
    experiment_values : list[float] or None
        Per-molecule target values. Required for LiON features and PLS reduction.
    n_components : int
        Number of components to extract.
    reduction : str
        Dimensionality reduction method: "pca" (unsupervised, default) or
        "pls" (supervised, requires experiment_values).

        PCA maximizes Var(X w) — fingerprint variance.
        PLS maximizes Cov(X w, y)^2 — covariance with the target.

        The first PLS weight vector solves:
            w_1 = argmax_w  (w' X' y)^2  s.t. ||w|| = 1
        which is the first left singular vector of X'y. Subsequent components
        are extracted from deflated residuals (NIPALS algorithm).

        When the target is a scalar (single Experiment_value per lipid), PLS
        reduces to maximizing the correlation between projected fingerprints
        and the response. This is equivalent to continuum regression at the
        PLS end of the PCA-OLS continuum (Stone & Brooks, 1990).

        scale=False is used because fingerprints are already StandardScaled
        upstream. PLS latent variables are scale-invariant for univariate y,
        so y does not require separate normalization.

        References
        ----------
        Wold, S., Sjostrom, M., & Eriksson, L.
        "PLS-regression: a basic tool of chemometrics."
        Chemometrics and Intelligent Laboratory Systems, 58(2), 2001,
        pp. 109-130.

        Geladi, P. & Kowalski, B.R.
        "Partial least-squares regression: a tutorial."
        Analytica Chimica Acta, 185, 1986, pp. 1-17.
    """
    if feature_type == "lion" and experiment_values is None:
        raise ValueError("experiment_values required for lion feature type")
    if reduction == "pls" and experiment_values is None:
        raise ValueError("experiment_values required for PLS reduction")

    if feature_type == "mfp":
        fp_scaled, fp_scaler = morgan_fingerprints(list_of_smiles)
    elif feature_type == "mordred":
        from .generate_mordred_descriptors import mordred_descriptors
        fp_scaled, fp_scaler = mordred_descriptors(list_of_smiles)
    elif feature_type == "lion":
        from .generate_LiON_fingerprints import lion_fingerprints
        fp_scaled, fp_scaler = lion_fingerprints(list_of_smiles, experiment_values)
    else:
        raise ValueError("Type of feature not found")

    # Clamp n_components to what the data supports
    max_components = min(fp_scaled.shape[0], fp_scaled.shape[1])
    if reduction == "pls":
        # PLS requires n_components < min(n_samples, n_features)
        max_components = max(max_components - 1, 1)
    n_components = min(n_components, max_components)

    if reduction == "pca":
        reducer = PCA(n_components=n_components, random_state=42)
        pc_matrix = reducer.fit_transform(fp_scaled)
    elif reduction == "pls":
        y = np.asarray(experiment_values, dtype=float)
        reducer = PLSRegression(n_components=n_components, scale=False)
        reducer.fit(fp_scaled, y)
        pc_matrix = reducer.transform(fp_scaled)
    else:
        raise ValueError(f"Unknown reduction method: {reduction!r}. Use 'pca' or 'pls'.")

    smiles_to_pc = {s: row for s, row in zip(list_of_smiles, pc_matrix)}
    pc_list = np.array([smiles_to_pc[s] for s in list_of_smiles])
    return pc_matrix, smiles_to_pc, pc_list, reducer, fp_scaler
