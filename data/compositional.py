"""Compositional data transforms for simplex-constrained molar ratios.

Implements ILR, ALR, and CLR transforms for mapping D simplex components
to unconstrained coordinates, enabling standard optimizers (L-BFGS-B) to
work directly without post-hoc projection.

References
----------
Egozcue, J.J., Pawlowsky-Glahn, V., Mateu-Figueras, G., & Barcelo-Vidal, C.
    (2003). "Isometric Logratio Transformations for Compositional Data
    Analysis." Mathematical Geology, 35(3), 279-300.

Aitchison, J. (1986). "The Statistical Analysis of Compositional Data."
    Chapman & Hall.

Pawlowsky-Glahn, V. & Egozcue, J.J. (2001). "Geometric approach to
    statistical analysis on the simplex." Stochastic Environmental
    Research and Risk Assessment, 15, 384-398.
"""

import json
import logging
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger("lnpbo")


def _to_fractions(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Normalize rows to sum to 1, adding epsilon to avoid log(0).

    Handles both percentage inputs (summing to ~100) and fraction inputs
    (summing to ~1). Zeros are replaced with eps before normalization.
    """
    X = np.asarray(X, dtype=np.float64)
    X = np.where(X <= 0, eps, X)
    row_sums = X.sum(axis=1, keepdims=True)
    return X / row_sums


def ilr_transform(X_simplex: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Isometric log-ratio transform: (n, D) simplex -> (n, D-1) unconstrained.

    Uses the Helmert sub-matrix basis (default orthonormal basis for ILR).
    The transform maps D simplex components to D-1 unconstrained real
    coordinates while preserving the Aitchison geometry.

    Parameters
    ----------
    X_simplex : array of shape (n, D)
        Compositional data. Can be percentages (summing to ~100) or
        fractions (summing to ~1). Zeros are replaced with eps.
    eps : float
        Small constant added to zeros before log transform.

    Returns
    -------
    X_ilr : array of shape (n, D-1)
        ILR-transformed coordinates in unconstrained R^(D-1).

    Reference: Egozcue et al. (2003), Eq. 3-5.
    """
    X = _to_fractions(X_simplex, eps=eps)
    D = X.shape[1]
    log_X = np.log(X)

    # Helmert sub-matrix basis: V is (D, D-1)
    V = _helmert_basis(D)

    # ILR coords = log(X) @ V
    return log_X @ V


def ilr_inverse(X_ilr: np.ndarray) -> np.ndarray:
    """Inverse ILR: (n, D-1) unconstrained -> (n, D) on the simplex.

    Returns compositions that sum to 1 (fractions, not percentages).

    Parameters
    ----------
    X_ilr : array of shape (n, D-1)
        ILR-transformed coordinates.

    Returns
    -------
    X_simplex : array of shape (n, D)
        Compositions on the simplex (rows sum to 1).

    Reference: Egozcue et al. (2003), inverse of Eq. 3-5.
    """
    X_ilr = np.asarray(X_ilr, dtype=np.float64)
    D = X_ilr.shape[1] + 1

    V = _helmert_basis(D)

    # Inverse: exp(X_ilr @ V^T), then close to simplex
    log_X = X_ilr @ V.T
    X = np.exp(log_X)
    row_sums = X.sum(axis=1, keepdims=True)
    return X / row_sums


def alr_transform(X_simplex: np.ndarray, ref_idx: int = -1, eps: float = 1e-8) -> np.ndarray:
    """Additive log-ratio transform: (n, D) simplex -> (n, D-1) unconstrained.

    Simpler than ILR but non-isometric (distances depend on choice of
    reference component).

    Parameters
    ----------
    X_simplex : array of shape (n, D)
        Compositional data (percentages or fractions).
    ref_idx : int
        Index of the reference component (denominator). Default: last.
    eps : float
        Small constant added to zeros before log transform.

    Returns
    -------
    X_alr : array of shape (n, D-1)
        ALR-transformed coordinates.

    Reference: Aitchison (1986), Ch. 4.
    """
    X = _to_fractions(X_simplex, eps=eps)
    D = X.shape[1]
    ref = X[:, ref_idx : ref_idx + 1] if ref_idx != -1 else X[:, -1:]
    non_ref = np.delete(X, ref_idx if ref_idx != -1 else D - 1, axis=1)
    return np.log(non_ref / ref)


def clr_transform(X_simplex: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Centered log-ratio transform: (n, D) simplex -> (n, D).

    Symmetric across all components but rank-deficient (output rows
    sum to zero, so only D-1 dimensions are independent).

    Parameters
    ----------
    X_simplex : array of shape (n, D)
        Compositional data (percentages or fractions).
    eps : float
        Small constant added to zeros before log transform.

    Returns
    -------
    X_clr : array of shape (n, D)
        CLR-transformed coordinates (rows sum to zero).

    Reference: Aitchison (1986), Ch. 4.
    """
    X = _to_fractions(X_simplex, eps=eps)
    log_X = np.log(X)
    geo_mean = log_X.mean(axis=1, keepdims=True)
    return log_X - geo_mean


def _helmert_basis(D: int) -> np.ndarray:
    """Construct the Helmert sub-matrix orthonormal basis for ILR.

    Returns V of shape (D, D-1) such that V^T V = I_{D-1}.
    Each column j (0-indexed) contrasts the geometric mean of the first
    j+1 components against component j+1.

    Reference: Egozcue et al. (2003), Eq. 4.
    """
    V = np.zeros((D, D - 1))
    for j in range(D - 1):
        # Column j: contrast first j+1 components vs component j+1
        k = j + 1  # number of components in the group
        coeff = 1.0 / np.sqrt(k * (k + 1))
        V[: j + 1, j] = coeff
        V[j + 1, j] = -k * coeff
    return V


def main() -> int:
    """Test ILR/ALR/CLR transforms on LNPDB molar ratios."""
    from LNPBO.data.study_utils import load_lnpdb_clean

    t0 = time.time()

    df = load_lnpdb_clean(drop_duplicates=False)
    df = df.dropna(subset=["IL_SMILES", "Experiment_value"]).reset_index(drop=True)

    ratio_cols = ["IL_molratio", "HL_molratio", "CHL_molratio", "PEG_molratio"]
    ratios = df[ratio_cols].values

    # --- Self-test: ILR inverse recovers original compositions ---
    ilr_coords = ilr_transform(ratios)
    recovered = ilr_inverse(ilr_coords)
    original_fracs = _to_fractions(ratios)
    max_err = np.abs(recovered - original_fracs).max()
    mean_err = np.abs(recovered - original_fracs).mean()
    logger.info("ILR inverse test: max_err=%.2e, mean_err=%.2e", max_err, mean_err)
    assert max_err < 1e-10, f"ILR inverse failed: max_err={max_err:.2e}"

    # ALR and CLR sanity checks
    alr_coords = alr_transform(ratios)
    clr_coords = clr_transform(ratios)
    clr_row_sums = np.abs(clr_coords.sum(axis=1)).max()
    logger.info("CLR row sum max deviation from 0: %.2e", clr_row_sums)
    assert clr_row_sums < 1e-12, f"CLR rows should sum to 0, got max={clr_row_sums:.2e}"
    logger.info("ILR shape: %s, ALR shape: %s, CLR shape: %s", ilr_coords.shape, alr_coords.shape, clr_coords.shape)

    results = {
        "inverse_test": {
            "max_error": float(max_err),
            "mean_error": float(mean_err),
            "passed": bool(max_err < 1e-10),
        },
        "clr_row_sum_test": {
            "max_deviation": float(clr_row_sums),
            "passed": bool(clr_row_sums < 1e-12),
        },
        "shapes": {
            "input": list(ratios.shape),
            "ilr": list(ilr_coords.shape),
            "alr": list(alr_coords.shape),
            "clr": list(clr_coords.shape),
        },
    }

    elapsed = time.time() - t0
    results["elapsed_seconds"] = round(elapsed, 1)

    out_path = Path(__file__).resolve().parent.parent / "diagnostics" / "ilr_transform_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    logger.info("Saved %s", out_path)
    logger.debug("Results:\n%s", json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
