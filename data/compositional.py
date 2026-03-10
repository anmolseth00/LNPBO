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
import time
from pathlib import Path

import numpy as np


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
    ref = X[:, ref_idx:ref_idx + 1] if ref_idx != -1 else X[:, -1:]
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
        V[:j + 1, j] = coeff
        V[j + 1, j] = -k * coeff
    return V


def main() -> int:
    """Test ILR/ALR/CLR transforms on LNPDB molar ratios and benchmark GP."""
    import torch
    from sklearn.metrics import r2_score
    from sklearn.preprocessing import StandardScaler

    from LNPBO.diagnostics.utils import (
        encode_lantern_il,
        lantern_il_feature_cols,
        load_lnpdb_clean,
        study_split,
    )
    from LNPBO.models.gp_surrogate import _predict, _train_sparse_gp
    from LNPBO.models.splits import scaffold_split

    t0 = time.time()

    df = load_lnpdb_clean(drop_duplicates=False)
    df = df.dropna(subset=["IL_SMILES", "Experiment_value"]).reset_index(drop=True)

    ratio_cols = ["IL_molratio", "HL_molratio", "CHL_molratio", "PEG_molratio"]
    ratios = df[ratio_cols].values

    # --- Self-test: ILR inverse recovers original compositions ---
    ilr_coords = ilr_transform(ratios)
    recovered = ilr_inverse(ilr_coords)
    # Scale recovered back to percentages for comparison
    original_fracs = _to_fractions(ratios)
    max_err = np.abs(recovered - original_fracs).max()
    mean_err = np.abs(recovered - original_fracs).mean()
    print(f"ILR inverse test: max_err={max_err:.2e}, mean_err={mean_err:.2e}")
    assert max_err < 1e-10, f"ILR inverse failed: max_err={max_err:.2e}"

    # ALR and CLR sanity checks
    alr_coords = alr_transform(ratios)
    clr_coords = clr_transform(ratios)
    clr_row_sums = np.abs(clr_coords.sum(axis=1)).max()
    print(f"CLR row sum max deviation from 0: {clr_row_sums:.2e}")
    assert clr_row_sums < 1e-12, f"CLR rows should sum to 0, got max={clr_row_sums:.2e}"
    print(f"ILR shape: {ilr_coords.shape}, ALR shape: {alr_coords.shape}, CLR shape: {clr_coords.shape}")

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
        "gp_comparison": {},
    }

    # --- GP benchmark: ILR vs raw ratios on scaffold split ---
    il_smiles = df["IL_SMILES"].tolist()
    train_idx, val_idx, test_idx = scaffold_split(il_smiles, sizes=(0.8, 0.1, 0.1), seed=42)
    train_idx = sorted(set(train_idx + val_idx))

    train_enc, test_enc, _ = encode_lantern_il(df, train_idx=train_idx, test_idx=test_idx, reduction="pca")
    feat_cols = lantern_il_feature_cols(train_enc)

    y_train = train_enc["Experiment_value"].values
    y_test = test_enc["Experiment_value"].values

    y_mean, y_std = y_train.mean(), max(y_train.std(), 1e-6)
    y_train_s = (y_train - y_mean) / y_std
    y_test_s = (y_test - y_mean) / y_std

    mol_train = train_enc[feat_cols].values
    mol_test = test_enc[feat_cols].values

    # Raw ratios
    raw_train = df.iloc[train_idx][ratio_cols].values
    raw_test = df.iloc[test_idx][ratio_cols].values

    # ILR ratios
    ilr_train = ilr_transform(raw_train)
    ilr_test = ilr_transform(raw_test)

    for ratio_label, r_train, r_test in [
        ("raw_ratios", raw_train, raw_test),
        ("ilr_ratios", ilr_train, ilr_test),
    ]:
        # Combine molecular features + ratio features
        r_scaler = StandardScaler().fit(r_train)
        r_train_s = r_scaler.transform(r_train)
        r_test_s = r_scaler.transform(r_test)

        m_scaler = StandardScaler().fit(mol_train)
        m_train_s = m_scaler.transform(mol_train)
        m_test_s = m_scaler.transform(mol_test)

        X_train_combined = np.hstack([m_train_s, r_train_s])
        X_test_combined = np.hstack([m_test_s, r_test_s])

        train_x = torch.tensor(X_train_combined, dtype=torch.float32)
        train_y = torch.tensor(y_train_s, dtype=torch.float32)
        test_x = torch.tensor(X_test_combined, dtype=torch.float32)

        for kernel_name in ["rbf", "matern52"]:
            label = f"{ratio_label}_{kernel_name}"
            print(f"  Training GP: {label} ...")
            model, likelihood = _train_sparse_gp(
                train_x, train_y, noise_init=1.0, fix_noise=False,
                kernel_name=kernel_name, epochs=30, batch_size=1024,
            )
            mu, sigma = _predict(model, likelihood, test_x)
            mu_np = mu.numpy()
            r2 = float(r2_score(y_test_s, mu_np))
            print(f"    {label}: R^2 = {r2:.4f}")
            results["gp_comparison"][label] = {"r2": r2, "n_features": int(X_train_combined.shape[1])}

    # --- Study-level split ---
    train_study_ids, test_study_ids = study_split(df, seed=42)
    s_train_mask = df["study_id"].isin(train_study_ids)
    s_test_mask = df["study_id"].isin(test_study_ids)
    s_train_idx = df.index[s_train_mask].tolist()
    s_test_idx = df.index[s_test_mask].tolist()

    if len(s_test_idx) > 100:
        s_train_enc, s_test_enc, _ = encode_lantern_il(
            df, train_idx=s_train_idx, test_idx=s_test_idx, reduction="pca",
        )
        s_feat_cols = lantern_il_feature_cols(s_train_enc)

        s_y_train = s_train_enc["Experiment_value"].values
        s_y_test = s_test_enc["Experiment_value"].values
        s_y_mean, s_y_std = s_y_train.mean(), max(s_y_train.std(), 1e-6)
        s_y_train_s = (s_y_train - s_y_mean) / s_y_std
        s_y_test_s = (s_y_test - s_y_mean) / s_y_std

        s_mol_train = s_train_enc[s_feat_cols].values
        s_mol_test = s_test_enc[s_feat_cols].values

        s_raw_train = df.iloc[s_train_idx][ratio_cols].values
        s_raw_test = df.iloc[s_test_idx][ratio_cols].values
        s_ilr_train = ilr_transform(s_raw_train)
        s_ilr_test = ilr_transform(s_raw_test)

        results["gp_comparison_study_split"] = {}

        for ratio_label, r_train, r_test in [
            ("raw_ratios", s_raw_train, s_raw_test),
            ("ilr_ratios", s_ilr_train, s_ilr_test),
        ]:
            r_scaler = StandardScaler().fit(r_train)
            r_train_s = r_scaler.transform(r_train)
            r_test_s = r_scaler.transform(r_test)

            m_scaler = StandardScaler().fit(s_mol_train)
            m_train_s = m_scaler.transform(s_mol_train)
            m_test_s = m_scaler.transform(s_mol_test)

            X_train_combined = np.hstack([m_train_s, r_train_s])
            X_test_combined = np.hstack([m_test_s, r_test_s])

            train_x = torch.tensor(X_train_combined, dtype=torch.float32)
            train_y = torch.tensor(s_y_train_s, dtype=torch.float32)
            test_x = torch.tensor(X_test_combined, dtype=torch.float32)

            for kernel_name in ["rbf", "matern52"]:
                label = f"{ratio_label}_{kernel_name}"
                print(f"  Training GP (study split): {label} ...")
                model, likelihood = _train_sparse_gp(
                    train_x, train_y, noise_init=1.0, fix_noise=False,
                    kernel_name=kernel_name, epochs=30, batch_size=1024,
                )
                mu, sigma = _predict(model, likelihood, test_x)
                mu_np = mu.numpy()
                r2 = float(r2_score(s_y_test_s, mu_np))
                print(f"    {label}: R^2 = {r2:.4f}")
                results["gp_comparison_study_split"][label] = {
                    "r2": r2,
                    "n_features": int(X_train_combined.shape[1]),
                }

    elapsed = time.time() - t0
    results["elapsed_seconds"] = round(elapsed, 1)

    out_path = Path("models") / "ilr_transform_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {out_path}")
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
