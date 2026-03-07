#!/usr/bin/env python3
"""Interpret IL count_mfp PCA components — what chemical variation do they capture?"""

import numpy as np
from scipy.stats import spearmanr

from LNPBO.data.compute_pcs import compute_pcs
from LNPBO.data.lnpdb_bridge import load_lnpdb_full


def main():
    db = load_lnpdb_full()
    df = db.df

    il_smiles = df["IL_SMILES"].dropna().unique().tolist()
    print(f"Unique IL SMILES: {len(il_smiles)}")

    pc_matrix, reducer, _fp_scaler, fp_scaled = compute_pcs(
        il_smiles, feature_type="count_mfp", n_components=5,
        reduction="pca", cache_name="IL",
    )

    print(f"PC matrix shape: {pc_matrix.shape}")
    print(f"Explained variance ratios: {[f'{v:.3f}' for v in reducer.explained_variance_ratio_]}")
    print(f"Cumulative: {[f'{v:.3f}' for v in np.cumsum(reducer.explained_variance_ratio_)]}")

    # Map IL SMILES -> mean transfection value
    il_to_val = df.groupby("IL_SMILES")["Experiment_value"].mean().to_dict()
    vals = np.array([il_to_val.get(s, np.nan) for s in il_smiles])
    valid = ~np.isnan(vals)

    print("\nSpearman correlations with mean Experiment_value:")
    for i in range(5):
        rho, p = spearmanr(pc_matrix[valid, i], vals[valid])
        print(f"  PC{i+1}: rho={rho:+.3f}, p={p:.2e}")

    # For each PC, show top/bottom molecules
    for pc_idx in range(5):
        pc_scores = pc_matrix[:, pc_idx]
        rho, _ = spearmanr(pc_scores[valid], vals[valid])

        top_idx = np.argsort(pc_scores)[-5:][::-1]
        bot_idx = np.argsort(pc_scores)[:5]

        print(f"\n{'='*70}")
        print(f"PC{pc_idx+1} (var={reducer.explained_variance_ratio_[pc_idx]:.3f}, rho={rho:+.3f})")
        print(f"{'='*70}")

        print(f"  High PC{pc_idx+1}:")
        for idx in top_idx:
            smi = il_smiles[idx]
            matches = df.loc[df["IL_SMILES"] == smi, "IL_name"]
            name = matches.iloc[0] if len(matches) > 0 else "?"
            val = il_to_val.get(smi, float("nan"))
            print(f"    score={pc_scores[idx]:>6.2f}  mean_val={val:>7.2f}  {name[:45]}")

        print(f"  Low PC{pc_idx+1}:")
        for idx in bot_idx:
            smi = il_smiles[idx]
            matches = df.loc[df["IL_SMILES"] == smi, "IL_name"]
            name = matches.iloc[0] if len(matches) > 0 else "?"
            val = il_to_val.get(smi, float("nan"))
            print(f"    score={pc_scores[idx]:>6.2f}  mean_val={val:>7.2f}  {name[:45]}")

    # What structural features does PC3 capture? Interpret via top loadings
    # Map fingerprint bits back to substructures using RDKit
    from rdkit import Chem
    from rdkit.Chem import AllChem

    print(f"\n\n{'='*70}")
    print("PC3 loading analysis — top fingerprint bits")
    print(f"{'='*70}")

    pc3_loadings = reducer.components_[2]
    top_bits = np.argsort(np.abs(pc3_loadings))[-15:][::-1]

    # Get bit info from a representative molecule
    example_mol = Chem.MolFromSmiles(il_smiles[0])
    bi = {}
    AllChem.GetMorganFingerprintAsBitVect(example_mol, 2, nBits=2048, bitInfo=bi)

    for bit in top_bits:
        # Count how many molecules have this bit set
        n_set = (fp_scaled[:, bit] != fp_scaled[:, bit].min()).sum()
        print(f"  bit {bit:>4d}: loading={pc3_loadings[bit]:>+.4f}  "
              f"set_in={n_set:>5d}/{len(il_smiles)} molecules")


if __name__ == "__main__":
    main()
