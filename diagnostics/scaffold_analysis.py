#!/usr/bin/env python3
"""Scaffold-level analysis of LNPBO benchmark predictions.

Part 1: Scaffold clustering of ILs (Murcko for ring-containing, head group for acyclic).
Part 2: Seen vs novel scaffold hit rate across 5 seeds.
Part 3: Partial correlation between prediction error and physicochemical descriptors.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from xgboost import XGBRegressor

from rdkit import Chem
from rdkit.Chem import Descriptors, QED
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from diagnostics.utils import load_lnpdb_clean, encode_lantern_il, lantern_il_feature_cols
from benchmarks.runner import prepare_benchmark_data


SEEDS = [42, 123, 456, 789, 2024]
N_SEED = 500
N_ROUNDS = 15
BATCH_SIZE = 12


def compute_scaffold(smiles: str, head_name: str | None = None) -> str:
    """Return Murcko scaffold SMILES for ring-containing ILs,
    or head-group label for acyclic ILs."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return f"head:{head_name}" if head_name and not pd.isna(head_name) else "acyclic_other"
    scaf = GetScaffoldForMol(mol)
    scaf_smi = Chem.MolToSmiles(scaf)
    if scaf_smi and scaf_smi.strip():
        return f"ring:{scaf_smi}"
    if head_name and not pd.isna(head_name):
        return f"head:{head_name}"
    return "acyclic_other"


def compute_rdkit_descriptors(smiles: str) -> dict:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {k: np.nan for k in ["MolLogP", "TPSA", "NumHDonors", "NumHAcceptors", "MolWt", "NumRotatableBonds", "QED"]}
    return {
        "MolLogP": Descriptors.MolLogP(mol),
        "TPSA": Descriptors.TPSA(mol),
        "NumHDonors": Descriptors.NumHDonors(mol),
        "NumHAcceptors": Descriptors.NumHAcceptors(mol),
        "MolWt": Descriptors.MolWt(mol),
        "NumRotatableBonds": Descriptors.NumRotatableBonds(mol),
        "QED": QED.qed(mol),
    }


def part1_scaffold_clustering(df: pd.DataFrame) -> dict:
    """Cluster ILs by Murcko scaffold (ring) or head group (acyclic)."""
    print("\n" + "=" * 70)
    print("PART 1: Scaffold Clustering of ILs")
    print("=" * 70)

    head_col = "IL_head_name" if "IL_head_name" in df.columns else None
    il_df = df[["IL_SMILES"]].copy()
    if head_col:
        il_df["IL_head_name"] = df[head_col]
    else:
        il_df["IL_head_name"] = None

    il_unique = il_df.drop_duplicates(subset=["IL_SMILES"]).copy()
    il_unique["scaffold"] = il_unique.apply(
        lambda r: compute_scaffold(r["IL_SMILES"], r.get("IL_head_name")), axis=1
    )

    ring_mask = il_unique["scaffold"].str.startswith("ring:")
    head_mask = il_unique["scaffold"].str.startswith("head:")
    acyclic_other_mask = il_unique["scaffold"] == "acyclic_other"

    n_ring = ring_mask.sum()
    n_head = head_mask.sum()
    n_other = acyclic_other_mask.sum()
    n_ring_scaffolds = il_unique.loc[ring_mask, "scaffold"].nunique()
    n_head_groups = il_unique.loc[head_mask, "scaffold"].nunique()

    print(f"\nUnique ILs: {len(il_unique)}")
    print(f"  Ring-containing: {n_ring} ({100*n_ring/len(il_unique):.1f}%) across {n_ring_scaffolds} Murcko scaffolds")
    print(f"  Acyclic (head group): {n_head} ({100*n_head/len(il_unique):.1f}%) across {n_head_groups} head groups")
    print(f"  Acyclic (no head name): {n_other} ({100*n_other/len(il_unique):.1f}%)")

    scaffold_counts = il_unique["scaffold"].value_counts()
    print(f"\nTop 10 scaffolds/head groups by IL count:")
    for i, (scaf, cnt) in enumerate(scaffold_counts.head(10).items()):
        print(f"  {i+1}. {scaf}: {cnt} ILs")

    smi_to_scaffold = dict(zip(il_unique["IL_SMILES"], il_unique["scaffold"]))
    df_scaffolded = df.copy()
    df_scaffolded["scaffold"] = df_scaffolded["IL_SMILES"].map(smi_to_scaffold)

    scaffold_formulation_counts = df_scaffolded["scaffold"].value_counts()
    print(f"\nTop 10 scaffolds by formulation count:")
    for i, (scaf, cnt) in enumerate(scaffold_formulation_counts.head(10).items()):
        n_ils = il_unique[il_unique["scaffold"] == scaf].shape[0]
        print(f"  {i+1}. {scaf}: {cnt} formulations ({n_ils} ILs)")

    mean_by_scaffold = df_scaffolded.groupby("scaffold")["Experiment_value"].agg(["mean", "std", "count"])
    mean_by_scaffold = mean_by_scaffold.sort_values("mean", ascending=False)
    print(f"\nTop 10 scaffolds by mean Experiment_value:")
    for i, (scaf, row) in enumerate(mean_by_scaffold.head(10).iterrows()):
        print(f"  {i+1}. {scaf}: mean={row['mean']:.3f}, std={row['std']:.3f}, n={int(row['count'])}")

    result = {
        "n_unique_ils": int(len(il_unique)),
        "n_ring_containing": int(n_ring),
        "n_acyclic_head": int(n_head),
        "n_acyclic_other": int(n_other),
        "n_ring_scaffolds": int(n_ring_scaffolds),
        "n_head_groups": int(n_head_groups),
        "top_scaffolds_by_formulations": {
            str(k): int(v) for k, v in scaffold_formulation_counts.head(20).items()
        },
    }
    return result, smi_to_scaffold


def part2_seen_vs_novel(df: pd.DataFrame, smi_to_scaffold: dict) -> dict:
    """Compare hit rate on seen vs novel scaffolds across 5 seeds."""
    print("\n" + "=" * 70)
    print("PART 2: Seen vs Novel Scaffold Hit Rate")
    print("=" * 70)

    results_per_seed = {}

    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")

        encoded, encoded_df, feature_cols, seed_idx, oracle_idx, top_k_values = prepare_benchmark_data(
            n_seed=N_SEED,
            random_seed=seed,
            reduction="pca",
            feature_type="lantern_il_only",
            data_df=df,
        )

        encoded_df["scaffold"] = encoded_df["IL_SMILES"].map(smi_to_scaffold)

        seed_scaffolds = set(encoded_df.loc[seed_idx, "scaffold"].dropna().unique())
        test_scaffolds = set(encoded_df.loc[oracle_idx, "scaffold"].dropna().unique())

        novel_scaffolds = test_scaffolds - seed_scaffolds
        seen_scaffolds = test_scaffolds & seed_scaffolds

        print(f"  Seed scaffolds: {len(seed_scaffolds)}")
        print(f"  Test scaffolds: {len(test_scaffolds)} ({len(seen_scaffolds)} seen, {len(novel_scaffolds)} novel)")

        test_df = encoded_df.loc[oracle_idx].copy()
        test_df["is_seen"] = test_df["scaffold"].isin(seed_scaffolds)

        X_train = encoded_df.loc[seed_idx, feature_cols].values
        y_train = encoded_df.loc[seed_idx, "Experiment_value"].values
        X_test = test_df[feature_cols].values
        y_test = test_df["Experiment_value"].values

        model = XGBRegressor(n_estimators=100, max_depth=6, random_state=seed)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        test_df["y_pred"] = y_pred
        test_df["y_true"] = y_test

        seed_result = {}
        for top_k in [50, 100]:
            actual_top_k_threshold = np.sort(y_test)[-top_k]
            is_actual_top = y_test >= actual_top_k_threshold

            pred_top_k_idx = np.argsort(y_pred)[-top_k:]
            is_pred_top = np.zeros(len(y_test), dtype=bool)
            is_pred_top[pred_top_k_idx] = True

            seen_mask = test_df["is_seen"].values
            novel_mask = ~seen_mask

            n_seen = seen_mask.sum()
            n_novel = novel_mask.sum()

            seen_actual_top = is_actual_top[seen_mask].sum()
            novel_actual_top = is_actual_top[novel_mask].sum()

            seen_hits = (is_pred_top & is_actual_top & seen_mask).sum()
            novel_hits = (is_pred_top & is_actual_top & novel_mask).sum()
            total_hits = (is_pred_top & is_actual_top).sum()

            seen_predicted = is_pred_top[seen_mask].sum()
            novel_predicted = is_pred_top[novel_mask].sum()

            seen_hit_rate = seen_hits / max(seen_actual_top, 1)
            novel_hit_rate = novel_hits / max(novel_actual_top, 1)
            total_hit_rate = total_hits / top_k

            print(f"  Top-{top_k}:")
            print(f"    Seen scaffolds:  {n_seen} test pts, {seen_actual_top} actual top, "
                  f"{seen_predicted} predicted top, {seen_hits} hits ({100*seen_hit_rate:.1f}% recall)")
            print(f"    Novel scaffolds: {n_novel} test pts, {novel_actual_top} actual top, "
                  f"{novel_predicted} predicted top, {novel_hits} hits ({100*novel_hit_rate:.1f}% recall)")
            print(f"    Overall: {total_hits}/{top_k} ({100*total_hit_rate:.1f}% recall)")

            seed_result[f"top_{top_k}"] = {
                "n_seen_test": int(n_seen),
                "n_novel_test": int(n_novel),
                "seen_actual_top": int(seen_actual_top),
                "novel_actual_top": int(novel_actual_top),
                "seen_predicted_top": int(seen_predicted),
                "novel_predicted_top": int(novel_predicted),
                "seen_hits": int(seen_hits),
                "novel_hits": int(novel_hits),
                "seen_recall": float(seen_hit_rate),
                "novel_recall": float(novel_hit_rate),
                "total_recall": float(total_hit_rate),
            }

        results_per_seed[str(seed)] = seed_result

    print("\n--- Aggregated across seeds ---")
    agg = {}
    for top_k in [50, 100]:
        key = f"top_{top_k}"
        seen_recalls = [results_per_seed[str(s)][key]["seen_recall"] for s in SEEDS]
        novel_recalls = [results_per_seed[str(s)][key]["novel_recall"] for s in SEEDS]
        total_recalls = [results_per_seed[str(s)][key]["total_recall"] for s in SEEDS]

        print(f"  Top-{top_k} recall:")
        print(f"    Seen:  {100*np.mean(seen_recalls):.1f} +/- {100*np.std(seen_recalls):.1f}%")
        print(f"    Novel: {100*np.mean(novel_recalls):.1f} +/- {100*np.std(novel_recalls):.1f}%")
        print(f"    Total: {100*np.mean(total_recalls):.1f} +/- {100*np.std(total_recalls):.1f}%")

        agg[key] = {
            "seen_recall_mean": float(np.mean(seen_recalls)),
            "seen_recall_std": float(np.std(seen_recalls)),
            "novel_recall_mean": float(np.mean(novel_recalls)),
            "novel_recall_std": float(np.std(novel_recalls)),
            "total_recall_mean": float(np.mean(total_recalls)),
            "total_recall_std": float(np.std(total_recalls)),
        }

    return {"per_seed": results_per_seed, "aggregated": agg}


def part3_partial_correlation(df: pd.DataFrame, smi_to_scaffold: dict) -> dict:
    """Partial correlation between XGB prediction error and physicochemical descriptors."""
    print("\n" + "=" * 70)
    print("PART 3: Partial Correlation with Physicochemical Descriptors")
    print("=" * 70)

    desc_names = ["MolLogP", "TPSA", "NumHDonors", "NumHAcceptors", "MolWt", "NumRotatableBonds", "QED"]

    all_seed_results = {}

    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")

        encoded, encoded_df, feature_cols, seed_idx, oracle_idx, top_k_values = prepare_benchmark_data(
            n_seed=N_SEED,
            random_seed=seed,
            reduction="pca",
            feature_type="lantern_il_only",
            data_df=df,
        )

        encoded_df["scaffold"] = encoded_df["IL_SMILES"].map(smi_to_scaffold)

        X_train = encoded_df.loc[seed_idx, feature_cols].values
        y_train = encoded_df.loc[seed_idx, "Experiment_value"].values
        X_test = encoded_df.loc[oracle_idx, feature_cols].values
        y_test = encoded_df.loc[oracle_idx, "Experiment_value"].values

        model = XGBRegressor(n_estimators=100, max_depth=6, random_state=seed)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        test_df = encoded_df.loc[oracle_idx].copy()
        test_df["pred_error"] = np.abs(y_test - y_pred)

        smiles_unique = test_df["IL_SMILES"].unique()
        desc_cache = {}
        for smi in smiles_unique:
            desc_cache[smi] = compute_rdkit_descriptors(smi)

        for d in desc_names:
            test_df[d] = test_df["IL_SMILES"].map(lambda s, d=d: desc_cache.get(s, {}).get(d, np.nan))

        valid = test_df.dropna(subset=desc_names + ["pred_error", "scaffold"]).copy()

        scaffold_codes, _ = pd.factorize(valid["scaffold"])
        valid["scaffold_code"] = scaffold_codes

        seed_corrs = {}
        for d in desc_names:
            x = valid[d].values
            y = valid["pred_error"].values
            z = valid["scaffold_code"].values

            rho_xy, _ = stats.spearmanr(x, y)

            rho_xz, _ = stats.spearmanr(x, z)
            rho_yz, _ = stats.spearmanr(y, z)

            denom = np.sqrt((1 - rho_xz**2) * (1 - rho_yz**2))
            if denom > 1e-10:
                partial_rho = (rho_xy - rho_xz * rho_yz) / denom
            else:
                partial_rho = np.nan

            n = len(valid)
            if not np.isnan(partial_rho) and n > 3:
                t_stat = partial_rho * np.sqrt((n - 3) / (1 - partial_rho**2 + 1e-12))
                p_val = 2 * stats.t.sf(np.abs(t_stat), df=n - 3)
            else:
                p_val = np.nan

            seed_corrs[d] = {
                "spearman_raw": float(rho_xy),
                "spearman_partial": float(partial_rho) if not np.isnan(partial_rho) else None,
                "p_value": float(p_val) if not np.isnan(p_val) else None,
                "n": int(n),
            }
            sig = "*" if (p_val is not None and not np.isnan(p_val) and p_val < 0.05) else ""
            print(f"  {d:22s}: raw_rho={rho_xy:+.4f}, partial_rho={partial_rho:+.4f}, p={p_val:.4e} {sig}")

        all_seed_results[str(seed)] = seed_corrs

    print("\n--- Aggregated partial correlations (mean +/- std across 5 seeds) ---")
    agg = {}
    for d in desc_names:
        partials = [all_seed_results[str(s)][d]["spearman_partial"] for s in SEEDS
                     if all_seed_results[str(s)][d]["spearman_partial"] is not None]
        raws = [all_seed_results[str(s)][d]["spearman_raw"] for s in SEEDS]
        if partials:
            mean_p = np.mean(partials)
            std_p = np.std(partials)
            mean_r = np.mean(raws)
            std_r = np.std(raws)
            print(f"  {d:22s}: raw={mean_r:+.4f}+/-{std_r:.4f}, partial={mean_p:+.4f}+/-{std_p:.4f}")
            agg[d] = {
                "raw_mean": float(mean_r),
                "raw_std": float(std_r),
                "partial_mean": float(mean_p),
                "partial_std": float(std_p),
            }
        else:
            print(f"  {d:22s}: no valid partial correlations")
            agg[d] = None

    return {"per_seed": all_seed_results, "aggregated": agg}


def main() -> int:
    print("Loading clean LNPDB...")
    df = load_lnpdb_clean(drop_duplicates=False)
    df = df.dropna(subset=["IL_SMILES", "Experiment_value"]).reset_index(drop=True)
    print(f"  {len(df)} formulations, {df['IL_SMILES'].nunique()} unique ILs")

    part1_result, smi_to_scaffold = part1_scaffold_clustering(df)
    part2_result = part2_seen_vs_novel(df, smi_to_scaffold)
    part3_result = part3_partial_correlation(df, smi_to_scaffold)

    results = {
        "part1_scaffold_clustering": part1_result,
        "part2_seen_vs_novel": part2_result,
        "part3_partial_correlation": part3_result,
    }

    out_path = Path(__file__).resolve().parent / "scaffold_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
