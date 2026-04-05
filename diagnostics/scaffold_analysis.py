#!/usr/bin/env python3
"""Scaffold-level analysis of LNPBO benchmark predictions.

Part 1: Scaffold clustering of ILs (Murcko for ring-containing, head group for acyclic).
Part 2: Seen vs novel scaffold hit rate across 5 seeds.
Part 3: Partial correlation between prediction error and physicochemical descriptors.
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import QED, Descriptors
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol
from scipy import stats
from xgboost import XGBRegressor

from LNPBO.benchmarks.runner import prepare_benchmark_data
from LNPBO.data.study_utils import load_lnpdb_clean

logger = logging.getLogger("lnpbo")

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
        keys = ["MolLogP", "TPSA", "NumHDonors", "NumHAcceptors", "MolWt", "NumRotatableBonds", "QED"]
        return {k: np.nan for k in keys}
    return {
        "MolLogP": Descriptors.MolLogP(mol),
        "TPSA": Descriptors.TPSA(mol),
        "NumHDonors": Descriptors.NumHDonors(mol),
        "NumHAcceptors": Descriptors.NumHAcceptors(mol),
        "MolWt": Descriptors.MolWt(mol),
        "NumRotatableBonds": Descriptors.NumRotatableBonds(mol),
        "QED": QED.qed(mol),
    }


def part1_scaffold_clustering(df: pd.DataFrame) -> tuple[dict, dict]:
    """Cluster ILs by Murcko scaffold (ring) or head group (acyclic)."""
    logger.info("=" * 70)
    logger.info("PART 1: Scaffold Clustering of ILs")
    logger.info("=" * 70)

    head_col = "IL_head_name" if "IL_head_name" in df.columns else None
    il_df = df[["IL_SMILES"]].copy()
    if head_col:
        il_df["IL_head_name"] = df[head_col]
    else:
        il_df["IL_head_name"] = None

    il_unique = il_df.drop_duplicates(subset=["IL_SMILES"]).copy()
    il_unique["scaffold"] = il_unique.apply(lambda r: compute_scaffold(r["IL_SMILES"], r.get("IL_head_name")), axis=1)

    ring_mask = il_unique["scaffold"].str.startswith("ring:")
    head_mask = il_unique["scaffold"].str.startswith("head:")
    acyclic_other_mask = il_unique["scaffold"] == "acyclic_other"

    n_ring = ring_mask.sum()
    n_head = head_mask.sum()
    n_other = acyclic_other_mask.sum()
    n_ring_scaffolds = il_unique.loc[ring_mask, "scaffold"].nunique()
    n_head_groups = il_unique.loc[head_mask, "scaffold"].nunique()

    logger.info("Unique ILs: %d", len(il_unique))
    logger.info(
        "  Ring-containing: %d (%.1f%%) across %d Murcko scaffolds",
        n_ring, 100 * n_ring / len(il_unique), n_ring_scaffolds,
    )
    logger.info(
        "  Acyclic (head group): %d (%.1f%%) across %d head groups",
        n_head, 100 * n_head / len(il_unique), n_head_groups,
    )
    logger.info("  Acyclic (no head name): %d (%.1f%%)", n_other, 100 * n_other / len(il_unique))

    scaffold_counts = il_unique["scaffold"].value_counts()
    logger.info("Top 10 scaffolds/head groups by IL count:")
    for i, (scaf, cnt) in enumerate(scaffold_counts.head(10).items()):
        logger.info("  %d. %s: %d ILs", i + 1, scaf, cnt)

    smi_to_scaffold = dict(zip(il_unique["IL_SMILES"], il_unique["scaffold"]))
    df_scaffolded = df.copy()
    df_scaffolded["scaffold"] = df_scaffolded["IL_SMILES"].map(smi_to_scaffold)

    scaffold_formulation_counts = df_scaffolded["scaffold"].value_counts()
    logger.info("Top 10 scaffolds by formulation count:")
    for i, (scaf, cnt) in enumerate(scaffold_formulation_counts.head(10).items()):
        n_ils = il_unique[il_unique["scaffold"] == scaf].shape[0]
        logger.info("  %d. %s: %d formulations (%d ILs)", i + 1, scaf, cnt, n_ils)

    mean_by_scaffold = df_scaffolded.groupby("scaffold")["Experiment_value"].agg(["mean", "std", "count"])
    mean_by_scaffold = mean_by_scaffold.sort_values("mean", ascending=False)
    logger.info("Top 10 scaffolds by mean Experiment_value:")
    for i, (scaf, row) in enumerate(mean_by_scaffold.head(10).iterrows()):
        logger.info("  %d. %s: mean=%.3f, std=%.3f, n=%d", i + 1, scaf, row["mean"], row["std"], int(row["count"]))

    result = {
        "n_unique_ils": len(il_unique),
        "n_ring_containing": int(n_ring),
        "n_acyclic_head": int(n_head),
        "n_acyclic_other": int(n_other),
        "n_ring_scaffolds": int(n_ring_scaffolds),
        "n_head_groups": int(n_head_groups),
        "top_scaffolds_by_formulations": {str(k): int(v) for k, v in scaffold_formulation_counts.head(20).items()},
    }
    return result, smi_to_scaffold


def part2_seen_vs_novel(df: pd.DataFrame, smi_to_scaffold: dict) -> dict:
    """Compare hit rate on seen vs novel scaffolds across 5 seeds."""
    logger.info("=" * 70)
    logger.info("PART 2: Seen vs Novel Scaffold Hit Rate")
    logger.info("=" * 70)

    results_per_seed = {}

    for seed in SEEDS:
        logger.info("--- Seed %d ---", seed)

        _encoded, encoded_df, feature_cols, seed_idx, oracle_idx, _top_k_values = prepare_benchmark_data(
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

        logger.info("  Seed scaffolds: %d", len(seed_scaffolds))
        logger.info("  Test scaffolds: %d (%d seen, %d novel)", len(test_scaffolds), len(seen_scaffolds), len(novel_scaffolds))

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

            logger.info("  Top-%d:", top_k)
            logger.info(
                "    Seen scaffolds:  %d test pts, %d actual top, "
                "%d predicted top, %d hits (%.1f%% recall)",
                n_seen, seen_actual_top, seen_predicted, seen_hits, 100 * seen_hit_rate,
            )
            logger.info(
                "    Novel scaffolds: %d test pts, %d actual top, "
                "%d predicted top, %d hits (%.1f%% recall)",
                n_novel, novel_actual_top, novel_predicted, novel_hits, 100 * novel_hit_rate,
            )
            logger.info("    Overall: %d/%d (%.1f%% recall)", total_hits, top_k, 100 * total_hit_rate)

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

    logger.info("--- Aggregated across seeds ---")
    agg = {}
    for top_k in [50, 100]:
        key = f"top_{top_k}"
        seen_recalls = [results_per_seed[str(s)][key]["seen_recall"] for s in SEEDS]
        novel_recalls = [results_per_seed[str(s)][key]["novel_recall"] for s in SEEDS]
        total_recalls = [results_per_seed[str(s)][key]["total_recall"] for s in SEEDS]

        logger.info("  Top-%d recall:", top_k)
        logger.info("    Seen:  %.1f +/- %.1f%%", 100 * np.mean(seen_recalls), 100 * np.std(seen_recalls))
        logger.info("    Novel: %.1f +/- %.1f%%", 100 * np.mean(novel_recalls), 100 * np.std(novel_recalls))
        logger.info("    Total: %.1f +/- %.1f%%", 100 * np.mean(total_recalls), 100 * np.std(total_recalls))

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
    logger.info("=" * 70)
    logger.info("PART 3: Partial Correlation with Physicochemical Descriptors")
    logger.info("=" * 70)

    desc_names = ["MolLogP", "TPSA", "NumHDonors", "NumHAcceptors", "MolWt", "NumRotatableBonds", "QED"]

    all_seed_results = {}

    for seed in SEEDS:
        logger.info("--- Seed %d ---", seed)

        _encoded, encoded_df, feature_cols, seed_idx, oracle_idx, _top_k_values = prepare_benchmark_data(
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
            test_df[d] = test_df["IL_SMILES"].map(lambda s, d=d, _dc=desc_cache: _dc.get(s, {}).get(d, np.nan))

        valid = test_df.dropna(subset=[*desc_names, "pred_error", "scaffold"]).copy()

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
            logger.info(f"  {d:22s}: raw_rho={rho_xy:+.4f}, partial_rho={partial_rho:+.4f}, p={p_val:.4e} {sig}")

        all_seed_results[str(seed)] = seed_corrs

    logger.info("--- Aggregated partial correlations (mean +/- std across 5 seeds) ---")
    agg = {}
    for d in desc_names:
        partials = [
            all_seed_results[str(s)][d]["spearman_partial"]
            for s in SEEDS
            if all_seed_results[str(s)][d]["spearman_partial"] is not None
        ]
        raws = [all_seed_results[str(s)][d]["spearman_raw"] for s in SEEDS]
        if partials:
            mean_p = np.mean(partials)
            std_p = np.std(partials)
            mean_r = np.mean(raws)
            std_r = np.std(raws)
            logger.info(f"  {d:22s}: raw={mean_r:+.4f}+/-{std_r:.4f}, partial={mean_p:+.4f}+/-{std_p:.4f}")
            agg[d] = {
                "raw_mean": float(mean_r),
                "raw_std": float(std_r),
                "partial_mean": float(mean_p),
                "partial_std": float(std_p),
            }
        else:
            logger.info("  %s: no valid partial correlations", d)
            agg[d] = None

    return {"per_seed": all_seed_results, "aggregated": agg}


def main() -> int:
    logger.info("Loading clean LNPDB...")
    df = load_lnpdb_clean(drop_duplicates=False)
    df = df.dropna(subset=["IL_SMILES", "Experiment_value"]).reset_index(drop=True)
    logger.info("  %d formulations, %d unique ILs", len(df), df["IL_SMILES"].nunique())

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
    logger.info("Results saved to %s", out_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
