#!/usr/bin/env python3
"""
LNPBO End-to-End Pipeline
==========================
Loads LNPDB data, encodes molecular features, builds formulation space,
and runs optimization to suggest new LNP formulations.

Default configuration: LANTERN features (count Morgan FP + RDKit descriptors),
PLS reduction, XGBoost greedy discrete scoring.

Usage:
    python pipeline.py                          # Full pipeline with defaults
    python pipeline.py --subset 500             # Use 500-row subset for fast iteration
    python pipeline.py --batch-size 12          # Suggest 12 formulations
    python pipeline.py --surrogate gp --feature-type mfp --reduction pca  # GP path
    python pipeline.py --feature-type lantern --reduction pls             # LANTERN + PLS

Requires the LNPDB repo cloned as a sibling directory or symlinked at data/LNPDB_repo.
"""

import argparse
import sys
import time
from pathlib import Path

# Ensure the project is importable as a package
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT.parent))

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="LNPBO: Bayesian Optimization Pipeline for LNP Design")
    parser.add_argument(
        "--subset", type=int, default=None, help="Use a random subset of N rows from LNPDB (for fast iteration)"
    )
    parser.add_argument("--batch-size", type=int, default=12, help="Number of formulations to suggest (default: 12)")
    parser.add_argument(
        "--acq-type",
        type=str,
        default="UCB",
        choices=["UCB", "EI", "LogEI", "LP_UCB", "LP_EI", "LP_LogEI"],
        help="Acquisition function type for GP surrogate (default: UCB)",
    )
    parser.add_argument("--kappa", type=float, default=5.0, help="UCB exploration parameter (default: 5.0)")
    parser.add_argument("--xi", type=float, default=0.01, help="EI exploration parameter (default: 0.01)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--output", type=str, default="pipeline_output.csv", help="Output CSV path for suggested formulations"
    )
    parser.add_argument("--study", type=str, default=None, help="Filter LNPDB to a specific publication PMID")
    parser.add_argument("--skip-bo", action="store_true", help="Skip BO step (just load data, encode, build space)")
    parser.add_argument(
        "--reduction",
        type=str,
        default="pls",
        choices=["pca", "pls", "none"],
        help="Dimensionality reduction for fingerprints (default: pls)",
    )
    parser.add_argument(
        "--surrogate",
        type=str,
        default="xgb",
        choices=["gp", "xgb", "xgb_ucb", "rf_ucb", "rf_ts", "gp_ucb"],
        help="Surrogate model (default: xgb). 'gp' uses continuous BO; others use discrete pool scoring.",
    )
    parser.add_argument(
        "--feature-type",
        type=str,
        default="lantern",
        choices=["mfp", "lantern", "count_mfp", "rdkit"],
        help="Feature type (default: lantern). lantern = count Morgan FP + RDKit descriptors.",
    )
    parser.add_argument(
        "--pool", type=str, default=None,
        help="Path to candidate pool CSV. Defaults to LNPDB minus training data.",
    )
    parser.add_argument(
        "--il-pcs-morgan", type=int, default=5, help="Number of Morgan fingerprint PCs for IL (default: 5)"
    )

    args = parser.parse_args()

    np.random.seed(args.seed)

    # -------------------------------------------------------------------------
    # Step 1: Load LNPDB data
    # -------------------------------------------------------------------------
    print("Step 1: Loading LNPDB database")

    t0 = time.time()

    from LNPBO.data.lnpdb_bridge import load_lnpdb_full

    dataset = load_lnpdb_full()
    df = dataset.df
    print(f"  Loaded {len(df):,} rows, {df['IL_name'].nunique()} ILs")

    # Optional: filter to a specific study
    if args.study:
        if "Publication_PMID" in df.columns:
            df = df[df["Publication_PMID"] == int(args.study)]
            print(f"\nFiltered to PMID {args.study}: {len(df):,} rows")
        else:
            print("\nWarning: Publication_PMID column not found, skipping study filter")

    # Optional: take a subset for fast iteration
    if args.subset and args.subset < len(df):
        df = df.sample(n=args.subset, random_state=args.seed).reset_index(drop=True)
        df["Formulation_ID"] = range(1, len(df) + 1)
        print(f"\nUsing random subset: {len(df):,} rows")

    # Track training LNP_IDs for pool exclusion
    training_lnp_ids = set(df["LNP_ID"].dropna()) if "LNP_ID" in df.columns else set()

    from LNPBO.data.dataset import Dataset, encode_kwargs_for_feature_type

    dataset = Dataset(df, source="lnpdb", name="LNPDB_pipeline")

    elapsed = time.time() - t0
    print(f"  ({elapsed:.1f}s)")

    # -------------------------------------------------------------------------
    # Step 2: Encode molecular features
    # -------------------------------------------------------------------------
    print("Step 2: Encoding molecular features")

    t0 = time.time()

    # Only encode variable components with >1 unique SMILES
    def _has_variable_smiles(role):
        return (
            df[f"{role}_name"].nunique() > 1 and f"{role}_SMILES" in df.columns and df[f"{role}_SMILES"].nunique() > 1
        )

    il_default_pcs = args.il_pcs_morgan if df["IL_name"].nunique() > 1 else 0
    hl_pcs = 3 if _has_variable_smiles("HL") else 0
    chl_pcs = 3 if _has_variable_smiles("CHL") else 0
    peg_pcs = 3 if _has_variable_smiles("PEG") else 0

    # Zero out PCs for non-variable roles
    encode_kwargs = encode_kwargs_for_feature_type(args.feature_type, il_pcs=il_default_pcs, other_pcs=0)
    for role, n in [("HL", hl_pcs), ("CHL", chl_pcs), ("PEG", peg_pcs)]:
        for key in list(encode_kwargs):
            if key.startswith(f"{role}_"):
                encode_kwargs[key] = n

    print(f"\nEncoding: {args.feature_type}, reduction={args.reduction}")

    encoded = dataset.encode_dataset(
        **encode_kwargs,
        encoding_csv_path="pipeline_encodings.csv",
        reduction=args.reduction,
    )

    enc_cols = [c for c in encoded.df.columns if "_pc" in c]
    print(f"  {len(encoded.df):,} rows, {len(enc_cols)} encoding columns ({time.time() - t0:.1f}s)")

    # -------------------------------------------------------------------------
    # Step 3: Build FormulationSpace
    # -------------------------------------------------------------------------
    print("Step 3: Building FormulationSpace")

    from LNPBO.space.formulation import FormulationSpace

    space = FormulationSpace.from_dataset(encoded)

    if args.skip_bo:
        print("\n--skip-bo flag set. Stopping before optimization.")
        print("\nPipeline validation successful! All modules loaded and wired up correctly.")
        return

    # -------------------------------------------------------------------------
    # Step 4: Run Optimization
    # -------------------------------------------------------------------------
    print(f"Step 4: Optimization (surrogate={args.surrogate}, feature={args.feature_type})")

    t0 = time.time()

    from LNPBO.optimization.optimizer import Optimizer

    # Build candidate pool for discrete surrogates
    candidate_pool = None
    if args.surrogate != "gp":
        if args.pool:
            import pandas as pd
            pool_df = pd.read_csv(args.pool)
            pool_dataset = Dataset(pool_df, source="lnpdb", name="candidate_pool")
            pool_encoded = pool_dataset.encode_dataset(
                **encode_kwargs, reduction=args.reduction,
                fitted_transformers_in=encoded.fitted_transformers,
            )
            candidate_pool = pool_encoded.df
        else:
            # Load full LNPDB as candidate pool, exclude training rows
            print("\n  Building candidate pool from full LNPDB...")
            full_dataset = load_lnpdb_full()
            if training_lnp_ids and "LNP_ID" in full_dataset.df.columns:
                pool_rows = full_dataset.df[~full_dataset.df["LNP_ID"].isin(training_lnp_ids)]
            else:
                pool_rows = full_dataset.df

            if pool_rows.empty:
                raise ValueError(
                    "No candidate pool available: all LNPDB formulations are in training. "
                    "Use --subset to reserve a portion for the candidate pool, "
                    "or --pool to provide a separate candidate pool CSV."
                )

            pool_dataset = Dataset(
                pool_rows.reset_index(drop=True),
                source="lnpdb", name="candidate_pool",
            )
            pool_encoded = pool_dataset.encode_dataset(
                **encode_kwargs, reduction=args.reduction,
                fitted_transformers_in=encoded.fitted_transformers,
            )
            candidate_pool = pool_encoded.df
            # Assign distinct Formulation_IDs that don't overlap with training
            max_train_id = int(encoded.df["Formulation_ID"].max()) if "Formulation_ID" in encoded.df.columns else 0
            candidate_pool = candidate_pool.copy()
            candidate_pool["Formulation_ID"] = range(max_train_id + 1, max_train_id + 1 + len(candidate_pool))
            print(f"  Candidate pool: {len(candidate_pool):,} formulations")

    optimizer = Optimizer(
        space=space,
        type=args.acq_type,
        kappa=args.kappa,
        xi=args.xi,
        batch_size=args.batch_size,
        random_seed=args.seed,
        surrogate=args.surrogate,
        candidate_pool=candidate_pool,
    )

    print(f"  Suggesting {args.batch_size} formulations...")

    suggestions = optimizer.suggest(output_csv=args.output)

    elapsed = time.time() - t0

    # Show only the new suggestions (last batch_size rows)
    new_rows = suggestions.tail(args.batch_size)

    display_cols = ["Formulation_ID", "Round"]
    for role in ["IL", "HL", "CHL", "PEG"]:
        if f"{role}_name" in new_rows.columns:
            display_cols.append(f"{role}_name")
        if f"{role}_molratio" in new_rows.columns:
            display_cols.append(f"{role}_molratio")
    if "IL_to_nucleicacid_massratio" in new_rows.columns:
        display_cols.append("IL_to_nucleicacid_massratio")
    display_cols.append("Experiment_value")

    display_cols = [c for c in display_cols if c in new_rows.columns]

    print(f"\nSuggested formulations (Round {new_rows['Round'].iloc[0]}):")
    print(new_rows[display_cols].to_string(index=False))
    print(f"\n{args.batch_size} suggestions written to {args.output} ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()
