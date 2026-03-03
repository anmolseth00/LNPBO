#!/usr/bin/env python3
"""
LNPBO End-to-End Pipeline
==========================
Loads LNPDB data, encodes molecular features, builds formulation space,
and runs Bayesian optimization to suggest new LNP formulations.

Usage:
    python pipeline.py                          # Full pipeline with defaults
    python pipeline.py --subset 500             # Use 500-row subset for fast iteration
    python pipeline.py --batch-size 12          # Suggest 12 formulations
    python pipeline.py --acq-type EI            # Use Expected Improvement
    python pipeline.py --il-pcs-morgan 5        # 5 Morgan fingerprint PCs for IL

Requires the LNPDB repo cloned as a sibling directory or symlinked at data/LNPDB_repo.
"""

import argparse
import sys
import time
from pathlib import Path

# Ensure the project is importable as a package
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT.parent))

# Import the project as a package (LNPBO.data, LNPBO.space, etc.)
PACKAGE_NAME = PROJECT_ROOT.name  # "LNPBO"

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
        help="Acquisition function type (default: UCB)",
    )
    parser.add_argument("--kappa", type=float, default=5.0, help="UCB exploration parameter (default: 5.0)")
    parser.add_argument("--xi", type=float, default=0.01, help="EI exploration parameter (default: 0.01)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--il-pcs-morgan", type=int, default=5, help="Number of Morgan fingerprint PCs for ionizable lipid (default: 5)"
    )
    parser.add_argument(
        "--output", type=str, default="pipeline_output.csv", help="Output CSV path for suggested formulations"
    )
    parser.add_argument("--study", type=str, default=None, help="Filter LNPDB to a specific publication PMID")
    parser.add_argument("--skip-bo", action="store_true", help="Skip BO step (just load data, encode, build space)")
    parser.add_argument(
        "--reduction",
        type=str,
        default="pca",
        choices=["pca", "pls"],
        help="Dimensionality reduction for fingerprints: pca (default) or pls (target-aware)",
    )

    args = parser.parse_args()

    np.random.seed(args.seed)

    # -------------------------------------------------------------------------
    # Step 1: Load LNPDB data
    # -------------------------------------------------------------------------
    print("=" * 70)
    print("STEP 1: Loading LNPDB database")
    print("=" * 70)

    t0 = time.time()

    from LNPBO.data.lnpdb_bridge import list_available_datasets, load_lnpdb_full

    # Show what's available
    available = list_available_datasets()
    print("\nAvailable data:")
    for key, val in available.items():
        if not isinstance(val, dict):
            continue
        if "rows" in val:
            print(f"  {key}: {val['rows']:,} rows")  # type: ignore[index]
        elif key == "single_split":
            parts = ", ".join(f"{k}={v:,}" for k, v in val.items())
            print(f"  {key}: {parts}")
        elif key == "heldout":
            for hname, hinfo in val.items():
                if isinstance(hinfo, dict):
                    print(
                        f"  heldout/{hname}: {hinfo.get('all_data_rows', '?')} rows, "  # type: ignore[call-overload]
                        f"heldout={hinfo.get('heldout_data_rows', '?')} rows"  # type: ignore[call-overload]
                    )

    # Load the full database
    dataset = load_lnpdb_full()
    df = dataset.df
    print(f"\nLoaded LNPDB: {len(df):,} rows x {len(df.columns)} columns")
    print(f"  Unique ILs:   {df['IL_name'].nunique()}")
    print(f"  Unique HLs:   {df['HL_name'].nunique()}")
    print(f"  Unique CHLs:  {df['CHL_name'].nunique()}")
    print(f"  Unique PEGs:  {df['PEG_name'].nunique()}")
    print(f"  Experiment_value range: [{df['Experiment_value'].min():.2f}, {df['Experiment_value'].max():.2f}]")

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

    # Rebuild dataset from filtered df
    from LNPBO.data.dataset import Dataset

    dataset = Dataset(df, source="lnpdb", name="LNPDB_pipeline")

    elapsed = time.time() - t0
    print(f"\nData loading complete ({elapsed:.1f}s)")

    # -------------------------------------------------------------------------
    # Step 2: Encode molecular features
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 2: Encoding molecular features")
    print("=" * 70)

    t0 = time.time()

    # Check what's variable
    print("\nVariable components:")
    for role in ["IL", "HL", "CHL", "PEG"]:
        nunique = df[f"{role}_name"].nunique()
        is_var = nunique > 1
        print(f"  {role}: {nunique} unique values {'(VARIABLE)' if is_var else '(FIXED)'}")

    print("\nVariable molar ratios:")
    for role in ["IL", "HL", "CHL", "PEG"]:
        col = f"{role}_molratio"
        nunique = df[col].nunique()
        is_var = nunique > 1
        rng = f"[{df[col].min():.4f}, {df[col].max():.4f}]"
        print(f"  {role}: {nunique} unique values {rng} {'(VARIABLE)' if is_var else '(FIXED)'}")

    il_mrna = df["IL_to_nucleicacid_massratio"]
    print(f"\nIL:mRNA mass ratio: {il_mrna.nunique()} unique values [{il_mrna.min():.1f}, {il_mrna.max():.1f}]")

    # Determine encoding PCs based on what's variable
    il_n_pcs_morgan = args.il_pcs_morgan if df["IL_name"].nunique() > 1 else 0

    # Only encode variable components with >1 unique SMILES
    def _has_variable_smiles(role):
        return (
            df[f"{role}_name"].nunique() > 1 and f"{role}_SMILES" in df.columns and df[f"{role}_SMILES"].nunique() > 1
        )

    hl_n_pcs = 3 if _has_variable_smiles("HL") else 0
    chl_n_pcs = 3 if _has_variable_smiles("CHL") else 0
    peg_n_pcs = 3 if _has_variable_smiles("PEG") else 0

    print(f"\nEncoding plan (reduction={args.reduction}):")
    print(f"  IL:  {il_n_pcs_morgan} Morgan components")
    print(f"  HL:  {hl_n_pcs} Morgan components")
    print(f"  CHL: {chl_n_pcs} Morgan components")
    print(f"  PEG: {peg_n_pcs} Morgan components")

    encoded = dataset.encode_dataset(
        IL_n_pcs_morgan=il_n_pcs_morgan,
        HL_n_pcs_morgan=hl_n_pcs,
        CHL_n_pcs_morgan=chl_n_pcs,
        PEG_n_pcs_morgan=peg_n_pcs,
        encoding_csv_path="pipeline_encodings.csv",
        reduction=args.reduction,
    )

    print(f"\nEncoded dataset: {len(encoded.df):,} rows x {len(encoded.df.columns)} columns")
    print(f"  Metadata keys: {list(encoded.metadata.keys())}")
    print(f"  Variable components: {encoded.metadata['variable_components']}")
    print(f"  Variable molratios: {encoded.metadata['variable_molratios']}")
    print(f"  Fitted transformers: {list(encoded.fitted_transformers.keys())}")

    # Show encoding columns
    enc_cols = [c for c in encoded.df.columns if "_pc" in c]
    if enc_cols:
        print(f"  Encoding columns: {enc_cols}")

    elapsed = time.time() - t0
    print(f"\nEncoding complete ({elapsed:.1f}s)")

    # -------------------------------------------------------------------------
    # Step 3: Build FormulationSpace
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 3: Building FormulationSpace")
    print("=" * 70)

    t0 = time.time()

    from LNPBO.space.formulation import FormulationSpace

    space = FormulationSpace.from_dataset(encoded)

    print("\nFormulationSpace created:")
    print(f"  Name: {space.name}")
    print("  Components:")
    for role in FormulationSpace.ROLES:
        n = len(space.components[role])
        print(f"    {role}: {n} unique lipids")
    print("  Molar ratio bounds:")
    for role, bounds in space.molratio_bounds.items():
        print(f"    {role}: [{bounds[0]:.4f}, {bounds[1]:.4f}]")
    il_mrna_preview = space.il_mrna_massratio_values[:5]
    suffix = "..." if len(space.il_mrna_massratio_values) > 5 else ""
    print(f"  IL:mRNA values: {il_mrna_preview}{suffix}")
    print(f"  Fixed values: {space.fixed_values}")
    print(f"  Parameters: {[type(p).__name__ for p in space.parameters]}")

    configs = space.get_configs()
    print("\n  BO config parameters:")
    for p in configs["parameters"]:
        print(f"    {p['name']} ({p['type']}): {len(p['columns'])} columns")

    elapsed = time.time() - t0
    print(f"\nSpace construction complete ({elapsed:.1f}s)")

    if args.skip_bo:
        print("\n--skip-bo flag set. Stopping before Bayesian optimization.")
        print("\nPipeline validation successful! All modules loaded and wired up correctly.")
        return

    # -------------------------------------------------------------------------
    # Step 4: Run Bayesian Optimization
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 4: Running Bayesian Optimization")
    print("=" * 70)

    t0 = time.time()

    from LNPBO.optimization.optimizer import Optimizer

    optimizer = Optimizer(
        space=space,
        type=args.acq_type,
        kappa=args.kappa,
        xi=args.xi,
        batch_size=args.batch_size,
        random_seed=args.seed,
    )

    print("\nOptimizer configured:")
    print(f"  Acquisition: {args.acq_type}")
    print(f"  Batch size:  {args.batch_size}")
    print(f"  Kappa:       {args.kappa}")
    print(f"  Seed:        {args.seed}")
    print(f"\nSuggesting {args.batch_size} new formulations...")

    suggestions = optimizer.suggest(output_csv=args.output)

    elapsed = time.time() - t0

    # -------------------------------------------------------------------------
    # Step 5: Display results
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 5: Results")
    print("=" * 70)

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

    print(f"\nBO complete ({elapsed:.1f}s)")
    print(f"Full results written to: {args.output}")

    # Summary stats
    print("\nSummary:")
    print(f"  Total rows (training + suggestions): {len(suggestions):,}")
    print(f"  New suggestions: {args.batch_size}")
    for role in ["IL", "HL", "CHL", "PEG"]:
        name_col = f"{role}_name"
        if name_col in new_rows.columns:
            unique_suggested = new_rows[name_col].nunique()
            print(f"  Unique {role}s suggested: {unique_suggested}")


if __name__ == "__main__":
    main()
