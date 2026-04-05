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
import logging
import time

import numpy as np

logger = logging.getLogger("lnpbo")


def main():
    parser = argparse.ArgumentParser(
        description="LNPBO: Bayesian Optimization Pipeline for LNP Design",
    )
    parser.add_argument(
        "--subset", type=int, default=None, help="Use a random subset of N rows from LNPDB (for fast iteration)"
    )
    parser.add_argument("--batch-size", type=int, default=12, help="Number of formulations to suggest (default: 12)")
    parser.add_argument(
        "--acq-type",
        type=str,
        default="UCB",
        choices=["UCB", "EI", "LogEI"],
        help="Acquisition function type (default: UCB)",
    )
    parser.add_argument(
        "--batch-strategy",
        type=str,
        default="kb",
        choices=["kb", "rkb", "lp", "ts", "qlogei", "greedy"],
        help="Batch strategy for GP surrogate (default: kb)",
    )
    parser.add_argument(
        "--normalize",
        type=str,
        default="copula",
        choices=["copula", "zscore", "none"],
        help="Target normalization (default: copula)",
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
        "--pool",
        type=str,
        default=None,
        help="Path to candidate pool CSV. Defaults to LNPDB minus training data.",
    )
    parser.add_argument(
        "--il-pcs-morgan", type=int, default=5, help="Number of Morgan fingerprint PCs for IL (default: 5)"
    )
    parser.add_argument(
        "--context-features",
        action="store_true",
        help="Include one-hot experimental context (cell type, target, RoA, etc.)",
    )

    args = parser.parse_args()

    np.random.seed(args.seed)

    # -------------------------------------------------------------------------
    # Step 1: Load LNPDB data
    # -------------------------------------------------------------------------
    logger.info("Step 1: Loading LNPDB database")

    t0 = time.time()

    from LNPBO.data.lnpdb_bridge import load_lnpdb_full

    full_dataset = load_lnpdb_full()
    df = full_dataset.df.copy()
    logger.info("  Loaded %s rows, %d ILs", f"{len(df):,}", df['IL_name'].nunique())

    # Optional: filter to a specific study
    if args.study:
        if "Publication_PMID" in df.columns:
            df = df[df["Publication_PMID"] == int(args.study)]
            logger.info("Filtered to PMID %s: %s rows", args.study, f"{len(df):,}")
        else:
            logger.warning("Publication_PMID column not found, skipping study filter")

    # Optional: take a subset for fast iteration
    if args.subset and args.subset < len(df):
        df = df.sample(n=args.subset, random_state=args.seed).reset_index(drop=True)
        df["Formulation_ID"] = range(1, len(df) + 1)
        logger.info("Using random subset: %s rows", f"{len(df):,}")

    from LNPBO.data.dataset import Dataset, encoders_for_feature_type

    dataset = Dataset(df, source="lnpdb", name="LNPDB_pipeline")

    elapsed = time.time() - t0
    logger.debug("  (%.1fs)", elapsed)

    # -------------------------------------------------------------------------
    # Step 2: Encode molecular features
    # -------------------------------------------------------------------------
    logger.info("Step 2: Encoding molecular features")

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

    enc = encoders_for_feature_type(args.feature_type, il_pcs=il_default_pcs, other_pcs=0)
    for role, n in [("HL", hl_pcs), ("CHL", chl_pcs), ("PEG", peg_pcs)]:
        if role in enc:
            enc[role] = {k: n for k in enc[role]}

    logger.info("Encoding: %s, reduction=%s", args.feature_type, args.reduction)

    encoded = dataset.encode_dataset(
        enc,
        encoding_csv_path="pipeline_encodings.csv",
        reduction=args.reduction,
    )

    enc_cols = [c for c in encoded.df.columns if "_pc" in c]
    logger.info("  %s rows, %d encoding columns (%.1fs)", f"{len(encoded.df):,}", len(enc_cols), time.time() - t0)

    # -------------------------------------------------------------------------
    # Step 3: Build FormulationSpace
    # -------------------------------------------------------------------------
    logger.info("Step 3: Building FormulationSpace")

    from LNPBO.space.formulation import FormulationSpace

    space = FormulationSpace.from_dataset(encoded)

    if args.skip_bo:
        print("\n--skip-bo flag set. Stopping before optimization.")
        print("\nPipeline validation successful! All modules loaded and wired up correctly.")
        return

    # -------------------------------------------------------------------------
    # Step 4: Run Optimization
    # -------------------------------------------------------------------------
    logger.info("Step 4: Optimization (surrogate=%s, feature=%s)", args.surrogate, args.feature_type)

    t0 = time.time()

    from LNPBO.cli._pool import build_candidate_pool
    from LNPBO.optimization.optimizer import Optimizer

    candidate_pool = build_candidate_pool(
        encoded, args.surrogate,
        pool_csv=args.pool,
        feature_type=args.feature_type,
        reduction=args.reduction,
    )

    optimizer = Optimizer(
        space=space,
        surrogate_type=args.surrogate,
        acquisition_type=args.acq_type,
        batch_strategy=args.batch_strategy,
        kappa=args.kappa,
        xi=args.xi,
        batch_size=args.batch_size,
        random_seed=args.seed,
        candidate_pool=candidate_pool,
        normalize=args.normalize,
        context_features=args.context_features,
    )

    logger.info("  Suggesting %d formulations...", args.batch_size)

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
