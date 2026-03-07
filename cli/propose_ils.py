from __future__ import annotations

from ..generative.il_propose import propose_ionizable_lipids


def add_propose_ils_command(subparsers):
    parser = subparsers.add_parser(
        "propose-ils",
        help="Propose new ionizable lipids with uncertainty-aware scoring",
    )

    parser.add_argument("--dataset", required=True, help="Path to LNPDB-format CSV with IL_SMILES + Experiment_value")
    parser.add_argument("--output", required=True, help="Output CSV path for proposed ILs")
    parser.add_argument(
        "--n-candidates", type=int, default=20000,
        help="Number of candidates to generate (default: 20000)",
    )
    parser.add_argument(
        "--n-output", type=int, default=100,
        help="Number of candidates to output (default: 100)",
    )
    parser.add_argument(
        "--diversity-pool", type=int, default=1000,
        help="Top-N by score for diversity selection (default: 1000)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--max-mutations", type=int, default=2, help="Max SELFIES mutations per candidate (default: 2)")
    parser.add_argument("--lcb-kappa", type=float, default=1.0, help="LCB weight on uncertainty (default: 1.0)")
    parser.add_argument(
        "--lcb-mode",
        choices=["std", "lower"],
        default="std",
        help="LCB mode: 'std' uses mean - kappa*std, 'lower' uses MAPIE lower bound (default: std)",
    )
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel jobs for model fitting (default: 1)")
    parser.add_argument(
        "--confidence-level",
        type=float,
        default=0.68,
        help="MAPIE confidence level for intervals (default: 0.68)",
    )

    parser.add_argument("--il-pcs-count-mfp", type=int, default=5, help="Count-MFP PCs for IL (default: 5)")
    parser.add_argument("--il-pcs-rdkit", type=int, default=5, help="RDKit PCs for IL (default: 5)")
    parser.add_argument(
        "--reduction",
        default="pls",
        choices=["pca", "pls", "none"],
        help="Dimensionality reduction method (default: pls)",
    )

    parser.add_argument(
        "--amine-smarts",
        default="[NX3;H0;!$(N-C=O);!$([nR])]",
        help="SMARTS for tertiary amine filter (default: protonatable tertiary amine)",
    )
    parser.add_argument("--mw-min", type=float, default=None, help="Minimum molecular weight (optional)")
    parser.add_argument("--mw-max", type=float, default=None, help="Maximum molecular weight (optional)")
    parser.add_argument("--logp-min", type=float, default=None, help="Minimum logP (optional)")
    parser.add_argument("--logp-max", type=float, default=None, help="Maximum logP (optional)")
    parser.add_argument("--max-atoms", type=int, default=None, help="Maximum atom count (optional)")
    parser.add_argument("--max-attempts", type=int, default=None, help="Maximum generation attempts (optional)")

    parser.set_defaults(func=run_propose_ils)


def run_propose_ils(args):
    df = propose_ionizable_lipids(
        dataset_path=args.dataset,
        n_candidates=args.n_candidates,
        n_output=args.n_output,
        diversity_pool=args.diversity_pool,
        random_seed=args.seed,
        max_mutations=args.max_mutations,
        lcb_kappa=args.lcb_kappa,
        lcb_mode=args.lcb_mode,
        n_jobs=args.n_jobs,
        confidence_level=args.confidence_level,
        n_pcs_count_mfp=args.il_pcs_count_mfp,
        n_pcs_rdkit=args.il_pcs_rdkit,
        reduction=args.reduction,
        amine_smarts=args.amine_smarts,
        mw_min=args.mw_min,
        mw_max=args.mw_max,
        logp_min=args.logp_min,
        logp_max=args.logp_max,
        max_atoms=args.max_atoms,
        max_attempts=args.max_attempts,
    )

    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df)} candidates to {args.output}")
