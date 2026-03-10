from ..data.dataset import Dataset
from ..optimization.optimizer import ACQUISITION_TYPES, ALL_BATCH_STRATEGIES, SURROGATE_TYPES, Optimizer
from ..space.formulation import FormulationSpace


def add_suggest_command(subparsers):
    parser = subparsers.add_parser(
        "suggest",
        help="Suggest next batch of formulations",
    )

    parser.add_argument("--dataset", required=True, help="Path to encoded dataset CSV (from 'encode' step)")
    parser.add_argument("--batch-size", type=int, default=12, help="Number of formulations to suggest (default: 12)")

    parser.add_argument(
        "--surrogate-type",
        default="xgb_ucb",
        choices=sorted(SURROGATE_TYPES),
        help="Surrogate model (default: xgb_ucb).",
    )
    parser.add_argument(
        "--acquisition-type",
        default="UCB",
        choices=sorted(ACQUISITION_TYPES),
        help="Acquisition function for GP surrogates (default: UCB).",
    )
    parser.add_argument(
        "--batch-strategy",
        default="kb",
        choices=sorted(ALL_BATCH_STRATEGIES),
        help="Batch selection strategy (default: kb). GP: kb/rkb/lp/ts/qlogei. Discrete: greedy/ts.",
    )
    parser.add_argument("--kappa", type=float, default=5.0, help="UCB exploration parameter (default: 5.0)")
    parser.add_argument("--xi", type=float, default=0.01, help="EI/LogEI exploration parameter (default: 0.01)")
    parser.add_argument("--seed", type=int, default=1, help="Random seed (default: 1)")
    parser.add_argument("--output", required=True, help="Output CSV path for suggested formulations")
    parser.add_argument(
        "--normalize",
        default="copula",
        choices=["copula", "zscore", "none"],
        help="Target normalization (default: copula).",
    )
    parser.add_argument(
        "--reduction",
        default="pca",
        choices=["pca", "pls", "none"],
        help="Dimensionality reduction: pca (default), pls, or none.",
    )
    parser.add_argument(
        "--feature-type",
        default="lantern",
        choices=["mfp", "count_mfp", "rdkit", "mordred", "unimol", "chemeleon", "lantern"],
        help="Feature type (default: lantern). lantern = count Morgan FP + RDKit descriptors.",
    )
    parser.add_argument(
        "--pool", default=None,
        help="Path to candidate pool CSV. If not set, uses the dataset itself.",
    )

    parser.set_defaults(func=run_suggest)


def run_suggest(args):
    dataset = Dataset.from_lnpdb_csv(args.dataset)

    encoded = dataset.encode_dataset(
        feature_type=args.feature_type, reduction=args.reduction,
    )

    space = FormulationSpace.from_dataset(encoded)

    # Build candidate pool
    candidate_pool = None
    if args.surrogate_type != "gp_sklearn":
        if args.pool:
            import pandas as pd
            pool_df = pd.read_csv(args.pool)
            pool_dataset = Dataset(pool_df, source="lnpdb", name="candidate_pool")
            pool_encoded = pool_dataset.encode_dataset(
                feature_type=args.feature_type,
                reduction=args.reduction,
                fitted_transformers_in=encoded.fitted_transformers,
            )
            candidate_pool = pool_encoded.df
        else:
            candidate_pool = encoded.df

    optimizer = Optimizer(
        space=space,
        surrogate_type=args.surrogate_type,
        acquisition_type=args.acquisition_type,
        batch_strategy=args.batch_strategy,
        kappa=args.kappa,
        xi=args.xi,
        random_seed=args.seed,
        batch_size=args.batch_size,
        candidate_pool=candidate_pool,
        normalize=args.normalize,
    )

    batch = optimizer.suggest(output_csv=args.output)
    print(f"Suggested {len(batch)} formulations.")
