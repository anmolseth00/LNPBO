from ..data.dataset import Dataset, encoders_for_feature_type
from ..optimization.optimizer import Optimizer
from ..space.formulation import FormulationSpace


def add_suggest_command(subparsers):
    parser = subparsers.add_parser(
        "suggest",
        help="Suggest next batch of formulations",
    )

    parser.add_argument("--dataset", required=True, help="Path to encoded dataset CSV (from 'encode' step)")
    parser.add_argument("--batch-size", type=int, default=12, help="Number of formulations to suggest (default: 12)")
    parser.add_argument(
        "--acq-type",
        default="UCB",
        choices=["UCB", "EI", "LogEI", "LP_UCB", "LP_EI", "LP_LogEI"],
        help="Acquisition function type for GP surrogate (default: UCB)",
    )
    parser.add_argument("--kappa", type=float, default=5.0, help="UCB exploration parameter (default: 5.0)")
    parser.add_argument("--xi", type=float, default=0.01, help="EI/PI exploration parameter (default: 0.01)")
    parser.add_argument("--seed", type=int, default=1, help="Random seed (default: 1)")
    parser.add_argument("--output", required=True, help="Output CSV path for suggested formulations")
    parser.add_argument(
        "--reduction",
        default="pls",
        choices=["pca", "pls", "none"],
        help="Dimensionality reduction: pca, pls (default), or none",
    )
    parser.add_argument(
        "--surrogate",
        default="xgb",
        choices=["gp", "xgb", "xgb_ucb", "rf_ucb", "rf_ts", "gp_ucb"],
        help="Surrogate model (default: xgb). 'gp' uses continuous BO; others use discrete pool scoring.",
    )
    parser.add_argument(
        "--feature-type",
        default="lantern",
        choices=["mfp", "lantern", "count_mfp", "rdkit"],
        help="Feature type (default: lantern). lantern = count Morgan FP + RDKit descriptors.",
    )
    parser.add_argument(
        "--pool", default=None,
        help="Path to candidate pool CSV for discrete surrogates. If not set, uses the dataset itself.",
    )

    parser.set_defaults(func=run_suggest)


def run_suggest(args):
    dataset = Dataset.from_lnpdb_csv(args.dataset)

    enc = encoders_for_feature_type(args.feature_type)

    encoded = dataset.encode_dataset(enc, reduction=args.reduction)

    space = FormulationSpace.from_dataset(encoded)

    # Build candidate pool for discrete surrogates
    candidate_pool = None
    if args.surrogate != "gp":
        if args.pool:
            import pandas as pd
            pool_df = pd.read_csv(args.pool)
            pool_dataset = Dataset(pool_df, source="lnpdb", name="candidate_pool")
            pool_encoded = pool_dataset.encode_dataset(
                enc,
                reduction=args.reduction,
                fitted_transformers_in=encoded.fitted_transformers,
            )
            candidate_pool = pool_encoded.df
        else:
            candidate_pool = encoded.df

    optimizer = Optimizer(
        space=space,
        type=args.acq_type,
        kappa=args.kappa,
        xi=args.xi,
        random_seed=args.seed,
        batch_size=args.batch_size,
        surrogate=args.surrogate,
        candidate_pool=candidate_pool,
    )

    batch = optimizer.suggest(output_csv=args.output)
    print(f"Suggested {len(batch)} formulations.")
