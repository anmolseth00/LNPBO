import logging

from ..data.dataset import Dataset
from ..optimization.optimizer import ACQUISITION_TYPES, ALL_BATCH_STRATEGIES, SURROGATE_TYPES, Optimizer
from ..space.formulation import FormulationSpace

logger = logging.getLogger("lnpbo")


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
        choices=["mfp", "count_mfp", "rdkit", "mordred", "unimol", "chemeleon", "lion", "agile", "lantern"],
        help="Feature type (default: lantern). lantern = count Morgan FP + RDKit descriptors.",
    )
    parser.add_argument(
        "--pool",
        default=None,
        help="Path to candidate pool CSV. If not set, uses the dataset itself.",
    )
    parser.add_argument(
        "--gp-engine",
        default="botorch",
        choices=["botorch", "sklearn"],
        help="GP backend (default: botorch). Only used when surrogate-type is 'gp'.",
    )
    parser.add_argument(
        "--kernel-type",
        default="matern",
        choices=["matern", "tanimoto", "aitchison", "dkl", "rf", "compositional", "robust"],
        help="GP kernel (default: matern).",
    )
    parser.add_argument(
        "--context-features",
        action="store_true",
        help="Include context features (e.g., cell type) in the surrogate model.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1e-6,
        help="Noise regularization for sklearn GP (default: 1e-6).",
    )
    parser.add_argument(
        "--surrogate-kwargs",
        default=None,
        help="JSON string of extra keyword arguments for the surrogate model.",
    )

    parser.set_defaults(func=run_suggest)


def run_suggest(args):
    import json

    from ._pool import build_candidate_pool

    dataset = Dataset.from_lnpdb_csv(args.dataset)

    encoded = dataset.encode_dataset(
        feature_type=args.feature_type,
        reduction=args.reduction,
    )

    space = FormulationSpace.from_dataset(encoded)

    candidate_pool = build_candidate_pool(
        encoded, args.surrogate_type,
        pool_csv=args.pool,
        feature_type=args.feature_type,
        reduction=args.reduction,
    )

    surrogate_kwargs = None
    if args.surrogate_kwargs:
        surrogate_kwargs = json.loads(args.surrogate_kwargs)

    optimizer = Optimizer(
        space=space,
        surrogate_type=args.surrogate_type,
        gp_engine=args.gp_engine,
        acquisition_type=args.acquisition_type,
        batch_strategy=args.batch_strategy,
        kappa=args.kappa,
        xi=args.xi,
        random_seed=args.seed,
        batch_size=args.batch_size,
        candidate_pool=candidate_pool,
        normalize=args.normalize,
        context_features=args.context_features,
        alpha=args.alpha,
        kernel_type=args.kernel_type,
        surrogate_kwargs=surrogate_kwargs,
    )

    batch = optimizer.suggest(output_csv=args.output)
    logger.info("Suggested %d formulations.", len(batch))
