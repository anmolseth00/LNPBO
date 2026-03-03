from ..data.dataset import Dataset
from ..space.formulation import FormulationSpace
from ..optimization.optimizer import Optimizer


def add_suggest_command(subparsers):
    parser = subparsers.add_parser(
        "suggest",
        help="Suggest next batch of formulations",
    )

    parser.add_argument("--dataset", required=True,
                        help="Path to encoded dataset CSV (from 'encode' step)")
    parser.add_argument("--batch-size", type=int, default=24,
                        help="Number of formulations to suggest (default: 24)")
    parser.add_argument("--acq-type", default="UCB",
                        choices=["UCB", "EI", "LogEI", "LP_UCB", "LP_EI", "LP_LogEI"],
                        help="Acquisition function type (default: UCB)")
    parser.add_argument("--kappa", type=float, default=5.0,
                        help="UCB exploration parameter (default: 5.0)")
    parser.add_argument("--xi", type=float, default=0.01,
                        help="EI/PI exploration parameter (default: 0.01)")
    parser.add_argument("--seed", type=int, default=1,
                        help="Random seed (default: 1)")
    parser.add_argument("--output", required=True,
                        help="Output CSV path for suggested formulations")
    parser.add_argument("--reduction", default="pca",
                        choices=["pca", "pls"],
                        help="Dimensionality reduction: pca (default) or pls (target-aware)")

    parser.set_defaults(func=run_suggest)


def run_suggest(args):
    dataset = Dataset.from_lnpdb_csv(args.dataset)

    # Encode dataset (required before building FormulationSpace).
    # If the CSV already contains encoding columns, encode_dataset
    # will detect single-valued components and skip them gracefully.
    encoded = dataset.encode_dataset(reduction=args.reduction)

    space = FormulationSpace.from_dataset(encoded)

    optimizer = Optimizer(
        space=space,
        type=args.acq_type,
        kappa=args.kappa,
        xi=args.xi,
        random_seed=args.seed,
        batch_size=args.batch_size,
    )

    batch = optimizer.suggest(output_csv=args.output)
    print(f"Suggested {len(batch)} formulations.")
