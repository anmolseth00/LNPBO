from ..data.dataset import Dataset
from ..optimization.optimizer import OptimizationSession
from ..space.formulation import FormulationSpace
from ..optimization.optimizer import Optimizer
import pickle


def add_suggest_command(subparsers):
    parser = subparsers.add_parser(
        "suggest",
        help="Suggest next batch of formulations",
    )

    parser.add_argument("--dataset", required=True)
    parser.add_argument("--space", required=True, help="Pickled FormulationSpace")
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--output", required=True)

    parser.set_defaults(func=run_suggest)


def run_suggest(args):
    dataset = Dataset.from_csv(args.dataset)

    with open(args.space, "rb") as f:
        space = pickle.load(f)

    optimizer = Optimizer(
        space=space,
        batch_size=args.batch_size,
    )

    session = OptimizationSession(
        dataset=dataset,
        space=space,
        optimizer=optimizer,
        start_round=args.round,
    )

    batch = session.suggest_next_batch()
    batch.to_csv(args.output, index=False)
