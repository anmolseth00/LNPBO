from ..optimization.serialization import save_surrogate
import pickle


def add_checkpoint_command(subparsers):
    parser = subparsers.add_parser(
        "checkpoint",
        help="Save GP surrogate model",
    )

    parser.add_argument("--gp", required=True)
    parser.add_argument("--scaler", required=True)
    parser.add_argument("--columns", required=True)
    parser.add_argument("--output", required=True)

    parser.set_defaults(func=run_checkpoint)


def run_checkpoint(args):
    with open(args.gp, "rb") as f:
        gp = pickle.load(f)

    with open(args.scaler, "rb") as f:
        scaler = pickle.load(f)

    with open(args.columns, "rb") as f:
        columns = pickle.load(f)

    save_surrogate(
        path=args.output,
        gp_model=gp,
        scaler=scaler,
        columns=columns,
        metadata={},
    )
