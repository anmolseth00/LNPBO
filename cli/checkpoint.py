import joblib
from ..optimization.serialization import save_surrogate


def add_checkpoint_command(subparsers):
    parser = subparsers.add_parser(
        "checkpoint",
        help="Save GP surrogate model",
    )

    parser.add_argument("--gp", required=True,
                        help="Path to serialized GP model (joblib format)")
    parser.add_argument("--scaler", required=True,
                        help="Path to serialized scaler (joblib format)")
    parser.add_argument("--columns", required=True,
                        help="Path to serialized column list (joblib format)")
    parser.add_argument("--output", required=True,
                        help="Output path for combined checkpoint file")

    parser.set_defaults(func=run_checkpoint)


def run_checkpoint(args):
    gp = joblib.load(args.gp)
    if not hasattr(gp, "predict"):
        raise TypeError(
            f"Loaded GP object ({type(gp).__name__}) does not have a predict method. "
            "Expected a fitted GaussianProcessRegressor or compatible model."
        )

    scaler = joblib.load(args.scaler)
    if not hasattr(scaler, "transform"):
        raise TypeError(
            f"Loaded scaler object ({type(scaler).__name__}) does not have a transform method. "
            "Expected a fitted StandardScaler or compatible transformer."
        )

    columns = joblib.load(args.columns)
    if not isinstance(columns, (list, tuple)):
        raise TypeError(
            f"Loaded columns object is {type(columns).__name__}, expected list or tuple."
        )

    save_surrogate(
        path=args.output,
        gp_model=gp,
        scaler=scaler,
        columns=columns,
        metadata={},
    )

    print(f"Checkpoint saved to {args.output}")
