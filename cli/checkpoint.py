"""CLI subcommand for saving and loading surrogate model checkpoints."""

import json

import joblib

from ..optimization.serialization import load_checkpoint, save_checkpoint


def add_checkpoint_command(subparsers):
    """Register the ``checkpoint`` subcommand with the argument parser.

    Args:
        subparsers: The ``argparse`` subparser group returned by
            ``ArgumentParser.add_subparsers()``.
    """
    parser = subparsers.add_parser(
        "checkpoint",
        help="Save or load surrogate model checkpoints",
    )

    sub = parser.add_subparsers(dest="checkpoint_action")

    # Save subcommand (directory format)
    save_parser = sub.add_parser("save", help="Save surrogate checkpoint (directory format)")
    save_parser.add_argument("--model", required=True, help="Path to serialized model")
    save_parser.add_argument("--surrogate-type", required=True, help="Surrogate family identifier")
    save_parser.add_argument("--columns", required=True, help="Path to serialized column list (joblib)")
    save_parser.add_argument("--output-dir", required=True, help="Output checkpoint directory")
    save_parser.add_argument("--scaler", default=None, help="Path to serialized scaler (joblib)")
    save_parser.add_argument("--round", type=int, default=0, help="Current BO round number")

    # Load subcommand
    load_parser = sub.add_parser("load", help="Load and inspect a checkpoint")
    load_parser.add_argument("checkpoint_dir", help="Path to checkpoint directory")

    parser.set_defaults(func=run_checkpoint)


def run_checkpoint(args):
    """Execute the checkpoint save/load action.

    Args:
        args: Parsed CLI arguments.
    """
    action = getattr(args, "checkpoint_action", None)

    if action == "save":
        _run_save(args)
    elif action == "load":
        _run_load(args)
    else:
        print("Usage: lnpbo checkpoint {save|load}")


def _run_save(args):
    """Save using the directory format."""
    model = joblib.load(args.model)
    columns = joblib.load(args.columns)
    scaler = joblib.load(args.scaler) if args.scaler else None

    save_checkpoint(
        checkpoint_dir=args.output_dir,
        model=model,
        surrogate_type=args.surrogate_type,
        feature_columns=columns,
        round_number=args.round,
        scaler=scaler,
    )
    print(f"Checkpoint saved to {args.output_dir}")


def _run_load(args):
    """Load and inspect a checkpoint directory."""
    model, meta, scaler = load_checkpoint(args.checkpoint_dir)
    print(json.dumps(meta, indent=2))
    print(f"\nModel type: {type(model).__name__}")
    if scaler is not None:
        print(f"Scaler type: {type(scaler).__name__}")
    print(f"Feature columns: {len(meta.get('feature_columns', []))}")
