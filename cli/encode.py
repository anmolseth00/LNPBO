from ..data.dataset import Dataset
import importlib


def add_encode_command(subparsers):
    parser = subparsers.add_parser(
        "encode",
        help="Encode raw formulation CSV into numeric features",
    )

    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--encoder",
        required=True,
        help="Python path to SMILES encoder function (e.g. mypkg.encoder:encode)",
    )

    parser.set_defaults(func=run_encode)


def run_encode(args):
    module_path, fn_name = args.encoder.split(":")
    module = importlib.import_module(module_path)
    encode_fn = getattr(module, fn_name)

    Dataset.encode_dataset(
        input_csv=args.input,
        output_csv=args.output,
        encode_smiles_fn=encode_fn,
    )
