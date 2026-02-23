from ..data.dataset import Dataset


def add_register_command(subparsers):
    parser = subparsers.add_parser(
        "register",
        help="Register experimental results into dataset",
    )

    parser.add_argument("--dataset", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--output", required=True)

    parser.set_defaults(func=run_register)


def run_register(args):
    dataset = Dataset.from_csv(args.dataset)
    results = Dataset.from_csv(args.results).df

    dataset.add_results(results)
    dataset.to_csv(args.output)
