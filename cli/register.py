import pandas as pd
from ..data.dataset import Dataset


def add_register_command(subparsers):
    parser = subparsers.add_parser(
        "register",
        help="Register experimental results into dataset",
    )

    parser.add_argument("--dataset", required=True,
                        help="Path to current dataset CSV")
    parser.add_argument("--results", required=True,
                        help="Path to CSV with completed experimental results "
                             "(must contain Formulation_ID, Round, Experiment_value)")
    parser.add_argument("--output", required=True,
                        help="Output path for updated dataset CSV")

    parser.set_defaults(func=run_register)


def run_register(args):
    dataset = Dataset.from_lnpdb_csv(args.dataset)

    results_df = pd.read_csv(args.results)

    required = {"Formulation_ID", "Round", "Experiment_value"}
    missing = required - set(results_df.columns)
    if missing:
        raise ValueError(
            f"Results CSV is missing required columns: {missing}. "
            f"Expected: {sorted(required)}"
        )

    if results_df["Experiment_value"].isna().any():
        n_missing = results_df["Experiment_value"].isna().sum()
        raise ValueError(
            f"{n_missing} row(s) in results CSV have missing Experiment_value. "
            "All results must include measured values."
        )

    updated = dataset.append_suggestions(results_df)
    updated.to_csv(args.output)
    print(f"Registered {len(results_df)} results. Updated dataset written to {args.output}")
