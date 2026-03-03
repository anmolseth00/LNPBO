from ..data.dataset import Dataset


def add_encode_command(subparsers):
    parser = subparsers.add_parser(
        "encode",
        help="Encode raw formulation CSV into numeric features",
    )

    parser.add_argument("--input", required=True,
                        help="Path to raw LNPDB-format CSV")
    parser.add_argument("--output", required=True,
                        help="Path for encoded output CSV")

    # IL encoding options
    parser.add_argument("--IL-n-pcs-morgan", type=int, default=0,
                        help="Number of Morgan fingerprint PCs for IL (default: 0)")
    parser.add_argument("--IL-n-pcs-mordred", type=int, default=0,
                        help="Number of Mordred descriptor PCs for IL (default: 0)")
    parser.add_argument("--IL-n-pcs-lion", type=int, default=0,
                        help="Number of LiON encoding PCs for IL (default: 0). "
                             "Cannot be combined with Morgan/Mordred for IL.")

    # HL encoding options
    parser.add_argument("--HL-n-pcs-morgan", type=int, default=0,
                        help="Number of Morgan fingerprint PCs for HL (default: 0)")
    parser.add_argument("--HL-n-pcs-mordred", type=int, default=0,
                        help="Number of Mordred descriptor PCs for HL (default: 0)")

    # CHL encoding options
    parser.add_argument("--CHL-n-pcs-morgan", type=int, default=0,
                        help="Number of Morgan fingerprint PCs for CHL (default: 0)")
    parser.add_argument("--CHL-n-pcs-mordred", type=int, default=0,
                        help="Number of Mordred descriptor PCs for CHL (default: 0)")

    # PEG encoding options
    parser.add_argument("--PEG-n-pcs-morgan", type=int, default=0,
                        help="Number of Morgan fingerprint PCs for PEG (default: 0)")
    parser.add_argument("--PEG-n-pcs-mordred", type=int, default=0,
                        help="Number of Mordred descriptor PCs for PEG (default: 0)")

    # Output control
    parser.add_argument("--only-encodings", action="store_true",
                        help="Output only the encoding lookup tables, not the full dataset")

    parser.set_defaults(func=run_encode)


def run_encode(args):
    dataset = Dataset.from_lnpdb_csv(args.input)

    # Validate IL encoding constraints upfront
    if args.IL_n_pcs_lion > 0 and (args.IL_n_pcs_morgan > 0 or args.IL_n_pcs_mordred > 0):
        raise ValueError(
            "LiON encoding cannot be combined with Morgan or Mordred for IL. "
            "Use either --IL-n-pcs-lion OR --IL-n-pcs-morgan/--IL-n-pcs-mordred."
        )

    encoded = dataset.encode_dataset(
        IL_n_pcs_morgan=args.IL_n_pcs_morgan,
        IL_n_pcs_mordred=args.IL_n_pcs_mordred,
        IL_n_pcs_lion=args.IL_n_pcs_lion,
        HL_n_pcs_morgan=args.HL_n_pcs_morgan,
        HL_n_pcs_mordred=args.HL_n_pcs_mordred,
        CHL_n_pcs_morgan=args.CHL_n_pcs_morgan,
        CHL_n_pcs_mordred=args.CHL_n_pcs_mordred,
        PEG_n_pcs_morgan=args.PEG_n_pcs_morgan,
        PEG_n_pcs_mordred=args.PEG_n_pcs_mordred,
        encoding_csv_path=args.output,
        only_encodings=args.only_encodings,
    )

    print(f"Encoded dataset written to {args.output}")
    print(f"  Rows: {len(encoded.df)}")
    print(f"  Columns: {len(encoded.df.columns)}")
