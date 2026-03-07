from ..data.dataset import Dataset


def add_encode_command(subparsers):
    parser = subparsers.add_parser(
        "encode",
        help="Encode raw formulation CSV into numeric features",
    )

    parser.add_argument("--input", required=True, help="Path to raw LNPDB-format CSV")
    parser.add_argument("--output", required=True, help="Path for encoded output CSV")

    # IL encoding options
    parser.add_argument(
        "--IL-n-pcs-morgan", type=int, default=0, help="Number of Morgan fingerprint PCs for IL (default: 0)"
    )
    parser.add_argument(
        "--IL-n-pcs-mordred", type=int, default=0, help="Number of Mordred descriptor PCs for IL (default: 0)"
    )
    parser.add_argument(
        "--IL-n-pcs-lion",
        type=int,
        default=0,
        help="Number of LiON encoding PCs for IL (default: 0). Cannot be combined with Morgan/Mordred for IL.",
    )
    parser.add_argument(
        "--IL-n-pcs-count-mfp", type=int, default=0, help="Number of count Morgan FP PCs for IL (default: 0)"
    )
    parser.add_argument(
        "--IL-n-pcs-rdkit", type=int, default=0, help="Number of RDKit descriptor PCs for IL (default: 0)"
    )
    parser.add_argument(
        "--IL-n-pcs-unimol", type=int, default=0, help="Number of Uni-Mol embedding PCs for IL (default: 0)"
    )

    # HL encoding options
    parser.add_argument(
        "--HL-n-pcs-morgan", type=int, default=0, help="Number of Morgan fingerprint PCs for HL (default: 0)"
    )
    parser.add_argument(
        "--HL-n-pcs-mordred", type=int, default=0, help="Number of Mordred descriptor PCs for HL (default: 0)"
    )
    parser.add_argument(
        "--HL-n-pcs-count-mfp", type=int, default=0, help="Number of count Morgan FP PCs for HL (default: 0)"
    )
    parser.add_argument(
        "--HL-n-pcs-rdkit", type=int, default=0, help="Number of RDKit descriptor PCs for HL (default: 0)"
    )
    parser.add_argument(
        "--HL-n-pcs-unimol", type=int, default=0, help="Number of Uni-Mol embedding PCs for HL (default: 0)"
    )

    # CHL encoding options
    parser.add_argument(
        "--CHL-n-pcs-morgan", type=int, default=0, help="Number of Morgan fingerprint PCs for CHL (default: 0)"
    )
    parser.add_argument(
        "--CHL-n-pcs-mordred", type=int, default=0, help="Number of Mordred descriptor PCs for CHL (default: 0)"
    )
    parser.add_argument(
        "--CHL-n-pcs-count-mfp", type=int, default=0, help="Number of count Morgan FP PCs for CHL (default: 0)"
    )
    parser.add_argument(
        "--CHL-n-pcs-rdkit", type=int, default=0, help="Number of RDKit descriptor PCs for CHL (default: 0)"
    )
    parser.add_argument(
        "--CHL-n-pcs-unimol", type=int, default=0, help="Number of Uni-Mol embedding PCs for CHL (default: 0)"
    )

    # PEG encoding options
    parser.add_argument(
        "--PEG-n-pcs-morgan", type=int, default=0, help="Number of Morgan fingerprint PCs for PEG (default: 0)"
    )
    parser.add_argument(
        "--PEG-n-pcs-mordred", type=int, default=0, help="Number of Mordred descriptor PCs for PEG (default: 0)"
    )
    parser.add_argument(
        "--PEG-n-pcs-count-mfp", type=int, default=0, help="Number of count Morgan FP PCs for PEG (default: 0)"
    )
    parser.add_argument(
        "--PEG-n-pcs-rdkit", type=int, default=0, help="Number of RDKit descriptor PCs for PEG (default: 0)"
    )
    parser.add_argument(
        "--PEG-n-pcs-unimol", type=int, default=0, help="Number of Uni-Mol embedding PCs for PEG (default: 0)"
    )

    # Reduction
    parser.add_argument(
        "--reduction",
        default="pls",
        choices=["pca", "pls", "none"],
        help="Dimensionality reduction: pca, pls (default), or none",
    )

    # Output control
    parser.add_argument(
        "--only-encodings", action="store_true", help="Output only the encoding lookup tables, not the full dataset"
    )

    parser.set_defaults(func=run_encode)


def run_encode(args):
    dataset = Dataset.from_lnpdb_csv(args.input)

    # Validate IL encoding constraints upfront
    if args.IL_n_pcs_lion > 0 and (args.IL_n_pcs_morgan > 0 or args.IL_n_pcs_mordred > 0):
        raise ValueError(
            "LiON encoding cannot be combined with Morgan or Mordred for IL. "
            "Use either --IL-n-pcs-lion OR --IL-n-pcs-morgan/--IL-n-pcs-mordred."
        )

    enc = {
        "IL": {
            "mfp": args.IL_n_pcs_morgan, "mordred": args.IL_n_pcs_mordred,
            "lion": args.IL_n_pcs_lion, "unimol": args.IL_n_pcs_unimol,
            "count_mfp": args.IL_n_pcs_count_mfp, "rdkit": args.IL_n_pcs_rdkit,
        },
        "HL": {
            "mfp": args.HL_n_pcs_morgan, "mordred": args.HL_n_pcs_mordred,
            "unimol": args.HL_n_pcs_unimol, "count_mfp": args.HL_n_pcs_count_mfp,
            "rdkit": args.HL_n_pcs_rdkit,
        },
        "CHL": {
            "mfp": args.CHL_n_pcs_morgan, "mordred": args.CHL_n_pcs_mordred,
            "unimol": args.CHL_n_pcs_unimol, "count_mfp": args.CHL_n_pcs_count_mfp,
            "rdkit": args.CHL_n_pcs_rdkit,
        },
        "PEG": {
            "mfp": args.PEG_n_pcs_morgan, "mordred": args.PEG_n_pcs_mordred,
            "unimol": args.PEG_n_pcs_unimol, "count_mfp": args.PEG_n_pcs_count_mfp,
            "rdkit": args.PEG_n_pcs_rdkit,
        },
    }

    encoded = dataset.encode_dataset(
        enc,
        encoding_csv_path=args.output,
        only_encodings=args.only_encodings,
        reduction=args.reduction,
    )

    print(f"Encoded dataset written to {args.output}")
    print(f"  Rows: {len(encoded.df)}")
    print(f"  Columns: {len(encoded.df.columns)}")
