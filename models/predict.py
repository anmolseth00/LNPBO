#!/usr/bin/env python3
"""Inference with the trained XGBoost model on LNPDB features.

Single prediction:
    python models/predict.py \
        --il-smiles "CCCCCCCCN(CC)CCCN(CC)CCC(=O)NCCCCCCCC" \
        --il-molratio 35 --hl-molratio 16 --chl-molratio 46.5 --peg-molratio 2.5 \
        --mass-ratio 10 --dose 0.5 \
        --hl-name DOPE --chl-name Cholesterol --peg-name DMG-PEG2000 \
        --cargo mRNA --cargo-type FLuc \
        --model-type HeLa --model-target in_vitro \
        --route in_vitro --mixing microfluidics \
        --buffer citrate --dialysis PBS \
        --batching individual

Batch prediction from CSV:
    python models/predict.py --csv input.csv --out predictions.csv

CSV must have column IL_SMILES and any subset of the formulation columns.
Missing columns are filled with 0 (continuous) or "Unknown" (categorical).
"""


import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from LNPBO.models.data import compute_morgan_fingerprints

MODEL_DIR = Path(__file__).resolve().parent / "runs" / "xgb_tuned"
MODEL_PATH = MODEL_DIR / "model.json"
METRICS_PATH = MODEL_DIR / "test_metrics.json"

FP_BITS = 2048
FP_RADIUS = 3

CONTINUOUS_COLS = [
    "IL_molratio",
    "HL_molratio",
    "CHL_molratio",
    "PEG_molratio",
    "IL_to_nucleicacid_massratio",
    "Dose_ug_nucleicacid",
]

# Categorical levels learned from the training split (scaffold_split seed=42,
# stratified random fallback, min_count=5). Order matters for feature alignment.
CAT_LEVELS = {
    "HL_name": ["14PA", "18PG", "DDAB", "DOPE", "DOTAP", "DSPC", "MDOA"],
    "CHL_name": ["Cholesterol"],
    "PEG_name": [
        "ALC-0159", "C16-Ceramide-PEG2000", "C8-Ceramide-PEG2000",
        "DMG-C-PEG2000", "DMG-PEG2000", "DMPE-PEG2000", "DSG-PEG2000",
        "DSPE-PEG2000", "Unknown",
    ],
    "Aqueous_buffer": ["Unknown", "acetate", "citrate"],
    "Dialysis_buffer": ["PBS", "water"],
    "Mixing_method": ["handmixed", "microfluidics"],
    "Model_type": [
        "A549", "BMDC", "BMDM", "BeWo_b30", "DC2.4", "HBEC_ALI",
        "HEK293T", "HeLa", "HepG2", "IGROV1", "Mouse_Ai14", "Mouse_B6",
        "Mouse_BALBc", "Mouse_CD1", "RAW264.7",
    ],
    "Model_target": [
        "bone_marrow", "heart", "in_vitro", "kidney", "liver", "lung",
        "lung_epithelium", "muscle", "spleen", "whole_body",
    ],
    "Route_of_administration": [
        "in_vitro", "intramuscular", "intratracheal", "intravenous",
    ],
    "Cargo": ["mRNA", "pDNA", "siRNA"],
    "Cargo_type": [
        "DNA_barcode", "FLuc", "FVII", "GFP", "base_editor", "hEPO",
        "peptide_barcode",
    ],
    "Experiment_batching": ["barcoded", "individual"],
}

N_CAT_ONEHOT = sum(len(v) for v in CAT_LEVELS.values())  # 65
N_FEATURES = FP_BITS + len(CONTINUOUS_COLS) + N_CAT_ONEHOT  # 2119


def encode_categoricals(df):
    parts = []
    for col, levels in CAT_LEVELS.items():
        vals = df[col] if col in df.columns else pd.Series([""] * len(df))
        for level in levels:
            parts.append((vals == level).astype(np.float32).values)
    return np.column_stack(parts) if parts else np.zeros((len(df), 0), dtype=np.float32)


def featurize(df):
    fps = compute_morgan_fingerprints(
        df["IL_SMILES"].tolist(), radius=FP_RADIUS, n_bits=FP_BITS,
    )

    cont = np.zeros((len(df), len(CONTINUOUS_COLS)), dtype=np.float32)
    for i, col in enumerate(CONTINUOUS_COLS):
        if col in df.columns:
            cont[:, i] = pd.to_numeric(df[col], errors="coerce").fillna(0).values

    cat = encode_categoricals(df)

    X = np.concatenate([fps, cont, cat], axis=1)
    assert X.shape[1] == N_FEATURES, f"Expected {N_FEATURES} features, got {X.shape[1]}"
    return X


def load_model(path=MODEL_PATH):
    model = xgb.XGBRegressor()
    model.load_model(str(path))
    return model


def predict(model, df):
    X = featurize(df)
    return model.predict(X)


def single_to_df(args):
    row = {
        "IL_SMILES": args.il_smiles,
        "IL_molratio": args.il_molratio,
        "HL_molratio": args.hl_molratio,
        "CHL_molratio": args.chl_molratio,
        "PEG_molratio": args.peg_molratio,
        "IL_to_nucleicacid_massratio": args.mass_ratio,
        "Dose_ug_nucleicacid": args.dose,
        "HL_name": args.hl_name or "",
        "CHL_name": args.chl_name or "",
        "PEG_name": args.peg_name or "",
        "Aqueous_buffer": args.buffer or "",
        "Dialysis_buffer": args.dialysis or "",
        "Mixing_method": args.mixing or "",
        "Model_type": args.model_type or "",
        "Model_target": args.model_target or "",
        "Route_of_administration": args.route or "",
        "Cargo": args.cargo or "",
        "Cargo_type": args.cargo_type or "",
        "Experiment_batching": args.batching or "",
    }
    return pd.DataFrame([row])


def main():
    parser = argparse.ArgumentParser(description="XGBoost inference for LNPDB", suggest_on_error=True)
    parser.add_argument("--model", type=str, default=str(MODEL_PATH))

    # Batch mode
    parser.add_argument("--csv", type=str, help="Input CSV for batch prediction")
    parser.add_argument("--out", type=str, help="Output CSV path (batch mode)")

    # Single mode
    parser.add_argument("--il-smiles", type=str)
    parser.add_argument("--il-molratio", type=float, default=0)
    parser.add_argument("--hl-molratio", type=float, default=0)
    parser.add_argument("--chl-molratio", type=float, default=0)
    parser.add_argument("--peg-molratio", type=float, default=0)
    parser.add_argument("--mass-ratio", type=float, default=0)
    parser.add_argument("--dose", type=float, default=0)
    parser.add_argument("--hl-name", type=str, default="")
    parser.add_argument("--chl-name", type=str, default="")
    parser.add_argument("--peg-name", type=str, default="")
    parser.add_argument("--buffer", type=str, default="")
    parser.add_argument("--dialysis", type=str, default="")
    parser.add_argument("--mixing", type=str, default="")
    parser.add_argument("--model-type", type=str, default="")
    parser.add_argument("--model-target", type=str, default="")
    parser.add_argument("--route", type=str, default="")
    parser.add_argument("--cargo", type=str, default="")
    parser.add_argument("--cargo-type", type=str, default="")
    parser.add_argument("--batching", type=str, default="")

    args = parser.parse_args()

    model = load_model(args.model)

    if args.csv:
        df = pd.read_csv(args.csv)
        if "IL_SMILES" not in df.columns:
            print("ERROR: CSV must have an IL_SMILES column", file=sys.stderr)
            sys.exit(1)
        preds = predict(model, df)
        df["predicted_z"] = preds
        if args.out:
            df.to_csv(args.out, index=False)
            print(f"Saved {len(df)} predictions to {args.out}")
        else:
            print(df[["IL_SMILES", "predicted_z"]].to_string(index=False))
    elif args.il_smiles:
        df = single_to_df(args)
        pred = predict(model, df)[0]
        print(f"Predicted z-scored Experiment_value: {pred:.4f}")
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
