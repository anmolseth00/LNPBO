from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT.parent))

from LNPBO.data.dataset import Dataset
from LNPBO.data.lnpdb_bridge import load_lnpdb_full

ASSAY_TYPES = [
    "in_vitro_single_formulation",
    "in_vitro_barcode_screen",
    "in_vivo_liver",
    "in_vivo_other",
    "unknown",
]


def _infer_assay_type_row(row: pd.Series) -> str:
    model = str(row.get("Model") or "").lower()
    route = str(row.get("Route_of_administration") or "").lower()
    target = str(row.get("Model_target") or "").lower()
    batching = str(row.get("Experiment_batching") or "").lower()

    in_vitro = model == "in_vitro" or route == "in_vitro" or target == "in_vitro"
    if in_vitro:
        if batching == "barcoded":
            return "in_vitro_barcode_screen"
        return "in_vitro_single_formulation"

    in_vivo = model == "in_vivo" or route in {
        "intravenous",
        "intramuscular",
        "intratracheal",
        "intradermal",
    }
    if in_vivo:
        if target == "liver":
            return "in_vivo_liver"
        return "in_vivo_other"

    return "unknown"


def add_assay_type(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["assay_type"] = df.apply(_infer_assay_type_row, axis=1)
    return df


def add_study_id(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Publication_PMID" in df.columns:
        study = df["Publication_PMID"].astype("string")
        if "Experiment_ID" in df.columns:
            study = study.fillna(df["Experiment_ID"].astype("string"))
        if "Publication_link" in df.columns:
            study = study.fillna(df["Publication_link"].astype("string"))
        df["study_id"] = study
    elif "Experiment_ID" in df.columns:
        df["study_id"] = df["Experiment_ID"].astype("string")
    elif "Publication_link" in df.columns:
        df["study_id"] = df["Publication_link"].astype("string")
    else:
        df["study_id"] = df.index.astype("string")
    return df


def load_lnpdb_clean(drop_duplicates: bool = False) -> pd.DataFrame:
    dataset = load_lnpdb_full(drop_duplicates=drop_duplicates)
    df = dataset.df
    df = add_study_id(df)
    df = add_assay_type(df)
    if "Formulation_ID" not in df.columns:
        df["Formulation_ID"] = range(1, len(df) + 1)
    return df


def encode_lantern_il(
    df: pd.DataFrame,
    train_idx: list[int] | None = None,
    test_idx: list[int] | None = None,
    il_pcs: int = 5,
    reduction: str = "pca",
):
    """Encode LANTERN IL-only features with optional train/test split."""
    if train_idx is None or test_idx is None:
        dataset = Dataset(df.copy(), source="lnpdb", name="lantern_il")
        encoded = dataset.encode_dataset(
            IL_n_pcs_count_mfp=il_pcs, IL_n_pcs_rdkit=il_pcs, reduction=reduction,
        )
        return encoded.df, encoded.fitted_transformers

    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()

    train_dataset = Dataset(train_df, source="lnpdb", name="lantern_il_train")
    train_encoded = train_dataset.encode_dataset(
        IL_n_pcs_count_mfp=il_pcs, IL_n_pcs_rdkit=il_pcs, reduction=reduction,
    )

    test_dataset = Dataset(test_df, source="lnpdb", name="lantern_il_test")
    test_encoded = test_dataset.encode_dataset(
        IL_n_pcs_count_mfp=il_pcs, IL_n_pcs_rdkit=il_pcs,
        reduction=reduction, fitted_transformers_in=train_encoded.fitted_transformers,
    )

    return train_encoded.df, test_encoded.df, train_encoded.fitted_transformers


def lantern_il_feature_cols(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c.startswith("IL_count_mfp_pc") or c.startswith("IL_rdkit_pc")]
    return sorted(cols)


def summarize_study_assay_types(study_df: pd.DataFrame) -> tuple[str, int]:
    counts = study_df["assay_type"].value_counts()
    if counts.empty:
        return "unknown", 0
    if len(counts) == 1:
        return counts.index[0], 1
    top = counts.index[0]
    if counts.iloc[0] / counts.sum() < 0.8:
        return "mixed", len(counts)
    return top, len(counts)
