import numpy as np
import pandas as pd

from LNPBO.data.dataset import Dataset
from LNPBO.data.lnpdb_bridge import load_lnpdb_full

ASSAY_TYPES = [
    "in_vitro_single_formulation",
    "in_vitro_barcode_screen",
    "in_vivo_liver",
    "in_vivo_other",
    "unknown",
]


def infer_assay_type_row(row: pd.Series) -> str:
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
    df["assay_type"] = df.apply(infer_assay_type_row, axis=1)
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
    enc = {"IL": {"count_mfp": il_pcs, "rdkit": il_pcs}}

    if train_idx is None or test_idx is None:
        dataset = Dataset(df.copy(), source="lnpdb", name="lantern_il")
        encoded = dataset.encode_dataset(enc, reduction=reduction)
        return encoded.df, encoded.fitted_transformers

    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()

    train_dataset = Dataset(train_df, source="lnpdb", name="lantern_il_train")
    train_encoded = train_dataset.encode_dataset(enc, reduction=reduction)

    test_dataset = Dataset(test_df, source="lnpdb", name="lantern_il_test")
    test_encoded = test_dataset.encode_dataset(
        enc, reduction=reduction, fitted_transformers_in=train_encoded.fitted_transformers,
    )

    return train_encoded.df, test_encoded.df, train_encoded.fitted_transformers


def lantern_il_feature_cols(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c.startswith("IL_count_mfp_pc") or c.startswith("IL_rdkit_pc")]
    return sorted(cols)


def build_study_type_map(df: pd.DataFrame) -> dict[str, str]:
    """Map each study_id to its dominant assay type."""
    result = {}
    for sid, sdf in df.groupby("study_id"):
        assay_type, _ = summarize_study_assay_types(sdf)
        result[str(sid)] = assay_type
    return result


def study_split(df_or_ids, study_to_type: dict[str, str] | None = None,
                *, seed: int = 42) -> tuple[set, set]:
    """Stratified 80/20 study-level split by assay type.

    Can be called two ways:
      study_split(df, seed=42)              — builds study_to_type internally
      study_split(ids, study_to_type, seed=N) — uses pre-built mapping
    """
    if isinstance(df_or_ids, pd.DataFrame):
        study_to_type = build_study_type_map(df_or_ids)
        all_ids = df_or_ids["study_id"].unique()
    else:
        all_ids = df_or_ids
        if study_to_type is None:
            raise ValueError("study_to_type required when passing study IDs")

    rng = np.random.RandomState(seed)
    train_ids: set = set()
    test_ids: set = set()
    for assay_type in sorted(set(study_to_type.values())):
        ids = [sid for sid in all_ids if study_to_type.get(str(sid)) == assay_type]
        rng.shuffle(ids)
        cut = max(1, int(0.8 * len(ids))) if len(ids) > 1 else len(ids)
        train_ids.update(ids[:cut])
        test_ids.update(ids[cut:])
    return train_ids, test_ids


def prepare_study_data(min_n: int = 5, reduction: str = "pca"):
    """Load LNPDB, encode LANTERN IL-only, and do a stratified study split.

    Shared by bradley_terry, groupdro_surrogate, and vrex_surrogate.
    """
    df = load_lnpdb_clean(drop_duplicates=False)
    df = df.dropna(subset=["IL_SMILES", "Experiment_value"]).reset_index(drop=True)

    study_sizes = df.groupby("study_id").size()
    keep_ids = study_sizes[study_sizes >= min_n].index
    df = df[df["study_id"].isin(keep_ids)].reset_index(drop=True)

    encoded, _ = encode_lantern_il(df, reduction=reduction)
    feat_cols = lantern_il_feature_cols(encoded)

    study_to_type = build_study_type_map(df)
    train_ids, test_ids = study_split(df["study_id"].unique(), study_to_type, seed=42)

    train_mask = df["study_id"].isin(train_ids)
    test_mask = df["study_id"].isin(test_ids)

    X = encoded[feat_cols].values.astype(np.float32)
    y = encoded["Experiment_value"].values.astype(np.float32)
    study_ids = df["study_id"].astype(str).values

    return X, y, study_ids, train_mask.values, test_mask.values


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
