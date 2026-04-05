"""
Bridge module to connect LNPBO's optimization framework to the full LNPDB database.

LNPDB (Lipid Nanoparticle Database) contains ~19,800 LNP formulations with
structure-function data for nucleic acid delivery. This module provides
functions to load, transform, and expose LNPDB data in the format expected
by LNPBO's Dataset class.

Data layout in LNPDB repo (data/LNPDB_for_LiON/):
    LNPDB.csv              -- full 19,797-row database (66 columns)
    all_data.csv           -- LiON main input (IL_SMILES, Experiment_value)
    all_data_all.csv       -- LiON extended  (24 columns incl. metadata)
    all_data_extra_x.csv   -- LiON one-hot encoded features (91 columns)
    all_data_metadata.csv  -- identifiers + IL_SMILES + Experiment_value
    all_data_weights.csv   -- sample weights
    single_split/          -- 70/15/15 train/val/test
    cv_splits_old/cv_{0..4}/   -- 5-fold CV (80/20)
    heldout/heldout_{name}/    -- heldout datasets with 5-fold CV each
"""

import logging
import os
from pathlib import Path

import pandas as pd

logger = logging.getLogger("lnpbo")

from .dataset import LNPDB_REQUIRED_COLUMNS, Dataset

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def get_lnpdb_path() -> Path:
    """Return the root of the LNPDB repository.

    Resolution order:
      1. LNPDB_PATH environment variable (if set)
      2. Symlink at  <LNPBO>/data/LNPDB_repo
      3. Sibling directory  ../../LNPDB  relative to this file
    Raises FileNotFoundError if none of the above exist.
    """
    # 1. Environment variable
    env = os.environ.get("LNPDB_PATH")
    if env:
        p = Path(env)
        if p.is_dir():
            return p

    this_dir = Path(__file__).resolve().parent  # <LNPBO>/data/

    # 2. Symlink
    symlink = this_dir / "LNPDB_repo"
    if symlink.is_dir():
        return symlink.resolve()

    # 3. Sibling repo
    sibling = this_dir.parent.parent / "LNPDB"
    if sibling.is_dir():
        return sibling

    raise FileNotFoundError(
        "Cannot locate LNPDB. Set the LNPDB_PATH env var, "
        "create the symlink at data/LNPDB_repo, "
        "or clone LNPDB as a sibling directory."
    )


def _lion_dir() -> Path:
    """Return the LNPDB_for_LiON directory."""
    return get_lnpdb_path() / "data" / "LNPDB_for_LiON"


# ---------------------------------------------------------------------------
# Column mapping:  LNPDB.csv  -->  LNPBO expected format
# ---------------------------------------------------------------------------

# LNPDB.csv already contains all LNPBO required columns with matching names.
# The only issue is NA-like strings ("NA", "") that pandas may not auto-detect
# when quoted in CSV.  We also need to handle the HL_name = "None" entries
# (which mean "no helper lipid") -- these should be kept as the string "None".

_LNPDB_CSV_USECOLS = [
    # Identifiers
    "LNP_ID",
    "Experiment_ID",
    "Formulation_ID",
    # Required by LNPBO
    "IL_name",
    "IL_SMILES",
    "IL_to_nucleicacid_massratio",
    "IL_molratio",
    "HL_name",
    "HL_SMILES",
    "HL_molratio",
    "CHL_name",
    "CHL_SMILES",
    "CHL_molratio",
    "PEG_name",
    "PEG_SMILES",
    "PEG_molratio",
    "Experiment_value",
    # Useful metadata
    "Model",
    "Model_type",
    "Model_target",
    "Route_of_administration",
    "Cargo",
    "Cargo_type",
    "Dose_ug_nucleicacid",
    "Aqueous_buffer",
    "Dialysis_buffer",
    "Mixing_method",
    "Experiment_batching",
    "Experiment_method",
    "Publication_link",
    "Publication_PMID",
    "IL_head_name",
]


def _read_lnpdb_csv(path: Path) -> pd.DataFrame:
    """Read the master LNPDB.csv handling its quoting / NA conventions."""
    usecols = [c for c in _LNPDB_CSV_USECOLS if c in _peek_columns(path)]
    df = pd.read_csv(
        path,
        usecols=usecols,
        na_values=["NA", ""],
        keep_default_na=True,
    )
    # Ensure component name columns never have NaN (use "None" for absent)
    for col in ["HL_name", "CHL_name", "PEG_name"]:
        if col in df.columns:
            df[col] = df[col].fillna("None")
    # IL_name must never be NaN
    if "IL_name" in df.columns:
        df = df.dropna(subset=["IL_name"])
    return df


def _peek_columns(path: Path) -> list[str]:
    """Read only the header of a CSV to discover column names."""
    return list(pd.read_csv(path, nrows=0).columns)


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------


def load_lnpdb_full(
    drop_missing_values: bool = True,
    drop_duplicates: bool = False,
    drop_unnormalized: bool = True,
    use_zscore_source: bool = True,
) -> Dataset:
    """Load the complete LNPDB dataset (~19,800 rows) as a Dataset.

    Parameters
    ----------
    drop_missing_values : bool
        If True, drop rows where any LNPDB_REQUIRED_COLUMNS value is NaN.
    drop_duplicates : bool
        If True, deduplicate on formulation composition. Use with care:
        the same formulation appears across multiple studies and should
        generally be kept to preserve study-level variation.
    drop_unnormalized : bool
        If True, remove formulations with unnormalized Experiment_value.
        LNPDB is z-scored per Experiment_ID (mean=0, std=1). Values >10
        are implausible z-scores. This filter is applied AFTER z-score
        replacement when use_zscore_source=True.
    use_zscore_source : bool
        If True (default), replace Experiment_value with the authoritative
        z-scored values from all_data_all.csv. LNPDB.csv contains a mix
        of raw and z-scored Experiment_value for ~8,800 rows due to the
        database being updated after all_data_all.csv was generated.
        The z-scoring is per Experiment_ID (≈per paper), covering all
        measurement methods within a study (luminescence, uptake, etc.).

    Returns
    -------
    Dataset
        Ready for use with LNPBO's optimization pipeline.
    """
    csv_path = _lion_dir() / "LNPDB.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"LNPDB.csv not found at {csv_path}")

    df = _read_lnpdb_csv(csv_path)

    if use_zscore_source:
        zscore_path = _lion_dir() / "all_data_all.csv"
        if zscore_path.exists():
            zscore_df = pd.read_csv(zscore_path, usecols=["LNP_ID", "Experiment_value"])
            zscore_map = zscore_df.set_index("LNP_ID")["Experiment_value"]
            mask = df["LNP_ID"].isin(zscore_map.index)
            n_replaced = mask.sum()
            df.loc[mask, "Experiment_value"] = df.loc[mask, "LNP_ID"].map(zscore_map)
            logger.info("Replaced Experiment_value with z-scored values for %d rows", n_replaced)

    if drop_unnormalized and "Experiment_value" in df.columns:
        bad_mask = df["Experiment_value"].abs() > 10
        n_bad = bad_mask.sum()
        if n_bad > 0:
            df = df[~bad_mask]
            logger.info("Dropped %d unnormalized formulations (|Experiment_value| > 10)", n_bad)

    # Validate required columns
    missing = set(LNPDB_REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"LNPDB.csv is missing required columns: {missing}")

    if drop_missing_values:
        df = df.dropna(subset=LNPDB_REQUIRED_COLUMNS)

    if drop_duplicates:
        from .dataset import columns_to_check_for_duplicates

        df = df.drop_duplicates(subset=[c for c in columns_to_check_for_duplicates if c in df.columns])

    df["Formulation_ID"] = range(1, len(df) + 1)
    df["Round"] = 0

    return Dataset(df, source="lnpdb", name="LNPDB_full")


def load_lnpdb_lion_data(
    variant: str = "all",
) -> dict[str, pd.DataFrame]:
    """Load the LiON-formatted data files from LNPDB.

    Parameters
    ----------
    variant : str
        Which data variant to load:
        - "all"        : all_data.csv  (IL_SMILES + Experiment_value)
        - "all_ext"    : all_data_all.csv  (24-col extended)
        - "all_extra_x": all_data_extra_x.csv  (one-hot features)
        - "all_meta"   : all_data_metadata.csv  (identifiers)
        - "all_weights": all_data_weights.csv  (sample weights)

    Returns
    -------
    dict
        Mapping of variant name to DataFrame.  When variant is "all" (default)
        all five files are loaded and returned together.
    """
    lion = _lion_dir()

    files = {
        "main": lion / "all_data.csv",
        "extended": lion / "all_data_all.csv",
        "extra_x": lion / "all_data_extra_x.csv",
        "metadata": lion / "all_data_metadata.csv",
        "weights": lion / "all_data_weights.csv",
    }

    variant_map = {
        "all": list(files.keys()),
        "all_ext": ["extended"],
        "all_extra_x": ["extra_x"],
        "all_meta": ["metadata"],
        "all_weights": ["weights"],
    }

    keys = variant_map.get(variant)
    if keys is None:
        raise ValueError(f"Unknown variant {variant!r}. Choose from: {list(variant_map.keys())}")

    result = {}
    for key in keys:
        path = files[key]
        if path.exists():
            result[key] = pd.read_csv(path)
        else:
            result[key] = None

    return result


def get_lnpdb_splits(
    split_type: str = "single",
    heldout_name: str | None = None,
    cv_index: int | None = None,
) -> dict[str, pd.DataFrame]:
    """Load pre-computed train/val/test splits from LNPDB_for_LiON.

    Parameters
    ----------
    split_type : str
        "single"  -- 70/15/15 train/val/test from single_split/
        "cv_old"  -- old 5-fold CV splits from cv_splits_old/
        "heldout" -- heldout dataset CV splits

    heldout_name : str, optional
        Required when split_type="heldout".
        One of: "BL_2023", "LM_2019", "SL_2020", "ZC_2023"

    cv_index : int, optional
        Which CV fold (0-4).  Required for "cv_old" and "heldout".
        Ignored for "single".

    Returns
    -------
    dict
        Keys depend on split_type:
        - "single": train, train_extra_x, train_metadata, train_weights,
                    val, val_extra_x, val_metadata, val_weights,
                    test, test_extra_x, test_metadata, test_weights
        - "cv_old": train, train_extra_x, ..., test, test_extra_x, ...
        - "heldout": train, test (CV fold), heldout_data, heldout_data_extra_x,
                     heldout_data_results (if available)
    """
    lion = _lion_dir()

    if split_type == "single":
        base = lion / "single_split"
        return _load_split_dir(base, has_val=True)

    elif split_type == "cv_old":
        if cv_index is None:
            raise ValueError("cv_index is required for cv_old splits")
        base = lion / "cv_splits_old" / f"cv_{cv_index}"
        return _load_split_dir(base, has_val=False)

    elif split_type == "heldout":
        if heldout_name is None:
            raise ValueError("heldout_name is required. Choose from: BL_2023, LM_2019, SL_2020, ZC_2023")
        if cv_index is None:
            raise ValueError("cv_index (0-4) is required for heldout splits")

        heldout_dir = lion / "heldout" / f"heldout_{heldout_name}"
        if not heldout_dir.is_dir():
            raise FileNotFoundError(f"Heldout directory not found: {heldout_dir}")

        cv_dir = heldout_dir / "cv_splits" / f"cv_{cv_index}"
        result = _load_split_dir(cv_dir, has_val=False)

        # Also load the heldout-level data
        for suffix in [
            "heldout_data",
            "heldout_data_extra_x",
            "heldout_data_metadata",
            "heldout_data_weights",
            "heldout_data_all",
            "all_data",
            "all_data_all",
            "all_data_extra_x",
            "all_data_metadata",
            "all_data_weights",
        ]:
            path = heldout_dir / f"{suffix}.csv"
            if path.exists():
                result[suffix] = pd.read_csv(path)

        # Load heldout predictions if available
        results_path = cv_dir / "heldout_data_results.csv"
        if results_path.exists():
            result["heldout_data_results"] = pd.read_csv(results_path)

        return result

    else:
        raise ValueError(f"Unknown split_type {split_type!r}. Choose from: single, cv_old, heldout")


def _load_split_dir(
    base: Path,
    has_val: bool = False,
) -> dict[str, pd.DataFrame]:
    """Load all CSV files from a split directory."""
    if not base.is_dir():
        raise FileNotFoundError(f"Split directory not found: {base}")

    result = {}
    partitions = ["train", "test"]
    if has_val:
        partitions.append("val")

    suffixes = ["", "_extra_x", "_metadata", "_weights"]

    for part in partitions:
        for suffix in suffixes:
            filename = f"{part}{suffix}.csv"
            path = base / filename
            if path.exists():
                result[f"{part}{suffix}"] = pd.read_csv(path)

    # Also load test_results if present
    test_results = base / "test_results.csv"
    if test_results.exists():
        result["test_results"] = pd.read_csv(test_results)

    # Fingerprints (only in single_split)
    fp = base / "fingerprints_all_data.csv"
    if fp.exists():
        result["fingerprints_all_data"] = pd.read_csv(fp)

    return result


# ---------------------------------------------------------------------------
# Dataset conversion helpers
# ---------------------------------------------------------------------------


def lion_split_to_dataset(
    split_type: str = "single",
    partition: str = "train",
    heldout_name: str | None = None,
    cv_index: int | None = None,
) -> Dataset:
    """Load a LiON split partition and convert it to a LNPBO Dataset.

    Since LiON split files only contain (IL_SMILES, Experiment_value),
    this function joins against LNPDB.csv to recover the full column set
    that LNPBO requires (IL_name, HL_name, molar ratios, etc.).

    The split CSVs and their companion metadata CSVs are row-aligned
    (same ordering, same length), so we use positional concatenation
    to attach LNP_ID, then use LNP_ID as a unique key to join against
    LNPDB.csv for the remaining columns.

    Parameters
    ----------
    split_type : str
        "single", "cv_old", or "heldout"
    partition : str
        "train", "val", or "test"
    heldout_name : str, optional
        Required for split_type="heldout"
    cv_index : int, optional
        Required for "cv_old" and "heldout"

    Returns
    -------
    Dataset
    """
    splits = get_lnpdb_splits(
        split_type=split_type,
        heldout_name=heldout_name,
        cv_index=cv_index,
    )

    if partition not in splits:
        raise ValueError(
            f"Partition {partition!r} not found. Available: {[k for k in splits if not k.startswith('_')]}"
        )

    split_df = splits[partition].copy()  # IL_SMILES, Experiment_value

    # The metadata CSV is row-aligned with the main split CSV.
    # Attach metadata columns (especially LNP_ID) via positional concat.
    meta_key = f"{partition}_metadata"
    if meta_key in splits and splits[meta_key] is not None:
        meta_df = splits[meta_key]
        # Only add columns not already present in split_df
        new_cols = [c for c in meta_df.columns if c not in split_df.columns]
        if new_cols:
            split_df = pd.concat(
                [split_df.reset_index(drop=True), meta_df[new_cols].reset_index(drop=True)],
                axis=1,
            )

    # Load full LNPDB to join additional columns
    full_csv = _lion_dir() / "LNPDB.csv"
    if not full_csv.exists():
        raise FileNotFoundError("LNPDB.csv needed to recover full columns for split data")

    full_df = _read_lnpdb_csv(full_csv)

    # Determine join key -- LNP_ID is unique and preferred
    if "LNP_ID" in split_df.columns and "LNP_ID" in full_df.columns:
        join_cols = ["LNP_ID"]
    else:
        join_cols = ["IL_SMILES", "Experiment_value"]

    # Columns we need from the full DB that are missing from split_df
    needed = set(LNPDB_REQUIRED_COLUMNS) - set(split_df.columns)
    # Also grab SMILES columns for encoding
    extra_want = {"IL_SMILES", "HL_SMILES", "CHL_SMILES", "PEG_SMILES"}
    needed = needed | (extra_want - set(split_df.columns))

    if needed:
        # Only fetch what we need plus join keys from full_df
        fetch_cols = list(needed | set(join_cols))
        fetch_cols = [c for c in fetch_cols if c in full_df.columns]

        lookup = full_df[fetch_cols].drop_duplicates(subset=join_cols)

        merged = split_df.merge(
            lookup,
            on=join_cols,
            how="left",
            suffixes=("", "_full"),
        )
        # Drop any duplicate columns from merge
        dup_cols = [c for c in merged.columns if c.endswith("_full")]
        merged = merged.drop(columns=dup_cols, errors="ignore")
    else:
        merged = split_df

    # Check if we now have the required columns
    still_missing = set(LNPDB_REQUIRED_COLUMNS) - set(merged.columns)
    if still_missing:
        raise ValueError(
            f"After joining with LNPDB.csv, still missing columns: "
            f"{still_missing}. The split data may not be recoverable."
        )

    # Fill component names that might be NaN
    for col in ["HL_name", "CHL_name", "PEG_name"]:
        if col in merged.columns:
            merged[col] = merged[col].fillna("None")

    # Drop rows still missing required values
    merged = merged.dropna(subset=LNPDB_REQUIRED_COLUMNS)

    # Assign formulation IDs and round
    merged["Formulation_ID"] = range(1, len(merged) + 1)
    merged["Round"] = 0

    name = f"LNPDB_{split_type}_{partition}"
    if heldout_name:
        name += f"_{heldout_name}"
    if cv_index is not None:
        name += f"_cv{cv_index}"

    return Dataset(merged, source="lnpdb", name=name)


# ---------------------------------------------------------------------------
# Discovery / listing
# ---------------------------------------------------------------------------


def list_available_datasets() -> dict[str, object]:
    """List all available LNPDB datasets and splits.

    Returns
    -------
    dict
        Structured summary of available data, e.g.::

            {
                "lnpdb_full": {"path": ..., "rows": 19797},
                "lion_all_data": {"path": ..., "rows": 18858},
                "single_split": {"train": 13883, "val": 2811, "test": 2164},
                "cv_splits_old": {0: {...}, 1: {...}, ...},
                "heldout": {
                    "BL_2023": {"all_data": ..., "heldout_data": ..., "cv_splits": [0..4]},
                    ...
                },
            }
    """
    lion = _lion_dir()
    info: dict[str, object] = {}

    # Full database
    lnpdb_csv = lion / "LNPDB.csv"
    if lnpdb_csv.exists():
        info["lnpdb_full"] = {
            "path": str(lnpdb_csv),
            "rows": _count_csv_rows(lnpdb_csv),
        }

    # LiON all_data
    all_data = lion / "all_data.csv"
    if all_data.exists():
        info["lion_all_data"] = {
            "path": str(all_data),
            "rows": _count_csv_rows(all_data),
        }

    # Single split
    ss = lion / "single_split"
    if ss.is_dir():
        info["single_split"] = {}
        for part in ["train", "val", "test"]:
            p = ss / f"{part}.csv"
            if p.exists():
                info["single_split"][part] = _count_csv_rows(p)

    # Old CV splits
    cv_old = lion / "cv_splits_old"
    if cv_old.is_dir():
        info["cv_splits_old"] = {}
        for i in range(5):
            cv_dir = cv_old / f"cv_{i}"
            if cv_dir.is_dir():
                fold_info = {}
                for part in ["train", "test"]:
                    p = cv_dir / f"{part}.csv"
                    if p.exists():
                        fold_info[part] = _count_csv_rows(p)
                info["cv_splits_old"][i] = fold_info

    # Heldout datasets
    heldout_base = lion / "heldout"
    if heldout_base.is_dir():
        info["heldout"] = {}
        for hdir in sorted(heldout_base.iterdir()):
            if hdir.is_dir() and hdir.name.startswith("heldout_"):
                name = hdir.name.replace("heldout_", "")
                hinfo: dict[str, object] = {}

                ad = hdir / "all_data.csv"
                if ad.exists():
                    hinfo["all_data_rows"] = _count_csv_rows(ad)

                hd = hdir / "heldout_data.csv"
                if hd.exists():
                    hinfo["heldout_data_rows"] = _count_csv_rows(hd)

                cv_dir = hdir / "cv_splits"
                if cv_dir.is_dir():
                    available_folds = sorted(
                        [
                            int(d.name.replace("cv_", ""))
                            for d in cv_dir.iterdir()
                            if d.is_dir() and d.name.startswith("cv_")
                        ]
                    )
                    hinfo["cv_folds"] = available_folds

                info["heldout"][name] = hinfo

    return info


def _count_csv_rows(path: Path) -> int:
    """Count data rows in a CSV (excluding header)."""
    with open(path) as f:
        return sum(1 for _ in f) - 1


# ---------------------------------------------------------------------------
# Convenience: load from LNPDB.csv directly via Dataset.from_lnpdb_csv
# ---------------------------------------------------------------------------


def lnpdb_csv_path() -> str:
    """Return the path to LNPDB.csv as a string, suitable for
    ``Dataset.from_lnpdb_csv(lnpdb_csv_path())``.
    """
    p = _lion_dir() / "LNPDB.csv"
    if not p.exists():
        raise FileNotFoundError(f"LNPDB.csv not found at {p}")
    return str(p)
