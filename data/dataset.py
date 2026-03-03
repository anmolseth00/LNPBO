from __future__ import annotations

import os

import pandas as pd

from .compute_pcs import compute_pcs

LNPDB_REQUIRED_COLUMNS = [
    "IL_name",
    "IL_to_nucleicacid_massratio",
    "IL_molratio",
    "HL_name",
    "HL_molratio",
    "CHL_name",
    "CHL_molratio",
    "PEG_name",
    "PEG_molratio",
    "Experiment_value",
]

INGREDIENT_COLUMNS = [
    "IL_name",
    "IL_SMILES",
    "IL_to_nucleicacid_massratio",
    "IL_molratio",
    "IL_to_nucleicacid_upper",
    "IL_to_nucleicacid_lower",
    "IL_upper",
    "IL_lower",
    "HL_name",
    "HL_SMILES",
    "HL_molratio",
    "HL_upper",
    "HL_lower",
    "CHL_name",
    "CHL_SMILES",
    "CHL_molratio",
    "CHL_upper",
    "CHL_lower",
    "PEG_name",
    "PEG_SMILES",
    "PEG_molratio",
    "PEG_upper",
    "PEG_lower",
]

columns_to_check_for_duplicates = [
    "IL_name",
    "HL_name",
    "CHL_name",
    "PEG_name",
    "IL_to_nucleicacid_massratio",
    "IL_molratio",
    "HL_molratio",
    "CHL_molratio",
    "PEG_molratio",
]


class Dataset:
    """
    Thin wrapper around a pandas DataFrame representing LNP experiments.

    Responsibilities:
    - schema validation
    - round / formulation tracking
    - append-only workflow
    """

    def __init__(
        self,
        df: pd.DataFrame,
        source: str = "lnpdb",
        metadata: dict | None = None,
        name="screen",
        encoders=None,
        fitted_transformers=None,
    ):
        self.df = df.copy()
        self.source = source
        self.metadata = metadata or {}
        self.name = name
        self.encoders = encoders
        self.fitted_transformers = fitted_transformers or {}

    @classmethod
    def from_lnpdb_csv(cls, path: str) -> Dataset:
        df = pd.read_csv(path)
        name = os.path.basename(path)

        missing = set(LNPDB_REQUIRED_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required LNPDB columns: {missing}")

        incomplete_mask = df[LNPDB_REQUIRED_COLUMNS].isna().any(axis=1)
        if incomplete_mask.any():
            bad_rows = df.index[incomplete_mask].tolist()
            raise ValueError(
                f"Incomplete rows detected at indices {bad_rows}. All required columns must be fully filled."
            )

        if df.duplicated(subset=columns_to_check_for_duplicates).any():
            n_before = len(df)
            # Average Experiment_value for duplicate formulations instead of arbitrary first-row selection
            non_dup_cols = [c for c in df.columns if c not in ["Experiment_value"]]
            group_cols = [c for c in columns_to_check_for_duplicates if c in df.columns]
            df = df.groupby(group_cols, as_index=False).agg(
                {c: "first" for c in non_dup_cols if c not in group_cols} | {"Experiment_value": "mean"}
            )
            print(f"Averaged {n_before - len(df)} duplicate formulations ({len(df)} remaining)")

        # Data quality warnings
        ev = df["Experiment_value"]
        ev_mean, ev_std = ev.mean(), ev.std()
        n_outliers = ((ev - ev_mean).abs() > 5 * ev_std).sum()
        if n_outliers > 0:
            print(f"Warning: {n_outliers} rows have Experiment_value > 5 std from mean")

        # Check molar ratio sums
        ratio_cols = [c for c in columns_to_check_for_duplicates if c.endswith("_molratio")]
        if ratio_cols:
            ratio_sums = df[ratio_cols].sum(axis=1)
            bad_sums = ((ratio_sums - ratio_sums.median()).abs() > 0.01).sum()
            if bad_sums > 0:
                print(
                    f"Warning: {bad_sums} rows have molar ratio sums deviating from median ({ratio_sums.median():.4f})"
                )

        df["Formulation_ID"] = range(1, len(df) + 1)
        df["Round"] = 0

        return cls(df, source="lnpdb", name=name)

    def encode_dataset(
        self,
        IL_n_pcs_morgan: int = 0,
        IL_n_pcs_mordred: int = 0,
        IL_n_pcs_lion: int = 0,
        HL_n_pcs_morgan: int = 0,
        HL_n_pcs_mordred: int = 0,
        CHL_n_pcs_morgan: int = 0,
        CHL_n_pcs_mordred: int = 0,
        PEG_n_pcs_morgan: int = 0,
        PEG_n_pcs_mordred: int = 0,
        encoding_csv_path: str | None = None,
        only_encodings: bool = False,
        reduction: str = "pca",
    ) -> Dataset:

        df = self.df.copy()

        def is_variable(col):
            return col in df.columns and df[col].nunique() > 1

        variable_components = {
            "IL": is_variable("IL_name"),
            "HL": is_variable("HL_name"),
            "CHL": is_variable("CHL_name"),
            "PEG": is_variable("PEG_name"),
        }

        variable_molratios = {
            "IL": is_variable("IL_molratio"),
            "HL": is_variable("HL_molratio"),
            "CHL": is_variable("CHL_molratio"),
            "PEG": is_variable("PEG_molratio"),
        }

        metadata = {
            "variable_components": variable_components,
            "variable_molratios": variable_molratios,
            "variable_il_mrna": is_variable("IL_to_nucleicacid_massratio"),
            "pcs": {"IL": {}, "HL": {}, "CHL": {}, "PEG": {}},
            "reduction": reduction,
        }

        # Prevent mixed IL encoding strategies
        if IL_n_pcs_lion > 0 and (IL_n_pcs_morgan > 0 or IL_n_pcs_mordred > 0):
            raise ValueError("LiON encoding cannot be combined with Morgan or Mordred for IL.")

        def unique_lipids(role: str) -> pd.DataFrame:
            smiles_col = f"{role}_SMILES"
            name_col = f"{role}_name"
            if smiles_col in df.columns:
                n_missing = df[smiles_col].isna().sum()
                if n_missing > 0:
                    missing_names = df.loc[df[smiles_col].isna(), name_col].unique()
                    suffix = "..." if len(missing_names) > 5 else ""
                    print(
                        f"Warning: {n_missing} {role} rows missing SMILES "
                        f"({len(missing_names)} unique lipids: {list(missing_names)[:5]}{suffix})"
                    )
            return (
                df[[f"{role}_name", f"{role}_SMILES"]]
                .dropna()
                .drop_duplicates()
                .rename(
                    columns={
                        f"{role}_name": "lipid_name",
                        f"{role}_SMILES": "SMILES",
                    }
                )
                .assign(lipid_type=role)
                .reset_index(drop=True)
            )

        def average_experiment_values(role: str) -> dict[str, float]:
            name_col = f"{role}_name"
            return df.dropna(subset=["Experiment_value"]).groupby(name_col)["Experiment_value"].mean().to_dict()

        fitted_transformers = {}

        def encode_lipid_table(lipid_df: pd.DataFrame, role: str, pcs_spec: dict):

            blocks = []
            smiles = lipid_df["SMILES"].tolist()

            # Per-lipid average target values (needed for LiON and PLS)
            exp_map = None
            needs_targets = pcs_spec.get("lion", 0) > 0 or reduction == "pls"
            if needs_targets:
                exp_map = average_experiment_values(role)
                if not exp_map:
                    raise ValueError(
                        f"Target values required for {role} encoding (reduction={reduction!r}) but none available."
                    )

            for enc_type, n in pcs_spec.items():
                if n <= 0:
                    continue

                exp_vals = None
                if enc_type == "lion" or reduction == "pls":
                    assert exp_map is not None
                    exp_vals = []
                    for name in lipid_df["lipid_name"]:
                        v = exp_map.get(name)
                        if v is None:
                            raise ValueError(
                                f"Missing Experiment_value for {role} lipid {name!r} "
                                f"required for {enc_type}/{reduction} encoding."
                            )
                        exp_vals.append(float(v))

                _, _, pc_list, reducer_obj, scaler_obj = compute_pcs(
                    smiles,
                    feature_type=enc_type,
                    n_components=n,
                    experiment_values=exp_vals,
                    reduction=reduction,
                )

                fitted_transformers[f"{role}_{enc_type}"] = {
                    "reducer": reducer_obj,
                    "scaler": scaler_obj,
                }

                cols = [f"{role}_{enc_type}_pc{i + 1}" for i in range(pc_list.shape[1])]

                blocks.append(pd.DataFrame(pc_list, columns=cols))

            if not blocks:
                return None

            return pd.concat([lipid_df.reset_index(drop=True), *blocks], axis=1)

        encoders = {
            "IL": {"mfp": IL_n_pcs_morgan, "mordred": IL_n_pcs_mordred, "lion": IL_n_pcs_lion},
            "HL": {"mfp": HL_n_pcs_morgan, "mordred": HL_n_pcs_mordred},
            "CHL": {"mfp": CHL_n_pcs_morgan, "mordred": CHL_n_pcs_mordred},
            "PEG": {"mfp": PEG_n_pcs_morgan, "mordred": PEG_n_pcs_mordred},
        }

        encoding_tables = []

        for role, pcs_spec in encoders.items():
            if not variable_components.get(role, False):
                continue

            lipid_df = unique_lipids(role)
            enc_df = encode_lipid_table(lipid_df, role, pcs_spec)

            if enc_df is not None:
                encoding_tables.append(enc_df)
                metadata["pcs"][role] = pcs_spec

                df = df.merge(
                    enc_df.drop(columns=["SMILES", "lipid_type"]),
                    left_on=f"{role}_name",
                    right_on="lipid_name",
                    how="left",
                ).drop(columns=["lipid_name"])

        # Save
        if encoding_csv_path:
            if only_encodings:
                if encoding_tables:
                    pd.concat(encoding_tables, axis=0).to_csv(encoding_csv_path, index=False)
            else:
                ordered_base_cols = [
                    "IL_name",
                    "IL_SMILES",
                    "IL_molratio",
                    "IL_to_nucleicacid_massratio",
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
                ]

                encoding_cols = []
                for role in ["IL", "HL", "CHL", "PEG"]:
                    role_cols = [
                        c
                        for c in df.columns
                        if c.startswith(f"{role}_mfp_")
                        or c.startswith(f"{role}_mordred_")
                        or c.startswith(f"{role}_lion_")
                    ]
                    encoding_cols.extend(sorted(role_cols))

                final_cols = ordered_base_cols + encoding_cols
                final_cols = [c for c in final_cols if c in df.columns]

                df[final_cols].to_csv(encoding_csv_path, index=False)

        return Dataset(
            df,
            source=self.source,
            metadata=metadata,
            name=self.name,
            encoders=encoders,
            fitted_transformers=fitted_transformers,
        )

    def load_encodings(self, encoding_csv_path: str) -> Dataset:
        enc = pd.read_csv(encoding_csv_path)
        assert len(self.df) == len(enc), (
            f"Row count mismatch: dataset has {len(self.df)} rows but encodings have {len(enc)} rows"
        )
        df = pd.concat([self.df.reset_index(drop=True), enc], axis=1)
        return Dataset(
            df,
            source=self.source,
            metadata=self.metadata,
            name=self.name,
            encoders=self.encoders,
            fitted_transformers=self.fitted_transformers,
        )

    @classmethod
    def from_ingredient_library(cls, path: str) -> Dataset:
        """
        Ingredient-only input (no Experiment_value).
        Intended ONLY for DoE initialization.
        """
        df = pd.read_csv(path)

        # copy only user-defined DOE options
        present = [c for c in INGREDIENT_COLUMNS if c in df.columns]
        df = df[present].copy()

        df["Experiment_value"] = pd.NA
        df["Round"] = 0
        df["Formulation_ID"] = -1  # assigned later

        return cls(df, source="ingredients")

    @classmethod
    def from_doe_options(cls, path: str) -> dict[str, dict]:
        df = pd.read_csv(path)

        options = {}
        for _, row in df.iterrows():
            role = row["role"]
            options[role] = {
                "names": row["names"].split(";") if pd.notna(row["names"]) else None,
                "ratios": ([float(x) for x in row["ratios"].split(";")] if pd.notna(row["ratios"]) else None),
                "bounds": ((row["ratio_lower"], row["ratio_upper"]) if pd.notna(row["ratio_lower"]) else None),
            }
        return options

    def validate_schema(self, strict: bool = True):
        missing = set(LNPDB_REQUIRED_COLUMNS) - set(self.df.columns)
        if missing and strict:
            raise ValueError(f"Dataset missing required columns: {missing}")
        return True

    def _ensure_df(self, x, prefix):
        if isinstance(x, pd.DataFrame):
            return x.add_prefix(prefix)
        return pd.DataFrame(x, columns=[f"{prefix}pc{i + 1}" for i in range(x.shape[1])])

    def max_round(self) -> int:
        return int(self.df["Round"].max()) if len(self.df) else 0

    def next_formulation_id(self) -> int:
        return int(self.df["Formulation_ID"].max()) + 1 if len(self.df) else 0

    def append_suggestions(self, df_new: pd.DataFrame) -> Dataset:
        """
        Append completed suggested formulations.

        - Copies encoding columns if lipid already exists
        - Prevents duplicates
        - Preserves modeling columns
        """

        df_new = df_new.copy()

        required = {"Formulation_ID", "Round", "Experiment_value"}
        missing = required - set(df_new.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        if df_new["Experiment_value"].isna().any():
            raise ValueError("All appended rows must have non-null Experiment_value.")

        existing_ids = set(self.df["Formulation_ID"])
        duplicate_ids = existing_ids.intersection(set(df_new["Formulation_ID"]))
        if duplicate_ids:
            raise ValueError(f"Duplicate Formulation_ID detected: {duplicate_ids}")

        roles = ["IL", "HL", "CHL", "PEG"]

        for role in roles:
            name_col = f"{role}_name"

            # Find encoding columns for this role
            encoding_cols = [
                c
                for c in self.df.columns
                if c.startswith(f"{role}_mfp_") or c.startswith(f"{role}_mordred_") or c.startswith(f"{role}_lion_")
            ]

            if not encoding_cols:
                continue  # nothing to copy

            # Build lookup table: lipid_name -> encoding row
            lookup = (
                self.df[[name_col, *encoding_cols]]
                .dropna(subset=[name_col])
                .drop_duplicates(subset=[name_col])
                .set_index(name_col)
            )

            # Drop any columns from df_new that overlap with lookup to prevent _x/_y suffixes
            overlap_cols = [c for c in df_new.columns if c in lookup.columns]
            if overlap_cols:
                df_new = df_new.drop(columns=overlap_cols)

            # Merge encodings into df_new
            df_new = df_new.merge(
                lookup,
                left_on=name_col,
                right_index=True,
                how="left",
            )

        feature_cols = [c for c in columns_to_check_for_duplicates if c in df_new.columns and c in self.df.columns]

        combined_check = pd.concat([self.df[feature_cols], df_new[feature_cols]], ignore_index=True)

        if combined_check.duplicated().any():
            raise ValueError("Duplicate formulations detected when appending suggestions.")

        all_cols = sorted(set(self.df.columns) | set(df_new.columns))

        df_old = self.df.reindex(columns=all_cols)
        df_new = df_new.reindex(columns=all_cols)

        combined = pd.concat([df_old, df_new], ignore_index=True)

        return Dataset(
            combined,
            source=self.source,
            metadata=self.metadata,
            name=self.name,
            encoders=self.encoders,
            fitted_transformers=self.fitted_transformers,
        )

    def to_csv(self, path: str):
        self.df.to_csv(path, index=False)
