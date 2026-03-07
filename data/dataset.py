import os
import warnings
from typing import Self

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

_ENC_PREFIXES = ("mfp_", "mordred_", "lion_", "unimol_", "count_mfp_", "rdkit_", "chemeleon_")

_FEATURE_TYPE_ENCODERS = {
    "mfp": ["mfp"],
    "count_mfp": ["count_mfp"],
    "rdkit": ["rdkit"],
    "mordred": ["mordred"],
    "unimol": ["unimol"],
    "chemeleon": ["chemeleon"],
    "lantern": ["count_mfp", "rdkit"],
}

# Maps encoder key -> flat kwarg suffix used in the deprecated API.
# Most are identity; the exception is "mfp" -> "morgan".
_ENCODER_TO_KWARG_SUFFIX = {
    "mfp": "morgan",
    "count_mfp": "count_mfp",
    "rdkit": "rdkit",
    "mordred": "mordred",
    "unimol": "unimol",
    "chemeleon": "chemeleon",
    "lion": "lion",
}

_ROLES = ("IL", "HL", "CHL", "PEG")


def encoders_for_feature_type(feature_type, il_pcs=5, other_pcs=3):
    """Build a nested ``encoders`` dict for a named feature type.

    Returns ``dict[str, dict[str, int]]`` suitable for passing as
    ``encoders=`` to ``Dataset.encode_dataset()``.

    Example::

        encoders_for_feature_type("lantern", il_pcs=5, other_pcs=0)
        # => {"IL": {"count_mfp": 5, "rdkit": 5},
        #     "HL": {"count_mfp": 0, "rdkit": 0}, ...}
    """
    encoder_keys = _FEATURE_TYPE_ENCODERS.get(feature_type)
    if encoder_keys is None:
        raise ValueError(f"Unknown feature type: {feature_type!r}")
    enc: dict[str, dict[str, int]] = {}
    for role in _ROLES:
        n = il_pcs if role == "IL" else other_pcs
        enc[role] = {k: n for k in encoder_keys}
    return enc


def encode_kwargs_for_feature_type(feature_type, il_pcs=5, other_pcs=3):
    """Map feature type name to flat ``encode_dataset()`` keyword arguments.

    .. deprecated::
        Use :func:`encoders_for_feature_type` and pass the result as
        ``encoders=`` to ``encode_dataset()`` instead.
    """
    encoder_keys = _FEATURE_TYPE_ENCODERS.get(feature_type)
    if encoder_keys is None:
        raise ValueError(f"Unknown feature type: {feature_type!r}")
    kwargs = {}
    for role in _ROLES:
        n = il_pcs if role == "IL" else other_pcs
        for k in encoder_keys:
            kwarg_suffix = _ENCODER_TO_KWARG_SUFFIX[k]
            kwargs[f"{role}_n_pcs_{kwarg_suffix}"] = n
    return kwargs


def _flat_kwargs_to_encoders(
    IL_n_pcs_morgan, IL_n_pcs_mordred, IL_n_pcs_lion, IL_n_pcs_unimol,
    IL_n_pcs_count_mfp, IL_n_pcs_rdkit, IL_n_pcs_chemeleon,
    HL_n_pcs_morgan, HL_n_pcs_mordred, HL_n_pcs_unimol,
    HL_n_pcs_count_mfp, HL_n_pcs_rdkit, HL_n_pcs_chemeleon,
    CHL_n_pcs_morgan, CHL_n_pcs_mordred, CHL_n_pcs_unimol,
    CHL_n_pcs_count_mfp, CHL_n_pcs_rdkit, CHL_n_pcs_chemeleon,
    PEG_n_pcs_morgan, PEG_n_pcs_mordred, PEG_n_pcs_unimol,
    PEG_n_pcs_count_mfp, PEG_n_pcs_rdkit, PEG_n_pcs_chemeleon,
):
    """Convert the 24 individual n_pcs kwargs into a nested encoders dict."""
    return {
        "IL": {
            "mfp": IL_n_pcs_morgan, "mordred": IL_n_pcs_mordred,
            "lion": IL_n_pcs_lion, "unimol": IL_n_pcs_unimol,
            "count_mfp": IL_n_pcs_count_mfp, "rdkit": IL_n_pcs_rdkit,
            "chemeleon": IL_n_pcs_chemeleon,
        },
        "HL": {
            "mfp": HL_n_pcs_morgan, "mordred": HL_n_pcs_mordred,
            "unimol": HL_n_pcs_unimol, "count_mfp": HL_n_pcs_count_mfp,
            "rdkit": HL_n_pcs_rdkit, "chemeleon": HL_n_pcs_chemeleon,
        },
        "CHL": {
            "mfp": CHL_n_pcs_morgan, "mordred": CHL_n_pcs_mordred,
            "unimol": CHL_n_pcs_unimol, "count_mfp": CHL_n_pcs_count_mfp,
            "rdkit": CHL_n_pcs_rdkit, "chemeleon": CHL_n_pcs_chemeleon,
        },
        "PEG": {
            "mfp": PEG_n_pcs_morgan, "mordred": PEG_n_pcs_mordred,
            "unimol": PEG_n_pcs_unimol, "count_mfp": PEG_n_pcs_count_mfp,
            "rdkit": PEG_n_pcs_rdkit, "chemeleon": PEG_n_pcs_chemeleon,
        },
    }


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
        raw_fingerprints=None,
    ):
        self.df = df.copy()
        self.source = source
        self.metadata = metadata or {}
        self.name = name
        self.encoders = encoders
        self.fitted_transformers = fitted_transformers or {}
        self.raw_fingerprints = raw_fingerprints or {}

    @classmethod
    def from_lnpdb_csv(cls, path: str) -> Self:
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
        encoders: dict[str, dict[str, int]] | None = None,
        *,
        IL_n_pcs_morgan: int = 0,
        IL_n_pcs_mordred: int = 0,
        IL_n_pcs_lion: int = 0,
        IL_n_pcs_unimol: int = 0,
        IL_n_pcs_count_mfp: int = 0,
        IL_n_pcs_rdkit: int = 0,
        IL_n_pcs_chemeleon: int = 0,
        HL_n_pcs_morgan: int = 0,
        HL_n_pcs_mordred: int = 0,
        HL_n_pcs_unimol: int = 0,
        HL_n_pcs_count_mfp: int = 0,
        HL_n_pcs_rdkit: int = 0,
        HL_n_pcs_chemeleon: int = 0,
        CHL_n_pcs_morgan: int = 0,
        CHL_n_pcs_mordred: int = 0,
        CHL_n_pcs_unimol: int = 0,
        CHL_n_pcs_count_mfp: int = 0,
        CHL_n_pcs_rdkit: int = 0,
        CHL_n_pcs_chemeleon: int = 0,
        PEG_n_pcs_morgan: int = 0,
        PEG_n_pcs_mordred: int = 0,
        PEG_n_pcs_unimol: int = 0,
        PEG_n_pcs_count_mfp: int = 0,
        PEG_n_pcs_rdkit: int = 0,
        PEG_n_pcs_chemeleon: int = 0,
        encoding_csv_path: str | None = None,
        only_encodings: bool = False,
        reduction: str = "pca",
        fitted_transformers_in: dict | None = None,
        fp_radius: int | None = None,
        fp_bits: int | None = None,
    ) -> Dataset:

        # Build the canonical encoders dict. If ``encoders`` is provided, use
        # it directly. Otherwise fall back to the individual kwargs (backward
        # compat — these are deprecated but still functional).
        any_kwarg_set = any(v != 0 for v in [
            IL_n_pcs_morgan, IL_n_pcs_mordred, IL_n_pcs_lion, IL_n_pcs_unimol,
            IL_n_pcs_count_mfp, IL_n_pcs_rdkit, IL_n_pcs_chemeleon,
            HL_n_pcs_morgan, HL_n_pcs_mordred, HL_n_pcs_unimol,
            HL_n_pcs_count_mfp, HL_n_pcs_rdkit, HL_n_pcs_chemeleon,
            CHL_n_pcs_morgan, CHL_n_pcs_mordred, CHL_n_pcs_unimol,
            CHL_n_pcs_count_mfp, CHL_n_pcs_rdkit, CHL_n_pcs_chemeleon,
            PEG_n_pcs_morgan, PEG_n_pcs_mordred, PEG_n_pcs_unimol,
            PEG_n_pcs_count_mfp, PEG_n_pcs_rdkit, PEG_n_pcs_chemeleon,
        ])

        if encoders is not None and any_kwarg_set:
            raise ValueError(
                "Cannot specify both 'encoders' and individual {ROLE}_n_pcs_{encoder} kwargs. "
                "Use one or the other."
            )

        if encoders is None:
            if any_kwarg_set:
                warnings.warn(
                    "Passing individual {ROLE}_n_pcs_{encoder} kwargs to encode_dataset() "
                    "is deprecated. Use encoders=encoders_for_feature_type(...) or pass "
                    "a dict[str, dict[str, int]] via the 'encoders' parameter instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
            encoders = _flat_kwargs_to_encoders(
                IL_n_pcs_morgan, IL_n_pcs_mordred, IL_n_pcs_lion, IL_n_pcs_unimol,
                IL_n_pcs_count_mfp, IL_n_pcs_rdkit, IL_n_pcs_chemeleon,
                HL_n_pcs_morgan, HL_n_pcs_mordred, HL_n_pcs_unimol,
                HL_n_pcs_count_mfp, HL_n_pcs_rdkit, HL_n_pcs_chemeleon,
                CHL_n_pcs_morgan, CHL_n_pcs_mordred, CHL_n_pcs_unimol,
                CHL_n_pcs_count_mfp, CHL_n_pcs_rdkit, CHL_n_pcs_chemeleon,
                PEG_n_pcs_morgan, PEG_n_pcs_mordred, PEG_n_pcs_unimol,
                PEG_n_pcs_count_mfp, PEG_n_pcs_rdkit, PEG_n_pcs_chemeleon,
            )

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
        il_enc = encoders.get("IL", {})
        if il_enc.get("lion", 0) > 0 and (il_enc.get("mfp", 0) > 0 or il_enc.get("mordred", 0) > 0):
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
        raw_fingerprints = {}

        def encode_lipid_table(lipid_df: pd.DataFrame, role: str, pcs_spec: dict):

            blocks = []
            smiles = lipid_df["SMILES"].tolist()

            # Per-lipid average target values (needed for LiON and PLS fitting)
            exp_map = None
            needs_targets = pcs_spec.get("lion", 0) > 0 or reduction == "pls"
            # Skip target computation when all encoders have pre-fitted reducers
            # (transform-only mode for pool encoding — targets not needed)
            has_all_fitted = fitted_transformers_in is not None and all(
                f"{role}_{et}" in fitted_transformers_in
                for et, n in pcs_spec.items() if n > 0
            )
            if needs_targets and not has_all_fitted:
                exp_map = average_experiment_values(role)
                if not exp_map:
                    raise ValueError(
                        f"Target values required for {role} encoding (reduction={reduction!r}) but none available."
                    )

            for enc_type, n in pcs_spec.items():
                if n <= 0:
                    continue

                # Reuse pre-fitted transformers if available (for consistent pool encoding)
                existing = fitted_transformers_in.get(f"{role}_{enc_type}") if fitted_transformers_in else None

                exp_vals = None
                if (enc_type == "lion" or reduction == "pls") and existing is None:
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

                fp_kw = {}
                if enc_type in ("mfp", "count_mfp"):
                    if fp_radius is not None:
                        fp_kw["fp_radius"] = fp_radius
                    if fp_bits is not None:
                        fp_kw["fp_bits"] = fp_bits

                pc_matrix, reducer_obj, scaler_obj, fp_scaled = compute_pcs(
                    smiles,
                    feature_type=enc_type,
                    n_components=n,
                    experiment_values=exp_vals,
                    reduction=reduction,
                    cache_name=role,
                    fitted_reducer=existing["reducer"] if existing else None,
                    fitted_scaler=existing["scaler"] if existing else None,
                    **fp_kw,
                )

                fitted_transformers[f"{role}_{enc_type}"] = {
                    "reducer": reducer_obj,
                    "scaler": scaler_obj,
                }

                if reduction == "pls" and fp_scaled is not None:
                    raw_fingerprints[f"{role}_{enc_type}"] = {
                        "fp_scaled": fp_scaled,
                        "smiles": smiles,
                        "lipid_names": lipid_df["lipid_name"].tolist(),
                        "n_components": n,
                    }

                cols = [f"{role}_{enc_type}_pc{i + 1}" for i in range(pc_matrix.shape[1])]

                blocks.append(pd.DataFrame(pc_matrix, columns=cols))

            if not blocks:
                return None

            return pd.concat([lipid_df.reset_index(drop=True), *blocks], axis=1)

        encoding_tables = []

        for role, pcs_spec in encoders.items():
            if not variable_components.get(role, False):
                continue

            lipid_df = unique_lipids(role)
            enc_df = encode_lipid_table(lipid_df, role, pcs_spec)

            if enc_df is not None:
                encoding_tables.append(enc_df)
                metadata["pcs"][role] = pcs_spec

                # Merge on (name, SMILES) to prevent many-to-many expansion
                # when the same lipid name maps to multiple SMILES.
                smiles_col = f"{role}_SMILES"
                merge_right = enc_df.drop(columns=["lipid_type"])
                if smiles_col in df.columns:
                    n_before = len(df)
                    df = df.merge(
                        merge_right,
                        left_on=[f"{role}_name", smiles_col],
                        right_on=["lipid_name", "SMILES"],
                        how="left",
                    ).drop(columns=["lipid_name", "SMILES"])
                    if len(df) != n_before:
                        raise RuntimeError(
                            f"Merge on ({role}_name, {smiles_col}) changed row count: "
                            f"{n_before} -> {len(df)}. Data has duplicate (name, SMILES) pairs."
                        )
                else:
                    df = df.merge(
                        merge_right.drop(columns=["SMILES"]),
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
                enc_prefixes = _ENC_PREFIXES
                for role in ["IL", "HL", "CHL", "PEG"]:
                    role_cols = [
                        c for c in df.columns
                        if any(c.startswith(f"{role}_{p}") for p in enc_prefixes)
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
            raw_fingerprints=raw_fingerprints,
        )

    def max_round(self) -> int:
        return int(self.df["Round"].max()) if len(self.df) else 0

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
            enc_prefixes = _ENC_PREFIXES
            encoding_cols = [
                c for c in self.df.columns
                if any(c.startswith(f"{role}_{p}") for p in enc_prefixes)
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
            raw_fingerprints=self.raw_fingerprints,
        )

    def refit_pls(self, training_indices: list[int], external_df=None) -> None:
        """Re-fit PLS using only training-set targets (prospective PLS).

        Computes per-lipid average Experiment_value from training rows only,
        re-fits PLSRegression on stored raw fingerprints, and updates PC
        columns in self.df in-place. If ``external_df`` is provided, the
        same PC columns are also updated there (avoids callers needing to
        manually sync columns after calling this method).

        This avoids target leakage: PLS is fit on observed targets only,
        not on the full oracle dataset.
        """
        import numpy as np
        from sklearn.cross_decomposition import PLSRegression

        if not self.raw_fingerprints:
            return

        train_df = self.df.loc[training_indices]

        for key, raw_data in self.raw_fingerprints.items():
            role, enc_type = key.split("_", 1)
            name_col = f"{role}_name"
            lipid_names = raw_data["lipid_names"]
            fp_scaled = raw_data["fp_scaled"]
            n_components = raw_data["n_components"]

            # Compute per-lipid average target from training data only
            train_means = train_df.dropna(subset=["Experiment_value"]).groupby(name_col)["Experiment_value"].mean()

            train_mask = []
            train_vals = []
            for name in lipid_names:
                v = train_means.get(name)
                if v is None:
                    train_mask.append(False)
                else:
                    train_mask.append(True)
                    train_vals.append(float(v))

            train_mask = np.asarray(train_mask, dtype=bool)
            if train_mask.sum() < 2:
                # Not enough labeled lipids to refit PLS for this role
                continue

            fp_train = fp_scaled[train_mask]
            y = np.asarray(train_vals, dtype=float)
            max_components = min(fp_train.shape[0], fp_train.shape[1])
            max_components = max(max_components - 1, 1)
            n_comp = min(n_components, max_components)

            reducer = PLSRegression(n_components=n_comp, scale=False)
            try:
                reducer.fit(fp_train, y)
                pc_matrix = reducer.transform(fp_scaled)
            except (ValueError, np.linalg.LinAlgError):
                # PLS can fail with degenerate features (e.g., 3 unique PEG SMILES);
                # skip re-fitting for this role and keep existing PC values
                continue

            # Update fitted_transformers
            if key in self.fitted_transformers:
                self.fitted_transformers[key]["reducer"] = reducer

            # Build lipid_name -> PC row lookup
            pc_lookup = {name: row for name, row in zip(lipid_names, pc_matrix)}
            cols = [f"{role}_{enc_type}_pc{i + 1}" for i in range(pc_matrix.shape[1])]

            # Update PC columns in self.df (and external_df if provided)
            col_map = {name: pc_lookup[name] for name in pc_lookup}
            for col_idx, col_name in enumerate(cols):
                mapping = {name: row[col_idx] for name, row in col_map.items()}
                self.df[col_name] = self.df[name_col].map(mapping)
                if external_df is not None and col_name in external_df.columns:
                    external_df[col_name] = self.df[col_name]

    def to_csv(self, path: str):
        self.df.to_csv(path, index=False)
