from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances

from ..data.dataset import Dataset
from ..space.formulation import FormulationSpace
from .bayesopt import perform_bayesian_optimization
from .doe import mixture_doe

DISCRETE_SURROGATES = {"xgb", "xgb_ucb", "rf_ucb", "rf_ts", "gp_ucb"}
ENC_PREFIXES = ["mfp_pc", "mordred_pc", "unimol_pc", "lion_pc", "count_mfp_pc", "rdkit_pc", "chemeleon_pc"]
CTX_PREFIX = "ctx_"


class Optimizer:
    """
    Minimal Bayesian optimizer for LNP design.
    Iteratively suggests new formulations using FormulationSpace as the source of truth.
    """

    def __init__(
        self,
        space: FormulationSpace,
        type: str = "UCB",
        kappa: float = 5.0,
        xi: float = 0.01,
        alpha: float = 1e-6,
        random_seed: int = 1,
        batch_size: int = 24,
        surrogate: str = "gp",
        candidate_pool: pd.DataFrame | None = None,
        context_features: bool = False,
    ):
        self.space = space
        self.type = type
        self.kappa = kappa
        self.xi = xi
        self.alpha = alpha
        self.random_seed = random_seed
        self.batch_size = batch_size
        self.surrogate = surrogate
        self.candidate_pool = candidate_pool
        self.context_features = context_features

    def suggest(self, output_csv: str | None = None) -> pd.DataFrame:
        if self.surrogate != "gp":
            return self.suggest_discrete(output_csv=output_csv)
        return self._suggest_gp(output_csv=output_csv)

    def suggest_discrete(self, output_csv: str | None = None) -> pd.DataFrame:
        """Suggest formulations by scoring a discrete candidate pool."""
        from .discrete import score_candidate_pool

        dataset = self.space._dataset
        assert dataset is not None
        assert self.candidate_pool is not None, "candidate_pool required for discrete surrogates"

        round_number = dataset.max_round() + 1

        # Identify feature columns from encoding prefixes
        enc_prefixes = ENC_PREFIXES
        feature_cols = []
        for role in ["IL", "HL", "CHL", "PEG"]:
            for prefix in enc_prefixes:
                role_cols = [c for c in dataset.df.columns if c.startswith(f"{role}_{prefix}")]
                feature_cols.extend(sorted(role_cols))
        for role in ["IL", "HL", "CHL", "PEG"]:
            col = f"{role}_molratio"
            if col in dataset.df.columns and dataset.df[col].nunique() > 1:
                feature_cols.append(col)
        if "IL_to_nucleicacid_massratio" in dataset.df.columns and dataset.df["IL_to_nucleicacid_massratio"].nunique() > 1:
            feature_cols.append("IL_to_nucleicacid_massratio")

        # Add one-hot encoded experimental context features
        ctx_levels = None
        if self.context_features:
            from ..data.context import encode_context

            dataset.df, ctx_cols, ctx_levels = encode_context(dataset.df)
            feature_cols.extend(ctx_cols)

        # Exclude already-evaluated Formulation_IDs from pool
        evaluated_ids = set(dataset.df["Formulation_ID"].dropna().astype(int))
        pool_df = self.candidate_pool[
            ~self.candidate_pool["Formulation_ID"].isin(evaluated_ids)
        ].copy()

        if pool_df.empty:
            raise ValueError("No candidates remaining in pool after excluding evaluated formulations.")

        # Encode context on pool using same levels as training data
        if self.context_features and ctx_levels is not None:
            from ..data.context import encode_context

            pool_df, _, _ = encode_context(pool_df, levels=ctx_levels)

        # Ensure feature columns exist in pool
        missing = [c for c in feature_cols if c not in pool_df.columns]
        if missing:
            raise ValueError(f"Candidate pool missing feature columns: {missing}")

        # Drop rows with missing feature values (training + pool)
        train_mask = dataset.df[feature_cols].notna().all(axis=1)
        if not train_mask.all():
            n_drop = int((~train_mask).sum())
            print(f"  Dropped {n_drop} training rows with missing features")
        train_df = dataset.df.loc[train_mask].copy()

        # Deduplicate pool by composition to avoid suggesting the same formulation twice.
        # Use lipid names + ratios (what the scientist actually makes), not feature vectors.
        composition_cols = [c for c in [
            "IL_name", "HL_name", "CHL_name", "PEG_name",
            "IL_molratio", "HL_molratio", "CHL_molratio", "PEG_molratio",
            "IL_to_nucleicacid_massratio",
        ] if c in pool_df.columns]
        if composition_cols:
            n_before = len(pool_df)
            pool_df = pool_df.drop_duplicates(subset=composition_cols).reset_index(drop=True)
            n_deduped = n_before - len(pool_df)
            if n_deduped > 0:
                print(f"  Deduplicated pool: {n_before:,} -> {len(pool_df):,} ({n_deduped:,} duplicates removed)")

        pool_mask = pool_df[feature_cols].notna().all(axis=1)
        if not pool_mask.all():
            n_drop = int((~pool_mask).sum())
            print(f"  Dropped {n_drop} pool rows with missing features")
            pool_df = pool_df.loc[pool_mask].copy()

        if pool_df.empty or train_df.empty:
            raise ValueError("No valid rows available after dropping missing features.")

        X_train = train_df[feature_cols].values
        y_train = train_df["Experiment_value"].values
        X_pool = pool_df[feature_cols].values

        top_indices, _ = score_candidate_pool(
            X_train, y_train, X_pool,
            surrogate=self.surrogate,
            batch_size=self.batch_size,
            kappa=self.kappa,
            random_seed=self.random_seed,
        )

        # Build batch dataframe from pool rows
        selected = pool_df.iloc[top_indices].copy()
        df_old = dataset.df.copy()

        if "Formulation_ID" not in df_old.columns:
            df_old["Formulation_ID"] = np.nan
        start_id = int(df_old["Formulation_ID"].max()) if df_old["Formulation_ID"].notna().any() else 0

        selected["Formulation_ID"] = np.arange(start_id + 1, start_id + 1 + len(selected))
        selected["Round"] = round_number
        selected["Experiment_value"] = np.nan

        # Combine old + new
        all_cols = sorted(set(df_old.columns) | set(selected.columns))
        df_old = df_old.reindex(columns=all_cols)
        selected = selected.reindex(columns=all_cols)
        df_final = pd.concat([df_old, selected], ignore_index=True)

        # Column ordering
        df_final = self._order_columns(df_final)

        if output_csv is not None:
            df_final.to_csv(output_csv, index=False)
            print(f"Suggested formulations written to {output_csv}")

        return df_final

    def _suggest_gp(self, output_csv: str | None = None) -> pd.DataFrame:
        """Original GP-based suggestion path."""
        dataset = self.space._dataset
        assert dataset is not None

        round_number = dataset.max_round() + 1

        meta = dataset.metadata
        any_components = any(meta["variable_components"].values())
        variable_ratios = sum(meta["variable_molratios"].values())

        if not any_components and variable_ratios < 2:
            print("No BO variables detected -> falling back to DoE")
            df_batch = pd.DataFrame(
                mixture_doe(
                    n_samples=self.batch_size,
                    components=["IL", "HL", "CHL", "PEG"],
                    bounds=self.space.molratio_bounds,
                    seed=self.random_seed,
                )
            )
        else:
            df_batch = perform_bayesian_optimization(
                data=dataset.df,
                formulation_space=self.space,
                round_number=round_number,
                acq_type=self.type,
                BATCH_SIZE=self.batch_size,
                RANDOM_STATE_SEED=self.random_seed,
                KAPPA=self.kappa,
                XI=self.xi,
                ALPHA=self.alpha,
            )

        df_old = dataset.df.copy()

        def decode_component(df_new, role, space):
            configs = space.get_configs()
            comp_params = [p for p in configs["parameters"] if p["type"] == "ComponentParameter" and p["name"] == role]
            name_col = f"{role}_name"
            smiles_col = f"{role}_SMILES"

            if not comp_params:
                df_new[name_col] = dataset.df[name_col].iloc[0]
                if smiles_col in dataset.df.columns:
                    df_new[smiles_col] = dataset.df[smiles_col].iloc[0]
                return df_new

            p = comp_params[0]
            feature_cols = p["columns"]

            ref_cols = [*feature_cols, name_col]
            if smiles_col in dataset.df.columns:
                ref_cols.append(smiles_col)

            ref = dataset.df[ref_cols].dropna().drop_duplicates(subset=feature_cols).reset_index(drop=True)

            X_ref = ref[feature_cols].to_numpy()
            names = ref[f"{role}_name"].tolist()
            smiles = ref[smiles_col].tolist() if smiles_col in ref.columns else None

            def decode_row(row):
                x = row[feature_cols].to_numpy().reshape(1, -1)
                idx = pairwise_distances(x, X_ref).argmin()
                if smiles is not None:
                    return names[idx], smiles[idx]
                return names[idx], np.nan

            decoded = df_new.apply(decode_row, axis=1, result_type="expand")
            df_new[name_col] = decoded.iloc[:, 0]
            df_new[smiles_col] = decoded.iloc[:, 1]
            return df_new

        for role in ["IL", "HL", "CHL", "PEG"]:
            df_batch = decode_component(df_batch, role, self.space)

        if not df_old.empty:
            ratio_cols = [c for c in df_batch.columns if c.endswith("_molratio")]
            ratio_cols.append("IL_to_nucleicacid_massratio")
            for col in ratio_cols:
                if col in df_old.columns:
                    reference_value = df_old[col].iloc[0]
                    if col in df_batch.columns:
                        df_batch[col] = df_batch[col].fillna(reference_value)
                    else:
                        df_batch[col] = reference_value

        if "Formulation_ID" not in df_old.columns:
            df_old["Formulation_ID"] = np.nan

        start_id = int(df_old["Formulation_ID"].max()) if df_old["Formulation_ID"].notna().any() else 0

        df_batch["Formulation_ID"] = np.arange(start_id + 1, start_id + 1 + len(df_batch))

        df_batch["Round"] = (
            int(df_old["Round"].max()) + 1 if "Round" in df_old.columns and df_old["Round"].notna().any() else 1
        )

        df_batch["Experiment_value"] = np.nan

        all_cols = sorted(set(df_old.columns) | set(df_batch.columns))
        df_old = df_old.reindex(columns=all_cols)
        df_batch = df_batch.reindex(columns=all_cols)

        df_final = pd.concat([df_old, df_batch], ignore_index=True)
        df_final = self._order_columns(df_final)

        if output_csv is not None:
            df_final.to_csv(output_csv, index=False)
            print(f"Suggested formulations written to {output_csv}")

        return df_final

    @staticmethod
    def _order_columns(df_final: pd.DataFrame) -> pd.DataFrame:
        def role_block(role: str):
            return [
                f"{role}_name",
                f"{role}_SMILES",
                f"{role}_molratio",
            ]

        ordered_cols = []
        ordered_cols += ["Formulation_ID", "Round"]
        ordered_cols += [
            "IL_name",
            "IL_SMILES",
            "IL_molratio",
            "IL_to_nucleicacid_massratio",
        ]
        for role in ["HL", "CHL", "PEG"]:
            ordered_cols += role_block(role)
        ordered_cols += ["Experiment_value"]

        for role in ["IL", "HL", "CHL", "PEG"]:
            enc_cols = [
                c
                for c in df_final.columns
                if c.startswith(f"{role}_mfp_")
                or c.startswith(f"{role}_mordred_")
                or c.startswith(f"{role}_lion_")
                or c.startswith(f"{role}_count_mfp_")
                or c.startswith(f"{role}_rdkit_")
                or c.startswith(f"{role}_unimol_")
                or c.startswith(f"{role}_chemeleon_")
            ]
            ordered_cols += sorted(enc_cols)

        remaining = [c for c in df_final.columns if c not in ordered_cols]
        ordered_cols += remaining

        # Filter to columns that actually exist
        ordered_cols = [c for c in ordered_cols if c in df_final.columns]
        return df_final[ordered_cols]
