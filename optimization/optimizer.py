from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances

from ..data.dataset import Dataset
from ..space.formulation import FormulationSpace
from .bayesopt import perform_bayesian_optimization
from .doe import mixture_doe


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
    ):
        self.space = space
        self.type = type
        self.kappa = kappa
        self.xi = xi
        self.alpha = alpha
        self.random_seed = random_seed
        self.batch_size = batch_size

        np.random.seed(random_seed)

    def suggest(self, output_csv: str | None = None) -> pd.DataFrame:
        """
        Suggest a batch of new formulations.

        Returns a DataFrame of suggested formulations. Optionally writes CSV.

        :param self: Optimizer object
        :param output_csv: suggestion save path
        :type output_csv: Optional[str]
        :return: suggestion dataframe
        :rtype: DataFrame
        """
        dataset = self.space._dataset
        assert dataset is not None

        round_number = dataset.max_round() + 1

        meta = dataset.metadata
        any_components = any(meta["variable_components"].values())
        variable_ratios = sum(meta["variable_molratios"].values())

        if not any_components and variable_ratios < 2:
            print("No BO variables detected → falling back to DoE")
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
            """
            Decode lipid name via nearest-neighbor in encoding space.
            Falls back to fixed component name when not variable.
            """
            configs = space.get_configs()

            # Locate ComponentParameter for this role
            comp_params = [p for p in configs["parameters"] if p["type"] == "ComponentParameter" and p["name"] == role]

            name_col = f"{role}_name"
            smiles_col = f"{role}_SMILES"

            # Fixed component -> copy name
            if not comp_params:
                df_new[name_col] = dataset.df[name_col].iloc[0]
                if smiles_col in dataset.df.columns:
                    df_new[smiles_col] = dataset.df[smiles_col].iloc[0]
                return df_new

            # Variable component -> decode
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

        # Mark new rows + add empty experiment column
        if "Formulation_ID" not in df_old.columns:
            df_old["Formulation_ID"] = np.nan

        start_id = int(df_old["Formulation_ID"].max()) if df_old["Formulation_ID"].notna().any() else 0

        df_batch["Formulation_ID"] = np.arange(start_id + 1, start_id + 1 + len(df_batch))

        df_batch["Round"] = (
            int(df_old["Round"].max()) + 1 if "Round" in df_old.columns and df_old["Round"].notna().any() else 1
        )

        df_batch["Experiment_value"] = np.nan

        # Concatenate vertically (old + new)
        all_cols = sorted(set(df_old.columns) | set(df_batch.columns))
        df_old = df_old.reindex(columns=all_cols)
        df_batch = df_batch.reindex(columns=all_cols)

        df_final = pd.concat([df_old, df_batch], ignore_index=True)

        # CLEAN COLUMN ORDERING
        def role_block(role: str):
            return [
                f"{role}_name",
                f"{role}_SMILES",
                f"{role}_molratio",
            ]

        ordered_cols = []

        # Meta first
        ordered_cols += ["Formulation_ID", "Round"]

        # IL block (special case: massratio)
        ordered_cols += [
            "IL_name",
            "IL_SMILES",
            "IL_molratio",
            "IL_to_nucleicacid_massratio",
        ]

        # Other roles
        for role in ["HL", "CHL", "PEG"]:
            ordered_cols += role_block(role)

        # Target
        ordered_cols += ["Experiment_value"]

        # Encoding columns grouped by role
        for role in ["IL", "HL", "CHL", "PEG"]:
            enc_cols = [
                c
                for c in df_final.columns
                if c.startswith(f"{role}_mfp_") or c.startswith(f"{role}_mordred_") or c.startswith(f"{role}_lion_")
            ]
            ordered_cols += sorted(enc_cols)

        # Append any remaining columns safely
        remaining = [c for c in df_final.columns if c not in ordered_cols]
        ordered_cols += remaining

        df_final = df_final[ordered_cols]

        # -------------------------------------------------
        # Write CSV
        # -------------------------------------------------

        if output_csv is not None:
            df_final.to_csv(output_csv, index=False)
            print(f"Suggested formulations written to {output_csv}")

        return df_final

    def update(self, csv_path: str) -> Dataset:
        """
        Read a filled-in suggestion CSV and append ONLY
        the most recent completed round.
        """

        df = pd.read_csv(csv_path)

        if "Experiment_value" not in df.columns or "Round" not in df.columns:
            raise ValueError("CSV must include 'Experiment_value' and 'Round' columns.")

        # Identify latest round present
        latest_round = df["Round"].max()

        # Select completed rows from that round
        df_new = df[(df["Round"] == latest_round) & df["Experiment_value"].notna()].copy()

        if df_new.empty:
            raise ValueError(f"No completed rows found for latest round ({latest_round}).")

        # Append to dataset
        assert self.space._dataset is not None
        updated_dataset = self.space._dataset.append_suggestions(df_new)

        # Update formulation space
        self.space.update(updated_dataset)

        return updated_dataset

    def state(self) -> dict:
        """
        Return basic internal state for debugging.
        """
        return {
            "type": self.type,
            "batch_size": self.batch_size,
            "kappa": self.kappa,
            "xi": self.xi,
            "alpha": self.alpha,
            "current_round": self.space._round_counter if hasattr(self.space, "_round_counter") else None,
        }
