import random
from typing import Self

import numpy as np

from ..data.dataset import Dataset
from .parameters import (
    ComponentParameter,
    DiscreteParameter,
    MixtureRatiosParameter,
)


class FormulationSpace:
    """
    Defines the Bayesian optimization search space for LNP formulations.
    Fully dataset-driven and role-symmetric.
    """

    ROLES = ("IL", "HL", "CHL", "PEG")

    def __init__(
        self,
        components: dict[str, list[dict]],
        molratio_bounds: dict[str, tuple[float, float]],
        il_mrna_massratio_values: list[float],
        target: str = "Experiment_value",
        dataset=None,
        component_pcs: dict[str, dict[str, int]] | None = None,
        fixed_values: dict[str, float] | None = None,
        normalize_molratios: bool = True,
        random_seed: int | None = None,
        name="screen",
    ):
        self._dataset: Dataset | None = dataset
        self.name = name

        self.components = components
        self.molratio_bounds = molratio_bounds
        self.il_mrna_massratio_values = il_mrna_massratio_values

        self.target = target
        self.component_pcs = component_pcs or {}
        self.fixed_values = fixed_values or {}
        self.normalize_molratios = normalize_molratios

        self._formulation_counter = 0
        self._round_counter = 0

        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

        self._validate_inputs()

        # Build parameter objects for BO
        self.parameters = self._build_parameters()

    def _build_parameters(self):
        """
        Build a list of BayesParameter objects from the current space configuration.
        Mirrors the logic in get_configs() but creates actual parameter objects.
        """
        params = []

        if self._dataset is None or not self._dataset.metadata:
            return params

        metadata = self._dataset.metadata
        encoders = self._dataset.encoders
        df = self._dataset.df

        if not encoders:
            return params

        # Component parameters
        for role in self.ROLES:
            if not metadata["variable_components"].get(role, False):
                continue

            pcs_dict = encoders[role]

            columns = []
            for encoder_name, n_pcs in pcs_dict.items():
                columns += [f"{role}_{encoder_name}_pc{i}" for i in range(1, n_pcs + 1)]

            if not columns:
                continue

            # Extract unique component PC values from the dataset
            valid_cols = [c for c in columns if c in df.columns]
            if not valid_cols:
                continue

            unique_vals = df[valid_cols].drop_duplicates().values
            bounds = np.column_stack(
                [
                    df[valid_cols].min().values,
                    df[valid_cols].max().values,
                ]
            )

            params.append(
                ComponentParameter(
                    name=role,
                    bounds=bounds,
                    valid_options=unique_vals,
                )
            )

        # Mixture ratio parameters
        columns_mixture = []
        for role in self.ROLES:
            if metadata["variable_molratios"].get(role, False):
                columns_mixture.append(f"{role}_molratio")

        if columns_mixture:
            nr_components = len(columns_mixture)
            mr_bounds = np.array(
                [
                    [self.molratio_bounds[role][0], self.molratio_bounds[role][1]]
                    for role in self.ROLES
                    if metadata["variable_molratios"].get(role, False)
                ]
            )
            params.append(
                MixtureRatiosParameter(
                    name="molratio",
                    nr_components=nr_components,
                    bounds=mr_bounds,
                )
            )

        # IL to mRNA mass ratio (discrete)
        if metadata.get("variable_il_mrna", False):
            domain = np.array(self.il_mrna_massratio_values)
            params.append(
                DiscreteParameter(
                    name="IL_to_nucleicacid_massratio",
                    domain=domain,
                )
            )

        return params

    # Validation
    def _validate_inputs(self):
        if set(self.components.keys()) != set(self.ROLES):
            raise ValueError(f"components must contain {self.ROLES}")

        if set(self.molratio_bounds.keys()) != set(self.ROLES):
            raise ValueError(f"molratio_bounds must contain {self.ROLES}")

    @classmethod
    def from_dataset(
        cls,
        dataset: Dataset,
        molratio_bounds_override: dict[str, tuple[float, float]] | None = None,
    ) -> Self:

        df = dataset.df
        meta = dataset.metadata

        if not meta:
            raise ValueError("Dataset must be encoded before FormulationSpace.")

        # Structural PCs
        component_pcs = {role: meta["pcs"].get(role, {}) for role in cls.ROLES if meta["variable_components"].get(role)}

        # Molratio bounds
        molratio_bounds = {}
        fixed_values = {}

        for role in cls.ROLES:
            col = f"{role}_molratio"

            if meta["variable_molratios"][role]:
                if molratio_bounds_override and role in molratio_bounds_override:
                    molratio_bounds[role] = molratio_bounds_override[role]
                else:
                    molratio_bounds[role] = (df[col].min(), df[col].max())
            else:
                val = df[col].iloc[0]
                molratio_bounds[role] = (val, val)
                fixed_values[col] = val

        # IL:mRNA (DISCRETE VALUES)
        il_mrna_values = sorted(df["IL_to_nucleicacid_massratio"].unique().tolist())

        # Components
        def component_records(role):
            cols = [f"{role}_name"]
            if f"{role}_SMILES" in df.columns:
                cols.append(f"{role}_SMILES")
            return (
                df[cols]
                .drop_duplicates()
                .rename(
                    columns={
                        f"{role}_name": "name",
                        f"{role}_SMILES": "smiles",
                    }
                )
                .to_dict("records")
            )

        components = {role: component_records(role) for role in cls.ROLES}

        return cls(
            components=components,
            molratio_bounds=molratio_bounds,
            il_mrna_massratio_values=il_mrna_values,
            component_pcs=component_pcs,
            fixed_values=fixed_values,
            target="Experiment_value",
            dataset=dataset,
            name=dataset.name,
        )

    def update(self, dataset: Dataset):
        """
        Update the internal formulation space from a Dataset.
        - Adds new lipids to components
        - Updates molratio bounds
        - Updates IL:mRNA values
        - Updates component PCs
        - Rebuilds self.parameters
        """
        self._dataset = dataset
        df = dataset.df
        meta = dataset.metadata

        # Update components
        for role in self.ROLES:
            current_names = {c["name"] for c in self.components[role]}
            # find new unique lipids
            cols = [f"{role}_name"]
            if f"{role}_SMILES" in df.columns:
                cols.append(f"{role}_SMILES")
            new_lipids = df[cols].drop_duplicates()
            for _, row in new_lipids.iterrows():
                if row[f"{role}_name"] not in current_names:
                    rec = {"name": row[f"{role}_name"]}
                    if f"{role}_SMILES" in row.index:
                        rec["smiles"] = row[f"{role}_SMILES"]
                    self.components[role].append(rec)

        # Update molratio bounds
        for role in self.ROLES:
            col = f"{role}_molratio"
            if meta["variable_molratios"].get(role, False):
                self.molratio_bounds[role] = (df[col].min(), df[col].max())
            else:
                self.molratio_bounds[role] = (df[col].iloc[0], df[col].iloc[0])
                self.fixed_values[col] = df[col].iloc[0]

        # Update IL:mRNA massratio values
        self.il_mrna_massratio_values = sorted(df["IL_to_nucleicacid_massratio"].dropna().unique().tolist())

        # Update PCs if present
        for role in self.ROLES:
            pcs_spec = meta.get("pcs", {}).get(role, {})
            if pcs_spec:
                self.component_pcs[role] = pcs_spec

        # Rebuild parameters
        self.parameters = self._build_parameters()

    # -------------------------
    # BO-facing API
    # -------------------------

    def get_configs(self):
        """
        Construct dictionary with configs for bayesopt
        """
        d = {
            "name": self.name,
            "target": "Experiment_value",
            "parameters": [],
        }

        assert self._dataset is not None
        metadata = self._dataset.metadata
        encoders = self._dataset.encoders
        assert encoders is not None

        # Component parameters
        for component in metadata["variable_components"]:
            if not metadata["variable_components"][component]:
                continue

            pcs_dict = encoders[component]
            df_cols = set(self._dataset.df.columns)

            columns = []
            for encoder_name, n_pcs in pcs_dict.items():
                # Use actual columns present in df (n_components may have been
                # clamped by compute_pcs when data is smaller than requested)
                enc_cols = [
                    f"{component}_{encoder_name}_pc{i}"
                    for i in range(1, n_pcs + 1)
                    if f"{component}_{encoder_name}_pc{i}" in df_cols
                ]
                columns += enc_cols

            if not columns:
                continue

            d["parameters"].append(
                {
                    "name": component,
                    "type": "ComponentParameter",
                    "columns": columns,
                }
            )

        # Mixture ratio parameters (only if at least one ratio varies)
        columns_mixture = []
        for molratio, is_var in metadata["variable_molratios"].items():
            if is_var:
                molratio = molratio + "_molratio"
                columns_mixture.append(molratio)
        if columns_mixture:
            param_dict = {
                "name": "molratio",
                "type": "MixtureRatiosParameter",
                "columns": columns_mixture,
            }
            d["parameters"].append(param_dict)

        # IL to mRNA mass ratio
        if metadata.get("variable_il_mrna", False):
            d["parameters"].append(
                {
                    "name": "IL_to_nucleicacid_massratio",
                    "type": "DiscreteParameter",
                    "columns": ["IL_to_nucleicacid_massratio"],
                }
            )
        return d

    def get_parameters(self):
        return self.parameters

    def get_target(self):
        return self.target

    def get_fixed_values(self):
        return self.fixed_values

    def new_round(self):
        self._round_counter += 1

    # Sampling (still symmetric)
    def sample_random(self, n: int):
        """
        Sample n random formulations in BO-compatible format.
        """
        return [self._sample_once() for _ in range(n)]

    def _sample_once(self):
        """
        Sample a single random formulation.
        Returns dict with keys exactly matching self.parameters' names:
        - ComponentParameter → np.array of component features
        - DiscreteParameter → single value
        - MixtureRatiosParameter → np.array of molratios
        """
        self._formulation_counter += 1
        sample = {
            "Formulation_ID": self._formulation_counter,
            "Round": self._round_counter,
        }

        # Components
        for p in self.parameters:
            if isinstance(p, ComponentParameter):
                # Pick a random component row
                idx = np.random.randint(len(p.unique_categories))
                sample[p.name] = p.unique_categories[idx]  # np.array of all columns

            elif isinstance(p, DiscreteParameter):
                sample[p.name] = random.choice(p.domain)

            elif isinstance(p, MixtureRatiosParameter):
                # Sample random ratios summing to sum_to
                raw = np.random.rand(len(p.domain))
                raw /= raw.sum()  # normalize
                raw *= p.sum_to
                sample[p.name] = raw

            else:
                raise TypeError(f"Unknown parameter type: {type(p)}")

        return sample
