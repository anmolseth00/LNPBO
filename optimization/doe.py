from __future__ import annotations
import itertools

# design = mixture_doe(
#     n_samples=16,
#     components=["IL", "HL", "CHL", "PEG"],
#     bounds={
#         "IL": (0.3, 0.6),
#         "HL": (0.05, 0.15),
#         "CHL": (0.2, 0.4),
#         "PEG": (0.01, 0.05),
#     },
#     method="simplex_random",
#     seed=42,
# )

# [
#   {"IL": 0.45, "HL": 0.10, "CHL": 0.40, "PEG": 0.05},
#   ...
# ]

import numpy as np
from typing import Dict, List, Tuple, Optional
from ..space.formulation import FormulationSpace
from ..data.dataset import Dataset

def generate_initial_batch(space: FormulationSpace) -> Dataset:
    df_new = ...  # existing DoE logic
    return Dataset(df_new, source="doe")

def _normalize_to_simplex(x: Dict[str, float]) -> Dict[str, float]:
    total = sum(x.values())
    return {k: v / total for k, v in x.items()}

def _random_simplex_samples(
    n: int,
    components: List[str],
    bounds: Dict[str, Tuple[float, float]],
    rng: np.random.Generator,
) -> List[Dict[str, float]]:
    samples = []

    while len(samples) < n:
        raw = {c: rng.uniform(*bounds[c]) for c in components}
        normed = _normalize_to_simplex(raw)

        # Check bounds after normalization
        if all(bounds[c][0] <= normed[c] <= bounds[c][1] for c in components):
            samples.append(normed)

    return samples

def _extreme_vertices(
    components: List[str],
    bounds: Dict[str, Tuple[float, float]],
) -> List[Dict[str, float]]:
    vertices = []

    for c in components:
        v = {k: bounds[k][0] for k in components}
        v[c] = bounds[c][1]
        vertices.append(_normalize_to_simplex(v))

    return vertices

def _centroid_axial(
    components: List[str],
    bounds: Dict[str, Tuple[float, float]],
    delta: float = 0.05,
) -> List[Dict[str, float]]:
    center = {
        c: np.mean(bounds[c]) for c in components
    }
    center = _normalize_to_simplex(center)

    design = [center]

    for c in components:
        for sign in [-1, 1]:
            point = center.copy()
            point[c] = np.clip(
                point[c] + sign * delta,
                bounds[c][0],
                bounds[c][1],
            )
            design.append(_normalize_to_simplex(point))

    return design

def mixture_doe(
    n_samples: int,
    components: List[str],
    bounds: Dict[str, Tuple[float, float]],
    method: str = "simplex_random",
    seed: Optional[int] = None,
    levels: int = 3,
) -> List[Dict[str, float]]:
    """
    Generate a mixture Design of Experiments.

    Parameters
    ----------
    n_samples
        Number of mixture points to generate

    components
        Component names (e.g. ["IL", "HL", "CHL", "PEG"])

    bounds
        Per-component (min, max) bounds

    method
        One of:
        - "simplex_random"
        - "extreme_vertices"
        - "centroid_axial"

    seed
        Random seed for reproducibility
    """

    rng = np.random.default_rng(seed)

    if method == "simplex_random":
        return _random_simplex_samples(n_samples, components, bounds, rng)

    if method == "extreme_vertices":
        return _extreme_vertices(components, bounds)

    if method == "centroid_axial":
        return _centroid_axial(components, bounds)
    
    if method == "full_factorial":
        return _full_factorial_mixture(
            components=components,
            bounds=bounds,
            levels=levels,
        )

    raise ValueError(f"Unknown mixture DOE method: {method}")

# mixtures = mixture_doe(
#     n_samples=12,
#     components=["IL", "HL", "CHL", "PEG"],
#     bounds=space.molratio_bounds,
#     method="simplex_random",
# )

# formulations = []
# for mix in mixtures:
#     f = space._sample_once()
#     f["IL_molratio"] = mix["IL"]
#     f["HL_molratio"] = mix["HL"]
#     f["CHL_molratio"] = mix["CHL"]
#     f["PEG_molratio"] = mix["PEG"]
#     formulations.append(f)

def _full_factorial_mixture(
    components: List[str],
    bounds: Dict[str, Tuple[float, float]],
    levels: int,
) -> List[Dict[str, float]]:
    """
    Full factorial design for mixture components with simplex constraint.

    Parameters
    ----------
    components
        Mixture components (e.g. ["IL", "HL", "CHL", "PEG"])

    bounds
        Per-component bounds

    levels
        Number of grid levels per component
    """

    grids = {
        c: np.linspace(bounds[c][0], bounds[c][1], levels)
        for c in components
    }

    candidates = []
    for values in itertools.product(*grids.values()):
        point = dict(zip(components, values))

        # Normalize to simplex
        total = sum(point.values())
        if total <= 0:
            continue

        point = {k: v / total for k, v in point.items()}

        # Enforce bounds after normalization
        if all(bounds[c][0] <= point[c] <= bounds[c][1] for c in components):
            candidates.append(point)

    return candidates


# # Generate all possible formulations using Full-Factorial Design.
# # # Note that if molarratio_spec == "range", the granularity variable is very important
# # as that will largely dictate how many formulations to test.
# # NEED TO EDIT TO ENSURE ANY FORMULATION HAS MOLAR RATIOS THAT SUM TO 1
# # Initialize lists for storing the final data

# formulations_list = []
# molar_ratios_list = []
# rna_ratios_list = []

# # Check molarratio_spec and define molar ratios accordingly
# if molarratio_spec == "select":
#     # Generate all possible combinations of
#     # components, molar ratios, and firstcomponent_rna_ratio
#     formulations = list(itertools.product(
#         first_component_type,
#         second_component_type,
#         third_component_type,
#         fourth_component_type,
#         fifth_component_type if fifth_component else [None]
#     ))

#     # Generate all combinations of molar ratios and firstcomponent_rna_ratio

#     molar_ratios = list(itertools.product(
#         first_component_molarratio,
#         second_component_molarratio,
#         third_component_molarratio,
#         fourth_component_molarratio,
#         fifth_component_molarratio if fifth_component else [None]
#     ))

 

#     # Combine molar ratios with RNA ratios
#     for rna_ratio in firstcomponent_rna_ratio:
#         for f in formulations:
#             for mr in molar_ratios:
#                 formulations_list.append(f)
#                 molar_ratios_list.append(mr)
#                 rna_ratios_list.append(rna_ratio)

# elif molarratio_spec == "range":
#     # Generate all possible combinations of components
#     formulations = list(itertools.product(
#         first_component_type,
#         second_component_type,
#         third_component_type,
#         fourth_component_type,
#         fifth_component_type if fifth_component else [None]
#     ))

#     # Generate discrete molar ratio values for each component based on granularity
#     first_molarratio_values = np.linspace(first_component_molarratio[0], first_component_molarratio[1], molarratio_granularity[0])
#     second_molarratio_values = np.linspace(second_component_molarratio[0], second_component_molarratio[1], molarratio_granularity[1])
#     third_molarratio_values = np.linspace(third_component_molarratio[0], third_component_molarratio[1], molarratio_granularity[2])
#     fourth_molarratio_values = np.linspace(fourth_component_molarratio[0], fourth_component_molarratio[1], molarratio_granularity[3])
 
#     if fifth_component:
#         fifth_molarratio_values = np.linspace(fifth_component_molarratio[0], fifth_component_molarratio[1], molarratio_granularity[4])
#     else:
#         fifth_molarratio_values = [None]

#     # Generate all combinations of these molar ratios

#     molar_ratios = list(itertools.product(
#         first_molarratio_values,
#         second_molarratio_values,
#         third_molarratio_values,
#         fourth_molarratio_values,
#         fifth_molarratio_values if fifth_component else [None]
#     ))

#     # Pair each formulation with each combination of molar ratios and RNA ratios
#     for rna_ratio in firstcomponent_rna_ratio:
#         for f in formulations:
#             for mr in molar_ratios:
#                 formulations_list.append(f)
#                 molar_ratios_list.append(mr)
#                 rna_ratios_list.append(rna_ratio)

# # Flatten the lists of tuples into columns
# formulation_df = pd.DataFrame({
#     "formulation_id": range(1, len(formulations_list) + 1),
#     "first_type": [f[0] for f in formulations_list],
#     "first_molarratio": [mr[0] for mr in molar_ratios_list],
#     "second_type": [f[1] for f in formulations_list],
#     "second_molarratio": [mr[1] for mr in molar_ratios_list],
#     "third_type": [f[2] for f in formulations_list],
#     "third_molarratio": [mr[2] for mr in molar_ratios_list],
#     "fourth_type": [f[3] for f in formulations_list],
#     "fourth_molarratio": [mr[3] for mr in molar_ratios_list],
#     "firstcomponent_rna_ratio": rna_ratios_list
# })

# if fifth_component:
#     formulation_df["fifth_type"] = [f[4] for f in formulations_list]
#     formulation_df["fifth_molarratio"] = [mr[4] for mr in molar_ratios_list]

# # Display the resulting DataFrame
# formulation_df

# # Assuming formulation_df is already defined
# screen_strategy = "all"  # "all" or "random" or "doe"
# num_formulations = 100  # Only needed if screen_strategy is "random"

# if screen_strategy == "all":
#     # Use all formulations
#     formulation_firstscreen_df = formulation_df.copy()

# elif screen_strategy == "random":
#     # Randomly select num_formulations rows from formulation_df
#     formulation_firstscreen_df = formulation_df.sample(n=num_formulations, random_state=42).reset_index(drop=True)

# #elif screen_strategy == "doe":
# #    # Example using DOE: Full Factorial Design as an example
# #    # Customize this depending on the DOE method you want to use
# #    num_factors = len(formulation_df.columns) - 1  # Minus formulation_id
# #    doe_matrix = ff2n(num_factors)  # Full factorial for num_factors
# #    doe_indices = np.random.choice(range(len(formulation_df)), size=len(doe_matrix), replace=False)

#     # Select the subset of formulations based on DOE indices
# #    formulation_firstscreen_df = formulation_df.iloc[doe_indices].reset_index(drop=True)

# # Add 'screen_id' as the first column
# formulation_firstscreen_df.insert(0, 'screen_id', 1)

# # Display the resulting DataFrame
# formulation_firstscreen_df