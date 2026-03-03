from __future__ import annotations
import itertools
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from ..space.formulation import FormulationSpace
from ..data.dataset import Dataset


def generate_initial_batch(
    space: FormulationSpace,
    n_samples: int = 24,
    method: str = "simplex_random",
    seed: Optional[int] = None,
    levels: int = 3,
) -> Dataset:
    """
    Generate an initial batch of LNP formulations via Design of Experiments.

    Uses the FormulationSpace to determine which components are variable
    and what molar ratio bounds to use, then produces a structured initial
    screen for Round 0.

    Parameters
    ----------
    space : FormulationSpace
        Defines available components, molar ratio bounds, IL:mRNA values.
    n_samples : int
        Number of formulations to generate.
    method : str
        DoE method: "simplex_random", "extreme_vertices", "centroid_axial", "full_factorial"
    seed : int, optional
        Random seed for reproducibility.
    levels : int
        Grid levels for full_factorial method.

    Returns
    -------
    Dataset
        A Dataset with n_samples rows, Round=0, Experiment_value=NaN.
    """
    rng = np.random.default_rng(seed)

    # Generate mixture ratios
    components = list(space.ROLES)
    bounds = {role: space.molratio_bounds[role] for role in components}

    # Infer target sum from bounds (midpoint of feasible sums)
    target_sum = sum(np.mean(bounds[c]) for c in components)

    mixtures = mixture_doe(
        n_samples=n_samples,
        components=components,
        bounds=bounds,
        method=method,
        seed=seed,
        levels=levels,
        target_sum=target_sum,
    )

    # Build formulation rows
    rows = []
    for i, mix in enumerate(mixtures):
        row = {
            "Formulation_ID": i + 1,
            "Round": 0,
            "Experiment_value": np.nan,
        }

        # Assign molar ratios
        for role in components:
            row[f"{role}_molratio"] = mix[role]

        # Sample component identities
        for role in components:
            lipids = space.components[role]
            chosen = lipids[rng.integers(len(lipids))]
            row[f"{role}_name"] = chosen["name"]
            if "smiles" in chosen:
                row[f"{role}_SMILES"] = chosen["smiles"]

        # Sample IL:mRNA mass ratio
        il_mrna_values = space.il_mrna_massratio_values
        row["IL_to_nucleicacid_massratio"] = il_mrna_values[
            rng.integers(len(il_mrna_values))
        ]

        rows.append(row)

    df = pd.DataFrame(rows)

    # Apply fixed values
    for k, v in space.get_fixed_values().items():
        df[k] = v

    return Dataset(df, source="doe", name=f"doe_{method}_{n_samples}")


def _normalize_to_target(x: Dict[str, float], target_sum: float) -> Dict[str, float]:
    """Normalize a point so its values sum to target_sum."""
    total = sum(x.values())
    if total <= 0:
        raise ValueError("Cannot normalize: total is <= 0")
    return {k: v * target_sum / total for k, v in x.items()}


def _random_simplex_samples(
    n: int,
    components: List[str],
    bounds: Dict[str, Tuple[float, float]],
    rng: np.random.Generator,
    target_sum: float = 1.0,
    max_iterations: int = 10000,
) -> List[Dict[str, float]]:
    """Sample from the bounded simplex using Dirichlet + rejection.

    Samples fractions from Dirichlet(1,...,1) then scales to target_sum.
    Rejects samples that fall outside per-component bounds.
    """
    samples = []
    iterations = 0
    k = len(components)

    while len(samples) < n:
        if iterations >= max_iterations:
            raise RuntimeError(
                f"_random_simplex_samples exceeded {max_iterations} iterations "
                f"with only {len(samples)}/{n} valid samples. "
                f"Bounds may be too tight for target_sum={target_sum:.4f}."
            )
        # Dirichlet gives uniform distribution on the unit simplex
        raw_fracs = rng.dirichlet(np.ones(k))
        # Scale to target sum
        point = {c: raw_fracs[i] * target_sum for i, c in enumerate(components)}

        # Check bounds
        if all(bounds[c][0] <= point[c] <= bounds[c][1] for c in components):
            samples.append(point)

        iterations += 1

    return samples


def _extreme_vertices(
    components: List[str],
    bounds: Dict[str, Tuple[float, float]],
    target_sum: float = 1.0,
) -> List[Dict[str, float]]:
    """Generate extreme vertex designs on the bounded simplex."""
    vertices = []

    for c in components:
        v = {k: bounds[k][0] for k in components}
        v[c] = bounds[c][1]
        normed = _normalize_to_target(v, target_sum)

        if all(bounds[k][0] <= normed[k] <= bounds[k][1] for k in components):
            vertices.append(normed)

    return vertices


def _centroid_axial(
    components: List[str],
    bounds: Dict[str, Tuple[float, float]],
    target_sum: float = 1.0,
    delta_frac: float = 0.05,
) -> List[Dict[str, float]]:
    """Generate centroid + axial design on the bounded simplex."""
    center = {c: np.mean(bounds[c]) for c in components}
    center = _normalize_to_target(center, target_sum)

    design = [center]

    # Scale delta relative to target_sum
    delta = delta_frac * target_sum

    for c in components:
        for sign in [-1, 1]:
            point = center.copy()
            point[c] = np.clip(
                point[c] + sign * delta,
                bounds[c][0],
                bounds[c][1],
            )
            normed = _normalize_to_target(point, target_sum)

            if all(bounds[k][0] <= normed[k] <= bounds[k][1] for k in components):
                design.append(normed)

    return design


def mixture_doe(
    n_samples: int,
    components: List[str],
    bounds: Dict[str, Tuple[float, float]],
    method: str = "simplex_random",
    seed: Optional[int] = None,
    levels: int = 3,
    target_sum: Optional[float] = None,
) -> List[Dict[str, float]]:
    """
    Generate a mixture Design of Experiments.

    Parameters
    ----------
    n_samples
        Number of mixture points to generate.
    components
        Component names (e.g. ["IL", "HL", "CHL", "PEG"]).
    bounds
        Per-component (min, max) bounds.
    method
        "simplex_random", "extreme_vertices", "centroid_axial", or "full_factorial".
    seed
        Random seed for reproducibility.
    levels
        Grid levels for full_factorial method.
    target_sum
        What the component values should sum to. If None, inferred from
        the midpoint of bounds (works for both fractions and percentages).
    """
    if target_sum is None:
        target_sum = sum(np.mean(bounds[c]) for c in components)

    rng = np.random.default_rng(seed)

    if method == "simplex_random":
        return _random_simplex_samples(n_samples, components, bounds, rng, target_sum)

    if method == "extreme_vertices":
        return _extreme_vertices(components, bounds, target_sum)

    if method == "centroid_axial":
        return _centroid_axial(components, bounds, target_sum)

    if method == "full_factorial":
        return _full_factorial_mixture(
            components=components,
            bounds=bounds,
            levels=levels,
            target_sum=target_sum,
        )

    raise ValueError(f"Unknown mixture DOE method: {method}")


def _full_factorial_mixture(
    components: List[str],
    bounds: Dict[str, Tuple[float, float]],
    levels: int,
    target_sum: float = 1.0,
) -> List[Dict[str, float]]:
    """Full factorial design with simplex constraint."""
    grids = {
        c: np.linspace(bounds[c][0], bounds[c][1], levels)
        for c in components
    }

    candidates = []
    for values in itertools.product(*grids.values()):
        point = dict(zip(components, values))

        total = sum(point.values())
        if total <= 0:
            continue

        # Normalize to target sum
        point = {k: v * target_sum / total for k, v in point.items()}

        if all(bounds[c][0] <= point[c] <= bounds[c][1] for c in components):
            candidates.append(point)

    return candidates
