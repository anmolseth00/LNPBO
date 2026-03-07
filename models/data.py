"""Dataset and DataLoader for LNPDB multi-component molecular data.

Loads LNPDB.csv (or all_data_all.csv), featurizes SMILES per lipid role,
and provides tabular features (molar ratios, mass ratio, one-hot encoded
metadata) alongside molecular graphs.
"""


from pathlib import Path

import numpy as np
import pandas as pd
import torch
from featurize import BatchMolGraph, MolGraph, batch_mol_graphs, mol_to_graph
from torch.utils.data import DataLoader, Dataset

# Columns used as tabular features (continuous, z-scored)
TABULAR_CONTINUOUS_COLS = [
    "IL_molratio",
    "HL_molratio",
    "CHL_molratio",
    "PEG_molratio",
    "IL_to_nucleicacid_massratio",
    "Dose_ug_nucleicacid",
]

# All categorical columns available for one-hot encoding.
# Matching the original Chemprop v1.7 LiON feature set (92 tabular features).
TABULAR_CATEGORICAL_COLS = [
    "HL_name",
    "CHL_name",
    "PEG_name",
    "Aqueous_buffer",
    "Dialysis_buffer",
    "Mixing_method",
    "Model_type",
    "Model_target",
    "Route_of_administration",
    "Cargo",
    "Cargo_type",
    "Experiment_batching",
]

# SMILES columns per component
SMILES_COLS = {
    "IL": "IL_SMILES",
    "HL": "HL_SMILES",
    "CHL": "CHL_SMILES",
    "PEG": "PEG_SMILES",
}

# Placeholder SMILES for missing components (single carbon -- minimal valid mol)
PLACEHOLDER_SMILES = "C"


def learn_categorical_levels(
    df: pd.DataFrame,
    cat_cols: list[str] | None = None,
    min_count: int = 5,
) -> dict[str, list[str]]:
    """Learn categorical levels from training data.

    Returns dict mapping column names to sorted lists of levels.
    Only includes levels appearing at least min_count times.
    """
    cat_cols = cat_cols or TABULAR_CATEGORICAL_COLS
    levels: dict[str, list[str]] = {}

    for col in cat_cols:
        if col not in df.columns:
            continue
        counts = df[col].value_counts()
        valid = counts[counts >= min_count].index.tolist()
        levels[col] = sorted(valid)

    return levels


def encode_categoricals(
    df: pd.DataFrame,
    levels: dict[str, list[str]],
) -> tuple[pd.DataFrame, list[str]]:
    """One-hot encode categorical columns using pre-learned levels.

    Unknown/null values get all-zeros (no category active).
    """
    new_cols: list[str] = []

    for col, col_levels in levels.items():
        if col not in df.columns:
            continue
        for level in col_levels:
            col_name = f"{col}__{level}"
            df[col_name] = (df[col] == level).astype(np.float32)
            new_cols.append(col_name)

    return df, new_cols


from models.splits import _scaffold, scaffold_split  # noqa: F401


class LNPDataset(Dataset):
    """PyTorch Dataset for LNPDB formulations.

    Each item returns:
      - component_graphs: dict[str, MolGraph] per lipid role
      - tabular: numpy array of continuous tabular features
      - target: Experiment_value (float)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        components: list[str] | None = None,
        tabular_cols: list[str] | None = None,
        categorical_cols: list[str] | None = None,
        target_col: str = "Experiment_value",
        tabular_mean: np.ndarray | None = None,
        tabular_std: np.ndarray | None = None,
        target_mean: float | None = None,
        target_std: float | None = None,
        graph_fn=None,
    ):
        self.df = df.reset_index(drop=True)
        self.components = components or ["IL"]
        self.tabular_cols = tabular_cols or []
        self.categorical_cols = categorical_cols or []
        self.target_col = target_col

        # Z-score normalization stats (from training set, continuous only)
        self.tabular_mean = tabular_mean
        self.tabular_std = tabular_std
        self.target_mean = target_mean if target_mean is not None else 0.0
        self.target_std = target_std if target_std is not None else 1.0

        _graph_fn = graph_fn or mol_to_graph

        # Pre-compute molecular graphs (cached)
        self._graphs: dict[str, list[MolGraph | None]] = {}
        for comp in self.components:
            smiles_col = SMILES_COLS[comp]
            graphs = []
            for smi in self.df[smiles_col]:
                if pd.isna(smi) or str(smi).strip() in ("", "None", "nan"):
                    graphs.append(_graph_fn(PLACEHOLDER_SMILES))
                else:
                    g = _graph_fn(str(smi))
                    if g is None:
                        graphs.append(_graph_fn(PLACEHOLDER_SMILES))
                    else:
                        graphs.append(g)
            self._graphs[comp] = graphs

        # Pre-compute tabular features: continuous (z-scored) + categorical (as-is)
        parts = []
        if self.tabular_cols:
            cont = self.df[self.tabular_cols].fillna(0.0).values.astype(np.float32)
            if self.tabular_mean is not None and self.tabular_std is not None:
                safe_std = np.where(self.tabular_std < 1e-8, 1.0, self.tabular_std)
                cont = (cont - self.tabular_mean) / safe_std
            parts.append(cont)
        if self.categorical_cols:
            cat = self.df[self.categorical_cols].fillna(0.0).values.astype(np.float32)
            parts.append(cat)
        if parts:
            self._tabular = np.concatenate(parts, axis=1)
        else:
            self._tabular = np.zeros((len(self.df), 0), dtype=np.float32)

        # Pre-compute targets
        self._targets = self.df[self.target_col].values.astype(np.float32)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[dict[str, MolGraph | None], np.ndarray, float]:
        graphs = {comp: self._graphs[comp][idx] for comp in self.components}
        tabular = self._tabular[idx]
        target = (self._targets[idx] - self.target_mean) / self.target_std
        return graphs, tabular, target

    def get_smiles(self, idx: int, component: str = "IL") -> str:
        return str(self.df[SMILES_COLS[component]].iloc[idx])


def collate_fn(
    batch: list[tuple[dict[str, MolGraph], np.ndarray, float]],
) -> tuple[dict[str, BatchMolGraph], torch.Tensor, torch.Tensor]:
    """Custom collate for multi-component molecular data."""
    component_names = list(batch[0][0].keys())

    # Separate graphs by component
    component_graph_lists: dict[str, list[MolGraph]] = {name: [] for name in component_names}
    tabulars = []
    targets = []

    for graphs, tab, target in batch:
        for name in component_names:
            component_graph_lists[name].append(graphs[name])
        tabulars.append(tab)
        targets.append(target)

    # Batch each component's graphs
    batched_graphs = {
        name: batch_mol_graphs(component_graph_lists[name])
        for name in component_names
    }

    tabular_tensor = torch.tensor(np.array(tabulars, dtype=np.float32))
    target_tensor = torch.tensor(np.array(targets, dtype=np.float32))

    return batched_graphs, tabular_tensor, target_tensor


def make_dataloader(
    dataset: LNPDataset,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=False,
    )


def load_lnpdb_dataframe(
    data_path: str | Path | None = None,
    components: list[str] | None = None,
) -> pd.DataFrame:
    """Load LNPDB data with all columns needed for training.

    When components include only IL, uses all_data_all.csv (has IL_SMILES +
    tabular + metadata). When multi-component SMILES are needed (HL, CHL, PEG),
    loads from LNPDB.csv which has all SMILES columns, then joins with
    all_data_all.csv for the extra tabular features.

    Resolution order:
      1. Explicit data_path
      2. LNPDB_PATH env var
      3. data/LNPDB_repo symlink
      4. Sibling ../LNPDB directory
    """
    import os

    if data_path is not None:
        p = Path(data_path)
        if p.is_file():
            return pd.read_csv(p)
        raise FileNotFoundError(f"Data file not found: {p}")

    components = components or ["IL"]
    needs_multi_smiles = any(c in components for c in ["HL", "CHL", "PEG"])

    this_dir = Path(__file__).resolve().parent.parent / "data"

    def _find_lion_dir() -> Path:
        env = os.environ.get("LNPDB_PATH")
        if env:
            p = Path(env) / "data" / "LNPDB_for_LiON"
            if p.is_dir():
                return p
        p = this_dir / "LNPDB_repo" / "data" / "LNPDB_for_LiON"
        if p.is_dir():
            return p
        p = this_dir.parent.parent / "LNPDB" / "data" / "LNPDB_for_LiON"
        if p.is_dir():
            return p
        raise FileNotFoundError(
            "Cannot locate LNPDB_for_LiON directory. Pass --data-path explicitly, "
            "set LNPDB_PATH env var, or ensure data/LNPDB_repo symlink exists."
        )

    lion_dir = _find_lion_dir()

    if not needs_multi_smiles:
        # Single-component mode: all_data_all.csv has everything we need
        p = lion_dir / "all_data_all.csv"
        if p.is_file():
            return pd.read_csv(p)
        raise FileNotFoundError(f"all_data_all.csv not found at {p}")

    # Multi-component mode: LNPDB.csv has all SMILES columns,
    # all_data_all.csv has z-scored Experiment_value + tabular features.
    # Merge on LNP_ID to get both.
    lnpdb_csv = lion_dir / "LNPDB.csv"
    if not lnpdb_csv.is_file():
        raise FileNotFoundError(f"LNPDB.csv not found at {lnpdb_csv}")

    all_data_csv = lion_dir / "all_data_all.csv"
    if not all_data_csv.is_file():
        raise FileNotFoundError(f"all_data_all.csv not found at {all_data_csv}")

    df_lnpdb = pd.read_csv(lnpdb_csv, low_memory=False)
    df_all = pd.read_csv(all_data_csv)

    # Verify required SMILES columns exist
    for comp in components:
        col = SMILES_COLS[comp]
        if col not in df_lnpdb.columns:
            raise ValueError(f"SMILES column {col} not found in LNPDB.csv")

    # Get SMILES from LNPDB.csv, everything else from all_data_all.csv
    smiles_cols_needed = [SMILES_COLS[c] for c in components if c != "IL"]
    keep_from_lnpdb = ["LNP_ID", *smiles_cols_needed]

    df = df_all.merge(
        df_lnpdb[keep_from_lnpdb].drop_duplicates("LNP_ID"),
        on="LNP_ID",
        how="inner",
    )

    return df
