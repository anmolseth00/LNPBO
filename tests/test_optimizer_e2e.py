"""End-to-end smoke tests for the Optimizer class.

Verifies that the full suggest() pipeline runs without error for
GP-LogEI, XGB, and RF-TS surrogates using synthetic data only.

Run: .venv/bin/python -m pytest tests/test_optimizer_e2e.py -v
"""

import numpy as np
import pandas as pd

from LNPBO.data.dataset import Dataset
from LNPBO.optimization.optimizer import Optimizer
from LNPBO.space.formulation import FormulationSpace

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

IL_SMILES_LIST = [
    "CCCCCCCCCCCCCCCCCCCC(=O)OCC(COC(=O)CCCCCCCCCCCCCCCCC)OC(=O)CCCCCCCCCCCCCCCCC",
    "CCCCCCCCCCCC(=O)OCC(COC(=O)CCCCCCCCCCC)OC(=O)CCCCCCCCCCC",
    "CCCCCCCCCC(=O)OCC(COC(=O)CCCCCCCC)OC(=O)CCCCCCCC",
]

HL_SMILES = "CCCCCCCCCCCCCCCCCC(=O)OC[C@H](COP(=O)([O-])OCC[N+](C)(C)C)OC(=O)CCCCCCCCCCCCCCCCC"
CHL_SMILES = "C1CCC2(CC1)CC1CCC3C(CCC4CC(O)CCC43C)C1CC2"
PEG_SMILES = "CCCCCCCCCCCCCCCCCCOC(=O)NCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCO"


def _make_full_pool(n_total=50, seed=42):
    """Build a synthetic DataFrame with 3 IL identities and varied ratios.

    Returns the full pool DataFrame with Formulation_ID 1..n_total and Round=0.
    The first n_seed rows serve as "observed" training data; the rest are
    unevaluated candidates. All four molar ratios vary across rows so the
    optimizer has real features to work with.
    """
    rng = np.random.RandomState(seed)
    il_indices = rng.choice(len(IL_SMILES_LIST), n_total)

    df = pd.DataFrame(
        {
            "Formulation_ID": np.arange(1, n_total + 1),
            "Round": 0,
            "IL_name": [f"IL_{i}" for i in il_indices],
            "IL_SMILES": [IL_SMILES_LIST[i] for i in il_indices],
            "HL_name": "DSPC",
            "HL_SMILES": HL_SMILES,
            "CHL_name": "Cholesterol",
            "CHL_SMILES": CHL_SMILES,
            "PEG_name": "DMG-PEG2000",
            "PEG_SMILES": PEG_SMILES,
            "IL_molratio": rng.uniform(30, 60, n_total),
            "HL_molratio": rng.uniform(5, 15, n_total),
            "CHL_molratio": rng.uniform(30, 45, n_total),
            "PEG_molratio": rng.uniform(1, 3, n_total),
            "IL_to_nucleicacid_massratio": 10.0,
            "Experiment_value": rng.randn(n_total),
        }
    )
    return df


def _build_optimizer(surrogate_type, acquisition_type, batch_size=5, seed=42):
    """Build Dataset -> encode -> FormulationSpace -> Optimizer.

    Simulates a realistic BO setup: n_seed rows are "observed" (training),
    the full n_total-row pool is the candidate pool. The optimizer excludes
    already-observed Formulation_IDs from the pool automatically.
    """
    n_seed = 15
    n_total = 50

    full_df = _make_full_pool(n_total=n_total, seed=seed)

    # The "observed" seed data: first n_seed rows with real Experiment_values
    seed_df = full_df.iloc[:n_seed].copy()
    seed_dataset = Dataset(seed_df, source="synthetic", name="smoke_test")
    encoded_seed = seed_dataset.encode_dataset()

    # The full pool includes all n_total rows (observed + unevaluated)
    pool_dataset = Dataset(full_df, source="synthetic", name="smoke_pool")
    encoded_pool = pool_dataset.encode_dataset()
    candidate_pool = encoded_pool.df.copy()

    space = FormulationSpace.from_dataset(encoded_seed)

    optimizer = Optimizer(
        space=space,
        surrogate_type=surrogate_type,
        acquisition_type=acquisition_type,
        candidate_pool=candidate_pool,
        batch_size=batch_size,
        random_seed=seed,
        normalize="none",
    )
    return optimizer, encoded_seed, candidate_pool


# ---------------------------------------------------------------------------
# NOTE: GP test runs first because sklearn surrogates with n_jobs=-1 fork
# child processes that import PyTorch. If GP runs after those forks, the
# PyTorch thread pool can deadlock. Running GP first avoids the issue.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 1. GP-LogEI smoke test (BoTorch path)
# ---------------------------------------------------------------------------


class TestOptimizerGpLogEI:
    def test_optimizer_gp_logei_synthetic(self):
        optimizer, encoded, _ = _build_optimizer(
            surrogate_type="gp",
            acquisition_type="LogEI",
            batch_size=5,
        )
        result = optimizer.suggest()

        assert isinstance(result, pd.DataFrame)
        n_original = len(encoded.df)
        n_new = len(result) - n_original
        assert n_new == 5, f"Expected 5 new suggestions, got {n_new}"
        new_rows = result.iloc[n_original:]
        assert new_rows["Experiment_value"].isna().all()


# ---------------------------------------------------------------------------
# 2. XGB-UCB smoke test
# ---------------------------------------------------------------------------


class TestOptimizerXgbUcb:
    def test_optimizer_xgb_ucb_synthetic(self):
        optimizer, encoded, _ = _build_optimizer(
            surrogate_type="xgb",
            acquisition_type="UCB",
            batch_size=5,
        )
        result = optimizer.suggest()

        assert isinstance(result, pd.DataFrame)
        n_original = len(encoded.df)
        n_new = len(result) - n_original
        assert n_new == 5, f"Expected 5 new suggestions, got {n_new}"
        new_rows = result.iloc[n_original:]
        assert new_rows["Experiment_value"].isna().all()


# ---------------------------------------------------------------------------
# 3. RF-TS smoke test
# ---------------------------------------------------------------------------


class TestOptimizerRfTs:
    def test_optimizer_rf_ts_synthetic(self):
        optimizer, encoded, _ = _build_optimizer(
            surrogate_type="rf_ts",
            acquisition_type="UCB",
            batch_size=5,
        )
        result = optimizer.suggest()

        assert isinstance(result, pd.DataFrame)
        n_original = len(encoded.df)
        n_new = len(result) - n_original
        assert n_new == 5, f"Expected 5 new suggestions, got {n_new}"
        new_rows = result.iloc[n_original:]
        assert new_rows["Experiment_value"].isna().all()


# ---------------------------------------------------------------------------
# 4. State update test: suggest -> update -> suggest again
# ---------------------------------------------------------------------------


class TestOptimizerSuggestUpdatesState:
    def test_optimizer_suggest_updates_state(self):
        batch_size = 5
        optimizer, encoded, candidate_pool = _build_optimizer(
            surrogate_type="rf_ucb",
            acquisition_type="UCB",
            batch_size=batch_size,
            seed=123,
        )

        # First suggestion round
        result1 = optimizer.suggest()
        n_original = len(encoded.df)
        new_rows1 = result1.iloc[n_original:].copy()
        assert len(new_rows1) == batch_size

        first_batch_ids = set(new_rows1["Formulation_ID"].astype(int))

        # Simulate experimental results for the suggested formulations
        rng = np.random.RandomState(999)
        new_rows1 = new_rows1.copy()
        new_rows1["Experiment_value"] = rng.randn(len(new_rows1))
        new_rows1["Round"] = 1

        # Update the dataset with completed results
        updated_dataset = encoded.append_suggestions(new_rows1)

        # Rebuild space and optimizer for the second round
        space2 = FormulationSpace.from_dataset(updated_dataset)
        optimizer2 = Optimizer(
            space=space2,
            surrogate_type="rf_ucb",
            acquisition_type="UCB",
            candidate_pool=candidate_pool,
            batch_size=batch_size,
            random_seed=123,
            normalize="none",
        )

        result2 = optimizer2.suggest()
        n_after_update = len(updated_dataset.df)
        new_rows2 = result2.iloc[n_after_update:]
        assert len(new_rows2) == batch_size

        second_batch_ids = set(new_rows2["Formulation_ID"].astype(int))

        # Second batch should not repeat any formulation IDs from the first
        overlap = first_batch_ids & second_batch_ids
        assert len(overlap) == 0, f"Second batch reused Formulation_IDs from first batch: {overlap}"
