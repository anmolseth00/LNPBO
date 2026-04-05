# Getting Started

This guide walks through three common use cases for LNPBO, from ratio-only optimization to full lipid identity screening.

---

## Prerequisites

Install LNPBO from source:

```bash
git clone https://github.com/evancollins1/LNPBO.git
cd LNPBO
pip install .
```

For GP-based optimization (BoTorch/GPyTorch):

```bash
pip install ".[gp]"
```

For all optional dependencies:

```bash
pip install ".[all]"
```

### LNPDB Setup

Benchmarks and LNPDB study examples require the [LNPDB](https://github.com/evancollins1/LNPDB) repo as a sibling directory, symlinked into the data folder:

```bash
cd ..
git clone https://github.com/evancollins1/LNPDB.git
cd LNPBO
ln -s ../LNPDB data/LNPDB_repo
```

---

## Input Data Format

LNPBO expects a CSV file following the [LNPDB](https://www.nature.com/articles/s41467-026-68818-1) column format. Each row is one LNP formulation with these columns:

| Column | Description |
|--------|-------------|
| `IL_name` | Ionizable lipid name |
| `IL_SMILES` | Ionizable lipid SMILES (optional if not varying IL) |
| `IL_molratio` | Ionizable lipid molar ratio |
| `IL_to_nucleicacid_massratio` | IL-to-nucleic acid mass ratio |
| `HL_name` | Helper lipid name |
| `HL_SMILES` | Helper lipid SMILES (optional if not varying HL) |
| `HL_molratio` | Helper lipid molar ratio |
| `CHL_name` | Cholesterol name |
| `CHL_SMILES` | Cholesterol SMILES (optional if not varying CHL) |
| `CHL_molratio` | Cholesterol molar ratio |
| `PEG_name` | PEG lipid name |
| `PEG_SMILES` | PEG lipid SMILES (optional if not varying PEG) |
| `PEG_molratio` | PEG lipid molar ratio |
| `Experiment_value` | Functional readout (e.g., transfection luminescence) |

SMILES columns are only required when lipid identities vary across formulations. When only ratios are being optimized, SMILES can be omitted.

---

## Example 1: Ratio-Only Optimization (Continuous BO)

When lipid identities are fixed and only molar ratios vary (e.g., optimizing IL/HL/CHL ratios and IL:mRNA mass ratio for a known lipid combination like cKK-E12 / DOPE / Cholesterol / DMG-PEG2000).

```python
from LNPBO.data.dataset import Dataset
from LNPBO.space.formulation import FormulationSpace
from LNPBO.optimization.optimizer import Optimizer

# Load initial screen
dataset = Dataset.from_lnpdb_csv("example1.csv")

# Encode (no molecular features needed for ratio-only)
encoded = dataset.encode_dataset(
    encoding_csv_path="example1_encodings.csv",
)

# Build formulation space
space = FormulationSpace.from_dataset(encoded)

# Initialize optimizer (continuous BO with sklearn GP)
optimizer = Optimizer(
    space=space,
    surrogate_type="gp",
    gp_engine="sklearn",
    candidate_pool=encoded.df,
    acquisition_type="UCB",
    kappa=5.0,
    random_seed=42,
    batch_size=24,
)

# Suggest next batch
suggestions = optimizer.suggest(output_csv="example1_round1.csv")
```

The output CSV appends the suggested formulations to the original data, with `Experiment_value` left blank for the scientist to fill in after testing.

---

## Example 2: Varying Ratios and Helper Lipid (Discrete Pool BO)

When varying HL identity alongside molar ratios, molecular encodings are needed to represent structural differences between lipids.

```python
from LNPBO.data.dataset import Dataset
from LNPBO.space.formulation import FormulationSpace
from LNPBO.optimization.optimizer import Optimizer

# Load initial screen
dataset = Dataset.from_lnpdb_csv("example2.csv")

# Encode with LANTERN (count MFP + RDKit, PCA to 5 PCs)
encoded = dataset.encode_dataset(
    feature_type="lantern",
    encoding_csv_path="example2_encodings.csv",
)

# Build formulation space and optimizer (discrete pool BO with GP)
space = FormulationSpace.from_dataset(encoded)
optimizer = Optimizer(
    space=space,
    surrogate_type="gp",
    candidate_pool=encoded.df,
    acquisition_type="UCB",
    kappa=5.0,
    random_seed=42,
    batch_size=24,
)

# Suggest next batch
suggestions = optimizer.suggest(output_csv="example2_round1.csv")
```

### Multi-Round Optimization

After testing the suggested formulations and recording results:

```python
# Reload with updated results
dataset = Dataset.from_lnpdb_csv("example2_round1_w_results.csv")
encoded = dataset.encode_dataset(
    feature_type="lantern",
    encoding_csv_path="example2_encodings_r2.csv",
)
space = FormulationSpace.from_dataset(encoded)
optimizer = Optimizer(
    space=space,
    surrogate_type="gp",
    candidate_pool=encoded.df,
    acquisition_type="UCB",
    kappa=5.0,
    random_seed=42,
    batch_size=24,
)

# Suggest round 2
suggestions = optimizer.suggest(output_csv="example2_round2.csv")
```

---

## Example 3: Varying Ratios, Helper Lipid, and Ionizable Lipid (Discrete Pool BO)

The most complex case: screening diverse IL and HL libraries while optimizing ratios.

```python
from LNPBO.data.dataset import Dataset
from LNPBO.space.formulation import FormulationSpace
from LNPBO.optimization.optimizer import Optimizer

# Load initial screen
dataset = Dataset.from_lnpdb_csv("example3.csv")

# Encode with LANTERN (default, best overall)
encoded = dataset.encode_dataset(
    feature_type="lantern",
    encoding_csv_path="example3_encodings.csv",
)

# Build formulation space
space = FormulationSpace.from_dataset(encoded)

# Initialize optimizer (discrete pool BO with GP)
optimizer = Optimizer(
    space=space,
    surrogate_type="gp",
    candidate_pool=encoded.df,
    acquisition_type="UCB",
    kappa=5.0,
    random_seed=42,
    batch_size=24,
)

# Suggest next batch
suggestions = optimizer.suggest(output_csv="example3_round1.csv")
```

### Alternative Encodings

```python
# Morgan fingerprints
encoded = dataset.encode_dataset(
    feature_type="mfp",
    encoding_csv_path="example3_mfp.csv",
)

# Mordred descriptors (requires: pip install "LNPBO[mordred]")
encoded = dataset.encode_dataset(
    feature_type="mordred",
    encoding_csv_path="example3_mordred.csv",
)
```

See [Molecular Encodings](encodings.md) for a full comparison of all encoding options.

---

## Using Tree-Based Surrogates

Tree-based surrogates (XGBoost, Random Forest, NGBoost) often outperform GP surrogates on diverse IL screening tasks:

```python
# NGBoost with UCB (best overall in benchmarks)
optimizer = Optimizer(
    space=space,
    candidate_pool=encoded.df,
    surrogate_type="ngboost",
    batch_size=24,
    random_seed=42,
)

# Random Forest with Thompson Sampling
optimizer = Optimizer(
    space=space,
    candidate_pool=encoded.df,
    surrogate_type="rf_ts",
    batch_size=24,
    random_seed=42,
)

# XGBoost greedy (fastest, no UQ)
optimizer = Optimizer(
    space=space,
    candidate_pool=encoded.df,
    surrogate_type="xgb",
    batch_size=24,
    random_seed=42,
)
```

See [Optimization Strategies](strategies.md) for details on all available surrogates and batch strategies.

---

## CLI Usage

LNPBO provides a command-line interface with five subcommands:

```bash
# Encode raw formulation CSV into numeric features (LANTERN-style)
lnpbo encode --input raw_data.csv --output encoded.csv \
    --IL-n-pcs-count-mfp 5 --IL-n-pcs-rdkit 5 \
    --HL-n-pcs-count-mfp 3 --HL-n-pcs-rdkit 3

# Suggest next batch of formulations
lnpbo suggest --dataset encoded.csv --output round1.csv \
    --surrogate-type xgb_ucb --batch-size 12

# Register experimental results back into dataset
lnpbo register --dataset round1.csv --results results.csv --output updated.csv

# Propose new ionizable lipids
lnpbo propose-ils --dataset data.csv --output proposed_ils.csv \
    --n-candidates 20000 --n-output 100

# Save surrogate model checkpoint
lnpbo checkpoint save --model model.joblib --surrogate-type gp \
    --columns cols.joblib --output-dir checkpoint_dir
```

The `suggest` command supports all surrogate types (`--surrogate-type`), acquisition functions (`--acquisition-type`), and batch strategies (`--batch-strategy`) available in the Python API.

---

## Next Steps

- [Molecular Encodings](encodings.md) -- Understand the 8+ encoding options and when to use each
- [Optimization Strategies](strategies.md) -- Compare surrogate models and batch strategies
- [API Reference](../api/optimization.md) -- Full API documentation
