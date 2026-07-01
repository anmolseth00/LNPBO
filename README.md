# LNPBO: Bayesian Optimization for Lipid Nanoparticle Design

<p align="center">
  <img src="LNPBO.png" alt="LNPBO overview" width="600px" align="middle"/>
</p>

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

Data-driven optimization of lipid nanoparticle (LNP) formulations.

---

## Table of Contents

- [Installation](#installation)
- [Quickstart](#quickstart)
- [Examples](#examples)
- [Choosing a Strategy](#choosing-a-strategy)
- [Molecular Encodings](#molecular-encodings)
- [License](#license)

---

## Installation

```bash
git clone https://github.com/evancollins1/LNPBO.git
cd LNPBO
uv sync
```

For GP support (BoTorch/GPyTorch):

```bash
uv sync --extra gp
```

Optional extras:

```bash
uv sync --extra bench      # NGBoost
uv sync --extra mordred    # Mordred descriptors
uv sync --extra all        # All optional dependencies
```

### LNPDB Setup

Working with the full [LNPDB](https://github.com/evancollins1/LNPDB) dataset (e.g., generating Uni-Mol or LiON embeddings) requires cloning it as a sibling directory and creating the expected symlink:

```bash
git clone https://github.com/evancollins1/LNPDB.git   # sibling to LNPBO/
cd LNPBO
ln -s ../LNPDB data/LNPDB_repo
```

### Development Installation

```bash
git clone https://github.com/evancollins1/LNPBO.git
cd LNPBO
uv sync --extra dev
```

Requires Python >= 3.10.

---

## Quickstart

Given a CSV of tested LNP formulations (see [input format](#input-format)), suggest the next batch to synthesize:

```python
from LNPBO.data.dataset import Dataset
from LNPBO.space.formulation import FormulationSpace
from LNPBO.optimization.optimizer import Optimizer

# Load and encode
dataset = Dataset.from_lnpdb_csv("my_lnps.csv")
encoded = dataset.encode_dataset(feature_type="lantern")
space = FormulationSpace.from_dataset(encoded)

# Optimize
optimizer = Optimizer(space=space, surrogate_type="gp", candidate_pool=encoded.df, acquisition_type="UCB", kappa=5.0, batch_size=12)
suggestions = optimizer.suggest(output_csv="round1.csv")
```

The output CSV appends the suggested formulations (with blank `Experiment_value`) to your original data. Test them in lab, fill in results, and re-run for the next round.

---

## Examples

The `examples/` folder contains Jupyter notebooks for the three synthetic use cases below; Example 4 runs the same workflow on real LNPDB data.

LNPBO supports two BO modes:

- **Discrete pool BO (retrospective or prospective)** - A surrogate model scores a fixed candidate pool and selects the most promising batch. Best for library screening where lipid identities vary. *(Examples 2, 3, 4)*
- **Continuous BO (prospective, real-time)** - A GP optimizes over continuous ratio bounds to suggest new formulations not in any existing pool. Best for ratio-only optimization with fixed lipid identities. *(Example 1)*

Examples 1–3 use synthetic data to illustrate the input format; Example 4 uses real LNPDB data.

<details>
<summary><b>Example 1: Ratio optimization (continuous BO, real-time)</b> &mdash; fixed lipid identities, varying molar ratios</summary>

A scientist has chosen lipid identities (cKK-E12, DOPE, Cholesterol, DMG-PEG2000) and wants to optimize IL, HL, and CHL molar ratios plus the IL:mRNA mass ratio.

```python
from LNPBO.data.dataset import Dataset
from LNPBO.space.formulation import FormulationSpace
from LNPBO.optimization.optimizer import Optimizer

dataset = Dataset.from_lnpdb_csv("examples/example1/example1.csv")
encoded = dataset.encode_dataset(encoding_csv_path="example1_enc.csv")
space = FormulationSpace.from_dataset(encoded)

optimizer = Optimizer(space=space, surrogate_type="gp", gp_engine="sklearn", acquisition_type="UCB", kappa=5.0, batch_size=24)
suggestions = optimizer.suggest(output_csv="example1_round1.csv")
```

Since no lipid identities vary, SMILES columns are optional and no molecular encoding is needed. The sklearn GP engine is used here for fast, continuous optimization over ratio bounds.

See: [`examples/example1/example1.ipynb`](examples/example1/example1.ipynb)

</details>

<details>
<summary><b>Example 2: Ratio + helper lipid optimization (discrete pool BO, retrospective or prospective)</b> &mdash; varying HL identity and molar ratios</summary>

A scientist fixes the IL identity but wants to explore alternative helper lipids while optimizing ratios. This requires molecular encoding to featurize HL structure.

```python
dataset = Dataset.from_lnpdb_csv("examples/example2/example2.csv")
encoded = dataset.encode_dataset(feature_type="lantern", encoding_csv_path="example2_enc.csv")
space = FormulationSpace.from_dataset(encoded)

optimizer = Optimizer(space=space, surrogate_type="gp", candidate_pool=encoded.df, acquisition_type="UCB", kappa=5.0, batch_size=24)
suggestions = optimizer.suggest(output_csv="example2_round1.csv")
```

**Multi-round optimization:** When round 1 results come back from the lab, reload and re-run:

```python
dataset = Dataset.from_lnpdb_csv("example2_round1_w_results.csv")
encoded = dataset.encode_dataset(feature_type="lantern", encoding_csv_path="example2_enc_r2.csv")
space = FormulationSpace.from_dataset(encoded)
optimizer = Optimizer(space=space, surrogate_type="gp", candidate_pool=encoded.df, acquisition_type="UCB", kappa=5.0, batch_size=24)
suggestions = optimizer.suggest(output_csv="example2_round2.csv")
```

See: [`examples/example2/example2.ipynb`](examples/example2/example2.ipynb)

</details>

<details>
<summary><b>Example 3: Full library screening (discrete pool BO, retrospective or prospective &mdash; tree surrogate)</b> &mdash; varying IL, HL identities and molar ratios</summary>

A scientist wants to screen a combinatorial library of ionizable lipids and helper lipids while co-optimizing molar ratios. This is the most complex scenario.

```python
dataset = Dataset.from_lnpdb_csv("examples/example3/example3.csv")
encoded = dataset.encode_dataset(feature_type="lantern", encoding_csv_path="example3_enc.csv")
space = FormulationSpace.from_dataset(encoded)

optimizer = Optimizer(space=space, surrogate_type="ngboost", candidate_pool=encoded.df, batch_size=24)
suggestions = optimizer.suggest(output_csv="example3_round1.csv")
```

Alternative encodings:

```python
# Morgan fingerprints (fast, 2048-bit)
encoded = dataset.encode_dataset(feature_type="mfp")

# Mordred descriptors (requires: uv sync --extra mordred)
encoded = dataset.encode_dataset(feature_type="mordred")
```

See: [`examples/example3/example3.ipynb`](examples/example3/example3.ipynb)

</details>

<details>
<summary><b>Example 4: Real LNPDB data (discrete pool BO)</b> &mdash; screening real formulations from the LNPDB database</summary>

With the [LNPDB Setup](#lnpdb-setup) symlink in place, point `from_lnpdb_csv` at an LNPDB-format CSV of real formulations (e.g., a study exported from `data/LNPDB_repo`) and run the same discrete pool BO workflow as Examples 2&ndash;3 on real data:

```python
from LNPBO.data.dataset import Dataset
from LNPBO.space.formulation import FormulationSpace
from LNPBO.optimization.optimizer import Optimizer

dataset = Dataset.from_lnpdb_csv("my_lnpdb_study.csv")
encoded = dataset.encode_dataset(feature_type="lantern")
space = FormulationSpace.from_dataset(encoded)

optimizer = Optimizer(space=space, surrogate_type="rf_ts", candidate_pool=encoded.df, batch_size=24)
suggestions = optimizer.suggest(output_csv="lnpdb_round1.csv")
```

</details>

### Input format

Each row is one LNP formulation. Required columns:

| Column | Description |
|--------|-------------|
| `IL_name` | Ionizable lipid name |
| `IL_SMILES` | Ionizable lipid SMILES (optional if IL identity is fixed) |
| `IL_molratio` | IL molar ratio |
| `IL_to_nucleicacid_massratio` | IL:nucleic acid mass ratio |
| `HL_name`, `HL_SMILES`, `HL_molratio` | Helper lipid |
| `CHL_name`, `CHL_SMILES`, `CHL_molratio` | Cholesterol |
| `PEG_name`, `PEG_SMILES`, `PEG_molratio` | PEG lipid |
| `Experiment_value` | Functional readout (e.g., transfection) |

---

## Choosing a Strategy

Suggested starting points:

| Scenario | Recommended strategy |
|----------|---------------------|
| Default / general use | `NGBoost-UCB` or `RF-TS` |
| Screening diverse IL libraries | Tree-based surrogates (RF, XGBoost, NGBoost) |
| Ratio-only optimization | `CASMOPolitan` or GP-based methods |
| Limited compute budget | `XGBoost-Greedy` (no uncertainty quantification needed) |

> **Note:** NGBoost is not a core dependency — install it with `uv sync --extra bench` before using `surrogate_type="ngboost"`.

```python
# NGBoost with UCB
optimizer = Optimizer(space=space, surrogate_type="ngboost", candidate_pool=encoded.df, batch_size=24)

# Random Forest with Thompson Sampling
optimizer = Optimizer(space=space, surrogate_type="rf_ts", candidate_pool=encoded.df, batch_size=24)

# CASMOPolitan (mixed continuous/categorical)
optimizer = Optimizer(space=space, surrogate_type="casmopolitan", candidate_pool=encoded.df, batch_size=24)
```

**Acquisition functions:** `"UCB"` (upper confidence bound), `"EI"` (expected improvement), `"LogEI"` (log expected improvement). `kappa` (UCB) and `xi` (EI/LogEI) control exploration vs. exploitation.

**Batch strategies:** `"kb"` (Kriging Believer), `"rkb"` (Resampling KB), `"lp"` (Local Penalization), `"ts"` (Thompson sampling), `"qlogei"` (q-Log Noisy EI), `"gibbon"` (GIBBON).

---

## Molecular Encodings

Pass any name below as the `feature_type` argument to `encode_dataset()`:

```python
encoded = dataset.encode_dataset(feature_type="lantern")
```

| `feature_type` | Description | Best for | Reference |
|----------|-------------|----------|-----------|
| `lantern` | Count Morgan FP + RDKit descriptors, PCA to 5 PCs | Default | [LANTERN, 2025](https://arxiv.org/abs/2507.03209) |
| `mfp` | Morgan fingerprints (2048-bit) | Fast baseline | [Rogers & Hahn, 2010](https://doi.org/10.1021/ci100050t) |
| `count_mfp` | Count-based Morgan fingerprints | When counts matter | [Rogers & Hahn, 2010](https://doi.org/10.1021/ci100050t) |
| `rdkit` | RDKit 2D descriptors | Physicochemical properties | [RDKit](https://www.rdkit.org) |
| `mordred` | Mordred 2D descriptors (`uv sync --extra mordred`) | Rich physicochemical features | [Moriwaki et al., 2018](https://doi.org/10.1186/s13321-018-0258-y) |
| `unimol` | Uni-Mol 3D molecular embeddings | 3D structure-aware | [Zhou et al., 2023](https://chemrxiv.org/doi/full/10.26434/chemrxiv-2022-jjm0j-v4) |
| `chemeleon` | CheMeleon embeddings | Pretrained chemical language model | [Burns et al., 2025](https://arxiv.org/html/2506.15792v1) |
| `lion` | LiON lipid-specific embeddings | Lipid-tailored representations | [Witten et al., 2025](https://www.nature.com/articles/s41587-024-02490-y) |
| `agile` | AGILE foundation model embeddings | Pretrained embeddings available | [Xu et al., 2024](https://www.nature.com/articles/s41467-024-50619-z) |

For per-role encoders and PC counts, see the [encodings guide](https://evancollins1.github.io/LNPBO/guide/encodings/).

---

## License

MIT License. See [LICENSE](LICENSE) for details.
