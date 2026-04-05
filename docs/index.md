# LNPBO

**Bayesian Optimization for Lipid Nanoparticle Formulation Design**

LNPBO is an open-source Python package for data-driven optimization of lipid nanoparticle (LNP) formulations. It implements Bayesian optimization with multiple surrogate models, acquisition functions, and molecular encodings, enabling researchers to efficiently navigate the combinatorial LNP design space.

Introduced in [Collins\*, Seth\* et al.](https://github.com/anmolseth00/LNPBO) (*under review*), LNPBO was benchmarked across 38 optimization strategies and 26 LNP studies from the [LNPDB](https://www.nature.com/articles/s41467-026-68818-1) database (~19,800 formulations).

---

## Key Features

- **Multiple surrogate models** -- Gaussian Process (BoTorch/GPyTorch), XGBoost, Random Forest, NGBoost, Deep Ensemble, CASMOPolitan, TabPFN, Ridge, SNGP, Laplace, Bradley-Terry, GroupDRO, VREx
- **Acquisition functions** -- UCB, EI, LogEI with batch strategies (Kriging Believer, Local Penalization, Thompson Sampling, qLogNoisyEI, GIBBON)
- **9 molecular encodings** -- Morgan FP, Count MFP, RDKit, Mordred, Uni-Mol, CheMeleon, AGILE, LiON, plus the composite LANTERN encoding
- **Discrete pool scoring** -- Direct scoring of candidate formulations without continuous relaxation
- **Custom kernels** -- Tanimoto (molecular fingerprints), Aitchison (compositional data), Deep Kernel Learning
- **Benchmarking suite** -- Statistical infrastructure for evaluating optimization strategies

---

## Installation

```bash
git clone https://github.com/evancollins1/LNPBO.git
cd LNPBO
pip install .
```

For GP support (BoTorch/GPyTorch):

```bash
pip install ".[gp]"
```

For all optional dependencies:

```bash
pip install ".[all]"
```

### LNPDB Setup

Many features (benchmarks, LNPDB study examples) require the [LNPDB](https://github.com/evancollins1/LNPDB) database. Clone it as a sibling directory and create the expected symlink:

```bash
cd ..
git clone https://github.com/evancollins1/LNPDB.git
cd LNPBO
ln -s ../LNPDB data/LNPDB_repo
```

### Development Installation

```bash
git clone https://github.com/evancollins1/LNPBO.git
git clone https://github.com/evancollins1/LNPDB.git
cd LNPBO
ln -s ../LNPDB data/LNPDB_repo
uv venv && source .venv/bin/activate
pip install -e ".[dev]"
```

---

## Quick Start

```python
from LNPBO.data.dataset import Dataset
from LNPBO.space.formulation import FormulationSpace
from LNPBO.optimization.optimizer import Optimizer

# Load your LNP screen data
dataset = Dataset.from_lnpdb_csv("my_screen.csv")

# Encode molecular features (LANTERN = count MFP + RDKit, PCA to 5 PCs)
encoded = dataset.encode_dataset(feature_type="lantern")

# Build the formulation space
space = FormulationSpace.from_dataset(encoded)

# Initialize optimizer and suggest next batch
optimizer = Optimizer(
    space=space,
    surrogate_type="gp",
    candidate_pool=encoded.df,
    acquisition_type="UCB",
    kappa=5.0,
    batch_size=24,
)
suggestions = optimizer.suggest(output_csv="round1.csv")
```

See the [Getting Started](guide/quickstart.md) guide for detailed examples.

---

## Choosing a Strategy

Based on a benchmark of 38 strategies across 26 LNP studies:

| Scenario | Recommended Strategy |
|----------|---------------------|
| Default / general use | `NGBoost-UCB` or `RF-TS` |
| Screening diverse IL libraries | Tree-based surrogates (RF, XGBoost, NGBoost) |
| Ratio-only optimization | `CASMOPolitan` or GP-based methods |
| Limited compute budget | `XGBoost-Greedy` (no uncertainty quantification) |

See [Optimization Strategies](guide/strategies.md) for details on each strategy family.

---

## CLI

LNPBO provides a command-line interface with five subcommands:

```bash
# Encode raw formulation CSV into numeric features
lnpbo encode --input data.csv --output encoded.csv \
    --IL-n-pcs-count-mfp 5 --IL-n-pcs-rdkit 5 --HL-n-pcs-count-mfp 3 --HL-n-pcs-rdkit 3

# Suggest next batch of formulations
lnpbo suggest --dataset data.csv --output round1.csv --surrogate-type xgb_ucb --batch-size 12

# Register experimental results back into dataset
lnpbo register --dataset round1.csv --results results.csv --output updated.csv

# Propose new ionizable lipids with uncertainty-aware scoring
lnpbo propose-ils --dataset data.csv --output proposed_ils.csv --n-candidates 20000 --n-output 100

# Save surrogate model checkpoint
lnpbo checkpoint save --model model.joblib --surrogate-type gp \
    --columns cols.joblib --output-dir checkpoint_dir
```

---

## Citation

**Benchmarking Optimization Strategies for Lipid Nanoparticle Design: 38 Strategy Configurations Across 26 Studies**

Evan Collins\*, Anmol Seth\*, Robert Langer, Daniel G. Anderson

*Under review*

---

## License

LNPBO is released under the [MIT License](https://github.com/anmolseth00/LNPBO/blob/main/LICENSE).
