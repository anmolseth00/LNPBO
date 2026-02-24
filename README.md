# Lipid Nanoparticle Bayesian Optimization (LNPBO)

<p align="center">
  <img src="LNPBO.png" alt="alt text" width="600px" align="middle"/>
</p>

*Figure: LNPBO is a Python implementation of Bayesian optimization tailored for lipid nanoparticle optimization.*

# Description

LNPBO is an open source method for implementing Bayesian optimization (BO) for lipid nanoparticle (LNP) optimization, as introduced in [our paper](#citation) (*under review*).

Things LNPBO can do:

- Suggest new LNPs to make based on initial batch

- Back-and-forth suggest new LNPs based on subsequent batches

To do these things, LNPBO encodes the inputted LNPs. Besides molar ratios, featurizing lipid identity involves generating encodings of Morgan fingerprints, Mordred descriptors, or LiON model fingerprints. The inputted LNPs are expected to follow a column format similar to what we introduced in [LNPDB](https://www.nature.com/articles/s41467-026-68818-1). We detail this input organization in the [Usage](#usage) section, where we walk through examples that illustrate what LNPBO can do.


Future version of LNPBO will also do these things:

- Suggest initial LNPs to make based on ingredient list

- Make landscape plots

_____

# Table of contents

- [Getting started / installation](#installation)
- [Usage](#usage)
- [Citation](#citation)

____

# Getting started / installation

To set up LNPBO locally, follow these steps.

We will first install this repository.

```
git clone https://github.com/evancollins1/LNPBO.git
```

We will next create the conda environment `lnpbo`.

```
conda create -n lnpbo python=3.11
conda activate lnpbo
conda install -c conda-forge chemprop=2.2.2 bayesian-optimization=3.2.0 mordred pandas jupyter
```

In order to use LiON fingerprint encodings, we will also need to create the conda environment `lnp_ml` as introduced in our [prior](https://github.com/evancollins1/LNPDB?tab=readme-ov-file#training--testing-lion-deep-learning-model) [repos](https://github.com/jswitten/LNP_ML?tab=readme-ov-file#install-python-dependencies). Make sure to `conda deactivate` from `lnpbo` before running the below chunk.

```
conda create -n lnp_ml python=3.8
conda activate lnp_ml
pip install chemprop==1.7.0
```

____

# Usage

We will now walk through the capabilities of LNPBO by discussing three example use cases, each with descriptions in this README as well as Jupyter notebooks to process the provided example data (see `/examples` folder).

1. Example 1: varying ratios

2. Example 2: varying ratios, helper lipid

3. Example 3: varying rations, helper lipid, ionizable lipid

Note that LNPBO can also handle other things (e.g., varying cholesterol and PEG lipid), but these three minimal examples provide the framework to understand what LNPBO can do. Also note that the transfection results for the example data provided in this repo are only for exercise purposes, as they have been synthetically generated. Note that IL denotes ionizable lipid, HL denotes helper lipid, CHL denotes cholesterol, and PEG denotes PEG lipid.

## Example 1: varying ratios

For example 1, let's say a scientist made and tested an initial batch of LNPs where only ratios were varied. More specifically, let's say that they kept constant the lipid identities (cKK-E12 IL, DOPE HL, Cholesterol, DMG-PEG2000) and only varied some lipid molar ratios (IL, HL, CHL) and the IL:mRNA mass ratio. This example could reflect when a scientist is confident in the lipid identities but wants to optimize their relative amounts. They now want LNPBO to suggest some number of new LNPs to make for the next batch based on the initial batch.

Take note of the organization of `/examples/example1/example1.csv`, which is the spreadsheet prepared by the scientist for the initial batch (round 0). This is the expected input organization for LNPBO. For each row (i.e., LNP formulation), the scientist should specify the following columns (i.e., structure-function data) in a .csv spreadsheet: 

- `IL_name`: name of ionizable lipid

- `IL_SMILES`: SMILES of ionizable lipid

- `IL_molratio`: molar ratio of ionizable lipid

- `IL_to_nucleicacid_massratio`: ionizable lipid-to-nucleic acid mass ratio

- `HL_name`: name of helper lipid

- `HL_SMILES`: SMILES of helper lipid

- `HL_molratio`: molar ratio of helper lipid

- `CHL_name`: name of cholesterol

- `CHL_SMILES`: SMILES of cholesterol

- `CHL_molratio`: molar ratio of cholesterol

- `PEG_name`: name of PEG lipid

- `PEG_SMILES`: SMILES of PEG lipid

- `PEG_molratio`: molar ratio of PEG lipid

- `Experiment_value`: functional value (e.g., transfection luminescence)


In cases like this one where the lipid identities are not being varied, entering the SMILES data is optional. 

For LNPBO to suggest new LNPs for the next batch based on this initial batch, follow the instructions below. Note that the code chunks below are also provided in `/examples/example1/example1.ipynb`.

Load packages and `example1.csv` (i.e., the spreadsheet prepared by the scientist for the initial batch).

```
import pandas as pd
import sys
from pathlib import Path
project_root = Path.cwd().resolve().parents[2]
sys.path.append(str(project_root))
from LNPBO.data.dataset import Dataset
from LNPBO.space.formulation import FormulationSpace
from LNPBO.optimization.optimizer import Optimizer

# Load dataset
dataset = Dataset.from_lnpdb_csv("example1.csv")
```

Next, encode the dataset and build the `FormulationSpace` object to be used for BO. `encoding_csv_path` specifies the directory for saving the encodings.

```
# Encode dataset
encoded_dataset = dataset.encode_dataset(
    encoding_csv_path="example1_encodings.csv",
)

# Build FormulationSpace for BO
space = FormulationSpace.from_dataset(encoded_dataset)
```

Next, we will initialize the `Optimizer`. `space` specifies the `FormulationSpace`. `type` specifies the acquisition function type, either "UCB" (upper confidence bound) or "EI" (expected improvement). If `type="UCB"`, then its `kappa` value should be defined. If `type="EI"`, then its `xi` value should be defined. `kappa` and `xi` balance exploration vs. exploitation, with higher values favoring exploration, as outlined nicely [here](https://github.com/bayesian-optimization/BayesianOptimization/blob/master/examples/exploitation_vs_exploration.ipynb). `alpha` specifies a small constant added to the diagonal of the kernel matrix of the Gaussian Process (GP), modeling experimental noise, with higher values reflecting noisier data. `random_seed` specifies the random seed. `batch_size` specifies the number of LNPs to suggest for the next batch.

```
# Initialize Optimizer
optimizer = Optimizer(
    space=space,
    type="UCB",
    kappa=5.0,
    alpha=1e-6,
    random_seed=42,
    batch_size=24
)
```

Finally, to suggest `batch_size` number of new LNPs for the next batch, run the following. `example1_round1.csv` specifies the directory for saving the BO results. Note that the resulting .csv appends the `batch_size` number of new LNPs to the bottom of the originally-inputted dataset (i.e., `example1.csv`). The `Experiment_value` (e.g., transfection luminescence) column for these new LNPs will be left blank, as it's now the scientist's job to test the suggested LNPs in lab.

```
# Perform first round of BO suggestions
round1_suggestions = optimizer.suggest(
    output_csv="example1_round1.csv"
)
```

If the scientist does ultimately come back with the `Experiment_value` results for the new round 1 LNPs, then it's possible for LNPBO to suggest another batch of LNPs based on the updated results. This use case is not described in example 1 but it is described in example 2, so see below if applicable.

## Example 2: varying ratios, helper lipid

For example 2, let's say a scientist made and tested an initial batch of LNPs where ratios and HL identity were varied. More specifically, let's say that They kept constant the IL identity, CHL identity, PEG identity, and PEG molar ratio. They varied the HL identity, IL molar ratio, HL molar ratio, CHL molar ratio, and IL:mRNA mass ratio. This example could reflect when a scientist is confident in the IL identity but wants to explore alternative HLs and optimize their relative amounts. They now want LNPBO to suggest some number of new LNPs to make for the next batch based on the initial batch.

Take note of the organization of `/examples/example2/example2.csv`, which is the spreadsheet prepared by the scientist for the initial batch (round 0). See example 1 above for details about the organization.

For LNPBO to suggest new LNPs for the next batch based on this initial batch, follow the instructions below. Note that the code chunks below are also provided in `/examples/example2/example2.ipynb`.

Load packages and `example2.csv` (i.e., the spreadsheet prepared by the scientist for the initial batch).

```
import pandas as pd
import sys
from pathlib import Path
project_root = Path.cwd().resolve().parents[2]
sys.path.append(str(project_root))
from LNPBO.data.dataset import Dataset
from LNPBO.space.formulation import FormulationSpace
from LNPBO.optimization.optimizer import Optimizer

# Load dataset
dataset = Dataset.from_lnpdb_csv("example2.csv")
```

Next, encode the dataset and build the `FormulationSpace` object to be used for BO. `HL_n_pcs_morgan` specifies the number of principal components to encode the HLs. `HL_n_pcs_mordred` specifies the number of principal components to encode the HLs. If both Morgan and Mordred are specified, then they will be combined. `encoding_csv_path` specifies the directory for saving the encodings.

```
# Encode dataset
encoded_dataset = dataset.encode_dataset(
    HL_n_pcs_morgan=5,
    HL_n_pcs_mordred=5,
    encoding_csv_path="example2_encodings.csv",
)

# Build FormulationSpace for BO
space = FormulationSpace.from_dataset(encoded_dataset)
```

Next, we will initialize the `Optimizer`. See example 1 above for details about the options for `Optimizer`.

```
# Initialize Optimizer
optimizer = Optimizer(
    space=space,
    type="UCB",
    kappa=5.0,
    alpha=1e-6,
    random_seed=42,
    batch_size=24
)
```

Finally, to suggest `batch_size` number of new LNPs for the next batch, run the following. `example2_round1.csv` specifies the directory for saving the BO results. Note that the resulting .csv appends the `batch_size` number of new LNPs to the bottom of the originally-inputted dataset (i.e., `example2.csv`). The `Experiment_value` (e.g., transfection luminescence) column for these new LNPs will be left blank, as it's now the scientist's job to test the suggested LNPs in lab.

```
# Perform first round of BO suggestions
round1_suggestions = optimizer.suggest(
    output_csv="example2_round1.csv"
)
```

If the scientist does ultimately come back with the `Experiment_value` results for the new round 1 LNPs, then it's possible for LNPBO to suggest another batch of LNPs based on the updated results. To do this, run the following, where in this example the scientist has saved these round 1 results in `example2_round1_w_results.csv`. The new suggested LNPs for round 2 are included in the outputted `example2_round2.csv`. Note that you can also adjust the optimizer at this point too (e.g., change kappa value or batch size).

```
# Wrap into Dataset
encoded_dataset = optimizer.update("example2_round1_w_results.csv")

# Suggest round 2
round2_suggestions = optimizer.suggest(output_csv="example2_round2.csv")
```

## Example 3: varying ratios, helper lipid, ionizable lipid

For example 3, let's say a scientist made and tested an initial batch of LNPs where molar ratios and IL & HL identities were varied. More specifically, let's say that they kept constant the CHL identity, PEG identity, PEG molar ratio, and IL:mRNA mass ratio. They varied the IL identity, HL identity, IL molar ratio, HL molar ratio, and CHL molar ratio. This example could reflect when a scientist wants to explore a library of ILs with different HLs and optimize their relative amounts. They now want LNPBO to suggest some number of new LNPs to make for the next batch based on the initial batch.

Take note of the organization of `/examples/example3/example3.csv`, which is the spreadsheet prepared by the scientist for the initial batch (round 0). See example 1 above for details about the organization.

Load packages and `example3.csv` (i.e., the spreadsheet prepared by the scientist for the initial batch).

```
import pandas as pd
import sys
from pathlib import Path
project_root = Path.cwd().resolve().parents[2]
sys.path.append(str(project_root))
from LNPBO.data.dataset import Dataset
from LNPBO.space.formulation import FormulationSpace
from LNPBO.optimization.optimizer import Optimizer

# Load dataset
dataset = Dataset.from_lnpdb_csv("example3.csv")
```

Next, encode the dataset and build the `FormulationSpace` object to be used for BO. `HL_n_pcs_morgan` specifies the number of principal components to encode the HLs, analagous for `IL_n_pcs_morgan`. `HL_n_pcs_mordred` specifies the number of principal components to encode the HLs, analagous for `IL_n_pcs_mordred`. If both Morgan and Mordred are specified for IL or HL, then they will be combined. `encoding_csv_path` specifies the directory for saving the encodings.

```
# Encode dataset
encoded_dataset = dataset.encode_dataset(
    HL_n_pcs_morgan=5,
    HL_n_pcs_mordred=5,
    IL_n_pcs_morgan=5,
    IL_n_pcs_mordred=5,
    encoding_csv_path="example3_encodings.csv",
)

# Build FormulationSpace for BO
space = FormulationSpace.from_dataset(encoded_dataset)
```

Alternatively, rather than encoding IL chemistry with Morgan fingerprints and Mordred descriptors, fingerprints from the LiON deep learning model can be used. These fingerprints are extracted from the penultimate linear layer of the LiON deep learning model trained on LNPDB, as introduced in our prior [repo](https://github.com/evancollins1/LNPDB).

```
# Encode dataset
encoded_dataset = dataset.encode_dataset(
    HL_n_pcs_morgan=5,
    HL_n_pcs_mordred=5,
    IL_n_pcs_lion=5,
    encoding_csv_path="example3_encodings.csv",
)

# Build FormulationSpace for BO
space = FormulationSpace.from_dataset(encoded_dataset)
```

Next, we will initialize the `Optimizer`. See example 1 above for details about the options for `Optimizer`.

```
# Initialize Optimizer
optimizer = Optimizer(
    space=space,
    type="UCB",
    kappa=5.0,
    alpha=1e-6,
    random_seed=42,
    batch_size=24
)
```

Finally, to suggest `batch_size` number of new LNPs for the next batch, run the following. `example3_round1.csv` specifies the directory for saving the BO results. Note that the resulting .csv appends the `batch_size` number of new LNPs to the bottom of the originally-inputted dataset (i.e., `example3.csv`). The `Experiment_value` (e.g., transfection luminescence) column for these new LNPs will be left blank, as it's now the scientist's job to test the suggested LNPs in lab.

```
# Perform first round of BO suggestions
round1_suggestions = optimizer.suggest(
    output_csv="example3_round1.csv"
)
```
If the scientist does ultimately come back with the `Experiment_value` results for the new round 1 LNPs, then it's possible for LNPBO to suggest another batch of LNPs based on the updated results. This use case is not described in example 1 but it is described in example 3, so see above if applicable.

______

# Citation

**A self-driving lab for the directed evolution of mRNA lipid nanoparticles**

Evan Collins, Rajith S. Manan, Samuel Detmer, Till Muser, Easwer Raman, Richard Zhu, Srinivas Balagopal, Yanshu Shi, Jungyong Ji, Akash Gupta, Yizong Hu, Suthathip Trongjit, Wontaek Chung, Haseeb Mughal, Jacob Witten, Anna Lapteva, Ashwin Pasupathy, Wonpil Im, Robert Langer, Daniel G. Anderson

*Under review*
