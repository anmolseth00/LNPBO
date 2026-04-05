# CLI Reference

LNPBO provides the `lnpbo` command-line interface with five subcommands for encoding, optimization, result registration, ionizable lipid proposal, and model checkpointing.

---

## `lnpbo encode`

Encode a raw LNPDB-format CSV into numeric features (molecular fingerprints, descriptors, embeddings) with optional dimensionality reduction.

```bash
# LANTERN-style encoding (count Morgan FP + RDKit descriptors, 5 PCs each for IL)
lnpbo encode --input raw_data.csv --output encoded.csv \
    --IL-n-pcs-count-mfp 5 --IL-n-pcs-rdkit 5 \
    --HL-n-pcs-count-mfp 3 --HL-n-pcs-rdkit 3

# Morgan fingerprints only
lnpbo encode --input raw_data.csv --output encoded.csv \
    --IL-n-pcs-morgan 5 --HL-n-pcs-morgan 3

# LiON embeddings (cannot combine with Morgan/Mordred for IL)
lnpbo encode --input raw_data.csv --output encoded.csv \
    --IL-n-pcs-lion 5 --HL-n-pcs-count-mfp 3
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--input` | *required* | Path to raw LNPDB-format CSV |
| `--output` | *required* | Path for encoded output CSV |
| `--{ROLE}-n-pcs-morgan` | 0 | Morgan fingerprint PCs (ROLE: IL, HL, CHL, PEG) |
| `--{ROLE}-n-pcs-mordred` | 0 | Mordred descriptor PCs |
| `--{ROLE}-n-pcs-count-mfp` | 0 | Count Morgan fingerprint PCs |
| `--{ROLE}-n-pcs-rdkit` | 0 | RDKit descriptor PCs |
| `--{ROLE}-n-pcs-unimol` | 0 | Uni-Mol embedding PCs |
| `--IL-n-pcs-lion` | 0 | LiON embedding PCs (IL only; cannot combine with Morgan/Mordred for IL) |
| `--reduction` | `pls` | Dimensionality reduction: `pca`, `pls`, or `none` |
| `--only-encodings` | false | Output only per-lipid encoding tables |

---

## `lnpbo suggest`

Suggest the next batch of formulations using a surrogate model and acquisition function.

```bash
lnpbo suggest --dataset encoded.csv --output round1.csv \
    --surrogate-type gp --gp-engine botorch --batch-size 12 --kappa 5.0
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | *required* | Path to encoded dataset CSV |
| `--output` | *required* | Output CSV path for suggestions |
| `--surrogate-type` | `xgb_ucb` | Surrogate model (see [Optimization Strategies](strategies.md) and [Surrogate Models](../surrogates.md)) |
| `--acquisition-type` | `UCB` | Acquisition function: `UCB`, `EI`, `LogEI` |
| `--batch-strategy` | `kb` | Batch strategy: `kb`, `rkb`, `lp`, `ts`, `qlogei`, `gibbon` (GP) or `greedy`, `ts` (discrete) |
| `--batch-size` | 12 | Number of formulations to suggest |
| `--kappa` | 5.0 | UCB exploration parameter |
| `--xi` | 0.01 | EI/LogEI exploration parameter |
| `--seed` | 1 | Random seed |
| `--normalize` | `copula` | Target normalization: `copula`, `zscore`, `none` |
| `--reduction` | `pca` | Dimensionality reduction: `pca`, `pls`, `none` (note: `encode` defaults to `pls`) |
| `--feature-type` | `lantern` | Molecular encoding type |
| `--pool` | dataset | Path to candidate pool CSV (uses dataset if not set) |
| `--gp-engine` | `botorch` | GP backend: `botorch`, `sklearn` |
| `--kernel-type` | `matern` | GP kernel: `matern`, `tanimoto`, `aitchison`, `dkl`, `rf`, `compositional`, `robust` |
| `--context-features` | false | Include context features in the surrogate model |
| `--alpha` | 1e-6 | Noise regularization for sklearn GP |
| `--surrogate-kwargs` | None | JSON string of extra surrogate keyword arguments |

---

## `lnpbo register`

Register completed experimental results back into the dataset for the next optimization round.

```bash
lnpbo register --dataset round1.csv --results results.csv --output updated.csv
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | *required* | Path to current dataset CSV |
| `--results` | *required* | Path to CSV with completed results (must contain `Formulation_ID`, `Round`, `Experiment_value`) |
| `--output` | *required* | Output path for updated dataset CSV |

---

## `lnpbo propose-ils`

Propose new ionizable lipids with uncertainty-aware scoring using generative chemistry.

```bash
lnpbo propose-ils --dataset data.csv --output proposed_ils.csv \
    --n-candidates 20000 --n-output 100 --lcb-kappa 1.0
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | *required* | Path to LNPDB-format CSV with `IL_SMILES` and `Experiment_value` |
| `--output` | *required* | Output CSV path for proposed ILs |
| `--n-candidates` | 20000 | Number of candidates to generate |
| `--n-output` | 100 | Number of candidates to output |
| `--diversity-pool` | 1000 | Top-N by score for diversity selection |
| `--seed` | 42 | Random seed |
| `--max-mutations` | 2 | Max SELFIES mutations per candidate |
| `--lcb-kappa` | 1.0 | LCB weight on uncertainty |
| `--lcb-mode` | `std` | LCB mode: `std` (mean - kappa\*std) or `lower` (MAPIE lower bound) |
| `--n-jobs` | 1 | Parallel jobs for model fitting |
| `--confidence-level` | 0.68 | MAPIE confidence level for intervals |
| `--il-pcs-count-mfp` | 5 | Count-MFP PCs for IL encoding |
| `--il-pcs-rdkit` | 5 | RDKit PCs for IL encoding |
| `--reduction` | `pls` | Dimensionality reduction: `pca`, `pls`, `none` |
| `--amine-smarts` | `[NX3;H0;...]` | SMARTS pattern for tertiary amine filter |
| `--mw-min` | None | Minimum molecular weight filter |
| `--mw-max` | None | Maximum molecular weight filter |
| `--logp-min` | None | Minimum logP filter |
| `--logp-max` | None | Maximum logP filter |
| `--max-atoms` | None | Maximum atom count filter |
| `--max-attempts` | None | Maximum generation attempts |

---

## `lnpbo checkpoint`

Save or load surrogate model checkpoints (directory format).

### `lnpbo checkpoint save`

```bash
lnpbo checkpoint save --model model.joblib --surrogate-type gp \
    --columns cols.joblib --output-dir checkpoint_dir \
    --scaler scaler.joblib --round 0
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | *required* | Path to serialized model |
| `--surrogate-type` | *required* | Surrogate family identifier (e.g., `gp`, `xgb`, `ngboost`) |
| `--columns` | *required* | Path to serialized column list (joblib) |
| `--output-dir` | *required* | Output checkpoint directory |
| `--scaler` | None | Path to serialized scaler (joblib) |
| `--round` | 0 | Current BO round number |

### `lnpbo checkpoint load`

```bash
lnpbo checkpoint load checkpoint_dir
```

| Argument | Default | Description |
|----------|---------|-------------|
| `checkpoint_dir` | *required* | Path to checkpoint directory |

Prints the checkpoint metadata (JSON), model type, scaler type, and feature column count.
