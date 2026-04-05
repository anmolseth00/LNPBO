# Data API

The data module handles dataset loading, molecular encoding, and dimensionality reduction for LNP formulations.

---

## Dataset

The core class for loading LNPDB-format CSV files, encoding lipid molecular features, and tracking optimization rounds.

::: LNPBO.data.dataset.Dataset
    options:
      members:
        - __init__
        - from_lnpdb_csv
        - encode_dataset
        - max_round
        - metadata

---

## Encoding Helpers

### encoders_for_feature_type

::: LNPBO.data.dataset.encoders_for_feature_type

---

## Dimensionality Reduction

### compute_pcs

::: LNPBO.data.compute_pcs.compute_pcs

---

## Constants

### Feature Type Encoders

The mapping from named feature types to their constituent encoders:

| Feature Type | Encoders | Description |
|-------------|----------|-------------|
| `mfp` | Morgan FP | Binary Morgan fingerprints (2048-bit) |
| `count_mfp` | Count MFP | Count-based Morgan fingerprints |
| `rdkit` | RDKit | RDKit 2D physicochemical descriptors |
| `mordred` | Mordred | Mordred 2D molecular descriptors |
| `unimol` | Uni-Mol | Uni-Mol pretrained embeddings |
| `chemeleon` | CheMeleon | CheMeleon foundation model embeddings |
| `lion` | LiON | LiON lipid-specific embeddings |
| `agile` | AGILE | AGILE foundation model embeddings |
| `lantern` | Count MFP + RDKit | Composite encoding, PCA-reduced (default) |

### Required Columns

::: LNPBO.data.dataset.LNPDB_REQUIRED_COLUMNS
