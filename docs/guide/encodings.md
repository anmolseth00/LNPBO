# Molecular Encodings

LNPBO supports multiple molecular encoding schemes for representing lipid identity. The choice of encoding affects how structural differences between lipids are captured by the surrogate model.

Encodings are applied per-component (IL, HL, CHL, PEG) and can be combined with PCA or PLS for dimensionality reduction.

---

## Overview

| Encoding | Key | Dimensionality | Description |
|----------|-----|----------------|-------------|
| Morgan FP | `mfp` | 2048 (binary) | Circular substructure fingerprints |
| Count Morgan FP | `count_mfp` | 2048 (integer) | Count-based Morgan fingerprints |
| RDKit Descriptors | `rdkit` | ~200 | 2D physicochemical descriptors |
| Mordred Descriptors | `mordred` | ~1600 | Comprehensive 2D molecular descriptors |
| Uni-Mol | `unimol` | 512 | 3D-aware pretrained molecular embeddings |
| CheMeleon | `chemeleon` | 768 | Foundation model molecular embeddings |
| AGILE | `agile` | 256 | Foundation model embeddings for lipids |
| LiON | `lion` | 256 | Lipid-specific pretrained embeddings |
| **LANTERN** | `lantern` | 5 PCs | Count MFP + RDKit, PCA-reduced (default) |

---

## LANTERN (Default)

LANTERN is the composite encoding used by default in LNPBO. It concatenates count Morgan fingerprints and RDKit descriptors, then applies PCA reduction to 5 principal components per component role.

```python
encoded = dataset.encode_dataset(feature_type="lantern")
```

LANTERN performed best overall in benchmarks, providing a good balance between structural resolution and dimensionality. The PCA reduction prevents overfitting with small training sets typical of early-stage LNP screens.

---

## Morgan Fingerprints (`mfp`)

Binary circular fingerprints based on atom neighborhoods of radius 3, hashed to 2048 bits. A standard cheminformatics representation.

```python
encoded = dataset.encode_dataset(feature_type="mfp")
```

Morgan fingerprints are fast to compute and provide a reasonable baseline encoding. They capture substructure presence/absence but lose count information.

**Reference:** Rogers, D. & Hahn, M. "Extended-connectivity fingerprints." *J. Chem. Inf. Model.* 50(5), 2010.

---

## Count Morgan Fingerprints (`count_mfp`)

Like Morgan fingerprints but retaining substructure counts instead of binary presence/absence. This preserves information about repeated substructures (e.g., multiple ester groups in ionizable lipids).

```python
encoded = dataset.encode_dataset(feature_type="count_mfp")
```

Count fingerprints are a component of the LANTERN composite encoding. They are especially useful with the Tanimoto kernel, which naturally handles count vectors.

---

## RDKit Descriptors (`rdkit`)

A set of approximately 200 RDKit 2D physicochemical descriptors including molecular weight, LogP, topological polar surface area, hydrogen bond counts, and ring system descriptors.

```python
encoded = dataset.encode_dataset(feature_type="rdkit")
```

RDKit descriptors capture global molecular properties complementary to the local substructure information in fingerprints. They are a component of the LANTERN composite encoding.

---

## Mordred Descriptors (`mordred`)

Comprehensive 2D molecular descriptors computed by the Mordred package (~1600 descriptors after dropping constant and correlated features). Requires the `mordred` optional dependency.

```python
# Requires: pip install "LNPBO[mordred]"
encoded = dataset.encode_dataset(feature_type="mordred")
```

Mordred provides the richest physicochemical feature set but at higher dimensionality. Best suited when larger training sets are available.

**Reference:** Moriwaki, H. et al. "Mordred: a molecular descriptor calculator." *J. Cheminformatics* 10(1), 2018.

---

## Uni-Mol (`unimol`)

3D-aware molecular representations from the Uni-Mol pretrained model. These embeddings capture conformational information beyond 2D structure.

```python
encoded = dataset.encode_dataset(feature_type="unimol")
```

Uni-Mol embeddings are pretrained on a large corpus of molecular structures and may capture conformational preferences relevant to LNP self-assembly.

**Reference:** Zhou, G. et al. "Uni-Mol: A Universal 3D Molecular Representation Learning Framework." *ICLR 2023*.

---

## CheMeleon (`chemeleon`)

Embeddings from the CheMeleon foundation model, which is trained on diverse chemical property prediction tasks.

```python
encoded = dataset.encode_dataset(feature_type="chemeleon")
```

CheMeleon embeddings may capture property-relevant molecular features learned from large-scale multi-task training.

---

## AGILE (`agile`)

Embeddings from the AGILE foundation model, designed for molecular property prediction with attention to lipid-relevant features.

```python
encoded = dataset.encode_dataset(feature_type="agile")
```

AGILE embeddings provide a pretrained representation that may capture lipid-specific structural patterns.

---

## LiON (`lion`)

Lipid-specific embeddings from the LiON model, trained on lipid nanoparticle activity data. These embeddings are tailored to the ionizable lipid design space.

```python
encoded = dataset.encode_dataset(feature_type="lion")
```

LiON was developed specifically for LNP applications, encoding structure-activity knowledge from lipid transfection data.

---

## Dimensionality Reduction

All encodings can be combined with PCA or PLS reduction via the `compute_pcs` function. The `encode_dataset` method handles this automatically.

### PCA (Default)

Unsupervised reduction that retains the top principal components of the fingerprint matrix.

### PLS

Supervised reduction using partial least squares regression. PLS components are linear combinations that maximize covariance with the target variable, potentially providing more informative low-dimensional representations for prediction.

```python
encoded = dataset.encode_dataset(
    feature_type="lantern",
    reduction="pls",
)
```

!!! note
    PLS reduction requires target values (`Experiment_value`) for fitting the projection. In prospective use, the PLS model is fit on the training data and applied to the candidate pool.

---

## Custom Encoding Configuration

For fine-grained control over encoders and PC counts per component role:

```python
from LNPBO.data.dataset import encoders_for_feature_type

# 5 PCs for IL, 3 for other components
encoders = encoders_for_feature_type("lantern", il_pcs=5, other_pcs=3)
# => {"IL": {"count_mfp": 5, "rdkit": 5},
#     "HL": {"count_mfp": 3, "rdkit": 3}, ...}

encoded = dataset.encode_dataset(encoders=encoders)
```

---

## Choosing an Encoding

| Scenario | Recommended Encoding |
|----------|---------------------|
| Default / general use | `lantern` |
| Small training set (< 50 formulations) | `lantern` (PCA-reduced) |
| Large training set (> 200 formulations) | `mordred` or `mfp` |
| Lipid-specific pretrained features | `lion` or `agile` |
| Fast baseline | `mfp` |
| Ratio-only optimization (no structural variation) | No encoding needed |
