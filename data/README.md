# data/

Data loading, molecular encoding, and LNPDB integration. The `Dataset` class handles CSV I/O, schema validation, duplicate averaging, and round tracking. `lnpdb_bridge.py` loads the full LNPDB database with z-score normalization.

Molecular encoding generators (`generate_*.py`) convert lipid SMILES into numeric feature vectors: Morgan fingerprints, RDKit descriptors, Mordred descriptors, LiON embeddings (via chemprop v1 subprocess), Uni-Mol embeddings, CheMeleon embeddings, and AGILE embeddings. `compute_pcs.py` applies PCA/PLS dimensionality reduction. The default LANTERN encoding combines count Morgan fingerprints + RDKit descriptors, reduced to 5 PCs per lipid role.
