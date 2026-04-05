# models/

Research surrogate models accessible through the Optimizer API but not part of the standard 37-strategy benchmark. Includes Deep Ensemble, SNGP, Laplace approximation, GroupDRO, V-REx, and Bradley-Terry. These are available via `--surrogate-type` for ad-hoc testing.

`models/experimental/` contains meta-learning surrogates (MAML, FSBO) and graph neural networks (MPNN, GPS-MPNN) that require additional dependencies.

The production surrogates used in the benchmark (RF, XGBoost, NGBoost, GP, CQR, Ridge) are implemented directly in `optimization/discrete.py` and `optimization/gp_bo.py`.
