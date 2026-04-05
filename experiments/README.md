# experiments/

Experiment reproduction configs and scripts for the paper. This directory contains everything needed to reproduce or extend the paper's experiments, but does not contain the benchmark engine itself (that's in `benchmarks/`).

- `ablations/` — JSON configs for 8 ablation studies (encoding, batch size, budget, PCA, kernel, kappa, warmup, small study)
- `run_ablation.py` — Unified ablation runner that reads a config and dispatches to the benchmark harness
- `data_integrity/` — Stratification audit, corrected study definitions (`studies.json`, `studies_with_ids.json`)
- `cross_study_transfer.py`, `calibration_analysis.py` — Standalone analysis experiments
- `analysis/` — Post-hoc ablation result analysis and figure generation
- `infrastructure/` — Setup scripts (chemprop v1 venv, COMET)
- `CATALOG.md` — Full experiment index with run commands and status
