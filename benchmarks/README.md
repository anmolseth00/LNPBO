# benchmarks/

Benchmark evaluation engine for the within-study retrospective simulation. `runner.py` runs the closed-loop BO simulation (seed pool, iterative acquisition, oracle reveal). `benchmark.py` orchestrates runs across studies, strategies, and seeds. `strategy_registry.py` defines all 38 strategy configurations and their display names. `constants.py` sets shared parameters (seeds, batch size, study thresholds). `stats.py` provides bootstrap CIs, Wilcoxon tests, BH-FDR correction, and effect sizes.

`baselines/` contains non-BO comparison strategies: predict-and-rank, AGILE foundation model, COMET reaction predictor, oracle ceiling, pure exploration, and diversity selection.

Analysis scripts (`analyze_*.py`, `gap_analysis.py`) process benchmark results into tables and figures. These are standalone scripts, not imported as library code.

See `experiments/CATALOG.md` for the full experiment index and `docs/guide/study_design.md` for the simulation protocol.
