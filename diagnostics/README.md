# diagnostics/

Statistical characterization of the LNPDB dataset itself — not BO benchmark results. Each script analyzes a different aspect of cross-study consistency and data quality:

- `compute_icc.py` — Intraclass correlation for functional assay reproducibility
- `anchor_analysis.py` — Consistency of anchor molecule rankings across studies
- `scaffold_analysis.py` — Murcko scaffold diversity as a structural proxy
- `partial_r2.py` — R-squared decomposition by LNP component (IL vs HL vs CHL vs PEG)
- `permutation_decomposition.py` — Permutation importance on XGBoost per component
- `icp_feature_selection.py` — Conformal feature selection (ANOVA + regression)
- `utility_consistency.py` — Spearman correlations on study-level rankings
- `study_audit.py` — Data integrity checks (missing values, outliers)

Each script produces a `.json` results file in this directory.
