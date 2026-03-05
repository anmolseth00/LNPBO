# Surrogate Model Training

Standalone model training and evaluation on LNPDB, independent of the BO loop.
All models predict z-scored `Experiment_value` (transfection efficiency).

**Best model:** XGBoost Optuna-tuned, R^2=0.376 (scaffold split, seed 42).

## Data Pipeline

All scripts share the same data loading and splitting:
- Source: LNPDB (~19,800 formulations) via `data.py` -> `load_lnpdb_dataframe()`
- Requires LNPDB repo at `../LNPDB` or symlinked at `data/LNPDB_repo`
- Default split: scaffold (Bemis-Murcko on IL_SMILES), seed=42, 80/10/10
- Features: Morgan FP (2048-bit, radius 3) + 6 continuous + 65 categorical one-hot = 2119 features (IL-only)
- Multi-component mode adds HL/CHL/PEG fingerprints: 8263 features total

## Scripts

### Training

| Script | Model | Command |
|--------|-------|---------|
| `train_xgb.py` | XGBoost baseline | `python models/train_xgb.py --save-dir models/runs/xgb_full --fp-bits 2048 --fp-radius 3` |
| `tune_xgb.py` | XGBoost + Optuna | `python models/tune_xgb.py` (100 trials, saves to `runs/xgb_tuned/`) |
| `train_lion.py` | D-MPNN | `python models/train_lion.py --save-dir models/runs/il_full --dropout 0.15 --patience 15` |
| `train_gps.py` | GPS-MPNN (D-MPNN + RWSE + attention) | `python models/train_gps.py --save-dir models/runs/gps_il` |
| `train_ensemble.py` | RF + ExtraTrees + XGBoost | `python models/train_ensemble.py` |

### Evaluation

| Script | Purpose | Command |
|--------|---------|---------|
| `eval_multiseed.py` | 5-seed RF/XGB comparison | `python models/eval_multiseed.py` |
| `eval_tuned_multiseed.py` | 5-seed with Optuna hyperparams | `python models/eval_tuned_multiseed.py` |
| `eval_multicomp_multiseed.py` | 5-seed multi-component (IL+HL+CHL+PEG) | `python models/eval_multicomp_multiseed.py` |

### Inference

| Script | Purpose | Command |
|--------|---------|---------|
| `predict.py` | Load trained XGBoost, predict | `python models/predict.py --il-smiles "CC..." --il-molratio 50 ...` |
| `predict.py` | Batch CSV prediction | `python models/predict.py --csv input.csv --out predictions.csv` |

Programmatic: `from models.predict import load_model, predict, featurize`

### Architecture Modules

| Module | Description |
|--------|-------------|
| `mpnn.py` | D-MPNN encoder (Yang et al., J Chem Inf Model 2019) |
| `gps_mpnn.py` | GPS-MPNN: D-MPNN + RWSE (Dwivedi et al. 2022) + global attention (Rampasek et al. 2022) |
| `featurize.py` | RDKit atom/bond featurization for molecular graphs |
| `data.py` | Dataset loading, scaffold splitting, tabular feature encoding |

## Results (seed 42, scaffold split)

| Run | Model | R^2 | Features | Command to Reproduce |
|-----|-------|-----|----------|---------------------|
| `xgb_tuned` | **XGBoost Optuna** | **0.376** | 2119 (IL-only) | `python models/tune_xgb.py` |
| `il_full` | D-MPNN (IL + cat + dropout) | 0.354 | IL graphs + 92 cat | `python models/train_lion.py --dropout 0.15 --patience 15 --no-categorical=False --save-dir models/runs/il_full` |
| `mpnn_multicomp_v2` | D-MPNN (4-component) | 0.355 | IL+HL+CHL+PEG graphs | `python models/train_lion.py --components IL HL CHL PEG --dropout 0.15 --save-dir models/runs/mpnn_multicomp_v2` |
| `xgb_full` | XGBoost default | 0.349 | 2119 (IL-only) | `python models/train_xgb.py --fp-bits 2048 --fp-radius 3 --learning-rate 0.03 --n-estimators 2000 --early-stopping 100 --save-dir models/runs/xgb_full` |
| `gps_il` | GPS-MPNN | 0.328 | IL graphs + RWSE | `python models/train_gps.py --save-dir models/runs/gps_il` |
| `ensemble` | RF+ET+XGB ensemble | 0.350 | 2119 (IL-only) | `python models/train_ensemble.py` |

Multi-component encoding (adding HL/CHL/PEG SMILES) provides no benefit: only 9 unique HLs, 16 CHLs, 14 PEGs -- not enough structural diversity for learned representations to exploit.

## Output Format

Each run saves to `models/runs/<name>/`:
- `test_metrics.json`: R^2, RMSE, MAE, hyperparameters, split info
- `results.png`: Predicted vs actual scatter plot
- `model.json` (XGBoost) or `best_model.pt` (MPNN): Serialized model weights
- `trial_history.json` (Optuna runs only): All trial results
