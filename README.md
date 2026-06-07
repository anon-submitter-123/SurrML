# SurrogateML
A framework for frozen-classifier inference under missing features via heteroscedastic Monte Carlo marginalization - no retraining required.

# Experiments
This repository contains the experimental code for SurrogateML, a missing-data wrapper for frozen machine-learning models. 
The core idea is to keep an already trained classifier fixed and use a conditional generator to produce plausible completions for missing features at test time.

## Repository Layout

- `baselines.py` - baseline missing-data methods, including mean imputation, KNN imputation, MICE, RF-Imp, MissForest-style imputation, and native-missing tree models.
- `data_loader.py` - dataset loaders and preprocessing for the benchmark datasets.
- `download_datasets.py` - helper script for downloading datasets that are not bundled locally.
- `missingness.py` - MCAR, MAR, and MNAR missingness mechanisms.
- `novel_generator.py` - HetIWAE / adversarial task-aware generator used by SurrogateML.
- `steering_v2.py` - soft Monte Carlo routing for frozen decision trees and random forests.
- `run_experiment.py` - main experiment runner and evaluation utilities.
- `run_ablations.py` - ablation experiments for generator and rerouting variants.
- `analyze_results.py` - utilities for summarizing experiment outputs.
- `experiment_config.py` - shared experiment configuration.
- `experiments/dataset_stats.py` - dataset distributional statistics used for benchmark diagnostics.
- `experiments/run_shift.py` and `experiments/run_shift_all.py` - distribution-shift experiments.
- `experiments/run_shift_rfimp_search.py` - search/evaluation script comparing SurrogateML against RF-Imp under source/target shifts.
- `test_*.py` - regression and protocol tests.

## Setup

Create an environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Some optional baselines require extra packages such as `xgboost`, `lightgbm`, `catboost`, and `xlrd`.

## Data

The loaders expect datasets under `Datasets/`. If data files are missing, run:

```bash
python download_datasets.py
```

Large raw datasets and generated artifacts should generally stay out of Git. 
The code is intended to regenerate experiment outputs from the downloaded datasets.

## Common Commands

Run the main benchmark experiments:

```bash
python run_experiment.py
```

Run baseline tests:

```bash
pytest test_baselines.py test_clean_train_protocol.py test_steering_v2.py
```

Generate dataset distribution statistics:

```bash
python experiments/dataset_stats.py
```

Run the matched corrupted-training control:

```bash
python run_matched_training_condition.py
```

Run the explicit RF-Imp baseline:

```bash
python run_rf_imputer_baseline.py
```

Run the degradation-focused distribution-shift comparison between SurrogateML and RF-Imp:

```bash
python experiments/run_shift_rfimp_search.py \
  --candidates california_dense_to_sparse,credit_ds_young_to_old \
  --epochs 30 \
  --n_mc 100 \
  --rf_estimators 100 \
  --rf_imp_estimators 50 \
  --rf_imp_max_iter 10 \
  --max_eval 1000 \
  --output_dir results_v3/shift_rfimp_degradation_full
```

## Notes

SurrogateML experiments can be computationally expensive because each prediction may average over many Monte Carlo imputations. 
For quick checks, reduce `--epochs`, `--n_mc`, or `--max_eval` where supported.
