# Food inspection recurrence heuristic

This branch implements a simple category-recurrence probability baseline for
the LA County food inspection project. It builds on the completed feature
engineering work without changing the historical features or dataset partitions.

For major CRF, minor CRF, and GRP separately, estimate the next-inspection rate
when the category was previously absent or present. The result is six training
lookup values, with training-prevalence fallback for unknown states. Smoothing
and binary cutoffs are selected on development data. No regression or neural
network is trained here.

## Setup

Tested with Python 3.12:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-heuristic.txt
```

## Run the workflow

```bash
# Clean and verify original inspection and violation CSVs.
.venv/bin/python src/prepare_data.py
.venv/bin/python src/verify_cleaning.py

# Construct strictly historical predictors and chronological partitions.
.venv/bin/python src/build_features.py
.venv/bin/python src/verify_features.py

# Estimate the lookup, select settings on development data, and evaluate.
.venv/bin/python src/train_heuristic.py

# Run data, feature, and heuristic checks.
.venv/bin/python -m unittest discover -s tests -v
```

If feature tables already exist, start with train_heuristic.py. Training opens
train.csv and dev.csv only. Scripts regenerate their own outputs; keep the raw
source CSVs unchanged.

## Project files

| Location | Purpose |
|---|---|
| `data/raw/` | Original inspection and violation CSVs |
| `data/processed/` | Normalized inspection/violation records, labels, and quality flags |
| `data/features/` | Training, development, test, feature schema, and split assignments |
| `src/prepare_data.py`, `src/verify_cleaning.py` | Cleaning and raw-source checks |
| `src/build_features.py`, `src/verify_features.py` | Historical features, chronological splits, and source checks |
| `configs/features.json` | Prediction dates and feature split boundaries |
| `configs/heuristic.json` | Smoothing candidates, threshold policy, budgets, and severity weights |
| `src/heuristic.py` | Three-input recurrence counts, smoothed probabilities, and JSON persistence |
| `src/train_heuristic.py` | Training-only estimation, development selection, and reports |
| `src/predict_heuristic.py` | Saved-model inference on precomputed previous-category flags |
| `src/heuristic_metrics.py` | Probability, classification, and monthly ranking metrics |
| `reports/` | Cleaning audits and feature/split documentation |
| `reports/heuristic/` | Six lookup values, smoothing trials, thresholds, metrics, predictions, and report |
| `models/heuristic/lookup.json` | Portable, inspectable model and source hashes |
| `reports/monthly_feature_statistics.csv` | Monthly feature-example counts and observed label frequencies |
| `tests/test_prepare_data.py` | Inherited cleaning and label-integrity tests |
| `tests/test_features.py` | Historical-feature, target-selection, and temporal-leakage tests |
| `tests/test_heuristic.py` | Counts, smoothing, fallbacks, serialization, thresholds, queues, and test-set exclusion |

## Feature construction rules

- Predict at month start using only earlier routine restaurant inspection records.
- Select the actual first routine target in a month before excluding uncertain
  labels. Require pre-cutoff history and do not invent negative outcomes for
  restaurants without an inspection.
- Keep facility/program identifiers separate. Preserve missing historical
  counts rather than replacing them with zero or skipping a more recent visit.
- Training: Apr 2024-Jun 2025. Development: Jul-Dec 2025, internally divided
  into tuning (Jul-Sep) and analysis (Oct-Dec). Test partition: Jan-Sep 2026.
- Preserve missing values in feature tables. The heuristic uses training
  prevalence for an unknown prior-category flag; it does not replace missing
  states with zero. No scaling or categorical encoding is needed.
- Feature tables also contain labels and audit metadata. The 38 approved raw
  feature names are listed in data/features/feature_schema.json; IDs and target
  outcomes must not become predictors in future models.
- The heuristic uses only last_present_major, last_present_minor, and
  last_present_grp. Labels are used for estimation/evaluation, never as
  prediction inputs. Earlier experiments examined test outcomes, so the dataset
  is not described as untouched; this run does not read the test partition.

See [cleaning details](README-cleaning.md) and
[feature rules and split counts](reports/feature_report.md). For the baseline,
read the [heuristic guide](README-heuristic.md) and
[results report](reports/heuristic/heuristic_report.md).

Generated feature tables are ignored by Git and can be regenerated. Original
raw/cleaned data and cleaning audits remain available. The core heuristic and
inference use NumPy/pandas; scikit-learn is used only for evaluation metrics.
