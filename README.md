# Food inspection feature engineering

This branch prepares historical features and chronological dataset partitions
for the LA County food inspection project.

The scope is data preparation: retain the inherited cleaning pipeline, engineer
38 historical features, create future-inspection labels and chronological
assignments, and verify their integrity. Prediction models, fitted preprocessing,
and model evaluation belong in subsequent branches.

## Setup

Tested with Python 3.12:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-features.txt
```

## Run the workflow

```bash
# Clean and verify original inspection and violation CSVs.
.venv/bin/python src/prepare_data.py
.venv/bin/python src/verify_cleaning.py

# Construct strictly historical predictors and chronological partitions.
.venv/bin/python src/build_features.py
.venv/bin/python src/verify_features.py

# Run cleaning and feature-construction checks.
.venv/bin/python -m unittest discover -s tests -v
```

If cleaned inspection tables already exist, start with build_features.py.
Scripts regenerate their own outputs; keep the raw source CSVs unchanged.

## Project files

| Location | Purpose |
|---|---|
| `data/raw/` | Original inspection and violation CSVs |
| `data/processed/` | Normalized inspection/violation records, labels, and quality flags |
| `data/features/` | Training, development, test, feature schema, and split assignments |
| `src/prepare_data.py`, `src/verify_cleaning.py` | Cleaning and raw-source checks |
| `src/build_features.py`, `src/verify_features.py` | Historical features, chronological splits, and source checks |
| `configs/features.json` | Prediction dates and feature split boundaries |
| `reports/` | Cleaning audits and feature/split documentation |
| `reports/monthly_feature_statistics.csv` | Monthly feature-example counts and observed label frequencies |
| `tests/test_prepare_data.py` | Inherited cleaning and label-integrity tests |
| `tests/test_features.py` | Historical-feature, target-selection, and temporal-leakage tests |

## Feature construction rules

- Predict at month start using only earlier routine restaurant inspection records.
- Select the actual first routine target in a month before excluding uncertain
  labels. Require pre-cutoff history and do not invent negative outcomes for
  restaurants without an inspection.
- Keep facility/program identifiers separate. Preserve missing historical
  counts rather than replacing them with zero or skipping a more recent visit.
- Training: Apr 2024-Jun 2025. Development: Jul-Dec 2025, internally divided
  into tuning (Jul-Sep) and analysis (Oct-Dec). Test partition: Jan-Sep 2026.
- Preserve missing values in feature tables. Imputation, scaling, categorical
  encoding, and model fitting are deferred to the subsequent modeling branches.
- Feature tables also contain labels and audit metadata. The 38 approved raw
  feature names are listed in data/features/feature_schema.json; IDs and target
  outcomes must not become predictors in future models.
- This branch computes no model predictions or performance metrics. Earlier
  experiments examined test outcomes, so the dataset is not described as untouched.

See [cleaning details](README-cleaning.md) and
[feature rules and split counts](reports/feature_report.md).

Generated feature tables are ignored by Git and can be regenerated. Original
raw/cleaned data and cleaning audits remain available. This branch's code needs
NumPy and pandas, without a machine-learning-library dependency.
