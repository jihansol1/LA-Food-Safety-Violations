# Category-recurrence probability baseline

This implementation belongs on `heuristic-prediction`. It inherits the existing
feature tables and leaves the `feature-engineering` branch unchanged.

## Inputs and outputs

The model reads only these three pre-cutoff fields:

- `last_present_major`
- `last_present_minor`
- `last_present_grp`

Each can be 0, 1, or missing. For each category, the model estimates the
next-inspection positive rate separately for previous state 0 and previous state
1. Six known-state probabilities are saved. Unknown prior states use the
category's training prevalence, rather than being interpreted as absence.

The three outputs are marginal probabilities; their sum need not be one. The
same model also produces binary decisions using development-selected thresholds.
The JSON model contains all training counts, priors, settings, and provenance.

## Estimate, select, and evaluate

```bash
.venv/bin/python -m pip install -r requirements-heuristic.txt
.venv/bin/python src/train_heuristic.py
.venv/bin/python -m unittest discover -s tests -v
```

The training script:

1. Reads train.csv and dev.csv, checking program-month uniqueness and cutoffs.
2. Uses Apr 2024-Jun 2025 training targets to count state/outcome pairs.
3. Compares alpha=0,5,20,100 on Jul-Sep development mean Brier score.
4. Uses the selected lookup's Jul-Sep probabilities to choose three F1 cutoffs.
5. Reports Oct-Dec development metrics using those frozen settings.
6. Separately reports fixed-0.5 accuracy/F1 to distinguish threshold policies.

The estimate is `(k + alpha*pi)/(n + alpha)`, where pi is training prevalence.
A zero-count group always falls back to pi, including when alpha=0. Exact
smoothing ties select smaller alpha. Exact threshold-F1 ties select the higher
cutoff. Future labels and other feature columns do not affect inference.

Test data are not opened by this script. Earlier experiments inspected test
outcomes, and Oct-Dec development has already informed development; neither is
presented as a newly untouched evaluation window.

## Predict from a saved model

```bash
.venv/bin/python src/predict_heuristic.py \
  --model models/heuristic/lookup.json \
  --input data/features/dev.csv \
  --output reports/heuristic/example_predictions.csv
```

Use `--threshold-policy fixed_0.5` for default cutoff decisions. Probabilities
and priority scores stay identical under either policy. Inputs must already
follow the historical feature-cutoff rules; this command does not reconstruct
histories from raw records or use outcomes as inputs.

## Results and artifacts

| File | Purpose |
|---|---|
| `models/heuristic/lookup.json` | Learned six-value lookup, fallback priors, thresholds, and provenance |
| `reports/heuristic/heuristic_report.md` | Method, actual lookup, development results, and limitations |
| `lookup_probabilities.csv` | Counts, unsmoothed rates, and selected probabilities |
| `smoothing_trials.csv` | Mean/per-label tuning Brier for every alpha |
| `decision_thresholds.csv` | Development-F1 and fixed cutoffs |
| `metrics_by_window.csv` | Train/tuning/analysis/combined metrics for both decision policies |
| `category_metrics.csv` | Per-label AP, Brier, accuracy, precision, recall, and F1 |
| `monthly_rankings.csv` | Per-month top-5/10/20% results and tie counts |
| `development_predictions.csv` | Later development probabilities, decisions, outcomes, and priority |
| `development_review_sample.csv` | Seeded 30-example sample for manual review |
| `experiment_summary.json` | Config, versions, counts, hashes, and test-exclusion record |

CSV/summary filenames in the table are under `reports/heuristic/` unless a
different location is shown. Optional command-line paths allow a separate
report/model output directory.

## Interpretation

Priority is `4*p_major + 2*p_minor + p_grp`. It measures weighted category
presence, not expected violation counts or total deducted points. Ranking is
evaluated within each observed inspection month, using ceiling budgets and a
seeded program-ID hash to resolve ties. Only a few probability levels are
possible, so coarse queues and large tie groups are expected.

F1-selected binary predictions may be degenerate (all positive), while the
probabilities still contain ranking information. Fixed-0.5 accuracy can be high
while missing every major violation. Report AP, Brier, and budget metrics along
with the decision policy rather than treating accuracy alone as success.
