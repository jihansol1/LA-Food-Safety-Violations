# Category-recurrence heuristic

This baseline estimates each category's next-inspection probability using only whether that category was present on the latest inspection day before the monthly cutoff. It consumes the existing feature tables; it does not change feature construction or fit feature coefficients.

## Method

For each category c and previous state s in {0,1}, calculate training counts n(c,s) and next-positive counts k(c,s). Let pi(c) be that category's prevalence among all training targets. The probability is (k + alpha*pi)/(n + alpha). An unobserved group or missing previous label uses pi(c). Unknown previous states are excluded from known-state groups, not treated as absent.

The three outputs are marginal probabilities and need not sum to one. No assumption of mutually exclusive labels is made. Six probability values cover the known-state cases; unknown states add the corresponding training-prior fallback.

## Data and selection

Training: 18,406 examples (Apr 2024-Jun 2025). Tuning: 3,744 (Jul-Sep 2025). Analysis: 3,409 (Oct-Dec 2025). The analysis window has been inspected in earlier development and is exploratory. The training script opens train.csv and dev.csv only. Earlier experiments evaluated test outcomes, but no test partition is read or evaluated here.

Compare alpha in [0, 5, 20, 100] using the mean of three tuning Brier scores (lower is better). Selected alpha: 0. Exact ties favor the smaller alpha. AP is not used to tune smoothing because this baseline supplies only a few distinct probability levels.

After choosing smoothing, select each binary threshold by tuning F1, with exact ties favoring the higher threshold. Both these decisions are frozen before reporting later analysis results. Fixed 0.5 decisions are also reported as a separate policy; do not mix the two accuracy definitions.

## Saved lookup

| Category | Previously present | Training n | Next positive | Raw rate | Selected probability |
|---|---:|---:|---:|---:|---:|
| major | 0 | 15,602 | 2,419 | 0.1550 | 0.1550 |
| major | 1 | 2,680 | 877 | 0.3272 | 0.3272 |
| minor | 0 | 9,789 | 4,195 | 0.4285 | 0.4285 |
| minor | 1 | 8,493 | 5,077 | 0.5978 | 0.5978 |
| grp | 0 | 665 | 532 | 0.8000 | 0.8000 |
| grp | 1 | 17,617 | 17,030 | 0.9667 | 0.9667 |

## Later development results

| Decision policy | Major AP | Minor AP | GRP AP | Macro AP | Mean Brier | Label accuracy | Exact match |
|---|---:|---:|---:|---:|---:|---:|---:|
| dev_f1 | 0.3537 | 0.5057 | 0.9517 | 0.6037 | 0.1747 | 57.32% | 15.93% |
| fixed_0.5 | 0.3537 | 0.5057 | 0.9517 | 0.6037 | 0.1747 | 74.12% | 39.37% |

AP means average precision (not trapezoidal PR-AUC). Brier measures probabilistic prediction quality; smaller is better. Label accuracy averages the three binary decisions; exact match requires all three to be correct. These measures answer different questions, and high GRP prevalence makes accuracy alone insufficient.

The tuning-F1 cutoffs produce every category present on every analysis example. This is an important weakness of the thresholded baseline, not a software error.

The fixed-0.5 policy predicts no major positives. Its overall accuracy must therefore be interpreted alongside zero major recall.

Probabilities and ranking metrics are identical under both decision policies.

The mean monthly major recall at a top-10% budget is 15.79%. Priority is 4*p_major + 2*p_minor + p_GRP, representing weighted category presence rather than counts or deducted points. Top-5/10/20% queues use ceiling rounding and a seeded hash of the known program ID for deterministic tie breaking. Monthly metrics are averaged; months with no major positives have undefined major recall/lift and are excluded from those averages. Many ties are intrinsic to this simple lookup.

## Reproduce

```bash
.venv/bin/python -m pip install -r requirements-heuristic.txt
.venv/bin/python src/train_heuristic.py
.venv/bin/python -m unittest discover -s tests -v
```

lookup_probabilities.csv contains all six groups. smoothing_trials.csv and decision_thresholds.csv record development choices. metrics_by_window.csv, category_metrics.csv, and monthly_rankings.csv contain complete train/development results. development_review_sample.csv is a seeded sample of 30 later development examples for manual review. models/heuristic/lookup.json is the portable model.
