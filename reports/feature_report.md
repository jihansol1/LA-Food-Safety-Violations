# Feature engineering and chronological splits

Features are constructed at a month-start cutoff using only routine restaurant inspection records strictly before that date. The target is the first actual routine visit in that month. Selection precedes quality filtering: an uncertain first target is excluded rather than replaced by a later visit. Programs need at least one earlier inspection day.

## History and missingness

The full inspection master supplies history, including uncertain visits. Scores are retained where numeric; uncertain category counts and labels remain missing. Frequencies use known days and the feature list includes history length and missing-label indicators. Same-day visits are collapsed without guessing their order: use the minimum score, maximum category count, and any-positive label, with all category values missing if any visit that day has uncertain labels. If the first target day has multiple visits, exclude that target month.

Only previously recorded restaurant risk/seat characteristics are used. Source publication timestamps are unavailable: inspection-date availability is an explicit assumption, so this is a historical simulation, not an operational deployment.

## Split sizes

| Split | Examples | Programs | Major positive | Minor positive | GRP positive |
|---|---:|---:|---:|---:|---:|
| train | 18,406 | 16,078 | 3,338 | 9,341 | 17,679 |
| dev | 7,153 | 6,923 | 2,182 | 3,520 | 6,797 |
| test | 12,633 | 11,875 | 4,444 | 6,553 | 11,936 |
| dev_tune | 3,744 | 3,712 | 1,091 | 1,981 | 3,565 |
| dev_analysis | 3,409 | 3,376 | 1,091 | 1,539 | 3,232 |

Training: Apr 2024–Jun 2025. Development: Jul–Dec 2025, internally split into tuning (Jul–Sep) and analysis (Oct–Dec). Test: Jan–Sep 2026. Earlier records initialize histories. Later monthly feature rows may incorporate earlier visits once they are observable. No model fitting or evaluation occurs here.

The feature builder exports the test partition without evaluating any prediction model. Earlier frozen models' test accuracies have already been observed in a separate requested evaluation; do not describe that test set as untouched. This branch only prepares feature tables and chronological assignments. Test label counts above are dataset statistics, not model results.

## Feature allowlist

- `n_prior_inspections`
- `n_prior_inspection_days`
- `n_prior_labeled_days`
- `n_prior_uncertain_days`
- `days_since_last_inspection`
- `days_since_first_observed_inspection`
- `last_score`
- `mean_score_last3_days`
- `min_score_last3_days`
- `score_change_last2_days`
- `last_day_labels_missing`
- `last_day_multiple_inspections`
- `n_labeled_last3_days`
- `prediction_month_sin`
- `prediction_month_cos`
- `last_count_major`
- `last_present_major`
- `historical_rate_major`
- `recent3_rate_major`
- `repeat_last2_major`
- `days_since_observed_major`
- `never_observed_major`
- `last_count_minor`
- `last_present_minor`
- `historical_rate_minor`
- `recent3_rate_minor`
- `repeat_last2_minor`
- `days_since_observed_minor`
- `never_observed_minor`
- `last_count_grp`
- `last_present_grp`
- `historical_rate_grp`
- `recent3_rate_grp`
- `repeat_last2_grp`
- `days_since_observed_grp`
- `never_observed_grp`
- `historical_risk_tier`
- `historical_seat_band`

## Limits

Examples represent programs actually inspected within each month. Lack of an inspection is never labeled negative. Counts represent the observed source window and inspection days, not lifetime records. IDs, target scores/grades, dates, and labels are excluded from model inputs. Temporal code-version differences remain a possible source of distribution shift.
