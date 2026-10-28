# Food inspection data cleaning report

This report describes a completed inspection-level cleaning run. It does not contain model results.

## Population

| Stage | Inspections | Facilities | Programs |
|---|---:|---:|---:|
| All source inspections | 100,386 | 43,087 | 47,270 |
| Routine restaurant inspections | 74,339 | 32,999 | 35,210 |
| Conservative modeling cohort | 73,632 | 32,820 | 35,027 |

Source coverage: 2023-10-01 through 2026-09-30.
Routine restaurant inspections excluded from the conservative cohort for quality: 707.

## Label prevalence in the conservative cohort

| Label | Positive | Negative | Positive percentage |
|---|---:|---:|---:|
| y_major | 16,792 | 56,840 | 22.81% |
| y_minor | 37,609 | 36,023 | 51.08% |
| y_grp | 70,433 | 3,199 | 95.66% |

Labels overlap: an inspection can be positive for multiple categories.

## Cleaning decisions

1. Raw CSV files remain unchanged. Headers use snake_case; surrounding whitespace is removed. IDs remain text. Dates are parsed explicitly as month/day/year. Scores and points are numeric. Output dates use ISO format and CSVs use UTF-8.
2. SERIAL NUMBER is the unique inspection join key. FACILITY ID + RECORD ID is retained as a program key so distinct programs within a facility are not merged.
3. Repeated exact violation rows are retained and flagged. Counts describe recorded rows, not deduplicated hazards or unique unsafe conditions. Their repetition is not proof of export error.
4. Primary labels require OUT OF COMPLIANCE status, a supported retail-food code family, and consistent points: major=4/11, minor=2, GRP=1. Both F0xx and F1xx code versions are recognized. Code ranges are a documented project mapping and require review when new versions appear.
5. Recognized administrative zero-point records and multiple-major 3-point penalties are retained separately and never counted as extra target violations. Zero-point substantive observations, code/point conflicts, unsupported codes, and unexpected statuses are unresolved; no severity is guessed.
6. Inspections with unresolved violation records or a mismatch between 100-score and total recorded points have all three labels withheld (blank), and are excluded from the conservative modeling cohort. They remain in the master tables and review queues. Recorded scores are never rewritten to force agreement.
7. Inspections without linked violations and with score 100 receive three zero labels, with an explicit inference flag. This is a documented dataset assumption supported by the perfect score; it does not establish that every hazard was absent or observed.
8. Main targets are routine inspections whose program description begins RESTAURANT. Current ACTIVE/INACTIVE status, missing grade, and owner names do not determine inclusion. Employee identifiers, names, addresses, grades, scores, and target violation counts are metadata/outcomes, not automatically approved model features.

## Repeated-row evidence

There are 209 repeated rows after normalization. Removing them changes point sums for 141 linked inspections. Of these, 137 reconcile with the recorded score when repeats are kept, versus 3 when removed. See duplicate_score_reconciliation.csv.

## Join coverage and exceptions

- 21 violation rows from 7 serial numbers have no inspection match. They are retained in violations_clean.csv and orphan_violations.csv, but cannot become inspection targets.
- 2,935 inspections have no linked violation rows; 2,935 of them have score 100.
- Review queue entries are retained for transparent follow-up. Resolving one later should trigger a fresh run, not a manual edit to a processed CSV.

### Quality flags among routine restaurant inspections

Flag counts may overlap.

| Flag | Inspections |
|---|---:|
| invalid_date | 0 |
| invalid_score | 0 |
| missing_program_identifiers | 0 |
| has_unresolved_violations | 668 |
| score_point_mismatch | 42 |
| missing_violation_records_nonperfect_score | 0 |

## Verification

Verified source hashes before/after processing, row preservation, unique inspection keys, one-to-one aggregate join, binary complete cohort labels, correspondence between counts and labels, and score/point reconciliation for all cohort rows. Unit tests exercise the cleaning rules separately.

## Limits and next steps

The cohort contains 73,632 inspection outcomes, not that many independent restaurants or ready-to-train examples. 38,598 have a strictly earlier clean routine restaurant visit within the downloaded window. A monthly prediction cutoff will further change historical eligibility. Counts represent observed history within this file's window, not lifetime history.

1. Review unresolved cases and document the conservative exclusions. Code-version changes may cause temporal measurement differences; compare prevalence by month using monthly_cohort_statistics.csv.
2. Construct features strictly before the chosen prediction cutoff; never use the target inspection's score, grade, or violation counts as inputs.
3. Define chronological train/dev/test periods and produce split-specific label statistics. Do not impute, scale, or fit encoders on the entire dataset.
4. Train the previous-score baseline and three logistic classifiers before adding XGBoost.

## Reproduction

From the repository root:

```bash
python -m pip install -r requirements-cleaning.txt
python src/prepare_data.py
python -m unittest discover -s tests -p 'test_prepare_data.py' -v
```

Optional --raw-dir, --output-dir, and --report-dir arguments override default project locations. Run the script against original inputs; output tables are regenerated deterministically.

## Source guidance

- LA County Retail Food Inspection Guide: https://publichealth.lacounty.gov/eh/inspection/retail-food-inspection-guide.htm
- The two local source filenames, encodings, sizes, row counts, and SHA-256 hashes are recorded in cleaning_summary.json.
