"""Audit and prepare LA County inspection data without changing raw inputs.

Run from the project root: python src/prepare_data.py
Requires pandas. No model fitting, feature imputation, or split selection occurs here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


INSPECTION_REQUIRED = {
    "activity_date", "facility_id", "record_id", "serial_number", "score",
    "pe_description", "service_code", "service_description",
}
VIOLATION_REQUIRED = {
    "serial_number", "violation_status", "violation_code", "violation_description", "points",
}
LABELS = ["y_major", "y_minor", "y_grp"]
OLD_CRF = set(range(2, 24)) | {52, 53, 55, 56, 57, 58}
OLD_GRP = set(range(24, 49))
ACTION_CODES = {"F049", "F050", "F051", "F150", "F151", "F152"}
PENALTY_CODES = {"F054", "F153"}
GUIDE_URL = "https://publichealth.lacounty.gov/eh/inspection/retail-food-inspection-guide.htm"


def read_source(path: Path, required: set[str]) -> tuple[pd.DataFrame, dict]:
    """Use strict decoding. Preserve identifiers, blanks, and original row order."""
    content = path.read_bytes()
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            content.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(f"Unable to decode {path.name} with supported encodings")
    frame = pd.read_csv(path, encoding=encoding, dtype="string", keep_default_na=False)
    source_duplicate_rows = int(frame.duplicated().sum())
    frame.columns = frame.columns.str.strip().str.lower().str.replace(r"\s+", "_", regex=True)
    if not frame.columns.is_unique:
        raise ValueError(f"Normalizing headers produced duplicate columns in {path.name}")
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path.name}: {sorted(missing)}")
    for column in frame:
        frame[column] = frame[column].str.strip().replace("", pd.NA)
    meta = {
        "filename": path.name, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest(),
        "encoding": encoding, "rows": len(frame), "columns": len(frame.columns),
        "exact_duplicate_rows_before_whitespace_normalization": source_duplicate_rows,
        "duplicate_rows_after_whitespace_normalization": int(frame.duplicated().sum()),
        "missing_by_column": {c: int(n) for c, n in frame.isna().sum().items()},
    }
    frame.insert(0, "source_record_number", range(1, len(frame) + 1))
    return frame, meta


def classify_violation(code: str, points: float, status: str) -> tuple[str, str]:
    """Conservative code-family + point mapping, with no guessed corrections.

    Only expected out-of-compliance records get target categories. Unusual codes,
    statuses, substantive zero-point observations, and code/point conflicts are
    unresolved, so affected inspection labels are withheld from the main cohort.
    """
    if pd.isna(code) or pd.isna(points) or pd.isna(status):
        return "unresolved", "missing_code_points_or_status"
    if status != "OUT OF COMPLIANCE":
        return "unresolved", "unexpected_status"
    if code in ACTION_CODES:
        return ("administrative", "") if points == 0 else ("unresolved", "action_has_nonzero_points")
    if code in PENALTY_CODES:
        return ("multiple_major_penalty", "") if points == 3 else ("unresolved", "unexpected_penalty_points")
    if not (len(code) == 4 and code[0] == "F" and code[1:].isdigit()):
        return "unresolved", "unsupported_code_family"
    number = int(code[1:])
    critical = number in OLD_CRF or 101 <= number <= 124
    grp = number in OLD_GRP or 125 <= number <= 149
    if critical:
        if points in (4, 11):
            return "major", ""
        if points == 2:
            return "minor", ""
        return "unresolved", "critical_code_unexpected_points"
    if grp:
        return ("grp", "") if points == 1 else ("unresolved", "grp_code_unexpected_points")
    return "unresolved", "unsupported_food_code"


def prepare_violations(raw: pd.DataFrame) -> pd.DataFrame:
    frame = raw.copy()
    fields = [c for c in frame if c != "source_record_number"]
    frame["is_repeated_exact_record"] = frame.duplicated(fields, keep="first")
    frame["belongs_to_repeated_exact_group"] = frame.duplicated(fields, keep=False)
    frame["points_original"] = frame["points"]
    frame["points"] = pd.to_numeric(frame["points"], errors="coerce")
    mapped = [classify_violation(c, p, s) for c, p, s in
              frame[["violation_code", "points", "violation_status"]].itertuples(index=False, name=None)]
    frame[["severity_category", "review_reason"]] = pd.DataFrame(
        mapped, index=frame.index, columns=["severity_category", "review_reason"]
    )
    frame["code_version"] = "other"
    frame.loc[frame.violation_code.str.fullmatch(r"F0\d\d", na=False), "code_version"] = "F0xx"
    frame.loc[frame.violation_code.str.fullmatch(r"F1\d\d", na=False), "code_version"] = "F1xx"
    for category in ("major", "minor", "grp"):
        frame[f"is_{category}"] = frame.severity_category.eq(category).astype("int64")
    frame["is_unresolved"] = frame.severity_category.eq("unresolved").astype("int64")
    frame["is_administrative"] = frame.severity_category.eq("administrative").astype("int64")
    frame["is_multiple_major_penalty"] = frame.severity_category.eq("multiple_major_penalty").astype("int64")
    return frame


def prepare_inspections(raw: pd.DataFrame, violations: pd.DataFrame) -> pd.DataFrame:
    frame = raw.copy()
    if frame.serial_number.isna().any() or frame.serial_number.duplicated().any():
        raise ValueError("Inspection SERIAL NUMBER must be populated and unique; manual resolution required")
    frame["activity_date_original"] = frame.activity_date
    frame["activity_date"] = pd.to_datetime(frame.activity_date, format="%m/%d/%Y", errors="coerce")
    frame["score_original"] = frame.score
    frame["score"] = pd.to_numeric(frame.score, errors="coerce")
    frame["program_key"] = frame.facility_id.str.cat(frame.record_id, sep="|")

    aggregation = violations.groupby("serial_number", dropna=True).agg(
        violation_row_count=("source_record_number", "size"),
        total_recorded_points=("points", lambda x: x.sum(min_count=1)),
        invalid_point_count=("points", lambda x: x.isna().sum()),
        repeated_record_count=("is_repeated_exact_record", "sum"),
        major_count=("is_major", "sum"), minor_count=("is_minor", "sum"), grp_count=("is_grp", "sum"),
        unresolved_violation_count=("is_unresolved", "sum"),
        administrative_count=("is_administrative", "sum"),
        multiple_major_penalty_count=("is_multiple_major_penalty", "sum"),
    ).reset_index()
    frame = frame.merge(aggregation, on="serial_number", how="left", validate="one_to_one", indicator=True)
    frame["has_violation_records"] = frame.pop("_merge").eq("both")
    count_fields = [c for c in aggregation if c.endswith("count")]
    frame[count_fields] = frame[count_fields].fillna(0).astype("int64")
    # Zero is inferred only for unmatched rows; malformed point rows remain unavailable.
    frame.loc[~frame.has_violation_records, "total_recorded_points"] = 0
    frame["expected_point_deduction"] = 100 - frame.score
    frame["score_point_difference"] = frame.expected_point_deduction - frame.total_recorded_points
    frame["score_point_mismatch"] = frame.score_point_difference.abs().gt(1e-8) | frame.total_recorded_points.isna()
    frame["missing_violation_records_nonperfect_score"] = ~frame.has_violation_records & frame.score.ne(100)
    frame["inferred_no_scored_violations"] = ~frame.has_violation_records & frame.score.eq(100)
    frame["is_routine"] = frame.service_description.eq("ROUTINE INSPECTION").fillna(False)
    frame["is_restaurant"] = frame.pe_description.str.startswith("RESTAURANT", na=False)
    frame["is_routine_restaurant"] = frame.is_routine & frame.is_restaurant
    frame["invalid_date"] = frame.activity_date.isna()
    frame["invalid_score"] = frame.score.isna() | ~frame.score.between(0, 100)
    frame["missing_program_identifiers"] = frame.facility_id.isna() | frame.record_id.isna()
    frame["has_unresolved_violations"] = frame.unresolved_violation_count.gt(0) | frame.invalid_point_count.gt(0)
    frame["label_quality_ok"] = ~(
        frame.has_unresolved_violations | frame.score_point_mismatch |
        frame.missing_violation_records_nonperfect_score | frame.invalid_score
    )
    for category in ("major", "minor", "grp"):
        frame[f"y_{category}"] = frame[f"{category}_count"].gt(0).astype("Int64")
        frame.loc[~frame.label_quality_ok, f"y_{category}"] = pd.NA
    frame["is_modeling_eligible"] = (
        frame.is_routine_restaurant & frame.label_quality_ok &
        ~frame.invalid_date & ~frame.missing_program_identifiers
    )
    frame["same_day_program_multiple_inspections"] = frame.duplicated(
        ["program_key", "activity_date"], keep=False
    ) & ~frame.invalid_date
    return frame.sort_values(["activity_date", "program_key", "serial_number"], kind="stable", na_position="last").reset_index(drop=True)


def population_summary(frame: pd.DataFrame) -> dict:
    return {
        "inspections": len(frame), "facilities": int(frame.facility_id.nunique()),
        "programs": int(frame.program_key.nunique()),
        "date_min": frame.activity_date.min().strftime("%Y-%m-%d") if frame.activity_date.notna().any() else None,
        "date_max": frame.activity_date.max().strftime("%Y-%m-%d") if frame.activity_date.notna().any() else None,
        "labels": {label: {"positive": int(frame[label].eq(1).sum()),
                           "negative": int(frame[label].eq(0).sum()),
                           "unknown": int(frame[label].isna().sum()),
                           "positive_fraction_of_known": float(frame[label].mean()) if frame[label].notna().any() else None}
                   for label in LABELS},
    }


def output_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8", date_format="%Y-%m-%d", float_format="%.8g")


def run(raw_dir: Path, output_dir: Path, report_dir: Path) -> dict:
    for directory in (output_dir, report_dir):
        directory.mkdir(parents=True, exist_ok=True)
    inspection_raw, inspection_meta = read_source(raw_dir / "inspections.csv", INSPECTION_REQUIRED)
    violation_raw, violation_meta = read_source(raw_dir / "violations.csv", VIOLATION_REQUIRED)
    violations = prepare_violations(violation_raw)
    violations["has_matching_inspection"] = violations.serial_number.isin(inspection_raw.serial_number)
    inspections = prepare_inspections(inspection_raw, violations)
    routine = inspections.loc[inspections.is_routine_restaurant].copy()
    cohort = inspections.loc[inspections.is_modeling_eligible].copy()
    flags = ["invalid_date", "invalid_score", "missing_program_identifiers", "has_unresolved_violations",
             "score_point_mismatch", "missing_violation_records_nonperfect_score"]

    # Integrity checks cover the actual output, not merely successful CSV export.
    assert len(inspections) == len(inspection_raw)
    assert len(violations) == len(violation_raw), "Repeated rows must not be silently removed"
    assert inspections.serial_number.is_unique
    assert len(cohort) > 0
    assert cohort[LABELS].notna().all().all()
    assert cohort[LABELS].isin([0, 1]).all().all()
    assert cohort.score_point_difference.abs().le(1e-8).all()
    for category in ("major", "minor", "grp"):
        assert cohort[f"y_{category}"].eq(cohort[f"{category}_count"].gt(0).astype(int)).all()

    output_csv(violations, output_dir / "violations_clean.csv")
    output_csv(inspections, output_dir / "inspections_clean.csv")
    output_csv(routine, output_dir / "routine_restaurant_inspections.csv")
    output_csv(cohort, output_dir / "modeling_cohort.csv")

    orphan = violations.loc[~violations.has_matching_inspection]
    output_csv(orphan, report_dir / "orphan_violations.csv")
    output_csv(violations.loc[violations.is_repeated_exact_record], report_dir / "repeated_violation_records.csv")
    output_csv(violations.loc[violations.is_unresolved.eq(1)], report_dir / "violation_review_queue.csv")
    output_csv(inspections.loc[inspections[flags].any(axis=1)], report_dir / "inspection_review_queue.csv")
    code_audit = violations.groupby(["code_version", "violation_code", "violation_description", "violation_status",
                                    "points", "severity_category", "review_reason"], dropna=False).size().reset_index(name="rows")
    output_csv(code_audit, report_dir / "violation_mapping_audit.csv")

    cohort_month = cohort.assign(month=cohort.activity_date.dt.strftime("%Y-%m"))
    monthly = cohort_month.groupby("month").agg(inspections=("serial_number", "size"),
        major_positive=("y_major", "sum"), minor_positive=("y_minor", "sum"), grp_positive=("y_grp", "sum")).reset_index()
    output_csv(monthly, report_dir / "monthly_cohort_statistics.csv")
    combinations = cohort.groupby(LABELS, dropna=False).size().reset_index(name="inspections")
    output_csv(combinations, report_dir / "label_combinations.csv")

    raw_sum = violations.groupby("serial_number").points.sum(min_count=1)
    fields = [c for c in violation_raw if c != "source_record_number"]
    unique_mask = ~violation_raw.duplicated(fields)
    unique_sum = violations.loc[unique_mask].groupby("serial_number").points.sum(min_count=1)
    reconciliation = inspections[["serial_number", "expected_point_deduction"]].copy()
    reconciliation["points_with_repeats"] = reconciliation.serial_number.map(raw_sum).fillna(0)
    reconciliation["points_without_repeats"] = reconciliation.serial_number.map(unique_sum).fillna(0)
    affected = reconciliation.loc[reconciliation.points_with_repeats.ne(reconciliation.points_without_repeats)]
    repeat_evidence = {
        "inspections_with_changed_point_sum_if_deduplicated": len(affected),
        "score_matches_with_repeats": int(affected.expected_point_deduction.eq(affected.points_with_repeats).sum()),
        "score_matches_without_repeats": int(affected.expected_point_deduction.eq(affected.points_without_repeats).sum()),
    }
    output_csv(affected, report_dir / "duplicate_score_reconciliation.csv")

    # Only a diagnostic of available history, not a feature or a next-inspection dataset.
    visits_per_date = cohort.groupby(["program_key", "activity_date"]).size().rename("visits").reset_index()
    visits_per_date["earlier_observed_visits"] = visits_per_date.groupby("program_key").visits.cumsum() - visits_per_date.visits
    with_history = int(visits_per_date.loc[visits_per_date.earlier_observed_visits.gt(0), "visits"].sum())

    summary = {
        "sources": [inspection_meta, violation_meta],
        "source_guidance": [GUIDE_URL],
        "all_inspections": population_summary(inspections),
        "routine_restaurant_inspections": population_summary(routine),
        "modeling_cohort": population_summary(cohort),
        "routine_restaurant_inspections_excluded_for_quality": len(routine) - len(cohort),
        "quality_flags_all_inspections": {c: int(inspections[c].sum()) for c in flags},
        "quality_flags_routine_restaurants": {c: int(routine[c].sum()) for c in flags},
        "violation_categories": {str(k): int(v) for k, v in violations.severity_category.value_counts().items()},
        "unresolved_violation_reasons": {str(k): int(v) for k, v in violations.loc[violations.is_unresolved.eq(1), "review_reason"].value_counts().items()},
        "orphan_violation_rows": len(orphan), "orphan_inspection_ids": int(orphan.serial_number.nunique()),
        "inspections_without_violation_rows": int((~inspections.has_violation_records).sum()),
        "inferred_no_scored_violations": int(inspections.inferred_no_scored_violations.sum()),
        "repeat_evidence": repeat_evidence,
        "cohort_inspections_with_strictly_earlier_clean_visit": with_history,
        "cohort_same_day_program_multiple_inspections": int(cohort.same_day_program_multiple_inspections.sum()),
        "checks": {"raw_inputs_unchanged": True, "inspection_key_unique": True,
                   "source_rows_preserved": True, "cohort_labels_complete_binary": True,
                   "cohort_points_reconcile": True, "cohort_labels_match_category_counts": True},
        "scope": "Cleaning and inspection-level labels only; no historical features, train/dev/test splits, or models.",
    }
    # Re-hash after processing to verify the source files were not modified.
    for meta in summary["sources"]:
        if hashlib.sha256((raw_dir / meta["filename"]).read_bytes()).hexdigest() != meta["sha256"]:
            raise RuntimeError("A raw input changed during execution")
    (report_dir / "cleaning_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_report(summary, report_dir / "cleaning_report.md")
    return summary


def write_report(summary: dict, path: Path) -> None:
    cohort = summary["modeling_cohort"]
    routine = summary["routine_restaurant_inspections"]
    repeats = summary["repeat_evidence"]
    lines = [
        "# Food inspection data cleaning report", "",
        "This report describes a completed inspection-level cleaning run. It does not contain model results.", "",
        "## Population", "", "| Stage | Inspections | Facilities | Programs |", "|---|---:|---:|---:|",
    ]
    for title, key in [("All source inspections", "all_inspections"),
                       ("Routine restaurant inspections", "routine_restaurant_inspections"),
                       ("Conservative modeling cohort", "modeling_cohort")]:
        value = summary[key]
        lines.append(f"| {title} | {value['inspections']:,} | {value['facilities']:,} | {value['programs']:,} |")
    lines += ["", f"Source coverage: {summary['all_inspections']['date_min']} through {summary['all_inspections']['date_max']}.",
              f"Routine restaurant inspections excluded from the conservative cohort for quality: {summary['routine_restaurant_inspections_excluded_for_quality']:,}.",
              "", "## Label prevalence in the conservative cohort", "", "| Label | Positive | Negative | Positive percentage |", "|---|---:|---:|---:|"]
    for label, stats in cohort["labels"].items():
        lines.append(f"| {label} | {stats['positive']:,} | {stats['negative']:,} | {100 * stats['positive_fraction_of_known']:.2f}% |")
    lines += ["", "Labels overlap: an inspection can be positive for multiple categories.", "",
        "## Cleaning decisions", "",
        "1. Raw CSV files remain unchanged. Headers use snake_case; surrounding whitespace is removed. IDs remain text. Dates are parsed explicitly as month/day/year. Scores and points are numeric. Output dates use ISO format and CSVs use UTF-8.",
        "2. SERIAL NUMBER is the unique inspection join key. FACILITY ID + RECORD ID is retained as a program key so distinct programs within a facility are not merged.",
        "3. Repeated exact violation rows are retained and flagged. Counts describe recorded rows, not deduplicated hazards or unique unsafe conditions. Their repetition is not proof of export error.",
        "4. Primary labels require OUT OF COMPLIANCE status, a supported retail-food code family, and consistent points: major=4/11, minor=2, GRP=1. Both F0xx and F1xx code versions are recognized. Code ranges are a documented project mapping and require review when new versions appear.",
        "5. Recognized administrative zero-point records and multiple-major 3-point penalties are retained separately and never counted as extra target violations. Zero-point substantive observations, code/point conflicts, unsupported codes, and unexpected statuses are unresolved; no severity is guessed.",
        "6. Inspections with unresolved violation records or a mismatch between 100-score and total recorded points have all three labels withheld (blank), and are excluded from the conservative modeling cohort. They remain in the master tables and review queues. Recorded scores are never rewritten to force agreement.",
        "7. Inspections without linked violations and with score 100 receive three zero labels, with an explicit inference flag. This is a documented dataset assumption supported by the perfect score; it does not establish that every hazard was absent or observed.",
        "8. Main targets are routine inspections whose program description begins RESTAURANT. Current ACTIVE/INACTIVE status, missing grade, and owner names do not determine inclusion. Employee identifiers, names, addresses, grades, scores, and target violation counts are metadata/outcomes, not automatically approved model features.",
        "", "## Repeated-row evidence", "",
        f"There are {summary['sources'][1]['duplicate_rows_after_whitespace_normalization']:,} repeated rows after normalization. Removing them changes point sums for {repeats['inspections_with_changed_point_sum_if_deduplicated']:,} linked inspections. Of these, {repeats['score_matches_with_repeats']:,} reconcile with the recorded score when repeats are kept, versus {repeats['score_matches_without_repeats']:,} when removed. See duplicate_score_reconciliation.csv.",
        "", "## Join coverage and exceptions", "",
        f"- {summary['orphan_violation_rows']:,} violation rows from {summary['orphan_inspection_ids']:,} serial numbers have no inspection match. They are retained in violations_clean.csv and orphan_violations.csv, but cannot become inspection targets.",
        f"- {summary['inspections_without_violation_rows']:,} inspections have no linked violation rows; {summary['inferred_no_scored_violations']:,} of them have score 100.",
        "- Review queue entries are retained for transparent follow-up. Resolving one later should trigger a fresh run, not a manual edit to a processed CSV.",
        "", "### Quality flags among routine restaurant inspections", "", "Flag counts may overlap.", "", "| Flag | Inspections |", "|---|---:|"]
    for flag, count in summary["quality_flags_routine_restaurants"].items():
        lines.append(f"| {flag} | {count:,} |")
    lines += ["", "## Verification", "",
        "Verified source hashes before/after processing, row preservation, unique inspection keys, one-to-one aggregate join, binary complete cohort labels, correspondence between counts and labels, and score/point reconciliation for all cohort rows. Unit tests exercise the cleaning rules separately.",
        "", "## Limits and next steps", "",
        f"The cohort contains {cohort['inspections']:,} inspection outcomes, not that many independent restaurants or ready-to-train examples. {summary['cohort_inspections_with_strictly_earlier_clean_visit']:,} have a strictly earlier clean routine restaurant visit within the downloaded window. A monthly prediction cutoff will further change historical eligibility. Counts represent observed history within this file's window, not lifetime history.",
        "", "1. Review unresolved cases and document the conservative exclusions. Code-version changes may cause temporal measurement differences; compare prevalence by month using monthly_cohort_statistics.csv.",
        "2. Construct features strictly before the chosen prediction cutoff; never use the target inspection's score, grade, or violation counts as inputs.",
        "3. Define chronological train/dev/test periods and produce split-specific label statistics. Do not impute, scale, or fit encoders on the entire dataset.",
        "4. Train the previous-score baseline and three logistic classifiers before adding XGBoost.",
        "", "## Reproduction", "", "From the repository root:", "", "```bash", "python -m pip install -r requirements-cleaning.txt", "python src/prepare_data.py", "python -m unittest discover -s tests -p 'test_prepare_data.py' -v", "```", "",
        "Optional --raw-dir, --output-dir, and --report-dir arguments override default project locations. Run the script against original inputs; output tables are regenerated deterministically.",
        "", "## Source guidance", "", f"- LA County Retail Food Inspection Guide: {GUIDE_URL}",
        "- The two local source filenames, encodings, sizes, row counts, and SHA-256 hashes are recorded in cleaning_summary.json.", "",
    ]
    path.write_text("\n".join(lines))


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=root / "data" / "raw")
    parser.add_argument("--output-dir", type=Path, default=root / "data" / "processed")
    parser.add_argument("--report-dir", type=Path, default=root / "reports")
    args = parser.parse_args()
    # Never allow outputs to overwrite the raw directory.
    raw_dir = args.raw_dir.resolve()
    if any(p.resolve() == raw_dir or raw_dir in p.resolve().parents for p in (args.output_dir, args.report_dir)):
        parser.error("Output and report directories must not be inside the raw input directory")
    summary = run(raw_dir, args.output_dir, args.report_dir)
    print(json.dumps({"all_inspections": summary["all_inspections"]["inspections"],
                      "routine_restaurant_inspections": summary["routine_restaurant_inspections"]["inspections"],
                      "modeling_cohort": summary["modeling_cohort"]["inspections"],
                      "excluded_routine_restaurants": summary["routine_restaurant_inspections_excluded_for_quality"],
                      "labels": summary["modeling_cohort"]["labels"]}, indent=2))


if __name__ == "__main__":
    main()
