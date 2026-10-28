"""Independently spot-check saved cohort labels against original CSV rows."""

import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=root / "data" / "raw")
    parser.add_argument("--output-dir", type=Path, default=root / "data" / "processed")
    parser.add_argument("--report-dir", type=Path, default=root / "reports")
    args = parser.parse_args()
    raw_dir = args.raw_dir.resolve()
    if args.report_dir.resolve() == raw_dir or raw_dir in args.report_dir.resolve().parents:
        parser.error("Reports must not be written inside the raw directory")
    summary = json.loads((args.report_dir / "cleaning_summary.json").read_text())
    encodings = {s["filename"]: s["encoding"] for s in summary["sources"]}
    inspections = pd.read_csv(raw_dir / "inspections.csv", encoding=encodings["inspections.csv"],
                              dtype=str, keep_default_na=False)
    violations = pd.read_csv(raw_dir / "violations.csv", encoding=encodings["violations.csv"],
                             dtype=str, keep_default_na=False)
    for frame in [inspections, violations]:
        for column in frame:
            frame[column] = frame[column].str.strip()
    cohort = pd.read_csv(args.output_dir / "modeling_cohort.csv", dtype=str, keep_default_na=False)
    assert len(cohort) == summary["modeling_cohort"]["inspections"]
    assert cohort.serial_number.is_unique
    assert cohort.service_code.eq("001").all()
    assert cohort[["y_major", "y_minor", "y_grp"]].isin(["0", "1"]).all().all()
    # Coverage-directed samples across positive, negative, and unmatched-perfect-score cases.
    groups = [cohort[cohort.y_major.eq("1")], cohort[cohort.y_major.eq("0")],
              cohort[cohort.y_minor.eq("0")], cohort[cohort.y_grp.eq("0")],
              cohort[cohort.inferred_no_scored_violations.eq("True")]]
    sample = pd.concat([group.sample(min(6, len(group)), random_state=467) for group in groups])
    sample = sample.drop_duplicates("serial_number")
    checks = []
    for row in sample.itertuples(index=False):
        source = inspections.loc[inspections["SERIAL NUMBER"].eq(row.serial_number)]
        findings = violations.loc[violations["SERIAL NUMBER"].eq(row.serial_number)]
        points = pd.to_numeric(findings["POINTS"])
        expected = [int(points.isin([4, 11]).any()), int(points.eq(2).any()), int(points.eq(1).any())]
        actual = [int(row.y_major), int(row.y_minor), int(row.y_grp)]
        assert expected == actual, f"Source label mismatch: {row.serial_number}"
        assert float(source["SCORE"].iloc[0]) == 100 - float(points.sum())
        checks.append(dict(serial_number=row.serial_number, raw_violation_rows=len(findings),
                           raw_point_sum=points.sum(), score=row.score,
                           y_major=actual[0], y_minor=actual[1], y_grp=actual[2], source_check_passed=True))
    pd.DataFrame(checks).to_csv(args.report_dir / "source_spot_checks.csv", index=False)
    print(f"Saved-file checks passed; {len(checks)} independent raw-source spot checks passed.")


if __name__ == "__main__":
    main()
