"""Independently check saved feature rows against original inspection histories.

This checks data construction, including test cutoffs, but never predicts test
outcomes or computes test model performance.
"""
import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=root / "data/processed/inspections_clean.csv")
    parser.add_argument("--features-dir", type=Path, default=root / "data/features")
    parser.add_argument("--report-dir", type=Path, default=root / "reports")
    args = parser.parse_args()
    source = pd.read_csv(args.input, dtype={"program_key": str, "serial_number": str}, low_memory=False)
    source["activity_date"] = pd.to_datetime(source.activity_date)
    source = source.loc[source.is_routine_restaurant & source.activity_date.notna()]
    groups = {key: group for key, group in source.groupby("program_key")}
    checks = []
    for split in ["train", "dev", "test"]:
        examples = pd.read_csv(args.features_dir / f"{split}.csv", dtype={"program_key": str, "serial_number": str})
        assert examples.serial_number.is_unique
        assert not examples.duplicated(["program_key", "prediction_date"]).any()
        assert (pd.to_datetime(examples.history_max_date) < pd.to_datetime(examples.prediction_date)).all()
        assert examples[["y_major", "y_minor", "y_grp"]].isin([0, 1]).all().all()
        for row in examples.sample(min(30, len(examples)), random_state=467).itertuples(index=False):
            cutoff = pd.Timestamp(row.prediction_date)
            program = groups[row.program_key]
            history = program.loc[program.activity_date < cutoff]
            latest_date = history.activity_date.max()
            latest = history.loc[history.activity_date.eq(latest_date)]
            assert len(history) == row.n_prior_inspections
            assert history.activity_date.nunique() == row.n_prior_inspection_days
            assert latest_date == pd.Timestamp(row.history_max_date)
            assert latest.score.min() == row.last_score
            month_targets = program.loc[(program.activity_date >= cutoff) &
                                        (program.activity_date < cutoff + pd.offsets.MonthBegin(1))]
            assert month_targets.activity_date.min() == pd.Timestamp(row.target_date)
            for category in ["major", "minor", "grp"]:
                value = getattr(row, f"last_count_{category}")
                if latest.label_quality_ok.all():
                    assert value == latest[f"{category}_count"].max()
                else:
                    assert pd.isna(value)
            checks.append(dict(split=split, serial_number=row.serial_number, history_cutoff_ok=True,
                               first_actual_target_ok=True, last_score_and_counts_ok=True))
    args.report_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(checks).to_csv(args.report_dir / "feature_source_spot_checks.csv", index=False)
    print(f"All saved partitions verified; {len(checks)} independent history-source spot checks passed.")


if __name__ == "__main__":
    main()
