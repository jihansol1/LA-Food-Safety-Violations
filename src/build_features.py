"""Build monthly pre-inspection features and fixed chronological partitions.

Reads the complete inspection master, not just clean targets. This prevents
uncertain intervening visits from being silently skipped. No fitting occurs here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

CATEGORIES = ["major", "minor", "grp"]
LABELS = [f"y_{c}" for c in CATEGORIES]
BASE_NUMERIC = [
    "n_prior_inspections", "n_prior_inspection_days", "n_prior_labeled_days",
    "n_prior_uncertain_days", "days_since_last_inspection", "days_since_first_observed_inspection",
    "last_score", "mean_score_last3_days", "min_score_last3_days", "score_change_last2_days",
    "last_day_labels_missing", "last_day_multiple_inspections", "n_labeled_last3_days",
    "prediction_month_sin", "prediction_month_cos",
]
CATEGORY_NUMERIC = [f"{prefix}_{category}" for category in CATEGORIES for prefix in
    ["last_count", "last_present", "historical_rate", "recent3_rate", "repeat_last2", "days_since_observed", "never_observed"]]
NUMERIC_FEATURES = BASE_NUMERIC + CATEGORY_NUMERIC
CATEGORICAL_FEATURES = ["historical_risk_tier", "historical_seat_band"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
BOOL_COLUMNS = ["is_routine_restaurant", "label_quality_ok", "is_modeling_eligible"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_inspections(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype="string")
    for col in BOOL_COLUMNS:
        if not frame[col].isin(["True", "False"]).all():
            raise ValueError(f"Unexpected boolean representation in {col}")
        frame[col] = frame[col].eq("True")
    frame["activity_date"] = pd.to_datetime(frame.activity_date, format="%Y-%m-%d", errors="coerce")
    for col in ["score", "major_count", "minor_count", "grp_count"] + LABELS:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    if frame.serial_number.isna().any() or frame.serial_number.duplicated().any():
        raise ValueError("Inspection keys must be populated and unique")
    return frame


def safe_mean(values):
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(finite.mean()) if len(finite) else np.nan


def risk_and_seats(description):
    if not isinstance(description, str):
        return "UNKNOWN", "UNKNOWN"
    tier = next((t for t in ["MODERATE", "HIGH", "LOW"] if f"{t} RISK" in description), "UNKNOWN")
    compact = description.replace(" ", "")
    seats = next((band for token, band in [("0-30", "0-30"), ("31-60", "31-60"),
                 ("61-150", "61-150"), ("151+", "151+")] if token in compact), "UNKNOWN")
    return tier, seats


def assign_split(date, config):
    if date < pd.Timestamp(config["train_start"]):
        return "history_only"
    if date < pd.Timestamp(config["dev_start"]):
        return "train"
    if date < pd.Timestamp(config["test_start"]):
        return "dev"
    if date < pd.Timestamp(config["test_end_exclusive"]):
        return "test"
    return "outside_window"


def make_examples(inspections: pd.DataFrame, config: dict):
    frame = inspections.loc[inspections.is_routine_restaurant & inspections.activity_date.notna() &
                            inspections.program_key.notna()].copy()
    frame = frame.sort_values(["program_key", "activity_date", "serial_number"], kind="stable")
    frame["prediction_date"] = frame.activity_date.dt.to_period("M").dt.to_timestamp()
    # Select the actual first inspection before applying target quality checks.
    targets = frame.drop_duplicates(["program_key", "prediction_date"], keep="first").copy()
    daily_counts = frame.groupby(["program_key", "activity_date"]).size()
    first_day_counts = pd.MultiIndex.from_frame(targets[["program_key", "activity_date"]]).map(daily_counts)
    targets["first_day_target_count"] = np.asarray(first_day_counts, dtype=int)

    aggregations = dict(score=("score", "min"), day_quality_ok=("label_quality_ok", "all"),
                        inspection_rows=("serial_number", "size"), description=("pe_description", "first"),
                        description_variants=("pe_description", "nunique"))
    for category in CATEGORIES:
        aggregations[f"{category}_count"] = (f"{category}_count", "max")
        aggregations[f"y_{category}"] = (f"y_{category}", "max")
    daily = frame.groupby(["program_key", "activity_date"], sort=True).agg(**aggregations).reset_index()
    daily.loc[daily.description_variants.gt(1), "description"] = "UNKNOWN"
    # If any same-day visit is uncertain, counts/labels for that day are unknown.
    for col in [f"{c}_count" for c in CATEGORIES] + LABELS:
        daily.loc[~daily.day_quality_ok, col] = np.nan
    daily_groups = {key: group.reset_index(drop=True) for key, group in daily.groupby("program_key", sort=False)}
    records, audit = [], []
    for key, group_targets in targets.groupby("program_key", sort=False):
        history = daily_groups[key]
        dates = history.activity_date.to_numpy(dtype="datetime64[ns]")
        arrays = {col: history[col].to_numpy() for col in ["score", "inspection_rows", "day_quality_ok"] +
                  [f"{c}_count" for c in CATEGORIES] + LABELS}
        for target in group_targets.itertuples(index=False):
            split = assign_split(target.activity_date, config)
            reason = "included"
            if split in ["history_only", "outside_window"]:
                reason = split
            elif target.first_day_target_count > 1:
                reason = "ambiguous_first_inspection_day"
            elif not target.is_modeling_eligible:
                reason = "uncertain_first_target"
            k = int(np.searchsorted(dates, np.datetime64(target.prediction_date), side="left"))
            if reason == "included" and k == 0:
                reason = "no_pre_cutoff_history"
            audit.append(dict(serial_number=target.serial_number, program_key=key,
                              prediction_date=target.prediction_date, target_date=target.activity_date,
                              split=split, decision=reason, prior_inspection_days=k))
            if reason != "included":
                continue
            cutoff = target.prediction_date
            last_date = pd.Timestamp(dates[k - 1])
            latest = history.iloc[k - 1]
            scores = np.asarray(arrays["score"][:k], dtype=float)
            recent_scores = scores[-3:]
            present_scores = recent_scores[np.isfinite(recent_scores)]
            quality = np.asarray(arrays["day_quality_ok"][:k], dtype=bool)
            row = dict(serial_number=target.serial_number, facility_id=target.facility_id,
                       record_id=target.record_id, program_key=key, prediction_date=cutoff,
                       target_date=target.activity_date, target_score_for_audit=target.score,
                       history_max_date=last_date, split=split,
                       dev_role=("tune" if target.activity_date < pd.Timestamp(config["dev_analysis_start"])
                                 else "analysis") if split == "dev" else "",
                       n_prior_inspections=int(np.sum(arrays["inspection_rows"][:k])),
                       n_prior_inspection_days=k, n_prior_labeled_days=int(quality.sum()),
                       n_prior_uncertain_days=int((~quality).sum()),
                       days_since_last_inspection=(cutoff - last_date).days,
                       days_since_first_observed_inspection=(cutoff - pd.Timestamp(dates[0])).days,
                       last_score=float(scores[-1]), mean_score_last3_days=safe_mean(recent_scores),
                       min_score_last3_days=float(present_scores.min()) if len(present_scores) else np.nan,
                       score_change_last2_days=float(scores[-1] - scores[-2]) if k >= 2 else np.nan,
                       last_day_labels_missing=int(not quality[-1]),
                       last_day_multiple_inspections=int(latest.inspection_rows > 1),
                       n_labeled_last3_days=int(quality[-3:].sum()),
                       prediction_month_sin=float(np.sin(2 * np.pi * cutoff.month / 12)),
                       prediction_month_cos=float(np.cos(2 * np.pi * cutoff.month / 12)))
            row["historical_risk_tier"], row["historical_seat_band"] = risk_and_seats(latest.description)
            for category in CATEGORIES:
                labels = np.asarray(arrays[f"y_{category}"][:k], dtype=float)
                positive = np.flatnonzero(labels == 1)
                row[f"last_count_{category}"] = float(arrays[f"{category}_count"][k - 1])
                row[f"last_present_{category}"] = float(labels[-1])
                row[f"historical_rate_{category}"] = safe_mean(labels)
                row[f"recent3_rate_{category}"] = safe_mean(labels[-3:])
                row[f"repeat_last2_{category}"] = (float(np.all(labels[-2:] == 1)) if k >= 2 and
                                                       np.isfinite(labels[-2:]).all() else np.nan)
                row[f"days_since_observed_{category}"] = (cutoff - pd.Timestamp(dates[positive[-1]])).days if len(positive) else np.nan
                row[f"never_observed_{category}"] = int(not len(positive))
                row[f"y_{category}"] = int(getattr(target, f"y_{category}"))
            records.append(row)
    examples = pd.DataFrame(records)
    if not examples.empty:
        examples = examples.sort_values(["prediction_date", "program_key"], kind="stable").reset_index(drop=True)
        verify_examples(examples, config)
    return examples, pd.DataFrame(audit)


def verify_examples(examples, config):
    assert examples.serial_number.is_unique
    assert not examples.duplicated(["program_key", "prediction_date"]).any()
    assert examples.history_max_date.lt(examples.prediction_date).all()
    assert examples.prediction_date.le(examples.target_date).all()
    assert (examples.prediction_date.dt.to_period("M") == examples.target_date.dt.to_period("M")).all()
    assert examples[LABELS].isin([0, 1]).all().all()
    assert examples.n_prior_inspection_days.ge(1).all()
    assert set(FEATURES).isdisjoint(LABELS + ["target_date", "target_score_for_audit", "serial_number", "program_key"])
    assert examples.target_date.map(lambda date: assign_split(date, config)).eq(examples.split).all()


def run(input_path: Path, config_path: Path, output_dir: Path, report_dir: Path):
    config = json.loads(config_path.read_text())
    boundaries = [pd.Timestamp(config[k]) for k in ["train_start", "dev_start", "dev_analysis_start", "test_start", "test_end_exclusive"]]
    if boundaries != sorted(set(boundaries)):
        raise ValueError("Chronological boundaries must be strictly increasing")
    input_hash = sha256(input_path)
    examples, audit = make_examples(read_inspections(input_path), config)
    if examples.empty:
        raise ValueError("No eligible prediction examples")
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    for split in ["train", "dev", "test"]:
        subset = examples.loc[examples.split.eq(split)]
        if subset.empty or any(subset[label].nunique() != 2 for label in LABELS):
            raise ValueError(f"Split {split} lacks enough label variation")
        subset.to_csv(output_dir / f"{split}.csv", index=False, date_format="%Y-%m-%d")
    examples[["serial_number", "program_key", "prediction_date", "target_date", "split", "dev_role"]].to_csv(
        output_dir / "split_assignments.csv", index=False, date_format="%Y-%m-%d")
    schema = dict(numeric_features=NUMERIC_FEATURES, categorical_features=CATEGORICAL_FEATURES,
                  label_columns=LABELS, excluded_metadata=[c for c in examples if c not in FEATURES + LABELS],
                  history_unit="inspection day; min score, max category count; all labels unknown if any same-day row uncertain")
    (output_dir / "feature_schema.json").write_text(json.dumps(schema, indent=2) + "\n")
    audit.to_csv(report_dir / "feature_eligibility_audit.csv", index=False, date_format="%Y-%m-%d")
    summaries = []
    populations = [(s, examples.loc[examples.split.eq(s)]) for s in ["train", "dev", "test"]]
    populations += [("dev_tune", examples.loc[examples.dev_role.eq("tune")]),
                    ("dev_analysis", examples.loc[examples.dev_role.eq("analysis")])]
    for name, subset in populations:
        row = dict(split=name, examples=len(subset), facilities=subset.facility_id.nunique(),
                   programs=subset.program_key.nunique(), start=subset.target_date.min().strftime("%Y-%m-%d"),
                   end=subset.target_date.max().strftime("%Y-%m-%d"))
        for category in CATEGORIES:
            row[f"{category}_positive"] = int(subset[f"y_{category}"].sum())
            row[f"{category}_negative"] = int(len(subset) - subset[f"y_{category}"].sum())
            row[f"{category}_prevalence"] = float(subset[f"y_{category}"].mean())
        summaries.append(row)
    stats = pd.DataFrame(summaries)
    stats.to_csv(report_dir / "split_statistics.csv", index=False)
    feature_missing = pd.DataFrame({s: examples.loc[examples.split.eq(s), FEATURES].isna().mean()
                                    for s in ["train", "dev", "test"]})
    feature_missing.index.name = "feature"
    feature_missing.to_csv(report_dir / "feature_missingness.csv")
    monthly = examples.groupby(examples.prediction_date.dt.strftime("%Y-%m")).agg(
        examples=("serial_number", "size"), major_positive=("y_major", "sum"),
        minor_positive=("y_minor", "sum"), grp_positive=("y_grp", "sum"))
    monthly.index.name = "prediction_month"
    monthly.to_csv(report_dir / "monthly_feature_statistics.csv")
    assert sha256(input_path) == input_hash
    summary = dict(config=config, input_sha256=input_hash, feature_count=len(FEATURES),
                   eligibility_decisions={str(k): int(v) for k, v in audit.decision.value_counts().items()},
                   splits=summaries, checks=dict(strict_history_cutoffs=True, unique_program_month=True,
                   first_actual_target_selected_before_quality=True, chronological_splits=True,
                   explicit_feature_allowlist=True, input_unchanged=True),
                   selection_population="Known programs actually receiving a routine inspection during the target month")
    (report_dir / "feature_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = ["# Feature engineering and chronological splits", "",
        "Features are constructed at a month-start cutoff using only routine restaurant inspection records strictly before that date. "
        "The target is the first actual routine visit in that month. Selection precedes quality filtering: an uncertain "
        "first target is excluded rather than replaced by a later visit. Programs need at least one earlier inspection day.", "",
        "## History and missingness", "",
        "The full inspection master supplies history, including uncertain visits. Scores are retained where numeric; "
        "uncertain category counts and labels remain missing. Frequencies use known days and the feature list includes "
        "history length and missing-label indicators. Same-day visits are collapsed without guessing their order: "
        "use the minimum score, maximum category count, and any-positive label, with all category values missing if any "
        "visit that day has uncertain labels. If the first target day has multiple visits, exclude that target month.", "",
        "Only previously recorded restaurant risk/seat characteristics are used. Source publication timestamps are unavailable: "
        "inspection-date availability is an explicit assumption, so this is a historical simulation, not an operational deployment.", "",
        "## Split sizes", "", "| Split | Examples | Programs | Major positive | Minor positive | GRP positive |",
        "|---|---:|---:|---:|---:|---:|"]
    for row in summaries:
        lines.append(f"| {row['split']} | {row['examples']:,} | {row['programs']:,} | {row['major_positive']:,} | {row['minor_positive']:,} | {row['grp_positive']:,} |")
    lines += ["", "Training: Apr 2024–Jun 2025. Development: Jul–Dec 2025, internally split into tuning (Jul–Sep) "
              "and analysis (Oct–Dec). Test: Jan–Sep 2026. Earlier records initialize histories. Later monthly "
              "feature rows may incorporate earlier visits once they are observable. No model fitting or evaluation occurs here.", "",
              "The feature builder exports the test partition without evaluating any prediction model. "
              "Earlier frozen models' test accuracies have already been observed in a separate requested evaluation; "
              "do not describe that test set as untouched. This branch only prepares feature tables and "
              "chronological assignments. Test label counts above are dataset statistics, not model results.", "",
              "## Feature allowlist", ""] + [f"- `{feature}`" for feature in FEATURES]
    lines += ["", "## Limits", "", "Examples represent programs actually inspected within each month. "
              "Lack of an inspection is never labeled negative. Counts represent the observed source "
              "window and inspection days, not lifetime records. IDs, target scores/grades, dates, and labels are excluded "
              "from model inputs. Temporal code-version differences remain a possible source of distribution shift.", ""]
    (report_dir / "feature_report.md").write_text("\n".join(lines))
    print(stats.to_string(index=False))
    return summary


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=root / "data/processed/inspections_clean.csv")
    parser.add_argument("--config", type=Path, default=root / "configs/features.json")
    parser.add_argument("--output-dir", type=Path, default=root / "data/features")
    parser.add_argument("--report-dir", type=Path, default=root / "reports")
    args = parser.parse_args()
    run(args.input, args.config, args.output_dir, args.report_dir)


if __name__ == "__main__":
    main()
