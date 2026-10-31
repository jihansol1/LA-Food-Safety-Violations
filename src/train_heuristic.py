"""Estimate recurrence probabilities on training data and select settings on dev.

The final-test partition is never opened. Earlier experiments' test exposure is
historical context, not an input to this heuristic or its selection procedure.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from heuristic import (CATEGORIES, INPUTS, LABELS, fit_lookup, lookup_table,
                       predict_probabilities, save_model, validate_inputs, validate_targets)
from heuristic_metrics import classification_metrics, monthly_ranking, select_thresholds


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_partition(path, expected_split):
    frame = pd.read_csv(path, dtype={"serial_number": str, "program_key": str, "facility_id": str,
                                   "record_id": str}, low_memory=False)
    validate_inputs(frame)
    validate_targets(frame)
    required = {"program_key", "prediction_date", "target_date", "history_max_date", "split"}
    if required - set(frame):
        raise ValueError(f"Missing partition audit metadata: {sorted(required - set(frame))}")
    if not frame.split.eq(expected_split).all():
        raise ValueError(f"Expected only {expected_split} examples")
    if frame.program_key.isna().any() or frame.duplicated(["program_key", "prediction_date"]).any():
        raise ValueError("Expected unique populated program-month keys")
    history, cutoff, target = [pd.to_datetime(frame[column], errors="raise") for column in
                               ["history_max_date", "prediction_date", "target_date"]]
    if not history.lt(cutoff).all() or not cutoff.le(target).all():
        raise ValueError("Invalid feature cutoff or target chronology")
    if not cutoff.dt.to_period("M").eq(target.dt.to_period("M")).all():
        raise ValueError("Targets must fall in their cutoff month")
    return frame


def validate_config(config):
    if config["selection_metric"] != "mean_brier_score" or config["threshold_metric"] != "per_label_f1":
        raise ValueError("Supported selection is mean Brier, followed by per-label F1 thresholds")
    grid = np.asarray(config["smoothing_grid"], dtype=float)
    if grid.ndim != 1 or len(grid) == 0 or not np.isfinite(grid).all() or np.any(grid < 0):
        raise ValueError("Smoothing candidates must be finite and nonnegative")
    weights = np.asarray(config["priority_weights"], dtype=float)
    if weights.shape != (3,) or not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("Expected three nonnegative severity weights")
    if not config["ranking_budgets"] or any(not 0 < budget <= 1 for budget in config["ranking_budgets"]):
        raise ValueError("Ranking budgets must be in (0,1]")


def choose_smoothing(training, tuning, grid):
    outcomes = validate_targets(tuning)
    trials, best_model, best_score = [], None, np.inf
    for alpha in sorted(set(float(value) for value in grid)):
        model = fit_lookup(training, alpha)
        probabilities = predict_probabilities(model, tuning)
        category_brier = np.mean((probabilities - outcomes) ** 2, axis=0)
        score = float(category_brier.mean())
        trials.append(dict(alpha=alpha, mean_brier=score,
                           **{f"{category}_brier": float(category_brier[index])
                              for index, category in enumerate(CATEGORIES)}))
        if score < best_score - 1e-12:
            best_model, best_score = model, score
        print(f"alpha={alpha:g}: tuning mean Brier={score:.6f}", flush=True)
    return best_model, pd.DataFrame(trials)


def output_predictions(frame, probabilities, model, weights):
    metadata = [column for column in ["serial_number", "program_key", "prediction_date", "target_date"] if column in frame]
    result = frame[metadata + INPUTS + LABELS].copy()
    tuned = (probabilities >= np.asarray(model["decision_thresholds"])).astype(int)
    fixed = (probabilities >= 0.5).astype(int)
    for index, category in enumerate(CATEGORIES):
        result[f"p_{category}"] = probabilities[:, index]
        result[f"pred_{category}"] = tuned[:, index]
        result[f"pred_{category}_at_0_5"] = fixed[:, index]
    result["priority_score"] = probabilities @ np.asarray(weights)
    return result


def run(features_dir, config_path, report_dir, model_path):
    config = json.loads(Path(config_path).read_text())
    validate_config(config)
    training = read_partition(features_dir / "train.csv", "train")
    development = read_partition(features_dir / "dev.csv", "dev")
    if "dev_role" not in development or not development.dev_role.isin(["tune", "analysis"]).all():
        raise ValueError("Development examples must have tune/analysis roles")
    tuning = development.loc[development.dev_role.eq("tune")].copy()
    analysis = development.loc[development.dev_role.eq("analysis")].copy()
    if tuning.empty or analysis.empty:
        raise ValueError("Both tuning and analysis windows must contain examples")
    if not pd.to_datetime(training.target_date).max() < pd.to_datetime(tuning.prediction_date).min():
        raise ValueError("All training targets must precede the first tuning cutoff")
    if not pd.to_datetime(tuning.target_date).max() < pd.to_datetime(analysis.prediction_date).min():
        raise ValueError("Tuning targets must precede the first analysis cutoff")
    report_dir.mkdir(parents=True, exist_ok=True)
    model, trials = choose_smoothing(training, tuning, config["smoothing_grid"])
    tuning_probabilities = predict_probabilities(model, tuning)
    model["decision_thresholds"] = select_thresholds(validate_targets(tuning), tuning_probabilities)
    train_hash, dev_hash = file_hash(features_dir / "train.csv"), file_hash(features_dir / "dev.csv")
    model["metadata"] = dict(train_sha256=train_hash, dev_sha256=dev_hash,
                             priority_weights=config["priority_weights"], ranking_seed=config["seed"],
                             smoothing_selection="Minimum tuning mean Brier; ties choose smaller alpha",
                             threshold_selection="Per-label tuning F1; exact ties choose larger threshold")
    save_model(model, model_path)
    lookup_table(model).to_csv(report_dir / "lookup_probabilities.csv", index=False)
    trials.to_csv(report_dir / "smoothing_trials.csv", index=False)
    threshold_rows = []
    for index, category in enumerate(CATEGORIES):
        threshold_rows.append(dict(category=category, dev_f1_threshold=model["decision_thresholds"][index], fixed_threshold=0.5))
    pd.DataFrame(threshold_rows).to_csv(report_dir / "decision_thresholds.csv", index=False)
    summaries, categories, monthly = [], [], []
    windows = {"train": training, "dev_tune": tuning, "dev_analysis": analysis, "dev_combined": development}
    for window, frame in windows.items():
        probabilities = predict_probabilities(model, frame)
        for policy, thresholds in [("dev_f1", model["decision_thresholds"]), ("fixed_0.5", [0.5, 0.5, 0.5])]:
            summary, detail = classification_metrics(frame, probabilities, thresholds, policy)
            summary["window"] = window
            detail.insert(0, "window", window)
            for budget in config["ranking_budgets"]:
                ranking, rank_detail = monthly_ranking(frame, probabilities, budget, config["priority_weights"], config["seed"])
                for metric in ["precision", "recall", "lift", "ndcg"]:
                    summary[f"{metric}_at_{round(100 * budget)}pct"] = ranking[metric]
                if policy == "dev_f1":
                    rank_detail.insert(0, "window", window)
                    monthly.append(rank_detail)
            summaries.append(summary)
            categories.append(detail)
        if window == "dev_analysis":
            prediction_frame = output_predictions(frame, probabilities, model, config["priority_weights"])
            prediction_frame.to_csv(report_dir / "development_predictions.csv", index=False)
            sample = prediction_frame.sample(min(30, len(prediction_frame)), random_state=config["seed"])
            sample.to_csv(report_dir / "development_review_sample.csv", index=False)
    metrics = pd.DataFrame(summaries)
    metrics.to_csv(report_dir / "metrics_by_window.csv", index=False)
    pd.concat(categories, ignore_index=True).to_csv(report_dir / "category_metrics.csv", index=False)
    pd.concat(monthly, ignore_index=True).to_csv(report_dir / "monthly_rankings.csv", index=False)
    analysis_results = metrics.loc[metrics.window.eq("dev_analysis")].astype(object)
    analysis_results = analysis_results.where(pd.notna(analysis_results), None)
    behavior = dict(
        tuned_positive_counts={category: int(prediction_frame[f"pred_{category}"].sum()) for category in CATEGORIES},
        fixed_positive_counts={category: int(prediction_frame[f"pred_{category}_at_0_5"].sum()) for category in CATEGORIES},
        all_tuned_labels_positive=bool(prediction_frame[[f"pred_{category}" for category in CATEGORIES]].eq(1).all().all()),
        no_fixed_major_positives=bool(prediction_frame.pred_major_at_0_5.eq(0).all()))
    model_summary = dict(config=config, method="category_recurrence", selected_alpha=model["smoothing_alpha"],
        decision_thresholds=model["decision_thresholds"], training_examples=len(training),
        tuning_examples=len(tuning), analysis_examples=len(analysis), inputs=INPUTS, labels=LABELS,
        train_sha256=train_hash, dev_sha256=dev_hash, model_sha256=file_hash(model_path),
        test_evaluated_this_run=False, test_partition_opened=False,
        development_analysis_results=analysis_results.to_dict("records"), development_behavior=behavior,
        versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, sklearn=sklearn.__version__),
        limitations=["Only six known-state probabilities, so many ranking ties", "Development analysis was already inspected in earlier work",
                     "Prior experiments examined test outcomes; it is not an untouched holdout",
                     "Observed inspected-program cohort rather than every eligible restaurant"])
    (report_dir / "experiment_summary.json").write_text(json.dumps(model_summary, indent=2, allow_nan=False) + "\n")
    if file_hash(features_dir / "train.csv") != train_hash or file_hash(features_dir / "dev.csv") != dev_hash:
        raise RuntimeError("A feature input changed during the run")
    write_report(model_summary, lookup_table(model), metrics, report_dir)
    print("\nDevelopment analysis:", flush=True)
    print(metrics.loc[metrics.window.eq("dev_analysis"),
        ["threshold_policy", "major_ap", "minor_ap", "grp_ap", "macro_ap", "mean_brier", "label_accuracy", "exact_match_accuracy", "recall_at_10pct"]].to_string(index=False))
    return model_summary


def write_report(summary, lookup, metrics, report_dir):
    lines = ["# Category-recurrence heuristic", "",
        "This baseline estimates each category's next-inspection probability using only whether that category "
        "was present on the latest inspection day before the monthly cutoff. It consumes the existing feature "
        "tables; it does not change feature construction or fit feature coefficients.", "",
        "## Method", "", "For each category c and previous state s in {0,1}, calculate training counts n(c,s) "
        "and next-positive counts k(c,s). Let pi(c) be that category's prevalence among all training targets. "
        "The probability is (k + alpha*pi)/(n + alpha). An unobserved group or missing previous label uses "
        "pi(c). Unknown previous states are excluded from known-state groups, not treated as absent.", "",
        "The three outputs are marginal probabilities and need not sum to one. No assumption of mutually "
        "exclusive labels is made. Six probability values cover the known-state cases; unknown states add the "
        "corresponding training-prior fallback.", "",
        "## Data and selection", "",
        f"Training: {summary['training_examples']:,} examples (Apr 2024-Jun 2025). Tuning: "
        f"{summary['tuning_examples']:,} (Jul-Sep 2025). Analysis: {summary['analysis_examples']:,} "
        "(Oct-Dec 2025). The analysis window has been inspected in earlier development and is exploratory. "
        "The training script opens train.csv and dev.csv only. Earlier experiments evaluated test outcomes, "
        "but no test partition is read or evaluated here.", "",
        f"Compare alpha in {summary['config']['smoothing_grid']} using the mean of three tuning Brier scores "
        f"(lower is better). Selected alpha: {summary['selected_alpha']:g}. Exact ties favor the smaller alpha. "
        "AP is not used to tune smoothing because this baseline supplies only a few distinct probability levels.", "",
        "After choosing smoothing, select each binary threshold by tuning F1, with exact ties favoring the "
        "higher threshold. Both these decisions are frozen before reporting later analysis results. Fixed "
        "0.5 decisions are also reported as a separate policy; do not mix the two accuracy definitions.", "",
        "## Saved lookup", "",
        "| Category | Previously present | Training n | Next positive | Raw rate | Selected probability |",
        "|---|---:|---:|---:|---:|---:|"]
    for row in lookup.to_dict("records"):
        raw = f"{row['raw_rate']:.4f}" if np.isfinite(row["raw_rate"]) else "n/a"
        lines.append(f"| {row['category']} | {row['previous_present']} | {row['training_examples']:,} | "
                     f"{row['next_positive']:,} | {raw} | {row['smoothed_probability']:.4f} |")
    lines += ["", "## Later development results", "",
        "| Decision policy | Major AP | Minor AP | GRP AP | Macro AP | Mean Brier | Label accuracy | Exact match |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in metrics.loc[metrics.window.eq("dev_analysis")].to_dict("records"):
        lines.append(f"| {row['threshold_policy']} | {row['major_ap']:.4f} | {row['minor_ap']:.4f} | {row['grp_ap']:.4f} | "
                     f"{row['macro_ap']:.4f} | {row['mean_brier']:.4f} | {100*row['label_accuracy']:.2f}% | {100*row['exact_match_accuracy']:.2f}% |")
    row = metrics.loc[metrics.window.eq("dev_analysis") & metrics.threshold_policy.eq("dev_f1")].iloc[0]
    lines += ["", "AP means average precision (not trapezoidal PR-AUC). Brier measures probabilistic prediction "
        "quality; smaller is better. Label accuracy averages the three binary decisions; exact match requires "
        "all three to be correct. These measures answer different questions, and high GRP prevalence makes "
        "accuracy alone insufficient.", ""]
    if summary["development_behavior"]["all_tuned_labels_positive"]:
        lines += ["The tuning-F1 cutoffs produce every category present on every analysis example. This is "
                  "an important weakness of the thresholded baseline, not a software error.", ""]
    if summary["development_behavior"]["no_fixed_major_positives"]:
        lines += ["The fixed-0.5 policy predicts no major positives. Its overall accuracy must therefore "
                  "be interpreted alongside zero major recall.", ""]
    recall_text = f"{100*row['recall_at_10pct']:.2f}%" if np.isfinite(row["recall_at_10pct"]) else "undefined"
    lines += ["Probabilities and ranking metrics are identical under both decision policies.", "",
        f"The mean monthly major recall at a top-10% budget is {recall_text}. "
        "Priority is 4*p_major + 2*p_minor + p_GRP, representing weighted category presence rather than counts "
        "or deducted points. Top-5/10/20% queues use ceiling rounding and a seeded hash of the known program ID "
        "for deterministic tie breaking. Monthly metrics are averaged; months with no major positives have "
        "undefined major recall/lift and are excluded from those averages. Many ties are intrinsic to this simple lookup.", "",
        "## Reproduce", "", "```bash", ".venv/bin/python -m pip install -r requirements-heuristic.txt",
        ".venv/bin/python src/train_heuristic.py", ".venv/bin/python -m unittest discover -s tests -v", "```", "",
        "lookup_probabilities.csv contains all six groups. smoothing_trials.csv and decision_thresholds.csv "
        "record development choices. metrics_by_window.csv, category_metrics.csv, and monthly_rankings.csv "
        "contain complete train/development results. development_review_sample.csv is a seeded sample of "
        "30 later development examples for manual review. models/heuristic/lookup.json is the portable model.", ""]
    (report_dir / "heuristic_report.md").write_text("\n".join(lines))


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path, default=root / "data/features")
    parser.add_argument("--config", type=Path, default=root / "configs/heuristic.json")
    parser.add_argument("--report-dir", type=Path, default=root / "reports/heuristic")
    parser.add_argument("--model-path", type=Path, default=root / "models/heuristic/lookup.json")
    args = parser.parse_args()
    run(args.features_dir, args.config, args.report_dir, args.model_path)


if __name__ == "__main__":
    main()
