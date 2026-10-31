"""Category-recurrence lookup baseline. NumPy/pandas only; no learned coefficients.

Inputs are the three latest known-at-cutoff category-presence features, including
missing values when the latest inspection labels are uncertain. Unknown states
use training prevalence rather than silently becoming absence or an older visit.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

CATEGORIES = ["major", "minor", "grp"]
INPUTS = [f"last_present_{category}" for category in CATEGORIES]
LABELS = [f"y_{category}" for category in CATEGORIES]


def validate_inputs(frame):
    missing = set(INPUTS) - set(frame)
    if missing:
        raise ValueError(f"Missing heuristic inputs: {sorted(missing)}")
    result = pd.DataFrame(index=frame.index)
    for column in INPUTS:
        try:
            result[column] = pd.to_numeric(frame[column], errors="raise")
        except (TypeError, ValueError) as error:
            raise ValueError(f"{column} must contain 0, 1, or missing values") from error
        if not result[column].dropna().isin([0, 1]).all():
            raise ValueError(f"{column} must contain 0, 1, or missing values")
    return result


def validate_targets(frame):
    if not len(frame) or set(LABELS) - set(frame) or not frame[LABELS].isin([0, 1]).all().all():
        raise ValueError("Training/evaluation targets must be complete nonempty binary indicator columns")
    return frame[LABELS].to_numpy(dtype=int)


def fit_lookup(training, alpha=0):
    alpha = float(alpha)
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("Smoothing alpha must be finite and nonnegative")
    previous = validate_inputs(training)
    outcomes = validate_targets(training)
    model = dict(format_version=1, method="category_recurrence", input_columns=INPUTS,
                 label_columns=LABELS, smoothing_alpha=alpha, training_examples=len(training),
                 decision_thresholds=[0.5, 0.5, 0.5], category_parameters={})
    for index, category in enumerate(CATEGORIES):
        prior = float(outcomes[:, index].mean())
        values = previous[INPUTS[index]]
        groups = {}
        for state in [0, 1]:
            mask = values.eq(state).fillna(False).to_numpy(dtype=bool)
            n = int(mask.sum())
            k = int(outcomes[mask, index].sum())
            probability = (k + alpha * prior) / (n + alpha) if n else prior
            groups[str(state)] = dict(training_examples=n, next_positive=k, probability=float(probability))
        model["category_parameters"][category] = dict(training_prevalence=prior,
            unknown_previous_examples=int(values.isna().sum()), groups=groups)
    validate_model(model)
    return model


def validate_model(model):
    if model.get("format_version") != 1 or model.get("method") != "category_recurrence":
        raise ValueError("Unsupported heuristic model format")
    if model.get("input_columns") != INPUTS or model.get("label_columns") != LABELS:
        raise ValueError("Unexpected heuristic input or output ordering")
    if not np.isfinite(model["smoothing_alpha"]) or model["smoothing_alpha"] < 0:
        raise ValueError("Invalid smoothing parameter")
    if int(model["training_examples"]) <= 0:
        raise ValueError("Invalid training example count")
    thresholds = np.asarray(model["decision_thresholds"], dtype=float)
    if thresholds.shape != (3,) or not np.isfinite(thresholds).all() or not ((thresholds >= 0) & (thresholds <= np.nextafter(1.0, np.inf))).all():
        raise ValueError("Expected three finite decision thresholds")
    for category in CATEGORIES:
        parameters = model["category_parameters"][category]
        prior = parameters["training_prevalence"]
        if not np.isfinite(prior) or not 0 <= prior <= 1:
            raise ValueError("Invalid training prevalence")
        counts = parameters["unknown_previous_examples"]
        if counts < 0:
            raise ValueError("Invalid unknown-state count")
        for state in ["0", "1"]:
            group = parameters["groups"][state]
            n, k, probability = group["training_examples"], group["next_positive"], group["probability"]
            if n < 0 or k < 0 or k > n or not np.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError("Invalid recurrence group counts or probability")
            expected = (k + model["smoothing_alpha"] * prior) / (n + model["smoothing_alpha"]) if n else prior
            if not np.isclose(probability, expected, rtol=1e-12, atol=1e-12):
                raise ValueError("Stored probability does not match training counts and smoothing")
            counts += n
        if counts != model["training_examples"]:
            raise ValueError("Group counts do not account for training examples")


def predict_probabilities(model, frame):
    validate_model(model)
    inputs = validate_inputs(frame)
    probabilities = np.empty((len(frame), 3), dtype=float)
    for index, category in enumerate(CATEGORIES):
        parameters = model["category_parameters"][category]
        probabilities[:, index] = parameters["training_prevalence"]
        for state in [0, 1]:
            mask = inputs[INPUTS[index]].eq(state).fillna(False).to_numpy(dtype=bool)
            probabilities[mask, index] = parameters["groups"][str(state)]["probability"]
    assert np.isfinite(probabilities).all() and ((probabilities >= 0) & (probabilities <= 1)).all()
    return probabilities


def predict_labels(model, frame, thresholds=None):
    selected = model["decision_thresholds"] if thresholds is None else thresholds
    selected = np.asarray(selected, dtype=float)
    if selected.shape != (3,) or not np.isfinite(selected).all() or not ((selected >= 0) & (selected <= np.nextafter(1.0, np.inf))).all():
        raise ValueError("Expected three finite decision thresholds")
    return (predict_probabilities(model, frame) >= selected).astype(int)


def lookup_table(model):
    validate_model(model)
    rows = []
    for category in CATEGORIES:
        parameters = model["category_parameters"][category]
        for state in [0, 1]:
            group = parameters["groups"][str(state)]
            n, k = group["training_examples"], group["next_positive"]
            rows.append(dict(category=category, previous_present=state, training_examples=n,
                             next_positive=k, next_negative=n-k, raw_rate=k/n if n else np.nan,
                             training_prevalence=parameters["training_prevalence"],
                             smoothed_probability=group["probability"],
                             unknown_previous_examples=parameters["unknown_previous_examples"]))
    return pd.DataFrame(rows)


def save_model(model, path):
    validate_model(model)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model, indent=2, allow_nan=False) + "\n")


def load_model(path):
    model = json.loads(Path(path).read_text())
    validate_model(model)
    return model
