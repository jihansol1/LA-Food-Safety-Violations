"""Classification/probability metrics and deterministic monthly inspection queues."""
import hashlib
import math

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             f1_score, precision_recall_curve, precision_score, recall_score)

from heuristic import CATEGORIES, LABELS, validate_targets


def select_thresholds(outcomes, probabilities):
    outcomes, probabilities = np.asarray(outcomes), np.asarray(probabilities)
    if outcomes.ndim != 2 or outcomes.shape[1] != 3 or len(outcomes) == 0 or probabilities.shape != outcomes.shape:
        raise ValueError("Expected aligned nonempty N x 3 outcomes and probabilities")
    if not np.isin(outcomes, [0, 1]).all() or not np.isfinite(probabilities).all() or not ((probabilities >= 0) & (probabilities <= 1)).all():
        raise ValueError("Invalid labels or probabilities")
    thresholds = []
    for index in range(3):
        y, p = outcomes[:, index], probabilities[:, index]
        if not y.any():
            thresholds.append(float(np.nextafter(p.max(), np.inf)))
            continue
        precision, recall, candidates = precision_recall_curve(y, p)
        f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-15)
        maximum = f1.max()
        choices = np.flatnonzero(np.isclose(f1, maximum, rtol=0, atol=1e-12))
        # Exact ties use the largest threshold, predicting fewer positives.
        thresholds.append(float(candidates[choices[-1]]))
    return thresholds


def classification_metrics(frame, probabilities, thresholds, policy):
    y = validate_targets(frame)
    p = np.asarray(probabilities)
    if p.shape != y.shape or not np.isfinite(p).all() or not ((p >= 0) & (p <= 1)).all():
        raise ValueError("Expected aligned N x 3 probabilities")
    thresholds = np.asarray(thresholds, dtype=float)
    if thresholds.shape != (3,) or not np.isfinite(thresholds).all() or not ((thresholds >= 0) & (thresholds <= np.nextafter(1.0, np.inf))).all():
        raise ValueError("Expected three finite decision thresholds")
    decisions = (p >= thresholds).astype(int)
    rows = []
    for index, category in enumerate(CATEGORIES):
        rows.append(dict(category=category, threshold_policy=policy, examples=len(frame),
            prevalence=float(y[:, index].mean()), average_precision=float(average_precision_score(y[:, index], p[:, index])),
            brier=float(brier_score_loss(y[:, index], p[:, index])),
            accuracy=float(accuracy_score(y[:, index], decisions[:, index])),
            precision=float(precision_score(y[:, index], decisions[:, index], zero_division=0)),
            recall=float(recall_score(y[:, index], decisions[:, index], zero_division=0)),
            f1=float(f1_score(y[:, index], decisions[:, index], zero_division=0)), threshold=float(thresholds[index])))
    summary = dict(threshold_policy=policy, examples=len(frame),
        macro_ap=float(np.mean([r["average_precision"] for r in rows])),
        mean_brier=float(np.mean([r["brier"] for r in rows])),
        macro_f1=float(np.mean([r["f1"] for r in rows])),
        label_accuracy=float((decisions == y).mean()), exact_match_accuracy=float((decisions == y).all(axis=1).mean()))
    for category, row in zip(CATEGORIES, rows):
        summary[f"{category}_ap"] = row["average_precision"]
    return summary, pd.DataFrame(rows)


def monthly_ranking(frame, probabilities, fraction, weights, seed=467):
    if not 0 < fraction <= 1:
        raise ValueError("Budget fraction must be in (0,1]")
    if frame.duplicated(["program_key", "prediction_date"]).any():
        raise ValueError("Ranking candidates must be unique program-months")
    y = validate_targets(frame)
    probabilities = np.asarray(probabilities)
    if probabilities.shape != y.shape or not np.isfinite(probabilities).all():
        raise ValueError("Invalid ranking probabilities")
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (3,) or not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("Expected three nonnegative severity weights")
    work = frame[["prediction_date", "program_key"] + LABELS].copy().reset_index(drop=True)
    work["priority"] = probabilities @ weights
    work["tie"] = work.program_key.map(lambda key: hashlib.sha256(f"{seed}|{key}".encode()).hexdigest())
    rows = []
    for month, group in work.groupby("prediction_date", sort=True):
        ranked = group.sort_values(["priority", "tie"], ascending=[False, True], kind="stable")
        k = max(1, math.ceil(len(group) * fraction))
        positives = int(group.y_major.sum())
        hits = int(ranked.iloc[:k].y_major.sum())
        prevalence = positives / len(group)
        gain = ranked[LABELS].to_numpy(dtype=float) @ weights
        discount = np.log2(np.arange(2, k + 2))
        ideal = float((np.sort(gain)[::-1][:k] / discount).sum())
        rows.append(dict(month=str(month), budget_fraction=fraction, candidates=len(group), k=k,
                         major_positive=positives, major_hits=hits, precision=hits/k,
                         recall=hits/positives if positives else np.nan,
                         lift=(hits/k)/prevalence if positives else np.nan,
                         ndcg=float((gain[:k]/discount).sum())/ideal if ideal else np.nan,
                         distinct_priority_scores=int(group.priority.nunique()),
                         largest_tie_group=int(group.priority.value_counts().max())))
    detail = pd.DataFrame(rows)
    summary = {metric: float(detail[metric].mean()) for metric in ["precision", "recall", "lift", "ndcg"]}
    summary.update(months=len(detail), months_with_major_positives=int(detail.major_positive.gt(0).sum()))
    return summary, detail
