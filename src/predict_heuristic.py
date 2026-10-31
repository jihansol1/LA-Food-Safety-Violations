"""Apply the saved JSON recurrence model to precomputed historical presence flags."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from heuristic import CATEGORIES, load_model, predict_labels, predict_probabilities


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold-policy", choices=["dev_f1", "fixed_0.5"], default="dev_f1")
    args = parser.parse_args()
    if args.output.resolve() in [args.input.resolve(), args.model.resolve()]:
        parser.error("Output must not overwrite the input or model")
    try:
        model = load_model(args.model)
        frame = pd.read_csv(args.input, dtype={"serial_number": str, "program_key": str}, low_memory=False)
        probabilities = predict_probabilities(model, frame)
        decisions = predict_labels(model, frame, [0.5] * 3 if args.threshold_policy == "fixed_0.5" else None)
    except ValueError as error:
        parser.error(str(error))
    metadata = [column for column in ["serial_number", "program_key", "prediction_date"] if column in frame]
    result = frame[metadata].copy()
    for index, category in enumerate(CATEGORIES):
        result[f"p_{category}"] = probabilities[:, index]
        result[f"pred_{category}"] = decisions[:, index]
    result["priority_score"] = probabilities @ np.asarray(model["metadata"]["priority_weights"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
