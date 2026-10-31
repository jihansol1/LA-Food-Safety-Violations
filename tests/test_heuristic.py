"""Tests of recurrence counts, smoothing, missing states, thresholds, and queues."""
from pathlib import Path
import contextlib
import copy
import io
import json
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from heuristic import (INPUTS, LABELS, fit_lookup, load_model, lookup_table,
                       predict_labels, predict_probabilities, save_model)
from heuristic_metrics import classification_metrics, monthly_ranking, select_thresholds
from train_heuristic import choose_smoothing, run


class HeuristicTests(unittest.TestCase):
    def setUp(self):
        self.training = pd.DataFrame(dict(last_present_major=[0, 0, 1, 1],
            last_present_minor=[0, 1, 0, 1], last_present_grp=[1, 1, 1, 1],
            y_major=[0, 0, 1, 0], y_minor=[0, 1, 1, 1], y_grp=[1, 1, 1, 0]))

    def test_training_counts_and_unsmoothed_rates(self):
        model = fit_lookup(self.training, 0)
        major = model["category_parameters"]["major"]
        self.assertEqual(major["training_prevalence"], 0.25)
        self.assertEqual(major["groups"]["0"]["training_examples"], 2)
        self.assertEqual(major["groups"]["0"]["probability"], 0)
        self.assertEqual(major["groups"]["1"]["probability"], 0.5)
        self.assertEqual(len(lookup_table(model)), 6)

    def test_smoothing_uses_training_prior(self):
        model = fit_lookup(self.training, 2)
        self.assertEqual(model["category_parameters"]["major"]["groups"]["0"]["probability"], 0.125)
        self.assertEqual(model["category_parameters"]["major"]["groups"]["1"]["probability"], 0.375)

    def test_unknown_state_uses_prior_instead_of_absence(self):
        model = fit_lookup(self.training, 0)
        case = pd.DataFrame(dict(last_present_major=[np.nan], last_present_minor=[np.nan], last_present_grp=[np.nan]))
        np.testing.assert_allclose(predict_probabilities(model, case), [[0.25, 0.75, 0.75]])

    def test_unobserved_training_group_uses_prior_even_without_smoothing(self):
        for alpha in [0, 20]:
            model = fit_lookup(self.training, alpha)
            case = pd.DataFrame(dict(last_present_major=[0], last_present_minor=[0], last_present_grp=[0]))
            self.assertEqual(predict_probabilities(model, case)[0, 2], 0.75)

    def test_unknown_training_states_are_not_counted_as_absent(self):
        source = self.training.copy()
        source.loc[0, "last_present_major"] = np.nan
        model = fit_lookup(source)
        parameters = model["category_parameters"]["major"]
        self.assertEqual(parameters["groups"]["0"]["training_examples"], 1)
        self.assertEqual(parameters["unknown_previous_examples"], 1)

    def test_three_probabilities_are_not_softmax(self):
        probabilities = predict_probabilities(fit_lookup(self.training, 5), self.training)
        self.assertEqual(probabilities.shape, (4, 3))
        self.assertTrue(np.isfinite(probabilities).all())
        self.assertTrue(((probabilities >= 0) & (probabilities <= 1)).all())
        self.assertFalse(np.allclose(probabilities.sum(axis=1), 1))

    def test_inference_ignores_future_labels_and_other_features(self):
        model = fit_lookup(self.training, 20)
        expected = predict_probabilities(model, self.training)
        changed = self.training.copy()
        changed[LABELS] = 1 - changed[LABELS]
        changed["last_score"] = 0
        changed["target_score_for_audit"] = 100
        np.testing.assert_allclose(predict_probabilities(model, changed), expected)

    def test_prediction_does_not_mutate_model(self):
        model = fit_lookup(self.training)
        original = copy.deepcopy(model)
        predict_probabilities(model, self.training)
        self.assertEqual(model, original)

    def test_serialization_round_trip(self):
        model = fit_lookup(self.training, 20)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lookup.json"
            save_model(model, path)
            loaded = load_model(path)
            self.assertEqual(loaded, model)
            np.testing.assert_allclose(predict_probabilities(loaded, self.training), predict_probabilities(model, self.training))

    def test_invalid_previous_states_targets_and_smoothing_are_rejected(self):
        for value in [2, -1, np.inf, "unknown"]:
            invalid = self.training.copy()
            invalid["last_present_major"] = invalid.last_present_major.astype(object)
            invalid.loc[0, "last_present_major"] = value
            with self.assertRaises(ValueError):
                fit_lookup(invalid)
        invalid = self.training.copy()
        invalid.loc[0, "y_major"] = np.nan
        with self.assertRaises(ValueError):
            fit_lookup(invalid)
        for alpha in [-1, np.inf, np.nan]:
            with self.assertRaises(ValueError):
                fit_lookup(self.training, alpha)

    def test_missing_columns_fail_instead_of_silent_fallback(self):
        model = fit_lookup(self.training)
        with self.assertRaises(ValueError):
            predict_probabilities(model, self.training.drop(columns="last_present_major"))

    def test_corrupt_model_probability_is_rejected(self):
        model = fit_lookup(self.training)
        model["category_parameters"]["major"]["groups"]["0"]["probability"] = 0.99
        with self.assertRaises(ValueError):
            predict_probabilities(model, self.training)

    def test_empty_prediction_table_is_supported(self):
        model = fit_lookup(self.training)
        self.assertEqual(predict_probabilities(model, self.training.iloc[:0]).shape, (0, 3))

    def test_smoothing_is_selected_by_tuning_brier(self):
        with contextlib.redirect_stdout(io.StringIO()):
            model, trials = choose_smoothing(self.training, self.training, [100, 20, 5, 0])
        self.assertEqual(model["smoothing_alpha"], trials.loc[trials.mean_brier.idxmin(), "alpha"])
        self.assertEqual(model["training_examples"], len(self.training))

    def test_tuning_outcomes_do_not_enter_training_counts(self):
        tuning = pd.concat([self.training] * 10, ignore_index=True)
        tuning[LABELS] = 1
        with contextlib.redirect_stdout(io.StringIO()):
            model, _ = choose_smoothing(self.training, tuning, [0, 100])
        self.assertEqual(model["training_examples"], 4)
        self.assertEqual(model["category_parameters"]["major"]["training_prevalence"], 0.25)
        self.assertEqual(model["category_parameters"]["major"]["groups"]["1"]["next_positive"], 1)

    def test_f1_thresholds_and_all_negative_case(self):
        outcomes = np.array([[0, 0, 0], [1, 1, 0]])
        probabilities = np.array([[0.1, 0.2, 1.0], [0.7, 0.8, 1.0]])
        thresholds = select_thresholds(outcomes, probabilities)
        np.testing.assert_allclose(thresholds[:2], [0.7, 0.8])
        self.assertGreater(thresholds[2], 1)
        self.assertFalse((probabilities[:, 2] >= thresholds[2]).any())

    def test_binary_policy_is_separate_from_probabilities(self):
        model = fit_lookup(self.training)
        before = predict_probabilities(model, self.training)
        predict_labels(model, self.training, [0.1, 0.2, 0.3])
        np.testing.assert_allclose(predict_probabilities(model, self.training), before)

    def test_classification_metrics_use_label_and_exact_match_accuracy(self):
        frame = self.training.copy()
        probabilities = frame[LABELS].to_numpy(dtype=float)
        summary, categories = classification_metrics(frame, probabilities, [0.5] * 3, "fixed_0.5")
        self.assertEqual(summary["label_accuracy"], 1)
        self.assertEqual(summary["exact_match_accuracy"], 1)
        self.assertEqual(summary["mean_brier"], 0)

    def test_ranking_ties_are_stable_under_row_reordering(self):
        frame = self.training.assign(program_key=["A", "B", "C", "D"], prediction_date="2025-10-01")
        probabilities = np.full((4, 3), 0.5)
        first, detail = monthly_ranking(frame, probabilities, 0.1, [4, 2, 1])
        second, _ = monthly_ranking(frame.iloc[::-1], probabilities, 0.1, [4, 2, 1])
        self.assertEqual(first, second)
        self.assertEqual(detail.iloc[0].k, 1)
        self.assertEqual(detail.iloc[0].largest_tie_group, 4)

    def test_ranking_without_major_positives_has_undefined_recall(self):
        frame = self.training.assign(y_major=0, program_key=["A", "B", "C", "D"], prediction_date="2025-10-01")
        result, _ = monthly_ranking(frame, np.full((4, 3), 0.5), 0.5, [4, 2, 1])
        self.assertTrue(np.isnan(result["recall"]))
        self.assertEqual(result["months_with_major_positives"], 0)

    def test_full_run_never_opens_test_and_exports_six_groups(self):
        def fixture(n, development):
            index = np.arange(n)
            frame = pd.DataFrame({INPUTS[k]: ((index + k) % 2).astype(float) for k in range(3)})
            for k in range(3):
                frame[LABELS[k]] = ((index + k) % 3 == 0).astype(int)
            frame["serial_number"] = [f"I{development}{i}" for i in index]
            frame["program_key"] = [f"P{development}{i}" for i in index]
            frame["history_max_date"] = "2025-03-01"
            frame["prediction_date"] = "2025-05-01"
            frame["target_date"] = "2025-05-15"
            frame["split"] = "dev" if development else "train"
            if development:
                frame["dev_role"] = ["tune"] * (n//2) + ["analysis"] * (n//2)
                for role, month in [("tune", "08"), ("analysis", "11")]:
                    frame.loc[frame.dev_role.eq(role), "prediction_date"] = f"2025-{month}-01"
                    frame.loc[frame.dev_role.eq(role), "target_date"] = f"2025-{month}-15"
            return frame
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            features = root / "features"
            features.mkdir()
            fixture(36, False).to_csv(features / "train.csv", index=False)
            fixture(24, True).to_csv(features / "dev.csv", index=False)
            poison = "never parse this final-test file\n"
            (features / "test.csv").write_text(poison)
            config = dict(smoothing_grid=[0, 5], selection_metric="mean_brier_score", threshold_metric="per_label_f1",
                          seed=467, priority_weights=[4, 2, 1], ranking_budgets=[0.05, 0.1, 0.2])
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config))
            with contextlib.redirect_stdout(io.StringIO()):
                summary = run(features, config_path, root / "reports", root / "models/lookup.json")
            self.assertFalse(summary["test_partition_opened"])
            self.assertEqual((features / "test.csv").read_text(), poison)
            self.assertEqual(len(pd.read_csv(root / "reports/lookup_probabilities.csv")), 6)
            self.assertEqual(len(pd.read_csv(root / "reports/smoothing_trials.csv")), 2)
            model = load_model(root / "models/lookup.json")
            self.assertEqual(model["training_examples"], 36)


if __name__ == "__main__":
    unittest.main()
