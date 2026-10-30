"""Tests of historical features, target selection, and chronological cutoffs."""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from build_features import FEATURES, LABELS, make_examples

CONFIG = {
    "train_start": "2024-01-01", "dev_start": "2025-07-01", "dev_analysis_start": "2025-10-01",
    "test_start": "2026-01-01", "test_end_exclusive": "2026-10-01",
}


def inspections(*overrides):
    rows = []
    for index, override in enumerate(overrides):
        row = dict(serial_number=f"I{index}", program_key="FA1|PR1", facility_id="FA1", record_id="PR1",
                   activity_date=pd.Timestamp("2024-01-15"), score=95.0,
                   pe_description="RESTAURANT (0-30) SEATS HIGH RISK", is_routine_restaurant=True,
                   label_quality_ok=True, is_modeling_eligible=True,
                   major_count=0.0, minor_count=2.0, grp_count=1.0, y_major=0.0, y_minor=1.0, y_grp=1.0)
        row.update(override)
        row["activity_date"] = pd.Timestamp(row["activity_date"])
        rows.append(row)
    return pd.DataFrame(rows)


class FeatureTests(unittest.TestCase):
    def test_current_month_inspection_does_not_enter_its_own_features(self):
        source = inspections({"activity_date": "2023-12-15", "score": 98},
                             {"activity_date": "2024-01-15", "score": 75})
        data, _ = make_examples(source, CONFIG)
        self.assertEqual(data.iloc[0].last_score, 98)
        self.assertLess(data.iloc[0].history_max_date, data.iloc[0].prediction_date)

    def test_prediction_on_first_day_does_not_use_same_day_history(self):
        source = inspections({"activity_date": "2023-12-15", "score": 98},
                             {"activity_date": "2024-01-01", "score": 70},
                             {"activity_date": "2024-01-20", "score": 80})
        data, _ = make_examples(source, CONFIG)
        self.assertEqual(len(data), 1)
        self.assertEqual(data.iloc[0].serial_number, "I1")
        self.assertEqual(data.iloc[0].last_score, 98)

    def test_uncertain_first_target_is_not_replaced_by_later_clean_visit(self):
        source = inspections({"activity_date": "2023-12-15"},
                             {"activity_date": "2024-01-05", "is_modeling_eligible": False},
                             {"activity_date": "2024-01-20"})
        data, audit = make_examples(source, CONFIG)
        self.assertTrue(data.empty)
        self.assertIn("uncertain_first_target", audit.decision.tolist())

    def test_no_history_month_excluded(self):
        data, audit = make_examples(inspections({}), CONFIG)
        self.assertTrue(data.empty)
        self.assertEqual(audit.iloc[0].decision, "no_pre_cutoff_history")

    def test_latest_uncertain_history_not_silently_skipped(self):
        source = inspections({"activity_date": "2023-11-15", "score": 99},
                             {"activity_date": "2023-12-15", "score": 80, "label_quality_ok": False,
                              "is_modeling_eligible": False, "y_major": np.nan, "y_minor": np.nan, "y_grp": np.nan},
                             {"activity_date": "2024-01-15"})
        data, _ = make_examples(source, CONFIG)
        self.assertEqual(data.iloc[0].last_score, 80)
        self.assertTrue(pd.isna(data.iloc[0].last_count_major))
        self.assertEqual(data.iloc[0].last_day_labels_missing, 1)
        self.assertEqual(data.iloc[0].n_prior_inspection_days, 2)

    def test_appending_future_records_cannot_change_earlier_features(self):
        source = inspections({"activity_date": "2023-12-15"}, {"activity_date": "2024-01-15"})
        before, _ = make_examples(source, CONFIG)
        future = inspections({"activity_date": "2024-03-15", "score": 50, "serial_number": "FUTURE"})
        after, _ = make_examples(pd.concat([source, future]), CONFIG)
        pd.testing.assert_frame_equal(before[FEATURES], after.loc[after.serial_number.eq("I1"), FEATURES].reset_index(drop=True))

    def test_changing_target_outcome_cannot_change_its_predictors(self):
        source = inspections({"activity_date": "2023-12-15"}, {"activity_date": "2024-01-15"})
        before, _ = make_examples(source, CONFIG)
        source.loc[1, ["score", "major_count", "y_major"]] = [50, 10, 1]
        after, _ = make_examples(source, CONFIG)
        pd.testing.assert_frame_equal(before[FEATURES], after[FEATURES])

    def test_ambiguous_first_target_day_excluded(self):
        source = inspections({"activity_date": "2023-12-15"},
                             {"activity_date": "2024-01-15"}, {"activity_date": "2024-01-15"})
        data, audit = make_examples(source, CONFIG)
        self.assertTrue(data.empty)
        self.assertIn("ambiguous_first_inspection_day", audit.decision.tolist())

    def test_same_day_history_uses_documented_aggregates(self):
        source = inspections({"activity_date": "2023-12-15", "score": 98, "major_count": 0, "y_major": 0},
                             {"activity_date": "2023-12-15", "score": 80, "major_count": 2, "y_major": 1},
                             {"activity_date": "2024-01-15"})
        data, _ = make_examples(source, CONFIG)
        self.assertEqual(data.iloc[0].last_score, 80)
        self.assertEqual(data.iloc[0].last_count_major, 2)
        self.assertEqual(data.iloc[0].n_prior_inspections, 2)
        self.assertEqual(data.iloc[0].n_prior_inspection_days, 1)

    def test_programs_at_same_facility_do_not_share_history(self):
        source = inspections({"activity_date": "2023-12-15"},
                             {"activity_date": "2024-01-15", "program_key": "FA1|PR2", "record_id": "PR2"})
        data, _ = make_examples(source, CONFIG)
        self.assertTrue(data.empty)

    def test_characteristics_come_from_history_not_target(self):
        source = inspections({"activity_date": "2023-12-15", "pe_description": "RESTAURANT (31-60) SEATS LOW RISK"},
                             {"activity_date": "2024-01-15", "pe_description": "RESTAURANT (151 + ) SEATS HIGH RISK"})
        data, _ = make_examples(source, CONFIG)
        self.assertEqual(data.iloc[0].historical_risk_tier, "LOW")
        self.assertEqual(data.iloc[0].historical_seat_band, "31-60")

    def test_split_boundaries_follow_targets(self):
        source = inspections({"activity_date": "2023-12-15"}, {"activity_date": "2025-06-30"},
                             {"activity_date": "2025-07-01"}, {"activity_date": "2026-01-01"})
        data, _ = make_examples(source, CONFIG)
        self.assertEqual(data.split.tolist(), ["train", "dev", "test"])

    def test_feature_allowlist_has_no_targets_or_ids(self):
        self.assertFalse(set(FEATURES) & set(LABELS + ["target_score_for_audit", "target_date", "program_key"]))



if __name__ == "__main__":
    unittest.main()
