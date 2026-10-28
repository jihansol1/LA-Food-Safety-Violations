"""Tests of substantive label, join, encoding, and repeated-row behavior."""

import importlib.util
from pathlib import Path
import tempfile
import unittest

import pandas as pd

spec = importlib.util.spec_from_file_location(
    "prepare_data", Path(__file__).resolve().parents[1] / "src" / "prepare_data.py"
)
clean = importlib.util.module_from_spec(spec)
spec.loader.exec_module(clean)


def inspection_rows(*rows):
    records = []
    for row in rows:
        base = dict(activity_date="01/15/2025", facility_id="FA1", record_id="PR1",
                    serial_number="I1", score="100", pe_description="RESTAURANT (0-30) SEATS HIGH RISK",
                    service_code="001", service_description="ROUTINE INSPECTION")
        base.update(row)
        records.append(base)
    result = pd.DataFrame(records, dtype="string")
    result.insert(0, "source_record_number", range(1, len(result) + 1))
    return result


def violation_rows(*rows):
    records = []
    for row in rows:
        base = dict(serial_number="I1", violation_status="OUT OF COMPLIANCE", violation_code="F134",
                    violation_description="NON-FOOD CONTACT SURFACES CLEAN", points="1")
        base.update(row)
        records.append(base)
    columns = ["serial_number", "violation_status", "violation_code", "violation_description", "points"]
    result = pd.DataFrame(records, columns=columns, dtype="string")
    result.insert(0, "source_record_number", range(1, len(result) + 1))
    return clean.prepare_violations(result)


class CleaningTests(unittest.TestCase):
    def test_legacy_and_new_critical_codes(self):
        for code in ["F014", "F115"]:
            for points, expected in [(2, "minor"), (4, "major"), (11, "major")]:
                self.assertEqual(clean.classify_violation(code, points, "OUT OF COMPLIANCE")[0], expected)

    def test_grp_cannot_become_major_from_bad_points(self):
        self.assertEqual(clean.classify_violation("F134", 4, "OUT OF COMPLIANCE")[0], "unresolved")
        self.assertEqual(clean.classify_violation("F115", 1, "OUT OF COMPLIANCE")[0], "unresolved")

    def test_penalty_and_actions_do_not_create_target_labels(self):
        for code in ["F054", "F153"]:
            self.assertEqual(clean.classify_violation(code, 3, "OUT OF COMPLIANCE")[0], "multiple_major_penalty")
        self.assertEqual(clean.classify_violation("F150", 0, "OUT OF COMPLIANCE")[0], "administrative")

    def test_zero_point_substantive_and_unusual_status_are_unresolved(self):
        for code, points, status in [("F014", 0, "OUT OF COMPLIANCE"),
                                     ("F014", 6, "OUT OF COMPLIANCE"),
                                     ("F052", 2, "NO"), ("W026", 0, "OUT OF COMPLIANCE")]:
            self.assertEqual(clean.classify_violation(code, points, status)[0], "unresolved")

    def test_repeated_records_preserved_when_points_reconcile(self):
        violations = violation_rows({}, {})
        frame = clean.prepare_inspections(inspection_rows({"score": "98"}), violations)
        self.assertEqual(len(violations), 2)
        self.assertEqual(frame.loc[0, "grp_count"], 2)
        self.assertEqual(frame.loc[0, "repeated_record_count"], 1)
        self.assertTrue(frame.loc[0, "is_modeling_eligible"])

    def test_left_join_retains_perfect_score_with_no_violation(self):
        frame = clean.prepare_inspections(inspection_rows({}), violation_rows())
        self.assertEqual(len(frame), 1)
        self.assertEqual(frame.loc[0, clean.LABELS].tolist(), [0, 0, 0])
        self.assertTrue(frame.loc[0, "inferred_no_scored_violations"])
        self.assertTrue(frame.loc[0, "is_modeling_eligible"])

    def test_missing_violation_with_bad_score_is_not_false_negative(self):
        frame = clean.prepare_inspections(inspection_rows({"score": "90"}), violation_rows())
        self.assertTrue(frame[clean.LABELS].isna().all().all())
        self.assertFalse(frame.loc[0, "is_modeling_eligible"])

    def test_unresolved_inspection_keeps_counts_but_withholds_labels(self):
        frame = clean.prepare_inspections(inspection_rows({"score": "94"}),
                                         violation_rows({"violation_code": "F014", "points": "6"}))
        self.assertTrue(frame.loc[0, "has_unresolved_violations"])
        self.assertTrue(frame[clean.LABELS].isna().all().all())
        self.assertEqual(len(frame), 1)

    def test_score_mismatch_is_not_repaired(self):
        frame = clean.prepare_inspections(inspection_rows({"score": "95"}), violation_rows({}))
        self.assertEqual(frame.loc[0, "score"], 95)
        self.assertTrue(frame.loc[0, "score_point_mismatch"])
        self.assertTrue(frame[clean.LABELS].isna().all().all())

    def test_orphan_cannot_create_inspection(self):
        frame = clean.prepare_inspections(inspection_rows({}), violation_rows({"serial_number": "ORPHAN"}))
        self.assertEqual(frame.serial_number.tolist(), ["I1"])

    def test_facility_programs_stay_distinct(self):
        frame = clean.prepare_inspections(inspection_rows({}, {"serial_number": "I2", "record_id": "PR2"}), violation_rows())
        self.assertEqual(frame.program_key.nunique(), 2)

    def test_nonroutine_and_markets_are_not_modeling_targets(self):
        frame = clean.prepare_inspections(inspection_rows(
            {"service_description": "OWNER INITIATED INSPECTION"},
            {"serial_number": "I2", "pe_description": "FOOD MKT RETAIL (1-1,999 SF) LOW RISK"}), violation_rows())
        self.assertFalse(frame.is_modeling_eligible.any())

    def test_invalid_date_or_missing_identifier_cannot_enter_cohort(self):
        frame = clean.prepare_inspections(inspection_rows(
            {"activity_date": "not-a-date"}, {"serial_number": "I2", "record_id": pd.NA}), violation_rows())
        self.assertFalse(frame.is_modeling_eligible.any())

    def test_duplicate_inspection_key_stops_the_pipeline(self):
        with self.assertRaises(ValueError):
            clean.prepare_inspections(inspection_rows({}, {}), violation_rows())

    def test_strict_decoding_whitespace_dates_and_leading_zero_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.csv"
            path.write_bytes('ACTIVITY DATE,SERVICE CODE,NAME\n 05/18/2024,001,Caf\u00e9\u2013Test\n'.encode("cp1252"))
            frame, metadata = clean.read_source(path, {"activity_date", "service_code"})
            self.assertEqual(metadata["encoding"], "cp1252")
            self.assertEqual(frame.loc[0, "service_code"], "001")
            self.assertEqual(frame.loc[0, "name"], "Caf\u00e9\u2013Test")
            self.assertFalse(pd.to_datetime(frame.activity_date, format="%m/%d/%Y").isna().any())


if __name__ == "__main__":
    unittest.main()
