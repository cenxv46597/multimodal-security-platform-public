import json
import unittest

from scripts.validate_sar_20261123_release import RECORD, validate_summary


class SarReleaseSummaryTests(unittest.TestCase):
    def setUp(self):
        self.record = json.loads(RECORD.read_text(encoding="utf-8"))

    def test_public_record(self):
        validate_summary(self.record)

    def test_rejects_unreconciled_candidates(self):
        self.record["candidates"]["pending"] -= 1
        with self.assertRaisesRegex(ValueError, "dispositions"):
            validate_summary(self.record)

    def test_rejects_premature_freeze(self):
        self.record["candidates"]["frozen"] = 1
        with self.assertRaisesRegex(ValueError, "frozen"):
            validate_summary(self.record)

    def test_rejects_incorrect_uncovered_count(self):
        self.record["overlap"]["uncovered_size_or_shape"] += 1
        with self.assertRaisesRegex(ValueError, "Uncovered"):
            validate_summary(self.record)

    def test_rejects_counting_failed_preflight_as_tested(self):
        self.record["overlap"]["mixed_size_preflight"]["tested"] = 1
        with self.assertRaisesRegex(ValueError, "preflight"):
            validate_summary(self.record)
