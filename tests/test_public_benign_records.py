import json
from pathlib import Path
import unittest

from scripts.public_benign_records import validate_qwen, validate_sar


RECORDS = Path(__file__).resolve().parents[1] / "experiments/records/2026-11-19"


def read_record(name):
    return json.loads((RECORDS / name).read_text(encoding="utf-8"))


class PublicBenignRecordTests(unittest.TestCase):
    def test_qwen_denominators(self):
        self.assertEqual(validate_qwen(read_record("qwen-benign.json")), {
            "completed": 12, "resource_blocked": 1, "timeout": 1,
            "preflight_blocked": 1, "not_run": 2,
        })

    def test_qwen_rejects_incorrect_exact_score(self):
        data = read_record("qwen-benign.json")
        row = next(row for row in data["records"] if row["id"] == "digit-neutral-image")
        row["exact_pass"] = True
        with self.assertRaisesRegex(ValueError, "exact score mismatch"):
            validate_qwen(data)

    def test_qwen_rejects_scored_timeout(self):
        data = read_record("qwen-benign.json")
        row = next(row for row in data["records"] if row["status"] == "timeout")
        row["semantic"] = "wrong"
        with self.assertRaisesRegex(ValueError, "non-completed"):
            validate_qwen(data)

    def test_sar_denominators(self):
        self.assertEqual(validate_sar(read_record("sar-summary.json")), (36, 4, 2))

    def test_sar_rejects_invalid_overlap(self):
        data = read_record("sar-summary.json")
        data["direct_query"]["paired_same_wrong_class"] = 3
        with self.assertRaisesRegex(ValueError, "invalid direct-query overlap"):
            validate_sar(data)
