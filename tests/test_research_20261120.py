import copy
import json
import unittest

from scripts.validate_research_20261120 import DIRECTORY, validate


def load(name):
    return json.loads((DIRECTORY / name).read_text(encoding="utf-8"))


class ResearchRecordTests(unittest.TestCase):
    def setUp(self):
        self.synthetic = load("qwen-synthetic.json")
        self.sar = load("sar-summary.json")

    def test_real_records(self):
        self.assertEqual(validate(self.synthetic, self.sar), 16)

    def test_capped_call_rejected(self):
        self.synthetic["records"][0]["capped"] = True
        with self.assertRaisesRegex(ValueError, "incomplete or capped"):
            validate(self.synthetic, self.sar)

    def test_invalid_paired_count_rejected(self):
        self.sar["direct_query_update"]["same_wrong_class_as_random"] = 5
        with self.assertRaisesRegex(ValueError, "invalid paired overlap"):
            validate(self.synthetic, self.sar)

    def test_invalid_baseline_denominator_rejected(self):
        altered = copy.deepcopy(self.sar)
        altered["qwen_sar_baseline"]["domain_prior"]["saved_wrong"] = 2
        with self.assertRaisesRegex(ValueError, "judgment denominator"):
            validate(self.synthetic, altered)
