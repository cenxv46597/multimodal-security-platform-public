import copy
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from validate_research_20261121 import validate


class RecordsTest(unittest.TestCase):
    def setUp(self):
        path = ROOT / 'experiments/records/2026-11-21/results.json'
        self.data = json.loads(path.read_text(encoding='utf-8'))

    def test_original(self):
        self.assertEqual(validate(self.data), 6)

    def test_rejects_incorrect_planned_denominator(self):
        data = copy.deepcopy(self.data)
        data['trusted_gate']['not_run'] = 3
        with self.assertRaises(ValueError):
            validate(data)

    def test_rejects_missing_paired_answer(self):
        data = copy.deepcopy(self.data)
        data['random_sign']['rows'][0]['random_sign_2'] = ''
        with self.assertRaises(ValueError):
            validate(data)

    def test_rejects_duplicate_prior_condition(self):
        path = ROOT / 'experiments/records/2026-11-21/sar-text-prior-review.json'
        prior = json.loads(path.read_text(encoding='utf-8'))
        prior['records'][1] = copy.deepcopy(prior['records'][0])
        with self.assertRaises(ValueError):
            validate(self.data, prior)


if __name__ == '__main__':
    unittest.main()
