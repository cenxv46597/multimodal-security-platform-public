"""Offline unit tests for the 2026-11-25 public SAR transfer release validator."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

from validate_sar_20261125_release import SUMMARY_PATH, validate_summary


class TestSar20261125Release(unittest.TestCase):
    def test_published_summary_and_scripts_pass(self):
        data = validate_summary(SUMMARY_PATH)
        self.assertEqual(data['date'], '2026-11-25')
        self.assertEqual(data['route1_visual_prototype_matching_16bit']['cohort17_scaling']['n_samples'], 17)

    def test_rejects_raw_sample_filename_leak(self):
        data = json.loads(SUMMARY_PATH.read_text(encoding='utf-8'))
        data['leaked_note'] = '0047270.jpg'
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'summary.json'
            p.write_text(json.dumps(data), encoding='utf-8')
            with self.assertRaises(ValueError):
                validate_summary(p)

    def test_rejects_inconsistent_class_sum(self):
        data = json.loads(SUMMARY_PATH.read_text(encoding='utf-8'))
        data['route1_visual_prototype_matching_16bit']['cohort17_scaling']['internvl2b_by_reference_class']['tank']['wrong'] = 4
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'summary.json'
            p.write_text(json.dumps(data), encoding='utf-8')
            with self.assertRaises(ValueError):
                validate_summary(p)


if __name__ == '__main__':
    unittest.main()
