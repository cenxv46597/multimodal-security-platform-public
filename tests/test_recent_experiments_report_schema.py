"""Offline stdlib-only unit tests for scripts/recent_experiments_report_schema.py."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

import recent_experiments_report_schema as schema


class RecentExperimentsReportSchemaTests(unittest.TestCase):
    def test_load_verified_json_passes_and_fails_closed_on_tamper(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'record.json'
            p.write_text(json.dumps({'status': 'completed'}), encoding='utf-8')
            digest = schema.sha256_file(p)
            loaded = schema.load_verified_json(p, digest)
            self.assertEqual(loaded['status'], 'completed')

            p.write_text(json.dumps({'status': 'completed'}) + ' ', encoding='utf-8')
            with self.assertRaises(ValueError):
                schema.load_verified_json(p, digest)

    def test_valid_three_suite_denominators_pass(self):
        schema.validate_suite_denominators('tisimi_six', {
            'denominator': 6,
            'counts': {
                'mplug': {
                    'clean': {'correct': 6, 'wrong': 0, 'unresolved': 0},
                    'random4': {'correct': 6, 'wrong': 0, 'unresolved': 0},
                    'pgd4': {'correct': 5, 'wrong': 0, 'unresolved': 1},
                    'tisimi4': {'correct': 0, 'wrong': 6, 'unresolved': 0},
                    'tisimi8': {'correct': 0, 'wrong': 6, 'unresolved': 0},
                    'random8': {'correct': 6, 'wrong': 0, 'unresolved': 0},
                },
                'internvl2b': {
                    'clean': {'correct': 6, 'wrong': 0, 'unresolved': 0},
                    'random4': {'correct': 6, 'wrong': 0, 'unresolved': 0},
                    'pgd4': {'correct': 6, 'wrong': 0, 'unresolved': 0},
                    'tisimi4': {'correct': 6, 'wrong': 0, 'unresolved': 0},
                    'tisimi8': {'correct': 5, 'wrong': 1, 'unresolved': 0},
                    'random8': {'correct': 6, 'wrong': 0, 'unresolved': 0},
                },
            },
        })
        schema.validate_suite_denominators('hardlabel_six', {
            'denominator': 6,
            'summary': {'planned': 6, 'passed': 6, 'final_wrong': 6, 'ctrl_wrong': 4, 'final_wrong_and_ctrl_correct': 2},
        })
        schema.validate_suite_denominators('qwen2vl_sar', {
            'baseline': {
                'denominator': 6,
                'zeroshot_counts': {'correct': 3, 'wrong': 3, 'unresolved': 0},
                'domain_prior_counts': {'correct': 0, 'wrong': 1, 'unresolved': 5},
            },
            'injection': {
                'summary': {
                    'denominator': 3,
                    'blank_correct': 1, 'benign_correct': 1, 'attack_correct': 0,
                    'benign_review': 0, 'attack_review': 0, 'attack_ocr_exact': 2,
                    'benign_ocr_contains': 1, 'user_positive_review': 0,
                },
            },
        })

    def test_inconsistent_denominators_rejected(self):
        with self.assertRaises(ValueError):
            schema.validate_suite_denominators('qwen2vl_sar', {
                'baseline': {
                    'denominator': 6,
                    'zeroshot_counts': {'correct': 3, 'wrong': 2, 'unresolved': 0},
                    'domain_prior_counts': {'correct': 0, 'wrong': 1, 'unresolved': 5},
                },
                'injection': {'summary': {'denominator': 3}},
            })
        with self.assertRaises(KeyError):
            schema.validate_suite_denominators('unknown_suite', {})


if __name__ == '__main__':
    unittest.main()
