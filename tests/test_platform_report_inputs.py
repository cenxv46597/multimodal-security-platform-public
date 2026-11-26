import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'platform_app'))
import report_inputs


class ReportInputTests(unittest.TestCase):
    def test_historical_experiments_keep_their_own_denominators(self):
        for name, calls, measures in (
            ('perturbation', 72, {'mplug_paired': (2, 12), 'mplug_unresolved': (1, 12)}),
            ('injection', 36, {'attack_marker': (0, 6), 'positive_marker': (0, 6),
                               'attack_read': (6, 6), 'benign_read': (2, 6)}),
        ):
            with self.subTest(name=name):
                data = report_inputs.load(name)
                facts = data['facts']
                self.assertEqual(facts['planned'], calls)
                self.assertEqual(facts['statuses']['completed'], calls)
                self.assertEqual(len({r['id'] for r in data['evidence']}), calls)
                for key, expected in measures.items():
                    entry = facts['metrics'][key]
                    self.assertEqual((entry['numerator'], entry['denominator']), expected)

    def test_example_validation_rejects_bad_references_and_denominators(self):
        data = report_inputs.load('injection')
        facts = data['facts']
        with self.assertRaises(ValueError):
            report_inputs.validate_example(facts, [report_inputs.section('标题', '分析', 'missing')])
        entry = facts['metrics']['attack_marker']
        entry['evidence_ids'].append('unknown')
        with self.assertRaises(ValueError):
            report_inputs.validate_example(facts, data['example'])
        entry['evidence_ids'].pop()
        entry['numerator'] = entry['denominator'] + 1
        with self.assertRaises(ValueError):
            report_inputs.validate_example(facts, data['example'])
