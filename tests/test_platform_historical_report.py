import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'platform_app'))
import historical_report


class HistoricalReportTests(unittest.TestCase):
    def test_existing_records_and_paired_denominators(self):
        report = historical_report.load_report()
        self.assertEqual(report['sample_count'], 12)
        self.assertEqual(report['counts']['mplug']['surrogate_step'], {'correct': 9, 'wrong': 2, 'unresolved': 1})
        self.assertEqual(report['counts']['internvl2b']['surrogate_step'], {'correct': 12, 'wrong': 0, 'unresolved': 0})
        self.assertEqual(report['paired']['mplug'], {'both_wrong': 0, 'attack_only_wrong': 2, 'random_only_wrong': 0, 'both_correct': 9, 'unresolved': 1})
        self.assertEqual(sum(report['paired']['internvl2b'].values()), 12)

    def test_inconsistent_semantic_answer_fails_closed(self):
        original = Path.read_text
        def altered(path, *args, **kwargs):
            content = original(path, *args, **kwargs)
            if path == historical_report.RUN / 'semantic_results.json':
                data = json.loads(content)
                data['items'][0]['judgments']['mplug']['clean']['answer'] = 'altered'
                return json.dumps(data)
            return content
        with patch.object(Path, 'read_text', altered):
            with self.assertRaises(ValueError):
                historical_report.load_report()

    def test_corrupt_image_hash_fails_closed(self):
        original = historical_report.sha
        def altered(path):
            return '0' * 64 if path.suffix.lower() in ('.jpg', '.png') else original(path)
        with patch.object(historical_report, 'sha', altered):
            with self.assertRaises(ValueError):
                historical_report.load_report()


if __name__ == '__main__':
    unittest.main()
