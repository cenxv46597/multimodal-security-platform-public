import sys
import unittest
import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'platform_app'))
import two_stage_report
import server


class TwoStageReportTests(unittest.TestCase):
    def test_existing_records_keep_stage_denominators_and_positive_control(self):
        report = two_stage_report.load()
        self.assertEqual(report['sample_count'], 1)
        self.assertEqual((report['status']['image_completed'], report['status']['image_planned']), (6, 6))
        self.assertEqual((report['status']['text_completed'], report['status']['text_planned']), (4, 4))
        self.assertEqual([r['answer'] for r in report['records'][:3]], ['3.', '3.', '3.'])
        self.assertEqual([r['answer'] for r in report['records'][6:]], ['3.', '3.', '3.', '9'])
        self.assertIn('不是 SAR', report['scope'])

    def test_changed_result_is_rejected(self):
        original = two_stage_report.sha
        def altered(path):
            if path.name == 'result.json' and path.parent.name == 'smol-stage2-03-attack-20261118':
                return '0' * 64
            return original(path)
        with patch.object(two_stage_report, 'sha', altered):
            with self.assertRaises(ValueError):
                two_stage_report.load()

    def test_read_only_http_route_returns_report_or_fails_closed(self):
        http = ThreadingHTTPServer(('127.0.0.1', 0), server.Handler)
        worker = threading.Thread(target=http.serve_forever, daemon=True)
        worker.start()
        url = f'http://127.0.0.1:{http.server_port}/api/two-stage-report'
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                data = json.load(response)
                self.assertEqual(len(data['records']), 10)
            with patch.object(two_stage_report, 'load', side_effect=ValueError('corrupt')):
                with self.assertRaises(urllib.error.HTTPError) as caught:
                    urllib.request.urlopen(url, timeout=5)
                self.assertEqual(caught.exception.code, 503)
        finally:
            http.shutdown()
            worker.join(timeout=5)
            http.server_close()


if __name__ == '__main__':
    unittest.main()
