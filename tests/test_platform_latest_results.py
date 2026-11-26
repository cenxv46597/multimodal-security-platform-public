"""Verify current results, partial denominators and fail-closed evidence checks."""
import sys
import threading
import unittest
import urllib.request
from pathlib import Path
from unittest import mock
import json

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'platform_app'))
import latest_results
import server


class LatestResultsTests(unittest.TestCase):
    def test_current_counts_and_incomplete_provider(self):
        data = latest_results.load()
        self.assertEqual([(x['wrong'], x['denominator']) for x in data['local_metrics']], [(0,17),(10,17),(10,17),(9,17),(7,17)])
        groups = {x['id']: x for x in data['cloud_groups']}
        self.assertEqual((groups['qwen_fewshot6']['completed'], groups['qwen_fewshot6']['planned']), (44,44))
        self.assertEqual([(x['wrong'], x['target_hit'], x['completed']) for x in groups['qwen_fewshot6']['metrics']], [(1,1,11),(2,0,11),(5,2,11),(7,3,11)])
        self.assertEqual((groups['doubao_fewshot6']['completed'],groups['doubao_fewshot6']['planned']), (28,48))
        self.assertEqual((groups['doubao_zeroshot']['completed'],groups['doubao_zeroshot']['planned']), (0,40))
        self.assertEqual([x['agree'] for x in data['prompt_metrics']], [4,2,4])
        self.assertEqual([x['target_hit'] for x in data['engineering8']], [0,0,1,0])
        self.assertEqual([x['correct'] for x in data['cloud_clean'][0]['metrics']], [6,9,11])

    def test_changed_cloud_evidence_rejected(self):
        real = latest_results._digest
        def altered(path):
            if path.name == '01_0047270.jpg.attempt2.json' and 'qwen_fewshot6' in str(path):
                return '0' * 64
            return real(path)
        with mock.patch.object(latest_results, '_digest', side_effect=altered):
            with self.assertRaises(ValueError):
                latest_results.load()

    def test_read_only_http_endpoint(self):
        httpd=server.ThreadingHTTPServer(('127.0.0.1',0),server.Handler)
        thread=threading.Thread(target=httpd.serve_forever,daemon=True);thread.start()
        try:
            opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(f'http://127.0.0.1:{httpd.server_address[1]}/api/latest-results',timeout=10) as response:
                data=json.loads(response.read().decode('utf-8'))
                self.assertEqual(data['cloud_groups'][0]['metrics'][3]['wrong'],7)
        finally:
            httpd.shutdown();httpd.server_close()

if __name__ == '__main__': unittest.main()
