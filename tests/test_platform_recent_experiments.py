"""Offline unit tests for platform_app/recent_experiments_report.py and its HTTP routes."""
import json
import shutil
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'platform_app'))

import recent_experiments_report
import server


class RecentExperimentsReportTests(unittest.TestCase):
    def test_all_five_suites_verify_hashes_and_denominators(self):
        data = recent_experiments_report.load_all()
        suites = data['suites']
        self.assertEqual(
            set(suites),
            {'tisimi_six', 'hardlabel_six', 'qwen2vl_sar', 'tisimi_ablation_six', 'engineering_cohort_v1'},
        )

        tis = suites['tisimi_six']
        self.assertEqual(tis['denominator'], 6)
        self.assertEqual(tis['counts']['mplug']['tisimi4']['wrong'], 6)
        self.assertEqual(tis['counts']['mplug']['tisimi4']['target_hit'], 5)
        self.assertEqual(tis['counts']['mplug']['tisimi8']['target_hit'], 6)
        self.assertEqual(tis['counts']['mplug']['random8']['wrong'], 0)
        self.assertEqual(tis['counts']['internvl2b']['tisimi4']['wrong'], 0)
        self.assertEqual(tis['counts']['internvl2b']['tisimi8']['wrong'], 1)
        self.assertEqual(tis['counts']['internvl2b']['tisimi8']['target_hit'], 1)
        self.assertEqual(tis['counts']['internvl2b']['random8']['wrong'], 0)

        abl = suites['tisimi_ablation_six']
        self.assertEqual(abl['denominator'], 6)
        self.assertEqual(abl['summary']['mplug_full_target_hit'], 6)
        self.assertEqual(abl['summary']['mplug_ablation_target_hit'], 6)
        self.assertEqual(abl['summary']['internvl2b_full_target_hit'], 1)
        self.assertEqual(abl['summary']['internvl2b_ablation_target_hit'], 1)
        self.assertEqual(abl['summary']['internvl2b_random_sign_1_correct'], 6)
        self.assertEqual(abl['summary']['internvl2b_random_sign_2_correct'], 6)
        self.assertEqual(abl['summary']['prior_failures'], 2)

        coh = suites['engineering_cohort_v1']
        self.assertEqual(coh['denominator'], 18)
        self.assertEqual(coh['status'], 'frozen_limited_engineering_evaluation')
        self.assertEqual(coh['class_counts'], {'ship': 2, 'aircraft': 3, 'car': 2, 'tank': 5, 'bridge': 4, 'harbor': 2})
        self.assertEqual(coh['excluded_hrsid_count'], 2)
        self.assertEqual(len(coh['items']), 18)

        hl = suites['hardlabel_six']
        self.assertEqual(hl['denominator'], 6)
        self.assertEqual(hl['summary']['planned'], 6)
        self.assertEqual(hl['summary']['passed'], 6)
        self.assertEqual(hl['summary']['final_wrong'], 6)
        self.assertEqual(hl['summary']['ctrl_wrong'], 4)
        self.assertEqual(hl['summary']['final_wrong_and_ctrl_correct'], 2)

        qw = suites['qwen2vl_sar']
        self.assertEqual(qw['baseline']['denominator'], 6)
        self.assertEqual(qw['baseline']['zeroshot_counts'], {'correct': 3, 'wrong': 3, 'unresolved': 0})
        self.assertEqual(qw['baseline']['domain_prior_counts'], {'correct': 0, 'wrong': 1, 'unresolved': 5})
        inj = qw['injection']['summary']
        self.assertEqual(inj['denominator'], 3)
        self.assertEqual(inj['attack_ocr_exact'], 2)
        self.assertEqual(inj['attack_review'], 0)
        self.assertEqual(inj['user_positive_review'], 0)

    def test_tampered_file_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake_root = Path(tmp)
            for rel in (
                'experiments/protocols/sar_tisimi_transfer_six_20261119.json',
                'runs/sar-tisimi-transfer-six-20261119/semantic_results.json',
                'runs/sar-tisimi-transfer-six-20261119/mplug-eval-part1.json',
                'runs/sar-tisimi-transfer-six-20261119/mplug-eval-part2.json',
                'runs/sar-tisimi-transfer-six-20261119/internvl2b-eval-part1.json',
                'runs/sar-tisimi-transfer-six-20261119/internvl2b-eval-part2.json',
            ):
                dst = fake_root / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / rel, dst)
            # Tamper with one evaluation file
            p = fake_root / 'runs/sar-tisimi-transfer-six-20261119/mplug-eval-part1.json'
            p.write_text(p.read_text(encoding='utf-8') + ' ', encoding='utf-8')
            with self.assertRaises(ValueError):
                recent_experiments_report.load_tisimi_six(fake_root)

    def test_read_only_http_routes(self):
        httpd = server.ThreadingHTTPServer(('127.0.0.1', 0), server.Handler)
        port = httpd.server_address[1]
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            with opener.open(f'http://127.0.0.1:{port}/api/recent-experiments', timeout=5) as resp:
                self.assertEqual(resp.status, 200)
                payload = json.loads(resp.read().decode('utf-8'))
                self.assertEqual(
                    set(payload['suites']),
                    {'tisimi_six', 'hardlabel_six', 'qwen2vl_sar', 'tisimi_ablation_six', 'engineering_cohort_v1'},
                )

            with opener.open(f'http://127.0.0.1:{port}/api/recent-experiments/engineering_cohort_v1', timeout=5) as resp:
                self.assertEqual(resp.status, 200)
                payload = json.loads(resp.read().decode('utf-8'))
                self.assertEqual(payload['id'], 'engineering_cohort_v1')
                self.assertEqual(payload['denominator'], 18)

            with opener.open(f'http://127.0.0.1:{port}/api/recent-experiments/qwen2vl_sar', timeout=5) as resp:
                self.assertEqual(resp.status, 200)
                payload = json.loads(resp.read().decode('utf-8'))
                self.assertEqual(payload['id'], 'qwen2vl_sar')

            with self.assertRaises(urllib.error.HTTPError) as ctx:
                opener.open(f'http://127.0.0.1:{port}/api/recent-experiments/nonexistent', timeout=5)
            self.assertEqual(ctx.exception.code, 404)
        finally:
            httpd.shutdown()
            httpd.server_close()


if __name__ == '__main__':
    unittest.main()
