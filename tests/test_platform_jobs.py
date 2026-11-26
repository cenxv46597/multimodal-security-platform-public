import json
import sqlite3
import sys
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch
from http.client import HTTPConnection

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'platform_app'))
import jobs
import server


class JobTests(unittest.TestCase):
    def setUp(self):
        self.conn = sqlite3.connect(':memory:', check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute('CREATE TABLE jobs (id TEXT PRIMARY KEY, status TEXT NOT NULL, created REAL NOT NULL, updated REAL NOT NULL, detail TEXT NOT NULL)')
        self.conn.execute('CREATE TABLE job_results (job_id TEXT NOT NULL, sample_index INTEGER NOT NULL, condition TEXT NOT NULL, status TEXT NOT NULL, detail TEXT NOT NULL, PRIMARY KEY (job_id, sample_index, condition))')
        self.job_id = 'a' * 32
        self.conn.execute('INSERT INTO jobs VALUES (?,?,?,?,?)', (self.job_id, 'queued', 1, 1, '{}'))
        self.patchers = [patch.object(jobs, 'db', return_value=self.conn),
                         patch.object(jobs, 'check_gpu', return_value=0),
                         patch.object(jobs, 'inputs', return_value=({'sample_id': 'test'}, 'question', 'ship', {k: {} for k in jobs.CONDITIONS}))]
        for p in self.patchers:
            p.start()

    def tearDown(self):
        for p in reversed(self.patchers):
            p.stop()
        self.conn.close()

    def test_completed_job_retains_three_answers(self):
        def run(row, question, condition, asset, directory, stop):
            return {'status': 'completed', 'answer': 'It is ship', 'truncated': False}
        jobs.work(self.job_id, threading.Event(), run)
        job = jobs.get(self.job_id)
        self.assertEqual(job['status'], 'completed')
        self.assertEqual([r['judgment'] for r in job['results']], ['correct'] * 3)
        self.assertIn('不新生成攻击', job['scope'])
        self.assertIn('不能评估稳定迁移', job['report'])

    def test_failure_stops_later_conditions(self):
        seen = []
        def run(row, question, condition, asset, directory, stop):
            seen.append(condition)
            return {'status': 'failed'}
        jobs.work(self.job_id, threading.Event(), run)
        job = jobs.get(self.job_id)
        self.assertEqual(job['status'], 'failed')
        self.assertEqual(seen, ['clean'])
        self.assertEqual(len(job['results']), 1)

    def test_batch_items_are_persisted_and_failure_stops_remaining(self):
        self.conn.execute('UPDATE jobs SET detail=? WHERE id=?', (json.dumps({'sample_indices': [0, 1]}), self.job_id))
        seen = []
        def run(row, question, condition, asset, directory, stop):
            seen.append((directory.parent.name, condition))
            if len(seen) == 5:
                return {'status': 'failed'}
            return {'status': 'completed', 'answer': 'It is ship', 'truncated': False}
        jobs.work(self.job_id, threading.Event(), run)
        job = jobs.get(self.job_id)
        self.assertEqual(job['status'], 'failed')
        self.assertEqual((job['planned_calls'], len(job['results'])), (6, 5))
        self.assertEqual(self.conn.execute('SELECT count(*) FROM job_results').fetchone()[0], 5)
        self.assertEqual(seen[-1], ('sample-01', 'surrogate_step'))
        summary = job['progress_summary']
        self.assertEqual(summary['planned_calls'], 6)
        self.assertEqual(summary['conditions']['surrogate_step'], {'planned': 2, 'completed': 1, 'failed_or_interrupted': 1, 'no_result_record': 0, 'correct': 1, 'wrong': 0, 'unresolved': 0})
        self.assertEqual(summary['conditions']['random_control']['no_result_record'], 1)
        self.assertEqual(summary['paired']['incomplete'], 1)
        self.assertEqual(summary['paired']['both_correct'], 1)

    def test_paired_counts_keep_unresolved_separate(self):
        self.conn.execute('UPDATE jobs SET detail=? WHERE id=?', (json.dumps({'sample_indices': [0, 1]}), self.job_id))
        def run(row, question, condition, asset, directory, stop):
            sample = directory.parent.name
            answer = 'It is ship'
            if sample == 'sample-00' and condition == 'surrogate_step': answer = 'It is aircraft'
            if sample == 'sample-01' and condition == 'surrogate_step': answer = 'ship or aircraft'
            if sample == 'sample-01' and condition == 'random_control': answer = 'It is aircraft'
            return {'status': 'completed', 'answer': answer, 'truncated': False}
        jobs.work(self.job_id, threading.Event(), run)
        job = jobs.get(self.job_id)
        self.assertEqual(job['status'], 'completed')
        self.assertEqual(job['paired_attack_random'], {'both_wrong': 0, 'attack_only_wrong': 1, 'random_only_wrong': 0, 'both_correct': 0, 'unresolved': 1})
        self.assertEqual(job['progress_summary']['conditions']['surrogate_step']['unresolved'], 1)
        self.assertEqual(job['progress_summary']['paired']['incomplete'], 0)

    def test_failed_atomic_save_does_not_publish_unpersisted_result(self):
        def run(*args):
            return {'status': 'completed', 'answer': 'It is ship', 'truncated': False}
        with patch.object(jobs, 'save_result', side_effect=sqlite3.OperationalError('simulated save failure')):
            jobs.work(self.job_id, threading.Event(), run)
        job = jobs.get(self.job_id)
        self.assertEqual(job['status'], 'failed')
        self.assertEqual(job['results'], [])
        self.assertEqual(self.conn.execute('SELECT count(*) FROM job_results').fetchone()[0], 0)
        self.assertEqual(job['progress_summary']['conditions']['clean']['no_result_record'], 1)

    def test_snapshot_update_failure_rolls_back_result_row(self):
        self.conn.commit()
        self.conn.execute("CREATE TRIGGER reject_snapshot BEFORE UPDATE ON jobs BEGIN SELECT RAISE(ABORT, 'simulated failure'); END")
        self.conn.commit()
        with self.assertRaises(sqlite3.IntegrityError):
            jobs.save_result(self.job_id, 0, 'clean', {'status': 'completed'}, {'results': []})
        self.assertEqual(self.conn.execute('SELECT count(*) FROM job_results').fetchone()[0], 0)
        self.assertEqual(self.conn.execute('SELECT detail FROM jobs WHERE id=?', (self.job_id,)).fetchone()[0], '{}')

    def test_interruption_preserves_partial_counts_on_recovery(self):
        stop = threading.Event()
        def run(*args):
            stop.set()
            return {'status': 'completed', 'answer': 'It is ship', 'truncated': False}
        jobs.work(self.job_id, stop, run)
        self.conn.execute("UPDATE jobs SET status='running' WHERE id=?", (self.job_id,))
        jobs.recover()
        job = jobs.get(self.job_id)
        self.assertEqual(job['status'], 'interrupted')
        self.assertEqual(job['progress_summary']['conditions']['clean']['completed'], 1)
        self.assertEqual(job['progress_summary']['conditions']['surrogate_step']['no_result_record'], 1)
        self.assertEqual(job['progress_summary']['paired']['incomplete'], 1)
        self.assertEqual(self.conn.execute('SELECT count(*) FROM job_results').fetchone()[0], 1)

    def test_sample_indices_are_validated_before_creating_job(self):
        for values in ([], [0, 0], [12], [-1], [True], '0'):
            with self.subTest(values=values), self.assertRaises(ValueError):
                jobs.create(values)

    def test_cancel_before_execution_is_interrupted(self):
        stop = threading.Event()
        stop.set()
        jobs.work(self.job_id, stop, lambda *args: self.fail('must not call'))
        self.assertEqual(jobs.get(self.job_id)['status'], 'interrupted')

    def test_ambiguous_and_truncated_answers_are_unresolved(self):
        self.assertEqual(jobs.classify('They are ship and tank', 'ship', False), 'unresolved')
        self.assertEqual(jobs.classify('It is ship', 'ship', True), 'unresolved')
        self.assertEqual(jobs.classify('It is aircraft', 'ship', False), 'wrong')

    def test_recovery_preserves_partial_results(self):
        self.conn.execute('UPDATE jobs SET status=?, detail=? WHERE id=?', ('running', json.dumps({'results': [{'status': 'completed'}]}), self.job_id))
        jobs.recover()
        job = jobs.get(self.job_id)
        self.assertEqual(job['status'], 'interrupted')
        self.assertEqual(len(job['results']), 1)
        self.assertNotIn('progress_summary', job)

    def test_running_job_exposes_phase_without_changing_status(self):
        self.conn.execute('UPDATE jobs SET status=?, detail=? WHERE id=?', ('running', json.dumps({'active_condition': 'clean'}), self.job_id))
        with patch.object(Path, 'read_text', return_value='{"phase":"inventory","monotonic":100}\n'), patch.object(jobs.time, 'monotonic', return_value=112):
            job = jobs.get(self.job_id)
        self.assertEqual(job['phase'], 'inventory')
        self.assertEqual(job['phase_elapsed_seconds'], 12)

    def test_http_workflow_with_simulated_model(self):
        self.conn.execute('DELETE FROM jobs')
        original_work = jobs.work
        def fake_run(row, question, condition, asset, directory, stop):
            return {'status': 'completed', 'answer': 'It is ship', 'truncated': False}
        def fake_work(job_id, stop):
            original_work(job_id, stop, fake_run)
        httpd = server.ThreadingHTTPServer(('127.0.0.1', 0), server.Handler)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            with patch.object(jobs, 'work', fake_work):
                conn = HTTPConnection('127.0.0.1', httpd.server_port, timeout=3)
                headers = {'Host': '127.0.0.1:8765', 'Origin': 'http://127.0.0.1:8765', 'Content-Type': 'application/json'}
                conn.request('GET', '/api/datasets/sar81?category=ship&page=2')
                response = conn.getresponse()
                self.assertEqual(response.status, 200)
                page = json.loads(response.read())
                self.assertEqual((page['filtered_total'], len(page['items'])), (15, 3))
                conn.request('GET', page['items'][0]['image_url'])
                response = conn.getresponse()
                self.assertEqual(response.status, 200)
                self.assertTrue(response.getheader('Content-Type').startswith('image/'))
                self.assertGreater(len(response.read()), 0)
                conn.request('GET', '/api/datasets/heldout243/images/0')
                response = conn.getresponse()
                self.assertEqual(response.status, 400)
                response.read()
                selection = {'goal': 'robustness', 'method': 'reeval', 'dataset': 'sar1', 'models': 'internvl2b'}
                conn.request('POST', '/api/jobs', json.dumps({'kind': 'local_reeval', 'selection': selection, 'sample_indices': [0, 1]}), headers)
                response = conn.getresponse()
                self.assertEqual(response.status, 202)
                job_id = json.loads(response.read())['id']
                for _ in range(100):
                    job = jobs.get(job_id)
                    if job['status'] not in ('queued', 'running'):
                        break
                    time.sleep(.01)
                self.assertEqual(job['status'], 'completed')
                self.assertEqual((job['sample_indices'], len(job['results'])), ([0, 1], 6))
                self.assertEqual(self.conn.execute('SELECT count(*) FROM job_results').fetchone()[0], 6)
                conn.request('GET', '/api/jobs/' + job_id)
                response = conn.getresponse()
                self.assertEqual(json.loads(response.read())['job']['status'], 'completed')
                conn.request('POST', '/api/jobs', json.dumps({'kind': 'local_reeval', 'selection': {**selection, 'dataset': 'sar12'}}), headers)
                response = conn.getresponse()
                self.assertEqual(response.status, 400)
                response.read()
                conn.request('POST', '/api/jobs', '{}', {**headers, 'Origin': 'https://untrusted.example'})
                response = conn.getresponse()
                self.assertEqual(response.status, 403)
                response.read()
                conn.close()
        finally:
            httpd.shutdown()
            httpd.server_close()
            thread.join()

    def test_http_failure_preserves_completed_and_unrun_results(self):
        self.conn.execute('DELETE FROM jobs')
        original_work = jobs.work
        seen = []
        finished = threading.Event()
        def fake_run(row, question, condition, asset, directory, stop):
            seen.append(condition)
            if condition == 'surrogate_step':
                return {'status': 'failed'}
            return {'status': 'completed', 'answer': 'It is ship', 'truncated': False}
        def fake_work(job_id, stop):
            try:
                original_work(job_id, stop, fake_run)
            finally:
                finished.set()
        httpd = server.ThreadingHTTPServer(('127.0.0.1', 0), server.Handler)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True); thread.start()
        try:
            with patch.object(jobs, 'work', fake_work):
                conn = HTTPConnection('127.0.0.1', httpd.server_port, timeout=3)
                headers = {'Host':'127.0.0.1:8765','Origin':'http://127.0.0.1:8765','Content-Type':'application/json'}
                selection = {'goal':'robustness','method':'reeval','dataset':'sar1','models':'internvl2b'}
                conn.request('POST','/api/jobs',json.dumps({'kind':'local_reeval','selection':selection}),headers)
                response=conn.getresponse(); self.assertEqual(response.status,202); job_id=json.loads(response.read())['id']
                self.assertTrue(finished.wait(5), '模拟任务未在时限内结束')
                conn.request('GET','/api/jobs/'+job_id)
                response=conn.getresponse(); job=json.loads(response.read())['job']
                self.assertEqual(job['status'],'failed')
                self.assertEqual(seen,['clean','surrogate_step'],job)
                self.assertEqual(job['progress_summary']['conditions']['clean']['completed'],1)
                self.assertEqual(job['progress_summary']['conditions']['surrogate_step']['failed_or_interrupted'],1)
                self.assertEqual(job['progress_summary']['conditions']['random_control']['no_result_record'],1)
                self.assertNotIn('report',job)
                conn.close()
        finally:
            httpd.shutdown(); httpd.server_close(); thread.join()


if __name__ == '__main__':
    unittest.main()
