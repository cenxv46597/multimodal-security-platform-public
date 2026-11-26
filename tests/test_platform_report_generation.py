import json
import sys
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'platform_app'))
import report_generation
import report_inputs
import saved_report


class ReportGenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.examples = {name: report_inputs.load(name) for name in ('perturbation', 'injection')}

    def test_projection_keeps_metrics_and_excludes_local_evidence(self):
        for name, data in self.examples.items():
            with self.subTest(name=name):
                facts = {**data['facts'], 'raw_answer': 'PRIVATE_ANSWER'}
                request = report_generation.build_request(facts)
                text = request['messages'][1]['content']
                self.assertNotIn('PRIVATE_ANSWER', text)
                self.assertNotIn('evidence_ids', text)
                self.assertNotIn('runs/', text)
                self.assertTrue(all(row['sample'] not in text for row in data['evidence']))
                projected = json.loads(text)
                self.assertEqual(projected['statuses'], facts['statuses'])
                for key, metric in projected['metrics'].items():
                    original = facts['metrics'][key]
                    self.assertEqual(metric, {k: original[k] for k in ('label', 'numerator', 'denominator')})

    def test_local_examples_bind_to_different_evidence_and_server_numbers(self):
        for name, data in self.examples.items():
            with self.subTest(name=name):
                facts = data['facts']
                draft = json.dumps({'sections': data['example']})
                result = report_generation.validate_response(facts, draft)
                self.assertEqual(result['limitations'], facts['limitations'])
                for section in result['sections']:
                    for metric in section['metrics']:
                        self.assertEqual(metric, {'id': metric['id'], **facts['metrics'][metric['id']]})
        with self.assertRaises(ValueError):
            report_generation.validate_response(
                self.examples['perturbation']['facts'],
                json.dumps({'sections': self.examples['injection']['example']}),
            )

    def test_validator_rejects_extra_fields_invalid_shapes_and_draft_numbers(self):
        facts = self.examples['injection']['facts']
        base = self.examples['injection']['example'][0]
        for invalid in (
            'not JSON',
            '{"sections": [], "sections": []}',
            json.dumps({'sections': [42]}),
            json.dumps({'sections': [{**base, 'denominator': 80}]}),
            json.dumps({'sections': [{**base, 'analysis': '错误示例，达到 100%'}]}),
            json.dumps({'sections': [{**base, 'metric_ids': ['attack_marker', 'attack_marker']}]}),
        ):
            with self.subTest(response=invalid), self.assertRaises(ValueError):
                report_generation.validate_response(facts, invalid)

    def test_deepseek_trial_sends_only_projected_injection_facts(self):
        data = self.examples['injection']
        response = {'model': 'deepseek-flash', 'choices': [{
            'finish_reason': 'stop',
            'message': {'content': json.dumps({'sections': data['example']})},
        }]}
        with patch.object(report_generation, 'build_opener') as opener:
            send = opener.return_value.open
            send.return_value = BytesIO(json.dumps(response).encode())
            output = report_generation.deepseek_trial(data['facts'], 'mock-key')
            send.assert_called_once()

            request = send.call_args.args[0]
            payload = json.loads(request.data)
            self.assertEqual(payload['messages'], report_generation.build_request(data['facts'])['messages'])
            self.assertEqual(request.full_url, 'https://api.deepseek.com/chat/completions')
            self.assertNotIn('mock-key', json.dumps(output))
            self.assertEqual(output['review_status'], 'pending_human_review')
            with self.assertRaises(ValueError):
                report_generation.deepseek_trial(data['facts'], '')
            with self.assertRaises(ValueError):
                report_generation.deepseek_trial(self.examples['perturbation']['facts'], 'mock-key')
            send.assert_called_once()

    def test_saved_report_verifies_stored_numbers_and_adds_review_metrics(self):
        data = self.examples['injection']
        saved = report_generation.validate_response(data['facts'], json.dumps({'sections': data['example']}))
        saved.update(provider='DeepSeek', model='deepseek-flash', review_status='pending_human_review')
        with patch.object(saved_report, 'read', return_value=saved):
            output = saved_report.load(data['facts'])
            self.assertEqual(output['sections'], saved['sections'])
            self.assertEqual([(m['numerator'], m['denominator']) for m in output['review_note']['metrics']], [(5, 6)] * 3)
            saved['sections'][0]['metrics'][0]['denominator'] = 7
            with self.assertRaises(ValueError):
                saved_report.load(data['facts'])
