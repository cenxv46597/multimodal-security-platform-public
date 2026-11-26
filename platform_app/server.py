"""Local-only research dashboard for an existing development run."""
import json
from collections import Counter
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit
import catalog
import datasets
import historical_report
import jobs
import latest_results
import report_inputs
import recent_experiments_report
import saved_report
import two_stage_report

ROOT = Path(__file__).resolve().parent.parent
STATIC = Path(__file__).resolve().parent / 'static'
RESULT = ROOT / 'runs/sar-twelve-repeat-20261113/semantic_results.json'
BASELINE = ROOT / 'runs/sar-expansion-81-20261112/semantic_results.json'


def load_report():
    return historical_report.load_report()


def load_clean_report():
    data = json.loads(BASELINE.read_text(encoding='utf-8'))
    models = {}
    for key in ('internvl1b', 'internvl2b', 'mplug', 'glm'):
        items = data['models'][key]['items']
        counts = dict(Counter(row['judgment'] for row in items))
        per_class = {}
        for row in items:
            label = row['reference']
            per_class.setdefault(label, {'total': 0, 'correct': 0, 'wrong': 0, 'unresolved': 0})
            per_class[label]['total'] += 1
            per_class[label][row['judgment']] += 1
        models[key] = {'total': len(items), 'counts': counts, 'per_class': per_class, 'items': items}
    return {'models': models, 'scope': '81 张 SAR 开发图；训练暴露及场景独立性未知，不是独立泛化成绩。'}


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC), **kwargs)

    def do_GET(self):
        parts = urlsplit(self.path)
        if parts.path == '/api/latest-results':
            try:
                self.send_json(latest_results.load(), 200)
            except (OSError, ValueError, KeyError, TypeError):
                self.send_json({'error': '最新实验记录缺失或无法核验'}, 503)
            return
        if parts.path == '/api/two-stage-report':
            try:
                self.send_json(two_stage_report.load(), 200)
            except (OSError, ValueError, KeyError, TypeError):
                self.send_json({'error': '合成图文报告无法核验'}, 503)
            return
        if parts.path == '/api/recent-experiments':
            try:
                self.send_json(recent_experiments_report.load_all(), 200)
            except (OSError, ValueError, KeyError, TypeError):
                self.send_json({'error': '近期 SAR 实验记录无法核验'}, 503)
            return
        if parts.path.startswith('/api/recent-experiments/'):
            suite_id = parts.path.removeprefix('/api/recent-experiments/')
            try:
                self.send_json(recent_experiments_report.load_suite(suite_id), 200)
            except KeyError:
                self.send_json({'error': '实验套件不存在'}, 404)
            except (OSError, ValueError, TypeError):
                self.send_json({'error': '近期 SAR 实验记录无法核验'}, 503)
            return
        if parts.path == '/api/datasets':
            try:
                self.send_json({'datasets': [datasets.summary(key) for key in datasets.CONFIG]}, 200)
            except (OSError, ValueError, KeyError, TypeError):
                self.send_json({'error': '开发数据集资料不可用'}, 503)
            return
        if parts.path.startswith('/api/datasets/'):
            segments = parts.path.strip('/').split('/')
            try:
                if len(segments) == 5 and segments[3] == 'images':
                    if not segments[4].isascii() or not segments[4].isdigit():
                        raise ValueError('图像不存在')
                    content, content_type = datasets.image(segments[2], int(segments[4]))
                    self.send_bytes(content, 200, content_type)
                    return
                if len(segments) == 3:
                    params = parse_qs(parts.query, keep_blank_values=True)
                    result = datasets.browse(segments[2], params.get('category', [''])[0], int(params.get('page', ['1'])[0]))
                    self.send_json(result, 200)
                    return
                raise ValueError('数据集路径无效')
            except (ValueError, TypeError):
                self.send_json({'error': '数据集、类别、页码或图像无效'}, 400)
            except (OSError, KeyError):
                self.send_json({'error': '本地开发数据不可用'}, 503)
            return
        if parts.path == '/api/jobs/latest':
            self.send_json({'job': jobs.latest()}, 200)
            return
        if parts.path.startswith('/api/jobs/'):
            job = jobs.get(parts.path.removeprefix('/api/jobs/'))
            self.send_json({'job': job} if job else {'error': '任务不存在'}, 200 if job else 404)
            return
        if parts.path == '/api/clean-report':
            try:
                self.send_json(load_clean_report(), 200)
            except (OSError, ValueError, KeyError, TypeError):
                self.send_json({'error': '81 图开发报告不可用或格式不正确'}, 503)
            return
        if parts.path in ('/api/options', '/api/validate'):
            params = parse_qs(parts.query, keep_blank_values=True)
            selection = {key: params.get(key, [''])[0] for key in ('goal', 'method', 'dataset', 'models')}
            try:
                if parts.path == '/api/options':
                    result = {'options': catalog.options(params.get('stage', [''])[0], selection)}
                else:
                    result = catalog.validate(selection)
                status = 200
            except ValueError as exc:
                result, status = {'error': str(exc)}, 400
            self.send_json(result, status)
            return
        if parts.path in ('/api/report-inputs/perturbation', '/api/report-inputs/injection'):
            try:
                report_id = parts.path.rsplit('/', 1)[1]
                data = report_inputs.load(report_id)
            except (OSError, ValueError, KeyError, TypeError):
                self.send_json({'error': '历史实验报告输入无法核验'}, 503)
                return
            if report_id == 'injection':
                try:
                    data['saved_report'] = saved_report.load(data['facts'])
                except FileNotFoundError:
                    data['saved_report_notice'] = '本地尚无已保存的 DeepSeek 草稿。'
                except (OSError, ValueError, KeyError, TypeError):
                    data['saved_report_notice'] = '已保存的 DeepSeek 草稿无法通过复核，请查看本地示例。'
            self.send_json(data, 200)
            return
        if self.path == '/api/report':
            try:
                body = json.dumps(load_report(), ensure_ascii=False).encode('utf-8')
                status = 200
            except (OSError, ValueError, KeyError, TypeError):
                body = json.dumps({'error': '本地实验记录不可用或格式不正确'}, ensure_ascii=False).encode('utf-8')
                status = 503
            self.send_bytes(body, status)
            return
        super().do_GET()

    def do_POST(self):
        # Only same-origin local UI requests may launch GPU work.
        host = self.headers.get('Host', '')
        origin = self.headers.get('Origin', '')
        if host not in ('127.0.0.1:8765', 'localhost:8765') or origin != 'http://' + host or self.headers.get('Content-Type', '').split(';')[0] != 'application/json':
            self.send_json({'error': '请求来源不允许'}, 403)
            return
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if length < 2 or length > 1024:
                raise ValueError('请求长度不正确')
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError('请求格式不正确')
            if self.path == '/api/jobs' and set(payload) in ({'kind', 'selection'}, {'kind', 'selection', 'sample_indices'}) and payload['kind'] == 'local_reeval' and isinstance(payload['selection'], dict):
                if catalog.validate(payload['selection'])['mode'] != 'local_reeval':
                    raise ValueError('所选配置不能启动本地复评。')
                self.send_json({'id': jobs.create(payload.get('sample_indices'))}, 202)
            elif self.path.startswith('/api/jobs/') and self.path.endswith('/cancel') and payload == {}:
                job_id = self.path[len('/api/jobs/'):-len('/cancel')]
                ok = jobs.cancel(job_id)
                self.send_json({'ok': True} if ok else {'error': '任务不在运行'}, 200 if ok else 409)
            else:
                self.send_json({'error': '不支持的任务请求'}, 400)
        except (ValueError, json.JSONDecodeError) as exc:
            self.send_json({'error': str(exc)}, 400)

    def send_json(self, data, status):
        self.send_bytes(json.dumps(data, ensure_ascii=False).encode('utf-8'), status)

    def send_bytes(self, body, status, content_type='application/json; charset=utf-8'):
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == '__main__':
    server = ThreadingHTTPServer(('127.0.0.1', 8765), Handler)
    jobs.recover()
    print('本地平台：http://127.0.0.1:8765', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
