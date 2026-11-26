"""Local SAR reevaluation of explicitly selected, existing development images."""
import hashlib
import json
import os
import re
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNTIME = ROOT / '.work/platform_execution_v1'
RUNS = ROOT / 'runs/vision-safety-capability-preflight-20261114'
PLAN = ROOT / 'experiments/protocols/sar_development_twelve_repeat_20261113.json'
SOURCE_DIR = ROOT / 'runs/sar-twelve-repeat-20261113'
SEMANTIC = ROOT / 'runs/sar-twelve-repeat-20261113/semantic_results.json'
PYTHON = ROOT / '.venv-vlm/Scripts/python.exe'
RUNNER = ROOT / 'scripts/sar_category_phase_runner_20261116.py'
sys.path.insert(0, str(ROOT / 'scripts'))
from sar_category_phase_watchdog_20261116 import make_job, stop_child

CONDITIONS = ('clean', 'surrogate_step', 'random_control')
LOCK = threading.Lock()
STOP = {}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def db():
    RUNTIME.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(RUNTIME / 'jobs.sqlite3', timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute('CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, status TEXT NOT NULL, created REAL NOT NULL, updated REAL NOT NULL, detail TEXT NOT NULL)')
    conn.execute('CREATE TABLE IF NOT EXISTS job_results (job_id TEXT NOT NULL, sample_index INTEGER NOT NULL, condition TEXT NOT NULL, status TEXT NOT NULL, detail TEXT NOT NULL, PRIMARY KEY (job_id, sample_index, condition), FOREIGN KEY (job_id) REFERENCES jobs(id))')
    return conn


def update(job_id, status, detail):
    with db() as conn:
        conn.execute('UPDATE jobs SET status=?, updated=?, detail=? WHERE id=?', (status, time.time(), json.dumps(detail, ensure_ascii=False), job_id))


def save_result(job_id, sample_index, condition, result, detail):
    """Atomically preserve each completed or failed item and the task snapshot."""
    with db() as conn:
        conn.execute('INSERT INTO job_results VALUES (?,?,?,?,?)',
                     (job_id, sample_index, condition, result['status'], json.dumps(result, ensure_ascii=False)))
        conn.execute('UPDATE jobs SET status=?, updated=?, detail=? WHERE id=?',
                     ('running', time.time(), json.dumps(detail, ensure_ascii=False), job_id))


def progress_summary(indices, results):
    """Describe planned, completed and uncompleted calls without scoring failures."""
    conditions = {}
    for condition in CONDITIONS:
        rows = [r for r in results if r.get('condition') == condition]
        completed = [r for r in rows if r.get('status') == 'completed']
        conditions[condition] = {
            'planned': len(indices), 'completed': len(completed),
            'failed_or_interrupted': len(rows) - len(completed),
            'no_result_record': len(indices) - len(rows),
            **{k: sum(r.get('judgment') == k for r in completed) for k in ('correct', 'wrong', 'unresolved')},
        }
    paired = {'both_wrong': 0, 'attack_only_wrong': 0, 'random_only_wrong': 0,
              'both_correct': 0, 'unresolved': 0, 'incomplete': 0}
    for index in indices:
        by_condition = {r['condition']: r for r in results if r.get('sample_index') == index}
        a, b = (by_condition.get(c) for c in ('surrogate_step', 'random_control'))
        if not a or not b or a.get('status') != 'completed' or b.get('status') != 'completed':
            paired['incomplete'] += 1
            continue
        attack, random = a.get('judgment'), b.get('judgment')
        key = ('unresolved' if attack not in ('correct', 'wrong') or random not in ('correct', 'wrong') else
               'both_wrong' if attack == random == 'wrong' else
               'attack_only_wrong' if attack == 'wrong' else
               'random_only_wrong' if random == 'wrong' else 'both_correct')
        paired[key] += 1
    return {'planned_calls': len(indices) * len(CONDITIONS), 'conditions': conditions, 'paired': paired}


def get(job_id):
    if not re.fullmatch(r'[0-9a-f]{32}', job_id or ''):
        return None
    with db() as conn:
        row = conn.execute('SELECT * FROM jobs WHERE id=?', (job_id,)).fetchone()
    if not row:
        return None
    job = {'id': row['id'], 'status': row['status'], 'created': row['created'], 'updated': row['updated'], **json.loads(row['detail'])}
    # Legacy snapshots lack sample indices and remain unchanged.
    if 'planned_calls' in job and 'sample_indices' in job:
        job['progress_summary'] = progress_summary(job['sample_indices'], job.get('results', []))
    condition = job.get('active_condition')
    if job['status'] == 'running' and condition in CONDITIONS:
        base = RUNS / ('platform-reeval-' + job_id)
        index = job.get('active_sample_index')
        phase_path = (base / f'sample-{index:02d}' / condition if isinstance(index, int) else base / condition) / 'phases.jsonl'
        try:
            lines = phase_path.read_text(encoding='utf-8').splitlines()
            event = json.loads(lines[-1]) if lines else None
            allowed = {'imports', 'inventory', 'model_loading', 'inference', 'generation_done'}
            if event and event.get('phase') in allowed and isinstance(event.get('monotonic'), (int, float)):
                job['phase'] = event['phase']
                job['phase_elapsed_seconds'] = max(0, int(time.monotonic() - event['monotonic']))
        except (OSError, ValueError, TypeError):
            pass
    return job


def latest():
    with db() as conn:
        row = conn.execute('SELECT id FROM jobs ORDER BY created DESC LIMIT 1').fetchone()
    return get(row['id']) if row else None


def recover():
    with db() as conn:
        rows = conn.execute("SELECT id, detail FROM jobs WHERE status IN ('queued','running')").fetchall()
        for row in rows:
            detail = json.loads(row['detail'])
            detail['message'] = '服务重启，中断的任务不会自动恢复。'
            conn.execute('UPDATE jobs SET status=?, updated=?, detail=? WHERE id=?', ('interrupted', time.time(), json.dumps(detail, ensure_ascii=False), row['id']))


def inputs(index=0):
    if type(index) is not int or not 0 <= index < 12:
        raise ValueError('开发样本索引必须为0至11')
    plan = json.loads(PLAN.read_text(encoding='utf-8'))
    source = json.loads((SOURCE_DIR / f'source_{index}.json').read_text(encoding='utf-8'))
    semantic = json.loads(SEMANTIC.read_text(encoding='utf-8'))
    row = plan['rows'][index]
    if source['status'] != 'passed' or source['sample_id'] != row['sample_id'] or semantic['items'][index]['id'] != row['sample_id']:
        raise ValueError('历史样本记录不匹配')
    assets = {'clean': {'path': row['image_path'], 'sha256': row['image_sha256']},
              **{c: source['assets'][c] for c in CONDITIONS[1:]}}
    for asset in assets.values():
        p = (ROOT / asset['path']).resolve()
        if not p.is_relative_to(ROOT) or not p.is_file() or digest(p) != asset['sha256']:
            raise ValueError('历史图像不存在或哈希不匹配')
    if not PYTHON.is_file() or not RUNNER.is_file():
        raise ValueError('本地推理环境或执行器不可用')
    return row, plan['question'], semantic['items'][index]['reference'], assets


def check_gpu():
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                            capture_output=True, text=True, timeout=10, check=True)
    values = [int(x.strip()) for x in result.stdout.splitlines() if x.strip()]
    if len(values) != 1 or values[0] != 0:
        raise ValueError('显卡未显示空闲，请关闭占用程序后再试。')
    return values[0]


def classify(answer, reference, truncated):
    if truncated:
        return 'unresolved'
    # Only unambiguous single-label short answers are automatically scored.
    match = re.fullmatch(r'\s*(?:it is\s+)?(ship|aircraft|car|tank|bridge|harbor)[.!]?\s*', answer, re.IGNORECASE)
    if not match:
        return 'unresolved'
    return 'correct' if match.group(1).lower() == reference else 'wrong'


def execute(row, question, condition, asset, directory, stop):
    directory.mkdir(parents=True, exist_ok=False)
    entry = {'sample_id': row['sample_id'] + '|' + condition, 'image_path': asset['path'],
             'image_sha256': asset['sha256'], 'question': question}
    protocol = {'question': question, 'generation': {'do_sample': False, 'max_new_tokens': 32}, 'rows': [entry]}
    protocol_path = directory / 'protocol.json'
    with protocol_path.open('x', encoding='utf-8') as f:
        json.dump(protocol, f, ensure_ascii=False, indent=2)
    result_path = directory / 'result.json'
    command = [str(PYTHON), str(RUNNER), '--model', 'internvl2b', '--output', str(result_path), '--protocol', str(protocol_path)]
    env = os.environ.copy()
    env['PYTHONPATH'] = str(ROOT / '.venv-internvl/Lib/site-packages') + os.pathsep + env.get('PYTHONPATH', '')
    env['HF_HUB_OFFLINE'] = env['TRANSFORMERS_OFFLINE'] = '1'
    start = time.monotonic()
    proc = job = None
    try:
        with (directory / 'stdout.log').open('x', encoding='utf-8') as stdout, (directory / 'stderr.log').open('x', encoding='utf-8') as stderr:
            proc = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr, env=env)
            job = make_job(proc)
            while proc.poll() is None:
                if stop.is_set() or time.monotonic() - start > 300:
                    stop_child(proc, job)
                    proc.wait()
                    return {'status': 'interrupted' if stop.is_set() else 'timeout'}
                phase_path = directory / 'phases.jsonl'
                if phase_path.exists():
                    try:
                        events = phase_path.read_text(encoding='utf-8').splitlines()
                        event = json.loads(events[-1]) if events else None
                    except (OSError, ValueError):
                        event = None
                    limits = {'imports': 90, 'inventory': 90, 'model_loading': 120, 'inference': 60, 'generation_done': 30}
                    if event and (event.get('phase') not in limits or time.monotonic() - event['monotonic'] > limits[event['phase']]):
                        stop_child(proc, job)
                        proc.wait()
                        return {'status': 'timeout', 'phase': event.get('phase')}
                time.sleep(.2)
            if proc.returncode != 0 or not result_path.exists():
                return {'status': 'failed', 'exit_code': proc.returncode}
            data = json.loads(result_path.read_text(encoding='utf-8'))
            predictions = data.get('predictions', [])
            if data.get('status') != 'completed' or len(predictions) != 1 or predictions[0].get('status') != 'ok' or predictions[0].get('sample_id') != entry['sample_id']:
                return {'status': 'failed', 'error': '结果缺失或状态不一致'}
            pred = predictions[0]
            return {'status': 'completed', 'answer': pred['answer'], 'truncated': pred['hit_token_limit'],
                    'result_path': str(result_path.relative_to(ROOT)), 'result_sha256': digest(result_path),
                    'protocol_sha256': digest(protocol_path)}
    finally:
        if proc and proc.poll() is None:
            stop_child(proc, job)
            proc.wait()
        if job:
            job[0].CloseHandle(job[1])


def work(job_id, stop, run=execute):
    existing = get(job_id)
    indices = existing.get('sample_indices', [0])
    detail = {'message': '正在核验历史输入。', 'results': [], 'planned_conditions': list(CONDITIONS),
              'sample_indices': indices, 'planned_calls': len(indices) * len(CONDITIONS),
              'scope': f'{len(indices)}张已使用开发图；本地目标；复用历史变化图，不新生成攻击。'}
    try:
        update(job_id, 'running', detail)
        samples = [(index, inputs(index)) for index in indices]
        detail['gpu_memory_used_before_mib'] = check_gpu()
        for index, (row, question, reference, assets) in samples:
            for condition in CONDITIONS:
                if stop.is_set():
                    update(job_id, 'interrupted', {**detail, 'message': '任务已停止，未完成条件保留。'})
                    return
                check_gpu()
                detail['message'] = f'正在复评第{index + 1}张：{condition}'
                detail['active_condition'] = condition
                detail['active_sample_index'] = index
                update(job_id, 'running', detail)
                directory = RUNS / ('platform-reeval-' + job_id) / f'sample-{index:02d}' / condition
                result = run(row, question, condition, assets[condition], directory, stop)
                result.update(condition=condition, sample_index=index, sample_id=row['sample_id'], reference=reference)
                if result['status'] == 'completed':
                    result['judgment'] = classify(result['answer'], reference, result['truncated'])
                next_detail = {**detail, 'results': [*detail['results'], result],
                               'active_condition': None, 'active_sample_index': None}
                save_result(job_id, index, condition, result, next_detail)
                detail = next_detail
                if result['status'] != 'completed':
                    update(job_id, 'interrupted' if result['status'] == 'interrupted' else 'failed', {**detail, 'message': '本项未完成，已停止后续调用。'})
                    return
        detail['message'] = f'{len(detail["results"])}项本地复评完成。'
        counts = {}
        for condition in CONDITIONS:
            c = {k: sum(r['condition'] == condition and r['judgment'] == k for r in detail['results']) for k in ('correct', 'wrong', 'unresolved')}
            counts[condition] = c
        detail['counts'] = counts
        paired = {'both_wrong': 0, 'attack_only_wrong': 0, 'random_only_wrong': 0, 'both_correct': 0, 'unresolved': 0}
        for index in indices:
            by_condition = {r['condition']: r['judgment'] for r in detail['results'] if r['sample_index'] == index}
            attack, random = by_condition['surrogate_step'], by_condition['random_control']
            key = ('unresolved' if 'unresolved' in (attack, random) else
                   'both_wrong' if attack == random == 'wrong' else
                   'attack_only_wrong' if attack == 'wrong' else
                   'random_only_wrong' if random == 'wrong' else 'both_correct')
            paired[key] += 1
        detail['paired_attack_random'] = paired
        detail['report'] = (f'本次对{len(indices)}张已用于开发的SAR图及既有单步变化图、随机对照分别进行本地InternVL 2B问答。'
                            + '；'.join(f"{c}：正确{counts[c]['correct']}、错误{counts[c]['wrong']}、无法判断{counts[c]['unresolved']} / {len(indices)}" for c in CONDITIONS)
                            + f"。变化图与随机对照同图比较：两者均错{paired['both_wrong']}、仅变化图错{paired['attack_only_wrong']}、仅随机图错{paired['random_only_wrong']}、两者均正确{paired['both_correct']}、至少一项无法判断{paired['unresolved']}。"
                            + '本次复用先前生成的图片，不是新的攻击生成；不能评估稳定迁移或闭源安全。自动判读仅适用于明确的简短单类别回答，其他回答记为无法判断；参考标签仍待复核。')
        update(job_id, 'completed', detail)
    except Exception as exc:
        update(job_id, 'failed', {**detail, 'message': '任务失败，保留已有记录。', 'error': str(exc)})
    finally:
        with LOCK:
            STOP.pop(job_id, None)


def create(sample_indices=None):
    if sample_indices is None:
        sample_indices = [0]
    if not isinstance(sample_indices, list) or not 1 <= len(sample_indices) <= 12 or any(type(i) is not int or not 0 <= i < 12 for i in sample_indices) or len(set(sample_indices)) != len(sample_indices):
        raise ValueError('请选择1至12个不重复的开发样本索引（0至11）。')
    with LOCK:
        with db() as conn:
            active = conn.execute("SELECT id FROM jobs WHERE status IN ('queued','running') LIMIT 1").fetchone()
            if active:
                raise ValueError('已有任务在运行，请等待或停止。')
            job_id = uuid.uuid4().hex
            now = time.time()
            conn.execute('INSERT INTO jobs VALUES (?,?,?,?,?)', (job_id, 'queued', now, now, json.dumps({'message': '任务已创建。', 'results': [], 'sample_indices': sample_indices}, ensure_ascii=False)))
        stop = threading.Event()
        STOP[job_id] = stop
        threading.Thread(target=work, args=(job_id, stop), daemon=True).start()
    return job_id


def cancel(job_id):
    with LOCK:
        stop = STOP.get(job_id)
        if not stop:
            return False
        stop.set()
        return True
