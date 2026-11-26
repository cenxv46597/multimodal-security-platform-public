"""One isolated preflight attempt with a wall-clock limit and manual stop file.

Use --simulate for CPU-only validation. A real invocation is deliberately
restricted to a previously unattempted benign list or OCR row.
"""
import argparse
import ctypes
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / 'runs/vision-safety-capability-preflight-20261114'
SOURCE = BASE / 'evaluation_protocol.json'
LIGHT_SOURCE = BASE / 'light-internvl1b-normal-20261115/protocol.json'
TEXT_SOURCE = BASE / 'text-only-account-preflight-20261115/protocol.json'
SECRET_SOURCE = BASE / 'text-secret-canary-20261115/protocol.json'
SAR_SOURCE = ROOT / 'runs/sar-category-injection-pilot-20261116/protocol.json'
RUNNER = ROOT / 'scripts/sar_category_phase_runner_20261116.py'
ALLOWED = {'account|visual_benign', 'account|risk_ocr'}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write_new(path, value):
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')


def classify_sar_result(path, sample_id):
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
        items = data.get('predictions', [])
        if data.get('status') != 'completed' or len(items) != 1 or items[0].get('sample_id') != sample_id:
            return 'invalid_or_incomplete_result'
        if items[0].get('status') != 'ok' or not isinstance(items[0].get('answer'), str):
            return 'inference_failure'
        return 'completed'
    except (OSError, ValueError, TypeError):
        return 'invalid_or_incomplete_result'


def make_job(proc):
    if os.name != 'nt':
        raise RuntimeError('process-tree control is implemented only for Windows')
    from ctypes import wintypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel.CreateJobObjectW.restype = wintypes.HANDLE
    kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel.AssignProcessToJobObject.restype = wintypes.BOOL
    kernel.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
    kernel.TerminateJobObject.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    kernel.SetInformationJobObject.restype = wintypes.BOOL
    class BasicLimit(ctypes.Structure):
        _fields_ = [('PerProcessUserTimeLimit', ctypes.c_int64), ('PerJobUserTimeLimit', ctypes.c_int64),
                    ('LimitFlags', wintypes.DWORD), ('MinimumWorkingSetSize', ctypes.c_size_t),
                    ('MaximumWorkingSetSize', ctypes.c_size_t), ('ActiveProcessLimit', wintypes.DWORD),
                    ('Affinity', ctypes.c_size_t), ('PriorityClass', wintypes.DWORD),
                    ('SchedulingClass', wintypes.DWORD)]
    class IoCounters(ctypes.Structure):
        _fields_ = [(name, ctypes.c_uint64) for name in ('ReadOperationCount', 'WriteOperationCount',
                    'OtherOperationCount', 'ReadTransferCount', 'WriteTransferCount', 'OtherTransferCount')]
    class ExtendedLimit(ctypes.Structure):
        _fields_ = [('BasicLimitInformation', BasicLimit), ('IoInfo', IoCounters),
                    ('ProcessMemoryLimit', ctypes.c_size_t), ('JobMemoryLimit', ctypes.c_size_t),
                    ('PeakProcessMemoryUsed', ctypes.c_size_t), ('PeakJobMemoryUsed', ctypes.c_size_t)]
    job = kernel.CreateJobObjectW(None, None)
    if not job:
        raise ctypes.WinError(ctypes.get_last_error())
    limits = ExtendedLimit()
    limits.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if not kernel.SetInformationJobObject(job, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
        err = ctypes.get_last_error()
        kernel.CloseHandle(job)
        raise ctypes.WinError(err)
    if not kernel.AssignProcessToJobObject(job, wintypes.HANDLE(proc._handle)):
        err = ctypes.get_last_error()
        kernel.CloseHandle(job)
        raise ctypes.WinError(err)
    return kernel, job


def stop_child(proc, job):
    if job:
        kernel, handle = job
        if not kernel.TerminateJobObject(handle, 1):
            raise ctypes.WinError(ctypes.get_last_error())
    elif proc.poll() is None:
        proc.kill()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--attempt', required=True, help='new simple directory name')
    p.add_argument('--timeout', type=float, required=True)
    p.add_argument('--sample', choices=sorted(ALLOWED))
    p.add_argument('--light', choices=['account|benign_ocr', 'account|benign_list'])
    p.add_argument('--text', action='store_true')
    p.add_argument('--secret', action='store_true')
    p.add_argument('--sar-row', help='one sample_id|condition from the prepared SAR protocol')
    p.add_argument('--prepare-only', action='store_true', help='validate and write one SAR row without launching a model')
    p.add_argument('--simulate', choices=['success', 'hang', 'failure', 'tree'])
    a = p.parse_args()
    if not a.attempt.isascii() or not a.attempt.replace('-', '').replace('_', '').isalnum():
        p.error('attempt must be a simple ASCII name')
    if not 0 < a.timeout <= 300:
        p.error('timeout must be in (0, 300] seconds')
    if sum(bool(x) for x in (a.sample, a.light, a.text, a.secret, a.sar_row, a.simulate)) != 1:
        p.error('choose exactly one run mode')
    if a.prepare_only and not a.sar_row:
        p.error('--prepare-only requires --sar-row')
    directory = BASE / a.attempt
    directory.mkdir(exist_ok=False)
    source_path = SAR_SOURCE if a.sar_row else SECRET_SOURCE if a.secret else TEXT_SOURCE if a.text else LIGHT_SOURCE if a.light else SOURCE
    source = json.loads(source_path.read_text(encoding='utf-8'))
    protocol = None
    if a.text or a.secret:
        protocol = source
        write_new(directory / 'protocol.json', protocol)
        python = ROOT / '.venv-vlm/Scripts/python.exe'
        command = [str(python), str(ROOT / '.work/text_safety_preflight_20261115.py'),
                   '--output', str(directory / 'result.json'), '--protocol', str(directory / 'protocol.json')]
    elif a.sample or a.light or a.sar_row:
        row = next(r for r in source['rows'] if r['sample_id'] == (a.sar_row or a.light or a.sample))
        image = ROOT / row['image_path']
        if digest(image) != row['image_sha256']:
            raise ValueError('image hash mismatch')
        protocol = {'question': source['question'], 'generation': source['generation'],
                    'rows': [row], 'source_protocol_sha256': digest(source_path),
                    'purpose': 'single independent attempt; not a completed 15-item protocol'}
        write_new(directory / 'protocol.json', protocol)
        python = ROOT / '.venv-vlm/Scripts/python.exe'
        command = [str(python), str(RUNNER), '--model', 'internvl1b' if a.light else 'internvl2b', '--output',
                   str(directory / 'result.json'), '--protocol', str(directory / 'protocol.json')]
    else:
        code = {'success': 'print("mock success")', 'hang': 'import time; time.sleep(60)',
                'failure': 'raise SystemExit(7)',
                'tree': 'import subprocess,sys,time; p=subprocess.Popen([sys.executable,"-c","import time; time.sleep(60)"]); print(p.pid,flush=True); time.sleep(60)'}[a.simulate]
        command = [sys.executable, '-c', code]
    if a.prepare_only:
        print('prepared_only', directory, flush=True)
        return 0
    child_env = os.environ.copy()
    if a.text or a.secret:
        child_env['HF_HOME'] = str(ROOT / '.cache/huggingface')
        child_env['HF_HUB_OFFLINE'] = '1'
        child_env['TRANSFORMERS_OFFLINE'] = '1'
    if a.light or a.text or a.secret or a.sar_row:
        extra = str(ROOT / '.venv-internvl/Lib/site-packages')
        child_env['PYTHONPATH'] = extra + (os.pathsep + child_env['PYTHONPATH'] if child_env.get('PYTHONPATH') else '')
    record = {'started_utc': now(), 'watchdog_pid': os.getpid(), 'timeout_seconds': a.timeout, 'sample': a.sar_row or a.light or a.sample,
              'simulation': a.simulate, 'source_protocol_sha256': digest(source_path),
              'extra_pythonpath': str(ROOT / '.venv-internvl/Lib/site-packages') if a.light or a.text or a.secret or a.sar_row else None,
              'watchdog_sha256': digest(Path(__file__)), 'runner_sha256': digest(RUNNER),
              'manual_stop': str(directory / 'STOP'), 'command': command}
    write_new(directory / 'started.json', record)
    start = time.monotonic()
    proc = None
    job = None
    reason = 'launch_error'
    error = None
    try:
        with (directory / 'stdout.log').open('x', encoding='utf-8') as out, (directory / 'stderr.log').open('x', encoding='utf-8') as err:
            proc = subprocess.Popen(command, cwd=ROOT, stdout=out, stderr=err, env=child_env)
            job = make_job(proc)
            while proc.poll() is None:
                if (directory / 'STOP').exists():
                    reason = 'manual_stop'
                    stop_child(proc, job)
                    break
                phase_file = directory / 'phases.jsonl'
                if phase_file.exists():
                    lines = phase_file.read_text(encoding='utf-8').splitlines()
                    if lines:
                        event = json.loads(lines[-1])
                        limits = {'imports': 90, 'inventory': 90, 'model_loading': 120, 'inference': 60, 'generation_done': 30}
                        if time.monotonic() - event['monotonic'] >= limits[event['phase']]:
                            reason = 'phase_timeout_' + event['phase']
                            stop_child(proc, job)
                            break
                if time.monotonic() - start >= a.timeout:
                    reason = 'timeout'
                    stop_child(proc, job)
                    break
                time.sleep(0.1)
            proc.wait()
            if reason not in ('manual_stop', 'timeout') and not reason.startswith('phase_timeout_'):
                reason = 'completed' if proc.returncode == 0 else 'process_failure'
                if a.sar_row and reason == 'completed':
                    reason = classify_sar_result(directory / 'result.json', a.sar_row)
    except BaseException as exc:
        if proc is not None and proc.poll() is None:
            stop_child(proc, job)
            proc.wait()
        reason = 'watchdog_interrupted' if isinstance(exc, KeyboardInterrupt) else 'watchdog_error'
        error = repr(exc)
    finally:
        if job:
            job[0].CloseHandle(job[1])
        write_new(directory / 'finished.json', {'finished_utc': now(), 'reason': reason,
                  'elapsed_seconds': round(time.monotonic() - start, 3),
                  'child_pid': proc.pid if proc else None,
                  'exit_code': proc.returncode if proc else None, 'error': error,
                  'result_exists': (directory / 'result.json').exists(),
                  'pending_exists': (directory / 'result.json.pending.json').exists()})
    print(reason, directory, flush=True)
    return 0 if reason == 'completed' else 1


if __name__ == '__main__':
    sys.exit(main())
