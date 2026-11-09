"""Acquire the pinned author EarthDial MS checkpoint with resumable SHA256 checks."""
import hashlib
import json
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / '.work/earthdial-preflight-20261109/checkpoint'
REPO = 'akshaydudhane/EarthDial_4B_MS'
REV = '835030e175a5b0861b0e9a050db20fcc1c2e0a88'
FILES = {
    'model-00001-of-00002.safetensors': (4957484336, 'c51e63404b7e2d78f914fa999058b533782153483a9b59e99aee8011879792a6'),
    'model-00002-of-00002.safetensors': (3336478024, 'ab2a47b175f75ecd5e55339417d63c5751f45018efdd7279d501193607392d06'),
}

def main():
    DEST.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.trust_env = False
    session.proxies = {'https': 'http://127.0.0.1:1011'}
    small = ['config.json', 'generation_config.json', 'model.safetensors.index.json',
             'special_tokens_map.json', 'tokenizer.model', 'tokenizer_config.json', 'added_tokens.json']
    for name in small:
        path = DEST / name
        if not path.exists():
            response = session.get(f'https://huggingface.co/{REPO}/resolve/{REV}/{name}', timeout=60)
            response.raise_for_status()
            path.write_bytes(response.content)
    for name, (size, expected) in FILES.items():
        path = DEST / name
        part = DEST / (name + '.part')
        if not path.exists():
            offset = part.stat().st_size if part.exists() else 0
            if offset > size:
                raise ValueError('Partial file exceeds expected size')
            failures = 0
            while offset < size:
                end = min(size - 1, offset + 64 * 1024 * 1024 - 1)
                headers = {'Range': f'bytes={offset}-{end}'}
                try:
                    with session.get(f'https://huggingface.co/{REPO}/resolve/{REV}/{name}', headers=headers,
                                     stream=True, timeout=(30, 120)) as response:
                        response.raise_for_status()
                        if response.status_code != 206 or response.headers.get('Content-Range') != f'bytes {offset}-{end}/{size}':
                            raise ValueError('Server did not honor requested range')
                        with part.open('ab') as out:
                            for block in response.iter_content(1024 * 1024):
                                if block:
                                    out.write(block)
                                    offset += len(block)
                    failures = 0
                except requests.RequestException as error:
                    failures += 1
                    print('interrupted, resuming at', offset, type(error).__name__, flush=True)
                    if failures >= 8:
                        raise
                print(name, offset, '/', size, flush=True)
            digest = hashlib.sha256()
            with part.open('rb') as f:
                for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
                    digest.update(block)
            if part.stat().st_size != size or digest.hexdigest() != expected:
                raise ValueError(f'Hash or size mismatch: {name}')
            part.rename(path)
        else:
            digest = hashlib.sha256()
            with path.open('rb') as f:
                for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
                    digest.update(block)
            if path.stat().st_size != size or digest.hexdigest() != expected:
                raise ValueError(f'Existing file mismatch: {name}')
        print('verified', name, flush=True)
    (DEST / 'acquisition.json').write_text(json.dumps({'repo': REPO, 'revision': REV, 'files': FILES}, indent=2)+'\n', encoding='utf8')
    print('acquisition complete', flush=True)

if __name__ == '__main__':
    main()
