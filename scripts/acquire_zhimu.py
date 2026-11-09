"""Download pinned Zhimu files with range resume and published LFS hash checks."""
import hashlib
import json
import argparse
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / '.work/zhimu-20261109/checkpoint'
REPO = 'AIR-CAS-kx/Zhimu-1.0'
REV = '35efd75d897b94e5971f523934efd72663106ce9'
INVENTORY = ROOT / '.work/zhimu-20261109/metadata/inventory.json'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shard', choices=['1','2','3','4','small'])
    args = parser.parse_args()
    metadata = json.loads(INVENTORY.read_text(encoding='utf8'))
    if metadata['repo'] != REPO or metadata['revision'] != REV:
        raise ValueError('Inventory revision mismatch')
    DEST.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.trust_env = False
    session.proxies = {'https':'http://127.0.0.1:1011'}
    files = [x for x in metadata['files'] if '/' not in x['rfilename'] and x['rfilename'] not in ('.gitattributes',)]
    if args.shard == 'small':
        files = [x for x in files if not x['rfilename'].endswith('.safetensors')]
    elif args.shard:
        files = [x for x in files if x['rfilename'] == f'model-0000{args.shard}-of-00004.safetensors']
        if len(files) != 1:
            raise ValueError('Expected one shard')
    for entry in files:
        name = entry['rfilename']
        size = entry['size']
        expected = entry.get('lfs', {}).get('sha256')
        path = DEST / name
        part = DEST / (name + '.part')
        if not path.exists():
            offset = part.stat().st_size if part.exists() else 0
            if offset > size:
                raise ValueError('Partial file too large: '+name)
            failures = 0
            while offset < size:
                end = min(size-1, offset+8*1024*1024-1)
                try:
                    with session.get(f'https://huggingface.co/{REPO}/resolve/{REV}/{name}',
                        headers={'Range':f'bytes={offset}-{end}'}, stream=True, timeout=(15,15)) as response:
                        response.raise_for_status()
                        if response.status_code != 206 or response.headers.get('Content-Range') != f'bytes {offset}-{end}/{size}':
                            raise ValueError('Wrong range for '+name)
                        with part.open('ab') as out:
                            for block in response.iter_content(1024*1024):
                                if block:
                                    out.write(block)
                                    offset += len(block)
                    failures = 0
                except requests.RequestException as error:
                    failures += 1
                    print('resuming', name, offset, type(error).__name__, flush=True)
                    if failures >= 8:
                        raise
                print(name, offset, '/', size, flush=True)
            if size == 0:
                part.touch()
            digest = hashlib.sha256()
            with part.open('rb') as f:
                for block in iter(lambda:f.read(4*1024*1024), b''):
                    digest.update(block)
            if part.stat().st_size != size or (expected and digest.hexdigest() != expected):
                raise ValueError('Size or LFS hash mismatch: '+name)
            part.rename(path)
        elif path.stat().st_size != size:
            raise ValueError('Existing size mismatch: '+name)
        print('acquired', name, flush=True)
    print('acquisition complete', flush=True)

if __name__ == '__main__':
    main()
