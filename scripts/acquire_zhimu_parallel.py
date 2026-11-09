"""Resume Zhimu shard downloads with verified, tracked parallel byte ranges."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time
import requests

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / '.work/zhimu-20261109/checkpoint'
INV = ROOT / '.work/zhimu-20261109/metadata/inventory.json'
REV = '35efd75d897b94e5971f523934efd72663106ce9'
CHUNK = 16*1024*1024

def main():
    p=argparse.ArgumentParser();p.add_argument('--shard',required=True,choices=['1','2','3','4']);a=p.parse_args()
    name=f'model-0000{a.shard}-of-00004.safetensors'
    meta=json.loads(INV.read_text(encoding='utf8'))
    if meta['revision']!=REV:raise ValueError('Unexpected revision')
    info=next(x for x in meta['files'] if x['rfilename']==name)
    size=info['size'];expected=info['lfs']['sha256'];dest=DEST/name;part=DEST/(name+'.part');statepath=DEST/(name+'.ranges.json')
    if dest.exists():
        if dest.stat().st_size!=size:raise ValueError('Existing size mismatch')
        print('already acquired',name,flush=True);return
    if statepath.exists():
        state=json.loads(statepath.read_text(encoding='utf8'))
        if state['size']!=size or state['sha256']!=expected or state['chunk']!=CHUNK:raise ValueError('Range metadata mismatch')
    else:
        prefix=part.stat().st_size if part.exists() else 0
        if prefix>size:raise ValueError('Invalid prefix')
        state={'size':size,'sha256':expected,'chunk':CHUNK,'prefix':prefix,'completed':[]}
        statepath.write_text(json.dumps(state),encoding='utf8')
    prefix=state['prefix']; completed=set(state['completed']);part.touch(exist_ok=True)
    todo=[(i,start,min(size-1,start+CHUNK-1)) for i,start in enumerate(range(prefix,size,CHUNK)) if i not in completed]
    def fetch(job):
        i,start,end=job
        for attempt in range(12):
            try:
                s=requests.Session();s.trust_env=False;s.proxies={'https':'http://127.0.0.1:1011'}
                with s.get(f'https://huggingface.co/AIR-CAS-kx/Zhimu-1.0/resolve/{REV}/{name}',
                           headers={'Range':f'bytes={start}-{end}'},timeout=(15,25)) as r:
                    r.raise_for_status()
                    if r.status_code!=206 or r.headers.get('Content-Range')!=f'bytes {start}-{end}/{size}' or len(r.content)!=end-start+1:
                        raise ValueError('Invalid range response')
                    return i,start,r.content
            except (requests.RequestException,ValueError):
                if attempt==11:raise
                time.sleep(min(2+attempt,8))
        raise RuntimeError('unreachable')
    with ThreadPoolExecutor(max_workers=1) as pool,part.open('r+b') as out:
        futures={pool.submit(fetch,job):job for job in todo}
        for future in as_completed(futures):
            i,start,data=future.result();out.seek(start);out.write(data);out.flush()
            completed.add(i);state['completed']=sorted(completed)
            tmp=statepath.with_suffix('.tmp');tmp.write_text(json.dumps(state),encoding='utf8');tmp.replace(statepath)
            if len(completed)%8==0:print(name,len(completed),'/',len(todo)+len(set(state['completed'])-set(i for i,_,_ in todo)),flush=True)
    digest=hashlib.sha256()
    with part.open('rb') as f:
        for data in iter(lambda:f.read(4*1024*1024),b''):digest.update(data)
    if part.stat().st_size!=size or digest.hexdigest()!=expected:raise ValueError('Final SHA256 mismatch')
    part.rename(dest);statepath.unlink();print('verified',name,flush=True)

if __name__=='__main__':main()
