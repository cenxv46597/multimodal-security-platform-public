import concurrent.futures,hashlib,io,json,pathlib,time,zlib
from urllib.parse import quote
import requests
from PIL import Image
root=pathlib.Path(__file__).resolve().parents[1];planpath=root/'experiments/protocols/sar_expansion_batch_20261112.json';raw=planpath.read_bytes();plan=json.loads(raw);out=root/'data/raw/sardet_expansion_20261112';out.mkdir(parents=True,exist_ok=True)
def get(item):
 name=item['image_name'];dest=out/name
 for attempt in range(1,4):
  try:
   if dest.exists():data=dest.read_bytes();status='existing'
   else:
    s=requests.Session();s.trust_env=False;url='https://www.kaggle.com/api/v1/datasets/download/greatbird/sardet-100k/'+quote(item['member'],safe='')+'?datasetVersionNumber=1';r=s.get(url,headers={'Range':f"bytes=0-{item['bytes']-1}",'Accept-Encoding':'identity'},timeout=35);r.raise_for_status();assert r.status_code==206 and r.headers.get('Content-Range')==f"bytes 0-{item['bytes']-1}/{item['bytes']}";data=r.content;status='downloaded'
   assert len(data)==item['bytes'] and f'{zlib.crc32(data)&0xffffffff:08x}'==item['crc32'];im=Image.open(io.BytesIO(data));im.load();assert im.size==(item['width'],item['height'])
   if status=='downloaded':dest.write_bytes(data)
   return {**item,'asset':{'file':dest.relative_to(root).as_posix(),'sha256':hashlib.sha256(data).hexdigest(),'pixel_sha256':hashlib.sha256(im.convert('RGB').tobytes()).hexdigest(),'format':im.format,'size':list(im.size)},'status':status}
  except Exception as e:
   error=f'{type(e).__name__}: {str(e)[:160]}'
   time.sleep(attempt)
 return {**item,'status':'failed','error':error}
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:results=list(pool.map(get,plan['items']))
report={'plan_sha256':hashlib.sha256(raw).hexdigest(),'scope':'Exploratory development data; no model run; source parent is not verified scene identity','items':results};path=root/'resources/manifests/sar_expansion_batch_acquisition_20261112.json';path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8');print('verified',sum(x['status']!='failed' for x in results),'failed',sum(x['status']=='failed' for x in results));print('failures',[(x['image_name'],x.get('error')) for x in results if x['status']=='failed'][:10])
