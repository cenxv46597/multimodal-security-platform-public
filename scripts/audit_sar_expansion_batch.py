import sys,json,pathlib,hashlib,zlib,collections,io
import numpy as np
from PIL import Image
sys.path.insert(0,'scripts')
from audit_sar_adaptation_near_duplicates import normalized_pixels,orient,ORIENTATIONS,compare_boxes
P=pathlib.Path; acquisition=P('resources/manifests/sar_expansion_batch_acquisition_20261112.json');b=acquisition.read_bytes();data=json.loads(b); rows=[]
for x in data['items']:
 assert x['status']!='failed';rows.append({'id':x['image_name'],'label':x['label'],'role':'new_'+x['partition'],'file':x['asset']['file'],'sha':x['asset']['sha256'],'coco':x['coco_image_id']})
for folder in ['sar_adaptation_v0','sar_clean_development_v0']:
 for ln in (P('data/processed')/folder/'records.jsonl').read_text(encoding='utf8').splitlines():
  x=json.loads(ln);rows.append({'id':x['sample_id'],'label':x['reference_answer'],'role':'old','file':x['asset']['file'],'sha':x['asset']['sha256'],'coco':x.get('coco_image_id',x.get('source_coco',{}).get('coco_image_id'))})
comp=P('.work/next-clean-screen-20261105/images/train-json-deflate.bin').read_bytes();plan=json.loads(P('resources/manifests/sar_adaptation_data_plan_20261107.json').read_text(encoding='utf8'));assert hashlib.sha256(comp).hexdigest()==plan['sources']['coco_train']['compressed_sha256'];coco=json.loads(zlib.decompress(comp,-15)); cat={x['id']:x['name'] for x in coco['categories']}; boxes=collections.defaultdict(list); catsets=collections.defaultdict(set)
for a in coco['annotations']:boxes[a['image_id']].append(a['bbox']);catsets[a['image_id']].add(cat[a['category_id']])
images=[];th=[]
for x in rows:
 data=P(x['file']).read_bytes();assert hashlib.sha256(data).hexdigest()==x['sha'];assert catsets[x['coco']]=={x['label']}; im=Image.open(io.BytesIO(data));im.load();gray=im.convert('L');images.append(np.asarray(gray));th.append(np.asarray(gray.resize((64,64),Image.Resampling.BOX)))
base=np.stack([normalized_pixels(t) for t in th]); nnew=len(data) if False else 150; hits=[]
for rot,flip in ORIENTATIONS:
 right=np.stack([normalized_pixels(orient(t,rot,flip)) for t in th]); matrix=base[:nnew]@right.T
 for i,j in zip(*np.where(matrix>=0.80)):
  if i==j or (j<nnew and j<i):continue
  key=(int(i),int(j));score=float(matrix[i,j]);hits.append((key,score,rot,flip))
best={}
for k,s,r,f in hits:
 if k not in best or s>best[k][0]:best[k]=(s,r,f)
result=[]
for (i,j),(s,r,f) in best.items():
 x,y=rows[i],rows[j];a=images[i];z=orient(images[j],r,f);native=None;err=None
 if a.shape==z.shape:native=float(normalized_pixels(a)@normalized_pixels(z));err=compare_boxes(boxes[x['coco']],boxes[y['coco']],images[j].shape,r,f)
 result.append({'left':x['id'],'right':y['id'],'left_role':x['role'],'right_role':y['role'],'left_label':x['label'],'right_label':y['label'],'thumbnail_ncc':round(s,6),'native_ncc':round(native,6) if native is not None else None,'rotation_ccw':r*90,'reflection':f,'box_error':err,'status':'high_confidence' if native is not None and native>=0.8 and err is not None and err<=3 and x['label']==y['label'] else 'needs_review'})
result.sort(key=lambda x:-x['thumbnail_ncc']);out={'scope':'150 new vs old 240 and internal 150, whole image rotations/reflections, 64x64 ncc>=0.80; excludes crop/translation/brightness issues and scene lineage; not human label validation','acquisition_sha256':hashlib.sha256(b).hexdigest(),'new_count':150,'old_count':240,'label_consistency':'all 390 match source COCO category set','candidate_pairs':result};P('resources/manifests/sar_expansion_duplicate_audit_20261112.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf8');print('candidates',len(result),'high confidence',sum(x['status']=='high_confidence' for x in result));print(json.dumps(result[:20],ensure_ascii=False))
