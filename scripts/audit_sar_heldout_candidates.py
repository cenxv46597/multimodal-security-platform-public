import sys,json,pathlib,hashlib,zlib,collections,io
import numpy as np
from PIL import Image
sys.path.insert(0,'scripts')
from audit_sar_adaptation_near_duplicates import normalized_pixels,orient,ORIENTATIONS,compare_boxes
P=pathlib.Path; ap=P('resources/manifests/sar_heldout_candidates_acquisition_20261112.json');raw=ap.read_bytes();d=json.loads(raw);rows=[]
for x in d['items']:
 assert x['status']!='failed';rows.append({'id':x['image_name'],'label':x['label'],'role':'test_candidate','file':x['asset']['file'],'sha':x['asset']['sha256'],'key':('val',x['coco_image_id'])})
for n in ['sar_expansion_batch_acquisition_20261112.json','sar_expansion_batch_v2_acquisition_20261112.json']:
 for x in json.loads((P('resources/manifests')/n).read_text(encoding='utf8'))['items']:
  assert x['status']!='failed';rows.append({'id':x['image_name'],'label':x['label'],'role':'previous_'+x['partition'],'file':x['asset']['file'],'sha':x['asset']['sha256'],'key':('train',x['coco_image_id'])})
for folder in ['sar_adaptation_v0','sar_clean_development_v0']:
 for ln in (P('data/processed')/folder/'records.jsonl').read_text(encoding='utf8').splitlines():
  x=json.loads(ln);rows.append({'id':x['sample_id'],'label':x['reference_answer'],'role':'old_development_or_training','file':x['asset']['file'],'sha':x['asset']['sha256'],'key':('train',x.get('coco_image_id',x.get('source_coco',{}).get('coco_image_id')))})
compressed=P('.work/next-clean-screen-20261105/images/train-json-deflate.bin').read_bytes();plan=json.loads(P('resources/manifests/sar_adaptation_data_plan_20261107.json').read_text(encoding='utf8'));assert hashlib.sha256(compressed).hexdigest()==plan['sources']['coco_train']['compressed_sha256'];vb=P('.work/sar-grouped-development-20261107/val.json').read_bytes();assert hashlib.sha256(vb).hexdigest()=='7427f3931312de2f0208e296de94914ba868e2924768cc2af4725d44ecbc7aa7';sources={'train':json.loads(zlib.decompress(compressed,-15)),'val':json.loads(vb)};boxes=collections.defaultdict(list);sets=collections.defaultdict(set)
for part,coco in sources.items():
 cats={x['id']:x['name'] for x in coco['categories']}
 for x in coco['annotations']:boxes[(part,x['image_id'])].append(x['bbox']);sets[(part,x['image_id'])].add(cats[x['category_id']])
images=[];thumb=[]
for x in rows:
 data=P(x['file']).read_bytes();assert hashlib.sha256(data).hexdigest()==x['sha'];assert sets[x['key']]=={x['label']}; im=Image.open(io.BytesIO(data));im.load();gray=im.convert('L');images.append(np.asarray(gray));thumb.append(np.asarray(gray.resize((64,64),Image.Resampling.BOX)))
base=np.stack([normalized_pixels(x) for x in thumb]);n=240;best={}
for rot,flip in ORIENTATIONS:
 right=np.stack([normalized_pixels(orient(x,rot,flip)) for x in thumb]);mat=base[:n]@right.T
 for i,j in zip(*np.where(mat>=.80)):
  if i==j or (j<n and j<i):continue
  key=(int(i),int(j));s=float(mat[i,j])
  if key not in best or s>best[key][0]:best[key]=(s,rot,flip)
results=[]
for (i,j),(s,r,f) in best.items():
 x,y=rows[i],rows[j];a=images[i];z=orient(images[j],r,f);native=err=None
 if a.shape==z.shape:native=float(normalized_pixels(a)@normalized_pixels(z));err=compare_boxes(boxes[x['key']],boxes[y['key']],images[j].shape,r,f)
 results.append({'left':x['id'],'right':y['id'],'left_role':x['role'],'right_role':y['role'],'left_label':x['label'],'right_label':y['label'],'thumbnail_ncc':round(s,6),'native_ncc':round(native,6) if native is not None else None,'rotation_ccw':r*90,'reflection':f,'box_error':err,'status':'high_confidence' if native is not None and native>=.8 and err is not None and err<=3 and x['label']==y['label'] else 'needs_review'})
results.sort(key=lambda x:-x['thumbnail_ncc']);out={'scope':'240 heldout candidates vs prior 690 and each other; 64px whole image rotation/reflection threshold .8, limited scope, no model evaluation','acquisition_sha256':hashlib.sha256(raw).hexdigest(),'label_consistency':'all 930 match corresponding train or val COCO label set, not human verified','candidate_pairs':results};P('resources/manifests/sar_heldout_duplicate_audit_20261112.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf8');print('pairs',len(results),'high confidence',sum(x['status']=='high_confidence' for x in results));print('roles',dict(collections.Counter(x['right_role'] for x in results)))
