import collections,hashlib,json,pathlib,re
P=pathlib.Path;root=P(__file__).resolve().parents[1];R=lambda x:root/x;sha=lambda x:hashlib.sha256(x).hexdigest();vp=R('.work/sar-grouped-development-20261107/val.json');vbytes=vp.read_bytes();assert sha(vbytes)=='7427f3931312de2f0208e296de94914ba868e2924768cc2af4725d44ecbc7aa7';v=json.loads(vbytes);mb=R('.work/sar-grouped-development-20261107/mapping.json').read_bytes();assert sha(mb)=='e0d814f6661657679ac355f52728df94cb0eb02d00756b7a0d97ce50aa787f55';rev={v:k for k,v in json.loads(mb).items()};ib=R('.work/next-clean-screen-20261105/images/kaggle-zip-index.json').read_bytes();assert sha(ib)=='a2a225097c6481d87035fa2b29f9a5732ed460abccbabdc6a1cca7739e32cb83';index={x['name']:x for x in json.loads(ib)['entries']};rx=re.compile(r'__[0-9]+(?:\.[0-9]+)?__-?[0-9]+___-?[0-9]+$');parent=lambda x:rx.sub('',pathlib.PurePosixPath(x.replace('\\','/')).with_suffix('').as_posix());old=[]
for n in ['sar_adaptation_v0','sar_clean_development_v0']:old.extend(json.loads(x) for x in R(f'data/processed/{n}/records.jsonl').read_text(encoding='utf8').splitlines())
exclude={parent(rev[x['image_name']]) for x in old}
for n in ['sar_expansion_batch_20261112.json','sar_expansion_batch_v2_20261112.json']:exclude.update(x['source_parent'] for x in json.loads(R('experiments/protocols/'+n).read_text(encoding='utf8'))['items'])
cats=collections.defaultdict(set)
for a in v['annotations']:cats[a['image_id']].add(a['category_id'])
names={x['id']:x['name'] for x in v['categories']};seed='sar-heldout-candidates-20261112-v1';pools=collections.defaultdict(list)
for x in v['images']:
 c=cats[x['id']]
 if len(c)!=1:continue
 label=names[next(iter(c))];g=parent(rev[x['file_name']])
 if g in exclude:continue
 member='SARDet_100K/JPEGImages/val/'+x['file_name'];e=index[member];rank=sha((seed+'|'+label+'|'+x['file_name']).encode());pools[label].append((rank,g,x,e))
items=[]; used=set()
for label in ['ship','aircraft','car','tank','bridge','harbor']:
 chosen=[]
 for rank,g,x,e in sorted(pools[label],key=lambda y:y[0]):
  if g in used:continue
  used.add(g);chosen.append((rank,g,x,e))
  if len(chosen)==40:break
 assert len(chosen)==40
 for rank,g,x,e in chosen:items.append({'partition':'test_candidate','label':label,'image_name':x['file_name'],'coco_image_id':x['id'],'width':x['width'],'height':x['height'],'source_parent':g,'rank_sha256':rank,'member':e['name'],'bytes':e['uncompressed_bytes'],'crc32':e['crc32']})
out={'status':'preselected candidate images, not frozen final test','seed':seed,'selection':'per class 40 single-annotated-category val images sorted by SHA256(seed|label|filename), one per mapped source parent, exclude prior source parents; no model outputs used','limitations':'SARChat exposure unknown, annotation may be incomplete, geographic independence unknown; near-duplicate screening and manual label verification required; no model use for development','val_sha256':sha(vbytes),'mapping_sha256':sha(mb),'index_sha256':sha(ib),'items':items};R('experiments/protocols/sar_heldout_candidates_20261112.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf8');print('candidates',len(items),'bytes',sum(x['bytes'] for x in items))
