import collections, hashlib, json, pathlib, re, zlib
P=pathlib.Path; root=P(__file__).resolve().parents[1]; R=lambda s:root/s
read=lambda path:json.loads(R(path).read_text(encoding='utf-8'))
sha=lambda b:hashlib.sha256(b).hexdigest()
seed='sar-expansion-20261112-v1'; plan=read('resources/manifests/sar_adaptation_data_plan_20261107.json'); raw=R('.work/next-clean-screen-20261105/images/train-json-deflate.bin').read_bytes();assert sha(raw)==plan['sources']['coco_train']['compressed_sha256'];train=json.loads(zlib.decompress(raw,-15)); vbytes=R('.work/sar-grouped-development-20261107/val.json').read_bytes();assert sha(vbytes)=='7427f3931312de2f0208e296de94914ba868e2924768cc2af4725d44ecbc7aa7';val=json.loads(vbytes)
mb=R('.work/sar-grouped-development-20261107/mapping.json').read_bytes();assert sha(mb)=='e0d814f6661657679ac355f52728df94cb0eb02d00756b7a0d97ce50aa787f55';mapping=json.loads(mb);rev={v:k for k,v in mapping.items()};assert len(rev)==len(mapping)
index_bytes=R('.work/next-clean-screen-20261105/images/kaggle-zip-index.json').read_bytes();assert sha(index_bytes)=='a2a225097c6481d87035fa2b29f9a5732ed460abccbabdc6a1cca7739e32cb83';index=json.loads(index_bytes); entries={x['name']:x for x in index['entries']}
rx=re.compile(r'__[0-9]+(?:\.[0-9]+)?__-?[0-9]+___-?[0-9]+$');parent=lambda x:rx.sub('',pathlib.PurePosixPath(x.replace('\\','/')).with_suffix('').as_posix()); old=[]
for n in ['sar_adaptation_v0','sar_clean_development_v0']:old.extend(json.loads(x) for x in R(f'data/processed/{n}/records.jsonl').read_text(encoding='utf8').splitlines())
excluded={parent(rev[x['image_name']]) for x in old}|{parent(rev[x['file_name']]) for x in val['images']}; cats=collections.defaultdict(set)
for x in train['annotations']:cats[x['image_id']].add(x['category_id'])
labels={x['id']:x['name'] for x in train['categories']}; pools=collections.defaultdict(list)
for x in train['images']:
 c=cats[x['id']]
 if len(c)!=1:continue
 g=parent(rev[x['file_name']])
 if g in excluded:continue
 member='SARDet_100K/JPEGImages/train/'+x['file_name'];e=entries[member];name=labels[next(iter(c))];rank=sha((seed+'|'+name+'|'+x['file_name']).encode());pools[name].append((rank,g,x,e))
selected=[]; used=set()
for name in ['ship','aircraft','car','tank','bridge','harbor']:
 candidates=sorted(pools[name],key=lambda z:z[0]); chosen=[]
 for rank,g,x,e in candidates:
  if g in used:continue
  used.add(g);chosen.append((rank,g,x,e))
  if len(chosen)==25:break
 assert len(chosen)==25
 for i,(rank,g,x,e) in enumerate(chosen):selected.append({'partition':'train' if i<20 else 'development','label':name,'image_name':x['file_name'],'coco_image_id':x['id'],'width':x['width'],'height':x['height'],'source_parent':g,'rank_sha256':rank,'member':e['name'],'bytes':e['uncompressed_bytes'],'crc32':e['crc32']})
out={'status':'fixed exploratory batch, not final test','seed':seed,'dataset':'greatbird/sardet-100k version 1','rule':'Single annotated category, exclude mapped parent in existing 240 and full author val, SHA256 rank by seed/category/filename, at most one image per parent, first 20 per class training then next 5 development, category order fixed. Parent identity is only a grouping clue; near-duplicate and geographic independence unresolved.','counts':{'train_per_class':20,'development_per_class':5},'val_annotation_sha256':sha(vbytes),'train_compressed_sha256':sha(raw),'mapping_sha256':sha(mb),'index_sha256':sha(index_bytes),'items':selected}
R('experiments/protocols/sar_expansion_batch_20261112.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf8');print('selected',len(selected),'bytes',sum(x['bytes'] for x in selected),'train',sum(x['partition']=='train' for x in selected),'dev',sum(x['partition']=='development' for x in selected))
