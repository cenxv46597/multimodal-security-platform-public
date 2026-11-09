"""Prepare two positive annotation-format checks; not a test set or model input."""
import hashlib
import json
import subprocess
from pathlib import Path
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT/'.work/sar-port-data-20261109'
OUT=ROOT/'resources/manifests/sar_port_positive_format_pilot_20261109.json'

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for x in iter(lambda:f.read(1024*1024),b''):h.update(x)
    return h.hexdigest()

def main():
    if OUT.exists():raise ValueError('Preserve existing output')
    files=json.loads((ROOT/'resources/manifests/sar_port_oil_image_probe_20261109.json').read_text(encoding='utf8'))
    archive_record=json.loads((ROOT/'resources/manifests/sar_port_subset_20261109.json').read_text(encoding='utf8'))
    for item in archive_record['files']:
        p=WORK/item['name']
        if not p.is_file() or p.stat().st_size!=item['size'] or sha(p)!=item['sha256']:raise ValueError('Downloaded file mismatch')
    rows=[]
    for record in files['files']:
        name=record['name'];image=WORK/name;fac=WORK/'fac_all/slice_fac'/name
        if image.stat().st_size!=record['size'] or sha(image)!=record['sha256']:raise ValueError('Image mismatch')
        result=subprocess.run(['7z','e','-so',str(WORK/'slice_func.rar'),'slice_func\\'+name],
                              capture_output=True,check=True)
        if not result.stdout.startswith(b'\x89PNG\r\n\x1a\n'):raise ValueError('Unexpected mask')
        target=WORK/'func_examples'/name;target.parent.mkdir(exist_ok=True);target.write_bytes(result.stdout)
        with Image.open(image) as im, Image.open(fac) as fm, Image.open(target) as um:
            im.load();fa=np.array(fm);ua=np.array(um)
            if im.mode!='L' or im.size!=(1024,1024) or fa.shape!=(1024,1024) or ua.shape!=(1024,1024):
                raise ValueError('Unexpected image or mask format')
            ys,xs=np.where(fa==2)
            if not len(xs):raise ValueError('No oil tank pixels')
            facilities={str(int(v)):int((fa==v).sum()) for v in np.unique(fa)}
            functions={str(int(v)):int((ua==v).sum()) for v in np.unique(ua)}
        rows.append({'image_path':image.relative_to(ROOT).as_posix(),'image_sha256':record['sha256'],
                     'facility_mask_path':fac.relative_to(ROOT).as_posix(),'facility_mask_sha256':sha(fac),
                     'functional_mask_path':target.relative_to(ROOT).as_posix(),'functional_mask_sha256':sha(target),
                     'source_name_group':name.split('_slice1024_')[0],
                     'oil_pixels':len(xs),'oil_union_bbox_xyxy_exclusive':[int(xs.min()),int(ys.min()),int(xs.max())+1,int(ys.max())+1],
                     'facility_value_pixel_counts':facilities,'functional_value_pixel_counts':functions,
                     'annotation_derived_reference':'oil tank mask present; unverified by human or independent source'})
    report={'status':'two positive examples for input and annotation format only; not a test set',
            'script_sha256':sha(Path(__file__)),'dataset_doi':files['doi'],'rows':rows,
            'candidate_question':'Does this SAR image contain any oil storage tanks? Answer in natural language.',
            'limitations':['Two positives and zero verified negatives cannot evaluate accuracy or false positives',
                           'Mask labels may be incomplete; images may contain other categories',
                           'Two filename groups are not proof of independent geographic scenes or unseen training data',
                           'Union bbox is only a visualization aid, not an individual object box',
                           'No model was run and these samples are not reserved as a final test set']}
    OUT.write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    for r in rows:print(r['source_name_group'],'oil pixels',r['oil_pixels'],'facility values',r['facility_value_pixel_counts'],'bbox',r['oil_union_bbox_xyxy_exclusive'])

if __name__=='__main__':main()
