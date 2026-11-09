"""Inventory downloaded facility masks, without scoring models or selecting test data."""
import collections
import hashlib
import json
from pathlib import Path
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT/'.work/sar-port-data-20261109'
SUBSET=ROOT/'resources/manifests/sar_port_subset_20261109.json'
OUT=ROOT/'resources/manifests/sar_port_facility_audit_20261109.json'

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def main():
    if OUT.exists():raise ValueError('Preserve previous audit')
    source=json.loads(SUBSET.read_text(encoding='utf8'))
    expected=next(f for f in source['files'] if f['name']=='slice_fac.rar')
    archive=WORK/'slice_fac.rar'
    if archive.stat().st_size!=expected['size'] or digest(archive)!=expected['sha256']:
        raise ValueError('Facility archive mismatch')
    files=sorted((WORK/'fac_all/slice_fac').glob('*.png'))
    if len(files)!=5267:raise ValueError('Unexpected file count')
    label_counts=collections.Counter();groups=collections.Counter();examples={};tank_count=0
    for f in files:
        with Image.open(f) as im:
            if im.size!=(1024,1024) or im.mode!='L':raise ValueError('Unexpected mask: '+f.name)
            hist=im.histogram()
        for index,n in enumerate(hist):
            if n:label_counts[index]+=1
        if hist[2]:
            tank_count+=1;group=f.name.split('_slice1024_')[0];groups[group]+=1
            examples.setdefault(group,{'file':f.name,'mask_sha256':digest(f),'oil_mask_pixels':hist[2]})
    report={'dataset_doi':source['doi'],'version':source['version'],
            'status':'format and source-name reconnaissance; not a test split',
            'facility_mask_archive_sha256':expected['sha256'],
            'script_sha256':digest(Path(__file__)),'mask_files':len(files),
            'images_containing_each_value':dict(label_counts),'oil_tank_value':2,
            'oil_tank_mask_count':tank_count,'source_name_group_rule':'filename before _slice1024_; proxy, not verified scene identity',
            'groups_with_oil_tank':dict(groups),'one_mask_example_per_group':examples,
            'limitations':['Mask value 2 interpreted using dataset description; annotation quality unverified',
                           'Filename prefixes are not guaranteed distinct geographic sites',
                           'Potential model training exposure and scene overlap not assessed',
                           'Segmentation images can contain multiple categories; not ready for single-label classification']}
    OUT.write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    print('masks',len(files),'tank masks',tank_count,'prefix groups',len(groups))
    for k,v in groups.items():print(k,v)

if __name__=='__main__':main()
