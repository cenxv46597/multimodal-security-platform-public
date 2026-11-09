"""Exploratory post-hoc diagnostic, not a replacement for the fixed strict gate."""
import argparse
import json
import re
from pathlib import Path
from evaluate_sar_clean_screen import score_samples
from evaluate_sar_grouped_development import load_panel
from run_vlm_clean_screen import ROOT, sha256

def main():
    p=argparse.ArgumentParser();p.add_argument('input',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    source=a.input.resolve();out=a.output.resolve()
    if not source.is_relative_to(ROOT/'runs') or not out.is_relative_to(ROOT/'runs') or out.exists():
        p.error('Read an existing run and choose a new output under runs')
    j=json.loads(source.read_text(encoding='utf8'))
    if j.get('status')!='completed':raise ValueError('Screen not complete')
    _,_,_,rows,_,_,labels,screen=load_panel(j['grouped_plan_sha256'])
    if score_samples(rows,j['predictions'],labels,screen)!=j['score']:raise ValueError('Original score mismatch')
    pattern=re.compile(r'it is (ship|aircraft|car|tank|bridge|harbor)',re.IGNORECASE)
    converted=[];changed=0
    for item in j['predictions']:
        item=dict(item); answer=item.get('answer');match=pattern.fullmatch(answer.strip()) if isinstance(answer,str) else None
        if match:item['answer']=match.group(1).lower();changed+=1
        converted.append(item)
    result={'source_sha256':sha256(source),'script_sha256':sha256(Path(__file__)),
            'rule':'Keep strict single labels; additionally accept exactly It is <one label>, ignoring case and outer whitespace, no punctuation.',
            'status':'post-hoc exploratory diagnostic, not original gate', 'converted':changed,
            'score':score_samples(rows,converted,labels,screen)}
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'converted':changed,'correct':result['score']['correct'],'valid_fraction':result['score']['valid_answer_fraction'],
                      'per_class':result['score']['per_class']}))

if __name__=='__main__':main()
