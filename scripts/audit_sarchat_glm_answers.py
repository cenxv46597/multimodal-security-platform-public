"""Adjudicate the observed GLM replies semantically, before reference comparison.

This exact-response review is not a general natural-language parser.
"""
import json
from collections import Counter
from pathlib import Path
from run_vlm_clean_screen import ROOT,sha256
from evaluate_sar_grouped_development import load_panel

PLAN='bc712ce9733fa633e3a6628cea93da0f3777291dd36f011a98ac0152c117f4f9'
SOURCE=ROOT/'runs/sar-sarchat-glm-edge-20261109/screen.json'
OUT=ROOT/'resources/manifests/sarchat_glm_edge_semantic_results_20261109.json'
LABELS=('ship','aircraft','car','tank','bridge','harbor')
REVIEW={x:x for x in LABELS}
REVIEW.update({f'it is {x}':x for x in LABELS})
REVIEW['they are harbor and tank']=None

def main():
    if OUT.exists():raise ValueError('Preserve existing audit')
    j=json.loads(SOURCE.read_text(encoding='utf8'))
    if j['status']!='completed' or j['grouped_plan_sha256']!=PLAN:raise ValueError('Unexpected source')
    _,_,_,rows,_,_,_,_=load_panel(PLAN)
    predictions={p['sample_id']:p for p in j['predictions']}
    if len(predictions)!=len(rows) or set(predictions)!={r['sample_id'] for r in rows}:raise ValueError('Prediction mismatch')
    results=[];classes={k:{'support':0,'correct':0,'wrong_label':0,'unresolved':0} for k in LABELS}
    for row in rows:
        pred=predictions[row['sample_id']];answer=pred.get('answer');key=answer.strip().lower() if isinstance(answer,str) else None
        if key is not None and key not in REVIEW:raise ValueError('Unreviewed answer: '+repr(answer))
        label=REVIEW.get(key) if pred['status']=='ok' else None
        ref=row['reference_answer'];outcome='unresolved' if label is None else ('correct' if label==ref else 'wrong_label')
        classes[ref]['support']+=1;classes[ref][outcome]+=1
        results.append({'sample_id':row['sample_id'],'answer':answer,'interpreted_label':label,'reference':ref,
                        'outcome':outcome,'hit_token_limit':pred.get('hit_token_limit')})
    count=Counter(x['outcome'] for x in results);total=len(rows)
    result={'status':'exploratory development screen; exact existing answers manually adjudicated by assistant, not independently reviewed',
            'source_sha256':sha256(SOURCE),'script_sha256':sha256(Path(__file__)),'protocol_sha256':j['protocol_sha256'],
            'model':j['model'],'counts':dict(count),'total':total,'accuracy_all_planned':count['correct']/total,
            'judgable_fraction':(total-count['unresolved'])/total,'per_class':classes,'items':results,
            'peak_per_sample_allocated_mib':max(p.get('peak_allocated_mib',0) for p in predictions.values()),
            'limitations':['Not independent test; training overlap and source confounding unresolved',
                           'Different prompt and token budget than previous models; ranking is not controlled',
                           'Semantic interpretations for this run reviewed by assistant only',
                           'No image-gradient or attack test']}
    OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print('counts',dict(count),'per class',classes,'peak',result['peak_per_sample_allocated_mib'])

if __name__=='__main__':main()
