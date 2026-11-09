"""Review actual replies in the shared natural-question development screen."""
import json
from collections import Counter
from pathlib import Path
from run_vlm_clean_screen import ROOT,sha256
from evaluate_sar_grouped_development import load_panel

PLAN='bc712ce9733fa633e3a6628cea93da0f3777291dd36f011a98ac0152c117f4f9'
PROTOCOL=ROOT/'experiments/protocols/sarchat_glm_edge_natural_screen_20261109.json'
SOURCES={'glm':'runs/sar-sarchat-glm-edge-20261109/screen.json',
         'internvl':'runs/sar-candidates-natural-20261109/internvl-screen.json',
         'mplug':'runs/sar-candidates-natural-20261109/mplug-screen.json'}
LABELS=('ship','aircraft','car','tank','bridge','harbor')
REVIEW={x:x for x in LABELS}
REVIEW.update({f'it is {x}':x for x in LABELS})
REVIEW.update({'they are harbor and tank':None,'they are ship and tank':None,'they are bridge and tank':None})

def main():
    out=ROOT/'resources/manifests/sar_candidates_natural_screen_20261109.json'
    if out.exists():raise ValueError('Preserve existing audit')
    protocol=json.loads(PROTOCOL.read_text(encoding='utf8'))
    _,_,_,rows,_,_,_,_=load_panel(PLAN);models={}
    for name,path in SOURCES.items():
        path=ROOT/path;j=json.loads(path.read_text(encoding='utf8'))
        if j['status']!='completed' or j['protocol_sha256']!=sha256(PROTOCOL) or j['grouped_plan_sha256']!=PLAN:
            raise ValueError('Protocol mismatch: '+name)
        if j['question']!=protocol['question']:raise ValueError('Question mismatch')
        preds={x['sample_id']:x for x in j['predictions']}
        if len(preds)!=60 or set(preds)!={r['sample_id'] for r in rows}:raise ValueError('Sample mismatch')
        per={x:{'support':0,'correct':0,'wrong_label':0,'unresolved':0} for x in LABELS};results=[]
        for row in rows:
            pred=preds[row['sample_id']];answer=pred.get('answer')
            key=answer.strip().lower() if isinstance(answer,str) else None
            if key is not None and key not in REVIEW:raise ValueError('Unreviewed answer: '+repr(answer))
            label=REVIEW.get(key) if pred['status']=='ok' else None
            ref=row['reference_answer'];result='unresolved' if label is None else ('correct' if label==ref else 'wrong_label')
            per[ref]['support']+=1;per[ref][result]+=1
            results.append({'sample_id':row['sample_id'],'answer':answer,'interpreted_label':label,'reference':ref,'outcome':result})
        counts=Counter(x['outcome'] for x in results)
        models[name]={'model':j['model'],'source_sha256':sha256(path),'counts':dict(counts),'total':len(rows),
                      'correct_fraction_all_planned':counts['correct']/len(rows),'per_class':per,'items':results,
                      'truncated':sum(bool(x.get('hit_token_limit')) for x in preds.values()),
                      'errors':sum(x['status']!='ok' for x in preds.values()),
                      'max_per_sample_allocated_mib':max(x.get('peak_allocated_mib',0) for x in preds.values())}
    result={'status':'exploratory development selection; assistant reviewed distinct replies, no independent review',
            'protocol_sha256':sha256(PROTOCOL),'script_sha256':sha256(Path(__file__)),
            'question':protocol['question'],'models':models,
            'limitations':['Same development panel, not independent test; possible SARChat training exposure',
                           'Each model uses native image preprocessing and templates; only question, sample and decoding budget shared',
                           'Assistant adjudicated observed answers only; not a general natural language evaluator',
                           'No new gradient or attack test']}
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    for name,v in models.items():print(name,v['counts'],'per_class',{k:x['correct'] for k,x in v['per_class'].items()})

if __name__=='__main__':main()
