"""Manual adjudication of the exact existing responses, independent of references.

This audit is not a general natural-language parser. Unlisted answers remain
unresolved; future responses require a separately specified review process.
"""
import json
from collections import Counter
from pathlib import Path
from evaluate_sar_grouped_development import load_panel
from run_vlm_clean_screen import ROOT, sha256

LABELS=('ship','aircraft','car','tank','bridge','harbor')
PLAN='bc712ce9733fa633e3a6628cea93da0f3777291dd36f011a98ac0152c117f4f9'
EXACT={label:label for label in LABELS}
EXACT.update({f'it is {label}':label for label in LABELS})
EXACT.update({'飞机':'aircraft','it is aircrafts':'aircraft',
              'the image shows a ship.':'ship',
              'the middle right portion features a small harbor.':'harbor',
              'there are 2 aircrafts':'aircraft',
              'there are 2 cars: one in the bottom left portion':'car',
              'there are 2 tanks: one in the top right portion':'tank',
              'there are 6 tanks: one in the bottom right portion':'tank',
              'there are 2 harbors: one in the middle center':'harbor',
              'in  large harbor, i can see a harbor. {':'harbor'})
SOURCES={'internvl':'runs/sar-sarchat-20261108/screen.json',
         'qwen2vl':'runs/sar-sarchat-qwen2vl-20261108/screen.json',
         'deepseekvl':'runs/sar-sarchat-deepseek-20261108/screen.json',
         'mplug':'runs/sar-sarchat-mplug-20261108/screen.json'}

def main():
    _,_,_,rows,_,_,_,_=load_panel(PLAN)
    all_results={}
    for name,relative in SOURCES.items():
        path=ROOT/relative;j=json.loads(path.read_text(encoding='utf8'))
        if j.get('status')!='completed' or j.get('grouped_plan_sha256',PLAN)!=PLAN:raise ValueError('Unexpected run')
        preds={x['sample_id']:x for x in j['predictions']}
        if len(preds)!=60 or set(preds)!={r['sample_id'] for r in rows}:raise ValueError('Mismatch')
        results=[];classes={k:{'support':0,'correct':0,'wrong_label':0,'unresolved':0} for k in LABELS}
        for row in rows:
            item=preds[row['sample_id']];answer=item.get('answer');ref=row['reference_answer']
            label=EXACT.get(answer.strip().lower()) if item.get('status')=='ok' and isinstance(answer,str) else None
            outcome='unresolved' if label is None else ('correct' if label==ref else 'wrong_label')
            classes[ref]['support']+=1;classes[ref][outcome]+=1
            results.append({'sample_id':row['sample_id'],'answer':answer,'interpreted_label':label,
                            'reference':ref,'outcome':outcome})
        counts=Counter(x['outcome'] for x in results)
        all_results[name]={'source_sha256':sha256(path),'counts':dict(counts),'per_class':classes,'items':results}
    out={'status':'post-hoc manual adjudication of exactly observed responses; not a general parser or independent test',
         'review_rule':'Accept clear unique category, including sentences, plurals and explicit counts; multi-category, unspecified and out-of-vocabulary responses unresolved. Labels interpreted before comparing reference.',
         'script_sha256':sha256(Path(__file__)),'grouped_plan_sha256':PLAN,'models':all_results}
    target=ROOT/'resources/manifests/sar_existing_answers_semantic_audit_20261108.json'
    if target.exists():raise ValueError('Preserve prior audit')
    target.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    for name,v in all_results.items():print(name,v['counts'],v['per_class'])

if __name__=='__main__':main()
