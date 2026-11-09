"""GLM-Edge SAR candidate screen using natural-language answers."""
import argparse
import json
import time
import traceback
from pathlib import Path
from run_vlm_clean_screen import ROOT,load_json,sha256,timed
from evaluate_sar_grouped_development import load_panel

SNAPSHOT=ROOT/'models/SARChat-GLM-edge-v-2B/master-20261109'
INVENTORY=ROOT/'resources/manifests/sarchat_glm_edge_inventory_20261109.json'
PROTOCOL=ROOT/'experiments/protocols/sarchat_glm_edge_natural_screen_20261109.json'
PLAN='bc712ce9733fa633e3a6628cea93da0f3777291dd36f011a98ac0152c117f4f9'
QUESTION=('What target category is shown in this SAR image? Choose one of: ship, aircraft, car, '
          'oil storage tank, bridge, harbor. Answer in a clear natural sentence.')

def main():
    import torch
    from PIL import Image
    from transformers import AutoTokenizer,AutoImageProcessor,AutoModelForCausalLM
    parser=argparse.ArgumentParser();parser.add_argument('--phase',required=True,choices=['probe','screen'])
    parser.add_argument('--probe-report',type=Path);parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();out=a.output.resolve()
    if not out.is_relative_to(ROOT/'runs') or out.exists():parser.error('Choose a new output under runs')
    out.parent.mkdir(parents=True,exist_ok=True)
    r={'status':'running','phase':a.phase,'runner_sha256':sha256(Path(__file__)),
       'inventory_sha256':sha256(INVENTORY),'protocol_sha256':sha256(PROTOCOL),'predictions':[]};start=time.perf_counter()
    try:
        inventory=load_json(INVENTORY)
        protocol=load_json(PROTOCOL)
        if protocol['question']!=QUESTION or protocol['model']!=inventory['model'] or protocol['grouped_plan_sha256']!=PLAN:
            raise ValueError('Screen protocol mismatch')
        for e in inventory['files']:
            p=(SNAPSHOT/e['path']).resolve()
            if not p.is_relative_to(SNAPSHOT) or p.stat().st_size!=e['size'] or sha256(p)!=e['sha256']:
                raise ValueError('Snapshot verification failed: '+e['path'])
        _,_,_,rows,_,_,labels,_=load_panel(PLAN)
        if a.phase=='screen':
            if a.probe_report is None:raise ValueError('Probe required')
            probe=load_json(a.probe_report)
            if probe['status']!='passed' or any(probe[k]!=r[k] for k in ('runner_sha256','inventory_sha256','protocol_sha256')):
                raise ValueError('Probe mismatch')
            r['probe_sha256']=sha256(a.probe_report)
        torch.set_num_threads(4)
        if not torch.cuda.is_available():raise RuntimeError('CUDA required')
        tokenizer=AutoTokenizer.from_pretrained(SNAPSHOT,local_files_only=True,trust_remote_code=False)
        processor=AutoImageProcessor.from_pretrained(SNAPSHOT,local_files_only=True,use_fast=False)
        model,info=AutoModelForCausalLM.from_pretrained(SNAPSHOT,local_files_only=True,trust_remote_code=True,
            use_safetensors=True,torch_dtype=torch.float16,attn_implementation='sdpa',output_loading_info=True)
        if any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):
            raise ValueError('Checkpoint load issue: '+str(info))
        model.eval().requires_grad_(False).to('cuda')
        r.update(model=inventory['model'],grouped_plan_sha256=PLAN,question=QUESTION,
                 execution={'dtype':'float16','attention':'sdpa','native_image_size':672,
                            'generation':{'do_sample':False,'max_new_tokens':32},'answer_rule':'semantic manual adjudication'},
                 gpu=torch.cuda.get_device_name())
        def infer(row):
            messages=[{'role':'user','content':[{'type':'image'},{'type':'text','text':QUESTION}]}]
            inputs=tokenizer.apply_chat_template(messages,add_generation_prompt=True,return_dict=True,
                                                  tokenize=True,return_tensors='pt').to('cuda')
            if int((inputs.input_ids==model.config.boi_token_id).sum())!=578:raise ValueError('Unexpected image placeholders')
            with Image.open(ROOT/row['asset']['file']) as image:
                pixel=processor(image.convert('RGB'),return_tensors='pt')['pixel_values'].to(device='cuda',dtype=torch.float16)
            if tuple(pixel.shape)!=(1,1,1,3,672,672) or not bool(torch.isfinite(pixel).all()):raise ValueError('Unexpected image tensor')
            with torch.inference_mode():
                generated=model.generate(**inputs,pixel_values=pixel,do_sample=False,max_new_tokens=32,
                                         return_dict_in_generate=True,output_logits=True)
                if not generated.logits or not all(bool(torch.isfinite(x).all()) for x in generated.logits):
                    raise ValueError('Nonfinite or missing logits')
            tokens=generated.sequences[:,inputs.input_ids.shape[1]:]
            return {'answer':tokenizer.decode(tokens[0].cpu().tolist(),skip_special_tokens=True).strip(),
                    'generated_tokens':int(tokens.shape[1]),'hit_token_limit':int(tokens.shape[1])>=32,
                    'generation_logits_finite':True}
        if a.phase=='probe':
            timed(r,'probe',lambda:infer(rows[0]),'cuda');r.update(status='passed',probe_sample_id=rows[0]['sample_id'])
        else:
            for row in rows:
                item={'sample_id':row['sample_id'],'status':'ok'}
                try:timed(item,'generation',lambda:infer(row),'cuda');item.update(item.pop('generation'))
                except Exception as e:item.update(status='error',answer=None,error_type=type(e).__name__,error=str(e))
                r['predictions'].append(item);out.write_text(json.dumps(r,indent=2)+'\n',encoding='utf8')
                print('Evaluated',len(r['predictions']),'/60',flush=True)
            r['status']='completed'
    except Exception as e:r.update(status='failed',error_type=type(e).__name__,error=str(e));traceback.print_exc()
    r['seconds']=time.perf_counter()-start
    if torch.cuda.is_available():r['last_peak_allocated_mib']=torch.cuda.max_memory_allocated()/2**20
    out.write_text(json.dumps(r,indent=2)+'\n',encoding='utf8');print(r['status'],out,flush=True)
    return int(r['status']=='failed')

if __name__=='__main__':raise SystemExit(main())
