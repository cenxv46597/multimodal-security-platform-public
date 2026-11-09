"""Offline SARChat mPLUG-Owl3 probe and fixed grouped development screen."""
import argparse
import json
import time
import traceback
from pathlib import Path
from run_vlm_clean_screen import ROOT, load_json, sha256, timed
from evaluate_sar_clean_screen import score_samples
from evaluate_sar_grouped_development import load_panel

SNAPSHOT=ROOT/'models/SARChat-mPLUG-Owl3-1B/master-20261108'
INVENTORY=ROOT/'resources/manifests/sarchat_mplug_inventory_20261108.json'
PLAN='bc712ce9733fa633e3a6628cea93da0f3777291dd36f011a98ac0152c117f4f9'

def main():
    import torch
    from PIL import Image
    from transformers import AutoModelForCausalLM,AutoTokenizer
    p=argparse.ArgumentParser();p.add_argument('--phase',required=True,choices=['probe','screen'])
    p.add_argument('--probe-report',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();output=a.output.resolve()
    if not output.is_relative_to(ROOT/'runs') or output.exists():p.error('Choose a new output under runs')
    output.parent.mkdir(parents=True,exist_ok=True)
    report={'status':'running','phase':a.phase,'runner_sha256':sha256(Path(__file__)),
            'inventory_sha256':sha256(INVENTORY),'predictions':[]};start=time.perf_counter()
    try:
        inventory=load_json(INVENTORY)
        for e in inventory['files']:
            path=(SNAPSHOT/e['path']).resolve()
            if not path.is_relative_to(SNAPSHOT) or path.stat().st_size!=e['size'] or sha256(path)!=e['sha256']:
                raise ValueError('Snapshot verification failed: '+e['path'])
        _,_,_,rows,_,_,labels,screen=load_panel(PLAN)
        if a.phase=='screen':
            if a.probe_report is None:raise ValueError('Screen requires probe')
            probe=load_json(a.probe_report)
            if probe['status']!='passed' or any(probe[k]!=report[k] for k in ('runner_sha256','inventory_sha256')):
                raise ValueError('Probe mismatch')
            report['probe_sha256']=sha256(a.probe_report)
        torch.set_num_threads(4)
        if not torch.cuda.is_available():raise RuntimeError('CUDA required')
        tokenizer=AutoTokenizer.from_pretrained(SNAPSHOT,local_files_only=True,trust_remote_code=False)
        model,info=AutoModelForCausalLM.from_pretrained(SNAPSHOT,local_files_only=True,trust_remote_code=True,
            use_safetensors=True,torch_dtype=torch.float16,attn_implementation='sdpa',output_loading_info=True)
        if any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):
            raise ValueError('Checkpoint load issue: '+str(info))
        model.eval().requires_grad_(False).to('cuda');processor=model.init_processor(tokenizer)
        report.update(model=inventory['model'],grouped_plan_sha256=PLAN,dtype='float16',gpu=torch.cuda.get_device_name(),
                      system='You are a helpful assistant.',attention='sdpa',cut_enable=True,image_size=384,
                      generation=screen['generation'],prompt=rows[0]['question'])
        def infer(row):
            messages=[{'role':'system','content':'You are a helpful assistant.'},
                      {'role':'user','content':'<|image|>\n'+row['question']}, {'role':'assistant','content':''}]
            with Image.open(ROOT/row['asset']['file']) as image:
                inputs=processor(messages,images=[image.convert('RGB')],videos=None,cut_enable=True).to('cuda')
            if inputs.pixel_values.shape[0]<1 or not torch.isfinite(inputs.pixel_values).all():
                raise ValueError('Unexpected image input')
            with torch.inference_mode():
                embeds=model.forward_image(inputs.pixel_values)
                if not bool(torch.isfinite(embeds).all()):raise ValueError('Nonfinite image embeddings')
                terminators=[tokenizer.convert_tokens_to_ids(x) for x in model.terminators]
                result=model.language_model.generate(input_ids=inputs.input_ids,image_embeds=embeds,
                    media_offset=inputs.media_offset,attention_mask=inputs.get('attention_mask'),
                    pad_token_id=0,eos_token_id=terminators, **screen['generation'],
                    return_dict_in_generate=True,output_logits=True)
                if not result.logits or not all(bool(torch.isfinite(x).all()) for x in result.logits):
                    raise ValueError('Nonfinite or missing logits')
            tokens=result.sequences[:,inputs.input_ids.shape[1]:]
            answer=tokenizer.decode(tokens[0].cpu().tolist(),skip_special_tokens=True).strip()
            return {'answer':answer,'generated_tokens':int(tokens.shape[1]),'generation_logits_finite':True,
                    'pixel_shape':list(inputs.pixel_values.shape)}
        if a.phase=='probe':
            timed(report,'probe',lambda:infer(rows[0]),'cuda');report.update(status='passed',probe_sample_id=rows[0]['sample_id'])
        else:
            for row in rows:
                item={'sample_id':row['sample_id'],'status':'ok'}
                try:timed(item,'generation',lambda:infer(row),'cuda');item.update(item.pop('generation'))
                except Exception as e:item.update(status='error',answer=None,error_type=type(e).__name__,error=str(e))
                report['predictions'].append(item);output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
                print('Evaluated',len(report['predictions']),'/60',flush=True)
            report['score']=score_samples(rows,report['predictions'],labels,screen);report['status']='completed'
    except Exception as e:
        report.update(status='failed',error_type=type(e).__name__,error=str(e));traceback.print_exc()
    report['seconds']=time.perf_counter()-start
    if torch.cuda.is_available():report['peak_allocated_mib']=torch.cuda.max_memory_allocated()/2**20
    output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf8');print(report['status'],output,flush=True)
    return int(report['status']=='failed')

if __name__=='__main__':raise SystemExit(main())
