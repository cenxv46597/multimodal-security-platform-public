"""Run a local Qwen2-VL SARChat probe or fixed development screen."""
import argparse
import json
import time
import traceback
from pathlib import Path
from run_vlm_clean_screen import ROOT, load_json, sha256, timed
from evaluate_sar_clean_screen import score_samples
from evaluate_sar_grouped_development import load_panel

SNAPSHOT = ROOT / 'models/SARChat-Qwen2VL-2B/master-20261108'
INVENTORY = ROOT / 'resources/manifests/sarchat_qwen2vl_inventory_20261108.json'
PLAN = 'bc712ce9733fa633e3a6628cea93da0f3777291dd36f011a98ac0152c117f4f9'
MAX_PIXELS = 256 * 28 * 28

def main():
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['probe','screen'], required=True)
    parser.add_argument('--probe-report',type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args(); output=args.output.resolve()
    if not output.is_relative_to(ROOT/'runs') or output.exists(): parser.error('Choose a new output in runs')
    output.parent.mkdir(parents=True,exist_ok=True)
    report={'status':'running','phase':args.phase,'runner_sha256':sha256(Path(__file__)),
            'inventory_sha256':sha256(INVENTORY),'predictions':[]}
    start=time.perf_counter()
    try:
        inventory=load_json(INVENTORY)
        for entry in inventory['files']:
            path=(SNAPSHOT/entry['path']).resolve()
            if not path.is_relative_to(SNAPSHOT) or path.stat().st_size != entry['size'] or sha256(path)!=entry['sha256']:
                raise ValueError('Snapshot verification failed: '+entry['path'])
        _,_,_,rows,_,_,labels,screen=load_panel(PLAN)
        if args.phase=='screen':
            if args.probe_report is None: raise ValueError('Screen requires a probe')
            probe=load_json(args.probe_report)
            if probe['status']!='passed' or any(probe[k]!=report[k] for k in ('runner_sha256','inventory_sha256')):
                raise ValueError('Probe mismatch')
            report['probe_sha256']=sha256(args.probe_report)
        torch.set_num_threads(4)
        if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
        processor=AutoProcessor.from_pretrained(SNAPSHOT, local_files_only=True, max_pixels=MAX_PIXELS, use_fast=False)
        model,info=Qwen2VLForConditionalGeneration.from_pretrained(SNAPSHOT,local_files_only=True,
            torch_dtype=torch.float16,attn_implementation='eager',output_loading_info=True)
        if any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):
            raise ValueError('Checkpoint load issue: '+str(info))
        model.eval().requires_grad_(False).to('cuda')
        report.update(model=inventory['model'], grouped_plan_sha256=PLAN, dtype='float16',
                      attention='eager', max_pixels=MAX_PIXELS, gpu=torch.cuda.get_device_name(),
                      generation=screen['generation'], prompt=rows[0]['question'])
        def infer(row):
            messages=[{'role':'user','content':[{'type':'image'},{'type':'text','text':row['question']}]}]
            prompt=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
            with Image.open(ROOT/row['asset']['file']) as image:
                rgb=image.convert('RGB'); inputs=processor(text=[prompt],images=[rgb],return_tensors='pt')
            inputs=inputs.to('cuda')
            with torch.inference_mode():
                result=model.generate(**inputs, **screen['generation'],return_dict_in_generate=True,output_logits=True)
                if not result.logits or not all(bool(torch.isfinite(x).all()) for x in result.logits):
                    raise ValueError('Nonfinite or missing logits')
            tokens=result.sequences[:,inputs.input_ids.shape[1]:]
            return {'answer':processor.batch_decode(tokens,skip_special_tokens=True,clean_up_tokenization_spaces=False)[0],
                    'generated_tokens':int(tokens.shape[1]),'generation_logits_finite':True,
                    'image_grid_thw':inputs.image_grid_thw.cpu().tolist()}
        if args.phase=='probe':
            timed(report,'probe',lambda:infer(rows[0]),'cuda'); report.update(status='passed',probe_sample_id=rows[0]['sample_id'])
        else:
            for row in rows:
                item={'sample_id':row['sample_id'],'status':'ok'}
                try:
                    timed(item,'generation',lambda:infer(row),'cuda');item.update(item.pop('generation'))
                except Exception as e: item.update(status='error',answer=None,error_type=type(e).__name__,error=str(e))
                report['predictions'].append(item)
                output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
                print('Evaluated',len(report['predictions']),'/60',flush=True)
            report['score']=score_samples(rows,report['predictions'],labels,screen);report['status']='completed'
    except Exception as e:
        report.update(status='failed',error_type=type(e).__name__,error=str(e));traceback.print_exc()
    report['seconds']=time.perf_counter()-start
    if torch.cuda.is_available(): report['peak_allocated_mib']=torch.cuda.max_memory_allocated()/2**20
    output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf8');print(report['status'],output,flush=True)
    return int(report['status']=='failed')

if __name__=='__main__': raise SystemExit(main())
