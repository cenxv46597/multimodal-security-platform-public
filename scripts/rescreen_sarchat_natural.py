"""Re-evaluate two candidate VLMs with the same natural-language question as GLM."""
import argparse
import json
import time
import traceback
from pathlib import Path
from run_vlm_clean_screen import ROOT,load_json,sha256,timed
from evaluate_sar_grouped_development import load_panel
from run_sar_proxy_candidate import IMAGE_MEAN,IMAGE_STD,IMAGE_SIZE,prepare_prompt

PLAN='bc712ce9733fa633e3a6628cea93da0f3777291dd36f011a98ac0152c117f4f9'
PROTOCOL=ROOT/'experiments/protocols/sarchat_glm_edge_natural_screen_20261109.json'
MODELS={
 'internvl':('models/SARChat-InternVL2.5-1B/6c963963','resources/manifests/sarchat_snapshot_inventory_20261108.json'),
 'mplug':('models/SARChat-mPLUG-Owl3-1B/master-20261108','resources/manifests/sarchat_mplug_inventory_20261108.json')}

def main():
    import torch
    from PIL import Image
    from transformers import AutoModel,AutoModelForCausalLM,AutoTokenizer
    from torchvision.transforms import Compose,Resize,ToTensor,Normalize,InterpolationMode
    p=argparse.ArgumentParser();p.add_argument('--model',choices=MODELS,required=True)
    p.add_argument('--phase',choices=['probe','screen'],required=True);p.add_argument('--probe-report',type=Path)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve()
    if not out.is_relative_to(ROOT/'runs') or out.exists():p.error('Choose a new output under runs')
    out.parent.mkdir(parents=True,exist_ok=True)
    snap=ROOT/MODELS[a.model][0];inventory=ROOT/MODELS[a.model][1]
    report={'status':'running','phase':a.phase,'model_key':a.model,'runner_sha256':sha256(Path(__file__)),
            'inventory_sha256':sha256(inventory),'protocol_sha256':sha256(PROTOCOL),'predictions':[]};start=time.perf_counter()
    try:
        metadata=load_json(inventory);protocol=load_json(PROTOCOL)
        if protocol['grouped_plan_sha256']!=PLAN:raise ValueError('Panel mismatch')
        for entry in metadata['files']:
            f=(snap/entry['path']).resolve()
            if not f.is_relative_to(snap) or f.stat().st_size!=entry['size'] or sha256(f)!=entry['sha256']:
                raise ValueError('Snapshot mismatch: '+entry['path'])
        _,_,_,rows,_,_,_,_=load_panel(PLAN)
        if a.phase=='screen':
            if a.probe_report is None:raise ValueError('Probe required')
            probe=load_json(a.probe_report)
            if probe['status']!='passed' or any(probe[k]!=report[k] for k in ('model_key','runner_sha256','inventory_sha256','protocol_sha256')):
                raise ValueError('Probe mismatch')
            report['probe_sha256']=sha256(a.probe_report)
        torch.set_num_threads(4);tokenizer=AutoTokenizer.from_pretrained(snap,local_files_only=True,trust_remote_code=False,use_fast=False)
        if a.model=='internvl':
            model,info=AutoModel.from_pretrained(snap,local_files_only=True,trust_remote_code=True,use_safetensors=True,
                torch_dtype=torch.float16,attn_implementation='eager',use_flash_attn=False,output_loading_info=True)
        else:
            model,info=AutoModelForCausalLM.from_pretrained(snap,local_files_only=True,trust_remote_code=True,use_safetensors=True,
                torch_dtype=torch.float16,attn_implementation='sdpa',output_loading_info=True)
        if any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):raise ValueError('Checkpoint mismatch')
        model.eval().requires_grad_(False).to('cuda');question=protocol['question'];gen=protocol['generation']
        report.update(model=metadata['model'],grouped_plan_sha256=PLAN,question=question,generation=gen,
                      gpu=torch.cuda.get_device_name(),dtype='float16')
        if a.model=='internvl':
            inputs,separator,template=prepare_prompt(model,tokenizer,question)
            inputs={k:v.to('cuda') for k,v in inputs.items()}
            transform=Compose([Resize((IMAGE_SIZE,IMAGE_SIZE),interpolation=InterpolationMode.BICUBIC),
                               ToTensor(),Normalize(IMAGE_MEAN,IMAGE_STD)])
        else:processor=model.init_processor(tokenizer)
        def infer(row):
            with Image.open(ROOT/row['asset']['file']) as image:
                rgb=image.convert('RGB')
                if a.model=='internvl':pixels=transform(rgb).unsqueeze(0).to(device='cuda',dtype=torch.float16)
                else:
                    messages=[{'role':'system','content':'You are a helpful assistant.'},
                              {'role':'user','content':'<|image|>\n'+question},{'role':'assistant','content':''}]
                    inp=processor(messages,images=[rgb],videos=None,cut_enable=True).to('cuda')
            with torch.inference_mode():
                if a.model=='internvl':
                    result=model.generate(pixel_values=pixels,**inputs,**gen,eos_token_id=template['eos_token_id'],
                                          return_dict_in_generate=True,output_logits=True)
                    tokens=result.sequences
                else:
                    emb=model.forward_image(inp.pixel_values)
                    end=[tokenizer.convert_tokens_to_ids(x) for x in model.terminators]
                    result=model.language_model.generate(input_ids=inp.input_ids,image_embeds=emb,media_offset=inp.media_offset,
                        attention_mask=inp.get('attention_mask'),pad_token_id=0,eos_token_id=end,**gen,
                        return_dict_in_generate=True,output_logits=True)
                    tokens=result.sequences[:,inp.input_ids.shape[1]:]
                if not result.logits or not all(bool(torch.isfinite(v).all()) for v in result.logits):raise ValueError('Nonfinite logits')
            answer=tokenizer.decode(tokens[0].cpu().tolist(),skip_special_tokens=True).strip()
            if a.model=='internvl':answer=answer.split(separator,1)[0]
            return {'answer':answer,'generated_tokens':int(tokens.shape[1]),
                    'hit_token_limit':int(tokens.shape[1])>=gen['max_new_tokens'],'generation_logits_finite':True}
        if a.phase=='probe':
            timed(report,'probe',lambda:infer(rows[0]),'cuda');report.update(status='passed',probe_sample_id=rows[0]['sample_id'])
        else:
            for row in rows:
                item={'sample_id':row['sample_id'],'status':'ok'}
                try:timed(item,'generation',lambda:infer(row),'cuda');item.update(item.pop('generation'))
                except Exception as e:item.update(status='error',answer=None,error_type=type(e).__name__,error=str(e))
                report['predictions'].append(item);out.write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
                print('Evaluated',len(report['predictions']),'/60',flush=True)
            report['status']='completed'
    except Exception as e:report.update(status='failed',error_type=type(e).__name__,error=str(e));traceback.print_exc()
    report['seconds']=time.perf_counter()-start
    out.write_text(json.dumps(report,indent=2)+'\n',encoding='utf8');print(report['status'],out,flush=True)
    return int(report['status']=='failed')

if __name__=='__main__':raise SystemExit(main())
