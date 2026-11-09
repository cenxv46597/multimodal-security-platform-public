"""One-image frozen-weight differentiability probe for verified SARChat mPLUG."""
import json
import time
import traceback
from pathlib import Path

from run_vlm_clean_screen import ROOT, load_json, sha256
from evaluate_sar_grouped_development import load_panel
from evaluate_sarchat_mplug import INVENTORY, SNAPSHOT, PLAN

def main():
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out=ROOT/'runs/sar-sarchat-mplug-20261109/gradient.json'
    if out.exists(): raise ValueError('Preserve existing output')
    out.parent.mkdir(parents=True,exist_ok=True)
    report={'status':'failed','script_sha256':sha256(Path(__file__)),
            'inventory_sha256':sha256(INVENTORY),'phase':'one-image engineering probe'}
    start=time.perf_counter()
    try:
        for e in load_json(INVENTORY)['files']:
            p=(SNAPSHOT/e['path']).resolve()
            if not p.is_relative_to(SNAPSHOT) or p.stat().st_size!=e['size'] or sha256(p)!=e['sha256']:
                raise ValueError('Snapshot verification failed: '+e['path'])
        _,_,_,rows,_,_,_,_=load_panel(PLAN)
        torch.set_num_threads(4)
        if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
        tok=AutoTokenizer.from_pretrained(SNAPSHOT,local_files_only=True,trust_remote_code=False)
        model,info=AutoModelForCausalLM.from_pretrained(SNAPSHOT,local_files_only=True,trust_remote_code=True,
            use_safetensors=True,torch_dtype=torch.float16,attn_implementation='sdpa',output_loading_info=True)
        if any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):
            raise ValueError('Checkpoint loading issue')
        model.eval().requires_grad_(False).to('cuda'); processor=model.init_processor(tok);row=rows[0]
        messages=[{'role':'system','content':'You are a helpful assistant.'},
                  {'role':'user','content':'<|image|>\n'+row['question']},{'role':'assistant','content':''}]
        with Image.open(ROOT/row['asset']['file']) as im:
            inputs=processor(messages,images=[im.convert('RGB')],videos=None,cut_enable=True).to('cuda')
        pixels=inputs.pixel_values.detach().to(torch.float16).requires_grad_(True)
        if pixels.shape[0]!=5 or not torch.isfinite(pixels).all():raise ValueError('Unexpected processed tiles')
        with torch.no_grad():
            reference=model.forward_image(pixels)
            candidate=model.vision2text_model(model.vision_model(pixels,output_hidden_states=True).hidden_states[-2])
            maxdiff=float((reference-candidate).abs().max())
        if not torch.allclose(reference,candidate,rtol=1e-3,atol=1e-3):raise ValueError('Visual feature mismatch')
        del reference,candidate
        target=tok(' ship',add_special_tokens=False,return_tensors='pt')['input_ids'].to('cuda')
        if target.numel()!=1:raise ValueError('Expected single target token')
        features=model.vision2text_model(model.vision_model(pixels,output_hidden_states=True).hidden_states[-2])
        result=model.language_model(input_ids=inputs.input_ids,image_embeds=features,media_offset=inputs.media_offset,
                                   attention_mask=inputs.get('attention_mask'),use_cache=False,return_dict=True)
        loss=F.cross_entropy(result.logits[:,-1,:].float(),target[:,0])
        grad,=torch.autograd.grad(loss,pixels)
        finite=bool(torch.isfinite(grad).all()); nonzero=int(torch.count_nonzero(grad))
        frozen=all(not p.requires_grad and p.grad is None for p in model.parameters())
        if not finite or not nonzero or not frozen or not bool(torch.isfinite(loss)):
            raise ValueError('Invalid gradient or frozen parameter check')
        report.update(status='passed',sample_id=row['sample_id'],reference=row['reference_answer'],
            target=' ship',objective='one next-token cross entropy; engineering check only',
            input='five preprocessed RGB image tiles',input_shape=list(pixels.shape),
            feature_max_abs_difference=maxdiff,loss=float(loss),gradient_finite=finite,
            gradient_nonzero=nonzero,gradient_max_abs=float(grad.abs().max()),weights_frozen=frozen)
    except Exception as e:
        report.update(error_type=type(e).__name__,error=str(e));traceback.print_exc()
    report['seconds']=time.perf_counter()-start
    if torch.cuda.is_available():report['peak_allocated_mib']=torch.cuda.max_memory_allocated()/2**20
    out.write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    print(json.dumps(report),flush=True)
    return int(report['status']!='passed')

if __name__=='__main__':raise SystemExit(main())
