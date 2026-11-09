"""Compare staged and full-batch input gradients for one verified mPLUG input."""
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
    from transformers import AutoModelForCausalLM,AutoTokenizer
    out=ROOT/'runs/sar-sarchat-mplug-20261109/staged_gradient_retry.json'
    if out.exists():raise ValueError('Preserve previous output')
    out.parent.mkdir(parents=True,exist_ok=True)
    report={'status':'failed','script_sha256':sha256(Path(__file__)),'inventory_sha256':sha256(INVENTORY)}
    start=time.perf_counter()
    try:
        for e in load_json(INVENTORY)['files']:
            p=(SNAPSHOT/e['path']).resolve()
            if not p.is_relative_to(SNAPSHOT) or p.stat().st_size!=e['size'] or sha256(p)!=e['sha256']:
                raise ValueError('Snapshot mismatch')
        _,_,_,rows,_,_,_,_=load_panel(PLAN);torch.set_num_threads(4)
        tok=AutoTokenizer.from_pretrained(SNAPSHOT,local_files_only=True,trust_remote_code=False)
        model,info=AutoModelForCausalLM.from_pretrained(SNAPSHOT,local_files_only=True,trust_remote_code=True,
            use_safetensors=True,torch_dtype=torch.float16,attn_implementation='sdpa',output_loading_info=True)
        if any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):raise ValueError('Load mismatch')
        model.eval().requires_grad_(False).to('cuda');processor=model.init_processor(tok);row=rows[0]
        messages=[{'role':'system','content':'You are a helpful assistant.'},
                  {'role':'user','content':'<|image|>\n'+row['question']},{'role':'assistant','content':''}]
        with Image.open(ROOT/row['asset']['file']) as im:
            inputs=processor(messages,images=[im.convert('RGB')],videos=None,cut_enable=True).to('cuda')
        pixels=inputs.pixel_values.detach().to(torch.float16)
        if tuple(pixels.shape)!=(5,3,384,384):raise ValueError('Unexpected tile layout')
        target=tok(' ship',add_special_tokens=False,return_tensors='pt')['input_ids'].to('cuda')
        if target.numel()!=1:raise ValueError('Target token mismatch')
        def visual(x):return model.vision2text_model(model.vision_model(x,output_hidden_states=True).hidden_states[-2])
        def objective(features):
            result=model.language_model(input_ids=inputs.input_ids,image_embeds=features,media_offset=inputs.media_offset,
                                        attention_mask=inputs.get('attention_mask'),use_cache=False,return_dict=True)
            return F.cross_entropy(result.logits[:,-1,:].float(),target[:,0])
        def clear_visual_cache():
            for m in model.modules():
                cache=getattr(m,'visual_cache',None)
                if isinstance(cache,dict):cache.clear()
        torch.cuda.reset_peak_memory_stats();t=time.perf_counter()
        with torch.no_grad():
            batch_features=visual(pixels)
            split_features=torch.cat([visual(pixels[i:i+1]) for i in range(5)],dim=0)
            max_feature_diff=float((batch_features-split_features).abs().max())
            feature_relative_l2=float(torch.linalg.vector_norm((batch_features-split_features).float())/torch.linalg.vector_norm(batch_features.float()))
        report.update(feature_max_abs_difference=max_feature_diff,feature_relative_l2=feature_relative_l2)
        print('feature relative L2',feature_relative_l2,'max',max_feature_diff,flush=True)
        if feature_relative_l2>0.01:raise ValueError('Split feature relative L2 exceeds 1%')
        del split_features
        features=batch_features.detach().requires_grad_(True);loss=objective(features)
        feature_grad,=torch.autograd.grad(loss,features);loss_staged=float(loss)
        clear_visual_cache();del loss,features,batch_features
        parts=[]
        for i in range(5):
            tile=pixels[i:i+1].detach().requires_grad_(True);y=visual(tile)
            g,=torch.autograd.grad(y,tile,grad_outputs=feature_grad[i:i+1])
            parts.append(g.detach().cpu());del tile,y,g
        staged=torch.cat(parts,dim=0).float();del parts,feature_grad
        torch.cuda.synchronize();staged_seconds=time.perf_counter()-t
        staged_peak=torch.cuda.max_memory_allocated()/2**20
        print('staged complete',round(staged_peak,2),'MiB',flush=True)
        torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();t=time.perf_counter()
        full_pixels=pixels.detach().requires_grad_(True);full_features=visual(full_pixels);full_loss=objective(full_features)
        full_grad,=torch.autograd.grad(full_loss,full_pixels);full_loss_value=float(full_loss)
        full_grad=full_grad.detach().cpu().float();clear_visual_cache()
        torch.cuda.synchronize();full_seconds=time.perf_counter()-t;full_peak=torch.cuda.max_memory_allocated()/2**20
        denom=float(torch.linalg.vector_norm(full_grad));diff=float(torch.linalg.vector_norm(staged-full_grad));
        cosine=float(F.cosine_similarity(staged.flatten(),full_grad.flatten(),dim=0))
        finite=bool(torch.isfinite(staged).all());frozen=all(not p.requires_grad and p.grad is None for p in model.parameters())
        report.update(status='passed' if finite and frozen and denom>0 and cosine>0.99 and diff/denom<0.1 else 'failed',
            sample_id=row['sample_id'],target=' ship',input_shape=list(pixels.shape),
            staged_loss=loss_staged,full_loss=full_loss_value,gradient_cosine=cosine,gradient_relative_l2=diff/denom,
            gradient_finite=finite,gradient_nonzero=int(torch.count_nonzero(staged)),weights_frozen=frozen,
            staged_peak_allocated_mib=staged_peak,full_peak_allocated_mib=full_peak,
            staged_seconds=staged_seconds,full_seconds=full_seconds,physical_vram_mib=torch.cuda.get_device_properties(0).total_memory/2**20)
    except Exception as e:report.update(error_type=type(e).__name__,error=str(e));traceback.print_exc()
    report['total_seconds']=time.perf_counter()-start
    out.write_text(json.dumps(report,indent=2)+'\n',encoding='utf8');print(json.dumps(report),flush=True)
    return int(report['status']!='passed')

if __name__=='__main__':raise SystemExit(main())
