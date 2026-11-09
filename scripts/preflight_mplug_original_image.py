"""Check an approximate differentiable original-image preprocessing path."""
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
    import numpy as np
    from PIL import Image
    from transformers import AutoModelForCausalLM,AutoTokenizer
    out=ROOT/'runs/sar-sarchat-mplug-20261109/original_image_probe_retry.json'
    if out.exists():raise ValueError('Preserve existing result')
    out.parent.mkdir(parents=True,exist_ok=True)
    report={'status':'failed','script_sha256':sha256(Path(__file__)),'inventory_sha256':sha256(INVENTORY)}
    start=time.perf_counter()
    try:
        for e in load_json(INVENTORY)['files']:
            p=(SNAPSHOT/e['path']).resolve()
            if not p.is_relative_to(SNAPSHOT) or p.stat().st_size!=e['size'] or sha256(p)!=e['sha256']:
                raise ValueError('Snapshot mismatch')
        _,_,_,rows,_,_,_,_=load_panel(PLAN);torch.set_num_threads(4);row=rows[0]
        tok=AutoTokenizer.from_pretrained(SNAPSHOT,local_files_only=True,trust_remote_code=False)
        model,info=AutoModelForCausalLM.from_pretrained(SNAPSHOT,local_files_only=True,trust_remote_code=True,
            use_safetensors=True,torch_dtype=torch.float16,attn_implementation='sdpa',output_loading_info=True)
        if any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):raise ValueError('Load mismatch')
        model.eval().requires_grad_(False).to('cuda');processor=model.init_processor(tok)
        messages=[{'role':'system','content':'You are a helpful assistant.'},
                  {'role':'user','content':'<|image|>\n'+row['question']},{'role':'assistant','content':''}]
        with Image.open(ROOT/row['asset']['file']) as im:
            if im.mode=='L':raw=np.array(im,copy=True)
            elif im.mode=='RGB':
                arr=np.array(im,copy=True)
                if not (np.array_equal(arr[:,:,0],arr[:,:,1]) and np.array_equal(arr[:,:,0],arr[:,:,2])):
                    raise ValueError('RGB channels are not identical')
                raw=arr[:,:,0].copy()
            else:raise ValueError('Probe requires grayscale or identical RGB channels')
            rgb=im.convert('RGB')
            selected=processor.image_processor.resizer(rgb,skip_resize=True)
            grid=processor.image_processor.anchors[int(selected)]
            inputs=processor(messages,images=[rgb],videos=None,cut_enable=True).to('cuda')
        if tuple(grid)!=(2,2) or tuple(inputs.pixel_values.shape)!=(5,3,384,384):raise ValueError('Expected 2x2 plus global')
        x=torch.tensor(raw,device='cuda',dtype=torch.float32).div(255).view(1,1,*raw.shape).requires_grad_(True)
        def preprocess(source):
            image=source.repeat(1,3,1,1)
            big=F.interpolate(image,size=(768,768),mode='bicubic',align_corners=False,antialias=True).clamp(0,1)
            global_view=F.interpolate(image,size=(384,384),mode='bicubic',align_corners=False,antialias=True).clamp(0,1)
            slices=[big[:,:,y:y+384,z:z+384] for y in (0,384) for z in (0,384)]
            return (torch.cat(slices+[global_view],dim=0)-0.5)/0.5
        tiles=preprocess(x);native=inputs.pixel_values.float()
        delta=tiles.detach()-native
        report.update(sample_id=row['sample_id'],raw_shape=list(x.shape),tile_shape=list(tiles.shape),
                      approx_pixel_mean_abs_error=float(delta.abs().mean()),approx_pixel_max_abs_error=float(delta.abs().max()),
                      note='Differentiable tensor bicubic approximates PIL bicubic; not bit-exact')
        def visual(p):return model.vision2text_model(model.vision_model(p,output_hidden_states=True).hidden_states[-2])
        with torch.no_grad():
            orig=model.forward_image(inputs.pixel_values)
            alternate=visual(tiles.detach().to(torch.float16))
            rel=float(torch.linalg.vector_norm((orig-alternate).float())/torch.linalg.vector_norm(orig.float()))
            def answer(feature):
                end=[tok.convert_tokens_to_ids(t) for t in model.terminators]
                ids=model.language_model.generate(input_ids=inputs.input_ids,image_embeds=feature,
                    media_offset=inputs.media_offset,attention_mask=inputs.get('attention_mask'),
                    pad_token_id=0,eos_token_id=end,do_sample=False,max_new_tokens=12)
                return tok.decode(ids[0,inputs.input_ids.shape[1]:],skip_special_tokens=True).strip()
            original_answer=answer(orig);alternate_answer=answer(alternate)
        report.update(feature_relative_l2=rel,native_answer=original_answer,approximate_answer=alternate_answer,
                      answers_match=original_answer==alternate_answer)
        clear=lambda:[m.visual_cache.clear() for m in model.modules() if isinstance(getattr(m,'visual_cache',None),dict)]
        clear();target=tok(' ship',add_special_tokens=False,return_tensors='pt')['input_ids'].to('cuda')
        if target.numel()!=1:raise ValueError('Unexpected target token')
        feat=alternate.detach().requires_grad_(True)
        output=model.language_model(input_ids=inputs.input_ids,image_embeds=feat,media_offset=inputs.media_offset,
                                    attention_mask=inputs.get('attention_mask'),use_cache=False,return_dict=True)
        loss=F.cross_entropy(output.logits[:,-1,:].float(),target[:,0]);fg,=torch.autograd.grad(loss,feat)
        clear();del output,feat,orig,alternate
        tile_grads=[]
        for i in range(5):
            tile=tiles[i:i+1].detach().to(torch.float16).requires_grad_(True)
            y=visual(tile);g,=torch.autograd.grad(y,tile,grad_outputs=fg[i:i+1]);tile_grads.append(g.detach().float());del tile,y,g
        grad,=torch.autograd.grad(tiles,x,grad_outputs=torch.cat(tile_grads,dim=0))
        finite=bool(torch.isfinite(grad).all());nonzero=int(torch.count_nonzero(grad));frozen=all(not p.requires_grad and p.grad is None for p in model.parameters())
        if not finite or not nonzero or not frozen:raise ValueError('Invalid gradient')
        report.update(status='passed',loss=float(loss),gradient_finite=finite,gradient_nonzero=nonzero,
                      gradient_max_abs=float(grad.abs().max()),weights_frozen=frozen)
    except Exception as e:report.update(error_type=type(e).__name__,error=str(e));traceback.print_exc()
    report['seconds']=time.perf_counter()-start
    if torch.cuda.is_available():report['peak_allocated_mib']=torch.cuda.max_memory_allocated()/2**20
    out.write_text(json.dumps(report,indent=2)+'\n',encoding='utf8');print(json.dumps(report),flush=True)
    return int(report['status']!='passed')

if __name__=='__main__':raise SystemExit(main())
