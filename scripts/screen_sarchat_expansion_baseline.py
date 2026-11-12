import json,pathlib,hashlib,time,torch
from PIL import Image
from torchvision.transforms import Compose,Resize,ToTensor,Normalize,InterpolationMode
from transformers import AutoModel,AutoTokenizer
from run_sar_proxy_candidate import IMAGE_SIZE,IMAGE_MEAN,IMAGE_STD,prepare_prompt
root=pathlib.Path(__file__).resolve().parents[1];snap=root/'models/SARChat-InternVL2.5-1B/6c963963';inventory=json.loads((root/'resources/manifests/sarchat_snapshot_inventory_20261108.json').read_text(encoding='utf8'))
for x in inventory['files']:
 f=snap/x['path'];assert f.stat().st_size==x['size'] and hashlib.sha256(f.read_bytes()).hexdigest()==x['sha256']
p=root/'resources/manifests/sar_expansion_conservative_subset_20261112.json';data=p.read_bytes();rows=[x for x in json.loads(data)['items'] if x['partition']=='development'];assert len(rows)==28
protocol=json.loads((root/'experiments/protocols/sarchat_glm_edge_natural_screen_20261109.json').read_text(encoding='utf8'));torch.set_num_threads(4);torch.cuda.reset_peak_memory_stats();start=time.perf_counter();tokenizer=AutoTokenizer.from_pretrained(snap,local_files_only=True,trust_remote_code=False,use_fast=False);model,info=AutoModel.from_pretrained(snap,local_files_only=True,trust_remote_code=True,use_safetensors=True,torch_dtype=torch.float16,attn_implementation='eager',use_flash_attn=False,output_loading_info=True);assert not any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs'));model.eval().requires_grad_(False).to('cuda');inputs,sep,template=prepare_prompt(model,tokenizer,protocol['question']);inputs={k:v.to('cuda') for k,v in inputs.items()};transform=Compose([Resize((IMAGE_SIZE,IMAGE_SIZE),interpolation=InterpolationMode.BICUBIC),ToTensor(),Normalize(IMAGE_MEAN,IMAGE_STD)]);results=[];out=root/'resources/manifests/sarchat_1b_expansion_development_baseline_20261112.json'
for row in rows:
 x={'image_name':row['image_name'],'reference':row['label']}
 try:
  im=Image.open(root/row['asset']['file']);im.load();pixels=transform(im.convert('RGB')).unsqueeze(0).to(device='cuda',dtype=torch.float16)
  with torch.inference_mode():
   result=model.generate(pixel_values=pixels,**inputs,**protocol['generation'],eos_token_id=template['eos_token_id'],return_dict_in_generate=True,output_logits=True)
  tokens=result.sequences;assert result.logits and all(bool(torch.isfinite(z).all()) for z in result.logits);x.update(answer=tokenizer.decode(tokens[0].cpu().tolist(),skip_special_tokens=True).strip().split(sep,1)[0],tokens=int(tokens.shape[1]),status='ok')
 except Exception as e:x.update(status='error',error=str(e),answer=None)
 results.append(x);out.write_text(json.dumps({'status':'running','subset_sha256':hashlib.sha256(data).hexdigest(),'question':protocol['question'],'generation':protocol['generation'],'predictions':results},ensure_ascii=False,indent=2)+'\n',encoding='utf8');print(f'{len(results)}/28',flush=True)
out.write_text(json.dumps({'status':'completed','subset_sha256':hashlib.sha256(data).hexdigest(),'question':protocol['question'],'generation':protocol['generation'],'predictions':results,'seconds':time.perf_counter()-start,'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20},ensure_ascii=False,indent=2)+'\n',encoding='utf8')
