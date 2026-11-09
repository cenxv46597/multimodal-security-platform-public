"""Screen pinned Zhimu on a fixed, balanced 12-image development subset."""
import hashlib
import importlib
import json
import sys
import time
import traceback
import types
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / '.work/zhimu-20261109/deps'),
                str(ROOT / '.work/earthdial-preflight-20261109/deps')]
from evaluate_sar_grouped_development import load_panel
from run_vlm_clean_screen import load_json, sha256

PLAN = 'bc712ce9733fa633e3a6628cea93da0f3777291dd36f011a98ac0152c117f4f9'
REV = '35efd75d897b94e5971f523934efd72663106ce9'
CHECKPOINT = ROOT / '.work/zhimu-20261109/checkpoint'
INVENTORY = ROOT / '.work/zhimu-20261109/metadata/inventory.json'
PROTOCOL = ROOT / 'experiments/protocols/sarchat_glm_edge_natural_screen_20261109.json'

def main():
    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoProcessor
    output = ROOT / 'runs/zhimu-20261109/pilot.json'
    if output.exists():
        raise ValueError('Output exists')
    output.parent.mkdir(parents=True, exist_ok=True)
    protocol = load_json(PROTOCOL)
    report = {'status':'running', 'model':'AIR-CAS-kx/Zhimu-1.0', 'revision':REV,
              'protocol_sha256':sha256(PROTOCOL), 'plan_sha256':PLAN, 'question':protocol['question'],
              'generation':protocol['generation'], 'selection':'first two validation rows per class',
              'predictions':[]}
    start = time.perf_counter()
    try:
        inv = load_json(INVENTORY)
        if inv['revision'] != REV:
            raise ValueError('Inventory mismatch')
        for entry in inv['files']:
            expected = entry.get('lfs', {}).get('sha256')
            if entry['rfilename'].endswith('.safetensors'):
                if not expected or sha256(CHECKPOINT / entry['rfilename']) != expected:
                    raise ValueError('Weight mismatch: '+entry['rfilename'])
        _,_,_,rows,_,_,labels,_ = load_panel(PLAN)
        counts = Counter()
        selected = []
        for row in rows:
            label = row['reference_answer']
            if counts[label] < 2:
                selected.append(row)
                counts[label] += 1
        if len(selected) != 12 or any(counts[x] != 2 for x in labels):
            raise ValueError('Panel mismatch')
        report['selected_ids'] = [x['sample_id'] for x in selected]
        torch.set_num_threads(4)
        name = 'zhimu_local_checkpoint'
        package = types.ModuleType(name)
        package.__path__ = [str(CHECKPOINT)]
        package.__package__ = name
        sys.modules[name] = package
        module = importlib.import_module(name+'.modeling_qwen3_vl')
        model, info = module.Qwen3VLForConditionalGeneration.from_pretrained(
            CHECKPOINT, local_files_only=True, torch_dtype=torch.float16, low_cpu_mem_usage=True,
            device_map='auto', max_memory={0:'4500MiB','cpu':'18GiB'}, attn_implementation='eager',
            output_loading_info=True)
        if any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):
            raise ValueError('Checkpoint loading issue: '+str(info))
        model.eval().requires_grad_(False)
        processor = AutoProcessor.from_pretrained(CHECKPOINT, trust_remote_code=True, local_files_only=True, use_fast=False)
        image_module = importlib.import_module(name+'.image_processing_qwen2_vl')
        processor.image_processor = image_module.Qwen2VLImageProcessor.from_pretrained(CHECKPOINT, local_files_only=True)
        report.update(device_map=model.hf_device_map, gpu=torch.cuda.get_device_name(), dtype='float16',
                      image_input='PIL grayscale conversion to 1xHxW uint8; SAR single-channel branch')
        target = [(n,m) for n,m in model.named_modules() if 'visual.patch_embed.extra_patch_embeds.ch_1.proj' in n]
        if len(target) != 1:
            raise ValueError('SAR image branch not found uniquely')
        for row in selected:
            with Image.open(ROOT / row['asset']['file']) as image:
                rgb = np.asarray(image.convert('RGB'))
                identical = bool(np.array_equal(rgb[:,:,0],rgb[:,:,1]) and np.array_equal(rgb[:,:,1],rgb[:,:,2]))
                pixels = np.asarray(image.convert('L'))[None,:,:]
            messages = [{'role':'user','content':[{'type':'image','image':pixels},
                        {'type':'text','text':protocol['question']}]}]
            inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                  return_dict=True, return_tensors='pt')
            if inputs['num_channels'].flatten().tolist() != [1]:
                raise ValueError('Wrong image branch input')
            called = []
            hook = target[0][1].register_forward_hook(lambda module, inp, out:called.append(True))
            try:
                inputs = inputs.to(model.device)
                length = inputs['input_ids'].shape[1]
                with torch.inference_mode():
                    result = model.generate(**inputs, **protocol['generation'], use_cache=True)
            finally:
                hook.remove()
            if not called:
                raise ValueError('SAR branch was not called')
            tokens = result[:,length:]
            answer = processor.batch_decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
            report['predictions'].append({'sample_id':row['sample_id'], 'answer':answer,
                'generated_tokens':int(tokens.shape[1]), 'hit_token_limit':int(tokens.shape[1])>=32,
                'rgb_channels_identical':identical, 'sar_branch_called':True})
            output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
            print('Evaluated',len(report['predictions']),'/12',flush=True)
        report['status'] = 'completed'
        report['peak_allocated_mib'] = torch.cuda.max_memory_allocated()/2**20
    except Exception as e:
        report.update(status='failed', error_type=type(e).__name__, error=str(e), traceback=traceback.format_exc())
        traceback.print_exc()
    report['seconds'] = time.perf_counter()-start
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(report['status'], output, flush=True)
    return int(report['status']!='completed')

if __name__ == '__main__':
    raise SystemExit(main())
