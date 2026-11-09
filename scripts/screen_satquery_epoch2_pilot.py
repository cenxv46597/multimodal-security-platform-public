"""Twelve-image exploratory screen of a separately published SatQuery adapter."""
import json
import time
import traceback
from pathlib import Path
from collections import Counter

from evaluate_sar_grouped_development import load_panel
from run_vlm_clean_screen import ROOT, load_json, sha256

PLAN = 'bc712ce9733fa633e3a6628cea93da0f3777291dd36f011a98ac0152c117f4f9'
ADAPTER_SHA = 'b3e048539d557dbef54708dc3b4edb4f068f9d3c50c0c035fef0e1ec737ed3c1'
BASE_SHA = '4e4666a35fc7a8672ef834b767ebc1a877d6ab277f88b139f3462ee44831bf16'

def main():
    import torch
    from PIL import Image
    from peft import PeftModel, get_peft_model_state_dict
    from safetensors.torch import load_file
    from torchvision.transforms import Compose, InterpolationMode, Normalize, Resize, ToTensor
    from transformers import AutoModel, AutoTokenizer

    output = ROOT / 'runs/satquery-epoch2-20261109/pilot-language-model-adapter.json'
    if output.exists():
        raise ValueError('Output already exists')
    output.parent.mkdir(parents=True, exist_ok=True)
    base = ROOT / '.work/satquery-20261108/base'
    adapter = ROOT / '.work/satquery-praneyaarora-20261109/adapter'
    protocol = load_json(ROOT / 'experiments/protocols/sarchat_glm_edge_natural_screen_20261109.json')
    report = {'status': 'running', 'model': 'Praneyaarora/satquery-internvl35-1b-epoch2-lora',
              'adapter_revision': 'bd2d41e568e30307718424082c24e3be64d0b76e',
              'adapter_sha256': ADAPTER_SHA, 'base_sha256': BASE_SHA, 'plan_sha256': PLAN,
              'question': protocol['question'], 'generation': protocol['generation'],
              'selection': 'First two validation rows per reference class in fixed panel order', 'predictions': []}
    start = time.perf_counter()
    try:
        if sha256(base / 'model.safetensors') != BASE_SHA or sha256(adapter / 'adapter_model.safetensors') != ADAPTER_SHA:
            raise ValueError('Checkpoint hash mismatch')
        _, _, _, rows, _, _, labels, _ = load_panel(PLAN)
        counts = Counter()
        selected = []
        for row in rows:
            label = row['reference_answer']
            if counts[label] < 2:
                selected.append(row)
                counts[label] += 1
        if len(selected) != 12 or any(counts[label] != 2 for label in labels):
            raise ValueError('Unexpected subset')
        report['selected_ids'] = [r['sample_id'] for r in selected]
        torch.set_num_threads(4)
        tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True, trust_remote_code=True, use_fast=False)
        model, info = AutoModel.from_pretrained(base, local_files_only=True, trust_remote_code=True,
            torch_dtype=torch.float16, use_flash_attn=False, output_loading_info=True)
        if any(info.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
            raise ValueError('Base checkpoint mismatch: ' + str(info))
        # This checkpoint targets the language model itself, not the outer VLM.
        wrapped = PeftModel.from_pretrained(model.language_model, adapter, local_files_only=True)
        expected = load_file(adapter / 'adapter_model.safetensors')
        actual = get_peft_model_state_dict(wrapped)
        if expected.keys() != actual.keys() or any(not torch.equal(expected[k], actual[k].cpu()) for k in expected):
            raise ValueError('Adapter tensors do not match checkpoint')
        model.language_model = wrapped
        model = model.eval().requires_grad_(False).to('cuda')
        transform = Compose([Resize((448, 448), interpolation=InterpolationMode.BICUBIC), ToTensor(),
                             Normalize((.485, .456, .406), (.229, .224, .225))])
        report.update(gpu=torch.cuda.get_device_name(), dtype='float16', adapter_tensors_verified=len(expected),
                      preprocessing='one 448x448 tile, ImageNet normalization')
        for row in selected:
            with Image.open(ROOT / row['asset']['file']) as image:
                pixels = transform(image.convert('RGB')).unsqueeze(0).to('cuda', dtype=torch.float16)
            with torch.inference_mode():
                answer = model.chat(tokenizer, pixels, '<image>\n' + protocol['question'],
                                    generation_config=dict(protocol['generation']))
            report['predictions'].append({'sample_id': row['sample_id'], 'answer': answer})
            output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
            print('Evaluated', len(report['predictions']), '/12', flush=True)
        report['status'] = 'completed'
        report['peak_allocated_mib'] = torch.cuda.max_memory_allocated() / 2**20
    except Exception as e:
        report.update(status='failed', error_type=type(e).__name__, error=str(e), traceback=traceback.format_exc())
        traceback.print_exc()
    report['seconds'] = time.perf_counter() - start
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    print(report['status'], output, flush=True)
    return int(report['status'] != 'completed')

if __name__ == '__main__':
    raise SystemExit(main())
