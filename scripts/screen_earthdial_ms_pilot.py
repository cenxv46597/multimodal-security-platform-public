"""Load author EarthDial code and screen a fixed 12-image development subset."""
import json
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / '.work/earthdial-preflight-20261109/source/src'),
                str(ROOT / '.work/earthdial-preflight-20261109/deps'),
                str(ROOT / '.work/internvl2b-deps')]
sys.path.extend([str(ROOT / '.venv-internvl/Lib/site-packages'),
                 str(ROOT / '.venv-satquery/Lib/site-packages')])  # keep the active environment's transformers
from evaluate_sar_grouped_development import load_panel
from run_vlm_clean_screen import load_json, sha256

PLAN = 'bc712ce9733fa633e3a6628cea93da0f3777291dd36f011a98ac0152c117f4f9'
WEIGHTS = {'model-00001-of-00002.safetensors': 'c51e63404b7e2d78f914fa999058b533782153483a9b59e99aee8011879792a6',
           'model-00002-of-00002.safetensors': 'ab2a47b175f75ecd5e55339417d63c5751f45018efdd7279d501193607392d06'}

def main():
    import torch
    from PIL import Image
    from torchvision.transforms import Compose, Resize, ToTensor, Normalize, InterpolationMode
    from transformers import AutoTokenizer
    from earthdial.model.internvl_chat.configuration_internvl_chat import InternVLChatConfig
    from earthdial.model.internvl_chat.modeling_internvl_chat import InternVLChatModel

    out = ROOT / 'runs/earthdial-ms-20261109/pilot-legacy-cache.json'
    if out.exists():
        raise ValueError('Output already exists')
    out.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = ROOT / '.work/earthdial-preflight-20261109/checkpoint'
    protocol = load_json(ROOT / 'experiments/protocols/sarchat_glm_edge_natural_screen_20261109.json')
    generation = dict(protocol['generation'])
    report = {'status': 'running', 'model': 'akshaydudhane/EarthDial_4B_MS',
              'revision': '835030e175a5b0861b0e9a050db20fcc1c2e0a88',
              'source_commit': '92c1260703ae8b3ddcac17fce6e8234de717308b',
              'plan_sha256': PLAN, 'question': protocol['question'], 'generation': generation,
              'selection': 'First two validation rows per class in fixed panel order', 'predictions': []}
    start = time.perf_counter()
    try:
        for name, expected in WEIGHTS.items():
            if sha256(checkpoint / name) != expected:
                raise ValueError('Checkpoint hash mismatch: '+name)
        _, _, _, rows, _, _, labels, _ = load_panel(PLAN)
        counts = Counter()
        selected = []
        for row in rows:
            if counts[row['reference_answer']] < 2:
                selected.append(row)
                counts[row['reference_answer']] += 1
        if len(selected) != 12 or any(counts[label] != 2 for label in labels):
            raise ValueError('Unexpected panel')
        report['selected_ids'] = [row['sample_id'] for row in selected]
        torch.set_num_threads(4)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, use_fast=False)
        config = InternVLChatConfig.from_pretrained(checkpoint, local_files_only=True)
        config.vision_config.use_flash_attn = False
        config.llm_config._attn_implementation = 'eager'
        model, info = InternVLChatModel.from_pretrained(checkpoint, config=config, local_files_only=True,
            torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map='auto',
            max_memory={0: '4500MiB', 'cpu': '20GiB'}, output_loading_info=True)
        if any(info.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
            raise ValueError('Checkpoint mismatch: '+str(info))
        model.eval().requires_grad_(False)
        # The bundled Phi3 implementation expects generation to start with no cache,
        # and creates its legacy cache internally; newer Transformers preallocate one.
        model.language_model._supports_cache_class = False
        report.update(gpu=torch.cuda.get_device_name(), dtype='float16', device_map=model.hf_device_map,
                      preprocessing='one 448x448 tile, ImageNet normalization', cache_mode='legacy compatibility')
        transform = Compose([Resize((448,448), interpolation=InterpolationMode.BICUBIC), ToTensor(),
                             Normalize((.485,.456,.406),(.229,.224,.225))])
        for row in selected:
            with Image.open(ROOT / row['asset']['file']) as image:
                pixels = transform(image.convert('RGB')).unsqueeze(0).to('cuda', dtype=torch.float16)
            with torch.inference_mode():
                answer = model.chat(tokenizer, pixels, '<image>\n'+protocol['question'],
                                    generation_config=dict(generation))
            report['predictions'].append({'sample_id': row['sample_id'], 'answer': answer})
            out.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n',encoding='utf8')
            print('Evaluated',len(report['predictions']),'/12',flush=True)
        report['status'] = 'completed'
        report['peak_allocated_mib'] = torch.cuda.max_memory_allocated() / 2**20
    except Exception as e:
        report.update(status='failed',error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc())
        traceback.print_exc()
    report['seconds'] = time.perf_counter()-start
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n',encoding='utf8')
    print(report['status'],out,flush=True)
    return int(report['status'] != 'completed')

if __name__ == '__main__':
    raise SystemExit(main())
