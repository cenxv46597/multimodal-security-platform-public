"""Rescreen local SARChat-Qwen on the existing natural-language panel."""
import argparse
import json
import time
import traceback
from pathlib import Path
from run_vlm_clean_screen import ROOT, load_json, sha256, timed
from evaluate_sar_grouped_development import load_panel

SNAP = ROOT / 'models/SARChat-Qwen2VL-2B/master-20261108'
INVENTORY = ROOT / 'resources/manifests/sarchat_qwen2vl_inventory_20261108.json'
PROTOCOL = ROOT / 'experiments/protocols/sarchat_glm_edge_natural_screen_20261109.json'
PLAN = 'bc712ce9733fa633e3a6628cea93da0f3777291dd36f011a98ac0152c117f4f9'

def main():
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    p = argparse.ArgumentParser()
    p.add_argument('--phase', choices=['probe', 'screen'], required=True)
    p.add_argument('--probe-report', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    out = a.output.resolve()
    if not out.is_relative_to(ROOT / 'runs') or out.exists():
        p.error('Choose a new output under runs')
    out.parent.mkdir(parents=True, exist_ok=True)
    report = {'status': 'running', 'phase': a.phase, 'runner_sha256': sha256(Path(__file__)),
              'inventory_sha256': sha256(INVENTORY), 'protocol_sha256': sha256(PROTOCOL), 'predictions': []}
    start = time.perf_counter()
    try:
        inventory = load_json(INVENTORY)
        protocol = load_json(PROTOCOL)
        if protocol['grouped_plan_sha256'] != PLAN:
            raise ValueError('Panel mismatch')
        for entry in inventory['files']:
            f = (SNAP / entry['path']).resolve()
            if not f.is_relative_to(SNAP) or f.stat().st_size != entry['size'] or sha256(f) != entry['sha256']:
                raise ValueError('Snapshot mismatch: ' + entry['path'])
        _, _, _, rows, _, _, _, _ = load_panel(PLAN)
        if a.phase == 'screen':
            if a.probe_report is None:
                raise ValueError('Probe required')
            probe = load_json(a.probe_report)
            if probe['status'] != 'passed' or any(probe[k] != report[k] for k in ('runner_sha256', 'inventory_sha256', 'protocol_sha256')):
                raise ValueError('Probe mismatch')
            report['probe_sha256'] = sha256(a.probe_report)
        torch.set_num_threads(4)
        processor = AutoProcessor.from_pretrained(SNAP, local_files_only=True, max_pixels=256*28*28, use_fast=False)
        model, info = Qwen2VLForConditionalGeneration.from_pretrained(SNAP, local_files_only=True,
            torch_dtype=torch.float16, attn_implementation='eager', output_loading_info=True)
        if any(info.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
            raise ValueError('Checkpoint loading mismatch')
        model.eval().requires_grad_(False).to('cuda')
        report.update(model=inventory['model'], grouped_plan_sha256=PLAN, question=protocol['question'],
                      generation=protocol['generation'], dtype='float16', attention='eager', max_pixels=256*28*28,
                      gpu=torch.cuda.get_device_name())
        def infer(row):
            messages = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': protocol['question']}]}]
            prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            with Image.open(ROOT / row['asset']['file']) as image:
                inputs = processor(text=[prompt], images=[image.convert('RGB')], return_tensors='pt').to('cuda')
            with torch.inference_mode():
                result = model.generate(**inputs, **protocol['generation'], return_dict_in_generate=True, output_logits=True)
            if not result.logits or not all(bool(torch.isfinite(x).all()) for x in result.logits):
                raise ValueError('Missing or nonfinite logits')
            tokens = result.sequences[:, inputs.input_ids.shape[1]:]
            return {'answer': processor.batch_decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip(),
                    'generated_tokens': int(tokens.shape[1]), 'hit_token_limit': int(tokens.shape[1]) >= protocol['generation']['max_new_tokens'],
                    'generation_logits_finite': True, 'image_grid_thw': inputs.image_grid_thw.cpu().tolist()}
        if a.phase == 'probe':
            timed(report, 'probe', lambda: infer(rows[0]), 'cuda')
            report.update(status='passed', probe_sample_id=rows[0]['sample_id'])
        else:
            for row in rows:
                item = {'sample_id': row['sample_id'], 'status': 'ok'}
                try:
                    timed(item, 'generation', lambda: infer(row), 'cuda')
                    item.update(item.pop('generation'))
                except Exception as e:
                    item.update(status='error', answer=None, error_type=type(e).__name__, error=str(e))
                report['predictions'].append(item)
                out.write_text(json.dumps(report, indent=2)+'\n', encoding='utf8')
                print('Evaluated', len(report['predictions']), '/60', flush=True)
            report['status'] = 'completed'
    except Exception as e:
        report.update(status='failed', error_type=type(e).__name__, error=str(e))
        traceback.print_exc()
    report['seconds'] = time.perf_counter()-start
    out.write_text(json.dumps(report, indent=2)+'\n', encoding='utf8')
    print(report['status'], out, flush=True)
    return int(report['status'] == 'failed')

if __name__ == '__main__':
    raise SystemExit(main())
