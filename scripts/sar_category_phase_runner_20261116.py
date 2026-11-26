"""Run the four retained SARChat models on two preselected port images."""
import argparse
import os
import json
import sys
sys.path.insert(0, str(__import__('pathlib').Path.cwd() / 'scripts'))
import time
import traceback
from pathlib import Path

def phase(name):
    output_arg = Path(sys.argv[sys.argv.index('--output') + 1])
    with (output_arg.parent / 'phases.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps({'phase': name, 'monotonic': time.monotonic()}) + '\n')
        f.flush()

phase('imports')
from run_vlm_clean_screen import ROOT, load_json, sha256
from run_sar_proxy_candidate import IMAGE_MEAN, IMAGE_STD, IMAGE_SIZE, prepare_prompt

sys.path.insert(0, str(ROOT / '.work/internvl2b-deps'))
PROTOCOL = ROOT / 'experiments/protocols/sar_port_positive_probe_20261111.json'
MODELS = {
    'internvl1b': ('SARChat-InternVL2.5-1B/6c963963', 'sarchat_snapshot_inventory_20261108.json', 'internvl'),
    'internvl2b': ('SARChat-InternVL2.5-2B/master-20261109', 'sarchat_internvl2b_inventory_20261109.json', 'internvl'),
    'mplug': ('SARChat-mPLUG-Owl3-1B/master-20261108', 'sarchat_mplug_inventory_20261108.json', 'mplug'),
    'glm': ('SARChat-GLM-edge-v-2B/master-20261109', 'sarchat_glm_edge_inventory_20261109.json', 'glm'),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, choices=MODELS)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--protocol', type=Path, default=PROTOCOL)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / 'runs/vision-safety-capability-preflight-20261114') or (output.exists() and not args.resume) or (args.resume and not output.exists()):
        parser.error('Choose a new output, or explicitly resume an existing output')
    protocol_path = args.protocol.resolve()
    if not protocol_path.is_relative_to(ROOT / 'runs/vision-safety-capability-preflight-20261114') or not protocol_path.is_file():
        parser.error('Choose a protocol under experiments/protocols')
    output.parent.mkdir(parents=True, exist_ok=True)
    snapshot_name, inventory_name, kind = MODELS[args.model]
    snapshot = ROOT / 'models' / snapshot_name
    inventory = ROOT / 'resources/manifests' / inventory_name
    protocol = load_json(protocol_path)
    result = {'status': 'running', 'model_key': args.model, 'protocol_sha256': sha256(protocol_path),
              'inventory_sha256': sha256(inventory), 'predictions': []}
    result['runner_sha256'] = sha256(Path(__file__))
    if args.resume:
        previous = load_json(output)
        for key in ('model_key', 'protocol_sha256', 'inventory_sha256', 'runner_sha256'):
            if previous.get(key) != result[key]: raise ValueError('Resume mismatch: '+key)
        if previous.get('status') == 'completed': raise ValueError('Already completed')
        prior = previous['predictions']
        if [v['sample_id'] for v in prior] != [v['sample_id'] for v in protocol['rows'][:len(prior)]]: raise ValueError('Invalid checkpoint order')
        old_status=previous.get('status')
        result = previous
        result['status'] = 'running'
        result.setdefault('resume_events', []).append({'previous_count':len(prior), 'previous_status':old_status})
    def save():
        temp=output.with_name(output.name+'.tmp')
        temp.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
        os.replace(temp,output)
    pending=output.with_name(output.name+'.pending.json')
    if pending.exists(): raise ValueError('Interrupted in-flight call recorded in '+str(pending)+'; inspect before continuing')
    started = time.perf_counter()
    try:
        import torch
        from PIL import Image
        from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, AutoImageProcessor
        from torchvision.transforms import Compose, Resize, ToTensor, Normalize, InterpolationMode
        phase('inventory')
        for entry in load_json(inventory)['files']:
            path = (snapshot / entry['path']).resolve()
            if not path.is_relative_to(snapshot) or path.stat().st_size != entry['size'] or sha256(path) != entry['sha256']:
                raise ValueError('Snapshot mismatch: ' + entry['path'])
        for row in protocol['rows']:
            if sha256(ROOT / row['image_path']) != row['image_sha256']:
                raise ValueError('Image hash mismatch')
        phase('model_loading')
        torch.set_num_threads(4)
        tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True,
                                                   trust_remote_code=(args.model == 'internvl2b'), use_fast=False)
        loader = AutoModel if kind == 'internvl' else AutoModelForCausalLM
        kwargs = {'local_files_only': True, 'trust_remote_code': True, 'use_safetensors': True,
                  'torch_dtype': torch.float16, 'attn_implementation': 'eager' if kind == 'internvl' else 'sdpa',
                  'output_loading_info': True}
        if kind == 'internvl':
            kwargs['use_flash_attn'] = False
        model, info = loader.from_pretrained(snapshot, **kwargs)
        if any(info.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
            raise ValueError('Checkpoint loading mismatch')
        model.eval().requires_grad_(False).to('cuda')
        question, generation = protocol['question'], protocol['generation']
        if kind == 'internvl':
            inputs, separator, template = prepare_prompt(model, tokenizer, question)
            inputs = {k: v.to('cuda') for k, v in inputs.items()}
            transform = Compose([Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=InterpolationMode.BICUBIC),
                                 ToTensor(), Normalize(IMAGE_MEAN, IMAGE_STD)])
        elif kind == 'mplug':
            processor = model.init_processor(tokenizer)
        else:
            processor = AutoImageProcessor.from_pretrained(snapshot, local_files_only=True, use_fast=False)
        result.update(model=load_json(inventory)['model'], question=question, generation=generation,
                      gpu=torch.cuda.get_device_name(), dtype='float16')
        for row in protocol['rows'][len(result['predictions']):]:
            phase('inference')
            question = row['question']
            if kind == 'internvl':
                inputs, separator, template = prepare_prompt(model, tokenizer, question)
                inputs = {k: v.to('cuda') for k, v in inputs.items()}
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); image_start = time.perf_counter()
            with pending.open('x',encoding='utf8') as f:
                json.dump({'sample_id':row['sample_id'],'attempt':'started','model_key':args.model},f)
            item = {'sample_id': row['sample_id'], 'status': 'ok'}
            try:
                with Image.open(ROOT / row['image_path']) as image:
                    item.update(raw_image_mode=image.mode, raw_image_size=list(image.size))
                    rgb = image.convert('RGB')
                    if kind == 'internvl':
                        pixels = transform(rgb).unsqueeze(0).to('cuda', dtype=torch.float16)
                    elif kind == 'mplug':
                        messages = [{'role': 'system', 'content': 'You are a helpful assistant.'},
                                    {'role': 'user', 'content': '<|image|>\n' + question},
                                    {'role': 'assistant', 'content': ''}]
                        batch = processor(messages, images=[rgb], videos=None, cut_enable=True).to('cuda')
                    else:
                        pixels = processor(rgb, return_tensors='pt')['pixel_values'].to('cuda', dtype=torch.float16)
                item['input_pixel_shape'] = list(batch.pixel_values.shape if kind == 'mplug' else pixels.shape)
                with torch.inference_mode():
                    if kind == 'internvl':
                        generated = model.generate(pixel_values=pixels, **inputs, **generation,
                                                   eos_token_id=template['eos_token_id'], return_dict_in_generate=True,
                                                   output_logits=True)
                        tokens = generated.sequences
                    elif kind == 'mplug':
                        embeds = model.forward_image(batch.pixel_values)
                        stop = [tokenizer.convert_tokens_to_ids(x) for x in model.terminators]
                        generated = model.language_model.generate(input_ids=batch.input_ids, image_embeds=embeds,
                            media_offset=batch.media_offset, attention_mask=batch.get('attention_mask'),
                            pad_token_id=0, eos_token_id=stop, **generation, return_dict_in_generate=True, output_logits=True)
                        tokens = generated.sequences[:, batch.input_ids.shape[1]:]
                    else:
                        messages = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': question}]}]
                        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_dict=True,
                                                              tokenize=True, return_tensors='pt').to('cuda')
                        generated = model.generate(**inputs, pixel_values=pixels, **generation,
                                                   return_dict_in_generate=True, output_logits=True)
                        tokens = generated.sequences[:, inputs.input_ids.shape[1]:]
                    if not generated.logits or not all(bool(torch.isfinite(x).all()) for x in generated.logits):
                        raise ValueError('Nonfinite generation logits')
                phase('generation_done')
                answer = tokenizer.decode(tokens[0].cpu().tolist(), skip_special_tokens=True).strip()
                if kind == 'internvl':
                    answer = answer.split(separator, 1)[0]
                item.update(answer=answer, generated_tokens=int(tokens.shape[1]),
                            hit_token_limit=int(tokens.shape[1]) >= generation['max_new_tokens'])
            except Exception as error:
                item.update(status='error', error_type=type(error).__name__, error=str(error))
            torch.cuda.synchronize()
            item.update(elapsed_seconds=time.perf_counter()-image_start, peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20, peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
            result['predictions'].append(item)
            save()
            pending.unlink()
            print(args.model, len(result['predictions']), '/', len(protocol['rows']), flush=True)
        result['status'] = 'completed'
    except Exception as error:
        result.update(status='failed', error_type=type(error).__name__, error=str(error))
        traceback.print_exc()
    result['seconds'] = time.perf_counter() - started
    save()
    return int(result['status'] != 'completed')


if __name__ == '__main__':
    raise SystemExit(main())
