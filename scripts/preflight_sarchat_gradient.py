"""Check one frozen SARChat input-gradient path after the fixed clean screen."""
import json
import time
import traceback
from pathlib import Path

from run_vlm_clean_screen import ROOT, load_json, sha256

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import Compose, InterpolationMode, Normalize, Resize, ToTensor
from transformers import AutoModel, AutoTokenizer

from evaluate_sarchat import INVENTORY, PROTOCOL, SNAPSHOT
from evaluate_sar_clean_screen import score_samples
from evaluate_sar_grouped_development import load_panel
from run_sar_proxy_candidate import IMAGE_MEAN, IMAGE_SIZE, IMAGE_STD, prepare_prompt

out = ROOT / 'runs/sar-sarchat-20261108/gradient_retry.json'
screen_path = ROOT / 'runs/sar-sarchat-20261108/screen.json'
if out.exists():
    raise SystemExit('Output already exists')
report = {'status': 'failed', 'screen_sha256': sha256(screen_path),
          'script_sha256': sha256(Path(__file__))}
start = time.perf_counter()
try:
    protocol, screen = load_json(PROTOCOL), load_json(screen_path)
    if screen['status'] != 'completed' or screen['protocol_sha256'] != sha256(PROTOCOL):
        raise ValueError('Screen or protocol mismatch')
    _, _, _, rows, _, _, labels, criteria = load_panel(protocol['grouped_plan_sha256'])
    score = score_samples(rows, screen['predictions'], labels, criteria)
    if score != screen['score'] or not score['gate']['passed']:
        raise ValueError('Independent rescoring or gate failed')
    for entry in load_json(INVENTORY)['files']:
        path = (SNAPSHOT / entry['path']).resolve()
        if not path.is_relative_to(SNAPSHOT) or path.stat().st_size != entry['size'] or sha256(path) != entry['sha256']:
            raise ValueError('Snapshot mismatch')
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT, local_files_only=True, trust_remote_code=False, use_fast=False)
    model, info = AutoModel.from_pretrained(SNAPSHOT, local_files_only=True, trust_remote_code=True,
        use_safetensors=True, torch_dtype=torch.float16, attn_implementation='eager', use_flash_attn=False,
        output_loading_info=True)
    if any(info.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
        raise ValueError('Checkpoint loading issue')
    model.eval().requires_grad_(False).to('cuda')
    row = rows[0]
    inputs, separator, template = prepare_prompt(model, tokenizer, row['question'])
    transform = Compose([Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=InterpolationMode.BICUBIC),
                         ToTensor(), Normalize(IMAGE_MEAN, IMAGE_STD)])
    with Image.open(ROOT / row['asset']['file']) as image:
        pixels = transform(image.convert('RGB')).unsqueeze(0).to(device='cuda', dtype=torch.float16).detach().requires_grad_(True)
    target_text = row['label'] if 'label' in row else score['sample_results'][0]['reference']
    target = tokenizer(target_text + separator, add_special_tokens=False, return_tensors='pt')['input_ids'].to('cuda')
    prefix = inputs['input_ids'].to('cuda')
    full_text = template['rendered_before_image_token_expansion'].replace('<image>', '<img>' + '<IMG_CONTEXT>' * model.num_image_token + '</img>', 1) + target_text + separator
    full = tokenizer(full_text, return_tensors='pt')['input_ids'].to('cuda')
    if not torch.equal(full, torch.cat((prefix, target), dim=1)):
        raise ValueError('Target tokenization does not match prompt boundary')
    ids = torch.cat((prefix, target[:, :-1]), dim=1)
    result = model(pixel_values=pixels, input_ids=ids, attention_mask=torch.ones_like(ids),
                   image_flags=torch.ones((1, 1), device='cuda', dtype=torch.long), use_cache=False, return_dict=True)
    logits = result.logits[:, prefix.shape[1]-1:prefix.shape[1]-1+target.shape[1], :].float()
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1))
    grad, = torch.autograd.grad(loss, pixels)
    finite = bool(torch.isfinite(grad).all())
    nonzero = int(torch.count_nonzero(grad))
    frozen = all(not p.requires_grad and p.grad is None for p in model.parameters())
    if not finite or not nonzero or not frozen or not bool(torch.isfinite(loss)):
        raise ValueError('Gradient or frozen-parameter check failed')
    report.update(status='passed', sample_id=row['sample_id'], reference=target_text,
                  input='normalized RGB tensor after resizing', shape=list(pixels.shape), dtype=str(pixels.dtype),
                  target_tokens=int(target.shape[1]), loss=float(loss), gradient_finite=finite,
                  gradient_nonzero=nonzero, gradient_max_abs=float(grad.abs().max()), weights_frozen=frozen)
except Exception as error:
    report.update(error_type=type(error).__name__, error=str(error))
    traceback.print_exc()
report['seconds'] = time.perf_counter() - start
if torch.cuda.is_available():
    report['peak_allocated_mib'] = torch.cuda.max_memory_allocated() / 2**20
out.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
print(json.dumps(report, ensure_ascii=False), flush=True)
raise SystemExit(report['status'] != 'passed')
