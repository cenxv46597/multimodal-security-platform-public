"""Comprehensive Single- vs. Dual- vs. Triple-Source & Cross-Family Ablation on the 6 SAR development images.

Generates and evaluates:
  1. Single-source controls (8 steps, 4/255 & 8/255):
     - mplug_single4, mplug_single8 (existing in runs/sar-tisimi-transfer-six-20261119)
     - iv1b_single4, iv1b_single8 (InternVL2.5-1B alone)
     - glm_single4, glm_single8 (GLM-edge-v-2B alone)
  2. Dual-source (mPLUG-1B + InternVL2.5-1B, existing in runs/sar-multisource-transfer-six-20261124):
     - mm_iv1b_sync4, mm_iv1b_sync8, mm_iv1b_relay8
     * Note: When evaluated on 3rd-family target GLM-edge-v-2B, this is a zero-family-overlap cross-family dual-source attack.
  3. Pure Cross-Family Dual-Source (mPLUG-1B + GLM-edge-v-2B -> attacking InternVL2.5-2B & InternVL2.5-1B):
     - cross_sync4, cross_sync8 (0.5 mPLUG-1B + 0.5 GLM-2B at every step)
     - cross_relay8 (Steps 1..4 on mPLUG-1B -> Steps 5..8 on GLM-2B)
  4. Triple-Source Ensemble (1/3 mPLUG-1B + 1/3 InternVL2.5-1B + 1/3 GLM-2B at every step):
     - triple_sync4, triple_sync8

All conditions are evaluated on all 4 models:
  - mplug (SARChat-mPLUG-Owl3-1B)
  - internvl1b (SARChat-InternVL2.5-1B)
  - glm (SARChat-GLM-edge-v-2B)
  - internvl2b (SARChat-InternVL2.5-2B)
"""

import copy
import gc
import json
import os
from pathlib import Path
import sys
import time

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '.venv-internvl/Lib/site-packages'))
sys.path.insert(0, str(ROOT / '.work/internvl2b-deps'))
sys.path.insert(0, str(ROOT / 'scripts'))

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose, InterpolationMode, Normalize, Resize, ToTensor
from transformers import AutoImageProcessor, AutoModel, AutoModelForCausalLM, AutoTokenizer

from evaluate_sarchat_mplug import INVENTORY as MPLUG_INV, SNAPSHOT as MPLUG_SNAP
from run_sar_proxy_candidate import IMAGE_MEAN, IMAGE_SIZE, IMAGE_STD, prepare_prompt
from run_vlm_clean_screen import load_json, sha256

PROTOCOL = ROOT / 'experiments/protocols/sar_tisimi_transfer_six_20261119.json'
SINGLE_MPLUG_DIR = ROOT / 'runs/sar-tisimi-transfer-six-20261119'
DUAL_MM_IV1B_DIR = ROOT / 'runs/sar-multisource-transfer-six-20261124'
OUT_DIR = ROOT / 'runs/sar-multisource-ablation-controls-20261124'

INTERNVL1B_SNAP = ROOT / 'models/SARChat-InternVL2.5-1B/6c963963'
INTERNVL1B_INV = ROOT / 'resources/manifests/sarchat_snapshot_inventory_20261108.json'
INTERNVL2B_SNAP = ROOT / 'models/SARChat-InternVL2.5-2B/master-20261109'
INTERNVL2B_INV = ROOT / 'resources/manifests/sarchat_internvl2b_inventory_20261109.json'
GLM_SNAP = ROOT / 'models/SARChat-GLM-edge-v-2B/master-20261109'
GLM_INV = ROOT / 'resources/manifests/sarchat_glm_edge_inventory_20261109.json'

SCALE_SCHEDULE = [0, 448, 384, 448, 0, 448, 384, 448]
VISUAL_WEIGHT = 10.0
GLM_CAT_IDS = {
    'ship': 15382,
    'aircraft': 25850,
    'car': 2261,
    'tank': 21525,
    'bridge': 23859,
    'harbor': 5485,
}


def make_ti_kernel(size: int = 5, sigma: float = 1.0) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32, device='cuda') - (size - 1) / 2.0
    g1d = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    g2d = torch.outer(g1d, g1d)
    g2d = g2d / g2d.sum()
    return g2d.view(1, 1, size, size)


def scale_pil(arr: np.ndarray, scale_size: int) -> np.ndarray:
    if scale_size <= 0:
        return arr
    h, w = arr.shape
    im = Image.fromarray(arr, mode='L')
    bottleneck = im.resize((scale_size, scale_size), Image.Resampling.BICUBIC)
    restored = bottleneck.resize((w, h), Image.Resampling.BICUBIC)
    return np.array(restored, copy=True)


def scale_tensor(x: torch.Tensor, scale_size: int) -> torch.Tensor:
    if scale_size <= 0:
        return x
    h, w = x.shape[-2:]
    b = F.interpolate(x, size=(scale_size, scale_size), mode='bicubic', align_corners=False, antialias=True).clamp(0, 1)
    return F.interpolate(b, size=(h, w), mode='bicubic', align_corners=False, antialias=True).clamp(0, 1)


def verify_inventory(snap: Path, inv_path: Path):
    inv = load_json(inv_path)
    for e in inv['files']:
        p = (snap / e['path']).resolve()
        expected_size = e.get('size', e.get('bytes'))
        if not p.is_relative_to(snap) or p.stat().st_size != expected_size or sha256(p) != e['sha256']:
            raise ValueError(f'Snapshot mismatch: {p}')


def load_mplug_cpu():
    verify_inventory(MPLUG_SNAP, MPLUG_INV)
    tok = AutoTokenizer.from_pretrained(MPLUG_SNAP, local_files_only=True, trust_remote_code=False)
    model, info = AutoModelForCausalLM.from_pretrained(
        MPLUG_SNAP, local_files_only=True, trust_remote_code=True,
        use_safetensors=True, torch_dtype=torch.float16, attn_implementation='sdpa',
        output_loading_info=True,
    )
    if any(info.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
        raise ValueError('mPLUG load mismatch')
    model.eval().requires_grad_(False)
    processor = model.init_processor(tok)
    return model, tok, processor


def load_internvl_cpu(snap: Path, inv: Path, trust_tok: bool):
    verify_inventory(snap, inv)
    tok = AutoTokenizer.from_pretrained(snap, local_files_only=True, trust_remote_code=trust_tok, use_fast=False)
    model, info = AutoModel.from_pretrained(
        snap, local_files_only=True, trust_remote_code=True,
        use_safetensors=True, torch_dtype=torch.float16, attn_implementation='eager',
        use_flash_attn=False, output_loading_info=True,
    )
    if any(info.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
        raise ValueError(f'InternVL load mismatch: {snap}')
    model.eval().requires_grad_(False)
    return model, tok


def load_glm_cpu():
    verify_inventory(GLM_SNAP, GLM_INV)
    tok = AutoTokenizer.from_pretrained(GLM_SNAP, local_files_only=True, trust_remote_code=False)
    proc = AutoImageProcessor.from_pretrained(GLM_SNAP, local_files_only=True, use_fast=False)
    model, info = AutoModelForCausalLM.from_pretrained(
        GLM_SNAP, local_files_only=True, trust_remote_code=True,
        use_safetensors=True, torch_dtype=torch.float16, attn_implementation='sdpa',
        output_loading_info=True,
    )
    if any(info.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
        raise ValueError('GLM load mismatch')
    model.eval().requires_grad_(False)
    return model, tok, proc


def clear_mplug_cache(model):
    for m in model.modules():
        if isinstance(getattr(m, 'visual_cache', None), dict):
            m.visual_cache.clear()


def mplug_native(processor, arr: np.ndarray, messages):
    rgb = Image.fromarray(arr.astype(np.uint8), mode='L').convert('RGB')
    selected = processor.image_processor.resizer(rgb, skip_resize=True)
    grid = processor.image_processor.anchors[int(selected)]
    inp = processor(copy.deepcopy(messages), images=[rgb], videos=None, cut_enable=True).to('cuda')
    if tuple(grid) != (2, 2) or tuple(inp.pixel_values.shape) != (5, 3, 384, 384):
        raise ValueError('Unexpected mPLUG preprocessing grid')
    return inp


def mplug_visual(model, p: torch.Tensor) -> torch.Tensor:
    return model.vision2text_model(model.vision_model(p, output_hidden_states=True).hidden_states[-2])


def mplug_preprocess(x: torch.Tensor) -> torch.Tensor:
    image = x.repeat(1, 3, 1, 1)
    big = F.interpolate(image, size=(768, 768), mode='bicubic', align_corners=False, antialias=True).clamp(0, 1)
    glob = F.interpolate(image, size=(384, 384), mode='bicubic', align_corners=False, antialias=True).clamp(0, 1)
    return (torch.cat([big[:, :, y:y + 384, z:z + 384] for y in (0, 384) for z in (0, 384)] + [glob], dim=0) - 0.5) / 0.5


def mplug_clean(model, tok, processor, arr: np.ndarray, messages):
    inp = mplug_native(processor, arr, messages)
    with torch.no_grad():
        feature = model.forward_image(inp.pixel_values)
        ids = model.language_model.generate(
            input_ids=inp.input_ids, image_embeds=feature, media_offset=inp.media_offset,
            attention_mask=inp.get('attention_mask'), pad_token_id=0,
            eos_token_id=[tok.convert_tokens_to_ids(t) for t in model.terminators],
            do_sample=False, max_new_tokens=32,
        )
        tokens = ids[:, inp.input_ids.shape[1]:]
        answer = tok.decode(tokens[0], skip_special_tokens=True).strip()
        end = {tok.convert_tokens_to_ids(t) for t in model.terminators}
        while tokens.shape[1] and int(tokens[0, -1]) in end:
            tokens = tokens[:, :-1]
        clear_mplug_cache(model)
        return answer, tokens.detach().cpu(), feature.detach().float().cpu()


def mplug_step_grad(model, processor, arr: np.ndarray, messages, target_cpu: torch.Tensor,
                    clean_token_id: int, adv_token_id: int, clean_vis_cpu: torch.Tensor,
                    scale_size: int, ti_kernel: torch.Tensor):
    target = target_cpu.to('cuda')
    clean_vis = clean_vis_cpu.to('cuda')
    arr_view = scale_pil(arr, scale_size)
    inp = mplug_native(processor, arr_view, messages)
    x = torch.tensor(arr, device='cuda', dtype=torch.float32).div(255).view(1, 1, *arr.shape).requires_grad_(True)
    approx = mplug_preprocess(scale_tensor(x, scale_size))
    nat = inp.pixel_values.float().detach()
    tiles = nat + (approx - approx.detach())
    with torch.no_grad():
        alt = mplug_visual(model, tiles.detach().to(torch.float16))
    clear_mplug_cache(model)
    feat = alt.detach().requires_grad_(True)
    prefix = target[:, :-1]
    length = inp.input_ids.shape[1]
    full = torch.cat((inp.input_ids, prefix), dim=1)
    mask = inp.get('attention_mask')
    if mask is None:
        mask = torch.ones_like(inp.input_ids)
    mask = torch.cat((mask, torch.ones_like(prefix)), dim=1)
    offset = torch.cat((inp.media_offset, inp.media_offset[:, -1:, :].expand(-1, prefix.shape[1], -1)), dim=1)
    output = model.language_model(
        input_ids=full, image_embeds=feat, media_offset=offset,
        attention_mask=mask, use_cache=False, return_dict=True,
    )
    scores = output.logits[:, length - 1:length - 1 + target.shape[1], :].float()
    cat_logits = scores[:, 2, :]
    clean_t = torch.tensor([clean_token_id], device='cuda', dtype=torch.long)
    adv_t = torch.tensor([adv_token_id], device='cuda', dtype=torch.long)
    loss_cat = F.cross_entropy(cat_logits, clean_t) - F.cross_entropy(cat_logits, adv_t)
    v_norm = F.normalize(feat.float(), dim=-1)
    c_norm = F.normalize(clean_vis, dim=-1)
    loss_vis = 1.0 - (v_norm * c_norm).sum(dim=-1).mean()
    loss = loss_cat + VISUAL_WEIGHT * loss_vis
    fg, = torch.autograd.grad(loss, feat)
    clear_mplug_cache(model)
    grads = []
    for i in range(5):
        tile = tiles[i:i + 1].detach().to(torch.float16).requires_grad_(True)
        v = mplug_visual(model, tile)
        g, = torch.autograd.grad(v, tile, grad_outputs=fg[i:i + 1])
        grads.append(g.detach().float())
        del tile, v, g
    raw_grad, = torch.autograd.grad(tiles, x, grad_outputs=torch.cat(grads, dim=0))
    ti_grad = F.conv2d(raw_grad, ti_kernel, padding=2)
    g_np = ti_grad.detach().cpu().numpy()[0, 0]
    clear_mplug_cache(model)
    return g_np, {'loss_cat': float(loss_cat), 'loss_vis': float(loss_vis), 'total_loss': float(loss)}


def internvl_native(tf, arr: np.ndarray) -> torch.Tensor:
    return tf(Image.fromarray(arr.astype(np.uint8), mode='L').convert('RGB')).unsqueeze(0).to('cuda', dtype=torch.float16)


def internvl_clean(model, tok, tf, inputs_gpu, sep, meta, arr: np.ndarray):
    pv = internvl_native(tf, arr)
    with torch.no_grad():
        feat = model.extract_feature(pv).detach().float().cpu()
        res = model.generate(
            pixel_values=pv, **inputs_gpu, do_sample=False, max_new_tokens=32,
            eos_token_id=meta['eos_token_id'], return_dict_in_generate=True,
        )
        tokens = res.sequences
        ans = tok.decode(tokens[0].cpu().tolist(), skip_special_tokens=True).strip().split(sep, 1)[0]
        while tokens.shape[1] and int(tokens[0, -1]) == meta['eos_token_id']:
            tokens = tokens[:, :-1]
        return ans, tokens.detach().cpu(), feat


def internvl_step_grad(model, tf, inputs_gpu, arr: np.ndarray, target_cpu: torch.Tensor,
                       clean_token_id: int, adv_token_id: int, clean_vis_cpu: torch.Tensor,
                       scale_size: int, ti_kernel: torch.Tensor):
    target = target_cpu.to('cuda')
    clean_vis = clean_vis_cpu.to('cuda')
    arr_view = scale_pil(arr, scale_size)
    pv = internvl_native(tf, arr_view)
    with torch.no_grad():
        feat0 = model.extract_feature(pv)
    feat = feat0.detach().requires_grad_(True)
    prefix = target[:, :-1]
    fullids = torch.cat((inputs_gpu['input_ids'], prefix), dim=1)
    fullmask = torch.cat((inputs_gpu['attention_mask'], torch.ones_like(prefix)), dim=1)
    emb = model.language_model.get_input_embeddings()(fullids).detach().clone()
    selected = fullids == model.img_context_token_id
    emb[selected] = feat.reshape(-1, feat.shape[-1])
    out = model.language_model(inputs_embeds=emb, attention_mask=fullmask, use_cache=False, return_dict=True)
    length = inputs_gpu['input_ids'].shape[1]
    scores = out.logits[:, length - 1:length - 1 + target.shape[1], :].float()
    cat_logits = scores[:, 2, :]
    clean_t = torch.tensor([clean_token_id], device='cuda', dtype=torch.long)
    adv_t = torch.tensor([adv_token_id], device='cuda', dtype=torch.long)
    loss_cat = F.cross_entropy(cat_logits, clean_t) - F.cross_entropy(cat_logits, adv_t)
    v_norm = F.normalize(feat.float(), dim=-1)
    c_norm = F.normalize(clean_vis, dim=-1)
    loss_vis = 1.0 - (v_norm * c_norm).sum(dim=-1).mean()
    loss = loss_cat + VISUAL_WEIGHT * loss_vis
    fg, = torch.autograd.grad(loss, feat)

    x = torch.tensor(arr, device='cuda', dtype=torch.float32).div(255).view(1, 1, *arr.shape).requires_grad_(True)
    scaled_x = scale_tensor(x, scale_size)
    sur = F.interpolate(scaled_x.repeat(1, 3, 1, 1), size=(IMAGE_SIZE, IMAGE_SIZE), mode='bicubic', align_corners=False, antialias=True).clamp(0, 1)
    mean = torch.tensor(IMAGE_MEAN, device='cuda').view(1, 3, 1, 1)
    std = torch.tensor(IMAGE_STD, device='cuda').view(1, 3, 1, 1)
    sur = (sur - mean) / std
    nat_f = pv.float().detach()
    tiles = nat_f + (sur - sur.detach())
    feat2 = model.extract_feature(tiles.to(torch.float16))
    raw_grad, = torch.autograd.grad(feat2, x, grad_outputs=fg)
    ti_grad = F.conv2d(raw_grad, ti_kernel, padding=2)
    g_np = ti_grad.detach().cpu().numpy()[0, 0]
    return g_np, {'loss_cat': float(loss_cat), 'loss_vis': float(loss_vis), 'total_loss': float(loss)}


def glm_native(proc, arr: np.ndarray) -> torch.Tensor:
    rgb = Image.fromarray(arr.astype(np.uint8), mode='L').convert('RGB')
    return proc(rgb, return_tensors='pt')['pixel_values'].to('cuda', dtype=torch.float16)


def glm_clean(model, tok, proc, inp_gpu, arr: np.ndarray):
    pv = glm_native(proc, arr)
    imgs = pv.reshape(1, 3, 672, 672)
    with torch.no_grad():
        vis = model.model.vision(imgs).detach().float().cpu()
        gen = model.generate(**inp_gpu, pixel_values=pv, do_sample=False, max_new_tokens=32, return_dict_in_generate=True)
        toks = gen.sequences[:, inp_gpu.input_ids.shape[1]:]
        ans = tok.decode(toks[0].cpu().tolist(), skip_special_tokens=True).strip()
        return ans, toks.detach().cpu(), vis


def glm_step_grad(model, proc, inp_gpu, arr: np.ndarray, clean_cat: str, target_cat: str,
                  clean_vis_cpu: torch.Tensor, scale_size: int, ti_kernel: torch.Tensor):
    clean_vis = clean_vis_cpu.to('cuda')
    arr_view = scale_pil(arr, scale_size)
    pv = glm_native(proc, arr_view)
    imgs = pv.reshape(1, 3, 672, 672)
    with torch.no_grad():
        feat0 = model.model.vision(imgs)
    feat = feat0.detach().requires_grad_(True)

    prefix = torch.tensor([[2089, 398]], device='cuda', dtype=torch.long)
    full_ids = torch.cat((inp_gpu.input_ids, prefix), dim=1)
    full_mask = torch.cat((inp_gpu.attention_mask, torch.ones_like(prefix)), dim=1)
    id_list = full_ids[0].tolist()
    boi_pos = id_list.index(model.config.boi_token_id)
    num_img_toks = id_list.count(model.config.boi_token_id)

    base_embeds = model.model.embed_tokens(full_ids).detach()
    inputs_embeds = torch.cat((base_embeds[:, :boi_pos], feat, base_embeds[:, boi_pos + num_img_toks:]), dim=1)

    cache_pos = torch.arange(0, inputs_embeds.shape[1], device='cuda')
    pos_ids = cache_pos.unsqueeze(0)
    causal_mask = model.model._update_causal_mask(full_mask, inputs_embeds, cache_pos, None, False)
    hs = inputs_embeds
    pos_emb = model.model.rotary_emb(hs, pos_ids)
    for layer in model.model.layers:
        hs = layer(
            hs, attention_mask=causal_mask, position_ids=pos_ids, past_key_value=None,
            output_attentions=False, use_cache=False, cache_position=cache_pos, position_embeddings=pos_emb,
        )[0]
    hs = model.model.norm(hs)
    cat_logits = model.lm_head(hs[:, -1:, :])[:, 0, :].float()
    clean_t = torch.tensor([GLM_CAT_IDS[clean_cat]], device='cuda', dtype=torch.long)
    adv_t = torch.tensor([GLM_CAT_IDS[target_cat]], device='cuda', dtype=torch.long)
    loss_cat = F.cross_entropy(cat_logits, clean_t) - F.cross_entropy(cat_logits, adv_t)
    v_norm = F.normalize(feat.float(), dim=-1)
    c_norm = F.normalize(clean_vis, dim=-1)
    loss_vis = 1.0 - (v_norm * c_norm).sum(dim=-1).mean()
    loss = loss_cat + VISUAL_WEIGHT * loss_vis
    fg, = torch.autograd.grad(loss, feat)

    x = torch.tensor(arr, device='cuda', dtype=torch.float32).div(255).view(1, 1, *arr.shape).requires_grad_(True)
    scaled_x = scale_tensor(x, scale_size)
    sur = (F.interpolate(scaled_x.repeat(1, 3, 1, 1), size=(672, 672), mode='bicubic', align_corners=False, antialias=True).clamp(0, 1) - 0.5) / 0.5
    nat_f = imgs.float().detach()
    tiles = nat_f + (sur - sur.detach())
    feat2 = model.model.vision(tiles.to(torch.float16))
    raw_grad, = torch.autograd.grad(feat2, x, grad_outputs=fg)
    ti_grad = F.conv2d(raw_grad, ti_kernel, padding=2)
    g_np = ti_grad.detach().cpu().numpy()[0, 0]
    return g_np, {'loss_cat': float(loss_cat), 'loss_vis': float(loss_vis), 'total_loss': float(loss)}


def save_and_verify_png(dest: Path, arr: np.ndarray, original: np.ndarray, max_budget: int):
    with dest.open('xb') as f:
        Image.fromarray(arr, mode='L').save(f, format='PNG')
    with Image.open(dest) as im:
        dec = np.array(im.convert('L'), copy=True)
    if not np.array_equal(dec, arr):
        raise ValueError(f'PNG roundtrip mismatch: {dest}')
    diff = np.abs(dec.astype(np.int16) - original)
    mx = int(diff.max())
    if mx > max_budget:
        raise ValueError(f'Budget exceeded on {dest}: {mx} > {max_budget}')
    return {
        'path': dest.relative_to(ROOT).as_posix(),
        'sha256': sha256(dest),
        'linf_8bit': mx,
        'mean_abs_difference_8bit': round(float(diff.mean()), 4),
        'changed_pixels': int(np.count_nonzero(diff)),
    }


def main():
    t0 = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=False)
    plan = load_json(PROTOCOL)
    question = plan['question']
    rows = plan['rows']
    torch.set_num_threads(4)
    ti_kernel = make_ti_kernel(5, 1.0)
    tf = Compose([Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=InterpolationMode.BICUBIC),
                  ToTensor(), Normalize(IMAGE_MEAN, IMAGE_STD)])
    mplug_messages = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': '<|image|>\n' + question},
        {'role': 'assistant', 'content': ''},
    ]
    glm_messages = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': question}]}]

    originals = {}
    mplug4_init = {}
    for r in rows:
        sid = r['sample_id']
        idx = r['index']
        p = ROOT / r['image_path']
        if sha256(p) != r['image_sha256']:
            raise ValueError(f'Hash mismatch: {sid}')
        with Image.open(p) as im:
            a = np.array(im.convert('L'), copy=True).astype(np.int16)
        originals[sid] = a
        with Image.open(SINGLE_MPLUG_DIR / f'{idx}_tisimi4.png') as im4:
            mplug4_init[sid] = np.array(im4.convert('L'), copy=True).astype(np.int16)

    print('phase: loading 3 source models to CPU RAM', flush=True)
    mplug_model, mplug_tok, mplug_proc = load_mplug_cpu()
    iv1b_model, iv1b_tok = load_internvl_cpu(INTERNVL1B_SNAP, INTERNVL1B_INV, trust_tok=False)
    iv1b_inputs_cpu, iv1b_sep, iv1b_meta = prepare_prompt(iv1b_model, iv1b_tok, question)
    glm_model, glm_tok, glm_proc = load_glm_cpu()
    glm_inp_cpu = glm_tok.apply_chat_template(glm_messages, add_generation_prompt=True, return_dict=True, tokenize=True, return_tensors='pt')

    # 0. Clean forwards for all 3 source models
    print('phase: clean forward on mPLUG-1B', flush=True)
    mplug_model.to('cuda')
    mplug_clean_state = {}
    for r in rows:
        sid = r['sample_id']
        ans, toks, vis = mplug_clean(mplug_model, mplug_tok, mplug_proc, originals[sid].astype(np.uint8), mplug_messages)
        mplug_clean_state[sid] = (ans, toks, vis)
    mplug_model.to('cpu')
    torch.cuda.empty_cache()

    print('phase: clean forward on InternVL2.5-1B + Control 1 (InternVL2.5-1B Single-Source 8 steps)', flush=True)
    iv1b_model.to('cuda')
    iv1b_inputs_gpu = {k: v.to('cuda') for k, v in iv1b_inputs_cpu.items()}
    iv1b_clean_state = {}
    iv1b_curr = {r['sample_id']: originals[r['sample_id']].copy() for r in rows}
    iv1b_mom = {r['sample_id']: np.zeros_like(originals[r['sample_id']], dtype=np.float32) for r in rows}
    iv1b_4 = {}
    for r in rows:
        sid = r['sample_id']
        ans, toks, vis = internvl_clean(iv1b_model, iv1b_tok, tf, iv1b_inputs_gpu, iv1b_sep, iv1b_meta, originals[sid].astype(np.uint8))
        iv1b_clean_state[sid] = (ans, toks, vis)

    for step, scale_size in enumerate(SCALE_SCHEDULE):
        step_size = 1 if step < 4 else 2
        bound = 4 if step < 4 else 8
        for r in rows:
            sid = r['sample_id']
            _, toks, vis = iv1b_clean_state[sid]
            g, _ = internvl_step_grad(
                iv1b_model, tf, iv1b_inputs_gpu, iv1b_curr[sid].astype(np.uint8),
                toks, r['clean_token_id'], r['target_token_id'], vis, scale_size, ti_kernel,
            )
            iv1b_mom[sid] = iv1b_mom[sid] + g / float(np.mean(np.abs(g)))
            delta = step_size * np.sign(iv1b_mom[sid]).astype(np.int16)
            iv1b_curr[sid] = np.clip(np.clip(iv1b_curr[sid] + delta, originals[sid] - bound, originals[sid] + bound), 0, 255).astype(np.int16)
            if step == 3:
                iv1b_4[sid] = iv1b_curr[sid].astype(np.uint8).copy()
        print(f'  iv1b_single step {step + 1}/8 done', flush=True)
    iv1b_model.to('cpu')
    torch.cuda.empty_cache()

    print('phase: clean forward on GLM-2B + GLM Single-Source (8 steps) + Cross-Family Relay (steps 5..8 on GLM-2B)', flush=True)
    glm_model.to('cuda')
    glm_inp_gpu = glm_inp_cpu.to('cuda')
    glm_clean_state = {}
    glm_curr = {r['sample_id']: originals[r['sample_id']].copy() for r in rows}
    glm_mom = {r['sample_id']: np.zeros_like(originals[r['sample_id']], dtype=np.float32) for r in rows}
    glm_4 = {}
    cross_relay_curr = {r['sample_id']: mplug4_init[r['sample_id']].copy() for r in rows}
    cross_relay_mom = {r['sample_id']: np.zeros_like(originals[r['sample_id']], dtype=np.float32) for r in rows}

    for r in rows:
        sid = r['sample_id']
        ans, toks, vis = glm_clean(glm_model, glm_tok, glm_proc, glm_inp_gpu, originals[sid].astype(np.uint8))
        glm_clean_state[sid] = (ans, toks, vis)

    for step, scale_size in enumerate(SCALE_SCHEDULE):
        t_s = time.perf_counter()
        step_size = 1 if step < 4 else 2
        bound = 4 if step < 4 else 8
        for r in rows:
            sid = r['sample_id']
            _, _, vis = glm_clean_state[sid]
            g, _ = glm_step_grad(
                glm_model, glm_proc, glm_inp_gpu, glm_curr[sid].astype(np.uint8),
                r['reference'], r['target_category'], vis, scale_size, ti_kernel,
            )
            glm_mom[sid] = glm_mom[sid] + g / float(np.mean(np.abs(g)))
            delta = step_size * np.sign(glm_mom[sid]).astype(np.int16)
            glm_curr[sid] = np.clip(np.clip(glm_curr[sid] + delta, originals[sid] - bound, originals[sid] + bound), 0, 255).astype(np.int16)
            if step == 3:
                glm_4[sid] = glm_curr[sid].astype(np.uint8).copy()

            if step >= 4:
                gr, _ = glm_step_grad(
                    glm_model, glm_proc, glm_inp_gpu, cross_relay_curr[sid].astype(np.uint8),
                    r['reference'], r['target_category'], vis, scale_size, ti_kernel,
                )
                cross_relay_mom[sid] = cross_relay_mom[sid] + gr / float(np.mean(np.abs(gr)))
                dr = 2 * np.sign(cross_relay_mom[sid]).astype(np.int16)
                cross_relay_curr[sid] = np.clip(np.clip(cross_relay_curr[sid] + dr, originals[sid] - 8, originals[sid] + 8), 0, 255).astype(np.int16)
        print(f'  glm_single (+ cross_relay) step {step + 1}/8 done in {round(time.perf_counter() - t_s, 2)}s', flush=True)
    glm_model.to('cpu')
    torch.cuda.empty_cache()

    # Phase 3: Synchronous Cross-Family Dual-Source (mPLUG + GLM) & Triple-Source (mPLUG + InternVL1B + GLM)
    print('phase: Synchronous Cross-Family Dual-Source (mPLUG+GLM) & Triple-Source (mPLUG+1B+GLM) steps 1..8', flush=True)
    cross_sync_curr = {r['sample_id']: originals[r['sample_id']].copy() for r in rows}
    cross_sync_mom = {r['sample_id']: np.zeros_like(originals[r['sample_id']], dtype=np.float32) for r in rows}
    cross_sync_4 = {}

    triple_sync_curr = {r['sample_id']: originals[r['sample_id']].copy() for r in rows}
    triple_sync_mom = {r['sample_id']: np.zeros_like(originals[r['sample_id']], dtype=np.float32) for r in rows}
    triple_sync_4 = {}

    for step, scale_size in enumerate(SCALE_SCHEDULE):
        t_s = time.perf_counter()
        # 1) mPLUG gradients for cross_sync and triple_sync
        mplug_model.to('cuda')
        g_mp_cross, g_mp_triple = {}, {}
        for r in rows:
            sid = r['sample_id']
            _, toks, vis = mplug_clean_state[sid]
            g1, _ = mplug_step_grad(
                mplug_model, mplug_proc, cross_sync_curr[sid].astype(np.uint8), mplug_messages,
                toks, r['clean_token_id'], r['target_token_id'], vis, scale_size, ti_kernel,
            )
            g_mp_cross[sid] = g1 / float(np.mean(np.abs(g1)))
            if np.array_equal(cross_sync_curr[sid], triple_sync_curr[sid]):
                g_mp_triple[sid] = g_mp_cross[sid]
            else:
                g2, _ = mplug_step_grad(
                    mplug_model, mplug_proc, triple_sync_curr[sid].astype(np.uint8), mplug_messages,
                    toks, r['clean_token_id'], r['target_token_id'], vis, scale_size, ti_kernel,
                )
                g_mp_triple[sid] = g2 / float(np.mean(np.abs(g2)))
        mplug_model.to('cpu')
        torch.cuda.empty_cache()

        # 2) InternVL-1B gradients for triple_sync
        iv1b_model.to('cuda')
        g_1b_triple = {}
        for r in rows:
            sid = r['sample_id']
            _, toks, vis = iv1b_clean_state[sid]
            g, _ = internvl_step_grad(
                iv1b_model, tf, iv1b_inputs_gpu, triple_sync_curr[sid].astype(np.uint8),
                toks, r['clean_token_id'], r['target_token_id'], vis, scale_size, ti_kernel,
            )
            g_1b_triple[sid] = g / float(np.mean(np.abs(g)))
        iv1b_model.to('cpu')
        torch.cuda.empty_cache()

        # 3) GLM-2B gradients for cross_sync and triple_sync
        glm_model.to('cuda')
        g_glm_cross, g_glm_triple = {}, {}
        for r in rows:
            sid = r['sample_id']
            _, _, vis = glm_clean_state[sid]
            g1, _ = glm_step_grad(
                glm_model, glm_proc, glm_inp_gpu, cross_sync_curr[sid].astype(np.uint8),
                r['reference'], r['target_category'], vis, scale_size, ti_kernel,
            )
            g_glm_cross[sid] = g1 / float(np.mean(np.abs(g1)))
            if np.array_equal(cross_sync_curr[sid], triple_sync_curr[sid]):
                g_glm_triple[sid] = g_glm_cross[sid]
            else:
                g2, _ = glm_step_grad(
                    glm_model, glm_proc, glm_inp_gpu, triple_sync_curr[sid].astype(np.uint8),
                    r['reference'], r['target_category'], vis, scale_size, ti_kernel,
                )
                g_glm_triple[sid] = g2 / float(np.mean(np.abs(g2)))
        glm_model.to('cpu')
        torch.cuda.empty_cache()

        # 4) Update both trajectories
        step_size = 1 if step < 4 else 2
        bound = 4 if step < 4 else 8
        for r in rows:
            sid = r['sample_id']
            # Cross-family dual-source (0.5 mPLUG + 0.5 GLM)
            fc = 0.5 * g_mp_cross[sid] + 0.5 * g_glm_cross[sid]
            cross_sync_mom[sid] = cross_sync_mom[sid] + fc
            dc = step_size * np.sign(cross_sync_mom[sid]).astype(np.int16)
            cross_sync_curr[sid] = np.clip(np.clip(cross_sync_curr[sid] + dc, originals[sid] - bound, originals[sid] + bound), 0, 255).astype(np.int16)
            if step == 3:
                cross_sync_4[sid] = cross_sync_curr[sid].astype(np.uint8).copy()

            # Triple-source (1/3 mPLUG + 1/3 InternVL-1B + 1/3 GLM)
            ft = (g_mp_triple[sid] + g_1b_triple[sid] + g_glm_triple[sid]) / 3.0
            triple_sync_mom[sid] = triple_sync_mom[sid] + ft
            dt = step_size * np.sign(triple_sync_mom[sid]).astype(np.int16)
            triple_sync_curr[sid] = np.clip(np.clip(triple_sync_curr[sid] + dt, originals[sid] - bound, originals[sid] + bound), 0, 255).astype(np.int16)
            if step == 3:
                triple_sync_4[sid] = triple_sync_curr[sid].astype(np.uint8).copy()

        print(f'  sync step {step + 1}/8 done in {round(time.perf_counter() - t_s, 2)}s', flush=True)

    # Save all new assets
    assets = {r['sample_id']: {} for r in rows}
    for r in rows:
        sid = r['sample_id']
        idx = r['index']
        orig = originals[sid]
        assets[sid]['iv1b_single4'] = save_and_verify_png(OUT_DIR / f'{idx}_iv1b_single4.png', iv1b_4[sid], orig, 4)
        assets[sid]['iv1b_single8'] = save_and_verify_png(OUT_DIR / f'{idx}_iv1b_single8.png', iv1b_curr[sid].astype(np.uint8), orig, 8)
        assets[sid]['glm_single4'] = save_and_verify_png(OUT_DIR / f'{idx}_glm_single4.png', glm_4[sid], orig, 4)
        assets[sid]['glm_single8'] = save_and_verify_png(OUT_DIR / f'{idx}_glm_single8.png', glm_curr[sid].astype(np.uint8), orig, 8)
        assets[sid]['cross_relay8'] = save_and_verify_png(OUT_DIR / f'{idx}_cross_relay8.png', cross_relay_curr[sid].astype(np.uint8), orig, 8)
        assets[sid]['cross_sync4'] = save_and_verify_png(OUT_DIR / f'{idx}_cross_sync4.png', cross_sync_4[sid], orig, 4)
        assets[sid]['cross_sync8'] = save_and_verify_png(OUT_DIR / f'{idx}_cross_sync8.png', cross_sync_curr[sid].astype(np.uint8), orig, 8)
        assets[sid]['triple_sync4'] = save_and_verify_png(OUT_DIR / f'{idx}_triple_sync4.png', triple_sync_4[sid], orig, 4)
        assets[sid]['triple_sync8'] = save_and_verify_png(OUT_DIR / f'{idx}_triple_sync8.png', triple_sync_curr[sid].astype(np.uint8), orig, 8)

    # Build complete evaluation paths per sample (including prior single-source mPLUG & prior dual-source mPLUG+1B & random8)
    eval_paths = {r['sample_id']: {} for r in rows}
    for r in rows:
        sid = r['sample_id']
        idx = r['index']
        for k, v in assets[sid].items():
            eval_paths[sid][k] = ROOT / v['path']
        # Also include existing assets so we evaluate them on GLM-2B and InternVL-1B
        eval_paths[sid]['random8'] = SINGLE_MPLUG_DIR / f'{idx}_random8.png'
        eval_paths[sid]['mplug_single4'] = SINGLE_MPLUG_DIR / f'{idx}_tisimi4.png'
        eval_paths[sid]['mplug_single8'] = SINGLE_MPLUG_DIR / f'{idx}_tisimi8.png'
        eval_paths[sid]['mm_iv1b_relay8'] = DUAL_MM_IV1B_DIR / f'{idx}_relay8.png'
        eval_paths[sid]['mm_iv1b_sync4'] = DUAL_MM_IV1B_DIR / f'{idx}_sync4.png'
        eval_paths[sid]['mm_iv1b_sync8'] = DUAL_MM_IV1B_DIR / f'{idx}_sync8.png'

    new_conds = list(assets[rows[0]['sample_id']].keys())
    all_conds = list(eval_paths[rows[0]['sample_id']].keys())
    answers = {r['sample_id']: {'mplug': {}, 'internvl1b': {}, 'glm': {}, 'internvl2b': {}} for r in rows}

    print('phase: evaluating on mPLUG-1B', flush=True)
    mplug_model.to('cuda')
    for r in rows:
        sid = r['sample_id']
        for c in new_conds:
            with Image.open(eval_paths[sid][c]) as im:
                arr = np.array(im.convert('L'), copy=True)
            ans, _, _ = mplug_clean(mplug_model, mplug_tok, mplug_proc, arr, mplug_messages)
            answers[sid]['mplug'][c] = ans
    del mplug_model
    gc.collect()
    torch.cuda.empty_cache()

    print('phase: evaluating on InternVL2.5-1B', flush=True)
    iv1b_model.to('cuda')
    for r in rows:
        sid = r['sample_id']
        for c in all_conds:
            with Image.open(eval_paths[sid][c]) as im:
                arr = np.array(im.convert('L'), copy=True)
            ans, _, _ = internvl_clean(iv1b_model, iv1b_tok, tf, iv1b_inputs_gpu, iv1b_sep, iv1b_meta, arr)
            answers[sid]['internvl1b'][c] = ans
    del iv1b_model
    gc.collect()
    torch.cuda.empty_cache()

    print('phase: evaluating on GLM-edge-v-2B', flush=True)
    glm_model.to('cuda')
    for r in rows:
        sid = r['sample_id']
        for c in all_conds:
            with Image.open(eval_paths[sid][c]) as im:
                arr = np.array(im.convert('L'), copy=True)
            ans, _, _ = glm_clean(glm_model, glm_tok, glm_proc, glm_inp_gpu, arr)
            answers[sid]['glm'][c] = ans
    del glm_model
    gc.collect()
    torch.cuda.empty_cache()

    print('phase: evaluating on Black-Box Target InternVL2.5-2B', flush=True)
    iv2b_model, iv2b_tok = load_internvl_cpu(INTERNVL2B_SNAP, INTERNVL2B_INV, trust_tok=True)
    iv2b_inputs_cpu, iv2b_sep, iv2b_meta = prepare_prompt(iv2b_model, iv2b_tok, question)
    iv2b_model.to('cuda')
    iv2b_inputs_gpu = {k: v.to('cuda') for k, v in iv2b_inputs_cpu.items()}
    for r in rows:
        sid = r['sample_id']
        for c in new_conds:
            with Image.open(eval_paths[sid][c]) as im:
                arr = np.array(im.convert('L'), copy=True)
            ans, _, _ = internvl_clean(iv2b_model, iv2b_tok, tf, iv2b_inputs_gpu, iv2b_sep, iv2b_meta, arr)
            answers[sid]['internvl2b'][c] = ans
    del iv2b_model
    gc.collect()
    torch.cuda.empty_cache()

    # Merge previously recorded answers for mplug & internvl2b on existing conditions
    prior_single = load_json(SINGLE_MPLUG_DIR / 'semantic_results.json')
    prior_dual = load_json(DUAL_MM_IV1B_DIR / 'summary.json')
    dual_by_sid = {x['sample_id']: x for x in prior_dual['items']}
    for m_key in ('mplug', 'internvl2b'):
        by_sid = {x['sample_id']: x for x in prior_single['models'][m_key]['items']}
        for r in rows:
            sid = r['sample_id']
            answers[sid][m_key]['random8'] = by_sid[sid]['conditions']['random8']['raw_answer']
            answers[sid][m_key]['mplug_single4'] = by_sid[sid]['conditions']['tisimi4']['raw_answer']
            answers[sid][m_key]['mplug_single8'] = by_sid[sid]['conditions']['tisimi8']['raw_answer']
            answers[sid][m_key]['mm_iv1b_relay8'] = dual_by_sid[sid]['answers'][m_key]['relay8']
            answers[sid][m_key]['mm_iv1b_sync4'] = dual_by_sid[sid]['answers'][m_key]['sync4']
            answers[sid][m_key]['mm_iv1b_sync8'] = dual_by_sid[sid]['answers'][m_key]['sync8']

    summary = {
        'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'protocol_sha256': sha256(PROTOCOL),
        'script_sha256': sha256(Path(__file__)),
        'elapsed_seconds': round(time.perf_counter() - t0, 3),
        'items': [
            {
                'sample_id': r['sample_id'],
                'index': r['index'],
                'reference': r['reference'],
                'target_category': r['target_category'],
                'new_assets': assets[r['sample_id']],
                'answers': answers[r['sample_id']],
            }
            for r in rows
        ],
    }
    (OUT_DIR / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print('ALL CONTROLS DONE in', summary['elapsed_seconds'], 's', flush=True)


if __name__ == '__main__':
    main()
