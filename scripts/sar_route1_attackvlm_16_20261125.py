"""Route 1 Pilot: AttackVLM-Style Targeted Visual Feature Prototype Matching (DI-TI-MI, Linf=16/255, 40 steps).
Pure vision-encoder optimization (no LLM logits loss!) to test zero-family-overlap cross-architecture transfer.

Conditions on the 6 SAR development images (all Linf <= 16/255, 8-bit grayscale PNG):
  1. random16       : uniform random noise in [-16, 16] control
  2. avlm_mplug16   : single-source mPLUG-1B vision encoder (40 steps)
  3. avlm_mm_glm16  : zero-InternVL dual-source (0.5 mPLUG-1B + 0.5 GLM-2B vision encoders)
  4. avlm_mm_iv1b16 : zero-GLM dual-source (0.5 mPLUG-1B + 0.5 InternVL2.5-1B vision encoders)
  5. avlm_iv1b_glm16: zero-mPLUG dual-source (0.5 InternVL2.5-1B + 0.5 GLM-2B vision encoders)

All 5 conditions are evaluated across all 4 full VLMs: mPLUG-1B, InternVL2.5-1B, GLM-edge-v-2B, InternVL2.5-2B.
"""

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
sys.path.insert(0, str(ROOT / '.work'))

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose, InterpolationMode, Normalize, Resize, ToTensor

from run_sar_proxy_candidate import IMAGE_MEAN, IMAGE_SIZE, IMAGE_STD, prepare_prompt
from run_vlm_clean_screen import load_json, sha256
from sar_multisource_ablation_controls_20261124 import (
    INTERNVL1B_INV, INTERNVL1B_SNAP, INTERNVL2B_INV, INTERNVL2B_SNAP,
    PROTOCOL,
    clear_mplug_cache, glm_clean, internvl_clean,
    load_glm_cpu, load_internvl_cpu, load_mplug_cpu, make_ti_kernel,
    mplug_clean, mplug_preprocess, mplug_visual, save_and_verify_png, scale_tensor,
)

DRAFT_POOL = ROOT / 'experiments/protocols/sar_mifgsm_expansion_draft_20261117.json'
OUT_DIR = ROOT / 'runs/sar-route1-attackvlm16-20261125'

EPS = 16
STEPS = 16
STEP_SIZE = 2
PROTO_PER_CLASS = 8
SCALE_POOL = [0, 384, 448, 512, 672]


def parse_category(raw: str):
    low = raw.lower()
    found = []
    for c in ('ship', 'aircraft', 'car', 'tank', 'bridge', 'harbor'):
        if c in low:
            found.append(c)
    if len(found) == 1:
        return found[0]
    return 'indeterminate'


def apply_input_diversity(x: torch.Tensor, scale_size: int, shift_y: int, shift_x: int) -> torch.Tensor:
    y = x
    if shift_y != 0 or shift_x != 0:
        y = torch.roll(y, shifts=(shift_y, shift_x), dims=(-2, -1))
    if scale_size > 0:
        y = scale_tensor(y, scale_size)
    return y


def mplug_embed_nograd(model, arr: np.ndarray) -> torch.Tensor:
    x = torch.tensor(arr, device='cuda', dtype=torch.float32).div(255).view(1, 1, *arr.shape)
    tiles = mplug_preprocess(x).to(torch.float16)
    with torch.no_grad():
        feats = []
        for i in range(5):
            feats.append(mplug_visual(model, tiles[i:i + 1]).float().mean(dim=1))
        emb = torch.cat(feats, dim=0).mean(dim=0, keepdim=True)
        clear_mplug_cache(model)
    return F.normalize(emb, dim=-1).detach().cpu()


def mplug_vis_grad(model, arr: np.ndarray, tgt_proto_cpu: torch.Tensor, clean_emb_cpu: torch.Tensor,
                   src_proto_cpu: torch.Tensor, scale_size: int, sy: int, sx: int, ti_kernel: torch.Tensor):
    tgt_p = tgt_proto_cpu.to('cuda')
    cln_e = clean_emb_cpu.to('cuda')
    src_p = src_proto_cpu.to('cuda')
    x = torch.tensor(arr, device='cuda', dtype=torch.float32).div(255).view(1, 1, *arr.shape).requires_grad_(True)
    x_div = apply_input_diversity(x, scale_size, sy, sx)
    tiles = mplug_preprocess(x_div)
    # Two-stage per-tile backward to keep VRAM low
    tile_means = []
    with torch.no_grad():
        for i in range(5):
            tile_means.append(mplug_visual(model, tiles[i:i + 1].detach().to(torch.float16)).float().mean(dim=1))
    clear_mplug_cache(model)
    stacked = torch.cat(tile_means, dim=0).detach().requires_grad_(True)
    emb = F.normalize(stacked.mean(dim=0, keepdim=True), dim=-1)
    loss = (emb * tgt_p).sum() - 0.5 * (emb * cln_e).sum() - 0.5 * (emb * src_p).sum()
    g_stacked, = torch.autograd.grad(loss, stacked)

    tile_grads = []
    for i in range(5):
        t = tiles[i:i + 1].detach().to(torch.float16).requires_grad_(True)
        v = mplug_visual(model, t).float().mean(dim=1)
        g, = torch.autograd.grad(v, t, grad_outputs=g_stacked[i:i + 1])
        tile_grads.append(g.detach().float())
        del t, v, g
    clear_mplug_cache(model)
    raw_grad, = torch.autograd.grad(tiles, x, grad_outputs=torch.cat(tile_grads, dim=0))
    ti_grad = F.conv2d(raw_grad, ti_kernel, padding=2)
    return ti_grad.detach().cpu().numpy()[0, 0], float(loss)


def iv1b_embed_nograd(model, arr: np.ndarray, mean_t: torch.Tensor, std_t: torch.Tensor) -> torch.Tensor:
    x = torch.tensor(arr, device='cuda', dtype=torch.float32).div(255).view(1, 1, *arr.shape)
    sur = F.interpolate(x.repeat(1, 3, 1, 1), size=(IMAGE_SIZE, IMAGE_SIZE), mode='bicubic', align_corners=False, antialias=True).clamp(0, 1)
    sur = (sur - mean_t) / std_t
    with torch.no_grad():
        emb = model.extract_feature(sur.to(torch.float16)).float().mean(dim=1)
    return F.normalize(emb, dim=-1).detach().cpu()


def iv1b_vis_grad(model, arr: np.ndarray, tgt_proto_cpu: torch.Tensor, clean_emb_cpu: torch.Tensor,
                  src_proto_cpu: torch.Tensor, scale_size: int, sy: int, sx: int,
                  mean_t: torch.Tensor, std_t: torch.Tensor, ti_kernel: torch.Tensor):
    tgt_p = tgt_proto_cpu.to('cuda')
    cln_e = clean_emb_cpu.to('cuda')
    src_p = src_proto_cpu.to('cuda')
    x = torch.tensor(arr, device='cuda', dtype=torch.float32).div(255).view(1, 1, *arr.shape).requires_grad_(True)
    x_div = apply_input_diversity(x, scale_size, sy, sx)
    sur = F.interpolate(x_div.repeat(1, 3, 1, 1), size=(IMAGE_SIZE, IMAGE_SIZE), mode='bicubic', align_corners=False, antialias=True).clamp(0, 1)
    sur = (sur - mean_t) / std_t
    v = model.extract_feature(sur.to(torch.float16)).float().mean(dim=1)
    emb = F.normalize(v, dim=-1)
    loss = (emb * tgt_p).sum() - 0.5 * (emb * cln_e).sum() - 0.5 * (emb * src_p).sum()
    raw_grad, = torch.autograd.grad(loss, x)
    ti_grad = F.conv2d(raw_grad, ti_kernel, padding=2)
    return ti_grad.detach().cpu().numpy()[0, 0], float(loss)


def glm_embed_nograd(model, arr: np.ndarray) -> torch.Tensor:
    x = torch.tensor(arr, device='cuda', dtype=torch.float32).div(255).view(1, 1, *arr.shape)
    sur = (F.interpolate(x.repeat(1, 3, 1, 1), size=(672, 672), mode='bicubic', align_corners=False, antialias=True).clamp(0, 1) - 0.5) / 0.5
    with torch.no_grad():
        emb = model.model.vision(sur.to(torch.float16)).float().mean(dim=1)
    return F.normalize(emb, dim=-1).detach().cpu()


def glm_vis_grad(model, arr: np.ndarray, tgt_proto_cpu: torch.Tensor, clean_emb_cpu: torch.Tensor,
                 src_proto_cpu: torch.Tensor, scale_size: int, sy: int, sx: int, ti_kernel: torch.Tensor):
    tgt_p = tgt_proto_cpu.to('cuda')
    cln_e = clean_emb_cpu.to('cuda')
    src_p = src_proto_cpu.to('cuda')
    x = torch.tensor(arr, device='cuda', dtype=torch.float32).div(255).view(1, 1, *arr.shape).requires_grad_(True)
    x_div = apply_input_diversity(x, scale_size, sy, sx)
    sur = (F.interpolate(x_div.repeat(1, 3, 1, 1), size=(672, 672), mode='bicubic', align_corners=False, antialias=True).clamp(0, 1) - 0.5) / 0.5
    v = model.model.vision(sur.to(torch.float16)).float().mean(dim=1)
    emb = F.normalize(v, dim=-1)
    loss = (emb * tgt_p).sum() - 0.5 * (emb * cln_e).sum() - 0.5 * (emb * src_p).sum()
    raw_grad, = torch.autograd.grad(loss, x)
    ti_grad = F.conv2d(raw_grad, ti_kernel, padding=2)
    return ti_grad.detach().cpu().numpy()[0, 0], float(loss)


def main():
    t0 = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plan = load_json(PROTOCOL)
    question = plan['question']
    rows = plan['rows']
    test_sids = {r['sample_id'] for r in rows}

    # Build 8 disjoint prototype images per class from development pool
    pool = load_json(DRAFT_POOL)['rows']
    proto_imgs = {c: [] for c in ('ship', 'aircraft', 'car', 'tank', 'bridge', 'harbor')}
    for p_row in pool:
        c = p_row['reference']
        if p_row['sample_id'] in test_sids or len(proto_imgs[c]) >= PROTO_PER_CLASS:
            continue
        with Image.open(ROOT / p_row['image_path']) as im:
            proto_imgs[c].append(np.array(im.convert('L'), copy=True))
    print('Prototype counts per class:', {k: len(v) for k, v in proto_imgs.items()}, flush=True)

    originals = {}
    for r in rows:
        with Image.open(ROOT / r['image_path']) as im:
            originals[r['sample_id']] = np.array(im.convert('L'), copy=True).astype(np.int16)

    torch.set_num_threads(4)
    ti_kernel = make_ti_kernel(5, 1.0)
    mean_t = torch.tensor(IMAGE_MEAN, device='cuda').view(1, 3, 1, 1)
    std_t = torch.tensor(IMAGE_STD, device='cuda').view(1, 3, 1, 1)
    tf = Compose([Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=InterpolationMode.BICUBIC),
                  ToTensor(), Normalize(IMAGE_MEAN, IMAGE_STD)])
    mplug_messages = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': '<|image|>\n' + question},
        {'role': 'assistant', 'content': ''},
    ]
    glm_messages = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': question}]}]

    print('phase: loading 3 source models to CPU RAM', flush=True)
    mplug_model, mplug_tok, mplug_proc = load_mplug_cpu()
    iv1b_model, iv1b_tok = load_internvl_cpu(INTERNVL1B_SNAP, INTERNVL1B_INV, trust_tok=False)
    iv1b_inputs_cpu, iv1b_sep, iv1b_meta = prepare_prompt(iv1b_model, iv1b_tok, question)
    glm_model, glm_tok, glm_proc = load_glm_cpu()
    glm_inp_cpu = glm_tok.apply_chat_template(glm_messages, add_generation_prompt=True, return_dict=True, tokenize=True, return_tensors='pt')

    # Compute class prototypes & clean embeddings for each vision encoder
    print('phase: computing 8-image class visual prototypes on mPLUG, InternVL-1B, GLM-2B', flush=True)
    mplug_model.to('cuda')
    mp_proto = {c: F.normalize(torch.cat([mplug_embed_nograd(mplug_model, a) for a in arrs], dim=0).mean(dim=0, keepdim=True), dim=-1)
                for c, arrs in proto_imgs.items()}
    mp_clean_emb = {r['sample_id']: mplug_embed_nograd(mplug_model, originals[r['sample_id']].astype(np.uint8)) for r in rows}
    mplug_model.to('cpu')
    torch.cuda.empty_cache()

    iv1b_model.to('cuda')
    iv_proto = {c: F.normalize(torch.cat([iv1b_embed_nograd(iv1b_model, a, mean_t, std_t) for a in arrs], dim=0).mean(dim=0, keepdim=True), dim=-1)
                for c, arrs in proto_imgs.items()}
    iv_clean_emb = {r['sample_id']: iv1b_embed_nograd(iv1b_model, originals[r['sample_id']].astype(np.uint8), mean_t, std_t) for r in rows}
    iv1b_model.to('cpu')
    torch.cuda.empty_cache()

    glm_model.to('cuda')
    gl_proto = {c: F.normalize(torch.cat([glm_embed_nograd(glm_model, a) for a in arrs], dim=0).mean(dim=0, keepdim=True), dim=-1)
                for c, arrs in proto_imgs.items()}
    gl_clean_emb = {r['sample_id']: glm_embed_nograd(glm_model, originals[r['sample_id']].astype(np.uint8)) for r in rows}
    glm_model.to('cpu')
    torch.cuda.empty_cache()

    # Initialize 2 key zero-family-overlap dual-source trajectories + random16
    trajs = ['avlm_mm_glm16', 'avlm_mm_iv1b16']
    curr = {tr: {r['sample_id']: originals[r['sample_id']].copy() for r in rows} for tr in trajs}
    mom = {tr: {r['sample_id']: np.zeros_like(originals[r['sample_id']], dtype=np.float32) for r in rows} for tr in trajs}
    rng_div = np.random.default_rng(2026112500)

    print(f'phase: running {STEPS}-step DI-TI-MI visual prototype matching (Linf={EPS}/255, step={STEP_SIZE}/255)', flush=True)
    for step in range(STEPS):
        t_s = time.perf_counter()
        scale_size = SCALE_POOL[step % len(SCALE_POOL)]
        sy = int(rng_div.integers(-8, 9))
        sx = int(rng_div.integers(-8, 9))

        # 1) mPLUG gradients for both trajectories
        mplug_model.to('cuda')
        g_mp = {tr: {} for tr in ('avlm_mm_glm16', 'avlm_mm_iv1b16')}
        for tr in g_mp:
            for r in rows:
                sid = r['sample_id']
                g, _ = mplug_vis_grad(
                    mplug_model, curr[tr][sid].astype(np.uint8),
                    mp_proto[r['target_category']], mp_clean_emb[sid], mp_proto[r['reference']],
                    scale_size, sy, sx, ti_kernel,
                )
                g_mp[tr][sid] = g / float(np.mean(np.abs(g)))
        mplug_model.to('cpu')
        torch.cuda.empty_cache()

        # 2) InternVL-1B gradients for avlm_mm_iv1b16
        iv1b_model.to('cuda')
        g_iv = {'avlm_mm_iv1b16': {}}
        for r in rows:
            sid = r['sample_id']
            g, _ = iv1b_vis_grad(
                iv1b_model, curr['avlm_mm_iv1b16'][sid].astype(np.uint8),
                iv_proto[r['target_category']], iv_clean_emb[sid], iv_proto[r['reference']],
                scale_size, sy, sx, mean_t, std_t, ti_kernel,
            )
            g_iv['avlm_mm_iv1b16'][sid] = g / float(np.mean(np.abs(g)))
        iv1b_model.to('cpu')
        torch.cuda.empty_cache()

        # 3) GLM-2B gradients for avlm_mm_glm16
        glm_model.to('cuda')
        g_gl = {'avlm_mm_glm16': {}}
        for r in rows:
            sid = r['sample_id']
            g, _ = glm_vis_grad(
                glm_model, curr['avlm_mm_glm16'][sid].astype(np.uint8),
                gl_proto[r['target_category']], gl_clean_emb[sid], gl_proto[r['reference']],
                scale_size, sy, sx, ti_kernel,
            )
            g_gl['avlm_mm_glm16'][sid] = g / float(np.mean(np.abs(g)))
        glm_model.to('cpu')
        torch.cuda.empty_cache()

        # 4) Momentum update + L_inf projection (16/255)
        for r in rows:
            sid = r['sample_id']
            orig = originals[sid]
            fused_map = {
                'avlm_mm_glm16': 0.5 * g_mp['avlm_mm_glm16'][sid] + 0.5 * g_gl['avlm_mm_glm16'][sid],
                'avlm_mm_iv1b16': 0.5 * g_mp['avlm_mm_iv1b16'][sid] + 0.5 * g_iv['avlm_mm_iv1b16'][sid],
            }
            for tr, gf in fused_map.items():
                mom[tr][sid] = mom[tr][sid] + gf
                delta = STEP_SIZE * np.sign(mom[tr][sid]).astype(np.int16)
                curr[tr][sid] = np.clip(np.clip(curr[tr][sid] + delta, orig - EPS, orig + EPS), 0, 255).astype(np.int16)

        print(f'  step {step + 1:02d}/{STEPS} done in {round(time.perf_counter() - t_s, 2)}s (scale={scale_size}, shift=({sy},{sx}))', flush=True)

    # Save random16 + 4 AttackVLM conditions
    conds = ['random16', *trajs]
    assets = {r['sample_id']: {} for r in rows}
    for r in rows:
        sid = r['sample_id']
        idx = r['index']
        orig = originals[sid]
        rng_r = np.random.default_rng(2026112516 + idx)
        rand16 = np.clip(orig + rng_r.integers(-EPS, EPS + 1, size=orig.shape, dtype=np.int16), 0, 255).astype(np.uint8)
        assets[sid]['random16'] = save_and_verify_png(OUT_DIR / f'{idx}_random16.png', rand16, orig, EPS)
        for tr in trajs:
            assets[sid][tr] = save_and_verify_png(OUT_DIR / f'{idx}_{tr}.png', curr[tr][sid].astype(np.uint8), orig, EPS)
    print('Saved and verified all 30 PNGs (5 conditions x 6 images, Linf <= 16/255)', flush=True)

    # Evaluate all 5 conditions on all 4 full VLMs
    answers = {r['sample_id']: {'mplug': {}, 'internvl1b': {}, 'glm': {}, 'internvl2b': {}} for r in rows}

    print('phase: evaluating on mPLUG-1B', flush=True)
    mplug_model.to('cuda')
    for r in rows:
        sid = r['sample_id']
        for c in conds:
            with Image.open(ROOT / assets[sid][c]['path']) as im:
                arr = np.array(im.convert('L'), copy=True)
            ans, _, _ = mplug_clean(mplug_model, mplug_tok, mplug_proc, arr, mplug_messages)
            answers[sid]['mplug'][c] = ans
    del mplug_model
    gc.collect()
    torch.cuda.empty_cache()

    print('phase: evaluating on InternVL2.5-1B', flush=True)
    iv1b_model.to('cuda')
    iv1b_inputs_gpu = {k: v.to('cuda') for k, v in iv1b_inputs_cpu.items()}
    for r in rows:
        sid = r['sample_id']
        for c in conds:
            with Image.open(ROOT / assets[sid][c]['path']) as im:
                arr = np.array(im.convert('L'), copy=True)
            ans, _, _ = internvl_clean(iv1b_model, iv1b_tok, tf, iv1b_inputs_gpu, iv1b_sep, iv1b_meta, arr)
            answers[sid]['internvl1b'][c] = ans
    del iv1b_model
    gc.collect()
    torch.cuda.empty_cache()

    print('phase: evaluating on GLM-edge-v-2B', flush=True)
    glm_model.to('cuda')
    glm_inp_gpu = glm_inp_cpu.to('cuda')
    for r in rows:
        sid = r['sample_id']
        for c in conds:
            with Image.open(ROOT / assets[sid][c]['path']) as im:
                arr = np.array(im.convert('L'), copy=True)
            ans, _, _ = glm_clean(glm_model, glm_tok, glm_proc, glm_inp_gpu, arr)
            answers[sid]['glm'][c] = ans
    del glm_model
    gc.collect()
    torch.cuda.empty_cache()

    print('phase: evaluating on InternVL2.5-2B', flush=True)
    iv2b_model, iv2b_tok = load_internvl_cpu(INTERNVL2B_SNAP, INTERNVL2B_INV, trust_tok=True)
    iv2b_inputs_cpu, iv2b_sep, iv2b_meta = prepare_prompt(iv2b_model, iv2b_tok, question)
    iv2b_model.to('cuda')
    iv2b_inputs_gpu = {k: v.to('cuda') for k, v in iv2b_inputs_cpu.items()}
    for r in rows:
        sid = r['sample_id']
        for c in conds:
            with Image.open(ROOT / assets[sid][c]['path']) as im:
                arr = np.array(im.convert('L'), copy=True)
            ans, _, _ = internvl_clean(iv2b_model, iv2b_tok, tf, iv2b_inputs_gpu, iv2b_sep, iv2b_meta, arr)
            answers[sid]['internvl2b'][c] = ans
    del iv2b_model
    gc.collect()
    torch.cuda.empty_cache()

    agg = {}
    for c in conds:
        agg[c] = {}
        for m in ('mplug', 'internvl1b', 'glm', 'internvl2b'):
            correct = wrong = indet = hit = 0
            for r in rows:
                sid = r['sample_id']
                cat = parse_category(answers[sid][m][c])
                if cat == 'indeterminate':
                    indet += 1
                elif cat == r['reference']:
                    correct += 1
                else:
                    wrong += 1
                    if cat == r['target_category']:
                        hit += 1
            agg[c][m] = {'correct': correct, 'wrong': wrong, 'indeterminate': indet, 'target_hit': hit}

    summary = {
        'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'eps': EPS, 'steps': STEPS, 'proto_per_class': PROTO_PER_CLASS,
        'protocol_sha256': sha256(PROTOCOL),
        'script_sha256': sha256(Path(__file__)),
        'elapsed_seconds': round(time.perf_counter() - t0, 3),
        'aggregate': agg,
        'items': [
            {
                'sample_id': r['sample_id'], 'index': r['index'],
                'reference': r['reference'], 'target_category': r['target_category'],
                'assets': assets[r['sample_id']], 'answers': answers[r['sample_id']],
            }
            for r in rows
        ],
    }
    out_path = OUT_DIR / 'summary.json'
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print('\n=== ROUTE 1: AttackVLM VISUAL PROTOTYPE MATCHING (Linf=16/255, 40 steps) ===', flush=True)
    for c in conds:
        parts = [f"{m}:{agg[c][m]['wrong']}/6(hit {agg[c][m]['target_hit']},ind {agg[c][m]['indeterminate']})"
                 for m in ('mplug', 'internvl1b', 'glm', 'internvl2b')]
        print(f'{c:16s} | ' + ' | '.join(parts), flush=True)
    print('ALL DONE in', summary['elapsed_seconds'], 's ->', out_path, flush=True)


if __name__ == '__main__':
    main()
