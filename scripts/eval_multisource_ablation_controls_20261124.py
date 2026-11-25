"""Evaluate all 15 single/dual/triple-source conditions on the 6 SAR development images across all 4 SARChat models."""

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
from torchvision.transforms import Compose, InterpolationMode, Normalize, Resize, ToTensor

from run_sar_proxy_candidate import IMAGE_MEAN, IMAGE_SIZE, IMAGE_STD, prepare_prompt
from run_vlm_clean_screen import load_json, sha256
from sar_multisource_ablation_controls_20261124 import (
    DUAL_MM_IV1B_DIR, INTERNVL1B_INV, INTERNVL1B_SNAP, INTERNVL2B_INV, INTERNVL2B_SNAP,
    OUT_DIR, PROTOCOL, SINGLE_MPLUG_DIR,
    glm_clean, internvl_clean, load_glm_cpu, load_internvl_cpu, load_mplug_cpu, mplug_clean,
)

COND_BUDGETS = {
    'random8': 8,
    'mplug_single4': 4,
    'mplug_single8': 8,
    'iv1b_single4': 4,
    'iv1b_single8': 8,
    'glm_single4': 4,
    'glm_single8': 8,
    'mm_iv1b_relay8': 8,
    'mm_iv1b_sync4': 4,
    'mm_iv1b_sync8': 8,
    'cross_relay8': 8,
    'cross_sync4': 4,
    'cross_sync8': 8,
    'triple_sync4': 4,
    'triple_sync8': 8,
}


def parse_category(raw: str):
    low = raw.lower()
    found = []
    for c in ('ship', 'aircraft', 'car', 'tank', 'bridge', 'harbor'):
        if c in low:
            found.append(c)
    if len(found) == 1:
        return found[0]
    return 'indeterminate'


def main():
    t0 = time.perf_counter()
    plan = load_json(PROTOCOL)
    question = plan['question']
    rows = plan['rows']
    torch.set_num_threads(4)
    tf = Compose([Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=InterpolationMode.BICUBIC),
                  ToTensor(), Normalize(IMAGE_MEAN, IMAGE_STD)])
    mplug_messages = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': '<|image|>\n' + question},
        {'role': 'assistant', 'content': ''},
    ]
    glm_messages = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': question}]}]

    # Verify all 15 conditions x 6 images
    eval_paths = {r['sample_id']: {} for r in rows}
    assets_meta = {r['sample_id']: {} for r in rows}
    for r in rows:
        sid = r['sample_id']
        idx = r['index']
        with Image.open(ROOT / r['image_path']) as im:
            orig = np.array(im.convert('L'), copy=True).astype(np.int16)

        path_map = {
            'random8': SINGLE_MPLUG_DIR / f'{idx}_random8.png',
            'mplug_single4': SINGLE_MPLUG_DIR / f'{idx}_tisimi4.png',
            'mplug_single8': SINGLE_MPLUG_DIR / f'{idx}_tisimi8.png',
            'mm_iv1b_relay8': DUAL_MM_IV1B_DIR / f'{idx}_relay8.png',
            'mm_iv1b_sync4': DUAL_MM_IV1B_DIR / f'{idx}_sync4.png',
            'mm_iv1b_sync8': DUAL_MM_IV1B_DIR / f'{idx}_sync8.png',
        }
        for c in ('iv1b_single4', 'iv1b_single8', 'glm_single4', 'glm_single8',
                  'cross_relay8', 'cross_sync4', 'cross_sync8', 'triple_sync4', 'triple_sync8'):
            path_map[c] = OUT_DIR / f'{idx}_{c}.png'

        for c, p in path_map.items():
            if not p.is_file():
                raise FileNotFoundError(f'Missing asset: {p}')
            with Image.open(p) as im:
                if im.format != 'PNG' or im.mode != 'L':
                    raise ValueError(f'Expected 8-bit grayscale PNG: {p}')
                arr = np.array(im, copy=True).astype(np.int16)
            if arr.shape != orig.shape:
                raise ValueError(f'Shape mismatch: {p}')
            diff = np.abs(arr - orig)
            mx = int(diff.max())
            if mx > COND_BUDGETS[c]:
                raise ValueError(f'Budget exceeded on {p}: {mx} > {COND_BUDGETS[c]}')
            eval_paths[sid][c] = p
            assets_meta[sid][c] = {
                'path': p.relative_to(ROOT).as_posix(),
                'sha256': sha256(p),
                'linf_8bit': mx,
                'mean_abs_difference_8bit': round(float(diff.mean()), 4),
            }
    print('Verified all 90 image-condition assets (15 conditions x 6 images)', flush=True)

    all_conds = list(COND_BUDGETS.keys())
    answers = {r['sample_id']: {'mplug': {}, 'internvl1b': {}, 'glm': {}, 'internvl2b': {}} for r in rows}

    # 1. mPLUG-1B
    ckpt_mplug = OUT_DIR / 'eval_mplug.json'
    if ckpt_mplug.is_file():
        saved = load_json(ckpt_mplug)
        for r in rows:
            answers[r['sample_id']]['mplug'] = saved[r['sample_id']]
        print('phase: loaded cached mPLUG-1B eval', flush=True)
    else:
        print('phase: evaluating on mPLUG-1B', flush=True)
        mplug_model, mplug_tok, mplug_proc = load_mplug_cpu()
        mplug_model.to('cuda')
        for r in rows:
            sid = r['sample_id']
            for c in all_conds:
                with Image.open(eval_paths[sid][c]) as im:
                    arr = np.array(im.convert('L'), copy=True)
                ans, _, _ = mplug_clean(mplug_model, mplug_tok, mplug_proc, arr, mplug_messages)
                answers[sid]['mplug'][c] = ans
        del mplug_model
        gc.collect()
        torch.cuda.empty_cache()
        ckpt_mplug.write_text(json.dumps({r['sample_id']: answers[r['sample_id']]['mplug'] for r in rows}, indent=2), encoding='utf-8')
        print('phase: mPLUG-1B done', flush=True)

    # 2. InternVL2.5-1B
    ckpt_iv1b = OUT_DIR / 'eval_internvl1b.json'
    if ckpt_iv1b.is_file():
        saved = load_json(ckpt_iv1b)
        for r in rows:
            answers[r['sample_id']]['internvl1b'] = saved[r['sample_id']]
        print('phase: loaded cached InternVL2.5-1B eval', flush=True)
    else:
        print('phase: evaluating on InternVL2.5-1B', flush=True)
        iv1b_model, iv1b_tok = load_internvl_cpu(INTERNVL1B_SNAP, INTERNVL1B_INV, trust_tok=False)
        iv1b_inputs_cpu, iv1b_sep, iv1b_meta = prepare_prompt(iv1b_model, iv1b_tok, question)
        iv1b_model.to('cuda')
        iv1b_inputs_gpu = {k: v.to('cuda') for k, v in iv1b_inputs_cpu.items()}
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
        ckpt_iv1b.write_text(json.dumps({r['sample_id']: answers[r['sample_id']]['internvl1b'] for r in rows}, indent=2), encoding='utf-8')
        print('phase: InternVL2.5-1B done', flush=True)

    # 3. GLM-edge-v-2B
    ckpt_glm = OUT_DIR / 'eval_glm.json'
    if ckpt_glm.is_file():
        saved = load_json(ckpt_glm)
        for r in rows:
            answers[r['sample_id']]['glm'] = saved[r['sample_id']]
        print('phase: loaded cached GLM-edge-v-2B eval', flush=True)
    else:
        print('phase: evaluating on GLM-edge-v-2B', flush=True)
        glm_model, glm_tok, glm_proc = load_glm_cpu()
        glm_inp_gpu = glm_tok.apply_chat_template(glm_messages, add_generation_prompt=True, return_dict=True, tokenize=True, return_tensors='pt').to('cuda')
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
        ckpt_glm.write_text(json.dumps({r['sample_id']: answers[r['sample_id']]['glm'] for r in rows}, indent=2), encoding='utf-8')
        print('phase: GLM-edge-v-2B done', flush=True)

    # 4. InternVL2.5-2B
    ckpt_iv2b = OUT_DIR / 'eval_internvl2b.json'
    if ckpt_iv2b.is_file():
        saved = load_json(ckpt_iv2b)
        for r in rows:
            answers[r['sample_id']]['internvl2b'] = saved[r['sample_id']]
        print('phase: loaded cached InternVL2.5-2B eval', flush=True)
    else:
        print('phase: evaluating on InternVL2.5-2B', flush=True)
        iv2b_model, iv2b_tok = load_internvl_cpu(INTERNVL2B_SNAP, INTERNVL2B_INV, trust_tok=True)
        iv2b_inputs_cpu, iv2b_sep, iv2b_meta = prepare_prompt(iv2b_model, iv2b_tok, question)
        iv2b_model.to('cuda')
        iv2b_inputs_gpu = {k: v.to('cuda') for k, v in iv2b_inputs_cpu.items()}
        for r in rows:
            sid = r['sample_id']
            for c in all_conds:
                with Image.open(eval_paths[sid][c]) as im:
                    arr = np.array(im.convert('L'), copy=True)
                ans, _, _ = internvl_clean(iv2b_model, iv2b_tok, tf, iv2b_inputs_gpu, iv2b_sep, iv2b_meta, arr)
                answers[sid]['internvl2b'][c] = ans
        del iv2b_model
        gc.collect()
        torch.cuda.empty_cache()
        ckpt_iv2b.write_text(json.dumps({r['sample_id']: answers[r['sample_id']]['internvl2b'] for r in rows}, indent=2), encoding='utf-8')
        print('phase: InternVL2.5-2B done', flush=True)

    # Aggregate counts per (condition, model)
    agg = {}
    for c in all_conds:
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
        'protocol_sha256': sha256(PROTOCOL),
        'eval_script_sha256': sha256(Path(__file__)),
        'elapsed_seconds': round(time.perf_counter() - t0, 3),
        'aggregate': agg,
        'items': [
            {
                'sample_id': r['sample_id'],
                'index': r['index'],
                'reference': r['reference'],
                'target_category': r['target_category'],
                'assets': assets_meta[r['sample_id']],
                'answers': answers[r['sample_id']],
            }
            for r in rows
        ],
    }
    out_path = OUT_DIR / 'summary.json'
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print('\n=== AGGREGATE TABLE (wrong/6, target_hit/6) ===', flush=True)
    for c in all_conds:
        line = f'{c:16s} | '
        parts = []
        for m in ('mplug', 'internvl1b', 'glm', 'internvl2b'):
            d = agg[c][m]
            parts.append(f"{m}:{d['wrong']}/6(hit {d['target_hit']},ind {d['indeterminate']})")
        print(line + ' | '.join(parts), flush=True)
    print('EVAL DONE in', summary['elapsed_seconds'], 's ->', out_path, flush=True)


if __name__ == '__main__':
    main()
