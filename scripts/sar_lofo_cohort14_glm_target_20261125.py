"""14-Image Frozen Engineering Cohort Zero-Family-Overlap Scaling Experiment:
Target Black-Box Model: SARChat-GLM-edge-v-2B (zero family overlap with mPLUG-1B and InternVL2.5-1B).
Compares on the 14 jointly clean-correct & format-compatible frozen cohort images (8 steps, Linf <= 8/255):
  1. random8       (reused from .work/sar_engineering_day06_random_controls_20261124)
  2. mplug_single8 (reused from runs/sar-paired-v1/day06-tisimi8-20261124)
  3. iv1b_single8  (NEW: single-source InternVL2.5-1B, 8-step TI-SI-MI)
  4. mm_iv1b_sync8 (NEW: synchronous dual-source 0.5 mPLUG-1B + 0.5 InternVL2.5-1B, 8-step TI-SI-MI)
Evaluates all 4 conditions on GLM-edge-v-2B (zero-overlap black-box target), mPLUG-1B, InternVL2.5-1B, and InternVL2.5-2B.
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
from torchvision.transforms import Compose, InterpolationMode, Normalize, Resize, ToTensor

from run_sar_proxy_candidate import IMAGE_MEAN, IMAGE_SIZE, IMAGE_STD, prepare_prompt
from run_vlm_clean_screen import load_json, sha256
from sar_multisource_ablation_controls_20261124 import (
    INTERNVL1B_INV, INTERNVL1B_SNAP, INTERNVL2B_INV, INTERNVL2B_SNAP,
    SCALE_SCHEDULE,
    glm_clean, internvl_clean, internvl_step_grad,
    load_glm_cpu, load_internvl_cpu, load_mplug_cpu, make_ti_kernel,
    mplug_clean, mplug_step_grad, save_and_verify_png,
)

METHOD_PROTO = ROOT / 'experiments/protocols/sar_paired_execution_20261123.json'
LOFO3_SUMMARY = ROOT / 'runs/sar-lofo-direction3-20261125/summary.json'
COHORT18 = ROOT / '.work/sar_engineering_evaluation_cohort_v1_20261123.json'
RANDOM17_SUMMARY = ROOT / '.work/sar_engineering_day06_random_controls_20261124/summary.json'
MPLUG17_SUMMARY = ROOT / 'runs/sar-paired-v1/day06-tisimi8-20261124/summary.json'
OUT_DIR = ROOT / 'runs/sar-lofo-cohort14-glm-target-20261125'


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
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    method = load_json(METHOD_PROTO)
    question = method['question']
    targets_map = method['attack']['targets']
    cat_tok_ids = method['attack']['category_token_ids']

    lofo3 = load_json(LOFO3_SUMMARY)
    eligible_ids = lofo3['cohort17_clean_probe']['joint_17_clean_eligible']
    cohort_map = {x['sample_id']: x for x in load_json(COHORT18)['items']}
    rand_map = {x['sample_id']: x['asset'] for x in load_json(RANDOM17_SUMMARY)['items']}
    mplug_map = {x['sample_id']: x['asset_8bit_path'] for x in load_json(MPLUG17_SUMMARY)['recorded']}

    rows = []
    for pos, sid in enumerate(eligible_ids):
        c = cohort_map[sid]
        ref = c['reference']
        tgt = targets_map[ref]
        rows.append({
            'pos': pos + 1,
            'sample_id': sid,
            'image_path': c['image_path'],
            'image_sha256': c['image_sha256'],
            'reference': ref,
            'target_category': tgt,
            'clean_token_id': cat_tok_ids[ref],
            'target_token_id': cat_tok_ids[tgt],
            'random8_path': rand_map[sid],
            'mplug_single8_path': mplug_map[sid],
        })
    print(f'Loaded {len(rows)} jointly eligible frozen cohort images (mPLUG+InternVL1B format-ok & GLM clean-ok)', flush=True)

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
    for r in rows:
        sid = r['sample_id']
        p = ROOT / r['image_path']
        if sha256(p) != r['image_sha256']:
            raise ValueError(f'Hash mismatch: {sid}')
        with Image.open(p) as im:
            originals[sid] = np.array(im.convert('L'), copy=True).astype(np.int16)

    # Check if generation assets already exist (resume-safe)
    need_gen = any(
        not (OUT_DIR / f"{r['pos']:02d}_{r['sample_id']}_iv1b_single8.png").is_file()
        or not (OUT_DIR / f"{r['pos']:02d}_{r['sample_id']}_mm_iv1b_sync8.png").is_file()
        for r in rows
    )

    mplug_model, mplug_tok, mplug_proc = load_mplug_cpu()
    iv1b_model, iv1b_tok = load_internvl_cpu(INTERNVL1B_SNAP, INTERNVL1B_INV, trust_tok=False)
    iv1b_inputs_cpu, iv1b_sep, iv1b_meta = prepare_prompt(iv1b_model, iv1b_tok, question)

    if need_gen:
        print('phase: clean forward on mPLUG-1B (14 images)', flush=True)
        mplug_model.to('cuda')
        mplug_clean_state = {}
        for r in rows:
            sid = r['sample_id']
            ans, toks, vis = mplug_clean(mplug_model, mplug_tok, mplug_proc, originals[sid].astype(np.uint8), mplug_messages)
            if toks[0].tolist() != [2132, 374, r['clean_token_id']]:
                raise ValueError(f'mPLUG token mismatch on {sid}: {toks[0].tolist()}')
            mplug_clean_state[sid] = (ans, toks, vis)
        mplug_model.to('cpu')
        torch.cuda.empty_cache()

        print('phase: clean forward on InternVL2.5-1B + iv1b_single8 (14 images x 8 steps)', flush=True)
        iv1b_model.to('cuda')
        iv1b_inputs_gpu = {k: v.to('cuda') for k, v in iv1b_inputs_cpu.items()}
        iv1b_clean_state = {}
        iv1b_curr = {r['sample_id']: originals[r['sample_id']].copy() for r in rows}
        iv1b_mom = {r['sample_id']: np.zeros_like(originals[r['sample_id']], dtype=np.float32) for r in rows}
        for r in rows:
            sid = r['sample_id']
            ans, toks, vis = internvl_clean(iv1b_model, iv1b_tok, tf, iv1b_inputs_gpu, iv1b_sep, iv1b_meta, originals[sid].astype(np.uint8))
            if toks[0].tolist() != [2132, 374, r['clean_token_id']]:
                raise ValueError(f'InternVL-1B token mismatch on {sid}: {toks[0].tolist()}')
            iv1b_clean_state[sid] = (ans, toks, vis)

        for step, scale_size in enumerate(SCALE_SCHEDULE):
            t_s = time.perf_counter()
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
            print(f'  iv1b_single step {step + 1}/8 done in {round(time.perf_counter() - t_s, 2)}s', flush=True)
        iv1b_model.to('cpu')
        torch.cuda.empty_cache()

        print('phase: mm_iv1b_sync8 (0.5 mPLUG-1B + 0.5 InternVL2.5-1B, 14 images x 8 steps)', flush=True)
        sync_curr = {r['sample_id']: originals[r['sample_id']].copy() for r in rows}
        sync_mom = {r['sample_id']: np.zeros_like(originals[r['sample_id']], dtype=np.float32) for r in rows}
        for step, scale_size in enumerate(SCALE_SCHEDULE):
            t_s = time.perf_counter()
            mplug_model.to('cuda')
            g1_all = {}
            for r in rows:
                sid = r['sample_id']
                _, toks, vis = mplug_clean_state[sid]
                g, _ = mplug_step_grad(
                    mplug_model, mplug_proc, sync_curr[sid].astype(np.uint8), mplug_messages,
                    toks, r['clean_token_id'], r['target_token_id'], vis, scale_size, ti_kernel,
                )
                g1_all[sid] = g / float(np.mean(np.abs(g)))
            mplug_model.to('cpu')
            torch.cuda.empty_cache()

            iv1b_model.to('cuda')
            g2_all = {}
            for r in rows:
                sid = r['sample_id']
                _, toks, vis = iv1b_clean_state[sid]
                g, _ = internvl_step_grad(
                    iv1b_model, tf, iv1b_inputs_gpu, sync_curr[sid].astype(np.uint8),
                    toks, r['clean_token_id'], r['target_token_id'], vis, scale_size, ti_kernel,
                )
                g2_all[sid] = g / float(np.mean(np.abs(g)))
            iv1b_model.to('cpu')
            torch.cuda.empty_cache()

            step_size = 1 if step < 4 else 2
            bound = 4 if step < 4 else 8
            for r in rows:
                sid = r['sample_id']
                fused = 0.5 * g1_all[sid] + 0.5 * g2_all[sid]
                sync_mom[sid] = sync_mom[sid] + fused
                delta = step_size * np.sign(sync_mom[sid]).astype(np.int16)
                sync_curr[sid] = np.clip(np.clip(sync_curr[sid] + delta, originals[sid] - bound, originals[sid] + bound), 0, 255).astype(np.int16)
            print(f'  mm_iv1b_sync step {step + 1}/8 done in {round(time.perf_counter() - t_s, 2)}s', flush=True)

        for r in rows:
            sid = r['sample_id']
            save_and_verify_png(OUT_DIR / f"{r['pos']:02d}_{sid}_iv1b_single8.png", iv1b_curr[sid].astype(np.uint8), originals[sid], 8)
            save_and_verify_png(OUT_DIR / f"{r['pos']:02d}_{sid}_mm_iv1b_sync8.png", sync_curr[sid].astype(np.uint8), originals[sid], 8)
        print('Saved and verified all 28 new PNGs (14 iv1b_single8 + 14 mm_iv1b_sync8)', flush=True)

    # Build paths for all 4 conditions
    conds = ['random8', 'mplug_single8', 'iv1b_single8', 'mm_iv1b_sync8']
    eval_paths = {}
    assets_meta = {}
    for r in rows:
        sid = r['sample_id']
        pmap = {
            'random8': ROOT / r['random8_path'],
            'mplug_single8': ROOT / r['mplug_single8_path'],
            'iv1b_single8': OUT_DIR / f"{r['pos']:02d}_{sid}_iv1b_single8.png",
            'mm_iv1b_sync8': OUT_DIR / f"{r['pos']:02d}_{sid}_mm_iv1b_sync8.png",
        }
        eval_paths[sid] = pmap
        assets_meta[sid] = {}
        for c, p in pmap.items():
            with Image.open(p) as im:
                arr = np.array(im.convert('L'), copy=True).astype(np.int16)
            diff = np.abs(arr - originals[sid])
            assets_meta[sid][c] = {
                'path': p.relative_to(ROOT).as_posix(),
                'sha256': sha256(p),
                'linf_8bit': int(diff.max()),
                'mean_abs_difference_8bit': round(float(diff.mean()), 4),
            }

    answers = {r['sample_id']: {'mplug': {}, 'internvl1b': {}, 'glm': {}, 'internvl2b': {}} for r in rows}

    # Evaluate on mPLUG-1B
    print('phase: evaluating 4 conditions on mPLUG-1B', flush=True)
    mplug_model.to('cuda')
    for r in rows:
        sid = r['sample_id']
        for c in conds:
            with Image.open(eval_paths[sid][c]) as im:
                arr = np.array(im.convert('L'), copy=True)
            ans, _, _ = mplug_clean(mplug_model, mplug_tok, mplug_proc, arr, mplug_messages)
            answers[sid]['mplug'][c] = ans
    del mplug_model
    gc.collect()
    torch.cuda.empty_cache()

    # Evaluate on InternVL2.5-1B
    print('phase: evaluating 4 conditions on InternVL2.5-1B', flush=True)
    iv1b_model.to('cuda')
    iv1b_inputs_gpu = {k: v.to('cuda') for k, v in iv1b_inputs_cpu.items()}
    for r in rows:
        sid = r['sample_id']
        for c in conds:
            with Image.open(eval_paths[sid][c]) as im:
                arr = np.array(im.convert('L'), copy=True)
            ans, _, _ = internvl_clean(iv1b_model, iv1b_tok, tf, iv1b_inputs_gpu, iv1b_sep, iv1b_meta, arr)
            answers[sid]['internvl1b'][c] = ans
    del iv1b_model
    gc.collect()
    torch.cuda.empty_cache()

    # Evaluate on Zero-Overlap Black-Box Target GLM-edge-v-2B
    print('phase: evaluating 4 conditions on Zero-Overlap Black-Box Target GLM-edge-v-2B', flush=True)
    glm_model, glm_tok, glm_proc = load_glm_cpu()
    glm_inp_gpu = glm_tok.apply_chat_template(glm_messages, add_generation_prompt=True, return_dict=True, tokenize=True, return_tensors='pt').to('cuda')
    glm_model.to('cuda')
    for r in rows:
        sid = r['sample_id']
        for c in conds:
            with Image.open(eval_paths[sid][c]) as im:
                arr = np.array(im.convert('L'), copy=True)
            ans, _, _ = glm_clean(glm_model, glm_tok, glm_proc, glm_inp_gpu, arr)
            answers[sid]['glm'][c] = ans
    del glm_model
    gc.collect()
    torch.cuda.empty_cache()

    # Evaluate on InternVL2.5-2B
    print('phase: evaluating 4 conditions on InternVL2.5-2B', flush=True)
    iv2b_model, iv2b_tok = load_internvl_cpu(INTERNVL2B_SNAP, INTERNVL2B_INV, trust_tok=True)
    iv2b_inputs_cpu, iv2b_sep, iv2b_meta = prepare_prompt(iv2b_model, iv2b_tok, question)
    iv2b_model.to('cuda')
    iv2b_inputs_gpu = {k: v.to('cuda') for k, v in iv2b_inputs_cpu.items()}
    for r in rows:
        sid = r['sample_id']
        for c in conds:
            with Image.open(eval_paths[sid][c]) as im:
                arr = np.array(im.convert('L'), copy=True)
            ans, _, _ = internvl_clean(iv2b_model, iv2b_tok, tf, iv2b_inputs_gpu, iv2b_sep, iv2b_meta, arr)
            answers[sid]['internvl2b'][c] = ans
    del iv2b_model
    gc.collect()
    torch.cuda.empty_cache()

    agg = {}
    n = len(rows)
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
        'n_samples': n,
        'script_sha256': sha256(Path(__file__)),
        'elapsed_seconds': round(time.perf_counter() - t0, 3),
        'aggregate': agg,
        'items': [
            {
                'pos': r['pos'], 'sample_id': r['sample_id'],
                'reference': r['reference'], 'target_category': r['target_category'],
                'assets': assets_meta[r['sample_id']], 'answers': answers[r['sample_id']],
            }
            for r in rows
        ],
    }
    out_path = OUT_DIR / 'summary.json'
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f'\n=== 14-IMAGE FROZEN COHORT ZERO-OVERLAP TRANSFER (Target: GLM-edge-v-2B, N={n}) ===', flush=True)
    for c in conds:
        parts = [f"{m}:{agg[c][m]['wrong']}/{n}(hit {agg[c][m]['target_hit']},ind {agg[c][m]['indeterminate']})"
                 for m in ('mplug', 'internvl1b', 'glm', 'internvl2b')]
        print(f'{c:16s} | ' + ' | '.join(parts), flush=True)
    print('ALL DONE in', summary['elapsed_seconds'], 's ->', out_path, flush=True)


if __name__ == '__main__':
    main()
