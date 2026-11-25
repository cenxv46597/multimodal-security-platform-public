"""Complete Leave-One-Family-Out Direction 3 on 6 dev images:
  - iv1b_glm_sync4 / iv1b_glm_sync8: 0.5 InternVL2.5-1B + 0.5 GLM-edge-v-2B at every step -> zero-overlap target mPLUG-1B
  - iv1b_glm_relay8: Steps 1..4 on InternVL2.5-1B (from iv1b_single4) -> Steps 5..8 on GLM-edge-v-2B
Evaluates on all 4 models (mPLUG-1B, InternVL2.5-1B, GLM-edge-v-2B, InternVL2.5-2B).
Also probes InternVL2.5-1B and GLM-edge-v-2B clean accuracy on the 17 frozen engineering cohort images.
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
    PROTOCOL, SCALE_SCHEDULE,
    glm_clean, glm_step_grad, internvl_clean, internvl_step_grad,
    load_glm_cpu, load_internvl_cpu, load_mplug_cpu, make_ti_kernel, mplug_clean,
    save_and_verify_png,
)

ABLATION_DIR = ROOT / 'runs/sar-multisource-ablation-controls-20261124'
OUT_DIR = ROOT / 'runs/sar-lofo-direction3-20261125'
DAY05 = ROOT / '.work/sar_engineering_day05_baseline_review_20261124.json'
COHORT18 = ROOT / '.work/sar_engineering_evaluation_cohort_v1_20261123.json'


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
    iv1b4_init = {}
    for r in rows:
        sid = r['sample_id']
        idx = r['index']
        with Image.open(ROOT / r['image_path']) as im:
            originals[sid] = np.array(im.convert('L'), copy=True).astype(np.int16)
        with Image.open(ABLATION_DIR / f'{idx}_iv1b_single4.png') as im4:
            iv1b4_init[sid] = np.array(im4.convert('L'), copy=True).astype(np.int16)

    # Load 17 frozen engineering images for clean probe
    day05 = load_json(DAY05)
    cohort_map = {x['sample_id']: x for x in load_json(COHORT18)['items']}
    cohort17 = [cohort_map[sid] for sid in day05['format_compatible_sample_ids_in_frozen_order']]

    print('phase: loading InternVL2.5-1B and GLM-edge-v-2B to CPU RAM', flush=True)
    iv1b_model, iv1b_tok = load_internvl_cpu(INTERNVL1B_SNAP, INTERNVL1B_INV, trust_tok=False)
    iv1b_inputs_cpu, iv1b_sep, iv1b_meta = prepare_prompt(iv1b_model, iv1b_tok, question)
    glm_model, glm_tok, glm_proc = load_glm_cpu()
    glm_inp_cpu = glm_tok.apply_chat_template(glm_messages, add_generation_prompt=True, return_dict=True, tokenize=True, return_tensors='pt')

    # Clean forward on InternVL2.5-1B (6 dev + 17 cohort)
    print('phase: clean forward on InternVL2.5-1B (6 dev + 17 frozen cohort)', flush=True)
    iv1b_model.to('cuda')
    iv1b_inputs_gpu = {k: v.to('cuda') for k, v in iv1b_inputs_cpu.items()}
    iv1b_clean_state = {}
    for r in rows:
        sid = r['sample_id']
        ans, toks, vis = internvl_clean(iv1b_model, iv1b_tok, tf, iv1b_inputs_gpu, iv1b_sep, iv1b_meta, originals[sid].astype(np.uint8))
        iv1b_clean_state[sid] = (ans, toks, vis)

    cohort17_iv1b = []
    for c_row in cohort17:
        with Image.open(ROOT / c_row['image_path']) as im:
            arr = np.array(im.convert('L'), copy=True)
        ans, toks, _ = internvl_clean(iv1b_model, iv1b_tok, tf, iv1b_inputs_gpu, iv1b_sep, iv1b_meta, arr)
        cat = parse_category(ans)
        cohort17_iv1b.append({
            'sample_id': c_row['sample_id'], 'reference': c_row['reference'],
            'raw_answer': ans, 'parsed': cat, 'correct': cat == c_row['reference'],
            'exact_format': ans == 'It is ' + c_row['reference'],
        })
    print(f"  Cohort-17 InternVL2.5-1B clean: {sum(x['correct'] for x in cohort17_iv1b)}/17 correct, {sum(x['exact_format'] for x in cohort17_iv1b)}/17 exact format", flush=True)
    iv1b_model.to('cpu')
    torch.cuda.empty_cache()

    # Clean forward on GLM-edge-v-2B (6 dev + 17 cohort) + Relay steps 5..8
    print('phase: clean forward on GLM-edge-v-2B (6 dev + 17 frozen cohort) + Relay steps 5..8', flush=True)
    glm_model.to('cuda')
    glm_inp_gpu = glm_inp_cpu.to('cuda')
    glm_clean_state = {}
    for r in rows:
        sid = r['sample_id']
        ans, toks, vis = glm_clean(glm_model, glm_tok, glm_proc, glm_inp_gpu, originals[sid].astype(np.uint8))
        glm_clean_state[sid] = (ans, toks, vis)

    cohort17_glm = []
    for c_row in cohort17:
        with Image.open(ROOT / c_row['image_path']) as im:
            arr = np.array(im.convert('L'), copy=True)
        ans, _, _ = glm_clean(glm_model, glm_tok, glm_proc, glm_inp_gpu, arr)
        cat = parse_category(ans)
        cohort17_glm.append({
            'sample_id': c_row['sample_id'], 'reference': c_row['reference'],
            'raw_answer': ans, 'parsed': cat, 'correct': cat == c_row['reference'],
        })
    print(f"  Cohort-17 GLM-edge-v-2B clean: {sum(x['correct'] for x in cohort17_glm)}/17 correct", flush=True)

    # Relay: Steps 5..8 on GLM-edge-v-2B starting from 4-step InternVL2.5-1B
    relay_curr = {r['sample_id']: iv1b4_init[r['sample_id']].copy() for r in rows}
    relay_mom = {r['sample_id']: np.zeros_like(originals[r['sample_id']], dtype=np.float32) for r in rows}
    for step in range(4, 8):
        scale_size = SCALE_SCHEDULE[step]
        for r in rows:
            sid = r['sample_id']
            _, _, vis = glm_clean_state[sid]
            g, _ = glm_step_grad(
                glm_model, glm_proc, glm_inp_gpu, relay_curr[sid].astype(np.uint8),
                r['reference'], r['target_category'], vis, scale_size, ti_kernel,
            )
            relay_mom[sid] = relay_mom[sid] + g / float(np.mean(np.abs(g)))
            delta = 2 * np.sign(relay_mom[sid]).astype(np.int16)
            relay_curr[sid] = np.clip(np.clip(relay_curr[sid] + delta, originals[sid] - 8, originals[sid] + 8), 0, 255).astype(np.int16)
        print(f'  iv1b_glm_relay step {step + 1}/8 done', flush=True)
    glm_model.to('cpu')
    torch.cuda.empty_cache()

    # Synchronous Dual-Source: 0.5 InternVL2.5-1B + 0.5 GLM-edge-v-2B (Steps 1..8)
    print('phase: Synchronous Dual-Source (0.5 InternVL2.5-1B + 0.5 GLM-2B) steps 1..8', flush=True)
    sync_curr = {r['sample_id']: originals[r['sample_id']].copy() for r in rows}
    sync_mom = {r['sample_id']: np.zeros_like(originals[r['sample_id']], dtype=np.float32) for r in rows}
    sync_4 = {}
    for step, scale_size in enumerate(SCALE_SCHEDULE):
        t_s = time.perf_counter()
        iv1b_model.to('cuda')
        g1_all = {}
        for r in rows:
            sid = r['sample_id']
            _, toks, vis = iv1b_clean_state[sid]
            g, _ = internvl_step_grad(
                iv1b_model, tf, iv1b_inputs_gpu, sync_curr[sid].astype(np.uint8),
                toks, r['clean_token_id'], r['target_token_id'], vis, scale_size, ti_kernel,
            )
            g1_all[sid] = g / float(np.mean(np.abs(g)))
        iv1b_model.to('cpu')
        torch.cuda.empty_cache()

        glm_model.to('cuda')
        g2_all = {}
        for r in rows:
            sid = r['sample_id']
            _, _, vis = glm_clean_state[sid]
            g, _ = glm_step_grad(
                glm_model, glm_proc, glm_inp_gpu, sync_curr[sid].astype(np.uint8),
                r['reference'], r['target_category'], vis, scale_size, ti_kernel,
            )
            g2_all[sid] = g / float(np.mean(np.abs(g)))
        glm_model.to('cpu')
        torch.cuda.empty_cache()

        step_size = 1 if step < 4 else 2
        bound = 4 if step < 4 else 8
        for r in rows:
            sid = r['sample_id']
            fused = 0.5 * g1_all[sid] + 0.5 * g2_all[sid]
            sync_mom[sid] = sync_mom[sid] + fused
            delta = step_size * np.sign(sync_mom[sid]).astype(np.int16)
            sync_curr[sid] = np.clip(np.clip(sync_curr[sid] + delta, originals[sid] - bound, originals[sid] + bound), 0, 255).astype(np.int16)
            if step == 3:
                sync_4[sid] = sync_curr[sid].astype(np.uint8).copy()
        print(f'  iv1b_glm_sync step {step + 1}/8 done in {round(time.perf_counter() - t_s, 2)}s', flush=True)

    # Save assets
    assets = {r['sample_id']: {} for r in rows}
    for r in rows:
        sid = r['sample_id']
        idx = r['index']
        orig = originals[sid]
        assets[sid]['iv1b_glm_relay8'] = save_and_verify_png(OUT_DIR / f'{idx}_iv1b_glm_relay8.png', relay_curr[sid].astype(np.uint8), orig, 8)
        assets[sid]['iv1b_glm_sync4'] = save_and_verify_png(OUT_DIR / f'{idx}_iv1b_glm_sync4.png', sync_4[sid], orig, 4)
        assets[sid]['iv1b_glm_sync8'] = save_and_verify_png(OUT_DIR / f'{idx}_iv1b_glm_sync8.png', sync_curr[sid].astype(np.uint8), orig, 8)

    conds = ['iv1b_glm_relay8', 'iv1b_glm_sync4', 'iv1b_glm_sync8']
    answers = {r['sample_id']: {'mplug': {}, 'internvl1b': {}, 'glm': {}, 'internvl2b': {}} for r in rows}

    # Eval on InternVL-1B
    print('phase: evaluating on InternVL2.5-1B', flush=True)
    iv1b_model.to('cuda')
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

    # Eval on GLM-2B
    print('phase: evaluating on GLM-edge-v-2B', flush=True)
    glm_model.to('cuda')
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

    # Eval on Zero-Overlap Target mPLUG-1B
    print('phase: evaluating on Zero-Overlap Black-Box Target mPLUG-1B', flush=True)
    mplug_model, mplug_tok, mplug_proc = load_mplug_cpu()
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

    # Eval on InternVL2.5-2B
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
        'protocol_sha256': sha256(PROTOCOL),
        'script_sha256': sha256(Path(__file__)),
        'elapsed_seconds': round(time.perf_counter() - t0, 3),
        'aggregate': agg,
        'cohort17_clean_probe': {
            'internvl1b': cohort17_iv1b,
            'glm': cohort17_glm,
            'internvl1b_correct': sum(x['correct'] for x in cohort17_iv1b),
            'internvl1b_exact_format': sum(x['exact_format'] for x in cohort17_iv1b),
            'glm_correct': sum(x['correct'] for x in cohort17_glm),
            'joint_17_clean_eligible': [
                a['sample_id'] for a, b in zip(cohort17_iv1b, cohort17_glm)
                if a['exact_format'] and b['correct']
            ],
        },
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
    print('\n=== LOFO DIRECTION 3 (InternVL1B + GLM -> zero-overlap mPLUG-1B) ===', flush=True)
    for c in conds:
        parts = [f"{m}:{agg[c][m]['wrong']}/6(hit {agg[c][m]['target_hit']},ind {agg[c][m]['indeterminate']})"
                 for m in ('mplug', 'internvl1b', 'glm', 'internvl2b')]
        print(f'{c:16s} | ' + ' | '.join(parts), flush=True)
    print(f"Cohort-17 joint eligible (mPLUG+InternVL1B format-ok & GLM clean-ok): {len(summary['cohort17_clean_probe']['joint_17_clean_eligible'])}/17", flush=True)
    print('DONE in', summary['elapsed_seconds'], 's', flush=True)


if __name__ == '__main__':
    main()
