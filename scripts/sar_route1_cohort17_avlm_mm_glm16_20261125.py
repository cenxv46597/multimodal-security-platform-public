"""17-Image Frozen Engineering Cohort Scaling Test for Route 1:
Target-Class Visual Prototype Matching (AttackVLM DI-TI-MI, Linf=16/255, 16 steps, step=2/255).
Zero-Family-Overlap Black-Box Surrogates: 0.5 mPLUG-1B + 0.5 GLM-edge-v-2B (ZERO InternVL involvement, ZERO LLM logits).
Primary Zero-Overlap Black-Box Target: SARChat-InternVL2.5-2B (17/17 clean correct).
Also evaluated on InternVL2.5-1B (zero-overlap), mPLUG-1B, and GLM-edge-v-2B, alongside strict random16 control.
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
    glm_clean, internvl_clean, load_glm_cpu, load_internvl_cpu, load_mplug_cpu,
    make_ti_kernel, mplug_clean, save_and_verify_png,
)
from sar_route1_attackvlm_16_20261125 import (
    DRAFT_POOL, EPS, PROTO_PER_CLASS, SCALE_POOL, STEPS, STEP_SIZE,
    glm_embed_nograd, glm_vis_grad, mplug_embed_nograd, mplug_vis_grad, parse_category,
)

METHOD_PROTO = ROOT / 'experiments/protocols/sar_paired_execution_20261123.json'
DAY05 = ROOT / '.work/sar_engineering_day05_baseline_review_20261124.json'
COHORT18 = ROOT / '.work/sar_engineering_evaluation_cohort_v1_20261123.json'
OUT_DIR = ROOT / 'runs/sar-route1-cohort17-avlm16-20261125'


def main():
    t0 = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    method = load_json(METHOD_PROTO)
    question = method['question']
    targets_map = method['attack']['targets']

    day05 = load_json(DAY05)
    cohort_map = {x['sample_id']: x for x in load_json(COHORT18)['items']}
    eligible_sids = day05['format_compatible_sample_ids_in_frozen_order']
    ord_map = {x['sample_id']: x['ordinal'] for x in day05['pair_eligibility']}

    rows = []
    for sid in eligible_sids:
        c = cohort_map[sid]
        ref = c['reference']
        rows.append({
            'frozen_ordinal': ord_map[sid],
            'sample_id': sid,
            'image_path': c['image_path'],
            'image_sha256': c['image_sha256'],
            'reference': ref,
            'target_category': targets_map[ref],
        })
    print(f'Loaded {len(rows)} frozen paired cohort images (ordinals {[r["frozen_ordinal"] for r in rows]})', flush=True)

    # Disjoint 8-image class prototypes from SARDet train development pool
    pool = load_json(DRAFT_POOL)['rows']
    test_sids = set(eligible_sids)
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
        sid = r['sample_id']
        p = ROOT / r['image_path']
        if sha256(p) != r['image_sha256']:
            raise ValueError(f'Image hash mismatch: {sid}')
        with Image.open(p) as im:
            originals[sid] = np.array(im.convert('L'), copy=True).astype(np.int16)

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

    need_gen = any(
        not (OUT_DIR / f"{r['frozen_ordinal']:02d}_{r['sample_id']}_random16.png").is_file()
        or not (OUT_DIR / f"{r['frozen_ordinal']:02d}_{r['sample_id']}_avlm_mm_glm16.png").is_file()
        for r in rows
    )

    mplug_model, mplug_tok, mplug_proc = load_mplug_cpu()
    glm_model, glm_tok, glm_proc = load_glm_cpu()
    glm_inp_cpu = glm_tok.apply_chat_template(glm_messages, add_generation_prompt=True, return_dict=True, tokenize=True, return_tensors='pt')

    if need_gen:
        print('phase: computing 8-image class visual prototypes on mPLUG-1B and GLM-2B', flush=True)
        mplug_model.to('cuda')
        mp_proto = {c: F.normalize(torch.cat([mplug_embed_nograd(mplug_model, a) for a in arrs], dim=0).mean(dim=0, keepdim=True), dim=-1)
                    for c, arrs in proto_imgs.items()}
        mp_clean_emb = {r['sample_id']: mplug_embed_nograd(mplug_model, originals[r['sample_id']].astype(np.uint8)) for r in rows}
        mplug_model.to('cpu')
        torch.cuda.empty_cache()

        glm_model.to('cuda')
        gl_proto = {c: F.normalize(torch.cat([glm_embed_nograd(glm_model, a) for a in arrs], dim=0).mean(dim=0, keepdim=True), dim=-1)
                    for c, arrs in proto_imgs.items()}
        gl_clean_emb = {r['sample_id']: glm_embed_nograd(glm_model, originals[r['sample_id']].astype(np.uint8)) for r in rows}
        glm_model.to('cpu')
        torch.cuda.empty_cache()

        curr = {r['sample_id']: originals[r['sample_id']].copy() for r in rows}
        mom = {r['sample_id']: np.zeros_like(originals[r['sample_id']], dtype=np.float32) for r in rows}
        rng_div = np.random.default_rng(2026112517)

        print(f'phase: generating avlm_mm_glm16 on 17 frozen images ({STEPS} steps, step={STEP_SIZE}/255, Linf={EPS}/255)', flush=True)
        for step in range(STEPS):
            t_s = time.perf_counter()
            scale_size = SCALE_POOL[step % len(SCALE_POOL)]
            sy = int(rng_div.integers(-8, 9))
            sx = int(rng_div.integers(-8, 9))

            mplug_model.to('cuda')
            g_mp = {}
            for r in rows:
                sid = r['sample_id']
                g, _ = mplug_vis_grad(
                    mplug_model, curr[sid].astype(np.uint8),
                    mp_proto[r['target_category']], mp_clean_emb[sid], mp_proto[r['reference']],
                    scale_size, sy, sx, ti_kernel,
                )
                g_mp[sid] = g / float(np.mean(np.abs(g)))
            mplug_model.to('cpu')
            torch.cuda.empty_cache()

            glm_model.to('cuda')
            g_gl = {}
            for r in rows:
                sid = r['sample_id']
                g, _ = glm_vis_grad(
                    glm_model, curr[sid].astype(np.uint8),
                    gl_proto[r['target_category']], gl_clean_emb[sid], gl_proto[r['reference']],
                    scale_size, sy, sx, ti_kernel,
                )
                g_gl[sid] = g / float(np.mean(np.abs(g)))
            glm_model.to('cpu')
            torch.cuda.empty_cache()

            for r in rows:
                sid = r['sample_id']
                orig = originals[sid]
                gf = 0.5 * g_mp[sid] + 0.5 * g_gl[sid]
                mom[sid] = mom[sid] + gf
                delta = STEP_SIZE * np.sign(mom[sid]).astype(np.int16)
                curr[sid] = np.clip(np.clip(curr[sid] + delta, orig - EPS, orig + EPS), 0, 255).astype(np.int16)

            print(f'  step {step + 1:02d}/{STEPS} done in {round(time.perf_counter() - t_s, 2)}s (scale={scale_size}, shift=({sy},{sx}))', flush=True)

        for r in rows:
            sid = r['sample_id']
            ord_id = r['frozen_ordinal']
            orig = originals[sid]
            rng_r = np.random.default_rng(2026112516 + ord_id)
            rand16 = np.clip(orig + rng_r.integers(-EPS, EPS + 1, size=orig.shape, dtype=np.int16), 0, 255).astype(np.uint8)
            save_and_verify_png(OUT_DIR / f'{ord_id:02d}_{sid}_random16.png', rand16, orig, EPS)
            save_and_verify_png(OUT_DIR / f'{ord_id:02d}_{sid}_avlm_mm_glm16.png', curr[sid].astype(np.uint8), orig, EPS)
        print('Saved and verified all 34 PNGs (17 random16 + 17 avlm_mm_glm16)', flush=True)

    conds = ['random16', 'avlm_mm_glm16']
    assets = {r['sample_id']: {} for r in rows}
    for r in rows:
        sid = r['sample_id']
        ord_id = r['frozen_ordinal']
        for c in conds:
            p = OUT_DIR / f'{ord_id:02d}_{sid}_{c}.png'
            with Image.open(p) as im:
                arr = np.array(im.convert('L'), copy=True).astype(np.int16)
            diff = np.abs(arr - originals[sid])
            assets[sid][c] = {
                'path': p.relative_to(ROOT).as_posix(),
                'sha256': sha256(p),
                'linf_8bit': int(diff.max()),
                'mean_abs_difference_8bit': round(float(diff.mean()), 4),
            }

    answers = {r['sample_id']: {'mplug': {}, 'glm': {}, 'internvl1b': {}, 'internvl2b': {}} for r in rows}

    # 1. mPLUG-1B
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

    # 2. GLM-2B
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

    # 3. Zero-Overlap Black-Box InternVL2.5-1B
    print('phase: evaluating on Zero-Overlap Black-Box InternVL2.5-1B', flush=True)
    iv1b_model, iv1b_tok = load_internvl_cpu(INTERNVL1B_SNAP, INTERNVL1B_INV, trust_tok=False)
    iv1b_inputs_cpu, iv1b_sep, iv1b_meta = prepare_prompt(iv1b_model, iv1b_tok, question)
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

    # 4. Primary Zero-Overlap Black-Box Target InternVL2.5-2B
    print('phase: evaluating on Primary Zero-Overlap Black-Box Target InternVL2.5-2B', flush=True)
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
    n = len(rows)
    for c in conds:
        agg[c] = {}
        for m in ('mplug', 'glm', 'internvl1b', 'internvl2b'):
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
        'n_samples': n, 'eps': EPS, 'steps': STEPS, 'step_size': STEP_SIZE,
        'proto_per_class': PROTO_PER_CLASS,
        'script_sha256': sha256(Path(__file__)),
        'elapsed_seconds': round(time.perf_counter() - t0, 3),
        'aggregate': agg,
        'items': [
            {
                'frozen_ordinal': r['frozen_ordinal'], 'sample_id': r['sample_id'],
                'reference': r['reference'], 'target_category': r['target_category'],
                'assets': assets[r['sample_id']], 'answers': answers[r['sample_id']],
            }
            for r in rows
        ],
    }
    out_path = OUT_DIR / 'summary.json'
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f'\n=== 17-IMAGE FROZEN COHORT: ROUTE 1 AttackVLM (ZERO InternVL, Linf=16/255, N={n}) ===', flush=True)
    for c in conds:
        parts = [f"{m}:{agg[c][m]['wrong']}/{n}(hit {agg[c][m]['target_hit']},ind {agg[c][m]['indeterminate']})"
                 for m in ('mplug', 'glm', 'internvl1b', 'internvl2b')]
        print(f'{c:16s} | ' + ' | '.join(parts), flush=True)
    print('ALL DONE in', summary['elapsed_seconds'], 's ->', out_path, flush=True)


if __name__ == '__main__':
    main()
