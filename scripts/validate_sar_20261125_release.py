"""Offline validator for the sanitized 2026-11-25 public SAR transfer release."""

import ast
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT / 'experiments/records/2026-11-25/summary.json'
FORBIDDEN_PATTERNS = [
    re.compile(r'[A-Za-z]:\\Users\\', re.IGNORECASE),
    re.compile(r'[A-Za-z]:\\Coding', re.IGNORECASE),
    re.compile(r'\b\d{7}\.(jpg|png)\b', re.IGNORECASE),
]


def validate_summary(path: Path = SUMMARY_PATH) -> dict:
    raw = path.read_text(encoding='utf-8')
    for pat in FORBIDDEN_PATTERNS:
        if pat.search(raw):
            raise ValueError(f'Sanitization violation ({pat.pattern}) in {path}')

    data = json.loads(raw)
    if data.get('date') != '2026-11-25':
        raise ValueError('Unexpected release date')

    # 1. Validate 6-image ablation table counts within denominator 6
    dev6 = data['dev6_multisource_ablation_8bit']
    n6 = dev6['n_samples']
    if n6 != 6 or dev6['total_model_calls'] != 360:
        raise ValueError('Unexpected 6-image ablation metadata')
    for cond, models in dev6['conditions'].items():
        for model, counts in models.items():
            if not (0 <= counts['hit'] <= counts['wrong'] <= n6):
                raise ValueError(f'Invalid dev6 counts for {cond}/{model}: {counts}')

    # 2. Validate 14-image frozen LOFO table counts within denominator 14
    c14 = data['cohort14_lofo_glm_target_8bit']
    n14 = c14['n_samples']
    if n14 != 14:
        raise ValueError('Unexpected cohort14 sample count')
    for cond, models in c14['conditions'].items():
        for model, counts in models.items():
            if not (0 <= counts['hit'] <= counts['wrong'] and counts['wrong'] + counts['indeterminate'] <= n14):
                raise ValueError(f'Invalid cohort14 counts for {cond}/{model}: {counts}')

    # 3. Validate Route 1 17-image scaling counts and per-class sum
    c17 = data['route1_visual_prototype_matching_16bit']['cohort17_scaling']
    if c17['n_samples'] != 17:
        raise ValueError('Unexpected cohort17 sample count')
    by_cls = c17['internvl2b_by_reference_class']
    if sum(v['planned'] for v in by_cls.values()) != 17:
        raise ValueError('Per-class planned sum must equal 17')
    if sum(v['wrong'] for v in by_cls.values()) != c17['full17_counts']['avlm_mm_glm16']['internvl2b']['wrong']:
        raise ValueError('Per-class wrong sum mismatches total InternVL2.5-2B wrong count')
    if sum(v['hit'] for v in by_cls.values()) != c17['full17_counts']['avlm_mm_glm16']['internvl2b']['hit']:
        raise ValueError('Per-class hit sum mismatches total InternVL2.5-2B hit count')

    for cond, models in c17['clean_correct_subset_counts'].items():
        for model, counts in models.items():
            if counts['denom'] != c17['clean_denominators'][model]:
                raise ValueError(f'Denominator mismatch for {cond}/{model}')
            if not (0 <= counts['hit'] <= counts['wrong'] <= counts['denom']):
                raise ValueError(f'Invalid clean subset counts for {cond}/{model}')

    # 4. Verify all published scripts exist, parse cleanly, and contain no forbidden strings
    for rel in data['published_scripts']:
        sp = (ROOT / rel).resolve()
        if not sp.is_relative_to(ROOT) or not sp.is_file():
            raise FileNotFoundError(f'Missing published script: {rel}')
        src = sp.read_text(encoding='utf-8')
        ast.parse(src, filename=rel)
        for pat in FORBIDDEN_PATTERNS:
            if pat.search(src):
                raise ValueError(f'Sanitization violation ({pat.pattern}) in {rel}')

    return data


def main() -> int:
    data = validate_summary()
    print(f"OK: validated {SUMMARY_PATH.relative_to(ROOT).as_posix()} and {len(data['published_scripts'])} scripts")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
