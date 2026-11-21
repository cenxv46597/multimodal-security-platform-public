"""Sanitized, stdlib-only schema and fail-closed hash verifier for read-only SAR experiment reports.

This public module mirrors the integrity-check logic used by the local dashboard's
read-only `/api/recent-experiments` adapter without bundling private SAR images,
sample IDs, raw model responses, or local run artifacts.
"""
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_verified_json(path: Path, expected_sha256: str) -> dict:
    """Load JSON only if its SHA-256 matches the expected digest; fail closed otherwise."""
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(f'SHA-256 mismatch for {path.name}')
    return json.loads(path.read_text(encoding='utf-8'))


def validate_suite_denominators(suite_id: str, payload: dict) -> None:
    """Verify denominator invariants for each read-only experiment suite."""
    if suite_id == 'tisimi_six':
        if payload.get('denominator') != 6:
            raise ValueError('tisimi_six denominator must be 6')
        for model_key in ('mplug', 'internvl2b'):
            for cond in ('clean', 'random4', 'pgd4', 'tisimi4', 'tisimi8', 'random8'):
                counts = payload['counts'][model_key][cond]
                total = counts['correct'] + counts['wrong'] + counts['unresolved']
                if total != 6:
                    raise ValueError(f'Count sum mismatch for {model_key}.{cond}: {total}')
    elif suite_id == 'hardlabel_six':
        summary = payload['summary']
        if payload.get('denominator') != 6 or summary['planned'] != 6:
            raise ValueError('hardlabel_six denominator must be 6')
        if summary['final_wrong_and_ctrl_correct'] > summary['final_wrong']:
            raise ValueError('Paired subset cannot exceed total final_wrong')
    elif suite_id == 'qwen2vl_sar':
        base = payload['baseline']
        if base.get('denominator') != 6:
            raise ValueError('qwen2vl_sar baseline denominator must be 6')
        for key in ('zeroshot_counts', 'domain_prior_counts'):
            c = base[key]
            if c['correct'] + c['wrong'] + c['unresolved'] != 6:
                raise ValueError(f'Baseline count mismatch in {key}')
        inj = payload['injection']['summary']
        if inj.get('denominator') != 3:
            raise ValueError('qwen2vl_sar injection denominator must be 3')
        for field in (
            'blank_correct', 'benign_correct', 'attack_correct',
            'benign_review', 'attack_review', 'attack_ocr_exact',
            'benign_ocr_contains', 'user_positive_review',
        ):
            if not 0 <= inj[field] <= 3:
                raise ValueError(f'Injection metric out of range: {field}')
    else:
        raise KeyError(f'Unknown suite_id: {suite_id}')
