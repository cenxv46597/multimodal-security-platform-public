"""Validate public November 21 summary denominators and paired answers."""

import json
from pathlib import Path


def validate(data, prior=None):
    control = data['random_sign']
    rows = control['rows']
    if len(rows) != control['planned_images'] or len({row['reference'] for row in rows}) != len(rows):
        raise ValueError('Incorrect image count or duplicate reference class')
    if control['completed_calls'] != len(rows) * control['draws_per_image'] or control['capped'] != 0:
        raise ValueError('Inconsistent random-sign call counts')
    for row in rows:
        if any(not isinstance(row[key], str) or not row[key].strip()
               for key in ('clean', 'tisimi8', 'random8', 'random_sign_1', 'random_sign_2')):
            raise ValueError('Missing paired answer')
    gate = data['trusted_gate']
    if gate['planned_calls'] != gate['completed_calls'] + gate['not_run']:
        raise ValueError('Inconsistent gate denominator')
    if len(gate['rows']) != gate['completed_calls']:
        raise ValueError('Inconsistent gate records')
    if prior is not None:
        if prior['calls_planned'] != 18 or prior['calls_completed'] != len(prior['records']) or prior['capped'] != 0:
            raise ValueError('Inconsistent prior review denominator')
        pairs = {(row['reference'], row['condition']) for row in prior['records']}
        if len(pairs) != 18 or any(row['capped'] or not row['answer'].strip() for row in prior['records']):
            raise ValueError('Incomplete or duplicate prior review record')
    return len(rows)


if __name__ == '__main__':
    path = Path(__file__).resolve().parents[1] / 'experiments/records/2026-11-21/results.json'
    data = json.loads(path.read_text(encoding='utf-8'))
    prior = json.loads((path.parent / 'sar-text-prior-review.json').read_text(encoding='utf-8'))
    print(f'Validated {validate(data, prior)} paired development images and gate and prior review counts.')
