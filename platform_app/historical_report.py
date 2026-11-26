"""Read-only, provenance-checked report for one existing development experiment."""
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / 'runs/sar-twelve-repeat-20261113'
PROTOCOL = ROOT / 'experiments/protocols/sar_development_twelve_repeat_eval_20261113.json'
MODELS = ('mplug', 'internvl2b')
CONDITIONS = ('clean', 'surrogate_step', 'random_control')
JUDGMENTS = ('correct', 'wrong', 'unresolved')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_report():
    protocol = json.loads(PROTOCOL.read_text(encoding='utf-8'))
    items = json.loads((RUN / 'semantic_results.json').read_text(encoding='utf-8'))['items']
    rows = protocol['rows']
    if len(items) != 12 or len({item['id'] for item in items}) != 12 or len(rows) != 36:
        raise ValueError('历史记录样本数量不一致')
    expected = {item['id'] + '|' + condition for item in items for condition in CONDITIONS}
    inputs = {row['sample_id']: row for row in rows}
    if len(inputs) != len(rows) or set(inputs) != expected:
        raise ValueError('历史输入与样本不匹配')
    for row in rows:
        path = (ROOT / row['image_path']).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file() or sha(path) != row['image_sha256']:
            raise ValueError('历史图片不存在或哈希不匹配')
    counts, pairs = {}, {}
    for model in MODELS:
        raw = json.loads((RUN / (model + '_eval.json')).read_text(encoding='utf-8'))
        predictions = raw['predictions']
        by_id = {prediction['sample_id']: prediction for prediction in predictions}
        if raw['status'] != 'completed' or raw['protocol_sha256'] != sha(PROTOCOL) or len(predictions) != 36 or len(by_id) != 36 or set(by_id) != expected:
            raise ValueError('历史原始回答或协议不一致')
        for item in items:
            for condition in CONDITIONS:
                judgment = item['judgments'][model][condition]
                prediction = by_id[item['id'] + '|' + condition]
                if (prediction['status'] != 'ok' or judgment['judgment'] not in JUDGMENTS
                    or judgment['answer'] != prediction['answer']
                    or judgment['hit_limit'] != prediction['hit_token_limit']):
                    raise ValueError('历史判读与原始回答不一致')
        counts[model] = {}
        for condition in CONDITIONS:
            tally = Counter(item['judgments'][model][condition]['judgment'] for item in items)
            counts[model][condition] = {key: tally[key] for key in JUDGMENTS}
        paired = Counter()
        for item in items:
            answer = item['judgments'][model]
            attack = answer['surrogate_step']['judgment']
            random = answer['random_control']['judgment']
            key = ('unresolved' if 'unresolved' in (attack, random) else
                   'both_wrong' if attack == random == 'wrong' else
                   'attack_only_wrong' if attack == 'wrong' else
                   'random_only_wrong' if random == 'wrong' else 'both_correct')
            paired[key] += 1
        pairs[model] = {key: paired[key] for key in ('both_wrong', 'attack_only_wrong', 'random_only_wrong', 'both_correct', 'unresolved')}
    return {'sample_count': len(items), 'counts': counts, 'paired': pairs,
            'scope': '12 张已知两模型共同答对的开发图；每类 2 张，不是独立泛化成绩。',
            'description': '原始权重；SAR 显示图像单步变化，最大 4/255，另设等预算随机对照；本地 InternVL2B 为目标。',
            'provenance': '已有历史结果；36 个输入文件哈希及两模型各 36 条原答与协议和语义记录核对一致。语义判读为既有单人记录，参考标签未完成领域复核。',
            'items': items}
