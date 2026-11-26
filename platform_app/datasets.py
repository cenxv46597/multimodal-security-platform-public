"""Read-only access to two explicitly registered SAR development sets."""
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLASSES = {'ship': '船', 'aircraft': '飞机', 'car': '汽车', 'tank': '储油罐', 'bridge': '桥', 'harbor': '港口'}
CONFIG = {
    'sar81': {'name': 'SAR 六类开发集 · 81 张', 'protocol': 'experiments/protocols/sar_expansion_81_development_20261112.json',
              'description': '用于四个本地模型的正常问答筛选，包含船、飞机、汽车、储油罐、桥和港口六类。',
              'limit': '已参与模型筛选；不是独立测试。参考标签、场景关联和模型训练暴露仍有不确定性。'},
    'sar332': {'name': 'SAR 六类扩充开发集 · 332 张', 'protocol': 'runs/sar-expanded-development-332-20261113/protocol.json',
               'description': '用于两模型的正常能力评价，并为后续本地扰动探索提供开发样本。',
               'limit': '已用于开发探索；不是独立测试。未完全排除场景关联、近重复或模型训练暴露。'},
}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def dataset(dataset_id):
    if dataset_id not in CONFIG:
        raise ValueError('数据集不可用')
    cfg = CONFIG[dataset_id]
    rows = read(ROOT / cfg['protocol'])['rows']
    if dataset_id == 'sar81':
        items = read(ROOT / 'runs/sar-expansion-81-20261112/semantic_results.json')['models']['internvl2b']['items']
        references = {item['sample_id']: item['reference'] for item in items}
        if len(references) != len(items):
            raise ValueError('开发集参考记录有重复')
    else:
        references = {item['sample_id']: item['reference'] for item in rows}
    expected_count = 81 if dataset_id == 'sar81' else 332
    if len(rows) != expected_count or len({r['sample_id'] for r in rows}) != len(rows) or set(references) != {r['sample_id'] for r in rows}:
        raise ValueError('开发集记录不完整或不匹配')
    entries = []
    for index, row in enumerate(rows):
        label = references[row['sample_id']]
        if label not in CLASSES:
            raise ValueError('未知类别')
        path = (ROOT / row['image_path']).resolve()
        roots = [ROOT / 'data/raw/sardet_expansion_20261112', ROOT / 'data/raw/sardet_expansion_v2_20261112']
        if not any(path.is_relative_to(root.resolve()) for root in roots) or path.suffix.lower() not in {'.png', '.jpg', '.jpeg', '.bmp'}:
            raise ValueError('开发图像路径不在允许范围')
        entries.append({'index': index, 'sample_id': row['sample_id'], 'label': label,
                        'path': path, 'sha256': row['image_sha256']})
    return cfg, entries


def summary(dataset_id):
    cfg, entries = dataset(dataset_id)
    counts = Counter(item['label'] for item in entries)
    return {'id': dataset_id, 'name': cfg['name'], 'description': cfg['description'], 'limit': cfg['limit'],
            'source': 'SARDet 相关数据，作者 train 分区', 'stage': '开发数据', 'total': len(entries),
            'classes': [{'id': k, 'name': v, 'count': counts[k]} for k, v in CLASSES.items()]}


def browse(dataset_id, category='', page=1, page_size=12):
    cfg, entries = dataset(dataset_id)
    if category and category not in CLASSES:
        raise ValueError('未知类别')
    if not isinstance(page, int) or page < 1 or page_size != 12:
        raise ValueError('无效页码')
    filtered = [r for r in entries if not category or r['label'] == category]
    pages = max(1, (len(filtered) + page_size - 1) // page_size)
    if page > pages:
        raise ValueError('页码超出范围')
    results = [{'sample_id': r['sample_id'], 'label': r['label'], 'label_name': CLASSES[r['label']],
                'image_url': f"/api/datasets/{dataset_id}/images/{r['index']}"}
               for r in filtered[(page - 1) * page_size:page * page_size]]
    return {'dataset': summary(dataset_id), 'page': page, 'pages': pages, 'filtered_total': len(filtered), 'items': results}


def image(dataset_id, index):
    _, entries = dataset(dataset_id)
    if not isinstance(index, int) or not 0 <= index < len(entries):
        raise ValueError('图像不存在')
    entry = entries[index]
    path = entry['path']
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != entry['sha256']:
        raise ValueError('图像哈希与协议不一致')
    return content, {'.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.bmp': 'image/bmp'}[path.suffix.lower()]
