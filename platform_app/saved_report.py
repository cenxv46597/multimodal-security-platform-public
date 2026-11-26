"""Read the saved DeepSeek trial and recheck its attached facts."""
import json

from report_generation import validate_response
from report_inputs import ROOT, read

SAVED = ROOT / '.work/deepseek-injection-report.json'
CLASSIFICATION_METRICS = [f'{condition}_correct' for condition in ('blank', 'benign', 'attack')]


def load(facts):
    saved = read(SAVED)
    if (facts['experiment'] != 'injection' or saved['provider'] != 'DeepSeek'
            or saved['review_status'] != 'pending_human_review'
            or not isinstance(saved['model'], str) or not saved['model'].strip()
            or saved['scope'] != facts['scope'] or saved['limitations'] != facts['limitations']):
        raise ValueError('保存报告的来源或范围不匹配')
    sections = []
    for item in saved['sections']:
        sections.append({
            'title': item['title'],
            'analysis': item['analysis'],
            'metric_ids': [entry['id'] for entry in item['metrics']],
        })
    validated = validate_response(facts, json.dumps({'sections': sections}, ensure_ascii=False))
    if validated['sections'] != saved['sections']:
        raise ValueError('保存报告的数字或证据与当前记录不匹配')
    return {
        'provider': saved['provider'],
        'model': saved['model'],
        'review_status': saved['review_status'],
        'sections': validated['sections'],
        'review_note': {
            'analysis': '下面三个指标用于复核草稿中的分类正确数。标记是否出现与分类是否正确采用各自的统计口径。',
            'metrics': [{'id': key, **facts['metrics'][key]} for key in CLASSIFICATION_METRICS],
        },
    }
