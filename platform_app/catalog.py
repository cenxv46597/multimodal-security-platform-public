"""Capabilities and compatible configurations of the local dashboard."""

GOALS = [
    {'id': 'robustness', 'name': '对抗鲁棒性与迁移', 'status': 'report_only'},
    {'id': 'injection', 'name': '提示词注入', 'status': 'planned'},
    {'id': 'jailbreak', 'name': '越狱提示', 'status': 'planned'},
]
METHODS = [{'id': 'reeval', 'name': '已有变化图与随机对照 · 单图本地复评',
            'goal': 'robustness', 'representation': 'sar_image', 'status': 'available'},
           {'id': 'step', 'name': '单步变化与随机对照 · 12图历史报告',
            'goal': 'robustness', 'representation': 'sar_image', 'status': 'report_only'}]
DATASETS = [{'id': 'sar12', 'name': 'SAR 开发图 · 六类共 12 张',
             'representation': 'sar_image', 'method': 'step', 'status': 'report_only'},
            {'id': 'sar1', 'name': 'SAR 已使用开发图 · 1 张',
             'representation': 'sar_image', 'method': 'reeval', 'status': 'available'}]
MODELS = [{'id': 'local', 'name': 'mPLUG 1B → InternVL2B · 均为本地模型',
           'representation': 'sar_image', 'dataset': 'sar12', 'status': 'report_only'},
          {'id': 'internvl2b', 'name': 'InternVL2B · 本地目标复评',
           'representation': 'sar_image', 'dataset': 'sar1', 'status': 'available'}]


def options(stage, selection):
    if stage == 'goals':
        return GOALS
    if stage == 'methods':
        if selection.get('goal') not in {item['id'] for item in GOALS}:
            raise ValueError('请先选择有效的研究目标。')
        return [item for item in METHODS if item['goal'] == selection['goal']]
    methods = options('methods', selection)
    method = next((item for item in methods if item['id'] == selection.get('method')), None)
    if method is None:
        raise ValueError('攻击方式与研究目标不兼容或尚未接入。')
    if stage == 'datasets':
        return [item for item in DATASETS if item['representation'] == method['representation'] and item['method'] == method['id']]
    datasets = options('datasets', selection)
    dataset = next((item for item in datasets if item['id'] == selection.get('dataset')), None)
    if dataset is None:
        raise ValueError('数据范围与攻击方式不兼容或尚未接入。')
    if stage == 'models':
        return [item for item in MODELS if item['representation'] == dataset['representation'] and item['dataset'] == dataset['id']]
    raise ValueError('未知的选择步骤。')


def validate(selection):
    models = options('models', selection)
    if not any(item['id'] == selection.get('models') for item in models):
        raise ValueError('模型组合与所选数据不兼容或尚未接入。')
    if selection['method'] == 'reeval':
        return {'valid': True, 'mode': 'local_reeval', 'execution_available': True,
                'message': '可运行一张已使用开发图的本地复评；复用已有变化图，不生成新攻击。'}
    return {'valid': True, 'mode': 'report_only', 'execution_available': False,
            'message': '配置可用于查看12张开发图的已有报告。'}
