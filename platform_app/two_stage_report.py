"""Read-only, fail-closed report for one fixed synthetic two-stage observation."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / 'runs/vision-safety-capability-preflight-20261114'
PLANS = (
    ('smolvlm_synthetic_injection_20261118.json', 'b318f3547d00af7a62b50677d5a34da6fecba015ae442d3c5f89c1103e4f8f74'),
    ('smol_two_stage_simulation_20261118.json', '1cbaf7c4b107a8f5168ea877eb204316bf8745f53a7c9eafc75a3b8e88d5779e'),
)
RESULTS = (
    ('smol-circle-01-blank-20261118', 'c75bd2f5ded48316168e51371e8fd5ba5fe2e92c8d83c997c0a2f28743f98dc9'),
    ('smol-circle-02-benign-20261118', '46f8d9841ca3fec6edb0356edfcce63f5667cb44aa59ba80067ad7393cb77655'),
    ('smol-circle-03-attack-20261118', '0e0490356af0d1c26fd4889beec22ff8d7415ca1e00e5a0974f3dfbe37218a8d'),
    ('smol-circle-04-benign-ocr-20261118', 'ac9ea949e498331ce50a4d0f892de6a4aac5bced85ca20fe07b3a7bcfc798080'),
    ('smol-circle-05-attack-ocr-20261118', '992458edc49037cd47732fcd1fa42d1618bb14b88626ac4c8dc2c52a110ed628'),
    ('smol-circle-06-user-positive-20261118', '820efa132e6b29d07469bf2cff574567e5aaefb0004659cc09b5b3c13026f7bf'),
    ('smol-stage2-01-blank-20261118', 'd7b6385a313d7b7ffba77095c956e461b2e997382a5ba241d076b46a9e9e7b99'),
    ('smol-stage2-02-benign-20261118', '6451eb055fd3cb609dcdc1cd812abb08cfbcaffacecf4db194f477d79faf449b'),
    ('smol-stage2-03-attack-20261118', '9280433e75a72ab17cd9d4769e109d699deb65469ba4473f41754cad82370f7b'),
    ('smol-stage2-04-user-positive-20261118', '5cc618057a4d198c5665c9ea915c00668632650b69da8b3130b1f9d53a9d1666'),
)
LABELS = ('空白栏计数', '普通文字栏计数', '指令文字栏计数', '普通文字独立转写', '指令文字独立转写', '可信用户正向',
          '空白数据', '普通文字', '指令文字', '可信用户正向')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def load():
    plans = []
    for name, expected in PLANS:
        path = ROOT / 'experiments/protocols' / name
        if sha(path) != expected:
            raise ValueError('总协议不一致')
        plans.append((path, read(path)))
    expected_rows = plans[0][1]['rows'] + plans[1][1]['rows']
    if len(expected_rows) != len(RESULTS):
        raise ValueError('计划数量不一致')
    records = []
    for index, ((name, expected_hash), label, row) in enumerate(zip(RESULTS, LABELS, expected_rows)):
        directory = RUNS / name
        result_path, protocol_path = directory / 'result.json', directory / 'protocol.json'
        if sha(result_path) != expected_hash:
            raise ValueError('原始结果不一致')
        protocol, result, finished = read(protocol_path), read(result_path), read(directory / 'finished.json')
        prediction, = result['predictions']
        plan_path = plans[0 if index < 6 else 1][0]
        image = (ROOT / row['image_path']).resolve()
        if (protocol['rows'] != [row] or protocol['source_protocol_sha256'] != sha(plan_path)
                or result['protocol_sha256'] != sha(protocol_path)
                or result['actual_question'] != row['question']
                or prediction['sample_id'] != row['sample_id']
                or result['status'] != 'completed' or prediction['status'] != 'ok'
                or prediction['hit_token_limit'] is not False or finished['reason'] != 'completed'
                or not image.is_relative_to(ROOT) or sha(image) != row['image_sha256']
                or (index >= 6 and result.get('input_modality') != 'text_only')):
            raise ValueError('单项协议、输入或状态不一致')
        records.append({'stage': '图文阶段' if index < 6 else '文字阶段', 'label': label,
                        'question': row['question'], 'answer': prediction['answer'],
                        'truncated': False, 'status': 'completed',
                        'source': result_path.relative_to(ROOT).as_posix()})
    for source in plans[1][1]['sources']:
        path = (ROOT / source['path']).resolve()
        if not path.is_relative_to(ROOT) or sha(path) != source['sha256']:
            raise ValueError('复用的图文阶段来源不一致')
        prediction, = read(path)['predictions']
        if prediction['answer'] != source['answer']:
            raise ValueError('复用的原答不一致')
    return {'title': '合成图文与模拟两阶段观察', 'sample_count': 1,
            'scope': '一张合成图、一个本地模型的开发观察；不是 SAR 测试或独立基准。',
            'status': {'image_planned': 6, 'image_completed': 6, 'text_planned': 4, 'text_completed': 4,
                       'failed': 0, 'interrupted': 0, 'not_run': 0, 'truncated': 0},
            'limitations': ['文字阶段的可信视觉观察由实验者提供，并非实时视觉模型生成。',
                            '图中文字在文字阶段仅以同一用户消息中的标签标记为不可信，并非系统级隔离。',
                            '独立转写不能证明计数调用中读取了文字；单例未出现目标回答不证明防护成功。',
                            '两阶段和端到端输入与提示不同，不能仅据结果断言因果关系。'],
            'records': records}
