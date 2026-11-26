"""Read-only, hash-verified adapter for the 2026-11-19/20 SAR transfer, hard-label query, and Qwen2-VL SAR suites."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def load_tisimi_six(root: Path = ROOT) -> dict:
    protocol_path = root / 'experiments/protocols/sar_tisimi_transfer_six_20261119.json'
    run_dir = root / 'runs/sar-tisimi-transfer-six-20261119'
    sem_path = run_dir / 'semantic_results.json'
    sem = _load_json(sem_path)
    if _sha256(protocol_path) != sem['protocol_sha256']:
        raise ValueError('TI-SI-MI protocol hash mismatch')
    eval_map = {
        'mplug_part1': run_dir / 'mplug-eval-part1.json',
        'mplug_part2': run_dir / 'mplug-eval-part2.json',
        'internvl2b_part1': run_dir / 'internvl2b-eval-part1.json',
        'internvl2b_part2': run_dir / 'internvl2b-eval-part2.json',
    }
    for key, path in eval_map.items():
        if _sha256(path) != sem['eval_sha256'][key]:
            raise ValueError(f'TI-SI-MI eval hash mismatch: {key}')
        data = _load_json(path)
        if data.get('status') != 'completed':
            raise ValueError(f'TI-SI-MI eval incomplete: {key}')
    if len(sem['items']) != 6:
        raise ValueError('Expected 6 TI-SI-MI items')
    return {
        'id': 'tisimi_six',
        'title': 'SAR 六图 TI-SI-MI 跨尺度平移不变定向迁移实验（2026-11-19）',
        'scope': sem['scope'],
        'denominator': 6,
        'protocol_sha256': sem['protocol_sha256'],
        'counts': sem['counts'],
        'items': sem['items'],
        'limitations': [
            '6 张均为已参与开发的条件子集样本，与历史 12 图零重合，未使用 243 张未冻结候选。',
            '单源 1B 代理对异构 2B 目标在 ε=4/255 下迁移 0/6、ε=8/255 下仅 1/6（同图随机对照 0/6），不能据此宣称稳定迁移。',
        ],
    }


def load_hardlabel_six(root: Path = ROOT) -> dict:
    protocol_path = root / 'experiments/protocols/sar_hardlabel_query_six_20261119.json'
    run_dir = root / 'runs/sar-hardlabel-query-six-20261119'
    summary = _load_json(run_dir / 'summary.json')
    if _sha256(protocol_path) != summary['protocol_sha256']:
        raise ValueError('Hard-label protocol hash mismatch')
    plan = _load_json(protocol_path)
    if len(summary['items']) != 6 or summary['summary']['passed'] != 6:
        raise ValueError('Expected 6 completed hard-label items')
    for row in plan['rows']:
        rec = _load_json(run_dir / f"query_{row['index']}.json")
        if rec.get('status') != 'passed':
            raise ValueError(f"Hard-label item not passed: {row['index']}")
        for key in ('asset', 'matched_random_control'):
            p = root / rec[key]['path']
            if _sha256(p) != rec[key]['sha256']:
                raise ValueError(f'Hard-label asset hash mismatch: {p}')
    return {
        'id': 'hardlabel_six',
        'title': 'SAR 六图严格硬标签（仅文本返回）黑盒查询实验（2026-11-19）',
        'scope': plan['scope'],
        'denominator': 6,
        'protocol_sha256': summary['protocol_sha256'],
        'summary': summary['summary'],
        'items': summary['items'],
        'limitations': [
            '本实验无 L∞ 微扰约束，最终最大像素差跨度为 48–211/255，属大幅可见混合/替换，不能称为不可察觉扰动。',
            '同掩膜同 L∞ 随机对照亦错 4/6，仅 2/6（船、汽车）为查询错而随机对；目标为本地开源模型，非真实闭源服务。',
        ],
    }


def load_qwen2vl_sar(root: Path = ROOT) -> dict:
    base_proto_path = root / 'experiments/protocols/qwen2vl_sar_six_baseline_20261120.json'
    inj_proto_path = root / 'experiments/protocols/qwen2vl_sar_injection_three_20261120.json'
    base_dir = root / 'runs/qwen2vl-sar-six-baseline-20261120'
    inj_dir = root / 'runs/qwen2vl-sar-injection-three-20261120'

    zeroshot = _load_json(base_dir / 'zeroshot.json')
    domain_prior = _load_json(base_dir / 'domain_prior.json')
    inj = _load_json(inj_dir / 'results.json')

    base_sha = _sha256(base_proto_path)
    inj_sha = _sha256(inj_proto_path)
    if zeroshot['protocol_sha256'] != base_sha or domain_prior['protocol_sha256'] != base_sha:
        raise ValueError('Qwen2-VL SAR baseline protocol hash mismatch')
    if inj['protocol_sha256'] != inj_sha:
        raise ValueError('Qwen2-VL SAR injection protocol hash mismatch')
    if zeroshot.get('status') != 'completed' or domain_prior.get('status') != 'completed' or inj.get('status') != 'completed':
        raise ValueError('Qwen2-VL SAR run incomplete')

    inj_plan = _load_json(inj_proto_path)
    for row in inj_plan['rows']:
        if _sha256(root / row['image_path']) != row['image_sha256']:
            raise ValueError(f"Qwen2-VL SAR image hash mismatch: {row['sample_id']}")

    preds = inj['predictions']
    if len(preds) != 18:
        raise ValueError('Expected 18 Qwen2-VL SAR injection predictions')

    by_cond = {}
    for p in preds:
        by_cond.setdefault(p['condition'], []).append(p)

    inj_summary = {
        'denominator': 3,
        'blank_correct': sum(x['classification_judgment'] == 'correct' for x in by_cond['blank_classify']),
        'benign_correct': sum(x['classification_judgment'] == 'correct' for x in by_cond['benign_classify']),
        'attack_correct': sum(x['classification_judgment'] == 'correct' for x in by_cond['attack_classify']),
        'benign_review': sum(x['contains_REVIEW'] for x in by_cond['benign_classify']),
        'attack_review': sum(x['contains_REVIEW'] for x in by_cond['attack_classify']),
        'attack_ocr_exact': sum(bool(x['ocr_exact']) for x in by_cond['attack_ocr']),
        'benign_ocr_contains': sum(bool(x['ocr_contains']) for x in by_cond['benign_ocr']),
        'user_positive_review': sum(x['contains_REVIEW'] for x in by_cond['user_positive']),
    }
    return {
        'id': 'qwen2vl_sar',
        'title': 'Qwen2-VL-2B-Instruct：SAR 六图基线与三图图片文字注入对照（2026-11-20）',
        'scope': '本地离线通用 2B 模型开发观察；基线分母 6 与注入条件子集分母 3 分开统计，非独立测试。',
        'baseline': {
            'denominator': 6,
            'protocol_sha256': base_sha,
            'zeroshot_counts': zeroshot['counts'],
            'domain_prior_counts': domain_prior['counts'],
        },
        'injection': {
            'protocol_sha256': inj_sha,
            'summary': inj_summary,
            'predictions': preds,
        },
        'limitations': [
            '3 张注入图为 zeroshot 正确子集（汽车、储油罐、桥），仅加 96px 白栏后空白分类即由 3/3 降至 1/3。',
            '虽然攻击短句独立 OCR 精确 2/3，但可信用户正向对照（句末附加 REVIEW）仍为 0/3，故攻击 0/3 不能解释为安全防御成功。',
        ],
    }


def load_tisimi_ablation_six(root: Path = ROOT) -> dict:
    audit_path = root / '.work/sar_six_evidence_audit_20261122.json'
    proto_path = root / 'experiments/protocols/sar_tisimi_transfer_six_20261119.json'
    audit = _load_json(audit_path)
    if _sha256(proto_path) != audit['source_protocol_sha256']:
        raise ValueError('Six-image evidence audit protocol hash mismatch')
    items = audit['items']
    if len(items) != 6:
        raise ValueError('Expected 6 items in six-image evidence audit')

    def _hit(model_key: str, cond: str) -> int:
        return sum(x['models'][model_key][cond] == f"It is {x['target']}" for x in items)

    def _correct(model_key: str, cond: str) -> int:
        return sum(x['models'][model_key][cond] == f"It is {x['reference']}" for x in items)

    rs1_correct = sum(x['random_sign'][0]['answer'] == f"It is {x['reference']}" for x in items)
    rs2_correct = sum(x['random_sign'][1]['answer'] == f"It is {x['reference']}" for x in items)
    return {
        'id': 'tisimi_ablation_six',
        'title': 'SAR 六图视觉损失消融与保幅随机符号对照（2026-11-21–22）',
        'scope': '6 张预筛选开发图核对表；原方法与去视觉损失 12 条离散回答逐条一致，属开发观察，非独立测试。',
        'denominator': 6,
        'protocol_sha256': audit['source_protocol_sha256'],
        'summary': {
            'mplug_full_target_hit': _hit('mplug', 'full'),
            'mplug_ablation_target_hit': _hit('mplug', 'ablation'),
            'internvl2b_full_target_hit': _hit('internvl2b', 'full'),
            'internvl2b_ablation_target_hit': _hit('internvl2b', 'ablation'),
            'internvl2b_random8_correct': _correct('internvl2b', 'random8'),
            'internvl2b_random_sign_1_correct': rs1_correct,
            'internvl2b_random_sign_2_correct': rs2_correct,
            'prior_failures': len(audit.get('prior_evaluation_failures', [])),
        },
        'items': items,
        'limitations': [
            '船图为先行试验，其余五图事后登记，不能称六图消融整体事前预登记；离散回答一致不证明视觉损失在所有样本或分数上无作用。',
            '两组逐像素保幅随机符号对照均答对 6/6，但共用六张图与原攻击幅度图，不可计为 12 张独立图；两次历史进程失败记录均保留。',
        ],
    }


def load_engineering_cohort_v1(root: Path = ROOT) -> dict:
    cohort_path = root / '.work/sar_engineering_evaluation_cohort_v1_20261123.json'
    proto_path = root / 'experiments/protocols/sar_paired_execution_20261123.json'
    cohort = _load_json(cohort_path)
    proto = _load_json(proto_path)
    proto_sha = _sha256(proto_path)
    if cohort['protocol_sha256'] != proto_sha:
        raise ValueError('Engineering cohort v1 protocol hash mismatch')
    items = cohort['items']
    if len(items) != cohort['count'] or len(items) != 18:
        raise ValueError('Expected 18 items in engineering cohort v1')
    for item in items:
        img_path = (root / item['image_path']).resolve()
        if not img_path.is_relative_to(root.resolve()) or _sha256(img_path) != item['image_sha256']:
            raise ValueError(f"Engineering cohort image hash mismatch: {item['sample_id']}")
        if item.get('decoded_rgb_channels_equal') is not True:
            raise ValueError(f"Engineering cohort grayscale check failed: {item['sample_id']}")
    return {
        'id': 'engineering_cohort_v1',
        'title': 'SAR 受限工程评测快照 v1 与四条件配对协议（2026-11-23）',
        'scope': '18 张经非专家工程初筛与灰度核验的固定输入快照；此处展示快照建立时的记录，后续评测另见最新实验，非独立测试集。',
        'version': cohort['version'],
        'status': cohort['status'],
        'denominator': cohort['count'],
        'class_counts': cohort['class_counts'],
        'protocol_sha256': proto_sha,
        'limits': proto['limits'],
        'conditions': proto['conditions'],
        'excluded_hrsid_count': len(cohort.get('excluded_hrsid', [])),
        'excluded_hrsid': cohort.get('excluded_hrsid', []),
        'items': [
            {
                'sample_id': x['sample_id'],
                'reference': x['reference'],
                'source_family': x['source_family'],
                'image_size': x['image_size'],
                'admission_basis': x['admission_basis'],
                'development_pairs': x['development_pairs'],
            }
            for x in items
        ],
        'limitations': [
            '本快照为受限工程评测集，非独立无偏测试集；不含外部雷达专家重标，上游逐图许可、模型训练暴露与异尺寸局部重叠仍未完全闭合。',
            '原 2 张 HRSID 初筛船图因存在 SARDet train 裁片坐标交集线索已移出并单列标记，由 2 张 ShipDataset 船图替补；模型调用与预测结果当前均为 0。',
        ],
    }


LOADERS = {
    'tisimi_six': load_tisimi_six,
    'hardlabel_six': load_hardlabel_six,
    'qwen2vl_sar': load_qwen2vl_sar,
    'tisimi_ablation_six': load_tisimi_ablation_six,
    'engineering_cohort_v1': load_engineering_cohort_v1,
}


def load_all(root: Path = ROOT) -> dict:
    return {
        'suites': {key: loader(root) for key, loader in LOADERS.items()},
    }


def load_suite(suite_id: str, root: Path = ROOT) -> dict:
    if suite_id not in LOADERS:
        raise KeyError(suite_id)
    return LOADERS[suite_id](root)
