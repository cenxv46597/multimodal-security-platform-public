"""Read-only, provenance-checked views of the November 24–26 experiments."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FILES = {
    'local_dual': ('runs/sar-route1-cohort17-avlm16-20261125/summary.json', '84708afd0806bf46fc5bb108d44d1ab5d0ecaac11dc8534a6a1ab34e3596db46'),
    'local_single': ('runs/sar-route1-controls17-20261125/summary.json', '931e01a7f28c9d6c5132e5f8df4d89fb86aeded746f9279f3af097410c08f95f'),
    'local_old': ('runs/sar-route1-oldmethod17-20261125/summary.json', 'f78fc896744478eea0e99394776aa86c9c46a9be92f69b00936fcc017be53fb1'),
    'cloud': ('runs/closed-source-transfer16-20261126/audit.json', '5065411a854d2d9ec9d296d3fc25451b00e87c08d976695e28fbc3b2777ba408'),
    'cloud_plan': ('runs/closed-source-transfer16-20261126/manifest.json', '75fe420377b1205bff463de85d979b8ae101c397226bcce0e6a464247626acd7'),
    'prompt': ('runs/qwen-prompt-compare14-20261126/assessment.json', '8830aebfb03c59a3b8405408c4203cf1efca825b281b50ab0a220a0056569a11'),
    'prompt_plan': ('runs/qwen-prompt-compare14-20261126/manifest.json', '1f617441058942a5ad00cd49df0f1f3d8180a576f30c5ea7ec94fec11727fe8f'),
    'pilot4': ('runs/qwen-limited-transfer4-20261126/assessment.json', '993336433f8f3c506a56bb43d08976576d5f72c5e5f6156dfb48194a226e322c'),
    'pilot14': ('runs/qwen-limited-transfer14-20261126/assessment.json', 'fd37e5dba2fea1e05d80652e51ecb6a3359cb5ed0709bf90fa5f593a71611120'),
    'clean14': ('runs/qwen-limited-clean14-20261126/assessment.json', '729c328f884ad7ef9dfb4fb883fcc3c0499279722eb6e4fe5313fbad89e5a477'),
    'engineering8': ('runs/vision-safety-capability-preflight-20261114/sar-engineering-day09-perturbed-20261124/assessment.json', '1905ffab929af3c708beff6bd8c36f7de0242f5726142ff33d61398d433a08fe'),
    'cloud_clean': ('runs/closed-source-clean17-20261126/audit-20261126.json', 'b4164fd2b27c93426ad0c0ffe0aa84c280089e63e5a2951d5c2789e7fbb9d71b'),
}


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(root, key):
    rel, expected = FILES[key]
    path = root / rel
    if _digest(path) != expected:
        raise ValueError(f'实验汇总哈希不一致: {key}')
    return json.loads(path.read_text(encoding='utf-8'))


def _verified_record(root, item):
    path = root / item['file']
    if not path.resolve().is_relative_to(root.resolve()) or _digest(path) != item['record_sha256']:
        raise ValueError('云端原答记录哈希不一致')
    data = json.loads(path.read_text(encoding='utf-8'))
    choice = data['response_json']['choices'][0]
    if choice['message']['content'] != item['answer'] or choice.get('finish_reason') != item['finish_reason']:
        raise ValueError('云端原答与判读不一致')


def load(root=ROOT):
    root = Path(root)
    data = {key: _read(root, key) for key in FILES}
    dual, single, old = (data[k] for k in ('local_dual', 'local_single', 'local_old'))
    if any(x['n_samples'] != 17 or len(x['items']) != 17 or x['eps'] != 16 for x in (dual, single, old)):
        raise ValueError('本地对照分母或预算不一致')
    local_conditions = [('幅度匹配随机', single, 'magnitude_random16'), ('mPLUG 单源', single, 'mplug_single16'),
                        ('GLM 单源', single, 'glm_single16'), ('双源视觉原型', dual, 'avlm_mm_glm16'),
                        ('旧方法双源', old, 'oldmethod16')]
    local_metrics = []
    for label, source, condition in local_conditions:
        value = source['aggregate'][condition]['internvl2b']
        if sum(value[k] for k in ('correct', 'wrong', 'indeterminate')) != 17:
            raise ValueError('本地结果计数不完整')
        wrong = hits = 0
        for row in source['items']:
            asset = row['assets'][condition]
            path = root / asset['path']
            if not path.resolve().is_relative_to(root.resolve()) or _digest(path) != asset['sha256']:
                raise ValueError('本地图像哈希不一致')
            answer = row['answers']['internvl2b'][condition]
            if answer not in {f'It is {x}' for x in ('ship','aircraft','car','tank','bridge','harbor')}:
                raise ValueError('本地原答需要重新判读')
            category = answer.removeprefix('It is ')
            wrong += category != row['reference']
            hits += category != row['reference'] and category == row['target_category']
        if wrong != value['wrong'] or hits != value['target_hit'] or value['indeterminate']:
            raise ValueError('本地原答与汇总不一致')
        local_metrics.append({'label': label, 'wrong': value['wrong'], 'target_hit': value['target_hit'], 'denominator': 17})

    cloud = data['cloud']
    if cloud['manifest_sha256'] != FILES['cloud_plan'][1]:
        raise ValueError('闭源计划哈希不一致')
    groups = []
    for key, title in [('qwen_fewshot6', '百炼 · 六示例'), ('doubao_fewshot6', '豆包 · 六示例'), ('doubao_zeroshot', '豆包 · 零样本')]:
        group, plan = cloud['groups'][key], data['cloud_plan']['groups'][key]
        if group['planned'] != plan['planned'] or len(group['items']) != plan['planned']:
            raise ValueError('闭源实验分母不一致')
        metrics = []
        labels = {'magnitude_random16':'幅度匹配随机', 'mplug_single16':'mPLUG 单源', 'glm_single16':'GLM 单源', 'avlm_mm_glm16':'双源视觉原型'}
        for condition, label in labels.items():
            rows = [x for x in group['items'] if x['condition'] == condition]
            if len(rows) != len(plan['rows']):
                raise ValueError('闭源条件缺失')
            for item in rows:
                if item.get('record_sha256'):
                    _verified_record(root, item)
            done = sum(x['status'] in ('correct', 'wrong', 'indeterminate', 'truncated') for x in rows)
            metrics.append({'label':label, 'wrong':sum(x['status']=='wrong' for x in rows),
                            'target_hit':sum(bool(x.get('target_hit')) for x in rows), 'completed':done, 'planned':len(rows)})
        groups.append({'id':key, 'title':title, 'planned':group['planned'],
                       'completed':sum(x['completed'] for x in metrics), 'metrics':metrics})

    prompt=data['prompt']
    if prompt['manifest_sha256'] != FILES['prompt_plan'][1] or len(prompt['items']) != 14:
        raise ValueError('提示实验计划不一致')
    prompt_metrics=[]
    for key,label in [('fewshot6','原六示例'), ('domain_prior','领域先验'), ('domain_plus_fewshot6','领域先验加六示例')]:
        count=prompt['levels'][key]
        if sum(count.get(k,0) for k in ('agrees','disagrees','truncated','indeterminate')) != 14:
            raise ValueError('提示结果分母不一致')
        prompt_metrics.append({'label':label,'agree':count.get('agrees',0),'denominator':14})
    for row in prompt['items']:
        for key in ('fewshot6','domain_prior','domain_plus_fewshot6'):
            path=(root/'runs/qwen-limited-clean14-20261126' if key=='fewshot6' else root/'runs/qwen-prompt-compare14-20261126'/key)/(row['sample_id']+'.json')
            if _digest(path) != row['conditions'][key]['record_sha256']:
                raise ValueError('提示实验原答哈希不一致')
            raw = json.loads(path.read_text(encoding='utf-8'))['response_json']['choices'][0]
            if raw['message']['content'] != row['conditions'][key]['answer'] or raw.get('finish_reason') != row['conditions'][key]['finish_reason']:
                raise ValueError('提示实验原答不一致')

    engineering8=[]
    for key,label in [('clean','原图'),('random8','随机对照'),('tisimi8','定向扰动'),('visual_ablation8','去视觉损失')]:
        source=data['engineering8']['models']['mplug']['conditions'][key]
        target=data['engineering8']['models']['internvl2b']['conditions'][key]
        if source['planned'] != 17 or target['planned'] != 17 or source['effective'] != 17 or target['effective'] != 17:
            raise ValueError('工程四条件分母不一致')
        engineering8.append({'label':label,'source_correct':source['correct'],'source_hit':source['specified_hits'],
                             'target_correct':target['correct'],'target_hit':target['specified_hits'],'denominator':17})
    cloud_clean=[]
    for provider,title in [('qwen','百炼原图能力'),('doubao','豆包原图能力')]:
        group=data['cloud_clean']['providers'][provider]
        summary_path=root/group['source_summary']
        if not summary_path.resolve().is_relative_to(root.resolve()) or _digest(summary_path)!=group['source_sha256']:
            raise ValueError('云端原图汇总哈希不一致')
        original=json.loads(summary_path.read_text(encoding='utf-8'))
        metrics=[]
        for level,label in [('zeroshot','零样本'),('domain_prior','领域先验'),('fewshot6','六示例')]:
            review=group['levels'][level]
            if len(review['items'])!=17:
                raise ValueError('云端原图分母不一致')
            lookup={item['sample_id']:item for item in original['levels'][level]['items']}
            for item in review['items']:
                ordinal=lookup[item['sample_id']]['frozen_ordinal']
                path=summary_path.parent/level/f"{ordinal:02d}_{item['sample_id']}.json"
                if _digest(path)!=item['record_sha256']:
                    raise ValueError('云端原图记录哈希不一致')
            counts=review['counts']
            metrics.append({'label':label,'correct':counts.get('correct',0),'truncated':counts.get('truncated',0),'denominator':17})
        cloud_clean.append({'title':title,'metrics':metrics})

    pilots=[]
    for key,title in [('pilot4','先行四图'),('pilot14','14 图扩展中的原图一致子集')]:
        source=data[key]; metrics=[]
        for cond,label in [('magnitude_random16','幅度匹配随机'),('mplug_single16','mPLUG 单源'),('glm_single16','GLM 单源'),('avlm_mm_glm16','双源视觉原型')]:
            result=source['modes'][cond]
            counts=result['counts']; n=len(result['items'])
            if n!=4 or sum(counts.get(k,0) for k in ('correct','wrong','indeterminate','uninterpretable','truncated'))!=n:
                raise ValueError('工程探索计数不完整')
            run = 'qwen-limited-transfer4-20261126' if key == 'pilot4' else 'qwen-limited-transfer14-20261126'
            for item in result['items']:
                path=root/'runs'/run/'responses'/cond/(item['sample_id']+'.json')
                if _digest(path)!=item['response_sha256']:
                    raise ValueError('工程探索原答哈希不一致')
            metrics.append({'label':label,'wrong':counts.get('wrong',0),'target_hit':counts.get('target_hit',0),'denominator':n})
        pilots.append({'title':title,'metrics':metrics})
    return {'title':'近期多源迁移与闭源评测', 'local_metrics':local_metrics, 'engineering8':engineering8, 'cloud_clean':cloud_clean, 'cloud_groups':groups,
            'prompt_metrics':prompt_metrics, 'pilots':pilots, 'clean14':{'planned':data['clean14']['planned'],'agree':data['clean14']['eligible_count']},
            'limitations':['历史只读记录，不会运行模型或发送请求。', '工程样本、提示和原图正确子集经过筛选；参考类别、场景独立性、逐图使用条件和训练暴露仍有限制。',
                           '错分与预设类别命中是不同指标；各批样本和分母不能直接合并。', '豆包实验未完成；本地 17 图与闭源 11 图属于不同目标模型和样本口径。',
                           '视觉原型双源在本地 17 图没有优于两个单源；闭源与小样本观察不能证明稳定泛化。']}
