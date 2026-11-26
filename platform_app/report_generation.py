"""Preview or validate a report, or explicitly run one DeepSeek injection-report trial."""
import argparse
import getpass
import json
import os
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener

import report_inputs

MAX_RESPONSE_CHARS = 8000
DEEPSEEK_URL = 'https://api.deepseek.com/chat/completions'
DEEPSEEK_MODEL = 'deepseek-flash'
SYSTEM_PROMPT = (
    '你负责撰写中文 SAR 应用安全分析，评估对象是实验中的模型行为。只使用输入中的汇总事实。'
    '先说明不可信输入在什么应用条件下可能影响下游，再以实验对照说明这个风险得到何种证据支持、还有什么无法判断。'
    '给出与该风险直接相关的防护或补证建议，并明确建议尚待验证。平台自身的账号、接口与部署安全另行评估。'
    '每段引用支撑它的指标编号。标题和正文不写数字，带数字的模型名用源模型或本地目标等角色称呼。'
    '平台会从指标填入模型名、数量和分母。'
    '遵守输入的解释规则，不作无依据的防护、因果或总体安全结论。'
    '只返回一个 JSON 对象，字段 sections 为数组，每项只包含 title、analysis、metric_ids。'
    '格式示例：{"sections":[{"title":"对照分析","analysis":"依据所引用的本轮对照展开。","metric_ids":["实际指标编号"]}]}。'
)


def build_request(facts):
    """Project verified facts onto the fields needed for writing."""
    metrics = {
        key: {field: item[field] for field in ('label', 'numerator', 'denominator')}
        for key, item in facts['metrics'].items()
    }
    conditions = []
    for item in facts['conditions']:
        condition = {key: item[key] for key in (
            'model', 'label', 'planned', 'completed', 'failed', 'interrupted', 'not_run', 'truncated'
        )}
        condition['judgments'] = (
            {key: item['judgments'][key] for key in ('correct', 'wrong', 'unresolved')}
            if item['judgments'] is not None else None
        )
        conditions.append(condition)
    content = {
        'schema_version': 1,
        'experiment': facts['experiment'],
        'stage': facts['stage'],
        'scope': facts['scope'],
        'planned': facts['planned'],
        'statuses': {key: facts['statuses'][key] for key in ('completed', 'failed', 'interrupted', 'not_run')},
        'conditions': conditions,
        'metrics': metrics,
        'rules': list(facts['rules']),
    }
    return {
        'messages': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': json.dumps(content, ensure_ascii=False)},
        ],
    }


def unique_object(pairs):
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError('报告 JSON 中存在重复字段')
    return result


def validate_response(facts, response):
    if not isinstance(response, str) or len(response) > MAX_RESPONSE_CHARS:
        raise ValueError('报告返回内容不是文本或过长')
    decoded = json.loads(response, object_pairs_hook=unique_object)
    if not isinstance(decoded, dict) or set(decoded) != {'sections'}:
        raise ValueError('报告返回应为仅含 sections 的 JSON 对象')
    sections = decoded['sections']
    if (not isinstance(sections, list) or not 1 <= len(sections) <= 6
            or any(not isinstance(item, dict) for item in sections)):
        raise ValueError('报告段落数量或类型无效')
    report_inputs.validate_example(facts, sections)
    for item in sections:
        if (len(item['title']) > 80 or len(item['analysis']) > 800
                or len(set(item['metric_ids'])) != len(item['metric_ids'])
                or any(char.isdecimal() for char in item['title'] + item['analysis'])):
            raise ValueError('报告段落过长、重复引用或在正文中直接提供了阿拉伯数字')
    return {
        'scope': facts['scope'],
        'limitations': facts['limitations'],
        'sections': [
            {
                'title': item['title'],
                'analysis': item['analysis'],
                'metrics': [
                    {'id': key, **facts['metrics'][key]} for key in item['metric_ids']
                ],
            }
            for item in sections
        ],
    }


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def deepseek_trial(facts, api_key):
    if facts['experiment'] != 'injection':
        raise ValueError('本次真实调用仅限六图注入汇总')
    if not api_key or any(char.isspace() for char in api_key):
        raise ValueError('缺少有效的 DeepSeek API 密钥')
    payload = {
        **build_request(facts),
        'model': DEEPSEEK_MODEL,
        'response_format': {'type': 'json_object'},
        'thinking': {'type': 'disabled'},
        'max_tokens': 900,
        'stream': False,
    }
    request = Request(
        DEEPSEEK_URL,
        data=json.dumps(payload, ensure_ascii=False).encode('utf-8'),
        headers={'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'},
        method='POST',
    )
    try:
        with build_opener(NoRedirect()).open(request, timeout=40) as response:
            body = response.read(65537)
    except HTTPError as error:
        raise ValueError(f'DeepSeek 调用失败（HTTP {error.code}）') from None
    except (URLError, TimeoutError):
        raise ValueError('DeepSeek 连接失败或超时') from None
    if len(body) > 65536:
        raise ValueError('DeepSeek 返回超过大小上限')
    try:
        envelope = json.loads(body)
        choice, = envelope['choices']
        if choice['finish_reason'] != 'stop':
            raise ValueError('DeepSeek 未正常完成生成')
        validated = validate_response(facts, choice['message']['content'])
    except (KeyError, TypeError, UnicodeError, ValueError):
        raise ValueError('DeepSeek 输出未通过报告格式或证据引用校验') from None
    usage = envelope.get('usage') or {}
    return {
        'provider': 'DeepSeek',
        'model': envelope.get('model', DEEPSEEK_MODEL),
        'usage': {key: usage[key] for key in ('prompt_tokens', 'completion_tokens', 'total_tokens') if key in usage},
        'review_status': 'pending_human_review',
        **validated,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment', choices=('perturbation', 'injection'))
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--preview', action='store_true', help='show proposed model request locally')
    modes.add_argument('--simulate', action='store_true', help='validate the saved local example')
    modes.add_argument('--response-file', type=Path, help='validate a local candidate JSON response')
    modes.add_argument('--deepseek', action='store_true', help='send the injection aggregate once to DeepSeek')
    parser.add_argument('--output', type=Path, help='save the result as a new UTF-8 JSON file')
    args = parser.parse_args()
    if args.deepseek and args.experiment != 'injection':
        parser.error('--deepseek is only available for injection')
    if args.output and (args.output.exists() or not args.output.parent.is_dir()):
        parser.error('--output requires a new file in an existing directory')
    data = report_inputs.load(args.experiment)
    facts = data['facts']
    if args.preview:
        output = build_request(facts)
    elif args.deepseek:
        key = os.environ.get('DEEPSEEK_API_KEY', '').strip()
        if not key and sys.stdin.isatty():
            key = getpass.getpass('DeepSeek API key (hidden): ').strip()
        try:
            output = deepseek_trial(facts, key)
        except ValueError as error:
            parser.exit(1, f'{error}\n')
    else:
        response = (args.response_file.read_text(encoding='utf-8') if args.response_file
                    else json.dumps({'sections': data['example']}, ensure_ascii=False))
        output = validate_response(facts, response)
    serialized = json.dumps(output, ensure_ascii=False, indent=2)
    if args.output:
        try:
            with args.output.open('x', encoding='utf-8') as file:
                file.write(serialized + '\n')
        except OSError as error:
            print(serialized)
            parser.exit(1, f'无法保存报告：{error}\n')
        print(f'报告已保存：{args.output}')
    else:
        print(serialized)


if __name__ == '__main__':
    main()
