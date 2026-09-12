"""Conversation figures bound only to the verified exp007 evidence package."""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "reports/experiments/exp007/evidence"
DEFAULT_OUTPUT = Path('/Users/vegetarianwolf/.codex/visualizations/2026/09/12/'
                      '01a09675-854f-74d3-b83e-551b231b00d0')
COST_POLICIES = [
    'exp001/original', 'exp001/legacy_rebased', 'exp002/primary', 'exp003/primary',
    'exp004/no_season', 'exp004/causal_season', 'exp005/beta_0.1',
    'exp006/primary', 'exp006/greedy_execution', 'exp007/regularized_42', 'exp007/cost_only_42',
]


def choose(rows, **where):
    matches = [r for r in rows if all(r.get(k) == v for k, v in where.items())]
    if len(matches) != 1:
        raise ValueError(f'Ambiguous evidence {where}: {len(matches)}')
    return matches[0]


def build(output=DEFAULT_OUTPUT):
    data = json.loads((EVIDENCE/'report_data.json').read_text())
    if not data['verification']['passed']:
        raise RuntimeError('Evidence must pass before presentation')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    costs = data['tables']['cost_history']
    rows = []
    for policy in COST_POLICIES:
        r = choose(costs, policy_id=policy)
        if abs(r['planned_cost']+r['emergency_cost']-r['total_cost']) > .01:
            raise ValueError('Cost components fail reconciliation')
        note = '' if r['ranking_allowed'] else '（异约束，仅并列）'
        rows.append({'label': r['label']+note, 'planned': r['planned_cost']/10000,
                     'emergency': r['emergency_cost']/10000,
                     'primary': policy == 'exp007/regularized_42',
                     'detail': f"{r['label']}；总费 {r['total_cost']:,.2f} 元；{r.get('status', '')}"})
    panels = [{'title': '实际购电总费', 'unit': '万元', 'axisTitle': '计划费 + 紧急费（万元）',
               'series': [{'key': 'planned', 'label': '计划费'}, {'key': 'emergency', 'label': '紧急费'}],
               'rows': rows}]
    forecasts = data['tables']['forecast_history']
    selected = []
    for experiment, name in [('exp001', 'midnight_rescored'), ('exp002', 'midnight_rescored'),
                             ('exp003', 'primary'), ('exp004', 'no_season'),
                             ('exp005', 'frozen_no_season'), ('exp006', 'primary'), ('exp007', 'regularized_42')]:
        candidates = [r for r in forecasts if r['experiment'] == experiment
                      and r['target'] == 'net_load' and r['population'] == 'all'
                      and (r.get('name') in (name, None) or r.get('policy_id') == experiment+'/'+name)
                      and r.get('seed', 42) == 42]
        if len(candidates) != 1:
            raise ValueError(f'Forecast evidence ambiguous: {experiment}/{name} {len(candidates)}')
        selected.append(candidates[0])
    for metric, title, unit in [('wape_pct', '净负荷预测 WAPE', '%'), ('rmse', '净负荷预测 RMSE', 'kW')]:
        panels.append({'title': title, 'unit': unit, 'axisTitle': title+f'（{unit}）',
                       'series': [{'key': 'value', 'label': title}],
                       'rows': [{'label': r['label'], 'value': r[metric],
                                 'primary': r['experiment'] == 'exp007',
                                 'detail': f"{r['label']}：{r[metric]:.6f} {unit}；exp007复用exp004无季节预测。"}
                                for r in selected]})
    time_rows = []
    for policy in ['exp001/legacy_rebased', 'exp002/primary', 'exp003/primary',
                   'exp004/no_season', 'exp006/primary', 'exp006/greedy_execution']:
        r = choose(costs, policy_id=policy)
        value = r.get('solve_execute_seconds')
        time_rows.append({'label': r['label']+'·调度执行', 'value': value,
                          'detail': '历史阶段累计秒数；设备及协议不同，仅描述性比较。'})
    current = choose(data['tables']['all_runs'], run_id='regularized_42')
    for key, label in [('training_seconds', 'exp007·RL训练'), ('planning_seconds', 'exp007·午夜规划'),
                       ('execution_seconds', 'exp007·实际执行')]:
        time_rows.append({'label': label, 'value': current[key], 'primary': True,
                          'detail': f"{label}：{current[key]:.6f} 秒，分别计时，不与墙钟重复相加。"})
    panels.append({'title': '计算阶段耗时', 'unit': '秒', 'axisTitle': '累计时间（秒；历史仅描述性比较）',
                   'series': [{'key': 'value', 'label': '秒'}], 'rows': time_rows})
    template = (ROOT/'reports/templates/exp006-inline.html').read_text()
    template = template.replace('q2-tree-history', 'q2-rl-history').replace(
        '第二问：费用、储能运行与预测误差', '第二问：RL 与历次实验').replace(
        '2025 年 2–12 月 · 334 日 · 主种子 42 · exp001 使用同物理口径重算',
        '2025 年 2–12 月 · 334 日 · 异约束行仅并列，不计算改善率')
    encoded = json.dumps({'panels': panels}, ensure_ascii=False, allow_nan=False).replace('</', '<\\/')
    (output/'q2-rl-history.html').write_text(template.replace('@@DATA@@', encoded))
    keys = ['exp004/no_season', 'exp006/primary', 'exp006/greedy_execution', 'exp007/regularized_42']
    labels = ['exp004 无季节', 'exp006 树DP', 'exp006 贪心对照', 'exp007 RL主组']
    battery = {'series': [{'key': k, 'label': label} for k, label in zip(keys, labels)], 'days': []}
    for date in data['battery']['dates']:
        values = {k: choose(data['battery']['random_curves'], policy_id=k, date=date)['power_kw'] for k in keys}
        if not all(len(v) == 144 for v in values.values()):
            raise ValueError('Random-day curves must retain all 144 samples')
        battery['days'].append({'date': date, 'values': values})
    template = (ROOT/'reports/templates/exp007-battery.inline.html').read_text()
    encoded = json.dumps(battery, ensure_ascii=False, allow_nan=False).replace('</', '<\\/')
    (output/'q2-rl-battery.html').write_text(template.replace('@@BATTERY@@', encoded))
    files = [output/'q2-rl-history.html', output/'q2-rl-battery.html']
    for path in files:
        if path.stat().st_size >= 1_000_000:
            raise ValueError('Conversation fragment exceeds size budget')
        content = path.read_text()
        if '<html' in content.lower() or '@@DATA@@' in content or '@@BATTERY@@' in content:
            raise ValueError('Unbound or non-fragment visualization')
    manifest = {'input_sha256': hashlib.sha256((EVIDENCE/'report_data.json').read_bytes()).hexdigest(),
                'outputs': [{'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
                             'bytes': p.stat().st_size} for p in files],
                'battery_days': data['battery']['dates'], 'random_seed': data['battery']['seed']}
    (EVIDENCE/'inline_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    build(parser.parse_args().output)
