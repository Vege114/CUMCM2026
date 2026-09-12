"""Regenerate exp005's report from the strict rerun, retaining the LP archive."""

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import scipy

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / 'reports/experiments/exp005'
RUN = ROOT / 'data/results/exp005/strict-mutual-exclusion-certified'
EVIDENCE = OUT / 'strict_rerun/evidence'
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))
import matplotlib  # noqa: E402

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from control_evidence import collect as collect_controls  # noqa: E402
from failure_evidence import collect as collect_failure  # noqa: E402
from prepare_workbook import periods  # noqa: E402


def read(path):
    return json.loads(path.read_text())


def dump(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False))


def rows(frame):
    return json.loads(frame.to_json(orient='records', double_precision=15))


def clock(slot):
    return f'{slot // 6:02}:{slot % 6 * 10:02}'


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--complete', action='store_true')
    parser.add_argument('--build', action='store_true')
    args = parser.parse_args()
    status = (read(RUN / 'status.json') if (RUN / 'status.json').exists()
              else read(RUN / 'pause_status.json') if (RUN / 'pause_status.json').exists() else {'status': 'running'})
    if args.complete and status['status'] == 'running':
        raise RuntimeError('replay still running; keep report updating')
    daily = pd.DataFrame(read(RUN / 'daily.json'))
    n = len(daily)
    protocol = read(RUN / 'protocol.json')
    baseline = dict(np.load(EVIDENCE / 'exp004_dispatch_2.npz'))
    raw, stage_rows, tree_rows = [], [], []
    audit_max = {key: 0. for key in ['balance', 'soc', 'overlap', 'ramp', 'bounds', 'ordinary_changed',
                                   'actual_changed', 'boundary_soc', 'node_overlap', 'node_residual', 'budget',
                                   'binary_residual', 'exclusive_residual', 'mip_gap']}
    prev_soc = protocol['initial_state']['soc']
    prev_power = protocol['initial_state']['previous_power_kw']
    totals = np.zeros(4)
    for i, date in enumerate(daily.date):
        folder = RUN / date
        z = dict(np.load(folder / 'actual.npz'))
        midnight = dict(np.load(folder / 'midnight.npz'))
        paths = dict(np.load(folder / 'scenarios.npz'))
        stages = [('midnight', -1, read(folder / 'midnight.json'))]
        stages += [('execution', t, log) for t, log in enumerate(read(folder / 'execution_logs.json'))]
        assert len(stages) == 145
        for scope, slot, meta in stages:
            assert len(meta['stages']) == 2 and meta['strict_mutual_exclusion']
            for layer, s in enumerate(meta['stages'], 1):
                assert s['status'] == 0
                stage_rows.append({'date': date, 'scope': scope, 'slot': slot, 'layer': layer, **s})
                for key, source in [('node_overlap', 'max_overlap_kwh'), ('node_residual', 'max_constraint_residual'),
                                    ('binary_residual', 'max_binary_residual'), ('exclusive_residual', 'max_exclusivity_residual'),
                                    ('mip_gap', 'mip_gap')]:
                    if s[source] is not None:
                        audit_max[key] = max(audit_max[key], s[source])
            audit_max['budget'] = max(audit_max['budget'], meta['second_cost'] - meta['budget'])
        eta = np.sqrt(.9)
        values = {
            'balance': np.max(np.abs(z['grid'] + z['actual'][:, 1] / 6 + z['discharge'] + z['emergency']
                                     - z['actual'][:, 0] / 6 - z['charge'] - z['surplus'])),
            'soc': np.max(np.abs(np.diff(z['states']) - eta * z['charge'] + z['discharge'] / eta)),
            'overlap': np.minimum(z['charge'], z['discharge']).max(),
            'ramp': np.abs(np.diff(np.r_[prev_power, z['net_power_kw']])).max(),
            'ordinary_changed': np.abs(z['grid'] - midnight['grid']).max(),
            'actual_changed': np.abs(z['actual'] - baseline['actual'][i]).max(),
            'boundary_soc': abs(z['states'][0] - prev_soc),
            'bounds': max(float(np.maximum(1200 - z['states'], 0).max()), float(np.maximum(z['states'] - 10800, 0).max()),
                          float(np.maximum(z['charge'] - 5000 / 6, 0).max()), float(np.maximum(z['discharge'] - 5000 / 6, 0).max()),
                          max(float(np.maximum(-z[k], 0).max()) for k in ['charge', 'discharge', 'grid', 'emergency', 'surplus']))}
        for key, val in values.items():
            audit_max[key] = max(audit_max[key], float(val))
        independent_fees = np.column_stack([z['grid'] * baseline['price'][i], 5 * z['emergency'] * baseline['price'][i]])
        np.testing.assert_allclose(independent_fees, z['fees'], atol=1e-6, rtol=0)
        np.testing.assert_allclose(independent_fees.sum(), daily.iloc[i].total_cost, atol=1e-6, rtol=0)
        totals += [z['grid'].sum(), z['emergency'].sum(), z['charge'].sum(), z['discharge'].sum()]
        raw.extend({'date': date, 'slot': t, 'hour': t / 6, 'interval': f'{clock(t)}—{clock(t + 1)}',
                    'grid_kwh': z['grid'][t], 'charge_kwh': z['charge'][t], 'discharge_kwh': z['discharge'][t],
                    'emergency_kwh': z['emergency'][t], 'surplus_kwh': z['surplus'][t],
                    'start_soc_kwh': z['states'][t], 'end_soc_kwh': z['states'][t + 1],
                    'power_kw': z['net_power_kw'][t], 'planned_cost': independent_fees[t, 0],
                    'emergency_cost': independent_fees[t, 1]} for t in range(144))
        tree_rows.append({'date': date, 'donors': len(paths['donor_origins']), 'paths': len(paths['paths_kw']),
                          'midnight_nodes': len(midnight['time']), 'fallback_point_only': bool(paths['fallback_point_only'])})
        prev_soc, prev_power = z['states'][-1], z['net_power_kw'][-1]
    for key, val in audit_max.items():
        if key not in ['ramp', 'mip_gap']:
            assert val <= 1e-6, (key, val)
    assert audit_max['ramp'] <= 1000 + 1e-6
    before = read(RUN / 'original_code_hashes_before.json')
    assert all(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == h for p, h in before.items())
    raw = pd.DataFrame(raw)
    specified_days = ['2025-03-20', '2025-06-21', '2025-09-23', '2025-12-21']
    specified_plan, specified_battery, specified_emergency, specified_soc = [], [], [], []
    for day in specified_days:
        subset = raw[raw.date == day]
        if len(subset) != 144:
            continue
        for t in [60, 72, 84, 96, 108, 120]:
            row = subset[subset.slot == t].iloc[0]
            specified_plan.append({'date': day, 'interval': row.interval, 'grid_kwh': row.grid_kwh})
        for block in range(6):
            a, b = block * 24, (block + 1) * 24
            part = subset[(subset.slot >= a) & (subset.slot < b)]
            specified_battery.append({'date': day, 'interval': f'{clock(a)}—{clock(b)}',
                                       'charge_kwh': float(part.charge_kwh.sum()), 'discharge_kwh': float(part.discharge_kwh.sum())})
        events = periods(subset.emergency_kwh.to_numpy())
        for a, b, energy in events or [(0, 0, 0.)]:
            specified_emergency.append({'date': day, 'interval': f'{clock(a)}—{clock(b)}' if b else '无', 'emergency_kwh': energy})
        specified_soc.append({'date': day, 'initial_soc_kwh': subset.iloc[0].start_soc_kwh,
                              'final_soc_kwh': subset.iloc[-1].end_soc_kwh})
    specified = {name: pd.DataFrame(value) for name, value in [
        ('specified_plan', specified_plan), ('specified_battery', specified_battery),
        ('specified_emergency', specified_emergency), ('specified_soc', specified_soc)]}
    for name, frame in specified.items():
        frame.to_csv(EVIDENCE / f'{name}.csv', index=False, encoding='utf-8-sig')
    daily['baseline_cost'] = baseline['fees'][:n].sum(axis=(1, 2))
    daily['difference_yuan'] = daily.total_cost - daily.baseline_cost
    daily['cumulative_difference'] = daily.difference_yuan.cumsum()
    daily['day_number'] = np.arange(1, n + 1)
    metrics = []
    for label, pc, ec, g, e in [
        ('exp004 无季节·原调度器', baseline['fees'][:n, :, 0].sum(), baseline['fees'][:n, :, 3].sum(), baseline['original'][:n].sum(), baseline['emergency'][:n].sum()),
        ('exp005 严格互斥·滚动规划', daily.planned_cost.sum(), daily.emergency_cost.sum(), totals[0], totals[1])]:
        metrics.append({'strategy': label, 'days': n, 'planned_cost': float(pc), 'adjustment_cost': 0.,
                        'emergency_cost': float(ec), 'total_cost': float(pc + ec), 'planned_kwh': float(g),
                        'emergency_kwh': float(e), 'total_wan': float((pc + ec) / 1e4)})
    checks = []
    names = {'balance': '实际供需平衡最大残差', 'soc': '实际SOC递推最大残差', 'overlap': '实际最大充放电重叠',
             'ramp': '实际最大相邻功率变化（含跨日）', 'bounds': '实际电量/SOC边界最大违反量',
             'ordinary_changed': '固定普通计划最大变更量', 'actual_changed': '实际供需与原始数据最大差',
             'boundary_soc': '跨日SOC衔接最大差', 'node_overlap': '两层全部规划节点最大重叠',
             'node_residual': '两层原约束最大残差', 'budget': '第二层超费用预算最大量',
             'binary_residual': '0/1变量最大整数残差', 'exclusive_residual': '互斥不等式最大残差',
             'mip_gap': '已完成求解最大相对gap'}
    for key, value in audit_max.items():
        if key == 'mip_gap':
            checks.append({'check': names[key], 'value': value, 'limit': None, 'passed': '原样披露'})
            continue
        limit = 1000. if key == 'ramp' else 1e-6
        checks.append({'check': names[key], 'value': value, 'limit': limit, 'passed': bool(value <= limit + (1e-6 if key == 'ramp' else 0))})
    checks.append({'check': '原实验源文件哈希变更数', 'value': 0, 'limit': 0, 'passed': True})
    same = dict(np.load(EVIDENCE / 'same_input_strict.npz'))
    same_meta = read(EVIDENCE / 'same_input_strict.json')
    old = dict(np.load(ROOT / 'data/results/exp005/beta0-boundary/2025-02-01/execution-081.npz'))
    old_meta = read(ROOT / 'data/results/exp005/beta0-boundary/2025-02-01/execution-081.json')
    compare = [{'model': label, 'max_overlap_kwh': meta['max_overlap_kwh'], 'tv_kw': meta['expected_tv_kw'],
                'emergency_cost': meta['second_cost'], 'budget': meta['budget']}
               for label, meta in [('原连续LP', old_meta), ('严格互斥MILP', same_meta)]]
    windows = pd.DataFrame([{'model': label, 'slot': t + 81, 'hour': (t + 81) / 6,
                            'charge_kwh': z['charge'][t], 'discharge_kwh': z['discharge'][t],
                            'power_kw': z['net_power_kw'][t], 'end_soc_kwh': z['end_soc'][t]}
                           for label, z in [('原连续LP', old), ('严格互斥MILP', same)] for t in range(63)])
    stage_frame = pd.DataFrame(stage_rows)
    stage_frame['absolute_objective_gap'] = np.maximum(stage_frame.objective - stage_frame.mip_dual_bound, 0)
    stage_frame['physical_absolute_gap'] = stage_frame.absolute_objective_gap * np.where(
        stage_frame.layer == 1, 1., 10000. * np.where(stage_frame.slot < 0, 144, 144 - stage_frame.slot))
    gap_rows = []
    for layer, label in [(1, '第一层费用 / 元'), (2, '第二层净功率总变差 / kW')]:
        group = stage_frame[stage_frame.layer == layer]
        gap_rows.append({'stage': label, 'max_absolute_gap': float(group.physical_absolute_gap.max()),
                         'max_reported_relative_gap': float(group.mip_gap.max()),
                         'unknown_gap_count': int(group.mip_gap.isna().sum()),
                         'numerical_retries': int(group.numerical_retry_count.sum())})
    gap_frame = pd.DataFrame(gap_rows)
    gap_frame.to_csv(EVIDENCE / 'optimality_gaps.csv', index=False, encoding='utf-8-sig')
    forecast_history = pd.read_csv(EVIDENCE / 'exp004_forecast_history.csv')
    forecast_history = forecast_history[forecast_history.population == 'all'].copy()
    selected = forecast_history[forecast_history.label == 'exp004 无季节'].copy()
    predictions = np.load(ROOT / 'data/results/exp004/predictions.npz')['no_season_seed_42']
    for target, pred, truth in [('load', predictions[:, :, 0], baseline['actual'][:, :, 0]),
                                ('pv', predictions[:, :, 1], baseline['actual'][:, :, 1]),
                                ('net_load', predictions[:, :, 0] - predictions[:, :, 1],
                                 baseline['actual'][:, :, 0] - baseline['actual'][:, :, 1])]:
        error = pred - truth
        old_score = selected[selected.target == target].iloc[0]
        np.testing.assert_allclose([np.abs(error).mean(), np.sqrt((error ** 2).mean()),
                                    100 * np.abs(error).sum() / np.abs(truth).sum()],
                                   [old_score.mae, old_score.rmse, old_score.wape_pct], rtol=1e-7, atol=1e-6)
    selected['label'] = 'exp005 沿用同一冻结预测'
    selected['experiment'] = 'exp005'
    forecast_history = pd.concat([forecast_history, selected], ignore_index=True)
    forecast_history['period_label'] = '2025-02-01—12-31，334日全部十分钟'
    forecast_history.to_csv(EVIDENCE / 'forecast_history.csv', index=False, encoding='utf-8-sig')
    timing_history = pd.read_csv(EVIDENCE / 'exp004_timings.csv')
    timing_history = pd.concat([timing_history, pd.DataFrame([
        {'experiment': 'exp005', 'label': 'exp005 严格互斥', 'stage': '新增训练', 'seconds': 0., 'groups': 0,
         'scope': '使用冻结预测，没有训练任务', 'source': 'protocol.json'},
        {'experiment': 'exp005', 'label': 'exp005 严格互斥', 'stage': '已完成求解器累计',
         'seconds': stage_frame.seconds.sum(), 'groups': n, 'scope': f'{n}日两层求解器计时，含数值复算；与历史组装执行计时范围不同',
         'source': 'solver_stages.csv'}])], ignore_index=True)
    timing_history.to_csv(EVIDENCE / 'timing_history.csv', index=False, encoding='utf-8-sig')
    environment = {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__,
                   'platform': platform.platform(), 'machine': platform.machine()}
    dump(EVIDENCE / 'environment.json', environment)
    history = pd.read_csv(EVIDENCE / 'exp004_cost_history.csv')
    history = history[['label', 'total_cost', 'planned_cost', 'emergency_cost', 'status']].copy()
    history['period'] = '2025-02-01—12-31（334天）'
    history.loc[len(history)] = ['exp005 严格互斥', daily.total_cost.sum(), daily.planned_cost.sum(), daily.emergency_cost.sum(),
                                 '新增硬爬坡/滚动控制；费用描述性并列', f'{daily.date.iloc[0]}—{daily.date.iloc[-1]}（{n}天）']
    for name, frame in [('daily', daily), ('actual_intervals', raw), ('solver_stages', stage_frame), ('tree_sizes', pd.DataFrame(tree_rows)),
                        ('same_input_comparison', pd.DataFrame(compare)), ('same_input_window', windows), ('audit', pd.DataFrame(checks)),
                        ('period_costs', pd.DataFrame(metrics)), ('history', history)]:
        frame.to_csv(EVIDENCE / f'{name}.csv', index=False, encoding='utf-8-sig')
    audit = {'checks': checks, 'completed_days': n, 'executed_intervals': n * 144,
             'solver_calls_audited': len(stage_rows), 'original_code_unchanged': True,
             'full_evaluation_complete': status['status'] == 'complete', 'run_status': status}
    dump(EVIDENCE / 'audit.json', audit)
    control_summary, control_daily, control_runs, control_audit = collect_controls(EVIDENCE)
    failure_text, stopped_prefix, stopped_stages = collect_failure(RUN, EVIDENCE)
    dump(OUT / 'strict-rerun-record.json', {
        'experiment_id': 'exp005', 'revision': 'strict-mutual-exclusion', 'status': status,
        'formal_334_day_result': status['status'] == 'complete', 'protocol': protocol,
        'completed_period': {'first_date': daily.date.iloc[0], 'last_date': daily.date.iloc[-1], 'days': n},
        'settlement': metrics, 'physical_audit': audit, 'environment': environment,
        'old_source_hashes': before,
        'new_source_hashes': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in sorted((OUT / 'strict_rerun').glob('*.py'))},
        'version_note': 'Old experimental source remains unmodified. New uncommitted adapters are identified by SHA-256; no claim that a Git commit includes them.'})
    dump(OUT / 'data_hashes.json', protocol['data_hashes'])
    for name, frame in [('control_summary', control_summary), ('control_daily', control_daily),
                        ('control_runs', control_runs), ('control_audit', control_audit)]:
        frame.to_csv(EVIDENCE / f'{name}.csv', index=False, encoding='utf-8-sig')
    tested_paths = [*(ROOT / path for path in before), ROOT / 'tests/test_q2_stochastic_lp.py',
                    OUT / 'strict_rerun/strict_backend.py', OUT / 'strict_rerun/test_strict_backend.py']
    test_hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in tested_paths}
    test_manifest = EVIDENCE / 'tested_source_hashes.json'
    if not test_manifest.exists() or read(test_manifest) != test_hashes:
        for name, cmd in [('original_tests', ['tests', 'test_q2_stochastic_lp.py']),
                          ('strict_tests', ['reports/experiments/exp005/strict_rerun', 'test_strict_backend.py'])]:
            test = subprocess.run([str(ROOT / '.venv/bin/python'), '-m', 'unittest', 'discover', '-s', cmd[0], '-p', cmd[1], '-v'],
                                  cwd=ROOT, capture_output=True, text=True, check=True)
            (EVIDENCE / f'{name}.txt').write_text(test.stdout + test.stderr)
        dump(test_manifest, test_hashes)
    figure_dir = OUT / 'strict_rerun/figures'
    figure_dir.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial Unicode MS', 'PingFang SC', 'DejaVu Sans'],
                         'axes.spines.top': False, 'axes.spines.right': False, 'axes.unicode_minus': False, 'font.size': 11, 'svg.fonttype': 'none'})
    def savefig(name):
        for suffix in ['png', 'svg']:
            plt.savefig(figure_dir / f'{name}.{suffix}', dpi=180, bbox_inches='tight')
        plt.close()
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True, sharey=True)
    for ax, label in zip(axes, ['原连续LP', '严格互斥MILP'], strict=True):
        sub = windows[(windows.model == label) & (windows.slot <= 99)]
        ax.plot(sub.hour, sub.charge_kwh, 'o-', markersize=3, color='#258a79', label='充电')
        ax.plot(sub.hour, sub.discharge_kwh, 'o-', markersize=3, color='#bf6548', label='放电')
        ax.axvspan(85 / 6, 86 / 6, color='#e8bd90', alpha=.2)
        ax.set_ylabel('kWh / 十分钟'); ax.set_title(label, loc='left'); ax.legend(frameon=False); ax.grid(axis='y', alpha=.2)
    axes[1].set_xlabel('2月1日时刻 / 小时；同一13:30状态与购电计划，仅新增互斥约束')
    fig.suptitle('原失败窗口：严格互斥消除重叠，总变差最优值保持一致', x=.1, ha='left')
    fig.tight_layout(); savefig('same-input')
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
    axes[0].plot(daily.day_number, daily.baseline_cost / 1e4, label='exp004 无季节·原调度器', color='#718095')
    axes[0].plot(daily.day_number, daily.total_cost / 1e4, label='exp005 严格互斥·滚动规划', color='#258a79')
    axes[0].set_ylabel('日购电费 / 万元'); axes[0].legend(frameon=False)
    axes[1].plot(daily.day_number, daily.cumulative_difference / 1e4, color='#bf6548')
    axes[1].axhline(0, color='#888', lw=1); axes[1].set_ylabel('累计费用差 / 万元'); axes[1].set_xlabel('从2月1日起的已完成日序号；正值表示本轮更贵')
    for ax in axes: ax.grid(axis='y', alpha=.2)
    fig.suptitle(f'同日期实际结算比较：{daily.date.iloc[0]}—{daily.date.iloc[-1]}，共{n}天', x=.1, ha='left')
    fig.tight_layout(); savefig('period-costs')
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.barh([v['strategy'] for v in metrics], [v['total_wan'] for v in metrics], color=['#718095', '#258a79'])
    for j, v in enumerate(metrics): ax.text(v['total_wan'], j, f" {v['total_wan']:.4f}", va='center')
    ax.set_xlim(0, max(v['total_wan'] for v in metrics) * 1.25); ax.set_xlabel('同一已完成评价期总购电费 / 万元')
    ax.set_title('结算日期一致；硬爬坡与控制器不同，仅作描述性比较', loc='left'); savefig('comparison')
    difference = float(daily.difference_yuan.sum())
    pct = difference / float(daily.baseline_cost.sum()) * 100
    finished = status['status'] == 'complete'
    scope_text = f'已完成{n}个完整日（{daily.date.iloc[0]}至{daily.date.iloc[-1]}），共{n * 144:,}个十分钟区间。'
    state_text = '334日回放完成，全部已完成求解通过互斥和物理核验。' if finished else '回放仍在继续，下述费用只覆盖已完成日期。'
    if status['status'] == 'solver_stopped':
        failure = status.get('failure', {})
        when = '午夜' if failure.get('slot') is None else clock(failure['slot'])
        reason = {0: '数值核验未通过', 1: '达到单层60秒求解时限', 2: '求解器判定不可行'}.get(failure.get('status'), '求解异常')
        state_text = f"回放在{failure.get('date', '')} {when}停止：{reason}。未执行未认证动作，未完成日不纳入下述完整日费用比较。完整原因和求解记录保存在严格重跑status.json中。"
    if status['status'] == 'paused_by_user':
        state_text = '按用户要求暂停计算，已完成日期及下一日初始状态已单独保存；0.1%和0.05%最优性差距的抽样计时结果见第八节。正式新实验尚未启动，等待用户决定。'
    control_text = ('对照沿用原实验的三个模式，并统一加上互斥：点计划＋点滚动、点计划＋场景滚动、场景计划＋场景滚动。'
                    '前两者比较日内控制的整体影响，后两者比较在同一场景控制方式下的午夜规划方案。'
                    '原点计划在午夜禁止预期紧急购电，场景计划允许按五倍价格计入预期补购，因此后两者差异还包含这一规划语义；不称作纯场景数量消融。'
                    '所有模式采用同一2月1日初始状态，之后各自连续传递真实SOC与末段功率。')
    if not control_summary.empty:
        common = int(control_summary.days.iloc[0])
        control_text += (f'三模式已共同完成{common}日（截至{control_summary.last_date.iloc[0]}），下表只累计这些相同日期，'
                         '各模式完整进度与停止状态另表列出。独立核验覆盖各模式全部已完成日。')
    else:
        control_text += '三模式尚未形成共同完整日，暂不计算消融差额。'
    for record in control_runs.to_dict('records'):
        if record['status'] == 'solver_stopped':
            position = '午夜' if pd.isna(record['failure_slot']) else clock(int(record['failure_slot']))
            stop_reason = {0: '数值核验未通过', 1: '单层60秒内未完成最优性认证', 2: '求解器判定不可行'}.get(record['failure_status'], '求解异常')
            gap = '未知' if pd.isna(record['failed_gap']) else f'{record["failed_gap"]:.6g}'
            control_text += (f'\n\n{record["strategy"]}在{record["failure_date"]} {position}第{int(record["failed_layer"])}层停止：'
                             f'{stop_reason}，停止时相对gap={gap}。候选未执行，未完成日期没有填入对照费用；'
                             '该对照未获得全年结果，不能据共同短期结果宣称全年场景收益。')
    sections = [
        ('结论与成绩', f'## 1. 结论与成绩\n\n本次按用户新决定严格禁止同时充放电。原实验源码未修改，通过独立求解适配器增加0/1互斥约束；报告仍归入exp005，原连续LP失败报告已归档。\n\n{scope_text}{state_text}\n\n已完成日期实际总购电费为{daily.total_cost.sum():,.4f}元，其中计划费{daily.planned_cost.sum():,.4f}元、紧急费{daily.emergency_cost.sum():,.4f}元、调整费0元。exp004无季节组同日期为{daily.baseline_cost.sum():,.4f}元，本轮差额{difference:+,.4f}元（{pct:+.2f}%）。该差额同时包含硬爬坡、场景与控制方式差异，不能归因为互斥本身。', ['period', 'daily', 'audit']),
        ('题目指标及信息边界', '## 2. 题目指标及信息边界\n\n负载与光伏功率以kW输入，乘1/6小时得到kWh。实际账单C=Σp(g+5e)：g为零点锁定的普通购电，e为实际紧急电量，计划全部计费。场景概率只用于规划期望，不进入真实账单；没有售电收入或调整费。\n\n单向效率√0.9；SOC为1200—10800 kWh，单向功率上限5000 kW，净功率相邻变化上限1000 kW。实际SOC和末段功率跨日连续。D5-A允许紧急电用于充电。当前十分钟供需仍采用稿中区间内计量反馈的平均近似，未来实际值不进入求解。\n\n重叠量=min(c,d)。数学互斥严格限制一者为零；浮点求解按原1e−6 kWh容差验收，微小数值残差照实保留，不净化数组。', ['audit']),
        ('数据与时间验证', f'## 3. 数据与时间验证\n\n沿用exp004/no_season_seed_42冻结预测，分支提交b42168d5271097762953f38d472d0ef5fe1a908d。预测档案SHA-256={protocol["archive_sha256"]}。未重训，附件3、4未进入本次模型。\n\n2月1日初始SOC为1421.7991105135516 kWh，上一段净充电功率172.76 kW，沿用共同一月预热。首日没有历史预测误差供体，退化为点路径；以后仅使用发布前已完整揭晓的日误差。每个完整日逐项核对实际供需、午夜计划和SOC衔接。\n\n原实验文件的运行前后哈希相同；适配脚本、数据和证据单独存放，原LP失败记录仍可复现。', ['tree', 'audit']),
        ('逐步技术讲解', '## 4. 逐步技术讲解\n\n计算顺序：冻结预测 → 已完成日联合误差 → 等权场景 → 已观测前缀信息树 → 午夜两层求解 → 固定普通计划逐段滚动求解 → 只执行当前动作 → 按实际量结算。\n\n原平衡式g+v+d+e=ℓ+c+w、SOC递推E后=E前+√0.9c−d/√0.9、净功率P=6(c−d)保持不变。每个节点新增z∈{0,1}，并加c≤(5000/6)z、d≤(5000/6)(1−z)。z=1允许充电，z=0允许放电，c=d=0允许待机。节点共享开关和动作，保留原非预见性结构。\n\n这使模型从连续LP变为MILP，但目标顺序不变：第一层最小预期费用；第二层在(1+0.001)C*+0.0001元预算内最小化预期净功率总变差。日内第一层只计算剩余紧急费用；β=0，没有第三层或终端奖励。每次都要求两层求解达到最优状态并检查约束。', ['same']),
        ('实验设置', '## 5. 实验设置\n\n最近最多56个完整历史日、最多16条路径、等权无放回抽样、种子42。06:00/12:00/18:00按可见前缀二分，每个子组至少2条路径；日内不更新权重。午夜与日内费用让步都为0.001，硬爬坡保持1000 kW。\n\n求解器SciPy/HiGHS MILP，每层60秒，mip_rel_gap请求值0；求解器按数值准则返回状态0，实际相对gap仍可能非零，报告按实测披露。先核验原始候选解，若物理残差超1e−6，则在该层剩余60秒预算内以整数可行性容差1e−10复算同一目标，绝不投影或净化。记录实际求解gap、时间、节点数和整数残差；超时或不可行时停止，不接受未认证解或自动放松约束。模型代码通过原模块调用，新增脚本只在求解器入口补充互斥矩阵。\n\n复用原11项测试，新增5项测试覆盖强制重叠的不可行反例、树节点互斥与状态继承、固定计划、D5-A紧急充电、未来实际扰动和补丁恢复。', ['tree', 'timing', 'audit']),
        ('结果及失败案例', f'## 6. 结果及失败案例\n\n对原2月1日13:30失败窗口保持相同状态、购电计划、预测与场景，仅加入互斥约束。最大重叠从566.5937811694077 kWh降为{same_meta["max_overlap_kwh"]:.8f} kWh；第二层总变差从{old_meta["expected_tv_kw"]:.8f}变为{same_meta["expected_tv_kw"]:.8f} kW，差异仅为浮点精度；剩余预期紧急费同为0.0001元。这个同输入对照表明原失败窗口存在同等目标值的互斥解。\n\n从2月1日重新回放时，每次求解都使用严格互斥，后续真实状态因而可以与原LP轨迹不同。首日完整日费23,907.3094元；不把同输入规划对照与重跑后的实际轨迹混为一谈。\n\n{scope_text}最贵已完成日为{daily.loc[daily.total_cost.idxmax(), "date"]}，费用{daily.total_cost.max():,.4f}元。已完成求解共{len(stage_rows):,}次，累计求解器计时{stage_frame.seconds.sum():,.2f}秒；原预测未重训，本轮新增训练时间为0。实际最大重叠{audit_max["overlap"]:.3e} kWh，全部两层节点最大重叠{audit_max["node_overlap"]:.3e} kWh。\n\n指定四日与正式334日购电Excel仅在相应日期完整回放和核验后输出；尚未完成的结果不填零。', ['same', 'window', 'actual', 'daily', 'timing', 'audit']),
        ('历次指标和技术路线对比', f'## 7. 历次指标和技术路线对比\n\nexp004只改预测，执行器仍为午夜确定性MILP加日内贪心；本轮固定同一预测，换成带硬爬坡的两层场景MILP和滚动执行。相同日期的实际费用可以描述性并列，不能把全部差额归因给充放电互斥，也不能宣称与旧协议严格同口径排名。\n\n截至{daily.date.iloc[-1]}的{n}个完整日，本轮相对exp004同日期费用差为{difference:+,.4f}元。原exp005连续LP只执行81段，没有完整日费，因此不绘制其费用改善率。历史exp001—exp004原登记或重算值照实保留，并明确各自评价期；未完成的本轮不与334日总额排序。\n\n当前预测与exp004无季节组完全相同，不声称预测精度提升。尚未完成点预测/场景及控制器的完整同协议消融，当前结果不能独立识别场景收益。', ['period', 'daily', 'history']),
        ('复现说明', '## 8. 复现说明\n\n严格重跑命令：`.venv/bin/python reports/experiments/exp005/strict_rerun/run.py --out .work/exp005-strict-reproduce`。输出目录须不存在；默认请求334日。原源码运行前后SHA-256保存在运行目录，新增求解适配器位于strict_rerun/strict_backend.py。\n\n新增测试：`.venv/bin/python -m unittest discover -s reports/experiments/exp005/strict_rerun -p test_strict_backend.py -v`。报告重建：`.venv/bin/python reports/experiments/exp005/strict_rerun/build_report.py --complete --build`，仅在回放完成或明确停止后使用complete。\n\n原LP报告归档于versions/lp-failure，原实验源码和失败输入保留。新证据包括每日午夜方案、逐段真实轨迹、两层求解日志、同输入互斥对照、独立物理/结算审计、代码哈希及测试日志。报告完成状态与策略是否完成334日评价分别标记。', ['audit'])]
    title, content, ids = sections[6]
    content = content.replace('尚未完成点预测/场景及控制器的完整同协议消融，当前结果不能独立识别场景收益。', '')
    sections[6] = (title, content + '\n\n' + control_text, ids + ['controls', 'control_runs', 'control_audit'])
    title, content, ids = sections[6]
    sections[6] = (title, content + '\n\n预测历史表单独采用全部334日冻结预测档案，覆盖负载、光伏和净负载。'
                   '本轮逐项重算并核对exp004无季节组MAE、RMSE与WAPE，数值相同。'
                   '预测档案完整不代表调度回放已完成。耗时另表保存历史阶段原值，未知项留空，设备和计时范围不同不计算速度提升率。',
                   ids + ['forecast_history', 'timing_history'])
    title, content, ids = sections[7]
    sections[7] = (title, content + '\n\n两项对照在上述命令中分别加入`--mode point`、`--mode point_scenario_control`，并使用独立的新输出目录。', ids)
    runtime_estimate = None
    timing_review = ROOT / 'data/results/exp005/gap-timing-review/estimate.json'
    if timing_review.exists():
        review = read(timing_review)
        runtime_estimate = pd.DataFrame([{'requested_gap': r['requested_gap'],
            'mature_extrapolation_hours': r['warmup_plus_mature_334_day_hours'],
            'planning_lower_hours': r['planning_range_hours'][0],
            'planning_upper_hours': r['planning_range_hours'][1],
            'estimated_speedup_fraction': r['estimated_speedup_fraction_vs_paired_zero_gap']}
            for r in review['rows']])
        runtime_estimate.to_csv(EVIDENCE / 'gap_runtime_estimate.csv', index=False)
        title, content, ids = sections[7]
        content += ('\n\n**暂停后的运行时间评估。** 原主方案55个完整日和点预测对照284个完整日已归档，'
                    '完整日状态可以复核；未完成日仅保留在暂停进程内存。正式近似重跑没有启动。'
                    '从已保存主方案选18个分层窗口，三档共54次两层求解全部通过原互斥和物理标准。'
                    '按本机单独运行、从头重算334日主方案估计：0.1%预留4—6小时，'
                    '0.05%预留4.5—6.5小时，原零差距约5—7小时；Excel和报告另预留15—30分钟。'
                    '这是有限样本的工程估计，不是保证完成的时限，也不包含另外对照的处理。'
                    '\n\n最优性差距仅改变每次求解的停止精度，二进制互斥、原物理容差和两层费用让步保持不变，'
                    '不是全年实际电费误差上限。原点计划＋场景执行对照的失败窗口在0.1%和0.05%下仍于60秒停止，'
                    '相对gap约85.126%，未执行失败候选。详细抽样、加权外推和压力检验见'
                    '`data/results/exp005/gap-timing-review/README.md`；保存进度见'
                    '`data/results/exp005/paused-before-gap-review/saved-progress.zip`。')
        sections[7] = (title, content, ids + ['gap_runtime_estimate'])
    title, content, ids = sections[4]
    gap_text = (f'已完成第一层的最大目标值与求解器下界之差为{gap_rows[0]["max_absolute_gap"]:.6g}元，'
                f'第二层换算为总变差后的最大差为{gap_rows[1]["max_absolute_gap"]:.6g} kW。'
                '当最优费用接近零时，相对gap可因极小分母显示为1，需结合绝对差阅读；不把未知差距写成零。'
                '0.1%仅是每次第二层相对于该次第一层费用解的局部预算，不构成全年实际账单的增幅保证。')
    sections[4] = (title, content + '\n\n' + gap_text, ids + ['gaps'])
    title, content, ids = sections[4]
    sections[4] = (title, content + f'\n\n本机环境：Python {environment["python"]}、NumPy {environment["numpy"]}、'
                   f'SciPy {environment["scipy"]}，{environment["platform"]}。预测种子42预先固定，未按本轮账单选种子或调参数。', ids)
    title, content, ids = sections[1]
    sections[1] = (title, content + '\n\n互斥只针对同一十分钟区间；同一个四小时汇总段内，可以先充电、后放电，'
                   '所以题目表2中两列同时为正不代表违反互斥。验收使用未汇总、未四舍五入的十分钟原始轨迹。', ids)
    title, content, ids = sections[2]
    sections[2] = (title, content + '\n\n此前已经查看过2025年全年实验结果，本轮应称为回顾性时序评价。'
                   '逐段信息边界仍按发布时刻执行，但不把这次设计后的重跑称为从未接触过的独立留出测试。', ids)
    title, content, ids = sections[3]
    sections[3] = (title, content + '\n\n严格互斥不保证两层问题只有一个最优轨迹。按用户要求保留β=0且不增加第三层，'
                   '因此保存本轮求解器返回的完整计划与实际执行数组；费用结论对应这条实际回放轨迹，'
                   '不能由两个规划目标值相同推断后续全年账单也相同。', ids)
    if failure_text:
        title, content, ids = sections[5]
        sections[5] = (title, content + '\n\n' + failure_text, ids + ['stopped_prefix', 'stopped_stages'])
    if len(specified_soc):
        title, content, ids = sections[5]
        content = content.replace('指定四日与正式334日购电Excel仅在相应日期完整回放和核验后输出；尚未完成的结果不填零。',
                                  f'题目指定四日中已有{len(specified_soc)}日完成回放，下表展示对应表1—表3及日初、日末SOC。正式Excel需334日全部完成并核验后生成。')
        sections[5] = (title, content, ids + list(specified))
    sources = {'period': (pd.DataFrame(metrics), 'period_costs.csv', '同一已完成日期的实际结算；硬爬坡与控制器不同。'),
               'daily': (daily, 'daily.csv', '完整日实际费用及exp004同日期对照，不含未完成日。'),
               'audit': (pd.DataFrame(checks), 'audit.csv', '完整日独立核对及全部已完成两层节点诊断。'),
               'tree': (pd.DataFrame(tree_rows), 'tree_sizes.csv', '每个午夜历史供体和信息树规模。'),
               'same': (pd.DataFrame(compare), 'same_input_comparison.csv', '原失败窗口，状态/计划/场景相同，仅新增互斥。'),
               'window': (windows, 'same_input_window.csv', '同输入第二层规划轨迹，非实际执行轨迹。'),
               'actual': (raw, 'actual_intervals.csv', '严格重跑逐段真实执行，末SOC与实际账单。'),
               'timing': (stage_frame.groupby('date').agg(solver_seconds=('seconds', 'sum'), calls=('seconds', 'size')).reset_index(), 'solver_stages.csv', '每完整日两层求解器时间总和；不含构树与文件写入。'),
               'history': (history, 'history.csv', '历史原值和评价期；协议差异限制排名。')}
    sources.update(controls=(control_summary, 'control_summary.csv', '三模式共同完整日期的实际结算；原点午夜方案禁止紧急购电。'),
                   control_runs=(control_runs, 'control_runs.csv', '各模式实际进度、停止位置、数值复算次数及已知gap。'),
                   control_audit=(control_audit, 'control_audit.csv', '各模式已完成日期逐区间独立核验与两层节点日志核验。'))
    sources['gaps'] = (gap_frame, 'optimality_gaps.csv', '目标值减求解器下界；第二层无量纲目标乘2×5000×该次剩余段数还原为kW。')
    sources['forecast_history'] = (forecast_history, 'forecast_history.csv', '全部334日午夜冻结预测重新评分；本轮与exp004无季节预测完全相同。')
    sources['timing_history'] = (timing_history, 'timing_history.csv', '历史计时原值与本轮已完成求解器累计；日期、设备、计时范围须一并阅读。')
    if runtime_estimate is not None:
        sources['gap_runtime_estimate'] = (runtime_estimate, 'gap_runtime_estimate.csv',
            '18个保存窗口的三档配对计时，加权至历史55日后按16路径成熟期外推；为工程估计，正式近似重跑未启动。')
    if failure_text:
        sources['stopped_prefix'] = (stopped_prefix, 'stopped_day_prefix.csv', '仅为停止当日此前已认证的执行段；失败候选未执行。')
        sources['stopped_stages'] = (stopped_stages, 'stopped_solver_stages.csv', '停止窗口的求解状态；超时不等同于数学不可行。')
    for name, frame in specified.items():
        sources[name] = (frame, f'{name}.csv', '题目指定日期的已完成主方案原始轨迹；普通计划、4小时充放电、SOC及连续紧急购电区间。')
    snapshot = read(OUT / 'app/src/data.json')
    delivery_status = ('complete' if finished else 'paused') if args.complete else 'updating'
    snapshot.update(title='exp005：严格互斥重跑', buildStatus=delivery_status,
                    generatedAt=datetime.now(UTC).isoformat(), report={'asOf': daily.date.iloc[-1]},
                    sections=[{'title': title, 'markdown': text, 'queryIds': ids} for title, text, ids in sections],
                    queries={key: {'rows': rows(frame), 'source': {'label': 'exp005严格重跑实测证据',
                              'files': ['strict_rerun/evidence/' + filename],
                              'metricDefinitions': [{'label': key, 'definition': definition}],
                              'evidenceFlow': [{'title': '核对原始数组', 'detail': '见strict_rerun/build_report.py的独立审计，未修改原实验源码。'}]}}
                             for key, (frame, filename, definition) in sources.items()})
    dump(OUT / 'app/src/data.json', snapshot)
    dump(OUT / 'reviewed.json', snapshot)
    text_blocks = ['# exp005：严格互斥重跑\n\n原实验源码保持不变，新增求解器入口互斥约束。']
    figures = {1: 'comparison', 6: 'same-input', 7: 'period-costs'}
    for i, (_, text, _) in enumerate(sections, 1):
        if i in figures: text += f'\n\n![{figures[i]}](strict_rerun/figures/{figures[i]}.png)'
        (OUT / f'section-{i}.md').write_text(text + '\n')
        text_blocks.append(text)
    (OUT / 'report.md').write_text('\n\n'.join(text_blocks) + '\n')
    (OUT / 'README.md').write_text(
        '# exp005：严格互斥重跑\n\n'
        '当前报告入口为report.html与report.md，仍按八节实验报告模板编写。'
        '原连续LP失败报告保存在versions/lp-failure；原实验源码未修改。\n\n'
        '新增实现位于strict_rerun：strict_backend.py在求解入口增加每节点0/1互斥，'
        'run.py调用原回放逻辑，build_report.py独立核验并重建报告。'
        '午夜和执行阶段仍只有费用、平稳性两个目标。物理残差超过1e-6的候选解不执行；'
        '同层可以在剩余60秒内收紧数值精度复算相同模型，不增加第三个目标。\n\n'
        f'{scope_text}{state_text}\n\n'
        '当前运行记录：data/results/exp005/strict-mutual-exclusion-certified。'
        'strict-mutual-exclusion为默认整数容差预检；strict-mutual-exclusion-tight为全局高精度预检。'
        '各次真实状态与费用独立保存，不混算。\n\n'
        '仅在完整334日通过核验后生成正式result2.xlsx。严格重跑、测试及报告重建命令见报告第八节。\n')
    if args.build:
        node = Path.home() / '.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node'
        plugin = Path.home() / '.codex/plugins/cache/openai-curated-remote/data-analytics/1.0.2/scripts/data-app.mjs'
        subprocess.run([str(node), str(plugin), 'build', '--project-dir', str(OUT / 'app'), '--separate-data'], check=True)
        offline = OUT / 'app/.data-app-offline/exports/report.html'
        subprocess.run([str(node), str(plugin), 'export-offline', '--project-dir', str(OUT / 'app'), '--output', str(offline)], check=True)
        shutil.copy2(offline, OUT / 'report.html')
        shutil.copytree(EVIDENCE, OUT / 'app/dist/strict_rerun/evidence', dirs_exist_ok=True)
    print(json.dumps({'days': n, 'run_status': status['status'], 'metrics': metrics, 'difference_yuan': difference,
                      'source_files_unchanged': len(before), 'audit_max': audit_max}, ensure_ascii=False))


if __name__ == '__main__':
    main()
