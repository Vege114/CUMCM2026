"""Describe a stopped strict run without treating its unfinished day as a full bill."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]


def collect(run, evidence):
    status_path = run / 'status.json'
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    if status.get('status') != 'solver_stopped':
        return '', pd.DataFrame(), pd.DataFrame()
    failure = status['failure']
    folder = run / failure['date']
    z = dict(np.load(folder / 'executed_prefix.npz'))
    n = len(z['charge'])
    protocol = json.loads((run / 'protocol.json').read_text())
    if status['completed_days']:
        daily = json.loads((run / 'daily.json').read_text())
        prior = dict(np.load(run / daily[-1]['date'] / 'actual.npz'))
        initial_soc, prior_power = prior['states'][-1], prior['net_power_kw'][-1]
    else:
        initial_soc, prior_power = protocol['initial_state']['soc'], protocol['initial_state']['previous_power_kw']
    assert abs(z['states'][0] - initial_soc) <= 1e-6
    if n:
        eta = np.sqrt(.9)
        residual = max(np.abs(np.diff(z['states']) - eta * z['charge'] + z['discharge'] / eta).max(),
                       np.abs(z['grid'] + z['actual'][:, 1] / 6 + z['discharge'] + z['emergency']
                              - z['actual'][:, 0] / 6 - z['charge'] - z['surplus']).max(),
                       np.maximum(1200 - z['states'], 0).max(), np.maximum(z['states'] - 10800, 0).max(),
                       np.maximum(z['charge'] - 5000 / 6, 0).max(), np.maximum(z['discharge'] - 5000 / 6, 0).max())
        residual = max(residual, *(np.maximum(-z[k], 0).max() for k in ['grid', 'charge', 'discharge', 'emergency', 'surplus']))
        baseline = dict(np.load(ROOT / 'reports/experiments/exp005/strict_rerun/evidence/exp004_dispatch_2.npz'))
        index = status['completed_days']
        np.testing.assert_allclose(z['actual'], baseline['actual'][index, :n], atol=0, rtol=0)
        plan = np.load(folder / 'midnight.npz')['grid']
        np.testing.assert_allclose(z['grid'], plan[:n], atol=0, rtol=0)
        price = baseline['price'][index, :n]
        np.testing.assert_allclose(z['fees'], np.column_stack([z['grid'] * price, 5 * z['emergency'] * price]), atol=1e-6, rtol=0)
        overlap = float(np.minimum(z['charge'], z['discharge']).max())
        ramp = float(np.abs(np.diff(np.r_[prior_power, z['net_power_kw']])).max())
        assert residual <= 1e-6 and overlap <= 1e-6 and ramp <= 1000 + 1e-6
    else:
        residual, overlap, ramp = 0., 0., 0.
    prefix = pd.DataFrame([{'date': failure['date'], 'slot': t, 'hour': t / 6,
                            'charge_kwh': z['charge'][t], 'discharge_kwh': z['discharge'][t],
                            'grid_kwh': z['grid'][t], 'emergency_kwh': z['emergency'][t],
                            'end_soc_kwh': z['states'][t + 1], 'power_kw': z['net_power_kw'][t]}
                           for t in range(n)])
    stages = pd.DataFrame([{'layer': i + 1, 'date': failure['date'], 'slot': failure.get('slot'),
                            'status': row['status'], 'seconds': row['seconds'],
                            'reported_relative_gap': row.get('mip_gap'),
                            'max_candidate_overlap_kwh': row.get('max_overlap_kwh'),
                            'candidate_executed': False}
                           for i, row in enumerate(failure.get('stages', []))])
    minute = failure.get('slot')
    when = '午夜计划' if minute is None else f'{minute // 6:02}:{minute % 6 * 10:02}'
    reason = {0: '数值验收未通过', 1: '达到60秒单层时限', 2: '求解器判定不可行'}.get(failure.get('status'), '求解异常')
    current_power = failure['state']['previous_power_kw']
    current_soc = failure['state']['soc']
    description = (f'本轮实际停止位置为{failure["date"]} {when}，原因是{reason}，'
                   f'失败窗口包含{failure["nodes"]}个节点、{failure["horizon"]}个剩余十分钟区间。'
                   f'进入该窗口的真实SOC={current_soc:.10f} kWh，前段净充电功率={current_power:.10f} kW。'
                   f'当日此前{n}段已执行且单独通过物理核验，最大重叠{overlap:.3e} kWh；失败候选未执行。'
                   '该未完成日不作为完整日费用参加比较，后续日期没有策略结果。')
    if failure.get('status') == 1:
        description += '超时不等于数学不可行，不能用超时解的费用代替可信第一层最优值。'
    evidence.mkdir(exist_ok=True)
    prefix.to_csv(evidence / 'stopped_day_prefix.csv', index=False, encoding='utf-8-sig')
    stages.to_csv(evidence / 'stopped_solver_stages.csv', index=False, encoding='utf-8-sig')
    (evidence / 'stopped_day_audit.json').write_text(json.dumps({
        'date': failure['date'], 'executed_intervals': n, 'max_residual': float(residual),
        'max_overlap_kwh': overlap, 'max_ramp_kw': ramp, 'passed': True,
        'scope': 'Only the accepted executed prefix; not a complete day or candidate certification'}, indent=2))
    return description, prefix, stages
