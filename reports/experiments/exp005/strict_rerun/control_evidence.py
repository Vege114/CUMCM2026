"""Independent audits and matched-date comparisons of existing replay modes."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from failure_evidence import collect as collect_failure

ROOT = Path(__file__).resolve().parents[4]
RESULTS = ROOT / 'data/results/exp005'
MODES = {
    'strict-mutual-exclusion-certified': ('场景计划＋场景滚动', 'scenario'),
    'strict-point-control': ('点计划＋点滚动', 'point'),
    'strict-point-scenario-control': ('点计划＋场景滚动', 'point_scenario_control'),
}


def read(path):
    return json.loads(path.read_text())


def collect(evidence):
    frames, run_rows, audits = {}, [], []
    baseline = dict(np.load(evidence / 'exp004_dispatch_2.npz'))
    for name, (label, mode) in MODES.items():
        folder = RESULTS / name
        if not (folder / 'daily.json').exists():
            continue
        daily = pd.DataFrame(read(folder / 'daily.json'))
        protocol = read(folder / 'protocol.json')
        assert protocol['mode'] == mode and protocol['strict_mutual_exclusion']
        status = (read(folder / 'status.json') if (folder / 'status.json').exists()
                  else read(folder / 'pause_status.json') if (folder / 'pause_status.json').exists() else {'status': 'running'})
        prev_soc = protocol['initial_state']['soc']
        prev_power = protocol['initial_state']['previous_power_kw']
        residual, overlap, node_overlap, ramp, retry_count, calls, gap = 0., 0., 0., 0., 0, 0, 0.
        for i, day in enumerate(daily.date):
            z = dict(np.load(folder / day / 'actual.npz'))
            plan = dict(np.load(folder / day / 'midnight.npz'))
            assert len(z['grid']) == 144 and len(z['states']) == 145
            eta = np.sqrt(.9)
            values = [np.abs(z['actual'] - baseline['actual'][i]).max(),
                      np.abs(z['grid'] - plan['grid']).max(), abs(z['states'][0] - prev_soc),
                      np.abs(np.diff(z['states']) - eta * z['charge'] + z['discharge'] / eta).max(),
                      np.abs(z['grid'] + z['actual'][:, 1] / 6 + z['discharge'] + z['emergency']
                             - z['actual'][:, 0] / 6 - z['charge'] - z['surplus']).max(),
                      np.maximum(1200 - z['states'], 0).max(), np.maximum(z['states'] - 10800, 0).max(),
                      np.maximum(z['charge'] - 5000 / 6, 0).max(), np.maximum(z['discharge'] - 5000 / 6, 0).max()]
            values += [np.maximum(-z[k], 0).max() for k in ['grid', 'charge', 'discharge', 'emergency', 'surplus']]
            residual = max(residual, *values)
            overlap = max(overlap, np.minimum(z['charge'], z['discharge']).max())
            ramp = max(ramp, np.abs(np.diff(np.r_[prev_power, z['net_power_kw']])).max())
            fees = np.column_stack([z['grid'] * baseline['price'][i], 5 * z['emergency'] * baseline['price'][i]])
            np.testing.assert_allclose(z['fees'], fees, atol=1e-6, rtol=0)
            np.testing.assert_allclose(daily.iloc[i].total_cost, fees.sum(), atol=1e-6, rtol=0)
            logs = [read(folder / day / 'midnight.json'), *read(folder / day / 'execution_logs.json')]
            assert len(logs) == 145
            for log in logs:
                assert log['second_cost'] <= log['budget'] + 1e-6
                assert len(log['stages']) == 2 and log['strict_mutual_exclusion']
                for stage in log['stages']:
                    assert stage['status'] == 0
                    assert stage['max_constraint_residual'] <= 1e-6
                    assert stage['max_exclusivity_residual'] <= 1e-6
                    assert stage['max_binary_residual'] <= 1e-6
                    node_overlap = max(node_overlap, stage['max_overlap_kwh'])
                    gap = max(gap, stage['mip_gap'] or 0.)
                    retry_count += stage['numerical_retry_count']
                    calls += 1
            prev_soc, prev_power = z['states'][-1], z['net_power_kw'][-1]
        assert residual <= 1e-6 and max(overlap, node_overlap) <= 1e-6 and ramp <= 1000 + 1e-6
        before = read(folder / 'original_code_hashes_before.json')
        assert all(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == value for p, value in before.items())
        failure = status.get('failure') or {}
        failed_stages = failure.get('stages') or []
        if failure:
            collect_failure(folder, evidence / f'control-failure-{mode}')
        run_rows.append({'strategy': label, 'mode': mode, 'completed_days': len(daily),
                         'last_date': daily.date.iloc[-1], 'status': status['status'],
                         'failure_date': failure.get('date'), 'failure_slot': failure.get('slot'),
                         'failure_status': failure.get('status'),
                         'failed_layer': len(failed_stages) if failed_stages else None,
                         'failed_gap': failed_stages[-1].get('mip_gap') if failed_stages else None,
                         'failed_candidate_executed': False,
                         'numerical_retries': retry_count, 'solver_calls': calls, 'max_reported_gap': gap})
        audits.append({'strategy': label, 'completed_days': len(daily), 'max_actual_residual': float(residual),
                       'max_actual_overlap_kwh': float(overlap), 'max_node_overlap_kwh': float(node_overlap),
                       'max_ramp_kw': float(ramp), 'passed': True})
        frames[label] = daily
    if len(frames) < 3:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(run_rows), pd.DataFrame(audits)
    common = min(len(df) for df in frames.values())
    reference = frames['点计划＋场景滚动'].iloc[:common]
    paired, summary = [], []
    for label, frame in frames.items():
        sub = frame.iloc[:common]
        assert sub.date.tolist() == reference.date.tolist()
        cost = float(sub.total_cost.sum())
        summary.append({'strategy': label, 'days': common, 'first_date': sub.date.iloc[0], 'last_date': sub.date.iloc[-1],
                        'planned_cost': float(sub.planned_cost.sum()), 'emergency_cost': float(sub.emergency_cost.sum()),
                        'total_cost': cost, 'emergency_kwh': float(sub.emergency_kwh.sum()),
                        'difference_vs_point_scenario': cost - float(reference.total_cost.sum())})
        for day, c in zip(sub.date, sub.total_cost, strict=True):
            paired.append({'date': day, 'strategy': label, 'total_cost': c})
    return pd.DataFrame(summary), pd.DataFrame(paired), pd.DataFrame(run_rows), pd.DataFrame(audits)
