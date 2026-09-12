"""Independent accounting and overlap diagnostics for the approved LP runs."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
RESULTS = ROOT / 'data/results/exp005'
OUT = Path(__file__).parent / 'evidence'
BASE = ROOT / 'reports/experiments/exp005/strict_rerun/evidence'


def read(path):
    return json.loads(path.read_text())


def collect():
    OUT.mkdir(exist_ok=True)
    baseline = dict(np.load(BASE / 'exp004_dispatch_2.npz'))
    raw_price = pd.read_csv(ROOT / 'data/raw/附件1.csv')['电价'].to_numpy()
    np.testing.assert_allclose(baseline['price'], np.broadcast_to(raw_price, baseline['price'].shape), atol=0, rtol=0)
    all_daily, all_actual, all_stage_daily, all_overlap, all_audits, summaries = [], [], [], [], [], []
    run_records = []
    for beta in [.1, .01]:
        folder = RESULTS / f'soft-penalty-beta-{beta}'
        if not (folder / 'daily.json').exists():
            continue
        daily = pd.DataFrame(read(folder / 'daily.json'))
        protocol = read(folder / 'protocol.json')
        status = read(folder / 'status.json') if (folder / 'status.json').exists() else read(folder / 'progress.json')
        n = len(daily)
        before = read(folder / 'original_code_hashes_before.json')
        assert all(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == h for p, h in before.items())
        assert protocol['config']['beta'] == beta and protocol['model_class'] == 'LP'
        assert protocol['config']['delta'] == protocol['config']['delta_exec'] == .001
        assert protocol['config']['ramp_kw'] == 1000
        assert hashlib.sha256((RESULTS.parent / 'exp004/predictions.npz').read_bytes()).hexdigest() == protocol['archive_sha256']
        prev_soc, prev_power = (protocol['initial_state'][k] for k in ['soc', 'previous_power_kw'])
        maxima = {k: 0. for k in ['balance', 'soc', 'bounds', 'fixed_grid', 'actual_source', 'cross_day_soc',
                                  'ramp', 'node_constraints', 'cost_budget', 'fees', 'integer_variables']}
        scope_aggregates = {}
        for i, day in enumerate(daily.date):
            z = dict(np.load(folder / day / 'actual.npz'))
            plan = dict(np.load(folder / day / 'midnight.npz'))
            assert z['grid'].shape == (144,) and z['states'].shape == (145,)
            eta = np.sqrt(.9)
            fees = np.column_stack([z['grid'] * baseline['price'][i], 5 * z['emergency'] * baseline['price'][i]])
            values = {
                'balance': np.abs(z['grid'] + z['actual'][:, 1] / 6 + z['discharge'] + z['emergency']
                                  - z['actual'][:, 0] / 6 - z['charge'] - z['surplus']).max(),
                'soc': np.abs(np.diff(z['states']) - eta * z['charge'] + z['discharge'] / eta).max(),
                'bounds': max(np.maximum(1200 - z['states'], 0).max(), np.maximum(z['states'] - 10800, 0).max(),
                              np.maximum(z['charge'] - 5000 / 6, 0).max(), np.maximum(z['discharge'] - 5000 / 6, 0).max(),
                              *(np.maximum(-z[k], 0).max() for k in ['grid', 'charge', 'discharge', 'surplus', 'emergency'])),
                'fixed_grid': np.abs(z['grid'] - plan['grid']).max(),
                'actual_source': np.abs(z['actual'] - baseline['actual'][i]).max(),
                'cross_day_soc': abs(z['states'][0] - prev_soc),
                'ramp': np.abs(np.diff(np.r_[prev_power, z['net_power_kw']])).max(),
                'fees': max(np.abs(fees - z['fees']).max(), abs(fees.sum() - daily.iloc[i].total_cost))}
            for key, value in values.items():
                maxima[key] = max(maxima[key], float(value))
            np.testing.assert_allclose(z['net_power_kw'], 6 * (z['charge'] - z['discharge']), atol=1e-6, rtol=0)
            overlap = np.minimum(z['charge'], z['discharge'])
            assert int((overlap > 1e-6).sum()) == daily.iloc[i].overlap_intervals
            np.testing.assert_allclose(overlap.sum(), daily.iloc[i].overlap_kwh, atol=1e-6, rtol=0)
            for t in range(144):
                all_actual.append({'beta': beta, 'date': day, 'month': day[:7], 'slot': t, 'hour': t / 6,
                    'charge_kwh': float(z['charge'][t]), 'discharge_kwh': float(z['discharge'][t]),
                    'overlap_kwh': float(overlap[t]), 'overlap_flag': int(overlap[t] > 1e-6),
                    'significant_overlap_kwh': float(overlap[t]) if overlap[t] > 1e-6 else 0.,
                    'grid_kwh': float(z['grid'][t]), 'emergency_kwh': float(z['emergency'][t]),
                    'end_soc_kwh': float(z['states'][t + 1]), 'net_power_kw': float(z['net_power_kw'][t]),
                    'total_cost': float(fees[t].sum())})
            logs = [('midnight', -1, read(folder / day / 'midnight.json'))]
            logs += [('execution', t, m) for t, m in enumerate(read(folder / day / 'execution_logs.json'))]
            assert len(logs) == 145
            day_aggregates = {}
            solver_seconds = 0.
            for scope, slot, meta in logs:
                assert meta['beta'] == beta and len(meta['stages']) == 2
                maxima['cost_budget'] = max(maxima['cost_budget'], meta['second_cost'] - meta['budget'])
                for layer, stage in enumerate(meta['stages'], 1):
                    assert stage['status'] == 0 and stage['integer_variables'] == 0
                    maxima['node_constraints'] = max(maxima['node_constraints'], stage['max_constraint_residual'])
                    solver_seconds += stage['seconds']
                    for aggregation in [day_aggregates, scope_aggregates]:
                        key = (scope, layer)
                        row = aggregation.setdefault(key, {'scope': scope, 'layer': layer, 'calls': 0,
                            'nodes': 0, 'overlap_nodes': 0, 'windows_with_overlap': 0,
                            'max_overlap_kwh': 0., 'expected_overlap_sum_kwh': 0., 'solver_seconds': 0.})
                        row['calls'] += 1
                        row['nodes'] += stage['nodes']
                        row['overlap_nodes'] += stage['overlap_nodes']
                        row['windows_with_overlap'] += int(stage['overlap_nodes'] > 0)
                        row['max_overlap_kwh'] = max(row['max_overlap_kwh'], stage['max_overlap_kwh'])
                        row['expected_overlap_sum_kwh'] += stage['expected_overlap_kwh']
                        row['solver_seconds'] += stage['seconds']
            np.testing.assert_allclose(solver_seconds, daily.iloc[i].solver_seconds, atol=1e-6, rtol=0)
            for row in day_aggregates.values():
                all_stage_daily.append(dict(beta=beta, date=day, **row))
            node_rows = read(folder / day / 'overlap_nodes.json')
            assert len(node_rows) == sum(r['overlap_nodes'] for r in day_aggregates.values())
            for row in node_rows:
                assert row['overlap_kwh'] > 1e-6
                np.testing.assert_allclose(row['overlap_kwh'], min(row['charge_kwh'], row['discharge_kwh']), atol=1e-6, rtol=0)
                all_overlap.append(dict(beta=beta, **row))
            prev_soc, prev_power = z['states'][-1], z['net_power_kw'][-1]
        for key, value in maxima.items():
            limit = 1000 + 1e-6 if key == 'ramp' else 0 if key == 'integer_variables' else 1e-6
            assert value <= limit, (beta, key, value)
            all_audits.append({'beta': beta, 'check': key, 'value': value, 'limit': limit, 'passed': True})
        daily['baseline_cost'] = baseline['fees'][:n].sum(axis=(1, 2))
        daily['day_number'] = np.arange(n) + 1
        daily['difference_yuan'] = daily.total_cost - daily.baseline_cost
        daily['cumulative_difference'] = daily.difference_yuan.cumsum()
        all_daily.append(daily)
        for row in scope_aggregates.values():
            summaries.append(dict(beta=beta, days=n, **row,
                                 overlap_node_rate_pct=100 * row['overlap_nodes'] / row['nodes']))
        run_records.append({'beta': beta, 'completed_days': n, 'last_date': daily.date.iloc[-1],
                            'status': status['status'], 'wall_seconds': status['seconds'],
                            'solver_seconds': float(daily.solver_seconds.sum()),
                            'original_source_files_unchanged': len(before)})
    frames = {'daily': pd.concat(all_daily, ignore_index=True), 'actual': pd.DataFrame(all_actual),
              'stage_daily': pd.DataFrame(all_stage_daily), 'overlap_nodes': pd.DataFrame(all_overlap),
              'audit': pd.DataFrame(all_audits), 'overlap_summary': pd.DataFrame(summaries),
              'runs': pd.DataFrame(run_records)}
    common = min(r['completed_days'] for r in run_records)
    for day in all_daily[0].date.iloc[:common]:
        with np.load(RESULTS / f'soft-penalty-beta-0.1/{day}/scenarios.npz') as main_s, np.load(RESULTS / f'soft-penalty-beta-0.01/{day}/scenarios.npz') as control_s:
            assert main_s.files == control_s.files
            for key in main_s.files:
                np.testing.assert_array_equal(main_s[key], control_s[key])
    frames['input_pairing'] = pd.DataFrame([{'check': '附件1电价逐段一致', 'days': 334, 'passed': True},
        {'check': '两β路径、供体、概率及场景诊断完全相同', 'days': common, 'passed': True}])
    for key, frame in frames.items():
        frame.to_csv(OUT / f'{key}.csv', index=False)
    return frames
