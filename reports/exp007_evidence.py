"""Freeze and audit exp007 report evidence without training or policy execution."""

import argparse
import hashlib
import json
import shutil
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MAIN_ROOT = Path('/Users/vegetarianwolf/Projects/CUMCM2026')
OUT = ROOT / 'data/results/exp007'
REPORT = ROOT / 'reports/experiments/exp007'
EVIDENCE = REPORT / 'evidence'
OLD = ROOT / 'reports/experiments/exp006/evidence'
ETA = float(np.sqrt(.9))
INITIAL_SOC = 1421.7991105135516
INITIAL_POWER = 172.75999999999976
DATES = pd.date_range('2025-02-01', '2025-12-31').strftime('%Y-%m-%d').tolist()
SPECIFIED = ['2025-03-20', '2025-06-21', '2025-09-23', '2025-12-21']
RANDOM_SEED = 20260912
RANDOM_DAYS = sorted(np.random.default_rng(RANDOM_SEED).choice(334, 4, replace=False).tolist())
RUNS = [f'{family}_{seed}' for family in ('regularized', 'cost_only') for seed in (42, 2026, 3407)]
LABELS = {run: f'exp007 {"正式·轻惩罚" if run == "regularized_42" else "轻惩罚" if run.startswith("regularized") else "仅费用"}·{run.rsplit("_", 1)[1]}' for run in RUNS}
CORE = ['exp004/no_season', 'exp006/primary', 'exp006/greedy_execution',
        'exp007/regularized_42', 'exp007/cost_only_42']
POWER_POLICIES = ['exp004/no_season', 'exp006/primary', 'exp006/greedy_execution'] + ['exp007/'+r for r in RUNS]


def read(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def records(frame):
    return json.loads(frame.to_json(orient='records', double_precision=15))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def owned_manifest_paths(table_names, source_records):
    """Only bind this generator's outputs and explicitly frozen input copies.

    Report, inline, figure, battery-export, browser, workbook, registration,
    test/lint and final QA are downstream products with separate manifests.
    Never discover ownership by recursively scanning the shared evidence folder.
    """
    candidates = {EVIDENCE/f'{name}.csv' for name in table_names}
    candidates.update(EVIDENCE/name for name in (
        'exp005_compatibility_audit.json', 'exp005_costs.csv',
        'historical_source_manifest.json', 'registry_coverage.json',
        'protocol.json', 'raw_run_manifest.json', 'raw_summary.csv',
        'raw_timings.csv', 'report_data.json',
    ))
    candidates.add(EVIDENCE/'battery/full_curves.npz')
    for policy in POWER_POLICIES:
        basename = policy.replace('/', '__')
        for suffix in ('full.csv', 'random_days.csv'):
            candidates.add(EVIDENCE/f'battery/{basename}_{suffix}')
    for source in source_records:
        if source.get('snapshot'):
            path = (REPORT/source['snapshot']).resolve()
            if not path.is_relative_to((EVIDENCE/'history').resolve()):
                raise ValueError(f'Unexpected frozen-source snapshot location: {path}')
            candidates.add(path)
    return sorted(path for path in candidates if path.is_file())


class Sources:
    def __init__(self):
        self.rows = {}

    def add(self, path, snapshot=None, purpose='source evidence'):
        path = Path(path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        row = {'source': str(path), 'sha256': sha256(path), 'bytes': path.stat().st_size,
               'purpose': purpose, 'snapshot': None}
        if snapshot:
            destination = EVIDENCE / snapshot
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() and sha256(destination) != row['sha256']:
                raise RuntimeError(f'Refusing to replace a changed frozen historical source: {destination}')
            if not destination.exists():
                shutil.copy2(path, destination)
            row['snapshot'] = str(destination.relative_to(REPORT))
        self.rows[str(path)] = row
        return row


def load_archive(path, price=None):
    with np.load(path) as archive:
        z = {key: archive[key].copy() for key in archive.files}
    if 'grid' in z:
        # Explicit exp005 adapter. Its fee archive has plan/emergency columns.
        z['original'] = z['grid'].copy()
        z['final'] = z['grid'].copy()
        z['price'] = np.broadcast_to(price, z['grid'].shape).copy()
        oldfees = z['fees']
        z['fees'] = np.stack((oldfees[:, :, 0], np.zeros_like(z['grid']),
                             np.zeros_like(z['grid']), oldfees[:, :, 1]), axis=-1)
    return z


def audit_dispatch(z, actual, price, allow_emergency_charge=False, check_initial=True):
    """Reconstruct physics and cash costs without calling any dispatch code."""
    expected = (334, 144)
    for key in ('original', 'final', 'charge', 'discharge', 'emergency', 'surplus', 'price'):
        if z[key].shape != expected:
            raise AssertionError(f'{key} has shape {z[key].shape}, expected {expected}')
    if z['states'].shape != (334, 145) or z['actual'].shape != (334, 144, 2):
        raise AssertionError('State or actual shape mismatch')
    for key in ('original', 'final', 'charge', 'discharge', 'emergency', 'surplus',
                'price', 'states', 'actual', 'fees'):
        if not np.isfinite(z[key]).all() or np.min(z[key]) < -1e-6:
            raise AssertionError(f'Nonfinite/negative {key}')
    q, c, d, e, w, s, p = (z[k] for k in ('final', 'charge', 'discharge', 'emergency',
                                          'surplus', 'states', 'price'))
    balance = q + (z['actual'][:, :, 1]-z['actual'][:, :, 0])/6+d+e-c-w
    fees = np.stack((p*q, np.zeros_like(q), np.zeros_like(q), 5*p*e), axis=-1)
    result = {
        'days': 334, 'intervals': 48096, 'source_actual_equal': bool(np.array_equal(z['actual'], actual)),
        'source_tariff_equal': bool(np.array_equal(p, np.broadcast_to(price, expected))),
        'midnight_purchase_unchanged': bool(np.array_equal(q, z['original'])),
        'max_balance_error_kwh': float(np.abs(balance).max()),
        'max_soc_error_kwh': float(np.abs(np.diff(s, axis=1)-ETA*c+d/ETA).max()),
        'max_cross_day_soc_error_kwh': float(np.abs(s[1:, 0]-s[:-1, -1]).max()),
        'initial_soc_error_kwh': float(abs(s[0, 0]-INITIAL_SOC)),
        'max_slot_fee_error_yuan': float(np.abs(fees-z['fees']).max()),
        'min_soc_kwh': float(s.min()), 'max_soc_kwh': float(s.max()),
        'max_charge_kw': float(6*c.max()), 'max_discharge_kw': float(6*d.max()),
        'simultaneous_slots': int(np.sum((c > 1e-6) & (d > 1e-6))),
        'max_overlap_kwh': float(np.minimum(c, d).max()),
        'emergency_charging_slots': int(np.sum((c > 1e-6) & (e > 1e-6))),
        'emergency_charging_allowed_in_source': allow_emergency_charge,
        'planned_cost': float(fees[:, :, 0].sum()), 'emergency_cost': float(fees[:, :, 3].sum()),
        'total_cost': float(fees.sum()), 'reported_cost_excludes_penalties_and_terminal_value': True,
    }
    tests = [result[k] for k in ('source_actual_equal', 'source_tariff_equal', 'midnight_purchase_unchanged')]
    tests += [result[k] <= 1e-6 for k in ('max_balance_error_kwh', 'max_soc_error_kwh',
                                         'max_cross_day_soc_error_kwh', 'max_slot_fee_error_yuan')]
    tests += [s.min() >= 1200-1e-6, s.max() <= 10800+1e-6,
              max(c.max(), d.max()) <= 5000/6+1e-6, result['simultaneous_slots'] == 0]
    if check_initial:
        tests.append(result['initial_soc_error_kwh'] <= 1e-6)
    if not allow_emergency_charge:
        tests.append(result['emergency_charging_slots'] == 0)
    result['passed'] = bool(all(tests))
    if not result['passed']:
        raise AssertionError(json.dumps(result, ensure_ascii=False))
    return result


def battery_metrics(z):
    c, d = np.maximum(z['charge'].ravel(), 0), np.maximum(z['discharge'].ravel(), 0)
    power = 6*(c-d)
    delta = np.diff(power)  # No artificial first zero; includes all 333 midnight boundaries.
    absdelta = np.abs(delta)
    tol = 6e-6  # Preserves the protocol's 1e-6 kWh zero tolerance.
    active = np.where(power > tol, 1, np.where(power < -tol, -1, 0))
    nonidle = active[active != 0]
    return {
        'charge_kwh': float(c.sum()), 'discharge_kwh': float(d.sum()),
        'throughput_kwh': float((c+d).sum()),
        'equivalent_full_cycles': float((ETA*c.sum()+d.sum()/ETA)/24000),
        'initial_soc': float(z['states'][0, 0]), 'final_soc': float(z['states'][-1, -1]),
        'simultaneous_slots': int(np.sum((c > 1e-6) & (d > 1e-6))),
        'direction_reversals': int(np.sum(nonidle[1:]*nonidle[:-1] == -1)),
        'nonidle_direction_reversals': int(np.sum(nonidle[1:]*nonidle[:-1] == -1)),
        'direct_adjacent_reversals': int(np.sum(active[1:]*active[:-1] == -1)),
        'mean_absolute_change_kw': float(absdelta.mean()),
        'rms_change_kw': float(np.sqrt(np.mean(delta**2))),
        'p95_absolute_change_kw': float(np.quantile(absdelta, .95)),
        'max_absolute_change_kw': float(absdelta.max()),
        'total_variation_kw': float(absdelta.sum()), 'power_variation_kw': float(absdelta.sum()),
        'large_jump_threshold_kw': 1000.,
        'large_jump_count': int(np.sum(absdelta > 1000)),
        'large_jump_pct': float(100*np.mean(absdelta > 1000)),
        'power_limit_hit_pct': float(100*np.mean(np.abs(power) >= 5000-tol)),
        'warmup_boundary_jump_kw': float(power[0]-INITIAL_POWER),
        'warmup_boundary_absolute_jump_kw': float(abs(power[0]-INITIAL_POWER)),
        'total_variation_including_warmup_kw': float(absdelta.sum()+abs(power[0]-INITIAL_POWER)),
        'active_slot_pct': float(100*np.mean(active != 0)),
        'first_power_kw': float(power[0]), 'last_power_kw': float(power[-1]),
        'zero_tolerance_kw': tol, 'power_limit_kw': 5000., 'points': len(power),
        'differences': len(delta), 'cross_midnight_differences': 333,
    }


def freeze_history(sources):
    """Freeze all six registrations and audit the two completed exp005 arms."""
    from experiments.problem2.exp003.data import Data

    data, original = Data(), Data(MAIN_ROOT)
    actual = data.actual[31*144:].reshape(334, 144, 2)
    coverage = []
    for exp in [f'exp{i:03d}' for i in range(1, 7)]:
        base = MAIN_ROOT if exp == 'exp005' else ROOT
        entry = sources.add(base/f'reports/registry/{exp}.json', f'history/registries/{exp}.json', 'original experiment role and protocol')
        coverage.append({'experiment': exp, **entry, 'included': True,
                         'restriction': 'different hard ramp / relaxed controller / emergency charging; descriptive only' if exp == 'exp005' else 'retain original/exploratory exclusions and original roles'})
    for name in ('cost_history', 'forecast_history', 'battery_history', 'timings'):
        sources.add(OLD/f'{name}.csv', f'history/exp006/{name}.csv', 'frozen historical comparison interface')
    raw_checks = []
    for name in data.hashes:
        a, b = ROOT/'data/raw'/name, MAIN_ROOT/'data/raw'/name
        sources.add(a, purpose='current official input'); sources.add(b, purpose='exp005 raw input')
        pa, pb = pd.read_csv(a), pd.read_csv(b)
        pd.testing.assert_frame_equal(pa, pb, check_exact=True)
        raw_checks.append({'file': name, 'current_sha256': sha256(a), 'exp005_sha256': sha256(b),
                           'same_bytes': a.read_bytes() == b.read_bytes(), 'same_table_values_and_time_labels': True,
                           'same_after_crlf_normalization': a.read_bytes().replace(b'\r\n', b'\n') == b.read_bytes().replace(b'\r\n', b'\n')})
    if not np.array_equal(data.actual, original.actual) or not np.array_equal(data.fixed_price, original.fixed_price):
        raise AssertionError('exp005 numerical input difference')
    pred = ROOT/'data/results/exp004/predictions.npz'
    pred5 = MAIN_ROOT/'data/results/exp004/predictions.npz'
    sources.add(pred, purpose='frozen predictor'); sources.add(pred5, purpose='exp005 frozen predictor')
    if sha256(pred) != sha256(pred5):
        raise AssertionError('exp005 frozen prediction bytes differ')
    exp005, audits, arrays = [], [], {}
    registry = read(MAIN_ROOT/'reports/registry/exp005.json')
    for beta in (.1, .01):
        tag = f'beta_{beta:g}'
        folder = MAIN_ROOT/f'data/results/exp005/soft-penalty-beta-{beta:g}'
        for file in ('dispatch.npz', 'daily.json', 'protocol.json', 'status.json'):
            sources.add(folder/file, f'history/exp005/{tag}/{file}', 'frozen exp005 completed arm')
        z = load_archive(EVIDENCE/f'history/exp005/{tag}/dispatch.npz', data.fixed_price)
        audit = audit_dispatch(z, actual, data.fixed_price, allow_emergency_charge=True)
        daily = pd.DataFrame(read(folder/'daily.json'))
        if daily.date.tolist() != DATES:
            raise AssertionError('exp005 date order differs')
        np.testing.assert_allclose(daily.total_cost, z['fees'].sum(axis=(1, 2)), atol=1e-6, rtol=0)
        registered = next(row for row in registry['metrics'] if row['beta'] == beta)
        if abs(audit['total_cost']-registered['total_cost']) > 1e-6:
            raise AssertionError('exp005 registered total differs from archive')
        np.testing.assert_allclose(z['net_power_kw'], 6*(z['charge']-z['discharge']), atol=1e-6, rtol=0)
        ramp = float(np.abs(np.diff(np.r_[INITIAL_POWER, z['net_power_kw'].ravel()])).max())
        if ramp > 1000+1e-6:
            raise AssertionError('exp005 registered hard ramp not satisfied')
        audit.update(beta=beta, date_order_equal=True, raw_input_checks=raw_checks,
                     forecast_sha256=sha256(pred), hard_ramp_kw=1000., max_ramp_kw=ramp,
                     ranking_allowed=False, source=str(folder/'dispatch.npz'),
                     interpretation='numerically same data and settlement; extra hard ramp, relaxed optimization and emergency charging change feasible controls')
        audits.append(audit)
        policy = f'exp005/{tag}'
        arrays[policy] = z
        exp005.append({'experiment': 'exp005', 'policy_id': policy, 'name': tag, 'beta': beta,
                       'label': f'exp005 LP·β={beta:g}', 'role': '当轮正式策略' if beta == .1 else '历史权重对照',
                       'seed': 42, 'forecast_seed': 42, 'days': 334, 'start_date': DATES[0], 'end_date': DATES[-1],
                       'total_cost': audit['total_cost'], 'planned_cost': audit['planned_cost'],
                       'emergency_cost': audit['emergency_cost'], 'planned_kwh': float(z['original'].sum()),
                       'emergency_kwh': float(z['emergency'].sum()), 'up_cost': 0., 'down_cost': 0.,
                       **battery_metrics(z), 'physically_comparable': False, 'ranking_allowed': False,
                       'comparable': False, 'exploratory': False,
                       'status': '仅描述：1000 kW硬爬坡、LP松弛及允许紧急充电；不排名',
                       'source': str((EVIDENCE/f'history/exp005/{tag}/dispatch.npz').relative_to(ROOT)),
                       'original_source': str(folder/'dispatch.npz')})
    for file in ('period.csv', 'runs.csv', 'audit.csv', 'timing_history.csv', 'forecast_history.csv'):
        path = MAIN_ROOT/'reports/experiments/exp005/soft_penalty/evidence'/file
        if path.is_file():
            sources.add(path, f'history/exp005/{file}', 'exp005 published full-period evidence')
    write_json(EVIDENCE/'exp005_compatibility_audit.json', {'passed': True, 'same_numeric_inputs': True,
               'same_time_labels': True, 'same_forecast_bytes': True, 'raw_files': raw_checks,
               'arms': audits, 'ranking_allowed': False})
    write_json(EVIDENCE/'registry_coverage.json', {'experiments': coverage,
               'note': 'All exp001–006 read; exp005 is completed but protocol-incompatible. No historical files rewritten.'})
    return pd.DataFrame(exp005), arrays, data


def relative_rows(costs, forecast, batteries):
    rows = []
    primary = costs[costs.policy_id == 'exp007/regularized_42'].iloc[0]
    def add(old, current, metric, unit, allowed, population=None, reason=None):
        previous, now = old.get(metric), current.get(metric)
        if pd.isna(previous) or pd.isna(now):
            return
        rows.append({'previous_policy_id': old.get('policy_id'), 'previous_experiment': old['experiment'],
                     'previous_label': old['label'], 'current_policy_id': current.get('policy_id'),
                     'current_experiment': 'exp007', 'current_label': current['label'],
                     'metric': metric, 'population': population, 'unit': unit,
                     'absolute_change_unit': '百分点' if metric == 'wape_pct' else unit,
                     'previous': previous, 'current': now,
                     'absolute_change': now-previous if allowed else None,
                     'relative_change_pct': 100*(now-previous)/abs(previous) if allowed and previous else None,
                     'comparable': bool(allowed), 'comparability_reason': reason or old.get('status'),
                     'previous_source': old['source'], 'current_source': current['source'],
                     'forecast_seed': 42, 'rl_seed': 42, 'period': f'{DATES[0]}/{DATES[-1]}'})
    for _, old in costs[costs.policy_id != primary.policy_id].iterrows():
        for metric, unit in [('total_cost', '元'), ('planned_cost', '元'), ('emergency_cost', '元'),
                             ('planned_kwh', 'kWh'), ('emergency_kwh', 'kWh')]:
            add(old, primary, metric, unit, bool(old.ranking_allowed))
    current = batteries[batteries.policy_id == primary.policy_id].iloc[0]
    for _, old in batteries[batteries.policy_id != primary.policy_id].iterrows():
        for metric in ('throughput_kwh', 'equivalent_full_cycles', 'direct_adjacent_reversals',
                       'nonidle_direction_reversals', 'mean_absolute_change_kw', 'rms_change_kw',
                       'p95_absolute_change_kw', 'max_absolute_change_kw', 'total_variation_kw',
                       'large_jump_pct', 'power_limit_hit_pct'):
            unit = 'kWh' if metric == 'throughput_kwh' else '%' if metric.endswith('pct') else '次数' if 'reversal' in metric else '循环' if 'cycles' in metric else 'kW/相邻10分钟槽'
            add(old, current, metric, unit, bool(old.ranking_allowed))
    for _, current in forecast[forecast.policy_id == primary.policy_id].iterrows():
        matching = forecast[(forecast.policy_id != primary.policy_id)
                            & (forecast.target == current.target) & (forecast.population == current.population)]
        for _, old in matching.iterrows():
            for metric in ('mae', 'rmse', 'wape_pct', 'bias'):
                add(old, current, metric, '%' if metric == 'wape_pct' else 'kW',
                    not bool(old.exploratory), f'{current.target}/{current.population}',
                    'same realized targets and midnight horizon; reused forecasts identical')
    return pd.DataFrame(rows)


def build_evidence(history_only=False, figures=True):
    began = time.perf_counter()
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    sources = Sources()
    exp005, arrays, data = freeze_history(sources)
    if history_only:
        exp005.to_csv(EVIDENCE/'exp005_costs.csv', index=False, float_format='%.15g')
        write_json(EVIDENCE/'historical_source_manifest.json', {'sources': list(sources.rows.values())})
        return {'exp005': exp005}
    manifest = read(OUT/'run_manifest.json')
    if manifest.get('complete') is not True:
        raise RuntimeError('Formal run is incomplete; refusing partial result report')
    protocol_file = ROOT/'experiments/problem2/rl_planning/protocol.approved.json'
    protocol = read(protocol_file)
    sources.add(protocol_file, purpose='approved frozen experiment protocol')
    summary = pd.read_csv(OUT/'summary.csv')
    if 'id' not in summary:
        summary['id'] = summary['run_id']
    if set(summary.id) != set(RUNS):
        raise RuntimeError('Formal matrix must contain exactly the six approved runs')
    actual = data.actual[31*144:].reshape(334, 144, 2)
    days, training, validations, verification = [], [], [], []
    for run in RUNS:
        folder = OUT/run
        for file in ('dispatch_2.npz', 'daily.csv', 'training.csv', 'planning_audit.json', 'completion.json', 'verification.json'):
            sources.add(folder/file, purpose='verified formal RL output')
        v = read(folder/'verification.json')
        if not (v.get('passed') or v.get('status') == 'passed'):
            raise RuntimeError(f'Failed independent verification: {run}')
        z = load_archive(folder/'dispatch_2.npz')
        audit = audit_dispatch(z, actual, data.fixed_price)
        verification.append({'id': run, 'label': LABELS[run], **audit, 'runner_verification': v})
        arrays['exp007/'+run] = z
        daily = pd.read_csv(folder/'daily.csv')
        if daily.date.tolist() != DATES:
            raise AssertionError(f'Incomplete time order: {run}')
        np.testing.assert_allclose(daily.total_cost, z['fees'].sum(axis=(1, 2)), atol=1e-6, rtol=0)
        rowindex = summary.index[summary.id == run][0]
        if abs(summary.loc[rowindex, 'total_cost']-audit['total_cost']) > 1e-5:
            raise AssertionError(f'Cost summary mismatch: {run}')
        for key, value in {**battery_metrics(z), 'planned_cost': audit['planned_cost'],
                           'emergency_cost': audit['emergency_cost'], 'total_cost': audit['total_cost'],
                           'planned_kwh': z['original'].sum(), 'emergency_kwh': z['emergency'].sum(),
                           'worst_date': str(daily.loc[daily.total_cost.idxmax(), 'date']),
                           'worst_day_cost': daily.total_cost.max(), 'days': 334}.items():
            summary.loc[rowindex, key] = value
        daily['run_id'], daily['name'], daily['policy_id'], daily['label'] = run, run, 'exp007/'+run, LABELS[run]
        daily['month'] = pd.to_datetime(daily.date).dt.month
        days.append(daily)
        t = pd.read_csv(folder/'training.csv')
        t['run_id'], t['label'], t['family'] = run, LABELS[run], 'regularized' if run.startswith('regularized') else 'cost_only'
        t['recorded_row'] = np.arange(1, len(t)+1)
        if 'transitions' in t:
            t['cumulative_transitions'] = t.transitions.cumsum()
        if 'rollout_seconds' in t and 'update_seconds' in t:
            t['seconds'] = t.rollout_seconds+t.update_seconds
        if 'cutoff_day' in t:
            t['cutoff_date'] = t.cutoff_day.map(lambda day: str((pd.Timestamp('2025-01-01')+pd.Timedelta(days=int(day))).date()))
            fields = [key for key in ('run_id', 'cutoff_day', 'cutoff_date', 'training_start_day',
                                      'training_end_day', 'history_days', 'max_observed_index') if key in t]
            validations += t[fields].drop_duplicates().to_dict('records')
        training.append(t)
        if (folder/'training_audit.json').exists():
            sources.add(folder/'training_audit.json', purpose='causal RL fit boundaries')
            content = read(folder/'training_audit.json')
            entries = content if isinstance(content, list) else content.get('updates', content.get('training', []))
            validations += [{'run_id': run, **r} for r in entries]
    summary['run_id'] = summary.id
    summary['name'] = summary.id
    summary['label'] = summary.id.map(LABELS)
    summary['experiment'] = 'exp007'
    summary['policy_id'] = 'exp007/'+summary.id
    summary['rl_seed'] = summary.id.map(lambda r: int(r.rsplit('_', 1)[1]))
    summary['seed'] = summary.rl_seed
    summary['forecast_seed'] = 42
    summary['family'] = summary.id.map(lambda r: 'regularized' if r.startswith('regularized') else 'cost_only')
    summary['role'] = summary.id.map(lambda r: '正式策略' if r == 'regularized_42' else '训练种子稳定性' if r.startswith('regularized') else '预声明仅费用消融')
    summary['ranking_allowed'], summary['comparable'], summary['physically_comparable'], summary['exploratory'] = True, True, True, False
    summary['status'] = '同物理结算口径；冻结同预测；PPO规划和互斥投影'
    summary['source'] = summary.id.map(lambda r: f'data/results/exp007/{r}/dispatch_2.npz')
    summary['start_date'], summary['end_date'] = DATES[0], DATES[-1]
    oldcost = pd.read_csv(EVIDENCE/'history/exp006/cost_history.csv')
    oldcost['comparable'] = oldcost.ranking_allowed
    costs = pd.concat([oldcost, exp005, summary], ignore_index=True)
    for key in ('total', 'planned', 'emergency'):
        costs[key+'_wan'] = costs[key+'_cost']/10000
    costs['days'] = costs.days.fillna(334)
    costs['start_date'] = costs.start_date.fillna(DATES[0])
    costs['end_date'] = costs.end_date.fillna(DATES[-1])
    for _, row in costs.iterrows():
        source = ROOT/str(row.source)
        if source.is_file():
            sources.add(source, purpose='cost comparison source')
    forecast = pd.read_csv(EVIDENCE/'history/exp006/forecast_history.csv')
    forecast['policy_id'] = forecast.experiment+'/'+forecast['name'].fillna('midnight_rescored')
    reused = []
    for name, label, exp in [('regularized_42', LABELS['regularized_42'], 'exp007'),
                              ('cost_only_42', LABELS['cost_only_42'], 'exp007'),
                              ('frozen_no_season', 'exp005 复用无季节预测', 'exp005')]:
        p = forecast[(forecast.experiment == 'exp004') & (forecast.name == 'no_season')].copy()
        p['experiment'], p['name'], p['label'], p['policy_id'] = exp, name, label, exp+'/'+name
        p['forecast_reused'], p['forecast_seed'] = True, 42
        reused.append(p)
    forecast = pd.concat([forecast, *reused], ignore_index=True)
    # Independently verify reused annual errors from the actual frozen tensors.
    with np.load(ROOT/'data/results/exp004/predictions.npz') as z:
        predictions = z['no_season_seed_42']
    for _, row in forecast[forecast.policy_id == 'exp007/regularized_42'].iterrows():
        j = 0 if row.target == 'load' else 1
        a = actual[:, :, j] if row.target != 'net_load' else actual[:, :, 0]-actual[:, :, 1]
        p = predictions[:, :, j] if row.target != 'net_load' else predictions[:, :, 0]-predictions[:, :, 1]
        mask = np.ones(a.shape, dtype=bool) if row.population == 'all' else actual[:, :, 1] > 1e-6
        errors = p[mask]-a[mask]
        recalculated = [np.mean(np.abs(errors)), np.sqrt(np.mean(errors**2)), 100*np.abs(errors).sum()/np.abs(a[mask]).sum()]
        np.testing.assert_allclose(recalculated, [row.mae, row.rmse, row.wape_pct], atol=1e-8, rtol=1e-9)
    for source in forecast.source.dropna().unique():
        if (ROOT/source).is_file():
            sources.add(ROOT/source, purpose='historical forecast scores')
    archive_map = {'exp001/original': 'data/results/exp001/dispatch_2.npz',
                   'exp002/primary': 'data/results/exp002/dispatch_2.npz',
                   'exp003/primary': 'data/results/exp003/dispatch_2.npz',
                   **{f'exp004/{v}': f'data/results/exp004/{v}/dispatch_2.npz' for v in ('no_season', 'causal_season', 'oracle_season')}}
    battery_old = pd.read_csv(EVIDENCE/'history/exp006/battery_history.csv').set_index('policy_id')
    battery_rows = []
    for _, row in costs.iterrows():
        policy = row.policy_id
        result = {key: row.get(key) for key in ('experiment', 'policy_id', 'label', 'role', 'ranking_allowed', 'comparable', 'exploratory', 'source', 'status')}
        if policy in battery_old.index:
            result.update(battery_old.loc[policy].to_dict())
        path = archive_map.get(policy, row.source if row.experiment == 'exp006' else None)
        if path and (ROOT/path).is_file():
            sources.add(ROOT/path, purpose='historical raw battery evidence')
            arrays[policy] = load_archive(ROOT/path)
        if policy in arrays:
            metrics = battery_metrics(arrays[policy])
            if policy == 'exp001/original':
                metrics['equivalent_full_cycles'] = None
                metrics['warmup_boundary_jump_kw'] = None
                metrics['warmup_boundary_absolute_jump_kw'] = None
                metrics['total_variation_including_warmup_kw'] = None
            result.update(metrics)
        battery_rows.append(result)
    batteries = pd.DataFrame(battery_rows)
    for policy in ('exp004/no_season', 'exp006/primary', 'exp006/greedy_execution'):
        audit = audit_dispatch(arrays[policy], actual, data.fixed_price)
        registered = costs[costs.policy_id == policy].iloc[0]
        if abs(registered.total_cost-audit['total_cost']) > 1e-5:
            raise AssertionError(f'Frozen core historical cost changed: {policy}')
        verification.append({'id': policy, 'label': registered.label, **audit})
    daily = pd.concat(days, ignore_index=True)
    monthly = daily.groupby(['run_id', 'name', 'policy_id', 'label', 'month'], as_index=False)[['total_cost', 'planned_cost', 'emergency_cost', 'planned_kwh', 'emergency_kwh']].sum()
    # Add the three same-predictor anchors from their raw fee arrays.
    historical_daily = []
    for policy in POWER_POLICIES[:3]:
        z = arrays[policy]
        label = str(costs.loc[costs.policy_id == policy, 'label'].iloc[0])
        for i, date in enumerate(DATES):
            historical_daily.append({'policy_id': policy, 'label': label, 'date': date,
                'month': int(date[5:7]), 'total_cost': float(z['fees'][i].sum()),
                'planned_cost': float(z['fees'][i, :, 0].sum()), 'emergency_cost': float(z['fees'][i, :, 3].sum()),
                'planned_kwh': float(z['original'][i].sum()), 'emergency_kwh': float(z['emergency'][i].sum())})
    history_days = pd.DataFrame(historical_daily)
    hm = history_days.groupby(['policy_id', 'label', 'month'], as_index=False)[['total_cost', 'planned_cost', 'emergency_cost', 'planned_kwh', 'emergency_kwh']].sum()
    monthly = pd.concat([monthly, hm], ignore_index=True)
    paired = daily[daily.run_id == 'regularized_42'][['date', 'total_cost']].merge(
        history_days[history_days.policy_id == 'exp004/no_season'][['date', 'total_cost']], on='date', suffixes=('_current', '_baseline'))
    paired['difference_yuan'] = paired.total_cost_current-paired.total_cost_baseline
    paired['relative_change_pct'] = 100*paired.difference_yuan/paired.total_cost_baseline.abs()
    selected_failure_dates = sorted(set(paired.nlargest(3, 'difference_yuan').date.tolist()+paired.nlargest(1, 'total_cost_current').date.tolist()))
    details = []
    for policy in CORE:
        z = arrays[policy]
        label = costs.loc[costs.policy_id == policy, 'label'].iloc[0]
        for date in sorted(set(SPECIFIED+selected_failure_dates)):
            i = DATES.index(date)
            for slot in range(144):
                details.append({'policy_id': policy, 'label': label, 'date': date, 'slot': slot,
                    'hour': (slot+1)/6, 'actual_net_kwh': float((actual[i, slot, 0]-actual[i, slot, 1])/6),
                    'forecast_net_kwh': float((predictions[i, slot, 0]-predictions[i, slot, 1])/6),
                    'planned_kwh': float(z['original'][i, slot]), 'emergency_kwh': float(z['emergency'][i, slot]),
                    'charge_kwh': float(z['charge'][i, slot]), 'discharge_kwh': float(z['discharge'][i, slot]),
                    'power_kw': float(6*(z['charge'][i, slot]-z['discharge'][i, slot])),
                    'soc_start_kwh': float(z['states'][i, slot]), 'soc_end_kwh': float(z['states'][i, slot+1])})
    seed_stats = []
    for family, group in summary.groupby('family', sort=False):
        for metric in ('total_cost', 'emergency_cost', 'emergency_kwh', 'throughput_kwh',
                       'equivalent_full_cycles', 'direct_adjacent_reversals', 'nonidle_direction_reversals', 'total_variation_kw'):
            seed_stats.append({'family': family, 'metric': metric, 'n': len(group),
                               'mean': group[metric].mean(), 'sample_std': group[metric].std(ddof=1),
                               'minimum': group[metric].min(), 'maximum': group[metric].max(),
                               'meaning': 'three RL seeds; sample standard deviation, not a confidence interval'})
    timings = pd.read_csv(EVIDENCE/'history/exp006/timings.csv')
    run_times = pd.read_csv(OUT/'timings.csv')
    if 'stage' in run_times:
        run_times['experiment'] = 'exp007'
        if 'label' not in run_times:
            key = 'id' if 'id' in run_times else 'run_id'
            run_times['label'] = run_times[key].map(LABELS)
        run_times['source'] = 'data/results/exp007/timings.csv'
    else:
        entries = []
        stage_labels = {'training_seconds': 'RL策略训练',
                        'dataset_seconds_this_invocation': '历史特征与训练数据准备',
                        'checkpoint_seconds_this_invocation': '检查点存储',
                        'risk_seconds': '评价日特征准备', 'planning_seconds': '策略推理',
                        'execution_seconds': '逐槽执行', 'wall_seconds_this_invocation': '本次组墙钟'}
        for row in run_times.to_dict('records'):
            name = row.get('id', row.get('run_id'))
            for key, value in row.items():
                if key in stage_labels and isinstance(value, (int, float)):
                    entries.append({'experiment': 'exp007', 'label': LABELS.get(name, name), 'name': name,
                                    'stage': stage_labels[key], 'seconds': value, 'raw_field': key,
                                    'scope': 'recorded cumulative or invocation timing as named; wall time contains component stages; not additive across all rows',
                                    'source': 'data/results/exp007/timings.csv'})
        run_times = pd.DataFrame(entries)
    timings = pd.concat([timings, run_times], ignore_index=True)
    archived_runs = pd.read_csv(EVIDENCE/'history/exp005/runs.csv')
    t5 = [{'experiment': 'exp005', 'label': f'exp005 LP·β={r.beta:g}', 'stage': stage,
           'seconds': seconds, 'groups': 334, 'scope': 'completed historical arm; solver time included in wall time; different control protocol',
           'source': str((EVIDENCE/'history/exp005/runs.csv').relative_to(ROOT))}
          for r in archived_runs.itertuples() for stage, seconds in [('单组墙钟', r.wall_seconds), ('LP求解累计', r.solver_seconds)]]
    timings = pd.concat([timings, pd.DataFrame(t5), pd.DataFrame([{
        'experiment': 'exp007', 'label': 'exp007 复用预测', 'stage': '新增预测训练', 'seconds': 0.,
        'groups': 0, 'scope': 'read frozen exp004 tensors; RL policy training recorded separately',
        'source': str(protocol_file.relative_to(ROOT))}])], ignore_index=True)
    frames = {'cost_history': costs, 'forecast_history': forecast, 'battery_history': batteries,
              'battery_metrics': batteries,
              'power_variation': batteries, 'relative_comparison': relative_rows(costs, forecast, batteries),
              'timings': timings, 'all_runs': summary, 'daily': daily, 'historical_daily': history_days,
              'monthly_cost': monthly, 'paired_daily': paired, 'detail': pd.DataFrame(details),
              'specified': daily[(daily.run_id == 'regularized_42') & daily.date.isin(SPECIFIED)],
              'seed_costs': summary, 'seed_summary': pd.DataFrame(seed_stats),
              'training': pd.concat(training, ignore_index=True), 'training_boundaries': pd.DataFrame(validations),
              'failure_days': paired.sort_values('difference_yuan', ascending=False),
              'verification': pd.DataFrame([{k: v for k, v in r.items() if k != 'runner_verification'} for r in verification])}
    for name in ('forecast_monthly', 'forecast_lead'):
        path = ROOT/f'data/results/exp004/{name}.csv'
        if path.is_file():
            sources.add(path, purpose='frozen forecast time-stratified scores')
            frame = pd.read_csv(path)
            frame = frame[(frame.seed == 42) & (frame.name == 'no_season')].copy()
            frame['forecast_reused'] = True
            frames[name] = frame
    battery = build_power_evidence(arrays, costs, sources)
    frames['battery_random_days'] = pd.DataFrame(battery.pop('random_rows'))
    for name, frame in frames.items():
        frame.to_csv(EVIDENCE/f'{name}.csv', index=False, float_format='%.15g')
    for file in ('run_manifest.json', 'summary.csv', 'timings.csv'):
        sources.add(OUT/file, purpose='formal matrix completion and timing')
        shutil.copy2(OUT/file, EVIDENCE/('raw_'+file))
    shutil.copy2(protocol_file, EVIDENCE/'protocol.json')
    figure_rows = []
    if figures:
        from reports.exp007_figures import build_figures
        figure_rows = build_figures(frames, battery)
    power_export_qa = verify_power_exports(battery, figures)
    payload = {'experiment_id': 'exp007', 'primary_run': 'regularized_42',
               'tables': {name: records(frame) for name, frame in frames.items()},
               'sources': list(sources.rows.values()), 'protocol': protocol,
               'verification': {'passed': True, 'runs': verification, 'exp005': read(EVIDENCE/'exp005_compatibility_audit.json'),
                                'power_export': power_export_qa},
               'figures': figure_rows, 'battery': battery,
               'failure_dates': selected_failure_dates, 'specified_dates': SPECIFIED,
               'scope': {'days': 334, 'points_per_policy': 48096, 'start': DATES[0], 'end': DATES[-1]},
               'definitions': {'cost': 'sum(price*planned_kwh + 5*price*emergency_kwh); no training penalties or terminal credits',
                   'forecast': 'frozen exp004 no_season seed42, identical predictions for all RL seeds',
                   'delta_power': 'P[t]-P[t-1], 48095 differences including midnight; first difference missing; warmup separately',
                   'seed_summary': 'three independently trained PPO seeds; sample standard deviation ddof=1; not a confidence interval',
                   'training': 'historical training episode scores, not held-out evaluation or policy optimality guarantees'},
               'generator': 'reports/exp007_evidence.py', 'no_training_or_solver_invocation': True}
    write_json(EVIDENCE/'report_data.json', payload)
    write_json(EVIDENCE/'evidence_manifest.json', {'passed': True, 'generator': 'reports/exp007_evidence.py',
        'generator_sha256': sha256(__file__), 'rows': {key: len(frame) for key, frame in frames.items()},
        'files': {str(p.relative_to(EVIDENCE)): sha256(p)
                  for p in owned_manifest_paths(frames, sources.rows.values())},
        'build_seconds': time.perf_counter()-began, 'no_training_or_solver_invocation': True})
    return frames


def verify_power_exports(battery, check_figures=True):
    """Read exported CSV/SVG back; confirm raw point counts and time semantics."""
    entries = []
    for curve in battery['full_curves']:
        frame = pd.read_csv(REPORT/curve['csv'])
        expected = np.asarray(curve['net_power_kw'])
        times = pd.to_datetime(frame.timestamp_end)
        assert len(frame) == 48096 and times.is_unique
        assert np.all(np.diff(times.astype('int64')) == 600*10**9)
        assert str(times.iloc[0]) == '2025-02-01 00:10:00'
        assert str(times.iloc[-1]) == '2026-01-01 00:00:00'
        assert pd.isna(frame.delta_power_kw.iloc[0]) and frame.delta_power_kw.iloc[1:].notna().all()
        np.testing.assert_allclose(frame.net_power_kw, expected, rtol=0, atol=1e-8)
        np.testing.assert_allclose(frame.charge_power_kw-frame.discharge_power_kw, expected, rtol=0, atol=1e-8)
        np.testing.assert_allclose(frame.delta_power_kw.iloc[1:], np.diff(expected), rtol=0, atol=1e-8)
        node_counts = None
        if check_figures:
            svg = REPORT/'figures'/(curve['policy_id'].replace('/', '__')+'-full.svg')
            root = ET.parse(svg).getroot()
            counts = [node.attrib.get('d', '').count('L')+node.attrib.get('d', '').count('M')
                      for node in root.iter() if node.tag.endswith('path')]
            node_counts = max(counts)
            assert node_counts == 48096, f'SVG point loss: {svg}'
        entries.append({'policy_id': curve['policy_id'], 'passed': True, 'csv_rows': len(frame),
                        'svg_raw_points': node_counts, 'first_delta_missing': True,
                        'consecutive_10_minute_timestamps': True, 'csv_sha256': sha256(REPORT/curve['csv'])})
    lookup = {r['policy_id']: np.asarray(r['net_power_kw']).reshape(334, 144) for r in battery['full_curves']}
    for curve in battery['random_curves']:
        np.testing.assert_array_equal(curve['power_kw'], lookup[curve['policy_id']][DATES.index(curve['date'])])
    result = {'passed': True, 'full_curves': entries, 'random_curves_checked': len(battery['random_curves']),
              'random_points_per_curve': 144, 'no_decimation_or_smoothing': True}
    write_json(EVIDENCE/'battery_export_qa.json', result)
    return result


def build_power_evidence(arrays, costs, sources):
    dates = pd.date_range('2025-02-01 00:10', periods=48096, freq='10min')
    full, random, random_rows, compressed = [], [], [], {}
    folder = EVIDENCE/'battery'
    folder.mkdir(parents=True, exist_ok=True)
    for policy in POWER_POLICIES:
        z = arrays[policy]
        row = costs[costs.policy_id == policy].iloc[0]
        power, c, d = (6*(z['charge']-z['discharge'])).ravel(), (6*z['charge']).ravel(), (6*z['discharge']).ravel()
        name = policy.replace('/', '__')
        compressed[name] = power
        frame = pd.DataFrame({'timestamp_end': dates.strftime('%Y-%m-%d %H:%M'),
                              'charge_power_kw': c, 'discharge_power_kw': d, 'net_power_kw': power,
                              'soc_start_kwh': z['states'][:, :-1].ravel(), 'soc_end_kwh': z['states'][:, 1:].ravel(),
                              'delta_power_kw': np.r_[np.nan, np.diff(power)]})
        csv = folder/f'{name}_full.csv'
        frame.to_csv(csv, index=False, float_format='%.15g')
        experiment, run_name = policy.split('/', 1)
        archive_path = ROOT/f'data/results/{experiment}/{run_name}/dispatch_2.npz'
        source = sources.add(archive_path, purpose='source of complete 48096-point battery curve')
        full.append({'policy_id': policy, 'label': row.label, 'role': row.role, 'point_count': 48096,
                     'csv': str(csv.relative_to(REPORT)), 'csv_sha256': sha256(csv),
                     'source': str(archive_path.relative_to(ROOT)), 'source_sha256': source['sha256'],
                     'rl_seed': int(run_name.rsplit('_', 1)[1]) if experiment == 'exp007' else None,
                     'forecast_seed': 42, 'period': '2025-02-01/2025-12-31', 'sampling_interval_minutes': 10,
                     'npz_key': name, 'net_power_kw': power.tolist(), 'ranking_allowed': True})
        sample_rows = []
        for day in RANDOM_DAYS:
            values = power[day*144:(day+1)*144]
            random.append({'policy_id': policy, 'label': row.label, 'date': DATES[day],
                           'power_kw': values.tolist(), 'hours_end': ((np.arange(144)+1)/6).tolist()})
            for slot in range(144):
                position = day*144+slot
                sample_rows.append({'policy_id': policy, 'label': row.label, 'date': DATES[day], 'slot': slot,
                    'hour_end': (slot+1)/6, 'timestamp_end': str(dates[position]),
                    'charge_power_kw': float(c[position]), 'discharge_power_kw': float(d[position]),
                    'net_power_kw': float(power[position])})
        pd.DataFrame(sample_rows).to_csv(folder/f'{name}_random_days.csv', index=False, float_format='%.15g')
        random_rows += sample_rows
    np.savez_compressed(folder/'full_curves.npz', **compressed)
    return {'seed': RANDOM_SEED, 'sampling_algorithm': 'numpy.random.default_rng(seed).choice(334, 4, replace=False), sorted indices',
            'sample_day_indices': RANDOM_DAYS, 'dates': [DATES[i] for i in RANDOM_DAYS],
            'policies': POWER_POLICIES, 'full_point_count': 48096, 'full_day_count': 334,
            'timestamp_semantics': 'interval end: 00:10 means 00:00–00:10; last sample 2026-01-01 00:00',
            'start_timestamp_end': str(dates[0]), 'end_timestamp_end': str(dates[-1]),
            'power_definition': '6*(charge_kwh-discharge_kwh); AC bus; positive charge; negative discharge',
            'no_smoothing': True, 'csv_files': [x['csv'] for x in full],
            'full_npz': 'evidence/battery/full_curves.npz', 'full_curves': full,
            'random_curves': random, 'random_rows': random_rows,
            'zero_tolerance_kw': 6e-6, 'big_jump_threshold_kw_per_10min': 1000.,
            'big_jump_is_descriptive_not_hard_limit': True, 'warmup_boundary_power_kw': INITIAL_POWER}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--history-only', action='store_true')
    parser.add_argument('--no-figures', action='store_true')
    options = parser.parse_args()
    result = build_evidence(options.history_only, not options.no_figures)
    print(json.dumps({'tables': {key: len(value) for key, value in result.items()},
                      'evidence': str(EVIDENCE)}, ensure_ascii=False))
