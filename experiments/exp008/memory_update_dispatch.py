"""Fixed load-energy-memory Q3/Q4-3 linkage, 30 days before a fixed gate.

Only the two Pareto-passing scenarios together authorize a 334-day run.
No forecast/decision parameter search, Q2 gate, or final report is involved.
"""
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from experiments.common.neural_v2.data import Data
from experiments.exp008.forecast_override_adapter import AuditedForecastOverride
from experiments.exp008.issued_residual_paths import issued_error_paths
from experiments.exp008.load_energy_memory import EnergyMemoryForecasts
from experiments.exp008.planner import Settings
from experiments.exp008.run import OUT as RESULTS, ROOT, initial_state, run_case
from experiments.exp008.verify import TOL, source_arrays, verify_arrays, verify_npz

OUT = RESULTS / 'memory_update_dispatch'
BASELINE = RESULTS / 'update_value_diagnostic/future1.5_d334_issued'
MODEL_ID = 'exp008_joint_cnn_seed42_ridge28_load_energy_memory_fixed_gain_half'
SETTINGS = Settings(future_shortfall_weight=1.5)
SOURCE_FILES = ['experiments/exp008/' + name + '.py' for name in (
    'memory_update_dispatch', 'run', 'forecast_override_adapter', 'issued_residual_paths',
    'planner', 'forecast', 'unified_forecast', 'load_energy_memory', 'neural_joint_calibration',
    'forecast_calibration', 'neural_joint', 'verify')]
SOURCE_FILES += ['experiments/common/neural_v2/data.py', 'experiments/common/neural_v2/physics.py']
ARTIFACTS = ['load_energy_memory/predictions.npz', 'load_energy_memory/protocol.json',
    'load_energy_memory/prediction_audit.json', 'load_energy_memory/verification.json',
    'neural_joint_calibration/joint_ridge28.npz', 'neural_joint_calibration/manifest.json',
    'neural_joint_calibration/joint_ridge28_audit.json', 'neural_joint/manifest.json']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def memory_override():
    forecasts = EnergyMemoryForecasts()
    with np.load(RESULTS / 'load_energy_memory/predictions.npz') as z:
        np.testing.assert_array_equal(forecasts.store.values, z['values'])
        np.testing.assert_array_equal(forecasts.store.origins, z['origins'])
    return AuditedForecastOverride(forecasts, MODEL_ID,
        [ROOT / 'experiments/exp008/load_energy_memory.py',
         RESULTS / 'load_energy_memory/predictions.npz',
         RESULTS / 'load_energy_memory/protocol.json',
         RESULTS / 'neural_joint_calibration/joint_ridge28.npz',
         RESULTS / 'neural_joint_calibration/manifest.json'])


def fixed_protocol():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'protocol.json').exists():
        raise FileExistsError('This fixed experiment already exists; keep its evidence immutable')
    sources, artifacts = {}, {}
    for name in SOURCE_FILES:
        src = ROOT / name
        dst = OUT / 'source_archive' / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        sources[name] = digest(src)
    for name in ARTIFACTS:
        src = RESULTS / name
        dst = OUT / 'forecast_archive' / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        artifacts[name] = digest(src)
    protocol = {'model_id': MODEL_ID, 'forecast_candidate_count': 1, 'memory_gain': .5,
        'hyperparameter_search': False, 'scenarios': ['3', '4-3'], 'pilot_days': 30,
        'issued_residuals': True, 'settings': asdict(SETTINGS), 'deadband_kwh': 20.,
        'ramp_limit': None, 'charge_mask': None, 'updates': True, 'seed': 42,
        'baseline': str(BASELINE), 'baseline_prefix_only_no_reexecution': True,
        'comparison_dates_pilot': ['2025-02-01', '2025-03-02'],
        'extension_gate': 'BOTH scenarios must each have cost <= baseline+1e-6 and non-idle reversals <= baseline, with at least one strict improvement and valid physical/causal checks',
        'Q2_8pct_gate_applied_to_these_scenarios': False,
        'forecast_scope': 'current issue and prior same-issue completed-day residuals use EnergyMemoryForecasts; official PV and causal price mechanisms unchanged',
        'source_sha256': sources, 'artifact_sha256': artifacts,
        'source_raw_data_sha256': Data().hashes,
        'final_model_selection': False, 'final_report_generation': False,
        'development_on_examined_2025_not_independent_test': True}
    write(OUT / 'protocol.json', protocol)
    return protocol


def verify_baseline(scenario, days):
    directory = OUT / f'days{days}' / scenario
    path = BASELINE / scenario / f'dispatch_{scenario}.npz'
    with np.load(path) as z:
        a = {key: z[key][:days].copy() for key in z.files}
    audit = json.loads((BASELINE / scenario / 'audit.json').read_text())
    audit = [row for row in audit if row['day'] < 31 + days]
    actual, price = source_arrays(scenario, 31, days)
    soc, power = initial_state(scenario)
    check = verify_arrays(a, scenario, expected_days=days, initial_soc=soc,
        initial_power_kw=power, initial_mode=int(np.sign(power)), source_actual=actual,
        source_price=price, audit_records=audit)
    assert check['passed'], check['errors']
    check.update(source=str(path), source_sha256=digest(path),
                 selected_prefix_days=days, baseline_reexecuted=False)
    write(directory / 'baseline_verification.json', check)
    return a, check


def causal_checks():
    rows, residual_rows = [], []
    # Rebuild the memory layer from changed actuals, not a memoized archive.
    # This tests the new layer as well as the intraday adapter.
    for scenario in ('3', '4-3'):
        for day, slot in ((31, 0), (32, 36), (44, 72), (60, 108)):
            data = Data()
            before = EnergyMemoryForecasts(data=data)
            changed_data = copy.copy(data)
            changed_data.actual = data.actual.copy()
            cutoff = day * 144 + slot
            changed_data.actual[cutoff:] += np.array([100000., 50000., 1000.])
            changed_data._forecasts = {k: np.asarray(v).copy() + (90000. if k > cutoff else 0.)
                                      for k, v in data.forecasts.items()}
            after = EnergyMemoryForecasts(data=changed_data)
            f0, f1 = before.get(day, slot, scenario), after.get(day, slot, scenario)
            r0 = issued_error_paths(before, day, slot, scenario=scenario)
            r1 = issued_error_paths(after, day, slot, scenario=scenario)
            for key in ('load_kw', 'pv_kw', 'price'):
                np.testing.assert_array_equal(f0[key], f1[key])
            for key in ('errors_kw', 'errors_kwh', 'price_errors', 'origins', 'label_stops_exclusive'):
                np.testing.assert_array_equal(r0[key], r1[key])
            assert not f1['audit']['known_future_price']
            assert r1['label_stops_exclusive'].max() <= day * 144
            assert f1['audit']['max_observed_index'] < cutoff
            row = f1['audit']['load_energy_memory']
            assert row['prior_label_end_exclusive'] is None or row['prior_label_end_exclusive'] <= day * 144
            rows.append({'scenario': scenario, 'day': day, 'slot': slot,
                'future_actual_channels_and_unreleased_official_PV_changed': True,
                'memory_store_rebuilt_from_changed_actual': True,
                'prediction_and_historical_residual_array_max_difference': 0.,
                'causal_memory_label_stop': row['prior_label_end_exclusive'], 'cutoff': cutoff})
    # Directly reconstruct the history after the pilot has accumulated memory
    # predictions: this catches accidentally reusing original-network errors.
    forecast = memory_override()
    for scenario in ('3', '4-3'):
        for slot in (0, 36, 72, 108):
            day = 60
            risk = issued_error_paths(forecast, day, slot, scenario=scenario)
            manual = []
            for origin in risk['origins']:
                old = int(origin // 144)
                f = forecast.get(old, slot, scenario)
                actual = forecast._observed(int(origin), (old + 1) * 144, day * 144 + slot)
                manual.append(((actual[:, 0] - f['load_kw']) - (actual[:, 1] - f['pv_kw'])) / 6)
                memory = f['audit'].get('load_energy_memory')
                if memory:
                    assert memory['prior_label_end_exclusive'] is None or memory['prior_label_end_exclusive'] <= old * 144
            np.testing.assert_array_equal(risk['errors_kwh'], manual)
            residual_rows.append({'scenario': scenario, 'day': day, 'slot': slot,
                'model_id': MODEL_ID, 'all_28_paths_reconstructed_from_same_issue_memory_forecast': True,
                'labels_end_exclusive_max': int(risk['label_stops_exclusive'].max())})
    result = {'passed': True, 'future_mutation_checks': rows, 'history_reconstruction_checks': residual_rows,
        'all_memory_labels_prior_complete_day': all(r['prior_label_end_exclusive'] is None or
            r['prior_label_end_exclusive'] <= r['origin'] for r in forecast.store.audit),
        'scope': 'MemoryForecasts rebuilt under prefix mutation, intraday correction/price/official-PV release and same-issue history; base joint CNN and Ridge28 training evidence copied separately'}
    assert result['all_memory_labels_prior_complete_day']
    write(OUT / 'causality_verification.json', result)
    return result


def compare(scenario, days):
    case = f'memory_update_dispatch/days{days}'
    completion = run_case(case, scenario, method='joint', settings=SETTINGS, deadband=20.,
        days=days, seed=42, updates=True, calibration=None, issued_residuals=True,
        forecast_override=memory_override())
    directory = OUT / f'days{days}' / scenario
    soc, power = initial_state(scenario)
    checked = verify_npz(directory / f'dispatch_{scenario}.npz', scenario,
        expected_days=days, initial_soc=soc, initial_mode=int(np.sign(power)),
        initial_power_kw=power, audit_path=directory / 'audit.json')
    assert checked['passed'], checked['errors']
    write(directory / 'verification.json', checked)
    baseline, old = verify_baseline(scenario, days)
    with np.load(directory / f'dispatch_{scenario}.npz') as z:
        values = {key: z[key].copy() for key in z.files}
    if days == 334:
        with np.load(OUT / 'days30' / scenario / f'dispatch_{scenario}.npz') as prior:
            for key in values:
                np.testing.assert_array_equal(values[key][:30], prior[key])
    audit = json.loads((directory / 'audit.json').read_text())
    assert len(audit) == days * 4
    for row in audit:
        assert row['residual_paths']['model_id'] == MODEL_ID
        assert row['residual_paths']['historical_predictions_use_same_override']
        assert 'frozen_cnn' not in row['residual_paths']['source']
        assert row['residual_paths']['label_stops_exclusive'][-1] <= row['day'] * 144
        assert not row['forecast']['known_future_price']
    costs = (old['recomputed_total_cost'], checked['recomputed_total_cost'])
    reversals = (old['battery_metrics']['direction_reversals'], checked['battery_metrics']['direction_reversals'])
    gate = (costs[1] <= costs[0] + TOL and reversals[1] <= reversals[0]
            and (costs[1] < costs[0] - TOL or reversals[1] < reversals[0]))
    result = {'scenario': scenario, 'days': days, 'baseline': old, 'memory': checked,
        'cost_change_yuan': costs[1] - costs[0], 'cost_change_pct': 100 * (costs[1] / costs[0] - 1),
        'direction_reversal_change': reversals[1] - reversals[0],
        'fee_component_change': {key: checked['billing'][key] - old['billing'][key] for key in old['billing']},
        'own_scenario_pareto_gate': bool(gate), 'same_execution_and_planning_parameters': True,
        'same_initial_soc_kwh': soc, 'own_continuous_soc': True,
        'all_issue_model_ids_and_residual_label_stops_verified': True,
        'baseline_model': 'original CNN without calibration; previous best update_value configuration',
        'candidate_model': MODEL_ID, 'Q2_target_applied': False,
        'full_first30_matches_pilot': days == 334}
    write(directory / 'paired_findings.json', result)
    daily = []
    for i in range(days):
        row = {'day': i + 31, 'date': str((pd.Timestamp('2025-02-01') + pd.Timedelta(days=i)).date())}
        for label, a in [('baseline', baseline), ('memory', values)]:
            for j, key in enumerate(('planned_cost', 'up_cost', 'down_cost', 'emergency_cost')):
                row[f'{label}_{key}'] = float(a['fees'][i, :, j].sum())
            row[f'{label}_total_cost'] = float(a['fees'][i].sum())
            row[f'{label}_throughput_kwh'] = float((a['charge'][i] + a['discharge'][i]).sum())
        row['cost_change_yuan'] = row['memory_total_cost'] - row['baseline_total_cost']
        daily.append(row)
    pd.DataFrame(daily).to_csv(directory / 'paired_daily.csv', index=False)
    print(json.dumps({'scenario': scenario, 'days': days, 'costs': costs, 'reversals': reversals,
                      'pareto_gate': gate, 'component_change': result['fee_component_change']}), flush=True)
    return result


def main():
    protocol = fixed_protocol()
    causal = causal_checks()
    pilots = [compare(scenario, 30) for scenario in ('3', '4-3')]
    gate = causal['passed'] and all(r['own_scenario_pareto_gate'] for r in pilots)
    write(OUT / 'extension_gate.json', {'passed': gate, 'pilot_days': 30,
        'rule': protocol['extension_gate'], 'scenario_gates': {r['scenario']: r['own_scenario_pareto_gate'] for r in pilots},
        'causality_passed': causal['passed']})
    annual = [compare(scenario, 334) for scenario in ('3', '4-3')] if gate else []
    write(OUT / 'summary.json', {'model_id': MODEL_ID, 'pilot': pilots, 'annual': annual,
        'extension_gate_passed': gate, 'full_year_run_performed': bool(annual),
        'stopped_after_fixed_pilot': not gate, 'Q2_target_applied': False,
        'final_model_selected': False, 'final_report_generated': False,
        'no_parameter_iteration': True})


if __name__ == '__main__':
    main()
