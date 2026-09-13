"""Independent source/history, hold-boundary, physical and causal replay audit."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from experiments.exp008.absolute_hgb_forecast_adapter import AbsoluteHGBForecasts
from experiments.exp008.forecast_absolute_hgb import PRIMARY, predict_day
from experiments.exp008.mode_minhold_physical import mode_state
from experiments.exp008.planner import execute
from experiments.exp008.run_hgb_mode_minhold import HOURLY, OUT, digest, save
from experiments.exp008.verify import battery_metrics, verify_npz
from experiments.problem2.exp003.data import ROOT

ETA = float(np.sqrt(.9))
LOW, HIGH, LIMIT = 1200., 10800., 5000/6


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key].copy() for key in z.files}


def audit(expected_days=334):
    arrays = load(OUT/'dispatch.npz')
    assert len(arrays['days']) == expected_days
    verification = verify_npz(OUT/'dispatch.npz', expected_days=expected_days, audit_path=OUT/'planning_audit.json')
    assert verification['passed'], verification['errors']
    provenance = json.loads((OUT/'complete_provenance.json').read_text())
    for name, expected in provenance['source_sha256'].items():
        assert digest(OUT/'source_archive'/name) == expected
    for name, expected in provenance['forecast_artifact_sha256'].items():
        assert digest(OUT/'forecast_archive'/name) == expected
    for name, expected in provenance['source_data_sha256'].items():
        assert digest(ROOT/'data/raw'/name) == expected
    assert digest(OUT/'issued_forecasts.npz') == provenance['issued_archive_sha256']
    archived = OUT/'forecast_archive'
    raw, ridge, issued = [load(path) for path in (archived/'direct_hgb_raw.npz',
                archived/'direct_hgb_ridge28.npz', OUT/'issued_forecasts.npz')]
    final = load(archived/f'{PRIMARY}.npz')
    np.testing.assert_array_equal(issued['values'], final['values'])
    for stage in (raw, ridge, issued):
        np.testing.assert_array_equal(stage['origins'], np.arange(31, 365)*144)
    actual = np.stack([pd.read_csv(ROOT/'data/raw'/name).iloc[:, 1:].to_numpy(float)
         for name in ('附件2_小区负载.csv', '附件2_光伏发电实际功率.csv')], axis=-1)
    reference = pd.read_csv(ROOT/'data/raw/附件1.csv').iloc[:, 1:].to_numpy(float)
    training = json.loads((archived/'training_audit.json').read_text())
    predictions = json.loads((archived/'prediction_audit.json').read_text())
    ra = json.loads((archived/'ridge28_audit.json').read_text())
    ma = json.loads((archived/'memory_half_audit.json').read_text())
    causal = json.loads((archived/'causality_verification.json').read_text())
    assert causal['passed'] and causal['all_postprocessing_labels_past']
    for record in training:
        asof = int((pd.Timestamp(2025, record['month'], 1)-pd.Timestamp(2025, 1, 1)).days)
        assert record['asof_day'] == asof and record['training_days'] == list(range(7, asof-7))
        assert record['validation_days'] == list(range(asof-7, asof))
        assert record['train_label_stop_exclusive'] == (asof-7)*144
        assert record['validation_label_stop_exclusive'] == asof*144
        assert not record['periodic_or_CNN_reference_columns']
        for model in record['models']:
            assert digest(archived/'models'/Path(model['model_path']).name) == model['model_sha256']
    expected = ridge['values'].copy()
    for i, day in enumerate(range(31, 365)):
        origin = day*144
        assert predictions[i]['day'] == day and predictions[i]['origin'] == origin
        assert predictions[i]['feature_last_actual_index'] == origin-1
        assert predictions[i]['training_label_stop'] < predictions[i]['validation_label_stop'] <= origin
        assert predictions[i]['no_CNN_periodic_official_PV_future_price_input']
        assert ra[i]['origin'] == origin and not ra[i]['current_truth_used']
        assert all(old+144 <= origin for old in ra[i]['history_origins'])
        assert ma[i]['origin'] == origin and ma[i]['gain'] == .5 and not ma[i]['current_truth_used']
        if i:
            correction = .5*float(np.mean(actual[day-1, :, 0]-ridge['values'][i-1, :, 0]))
            expected[i, :, 0] = np.maximum(0, expected[i, :, 0]+correction)
            assert ma[i]['load_correction_kw'] == correction
            assert ma[i]['prior_label_end_exclusive'] == origin
    np.testing.assert_array_equal(expected, issued['values'])
    forecast = AbsoluteHGBForecasts()
    np.testing.assert_array_equal(forecast.store.values, issued['values'])

    def midnight(day):
        if day >= 31:
            return issued['values'][day-31]
        value = reference[:, 1:3].copy()
        for channel, lag in ((0, 7), (1, 1)):
            old = day-lag if day >= lag else day-1
            if old >= 0:
                value[:, channel] = actual[old, :, channel]
        return value

    audits = json.loads((OUT/'planning_audit.json').read_text())
    assert [a['day'] for a in audits] == list(range(31, 31+expected_days))
    max_error = 0.
    for i, day in enumerate(arrays['days']):
        day = int(day)
        p = load(OUT/f'planning_day{day}.npz')
        history = np.arange(day-28, day)
        residual = np.stack([((actual[old, :, 0]-midnight(old)[:, 0])
                        -(actual[old, :, 1]-midnight(old)[:, 1]))/6 for old in history])
        current = midnight(day)
        paths = (current[:, 0]-current[:, 1])[None, :]/6+residual
        error = float(np.max(np.abs(p['all_net_paths']-paths)))
        np.testing.assert_array_equal(p['selected_scenario_indices'], np.linspace(0, 27, 3).astype(int))
        assert audits[i]['history']['model_id'] == PRIMARY
        assert audits[i]['history']['max_observed_index'] == day*144-1
        assert audits[i]['history']['fallback_days'] == history[history < 31].tolist()
        assert audits[i]['forecast']['selected_model_id'] == PRIMARY
        assert audits[i]['forecast']['load_method'] == 'direct_absolute_HGB_Ridge28_memory_half'
        assert float(p['initial_soc']) == arrays['states'][i, 0]
        if i:
            assert float(p['initial_soc']) == arrays['states'][i-1, -1]
            assert int(p['previous_planned_mode']) == state['final_planned_mode']
            assert int(p['previous_planned_run_slots']) == state['final_planned_run_slots']
        else:
            assert int(p['previous_planned_mode']) == 1 and int(p['previous_planned_run_slots']) == 1
        mask = p['allowed_charge'].astype(bool)
        np.testing.assert_array_equal(mask, arrays['allowed_charge'][i])
        state, flips = mode_state(mask, int(p['previous_planned_mode']), int(p['previous_planned_run_slots']))
        np.testing.assert_array_equal(flips, p['planned_flips'])
        c, d, e, w, s = [p['scenario_'+key] for key in ('charge', 'discharge', 'emergency', 'surplus', 'states')]
        net = paths[p['selected_scenario_indices']]
        error = max(error, float(np.max(np.abs(p['purchase'][None, :]+d+e-c-w-net))),
                    float(np.max(np.abs(np.diff(s, axis=1)-ETA*c+d/ETA))))
        assert min(c.min(), d.min(), e.min(), w.min()) >= -1e-6
        assert s.min() >= LOW-1e-6 and s.max() <= HIGH+1e-6
        assert max(c.max(), d.max()) <= LIMIT+1e-6
        assert not np.any((c > 1e-6) & (d > 1e-6)) and not np.any((c > 1e-6) & (e > 1e-6))
        assert np.max(c[:, ~mask], initial=0.) < 1e-6 and np.max(d[:, mask], initial=0.) < 1e-6
        assert np.all(e <= np.maximum(net, 0)+1e-6)
        assert np.all(p['purchase'] <= np.maximum(net.max(0), 0)+LIMIT+1e-6)
        q, price = arrays['original'][i], arrays['price'][i]
        soc = arrays['states'][i, 0]
        for t in range(144):
            balance = q[t]+(actual[day, t, 1]-actual[day, t, 0])/6
            charge = min(max(balance, 0), LIMIT, max(0, (HIGH-soc)/ETA)) if mask[t] else 0.
            discharge = min(max(-balance, 0), LIMIT, max(0, (soc-LOW)*ETA)) if not mask[t] else 0.
            flows = [charge, discharge, max(0, -balance-discharge), max(0, balance-charge)]
            stored = [arrays[key][i, t] for key in ('charge', 'discharge', 'emergency', 'surplus')]
            error = max(error, float(np.max(np.abs(np.array(flows)-stored))))
            soc += ETA*charge-discharge/ETA
        error = max(error, abs(soc-arrays['states'][i, -1]))
        for cutoff in (1, 71, 143):
            polluted = actual[day].copy()
            polluted[cutoff:] += np.array([50000., 30000.])
            mutated = execute(q, polluted, price, float(p['initial_soc']), charge_mask=mask, charge_deadband=0.)
            for key in ('charge', 'discharge', 'emergency', 'surplus'):
                error = max(error, float(np.max(np.abs(mutated[key][:cutoff]-arrays[key][i, :cutoff]))))
            error = max(error, float(np.max(np.abs(mutated['states'][:cutoff+1]-arrays['states'][i, :cutoff+1]))))
        assert error < 1e-6
        max_error = max(max_error, error)
    future_checks = []
    for day in (31, 32, 59, 90, 151, 243, 364):
        if day >= 31+expected_days:
            continue
        mutant = copy.copy(forecast)
        mutant.data = copy.copy(forecast.data)
        mutant.data.actual = forecast.data.actual.copy()
        mutant.data.actual[day*144:] += np.array([50000., 30000., 2.])
        mutant.store = copy.copy(forecast.store)
        mutant.store.values = forecast.store.values.copy()
        mutant.store.values[day-31+1:] += 80000.
        before, after = forecast.get(day), mutant.get(day)
        for key in ('load_kw', 'pv_kw', 'price'):
            np.testing.assert_array_equal(before[key], after[key])
        np.testing.assert_array_equal(forecast.net_error_paths(day)['errors_kwh'], mutant.net_error_paths(day)['errors_kwh'])
        month = int((pd.Timestamp('2025-01-01')+pd.Timedelta(days=day)).month)
        record = next(row for row in training if row['month'] == month)
        models = [joblib.load(archived/'models'/Path(row['model_path']).name) for row in record['models']]
        predicted = predict_day(forecast.data, day, models)
        np.testing.assert_array_equal(predicted, raw['values'][day-31])
        np.testing.assert_array_equal(predicted, predict_day(mutant.data, day, models))
        future_checks.append({'day': day, 'raw_model_reload_and_future_mutation': True,
            'issued_and_full28_historical_paths_future_mutation': True})
    prefix_equal = None
    if expected_days == 334:
        first3 = load(OUT/'first3_dispatch.npz')
        prefix_equal = {key: bool(np.array_equal(value, arrays[key][:3])) for key, value in first3.items()}
        assert all(prefix_equal.values())
    flat = arrays['allowed_charge'].ravel()
    changes = np.flatnonzero(flat[1:] != flat[:-1])+1
    assert len(changes) < 2 or np.diff(changes).min() >= 3
    output = {'passed': True, 'days': expected_days, 'verification': verification,
        'all_source_and_forecast_model_hashes_passed': True, 'prediction_cutoffs_checked': 334,
        'historical_paths_rebuilt_independently': True, 'all_scenario_physics_passed': True,
        'continuous_SOC_and_planned_mode_age_passed': True, 'maximum_numerical_error': max_error,
        'execution_future_mutation_count': expected_days*3, 'forecast_future_checks': future_checks,
        'prefix_arrays_equal': prefix_equal, 'planned_changes_off_hour': int(np.sum(changes % 6 != 0)),
        'planned_changes': len(changes), 'last_planned_state': state,
        'nonanticipative_recourse_certificate': False}
    save(OUT/'independent_audit.json', output)
    pieces = [load(path) for path in sorted((ROOT/'data/results/exp006/primary/checkpoints').glob('chunk_*.npz'))]
    baseline = {key: np.concatenate([piece[key] for piece in pieces])[:expected_days] for key in pieces[0]}
    comparisons = [('exp006', baseline), ('direct_hgb_memory_hold3', arrays)]
    hourly_available = False
    if (HOURLY/'dispatch.npz').exists():
        hourly = load(HOURLY/'dispatch.npz')
        if len(hourly['states']) >= expected_days:
            hourly = {key: value[:expected_days] for key, value in hourly.items()}
            for day in arrays['days']:
                old, new = load(HOURLY/f'planning_day{day}.npz'), load(OUT/f'planning_day{day}.npz')
                np.testing.assert_array_equal(old['all_net_paths'], new['all_net_paths'])
                np.testing.assert_array_equal(old['selected_scenario_indices'], new['selected_scenario_indices'])
            comparisons.insert(1, ('direct_hgb_memory_hourly', hourly))
            hourly_available = True
    annual, monthly = [], []
    dates = pd.date_range('2025-02-01', periods=expected_days)
    for name, value in comparisons:
        annual.append({'name': name, 'total_cost': float(value['fees'].sum()),
            'planned_cost': float(value['fees'][:, :, 0].sum()), 'emergency_cost': float(value['fees'][:, :, 3].sum()),
            **battery_metrics(value)})
        mode, power = 1, 172.76
        for month in sorted(set(dates.month)):
            sub = {key: array[dates.month == month] for key, array in value.items()}
            battery = battery_metrics(sub, initial_mode=mode, initial_power_kw=power)
            monthly.append({'name': name, 'month': int(month), 'total_cost': float(sub['fees'].sum()), **battery})
            mode, power = battery['final_mode'], battery['final_power_kw']
    pd.DataFrame(annual).to_csv(OUT/'paired_comparison.csv', index=False)
    pd.DataFrame(monthly).to_csv(OUT/'monthly_comparison.csv', index=False)
    findings = {'completed_days': expected_days, 'independent_audit_passed': True, 'metrics': annual,
        'same_HGB_hourly_comparator_available': hourly_available, 'goal': verification['goal'],
        'mip_gap_mean': float(np.mean([row['mip']['mip_gap'] for row in audits])),
        'mip_gap_maximum': float(max(row['mip']['mip_gap'] for row in audits)),
        'all_334_predeclared': True, 'partial_cost_and_count_not_selection_gate': True}
    save(OUT/'paired_findings.json', findings)
    print(json.dumps(findings, indent=2), flush=True)
    return output


if __name__ == '__main__':
    audit()
