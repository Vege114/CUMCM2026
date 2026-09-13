"""Independent source/history, hold-boundary, physical and causal replay audit."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.exp008.forecast_hgb_extra_trees_half import PRIMARY, RAW, RIDGE, OUT as FORECAST_OUT, HGB, EXTRA
from experiments.exp008.forecast_calibration import _calibrate, CONFIG
from experiments.exp008.mode_budget_hold1_physical import mode_state
from experiments.exp008.planner import execute
from experiments.exp008.run_blend_mode_budget_hold1 import HGB_HOLD1, OUT, digest, save, BlendForecasts, source_consistency
from experiments.exp008.verify import battery_metrics, verify_npz
from experiments.problem2.exp003.data import ROOT

ETA = float(np.sqrt(.9))
LOW, HIGH, LIMIT = 1200., 10800., 5000/6


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key].copy() for key in z.files}


def operation_segments(arrays):
    """Whole-horizon runs, clipped at evaluation edges; idle splits real bouts."""
    def lengths(values, exclude_zero=False):
        boundaries = np.r_[0, np.flatnonzero(values[1:] != values[:-1])+1, len(values)]
        sizes = np.diff(boundaries)
        return sizes[values[boundaries[:-1]] != 0] if exclude_zero else sizes

    net = (arrays['charge']-arrays['discharge']).ravel()
    actual = lengths(np.where(np.abs(net) > 1e-6, np.sign(net), 0), exclude_zero=True)
    planned = lengths(arrays['allowed_charge'].ravel().astype(int)) if 'allowed_charge' in arrays else None
    result = {}
    for name, run_lengths in [('planned_mode', planned), ('actual_contiguous_active', actual)]:
        result[f'{name}_segments'] = len(run_lengths) if run_lengths is not None else None
        for slots in (1, 2, 3):
            result[f'{name}_segments_{slots}slots'] = int(np.sum(run_lengths == slots)) if run_lengths is not None else None
        result[f'{name}_segments_under3slots'] = int(np.sum(run_lengths < 3)) if run_lengths is not None else None
        result[f'{name}_median_slots'] = float(np.median(run_lengths)) if run_lengths is not None and len(run_lengths) else None
        result[f'{name}_minimum_slots'] = int(np.min(run_lengths)) if run_lengths is not None and len(run_lengths) else None
    return result


def independently_verify_switch_budget(arrays):
    """Each actual sign change requires a distinct flip in the allowed mask.

    Compressing a +/- mask by deleting idle execution slots cannot increase
    variation. Include both the known initial planned and actual sign +1.
    The disjoint intervals between consecutive nonidle signs give an explicit
    injection from actual reversals to planned flips, including midnight.
    """
    mask = np.asarray(arrays['allowed_charge'], bool)
    planned = np.where(mask.ravel(), 1, -1)
    flips = np.flatnonzero(np.diff(np.r_[1, planned]) != 0)
    by_day = np.bincount(flips // 144, minlength=len(mask))
    assert np.all(by_day <= 8)
    charge, discharge = arrays['charge'].ravel(), arrays['discharge'].ravel()
    nonidle = np.flatnonzero(np.abs(charge-discharge) > 1e-6)
    signs = np.sign((charge-discharge)[nonidle]).astype(int)
    assert np.array_equal(signs, planned[nonidle])
    previous_index, previous_sign = -1, 1
    witness = []
    for index, sign in zip(nonidle, signs):
        if sign != previous_sign:
            crossing = flips[(flips > previous_index) & (flips <= index)]
            assert len(crossing) > 0
            witness.append(int(crossing[0]))
        previous_index, previous_sign = int(index), int(sign)
    assert len(witness) == len(set(witness))
    assert len(witness) <= len(flips) <= 8*len(mask)
    return {'passed': True, 'daily_planned_switch_counts': by_day.astype(int).tolist(),
        'planned_changes_including_initial_and_midnight': len(flips),
        'actual_reversals_including_initial': len(witness),
        'unique_planned_flip_witnesses': witness, 'bound_for_observed_days': 8*len(mask),
        'full334_upper_bound': 2672, 'below_exp0062729_by_construction': True,
        'proof': 'Nonidle actual signs are an order-preserving subsequence of allowed signs; deleting idle slots cannot increase sign changes.'}


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
    source_consistency()
    archived = OUT/'forecast_archive'/FORECAST_OUT.name
    raw, ridge, issued = [load(path) for path in (archived/f'{RAW}.npz',
                archived/f'{RIDGE}.npz', OUT/'issued_forecasts.npz')]
    final = load(archived/f'{PRIMARY}.npz')
    np.testing.assert_array_equal(issued['values'], final['values'])
    for stage in (raw, ridge, issued):
        np.testing.assert_array_equal(stage['origins'], np.arange(31, 365)*144)
    hgbraw = load(OUT/'forecast_archive'/HGB.name/'direct_hgb_raw.npz')
    extraraw = load(OUT/'forecast_archive'/EXTRA.name/'absolute_extra_trees_raw.npz')
    np.testing.assert_array_equal(raw['values'], (hgbraw['values']+extraraw['values'])/2)
    for family in (HGB, EXTRA):
        folder = OUT/'forecast_archive'/family.name
        assert json.loads((folder/'independent_verification.json').read_text())['passed']
        assert json.loads((folder/'causality_verification.json').read_text())['passed']
        training = json.loads((folder/'training_audit.json').read_text())
        assert len(training) == 11
        for record in training:
            asof = int((pd.Timestamp(2025, record['month'], 1)-pd.Timestamp(2025,1,1)).days)
            assert record['training_days'] == list(range(7,asof-7))
            assert record['validation_days'] == list(range(asof-7,asof))
            assert record['train_label_stop_exclusive'] == (asof-7)*144
            assert record['validation_label_stop_exclusive'] == asof*144
            for model in record['models']:
                assert digest(folder/'models'/Path(model['model_path']).name) == model['model_sha256']
    actual = np.stack([pd.read_csv(ROOT/'data/raw'/name).iloc[:, 1:].to_numpy(float)
         for name in ('附件2_小区负载.csv', '附件2_光伏发电实际功率.csv')], axis=-1)
    reference = pd.read_csv(ROOT/'data/raw/附件1.csv').iloc[:, 1:].to_numpy(float)
    ra = json.loads((archived/'ridge_audit.json').read_text())
    ma = json.loads((archived/'memory_audit.json').read_text())
    causal = json.loads((archived/'causality_verification.json').read_text())
    assert causal['passed'] and causal['all334_postprocessing_labels_before_issue']
    assert json.loads((archived/'independent_verification.json').read_text())['passed']
    expected = ridge['values'].copy()
    for i, day in enumerate(range(31, 365)):
        origin = day*144
        recalibrated, _ = _calibrate(actual.reshape(-1,2),raw['origins'],raw['values'],i,CONFIG['ridge_28'])
        np.testing.assert_array_equal(recalibrated,ridge['values'][i])
        assert ra[i]['origin'] == origin and not ra[i]['current_truth_used']
        assert all(old+144 <= origin for old in ra[i]['history_origins'])
        assert ma[i]['origin'] == origin and ma[i]['gain'] == .5 and not ma[i]['current_truth_used']
        if i:
            correction = .5*float(np.mean(actual[day-1, :, 0]-ridge['values'][i-1, :, 0]))
            expected[i, :, 0] = np.maximum(0, expected[i, :, 0]+correction)
            assert ma[i]['load_correction_kw'] == correction
            assert ma[i]['prior_label_end_exclusive'] == origin
    np.testing.assert_array_equal(expected, issued['values'])
    forecast = BlendForecasts()
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
    state = {'final_planned_mode': 1, 'final_planned_run_slots': 1}
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
        assert audits[i]['forecast']['load_method'] == PRIMARY
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
        assert int(np.sum(flips)) <= 8
        np.testing.assert_allclose(p['solved_flip_values'], flips, atol=1e-6, rtol=0)
        assert audits[i]['mip']['daily_planned_switch_cap'] == 8
        assert audits[i]['mip']['switching_cost'] == 0.
        assert audits[i]['mip']['exact_binary_XOR']
        assert audits[i]['mip']['hold_slots'] == 1
        assert audits[i]['mip']['inherited_locked_prefix_slots'] == 0
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
        mutant.store.values[day-31+1:] += [80000.,30000.]
        before, after = forecast.get(day), mutant.get(day)
        for key in ('load_kw', 'pv_kw', 'price'):
            np.testing.assert_array_equal(before[key], after[key])
        np.testing.assert_array_equal(forecast.net_error_paths(day)['errors_kwh'], mutant.net_error_paths(day)['errors_kwh'])
        future_checks.append({'day': day, 'signed_raw_models_causality_inherited': True,
            'issued_and_full28_historical_paths_asymmetric_future_mutation': True})
    prefix_equal = None
    if expected_days == 334:
        first3 = load(OUT/'first3_dispatch.npz')
        prefix_equal = {key: bool(np.array_equal(value, arrays[key][:3])) for key, value in first3.items()}
        assert all(prefix_equal.values())
        prefix_hashes = json.loads((OUT/'first3_planning_hashes.json').read_text())
        assert all(digest(OUT/f'planning_day{day}.npz') == h for day, h in prefix_hashes.items())
    switch_budget = independently_verify_switch_budget(arrays)
    flat = arrays['allowed_charge'].ravel()
    changes = np.flatnonzero(flat[1:] != flat[:-1])+1
    assert len(changes) < 2 or np.diff(changes).min() >= 1
    output = {'passed': True, 'days': expected_days, 'verification': verification,
        'all_source_and_forecast_model_hashes_passed': True, 'prediction_cutoffs_checked': 334,
        'historical_paths_rebuilt_independently': True, 'all_scenario_physics_passed': True,
        'continuous_SOC_and_planned_mode_age_passed': True, 'maximum_numerical_error': max_error,
        'execution_future_mutation_count': expected_days*3, 'forecast_future_checks': future_checks,
        'prefix_arrays_equal': prefix_equal, 'planned_changes_off_hour': int(np.sum(changes % 6 != 0)),
        'switch_budget_certificate': switch_budget, 'planned_changes': len(changes), 'last_planned_state': state,
        'nonanticipative_recourse_certificate': False}
    save(OUT/'independent_audit.json', output)
    pieces = [load(path) for path in sorted((ROOT/'data/results/exp006/primary/checkpoints').glob('chunk_*.npz'))]
    baseline = {key: np.concatenate([piece[key] for piece in pieces])[:expected_days] for key in pieces[0]}
    comparisons = [('exp006', baseline), ('hgb_extra_trees_half_hold1_cap8_kappa0', arrays)]
    old = load(HGB_HOLD1/'dispatch.npz')
    assert len(old['states']) == 334
    old = {key:value[:expected_days] for key,value in old.items()}
    old_protocol = json.loads((HGB_HOLD1/'evaluation_protocol.json').read_text())
    new_protocol = json.loads((OUT/'evaluation_protocol.json').read_text())
    matched_keys = ['evaluation_days','evaluation_start_day','evaluation_stop_exclusive',
        'mode_resolution_slots','minimum_planned_mode_hold_slots','initial_soc',
        'initial_planned_mode','initial_planned_run_slots','initial_two_slots_locked',
        'scenarios','scenario_selection','refinement_history_days','refinement_maxiter',
        'switching','daily_planned_switch_cap','wear','variation','deadband',
        'mip_seconds','target_mip_gap','terminal','actual_execution']
    assert all(old_protocol[key] == new_protocol[key] for key in matched_keys)
    for day in arrays['days']:
        a, b = load(HGB_HOLD1/f'planning_day{day}.npz'), load(OUT/f'planning_day{day}.npz')
        np.testing.assert_array_equal(a['selected_scenario_indices'], b['selected_scenario_indices'])
    comparisons.insert(1, ('original_HGB_hold1_cap8_kappa0',old))
    annual, monthly = [], []
    dates = pd.date_range('2025-02-01', periods=expected_days)
    for name, value in comparisons:
        annual.append({'name': name, 'total_cost': float(value['fees'].sum()),
            'planned_cost': float(value['fees'][:, :, 0].sum()), 'emergency_cost': float(value['fees'][:, :, 3].sum()),
            **battery_metrics(value), **operation_segments(value)})
        mode, power = 1, 172.76
        for month in sorted(set(dates.month)):
            sub = {key: array[dates.month == month] for key, array in value.items()}
            battery = battery_metrics(sub, initial_mode=mode, initial_power_kw=power)
            monthly.append({'name': name, 'month': int(month), 'total_cost': float(sub['fees'].sum()),
                            **battery, **operation_segments(sub)})
            mode, power = battery['final_mode'], battery['final_power_kw']
    pd.DataFrame(annual).to_csv(OUT/'paired_comparison.csv', index=False)
    pd.DataFrame(monthly).to_csv(OUT/'monthly_comparison.csv', index=False)
    findings = {'switch_budget_certificate': switch_budget, 'completed_days': expected_days, 'independent_audit_passed': True, 'metrics': annual,
        'original_HGB_hold1_planning_configuration_matched': matched_keys,
        'only_forecast_pipeline_and_own_issued_errors_change': True,
        'segment_definition': 'Run lengths clipped at evaluation edges; planned modes include idle slots; actual active same-sign bouts split at idle slots.',
        'switching_budget_is_not_a_battery_lifetime_improvement_certificate': True,
        'goal': verification['goal'],
        'mip_gap_mean': float(np.mean([row['mip']['mip_gap'] for row in audits])),
        'mip_gap_maximum': float(max(row['mip']['mip_gap'] for row in audits)),
        'all_334_predeclared': True, 'partial_cost_and_count_not_selection_gate': True}
    save(OUT/'paired_findings.json', findings)
    print(json.dumps(findings, indent=2), flush=True)
    source_consistency()
    return output


if __name__ == '__main__':
    audit()
