"""One simultaneous-prefix historical envelope; fixed full334 LP replay.

This is an empirical risk reference, not a load/PV point forecast or a proven
joint chance constraint. Inventory efficiency/saturation and the intended-SOC
feedback floor prevent interpreting cumulative risk as exact storage need.
"""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd

from experiments.exp008.absolute_hgb_forecast_adapter import AbsoluteHGBForecasts
from experiments.exp008.controller_candidate import (
    Data, INITIAL_SOC, ETA, MIN_SOC, MAX_SOC, POWER_ENERGY, plan_inventory, execute_inventory,
)
from experiments.exp008.verify import verify_npz

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data/results/exp008/simultaneous_prefix_envelope'
PRIOR = ROOT / 'data/results/exp008/cumulative_risk_planning'
TREE = ROOT / 'data/results/exp008/forecast_absolute_hgb/lp_bridge/direct_hgb_ridge28_memory_tree28_q08_buffer500_334days'
SPEC = {'quantile': .8, 'controller': 'greedy', 'state_buffer': 500., 'wear': .002}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_hash(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def save(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def trajectory(forecast_net, errors):
    forecast_net, errors = np.asarray(forecast_net), np.asarray(errors)
    if errors.shape != (28, 144) or forecast_net.shape != (144,):
        raise ValueError('This fixed candidate requires 28 complete 144-slot error paths')
    cumulative = np.cumsum(errors, axis=1)
    mean = cumulative.mean(axis=0)
    sigma = np.maximum(cumulative.std(axis=0, ddof=0), 10.)
    scores = np.max((cumulative - mean) / sigma, axis=1)
    score_quantile = float(np.quantile(scores, .8, method='linear'))
    adjustment = mean + score_quantile * sigma
    increments = np.diff(np.r_[0., adjustment])
    risk = forecast_net + increments
    np.testing.assert_allclose(np.cumsum(risk - forecast_net), adjustment, rtol=0, atol=1e-8)
    np.testing.assert_allclose(risk.sum() - forecast_net.sum(), adjustment[-1], rtol=0, atol=1e-8)
    covered = np.all(cumulative <= adjustment + 1e-8, axis=1)
    np.testing.assert_array_equal(covered, scores <= score_quantile + 1e-10)
    return risk, {'mean': mean, 'sigma': sigma, 'path_scores': scores,
        'score_quantile': np.array(score_quantile), 'envelope': adjustment, 'increments': increments,
        'history_simultaneously_covered': covered,
        'history_pointwise_coverage': np.mean(cumulative <= adjustment + 1e-8, axis=0)}


def floor_diagnostic(detail):
    """Same-state instantaneous unmet demand due to the intended-SOC floor.

    This does not rerun a no-floor policy; spending this energy now changes
    subsequent SOC, so the summed cost proxy is not a realizable saving.
    """
    net = (detail['actual'][..., 0] - detail['actual'][..., 1]) / 6
    deficit = np.maximum(net - detail['original'], 0.)
    floor = np.maximum(MIN_SOC, detail['intended_states'][..., 1:] - 500.)
    soc = detail['states'][..., :-1]
    physical = np.minimum(np.minimum(deficit, POWER_ENERGY), np.maximum(0., (soc - MIN_SOC) * ETA))
    limited = np.minimum(np.minimum(deficit, POWER_ENERGY), np.maximum(0., (soc - floor) * ETA))
    np.testing.assert_allclose(detail['discharge'], limited, rtol=0, atol=1e-8)
    extra = np.maximum(physical - limited, 0.)
    assert np.all(extra <= detail['emergency'] + 1e-8)
    assert np.min(floor) >= MIN_SOC and np.max(floor) <= MAX_SOC
    return {'floor_kwh': floor, 'same_state_unrestricted_discharge_kwh': physical,
        'extra_unserved_demand_due_floor_kwh': extra}, {
        'floor_min_kwh': float(floor.min()), 'floor_mean_kwh': float(floor.mean()), 'floor_max_kwh': float(floor.max()),
        'floor_binding_deficit_slots': int(np.sum(extra > 1e-6)),
        'deficit_slots_with_floor_above_actual_soc': int(np.sum((deficit > 1e-6) & (floor > soc + 1e-6))),
        'same_state_extra_unserved_demand_due_floor_kwh': float(extra.sum()),
        'same_state_extra_emergency_fee_proxy_yuan': float((5 * detail['price'] * extra).sum()),
        'interpretation': 'same-state instantaneous diagnostic; summed fee is not a no-floor policy saving'}


def protocol():
    if OUT.exists():
        raise FileExistsError('The single fixed-envelope experiment may not overwrite prior results')
    OUT.mkdir(parents=True)
    prior_protocol = json.loads((PRIOR / 'protocol.json').read_text())
    assert prior_protocol['fixed_spec'] == SPEC
    files = [Path(__file__), ROOT / 'experiments/exp008/controller_candidate.py',
        ROOT / 'experiments/exp008/absolute_hgb_forecast_adapter.py', ROOT / 'experiments/exp008/forecast.py',
        ROOT / 'experiments/exp008/verify.py']
    inputs = [PRIOR / 'risk_inputs.npz', PRIOR / 'protocol.json', PRIOR / 'point/dispatch_2.npz',
        PRIOR / 'prefix/dispatch_2.npz', TREE / 'dispatch_2.npz',
        ROOT / 'data/results/exp008/forecast_absolute_hgb/direct_hgb_ridge28_memory_half.npz']
    config = {'candidate_count': 1, 'formal_days': 334, 'quantile': .8, 'quantile_method': 'linear',
        'history': 'same previous28 complete issued HGB errors with explicitly labelled January periodic fallback',
        'formula': 'C_jt=cumsum(error_jt); mu=mean(C); sigma=max(std(C,ddof0),10kWh); z_j=max_t((C_jt-mu_t)/sigma_t); z*=quantile(z,.8); a=mu+z*sigma; r=f+diff([0,a])',
        'sigma_floor_kwh': 10., 'std_ddof': 0, 'negative_increments_or_risk_clipped': False,
        'risk_is_not_a_load_PV_point_forecast': True,
        'not_a_true_joint_chance_constraint': True, 'eta_and_saturation_mapping_not_exact': True,
        'empirical_coverage': 'report actual finite-sample simultaneous coverage; do not label interpolation as exactly80pct',
        'fixed_spec': SPEC, 'initial_soc_kwh': INITIAL_SOC, 'own_continuous_realized_soc': True,
        'policy_link': 'r changes both purchases and intended SOC; state_buffer500 follows changed intended_SOC[t+1]',
        'floor_diagnostic': 'same-state maximum physical discharge minus floor-limited discharge; not a counterfactual full-policy saving',
        'all334_predeclared_no_partial_gate': True, 'development_not_independent_test': True,
        'source_sha256': {str(p.relative_to(ROOT)): digest(p) for p in files},
        'input_sha256': {str(p): digest(p) for p in inputs}, 'raw_data_sha256': Data().hashes}
    save(OUT / 'protocol.json', config)
    for path in files:
        target = OUT / 'sources' / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    return config


def comparison(own_check, own_arrays):
    entries = [('tree', TREE), ('point', PRIOR / 'point'), ('prefix', PRIOR / 'prefix')]
    result, rows = {}, []
    for name, folder in entries:
        checked = verify_npz(folder / 'dispatch_2.npz', audit_path=folder / 'audit.json')
        assert checked['passed'], checked['errors']
        with np.load(folder / 'dispatch_2.npz') as z:
            arrays = {key: z[key].copy() for key in z.files}
        _, floor = floor_diagnostic(arrays)
        result[name] = {'verification': checked, 'floor_diagnostic': floor}
        rows.append({'name': name, **checked['billing'], **checked['battery_metrics'], **floor})
    _, own_floor = floor_diagnostic(own_arrays)
    result['simultaneous_prefix'] = {'verification': own_check, 'floor_diagnostic': own_floor}
    rows.append({'name': 'simultaneous_prefix', **own_check['billing'], **own_check['battery_metrics'], **own_floor})
    pd.DataFrame(rows).to_csv(OUT / 'comparison.csv', index=False)
    save(OUT / 'comparison.json', result)
    return result


def run():
    config = protocol()
    data, forecast = Data(), AbsoluteHGBForecasts()
    with np.load(PRIOR / 'risk_inputs.npz') as z:
        prior_risk = {k: z[k].copy() for k in z.files}
    risks, records, audits, coverage_rows = [], [], [], []
    for day in range(31, 365):
        issue, history = forecast.get(day, scenario='2'), forecast.net_error_paths(day, '2', limit=28)
        f = (issue['load_kw'] - issue['pv_kw']) / 6
        errors = history['errors_kwh']
        np.testing.assert_array_equal(errors, prior_risk['errors'][day - 31])
        np.testing.assert_array_equal(f, prior_risk['forecast_net'][day - 31])
        assert history['audit']['max_observed_index'] < day * 144
        assert np.all(history['origins'] + 144 <= day * 144)
        risk, record = trajectory(f, errors)
        risks.append(risk)
        records.append(record)
        cumulative = np.cumsum(errors, axis=1)
        for method in ['point', 'prefix', 'simultaneous_prefix']:
            envelope = (record['envelope'] if method == 'simultaneous_prefix' else
                        np.cumsum(prior_risk[method][day - 31] - f))
            historical_covered = np.all(cumulative <= envelope + 1e-8, axis=1)
            coverage_rows.append({'day': day, 'method': method,
                'historical_paths_covered_at_all_prefixes': int(historical_covered.sum()),
                'historical_simultaneous_prefix_coverage': float(historical_covered.mean()),
                'historical_mean_pointwise_prefix_coverage': float(np.mean(cumulative <= envelope + 1e-8))})
        audits.append({'day': day, 'information_cutoff': day * 144, 'max_observed_index': day * 144 - 1,
            'forecast': issue['audit'], 'history': history['audit'], 'training_origins': history['origins'].tolist(),
            'risk_representation': 'simultaneous_prefix_empirical_envelope_not_point_forecast',
            'risk_sha256': array_hash(risk), 'score_quantile': float(record['score_quantile']),
            'historical_simultaneously_covered_count': int(record['history_simultaneously_covered'].sum()),
            'negative_envelope_increments': int(np.sum(record['increments'] < 0)),
            'risk_negative_slots': int(np.sum(risk < 0)),
            'joint_chance_certificate': False, 'purchase_locked_before_actual_read': True})
    risk_values = np.stack(risks)
    np.savez_compressed(OUT / 'risk_inputs.npz', risks=risk_values, forecast_net=prior_risk['forecast_net'],
        errors=prior_risk['errors'], days=np.arange(31, 365),
        **{key: np.stack([record[key] for record in records]) for key in records[0]})
    pd.DataFrame(coverage_rows).to_csv(OUT / 'historical_coverage_comparison.csv', index=False)
    mutations = []
    for day in (31, 32, 60, 151, 243, 364):
        changed = copy.copy(forecast.data)
        changed.actual = forecast.data.actual.copy()
        changed.actual[day * 144:] += np.array([70000., 30000., 1000.])
        after = AbsoluteHGBForecasts(changed)
        after.store.values[after.store.origins > day * 144] += 60000.
        issue, history = after.get(day, scenario='2'), after.net_error_paths(day)
        f = (issue['load_kw'] - issue['pv_kw']) / 6
        value, details = trajectory(f, history['errors_kwh'])
        np.testing.assert_array_equal(value, risk_values[day - 31])
        for key, array in records[day - 31].items():
            np.testing.assert_array_equal(details[key], array)
        mutations.append({'day': day, 'future_actual_and_future_forecast_mutation_all_envelope_arrays_identical': True})
    save(OUT / 'forecast_risk_causality.json', {'passed': True, 'checks': mutations})
    soc, mode, parts, daily, floor_arrays, actual_coverage = INITIAL_SOC, 1, [], [], [], []
    for i, day in enumerate(range(31, 365)):
        plan = plan_inventory(np.repeat(risk_values[i, :, None], 9, axis=1), data.fixed_price, soc, SPEC, final=day == 364)
        np.testing.assert_array_equal(plan['net'], risk_values[i])
        audits[i].update(initial_soc=soc, initial_mode=mode, purchase_sha256=array_hash(plan['purchase']),
            intended_soc_sha256=array_hash(plan['states']))
        actual = data.actual[day * 144:(day + 1) * 144]
        detail, mode = execute_inventory(plan, actual, data.fixed_price, soc, mode, SPEC)
        floor_values, floor = floor_diagnostic(detail)
        floor_arrays.append(floor_values)
        parts.append(detail)
        daily.append({'day': day, 'date': str((pd.Timestamp('2025-01-01') + pd.Timedelta(days=day)).date()),
            'total_cost': float(detail['fees'].sum()), 'planned_cost': float(detail['fees'][:, 0].sum()),
            'emergency_cost': float(detail['fees'][:, 3].sum()), 'initial_soc': soc,
            'final_soc': float(detail['states'][-1]), **floor})
        actual_error_prefix = np.cumsum((actual[:, 0] - actual[:, 1]) / 6 - prior_risk['forecast_net'][i])
        for method in ['point', 'prefix', 'simultaneous_prefix']:
            envelope = records[i]['envelope'] if method == 'simultaneous_prefix' else np.cumsum(prior_risk[method][i] - prior_risk['forecast_net'][i])
            actual_coverage.append({'day': day, 'method': method,
                'realized_all_prefixes_covered': bool(np.all(actual_error_prefix <= envelope + 1e-8)),
                'realized_pointwise_prefix_coverage': float(np.mean(actual_error_prefix <= envelope + 1e-8)),
                'realized_endpoint_covered': bool(actual_error_prefix[-1] <= envelope[-1] + 1e-8)})
        soc = daily[-1]['final_soc']
    arrays = {key: np.stack([part[key] for part in parts]) for key in parts[0]}
    arrays['days'] = np.arange(31, 365)
    np.savez_compressed(OUT / 'dispatch_2.npz', **arrays)
    np.savez_compressed(OUT / 'floor_diagnostics.npz', **{key: np.stack([row[key] for row in floor_arrays]) for key in floor_arrays[0]})
    save(OUT / 'audit.json', audits)
    pd.DataFrame(daily).to_csv(OUT / 'daily.csv', index=False)
    pd.DataFrame(actual_coverage).to_csv(OUT / 'realized_prefix_coverage.csv', index=False)
    check = verify_npz(OUT / 'dispatch_2.npz', audit_path=OUT / 'audit.json')
    assert check['passed'], check['errors']
    save(OUT / 'independent_physical_verification.json', check)
    prefix_tests = []
    for day in (31, 90, 151, 243, 364):
        i = day - 31
        plan = plan_inventory(np.repeat(risk_values[i, :, None], 9, axis=1), data.fixed_price,
                              float(arrays['states'][i, 0]), SPEC, final=day == 364)
        for new, stored in [('purchase', 'original'), ('charge', 'intended_charge'),
                            ('discharge', 'intended_discharge'), ('states', 'intended_states')]:
            np.testing.assert_array_equal(plan[new], arrays[stored][i])
        for stop in (1, 36, 108):
            observed = arrays['actual'][i].copy()
            observed[stop:] += np.array([80000., 30000.])
            result, _ = execute_inventory(plan, observed, data.fixed_price, float(arrays['states'][i, 0]),
                                          audits[i]['initial_mode'], SPEC)
            for key in ('charge', 'discharge', 'emergency', 'surplus', 'fees'):
                np.testing.assert_array_equal(result[key][:stop], arrays[key][i, :stop])
            np.testing.assert_array_equal(result['states'][:stop + 1], arrays['states'][i, :stop + 1])
            prefix_tests.append({'day': day, 'mutated_from_slot': stop, 'planning_arrays_exact_and_execution_prefix_unchanged': True})
    save(OUT / 'execution_causality.json', {'passed': True, 'checks': prefix_tests})
    compared = comparison(check, arrays)
    historical = pd.DataFrame(coverage_rows).groupby('method').mean(numeric_only=True).drop(columns='day')
    realized = pd.DataFrame(actual_coverage).groupby('method').mean(numeric_only=True).drop(columns='day')
    historical.to_csv(OUT / 'historical_coverage_means.csv')
    realized.to_csv(OUT / 'realized_coverage_means.csv')
    sources_ok = all(digest(ROOT / path) == expected for path, expected in config['source_sha256'].items())
    inputs_ok = all(digest(path) == expected for path, expected in config['input_sha256'].items())
    assert sources_ok and inputs_ok
    result = {'complete': True, 'days': 334, 'candidate_count': 1, 'verification': check,
        'historical_coverage': historical.to_dict('index'), 'realized_coverage': realized.to_dict('index'),
        'cost_changes_yuan': {name: check['recomputed_total_cost'] - compared[name]['verification']['recomputed_total_cost'] for name in ['tree', 'point', 'prefix']},
        'floor_diagnostic': compared['simultaneous_prefix']['floor_diagnostic'],
        'negative_increments_retained': int(sum(np.sum(row['increments'] < 0) for row in records)),
        'negative_risk_slots_retained': int(np.sum(risk_values < 0)),
        'all_endpoint_identities_passed': True, 'future_mutation_checks_passed': True,
        'source_and_input_hashes_unchanged': True, 'risk_is_not_point_forecast': True,
        'not_a_joint_chance_or_exact_storage_risk_certificate': True, 'final_model_selected': False}
    save(OUT / 'summary.json', result)
    print(json.dumps({key: value for key, value in result.items() if key != 'verification'}, indent=2), flush=True)
    print(json.dumps({'billing': check['billing'], 'battery': check['battery_metrics']}, indent=2), flush=True)


if __name__ == '__main__':
    run()
