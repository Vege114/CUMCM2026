"""Immutable independent supplement; no signed experiment or dispatch changes."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.exp008.absolute_hgb_forecast_adapter import AbsoluteHGBForecasts
from experiments.exp008.cumulative_risk_planning import trajectory
from experiments.exp008.verify import verify_npz
from experiments.problem2.exp003.data import ROOT

BASE = ROOT/'data/results/exp008/cumulative_risk_planning'
ORACLE = ROOT/'data/results/exp008/oracle_diagnostic_cumulative_risk'
OUT = ROOT/'data/results/exp008/cumulative_risk_supplement_v1'
ETA = float(np.sqrt(.9))
LOW, HIGH, LIMIT, TARGET = 1200., 10800., 5000/6, 12940956.8791


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def load(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def q80(values):
    """Independently spell out NumPy's linear empirical interpolation."""
    ordered = np.sort(np.asarray(values, float), axis=0)
    index = .8*(len(ordered)-1)
    lo = int(np.floor(index)); hi = int(np.ceil(index)); fraction = index-lo
    return ordered[lo]+fraction*(ordered[hi]-ordered[lo])


def rebuild(forecast, errors, method):
    if method == 'prefix':
        cumulative = q80(np.cumsum(errors, axis=1))
        increments = np.diff(np.r_[0., cumulative])
    else:
        increments = q80(errors)
        cumulative = np.cumsum(increments)
    return forecast+increments, cumulative, increments


def mathematical_checks():
    f = np.array([100., 90., 120., 80.])
    single = np.array([[4., -9., 12., -3.]])
    p, _ = trajectory(f, single, 'prefix')
    point, _ = trajectory(f, single, 'point')
    np.testing.assert_array_equal(p, f+single[0])
    np.testing.assert_array_equal(p, point)
    zero = np.zeros((28, 4))
    for method in ('prefix', 'point'):
        np.testing.assert_array_equal(trajectory(f, zero, method)[0], f)
    coupled = np.r_[np.full((14, 2), 10.), np.full((14, 2), -10.)]
    shuffled = coupled.copy(); shuffled[:, 1] = shuffled[::-1, 1]
    prefix0 = trajectory(np.zeros(2), coupled, 'prefix')[0]
    prefix1 = trajectory(np.zeros(2), shuffled, 'prefix')[0]
    point0 = trajectory(np.zeros(2), coupled, 'point')[0]
    point1 = trajectory(np.zeros(2), shuffled, 'point')[0]
    np.testing.assert_array_equal(point0, point1)
    assert not np.array_equal(prefix0, prefix1)
    for method in ('prefix', 'point'):
        np.testing.assert_array_equal(trajectory(np.zeros(2), coupled, method)[0],
                                      trajectory(np.zeros(2), coupled[::-1], method)[0])
    disjoint = np.zeros((28, 2)); disjoint[:4, 0] = 10.; disjoint[4:8, 1] = 10.
    greater_prefix = float(trajectory(np.zeros(2), disjoint, 'prefix')[0].sum())
    lesser_point = float(trajectory(np.zeros(2), disjoint, 'point')[0].sum())
    assert greater_prefix == 10. and lesser_point == 0.
    lesser_prefix = float(prefix1.sum()); greater_point = float(point1.sum())
    assert lesser_prefix == 0. and greater_point == 20.
    return {'single_path_identity': True, 'zero_error_identity': True,
        'whole_history_row_permutation_invariant': True,
        'columnwise_history_coupling_shuffle_changes_prefix_not_point': True,
        'no_uniform_quantile_sum_ordering_examples': {
            'disjoint4_plus4_out_of28_positive_paths': {'cumulative_quantile': greater_prefix, 'sum_slot_quantiles': lesser_point},
            'balanced_antithetic_paths': {'cumulative_quantile': lesser_prefix, 'sum_slot_quantiles': greater_point}}}


def method_audit(name, inputs):
    detail = load(BASE/name/'dispatch_2.npz')
    verified = verify_npz(BASE/name/'dispatch_2.npz', expected_days=334, audit_path=BASE/name/'audit.json')
    assert verified['passed'], verified['errors']
    q, c, d, states = [detail[key] for key in ('original', 'intended_charge', 'intended_discharge', 'intended_states')]
    risk = inputs[name]
    spill = q-c+d-risk
    soc_error = np.diff(states, axis=1)-ETA*c+d/ETA
    assert np.max(np.abs(soc_error)) < 1e-6
    np.testing.assert_array_equal(states[:, 0], detail['states'][:, 0])
    assert min(q.min(), c.min(), d.min(), spill.min()) >= -1e-6
    assert states.min() >= LOW-1e-6 and states.max() <= HIGH+1e-6
    assert max(c.max(), d.max()) <= LIMIT+1e-6
    overlap = np.minimum(c, d)
    assert np.max(overlap) < 1e-6
    planned = {'maximum_SOC_equation_error_kwh': float(np.abs(soc_error).max()),
        'reconstructed_spill_minimum_kwh': float(spill.min()),
        'maximum_planned_charge_discharge_overlap_kwh': float(overlap.max()),
        'planned_soc_minimum_kwh': float(states.min()), 'planned_soc_maximum_kwh': float(states.max()),
        'planned_power_limit_kwh': LIMIT, 'all_initial_planned_states_equal_actual_initial': True,
        'passed': True, 'not_claiming_binary_mutex_in_continuous_LP': True}
    floor = np.empty_like(q); physical = np.empty_like(q); withheld = np.empty_like(q)
    max_discharge_error = 0.
    for i in range(334):
        for t in range(144):
            before = float(detail['states'][i, t])
            shortage = max(0., float((detail['actual'][i, t, 0]-detail['actual'][i, t, 1])/6-q[i, t]))
            floor[i, t] = max(LOW, float(states[i, t+1])-500.)
            physical[i, t] = min(shortage, LIMIT, max(0., (before-LOW)*ETA))
            limited = min(shortage, LIMIT, max(0., (before-floor[i, t])*ETA))
            max_discharge_error = max(max_discharge_error, abs(limited-detail['discharge'][i, t]))
            withheld[i, t] = max(0., physical[i, t]-limited)
    assert max_discharge_error < 1e-8
    assert np.all(withheld <= detail['emergency']+1e-8)
    deficit = (detail['actual'][..., 0]-detail['actual'][..., 1])/6-q
    gross_fee = 5*detail['price']*withheld
    floor_result = {'floor_min_kwh': float(floor.min()), 'floor_mean_kwh': float(floor.mean()),
        'floor_max_kwh': float(floor.max()), 'floor_binding_deficit_slots': int(np.sum(withheld > 1e-6)),
        'deficit_slots_with_floor_above_actual_soc': int(np.sum((deficit > 1e-6)&(floor > detail['states'][:, :-1]+1e-6))),
        'same_state_extra_unserved_demand_due_floor_kwh': float(withheld.sum()),
        'same_state_extra_emergency_fee_proxy_yuan': float(gross_fee.sum()),
        'maximum_scalar_discharge_rebuild_error_kwh': max_discharge_error,
        'interpretation': 'same observed SOC; instantaneous conditional withheld demand; not realizable annual savings'}
    np.savez_compressed(OUT/f'{name}_planned_and_floor_diagnostics.npz',
        reconstructed_planned_spill_kwh=spill, floor_kwh=floor,
        same_state_unrestricted_discharge_kwh=physical, extra_unserved_demand_due_floor_kwh=withheld,
        same_state_extra_emergency_fee_proxy_yuan=gross_fee, days=inputs['days'])
    actual_error = (detail['actual'][..., 0]-detail['actual'][..., 1])/6-inputs['forecast_net']
    actual_cumulative = np.cumsum(actual_error, axis=1)
    margin = np.cumsum(risk-inputs['forecast_net'], axis=1)
    heldout = actual_cumulative <= margin+1e-8
    history_cumulative = np.cumsum(inputs['errors'], axis=2)
    historical = history_cumulative <= margin[:, None, :]+1e-8
    daily = pd.DataFrame({'day': inputs['days'],
        'date': pd.date_range('2025-02-01', '2025-12-31').astype(str),
        'heldout_pointwise_prefix_coverage': heldout.mean(axis=1),
        'heldout_whole_path_covered': heldout.all(axis=1),
        'heldout_dayend_covered': heldout[:, -1],
        'historical_pointwise_prefix_coverage': historical.mean(axis=(1, 2)),
        'historical_whole_path_coverage': historical.all(axis=2).mean(axis=1),
        'floor_binding_slots': (withheld > 1e-6).sum(axis=1),
        'same_state_withheld_kwh': withheld.sum(axis=1), 'gross_floor_fee_proxy': gross_fee.sum(axis=1)})
    daily.to_csv(OUT/f'{name}_coverage_and_floor_daily.csv', index=False)
    coverage = {'heldout_at_issue_cumulative_pointwise_coverage': float(heldout.mean()),
        'heldout_at_issue_whole_path_coverage': float(heldout.all(axis=1).mean()),
        'heldout_at_issue_dayend_coverage': float(heldout[:, -1].mean()),
        'historical_cumulative_pointwise_coverage': float(historical.mean()),
        'historical_whole_path_coverage': float(historical.all(axis=2).mean()),
        'quantile_curve_type': 'marginal prefix quantile' if name == 'prefix' else 'cumulative sum of slotwise quantiles',
        'not_an_untouched_2025_test': True, 'no_joint_chance_guarantee': True}
    return {'independent_source_physics_billing': verified, 'planned_physics': planned,
            'floor': floor_result, 'coverage': coverage}


def main():
    if OUT.exists():
        raise FileExistsError('Supplementary evidence is immutable; use a new version if needed')
    OUT.mkdir(parents=True)
    protected = [ROOT/'experiments/exp008/cumulative_risk_planning.py']
    protected += [BASE/name for name in ('source_snapshot.py', 'protocol.json', 'risk_inputs.npz',
                                       'causality_verification.json', 'summary.json')]
    protected += [BASE/method/name for method in ('prefix', 'point')
                  for name in ('dispatch_2.npz', 'audit.json', 'summary.json', 'daily.csv')]
    hashes = {str(path.relative_to(ROOT)): digest(path) for path in protected}
    save(OUT/'protocol.json', {'supplement_only': True, 'original_artifacts_read_only_sha256': hashes,
        'why_new_test': 'original future-store +60000 on both channels cancels in net; use asymmetric +[60000,20000]',
        'future_mutation_days': [31, 32, 60, 151, 243, 364],
        'source_sha256': digest(__file__), 'new_policies_or_annual_plans_run': False})
    (OUT/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    inputs = load(BASE/'risk_inputs.npz')
    math_checks = mathematical_checks()
    maximum_risk_rebuild_error = 0.
    for i in range(334):
        f, errors = inputs['forecast_net'][i], inputs['errors'][i]
        assert np.isfinite(f).all() and np.isfinite(errors).all()
        for method in ('prefix', 'point'):
            risk, margin, increments = rebuild(f, errors, method)
            maximum_risk_rebuild_error = max(maximum_risk_rebuild_error, float(np.abs(risk-inputs[method][i]).max()))
            np.testing.assert_allclose(risk, inputs[method][i], rtol=0, atol=1e-8)
            if method == 'prefix':
                np.testing.assert_allclose(np.cumsum(risk-f), margin, rtol=0, atol=1e-8)
                assert np.all(increments >= errors.min(axis=0)-1e-8)
                assert np.all(increments <= errors.max(axis=0)+1e-8)
    math_checks.update(all334_prefix_increment_minmax_bounds=True,
        all334_telescope_identities=True, maximum_independent_risk_rebuild_error_kwh=maximum_risk_rebuild_error)
    mutation = []
    original = AbsoluteHGBForecasts()
    for day in (31, 32, 60, 151, 243, 364):
        for kind in ('future_store_only', 'future_actual_only', 'both'):
            data = copy.copy(original.data); data.actual = original.data.actual.copy()
            if kind != 'future_store_only':
                data.actual[day*144:] += np.array([70000., 30000., 1000.])
            changed = AbsoluteHGBForecasts(data)
            future = changed.store.origins > day*144
            if kind != 'future_actual_only':
                changed.store.values[future] += np.array([60000., 20000.])
            issue = changed.get(day, scenario='2'); history = changed.net_error_paths(day, '2', limit=28)
            before = original.get(day, scenario='2')
            for key in ('load_kw', 'pv_kw', 'price'):
                np.testing.assert_array_equal(issue[key], before[key])
            np.testing.assert_array_equal(history['errors_kwh'], inputs['errors'][day-31])
            f = (issue['load_kw']-issue['pv_kw'])/6
            for method in ('prefix', 'point'):
                risk, _, _ = rebuild(f, history['errors_kwh'], method)
                np.testing.assert_allclose(risk, inputs[method][day-31], rtol=0, atol=1e-8)
            mutation.append({'day': day, 'kind': kind, 'passed': True,
                'future_store_rows_mutated': int(future.sum()) if kind != 'future_actual_only' else 0,
                'future_store_net_change_kw': 40000. if kind != 'future_actual_only' and future.any() else 0.,
                'store_mutation_nonvacuous': bool(future.any()) if kind != 'future_actual_only' else None})
    methods = {name: method_audit(name, inputs) for name in ('prefix', 'point')}
    bounds = []
    for method in ('prefix', 'point'):
        bound = json.loads((ORACLE/method/'bound.json').read_text())
        assert bound['source_sha256'] == digest(BASE/method/'dispatch_2.npz')
        assert bound['oracle_uses_future_actual'] and not bound['eligible_as_causal_strategy']
        assert abs(bound['minimum_total_cost_yuan']-bound['total_dual_bound_yuan']) < 1e-5
        bounds.append({'method': method, 'source_bound_sha256': digest(ORACLE/method/'bound.json'),
            'actual_cost_yuan': bound['candidate_actual_total_cost_yuan'],
            'fixed_Q_perfect_future_minimum_cost_yuan': bound['minimum_total_cost_yuan'],
            'total_dual_bound_yuan': bound['total_dual_bound_yuan'],
            'maximum_execution_only_saving_yuan': bound['maximum_saving_by_execution_only_yuan'],
            'dual_bound_above8pct_target_yuan': bound['total_dual_bound_yuan']-TARGET,
            'fixed_Q_execution_only_8pct_ruled_out': bound['total_dual_bound_yuan'] > TARGET,
            'not_a_causal_strategy_and_no_oracle_actions_exported': True})
    pd.DataFrame(bounds).to_csv(OUT/'fixed_Q_oracle_conclusions.csv', index=False)
    assert all(digest(ROOT/name) == expected for name, expected in hashes.items())
    result = {'passed': True, 'signed_original_artifacts_unchanged': True, 'math_checks': math_checks,
        'asymmetric_future_mutation_checks': mutation,
        'day364_has_no_later_issued_store_rows': True, 'methods': methods,
        'fixed_Q_oracle_conclusions': bounds, 'new_policies_or_annual_plans_run': False}
    save(OUT/'supplementary_audit.json', result)
    print(json.dumps({'passed': True, 'methods': {name: {'floor': value['floor'], 'coverage': value['coverage']}
                     for name, value in methods.items()}, 'fixed_Q_oracle': bounds}, indent=2), flush=True)
    return result


if __name__ == '__main__':
    main()
