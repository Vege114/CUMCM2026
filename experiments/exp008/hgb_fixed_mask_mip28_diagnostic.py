"""Five fixed-day MIP28 initializations under archived HGB hourly mode masks.

No new forecasts, mode choices, strategy sweep or continuous-policy replay.
Original and new purchases are selected only by the same historical greedy
objective. Each day retains the archived policy's own initial SOC and mask.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import shutil
from time import perf_counter
import numpy as np
import pandas as pd

from experiments.common.neural_v2.physics import LinearModel
from experiments.exp008.absolute_hgb_forecast_adapter import AbsoluteHGBForecasts
from experiments.exp008.closed_loop import objective, optimize
from experiments.exp008.planner import ETA, LOW, HIGH, LIMIT, execute
from experiments.exp008.verify import verify_arrays
from experiments.problem2.exp003.data import Data, ROOT

OUT = ROOT / 'data/results/exp008/hgb_fixed_mask_mip28_diagnostic'
SOURCE = ROOT / 'data/results/exp008/mode_planning_physical/direct_hgb_memory_physical3_334days'
DAYS = (31, 90, 151, 243, 364)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_hash(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def save(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def physical_check(q, paths, mask, states, charge, discharge, emergency, surplus):
    balance = q[None] + discharge + emergency - charge - surplus - paths
    transition = np.diff(states, axis=1) - ETA * charge + discharge / ETA
    arrays = [q, charge, discharge, emergency, surplus]
    result = {'balance_error_kwh': float(np.abs(balance).max()),
        'soc_equation_error_kwh': float(np.abs(transition).max()),
        'simultaneous_charge_discharge_slots': int(np.sum((charge > 1e-6) & (discharge > 1e-6))),
        'emergency_charge_slots': int(np.sum((charge > 1e-6) & (emergency > 1e-6))),
        'discharge_into_surplus_slots': int(np.sum((discharge > 1e-6) & (surplus > 1e-6))),
        'charge_in_discharge_mode_slots': int(np.sum(charge[:, ~mask] > 1e-6)),
        'discharge_in_charge_mode_slots': int(np.sum(discharge[:, mask] > 1e-6)),
        'minimum_energy': float(min(np.min(value) for value in arrays)),
        'minimum_soc': float(states.min()), 'maximum_soc': float(states.max()),
        'maximum_charge_discharge_kwh': float(max(charge.max(), discharge.max()))}
    result['passed'] = bool(max(result['balance_error_kwh'], result['soc_equation_error_kwh']) < 1e-6
        and all(result[key] == 0 for key in ['simultaneous_charge_discharge_slots', 'emergency_charge_slots',
            'discharge_into_surplus_slots', 'charge_in_discharge_mode_slots', 'discharge_in_charge_mode_slots'])
        and result['minimum_energy'] >= -1e-6 and states.min() >= LOW - 1e-6 and states.max() <= HIGH + 1e-6
        and result['maximum_charge_discharge_kwh'] <= LIMIT + 1e-6)
    return result


def fixed_mask_mip(paths, price, initial_soc, mask, previous_mode, final):
    paths, price, mask = np.asarray(paths), np.asarray(price), np.asarray(mask, bool)
    k, n = paths.shape
    assert (k, n) == (28, 144) and mask.shape == (144,)
    assert all(np.all(mask[start:start + 6] == mask[start]) for start in range(0, 144, 6))
    m = LinearModel()
    q = m.variables((n,), upper=np.maximum(paths.max(axis=0), 0) + LIMIT, cost=price)
    c = m.variables((k, n), upper=LIMIT * mask[None], cost=.002 / k)
    d = m.variables((k, n), upper=LIMIT * (~mask)[None], cost=.002 / k)
    e_upper = np.maximum(paths, 0)
    e = m.variables((k, n), upper=e_upper, cost=5 * price[None] / k)
    w = m.variables((k, n))
    s = m.variables((k, n), lower=LOW, upper=HIGH)
    # No mode variables remain. This binary only separates surplus charging
    # from emergency purchases on scenario slots where charging is permitted.
    active_t = np.flatnonzero(mask)
    charging_branch = m.variables((k, len(active_t)), upper=1., integer=True)
    branch_lookup = {int(t): b for b, t in enumerate(active_t)}
    terminal = 0. if final else float(price.min()) / ETA
    for index in s[:, -1]:
        m.objective[int(index)] = -terminal / k
    for j in range(k):
        for t in range(n):
            m.constraint([(q[t], 1), (c[j, t], -1), (d[j, t], 1), (e[j, t], 1), (w[j, t], -1)],
                         paths[j, t], paths[j, t])
            terms = [(s[j, t], 1), (c[j, t], -ETA), (d[j, t], 1 / ETA)]
            if t:
                terms.append((s[j, t - 1], -1))
            rhs = initial_soc if t == 0 else 0.
            m.constraint(terms, rhs, rhs)
            if mask[t]:
                z = charging_branch[j, branch_lookup[t]]
                m.constraint([(c[j, t], 1), (z, -LIMIT)], upper=0.)
                m.constraint([(e[j, t], 1), (z, e_upper[j, t])], upper=e_upper[j, t])
    values, metadata = m.solve(seconds=30., gap=.002)
    mode_blocks = mask[::6].astype(int)
    flips = int(np.count_nonzero(np.diff(np.r_[int(previous_mode > 0), mode_blocks])))
    switch_constant = 50. * flips
    metadata.update(mode_mask_fixed=True, mode_decision_variables=0, scenario_count=28,
        terminal_coefficient=terminal, fixed_switch_count=flips, fixed_switch_cost=switch_constant,
        seconds_limit=30., relative_gap_target=.002, warm_start_used=False,
        warm_start_omitted_reason='existing LinearModel.solve and scipy.milp expose no incumbent parameter',
        recourse_knows_own_scenario_future=True, nonanticipativity_certificate=False)
    if values is None:
        return None, metadata
    q_value = np.maximum(values[q], 0.)
    states = np.column_stack((np.full(k, initial_soc), values[s]))
    checks = physical_check(q_value, paths, mask, states, values[c], values[d], values[e], values[w])
    if not checks['passed']:
        raise AssertionError(checks)
    planned_cost = float(price @ q_value)
    emergency_cost = float(np.mean(np.sum(5 * price * values[e], axis=1)))
    wear = float(.002 * np.mean(np.sum(values[c] + values[d], axis=1)))
    terminal_credit = float(terminal * states[:, -1].mean())
    primal = planned_cost + emergency_cost + wear - terminal_credit
    direct_primal = float(np.dot(np.asarray(m.objective), values))
    assert abs(primal - direct_primal) < 1e-6
    dual = metadata['dual_bound']
    metadata.update(physical_checks=checks, planned_cost=planned_cost,
        expected_emergency_cost=emergency_cost, expected_wear=wear, terminal_credit=terminal_credit,
        primal_without_switch_constant=primal, primal_with_switch_constant=primal + switch_constant,
        dual_with_switch_constant=None if dual is None else dual + switch_constant,
        primal_dual_absolute_gap=None if dual is None else primal - dual,
        final_soc_mean=float(states[:, -1].mean()), final_soc_min=float(states[:, -1].min()),
        final_soc_max=float(states[:, -1].max()))
    return {'purchase': q_value, 'allowed_charge': mask.copy(), 'scenario_charge': values[c],
        'scenario_discharge': values[d], 'scenario_emergency': values[e], 'scenario_surplus': values[w],
        'scenario_states': states, 'charging_branch': values[charging_branch]}, metadata


def greedy_independent(q, paths, price, initial_soc, mask, terminal):
    """Independent scalar per-path execution and direct objective recomputation."""
    k, n = paths.shape
    c, d, e, w = [np.zeros((k, n)) for _ in range(4)]
    s = np.full((k, n + 1), initial_soc)
    for j in range(k):
        for t in range(n):
            balance = q[t] - paths[j, t]
            if balance >= 0 and mask[t]:
                c[j, t] = min(balance, LIMIT, max(0., (HIGH - s[j, t]) / ETA))
            if balance < 0 and not mask[t]:
                d[j, t] = min(-balance, LIMIT, max(0., (s[j, t] - LOW) * ETA))
            e[j, t] = max(0., -balance - d[j, t])
            w[j, t] = max(0., balance - c[j, t])
            s[j, t + 1] = s[j, t] + ETA * c[j, t] - d[j, t] / ETA
    value = float(price @ q + np.mean(np.sum(5 * price * e + .002 * (c + d), axis=1))
                  - terminal * np.mean(s[:, -1] - LOW))
    expected, _ = objective(q, paths, price, initial_soc, charge_mask=mask,
                           throughput=.002, variation=0., terminal=terminal, deadband=0.)
    assert abs(value - expected) < 1e-7, (value, expected)
    check = physical_check(q, paths, mask, s, c, d, e, w)
    assert check['passed'], check
    return {'objective': value, 'planned_cost': float(price @ q),
        'expected_emergency_cost': float(np.mean(np.sum(5 * price * e, axis=1))),
        'expected_throughput_penalty': float(.002 * np.mean(np.sum(c + d, axis=1))),
        'terminal_credit_from_LOW': float(terminal * np.mean(s[:, -1] - LOW)),
        'final_soc_mean': float(s[:, -1].mean()), 'physical_checks': check}, {
        'charge': c, 'discharge': d, 'emergency': e, 'surplus': w, 'states': s}


def replay_one(q, actual, price, initial_soc, mask, day, initial_mode):
    detail = execute(q, actual, price, initial_soc, charge_deadband=0., charge_mask=mask)
    detail.update(original=q.copy(), final=q.copy(), actual=actual.copy(), price=price.copy(), allowed_charge=mask.copy())
    detail['fees'] = np.stack((q * price, np.zeros(144), np.zeros(144), 5 * detail['emergency'] * price), axis=-1)
    arrays = {key: value[None] for key, value in detail.items()}
    arrays['days'] = np.array([day])
    check = verify_arrays(arrays, expected_days=1, start_day=day, initial_soc=initial_soc,
                          initial_mode=initial_mode, source_actual=actual[None], source_price=price)
    assert check['passed'], check['errors']
    return detail, check


def protocol():
    if OUT.exists():
        raise FileExistsError('Five-day diagnostic may not overwrite or silently rerun')
    OUT.mkdir(parents=True)
    inputs = [SOURCE / 'dispatch.npz', SOURCE / 'planning_audit.json', SOURCE / 'complete_provenance.json']
    inputs += [SOURCE / f'planning_day{day}.npz' for day in DAYS]
    source_files = ['experiments/exp008/hgb_fixed_mask_mip28_diagnostic.py',
        'experiments/common/neural_v2/physics.py', 'experiments/exp008/mode_planning_physical.py',
        'experiments/exp008/closed_loop.py', 'experiments/exp008/planner.py', 'experiments/exp008/verify.py']
    hashes = {}
    for source in source_files:
        path = ROOT / source
        destination = OUT / 'source_archive' / source
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        hashes[source] = digest(path)
    cfg = {'days_preselected': list(DAYS), 'candidate_count': 1,
        'source': str(SOURCE), 'all_case_SOC_mask_paths_from_same_original_archive': True,
        'frozen_input_fields': ['initial_soc', 'initial_mode', 'allowed_charge', 'all_net_paths', 'price'],
        'scenario_count': 28, 'mode_variables_removed': True, 'fixed_mode_switch_cost_constant': True,
        'emergency_charging_forbidden_with_scenario_slot_binary': True,
        'mip_seconds': 30., 'mip_relative_gap': .002, 'warm_start': False,
        'mip_terminal': '0 final day else min(price)/ETA times mean final SOC (unchanged)',
        'greedy_refinement': {'maxiter': 120, 'wear': .002, 'variation': 0., 'deadband': 0.,
                              'terminal': '0 final day else .45 times mean(final_SOC-LOW)'},
        'original_refinement': 'reproduce from archived original 3-scenario MIP purchase, same mask and all28 paths',
        'selection': 'lower independently recomputed same historical greedy objective of original/refined_new; ties original',
        'selection_commit_before_actual_read': True,
        'no_incumbent_policy': 'record failure and stop remaining cases; never extend30seconds',
        'not_continuous_strategy': True, 'actual_fees_only_diagnostic_not_selection': True,
        'recourse_nonanticipativity_not_certified': True,
        'input_sha256': {str(path): digest(path) for path in inputs}, 'source_sha256': hashes}
    save(OUT / 'protocol.json', cfg)
    return cfg


def run():
    cfg = protocol()
    forecast = AbsoluteHGBForecasts()
    data = Data()
    rows = []
    for day in DAYS:
        directory = OUT / f'day_{day}'
        directory.mkdir()
        with np.load(SOURCE / f'planning_day{day}.npz') as z:
            paths, mask = z['all_net_paths'].copy(), z['allowed_charge'].astype(bool)
            old_initial, soc, mode, price = z['purchase'].copy(), float(z['initial_soc']), int(z['initial_mode']), z['price'].copy()
        with np.load(SOURCE / 'dispatch.npz') as z:
            old_q = z['original'][day - 31].copy()
            assert float(z['states'][day - 31, 0]) == soc
            np.testing.assert_array_equal(mask, z['allowed_charge'][day - 31])
        issue, history = forecast.get(day, scenario='2'), forecast.net_error_paths(day)
        expected = (issue['load_kw'] - issue['pv_kw'])[None] / 6 + history['errors_kwh']
        np.testing.assert_array_equal(paths, expected)
        assert history['audit']['max_observed_index'] < day * 144
        terminal = 0. if day == 364 else .45
        args = dict(charge_mask=mask, throughput=.002, variation=0., terminal=terminal, maxiter=120, deadband=0.)
        print('RECONSTRUCT_ORIGINAL', day, flush=True)
        original = optimize(old_initial, paths, price, soc, **args)
        np.testing.assert_array_equal(original['purchase'], old_q)
        old_history, old_history_arrays = greedy_independent(old_q, paths, price, soc, mask, terminal)
        np.savez_compressed(directory / 'original_historical_greedy.npz', purchase=old_q, **old_history_arrays)
        save(directory / 'original_reconstruction.json', {
            'purchase_exact_archive_match': True, 'refinement': original['metadata'],
            'historical_objective': old_history, 'same_model_paths_exact': True,
            'history_information_audit': history['audit']})
        print('SOLVE_MIP28', day, flush=True)
        new, metadata = fixed_mask_mip(paths, price, soc, mask, mode, day == 364)
        save(directory / 'mip_metadata.json', metadata)
        if new is None:
            save(OUT / 'summary.json', {'completed_cases': rows, 'failed_day': day,
                'no_feasible_incumbent_within30seconds': True, 'stopped_remaining_cases': True,
                'no_time_limit_extension': True, 'worth_full_trial': False})
            print('STOP_NO_INCUMBENT', day, flush=True)
            return
        np.savez_compressed(directory / 'mip28_historical_scenarios.npz', **new,
                            all_net_paths=paths, initial_soc=np.array(soc), price=price)
        refined = optimize(new['purchase'], paths, price, soc, **args)
        new_history, new_history_arrays = greedy_independent(refined['purchase'], paths, price, soc, mask, terminal)
        raw_new_history, _ = greedy_independent(new['purchase'], paths, price, soc, mask, terminal)
        np.savez_compressed(directory / 'new_historical_greedy.npz', purchase=refined['purchase'], **new_history_arrays)
        selected_new = new_history['objective'] < old_history['objective']
        selected_q = refined['purchase'] if selected_new else old_q
        selected_name = 'MIP28_then_greedy120' if selected_new else 'original_MIP3_then_greedy120'
        # Persist the binding historical-only selection before actual replay.
        selection = {'day': day, 'selected': selected_name, 'actual_not_used_to_select': True,
            'information_cutoff_exclusive': day * 144, 'initial_soc': soc, 'initial_mode': mode,
            'mode_mask_sha256': array_hash(mask), 'historical_paths_sha256': array_hash(paths),
            'selected_purchase_sha256': array_hash(selected_q),
            'original_history': old_history, 'MIP28_raw_history': raw_new_history,
            'new_refined_history': new_history, 'new_refinement_metadata': refined['metadata']}
        save(directory / 'selection_locked_before_actual.json', selection)
        np.savez_compressed(directory / 'selected_purchase_before_actual.npz', purchase=selected_q)
        # No current actual data are accessed until selection has been locked.
        actual = data.actual[day * 144:(day + 1) * 144].copy()
        old_detail, old_check = replay_one(old_q, actual, price, soc, mask, day, mode)
        selected_detail, selected_check = replay_one(selected_q, actual, price, soc, mask, day, mode)
        with np.load(SOURCE / 'dispatch.npz') as z:
            exact_checks = {key: bool(np.array_equal(value, z[key][day - 31])) for key, value in old_detail.items()}
        assert all(exact_checks.values()), exact_checks
        save(directory / 'actual_independent_verification.json', {'original': old_check, 'selected': selected_check,
            'original_all_replayed_arrays_exact_archive_match': exact_checks,
            'selection_file_sha256': digest(directory / 'selection_locked_before_actual.json')})
        # These are explicitly isolated day diagnostics, not final dispatch index entries.
        np.savez_compressed(directory / 'isolated_selected_replay.npz', **selected_detail)
        row = {'day': day, 'selected': selected_name, 'historical_objective_original': old_history['objective'],
            'historical_objective_MIP28_raw': raw_new_history['objective'],
            'historical_objective_MIP28_refined': new_history['objective'],
            'historical_objective_improvement': old_history['objective'] - min(old_history['objective'], new_history['objective']),
            'MIP_seconds': metadata['seconds'], 'MIP_gap': metadata['mip_gap'],
            'actual_fee_original': old_check['recomputed_total_cost'],
            'actual_fee_selected': selected_check['recomputed_total_cost'],
            'actual_fee_change': selected_check['recomputed_total_cost'] - old_check['recomputed_total_cost'],
            'actual_final_SOC_original': old_check['battery_metrics']['final_soc'],
            'actual_final_SOC_selected': selected_check['battery_metrics']['final_soc'],
            'actual_direction_reversals_original': old_check['battery_metrics']['direction_reversals'],
            'actual_direction_reversals_selected': selected_check['battery_metrics']['direction_reversals'],
            'initial_soc': soc, 'independent_physics_passed': True}
        rows.append(row)
        save(directory / 'summary.json', row)
        print('COMPLETE', json.dumps(row), flush=True)
    unchanged = all(digest(path) == expected for path, expected in cfg['input_sha256'].items())
    assert unchanged and all(digest(ROOT / path) == expected for path, expected in cfg['source_sha256'].items())
    improvement = sum(row['historical_objective_improvement'] for row in rows)
    summary = {'cases': rows, 'all_five_cases_complete': len(rows) == 5,
        'selected_new_cases': sum(row['selected'] == 'MIP28_then_greedy120' for row in rows),
        'sum_isolated_historical_objective_improvement': improvement,
        'isolated_actual_fee_change_sum': sum(row['actual_fee_change'] for row in rows),
        'isolated_actual_fee_sum_is_not_continuous_policy_performance': True,
        'sources_inputs_and_original_archive_unchanged': unchanged,
        'all_current_truth_access_after_historical_selection_commit': True,
        'all_same_forecast_histories_and_original_refine_reconstructed_exactly': True,
        'all_scenario_and_greedy_physics_passed': True,
        'worth_full_trial_on_historical_objective': bool(improvement > 0),
        'full_trial_not_run_or_authorized_by_this_diagnostic': True}
    save(OUT / 'summary.json', summary)
    pd.DataFrame(rows).to_csv(OUT / 'five_isolated_days.csv', index=False)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    run()
