"""Five fixed conditional days: align existing mode boundaries to all 28 paths."""

from __future__ import annotations

import hashlib
import json
import shutil
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.exp008.closed_loop import ETA, HIGH, LIMIT, LOW, objective, optimize
from experiments.exp008.mode_budget_physical import mode_state
from experiments.exp008.verify import battery_metrics

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'data/results/exp008/mode_budget_physical/direct_hgb_memory_hold3_cap8_kappa0_334days'
OUT = ROOT/'data/results/exp008/mode_boundary_alignment/fixed5_hold3_cap8_full28'
DAYS = (31, 90, 151, 243, 364)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))


def load(path):
    with np.load(path) as values:
        return {key: values[key] for key in values.files}


def build_mask(boundaries, previous_mode, n=144):
    mask = np.full(n, previous_mode > 0, bool)
    for boundary in boundaries:
        mask[boundary:] = ~mask[boundary:]
    return mask


def history_replay(q, paths, price, initial_soc, mask, terminal):
    """Exact strict-mask physical replay, independently of objective derivatives."""
    k, n = paths.shape
    flows = {key: np.zeros((k, n)) for key in ('charge', 'discharge', 'emergency', 'surplus')}
    states = np.full((k, n+1), initial_soc)
    for t in range(n):
        balance = q[t]-paths[:, t]
        soc = states[:, t]
        if mask[t]:
            flows['charge'][:, t] = np.minimum.reduce((np.maximum(balance, 0),
                np.full(k, LIMIT), np.maximum(0, (HIGH-soc)/ETA)))
        else:
            flows['discharge'][:, t] = np.minimum.reduce((np.maximum(-balance, 0),
                np.full(k, LIMIT), np.maximum(0, (soc-LOW)*ETA)))
        c, d = flows['charge'][:, t], flows['discharge'][:, t]
        flows['emergency'][:, t] = np.maximum(0, -balance-d)
        flows['surplus'][:, t] = np.maximum(0, balance-c)
        states[:, t+1] = soc+ETA*c-d/ETA
    c, d, e, w = [flows[key] for key in ('charge', 'discharge', 'emergency', 'surplus')]
    error = float(np.abs(q[None, :]+d+e-c-w-paths).max())
    assert error < 1e-6 and states.min() >= LOW-1e-6 and states.max() <= HIGH+1e-6
    assert max(c.max(), d.max()) <= LIMIT+1e-6
    assert not np.any((c > 1e-6) & ((d > 1e-6) | (e > 1e-6)))
    expected = float(q@price+np.mean(np.sum(5*e*price+.002*(c+d), axis=1))
                     -terminal*np.mean(states[:, -1]-LOW))
    analytic = objective(q, paths, price, initial_soc, charge_mask=mask, throughput=.002,
                         variation=0., terminal=terminal, deadband=0.)[0]
    np.testing.assert_allclose(expected, analytic, atol=1e-6, rtol=0.)
    flows['states'] = states
    return flows, {'objective': expected, 'physical_passed': True,
        'maximum_balance_error_kwh': error, 'expected_final_soc': float(states[:, -1].mean()),
        'expected_emergency_cost': float(np.mean(np.sum(5*e*price, axis=1))),
        'expected_throughput_kwh': float(np.mean(np.sum(c+d, axis=1)))}


def select(day, archived_objective):
    """Consumes frozen planning inputs and Q only; no current actual argument."""
    p = load(BASE/f'planning_day{day}.npz')
    with np.load(BASE/'dispatch.npz') as old:
        archived_q = old['original'][day-31].copy()
        np.testing.assert_array_equal(old['allowed_charge'][day-31], p['allowed_charge'])
    paths, price, soc = p['all_net_paths'], p['price'], float(p['initial_soc'])
    previous, age = int(p['previous_planned_mode']), int(p['previous_planned_run_slots'])
    mask = p['allowed_charge'].copy()
    terminal = 0. if day == 364 else .45
    args = {'throughput': .002, 'variation': 0., 'terminal': terminal, 'deadband': 0.}
    reproduced = optimize(p['purchase'], paths, price, soc, charge_mask=mask, maxiter=120, **args)
    np.testing.assert_array_equal(reproduced['purchase'], archived_q)
    np.testing.assert_allclose(reproduced['metadata']['objective'], archived_objective, atol=1e-6, rtol=0.)
    q = archived_q.copy()
    boundaries = np.flatnonzero(np.diff(np.r_[previous > 0, mask])).tolist()
    initial_boundary_count = len(boundaries)
    initial_flows, initial = history_replay(q, paths, price, soc, mask, terminal)
    np.savez_compressed(OUT/f'day{day}_step0.npz', q=q, mask=mask, boundaries=boundaries,
        paths=paths, price=price, initial_soc=soc, previous_planned_mode=previous,
        previous_planned_run_slots=age, **initial_flows)
    iterations = []
    for iteration in range(1, 4):
        old_value = objective(q, paths, price, soc, charge_mask=mask, **args)[0]
        candidates, best = [], None
        for boundary_index, original_position in enumerate(boundaries):
            for displacement in (-3, -2, -1, 1, 2, 3):
                proposed = boundaries.copy()
                proposed[boundary_index] += displacement
                record = {'boundary_index': boundary_index, 'from': original_position,
                    'to': proposed[boundary_index], 'displacement': displacement,
                    'boundaries': proposed}
                if proposed[0] < 0 or proposed[-1] >= 144 or any(b <= a for a, b in pairwise(proposed)):
                    record.update(feasible=False, reason='boundary outside day or collision/crossing')
                else:
                    candidate_mask = build_mask(proposed, previous)
                    try:
                        state, flips = mode_state(candidate_mask, previous, age, 3)
                        assert int(flips.sum()) == initial_boundary_count <= 8
                        assert np.all(candidate_mask[:max(0, 3-age)] == (previous > 0))
                    except AssertionError:
                        record.update(feasible=False, reason='hold3, inherited lock or fixed flip count')
                    else:
                        value = float(objective(q, paths, price, soc, charge_mask=candidate_mask, **args)[0])
                        record.update(feasible=True, objective=value, history_only=True,
                                      next_day_required_hold_slots=state['next_day_required_hold_slots'])
                        if value < old_value-1e-6 and (best is None or value < best['objective']-1e-6):
                            best = dict(record)
                candidates.append(record)
        record = {'iteration': iteration, 'q_source': f'day{day}_step{iteration-1}.npz',
            'old_objective': float(old_value), 'all_candidates': candidates, 'accepted': best}
        if best is None:
            record['stop_reason'] = 'No feasible existing-boundary move strictly improves full28 objective'
            iterations.append(record)
            break
        boundaries = best['boundaries']
        mask = build_mask(boundaries, previous)
        proposed_q = optimize(q, paths, price, soc, charge_mask=mask, maxiter=120, **args)
        refined_value = float(objective(proposed_q['purchase'], paths, price, soc, charge_mask=mask, **args)[0])
        accepted_q = refined_value <= best['objective']
        record.update(refinement=proposed_q['metadata'], refinement_recomputed_objective=refined_value,
                      refined_q_accepted=accepted_q, preceding_q_preserved_if_objective_worse=True)
        if accepted_q:
            q = proposed_q['purchase']
        flows, replay = history_replay(q, paths, price, soc, mask, terminal)
        assert replay['objective'] < old_value-1e-6
        record['post_move_and_refinement'] = replay
        np.savez_compressed(OUT/f'day{day}_step{iteration}.npz', q=q, mask=mask,
            boundaries=boundaries, paths=paths, price=price, initial_soc=soc,
            previous_planned_mode=previous, previous_planned_run_slots=age,
            proposed_refinement_q=proposed_q['purchase'], **flows)
        iterations.append(record)
    final_flows, final = history_replay(q, paths, price, soc, mask, terminal)
    final_state, final_flips = mode_state(mask, previous, age, 3)
    assert final_flips.sum() == initial_boundary_count <= 8
    np.savez_compressed(OUT/f'day{day}_locked_final.npz', q=q, mask=mask,
        boundaries=boundaries, paths=paths, price=price, initial_soc=soc,
        previous_planned_mode=previous, previous_planned_run_slots=age, **final_flows)
    result = {'day': day, 'raw_MIP_not_rerun': True, 'original_refine120_Q_reproduced_exactly': True,
        'original_objective': initial, 'final_objective': final, 'iterations': iterations,
        'accepted_moves': sum(item['accepted'] is not None for item in iterations),
        'fixed_planned_flip_count': initial_boundary_count, 'final_planned_state': final_state,
        'final_q_sha256': hashlib.sha256(q.tobytes()).hexdigest(),
        'current_actual_loaded_during_selection': False}
    save(OUT/f'day{day}_selection.json', result)
    print('SELECTED', day, initial['objective'], final['objective'], result['accepted_moves'], flush=True)
    return result


def run():
    if OUT.exists():
        raise FileExistsError('Never overwrite a completed or partial fixed diagnostic')
    OUT.mkdir(parents=True)
    save(OUT/'protocol.json', {'days': DAYS, 'conditional_fixed_original_year_SOC_mode_age': True,
        'continuous_year_evaluation': False, 'candidate_parameters_changed': False,
        'full_pipeline': 'direct_HGB_Ridge28_memory_half', 'MIP_or_forecast_not_repeated': True,
        'history_paths': 28, 'maximum_accepted_moves': 3,
        'boundary_displacements': [-3, -2, -1, 1, 2, 3], 'boundary_count_never_changes': True,
        'midnight_boundary_included_and_can_shift_within_day': True,
        'acceptance': 'best feasible strict full28 greedy objective decrease exceeding numeric 1e-6',
        'mode_hold': 3, 'daily_flip_cap': 8, 'previous_planned_mode_age_from_original_year': True,
        'greedy_refinement_maxiter': 120, 'wear': .002, 'variation': 0., 'deadband': 0.,
        'terminal': '.45 except final evaluated day364 zero',
        'refinement_worse_than_previous_Q_rejected': True,
        'all_five_selections_locked_before_current_actual_file_loaded': True,
        'actual_cost_not_a_mode_or_purchase_selection_criterion': True,
        'development_year_not_independent_test': True})
    sources = [Path(__file__), ROOT/'experiments/exp008/closed_loop.py',
               ROOT/'experiments/exp008/mode_budget_physical.py', ROOT/'experiments/exp008/verify.py']
    hashes = {}
    for source in sources:
        destination = OUT/'source_archive'/source.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        hashes[str(source.relative_to(ROOT))] = sha(destination)
    audits = json.loads((BASE/'planning_audit.json').read_text())
    selection = [select(day, audits[day-31]['greedy_refinement']['objective']) for day in DAYS]
    locked = {str(day): sha(OUT/f'day{day}_locked_final.npz') for day in DAYS}
    save(OUT/'all_selections_locked_before_actual.json', {'locked_sha256': locked,
        'source_sha256': hashes, 'current_actual_not_loaded': True,
        'base_dispatch_sha256': sha(BASE/'dispatch.npz'),
        'base_planning_inputs_sha256': {str(day): sha(BASE/f'planning_day{day}.npz') for day in DAYS}})

    # Only after every choice is locked, obtain the actual rows for evaluation.
    actual = np.stack([pd.read_csv(ROOT/'data/raw'/name).iloc[:, 1:].to_numpy(float)
        for name in ('附件2_小区负载.csv', '附件2_光伏发电实际功率.csv')], axis=-1)
    base = load(BASE/'dispatch.npz')
    all_actual_power = 6*(base['charge']-base['discharge']).ravel()
    rows = []
    for day in DAYS:
        assert sha(OUT/f'day{day}_locked_final.npz') == locked[str(day)]
        prior_power = all_actual_power[:(day-31)*144]
        active_prior = prior_power[np.abs(prior_power) > 1e-6]
        initial_mode = int(np.sign(active_prior[-1])) if len(active_prior) else 1
        initial_power = float(prior_power[-1]) if len(prior_power) else 172.75999999999976
        for name, path in [('original', OUT/f'day{day}_step0.npz'),
                           ('aligned', OUT/f'day{day}_locked_final.npz')]:
            p = load(path)
            net = (actual[day, :, 0]-actual[day, :, 1])[None, :]/6
            flows, _replay = history_replay(p['q'], net, p['price'], float(p['initial_soc']), p['mask'],
                                          0. if day == 364 else .45)
            if name == 'original':
                for key, value in flows.items():
                    np.testing.assert_array_equal(value[0], base[key][day-31])
            flows['original'] = p['q'][None, :]
            flows['price'] = p['price'][None, :]
            flows['fees'] = np.stack((flows['original']*flows['price'], np.zeros((1, 144)),
                np.zeros((1, 144)), 5*flows['emergency']*flows['price']), axis=-1)
            flows['allowed_charge'] = p['mask'][None, :]
            np.savez_compressed(OUT/f'day{day}_{name}_actual_replay.npz', **flows)
            rows.append({'day': day, 'date': str((pd.Timestamp('2025-01-01')+pd.Timedelta(days=day)).date()),
                'name': name, 'total_cost': float(flows['fees'].sum()),
                'planned_cost': float(flows['fees'][:, :, 0].sum()),
                'emergency_cost': float(flows['fees'][:, :, 3].sum()),
                'history_objective': float(next(item for item in selection if item['day'] == day)
                    ['original_objective' if name == 'original' else 'final_objective']['objective']),
                **battery_metrics(flows, initial_mode=initial_mode, initial_power_kw=initial_power)})
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT/'conditional_daily_comparison.csv', index=False)
    paired = frame.pivot(index='day', columns='name')
    findings = {'conditional_days': DAYS, 'continuous_year_result': False,
        'all_original_refinement_Q_arrays_reproduced_exactly': True,
        'all_original_actual_flows_reproduced_exactly': True,
        'all_history_objectives_independently_replayed': True,
        'all_actual_reads_after_five_selections_locked': True,
        'original_total_cost': float(frame[frame.name == 'original'].total_cost.sum()),
        'aligned_total_cost': float(frame[frame.name == 'aligned'].total_cost.sum()),
        'history_objective_improvement': float(sum(item['original_objective']['objective']-item['final_objective']['objective'] for item in selection)),
        'accepted_moves': {str(item['day']): item['accepted_moves'] for item in selection},
        'per_day_final_SOC_delta_kwh': {str(day): float(paired.loc[day, ('final_soc', 'aligned')]-paired.loc[day, ('final_soc', 'original')]) for day in DAYS},
        'state_boundary_limit': 'Each day starts from original annual SOC and actual/plan mode; differences in final SOC are not propagated across these separated diagnostic days.',
        'no_full_year_goal_claim': True}
    save(OUT/'findings.json', findings)
    print(json.dumps(findings, indent=2), flush=True)


if __name__ == '__main__':
    run()
