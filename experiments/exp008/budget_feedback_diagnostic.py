"""Predeclared fixed-five-day Q/policy diagnostic; no annual extrapolation."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.exp008 import budget_feedback as policy
from experiments.exp008.full28_mode_diagnostic_audit import sha
from experiments.exp008.verify import battery_metrics

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'data/results/exp008/mode_budget_hold1_physical/direct_hgb_memory_hold1_cap8_kappa0_334days'
OUT = ROOT / 'data/results/exp008/budget_feedback/fixed5_hgb_cap8_grid200_blocks8_step50'
DAYS = (31, 90, 151, 243, 364)


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def load(path):
    with np.load(path) as z:
        return {k: z[k].copy() for k in z.files}


def check_flow(q, paths, price, flow, initial_soc, initial_mode):
    c, d, e, w, s = [flow[key] for key in ('charge', 'discharge', 'emergency', 'surplus', 'states')]
    b = q[None, :] - paths
    error = max(float(np.abs(b + d + e - c - w).max()),
        float(np.abs(np.diff(s, axis=1) - policy.ETA * c + d / policy.ETA).max()))
    assert error < 1e-6
    assert min(c.min(), d.min(), e.min(), w.min()) >= -1e-6
    assert max(c.max(), d.max()) <= policy.LIMIT + 1e-6
    assert s.min() >= policy.LOW - 1e-6 and s.max() <= policy.HIGH + 1e-6
    assert np.all(s[:, 0] == initial_soc)
    assert not np.any((c > 1e-6) & ((d > 1e-6) | (e > 1e-6)))
    assert np.all(c <= np.maximum(b, 0.) + 1e-6)
    assert np.all(d <= np.maximum(-b, 0.) + 1e-6)
    np.testing.assert_allclose(flow['fees'][:, :, 0], np.broadcast_to(q * price, c.shape), atol=0., rtol=0.)
    np.testing.assert_allclose(flow['fees'][:, :, 3], 5 * e * price, atol=0., rtol=0.)
    assert np.all(flow['fees'][:, :, 1:3] == 0.)
    if 'remaining_budget' in flow:
        previous = np.full(len(paths), initial_mode, dtype=int)
        used = np.zeros(len(paths), dtype=int)
        assert np.all(flow['remaining_budget'][:, 0] == 8)
        assert np.all(flow['modes'][:, 0] == initial_mode)
        for t in range(len(q)):
            sign = np.sign(c[:, t] - d[:, t]).astype(int)
            active = c[:, t] + d[:, t] > 1e-8
            used += active & (sign != previous)
            previous = np.where(active, sign, previous)
            np.testing.assert_array_equal(previous, flow['modes'][:, t + 1])
            np.testing.assert_array_equal(8 - used, flow['remaining_budget'][:, t + 1])
        assert np.all(used <= 8)
    return error


def prefix_checks(q, p, grid, values, model, baseline):
    checks = []
    for stop in (1, 36, 108):
        mutated = p['all_net_paths'].copy()
        mutated[:, stop:] += np.linspace(1e4, 7e4, 144 - stop)[None, :]
        changed = policy.execute_paths(q, mutated, p['price'], float(p['initial_soc']), grid, values, model,
            initial_mode=int(p['initial_real_mode']))
        for key in ('charge', 'discharge', 'emergency', 'surplus', 'observed_error_bins'):
            np.testing.assert_array_equal(baseline[key][:, :stop], changed[key][:, :stop])
        for key in ('states', 'modes', 'remaining_budget'):
            np.testing.assert_array_equal(baseline[key][:, :stop + 1], changed[key][:, :stop + 1])
        checks.append(dict(cutoff=stop, changed_paths=28, passed=True))
    return checks


def evaluate(q, p, model, terminal):
    grid, values, meta = policy.value_functions(q, model, p['price'], grid_kwh=200., wear=.002, terminal=terminal, budget=8)
    flow = policy.execute_paths(q, p['all_net_paths'], p['price'], float(p['initial_soc']), grid, values, model,
        initial_mode=int(p['initial_real_mode']))
    physical_error = check_flow(q, p['all_net_paths'], p['price'], flow, float(p['initial_soc']), int(p['initial_real_mode']))
    objective = policy.historical_objective(flow, wear=.002, terminal=terminal)
    expected = float(q @ p['price'] + policy.initial_value(grid, values, model,
        float(p['initial_soc']), mode=int(p['initial_real_mode']), remaining=8))
    report = dict(historical_policy_objective=objective, DP_initial_expected_objective=expected,
        history_minus_DP_objective=objective-expected,
        expected_historical_cash_fee=float(np.mean(flow['fees'].sum(axis=(1, 2)))),
        expected_historical_emergency_fee=float(np.mean(flow['fees'][:, :, 3].sum(axis=1))),
        expected_historical_final_soc=float(np.mean(flow['states'][:, -1])),
        expected_historical_throughput=float(np.mean((flow['charge'] + flow['discharge']).sum(axis=1))),
        maximum_historical_used_budget=int(np.max(8 - flow['remaining_budget'][:, -1])),
        physical_error_kwh=physical_error, DP_metadata=meta)
    return dict(q=q.copy(), grid=grid, values=values, flow=flow, report=report)


def run():
    protocol = json.loads((OUT / 'protocol.json').read_text())
    assert protocol['days'] == list(DAYS) and protocol['maximum_Q_evaluations_per_day'] == 17
    if (OUT / 'source_manifest.json').exists():
        raise FileExistsError('Never overwrite a begun or completed budget-feedback diagnostic')
    inputs = {str(path.relative_to(BASE)): sha(path) for path in [BASE / 'dispatch.npz',
        BASE / 'issued_forecasts.npz', BASE / 'planning_audit.json', *[BASE / f'planning_day{d}.npz' for d in DAYS]]}
    sources = [Path(__file__), ROOT / 'experiments/exp008/budget_feedback.py',
        ROOT / 'experiments/exp008/markov_feedback.py', ROOT / 'experiments/exp008/markov_tail_feedback.py',
        ROOT / 'experiments/exp008/full28_mode_diagnostic_audit.py', ROOT / 'experiments/exp008/verify.py']
    source_hashes = {}
    for path in sources:
        destination = OUT / 'source_archive' / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        source_hashes[str(path.relative_to(ROOT))] = sha(path)
    save(OUT / 'source_manifest.json', dict(input_files=inputs, source_files=source_hashes,
        protocol_sha256=sha(OUT / 'protocol.json')))
    save(OUT / 'functional_checks.json', policy.checks())
    source = load(BASE / 'dispatch.npz')
    issued = load(BASE / 'issued_forecasts.npz')
    prior_power = 6 * (source['charge'] - source['discharge']).ravel()
    prepared = {}
    # Initial functional checks for all five days precede any coordinate sweep.
    for day in DAYS:
        p = load(BASE / f'planning_day{day}.npz')
        prefix = prior_power[:(day - 31) * 144]
        nonidle = prefix[np.abs(prefix) > 1e-6]
        p['initial_real_mode'] = int(np.sign(nonidle[-1])) if len(nonidle) else 1
        p['initial_power_kw'] = float(prefix[-1]) if len(prefix) else 172.75999999999976
        row = int(np.flatnonzero(issued['origins'] == day * 144)[0])
        forecast = (issued['values'][row, :, 0] - issued['values'][row, :, 1]) / 6
        errors = p['all_net_paths'] - forecast[None, :]
        origins = np.arange(day - 28, day) * 144
        model = policy.fit_error_model(forecast, errors, points=3, history_origins=origins, cutoff=day * 144)
        terminal = 0. if day == 364 else .45
        q = source['original'][day - 31].copy()
        initial = evaluate(q, p, model, terminal)
        initial['report']['prefix_mutations'] = prefix_checks(q, p, initial['grid'], initial['values'], model, initial['flow'])
        prepared[day] = dict(p=p, model=model, terminal=terminal, initial=initial)
        np.savez_compressed(OUT / f'day{day}_inputs.npz', **p, forecast_net=forecast,
            historical_errors=errors, history_origins=origins, original_Q=q)
        np.savez_compressed(OUT / f'day{day}_model.npz', **{k: v for k, v in model.items() if k != 'metadata'})
        save(OUT / f'day{day}_initial_policy_check.json', dict(model_metadata=model['metadata'], **initial['report']))
        print('INITIAL_CHECKED', day, initial['report']['historical_policy_objective'],
            initial['report']['DP_initial_expected_objective'], flush=True)
    save(OUT / 'all_five_initial_checks_passed.json', dict(days=list(DAYS), passed=True, actual_CSV_loaded=False))
    selections = []
    for day in DAYS:
        item = prepared[day]
        p, model, terminal, initial = [item[k] for k in ('p', 'model', 'terminal', 'initial')]
        best = initial
        theta = np.zeros(8)
        qbase = initial['q']
        candidates = [dict(index=0, theta=theta.tolist(), **initial['report'])]
        all_q = [qbase.copy()]
        coordinate_steps = []
        best_index = 0
        for block in range(8):
            chosen_theta = theta.copy()
            step_record = dict(block=block, original_theta=theta.tolist(), before_best_index=best_index, candidates=[])
            # Both signs are centered on the same preceding block-coordinate state.
            for sign in (-1, 1):
                candidate_theta = theta.copy()
                candidate_theta[block] = np.clip(candidate_theta[block] + 50 * sign, -policy.LIMIT, policy.LIMIT)
                q = np.maximum(0., qbase + np.repeat(candidate_theta, 18))
                candidate = evaluate(q, p, model, terminal)
                index = len(candidates)
                candidates.append(dict(index=index, theta=candidate_theta.tolist(), **candidate['report']))
                all_q.append(q)
                step_record['candidates'].append(index)
                if candidate['report']['historical_policy_objective'] < best['report']['historical_policy_objective'] - 1e-7:
                    best, chosen_theta, best_index = candidate, candidate_theta, index
            theta = chosen_theta
            step_record['after_best_index'] = best_index
            coordinate_steps.append(step_record)
        assert len(candidates) == 17
        assert best['report']['historical_policy_objective'] <= initial['report']['historical_policy_objective'] + 1e-7
        prefix = prefix_checks(best['q'], p, best['grid'], best['values'], model, best['flow'])
        for label, choice in (('initial', initial), ('selected', best)):
            np.savez_compressed(OUT / f'day{day}_{label}_policy.npz', q=choice['q'], grid=choice['grid'], values=choice['values'])
            np.savez_compressed(OUT / f'day{day}_{label}_history.npz', **choice['flow'])
        np.savez_compressed(OUT / f'day{day}_all_candidate_Q.npz', q=np.stack(all_q),
            theta=np.asarray([r['theta'] for r in candidates]))
        report = dict(day=day, candidates=candidates, coordinate_steps=coordinate_steps,
            selected_index=best_index, selected_theta_kwh=theta.tolist(), selected_report=best['report'],
            initial_report=initial['report'], selected_prefix_checks=prefix,
            selected_policy_sha256=sha(OUT / f'day{day}_selected_policy.npz'),
            historical_selection_complete_before_current_actual=True,
            initial_real_mode=int(p['initial_real_mode']), initial_power_kw=float(p['initial_power_kw']))
        save(OUT / f'day{day}_selection.json', report)
        selections.append(report)
        prepared[day]['selected'] = best
        print('BUDGET_SELECTED', day, best_index, initial['report']['historical_policy_objective'],
            best['report']['historical_policy_objective'], flush=True)
    save(OUT / 'all_choices_locked_before_actual.json', dict(actual_CSV_loaded=False,
        choices={str(r['day']): r['selected_policy_sha256'] for r in selections}))
    # Only after every midnight decision has been fixed do actual CSVs enter.
    actual = np.stack([pd.read_csv(ROOT / 'data/raw' / name).iloc[:, 1:].to_numpy(float)
        for name in ('附件2_小区负载.csv', '附件2_光伏发电实际功率.csv')], axis=-1)
    rows = []
    for day in DAYS:
        item, p = prepared[day], prepared[day]['p']
        net = ((actual[day, :, 0] - actual[day, :, 1]) / 6)[None, :]
        for label in ('source_hold1', 'initial', 'selected'):
            if label == 'source_hold1':
                flow = {key: source[key][day - 31:day - 30].copy() for key in
                    ('charge', 'discharge', 'emergency', 'surplus', 'states', 'fees')}
                q = source['original'][day - 31]
            else:
                choice = item[label]
                q = choice['q']
                flow = policy.execute_paths(q, net, p['price'], float(p['initial_soc']), choice['grid'],
                    choice['values'], item['model'], initial_mode=int(p['initial_real_mode']))
            error = check_flow(q, net, p['price'], flow, float(p['initial_soc']), int(p['initial_real_mode']))
            np.savez_compressed(OUT / f'day{day}_{label}_actual_replay.npz', **flow)
            metrics = battery_metrics(flow, initial_mode=int(p['initial_real_mode']), initial_power_kw=float(p['initial_power_kw']))
            # Existing generic starts count begins at idle. Also expose the
            # observed previous-slot-aware episode count for matched daily use.
            signs = np.where(np.abs(flow['charge'][0] - flow['discharge'][0]) > 1e-6,
                np.sign(flow['charge'][0] - flow['discharge'][0]), 0).astype(int)
            prior = int(np.sign(p['initial_power_kw'])) if abs(p['initial_power_kw']) > 6e-6 else 0
            starts_c = int(np.sum((signs == 1) & (np.r_[prior, signs[:-1]] != 1)))
            starts_d = int(np.sum((signs == -1) & (np.r_[prior, signs[:-1]] != -1)))
            rows.append(dict(day=day, label=label, total_cost=float(flow['fees'].sum()),
                plan_cost=float(flow['fees'][:, :, 0].sum()), emergency_cost=float(flow['fees'][:, :, 3].sum()),
                physical_error=error, charge_starts_with_previous_slot=starts_c,
                discharge_starts_with_previous_slot=starts_d, episodes_with_previous_slot=starts_c + starts_d,
                **metrics))
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / 'conditional_comparison.csv', index=False)
    pairs = []
    for day in DAYS:
        a = frame[(frame.day == day) & (frame.label == 'source_hold1')].iloc[0]
        b = frame[(frame.day == day) & (frame.label == 'initial')].iloc[0]
        c = frame[(frame.day == day) & (frame.label == 'selected')].iloc[0]
        row = dict(day=day)
        for label, reference in (('vs_source_hold1', a), ('vs_same_Q_adaptive_policy', b)):
            row[label] = {key + '_delta': float(c[key] - reference[key]) for key in
                ('total_cost', 'plan_cost', 'emergency_cost', 'throughput_kwh', 'active_slots', 'final_soc',
                 'direction_reversals_including_warmup_boundary', 'episodes_with_previous_slot')}
        pairs.append(row)
    assert inputs == {name: sha(BASE / name) for name in inputs}
    assert source_hashes == {name: sha(ROOT / name) for name in source_hashes}
    summary = dict(complete=True, fixed_days=list(DAYS), actual_current_information_not_used_in_choice=True,
        maximum_evaluations_per_day=17, all_historical_candidate_physics_checked=True,
        all_fixed_sources_unchanged=True, all_real_budgets_at_most8=True, paired=pairs,
        totals={label: {key: float(frame[frame.label == label][key].sum()) for key in
            ('total_cost', 'plan_cost', 'emergency_cost', 'throughput_kwh', 'active_slots',
             'direction_reversals_including_warmup_boundary', 'episodes_with_previous_slot')}
            for label in ('source_hold1', 'initial', 'selected')},
        annual_savings_not_evaluated=True, goal_gate_not_evaluated=True,
        limitations=['Independent original day starting states; changed day-end inventory not propagated.',
            'Direction budget is hard in execution; model distribution, SOC grid and value interpolation remain approximations.',
            'Historical policy bill chooses Q; the same history also fits disturbance transitions, so this is development optimization.'])
    save(OUT / 'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    run()
