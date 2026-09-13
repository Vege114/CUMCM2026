"""Five fixed conditional days: shared variable modes over all 28 past paths."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.exp008.closed_loop import ETA, LOW, objective, optimize
from experiments.exp008.fixed_purchase_bounds import sha256
from experiments.exp008.mode_boundary_alignment import history_replay
from experiments.exp008.mode_budget_hold1_physical import mode_state, plan
from experiments.exp008.verify import battery_metrics

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'data/results/exp008/mode_budget_hold1_physical/direct_hgb_memory_hold1_cap8_kappa0_334days'
OUT = ROOT/'data/results/exp008/full28_mode_diagnostic/hold1_cap8_fixed5_60seconds'
DAYS = (31, 90, 151, 243, 364)


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def load(path):
    with np.load(path) as z:
        return {key: z[key].copy() for key in z.files}


def snapshot(day, label, q, mask, p):
    terminal = 0. if day == 364 else .45
    flows, metric = history_replay(q, p['all_net_paths'], p['price'], float(p['initial_soc']), mask, terminal)
    state, flips = mode_state(mask, int(p['previous_planned_mode']), int(p['previous_planned_run_slots']))
    assert int(flips.sum()) <= 8 and state['next_day_required_hold_slots'] == 0
    np.savez_compressed(OUT/f'day{day}_{label}.npz', q=q, mask=mask,
        paths=p['all_net_paths'], price=p['price'], initial_soc=p['initial_soc'], **flows)
    metric.update(planned_switches_including_midnight=int(flips.sum()), final_planned_state=state,
        changed_mask_slots_vs_source3=int(np.sum(mask != p['allowed_charge'])))
    return metric


def select(day, source_audit):
    p = load(BASE/f'planning_day{day}.npz')
    assert p['all_net_paths'].shape == (28, 144)
    np.testing.assert_array_equal(p['selected_scenario_indices'], np.linspace(0, 27, 3).astype(int))
    with np.load(BASE/'dispatch.npz') as z:
        source_q = z['original'][day-31].copy()
        np.testing.assert_array_equal(p['allowed_charge'], z['allowed_charge'][day-31])
    paths, price, soc = p['all_net_paths'], p['price'], float(p['initial_soc'])
    previous, age = int(p['previous_planned_mode']), int(p['previous_planned_run_slots'])
    args = {'throughput': .002, 'variation': 0., 'terminal': 0. if day == 364 else .45, 'deadband': 0.}
    reproduced = optimize(p['purchase'], paths, price, soc, charge_mask=p['allowed_charge'], maxiter=120, **args)
    np.testing.assert_array_equal(reproduced['purchase'], source_q)
    np.testing.assert_allclose(reproduced['metadata']['objective'], source_audit['greedy_refinement']['objective'], atol=1e-6, rtol=0.)
    report = {'day': day, 'original_refine120_Q_reproduced_exactly': True,
        'source3_MIP_metadata': source_audit['mip'], 'source3_refinement_metadata': source_audit['greedy_refinement'],
        'source3_initial': snapshot(day, 'source3_initial', p['purchase'], p['allowed_charge'], p),
        'source3_refined': snapshot(day, 'source3_refined', source_q, p['allowed_charge'], p),
        'current_actual_used': False}
    try:
        initialized = plan(paths, price, soc, previous, age, seconds=60., gap=.002, final=day == 364)
    except (RuntimeError, AssertionError) as exc:
        report.update(full28_feasible=False, failure=str(exc), no_added_time_or_parameter_change=True)
        save(OUT/f'day{day}_selection.json', report)
        print('FULL28_FAILED', day, str(exc), flush=True)
        return report
    meta = initialized['metadata']
    assert meta['scenario_count'] == 28 and meta['hold_slots'] == 1
    assert meta['daily_planned_switch_cap'] == 8 and meta['switching_cost'] == 0.
    assert initialized['allowed_charge'].shape == (144,)
    np.savez_compressed(OUT/f'day{day}_full28_MIP.npz', **{k: v for k, v in initialized.items() if k != 'metadata'})
    c, d, e, w, states = [initialized['scenario_'+key] for key in ('charge', 'discharge', 'emergency', 'surplus', 'states')]
    error = max(float(np.abs(initialized['purchase'][None, :]+d+e-c-w-paths).max()),
        float(np.abs(np.diff(states, axis=1)-ETA*c+d/ETA).max()))
    assert error < 1e-6 and np.all(states[:, 0] == soc)
    assert not np.any((c > 1e-6) & ((d > 1e-6) | (e > 1e-6)))
    mask = initialized['allowed_charge']
    assert np.all(c[:, ~mask] <= 1e-6) and np.all(d[:, mask] <= 1e-6)
    initial = snapshot(day, 'full28_initial', initialized['purchase'], mask, p)
    refined = optimize(initialized['purchase'], paths, price, soc, charge_mask=mask, maxiter=120, **args)
    refined_metric = snapshot(day, 'full28_refined', refined['purchase'], mask, p)
    choose_refined = refined_metric['objective'] <= initial['objective']
    selected = 'full28_refined' if choose_refined else 'full28_initial'
    # Same-terminal proxy/greedy comparison is diagnostic; it never sees actual.
    mip_terminal = 0. if day == 364 else float(price.min())/ETA
    recourse_value = float(initialized['purchase']@price+np.mean(np.sum(5*e*price+.002*(c+d), axis=1))
                           -mip_terminal*np.mean(states[:, -1]-LOW))
    greedy_mip_terminal = float(objective(initialized['purchase'], paths, price, soc,
        charge_mask=mask, throughput=.002, variation=0., terminal=mip_terminal, deadband=0.)[0])
    report.update(full28_feasible=True, full28_MIP_metadata=meta,
        full28_independent_scenario_physical_error=error,
        full28_initial=initial, full28_refined=refined_metric, full28_refinement_metadata=refined['metadata'],
        selected=selected, selection_criterion='minimum full28 exact greedy historical objective of new initial/refined only',
        selected_objective=(refined_metric if choose_refined else initial)['objective'],
        full28_MIP_recourse_at_MIP_terminal=recourse_value,
        same_Q_greedy_full28_at_MIP_terminal=greedy_mip_terminal,
        recourse_optimism_at_matched_terminal=greedy_mip_terminal-recourse_value,
        source3_is_frozen_comparator_not_actual_based_fallback=True,
        selected_sha256=sha256(OUT/f'day{day}_{selected}.npz'))
    save(OUT/f'day{day}_selection.json', report)
    print('FULL28_SELECTED', day, meta['seconds'], meta['mip_gap'], report['source3_refined']['objective'], report['selected_objective'], flush=True)
    return report


def run():
    if OUT.exists():
        raise FileExistsError('Never overwrite a conditional full28 diagnostic')
    OUT.mkdir(parents=True)
    protected = {str(path.relative_to(BASE)): sha256(path) for path in sorted(BASE.rglob('*')) if path.is_file()}
    save(OUT/'protocol.json', {'days': DAYS, 'maximum_configurations': 5, 'continuous_year': False,
        'initial_state': 'frozen hold1 cap8 annual SOC, previous planned mode/age for each selected day',
        'common_mode_variables': 144, 'all28_historical_paths_jointly_select_purchase_and_mode': True,
        'mode_is_not_fixed_to_source3': True, 'daily_planned_switch_cap': 8, 'minimum_hold_slots': 1,
        'switching_penalty': 0., 'MIP_seconds_per_day': 60., 'MIP_relative_gap_target': .002,
        'scenario_emergency_charging_and_charge_discharge_overlap_forbidden': True,
        'greedy_refinement_maxiter': 120, 'wear': .002, 'variation': 0., 'deadband': 0.,
        'terminal': 'MIP min known price/ETA, greedy .45; both0 at day364',
        'selection': 'minimum full28 exact greedy historical objective among new initial/refined; source3 only paired comparator',
        'all_five_selections_locked_before_current_actual_loaded': True,
        'failed_MIP_days_recorded_without_extra_time': True,
        'source3_time_budget_was5seconds': True, 'no_annual_extrapolation': True,
        'scenario_recourse_not_a_nonanticipative_tree_certificate': True,
        'protected_source_tree_sha256': protected})
    source_files = [Path(__file__), ROOT/'experiments/exp008/mode_budget_hold1_physical.py',
        ROOT/'experiments/exp008/closed_loop.py', ROOT/'experiments/exp008/mode_boundary_alignment.py',
        ROOT/'experiments/common/neural_v2/physics.py', ROOT/'experiments/exp008/verify.py']
    hashes = {}
    for path in source_files:
        destination = OUT/'source_archive'/path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        hashes[str(path.relative_to(ROOT))] = sha256(destination)
    save(OUT/'source_sha256.json', hashes)
    audit = json.loads((BASE/'planning_audit.json').read_text())
    choices = [select(day, audit[day-31]) for day in DAYS]
    save(OUT/'all_choices_locked_before_actual.json', {'current_actual_loaded': False,
        'choices': {str(r['day']): {'feasible': r['full28_feasible'], 'selected_sha256': r.get('selected_sha256')} for r in choices}})
    # Actual rows become available to this diagnostic only after all choices lock.
    actual = np.stack([pd.read_csv(ROOT/'data/raw'/name).iloc[:, 1:].to_numpy(float)
        for name in ('附件2_小区负载.csv', '附件2_光伏发电实际功率.csv')], axis=-1)
    base = load(BASE/'dispatch.npz')
    prior_power = 6*(base['charge']-base['discharge']).ravel()
    rows = []
    for report in choices:
        day = report['day']
        seen = prior_power[:(day-31)*144]
        nonidle = seen[np.abs(seen) > 1e-6]
        initial_mode = int(np.sign(nonidle[-1])) if len(nonidle) else 1
        initial_power = float(seen[-1]) if len(seen) else 172.75999999999976
        labels = ['source3_initial', 'source3_refined']
        if report['full28_feasible']:
            assert sha256(OUT/f'day{day}_{report["selected"]}.npz') == report['selected_sha256']
            labels += ['full28_initial', 'full28_refined']
        for label in labels:
            p = load(OUT/f'day{day}_{label}.npz')
            net = (actual[day, :, 0]-actual[day, :, 1])[None, :]/6
            flows, _ = history_replay(p['q'], net, p['price'], float(p['initial_soc']), p['mask'], 0. if day == 364 else .45)
            if label == 'source3_refined':
                for key, value in flows.items():
                    np.testing.assert_array_equal(value[0], base[key][day-31])
            flows['fees'] = np.stack((p['q'][None, :]*p['price'], np.zeros((1, 144)),
                                     np.zeros((1, 144)), 5*flows['emergency']*p['price']), axis=-1)
            np.savez_compressed(OUT/f'day{day}_{label}_actual_replay.npz', **flows)
            rows.append({'day': day, 'label': label, 'selected_new': label == report.get('selected'),
                'total_cost': float(flows['fees'].sum()), 'planned_cost': float(flows['fees'][:, :, 0].sum()),
                'emergency_cost': float(flows['fees'][:, :, 3].sum()),
                'historical_full28_objective': report[label]['objective'],
                'mask_changes_vs_source3': report[label]['changed_mask_slots_vs_source3'],
                **battery_metrics(flows, initial_mode=initial_mode, initial_power_kw=initial_power)})
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT/'conditional_comparison.csv', index=False)
    assert protected == {str(path.relative_to(BASE)): sha256(path) for path in sorted(BASE.rglob('*')) if path.is_file()}
    summary = {'complete': True, 'fixed_days': DAYS, 'full28_feasible_days': sum(x['full28_feasible'] for x in choices),
        'all_source_Q_and_original_actual_replays_exact': True,
        'all_current_actual_reads_after_choices_locked': True, 'protected_archives_unchanged': True,
        'continuous_annual_cost_not_evaluated': True, 'no_annual_extrapolation': True,
        'paired': []}
    for report in choices:
        if not report['full28_feasible']:
            continue
        day = report['day']
        a = frame[(frame.day == day) & frame.selected_new].iloc[0]
        b = frame[(frame.day == day) & (frame.label == 'source3_refined')].iloc[0]
        summary['paired'].append({'day': day, 'historical_objective_delta': float(a.historical_full28_objective-b.historical_full28_objective),
            'actual_cost_delta': float(a.total_cost-b.total_cost), 'actual_end_SOC_delta_kwh': float(a.final_soc-b.final_soc),
            'actual_direction_reversals_delta': int(a.direction_reversals-b.direction_reversals),
            'mask_changed_slots': int(a.mask_changes_vs_source3),
            'MIP_seconds': report['full28_MIP_metadata']['seconds'], 'MIP_gap': report['full28_MIP_metadata']['mip_gap']})
    save(OUT/'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    run()
