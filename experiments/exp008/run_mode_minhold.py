"""Fixed 3-day computational gate, then at most 30 continuous days if stable."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.exp008.closed_loop import optimize
from experiments.exp008.controller_candidate import INITIAL_SOC
from experiments.exp008.mode_minhold_physical import mode_state, plan
from experiments.exp008.neural_joint_dispatch import JointForecasts
from experiments.exp008.planner import ETA, execute
from experiments.exp008.verify import battery_metrics, verify_npz
from experiments.problem2.exp003.data import Data, ROOT

OUT = ROOT/'data/results/exp008/mode_minhold_physical/hold3_s3_5sec'
BASE = ROOT/'data/results/exp008/mode_planning_physical/joint_ridge28_refined_s3_30days'


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2))


def run():
    OUT.mkdir(exist_ok=True, parents=True)
    if (OUT/'summary.json').exists():
        raise FileExistsError('Completed hold3 experiment is immutable')
    data, forecast = Data(), JointForecasts()
    protocol = {'candidate_count': 1, 'forecast': 'monthly_Joint_then_fixed_Ridge28_no_memory',
        'scenario_count': 3, 'scenario_selection': 'equally_spaced_in_complete_history_origins',
        'common_mode_resolution_slots': 1, 'minimum_planned_mode_hold_slots': 3,
        'mode_hold_is_planned_mask_not_guaranteed_nonidle_battery_activity': True,
        'initial_planned_mode': 1, 'initial_planned_run_slots': 1,
        'initial_state_interpretation': 'warmup ended charging; conservatively count one known slot and lock first2 slots',
        'cross_midnight': 'carry exact last planned mode and its completed run length; lock pending initial slots',
        'switching_cost': 50., 'wear': .002, 'mip_seconds': 5., 'target_mip_gap': .002,
        'greedy_refinement': 'all28 complete daily paths, maxiter120, wear.002, TV0, terminal.45, deadband0',
        'actual_execution': 'strict_mask_and_deadband0_with_continuous_SOC',
        'computational_gate': 'all first3 days feasible, physically checked, mip_gap<=.05',
        'if_gate_passes': 'continue same fixed configuration and all states to30 days',
        'if_gate_fails': 'stop_without_time_limit_or_parameter_search',
        'nonanticipative_recourse_certificate': False,
        'development_year_not_independent_test': True,
        'forecast_values_sha256': hashlib.sha256(forecast.store.values.tobytes()).hexdigest(),
        'source_data_hashes': data.hashes, 'source_sha256': {}}
    names = ['experiments/exp008/'+name+'.py' for name in (
        'run_mode_minhold', 'mode_minhold_physical', 'closed_loop', 'planner',
        'forecast', 'neural_joint_dispatch', 'neural_joint_calibration', 'verify')]
    names += ['experiments/common/neural_v2/physics.py', 'experiments/problem2/exp003/data.py']
    for name in names:
        path = ROOT/name
        protocol['source_sha256'][name] = hashlib.sha256(path.read_bytes()).hexdigest()
        saved = OUT/'source_archive'/name
        saved.parent.mkdir(exist_ok=True, parents=True)
        saved.write_bytes(path.read_bytes())
    write_json(OUT/'protocol.json', protocol)
    np.savez_compressed(OUT/'issued_forecasts.npz', origins=forecast.store.origins, values=forecast.store.values)
    parts, audits, rows = [], [], []
    soc, planned_mode, planned_age = INITIAL_SOC, 1, 1
    computation_gate, failure = None, None
    for day in range(31, 61):
        issue = forecast.get(day, scenario='2')
        errors = forecast.net_error_paths(day, '2', limit=28)
        paths = (issue['load_kw']-issue['pv_kw'])[None, :]/6+errors['errors_kwh']
        selected = np.linspace(0, len(paths)-1, 3).astype(int)
        try:
            initialized = plan(paths[selected], data.fixed_price, soc, planned_mode, planned_age)
        except (RuntimeError, AssertionError) as exc:
            failure = {'day': day, 'completed_days': len(parts), 'error': str(exc),
                       'no_extra_time_or_parameter_changes': True}
            write_json(OUT/'failure.json', failure)
            print(json.dumps(failure), flush=True)
            break
        metadata = initialized['metadata']
        state, _ = mode_state(initialized['allowed_charge'], planned_mode, planned_age)
        np.savez_compressed(OUT/f'planning_day{day}.npz',
            **{k: v for k, v in initialized.items() if k != 'metadata'},
            all_net_paths=paths, selected_scenario_indices=selected,
            initial_soc=soc, previous_planned_mode=planned_mode, previous_planned_run_slots=planned_age,
            price=data.fixed_price)
        refined = optimize(initialized['purchase'], paths, data.fixed_price, soc,
            charge_mask=initialized['allowed_charge'], throughput=.002, variation=0.,
            terminal=.45, deadband=0., maxiter=120)
        q = refined['purchase']
        observed = data.actual[day*144:(day+1)*144]
        detail = execute(q, observed, data.fixed_price, soc,
                         charge_deadband=0., charge_mask=initialized['allowed_charge'])
        detail.update(original=q, final=q.copy(), actual=observed.copy(), price=data.fixed_price.copy(),
                      allowed_charge=initialized['allowed_charge'].copy())
        detail['fees'] = np.stack((q*data.fixed_price, np.zeros(144), np.zeros(144),
                                   5*detail['emergency']*data.fixed_price), axis=-1)
        parts.append(detail)
        audits.append({'day': day, 'forecast': issue['audit'], 'history': errors['audit'],
                       'mip': metadata, 'greedy_refinement': refined['metadata'],
                       'planned_boundary_state_before': [planned_mode, planned_age],
                       'planned_boundary_state_after': [state['final_planned_mode'], state['final_planned_run_slots']]})
        rows.append({'day': day, 'date': str((pd.Timestamp('2025-01-01')+pd.Timedelta(days=day)).date()),
            'total_cost': float(detail['fees'].sum()), 'emergency_cost': float(detail['fees'][:, 3].sum()),
            'initial_soc': soc, 'final_soc': float(detail['states'][-1]),
            'mip_gap': metadata['mip_gap'], 'mip_seconds': metadata['seconds'],
            'planned_changes': state['planned_changes_including_boundary'],
            'next_day_pending_mode_hold': state['next_day_required_hold_slots']})
        soc = rows[-1]['final_soc']
        planned_mode, planned_age = state['final_planned_mode'], state['final_planned_run_slots']
        print('DAY', day, 'cost', rows[-1]['total_cost'], 'gap', metadata['mip_gap'],
              'mode_state', planned_mode, planned_age, flush=True)
        if len(parts) == 3:
            gaps = [row['mip_gap'] for row in rows]
            computation_gate = {'passed': all(g is not None and g <= .05 for g in gaps),
                'days': 3, 'gaps': gaps, 'all_scenario_physical_checks_passed': True,
                'continued_to30': all(g is not None and g <= .05 for g in gaps)}
            write_json(OUT/'computational_gate.json', computation_gate)
            if not computation_gate['passed']:
                break
    if not parts:
        summary = {'completed_days': 0, 'computational_gate_passed': False,
                   'failure': failure, 'annual_effect_claimed': False}
        write_json(OUT/'summary.json', summary)
        return summary
    arrays = {key: np.stack([part[key] for part in parts]) for key in parts[0]}
    arrays['days'] = np.arange(31, 31+len(parts))
    np.savez_compressed(OUT/'dispatch.npz', **arrays)
    write_json(OUT/'planning_audit.json', audits)
    pd.DataFrame(rows).to_csv(OUT/'daily.csv', index=False)
    verification = verify_npz(OUT/'dispatch.npz', expected_days=len(parts), audit_path=OUT/'planning_audit.json')
    if not verification['passed']:
        raise AssertionError(verification['errors'])
    with np.load(BASE/'dispatch.npz') as saved:
        baseline = {name: saved[name][:len(parts)].copy() for name in saved.files if name != 'days'}
    battery, old_battery = battery_metrics(arrays), battery_metrics(baseline)
    baseline_cost = float(baseline['fees'].sum())
    cost = float(arrays['fees'].sum())
    compensation = max(0., old_battery['final_soc']-battery['final_soc'])*ETA*5*float(data.fixed_price.max())
    summary = {'completed_days': len(parts), 'total_cost': cost, 'baseline_cost': baseline_cost,
        'cost_change': cost-baseline_cost, 'battery': battery, 'baseline_battery': old_battery,
        'lower_final_SOC_compensation_upper_bound': compensation,
        'conservative_savings': baseline_cost-cost-compensation,
        'computational_gate': computation_gate, 'failure': failure, 'verification': verification,
        'last_planned_mode': planned_mode, 'last_planned_run_slots': planned_age,
        'next_day_mode_hold_remaining': max(0, 3-planned_age),
        'all_cross_day_planned_holds_verified': True, 'annual_effect_claimed': False}
    write_json(OUT/'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == '__main__':
    run()
