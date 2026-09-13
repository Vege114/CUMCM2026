"""Fixed-five numerical resolution check, not an annual strategy selection."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.exp008 import budget_feedback as policy
from experiments.exp008.budget_feedback_diagnostic import check_flow, load
from experiments.problem2.exp003.data import Data

BASE = Path('data/results/exp008/budget_feedback/fixed5_hgb_cap8_grid200_blocks8_step50')
OUT = Path('data/results/exp008/budget_feedback/grid100_fixed5_diagnostic')
DAYS = (31, 90, 151, 243, 364)


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def numerical_metrics(q, p, flow, day):
    power = flow['charge']-flow['discharge']
    modes = flow['modes']
    # Count contiguous charge/discharge starts separately from nonidle reversals.
    activity = np.sign(power)
    previous = np.concatenate((np.full((len(power), 1), np.sign(float(p['initial_power_kw']))), activity[:, :-1]), axis=1)
    return {'day': day, 'cash_fee': float(flow['fees'].sum()),
        'emergency_fee': float(flow['fees'][:, :, 3].sum()),
        'direction_reversals_including_boundary': int(np.sum(np.diff(modes, axis=1) != 0)),
        'activity_episode_starts': int(np.sum((activity != 0) & (activity != previous))),
        'active_slots': int(np.sum(np.abs(power) > 1e-6)),
        'throughput_kwh': float((flow['charge']+flow['discharge']).sum()),
        'final_soc_kwh': float(flow['states'][0, -1])}


def main():
    if OUT.exists():
        raise FileExistsError('Completed or begun numerical checks are immutable')
    OUT.mkdir(parents=True)
    required = [BASE/'independent_audit.json']
    required += [BASE/f'day{day}_{suffix}.npz' for day in DAYS
                 for suffix in ('inputs', 'model', 'initial_policy', 'initial_actual_replay')]
    hashes = {str(p): digest(p) for p in required}
    save(OUT/'protocol.json', {'days': DAYS, 'grid_control_kwh': 200, 'grid_candidate_kwh': 100,
        'purchases_and_initial_states': 'fixed already signed five-day initial Q diagnostic',
        'Q_reoptimization': False, 'all_models_and_DP_other_settings_unchanged': True,
        'policies_locked_before_actual_access': True, 'finer_grid_not_assumed_to_improve_actual_cost': True,
        'annual_extrapolation_or_model_selection': False, 'inputs_sha256': hashes,
        'source_sha256': digest(__file__)})
    sources = ('budget_grid_diagnostic.py', 'budget_feedback.py', 'budget_feedback_diagnostic.py')
    for name in sources:
        (OUT/name).write_bytes((Path(__file__).parent/name).read_bytes())
    prepared = {}
    history_rows = []
    for day in DAYS:
        p = load(BASE/f'day{day}_inputs.npz')
        model = load(BASE/f'day{day}_model.npz')
        original = load(BASE/f'day{day}_initial_policy.npz')
        q = original['q']; terminal = 0. if day == 364 else .45
        grid, values, metadata = policy.value_functions(q, model, p['price'], grid_kwh=100., terminal=terminal)
        flow = policy.execute_paths(q, p['all_net_paths'], p['price'], float(p['initial_soc']), grid, values, model,
                                    initial_mode=int(p['initial_real_mode']))
        old = policy.execute_paths(q, p['all_net_paths'], p['price'], float(p['initial_soc']), original['grid'], original['values'], model,
                                   initial_mode=int(p['initial_real_mode']))
        error = check_flow(q, p['all_net_paths'], p['price'], flow, float(p['initial_soc']), int(p['initial_real_mode']))
        row = {'day': day, 'history_objective_grid100': policy.historical_objective(flow, terminal=terminal),
            'history_objective_grid200': policy.historical_objective(old, terminal=terminal),
            'max_physics_error': error, 'metadata': metadata,
            'largest_value_difference_on_shared_grid_yuan': float(np.max(np.abs(values[..., ::2]-original['values'])))}
        np.savez_compressed(OUT/f'day{day}_policy.npz', q=q, grid=grid, values=values)
        save(OUT/f'day{day}_history.json', row)
        history_rows.append(row); prepared[day] = (p, model, q, grid, values, original)
    save(OUT/'all_policies_locked.json', {str(day): digest(OUT/f'day{day}_policy.npz') for day in DAYS})
    data = Data(); rows = []; mutations = []
    for day, (p, model, q, grid, values, original) in prepared.items():
        actual = data.actual[day*144:(day+1)*144, :2]
        net = ((actual[:, 0]-actual[:, 1])/6)[None, :]
        new = policy.execute_paths(q, net, p['price'], float(p['initial_soc']), grid, values, model,
                                   initial_mode=int(p['initial_real_mode']))
        old = policy.execute_paths(q, net, p['price'], float(p['initial_soc']), original['grid'], original['values'], model,
                                   initial_mode=int(p['initial_real_mode']))
        signed = load(BASE/f'day{day}_initial_actual_replay.npz')
        for key in signed:
            np.testing.assert_array_equal(old[key], signed[key])
        check_flow(q, net, p['price'], new, float(p['initial_soc']), int(p['initial_real_mode']))
        np.savez_compressed(OUT/f'day{day}_actual_replay.npz', **new)
        rows.extend([{'grid_kwh': resolution, **numerical_metrics(q, p, flow, day)}
                     for resolution, flow in ((200, old), (100, new))])
        for stop in (1, 36, 108):
            changed = net.copy(); changed[:, stop:] += 10000.+np.arange(144-stop)
            f = policy.execute_paths(q, changed, p['price'], float(p['initial_soc']), grid, values, model,
                                      initial_mode=int(p['initial_real_mode']))
            for key in ('charge', 'discharge', 'emergency', 'surplus', 'observed_error_bins'):
                np.testing.assert_array_equal(f[key][:, :stop], new[key][:, :stop])
            for key in ('states', 'modes', 'remaining_budget'):
                np.testing.assert_array_equal(f[key][:, :stop+1], new[key][:, :stop+1])
            mutations.append({'day': day, 'future_net_mutated_from_slot': stop, 'prefix_identical': True})
    frame = pd.DataFrame(rows); frame.to_csv(OUT/'matched_metrics.csv', index=False)
    assert all(digest(p) == h for p, h in hashes.items())
    delta = frame[frame.grid_kwh == 100].set_index('day').drop(columns='grid_kwh')-frame[frame.grid_kwh == 200].set_index('day').drop(columns='grid_kwh')
    delta.to_csv(OUT/'deltas.csv')
    result = {'complete': True, 'all_source_hashes_unchanged': True, 'future_mutation_checks': mutations,
        'sum_fixed5_cash_fee_change_yuan': float(delta.cash_fee.sum()),
        'sum_fixed5_historical_objective_change_yuan': sum(r['history_objective_grid100']-r['history_objective_grid200'] for r in history_rows),
        'no_continuous_annual_result': True, 'final_model_selection': False}
    save(OUT/'summary.json', result); print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
