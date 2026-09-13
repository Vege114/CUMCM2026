"""Independent saved-array audit; never optimizes against actual observations."""

from __future__ import annotations

import hashlib
import json

import numpy as np

from experiments.exp008.mode_minhold_physical import mode_state
from experiments.exp008.neural_joint_dispatch import JointForecasts
from experiments.exp008.planner import ETA, HIGH, LIMIT, LOW, execute
from experiments.exp008.run_mode_minhold import BASE, OUT
from experiments.problem2.exp003.data import ROOT


def run():
    protocol = json.loads((OUT/'protocol.json').read_text())
    with np.load(OUT/'dispatch.npz') as packed:
        arrays = {key: packed[key] for key in packed.files}
    forecast = JointForecasts()
    checks = []
    max_error = 0.
    for i, day in enumerate(arrays['days']):
        day = int(day)
        with np.load(OUT/f'planning_day{day}.npz') as packed:
            p = {key: packed[key] for key in packed.files}
        issue = forecast.get(day, scenario='2')
        expected = ((issue['load_kw']-issue['pv_kw'])[None, :]/6
                    + forecast.net_error_paths(day, '2', limit=28)['errors_kwh'])
        errors = [float(np.max(np.abs(p['all_net_paths']-expected)))]
        with np.load(BASE/f'planning_day{day}.npz') as old:
            errors.append(float(np.max(np.abs(old['all_net_paths']-p['all_net_paths']))))
            np.testing.assert_array_equal(old['selected_scenario_indices'], p['selected_scenario_indices'])
            np.testing.assert_array_equal(old['price'], p['price'])
        mask = p['allowed_charge'].astype(bool)
        np.testing.assert_array_equal(mask, arrays['allowed_charge'][i])
        assert float(p['initial_soc']) == arrays['states'][i, 0]
        if i:
            assert float(p['initial_soc']) == arrays['states'][i-1, -1]
            assert int(p['previous_planned_mode']) == state['final_planned_mode']
            assert int(p['previous_planned_run_slots']) == state['final_planned_run_slots']
        state, flips = mode_state(mask, int(p['previous_planned_mode']), int(p['previous_planned_run_slots']))
        np.testing.assert_array_equal(flips, p['planned_flips'])
        c, d, e, w, s = [p['scenario_'+key] for key in ('charge', 'discharge', 'emergency', 'surplus', 'states')]
        net = p['all_net_paths'][p['selected_scenario_indices']]
        errors.extend((float(np.max(np.abs(p['purchase'][None, :]+d+e-c-w-net))),
                       float(np.max(np.abs(np.diff(s, axis=1)-ETA*c+d/ETA)))))
        assert c.min() >= -1e-6 and d.min() >= -1e-6 and e.min() >= -1e-6 and w.min() >= -1e-6
        assert s.min() >= LOW-1e-6 and s.max() <= HIGH+1e-6
        assert max(c.max(), d.max()) <= LIMIT+1e-6
        assert not np.any((c > 1e-6) & (d > 1e-6))
        assert not np.any((c > 1e-6) & (e > 1e-6))
        assert np.max(c[:, ~mask], initial=0.) < 1e-6
        assert np.max(d[:, mask], initial=0.) < 1e-6
        q, observed, price = [arrays[key][i] for key in ('original', 'actual', 'price')]
        replay = execute(q, observed, price, float(p['initial_soc']), charge_mask=mask, charge_deadband=0.)
        for key in ('charge', 'discharge', 'emergency', 'surplus', 'states'):
            errors.append(float(np.max(np.abs(replay[key]-arrays[key][i]))))
        # Holding the published purchase/mask fixed, mutations after a chosen
        # observation cutoff must leave every previously executed slot intact.
        for cutoff in (1, 71, 143):
            changed = observed.copy()
            changed[cutoff:, 0] = changed[cutoff:, 0]*7+50000
            changed[cutoff:, 1] = changed[cutoff:, 1]*.01
            polluted = execute(q, changed, price, float(p['initial_soc']), charge_mask=mask, charge_deadband=0.)
            for key in ('charge', 'discharge', 'emergency', 'surplus'):
                errors.append(float(np.max(np.abs(polluted[key][:cutoff]-replay[key][:cutoff]))))
            errors.append(float(np.max(np.abs(polluted['states'][:cutoff+1]-replay['states'][:cutoff+1]))))
        assert max(errors) < 1e-6
        max_error = max(max_error, *errors)
        checks.append({'day': day, 'maximum_error': max(errors), 'passed': True, **state})
    hashes = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest
              for name, digest in protocol['source_sha256'].items()}
    assert all(hashes.values())
    flat = arrays['allowed_charge'].ravel()
    indices = np.flatnonzero(flat[1:] != flat[:-1])+1
    assert len(indices) < 2 or np.diff(indices).min() >= 3
    output = {'passed': True, 'days': len(checks), 'scenario_paths_per_day': 3,
        'all_historical_paths_rebuilt': True, 'baseline_forecasts_paths_selection_and_prices_identical': True,
        'baseline_dispatch_sha256': hashlib.sha256((BASE/'dispatch.npz').read_bytes()).hexdigest(),
        'continuous_SOC_and_planned_mode_state': True,
        'execution_future_mutation_checks': 3*len(checks), 'maximum_numerical_error': max_error,
        'source_hash_checks': hashes, 'day_checks': checks,
        'planned_changes_off_hour_boundary': int(np.sum(indices % 6 != 0)),
        'planned_changes_within_evaluation': len(indices),
        'minimum_completed_mode_duration_slots': int(np.diff(indices).min()) if len(indices)>1 else None,
        'end_of_trial_pending_hold_slots': state['next_day_required_hold_slots'],
        'scenario_recourse_is_optimistic_not_a_nonanticipative_policy_certificate': True}
    (OUT/'independent_audit.json').write_text(json.dumps(output, indent=2))
    print(json.dumps({key: value for key, value in output.items() if key not in ('day_checks', 'source_hash_checks')}, indent=2))
    return output


if __name__ == '__main__':
    run()
