"""Independent archive, input identity and configuration audit for Q3/Q4-3.

This module imports no optimizer, predictor or exp009 runner. It reuses only
the existing independent physical and billing verifier and reads frozen files.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.exp008.verify import verify_npz

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data/results/exp009/update_cost_only'
BASE = ROOT / 'data/results/exp008/absolute_hgb_update_dispatch/full334'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n')


def audit_scenario(scenario):
    if scenario not in ('3', '4-3'):
        raise ValueError('Only Q3 and Q4-3 are in this controlled experiment')
    directory, baseline = OUT / 'full334' / scenario, BASE / scenario
    protocol = json.loads((OUT / 'run_protocol.json').read_text())
    assert digest(ROOT / 'experiments/exp009/protocol.json') == protocol['root_protocol_sha256']
    warmup = ROOT / f'data/results/exp002/warmup_{scenario}.npz'
    with np.load(warmup, allow_pickle=False) as z:
        soc = float(z['states'][-1, -1])
        power = float(6 * (z['charge'][-1, -1] - z['discharge'][-1, -1]))
    args = dict(scenario=scenario, expected_days=334, initial_soc=soc,
                initial_mode=int(np.sign(power)), initial_power_kw=power)
    verified = []
    for where in (baseline, directory):
        result = verify_npz(where / f'dispatch_{scenario}.npz',
                            audit_path=where / 'audit.json', **args)
        assert result['passed'], result['errors']
        verified.append(result)
    old, new = verified
    old_completion = json.loads((baseline / 'completion.json').read_text())
    completion = json.loads((directory / 'completion.json').read_text())
    a, b = old_completion['config'], completion['config']
    for key in a:
        if key not in ('case', 'settings', 'deadband', 'forecast_override'):
            assert a[key] == b[key], ('unexpected_config_difference', key)
    expected_settings = dict(a['settings'], throughput_penalty=0., variation_penalty=0.)
    assert b['settings'] == expected_settings and b['deadband'] == 0.
    assert Path(b['case']).resolve() == OUT / 'full334'
    # The original was executed as __main__; importing that exact source now
    # changes only Python's descriptive module qualifier, never its class code.
    identity_a, identity_b = dict(a['forecast_override']), dict(b['forecast_override'])
    class_a = identity_a.pop('implementation_class')
    class_b = identity_b.pop('implementation_class')
    assert class_a.rsplit('.', 1)[-1] == class_b.rsplit('.', 1)[-1] == 'LinkedHGBForecasts'
    assert identity_a == identity_b, 'forecast_source_or_value_identity_changed'
    before = json.loads((baseline / 'audit.json').read_text())
    after = json.loads((directory / 'audit.json').read_text())
    assert len(before) == len(after) == 334 * 4
    rows = []
    invariant_solver = ('method', 'scenarios', 'nonanticipative_recourse_certificate',
                        'next_update_slot', 'future_shortfall_weight',
                        'planning_emergency_weights', 'opportunity_proxy_active',
                        'downward_revision_dominated', 'variables', 'constraints', 'binaries')
    for idx, (x, y) in enumerate(zip(before, after)):
        day, slot = 31 + idx // 4, (idx % 4) * 36
        assert (x['day'], x['slot']) == (y['day'], y['slot']) == (day, slot)
        for key in ('information_cutoff', 'forecast', 'original_locked', 'executed_until',
                    'residual_paths', 'training_origins', 'max_observed_index',
                    'issued_prediction_sha256'):
            assert x[key] == y[key], (scenario, day, slot, key)
        assert y['information_cutoff'] == day * 144 + slot
        assert y['executed_until'] == day * 144 + slot + 36
        assert y['solver']['feasible'] and y['solver']['status'] == 0
        for key in invariant_solver:
            assert x['solver'][key] == y['solver'][key], (scenario, day, slot, key)
        assert y['residual_paths']['label_stops_exclusive'][-1] <= day * 144
        assert not y['forecast']['known_future_price']
        row = {'scenario': scenario, 'day': day, 'slot': slot, 'matched': True}
        for key, sha in y['issued_prediction_sha256'].items():
            row[f'{key}_sha256'] = sha
        for key in ('errors_kwh_sha256', 'price_errors_sha256'):
            row[key] = y['residual_paths'][key]
        # Matching all 28 paths and the fixed original linspace rule implies
        # identical seven planning paths; their indices are recorded explicitly.
        n = len(y['training_origins'])
        chosen = np.linspace(0, n - 1, 7).astype(int)
        row['representative_history_indices'] = json.dumps(chosen.tolist())
        row['representative_history_origins'] = json.dumps(np.asarray(y['training_origins'])[chosen].tolist())
        rows.append(row)
    with np.load(baseline / f'dispatch_{scenario}.npz', allow_pickle=False) as zold, \
            np.load(directory / f'dispatch_{scenario}.npz', allow_pickle=False) as znew:
        for key in ('days', 'actual', 'price'):
            np.testing.assert_array_equal(zold[key], znew[key])
        states_changed = int(np.count_nonzero(np.abs(zold['states'][:, 0] - znew['states'][:, 0]) > 1e-6))
        source_fees = znew['fees'].sum(axis=(1, 2))
        final_soc = float(znew['states'][-1, -1])
    daily = pd.read_csv(directory / 'daily.csv')
    np.testing.assert_allclose(source_fees, daily['total_cost'], rtol=0, atol=1e-7)
    assert np.isclose(completion['total_cost'], new['recomputed_total_cost'], rtol=0, atol=1e-6)
    assert completion['complete'] and completion['config']['days'] == 334
    pd.DataFrame(rows).to_csv(directory / 'issue_hash_comparison.csv', index=False)
    battery_deltas = {k: new['battery_metrics'][k] - old['battery_metrics'][k]
                      for k in ('direction_reversals', 'charge_starts', 'discharge_starts',
                                'active_slots', 'throughput_kwh', 'equivalent_full_cycles',
                                'power_ramp_total_kw')}
    timing = {
        'exp008_run_wall_seconds': old_completion['wall_seconds'],
        'exp009_run_wall_seconds': completion['wall_seconds'],
        'exp008_lp_solver_seconds_sum': sum(r['solver']['seconds'] for r in before),
        'exp009_lp_solver_seconds_sum': sum(r['solver']['seconds'] for r in after),
        'exp008_planning_seconds_sum': sum(r['solver']['planning_seconds'] for r in before),
        'exp009_planning_seconds_sum': sum(r['solver']['planning_seconds'] for r in after),
        'exp009_daily_loop_seconds_sum': float(daily['seconds'].sum()),
        'forecast_training_seconds': 0., 'new_forecast_fits': 0,
        'interpretation': 'Wall/LP/planning are distinct nested stages; runs are from different sessions with concurrent work, so descriptive timing is not a controlled hardware speedup claim.',
    }
    result = {
        'passed': True, 'scenario': scenario, 'complete': True, 'days': 334,
        'baseline': old, 'candidate': new,
        'cost_change_yuan': new['recomputed_total_cost'] - old['recomputed_total_cost'],
        'cost_change_pct': 100 * (new['recomputed_total_cost'] / old['recomputed_total_cost'] - 1),
        'billing_component_changes': {k: new['billing'][k] - old['billing'][k] for k in old['billing']},
        'battery_metric_changes': battery_deltas,
        'episodes': {'exp008': old['battery_metrics']['charge_starts'] + old['battery_metrics']['discharge_starts'],
                     'exp009': new['battery_metrics']['charge_starts'] + new['battery_metrics']['discharge_starts']},
        'all_issue_hashes_equal': True, 'matched_issue_count': len(rows),
        'hashes_per_issue': ['load_kw', 'pv_kw', 'price', '28_net_error_paths', '28_price_error_paths'],
        'all_forecast_metadata_and_history_origins_equal': True,
        'all_frozen_planner_dimensions_and_release_weights_equal': True,
        'identity_module_qualifier_only': {'exp008': class_a, 'exp009': class_b},
        'sole_numerical_changes': {'throughput_penalty': [a['settings']['throughput_penalty'], 0.],
                                  'variation_penalty': [a['settings']['variation_penalty'], 0.],
                                  'charge_deadband_kwh': [a['deadband'], 0.]},
        'own_continuous_soc_verified': True, 'initial_soc_days_differing_from_exp008': states_changed,
        'final_soc_kwh': final_soc, 'solver_status_counts': {'optimal': len(after)},
        'solver_gap': {'kind': 'continuous_LP', 'mip_gap_not_applicable': True,
                       'max_constraint_residual': max(r['solver']['constraint_residual'] for r in after)},
        'timing': timing,
        'scope': 'Single predeclared controlled treatment on the repeatedly used 2025 development year; no untouched holdout, no candidate selection, no Q2 8% gate.',
        'planning_limitation': 'The inherited seven-path LP has scenario-dependent battery recourse; it is an optimistic approximation, not a certified nonanticipative scenario tree. Actual greedy execution is independently physically verified.',
        'source_sha256': {str(p.relative_to(ROOT)): digest(p) for p in
                          (baseline / 'audit.json', baseline / 'completion.json',
                           directory / 'audit.json', directory / 'completion.json', warmup)},
    }
    save(directory / 'independent_verification.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenario', choices=['3', '4-3'], required=True)
    result = audit_scenario(parser.parse_args().scenario)
    print(json.dumps({k: result[k] for k in ('scenario', 'passed', 'cost_change_yuan', 'cost_change_pct')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
