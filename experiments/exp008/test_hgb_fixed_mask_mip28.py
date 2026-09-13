"""Independent input/selection replay for the five archived MIP diagnostics.

Uses the existing feasible MIP candidates: does not rerun solver or extend any
time limit. Future perturbation checks legal inputs and historical selection.
"""
import copy
import json
import numpy as np

from experiments.exp008.hgb_fixed_mask_mip28_diagnostic import (
    OUT, SOURCE, DAYS, AbsoluteHGBForecasts, Data, ROOT, array_hash, digest,
    greedy_independent, physical_check, save,
)


def main():
    rows = []
    for day in DAYS:
        directory = OUT / f'day_{day}'
        selection = json.loads((directory / 'selection_locked_before_actual.json').read_text())
        with np.load(SOURCE / f'planning_day{day}.npz') as z:
            paths = z['all_net_paths'].copy()
            soc, mask, price = float(z['initial_soc']), z['allowed_charge'].astype(bool), z['price'].copy()
        original, changed = AbsoluteHGBForecasts(), AbsoluteHGBForecasts()
        changed.data = copy.copy(changed.data)
        changed.data.actual = changed.data.actual.copy()
        changed.data.actual[day * 144:] += np.array([90000., 70000., 5000.])
        changed.store = copy.copy(changed.store)
        changed.store.values = changed.store.values.copy()
        changed.store.values[changed.store.origins > day * 144] += 80000.
        history0, history1 = original.net_error_paths(day), changed.net_error_paths(day)
        issue0, issue1 = original.get(day), changed.get(day)
        np.testing.assert_array_equal(history0['errors_kwh'], history1['errors_kwh'])
        for key in ('load_kw', 'pv_kw', 'price'):
            np.testing.assert_array_equal(issue0[key], issue1[key])
        rebuilt = (issue1['load_kw'] - issue1['pv_kw'])[None] / 6 + history1['errors_kwh']
        np.testing.assert_array_equal(paths, rebuilt)
        terminal = 0. if day == 364 else .45
        with np.load(directory / 'original_historical_greedy.npz') as z:
            q_old = z['purchase'].copy()
            old, old_arrays = greedy_independent(q_old, rebuilt, price, soc, mask, terminal)
            for key, value in old_arrays.items():
                np.testing.assert_array_equal(value, z[key])
        with np.load(directory / 'new_historical_greedy.npz') as z:
            q_new = z['purchase'].copy()
            new, new_arrays = greedy_independent(q_new, rebuilt, price, soc, mask, terminal)
            for key, value in new_arrays.items():
                np.testing.assert_array_equal(value, z[key])
        selected = q_new if new['objective'] < old['objective'] else q_old
        assert array_hash(selected) == selection['selected_purchase_sha256']
        with np.load(directory / 'mip28_historical_scenarios.npz') as z:
            check = physical_check(z['purchase'], rebuilt, mask, z['scenario_states'], z['scenario_charge'],
                z['scenario_discharge'], z['scenario_emergency'], z['scenario_surplus'])
        assert check['passed']
        rows.append({'day': day, 'future_actual_and_future_forecasts_mutated': True,
            'midnight_and_all28_history_paths_unchanged': True,
            'historical_greedy_all_arrays_exact': True, 'selection_unchanged': True,
            'MIP_scenario_physics_rechecked': True, 'solver_repeated': False})
    protocol = json.loads((OUT / 'protocol.json').read_text())
    assert all(digest(path) == h for path, h in protocol['input_sha256'].items())
    assert all(digest(ROOT / path) == h for path, h in protocol['source_sha256'].items())
    save(OUT / 'independent_future_and_selection_check.json', {'passed': True, 'cases': rows,
        'all_original_sources_and_archives_unchanged': True,
        'solver_future_invariance_scope': 'same exact problem inputs, no repeated MIP solving',
        'not_continuous_policy_evaluation': True})
    print('Passed: five future-mutation problem-input checks, historical selections, full greedy arrays and MIP physical rechecks.')


if __name__ == '__main__':
    main()
