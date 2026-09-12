"""Bounded, paired solver timing only; never advances a dispatch policy."""

import csv
import hashlib
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from gap_backend import GapBackend  # noqa: E402
from run import hashes  # noqa: E402

from experiments.problem2.exp003.data import Data  # noqa: E402
from experiments.problem2.stochastic_lp.model import Config, SolveError, State, Tree  # noqa: E402
from experiments.problem2.stochastic_lp.scenarios import information_tree  # noqa: E402

RUN = ROOT / 'data/results/exp005/strict-mutual-exclusion-certified'
OUT = ROOT / 'data/results/exp005/gap-timing-review'


def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2))


def main():
    OUT.mkdir(exist_ok=False)
    before = hashes()
    save('original_code_hashes_before.json', before)
    windows = json.loads((ROOT / '.work/exp005/runtime_windows.json').read_text())
    daily = json.loads((RUN / 'daily.json').read_text())
    dates = [row['date'] for row in daily]
    protocol = json.loads((RUN / 'protocol.json').read_text())
    price = Data().fixed_price
    groups = {'midnight': [], 'rolling_lt_0.2s': [], 'rolling_0.2_to_0.75s': [],
              'rolling_0.75_to_2s': [], 'rolling_2_to_7s': [], 'rolling_ge_7s': []}
    for window in windows:
        key = ('midnight' if window['slot'] == -1 else
               list(groups)[1 + int(np.searchsorted([.2, .75, 2., 7.], window['seconds'], side='right'))])
        groups[key].append(window)
    selected = []
    for key, rows in groups.items():
        rows.sort(key=lambda r: r['seconds'])
        # Three duration quantiles from each disjoint timing stratum.
        for fraction in [1 / 6, 1 / 2, 5 / 6]:
            row = dict(rows[min(len(rows) - 1, int(len(rows) * fraction))])
            row.update(stratum=key, stratum_count=len(rows))
            selected.append(row)
    save('design.json', {'scope': '18 paired saved windows; no policy replay or new official results',
         'gaps': [0., .0005, .001], 'physical_tolerance': 1e-6, 'time_limit_per_stage': 60,
         'delta_and_delta_exec_unchanged': .001, 'selection': 'three duration quantiles per disjoint stratum',
         'population_windows': len(windows), 'population_days': len(daily), 'selected': selected,
         'strata': {k: {'count': len(v), 'historical_seconds': sum(r['seconds'] for r in v)} for k, v in groups.items()},
         'forecast_sha256': hashlib.sha256((ROOT / 'data/results/exp004/predictions.npz').read_bytes()).hexdigest()})
    results = []
    started = perf_counter()
    for number, window in enumerate(selected):
        day, slot = window['date'], window['slot']
        index = dates.index(day)
        folder = RUN / day
        plan, actual = dict(np.load(folder / 'midnight.npz')), dict(np.load(folder / 'actual.npz'))
        previous_power = (protocol['initial_state']['previous_power_kw'] if index == 0 else
                          float(np.load(RUN / dates[index - 1] / 'actual.npz')['net_power_kw'][-1]))
        if slot == -1:
            tree = Tree(plan['time'], plan['parent'], plan['probability'], plan['supply_kw'], 144)
            state, fixed, prices = State(float(actual['states'][0]), previous_power), None, price
        else:
            bundle = dict(np.load(folder / 'scenarios.npz'))
            tree, _ = information_tree(bundle['paths_kw'], bundle['scale_kw'], slot, actual['actual'][slot])
            state = State(float(actual['states'][slot]), previous_power if slot == 0 else float(actual['net_power_kw'][slot - 1]))
            fixed, prices = plan['grid'][slot:], price[slot:]
        assert len(tree.time) == window['nodes']
        input_name = f'{day}-{slot + 1:03d}'
        np.savez_compressed(OUT / f'{input_name}.npz', time=tree.time, parent=tree.parent,
                            probability=tree.probability, supply_kw=tree.supply_kw, price=prices,
                            fixed_grid=np.array([]) if fixed is None else fixed,
                            state=np.array([state.soc, state.previous_power_kw]))
        gaps = [0., .0005, .001]
        gaps = gaps[number % 3:] + gaps[:number % 3]
        for gap in gaps:
            backend = GapBackend(gap)
            began = perf_counter()
            row = dict(window, requested_gap=gap, order=len(results))
            try:
                result = backend.solve_tree(tree, prices, state, Config(), fixed)
                row.update(success=True, metadata=result['metadata'])
                np.savez_compressed(OUT / f'{input_name}-gap-{gap}.npz',
                                    **{k: v for k, v in result.items() if k != 'metadata'})
            except SolveError as error:
                row.update(success=False, error=str(error), stages=backend.logs)
            row['wall_seconds'] = perf_counter() - began
            results.append(row)
            save('results.json', results)
            print(json.dumps({k: row[k] for k in ['date', 'slot', 'stratum', 'requested_gap', 'success', 'wall_seconds']})
                  + f' completed={len(results)}/54 elapsed={perf_counter() - started:.1f}s', flush=True)
    # A known difficult control window is stress evidence, excluded from the
    # main-policy runtime extrapolation. Keep the original sixty-second cap.
    failed = ROOT / 'data/results/exp005/strict-point-scenario-control/2025-02-23'
    failure = json.loads((failed / 'solver_failure.json').read_text())
    z = dict(np.load(failed / 'failed_input.npz'))
    tree = Tree(z['time'], z['parent'], z['probability'], z['supply_kw'], failure['horizon'])
    stress = []
    for gap in [.001, .0005]:
        backend, began = GapBackend(gap), perf_counter()
        row = {'requested_gap': gap, 'date': '2025-02-23', 'slot': 8, 'mode': 'point_scenario_control'}
        try:
            result = backend.solve_tree(tree, z['price'], State(**failure['state']), Config(), z['fixed_grid'])
            row.update(success=True, metadata=result['metadata'])
        except SolveError as error:
            row.update(success=False, error=str(error), stages=backend.logs)
        row['wall_seconds'] = perf_counter() - began
        stress.append(row)
        save('stress.json', stress)
        print(f'STRESS gap={gap} success={row["success"]} seconds={row["wall_seconds"]:.2f}', flush=True)
    after = hashes()
    assert before == after
    save('original_code_hashes_after.json', after)
    save('status.json', {'status': 'benchmark_complete', 'paired_windows': len(selected),
         'main_solves': len(results), 'stress_solves': len(stress), 'seconds': perf_counter() - started,
         'original_code_unchanged': before == after, 'full_rerun_started': False})
    with (OUT / 'timings.csv').open('w') as f:
        keys = ['date', 'slot', 'stratum', 'stratum_count', 'seconds', 'requested_gap', 'success', 'wall_seconds']
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows({k: row[k] for k in keys} for row in results)


if __name__ == '__main__':
    main()
