"""Same saved inputs, original continuous LP, second-layer turnover weights."""

import csv
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent.parent / 'strict_rerun'))
from run import hashes  # noqa: E402

from experiments.problem2.exp003.data import Data  # noqa: E402
from experiments.problem2.stochastic_lp.model import Config, SolveError, State, Tree, solve_tree  # noqa: E402

OUT = ROOT / 'data/results/exp005/soft-penalty-preflight'


def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2))


def main():
    OUT.mkdir(exist_ok=False)
    before = hashes()
    save('original_code_hashes_before.json', before)
    prior = ROOT / 'data/results/exp005/gap-timing-review'
    design = json.loads((prior / 'design.json').read_text())
    inputs = []
    for window in design['selected']:
        name = f'{window["date"]}-{window["slot"] + 1:03d}'
        z = dict(np.load(prior / f'{name}.npz'))
        inputs.append((name, z, window))
    old = ROOT / 'data/results/exp005/beta0-boundary'
    failure = json.loads((old / 'certification_failure.json').read_text())
    z = dict(np.load(old / '2025-02-01/execution-081.npz'))
    z.update(price=Data().fixed_price[81:], fixed_grid=z['grid'],
             state=np.array(list(failure['initial_state'].values())))
    inputs.append(('original-lp-overlap', z, {'scope': 'original_failed_input'}))
    folder = ROOT / 'data/results/exp005/strict-point-scenario-control/2025-02-23'
    f = json.loads((folder / 'solver_failure.json').read_text())
    z = dict(np.load(folder / 'failed_input.npz'))
    z.update(state=np.array([f['state']['soc'], f['state']['previous_power_kw']]))
    inputs.append(('strict-control-timeout', z, {'scope': 'strict_failed_input'}))
    rows = []
    began = perf_counter()
    for name, z, context in inputs:
        tree = Tree(z['time'], z['parent'], z['probability'], z['supply_kw'], len(z['price']))
        np.savez_compressed(OUT / f'{name}-input.npz', **z)
        for beta in [0., .001, .01, .1, 1.]:
            started = perf_counter()
            row = dict(window=name, beta=beta, context=context)
            try:
                result = solve_tree(tree, z['price'], State(*z['state']), Config(beta=beta),
                                    z['fixed_grid'] if len(z['fixed_grid']) else None)
                overlap = np.minimum(result['charge'], result['discharge'])
                throughput = result['charge'] + result['discharge']
                row.update(success=True, metadata=result['metadata'], nodes=len(tree.time),
                           overlap_nodes=int((overlap > 1e-6).sum()),
                           max_overlap_kwh=float(overlap.max()),
                           expected_overlap_kwh=float(tree.probability @ overlap),
                           expected_throughput_kwh=float(tree.probability @ throughput),
                           current_overlap_kwh=float(overlap[0]))
                assert all(s['max_constraint_residual'] <= 1e-6 for s in result['metadata']['stages'])
                assert result['metadata']['stage_count'] == 2
                assert all(s['integer_variables'] == 0 for s in result['metadata']['stages'])
                np.savez_compressed(OUT / f'{name}-beta-{beta}.npz',
                                    **{k: v for k, v in result.items() if k != 'metadata'})
            except SolveError as error:
                row.update(success=False, error=str(error))
            row['wall_seconds'] = perf_counter() - started
            rows.append(row)
            save('results.json', rows)
        print(f'{name} completed={len(rows)}/{len(inputs) * 5} seconds={perf_counter() - began:.2f}', flush=True)
    assert hashes() == before
    save('original_code_hashes_after.json', hashes())
    save('status.json', {'status': 'preflight_complete', 'windows': len(inputs), 'solves': len(rows),
                        'seconds': perf_counter() - began, 'original_code_unchanged': True,
                        'model_class': 'LP', 'full_rerun_started': False,
                        'penalty': 'normalized throughput, second layer only', 'betas': [0., .001, .01, .1, 1.]})
    with (OUT / 'summary.csv').open('w') as stream:
        keys = ['window', 'beta', 'success', 'nodes', 'overlap_nodes', 'max_overlap_kwh',
                'expected_overlap_kwh', 'current_overlap_kwh', 'expected_throughput_kwh', 'wall_seconds']
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows({k: r.get(k) for k in keys} for r in rows)


if __name__ == '__main__':
    main()
