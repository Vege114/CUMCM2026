"""Re-solve two recorded windows with strict mutual exclusion, same inputs."""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from strict_backend import StrictBackend  # noqa: E402

from experiments.problem2.exp003.data import Data  # noqa: E402
from experiments.problem2.stochastic_lp.model import Config, State, Tree  # noqa: E402

out = Path(__file__).parent / 'evidence'
old = ROOT / 'data/results/exp005/beta0-boundary'
failure = json.loads((old / 'certification_failure.json').read_text())
z = dict(np.load(old / '2025-02-01/execution-081.npz'))
tree = Tree(z['time'], z['parent'], z['probability'], z['supply_kw'], 63)
result = StrictBackend().solve_tree(tree, Data().fixed_price[81:], State(**failure['initial_state']), Config(), z['grid'])
np.savez_compressed(out / 'same_input_strict.npz', time=z['time'], parent=z['parent'],
                    probability=z['probability'], supply_kw=z['supply_kw'],
                    **{k: v for k, v in result.items() if k != 'metadata'})
(out / 'same_input_strict.json').write_text(json.dumps(result['metadata'], ensure_ascii=False, indent=2))
print('Original LP failure, strict overlap:', result['metadata']['max_overlap_kwh'])
folder = ROOT / 'data/results/exp005/strict-mutual-exclusion/2025-02-21'
failure = json.loads((folder / 'solver_failure.json').read_text())
z = dict(np.load(folder / 'failed_input.npz'))
tree = Tree(z['time'], z['parent'], z['probability'], z['supply_kw'], failure['horizon'])
result = StrictBackend().solve_tree(tree, z['price'], State(**failure['state']), Config(), z['fixed_grid'])
np.savez_compressed(out / 'numerical_boundary_tight.npz', **{k: v for k, v in result.items() if k != 'metadata'})
(out / 'numerical_boundary_tight.json').write_text(json.dumps(result['metadata'], ensure_ascii=False, indent=2))
print('Integer-tolerance boundary, tight overlap:', result['metadata']['max_overlap_kwh'])
