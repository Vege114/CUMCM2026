"""Re-run exp005 without changing any existing experimental source file."""

# These callbacks are consumed synchronously before the daily loop advances.
# ruff: noqa: B023

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import numpy as np
from scipy.sparse import save_npz

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from strict_backend import StrictBackend  # noqa: E402

from experiments.problem2.exp003.data import Data  # noqa: E402
from experiments.problem2.stochastic_lp import run as original  # noqa: E402
from experiments.problem2.stochastic_lp.model import Config, SolveError, State, settle  # noqa: E402
from experiments.problem2.stochastic_lp.scenarios import sample_paths  # noqa: E402


def hashes():
    paths = list((ROOT / 'experiments/problem2/stochastic_lp').glob('*.py'))
    paths += list((ROOT / 'experiments/problem2/exp004').glob('*.py'))
    paths += [ROOT / 'experiments/problem2/exp003/dispatch.py', ROOT / 'experiments/problem2/linear_planning/model.py']
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--days', type=int, default=334)
    parser.add_argument('--mode', choices=['scenario', 'point', 'point_scenario_control'], default='scenario')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or not 1 <= args.days <= 334:
        parser.error('new output directory and days in 1..334 required')
    args.out.mkdir(parents=True)
    save = original.save_json
    before = hashes()
    save(args.out / 'original_code_hashes_before.json', before)
    data, cfg, backend = Data(), Config(), StrictBackend()
    source = ROOT / 'data/results/exp004'
    archive = source / 'predictions.npz'
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    manifest = json.loads((source / 'prediction_manifest.json').read_text())
    assert digest == manifest['archive_sha256'] and manifest['complete']
    with np.load(archive) as z:
        forecasts, origins, masks = (z[k].copy() for k in ['no_season_seed_42', 'origins', 'no_season_mask'])
    with np.load(source / 'no_season/warmup_2.npz') as z:
        state = State(float(z['states'][-1, -1]), float((z['charge'][-1, -1] - z['discharge'][-1, -1]) * 6))
    protocol = json.loads((ROOT / 'data/results/exp005/beta0-boundary/protocol.json').read_text())
    protocol.update(config=asdict(cfg), initial_state=asdict(state), model_class='MILP', mode=args.mode,
                    strict_mutual_exclusion=True, mip_rel_gap_requested=0.0,
                    integer_tolerance_retry=[1e-6, 1e-10], numerical_retry_same_model=True,
                    user_amendment='Strictly forbid simultaneous charge/discharge; keep original experimental code unchanged; rerun and regenerate report',
                    code_access='original imported unchanged; separate solver-boundary adapter',
                    forecast_archive_sha256=digest, requested_days=args.days)
    save(args.out / 'protocol.json', protocol)
    records, details = [], []
    began = perf_counter()
    failure = None
    for i in range(args.days):
        origin, day = int(origins[i]), int(origins[i] // 144)
        day_name = str(date(2025, 1, 1) + timedelta(days=day))
        folder = args.out / day_name
        folder.mkdir()
        bundle = sample_paths(forecasts[i], forecasts[:i], data.actual[origins[:i, None] + np.arange(144)], origins[:i], origin, masks[i])
        np.savez_compressed(folder / 'scenarios.npz', **bundle)
        initial = state
        actions, observations, logs, midnight = [], [], [], []

        def observe(t):
            value = data.actual[origin + t].copy()
            observations.append(value)
            return value

        def recorded_solve(tree, prices, current_state, current_cfg, fixed_grid=None, **kwargs):
            scope = 'midnight' if fixed_grid is None else 'execution'
            slot = None if fixed_grid is None else len(actions)
            try:
                result = backend.solve_tree(tree, prices, current_state, current_cfg, fixed_grid, **kwargs)
            except SolveError:
                np.savez_compressed(folder / 'failed_input.npz', time=tree.time, parent=tree.parent,
                                    probability=tree.probability, supply_kw=tree.supply_kw, price=prices,
                                    fixed_grid=np.array([]) if fixed_grid is None else fixed_grid)
                problem = backend.last_problem
                save_npz(folder / 'failed_constraint_matrix.npz', problem['matrix'])
                np.savez_compressed(folder / 'failed_milp.npz', **{k: v for k, v in problem.items() if k != 'matrix'})
                save(folder / 'solver_failure.json', {'date': day_name, 'scope': scope, 'slot': slot,
                      'state': asdict(current_state), 'horizon': tree.horizon, 'nodes': len(tree.time),
                      'stages': backend.logs, 'status': int(backend.last_result.status)})
                raise
            if result['metadata']['max_overlap_kwh'] > cfg.tolerance:
                raise SolveError('strict solution has overlap above tolerance')
            if fixed_grid is None:
                midnight.append(result)
                np.savez_compressed(folder / 'midnight.npz', time=tree.time, parent=tree.parent,
                                    probability=tree.probability, supply_kw=tree.supply_kw,
                                    **{k: v for k, v in result.items() if k != 'metadata'})
                save(folder / 'midnight.json', result['metadata'])
            else:
                actions.append({k: float(result[k][0]) for k in ['charge', 'discharge', 'emergency', 'surplus', 'end_soc', 'net_power_kw']})
                logs.append(result['metadata'])
                if i == 0 and slot == 81:
                    np.savez_compressed(folder / 'former_failure_window.npz', time=tree.time, parent=tree.parent,
                                        probability=tree.probability, supply_kw=tree.supply_kw,
                                        **{k: v for k, v in result.items() if k != 'metadata'})
                    save(folder / 'former_failure_window.json', result['metadata'])
                if slot % 36 == 0:
                    print(f'{day_name} slot={slot} nodes={len(tree.time)} elapsed={perf_counter() - began:.1f}s', flush=True)
            return result

        try:
            with patch.object(original, 'solve_tree', recorded_solve):
                plan, detail, state, day_logs = original.replay_day(forecasts[i], bundle, data.fixed_price, observe, state, cfg, mode=args.mode)
        except SolveError as error:
            failure = json.loads((folder / 'solver_failure.json').read_text()) if (folder / 'solver_failure.json').exists() else {'reason': str(error)}
            n = len(actions)
            partial = {k: np.array([row[k] for row in actions]) for k in ['charge', 'discharge', 'emergency', 'surplus', 'net_power_kw']}
            partial['states'] = np.r_[initial.soc, [row['end_soc'] for row in actions]]
            partial['actual'] = np.array(observations[:n]).reshape(n, 2)
            partial['grid'] = midnight[0]['grid'][:n] if midnight else np.array([])
            partial['fees'] = settle(partial['grid'], partial['emergency'], data.fixed_price[:n])
            np.savez_compressed(folder / 'executed_prefix.npz', **partial)
            save(folder / 'execution_logs.json', logs)
            print(f'STOP: {day_name} completed intervals={n}: {error}', flush=True)
            break
        np.savez_compressed(folder / 'actual.npz', **detail)
        save(folder / 'execution_logs.json', day_logs)
        fees = detail['fees'].sum(axis=0)
        row = {'date': day_name, 'planned_cost': float(fees[0]), 'emergency_cost': float(fees[1]),
               'total_cost': float(fees.sum()), 'planned_kwh': float(detail['grid'].sum()),
               'emergency_kwh': float(detail['emergency'].sum()), 'initial_soc': initial.soc, 'final_soc': state.soc,
               'tv_kw': float(np.abs(np.diff(np.r_[initial.previous_power_kw, detail['net_power_kw']])).sum()),
               'plan_expected_cost': plan['metadata']['second_cost'],
               'max_overlap_kwh': float(np.minimum(detail['charge'], detail['discharge']).max())}
        records.append(row)
        details.append(detail)
        save(args.out / 'daily.json', records)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    after = hashes()
    save(args.out / 'original_code_hashes_after.json', after)
    assert before == after, 'original experimental source changed'
    if details:
        np.savez_compressed(args.out / 'dispatch.npz', **{k: np.stack([d[k] for d in details]) for k in details[0]})
    status = {'status': ('complete' if args.days == 334 else 'smoke_passed') if failure is None else 'solver_stopped',
              'failure': failure, 'completed_days': len(records), 'seconds': perf_counter() - began,
              'original_code_unchanged': before == after}
    save(args.out / 'status.json', status)
    print(json.dumps(status, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
