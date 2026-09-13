"""Continuous Q2 replay with the approved second-layer turnover penalty.

Original model, scenarios and replay code are imported without modification.
Overlap is measured, never projected away, and no longer stops this relaxed
model's simulation. Other solver, physical and information checks remain.
"""

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

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from experiments.problem2.exp003.data import Data  # noqa: E402
from experiments.problem2.stochastic_lp import model, run as original  # noqa: E402
from experiments.problem2.stochastic_lp.model import Config, SolveError, State, settle  # noqa: E402
from experiments.problem2.stochastic_lp.scenarios import sample_paths  # noqa: E402


def hashes():
    paths = list((ROOT / 'experiments/problem2/stochastic_lp').glob('*.py'))
    paths += list((ROOT / 'experiments/problem2/exp004').glob('*.py'))
    paths += [ROOT / 'experiments/problem2/exp003/dispatch.py', ROOT / 'experiments/problem2/linear_planning/model.py']
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--beta', type=float, choices=[.01, .1], required=True)
    parser.add_argument('--days', type=int, default=334)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or not 1 <= args.days <= 334:
        parser.error('new output directory and days in 1..334 required')
    args.out.mkdir(parents=True)
    save = original.save_json
    before = hashes()
    save(args.out / 'original_code_hashes_before.json', before)
    data, cfg = Data(), Config(beta=args.beta)
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
    protocol.update(config=asdict(cfg), initial_state=asdict(state), model_class='LP', mode='scenario',
                    strict_mutual_exclusion=False, overlap_policy='Measure raw overlap and continue relaxed simulation',
                    user_amendment='Remove integer switch; beta=0.1 main, beta=0.01 sensitivity; retain original code and report all overlap',
                    forecast_archive_sha256=digest, requested_days=args.days,
                    penalty_scope='Second layer normalized total charge plus discharge; also penalizes normal throughput',
                    output_semantics='Continuous-relaxation policy simulation, not a mathematical guarantee of physical exclusivity')
    protocol['D'].update(D2='B: approved beta', D4='User amendment: report overlap, continue LP simulation')
    save(args.out / 'protocol.json', protocol)
    records, details = [], []
    began = perf_counter()
    failure = None
    for i in range(args.days):
        origin = int(origins[i])
        day_name = str(date(2025, 1, 1) + timedelta(days=origin // 144))
        folder = args.out / day_name
        folder.mkdir()
        bundle = sample_paths(forecasts[i], forecasts[:i], data.actual[origins[:i, None] + np.arange(144)], origins[:i], origin, masks[i])
        np.savez_compressed(folder / 'scenarios.npz', **bundle)
        initial = state
        actions, observations, logs, midnight = [], [], [], []
        overlap_rows = []
        worst_saved = -1.

        def observe(t):
            value = data.actual[origin + t].copy()
            observations.append(value)
            return value

        def recorded_solve(tree, prices, current_state, current_cfg, fixed_grid=None, **kwargs):
            nonlocal worst_saved
            scope = 'midnight' if fixed_grid is None else 'execution'
            slot = None if fixed_grid is None else len(actions)
            try:
                result = model.solve_tree(tree, prices, current_state, current_cfg, fixed_grid, **kwargs)
            except SolveError as error:
                np.savez_compressed(folder / 'failed_input.npz', time=tree.time, parent=tree.parent,
                                    probability=tree.probability, supply_kw=tree.supply_kw, price=prices,
                                    fixed_grid=np.array([]) if fixed_grid is None else fixed_grid)
                save(folder / 'solver_failure.json', {'date': day_name, 'scope': scope, 'slot': slot,
                     'state': asdict(current_state), 'horizon': tree.horizon, 'nodes': len(tree.time), 'reason': str(error)})
                raise
            meta = result['metadata']
            meta.update(model_class='LP', strict_mutual_exclusion=False, beta=args.beta,
                        physical_acceptance_tolerance=1e-6, overlap_is_diagnostic=True)
            for layer, prefix in [(1, 'first_'), (2, '')]:
                c, d = result[prefix + 'charge'], result[prefix + 'discharge']
                overlap = np.minimum(c, d)
                stage = meta['stages'][layer - 1]
                assert stage['integer_variables'] == 0 and stage['max_constraint_residual'] <= 1e-6
                stage.update(overlap_nodes=int((overlap > cfg.tolerance).sum()),
                             expected_overlap_kwh=float(tree.probability @ overlap),
                             expected_throughput_kwh=float(tree.probability @ (c + d)),
                             current_overlap_kwh=float(overlap[0]),
                             horizon=tree.horizon)
                for node in np.flatnonzero(overlap > cfg.tolerance):
                    overlap_rows.append({'date': day_name, 'scope': scope, 'replay_slot': slot,
                         'layer': layer, 'node': int(node), 'future_slot': int(tree.time[node] + (slot or 0)),
                         'probability': float(tree.probability[node]), 'charge_kwh': float(c[node]),
                         'discharge_kwh': float(d[node]), 'overlap_kwh': float(overlap[node]),
                         'end_soc_kwh': float(result[prefix + 'end_soc'][node]),
                         'net_power_kw': float(result[prefix + 'net_power_kw'][node])})
            if meta['max_overlap_kwh'] > worst_saved:
                worst_saved = meta['max_overlap_kwh']
                np.savez_compressed(folder / 'largest_second_layer_overlap.npz', time=tree.time, parent=tree.parent,
                                    probability=tree.probability, supply_kw=tree.supply_kw,
                                    **{k: v for k, v in result.items() if k != 'metadata'})
                save(folder / 'largest_second_layer_overlap.json', {'scope': scope, 'slot': slot,
                      'state': asdict(current_state), 'metadata': meta})
            if fixed_grid is None:
                midnight.append(result)
                np.savez_compressed(folder / 'midnight.npz', time=tree.time, parent=tree.parent,
                                    probability=tree.probability, supply_kw=tree.supply_kw,
                                    **{k: v for k, v in result.items() if k != 'metadata'})
                save(folder / 'midnight.json', meta)
            else:
                actions.append({k: float(result[k][0]) for k in ['charge', 'discharge', 'emergency', 'surplus', 'end_soc', 'net_power_kw']})
                logs.append(meta)
            return result

        try:
            with patch.object(original, 'solve_tree', recorded_solve):
                plan, detail, state, day_logs = original.replay_day(forecasts[i], bundle, data.fixed_price, observe, state, cfg)
        except SolveError as error:
            failure = json.loads((folder / 'solver_failure.json').read_text())
            n = len(actions)
            partial = {k: np.array([r[k] for r in actions]) for k in ['charge', 'discharge', 'emergency', 'surplus', 'net_power_kw']}
            partial.update(states=np.r_[initial.soc, [r['end_soc'] for r in actions]],
                           actual=np.array(observations[:n]).reshape(n, 2),
                           grid=midnight[0]['grid'][:n] if midnight else np.array([]))
            partial['fees'] = settle(partial['grid'], partial['emergency'], data.fixed_price[:n])
            np.savez_compressed(folder / 'executed_prefix.npz', **partial)
            save(folder / 'execution_logs.json', logs)
            save(folder / 'overlap_nodes.json', overlap_rows)
            print(f'STOP {day_name} slot={n}: {error}', flush=True)
            break
        np.savez_compressed(folder / 'actual.npz', **detail)
        save(folder / 'execution_logs.json', day_logs)
        save(folder / 'overlap_nodes.json', overlap_rows)
        fees = detail['fees'].sum(axis=0)
        overlap = np.minimum(detail['charge'], detail['discharge'])
        row = {'date': day_name, 'beta': args.beta, 'planned_cost': float(fees[0]),
               'emergency_cost': float(fees[1]), 'total_cost': float(fees.sum()),
               'planned_kwh': float(detail['grid'].sum()), 'emergency_kwh': float(detail['emergency'].sum()),
               'initial_soc': initial.soc, 'final_soc': state.soc,
               'previous_power_kw': initial.previous_power_kw, 'final_power_kw': state.previous_power_kw,
               'tv_kw': float(np.abs(np.diff(np.r_[initial.previous_power_kw, detail['net_power_kw']])).sum()),
               'plan_expected_cost': plan['metadata']['second_cost'],
               'overlap_intervals': int((overlap > cfg.tolerance).sum()),
               'overlap_kwh': float(overlap.sum()), 'max_overlap_kwh': float(overlap.max()),
               'throughput_kwh': float((detail['charge'] + detail['discharge']).sum()),
               'solver_seconds': sum(s['seconds'] for m in [plan['metadata'], *day_logs] for s in m['stages'])}
        records.append(row)
        details.append(detail)
        save(args.out / 'daily.json', records)
        save(args.out / 'progress.json', {'status': 'running', 'completed_days': len(records),
             'last_completed_date': day_name, 'next_day_index': i + 1,
             'next_day_initial_state': asdict(state), 'seconds': perf_counter() - began})
        print(json.dumps(row, ensure_ascii=False) + f' elapsed={perf_counter() - began:.2f}s', flush=True)
    after = hashes()
    save(args.out / 'original_code_hashes_after.json', after)
    assert before == after
    if details:
        np.savez_compressed(args.out / 'dispatch.npz', **{k: np.stack([d[k] for d in details]) for k in details[0]})
    status = {'status': ('complete' if args.days == 334 else 'smoke_passed') if failure is None else 'solver_stopped',
              'failure': failure, 'completed_days': len(records), 'seconds': perf_counter() - began,
              'beta': args.beta, 'original_code_unchanged': True,
              'overlap_policy': 'Report without projection; continuous simulation'}
    save(args.out / 'status.json', status)
    print(json.dumps(status, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
