"""Run Q2 or Q4-2 continuously from 2025-01-01 with initial SOC 6000 kWh.

Use --scenario 2 or --scenario 4-2 and --days 365 for a full-year run.
Daily files retain boundary states and support resuming the same output directory.
Use --out to select a new directory when recomputing archived results.
"""
import argparse
import hashlib
import json
import platform
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import scipy

from experiments.common.neural_v2.data import ROOT
from experiments.exp008.closed_loop import optimize
from experiments.exp008.mode_budget_hold1_physical import plan as q2_plan
from experiments.exp008.mode_planning_physical import plan as q42_plan
from experiments.exp008.planner import execute
from experiments.exp008.verify import verify_arrays, battery_metrics
from experiments.exp009.q12_forecast import YearForecasts


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def read(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def run(scenario, days, output=None):
    out = ROOT/'data/results/exp009'/('q2' if scenario == '2' else 'q4_2') if output is None else Path(output)
    out.mkdir(parents=True, exist_ok=True)
    chunks = out/'daily_chunks'
    chunks.mkdir(exist_ok=True)
    fc = YearForecasts(scenario)
    config = {'scenario': scenario, 'start_day': 0, 'initial_soc_kwh': 6000.,
        'initial_mode': 1, 'initial_mode_age': 1, 'mode_boundary': 'planned mode and age' if scenario=='2' else 'previous nonidle actual mode',
        'seed': 42, 'mip_seconds': 5., 'target_mip_gap': .002, 'scenario_count_up_to': 3,
        'refinement_history_days': 28, 'refinement_maxiter': 120, 'throughput': .002,
        'deadband': 0., 'terminal_value_refinement': .45, 'year_end_terminal_value': 0.,
        'forecast_path': str(fc.store.path.relative_to(ROOT)), 'forecast_sha256': fc.store.sha256,
        'cold_start': 'day0 attachment1; January load lag7/fallback lag1 and PV lag1; own completed daily errors including day0; no dispatch warmup',
        'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__).resolve(), ROOT/'experiments/exp009/q12_forecast.py',
                ROOT/'experiments/exp008/closed_loop.py', ROOT/'experiments/exp008/mode_budget_hold1_physical.py',
                ROOT/'experiments/exp008/mode_planning_physical.py', ROOT/'experiments/exp008/planner.py')},
        'raw_input_sha256': fc.data.hashes, 'python': platform.python_version(),
        'numpy': np.__version__, 'scipy': scipy.__version__, 'threads': 1}
    if (out/'protocol.json').exists():
        assert json.loads((out/'protocol.json').read_text()) == config, 'changed protocol; choose fresh output'
    else:
        write(out/'protocol.json', config)
    parts, audits, rows = [], [], []
    soc, mode, age = 6000., 1, 1
    began = perf_counter()
    for day in range(days):
        part_path, audit_path = chunks/f'day_{day:03d}.npz', chunks/f'day_{day:03d}.json'
        if part_path.exists() and audit_path.exists():
            detail, record = read(part_path), json.loads(audit_path.read_text())
            assert abs(float(detail['states'][0])-soc)<1e-7
            assert record['boundary_before'] == [mode, age]
        else:
            start = perf_counter()
            issue, history = fc.get(day), fc.net_error_paths(day)
            paths = (issue['load_kw']-issue['pv_kw'])[None, :]/6+history['errors_kwh']
            selected = np.linspace(0, len(paths)-1, min(3, len(paths))).astype(int)
            price = issue['price']
            if scenario == '2':
                planned = q2_plan(paths[selected], price, soc, mode, age, final=day==364)
            else:
                planned = q42_plan(paths[selected], price, soc, mode, block_slots=6,
                    switching=50., wear=.002, seconds=5., gap=.002, final=day==364)
            refined = optimize(planned['purchase'], paths, price, soc,
                charge_mask=planned['allowed_charge'], throughput=.002, variation=0.,
                terminal=0. if day==364 else .45, deadband=0., maxiter=120)
            q = refined['purchase']
            observed = fc.data.actual[day*144:(day+1)*144]
            detail = execute(q, observed[:, :2], price, soc,
                charge_mask=planned['allowed_charge'], charge_deadband=0.)
            real_price = price if scenario == '2' else observed[:, 2]
            detail.update(original=q, final=q.copy(), actual=observed.copy(), price=real_price.copy(),
                allowed_charge=planned['allowed_charge'].copy())
            detail['fees'] = np.stack((q*real_price, np.zeros(144), np.zeros(144),
                5*detail['emergency']*real_price), axis=-1)
            if scenario == '2':
                next_mode, next_age = planned['metadata']['final_planned_mode'], planned['metadata']['final_planned_run_slots']
            else:
                signs = np.sign(detail['charge']-detail['discharge'])
                next_mode, next_age = int(signs[signs!=0][-1]) if np.any(signs) else mode, 1
            record = {'day': day, 'information_cutoff': day*144, 'forecast': issue['audit'],
                'history': history['audit'], 'mip': planned['metadata'], 'refinement': refined['metadata'],
                'boundary_before': [mode, age], 'boundary_after': [next_mode, next_age],
                'seconds': perf_counter()-start, 'purchase_locked_before_current_actual_read': True}
            np.savez_compressed(chunks/f'planning_{day:03d}.npz',
                **{k: v for k, v in planned.items() if k != 'metadata'},
                all_net_paths=paths, selected_scenario_indices=selected,
                predicted_price=price, refined_purchase=q, initial_soc=soc)
            np.savez_compressed(part_path, **detail)
            write(audit_path, record)
        soc = float(detail['states'][-1])
        mode, age = record['boundary_after']
        parts.append(detail); audits.append(record)
        rows.append({'day': day, 'date': str((pd.Timestamp('2025-01-01')+pd.Timedelta(days=day)).date()),
            'total_cost': float(detail['fees'].sum()), 'planned_cost': float(detail['fees'][:,0].sum()),
            'emergency_cost': float(detail['fees'][:,3].sum()), 'emergency_kwh': float(detail['emergency'].sum()),
            'initial_soc': float(detail['states'][0]), 'final_soc': soc,
            'mip_gap': record['mip']['mip_gap'], 'mip_seconds': record['mip']['seconds'],
            'refinement_seconds': record['refinement']['planning_seconds']})
        if day < 3 or (day+1)%10 == 0:
            print(scenario, 'DAY', day, 'cost', rows[-1]['total_cost'], 'seconds', round(perf_counter()-began, 1), flush=True)
    arrays = {k: np.stack([part[k] for part in parts]) for k in parts[0]}
    arrays['days'] = np.arange(days)
    np.savez_compressed(out/'dispatch.npz', **arrays)
    write(out/'planning_audit.json', audits)
    pd.DataFrame(rows).to_csv(out/'daily.csv', index=False)
    actual = fc.data.actual[:days*144].reshape(days, 144, 3)
    check = verify_arrays(arrays, scenario, expected_days=days, start_day=0, initial_soc=6000.,
        initial_mode=0, initial_power_kw=0., source_actual=actual,
        source_price=fc.data.fixed_price if scenario=='2' else actual[:,:,2], audit_records=audits)
    assert check['passed'], check['errors']
    write(out/'verification.json', check)
    summary = {'scenario': scenario, 'days': days, 'complete': days==365,
        'full_year_cost': float(arrays['fees'].sum()), 'january_cost': float(arrays['fees'][:31].sum()),
        'feb_dec_cost': float(arrays['fees'][31:].sum()), 'feb_initial_soc': float(arrays['states'][31,0]) if days>31 else None,
        'last_soc': soc, 'battery': battery_metrics(arrays, initial_mode=0, initial_power_kw=0.),
        'mip_seconds_sum': sum(r['mip_seconds'] for r in rows),
        'refinement_seconds_sum': sum(r['refinement_seconds'] for r in rows),
        'mip_gap_max': max(r['mip_gap'] for r in rows), 'mip_gap_mean': float(np.mean([r['mip_gap'] for r in rows])),
        'verified': True, 'forecast_retrained': False}
    if days>31:
        subset={k:v[31:] for k,v in arrays.items()}
        summary['feb_dec_battery'] = battery_metrics(subset, initial_mode=0, initial_power_kw=0.)
    write(out/'summary.json', summary)
    print('DONE', json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenario', choices=['2', '4-2'], required=True)
    parser.add_argument('--days', type=int, default=365)
    parser.add_argument('--out', type=Path)
    args=parser.parse_args()
    run(args.scenario, args.days, args.out)
