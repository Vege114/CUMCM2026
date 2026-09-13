"""Shared hourly battery modes and day-ahead purchase scenario MILP.

Historical scenario recourse is still optimistic, explicitly not a full
nonanticipative policy. Shared modes at least constrain the scenario and live
controllers to the same physical direction throughout each hour.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from experiments.common.neural_v2.physics import LinearModel
from experiments.exp008.controller_candidate import Data, INITIAL_SOC
from experiments.exp008.forecast import Forecasts
from experiments.exp008.unified_forecast import UnifiedForecasts
from experiments.exp008.planner import ETA, LOW, HIGH, LIMIT, execute
from experiments.exp008.verify import verify_npz, battery_metrics

OUT = Path('data/results/exp008/mode_planning')


def plan(net_paths, price, soc, previous_mode=1, *, block_slots=6,
         switching=50., wear=.002, seconds=5., gap=.002, final=False):
    paths = np.asarray(net_paths, dtype=float)
    k, n = paths.shape
    price = np.asarray(price, dtype=float)
    nb = (n+block_slots-1)//block_slots
    m = LinearModel()
    q = m.variables((n,), cost=price)
    mode = m.variables((nb,), upper=1, integer=True)
    flip = m.variables((nb,), cost=switching)
    c = m.variables((k,n), upper=LIMIT, cost=wear/k)
    d = m.variables((k,n), upper=LIMIT, cost=wear/k)
    e = m.variables((k,n), cost=5*price[None,:]/k)
    w = m.variables((k,n))
    s = m.variables((k,n), lower=LOW, upper=HIGH)
    for idx in s[:,-1]:
        m.objective[int(idx)] = 0 if final else -float(price.min())/ETA/k
    for b in range(nb):
        terms = [(mode[b],1.)]
        rhs = float(previous_mode > 0) if b == 0 else 0.
        if b:
            terms.append((mode[b-1],-1.))
        m.constraint(terms+[(flip[b],-1.)], upper=rhs)
        m.constraint([(i,-v) for i,v in terms]+[(flip[b],-1.)], upper=-rhs)
    for j in range(k):
        for t in range(n):
            m.constraint([(q[t],1),(c[j,t],-1),(d[j,t],1),(e[j,t],1),(w[j,t],-1)],
                         paths[j,t], paths[j,t])
            terms = [(s[j,t],1),(c[j,t],-ETA),(d[j,t],1/ETA)]
            if t: terms.append((s[j,t-1],-1))
            rhs = soc if t == 0 else 0.
            m.constraint(terms, rhs, rhs)
            b = t//block_slots
            m.constraint([(c[j,t],1),(mode[b],-LIMIT)], upper=0.)
            m.constraint([(d[j,t],1),(mode[b],LIMIT)], upper=LIMIT)
    x, metadata = m.solve(seconds=seconds, gap=gap)
    if x is None:
        raise RuntimeError(f'No feasible shared-mode plan: {metadata}')
    metadata.update(method='shared_hourly_mode_scenario_milp',
                    scenario_count=k, nonanticipative_recourse_certificate=False,
                    planned_mode_changes=int(np.round(x[flip]).sum()))
    return dict(purchase=np.maximum(x[q],0),
                allowed_charge=(x[mode]>.5)[np.arange(n)//block_slots],
                metadata=metadata)


def run(name, days=30, calibration='ridge_28', switching=50.,
        scenarios=7, block_slots=6, seconds=5., deadband=0., refinement=False,
        rank_paths=False):
    directory = OUT/name
    directory.mkdir(parents=True, exist_ok=True)
    if (directory/'summary.json').exists():
        raise FileExistsError(f'Completed immutable experiment: {name}')
    cfg = dict(name=name, days=days, calibration=calibration, switching=switching,
               scenarios=scenarios, block_slots=block_slots, seconds=seconds,
               deadband=deadband, refinement=refinement, rank_paths=rank_paths,
               development_on_evaluation_year=True,
               source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (directory/'config.json').write_text(json.dumps(cfg, indent=2))
    (directory/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    forecast = UnifiedForecasts(calibration=calibration) if calibration else Forecasts()
    data = Data()
    risk=None
    if rank_paths:
        from experiments.problem2.tree_planning.risk import TreeResidualScenarios
        risk=TreeResidualScenarios(forecast.store.origins,forecast.store.values,data.actual,data.fixed_price)
    soc, mode, daily, details, audits = INITIAL_SOC, 1, [], [], []
    began = time.perf_counter()
    for day in range(31,31+days):
        issue = forecast.get(day, scenario='2')
        net = (issue['load_kw']-issue['pv_kw'])/6
        errors = forecast.net_error_paths(day, '2', limit=28)
        paths = errors['errors_kwh']
        all_demands=net[None,:]+paths
        risk_audit=None
        if risk is not None:
            support,risk_audit=risk.for_day(day)
            ranks=(rankdata(paths,axis=0)-.5)/len(paths)
            levels=(np.arange(9)+.5)/9
            all_demands=np.stack([np.interp(ranks[:,t],levels,support[t]) for t in range(144)],axis=1)
        # Preserve within-day paths, choose equally spaced past origins.
        demands=all_demands
        if len(demands) > scenarios:
            demands = demands[np.linspace(0,len(demands)-1,scenarios).astype(int)]
        result = plan(demands, data.fixed_price, soc, mode,
                      block_slots=block_slots, switching=switching,
                      seconds=seconds, final=day == 364)
        if refinement:
            from experiments.exp008.closed_loop import optimize
            refined = optimize(result['purchase'], all_demands,
                               data.fixed_price,soc,charge_mask=result['allowed_charge'],
                               throughput=.002,variation=0.,terminal=0 if day == 364 else .45,
                               maxiter=120,deadband=deadband)
            result['purchase'] = refined['purchase']
            result['metadata']['greedy_refinement'] = refined['metadata']
        actual = data.actual[day*144:(day+1)*144]
        q = result['purchase']
        detail = execute(q, actual, data.fixed_price, soc,
                         charge_deadband=deadband,
                         charge_mask=result['allowed_charge'])
        detail.update(original=q, final=q.copy(), actual=actual.copy(),
                      price=data.fixed_price.copy())
        detail['fees'] = np.stack((q*data.fixed_price,np.zeros(144),np.zeros(144),
                                  5*detail['emergency']*data.fixed_price),axis=-1)
        soc = float(detail['states'][-1])
        signs = np.sign(detail['charge']-detail['discharge'])
        nonzero = signs[signs != 0]
        if len(nonzero): mode = int(nonzero[-1])
        audits.append(dict(day=day, forecast=issue['audit'], history=errors['audit'],
                           risk=risk_audit,solver=result['metadata']))
        daily.append(dict(day=day, total_cost=float(detail['fees'].sum()),
                          plan_cost=float(detail['fees'][:,0].sum()),
                          emergency_cost=float(detail['fees'][:,3].sum()),
                          **result['metadata']))
        details.append(detail)
        if (day-30)%10 == 0:
            print(name, day-30, float(sum(d['total_cost'] for d in daily)),flush=True)
    arrays = {key:np.stack([d[key] for d in details]) for key in details[0]}
    arrays['days'] = np.arange(31,31+days)
    np.savez_compressed(directory/'dispatch.npz', **arrays)
    pd.DataFrame(daily).to_csv(directory/'daily.csv',index=False)
    (directory/'planning_audit.json').write_text(json.dumps(audits,indent=2))
    check = verify_npz(directory/'dispatch.npz',scenario='2',expected_days=days,start_day=31,
                       initial_soc=INITIAL_SOC,audit_path=directory/'planning_audit.json')
    summary = dict(config=cfg, total_cost=float(arrays['fees'].sum()),
                   elapsed_seconds=time.perf_counter()-began,
                   verification=check, battery=battery_metrics(arrays))
    (directory/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary),flush=True)
    return summary


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--name', required=True)
    p.add_argument('--days', type=int, default=30)
    p.add_argument('--calibration', default='ridge_28')
    p.add_argument('--switching', type=float, default=50.)
    p.add_argument('--scenarios', type=int, default=7)
    p.add_argument('--block-slots', type=int, default=6)
    p.add_argument('--seconds', type=float, default=5.)
    p.add_argument('--deadband', type=float, default=0.)
    p.add_argument('--refinement', action='store_true')
    p.add_argument('--rank-paths', action='store_true')
    run(**vars(p.parse_args()))
