"""Exploratory causal Q2 controllers; fixed forecasts and same exp006 physics.

This is development on the reported year, not an untouched validation set.
No actual value from the execution day is available to ``plan_inventory``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import lil_matrix, coo_matrix
from scipy.stats import rankdata

from experiments.problem2.exp003.data import Data
from experiments.problem2.exp004.predict import ForecastStore
from experiments.problem2.tree_planning.model import ETA, MIN_SOC, MAX_SOC, POWER_ENERGY
from experiments.problem2.tree_planning.risk import TreeResidualScenarios
from experiments.problem2.tree_planning.verify import battery_metrics

OUT = Path(__file__).resolve().parents[2] / 'data/results/exp008/controller_candidates'
INITIAL_SOC = 1421.7991105135516
BASELINE = 14066257.477300335
CHEAPEST = 13534196.0


def charging_mask(spec, net, price):
    t = np.arange(len(price))
    start, stop = spec.get('solar_start', 54), spec.get('solar_stop', 96)
    return (t < 34) | ((t >= start) & (t < stop)) | (t >= 131)


def plan_inventory(support, price, soc, spec, final=False, error_paths=None, base_forecast=None):
    """Continuous inventory LP with separate AC flows and physical efficiency."""
    if spec.get('planner') == 'joint':
        from experiments.exp008.planner import plan, Settings
        keys = Settings.__dataclass_fields__
        settings = Settings(**{k:v for k,v in spec.items() if k in keys})
        if spec.get('path_type') == 'rank' and len(error_paths) > 1:
            ranks = (rankdata(error_paths, axis=0)-.5)/len(error_paths)
            quantile_levels = (np.arange(9)+.5)/9
            net = np.median(support,axis=1)
            demands = np.stack([np.interp(ranks[:,t],quantile_levels,support[t]) for t in range(144)],axis=1)
            error_paths = demands-net
            forecast = {'load_kw':np.maximum(net,0)*6,'pv_kw':np.maximum(-net,0)*6,'price':price}
        else:
            forecast = {'load_kw':base_forecast[:,0], 'pv_kw':base_forecast[:,1], 'price':price}
            net = (base_forecast[:,0]-base_forecast[:,1])/6
        allowed_charge = charging_mask(spec,net,price)
        if spec.get('controller') == 'adaptive_blocks':
            reference = plan_inventory(support,price,soc,{'quantile':spec.get('quantile',.8),'wear':spec.get('mode_wear',.02)},final)
            direction = np.sign(reference['charge']-reference['discharge'])
            last = 1
            for t in range(len(direction)):
                last = direction[t] or last
                direction[t] = last
            hold = spec.get('mode_min_slots',3)
            starts = np.r_[0,np.flatnonzero(np.diff(direction))+1,len(direction)]
            for a,b in zip(starts[:-1],starts[1:]):
                if b-a < hold and a:
                    direction[a:b] = direction[a-1]
            allowed_charge = direction > 0
        mask = allowed_charge if spec.get('controller') in ('fixed_blocks','adaptive_blocks') else None
        result = plan(forecast,soc,error_paths=error_paths,settings=settings,final_day=final,charge_mask=mask)
        result.update(net=net,allowed_charge=allowed_charge)
        return result
    n = len(price)
    # Variable blocks: q, charge, discharge, spill, next SOC.
    obj = np.r_[price, np.full(n, spec.get('wear', .002)),
                np.full(n, spec.get('wear', .002)), np.full(n, 1e-7), np.zeros(n)]
    obj[-1] = 0 if final else -float(price.min()) / ETA
    eq = lil_matrix((2*n, 5*n))
    quantile = spec.get('quantile', .7)
    net = np.quantile(support, quantile, axis=1)
    rhs = np.r_[net, np.zeros(n)]
    rhs[n] = soc
    bounds = [(0, None)] * n + [(0, POWER_ENERGY)] * (2*n) + [(0, None)] * n + [(MIN_SOC, MAX_SOC)] * n
    # Pool forecast uncertainty in stored energy and empty capacity instead of
    # purchasing a separate safety quantile at every expensive ten-minute slot.
    reserve=spec.get('soc_reserve',0.)
    headroom=spec.get('soc_headroom',0.)
    for t in range(n):
        bottom=min(MIN_SOC+reserve,soc+ETA*POWER_ENERGY*(t+1))
        top=max(MAX_SOC-headroom,soc-POWER_ENERGY*(t+1)/ETA)
        if final:
            bottom=max(MIN_SOC,bottom-reserve*max(0,(t-120)/23))
        if bottom>top:
            raise ValueError('Reserve and headroom leave no feasible inventory interval')
        bounds[4*n+t]=(bottom,top)
    allowed_charge = charging_mask(spec, net, price)
    if spec.get('controller') == 'fixed_blocks':
        for t in range(n):
            bounds[(2 if allowed_charge[t] else 1)*n+t] = (0, 0)
    for t in range(n):
        eq[t, t] = 1
        eq[t, n+t] = -1
        eq[t, 2*n+t] = 1
        eq[t, 3*n+t] = -1
        eq[n+t, 4*n+t] = 1
        eq[n+t, n+t] = -ETA
        eq[n+t, 2*n+t] = 1/ETA
        if t:
            eq[n+t, 4*n+t-1] = -1
    result = linprog(obj, A_eq=eq.tocsr(), b_eq=rhs, bounds=bounds, method='highs')
    if not result.success:
        raise RuntimeError(result.message)
    x = result.x
    return {'purchase': x[:n], 'charge': x[n:2*n], 'discharge': x[2*n:3*n],
            'states': np.r_[soc, x[4*n:]], 'net': net, 'allowed_charge': allowed_charge}


def execute_inventory(plan, actual, price, soc, mode, spec):
    n = len(price)
    q = plan['purchase']
    c, d, e, w = (np.zeros(n) for _ in range(4))
    states = np.r_[soc, np.zeros(n)]
    min_charge = spec.get('charge_deadband', 0.)
    min_discharge = spec.get('discharge_deadband', 0.)
    reserve = spec.get('reserve', 0.)
    for t in range(n):
        # Current observation only. Whole actual array is merely replay input.
        balance = q[t] + (actual[t, 1] - actual[t, 0])/6
        if balance >= 0:
            request = min(balance, POWER_ENERGY, (MAX_SOC-states[t])/ETA)
            if spec.get('controller') == 'planned_direction' and plan['charge'][t] < 1e-5:
                request = 0
            if spec.get('controller') == 'macro' and not (t < 42 or 60 <= t < 96 or t >= 138):
                request = 0
            if spec.get('controller') in ('fixed_blocks','adaptive_blocks') and not plan['allowed_charge'][t]:
                request = 0
            threshold = min_charge if mode == -1 else spec.get('holding_deadband', 0.)
            c[t] = request if request >= threshold else 0
            w[t] = balance-c[t]
        else:
            floor = MIN_SOC
            if reserve and price[t] < np.max(price[t:]):
                floor = min(MAX_SOC, MIN_SOC + reserve)
            if 'state_buffer' in spec:
                floor = max(MIN_SOC, plan['states'][t+1]-spec['state_buffer'])
            request = min(-balance, POWER_ENERGY, max(0, (states[t]-floor)*ETA))
            if spec.get('controller') in ('fixed_blocks','adaptive_blocks') and plan['allowed_charge'][t]:
                request = 0
            d[t] = request if request >= min_discharge else 0
            e[t] = -balance-d[t]
        states[t+1] = states[t] + ETA*c[t] - d[t]/ETA
        if c[t] or d[t]:
            mode = 1 if c[t] else -1
    fees = np.stack([q*price, np.zeros(n), np.zeros(n), 5*e*price], axis=-1)
    detail = dict(original=q, final=q.copy(), charge=c, discharge=d, emergency=e,
                  surplus=w, states=states, fees=fees, actual=actual.copy(), price=price.copy(),
                  intended_charge=plan['charge'], intended_discharge=plan['discharge'],
                  intended_states=plan['states'])
    return detail, mode


def risk_cache(data, store):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT/'risk_inputs.npz'
    if path.exists():
        with np.load(path) as z:
            return z['supports'].copy(), json.loads((OUT/'risk_audit.json').read_text())
    risk = TreeResidualScenarios(store.origins, store.values, data.actual, data.fixed_price)
    support, audit = [], []
    for day in range(31, 365):
        values, info = risk.for_day(day)
        support.append(values)
        audit.append(info)
    np.savez_compressed(path, supports=np.stack(support))
    (OUT/'risk_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2))
    return np.stack(support), audit


def perfect_information_bound(data):
    """Full-horizon perfect-information relaxation; never an eligible controller."""
    started = time.perf_counter()
    actual = data.actual[31*144:]
    net = (actual[:,0]-actual[:,1])/6
    n = len(net)
    p = np.tile(data.fixed_price,334)
    t = np.arange(n)
    rows = np.r_[t,t,t,t,n+t,n+t,n+t,n+t[1:]]
    cols = np.r_[t,n+t,2*n+t,3*n+t,4*n+t,n+t,2*n+t,4*n+t[:-1]]
    vals = np.r_[np.ones(n),-np.ones(n),np.ones(n),-np.ones(n),np.ones(n),
                 np.full(n,-ETA),np.full(n,1/ETA),-np.ones(n-1)]
    mat = coo_matrix((vals,(rows,cols)),shape=(2*n,5*n)).tocsr()
    rhs = np.r_[net,np.zeros(n)]
    rhs[n] = INITIAL_SOC
    bounds = [(0,None)]*n+[(0,POWER_ENERGY)]*(2*n)+[(0,None)]*n+[(MIN_SOC,MAX_SOC)]*n
    result = linprog(np.r_[p,np.zeros(4*n)],A_eq=mat,b_eq=rhs,bounds=bounds,method='highs')
    if not result.success:
        raise RuntimeError(result.message)
    summary = {'role':'ineligible perfect-information lower bound','total_cost':float(result.fun),
               'days':334,'initial_soc':INITIAL_SOC,'final_soc':float(result.x[-1]),
               'max_savings_vs_exp006_pct':100*(1-result.fun/BASELINE),
               'elapsed_seconds':time.perf_counter()-started,'uses_future_actuals':True}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'perfect_information_bound.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary),flush=True)
    return summary


def evaluate(spec, data=None, supports=None, audits=None):
    data = Data() if data is None else data
    if supports is None:
        supports, audits = risk_cache(data, ForecastStore('no_season', seed=42))
    run_dir = OUT/spec['id']
    if (run_dir/'summary.json').exists():
        cached = json.loads((run_dir/'summary.json').read_text())
        if cached['spec'] != spec:
            raise ValueError(f"Refusing to reuse run id with changed settings: {spec['id']}")
        return cached
    start = time.perf_counter()
    source_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    soc, mode, details, daily, execution_audit = INITIAL_SOC, 1, [], [], []
    store = ForecastStore('no_season',seed=42) if spec.get('planner') == 'joint' or spec.get('optimize') else None
    if spec.get('calibration'):
        from experiments.exp008.forecast_calibration import CalibratedStore
        store = CalibratedStore(spec['calibration'])
    stop_day = 31+spec.get('development_days',334)
    for i, day in enumerate(range(31,stop_day)):
        errors, forecast = None, None
        if store is not None:
            ids = np.flatnonzero(store.origins+144 <= day*144)[-28:]
            forecast = store.get(day*144)
            if len(ids):
                observed = data.actual[store.origins[ids,None]+np.arange(144)]
                diff = (observed-store.values[ids])/6
                errors = diff[:,:,0]-diff[:,:,1]
            else:
                net = (forecast[:,0]-forecast[:,1])/6
                errors = supports[i].T-net[None,:]
        plan = plan_inventory(supports[i], data.fixed_price, soc, spec, final=day == 364,
                              error_paths=errors,base_forecast=forecast)
        if spec.get('optimize'):
            from experiments.exp008.closed_loop import optimize
            net = (forecast[:,0]-forecast[:,1])/6
            paths = net[None,:] + errors
            if spec.get('path_type') == 'rank' and len(errors)>1:
                ranks = (rankdata(errors,axis=0)-.5)/len(errors)
                levels = (np.arange(9)+.5)/9
                paths = np.stack([np.interp(ranks[:,t],levels,supports[i,t]) for t in range(144)],axis=1)
            max_paths = spec.get('scenarios',28)
            if len(paths)>max_paths:
                paths = paths[np.linspace(0,len(paths)-1,max_paths).astype(int)]
            mask = plan['allowed_charge'] if spec.get('controller') in ('fixed_blocks','adaptive_blocks') else None
            result = optimize(plan['purchase'],paths,data.fixed_price,soc,charge_mask=mask,
                              throughput=spec.get('wear',.005),variation=spec.get('variation_penalty',.001),
                              terminal=0 if day==364 else data.fixed_price.min()/ETA,
                              deadband=spec.get('charge_deadband',0),maxiter=spec.get('maxiter',100),
                              emergency_weight=spec.get('emergency_weight',5))
            plan['purchase'] = result['purchase']
            plan['optimizer_metadata'] = result['metadata']
        old_mode = mode
        if spec.get('impulse_feedback'):
            from experiments.exp008.impulse_feedback import value_functions, execute as impulse_execute
            grid, values, value_meta = value_functions(plan['purchase'], supports[i], data.fixed_price,
                grid_kwh=spec.get('feedback_grid',100.),wear=spec.get('wear',.002),
                switching=spec.get('switching',50.),terminal=0 if day==364 else data.fixed_price.min()/ETA)
            detail,mode = impulse_execute(plan['purchase'],data.actual[day*144:(day+1)*144],data.fixed_price,
                soc,grid,values,wear=spec.get('wear',.002),switching=spec.get('switching',50.),initial_mode=mode)
            plan['optimizer_metadata'] = value_meta
        elif spec.get('value_feedback'):
            from experiments.exp008.battery_feedback import value_functions, execute as value_execute
            mask = plan['allowed_charge'] if spec.get('controller') in ('fixed_blocks','adaptive_blocks') else None
            grid, values, value_meta = value_functions(plan['purchase'], supports[i], data.fixed_price,
                grid_kwh=spec.get('feedback_grid',100.),wear=spec.get('wear',.002),charge_mask=mask,
                terminal=0 if day==364 else data.fixed_price.min()/ETA,
                charge_deadband=spec.get('charge_deadband',0.))
            detail = value_execute(plan['purchase'],data.actual[day*144:(day+1)*144],data.fixed_price,
                soc,grid,values,wear=spec.get('wear',.002),charge_mask=mask,
                charge_deadband=spec.get('charge_deadband',0.))
            signs=np.sign(detail['charge']-detail['discharge'])
            if np.any(signs):
                mode=int(signs[signs!=0][-1])
            plan['optimizer_metadata'] = value_meta
        else:
            detail, mode = execute_inventory(plan, data.actual[day*144:(day+1)*144], data.fixed_price, soc, mode, spec)
        details.append(detail)
        row = {'day':day, 'date':str((pd.Timestamp('2025-01-01')+pd.Timedelta(days=day)).date()),
               'total_cost':float(detail['fees'].sum()), 'planned_cost':float(detail['fees'][:,0].sum()),
               'emergency_cost':float(detail['fees'][:,3].sum()), 'initial_soc':soc,
               'final_soc':float(detail['states'][-1])}
        daily.append(row)
        execution_audit.append({**audits[i], 'forecast_origin':day*144,
                               'execution_initial_soc':soc,'execution_initial_mode':old_mode,
                               'execution_final_soc':row['final_soc'],'execution_final_mode':mode,
                               'purchase_locked_before_actual_read':True,
                               'optimizer_metadata':plan.get('optimizer_metadata')})
        soc = row['final_soc']
    arrays = {k:np.stack([d[k] for d in details]) for k in details[0]}
    battery = battery_metrics(arrays, 1, 172.75999999999976)
    summary = {'spec':spec,'days':stop_day-31,'total_cost':float(arrays['fees'].sum()),
               'planned_cost':float(arrays['fees'][:,:,0].sum()),
               'emergency_cost':float(arrays['fees'][:,:,3].sum()),
               'emergency_kwh':float(arrays['emergency'].sum()),
               'surplus_kwh':float(arrays['surplus'].sum()),**battery,
               'elapsed_seconds':time.perf_counter()-start,'evaluation_role':'2025 development; not untouched validation'}
    summary['controller_source_sha256_at_start'] = source_sha256
    baseline = BASELINE
    if stop_day != 365:
        baseline = float(pd.read_csv(OUT.parents[1]/'exp006/primary/daily.csv').iloc[:stop_day-31]['total_cost'].sum())
    summary['baseline_cost_for_evaluation_days'] = baseline
    summary['target_eligible'] = stop_day == 365
    summary['improvement_vs_exp006_pct'] = 100*(1-summary['total_cost']/baseline)
    summary['improvement_vs_cheapest_pct'] = 100*(1-summary['total_cost']/CHEAPEST) if stop_day==365 else None
    run_dir.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(run_dir/'dispatch_2.npz',**arrays)
    pd.DataFrame(daily).to_csv(run_dir/'daily.csv',index=False)
    (run_dir/'planning_audit.json').write_text(json.dumps({'days':execution_audit},ensure_ascii=False,indent=2))
    (run_dir/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print(json.dumps(summary,ensure_ascii=False),flush=True)
    return summary


def verify_completed():
    from experiments.problem2.tree_planning.verify import verify_arrays
    data = Data()
    rows = []
    for path in sorted(OUT.glob('*/summary.json')):
        summary = json.loads(path.read_text())
        if summary['days'] != 334:
            continue
        with np.load(path.parent/'dispatch_2.npz') as archive:
            arrays = {k:archive[k].copy() for k in archive.files}
        if summary['spec'].get('planner') == 'joint':
            # Scenario means are surrogate intentions, not a single physical
            # action; verify only actual physical dispatch for these runs.
            arrays = {k:v for k,v in arrays.items() if not k.startswith('intended_')}
        daily = pd.read_csv(path.parent/'daily.csv').to_dict('records')
        audits = json.loads((path.parent/'planning_audit.json').read_text())['days']
        report = verify_arrays(arrays,daily,audits,source_actual=data.actual[31*144:].reshape(334,144,2),
                               source_price=data.fixed_price)
        report['planning_intentions_are_scenario_means'] = summary['spec'].get('planner') == 'joint'
        (path.parent/'verification.json').write_text(json.dumps(report,indent=2))
        rows.append({'id':summary['spec']['id'],'passed':report['passed'],'errors':report['errors']})
    (OUT/'verification_summary.json').write_text(json.dumps(rows,indent=2))
    print(json.dumps({'runs':len(rows),'all_passed':all(r['passed'] for r in rows),'failed':[r for r in rows if not r['passed']]}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', default='first')
    args = parser.parse_args()
    data = Data()
    if args.batch == 'bound':
        perfect_information_bound(data)
        return
    if args.batch == 'verify':
        verify_completed()
        return
    supports,audits = risk_cache(data,ForecastStore('no_season',seed=42))
    if args.batch == 'first':
        specs = [{'id':f'lp_q{q:.2f}_greedy','quantile':q} for q in [.5,.6,.7,.8]]
    elif args.batch == 'deadband':
        specs = [{'id':f'lp_q{q:.2f}_db{db}','quantile':q,'charge_deadband':db}
                 for q in [.5,.6,.7] for db in [50,150,300]]
    elif args.batch == 'macro':
        specs = [{'id':f'lp_q{q:.2f}_{control}','quantile':q,'controller':control}
                 for q in [.5,.6,.7,.8] for control in ['planned_direction','macro']]
    elif args.batch == 'blocks':
        specs = [{'id':f'blocks_q{q:.2f}_solar{start}', 'quantile':q, 'controller':'fixed_blocks', 'solar_start':start}
                 for q in [.6,.7,.8,.85] for start in [48,54,60,64]]
    elif args.batch == 'joint':
        specs = [{'id':f'joint_{kind}_e{ew}', 'planner':'joint', 'path_type':kind,'emergency_weight':ew,'scenarios':7}
                 for kind in ['raw','rank'] for ew in [5,7]]
    elif args.batch == 'joint_blocks':
        specs = [{'id':f'joint_{kind}_{control}_db{db}', 'planner':'joint', 'path_type':kind,
                  'emergency_weight':5,'scenarios':7,'controller':control,'solar_start':60,
                  'charge_deadband':db,'holding_deadband':db}
                 for kind in ['rank'] for control in ['fixed_blocks','adaptive_blocks'] for db in [0,100]]
    elif args.batch == 'reserve':
        specs = ([{'id':f'lp_q{q:.2f}_reserve{r}','quantile':q,'reserve':r}
                  for q in [.7,.75,.8,.85] for r in [500,1000,2000]] +
                 [{'id':f'lp_q{q:.2f}_statebuffer{r}','quantile':q,'state_buffer':r}
                  for q in [.75,.8,.85] for r in [0,500,1000,2000]])
    elif args.batch == 'closed_pilot':
        specs = [{'id':f'closed_pilot_{kind}_{control}', 'optimize':True,'path_type':kind,
                  'quantile':.8,'controller':control,'solar_start':60,'development_days':30,
                  'maxiter':100,'scenarios':28}
                 for kind in ['raw','rank'] for control in ['greedy','fixed_blocks']]
    else:
        specs = json.loads(Path(args.batch).read_text())
    rows = [evaluate(s,data,supports,audits) for s in specs]
    all_rows = [json.loads(p.read_text()) for p in OUT.glob('*/summary.json')]
    pd.DataFrame([{**r, 'spec':json.dumps(r['spec'])} for r in all_rows if r['days']==334]).to_csv(OUT/'candidate_summary.csv',index=False)
    pd.DataFrame([{**r, 'spec':json.dumps(r['spec'])} for r in all_rows if r['days']!=334]).to_csv(OUT/'pilot_summary.csv',index=False)
    print('BEST',min(rows,key=lambda r:r['total_cost'])['spec'],flush=True)


if __name__ == '__main__':
    main()
