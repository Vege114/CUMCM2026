"""Simplified low-dimensional fixed-purchase optimization inspired by ARBO-DART.

This is our block correction + local L-BFGS implementation, not the paper's GP
Bayesian optimization, adaptive partition scheme, or global convergence claim.
Only complete issued historical days enter the midnight objective. All reported
2025 results are development results, not an untouched test set.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata

from experiments.exp008.closed_loop import objective
from experiments.exp008.controller_candidate import (
    INITIAL_SOC,
    execute_inventory,
    plan_inventory,
)
from experiments.exp008.forecast_calibration import CalibratedStore
from experiments.problem2.exp003.data import ROOT, Data
from experiments.problem2.tree_planning.model import ETA
from experiments.problem2.tree_planning.risk import TreeResidualScenarios
from experiments.problem2.tree_planning.verify import battery_metrics, verify_arrays

OUT = ROOT / 'data/results/exp008/block_purchase'


def forecast_store(name):
    if name.startswith('calendar_ridge'):
        from experiments.exp008.calendar_forecast import CalendarStore
        return CalendarStore(name)
    return CalibratedStore(name)


def basis(blocks, n=144):
    """Fixed equal-duration indicator blocks; B has shape (n, blocks)."""
    if blocks not in (4, 8, 24) or n % blocks:
        raise ValueError('Use 4, 8, or 24 equal-duration blocks')
    return np.eye(blocks)[np.arange(n)//(n//blocks)]


def optimize_blocks(initial, paths, price, soc, spec, mask=None, final=False):
    start = time.perf_counter()
    bmat = basis(spec['blocks'], len(price))
    options = {'charge_mask':mask, 'throughput':spec.get('wear', .005),
               'variation':spec.get('variation', .001),
               'terminal':0 if final else float(price.min())/ETA,
               'deadband':spec.get('charge_deadband', 0.),
               'emergency_weight':spec.get('emergency_weight', 5.)}
    best = {'value': float('inf'), 'theta': np.zeros(spec['blocks'])}

    def fun(theta):
        raw = initial+np.einsum('ij,j->i',bmat,theta,optimize=False)
        purchase = np.maximum(raw, 0.)
        value, gradient = objective(purchase, paths, price, soc, **options)
        # At exact zero, use the right derivative, consistent with a feasible
        # positive purchase perturbation. The objective is still nonsmooth.
        grad_theta = np.einsum('ij,i->j',bmat,gradient*(raw >= 0.),optimize=False)
        if value < best['value']:
            best.update(value=float(value), theta=theta.copy())
        return value, grad_theta

    initial_value = fun(np.zeros(spec['blocks']))[0]
    bound = spec.get('theta_bound_kwh', 833.3333333333)
    result = minimize(fun, np.zeros(spec['blocks']), jac=True, method='L-BFGS-B',
                      bounds=[(-bound, bound)]*spec['blocks'],
                      options={'maxiter':spec.get('maxiter', 400), 'maxls':30,
                               'ftol':1e-9, 'gtol':1e-5})
    theta = best['theta']
    return np.maximum(0., initial+np.einsum('ij,j->i',bmat,theta,optimize=False)), {
        'method':'fixed_block_correction_lbfgsb', 'theta_kwh':theta.tolist(),
        'block_edges_slots':np.arange(0, 145, 144//spec['blocks']).tolist(),
        'success':bool(result.success), 'message':str(result.message),
        'iterations':int(result.nit), 'evaluations':int(result.nfev),
        'initial_training_objective_yuan':float(initial_value),
        'selected_training_objective_yuan':best['value'],
        'training_objective_improvement_yuan':float(initial_value-best['value']),
        'global_optimality_certificate':False,
        'same_greedy_controller_in_training_and_replay':True,
        'seconds':time.perf_counter()-start,
    }


def load_inputs(calibration, days, data):
    store = forecast_store(calibration)
    risk = TreeResidualScenarios(store.origins, store.values, data.actual, data.fixed_price)
    support, audit, paths = [], [], []
    for day in range(31, 31+days):
        values, info = risk.for_day(day)
        ids = np.flatnonzero(store.origins+144 <= day*144)[-28:]
        forecast = store.get(day*144)
        net = (forecast[:,0]-forecast[:,1])/6
        if len(ids):
            slots = store.origins[ids,None]+np.arange(144)
            assert slots.max() < day*144
            observed = data.actual[slots]
            residual = (observed-store.values[ids])/6
            error = residual[:,:,0]-residual[:,:,1]
            demands = net[None,:]+error
        else:
            demands = values.T.copy()
        support.append(values)
        paths.append(demands)
        info.update(forecast_calibration=calibration,
                    forecast_net_kwh=net.tolist(),
                    path_training_origins=store.origins[ids].tolist(),
                    path_fallback_to_causal_tree=len(ids)==0,
                    forecast_values_sha256=hashlib.sha256(store.values.tobytes()).hexdigest())
        audit.append(info)
    return np.stack(support), paths, audit


def optimize_dp_blocks(initial, support, price, soc, spec, mask=None, final=False, mode=1, model=None):
    """Small bounded coordinate search with a newly solved DP at every Q."""
    markov = spec.get('optimizer')=='markov_coordinate' and model is not None
    impulse = spec.get('optimizer') in ('impulse_coordinate','markov_coordinate')
    if markov:
        if spec.get('markov_tail_points'):
            from experiments.exp008.markov_tail_feedback import (
                initial_value as markov_initial_value,
            )
            from experiments.exp008.markov_tail_feedback import value_functions
        else:
            from experiments.exp008.markov_feedback import initial_value as markov_initial_value
            from experiments.exp008.markov_feedback import value_functions
    elif impulse:
        from experiments.exp008.impulse_feedback import value_functions
        if mask is not None:
            raise ValueError('Impulse DP uses economic switching, not a hard time mask')
    else:
        from experiments.exp008.battery_feedback import value_functions
    started = time.perf_counter()
    matrix = basis(spec['blocks'])
    cache = {}
    terminal = 0 if final else float(price.min())/ETA
    wear = spec.get('wear',.002)

    def evaluate(theta):
        key=tuple(float(x) for x in theta)
        if key not in cache:
            purchase=np.maximum(0.,initial+np.einsum('ij,j->i',matrix,theta,optimize=False))
            kwargs={'grid_kwh':spec.get('dp_grid_kwh',200.),'wear':wear,'terminal':terminal}
            if impulse:
                kwargs['switching']=spec.get('switching',300.)
            else:
                kwargs.update(charge_mask=mask,charge_deadband=spec.get('charge_deadband',0.))
            grid,values,meta=value_functions(purchase,model if markov else support,price,**kwargs)
            if markov:
                continuation=markov_initial_value(grid,values,model,soc,mode=mode)
            else:
                initial_slice=values[0,mode+1] if impulse else values[0]
                continuation=np.interp(soc,grid,initial_slice)
            cost=float(np.dot(purchase,price)+continuation)
            cache[key]=(cost,purchase,grid,values,meta)
        return cache[key]

    theta=np.zeros(spec['blocks'])
    initial_value=evaluate(theta)[0]
    best=evaluate(theta)
    bound=spec.get('theta_bound_kwh',833.3333333333)
    rounds=spec.get('coordinate_rounds',1)
    for iteration in range(rounds):
        step=spec.get('coordinate_step_kwh',50.)/(2**iteration)
        for j in range(spec['blocks']):
            selected=theta.copy()
            for sign in (-1,1):
                proposed=theta.copy()
                proposed[j]=np.clip(proposed[j]+sign*step,-bound,bound)
                candidate=evaluate(proposed)
                if candidate[0]<best[0]-1e-7:
                    best,selected=candidate,proposed
            theta=selected
    return best[1],best[2],best[3],{
        'method':('fixed_block_correction_markov_coordinate' if markov else
                  'fixed_block_correction_impulse_coordinate' if impulse else 'fixed_block_correction_dp_coordinate'),
        'theta_kwh':theta.tolist(),
        'block_edges_slots':np.arange(0,145,144//spec['blocks']).tolist(),
        'success':True,'message':'Completed bounded coordinate sweep; no optimum certificate',
        'iterations':rounds,'evaluations':len(cache),
        'initial_training_objective_yuan':initial_value,
        'selected_training_objective_yuan':best[0],
        'training_objective_improvement_yuan':initial_value-best[0],
        'global_optimality_certificate':False,
        'same_greedy_controller_in_training_and_replay':False,
        'same_value_dp_in_training_and_replay':True,
        'initial_mode_for_value':mode,
        'switching_yuan_auxiliary':spec.get('switching',300.) if impulse else 0.,
        'disturbance_assumption':('conditional residual-state Markov model' if markov
                                  else 'independent conditional slot marginals'),
        'markov_model_metadata':model['metadata'] if markov else None,
        'dp_grid_kwh':spec.get('dp_grid_kwh',200.),
        'seconds':time.perf_counter()-started,
    }


def paths_for_day(raw_paths, support, path_type, scenarios=28):
    paths = raw_paths
    if path_type == 'rank' and len(paths)>1:
        # Rank of demand equals rank of residual when subtracting a common
        # forecast for the current day, preserving path temporal ordering.
        ranks = (rankdata(paths, axis=0)-.5)/len(paths)
        levels = (np.arange(9)+.5)/9
        paths = np.stack([np.interp(ranks[:,t], levels, support[t])
                          for t in range(144)], axis=1)
    elif path_type != 'raw':
        raise ValueError('path_type must be raw or rank')
    if len(paths)>scenarios:
        paths = paths[np.linspace(0,len(paths)-1,scenarios).astype(int)]
    return paths


def run(spec, data=None, inputs=None):
    directory = OUT/spec['id']
    if (directory/'summary.json').exists():
        saved = json.loads((directory/'summary.json').read_text())
        if saved['spec'] != spec:
            raise ValueError('Existing run id has different settings')
        return saved
    days = spec.get('days',30)
    if not 1 <= days <= 334:
        raise ValueError('days must fit February-December 2025')
    data = Data() if data is None else data
    support, paths, input_audit = (load_inputs(spec['calibration'],days,data)
                                    if inputs is None else inputs)
    source_hashes = {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in (
        'experiments/exp008/block_purchase.py', 'experiments/exp008/closed_loop.py',
        'experiments/exp008/controller_candidate.py','experiments/exp008/forecast_calibration.py',
        'experiments/exp008/battery_feedback.py','experiments/exp008/impulse_feedback.py')}
    if spec['calibration'].startswith('calendar_ridge'):
        source_hashes['experiments/exp008/calendar_forecast.py']=hashlib.sha256(
            (ROOT/'experiments/exp008/calendar_forecast.py').read_bytes()).hexdigest()
    if spec.get('optimizer')=='markov_coordinate':
        source_hashes['experiments/exp008/markov_feedback.py']=hashlib.sha256(
            (ROOT/'experiments/exp008/markov_feedback.py').read_bytes()).hexdigest()
        model_store=forecast_store(spec['calibration'])
        if spec.get('markov_shrinkage',12.) != 12.:
            raise ValueError('model_for_day adapter currently uses shrinkage=12')
        if spec.get('markov_tail_points'):
            source_hashes['experiments/exp008/markov_tail_feedback.py']=hashlib.sha256(
                (ROOT/'experiments/exp008/markov_tail_feedback.py').read_bytes()).hexdigest()
    begin = time.perf_counter()
    soc, mode = INITIAL_SOC, 1
    details, daily, audits = [], [], []
    execution_spec = dict(spec, holding_deadband=spec.get('charge_deadband',0.))
    for i, day in enumerate(range(31,31+days)):
        nominal = plan_inventory(support[i],data.fixed_price,soc,execution_spec,final=day==364)
        original_mode = mode
        mask = nominal['allowed_charge'] if spec.get('controller')=='fixed_blocks' else None
        available_paths = paths_for_day(paths[i],support[i],spec.get('path_type','raw'),spec.get('scenarios',28))
        model=None
        if spec.get('optimizer')=='markov_coordinate':
            if spec.get('markov_tail_points'):
                from experiments.exp008.markov_tail_feedback import model_for_day
                model=model_for_day(data,model_store,day,independent=spec.get('markov_independent',False),
                                    points=spec['markov_tail_points'])
            else:
                from experiments.exp008.markov_feedback import model_for_day
                model=model_for_day(data,model_store,day,independent=spec.get('markov_independent',False))
            model['metadata']['forecast_calibration']=spec['calibration']
            if not model['metadata'].get('fallback',False):
                model['metadata']['residual_source']='issued_' + spec['calibration'] + '_forecast'
        if spec.get('optimizer') in ('dp_coordinate','impulse_coordinate','markov_coordinate'):
            purchase,grid,values,meta=optimize_dp_blocks(nominal['purchase'],support[i],data.fixed_price,
                                                       soc,spec,mask,final=day==364,mode=mode,model=model)
        else:
            purchase, meta = optimize_blocks(nominal['purchase'],available_paths,data.fixed_price,
                                             soc,spec,mask,final=day==364)
        nominal['purchase'] = purchase
        # Current actual observations enter only after the complete purchase
        # vector has been selected and locked by the historical-path objective.
        actual = data.actual[day*144:(day+1)*144]
        if spec.get('optimizer')=='markov_coordinate' and model is not None:
            from experiments.exp008.markov_feedback import execute as execute_feedback
            detail,mode=execute_feedback(purchase,actual,data.fixed_price,soc,grid,values,model,
                wear=spec.get('wear',.002),switching=spec.get('switching',300.),initial_mode=mode)
        elif spec.get('optimizer') in ('impulse_coordinate','markov_coordinate'):
            from experiments.exp008.impulse_feedback import execute as execute_feedback
            detail,mode=execute_feedback(purchase,actual,data.fixed_price,soc,grid,values,
                wear=spec.get('wear',.002),switching=spec.get('switching',300.),initial_mode=mode)
        elif spec.get('optimizer')=='dp_coordinate':
            from experiments.exp008.battery_feedback import execute as execute_feedback
            detail=execute_feedback(purchase,actual,data.fixed_price,soc,grid,values,
                wear=spec.get('wear',.002),charge_mask=mask,charge_deadband=spec.get('charge_deadband',0.))
            nonzero=np.sign(detail['charge']-detail['discharge'])
            nonzero=nonzero[nonzero!=0]
            mode=int(nonzero[-1]) if len(nonzero) else mode
        else:
            detail, mode = execute_inventory(nominal,actual,data.fixed_price,soc,mode,execution_spec)
        detail.pop('intended_charge',None)
        detail.pop('intended_discharge',None)
        detail.pop('intended_states',None)
        details.append(detail)
        row = {'day':day,'date':str((pd.Timestamp('2025-01-01')+pd.Timedelta(days=day)).date()),
               'total_cost':float(detail['fees'].sum()), 'planned_cost':float(detail['fees'][:,0].sum()),
               'emergency_cost':float(detail['fees'][:,3].sum()), 'initial_soc':soc,
               'final_soc':float(detail['states'][-1]),'planning_seconds':meta['seconds']}
        daily.append(row)
        audits.append({**input_audit[i],'forecast_origin':day*144,
                       'execution_initial_soc':soc,'execution_initial_mode':original_mode,
                       'execution_final_mode':mode,'execution_final_soc':row['final_soc'],
                       'purchase_locked_before_actual_read':True,'optimizer':meta})
        soc = row['final_soc']
        if (i+1)%10 == 0:
            print(f"PROGRESS {spec['id']} {i+1}/{days} elapsed={time.perf_counter()-begin:.1f}s",flush=True)
    arrays = {k:np.stack([d[k] for d in details]) for k in details[0]}
    report = verify_arrays(arrays,daily,audits,expected_days=days,
                           source_actual=data.actual[31*144:(31+days)*144].reshape(days,144,2),
                           source_price=data.fixed_price)
    if not report['passed']:
        raise RuntimeError(report['errors'])
    battery = battery_metrics(arrays,1,172.75999999999976)
    baseline = pd.read_csv(ROOT/'data/results/exp006/primary/daily.csv').iloc[:days]
    baseline_cost = float(baseline.total_cost.sum())
    with np.load(ROOT/'data/results/exp006/primary/dispatch_2.npz') as archive:
        base_battery = battery_metrics({k:archive[k][:days] for k in archive.files},1,172.75999999999976)
    summary = {'spec':spec,'days':days,'total_cost':float(arrays['fees'].sum()),
               'planned_cost':float(arrays['fees'][:,:,0].sum()),
               'emergency_cost':float(arrays['fees'][:,:,3].sum()),
               'emergency_kwh':float(arrays['emergency'].sum()),
               'surplus_kwh':float(arrays['surplus'].sum()),**battery,
               'baseline_cost_same_days':baseline_cost,
               'baseline_direction_reversals_same_days':base_battery['direction_reversals'],
               'baseline_active_slots_same_days':base_battery['active_slots'],
               'improvement_vs_exp006_pct':100*(1-float(arrays['fees'].sum())/baseline_cost),
               'target_eligible':days==334,'verified':True,'seconds':time.perf_counter()-begin,
               'optimizer_success_days':sum(a['optimizer']['success'] for a in audits),
               'total_training_objective_improvement_yuan':sum(a['optimizer']['training_objective_improvement_yuan'] for a in audits),
               'source_sha256':source_hashes,
               'evaluation_role':'2025 development; not untouched validation',
               'literature_implementation':'Simplified fixed block correction; not full ARBO-DART'}
    directory.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(directory/'dispatch_2.npz',**arrays)
    pd.DataFrame(daily).to_csv(directory/'daily.csv',index=False)
    (directory/'planning_audit.json').write_text(json.dumps({'days':audits},indent=2))
    (directory/'verification.json').write_text(json.dumps(report,indent=2))
    (directory/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary),flush=True)
    return summary


def check_gradient():
    rng = np.random.default_rng(8103)
    initial = rng.uniform(200,500,144)
    paths = rng.uniform(300,800,(7,144))
    price = rng.uniform(.4,1.4,144)
    matrix = basis(8)
    theta = rng.uniform(-20,20,8)
    purchase = initial+np.einsum('ij,j->i',matrix,theta,optimize=False)
    _, grad = objective(purchase,paths,price,6000.)
    analytic = np.einsum('ij,i->j',matrix,grad,optimize=False)
    numeric = np.zeros(8)
    step=1e-4
    for j in range(8):
        perturb = matrix[:,j]*step
        numeric[j] = (objective(purchase+perturb,paths,price,6000.)[0]
                      -objective(purchase-perturb,paths,price,6000.)[0])/(2*step)
    error = float(np.max(np.abs(analytic-numeric)))
    assert error<1e-5,error
    return {'finite_difference_max_error':error,'passed':True}


def annual_report():
    """Independent source/physics audit and monthly matched comparison."""
    from experiments.exp008.verify import battery_metrics as independent_metrics
    from experiments.exp008.verify import verify_npz

    annual = []
    monthly = []
    cases = [('exp006_primary', ROOT/'data/results/exp006/primary')]
    for rounds in (0,1):
        name=f'ridge28_markov300_blocks8_rounds{rounds}_step50_full334'
        cases.append((name,OUT/name))
    for name,directory in cases:
        report=verify_npz(directory/'dispatch_2.npz',expected_days=334,
                          audit_path=directory/'planning_audit.json' if name!='exp006_primary' else None)
        if name!='exp006_primary':
            audit=json.loads((directory/'planning_audit.json').read_text())['days']
            historical_checks=[]
            for row in audit:
                model=row['optimizer']['markov_model_metadata']
                cutoff=model['information_cutoff_exclusive']
                historical_checks.append(cutoff==row['forecast_origin'] and
                    all(origin+144<=cutoff for origin in model['history_origins']) and
                    model['maximum_historical_label']<cutoff)
            report['markov_complete_history_cutoff_checks']={
                'days':len(historical_checks),'all_passed':all(historical_checks),
                'scope':'saved complete-horizon provenance; does not certify unrecorded program access'}
            report['passed']=report['passed'] and all(historical_checks)
            (directory/'independent_verification.json').write_text(json.dumps(report,indent=2))
        if not report['passed']:
            raise RuntimeError(f'Independent audit failed for {name}: {report["errors"]}')
        annual.append(dict(case=name,**report['billing'],**report['battery_metrics'],
                           verified=report['passed'],goal_passed=report.get('goal',{}).get('passed',False)))
        with np.load(directory/'dispatch_2.npz') as archive:
            arrays={key:archive[key] for key in archive.files}
        dates=pd.date_range('2025-02-01','2025-12-31')
        prior_mode,prior_power=1,172.75999999999976
        for month in sorted(set(dates.month)):
            select=dates.month==month
            a={key:value[select] for key,value in arrays.items()}
            metrics=independent_metrics(a,initial_mode=prior_mode,initial_power_kw=prior_power)
            monthly.append(dict(case=name,month=f'2025-{month:02d}',
                total_cost=float(a['fees'].sum()),planned_cost=float(a['fees'][...,0].sum()),
                emergency_cost=float(a['fees'][...,3].sum()),surplus_kwh=float(a['surplus'].sum()),
                emergency_kwh=float(a['emergency'].sum()),**metrics))
            prior_mode,prior_power=metrics['final_mode'],metrics['final_power_kw']
    frame=pd.DataFrame(monthly)
    for metric in ('total_cost','planned_cost','emergency_cost','throughput_kwh','active_slots',
                   'direction_reversals_including_warmup_boundary','power_ramp_total_kw_including_warmup_boundary'):
        base=frame[frame['case']=='exp006_primary'].set_index('month')[metric]
        control=frame[frame['case']==cases[1][0]].set_index('month')[metric]
        frame[metric+'_delta_vs_exp006']=frame[metric]-frame['month'].map(base)
        frame[metric+'_delta_vs_uncorrected']=frame[metric]-frame['month'].map(control)
    frame.to_csv(OUT/'markov_annual_monthly_comparison.csv',index=False)
    pd.DataFrame(annual).to_csv(OUT/'markov_annual_comparison.csv',index=False)
    print(json.dumps(annual,indent=2))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--batch',default='pilot')
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if args.batch=='check':
        result=check_gradient()
        (OUT/'gradient_check.json').write_text(json.dumps(result,indent=2))
        print(json.dumps(result));return
    if args.batch=='annual_report':
        annual_report();return
    if args.batch in ('pilot','mask_pilot'):
        control = 'greedy' if args.batch=='pilot' else 'fixed_blocks'
        specs=[{'id':f'{cal}_{control}_blocks{blocks}_pilot30','calibration':cal,
                'blocks':blocks,'quantile':.8,'controller':control,'solar_start':60,
                'days':30,'path_type':'raw','scenarios':28,'maxiter':400}
               for cal in ('base','ridge_28') for blocks in (4,8,24)]
    elif args.batch=='dp_pilot':
        specs=[{'id':'ridge28_dp_blocks4_step50_pilot30','calibration':'ridge_28',
                'blocks':4,'quantile':.8,'controller':'greedy','days':30,
                'path_type':'raw','optimizer':'dp_coordinate','coordinate_rounds':1,
                'coordinate_step_kwh':50.,'dp_grid_kwh':200.,'wear':.002}]
    elif args.batch=='impulse_pilot':
        specs=[{'id':f'ridge28_impulse{switch}_blocks4_step{step}_pilot30','calibration':'ridge_28',
                'blocks':4,'quantile':.8,'controller':'greedy','days':30,
                'path_type':'raw','optimizer':'impulse_coordinate','coordinate_rounds':1,
                'coordinate_step_kwh':float(step),'dp_grid_kwh':200.,'wear':.002,'switching':float(switch)}
               for switch in (100,300) for step in (50,100)]
    elif args.batch=='markov_pilot':
        specs=[{'id':f'ridge28_markov{switch}_blocks4_rounds{rounds}_step50_pilot30','calibration':'ridge_28',
                'blocks':4,'quantile':.8,'controller':'greedy','days':30,
                'path_type':'raw','optimizer':'markov_coordinate','coordinate_rounds':rounds,
                'coordinate_step_kwh':50.,'dp_grid_kwh':200.,'wear':.002,'switching':float(switch)}
               for switch in (100,300) for rounds in (0,1)]
    else:
        specs=json.loads(Path(args.batch).read_text())
    data=Data();cache={};results=[]
    for spec in specs:
        key=(spec['calibration'],spec.get('days',30))
        if key not in cache:
            cache[key]=load_inputs(*key,data)
        results.append(run(spec,data,cache[key]))
    rows=[json.loads(p.read_text()) for p in OUT.glob('*/summary.json')]
    for days in sorted({r['days'] for r in rows}):
        pd.DataFrame([{k:v for k,v in r.items() if k!='source_sha256'} for r in rows if r['days']==days]).to_csv(OUT/f'summary_{days}days.csv',index=False)
    print('BEST',min(results,key=lambda r:r['total_cost'])['spec'],flush=True)


if __name__=='__main__':
    main()
