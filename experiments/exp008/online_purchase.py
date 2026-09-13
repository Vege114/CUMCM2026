"""Fixed-parameter daily online AdaGrad corrections to hourly purchases.

Current-day purchases are locked before observations. One complete realized
day updates the correction for the following midnight only. The zero and
online arms each carry their own realized SOC; no monthly hindsight fitting
or current-day optimization with actual labels is performed.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from experiments.exp008.closed_loop import objective
from experiments.exp008.controller_candidate import INITIAL_SOC, Data, plan_inventory
from experiments.exp008.forecast_calibration import CalibratedStore
from experiments.exp008.planner import ETA, execute
from experiments.exp008.verify import battery_metrics, verify_npz
from experiments.problem2.tree_planning.risk import TreeResidualScenarios

OUT=Path('data/results/exp008/online_purchase')
CONFIG={'calibration':'ridge_28','history_days':28,'quantile':.8,
        'blocks':24,'slots_per_block':6,'step_kwh':20.,'theta_bound_kwh':150.,
        'decay':.995,'adagrad_epsilon':1e-8,'wear':0.,'variation':0.,'deadband':0.,
        'theta_unit':'kWh per 10-minute slot, shared across six slots of the hour',
        'terminal':'minimum known fixed tariff / sqrt(.9)',
        'gradient_at_zero_purchase':'zero projection derivative as predeclared',
        'update':'theta_next=clip(decay*theta-step*gradient/sqrt(cumulative_gradient_squared),bounds)',
        'parameter_selection':'fixed before evaluation; no hyperparameter sweep',
        'parameter_availability':'gradient from completed day d first usable at midnight d+1',
        'evaluation_role':'2025 development exploration; not independent validation'}


def purchase(initial,theta):
    return np.maximum(0.,np.asarray(initial)+np.repeat(theta,6))


def realized_gradient(q,actual,price,initial_soc):
    net=(actual[:,0]-actual[:,1])/6
    value,gradient=objective(q,net[None,:],price,initial_soc,
                             throughput=0.,variation=0.,terminal=float(price.min())/ETA,
                             deadband=0.)
    hourly=(gradient*(q>0.)).reshape(24,6).sum(axis=1)
    return value,hourly


def update(theta,accumulator,gradient):
    accumulator=accumulator+gradient**2
    theta=np.clip(CONFIG['decay']*theta-CONFIG['step_kwh']*gradient/
                  (np.sqrt(accumulator)+CONFIG['adagrad_epsilon']),
                  -CONFIG['theta_bound_kwh'],CONFIG['theta_bound_kwh'])
    return theta,accumulator


def simulate(data,store,days,arm,directory=None):
    if arm not in ('zero','online'):
        raise ValueError('arm must be zero or online')
    risk=TreeResidualScenarios(store.origins,store.values,data.actual,data.fixed_price)
    soc,mode=INITIAL_SOC,1
    theta,accumulator=np.zeros(24),np.zeros(24)
    details,daily,audits=[],[],[]
    began=perf_counter()
    for i,day in enumerate(range(31,31+days)):
        origin=day*144
        support,risk_audit=risk.for_day(day)
        nominal=plan_inventory(support,data.fixed_price,soc,
                                {'quantile':.8,'wear':0.,'controller':'greedy'},final=day==364)
        before=theta.copy();previous_accumulator=accumulator.copy()
        q=purchase(nominal['purchase'],before)
        # The complete purchase vector above is finalized before this day's
        # actuals are exposed to either execution or the after-close update.
        actual=data.actual[origin:origin+144]
        detail=execute(q,actual,data.fixed_price,soc,charge_deadband=0.)
        value,gradient=realized_gradient(q,actual,data.fixed_price,soc)
        if arm=='online':
            theta,accumulator=update(before,accumulator,gradient)
        detail.update(original=q,final=q.copy(),actual=actual.copy(),price=data.fixed_price.copy(),
                      initial_purchase=nominal['purchase'],theta_before=before,
                      theta_after=theta.copy(),hourly_gradient=gradient,
                      accumulator_before=previous_accumulator,accumulator_after=accumulator.copy())
        detail['fees']=np.stack((q*data.fixed_price,np.zeros(144),np.zeros(144),
                                5*detail['emergency']*data.fixed_price),axis=-1)
        details.append(detail)
        last_soc=float(detail['states'][-1])
        direct=float(detail['fees'].sum())-float(data.fixed_price.min())/ETA*(last_soc-1200.)
        if abs(direct-value)>1e-6:
            raise AssertionError('Online update objective differs from actual greedy execution')
        daily.append({'day':day,'date':str((pd.Timestamp('2025-01-01')+pd.Timedelta(days=day)).date()),
                      'total_cost':float(detail['fees'].sum()),'planned_cost':float(detail['fees'][:,0].sum()),
                      'emergency_cost':float(detail['fees'][:,3].sum()),
                      'initial_soc':soc,'final_soc':last_soc,
                      'after_close_learning_objective':value,
                      'theta_l2_before':float(np.linalg.norm(before)),
                      'theta_max_abs_before':float(np.abs(before).max())})
        audits.append({'day':day,'origin':origin,'information_cutoff':origin,
                       'forecast_origin':origin,'forecast_calibration':'ridge_28',
                       'training_origins':risk_audit['training_origins'],
                       'max_observed_index':origin-1,'tree':risk_audit,
                       'parameters_applied_origin':origin,
                       'parameter_last_update_label':origin-1 if i and arm=='online' else None,
                       'theta_before':before.tolist(),'initial_soc':soc,
                       'initial_mode':mode,'purchase_locked_before_current_actual':True,
                       'after_close_update':{'performed':arm=='online',
                           'information_cutoff_exclusive':origin+144,
                           'gradient_actual_first_index':origin,'gradient_actual_last_index':origin+143,
                           'first_permitted_issue_origin':origin+144,
                           'theta_after':theta.tolist(),'accumulator_after':accumulator.tolist()}})
        soc=last_soc
        signs=np.sign(detail['charge']-detail['discharge']);nonzero=signs[signs!=0]
        if len(nonzero):
            mode=int(nonzero[-1])
        if directory is not None and (i+1)%10==0:
            print(arm,i+1,'cost',sum(row['total_cost'] for row in daily),
                  'seconds',perf_counter()-began,flush=True)
    arrays={key:np.stack([detail[key] for detail in details]) for key in details[0]}
    arrays['days']=np.arange(31,31+days)
    if directory is not None:
        directory.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(directory/'dispatch.npz',**arrays)
        pd.DataFrame(daily).to_csv(directory/'daily.csv',index=False)
        (directory/'audit.json').write_text(json.dumps(audits,indent=2))
        check=verify_npz(directory/'dispatch.npz',expected_days=days,audit_path=directory/'audit.json')
        check['online_parameter_chronology']={
            'all_preissued_updates':all(a['parameter_last_update_label'] is None or
                                       a['parameter_last_update_label']<a['origin'] for a in audits),
            'all_current_gradients_only_for_next_day':all(a['after_close_update']['gradient_actual_last_index']<
                     a['after_close_update']['first_permitted_issue_origin'] for a in audits),
            'theta_handoff_matches':bool(np.array_equal(arrays['theta_before'][1:],arrays['theta_after'][:-1]))}
        assert check['passed'] and all(check['online_parameter_chronology'].values()),check
        (directory/'verification.json').write_text(json.dumps(check,indent=2))
        summary={'arm':arm,'days':days,'total_cost':float(arrays['fees'].sum()),
                 'planned_cost':float(arrays['fees'][:,:,0].sum()),
                 'emergency_cost':float(arrays['fees'][:,:,3].sum()),
                 'battery':battery_metrics(arrays),'verification_passed':check['passed'],
                 'online_parameter_chronology':check['online_parameter_chronology'],
                 'final_theta':theta.tolist(),'seconds':perf_counter()-began}
        (directory/'summary.json').write_text(json.dumps(summary,indent=2))
    return arrays,daily,audits


def check():
    rng=np.random.default_rng(6732)
    initial=rng.uniform(200,600,144);theta=rng.uniform(-20,20,24)
    initial[:6]=50.;theta[0]=-100.
    actual=np.column_stack((rng.uniform(500,7000,144),rng.uniform(0,2500,144)))
    price=rng.uniform(.4,1.4,144)
    q=purchase(initial,theta)
    _,gradient=realized_gradient(q,actual,price,5000.)
    numerical=[]
    for j in range(24):
        step=np.zeros(24);step[j]=1e-4
        plus=realized_gradient(purchase(initial,theta+step),actual,price,5000.)[0]
        minus=realized_gradient(purchase(initial,theta-step),actual,price,5000.)[0]
        numerical.append((plus-minus)/2e-4)
    error=float(np.max(np.abs(gradient-numerical)))
    assert error<1e-5 and gradient[0]==0.
    data,store=Data(),CalibratedStore('ridge_28')
    before,_,_=simulate(data,store,5,'online')
    changed=copy.copy(data);changed.actual=data.actual.copy()
    changed.actual[34*144:,0]+=7000.
    after,_,_=simulate(changed,store,5,'online')
    # Day34 purchases were locked before any day34 realization; day35 can
    # change through both observed history and the newly updated parameter.
    prefix_error=float(np.max(np.abs(before['original'][:4]-after['original'][:4])))
    assert prefix_error==0.
    assert np.array_equal(before['theta_before'][0],np.zeros(24))
    assert np.array_equal(before['theta_before'][1:],before['theta_after'][:-1])
    return {'passed':True,'hourly_projection_gradient_max_error':error,
            'inactive_projection_hour_gradient':float(gradient[0]),
            'current_and_future_actual_mutation_purchase_prefix_error':prefix_error,
            'initial_theta_zero':True,'theta_applied_next_day_only':True}


def run(days=60):
    directory=OUT/f'fixed_{days}days'
    if (directory/'comparison.json').exists():
        raise FileExistsError('Completed fixed online experiment is immutable')
    directory.mkdir(parents=True,exist_ok=True)
    (directory/'protocol.json').write_text(json.dumps({**CONFIG,'days':days},indent=2))
    sources=['online_purchase','closed_loop','controller_candidate','planner','forecast_calibration']
    hashes={}
    for source in sources:
        path=Path(f'experiments/exp008/{source}.py')
        hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        (directory/f'{source}_snapshot.py').write_bytes(path.read_bytes())
    data,store=Data(),CalibratedStore('ridge_28')
    (directory/'provenance.json').write_text(json.dumps({'source_sha256':hashes,
        'source_data_sha256':data.hashes,'forecast_values_sha256':hashlib.sha256(store.values.tobytes()).hexdigest()},indent=2))
    (directory/'checks.json').write_text(json.dumps(check(),indent=2))
    for arm in ('zero','online'):
        simulate(data,store,days,arm,directory/arm)
    summaries={arm:json.loads((directory/arm/'summary.json').read_text()) for arm in ('zero','online')}
    delta=summaries['online']['total_cost']-summaries['zero']['total_cost']
    comparison={'days':days,'arms':summaries,'online_minus_zero_cost_yuan':delta,
        'online_reduction_vs_zero_pct':100*(1-summaries['online']['total_cost']/summaries['zero']['total_cost']),
        'cost_advantage':delta<0,'continue_only_if_cost_advantage':True,
        'parameters_tuned_after_results':False,'annual_goal_claimed':False,
        'each_arm_has_its_own_continuous_soc':True}
    (directory/'comparison.json').write_text(json.dumps(comparison,indent=2))
    pd.DataFrame({'arm':arm,'total_cost':r['total_cost'],**r['battery']}
                 for arm,r in summaries.items()).to_csv(directory/'comparison.csv',index=False)
    print(json.dumps(comparison,indent=2),flush=True)
    return comparison


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--days',type=int,default=60)
    parser.add_argument('--check',action='store_true');args=parser.parse_args()
    print(json.dumps(check(),indent=2)) if args.check else run(args.days)
