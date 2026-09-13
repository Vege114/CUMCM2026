"""Shared hourly modes with scenario emergency charging explicitly forbidden.

This repairs a physical feasible-set error in the previous scenario surrogate.
Scenario recourse still knows its own future path and remains an optimistic
approximation; the physical repair does not certify nonanticipativity.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import shutil
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from experiments.common.neural_v2.physics import LinearModel
from experiments.exp008.closed_loop import optimize
from experiments.exp008.controller_candidate import INITIAL_SOC, Data
from experiments.exp008.planner import ETA, HIGH, LIMIT, LOW, execute
from experiments.exp008.unified_forecast import UnifiedForecasts
from experiments.exp008.verify import battery_metrics, verify_npz

OUT = Path('data/results/exp008/mode_planning_physical')


def plan(net_paths, price, soc, previous_mode=1, *, block_slots=6,
         switching=50., wear=.002, seconds=5., gap=.002, final=False):
    paths=np.asarray(net_paths,float)
    price=np.asarray(price,float)
    k,n=paths.shape
    if price.shape!=(n,) or not np.all(price>0) or not np.isfinite(paths).all():
        raise ValueError('finite scenario net kWh and matching positive price required')
    if not LOW<=soc<=HIGH or block_slots<1:
        raise ValueError('invalid initial SOC or block length')
    nb=(n+block_slots-1)//block_slots
    m=LinearModel()
    # Larger common purchases are entirely wasted in every scenario and are
    # dominated under strictly positive fixed-plan purchase prices.
    q_upper=np.maximum(paths.max(axis=0),0.)+LIMIT
    q=m.variables((n,),upper=q_upper,cost=price)
    mode=m.variables((nb,),upper=1.,integer=True)
    flip=m.variables((nb,),cost=switching)
    c=m.variables((k,n),upper=LIMIT,cost=wear/k)
    d=m.variables((k,n),upper=LIMIT,cost=wear/k)
    # With emergency energy unable to charge, q>=0 and discharge>=0,
    # physically useful emergency energy cannot exceed positive net demand.
    emergency_upper=np.maximum(paths,0.)
    e=m.variables((k,n),upper=emergency_upper,cost=5*price[None,:]/k)
    w=m.variables((k,n))
    s=m.variables((k,n),lower=LOW,upper=HIGH)
    may_charge=m.variables((k,n),upper=1.,integer=True)
    for idx in s[:,-1]:
        m.objective[int(idx)]=0. if final else -float(price.min())/ETA/k
    for b in range(nb):
        terms=[(mode[b],1.)]
        rhs=float(previous_mode>0) if b==0 else 0.
        if b:
            terms.append((mode[b-1],-1.))
        m.constraint(terms+[(flip[b],-1.)],upper=rhs)
        m.constraint([(i,-v) for i,v in terms]+[(flip[b],-1.)],upper=-rhs)
    for j in range(k):
        for t in range(n):
            m.constraint([(q[t],1),(c[j,t],-1),(d[j,t],1),(e[j,t],1),(w[j,t],-1)],
                         paths[j,t],paths[j,t])
            terms=[(s[j,t],1),(c[j,t],-ETA),(d[j,t],1/ETA)]
            if t:
                terms.append((s[j,t-1],-1.))
            rhs=soc if t==0 else 0.
            m.constraint(terms,rhs,rhs)
            block=t//block_slots
            m.constraint([(c[j,t],1),(mode[block],-LIMIT)],upper=0.)
            m.constraint([(d[j,t],1),(mode[block],LIMIT)],upper=LIMIT)
            m.constraint([(c[j,t],1),(may_charge[j,t],-LIMIT)],upper=0.)
            m.constraint([(e[j,t],1),(may_charge[j,t],emergency_upper[j,t])],
                         upper=emergency_upper[j,t])
    x,metadata=m.solve(seconds=seconds,gap=gap)
    if x is None:
        raise RuntimeError(f'No feasible physical shared-mode incumbent: {metadata}')
    states=np.column_stack((np.full(k,soc),x[s]))
    balance=x[q][None,:]+x[d]+x[e]-x[c]-x[w]-paths
    state_error=np.diff(states,axis=1)-ETA*x[c]+x[d]/ETA
    check={'maximum_balance_error_kwh':float(np.abs(balance).max()),
           'maximum_soc_equation_error_kwh':float(np.abs(state_error).max()),
           'emergency_charging_slots':int(np.sum((x[c]>1e-6)&(x[e]>1e-6))),
           'simultaneous_charge_discharge_slots':int(np.sum((x[c]>1e-6)&(x[d]>1e-6))),
           'minimum_soc_kwh':float(states.min()),'maximum_soc_kwh':float(states.max()),
           'maximum_charge_or_discharge_kwh':float(max(x[c].max(),x[d].max())),
           'minimum_variable':float(x.min())}
    check['passed']=(max(check['maximum_balance_error_kwh'],check['maximum_soc_equation_error_kwh'])<1e-6
                     and check['emergency_charging_slots']==0
                     and check['simultaneous_charge_discharge_slots']==0
                     and states.min()>=LOW-1e-6 and states.max()<=HIGH+1e-6
                     and check['maximum_charge_or_discharge_kwh']<=LIMIT+1e-6
                     and check['minimum_variable']>=-1e-6)
    if not check['passed']:
        raise AssertionError(check)
    metadata.update(method='shared_mode_scenario_milp_without_emergency_charging',
                    scenario_count=k,nonanticipative_recourse_certificate=False,
                    planned_mode_changes=int(np.round(x[flip]).sum()),
                    scenario_physical_checks=check,
                    emergency_charge_complementarity='scenario-slot binary',
                    common_purchase_bound='max(max_scenario_net,0)+AC_charge_limit',
                    emergency_bound='max(scenario_net,0)')
    return {'purchase':np.maximum(x[q],0.),'allowed_charge':(x[mode]>.5)[np.arange(n)//block_slots],
            'scenario_charge':x[c],'scenario_discharge':x[d],'scenario_emergency':x[e],
            'scenario_surplus':x[w],'scenario_states':states,'metadata':metadata}


def counterexample():
    paths=np.array([[-1000.,0.]]*6+[[0.,1000.]])
    result=plan(paths,np.array([.4,1.]),LOW,block_slots=1,switching=0.,final=True)
    return {'purchase':result['purchase'].tolist(),
            'allowed_charge':result['allowed_charge'].tolist(),
            'scenario_charge':result['scenario_charge'].tolist(),
            'scenario_emergency':result['scenario_emergency'].tolist(),
            'metadata':result['metadata']}


def run(days=3,scenarios=7,checkpoint_days=None,forecast_override=None,case_name=None,
        prefix_directory=None):
    directory=OUT/(case_name or (f'refined_{days}days' if scenarios==7 else f'refined_s{scenarios}_{days}days'))
    directory.mkdir(parents=True,exist_ok=True)
    if (directory/'summary.json').exists():
        raise FileExistsError('Completed experiments are immutable')
    cfg={'days':days,'calibration':getattr(forecast_override,'calibration','ridge_28'),
         'scenarios':scenarios,'block_slots':6,
         'switching':50.,'wear':.002,'seconds':5.,'gap':.002,
         'refinement':True,'maxiter':120,'deadband':0.,
         'repair':'hard_no_scenario_emergency_charging',
         'scenario_selection':'deterministic_equally_spaced_complete_historical_origins',
         'initialization_scenario_count_also_changed':scenarios!=7,
         'refinement_uses_all_28_historical_paths':True,
         'checkpoint_days':checkpoint_days,'checkpoint_maximum_gap_threshold':.05,
         'nonanticipative_recourse_certificate':False,
         'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (directory/'config.json').write_text(json.dumps(cfg,indent=2))
    (directory/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    data=Data()
    forecast=forecast_override if forecast_override is not None else UnifiedForecasts(calibration='ridge_28')
    dependencies=('closed_loop','planner','unified_forecast','forecast','forecast_calibration')
    provenance={'source_data_hashes':data.hashes,
                'calibrated_forecast_values_sha256':hashlib.sha256(forecast.store.values.tobytes()).hexdigest(),
                'dependency_source_sha256':{}}
    for dependency in dependencies:
        source=Path(f'experiments/exp008/{dependency}.py')
        provenance['dependency_source_sha256'][str(source)]=hashlib.sha256(source.read_bytes()).hexdigest()
        (directory/f'{dependency}_snapshot.py').write_bytes(source.read_bytes())
    (directory/'provenance.json').write_text(json.dumps(provenance,indent=2))
    soc,mode=INITIAL_SOC,1
    rows,details,audits=[],[],[]
    prefix_count=0
    if prefix_directory is not None:
        prefix=Path(prefix_directory)
        old_cfg=json.loads((prefix/'config.json').read_text())
        excluded={'days','checkpoint_days','source_sha256'}
        for key,value in cfg.items():
            if key not in excluded and value!=old_cfg.get(key):
                raise AssertionError(f'Prefix configuration mismatch: {key}')
        old_provenance=json.loads((prefix/'provenance.json').read_text())
        if old_provenance!=provenance:
            raise AssertionError('Prefix data, forecasts or dependency sources changed')
        def plan_ast(source):
            return ast.dump(next(node for node in ast.parse(source).body
                                 if isinstance(node,ast.FunctionDef) and node.name=='plan'))
        if plan_ast((prefix/'source_snapshot.py').read_text())!=plan_ast(Path(__file__).read_text()):
            raise AssertionError('Prefix physical planning function changed')
        with np.load(prefix/'dispatch.npz',allow_pickle=False) as archive:
            prefix_count=len(archive['days'])
            if prefix_count>=days or not np.array_equal(archive['days'],np.arange(31,31+prefix_count)):
                raise ValueError('Prefix must be shorter, contiguous and start at day 31')
            details=[{key:archive[key][i].copy() for key in archive.files if key!='days'}
                     for i in range(prefix_count)]
        prefix_check=verify_npz(prefix/'dispatch.npz',expected_days=prefix_count,
                                audit_path=prefix/'planning_audit.json')
        if not prefix_check['passed']:
            raise AssertionError(prefix_check['errors'])
        audits=json.loads((prefix/'planning_audit.json').read_text())
        rows=[{'day':a['day'],'total_cost':float(d['fees'].sum()),
               'planned_cost':float(d['fees'][:,0].sum()),
               'emergency_cost':float(d['fees'][:,3].sum()),
               'mip':a['mip'],'refinement':a['greedy_refinement']}
              for a,d in zip(audits,details,strict=True)]
        for day in range(31,31+prefix_count):
            shutil.copy2(prefix/f'planning_day{day}.npz',directory/f'planning_day{day}.npz')
        soc=float(details[-1]['states'][-1])
        modes=np.sign(np.concatenate([d['charge']-d['discharge'] for d in details]))
        nonzero=modes[modes!=0]
        mode=int(nonzero[-1]) if len(nonzero) else 1
        (directory/'prefix_manifest.json').write_text(json.dumps({
            'directory':str(prefix),'days':prefix_count,'exact_arrays_reused':True,
            'continuation_initial_soc':soc,'continuation_initial_mode':mode,
            'dispatch_sha256':hashlib.sha256((prefix/'dispatch.npz').read_bytes()).hexdigest(),
            'physical_plan_ast_unchanged':True,'configuration_unchanged':True,
            'data_forecasts_dependencies_unchanged':True,'independent_verification':prefix_check},indent=2))
    began=perf_counter()
    for day in range(31+prefix_count,31+days):
        issue=forecast.get(day,scenario='2')
        errors=forecast.net_error_paths(day,'2',limit=28)
        paths=(issue['load_kw']-issue['pv_kw'])[None,:]/6+errors['errors_kwh']
        selected=np.linspace(0,len(paths)-1,min(scenarios,len(paths))).astype(int)
        try:
            result=plan(paths[selected],data.fixed_price,soc,mode,final=day==364)
        except (RuntimeError,AssertionError) as error:
            (directory/'failure.json').write_text(json.dumps({'day':day,'error':str(error),
                'completed_days':len(details),'stopped_before_remaining_days':True},indent=2))
            raise
        np.savez_compressed(directory/f'planning_day{day}.npz',
            **{key:value for key,value in result.items() if key!='metadata'},
            all_net_paths=paths,selected_scenario_indices=selected,initial_soc=np.array(soc),
            initial_mode=np.array(mode),price=data.fixed_price)
        refined=optimize(result['purchase'],paths,data.fixed_price,soc,
            charge_mask=result['allowed_charge'],throughput=.002,variation=0.,
            terminal=0. if day==364 else .45,maxiter=120,deadband=0.)
        q=refined['purchase']
        actual=data.actual[day*144:(day+1)*144]
        detail=execute(q,actual,data.fixed_price,soc,charge_deadband=0.,charge_mask=result['allowed_charge'])
        detail.update(original=q,final=q.copy(),actual=actual.copy(),price=data.fixed_price.copy(),
                      allowed_charge=result['allowed_charge'].copy())
        detail['fees']=np.stack((q*data.fixed_price,np.zeros(144),np.zeros(144),
                                5*detail['emergency']*data.fixed_price),axis=-1)
        details.append(detail)
        audits.append({'day':day,'forecast':issue['audit'],'history':errors['audit'],
                       'mip':result['metadata'],'greedy_refinement':refined['metadata']})
        rows.append({'day':day,'total_cost':float(detail['fees'].sum()),
                     'planned_cost':float(detail['fees'][:,0].sum()),
                     'emergency_cost':float(detail['fees'][:,3].sum()),
                     'mip':result['metadata'],'refinement':refined['metadata']})
        soc=float(detail['states'][-1])
        signs=np.sign(detail['charge']-detail['discharge']);nonzero=signs[signs!=0]
        if len(nonzero):
            mode=int(nonzero[-1])
        print('physical day',day,'cost',rows[-1]['total_cost'],'seconds',perf_counter()-began,flush=True)
        if days>30 and len(details)%10==0:
            progress={key:np.stack([d[key] for d in details]) for key in details[0]}
            progress['days']=np.arange(31,day+1)
            np.savez_compressed(directory/'progress_dispatch.npz',**progress)
            (directory/'progress_audit.json').write_text(json.dumps(audits,indent=2))
            pd.DataFrame(rows).to_csv(directory/'progress_daily.csv',index=False)
        if checkpoint_days is not None and len(details)==checkpoint_days:
            gaps=[audit['mip']['mip_gap'] for audit in audits]
            checkpoint={'completed_days':len(details),
                        'feasible_days':sum(audit['mip']['feasible'] for audit in audits),
                        'mip_gaps':gaps,'mip_seconds':[audit['mip']['seconds'] for audit in audits],
                        'maximum_gap_threshold':.05,
                        'continue':all(gap is not None and gap<=.05 for gap in gaps),
                        'cost_so_far':sum(row['total_cost'] for row in rows)}
            (directory/'checkpoint.json').write_text(json.dumps(checkpoint,indent=2))
            print('CHECKPOINT',json.dumps(checkpoint),flush=True)
            if not checkpoint['continue']:
                return checkpoint
    arrays={key:np.stack([detail[key] for detail in details]) for key in details[0]}
    arrays['days']=np.arange(31,31+days)
    np.savez_compressed(directory/'dispatch.npz',**arrays)
    pd.DataFrame(rows).to_csv(directory/'daily.csv',index=False)
    (directory/'planning_audit.json').write_text(json.dumps(audits,indent=2))
    check=verify_npz(directory/'dispatch.npz',expected_days=days,audit_path=directory/'planning_audit.json')
    if not check['passed']:
        raise AssertionError(check['errors'])
    if prefix_count:
        with np.load(Path(prefix_directory)/'dispatch.npz',allow_pickle=False) as old:
            prefix_equal={key:bool(np.array_equal(old[key],arrays[key][:prefix_count])) for key in old.files}
        if not all(prefix_equal.values()):
            raise AssertionError(prefix_equal)
        (directory/'prefix_array_equality.json').write_text(json.dumps(prefix_equal,indent=2))
    summary={'config':cfg,'total_cost':float(arrays['fees'].sum()),
             'battery':battery_metrics(arrays),'verification':check,
             'seconds':perf_counter()-began,
             'scenario_physical_checks':[a['mip']['scenario_physical_checks'] for a in audits],
             'mip_solver_metadata':[a['mip'] for a in audits]}
    (directory/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true')
    parser.add_argument('--days',type=int,default=3)
    parser.add_argument('--scenarios',type=int,default=7)
    parser.add_argument('--checkpoint-days',type=int)
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if args.check:
        report=counterexample()
        (OUT/'counterexample_fixed.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report,indent=2))
    else:
        run(args.days,args.scenarios,args.checkpoint_days)
