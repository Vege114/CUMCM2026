"""Fixed historical path reduction before physical common-mode planning.

The physical MILP below is copied from mode_planning_physical.plan, with only
scenario expectation weights added. The paired default-weight equivalence is
checked before use. This module never changes an existing running planner.
"""
from pathlib import Path
from time import perf_counter
import hashlib
import json
import shutil

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, cut_tree

from experiments.common.neural_v2.physics import LinearModel
from experiments.exp008.planner import ETA,HIGH,LIMIT,LOW,execute
from experiments.exp008.closed_loop import optimize
from experiments.exp008.controller_candidate import INITIAL_SOC,Data
from experiments.exp008.neural_joint_dispatch import JointForecasts
from experiments.exp008.verify import verify_npz

OUT=Path('data/results/exp008/weighted_mode_planning')


def reduce_paths(paths, price, clusters=3):
    """Ward clusters in fixed slot and cumulative inventory error geometry.

    One half of the distance is tariff-weighted interval energy; the other is
    prefix energy divided by sqrt(horizon), preventing units from overwhelming
    the interval component. No fitted scale, target-day actual, or cost label.
    """
    paths,price=np.asarray(paths,float),np.asarray(price,float)
    if paths.ndim!=2 or price.shape!=(paths.shape[1],) or not np.isfinite(paths).all():
        raise ValueError('Finite full historical paths and matching prices required')
    count,n=paths.shape
    if not count or clusters<1 or np.any(price<=0):
        raise ValueError('Positive counts and prices required')
    geometry=np.c_[paths*np.sqrt(price/price.mean()),np.cumsum(paths,axis=1)/np.sqrt(n)]
    labels=np.arange(count) if count<=clusters else cut_tree(
        linkage(geometry,method='ward'),n_clusters=clusters).ravel()
    selected,weights=[],[]
    for label in np.unique(labels):
        ids=np.flatnonzero(labels==label)
        center=geometry[ids].mean(axis=0)
        distance=((geometry[ids]-center)**2).sum(axis=1)
        selected.append(int(ids[np.argmin(distance)]))
        weights.append(len(ids)/count)
    order=np.argsort(selected)
    selected=np.asarray(selected)[order];weights=np.asarray(weights)[order]
    return selected,weights,{'method':'Ward then within-cluster medoid',
        'cluster_labels':labels.tolist(),'selected_indices':selected.tolist(),
        'weights':weights.tolist(),'historical_path_count':count,
        'distance':'squared L2 on [net*sqrt(p/mean(p)),cumsum(net)/sqrt(144)]',
        'target_actual_used':False,'selection_parameter_fitting':False}

def plan_weighted(net_paths, price, soc, previous_mode=1, *, weights=None, block_slots=6,
         switching=50., wear=.002, seconds=5., gap=.002, final=False):
    paths=np.asarray(net_paths,float)
    price=np.asarray(price,float)
    k,n=paths.shape
    weights=np.full(k,1/k) if weights is None else np.asarray(weights,float)
    if weights.shape!=(k,) or not np.all(weights>0) or not np.isclose(weights.sum(),1.):
        raise ValueError("Strictly positive scenario weights must sum to one")
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
    c=m.variables((k,n),upper=LIMIT,cost=wear*weights[:,None])
    d=m.variables((k,n),upper=LIMIT,cost=wear*weights[:,None])
    # With emergency energy unable to charge, q>=0 and discharge>=0,
    # physically useful emergency energy cannot exceed positive net demand.
    emergency_upper=np.maximum(paths,0.)
    e=m.variables((k,n),upper=emergency_upper,cost=5*price[None,:]*weights[:,None])
    w=m.variables((k,n))
    s=m.variables((k,n),lower=LOW,upper=HIGH)
    may_charge=m.variables((k,n),upper=1.,integer=True)
    for j,idx in enumerate(s[:,-1]):
        m.objective[int(idx)]=0. if final else -float(price.min())/ETA*weights[j]
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
                    scenario_count=k,scenario_weights=weights.tolist(),nonanticipative_recourse_certificate=False,
                    planned_mode_changes=int(np.round(x[flip]).sum()),
                    scenario_physical_checks=check,
                    emergency_charge_complementarity='scenario-slot binary',
                    common_purchase_bound='max(max_scenario_net,0)+AC_charge_limit',
                    emergency_bound='max(scenario_net,0)')
    return {'purchase':np.maximum(x[q],0.),'allowed_charge':(x[mode]>.5)[np.arange(n)//block_slots],
            'scenario_charge':x[c],'scenario_discharge':x[d],'scenario_emergency':x[e],
            'scenario_surplus':x[w],'scenario_states':states,'metadata':metadata}


def run(days=30):
    directory=OUT/f'ward3_joint_ridge28_deadband0_{days}days'
    directory.mkdir(parents=True,exist_ok=True)
    if (directory/'summary.json').exists():
        raise FileExistsError('Completed results are immutable')
    cfg={'days':days,'scenario_reduction':'fixed Ward3 medoids and population weights',
         'forecast':'joint_ridge28','block_slots':6,'switching':50.,'wear':.002,
         'mip_seconds':5.,'mip_gap':.002,'refinement_all_history_paths':28,
         'maxiter':120,'terminal_refinement':.45,'deadband':0.,
         'evaluation_role':'2025 development, not independent validation',
         'predeclared_extension_gate':'0.3% same-period bill saving, no terminal inventory depletion, reversals below exp006',
         'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (directory/'config.json').write_text(json.dumps(cfg,indent=2))
    data,forecast=Data(),JointForecasts()
    provenance={'data_sha256':data.hashes,
         'forecast_values_sha256':hashlib.sha256(forecast.store.values.tobytes()).hexdigest(),
         'sources':{}}
    source_dir=directory/'sources';source_dir.mkdir(exist_ok=True)
    for name in ('weighted_mode_planning','neural_joint_dispatch','neural_joint_calibration',
                 'neural_joint','forecast','forecast_calibration','closed_loop','planner','verify'):
        source=Path(__file__).parent/f'{name}.py'
        provenance['sources'][name]=hashlib.sha256(source.read_bytes()).hexdigest()
        shutil.copy2(source,source_dir/source.name)
    (directory/'provenance.json').write_text(json.dumps(provenance,indent=2))
    soc,mode=INITIAL_SOC,1
    details,audits,rows=[],[],[]
    began=perf_counter()
    for day in range(31,31+days):
        issue=forecast.get(day,scenario='2')
        history=forecast.net_error_paths(day,'2',limit=28)
        paths=(issue['load_kw']-issue['pv_kw'])[None,:]/6+history['errors_kwh']
        selected,weights,reduction=reduce_paths(paths,data.fixed_price)
        planned=plan_weighted(paths[selected],data.fixed_price,soc,mode,weights=weights,
                              final=day==364)
        refined=optimize(planned['purchase'],paths,data.fixed_price,soc,
            charge_mask=planned['allowed_charge'],throughput=.002,variation=0.,
            terminal=0. if day==364 else .45,maxiter=120,deadband=0.)
        q=refined['purchase']
        np.savez_compressed(directory/f'planning_day{day}.npz',
            **{key:value for key,value in planned.items() if key!='metadata'},
            all_net_paths=paths,selected_scenario_indices=selected,weights=weights,
            initial_soc=np.array(soc),initial_mode=np.array(mode),price=data.fixed_price)
        observed=data.actual[day*144:(day+1)*144]
        detail=execute(q,observed,data.fixed_price,soc,charge_deadband=0.,charge_mask=planned['allowed_charge'])
        detail.update(original=q,final=q.copy(),actual=observed.copy(),price=data.fixed_price.copy(),
                      allowed_charge=planned['allowed_charge'].copy())
        detail['fees']=np.stack((q*data.fixed_price,np.zeros(144),np.zeros(144),
                                 5*detail['emergency']*data.fixed_price),axis=-1)
        details.append(detail)
        audits.append({'day':day,'forecast':issue['audit'],'history':history['audit'],
                       'reduction':reduction,'mip':planned['metadata'],
                       'greedy_refinement':refined['metadata']})
        soc=float(detail['states'][-1])
        nonzero=np.sign(detail['charge']-detail['discharge'])
        nonzero=nonzero[nonzero!=0]
        if len(nonzero):
            mode=int(nonzero[-1])
        rows.append({'day':day,'total_cost':float(detail['fees'].sum()),
                     'planned_cost':float(detail['fees'][:,0].sum()),
                     'emergency_cost':float(detail['fees'][:,3].sum()),
                     'final_soc':soc,'mip_gap':planned['metadata']['mip_gap']})
        print('ward3',day,rows[-1]['total_cost'],'seconds',perf_counter()-began,flush=True)
    arrays={key:np.stack([d[key] for d in details]) for key in details[0]}
    arrays['days']=np.arange(31,31+days)
    np.savez_compressed(directory/'dispatch.npz',**arrays)
    (directory/'planning_audit.json').write_text(json.dumps(audits,indent=2))
    pd.DataFrame(rows).to_csv(directory/'daily.csv',index=False)
    checked=verify_npz(directory/'dispatch.npz',expected_days=days,
                        audit_path=directory/'planning_audit.json')
    if not checked['passed']:
        raise AssertionError(checked['errors'])
    summary={'config':cfg,'total_cost':float(arrays['fees'].sum()),
             'verification':checked,'battery':checked['battery_metrics'],
             'seconds':perf_counter()-began}
    if days==30:
        reference=json.loads(Path('data/results/exp008/mode_planning_physical/'
            'joint_ridge28_refined_s3_30days/summary.json').read_text())
        delta=summary['total_cost']-reference['total_cost']
        summary['paired_comparison']={'reference_total_cost':reference['total_cost'],
            'cost_delta':delta,'cost_change_pct':100*delta/reference['total_cost'],
            'reference_final_soc':reference['battery']['final_soc'],
            'reference_reversals':reference['battery']['direction_reversals'],
            'extend_gate_passed':bool(delta<=-.003*reference['total_cost']
                and checked['battery_metrics']['final_soc']>=reference['battery']['final_soc']-1e-6
                and checked['battery_metrics']['direction_reversals']<267)}
    (directory/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps({key:value for key,value in summary.items() if key!='verification'},indent=2))
    return summary


if __name__=='__main__':
    run()
