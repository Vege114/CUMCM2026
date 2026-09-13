"""Fixed cumulative-energy quantile risk trajectory and a matched point control.

This is an approximate inventory-risk representation, not a joint chance
constraint certificate. Differencing a prefix quantile is not a quantile of
individual errors; efficiency and saturation also limit the interpretation.
"""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.exp008.absolute_hgb_forecast_adapter import AbsoluteHGBForecasts
from experiments.exp008.controller_candidate import Data, INITIAL_SOC, plan_inventory, execute_inventory
from experiments.exp008.verify import verify_npz

OUT=Path('data/results/exp008/cumulative_risk_planning')
SPEC={'quantile':.8,'controller':'greedy','state_buffer':500.,'wear':.002}


def save(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def trajectory(forecast_net,errors,method):
    forecast_net=np.asarray(forecast_net,float);errors=np.asarray(errors,float)
    if errors.ndim!=2 or errors.shape[1]!=len(forecast_net) or not len(errors):
        raise ValueError('Complete joint historical error paths required')
    if method=='prefix':
        adjustment=np.quantile(np.cumsum(errors,axis=1),.8,axis=0,method='linear')
        risk=forecast_net+np.diff(np.r_[0.,adjustment])
        np.testing.assert_allclose(np.cumsum(risk-forecast_net),adjustment,rtol=0,atol=1e-8)
    elif method=='point':
        adjustment=np.quantile(errors,.8,axis=0,method='linear')
        risk=forecast_net+adjustment
    else:
        raise ValueError(method)
    return risk,adjustment


def run(method,data,forecast,prepared,audits):
    directory=OUT/method
    directory.mkdir()
    soc,mode=INITIAL_SOC,1
    parts,rows=[],[]
    for i,day in enumerate(range(31,365)):
        risk=prepared[method][i]
        plan=plan_inventory(np.repeat(risk[:,None],9,axis=1),data.fixed_price,soc,SPEC,final=day==364)
        # The true current-day load/PV is accessed only after Q and intended
        # states are determined. Executor only observes the current slot.
        observed=data.actual[day*144:(day+1)*144]
        detail,mode=execute_inventory(plan,observed,data.fixed_price,soc,mode,SPEC)
        parts.append(detail)
        rows.append({'day':day,'total_cost':float(detail['fees'].sum()),
                     'planned_cost':float(detail['fees'][:,0].sum()),
                     'emergency_cost':float(detail['fees'][:,3].sum()),
                     'initial_soc':soc,'final_soc':float(detail['states'][-1])})
        soc=rows[-1]['final_soc']
    arrays={key:np.stack([d[key] for d in parts]) for key in parts[0]}
    arrays['days']=np.arange(31,365)
    np.savez_compressed(directory/'dispatch_2.npz',**arrays)
    own=copy.deepcopy(audits)
    for row in own:
        row.update(risk_representation=method,quantile_level=.8,
                   quantile_is_of_complete_error_sample_not_nine_tree_atoms=True,
                   purchase_locked_before_actual=True,joint_chance_certificate=False)
    save(directory/'audit.json',own)
    pd.DataFrame(rows).to_csv(directory/'daily.csv',index=False)
    check=verify_npz(directory/'dispatch_2.npz',audit_path=directory/'audit.json')
    assert check['passed'],check['errors']
    # Verify a subset of the execution prefix under arbitrary future actuals.
    prefix_checks=[]
    for i in (0,59,150,242,333):
        for stop in (1,36,108):
            observed=arrays['actual'][i].copy();observed[stop:]+=np.array([80000.,30000.])
            plan={'purchase':arrays['original'][i],'charge':arrays['intended_charge'][i],
                  'discharge':arrays['intended_discharge'][i],'states':arrays['intended_states'][i]}
            d,_=execute_inventory(plan,observed,data.fixed_price,float(arrays['states'][i,0]),1,SPEC)
            for key in ('charge','discharge','emergency','surplus'):
                np.testing.assert_array_equal(d[key][:stop],arrays[key][i,:stop])
            prefix_checks.append({'day':i+31,'mutated_from_slot':stop,'prefix_flows_identical':True})
    result={'method':method,'complete':True,'days':334,'verification':check,
            'execution_prefix_checks':prefix_checks,'final_model_selection':False}
    save(directory/'summary.json',result)
    print(method,check['billing'],check['battery_metrics']['direction_reversals'],flush=True)
    return result


def main():
    if OUT.exists():
        raise FileExistsError('Development comparisons are immutable')
    OUT.mkdir(parents=True)
    save(OUT/'protocol.json',{'days':334,'methods':['prefix','point'],'true_empirical_quantile':.8,
        'history':'same preceding28 completed-day HGB pipeline net errors with labelled January fallback',
        'prefix_formula':'r_t=f_t+Delta quantile_j(sum_{u<=t} error_ju,0.8)',
        'point_control_formula':'r_t=f_t+quantile_j(error_jt,0.8)',
        'fixed_spec':SPEC,'same_daily_LP_and_actual_executor':True,'own_continuous_SOC':True,
        'risk_representation_is_not_point_load_PV_forecast':True,
        'not_a_joint_chance_constraint_certificate':True,
        'efficiency_and_saturation_risk_mapping_not_exact':True,
        'tree_baseline_not_single_factor':'raw joint history replaces conditional tree marginals; genuine .8 rather than .8 interpolation of nine atoms',
        'all334_predeclared_no_partial_performance_gate':True,'development_not_independent_test':True,
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    (OUT/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    data,forecast=Data(),AbsoluteHGBForecasts()
    prepared={name:[] for name in ('prefix','point')};errors=[];bases=[];audits=[]
    for day in range(31,365):
        issue=forecast.get(day,scenario='2');h=forecast.net_error_paths(day,'2',limit=28)
        f=(issue['load_kw']-issue['pv_kw'])/6
        assert h['audit']['max_observed_index']<day*144
        for name in prepared:
            prepared[name].append(trajectory(f,h['errors_kwh'],name)[0])
        errors.append(h['errors_kwh']);bases.append(f)
        audits.append({'day':day,'information_cutoff':day*144,
            'max_observed_index':day*144-1,'forecast':issue['audit'],
            'history':h['audit'],'training_origins':h['origins'].tolist()})
    prepared={k:np.stack(v) for k,v in prepared.items()}
    np.savez_compressed(OUT/'risk_inputs.npz',**prepared,errors=np.stack(errors),
                        forecast_net=np.stack(bases),days=np.arange(31,365))
    mutation=[]
    for day in (31,32,60,151,243,364):
        changed=copy.copy(forecast.data);changed.actual=forecast.data.actual.copy()
        changed.actual[day*144:]+=[70000.,30000.,1000.]
        after=AbsoluteHGBForecasts(changed)
        after.store.values[after.store.origins>day*144]+=60000.
        issue=after.get(day,scenario='2');h=after.net_error_paths(day,'2',limit=28)
        f=(issue['load_kw']-issue['pv_kw'])/6
        for name in prepared:
            np.testing.assert_array_equal(trajectory(f,h['errors_kwh'],name)[0],prepared[name][day-31])
        mutation.append({'day':day,'future_actual_and_future_forecasts_leave_both_risk_trajectories_identical':True})
    save(OUT/'causality_verification.json',{'passed':True,'checks':mutation})
    results=[run(name,data,forecast,prepared,audits) for name in ('prefix','point')]
    save(OUT/'summary.json',{'complete':True,'results':results,'no_final_model_selected':True})


if __name__=='__main__':
    main()
