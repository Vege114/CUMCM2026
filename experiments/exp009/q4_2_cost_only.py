"""One frozen-input annual Q4-2 control: remove only switch/throughput penalties.

No forecast fitting, action search against realized outcomes, or candidate selection.
The unchanged exp008 physical initializer and greedy refinement are imported.
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

from experiments.common.neural_v2.data import Data
from experiments.exp008.mode_planning_physical import plan
from experiments.exp008.closed_loop import optimize
from experiments.exp008.planner import execute
from experiments.exp008.verify import verify_arrays
from experiments.exp008.run_q4_hgb_linked_price_physical import local_dependency_closure

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'data/results/exp009/q4_2_cost_only'
BASE=ROOT/'data/results/exp008/q4_hgb_linked_price_physical'
HGB=ROOT/'data/results/exp008/forecast_absolute_hgb/direct_hgb_ridge28_memory_half.npz'
PRICE=ROOT/'data/results/exp008/hgb_linked_price_forecast/price_predictions.npz'
WARMUP=ROOT/'data/results/exp002/warmup_4-2.npz'
GLOBAL_PROTOCOL=ROOT/'experiments/exp009/protocol.json'
CONFIG={'scenario':'4-2','days':334,'start_day':31,'seed':42,'candidate_count':1,
 'block_slots':6,'initialization_scenarios':3,'refinement_history_days':28,
 'scenario_selection':'deterministic equally spaced historical origins',
 'switching':0.,'wear':0.,'throughput':0.,'variation':0.,'deadband':0.,
 'seconds':5.,'gap':.002,'refinement_maxiter':120,'refinement_terminal_value':.45,
 'final_day_terminal_value':0.,'initializer_terminal_value':'min(current predicted price)/ETA, unchanged; final day zero',
 'changed_from_exp008':{'switching':[50.,0.],'wear_and_refinement_throughput':[.002,0.]},
 'unchanged':'forecast values, historical residuals, common hourly masks, physical constraints, scenario count, time/gap budgets, refinement algorithm, terminal values, strict-mask execution and settlement',
 'price_information':'same scalar linked-HGB midnight price in every scenario; historical price residuals are archived but not used by the scalar objective',
 'frozen_input_policy':'read only exogenous arrays from original planning archives, never old purchase/masks/SOC/recourse actions',
 'partial_actual_cost_or_battery_counts_are_gates':False,'feasibility_gate_days':3,
 'no_retraining':True,'no_realized_outcome_selection':True,
 'global_optimality_certificate':False,'nonanticipative_recourse_certificate':False,
 'terminal_value_is_retained_economic_inventory_approximation':True}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def array_sha(value):return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
def write(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def read_npz(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def rel(path):return str(Path(path).relative_to(ROOT))

class FrozenInputs:
    """Only frozen exogenous inputs are loaded from the exp008 planning files."""
    def __init__(self):
        with np.load(HGB,allow_pickle=False) as z:
            self.values=z['values'].copy();self.origins=z['origins'].copy()
        with np.load(PRICE,allow_pickle=False) as z:
            self.prices=z['linked_hgb'].copy()
            np.testing.assert_array_equal(z['origins'],self.origins)
        np.testing.assert_array_equal(self.origins,np.arange(31,365)*144)
        assert self.values.shape==(334,144,2) and self.prices.shape==(334,144)
        self.audits=json.loads((BASE/'full334/source_corrected_audit.json').read_text())
        assert [x['day'] for x in self.audits]==list(range(31,365))
        self.paths={day:BASE/f'daily_chunks/planning_day_{day}.npz' for day in range(31,365)}
        self.hashes={day:sha(path) for day,path in self.paths.items()}

    def get(self,day):
        i=day-31;origin=day*144;path=self.paths[day]
        assert sha(path)==self.hashes[day]
        names=('all_net_paths','predicted_price','selected_scenario_indices','all_price_error_paths')
        with np.load(path,allow_pickle=False) as z:x={k:z[k].copy() for k in names}
        x['load_kw']=self.values[i,:,0].copy();x['pv_kw']=self.values[i,:,1].copy()
        np.testing.assert_array_equal(x['predicted_price'],self.prices[i])
        expected=np.linspace(0,27,3).astype(int)
        np.testing.assert_array_equal(x['selected_scenario_indices'],expected)
        assert x['all_net_paths'].shape==(28,144)
        assert x['all_price_error_paths'].shape==(28,144)
        assert np.isfinite(x['all_net_paths']).all() and np.all(x['predicted_price']>0)
        x['history_origins']=np.arange(day-28,day)*144
        a=self.audits[i]
        assert a['forecast']['information_cutoff_exclusive']==origin
        assert a['forecast']['max_observed_index']<origin
        assert a['forecast']['price_last_label']<origin
        assert not a['forecast']['known_future_price']
        assert a['history']['information_cutoff_exclusive']==origin
        assert a['history']['max_observed_index']<origin
        np.testing.assert_array_equal(a['selected_history_origins'],x['history_origins'][expected])
        x['forecast_audit']=copy.deepcopy(a['forecast']);x['history_audit']=copy.deepcopy(a['history'])
        x['source']=rel(path);x['source_sha256']=self.hashes[day]
        x['issued_array_sha256']={k:array_sha(x[k]) for k in names+('load_kw','pv_kw')}
        return x

def finish(out,details,audits,baseline,data,initial_soc,initial_mode,initial_power):
    days=len(details);directory=out/('feasibility3' if days==3 else 'full334')
    arrays={key:np.stack([d[key] for d in details]) for key in details[0]}
    arrays['days']=np.arange(31,31+days)
    actual=data.actual[31*144:(31+days)*144].reshape(days,144,3)
    args=dict(scenario='4-2',expected_days=days,initial_soc=initial_soc,
        initial_mode=initial_mode,initial_power_kw=initial_power,source_actual=actual,source_price=actual[...,2])
    check=verify_arrays(arrays,audit_records=audits,**args)
    old=verify_arrays({k:v[:days] for k,v in baseline.items()},**args)
    assert check['passed'],check['errors'];assert old['passed'],old['errors']
    before,after=old['billing']['total_cost'],check['billing']['total_cost']
    summary={'days':days,'complete':days==334,'scenario':'4-2','config':CONFIG,
        **check['billing'],'battery':check['battery_metrics'],
        'exp008_billing':old['billing'],'exp008_battery':old['battery_metrics'],
        'total_delta_yuan':after-before,'cost_reduction_pct':100*(1-after/before),
        'battery_deltas':{k:check['battery_metrics'][k]-v for k,v in old['battery_metrics'].items()
            if isinstance(v,(float,int)) and isinstance(check['battery_metrics'].get(k),(float,int))},
        'verified':True,'mip_feasible_days':sum(a['mip']['feasible'] for a in audits),
        'refinement_success_days':sum(a['refinement']['success'] for a in audits),
        'comparison_scope':'paired fixed external forecasts and unchanged feasible set/algorithm/budget; only switch and throughput soft penalties removed; realized price only for settlement',
        'partial_actual_metric_gate':False,'cost_guaranteed_to_decrease':False,
        'global_optimization_certificate':False,'nonanticipative_scenario_recourse_certificate':False}
    directory.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(directory/'dispatch_4-2.npz',**arrays)
    pd.DataFrame([{'day':31+i,'date':str((pd.Timestamp('2025-02-01')+pd.Timedelta(days=i)).date()),
        'total_cost':float(d['fees'].sum()),'planned_cost':float(d['fees'][:,0].sum()),
        'emergency_cost':float(d['fees'][:,3].sum()),'initial_soc':float(d['states'][0]),
        'final_soc':float(d['states'][-1])} for i,d in enumerate(details)]).to_csv(directory/'daily.csv',index=False)
    for name,value in [('audit',audits),('summary',summary),('verification',check),('baseline_verification',old)]:
        write(directory/f'{name}.json',value)
    print('STAGE',json.dumps({'days':days,'total_cost':after,'delta_yuan':after-before,'passed':True}),flush=True)
    return summary

def run(out=OUT):
    out=Path(out)
    if not GLOBAL_PROTOCOL.is_file():raise FileNotFoundError('exp009 global protocol must be frozen before running')
    if (out/'protocol.json').exists():raise FileExistsError('This fixed candidate has begun; no silent overwrite or parameter resume')
    frozen=FrozenInputs();data=Data();warmup=read_npz(WARMUP)
    soc=float(warmup['states'][-1,-1]);power=float(6*(warmup['charge'][-1,-1]-warmup['discharge'][-1,-1]))
    nonidle=np.sign((warmup['charge']-warmup['discharge']).ravel());nonidle=nonidle[nonidle!=0]
    mode=int(nonidle[-1]) if len(nonidle) else 0
    assert abs(soc-1390.382746315672)<1e-8
    initial_soc,initial_mode=soc,mode
    baseline_path=BASE/'full334/dispatch_4-2.npz';baseline=read_npz(baseline_path)
    sources=set(local_dependency_closure())|{Path(__file__).resolve(),GLOBAL_PROTOCOL,
        ROOT/'experiments/exp009/q4_2_cost_only_audit.py',ROOT/'pyproject.toml',ROOT/'uv.lock'}
    locked={rel(p):sha(p) for p in sorted(sources) if p.is_file()}
    artifacts={rel(p):sha(p) for p in [HGB,PRICE,WARMUP,baseline_path,BASE/'full334/source_corrected_audit.json']}
    provenance={'config':CONFIG,'global_protocol_sha256':sha(GLOBAL_PROTOCOL),
        'source_hashes':locked,'artifact_sha256':artifacts,'raw_source_hashes':data.hashes,
        'frozen_planning_source_sha256':{rel(frozen.paths[d]):frozen.hashes[d] for d in frozen.paths},
        'initial_soc_kwh':soc,'initial_mode':mode,'initial_power_kw':power,
        'issued_values_sha256':array_sha(frozen.values),'issued_price_sha256':array_sha(frozen.prices),
        'old_decision_arrays_never_loaded_for_action':True,'weights_loaded':False,
        'no_retraining_or_price_regression':True,'candidate_count':1}
    write(out/'protocol.json',provenance)
    for name in locked:
        p=ROOT/name;snapshot=out/'source_snapshots'/name;snapshot.parent.mkdir(parents=True,exist_ok=True)
        snapshot.write_bytes(p.read_bytes())
    np.savez_compressed(out/'issued_forecasts.npz',origins=frozen.origins,values=frozen.values,price=frozen.prices)
    chunks=out/'daily_chunks';chunks.mkdir(exist_ok=True)
    details=[];audits=[];began=perf_counter()
    try:
        for day in range(31,365):
            x=frozen.get(day);selected=x['selected_scenario_indices']
            initialized=plan(x['all_net_paths'][selected],x['predicted_price'],soc,mode,
                block_slots=6,switching=0.,wear=0.,seconds=5.,gap=.002,final=day==364)
            refined=optimize(initialized['purchase'],x['all_net_paths'],x['predicted_price'],soc,
                charge_mask=initialized['allowed_charge'],throughput=0.,variation=0.,
                terminal=0. if day==364 else .45,maxiter=120,deadband=0.)
            q=refined['purchase'];mask=initialized['allowed_charge'];hourly=mask[::6]
            assert np.array_equal(mask,np.repeat(hourly,6))
            mip=initialized['metadata'].copy()
            mip['unpenalized_flip_auxiliary_sum_not_a_behavior_metric']=mip.pop('planned_mode_changes')
            mip['planned_mode_changes']=int(np.count_nonzero(hourly[1:]!=hourly[:-1])+(bool(hourly[0])!=(mode>0)))
            mip['planned_mode_change_definition']='computed from hourly mask including previous real nonidle direction; never unpenalized auxiliary variables'
            # q and the complete mask are locked before current actual observations.
            observed=data.actual[day*144:(day+1)*144,:2]
            detail=execute(q,observed,x['predicted_price'],soc,charge_deadband=0.,charge_mask=mask)
            actual=data.actual[day*144:(day+1)*144].copy();price=actual[:,2]
            detail.update(original=q,final=q.copy(),actual=actual,price=price.copy(),allowed_charge=mask.copy())
            detail['fees']=np.stack((q*price,np.zeros(144),np.zeros(144),5*detail['emergency']*price),axis=-1)
            audit={'day':day,'information_cutoff':day*144,'forecast':x['forecast_audit'],'history':x['history_audit'],
                'mip':mip,'refinement':refined['metadata'],'initial_soc':soc,'initial_mode':mode,
                'selected_history_origins':x['history_origins'][selected].tolist(),
                'history_origins':x['history_origins'].tolist(),'frozen_source':x['source'],
                'frozen_source_sha256':x['source_sha256'],'issued_array_sha256':x['issued_array_sha256'],
                'purchase_locked_before_current_actual_read':True,'actual_price_used_only_in_settlement':True,
                'optimization_price_source':'same exp008 frozen linked-HGB midnight forecast',
                'realized_outcomes_used_to_select_or_change_parameters':False}
            np.savez_compressed(chunks/f'planning_day_{day}.npz',initial_purchase=initialized['purchase'],
                refined_purchase=q,allowed_charge=mask,initial_soc=np.array(soc),initial_mode=np.array(mode),
                all_net_paths=x['all_net_paths'],predicted_price=x['predicted_price'],
                selected_scenario_indices=selected,all_price_error_paths=x['all_price_error_paths'],
                load_kw=x['load_kw'],pv_kw=x['pv_kw'],history_origins=x['history_origins'],
                **{key:initialized[key] for key in ('scenario_charge','scenario_discharge','scenario_emergency','scenario_surplus','scenario_states')})
            np.savez_compressed(chunks/f'day_{day}.npz',**detail);write(chunks/f'day_{day}.json',audit)
            details.append(detail);audits.append(audit);soc=float(detail['states'][-1])
            signs=np.sign(detail['charge']-detail['discharge']);active=signs[signs!=0]
            if len(active):mode=int(active[-1])
            if len(details)==3:
                first=finish(out,details,audits,baseline,data,initial_soc,initial_mode,power)
                assert first['verified'] and first['mip_feasible_days']==3
                write(out/'first3_feasibility_gate.json',{'passed':True,'actual_cost_or_counts_used_for_gate':False})
            if len(details)%5==0:
                print('PROGRESS',json.dumps({'days':len(details),'seconds':perf_counter()-began,
                    'total_cost':sum(float(a['fees'].sum()) for a in details),'last_mip_gap':mip['mip_gap']}),flush=True)
    except BaseException as error:
        write(out/'interruption.json',{'completed_days':len(details),'next_day':31+len(details),
            'exception':type(error).__name__,'message':str(error),'complete':False,
            'elapsed_seconds':perf_counter()-began,'no_parameter_update_or_candidate_selection':True})
        raise
    summary=finish(out,details,audits,baseline,data,initial_soc,initial_mode,power)
    first=read_npz(out/'feasibility3/dispatch_4-2.npz');full=read_npz(out/'full334/dispatch_4-2.npz')
    prefix={k:bool(np.array_equal(v,full[k][:3])) for k,v in first.items()};assert all(prefix.values())
    write(out/'first3_prefix_equality.json',prefix)
    assert all(sha(ROOT/p)==h for p,h in locked.items())
    assert all(sha(ROOT/p)==h for p,h in artifacts.items())
    assert all(sha(frozen.paths[d])==frozen.hashes[d] for d in frozen.paths)
    write(out/'completion.json',{'complete':True,'days':334,'wall_seconds':perf_counter()-began,
        'sources_and_forecast_artifacts_unchanged':True,'all334_inputs_exactly_exp008':True,
        'no_retraining':True,'no_partial_cost_selection':True,'config':CONFIG})
    return summary

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,default=OUT)
    args=parser.parse_args();run(args.out)
