"""Independent frozen-source, actual-billing and causal-prefix audit for Q4-2."""
from pathlib import Path
import argparse
import json
import numpy as np

from experiments.common.neural_v2.data import Data
from experiments.exp008.planner import execute
from experiments.exp008.verify import battery_metrics
from experiments.exp009.q4_2_cost_only import ROOT,OUT,BASE,HGB,PRICE,WARMUP,CONFIG,sha,array_sha,write,read_npz

def independent_equations(a,data):
    c,d,e,w=(a[k] for k in ('charge','discharge','emergency','surplus'))
    actual=data.actual[31*144:].reshape(334,144,3);price=actual[:,:,2]
    np.testing.assert_array_equal(a['actual'],actual);np.testing.assert_array_equal(a['price'],price)
    np.testing.assert_array_equal(a['original'],a['final'])
    expected=np.stack((a['original']*price,np.zeros_like(c),np.zeros_like(c),5*e*price),axis=-1)
    balance=a['final']+e+d+actual[:,:,1]/6-actual[:,:,0]/6-c-w
    transition=np.diff(a['states'],axis=1)-np.sqrt(.9)*c+d/np.sqrt(.9)
    mask=a['allowed_charge'].astype(bool)
    values={'balance_max_abs_kwh':float(abs(balance).max()),'state_equation_max_abs_kwh':float(abs(transition).max()),
        'midnight_SOC_max_abs_kwh':float(abs(a['states'][1:,0]-a['states'][:-1,-1]).max()),
        'fee_max_abs_yuan':float(abs(expected-a['fees']).max()),
        'simultaneous_slots':int(np.count_nonzero((c>1e-6)&(d>1e-6))),
        'emergency_charging_slots':int(np.count_nonzero((c>1e-6)&(e>1e-6))),
        'charge_forbidden_mask_slots':int(np.count_nonzero((c>1e-6)&~mask)),
        'discharge_forbidden_mask_slots':int(np.count_nonzero((d>1e-6)&mask)),
        'cost_components_yuan':expected.sum(axis=(0,1)).tolist(),'total_cost_yuan':float(expected.sum()),
        'minimum_soc_kwh':float(a['states'].min()),'maximum_soc_kwh':float(a['states'].max()),
        'maximum_charge_discharge_kwh':float(max(c.max(),d.max())),
        'minimum_nonnegative_flow_kwh':float(min(a[k].min() for k in ('original','final','charge','discharge','emergency','surplus')))}
    assert all(values[k]<1e-6 for k in ('balance_max_abs_kwh','state_equation_max_abs_kwh','midnight_SOC_max_abs_kwh','fee_max_abs_yuan'))
    assert all(values[k]==0 for k in ('simultaneous_slots','emergency_charging_slots','charge_forbidden_mask_slots','discharge_forbidden_mask_slots'))
    assert values['minimum_soc_kwh']>=1200-1e-6 and values['maximum_soc_kwh']<=10800+1e-6
    assert values['maximum_charge_discharge_kwh']<=5000/6+1e-6 and values['minimum_nonnegative_flow_kwh']>=-1e-6
    values['passed']=True
    return values

def run(out=OUT):
    out=Path(out);protocol=json.loads((out/'protocol.json').read_text())
    assert protocol['config']==CONFIG
    assert all(sha(ROOT/p)==h for p,h in protocol['source_hashes'].items())
    assert all(sha(ROOT/p)==h for p,h in protocol['artifact_sha256'].items())
    assert all(sha(ROOT/p)==h for p,h in protocol['frozen_planning_source_sha256'].items())
    a=read_npz(out/'full334/dispatch_4-2.npz');old=read_npz(BASE/'full334/dispatch_4-2.npz')
    np.testing.assert_array_equal(a['days'],np.arange(31,365))
    data=Data();direct=independent_equations(a,data);previous=independent_equations(old,data)
    hgb=read_npz(HGB);prices=read_npz(PRICE);issued=read_npz(out/'issued_forecasts.npz')
    np.testing.assert_array_equal(issued['values'],hgb['values']);np.testing.assert_array_equal(issued['price'],prices['linked_hgb'])
    np.testing.assert_array_equal(issued['origins'],hgb['origins'])
    audits=json.loads((out/'full334/audit.json').read_text());assert len(audits)==334
    warmup=read_npz(WARMUP);soc=float(warmup['states'][-1,-1]);real=np.sign((warmup['charge']-warmup['discharge']).ravel());real=real[real!=0]
    mode=int(real[-1]) if len(real) else 0
    actual=data.actual.reshape(365,144,3)
    # January is the archived causal periodic cold start; no learned future model.
    historic_predictions={day:np.column_stack((actual[day-7 if day>=7 else day-1,:,0],actual[day-1,:,1])) for day in range(3,31)}
    historic_predictions.update({day:issued['values'][day-31] for day in range(31,365)})
    equal_rows=[];prefix=[]
    for i,day in enumerate(range(31,365)):
        path=out/f'daily_chunks/planning_day_{day}.npz';p=read_npz(path)
        source=BASE/f'daily_chunks/planning_day_{day}.npz'
        with np.load(source,allow_pickle=False) as z:
            for key in ('all_net_paths','predicted_price','selected_scenario_indices','all_price_error_paths'):
                np.testing.assert_array_equal(p[key],z[key])
        np.testing.assert_array_equal(p['load_kw'],issued['values'][i,:,0]);np.testing.assert_array_equal(p['pv_kw'],issued['values'][i,:,1])
        np.testing.assert_array_equal(p['predicted_price'],issued['price'][i])
        history=np.arange(day-28,day);truth=actual[history,:,:2]
        past=np.stack([historic_predictions[int(d)] for d in history]);errors=truth-past
        expected=(issued['values'][i,:,0]-issued['values'][i,:,1])[None,:]/6+(errors[:,:,0]-errors[:,:,1])/6
        np.testing.assert_array_equal(p['all_net_paths'],expected)
        np.testing.assert_array_equal(p['history_origins'],history*144)
        assert np.all(p['history_origins']+144<=day*144)
        np.testing.assert_allclose(p['initial_soc'],soc,rtol=0,atol=1e-8);assert int(p['initial_mode'])==mode
        np.testing.assert_array_equal(p['refined_purchase'],a['original'][i]);np.testing.assert_array_equal(p['allowed_charge'],a['allowed_charge'][i])
        hour=p['allowed_charge'][::6];np.testing.assert_array_equal(np.repeat(hour,6),p['allowed_charge'])
        flips=int(np.count_nonzero(hour[1:]!=hour[:-1])+(bool(hour[0])!=(mode>0)))
        assert audits[i]['mip']['planned_mode_changes']==flips
        for key,h in audits[i]['issued_array_sha256'].items():assert array_sha(p[key])==h
        replay=execute(p['refined_purchase'],actual[day,:,:2],p['predicted_price'],soc,charge_deadband=0.,charge_mask=p['allowed_charge'])
        for key in ('charge','discharge','emergency','surplus','states'):np.testing.assert_array_equal(replay[key],a[key][i])
        if day in (31,32,33):
            for cutoff in (1,36,72,108,143):
                mutated=actual[day,:,:2].copy();mutated[cutoff:]+=np.array([1e7,5e6])
                alternate=execute(p['refined_purchase'],mutated,p['predicted_price'],soc,charge_deadband=0.,charge_mask=p['allowed_charge'])
                for key in ('charge','discharge','emergency','surplus'):np.testing.assert_array_equal(alternate[key][:cutoff],replay[key][:cutoff])
                np.testing.assert_array_equal(alternate['states'][:cutoff+1],replay['states'][:cutoff+1])
                prefix.append({'day':day,'future_mutation_after_slot':cutoff,'passed':True})
        equal_rows.append({'day':day,'frozen_source_sha256':sha(source),'new_planning_archive_sha256':sha(path),
            'issued_prediction_and_history_exact':True,'historical_error_reconstruction_exact':True,
            'initial_soc_own_continuous':True,'previous_mode_own_real_nonidle':True,'mask_flips_from_mask':flips})
        soc=float(a['states'][i,-1]);real=np.sign(a['charge'][i]-a['discharge'][i]);real=real[real!=0]
        if len(real):mode=int(real[-1])
    first=read_npz(out/'feasibility3/dispatch_4-2.npz')
    for key,v in first.items():np.testing.assert_array_equal(v,a[key][:3])
    result={'passed':True,'days':334,'source_hashes_before_after_passed':True,
        'actual_equations':direct,'exp008_actual_equations':previous,
        'new_battery':battery_metrics(a),'exp008_battery':battery_metrics(old),
        'total_delta_yuan':direct['total_cost_yuan']-previous['total_cost_yuan'],
        'cost_reduction_pct':100*(1-direct['total_cost_yuan']/previous['total_cost_yuan']),
        'forecast_and_history_days':equal_rows,'execution_future_mutation_prefix_checks':prefix,
        'initial_SOC_same_warmup_then_own_continuous':True,'first3_full_prefix_exact':True,
        'old_decisions_not_used_as_action_inputs':True,'all_forecast_arrays_exact_exp008':True,
        'no_retraining_no_actual_outcome_selection':True,
        'cost_only_scope':'remove switch/throughput soft weights; retain economic terminal value, physical constraints and fixed approximate planning algorithm',
        'no_global_or_nonanticipative_recourse_certificate':True,'audit_script_sha256':sha(Path(__file__))}
    write(out/'independent_audit.json',result)
    print(json.dumps({k:result[k] for k in ('passed','days','total_delta_yuan','cost_reduction_pct')},ensure_ascii=False))
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,default=OUT)
    run(parser.parse_args().out)
