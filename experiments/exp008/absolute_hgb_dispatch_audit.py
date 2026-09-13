"""Independent source, HGB-stage, memory, risk-path and physical trace audit."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.exp008.verify import battery_metrics, verify_npz

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'data/results/exp008/mode_planning_physical'
PRIMARY='direct_hgb_ridge28_memory_half'
ETA=float(np.sqrt(.9));LOW=1200.;HIGH=10800.;LIMIT=5000/6


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    with np.load(path,allow_pickle=False) as archive:
        return {key:archive[key].copy() for key in archive.files}


def main(days=334):
    directory=OUT/f'direct_hgb_memory_physical3_{days}days'
    archived=directory/'forecast_archive'
    detail=load(directory/'dispatch.npz')
    check=verify_npz(directory/'dispatch.npz',expected_days=days,audit_path=directory/'planning_audit.json')
    assert check['passed'],check['errors']
    provenance=json.loads((directory/'complete_provenance.json').read_text())
    for name,expected_hash in provenance['source_sha256'].items():
        assert digest(directory/'source_archive'/name)==expected_hash
    for name,expected_hash in provenance['forecast_artifact_sha256'].items():
        assert digest(archived/name)==expected_hash
    for name,expected_hash in provenance['source_data_sha256'].items():
        assert digest(ROOT/'data/raw'/name)==expected_hash
    assert provenance['block_slots']==6 and provenance['refinement_maxiter']==120
    assert provenance['deadband']==0 and provenance['scenarios']==3
    issued=load(directory/'issued_forecasts.npz')
    assert digest(directory/'issued_forecasts.npz')==provenance['issued_archive_sha256']
    raw=load(archived/'direct_hgb_raw.npz');ridge=load(archived/'direct_hgb_ridge28.npz')
    final=load(archived/f'{PRIMARY}.npz')
    for stage in (issued,raw,ridge,final):
        np.testing.assert_array_equal(stage['origins'],np.arange(31,365)*144)
    np.testing.assert_array_equal(issued['values'],final['values'])
    actual=np.stack([pd.read_csv(ROOT/'data/raw'/name).iloc[:,1:].to_numpy(float)
                     for name in ('附件2_小区负载.csv','附件2_光伏发电实际功率.csv')],axis=-1)
    reference=pd.read_csv(ROOT/'data/raw/附件1.csv').iloc[:,1:].to_numpy(float)
    training=json.loads((archived/'training_audit.json').read_text())
    predictions=json.loads((archived/'prediction_audit.json').read_text())
    ridge_audit=json.loads((archived/'ridge28_audit.json').read_text())
    memory_audit=json.loads((archived/'memory_half_audit.json').read_text())
    assert len(training)==11 and len(predictions)==len(ridge_audit)==len(memory_audit)==334
    for record in training:
        month=record['month'];asof=(pd.Timestamp(2025,month,1)-pd.Timestamp(2025,1,1)).days;cutoff=asof-7
        assert record['asof_day']==asof and record['training_days']==list(range(7,cutoff))
        assert record['validation_days']==list(range(cutoff,asof))
        assert record['train_label_stop_exclusive']==cutoff*144
        assert record['validation_label_stop_exclusive']==asof*144
        assert not record['periodic_or_CNN_reference_columns']
        for model in record['models']:
            assert model['reload_identical']
            assert model['model_sha256']==digest(archived/'models'/Path(model['model_path']).name)
    expected=ridge['values'].copy()
    for i,day in enumerate(range(31,365)):
        origin=day*144;prediction=predictions[i];ra=ridge_audit[i];ma=memory_audit[i]
        assert prediction['day']==day and prediction['origin']==origin
        assert prediction['training_label_stop']<prediction['validation_label_stop']<=origin
        assert prediction['feature_last_actual_index']==prediction['daylight_last_actual_index']==origin-1
        assert prediction['no_CNN_periodic_official_PV_future_price_input']
        assert prediction['raw_prediction_sha256']==hashlib.sha256(raw['values'][i].tobytes()).hexdigest()
        assert ra['origin']==origin and not ra['current_truth_used']
        assert ra['history_last_label'] is None or ra['history_last_label']<origin
        assert all(old+144<=origin for old in ra['history_origins'])
        assert ma['origin']==origin and ma['gain']==.5 and not ma['current_truth_used']
        assert not ma['recursive_corrected_output_residual']
        if i:
            correction=.5*float(np.mean(actual[day-1,:,0]-ridge['values'][i-1,:,0]))
            expected[i,:,0]=np.maximum(0,expected[i,:,0]+correction)
            assert ma['prior_forecast_origin']==origin-144 and ma['prior_label_end_exclusive']==origin
            assert ma['load_correction_kw']==correction
    np.testing.assert_array_equal(expected,issued['values'])
    future_checks=json.loads((archived/'causality_verification.json').read_text())
    assert future_checks['passed'] and future_checks['all_postprocessing_labels_past']

    def midnight(day):
        if day>=31:
            return expected[day-31]
        value=reference[:,1:3].copy()
        for channel,lag in ((0,7),(1,1)):
            old=day-lag if day>=lag else day-1
            if old>=0:
                value[:,channel]=actual[old,:,channel]
        return value

    audits=json.loads((directory/'planning_audit.json').read_text())
    assert [a['day'] for a in audits]==list(range(31,31+days))
    max_path=max_replay=max_ce=max_cd=max_balance=max_state=0.;previous_mode=1
    for i,day in enumerate(range(31,31+days)):
        planned=load(directory/f'planning_day{day}.npz');history=np.arange(day-28,day)
        residual=np.stack([((actual[old,:,0]-midnight(old)[:,0])
                            -(actual[old,:,1]-midnight(old)[:,1]))/6 for old in history])
        current=midnight(day);net=(current[:,0]-current[:,1])[None,:]/6+residual
        max_path=max(max_path,float(np.max(np.abs(net-planned['all_net_paths']))))
        assert max_path<1e-9
        selected=np.linspace(0,27,3).astype(int)
        np.testing.assert_array_equal(planned['selected_scenario_indices'],selected)
        assert audits[i]['forecast']['selected_model_id']==PRIMARY
        assert audits[i]['forecast']['direct_hgb_prediction']==predictions[i]
        assert audits[i]['forecast']['load_energy_memory']==memory_audit[i]
        assert audits[i]['history']['model_id']==PRIMARY
        assert audits[i]['history']['information_cutoff_exclusive']==day*144
        assert audits[i]['history']['max_observed_index']==day*144-1
        assert audits[i]['history']['fallback_days']==history[history<31].tolist()
        mask=planned['allowed_charge'];np.testing.assert_array_equal(mask,detail['allowed_charge'][i])
        assert np.all(mask.reshape(24,6)==mask[::6,None])
        assert int(planned['initial_mode'])==previous_mode and float(planned['initial_soc'])==detail['states'][i,0]
        q,c,d,e,w,s=(planned[k] for k in ('purchase','scenario_charge','scenario_discharge',
                                        'scenario_emergency','scenario_surplus','scenario_states'))
        max_balance=max(max_balance,float(np.max(np.abs(q[None,:]+d+e-c-w-net[selected]))))
        max_state=max(max_state,float(np.max(np.abs(np.diff(s,axis=1)-ETA*c+d/ETA))))
        max_ce=max(max_ce,float(np.maximum(0,np.minimum(c,e)).max()))
        max_cd=max(max_cd,float(np.maximum(0,np.minimum(c,d)).max()))
        assert max(max_balance,max_state,max_ce,max_cd)<1e-6
        assert s.min()>=LOW-1e-6 and s.max()<=HIGH+1e-6
        assert max(c.max(),d.max())<=LIMIT+1e-6 and min(c.min(),d.min(),e.min(),w.min(),q.min())>=-1e-6
        assert np.max(c[:,~mask],initial=0)<1e-6 and np.max(d[:,mask],initial=0)<1e-6
        assert np.all(e<=np.maximum(net[selected],0)+1e-6)
        assert np.all(q<=np.maximum(net[selected].max(axis=0),0)+LIMIT+1e-6)
        np.testing.assert_allclose(s[:,0],detail['states'][i,0],atol=1e-9,rtol=0)
        soc=detail['states'][i,0]
        for t in range(144):
            balance=detail['final'][i,t]+(actual[day,t,1]-actual[day,t,0])/6
            charge=min(max(balance,0),LIMIT,max(0,(HIGH-soc)/ETA)) if mask[t] else 0.
            discharge=min(max(-balance,0),LIMIT,max(0,(soc-LOW)*ETA)) if not mask[t] else 0.
            flow=np.array([charge,discharge,max(0,-balance-discharge),max(0,balance-charge)])
            stored=[detail[k][i,t] for k in ('charge','discharge','emergency','surplus')]
            max_replay=max(max_replay,float(np.max(np.abs(flow-stored))))
            soc+=ETA*charge-discharge/ETA
        assert abs(soc-detail['states'][i,-1])<1e-9 and max_replay<1e-9
        nonzero=np.sign(detail['charge'][i]-detail['discharge'][i]);nonzero=nonzero[nonzero!=0]
        if len(nonzero):
            previous_mode=int(nonzero[-1])
    prefix_equal={}
    if days==334:
        prefix=OUT/'direct_hgb_memory_physical3_3days';old=load(prefix/'dispatch.npz')
        prefix_equal={key:bool(np.array_equal(value,detail[key][:3])) for key,value in old.items()}
        assert all(prefix_equal.values()) and audits[:3]==json.loads((prefix/'planning_audit.json').read_text())
        for day in range(31,34):
            assert digest(directory/f'planning_day{day}.npz')==digest(prefix/f'planning_day{day}.npz')
    check.update(all_HGB_model_and_forecast_archive_hashes_passed=True,
                 monthly_training_cutoffs_checked=11,prediction_postprocessing_cutoffs_checked=334,
                 shared_memory_formula_checked_all334_days=True,
                 maximum_historical_path_error_kwh=max_path,maximum_greedy_replay_error_kwh=max_replay,
                 maximum_scenario_balance_error_kwh=max_balance,maximum_scenario_state_error_kwh=max_state,
                 maximum_scenario_emergency_charge_overlap_kwh=max_ce,
                 maximum_scenario_charge_discharge_overlap_kwh=max_cd,
                 prefix_arrays_equal=prefix_equal,nonanticipative_recourse_certificate=False,source_sha256=digest(__file__))
    (directory/'independent_verification.json').write_text(json.dumps(check,indent=2))
    (directory/'independent_audit_snapshot.py').write_bytes(Path(__file__).read_bytes())
    comparator=load(OUT/'load_memory_half_physical3_334days/dispatch.npz')
    comparator={key:value[:days] for key,value in comparator.items()}
    pieces=[load(p) for p in sorted((ROOT/'data/results/exp006/primary/checkpoints').glob('chunk_*.npz'))]
    baseline={key:np.concatenate([p[key] for p in pieces])[:days] for key in pieces[0]}
    comparison=[];monthly=[];dates=pd.date_range('2025-02-01',periods=days)
    for name,value in [('exp006',baseline),('joint_ridge28_memory_half',comparator),(PRIMARY,detail)]:
        comparison.append({'name':name,'total_cost':float(value['fees'].sum()),
                           'planned_cost':float(value['fees'][:,:,0].sum()),
                           'emergency_cost':float(value['fees'][:,:,3].sum()),**battery_metrics(value)})
        mode=1;power=172.75999999999976
        for month in sorted(set(dates.month)):
            sub={key:array[dates.month==month] for key,array in value.items()}
            battery=battery_metrics(sub,initial_mode=mode,initial_power_kw=power)
            monthly.append({'name':name,'month':month,'total_cost':float(sub['fees'].sum()),**battery})
            mode=battery['final_mode'];power=battery['final_power_kw']
    pd.DataFrame(comparison).to_csv(directory/'paired_comparison.csv',index=False)
    pd.DataFrame(monthly).to_csv(directory/'monthly_comparison.csv',index=False)
    before,after=comparison[1:]
    findings={'days':days,'independent_verification_passed':check['passed'],'metrics':comparison,
              'cost_change_vs_joint_memory_yuan':after['total_cost']-before['total_cost'],
              'cost_change_vs_joint_memory_pct':100*(after['total_cost']/before['total_cost']-1),
              'goal':check['goal'],'short_window_cost_selection_gate':False,
              'full_pipeline_comparison_not_raw_HGB_claim':True,'predeclared_evaluation_days':334,
              'cost_gap_to8pct_yuan':after['total_cost']-check['goal']['cost_threshold_yuan'] if days==334 else None,
              'mip_feasible_days':sum(a['mip']['feasible'] for a in audits),
              'maximum_mip_gap':max(a['mip']['mip_gap'] for a in audits),
              'refinement_success_days':sum(a['greedy_refinement']['success'] for a in audits),
              'refinement_maxiter':120,'hourly_mode_block_slots':6}
    (directory/'paired_findings.json').write_text(json.dumps(findings,indent=2))
    print(json.dumps(findings,indent=2),flush=True)
    return findings


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--days',type=int,default=334)
    main(parser.parse_args().days)
