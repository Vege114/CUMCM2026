"""Independent memory, scenario, execution and billing audit from source CSVs."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.exp008.verify import battery_metrics, verify_npz

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'data/results/exp008/mode_planning_physical'
ETA=float(np.sqrt(.9));LOW=1200.;HIGH=10800.;LIMIT=5000/6


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    with np.load(path,allow_pickle=False) as archive:
        return {key:archive[key].copy() for key in archive.files}


def main(days=30):
    directory=OUT/f'load_memory_half_physical3_{days}days'
    detail=load(directory/'dispatch.npz')
    check=verify_npz(directory/'dispatch.npz',expected_days=days,audit_path=directory/'planning_audit.json')
    assert check['passed'],check['errors']
    provenance=json.loads((directory/'complete_provenance.json').read_text())
    for name,expected in provenance['source_sha256'].items():
        assert digest(directory/'source_archive'/name)==expected
    for name,expected in provenance['artifact_sha256'].items():
        assert digest(directory/'forecast_archive'/name)==expected
    for name,expected in provenance['source_data_sha256'].items():
        assert digest(ROOT/'data/raw'/name)==expected
    issued=load(directory/'issued_memory_forecasts.npz')
    assert digest(directory/'issued_memory_forecasts.npz')==provenance['archive_sha256']
    base=load(directory/'forecast_archive/neural_joint_calibration/joint_ridge28.npz')
    np.testing.assert_array_equal(issued['base_values'],base['values'])
    np.testing.assert_array_equal(issued['origins'],base['origins'])
    np.testing.assert_array_equal(issued['origins'],np.arange(31,365)*144)
    neural_proof_path=OUT/'joint_ridge28_refined_s3_334days/independent_annual_verification.json'
    neural_proof=json.loads(neural_proof_path.read_text())
    assert neural_proof['passed'] and len(neural_proof['monthly_training_checks'])==11
    reference_base=load(OUT/'joint_ridge28_refined_s3_334days/dispatch.npz')
    assert reference_base['actual'].shape[0]==334
    signed_base=ROOT/'data/results/exp008/neural_joint_calibration/joint_ridge28.npz'
    assert digest(signed_base)==digest(directory/'forecast_archive/neural_joint_calibration/joint_ridge28.npz')
    actual=np.stack([pd.read_csv(ROOT/'data/raw'/name).iloc[:,1:].to_numpy(float)
                     for name in ('附件2_小区负载.csv','附件2_光伏发电实际功率.csv')],axis=-1)
    reference=pd.read_csv(ROOT/'data/raw/附件1.csv').iloc[:,1:].to_numpy(float)
    expected=base['values'].copy()
    memory_audit=json.loads((directory/'forecast_archive/load_energy_memory/prediction_audit.json').read_text())
    for i,day in enumerate(range(31,365)):
        record=memory_audit[i]
        assert record['origin']==day*144 and record['gain']==.5 and not record['current_truth_used']
        assert record['pv_unchanged'] and not record['recursive_corrected_output_residual']
        if i:
            correction=.5*float(np.mean(actual[day-1,:,0]-base['values'][i-1,:,0]))
            expected[i,:,0]=np.maximum(0,expected[i,:,0]+correction)
            assert record['prior_forecast_origin']==(day-1)*144
            assert record['prior_label_end_exclusive']==day*144
            assert record['prior_max_observed_index']==day*144-1
            assert record['load_correction_kw']==correction
    np.testing.assert_array_equal(issued['values'],expected)

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
    previous_mode=1;max_path=max_replay=max_ce=max_cd=0.
    scenario_rows=[]
    for i,day in enumerate(range(31,31+days)):
        planned=load(directory/f'planning_day{day}.npz')
        history=np.arange(day-28,day)
        residual=np.stack([((actual[old,:,0]-midnight(old)[:,0])
                             -(actual[old,:,1]-midnight(old)[:,1]))/6 for old in history])
        current=midnight(day)
        net=(current[:,0]-current[:,1])[None,:]/6+residual
        max_path=max(max_path,float(np.max(np.abs(net-planned['all_net_paths']))))
        assert max_path<1e-9
        selected=np.linspace(0,27,3).astype(int)
        np.testing.assert_array_equal(planned['selected_scenario_indices'],selected)
        assert audits[i]['forecast']['load_energy_memory']==memory_audit[i]
        assert audits[i]['history']['information_cutoff_exclusive']==day*144
        assert audits[i]['history']['max_observed_index']==day*144-1
        assert audits[i]['history']['fallback_days']==history[history<31].tolist()
        mask=planned['allowed_charge']
        np.testing.assert_array_equal(mask,detail['allowed_charge'][i])
        assert np.all(mask.reshape(24,6)==mask[::6,None])
        assert int(planned['initial_mode'])==previous_mode
        assert float(planned['initial_soc'])==detail['states'][i,0]
        q,c,d,e,w,s=(planned[k] for k in ('purchase','scenario_charge','scenario_discharge',
                                        'scenario_emergency','scenario_surplus','scenario_states'))
        balance=float(np.max(np.abs(q[None,:]+d+e-c-w-net[selected])))
        state=float(np.max(np.abs(np.diff(s,axis=1)-ETA*c+d/ETA)))
        max_ce=max(max_ce,float(np.maximum(0,np.minimum(c,e)).max()))
        max_cd=max(max_cd,float(np.maximum(0,np.minimum(c,d)).max()))
        assert max(balance,state,max_ce,max_cd)<1e-6
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
        scenario_rows.append({'day':day,'scenario_state_error_kwh':state,
                              'mip_gap':audits[i]['mip']['mip_gap'],
                              'archive_sha256':digest(directory/f'planning_day{day}.npz')})
    prefix_equal={}
    if days==334:
        prefix=OUT/'load_memory_half_physical3_30days'
        old=load(prefix/'dispatch.npz')
        prefix_equal={key:bool(np.array_equal(value,detail[key][:30])) for key,value in old.items()}
        assert all(prefix_equal.values())
        assert audits[:30]==json.loads((prefix/'planning_audit.json').read_text())
        for day in range(31,61):
            assert digest(directory/f'planning_day{day}.npz')==digest(prefix/f'planning_day{day}.npz')
        amendment=json.loads((directory/'amended_evaluation_protocol.json').read_text())
        pilot_gate=json.loads((prefix/'paired_findings.json').read_text())
        assert not pilot_gate['annual_extension_gate_passed']
        assert not amendment['original_30day_extension_gate_passed']
        assert amendment['original_30day_gate_preserved_not_reclassified']
        assert not amendment['candidate_hyperparameters_changed']
        assert amendment['original_gate_artifact_sha256']==digest(prefix/'paired_findings.json')
    check.update(memory_forecast_formula_checked_all_334_days=True,
                 source_and_dependency_archive_hashes_passed=True,
                 memory_gain=.5,historical_risk_paths_checked=days*28,
                 maximum_historical_path_error_kwh=max_path,maximum_greedy_replay_error_kwh=max_replay,
                 maximum_scenario_emergency_charge_overlap_kwh=max_ce,
                 maximum_scenario_charge_discharge_overlap_kwh=max_cd,
                 scenario_checks=scenario_rows,prefix_arrays_equal=prefix_equal,
                 shared_joint_training_evidence_sha256=digest(neural_proof_path),
                 shared_joint_forecast_archive_identical=True,
                 nonanticipative_recourse_certificate=False,source_sha256=digest(__file__))
    (directory/'independent_verification.json').write_text(json.dumps(check,indent=2))
    (directory/'independent_audit_snapshot.py').write_bytes(Path(__file__).read_bytes())
    (directory/'shared_joint_training_evidence.json').write_bytes(neural_proof_path.read_bytes())
    joint=load(OUT/f'joint_ridge28_refined_s3_{days}days/dispatch.npz')
    pieces=[load(p) for p in sorted((ROOT/'data/results/exp006/primary/checkpoints').glob('chunk_*.npz'))]
    baseline={key:np.concatenate([p[key] for p in pieces])[:days] for key in pieces[0]}
    comparison=[];monthly=[]
    dates=pd.date_range('2025-02-01',periods=days)
    for name,value in [('exp006',baseline),('joint_ridge28',joint),('memory_half',detail)]:
        metrics=battery_metrics(value)
        comparison.append({'name':name,'total_cost':float(value['fees'].sum()),
                           'planned_cost':float(value['fees'][:,:,0].sum()),
                           'emergency_cost':float(value['fees'][:,:,3].sum()),**metrics})
        mode=1;power=172.75999999999976
        for month in sorted(set(dates.month)):
            sub={key:array[dates.month==month] for key,array in value.items()}
            battery=battery_metrics(sub,initial_mode=mode,initial_power_kw=power)
            monthly.append({'name':name,'month':month,'total_cost':float(sub['fees'].sum()),**battery})
            mode=battery['final_mode'];power=battery['final_power_kw']
    pd.DataFrame(comparison).to_csv(directory/'paired_comparison.csv',index=False)
    pd.DataFrame(monthly).to_csv(directory/'monthly_comparison.csv',index=False)
    base_row,joint_row,memory_row=comparison
    gate=memory_row['total_cost']<joint_row['total_cost'] and memory_row['direction_reversals']<base_row['direction_reversals']
    findings={'days':days,'metrics':comparison,'cost_change_vs_joint_yuan':memory_row['total_cost']-joint_row['total_cost'],
              'cost_change_vs_joint_pct':100*(memory_row['total_cost']/joint_row['total_cost']-1),
              'annual_extension_gate_passed':gate if days==30 else None,
              'annual_extension_rule':'30day expense below fixed joint30 AND reversals below same-period exp006',
              'independent_verification_passed':check['passed'],'goal':check['goal'],
              'cost_gap_to_8pct_yuan':memory_row['total_cost']-check['goal']['cost_threshold_yuan'] if days==334 else None,
              'prediction_and_history_memory_gain':.5,'no_new_hyperparameter_search':True,
              'scenario_count':3,'hourly_modes':True,'refinement_maxiter':120,
              'refinement_success_days':sum(a['greedy_refinement']['success'] for a in audits),
              'mip_feasible_days':sum(a['mip']['feasible'] for a in audits),
              'maximum_mip_gap':max(a['mip']['mip_gap'] for a in audits),
              'sum_archived_mip_seconds':sum(a['mip']['seconds'] for a in audits),
              'sum_archived_refinement_seconds':sum(a['greedy_refinement']['planning_seconds'] for a in audits)}
    (directory/'paired_findings.json').write_text(json.dumps(findings,indent=2))
    print(json.dumps(findings,indent=2),flush=True)
    return findings


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--days',type=int,default=30)
    main(parser.parse_args().days)
