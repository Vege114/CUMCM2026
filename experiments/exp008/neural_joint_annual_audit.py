"""Independent archived-array audit and annual method comparison.

Does not import the optimizer, physical planner, forecast adapter or executor.
Forecast training/mutation evidence is linked separately from trace arithmetic.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.exp008.verify import battery_metrics, source_arrays, verify_npz

ROOT=Path(__file__).resolve().parents[2]
RESULTS=ROOT/'data/results/exp008'
CASE=RESULTS/'mode_planning_physical/joint_ridge28_refined_s3_334days'
PREFIX=RESULTS/'mode_planning_physical/joint_ridge28_refined_s3_30days'
ETA=float(np.sqrt(.9));LOW=1200.;HIGH=10800.;LIMIT=5000/6


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    with np.load(path,allow_pickle=False) as archive:
        return {key:archive[key].copy() for key in archive.files}


def main():
    detail=load(CASE/'dispatch.npz')
    source,price=source_arrays('2')
    check=verify_npz(CASE/'dispatch.npz',expected_days=334,
                     audit_path=CASE/'planning_audit.json')
    assert check['passed'],check['errors']
    prefix=load(PREFIX/'dispatch.npz')
    prefix_checks={key:bool(np.array_equal(value,detail[key][:30])) for key,value in prefix.items()}
    assert all(prefix_checks.values())
    archives=load(RESULTS/'neural_joint_calibration/joint_ridge28.npz')
    np.testing.assert_array_equal(archives['origins'],np.arange(31,365)*144)
    provenance=json.loads((CASE/'provenance.json').read_text())
    for filename,expected in provenance['source_data_hashes'].items():
        assert digest(ROOT/'data/raw'/filename)==expected
    assert hashlib.sha256(archives['values'].tobytes()).hexdigest()==provenance['calibrated_forecast_values_sha256']
    cal_manifest=json.loads((RESULTS/'neural_joint_calibration/manifest.json').read_text())
    assert digest(RESULTS/'neural_joint_calibration/joint_ridge28.npz')==cal_manifest['joint_calibrated_archive_sha256']
    annual_actual=np.stack([pd.read_csv(ROOT/'data/raw'/name).iloc[:,1:].to_numpy(float)
                           for name in ('附件2_小区负载.csv','附件2_光伏发电实际功率.csv')],axis=-1)
    reference=pd.read_csv(ROOT/'data/raw/附件1.csv').iloc[:,1:].to_numpy(float)[:,1:3]

    def midnight(day):
        if day>=31:
            return archives['values'][day-31]
        prediction=reference.copy()
        for channel,lag in ((0,7),(1,1)):
            old=day-lag if day>=lag else day-1
            if old>=0:
                prediction[:,channel]=annual_actual[old,:,channel]
        return prediction

    audit=json.loads((CASE/'planning_audit.json').read_text())
    assert [row['day'] for row in audit]==list(range(31,365))
    assert audit[:30]==json.loads((PREFIX/'planning_audit.json').read_text())
    prefix_planning_hashes={str(day):digest(CASE/f'planning_day{day}.npz')==digest(PREFIX/f'planning_day{day}.npz')
                            for day in range(31,61)}
    assert all(prefix_planning_hashes.values())
    previous_mode=1
    maximum_path_error=0.;maximum_replay_error=0.
    scenario_rows=[]
    for index,day in enumerate(range(31,365)):
        plan=load(CASE/f'planning_day{day}.npz')
        history=np.arange(day-28,day)
        errors=np.stack([((annual_actual[old,:,0]-midnight(old)[:,0])
                          -(annual_actual[old,:,1]-midnight(old)[:,1]))/6 for old in history])
        issued=midnight(day)
        expected_paths=(issued[:,0]-issued[:,1])[None,:]/6+errors
        maximum_path_error=max(maximum_path_error,float(np.max(np.abs(expected_paths-plan['all_net_paths']))))
        assert maximum_path_error<1e-9
        selected=np.linspace(0,27,3).astype(int)
        np.testing.assert_array_equal(plan['selected_scenario_indices'],selected)
        assert all((history+1)*144<=day*144)
        assert audit[index]['history']['information_cutoff_exclusive']==day*144
        assert audit[index]['history']['max_observed_index']==day*144-1
        assert audit[index]['history']['fallback_days']==history[history<31].tolist()
        assert audit[index]['forecast']['origin']==day*144
        assert audit[index]['forecast']['base_forecast']=='exp008_joint_shared_cnn'
        assert int(plan['initial_mode'])==previous_mode
        assert float(plan['initial_soc'])==detail['states'][index,0]
        np.testing.assert_array_equal(plan['price'],price)
        mask=plan['allowed_charge']
        np.testing.assert_array_equal(mask,detail['allowed_charge'][index])
        assert np.all(mask.reshape(24,6)==mask[::6,None])
        q,c,d,e,w,s=(plan[key] for key in ('purchase','scenario_charge','scenario_discharge',
                                          'scenario_emergency','scenario_surplus','scenario_states'))
        net=plan['all_net_paths'][selected]
        scenario_balance=float(np.max(np.abs(q[None,:]+d+e-c-w-net)))
        state=float(np.max(np.abs(np.diff(s,axis=1)-ETA*c+d/ETA)))
        assert max(scenario_balance,state)<1e-6
        assert s.min()>=LOW-1e-6 and s.max()<=HIGH+1e-6
        assert max(c.max(),d.max())<=LIMIT+1e-6
        assert min(c.min(),d.min(),e.min(),w.min(),q.min())>=-1e-6
        assert not np.any((c>1e-6)&(e>1e-6))
        assert not np.any((c>1e-6)&(d>1e-6))
        assert np.max(c[:,~mask],initial=0.)<1e-6 and np.max(d[:,mask],initial=0.)<1e-6
        assert np.all(e<=np.maximum(net,0)+1e-6)
        assert np.all(q<=np.maximum(net.max(axis=0),0)+LIMIT+1e-6)
        np.testing.assert_allclose(s[:,0],detail['states'][index,0],atol=1e-9,rtol=0)
        # Independently replay greedy flows using only present balance and SOC.
        soc=detail['states'][index,0]
        for t in range(144):
            balance=detail['final'][index,t]+(source[index,t,1]-source[index,t,0])/6
            charge=min(max(balance,0),LIMIT,max(0,(HIGH-soc)/ETA)) if mask[t] else 0.
            discharge=min(max(-balance,0),LIMIT,max(0,(soc-LOW)*ETA)) if not mask[t] else 0.
            expected=(charge,discharge,max(0,-balance-discharge),max(0,balance-charge))
            stored=[detail[key][index,t] for key in ('charge','discharge','emergency','surplus')]
            maximum_replay_error=max(maximum_replay_error,float(np.max(np.abs(np.asarray(expected)-stored))))
            soc+=ETA*charge-discharge/ETA
        maximum_replay_error=max(maximum_replay_error,abs(soc-detail['states'][index,-1]))
        assert maximum_replay_error<1e-9
        nonzero=np.sign(detail['charge'][index]-detail['discharge'][index])
        nonzero=nonzero[nonzero!=0]
        if len(nonzero):
            previous_mode=int(nonzero[-1])
        scenario_rows.append({'day':day,'scenarios':len(selected),'balance_error_kwh':scenario_balance,
                              'state_error_kwh':state,'no_emergency_charging':True,
                              'no_simultaneous_charge_discharge':True,
                              'mip_gap':audit[index]['mip']['mip_gap'],
                              'mip_feasible':audit[index]['mip']['feasible'],
                              'refinement_success':audit[index]['greedy_refinement']['success'],
                              'archive_sha256':digest(CASE/f'planning_day{day}.npz')})
    calibration=json.loads((RESULTS/'neural_joint_calibration/joint_ridge28_audit.json').read_text())
    assert len(calibration['days'])==334
    for day,entry in zip(range(31,365),calibration['days'],strict=True):
        assert entry['origin']==day*144
        assert entry['max_actual_index']<day*144 and not entry['current_truth_used']
        assert all(old+144<=day*144 for old in entry['history_origins'])
        assert entry['history_last_label'] is None or entry['history_last_label']<day*144
    neural=json.loads((RESULTS/'neural_joint/findings.json').read_text())
    assert len(neural['all_monthly_checks'])==11
    assert all(all(row[key] for key in ('historical_split_passed','save_reload_passed','future_features_passed'))
               for row in neural['all_monthly_checks'])
    neural_manifest=json.loads((RESULTS/'neural_joint/manifest.json').read_text())
    run_directory=Path(neural_manifest['run_directory'])
    if not run_directory.is_absolute():
        run_directory=ROOT/run_directory
    raw_neural=load(RESULTS/'neural_joint/predictions.npz')
    assert digest(RESULTS/'neural_joint/predictions.npz')==neural_manifest['archive_sha256']
    training_checks=[]
    for month in range(2,13):
        stem=run_directory/f'joint_m{month:02d}_s42'
        model=json.loads(stem.with_suffix('.json').read_text())
        issue=(pd.Timestamp(2025,month,1)-pd.Timestamp(2025,1,1)).days
        cutoff=issue-7
        assert model['asof_day']==issue and model['scaler_cutoff_day']==cutoff
        assert model['train_days']==list(range(7,cutoff))
        assert model['validation_days']==list(range(cutoff,issue))
        assert model['train_latest_label_exclusive']==cutoff*144
        assert model['validation_latest_label_exclusive']==issue*144
        assert model['net_scaler_latest_label_exclusive']==cutoff*144
        observed=annual_actual[:cutoff,:,0]-annual_actual[:cutoff,:,1]
        assert abs(model['net_scale_kw']-max(1.,float(observed.std())))<1e-9
        assert model['weights_sha256']==digest(stem.with_suffix('.keras'))
        assert model['archive_sha256']==digest(stem.with_suffix('.npz'))
        monthly_predictions=load(stem.with_suffix('.npz'))
        np.testing.assert_array_equal(monthly_predictions['predictions'],
                                      raw_neural['values'][monthly_predictions['days']-31])
        assert model['signature']==neural_manifest['signature']
        assert model['save_reload_max_error']==0. and not model['uses_future_seasonal_information']
        assert model['future_feature_pollution_check']['passed']
        training_checks.append({'month':month,'training_cutoff_exclusive':cutoff*144,
                                'validation_cutoff_exclusive':issue*144,
                                'model_archive_and_weights_hashes_passed':True,
                                'model_signature':model['signature']})
    np.testing.assert_array_equal(raw_neural['origins'],archives['origins'])
    check.update(prefix_arrays_exact=prefix_checks,scenario_checks_passed=True,
                 prefix_planning_archives_identical=prefix_planning_hashes,
                 prefix_audit_records_identical=True,
                 scenarios_checked=334*3,scenario_day_checks=scenario_rows,
                 complete_historical_joint_residual_paths_verified=334*28,
                 maximum_net_path_error_kwh=maximum_path_error,
                 current_actual_only_greedy_replay_max_error_kwh=maximum_replay_error,
                 calibration_complete_day_cutoffs_checked=334,
                 monthly_training_checks=training_checks,
                 neural_monthly_training_and_mutation_evidence=str(RESULTS/'neural_joint/findings.json'),
                 neural_evidence_sha256=digest(RESULTS/'neural_joint/findings.json'),
                 source_sha256=digest(__file__),
                 nonanticipative_scenario_recourse_certificate=False)
    (CASE/'independent_annual_verification.json').write_text(json.dumps(check,indent=2))
    pd.DataFrame(scenario_rows).to_csv(CASE/'scenario_daily_checks.csv',index=False)
    baseline_parts=[load(path) for path in sorted((ROOT/'data/results/exp006/primary/checkpoints').glob('chunk_*.npz'))]
    baseline={key:np.concatenate([a[key] for a in baseline_parts]) for key in baseline_parts[0]}
    assert baseline['charge'].shape==(334,144)
    assert abs(float(baseline['fees'].sum())-14066257.477256786)<1e-6
    annual=[];monthly=[]
    dates=pd.date_range('2025-02-01','2025-12-31')
    for name,value in [('exp006_primary',baseline),('joint_ridge28_physical3',detail)]:
        annual.append({'name':name,'total_cost':float(value['fees'].sum()),**battery_metrics(value)})
        previous_mode=1;previous_power=172.75999999999976
        for month in range(2,13):
            subset={key:array[dates.month==month] for key,array in value.items()}
            metrics=battery_metrics(subset,initial_mode=previous_mode,initial_power_kw=previous_power)
            monthly.append({'name':name,'month':month,'total_cost':float(subset['fees'].sum()),
                            'planned_cost':float(subset['fees'][:,:,0].sum()),
                            'emergency_cost':float(subset['fees'][:,:,3].sum()),**metrics})
            previous_mode=metrics['final_mode'];previous_power=metrics['final_power_kw']
    pd.DataFrame(annual).to_csv(CASE/'annual_method_comparison.csv',index=False)
    pd.DataFrame(monthly).to_csv(CASE/'monthly_method_comparison.csv',index=False)
    summary={'total_cost':check['recomputed_total_cost'],'billing':check['billing'],
             'battery':check['battery_metrics'],'goal':check['goal'],
             'independent_verification_passed':check['passed'],
             'prefix_30days_identical':all(prefix_checks.values()),
             'scenario_emergency_charge_overlap_slots':0,
             'cost_gap_to_8pct_yuan':check['recomputed_total_cost']-check['goal']['cost_threshold_yuan'],
             'annual_comparison_is_method_comparison_not_pure_network_ablation':True,
             'sum_archived_mip_seconds':sum(row['mip']['seconds'] for row in audit),
             'sum_archived_refinement_seconds':sum(row['greedy_refinement']['planning_seconds'] for row in audit),
             'refinement_success_days':sum(row['greedy_refinement']['success'] for row in audit),
             'refinement_maxiter':120,
             'continuation_wall_seconds_excludes_reused_30_day_prefix':True,
             'neural_candidate_count':1,'no_further_hyperparameter_tuning':True}
    (CASE/'annual_findings.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)
    return summary


if __name__=='__main__':
    main()
