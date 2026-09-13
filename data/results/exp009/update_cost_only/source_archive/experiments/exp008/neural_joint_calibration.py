"""Fair fixed Ridge28 postprocessing for the original and joint CNN archives."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.exp008.forecast_calibration import CONFIG, CalibratedStore, _calibrate
from experiments.problem2.exp003.data import ROOT, Data
from experiments.problem2.exp004.predict import ForecastStore

OUT=ROOT/'data/results/exp008/neural_joint_calibration'
JOINT=ROOT/'data/results/exp008/neural_joint'


class JointStore:
    def __init__(self,calibrated=False):
        directory=OUT if calibrated else JOINT
        filename='joint_ridge28.npz' if calibrated else 'predictions.npz'
        meta=json.loads((directory/'manifest.json').read_text())
        if not meta['complete']:
            raise ValueError('All eleven monthly joint outputs must be complete')
        path=directory/filename
        expected=meta['joint_calibrated_archive_sha256'] if calibrated else meta['archive_sha256']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise ValueError('Joint forecast archive hash mismatch')
        with np.load(path) as archive:
            self.origins=archive['origins'].copy();self.values=archive['values'].copy()
        self.name='joint_ridge28' if calibrated else 'joint_raw'
        self.lookup={int(origin):i for i,origin in enumerate(self.origins)}

    def get(self,origin):
        return self.values[self.lookup[int(origin)]].copy()


def score(values,truth,price):
    error=values-truth;net=error[:,:,0]-error[:,:,1]
    high=price>=np.quantile(price,.75)
    energy=net.sum(axis=1)/6;prefix=np.cumsum(net/6,axis=1)
    return {'net_rmse_kw':float(np.sqrt(np.mean(net**2))),
            'price_weighted_net_rmse_kw':float(np.sqrt(np.mean(price*net**2)/price.mean())),
            'load_rmse_kw':float(np.sqrt(np.mean(error[:,:,0]**2))),
            'pv_rmse_kw':float(np.sqrt(np.mean(error[:,:,1]**2))),
            'net_bias_kw':float(net.mean()),'high_price_threshold':float(np.quantile(price,.75)),
            'high_price_net_rmse_kw':float(np.sqrt(np.mean(net[:,high]**2))),
            'high_price_net_mae_kw':float(np.abs(net[:,high]).mean()),
            'high_price_net_bias_kw':float(net[:,high].mean()),
            'daily_net_energy_rmse_kwh':float(np.sqrt(np.mean(energy**2))),
            'daily_net_energy_mae_kwh':float(np.abs(energy).mean()),
            'intraday_cumulative_error_rmse_kwh':float(np.sqrt(np.mean(prefix**2))),
            'cumulative_net_energy_error_kwh':float(energy.sum())}


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    cfg={'network_candidate_count':1,'calibrator':'unmodified_exp008_Ridge28',
         'calibrator_config':CONFIG['ridge_28'],'data_explicit':True,'use_cache':False,
         'base_store_injection':True,'no_existing_forecast_or_calibration_cache_overwritten':True,
         'same_postprocessing_for_both_architectures':True,
         'hyperparameters_changed':False,'development_on_2025_not_independent_validation':True,
         'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         'calibrator_source_sha256':hashlib.sha256((ROOT/'experiments/exp008/forecast_calibration.py').read_bytes()).hexdigest()}
    (OUT/'protocol.json').write_text(json.dumps(cfg,indent=2))
    data=Data();original=ForecastStore('no_season',seed=42);joint=JointStore()
    np.testing.assert_array_equal(original.origins,joint.origins)
    calibrated={}
    for name,store in [('original',original),('joint',joint)]:
        result=CalibratedStore('ridge_28',seed=42,data=data,directory=OUT/name,use_cache=False,_base_store=store)
        calibrated[name]=result
        path=OUT/f'{name}_ridge28.npz'
        np.savez_compressed(path,origins=result.origins,values=result.values,delta=result.delta)
        (OUT/f'{name}_ridge28_audit.json').write_text(json.dumps({
            'config':CONFIG['ridge_28'],'days':result.audit,
            'base_forecast_sha256':hashlib.sha256(store.values.tobytes()).hexdigest(),
            'archive_sha256':hashlib.sha256(path.read_bytes()).hexdigest()},indent=2))
    cached_original=CalibratedStore('ridge_28').values
    original_parity=float(np.max(np.abs(cached_original-calibrated['original'].values)))
    assert original_parity==0.
    pollution=[]
    for day in (31,62,243,364):
        index=joint.lookup[day*144]
        values,_=_calibrate(data.actual,joint.origins,joint.values,index,CONFIG['ridge_28'])
        changed=data.actual.copy();changed[day*144:]+=70000.
        mutated,_=_calibrate(changed,joint.origins,joint.values,index,CONFIG['ridge_28'])
        error=float(np.max(np.abs(values-mutated)))
        assert error==0.
        pollution.append({'day':day,'current_and_future_actual_mutation_output_error_kw':error})
    truth=data.actual[joint.origins[:,None]+np.arange(144)]
    dates=pd.date_range('2025-02-01','2025-12-31')
    annual=[];monthly=[];daily=[]
    for name,values in [('original_raw',original.values),('joint_raw',joint.values),
                        ('original_ridge28',calibrated['original'].values),('joint_ridge28',calibrated['joint'].values)]:
        annual.append({'name':name,**score(values,truth,data.fixed_price)})
        for month in range(2,13):
            ids=dates.month==month
            monthly.append({'name':name,'month':month,**score(values[ids],truth[ids],data.fixed_price)})
        for i,date in enumerate(dates):
            daily.append({'name':name,'date':str(date.date()),**score(values[i:i+1],truth[i:i+1],data.fixed_price)})
    pd.DataFrame(annual).to_csv(OUT/'annual_metrics.csv',index=False)
    pd.DataFrame(monthly).to_csv(OUT/'monthly_metrics.csv',index=False)
    pd.DataFrame(daily).to_csv(OUT/'daily_metrics.csv',index=False)
    reference,candidate=annual[2],annual[3]
    manifest={'complete':True,'days':334,'metrics':annual,
        'original_recomputed_vs_cached_max_error_kw':original_parity,
        'future_actual_pollution_checks':pollution,
        'all_calibration_labels_are_prior_complete_days':all(
            a['history_last_label'] is None or a['history_last_label']<a['origin']
            for store in calibrated.values() for a in store.audit),
        'joint_calibrated_archive_sha256':hashlib.sha256((OUT/'joint_ridge28.npz').read_bytes()).hexdigest(),
        'joint_calibrated_net_rmse_below_original_calibrated':candidate['net_rmse_kw']<reference['net_rmse_kw'],
        'joint_calibrated_weighted_rmse_below_original_calibrated':candidate['price_weighted_net_rmse_kw']<reference['price_weighted_net_rmse_kw'],
        'joint_calibrated_high_price_rmse_below_original_calibrated':candidate['high_price_net_rmse_kw']<reference['high_price_net_rmse_kw']}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest,indent=2),flush=True)
    return manifest


if __name__=='__main__':
    run()
