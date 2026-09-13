"""One causal PV-scale normalization candidate; fixed original load HGB.

The PV model sees power relative to the 95th percentile of strictly past
positive PV over 28 days. This is an empirical scale, not known rated power
or a measured clear-sky envelope. No annual maximum or future weather enters.
"""
import copy
import hashlib
import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from experiments.exp008.forecast_absolute_hgb import (
    AbsoluteHGBStore, ArrayStore, OUT as BASE, PRIMARY, array_hash,
    features_for_day as absolute_features)
from experiments.exp008.forecast_calibration import CalibratedStore, _daylight, _calibrate, CONFIG
from experiments.exp008.forecast_shape_diagnostic import summary as score
from experiments.exp008.load_energy_memory import EnergyMemoryStore, correct_day
from experiments.problem2.exp003.data import Data
from experiments.problem2.exp004.data import split_days, age_weights

OUT=Path('data/results/exp008/forecast_scaled_pv_hgb')


def save(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def features(data,day):
    x,names=absolute_features(data,day)
    past=data.actual[max(0,day-28)*144:day*144,1]
    positive=past[past>0]
    scale=max(100.,float(np.quantile(positive,.95))) if len(positive) else 100.
    ratio=1000./scale
    x=x.copy()
    byname={n:i for i,n in enumerate(names)}
    for i,name in enumerate(names):
        if name.endswith('_pv'):
            x[:,i]*=ratio
    h=data.actual[(day-7)*144:day*144,:2].reshape(7,144,2)
    scaled_net=h[:,:,0]-ratio*h[:,:,1]
    for i,name in enumerate(names):
        if name.endswith('_net'):
            if name=='std7_net':
                x[:,i]=scaled_net.std(axis=0)
            elif name=='yesterday_std_net':
                x[:,i]=scaled_net[-1].std()
            else:
                prefix=name[:-4]
                x[:,i]=x[:,byname[prefix+'_load']]-x[:,byname[prefix+'_pv']]
    return x,scale,names


def fit(data,month,all_x,scales,configuration):
    train,val,formal,cutoff=split_days(month)
    labels=data.actual.reshape(365,144,2)[:,:,1]
    ytrain=(labels[train]/scales[train-7,None]*1000.).ravel()
    yval=(labels[val]/scales[val-7,None]*1000.).ravel()
    xt=all_x[train-7].reshape(-1,all_x.shape[-1])
    xv=all_x[val-7].reshape(-1,all_x.shape[-1])
    model=HistGradientBoostingRegressor(**configuration)
    began=perf_counter()
    with threadpool_limits(limits=1):
        model.fit(xt,ytrain,sample_weight=np.repeat(age_weights(train,cutoff,90),144),
                  X_val=xv,y_val=yval)
    audit={'month':month,'training_days':train.tolist(),'validation_days':val.tolist(),
           'formal_days':formal.tolist(),'asof_day':int(formal[0]),
           'maximum_label_exclusive':int(formal[0]*144),'training_seconds':perf_counter()-began,
           'iterations':int(model.n_iter_),'training_features_sha256':array_hash(xt),
           'validation_features_sha256':array_hash(xv),'training_targets_sha256':array_hash(ytrain),
           'validation_targets_sha256':array_hash(yval)}
    return model,audit


def main():
    if OUT.exists():
        raise FileExistsError('Existing development results are immutable')
    OUT.mkdir(parents=True)
    config=json.loads((BASE/'protocol.json').read_text())['model_configuration']
    save(OUT/'protocol.json',{'one_fixed_candidate':True,'model_configuration':config,
        'PV_target_and_PV_historical_feature_scale':'1000 / max(100,95th quantile of positive PV from past <=28 completed days)',
        'mixed_net_features':'recomputed load minus rescaled PV; all 31 feature names retained as analogous features',
        'raw_load_model':'unchanged original absolute HGB archive',
        'postprocessing':'same fixed Ridge28 and nonrecursive .5 load memory, both refitted to new raw outputs',
        'monthly_training_and_last7_validation':True,'historical_weight_half_life_days':90,
        'no_annual_capacity_or_future_weather_used':True,'development_not_independent_test':True,
        'continuation_gate':'net RMSE, high-price net RMSE, daily-energy RMSE and prefix-energy RMSE all improve versus original HGB full pipeline before any LP bridge',
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    (OUT/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    data=Data(); raw_base=AbsoluteHGBStore('direct_hgb_raw')
    inputs=[features(data,day) for day in range(7,365)]
    all_x=np.stack([r[0] for r in inputs]); scales=np.array([r[1] for r in inputs])
    values=raw_base.values.copy(); models={}; audits=[]
    for month in range(2,13):
        model,audit=fit(data,month,all_x,scales,config)
        models[month]=model
        days=np.asarray(audit['formal_days'])
        with threadpool_limits(limits=1):
            predictions=model.predict(all_x[days-7].reshape(-1,all_x.shape[-1])).reshape(-1,144)
        for i,day in enumerate(days):
            values[day-31,:,1]=np.maximum(0.,predictions[i]*scales[day-7]/1000.)*_daylight(data.actual,int(day))
        path=OUT/f'pv_m{month:02}.joblib'; joblib.dump(model,path)
        with threadpool_limits(limits=1):
            reloaded=joblib.load(path).predict(all_x[days-7].reshape(-1,all_x.shape[-1])).reshape(-1,144)
        np.testing.assert_array_equal(reloaded,predictions)
        audit['model_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
        audits.append(audit)
        print('scaled PV month',month,'iterations',audit['iterations'],flush=True)
    np.testing.assert_array_equal(values[:,:,0],raw_base.values[:,:,0])
    raw=ArrayStore(values,raw_base.origins,'scaled_pv_hgb_raw')
    ridge=CalibratedStore('ridge_28',data=data,use_cache=False,_base_store=raw,
                         directory=OUT/'calibration')
    memory=EnergyMemoryStore(data=data,base_store=ridge)
    for a in memory.audit:
        a['underlying_forecast']='scaled_PV_absolute_HGB_with_unchanged_raw_load_then_Ridge28'
    stages={'raw':raw,'ridge':ridge,'memory':memory}
    for name,store in stages.items():
        np.savez_compressed(OUT/f'{name}.npz',values=store.values,origins=store.origins)
    np.savez_compressed(OUT/'features.npz',values=all_x,scales=scales,days=np.arange(7,365))
    save(OUT/'training_audit.json',audits);save(OUT/'ridge_audit.json',ridge.audit)
    save(OUT/'memory_audit.json',memory.audit)
    verification=[]
    for day in (31,32,59,90,151,243,364):
        changed=copy.copy(data); changed.actual=data.actual.copy()
        changed.actual[day*144:]+=np.array([50000.,90000.])
        x,s,_=features(changed,day)
        np.testing.assert_array_equal(x,all_x[day-7]); assert s==scales[day-7]
        month=int((pd.Timestamp('2025-01-01')+pd.Timedelta(days=day)).month)
        with threadpool_limits(limits=1):
            pv=np.maximum(0.,models[month].predict(x)*s/1000.)*_daylight(changed.actual,day)
        np.testing.assert_array_equal(pv,values[day-31,:,1])
        i=day-31
        future=values.copy();future[i+1:]+=80000.
        a,_=_calibrate(data.actual,raw.origins,values,i,CONFIG['ridge_28'])
        b,_=_calibrate(changed.actual,raw.origins,future,i,CONFIG['ridge_28'])
        np.testing.assert_array_equal(a,b)
        f=ridge.values.copy();f[i+1:]+=70000.
        a,_=correct_day(data.actual,raw.origins,ridge.values,i)
        b,_=correct_day(changed.actual,raw.origins,f,i)
        np.testing.assert_array_equal(a,b)
        verification.append({'day':day,'features_scale_prediction_and_postprocessing_unchanged':True})
    truth=data.actual[raw.origins[:,None]+np.arange(144)]
    records=[]
    for name,v in [('original_HGB',AbsoluteHGBStore().values),('scaled_PV_HGB',memory.values)]:
        for channel,e in [('load',v[:,:,0]-truth[:,:,0]),('pv',v[:,:,1]-truth[:,:,1]),
                          ('net',(v[:,:,0]-v[:,:,1])-(truth[:,:,0]-truth[:,:,1]))]:
            records.append({'name':name,'channel':channel,**score(e,data.fixed_price)})
    pd.DataFrame(records).to_csv(OUT/'metrics.csv',index=False)
    old,new=[r for r in records if r['channel']=='net']
    keys=['rmse_kw','high_price_rmse_kw','daily_energy_rmse_kwh','cumulative_error_rmse_kwh']
    gate=all(new[k]<old[k] for k in keys)
    save(OUT/'summary.json',{'complete':True,'metrics':records,'continuation_gate_passed':gate,
        'future_mutation_checks':verification,'all11_model_reload_exact':True,
        'raw_load_unchanged':True,'base_raw_values_sha256':array_hash(raw_base.values),
        'lp_bridge_performed':False})
    print(json.dumps({'net_metrics':[old,new],'gate':gate},indent=2),flush=True)


if __name__=='__main__':
    main()
