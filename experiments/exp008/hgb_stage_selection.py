"""One monthly HGB iteration policy selected solely on the past validation week.

The previous 100-iteration ceiling censored several still-improving load
fits. This candidate raises it to 400 with unchanged patience and explicitly
uses the best validation stage, rather than the final patience-tail stage.
Both are declared changes; this is not a correction to historical results.
"""
import copy
import itertools
import json
import hashlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments.exp008.forecast_absolute_hgb import (
    OUT as BASE, AbsoluteHGBStore, ArrayStore, features_for_day, fit_month, array_hash)
from experiments.exp008.forecast_calibration import CalibratedStore, _daylight, _calibrate, CONFIG
from experiments.exp008.forecast_shape_diagnostic import summary as score
from experiments.exp008.load_energy_memory import EnergyMemoryStore, correct_day
from experiments.problem2.exp003.data import Data

OUT=Path('data/results/exp008/hgb_stage_selection')


def save(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def best_stage(model):
    # Stage zero is intercept-only; require at least one fitted tree.
    return int(np.argmax(model.validation_score_[1:])+1)


def predict_at_stage(model,x,stage):
    if not 1<=stage<=model.n_iter_:
        raise ValueError('stage must be a fitted boosting iteration')
    with threadpool_limits(limits=1):
        return next(itertools.islice(model.staged_predict(x),stage-1,stage))


def outputs(data,day,models,stages):
    x,_=features_for_day(data,day)
    value=np.column_stack([predict_at_stage(m,x,s) for m,s in zip(models,stages,strict=True)])
    value=np.maximum(value,0.)
    value[:,1]*=_daylight(data.actual,day)
    return value


def main():
    if OUT.exists():
        raise FileExistsError('Historical experiments are immutable')
    OUT.mkdir(parents=True)
    protocol=json.loads((BASE/'protocol.json').read_text())
    protocol['model_configuration']['max_iter']=400
    save(OUT/'protocol.json',{'model_configuration':protocol['model_configuration'],
        'features':protocol['features'],'changes':['max_iter100_to400','best_prior_validation_stage'],
        'patience_unchanged':True,'all_other_training_features_and_postprocessing_unchanged':True,
        'stage_selection':'argmax previous seven validation days score, exclude intercept stage',
        'one_fixed_candidate':True,'development_not_independent_test':True,
        'continuation_gate':'all four net RMSE/high-price/daily-energy/prefix RMSE improve over original complete HGB pipeline',
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    (OUT/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    data=Data();base=AbsoluteHGBStore()
    x=np.stack([features_for_day(data,d)[0] for d in range(7,365)])
    values=np.empty_like(base.values);fitted={};audits=[]
    for month in range(2,13):
        models,audit=fit_month(data,month,x,protocol)
        stages=[best_stage(model) for model in models]
        fitted[month]=(models,stages)
        audit['selected_stages']=stages
        for day in audit['formal_days']:
            values[day-31]=outputs(data,day,models,stages)
        path=OUT/f'm{month:02}.joblib';joblib.dump({'models':models,'stages':stages},path)
        loaded=joblib.load(path); day=audit['formal_days'][0]
        np.testing.assert_array_equal(values[day-31],outputs(data,day,loaded['models'],loaded['stages']))
        audit['model_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
        audits.append(audit)
        print('stage selection month',month,'fitted',[m.n_iter_ for m in models],'selected',stages,flush=True)
    raw=ArrayStore(values,base.origins,'hgb_best_validation_stage_raw')
    ridge=CalibratedStore('ridge_28',data=data,use_cache=False,_base_store=raw,directory=OUT/'calibration')
    memory=EnergyMemoryStore(data=data,base_store=ridge)
    for a in memory.audit:
        a['underlying_forecast']='best_validation_stage_absolute_HGB_then_same_Ridge28'
    for name,store in [('raw',raw),('ridge',ridge),('memory',memory)]:
        np.savez_compressed(OUT/f'{name}.npz',values=store.values,origins=store.origins)
    save(OUT/'training_audit.json',audits);save(OUT/'ridge_audit.json',ridge.audit);save(OUT/'memory_audit.json',memory.audit)
    checks=[]
    for day in (31,32,59,90,151,243,364):
        changed=copy.copy(data);changed.actual=data.actual.copy();changed.actual[day*144:]+=[30000.,70000.]
        month=int((pd.Timestamp('2025-01-01')+pd.Timedelta(days=day)).month)
        models,stages=fitted[month]
        np.testing.assert_array_equal(values[day-31],outputs(changed,day,models,stages))
        i=day-31;future=values.copy();future[i+1:]+=50000.
        a,_=_calibrate(data.actual,base.origins,values,i,CONFIG['ridge_28'])
        b,_=_calibrate(changed.actual,base.origins,future,i,CONFIG['ridge_28'])
        np.testing.assert_array_equal(a,b)
        future=ridge.values.copy();future[i+1:]+=50000.
        a,_=correct_day(data.actual,base.origins,ridge.values,i)
        b,_=correct_day(changed.actual,base.origins,future,i)
        np.testing.assert_array_equal(a,b)
        checks.append({'day':day,'features_predictions_and_postprocessing_future_mutation_unchanged':True})
    truth=data.actual[base.origins[:,None]+np.arange(144)]
    metrics=[]
    for name,v in [('original_HGB',base.values),('best_stage_HGB',memory.values)]:
        for channel,e in [('load',v[:,:,0]-truth[:,:,0]),('pv',v[:,:,1]-truth[:,:,1]),
                         ('net',(v[:,:,0]-v[:,:,1])-(truth[:,:,0]-truth[:,:,1]))]:
            metrics.append({'name':name,'channel':channel,**score(e,data.fixed_price)})
    pd.DataFrame(metrics).to_csv(OUT/'metrics.csv',index=False)
    old,new=[r for r in metrics if r['channel']=='net']
    gate=all(new[k]<old[k] for k in ['rmse_kw','high_price_rmse_kw','daily_energy_rmse_kwh','cumulative_error_rmse_kwh'])
    save(OUT/'summary.json',{'complete':True,'metrics':metrics,'continuation_gate_passed':gate,
        'future_mutation_checks':checks,'base_values_sha256':array_hash(base.values),
        'all_monthly_first_day_reload_exact':True,'LP_bridge_performed':False})
    print(json.dumps({'net_metrics':[old,new],'gate':gate},indent=2),flush=True)


if __name__=='__main__':
    main()
