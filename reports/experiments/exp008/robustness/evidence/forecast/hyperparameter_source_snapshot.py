"""Small genuine tree-parameter perturbation supplement to forecast_analysis."""
import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor
from threadpoolctl import threadpool_limits

from experiments.exp008.robustness.forecast_analysis import (
    OUT, OLD, ROOT, BASE, FOLDS, Data, AbsoluteHGBStore, AbsoluteExtraTreesStore,
    MODEL_CONFIG, _daylight, digest, array_hash, features_for_day, historical_labels,
    age_weights, postprocess, metric_rows, verify_metrics, date, write_json,
)


SETTINGS = [
    dict(name='hgb_learning_rate_0.064',family='hgb',parameter='learning_rate',value=.064,default=.08),
    dict(name='hgb_learning_rate_0.096',family='hgb',parameter='learning_rate',value=.096,default=.08),
    dict(name='extra_trees_min_samples_leaf_8',family='extra_trees',parameter='min_samples_leaf',value=8,default=10),
    dict(name='extra_trees_min_samples_leaf_12',family='extra_trees',parameter='min_samples_leaf',value=12,default=10),
]


def run():
    if (OUT/'hyperparameter_protocol.json').exists():
        raise FileExistsError('Supplement already exists')
    assert json.loads((OUT/'summary.json').read_text())['complete']
    began=perf_counter()
    data=Data()
    hgb=AbsoluteHGBStore('direct_hgb_raw');et=AbsoluteExtraTreesStore('absolute_extra_trees_raw')
    raw0=.5*hgb.values+.5*et.values
    with np.load(OLD/'forecast_hgb_extra_trees_half/hgb_extra_trees_half_ridge28.npz') as z:
        ridge0=z['values'].copy()
    hgbcfg=json.loads((OLD/'forecast_absolute_hgb/protocol.json').read_text())['model_configuration']
    features=np.stack([features_for_day(data,d)[0] for d in range(7,365)])
    protocol=dict(settings=SETTINGS,fold_months=FOLDS,validation_days=7,test_days=28,
        fits=12,models_fitted_per_setting_fold=2,
        rule='Only changed family is genuinely refitted for each setting/fold; unchanged family uses the signed original monthly issued predictions. Both baseline families were separately genuinely refitted and reproduced by forecast_analysis.',
        fixed='Original31features,seed42,expandingtrain,agehalfLife90,rawblend0.5,Ridge28andgain0.5',
        warm_start='Shared causal frozen issued prefix before each fold; new predictions from fold start',
        role='2025development_sensitivity_not_independent_test; no final model replacement or score-based selection',
        source_sha256=digest(__file__),data_sha256=data.hashes,
        input_sha256={str(p.relative_to(ROOT)):digest(p) for p in [OUT/'protocol.json',OUT/'summary.json',OUT/'cv_metrics.csv',
            ROOT/'experiments/exp008/robustness/forecast_analysis.py',
            OLD/'forecast_absolute_hgb/direct_hgb_raw.npz',OLD/'forecast_absolute_extra_trees/absolute_extra_trees_raw.npz',
            OLD/'forecast_hgb_extra_trees_half/hgb_extra_trees_half_ridge28.npz']})
    write_json('hyperparameter_protocol.json',protocol)
    annual,daily,audits=[],[],[]
    cv=pd.read_csv(OUT/'cv_metrics.csv')
    for _,r in cv[(cv.validation_days==7)&(cv.test_days==28)].iterrows():
        annual.append(dict(name='baseline',family='baseline',parameter='baseline',value=0.,default=0.,**r.to_dict()))
    for month in FOLDS:
        asof=int((pd.Timestamp(2025,month,1)-pd.Timestamp('2025-01-01')).days)
        cutoff=asof-7;train=np.arange(7,cutoff);validation=np.arange(cutoff,asof)
        test=np.arange(asof,asof+28);start=asof-31
        x=features[train-7].reshape(-1,31);xval=features[validation-7].reshape(-1,31)
        y=historical_labels(data,train,cutoff*144).reshape(-1,2)
        yval=historical_labels(data,validation,asof*144).reshape(-1,2)
        xp=features[test-7].reshape(-1,31)
        weights=np.repeat(age_weights(train,cutoff,90),144)
        for setting in SETTINGS:
            t=perf_counter();config=dict(hgbcfg if setting['family']=='hgb' else MODEL_CONFIG)
            config[setting['parameter']]=setting['value']
            ctor=HistGradientBoostingRegressor if setting['family']=='hgb' else ExtraTreesRegressor
            predictions=[];models=[]
            for c,channel in enumerate(['load','pv']):
                model=ctor(**config)
                with threadpool_limits(limits=1):
                    if setting['family']=='hgb':model.fit(x,y[:,c],sample_weight=weights,X_val=xval,y_val=yval[:,c])
                    else:model.fit(x,y[:,c],sample_weight=weights)
                    pred=model.predict(xp)
                    valpred=model.predict(xval)
                path=OUT/'models'/f'hyper_m{month:02}_{setting["name"]}_{channel}.joblib'
                joblib.dump(model,path,compress=3)
                predictions.append(pred)
                models.append(dict(channel=channel,model_sha256=digest(path),model_file=str(path.relative_to(ROOT)),
                    iterations=int(model.n_iter_) if setting['family']=='hgb' else None,
                    validation_rmse_kw=float(np.sqrt(np.mean((valpred-yval[:,c])**2)))))
                del model
            new=np.maximum(np.column_stack(predictions).reshape(28,144,2),0.)
            for j,d in enumerate(test):new[j,:,1]*=_daylight(data.actual,int(d))
            other=(et.values if setting['family']=='hgb' else hgb.values)[start:start+28]
            raw=raw0[:start+28].copy();raw[start:]=.5*new+.5*other
            final,_,_=postprocess(data.actual,hgb.origins[:len(raw)],raw,BASE,ridge_initial=ridge0[:len(raw)],start=start)
            final=final[start:]
            truth=data.actual[test[:,None]*144+np.arange(144),:2]
            scores=list(metric_rows(final,truth,data.fixed_price));verify_metrics(final,truth,scores)
            for r in scores:annual.append(dict(**setting,fold_month=month,validation_days=7,test_days=28,
                train_start=date(train[0]),train_end=date(train[-1]),validation_start=date(validation[0]),
                validation_end=date(validation[-1]),test_start=date(asof),test_end=date(asof+27),**r))
            for j,d in enumerate(test):
                for r in metric_rows(final[j:j+1],truth[j:j+1],data.fixed_price):
                    daily.append(dict(name=setting['name'],fold_month=month,date=date(d),day=int(d),**r))
            archive=OUT/f'hyper_m{month:02}_{setting["name"]}.npz'
            np.savez_compressed(archive,origins=test*144,values=final)
            audits.append(dict(**setting,fold_month=month,genuine_refit=True,models=models,
                training_last_label_exclusive=cutoff*144,validation_last_label_exclusive=asof*144,first_test_issue=asof*144,
                xtrain_sha256=array_hash(x),ytrain_sha256=array_hash(y),xval_sha256=array_hash(xval),yval_sha256=array_hash(yval),
                training_and_validation_inputs_same_as_audited_causal_baseline=True,
                forecast_sha256=digest(archive),seconds=perf_counter()-t))
            print(json.dumps(dict(stage='hyperparameters',month=month,name=setting['name'],seconds=perf_counter()-t,net_rmse_kw=scores[-1]['rmse_kw'])),flush=True)
    baseline={(r['fold_month'],r['channel']):r for r in annual if r['name']=='baseline'}
    for r in annual:
        for field in ['rmse_kw','mae_kw','daily_energy_rmse_kwh','cumulative_error_rmse_kwh']:
            r[field+'_change_pct']=100*(r[field]/baseline[r['fold_month'],r['channel']][field]-1)
    pd.DataFrame(annual).to_csv(OUT/'hyperparameter_metrics.csv',index=False)
    pd.DataFrame(daily).to_csv(OUT/'hyperparameter_daily.csv',index=False)
    write_json('hyperparameter_training_audit.json',audits)
    assert digest(__file__)==protocol['source_sha256']
    assert all(digest(ROOT/p)==h for p,h in protocol['input_sha256'].items())
    write_json('hyperparameter_summary.json',dict(complete=True,seconds=perf_counter()-began,
        settings=len(SETTINGS),folds=len(FOLDS),genuine_refits=len(audits),saved_models=sum(len(a['models']) for a in audits),
        all_inputs_unchanged=True,development_not_independent_test=True,
        net_rmse_relative_changes_pct=[dict(name=r['name'],fold_month=r['fold_month'],change_pct=r['rmse_kw_change_pct']) for r in annual if r['channel']=='net']))
    (OUT/'hyperparameter_source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    write_json('hyperparameter_output_manifest.json',dict(files_sha256={str(p.relative_to(OUT)):digest(p) for p in OUT.rglob('*') if p.is_file() and (p.name.startswith('hyper_') or p.name.startswith('hyperparameter_')) and p.name!='hyperparameter_output_manifest.json'}))


if __name__=='__main__':run()
