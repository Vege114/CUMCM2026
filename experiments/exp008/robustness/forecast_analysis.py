"""Reproducible Q2 local sensitivity and genuinely refitted temporal CV.

Run from repository root with .venv/bin/python -m
experiments.exp008.robustness.forecast_analysis. Frozen evidence is read-only.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from experiments.exp008.forecast_absolute_hgb import (
    AbsoluteHGBStore, array_hash, digest, features_for_day, historical_labels,
)
from experiments.exp008.forecast_absolute_extra_trees import AbsoluteExtraTreesStore, MODEL_CONFIG
from experiments.exp008.forecast_calibration import _calibrate, _features, _daylight
from experiments.exp008.forecast_shape_diagnostic import summary as error_summary
from experiments.exp008.load_energy_memory import correct_day
from experiments.problem2.exp003.data import Data, ROOT
from experiments.problem2.exp004.data import age_weights

OUT = ROOT / 'data/results/exp008/robustness/forecast'
OLD = ROOT / 'data/results/exp008'
BLEND = OLD / 'forecast_hgb_extra_trees_half'
EPOCH = pd.Timestamp('2025-01-01')
BASE = dict(weight=.5, window=28, penalty_multiplier=1., gain=.5)
GRIDS = dict(weight=[.3,.4,.5,.6,.7], window=[14,21,28,35,42],
             penalty_multiplier=[.5,.75,1.,1.25,1.5,2.], gain=[.25,.4,.5,.6,.75])
FOLDS = [4,7,10]
VALIDATIONS = [7,14,21]
HORIZONS = [7,14,28]
INPUTS = {}


def write_json(name, payload):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def register(path):
    INPUTS[str(Path(path).relative_to(ROOT))] = digest(path)
    return Path(path)


def date(day):
    return str((EPOCH + pd.Timedelta(days=int(day))).date())


def candidates():
    result = [dict(name='baseline', factor='baseline', factor_value=0., **BASE)]
    for factor, values in GRIDS.items():
        for value in values:
            if value == BASE[factor]:
                continue
            config = dict(BASE)
            config[factor] = value
            result.append(dict(name=f'{factor}_{value:g}', factor=factor, factor_value=value, **config))
    return result


def ridge_day(actual, origins, raw, index, window=28, penalty_multiplier=1.):
    """Same arithmetic as frozen Ridge; allow only declared penalty scaling."""
    if penalty_multiplier == 1.:
        return _calibrate(actual, origins, raw, index, dict(method='ridge', window=window))
    origin = int(origins[index])
    day = origin // 144
    hist = np.arange(max(0,index-window), index, dtype=int)
    audit = dict(origin=origin, history_last_label=int(origins[hist[-1]]+143) if len(hist) else None,
                 window_days=window, history_count=len(hist), current_truth_used=False)
    if not len(hist):
        return raw[index].copy(), audit
    assert np.all(origins[hist]+144 <= origin)
    observed = np.stack([actual[int(origins[h]):int(origins[h])+144,:2] for h in hist])
    residual = observed - raw[hist]
    weights = 2 ** (-np.arange(len(hist)-1,-1,-1) / max(1,window/2))
    delta = np.zeros((144,2))
    for c in range(2):
        x = np.concatenate([_features(actual,int(origins[h]//144),raw[h],c) for h in hist])
        target = residual[:,:,c].ravel()/1000
        w = np.repeat(weights,144)
        if c == 1:
            active = ((raw[hist,:,1]>1)|(observed[:,:,1]>1)).ravel()
            w *= np.where(active,1.,.1)
        penalty = np.full(x.shape[1],20.)
        penalty[0] = 2.
        penalty *= penalty_multiplier
        gram = np.einsum('ni,nj,n->ij',x,x,w,optimize=False)
        rhs = np.einsum('ni,n,n->i',x,w,target,optimize=False)
        coef = np.linalg.solve(gram+np.diag(penalty),rhs)
        xt = _features(actual,day,raw[index],c)
        correction = 1000*np.einsum('ni,i->n',xt,coef,optimize=False)
        bound = max(100.,3*float(np.sqrt(np.mean(residual[:,:,c]**2))))
        delta[:,c] = np.clip(correction,-bound,bound)
    output = np.maximum(0.,raw[index]+delta)
    output[:,1] *= _daylight(actual,day)
    return output,audit


def memory_day(actual, origins, ridge, index, gain=.5):
    if gain == .5:
        return correct_day(actual,origins,ridge,index)
    origin = int(origins[index])
    prior = np.flatnonzero(origins+144 == origin)
    output = ridge[index].copy()
    audit = dict(origin=origin, prior_label_end_exclusive=None, gain=gain)
    if len(prior):
        j = int(prior[-1]); start = int(origins[j])
        assert j < index and start+144 <= origin
        residual = float(np.mean(actual[start:start+144,0]-ridge[j,:,0]))
        output[:,0] = np.maximum(0.,output[:,0]+gain*residual)
        audit['prior_label_end_exclusive'] = start+144
    return output,audit


def postprocess(actual, origins, raw, config, ridge_initial=None, start=0):
    ridge = np.empty_like(raw) if ridge_initial is None else ridge_initial.copy()
    audits = []
    for i in range(start,len(origins)):
        ridge[i],audit = ridge_day(actual,origins,raw,i,config['window'],config['penalty_multiplier'])
        assert audit['history_last_label'] is None or audit['history_last_label'] < origins[i]
        audits.append(audit)
    values = np.empty_like(raw)
    for i in range(len(origins)):
        values[i],audit = memory_day(actual,origins,ridge,i,config['gain'])
        assert audit['prior_label_end_exclusive'] is None or audit['prior_label_end_exclusive'] <= origins[i]
    return values,ridge,audits


def metric_rows(values, truth, price):
    for c,channel in enumerate(('load','pv','net')):
        p = values[:,:,c] if c < 2 else values[:,:,0]-values[:,:,1]
        y = truth[:,:,c] if c < 2 else truth[:,:,0]-truth[:,:,1]
        e = p-y
        yield dict(channel=channel,n=e.size,days=len(e),
                   wape_pct=float(100*np.abs(e).sum()/np.abs(y).sum()),
                   squared_error_sum_kw2=float(np.square(e).sum()),
                   absolute_error_sum_kw=float(np.abs(e).sum()),
                   actual_absolute_sum_kw=float(np.abs(y).sum()),
                   **error_summary(e,price))


def verify_metrics(values, truth, rows):
    """Independent Python fsum oracle, separate from vectorized score helper."""
    for c,r in enumerate(rows):
        p = values[:,:,c] if c<2 else values[:,:,0]-values[:,:,1]
        y = truth[:,:,c] if c<2 else truth[:,:,0]-truth[:,:,1]
        e = (p-y).ravel().tolist()
        np.testing.assert_allclose(r['rmse_kw'],math.sqrt(math.fsum(v*v for v in e)/len(e)),rtol=1e-14)
        np.testing.assert_allclose(r['mae_kw'],math.fsum(abs(v) for v in e)/len(e),rtol=1e-14)


def sensitivity(data,hgb,extra,origins,truth):
    annual,daily,monthly,checks = [],[],[],[]
    cache = {}
    dates = pd.date_range('2025-02-01','2025-12-31')
    with np.load(register(BLEND/'hgb_extra_trees_half_ridge28_memory.npz')) as z:
        saved = z['values'].copy()
    with np.load(register(BLEND/'hgb_extra_trees_half_ridge28.npz')) as z:
        saved_ridge = z['values'].copy()
    for cfg in candidates():
        began = perf_counter()
        raw = cfg['weight']*hgb+(1-cfg['weight'])*extra
        key = (cfg['weight'],cfg['window'],cfg['penalty_multiplier'])
        if key in cache:
            ridge = cache[key]
            values = np.stack([memory_day(data.actual,origins,ridge,i,cfg['gain'])[0] for i in range(len(origins))])
        else:
            values,ridge,_ = postprocess(data.actual,origins,raw,cfg)
            cache[key] = ridge
        if cfg['name'] == 'baseline':
            np.testing.assert_array_equal(values,saved)
            np.testing.assert_array_equal(ridge,saved_ridge)
        scores = list(metric_rows(values,truth,data.fixed_price))
        verify_metrics(values,truth,scores)
        seconds = perf_counter()-began
        for r in scores:
            annual.append(dict(**cfg,seconds=seconds,**r))
        for i,d in enumerate(dates):
            for r in metric_rows(values[i:i+1],truth[i:i+1],data.fixed_price):
                daily.append(dict(name=cfg['name'],date=str(d.date()),day=i+31,**r))
        for m in range(2,13):
            mask = dates.month == m
            for r in metric_rows(values[mask],truth[mask],data.fixed_price):
                monthly.append(dict(name=cfg['name'],month=m,**r))
        # Mutation check for every setting at seven declared boundary/interior days.
        for day in [31,32,59,90,151,243,364]:
            i = day-31
            actual_mutated = data.actual.copy(); actual_mutated[day*144:] += [70000.,30000.]
            raw_mutated = raw.copy(); raw_mutated[i+1:] += [50000.,90000.]
            a,_ = ridge_day(data.actual,origins,raw,i,cfg['window'],cfg['penalty_multiplier'])
            b,_ = ridge_day(actual_mutated,origins,raw_mutated,i,cfg['window'],cfg['penalty_multiplier'])
            np.testing.assert_array_equal(a,b)
            ridge_mutated = ridge.copy();ridge_mutated[i+1:] += [60000.,20000.]
            a,_ = memory_day(data.actual,origins,ridge,i,cfg['gain'])
            b,_ = memory_day(actual_mutated,origins,ridge_mutated,i,cfg['gain'])
            np.testing.assert_array_equal(a,b)
            checks.append(dict(name=cfg['name'],day=day,date=date(day),future_mutation_max_difference_kw=0.))
        np.savez_compressed(OUT/(cfg['name']+'.npz'),origins=origins,values=values)
        print(json.dumps(dict(stage='sensitivity',name=cfg['name'],seconds=seconds,net_rmse_kw=scores[-1]['rmse_kw'])),flush=True)
    baseline = {r['channel']:r for r in annual if r['name']=='baseline'}
    for r in annual:
        for metric in ['rmse_kw','mae_kw','daily_energy_rmse_kwh','cumulative_error_rmse_kwh']:
            r[metric+'_change_pct'] = 100*(r[metric]/baseline[r['channel']][metric]-1)
    for name,rows in [('sensitivity_annual',annual),('sensitivity_daily',daily),('sensitivity_monthly',monthly),('sensitivity_causality_checks',checks)]:
        pd.DataFrame(rows).to_csv(OUT/(name+'.csv'),index=False)
    return annual,saved_ridge


def temporal_cv(data,hgb,extra,origins,saved_ridge):
    features = np.stack([features_for_day(data,d)[0] for d in range(7,365)])
    original_features_hash = json.loads(register(OLD/'forecast_absolute_hgb/provenance.json').read_text())['feature_tensor_sha256']
    assert array_hash(features) == original_features_hash
    hgb_cfg = json.loads(register(OLD/'forecast_absolute_hgb/protocol.json').read_text())['model_configuration']
    old = .5*hgb+.5*extra
    all_scores,all_daily,audits = [],[],[]
    pilot_seconds = None
    for month in FOLDS:
        asof = int((pd.Timestamp(2025,month,1)-EPOCH).days)
        test = np.arange(asof,asof+28)
        for val_days in VALIDATIONS:
            began = perf_counter()
            cutoff = asof-val_days
            train = np.arange(7,cutoff)
            validation = np.arange(cutoff,asof)
            assert train[-1] < validation[0] <= validation[-1] < test[0]
            xtrain = features[train-7].reshape(-1,31)
            xval = features[validation-7].reshape(-1,31)
            ytrain = historical_labels(data,train,cutoff*144).reshape(-1,2)
            yval = historical_labels(data,validation,asof*144).reshape(-1,2)
            weights = np.repeat(age_weights(train,cutoff,90),144)
            xforecast = features[test-7].reshape(-1,31)
            predictions,model_rows = {},[]
            for family,constructor,config in [('hgb',HistGradientBoostingRegressor,hgb_cfg),('extra_trees',ExtraTreesRegressor,MODEL_CONFIG)]:
                channel_predictions = []
                for c,channel in enumerate(('load','pv')):
                    model = constructor(**config)
                    tick = perf_counter()
                    with threadpool_limits(limits=1):
                        if family == 'hgb':
                            model.fit(xtrain,ytrain[:,c],sample_weight=weights,X_val=xval,y_val=yval[:,c])
                        else:
                            model.fit(xtrain,ytrain[:,c],sample_weight=weights)
                        val_pred = model.predict(xval)
                        pred = model.predict(xforecast)
                    model_file = OUT/'models'/f'm{month:02}_val{val_days}_{family}_{channel}.joblib'
                    model_file.parent.mkdir(exist_ok=True)
                    joblib.dump(model,model_file,compress=3)
                    restored = joblib.load(model_file)
                    with threadpool_limits(limits=1):
                        reload_pred = restored.predict(xforecast)
                    np.testing.assert_allclose(pred,reload_pred,rtol=0,atol=1e-8)
                    channel_predictions.append(pred)
                    model_rows.append(dict(family=family,channel=channel,
                        model_sha256=digest(model_file),model_file=str(model_file.relative_to(ROOT)),
                        fit_predict_save_reload_seconds=perf_counter()-tick,
                        iterations=int(model.n_iter_) if family=='hgb' else None,
                        validation_rmse_kw=float(np.sqrt(np.mean((val_pred-yval[:,c])**2))),
                        max_reload_difference_kw=float(np.max(np.abs(pred-reload_pred))),
                        parameters=model.get_params()))
                    del model,restored
                raw_family = np.column_stack(channel_predictions).reshape(28,144,2)
                raw_family = np.maximum(raw_family,0.)
                for j,d in enumerate(test): raw_family[j,:,1] *= _daylight(data.actual,int(d))
                predictions[family] = raw_family
            raw = old[:asof+28-31].copy()
            start = asof-31
            raw[start:] = .5*predictions['hgb']+.5*predictions['extra_trees']
            raw_delta = float(np.max(np.abs(raw[start:]-old[start:start+28])))
            if val_days == 7:
                np.testing.assert_allclose(raw[start:],old[start:start+28],rtol=0,atol=1e-8)
            final,ridge,_ = postprocess(data.actual,origins[:len(raw)],raw,BASE,
                ridge_initial=saved_ridge[:len(raw)],start=start)
            final = final[start:]
            if val_days == 7:
                with np.load(OUT/'baseline.npz') as z: expected = z['values'][start:start+28]
                np.testing.assert_allclose(final,expected,rtol=0,atol=1e-8)
                final_delta = float(np.max(np.abs(final-expected)))
            else:
                final_delta = None
            # Prove feature and fitting inputs unchanged under future actual mutation.
            changed = copy.copy(data);changed.actual = data.actual.copy()
            changed.actual[asof*144:] += [50000.,30000.]
            past_changed = np.stack([features_for_day(changed,int(d))[0] for d in np.r_[train,validation]])
            np.testing.assert_array_equal(past_changed,features[np.r_[train,validation]-7])
            np.testing.assert_array_equal(historical_labels(changed,train,cutoff*144).reshape(-1,2),ytrain)
            np.testing.assert_array_equal(historical_labels(changed,validation,asof*144).reshape(-1,2),yval)
            np.testing.assert_array_equal(features_for_day(changed,asof)[0],features[asof-7])
            truth = data.actual[test[:,None]*144+np.arange(144),:2]
            for horizon in HORIZONS:
                scores = list(metric_rows(final[:horizon],truth[:horizon],data.fixed_price))
                verify_metrics(final[:horizon],truth[:horizon],scores)
                for r in scores:
                    all_scores.append(dict(fold_month=month,validation_days=val_days,test_days=horizon,
                        train_start=date(train[0]),train_end=date(train[-1]),
                        validation_start=date(validation[0]),validation_end=date(validation[-1]),
                        test_start=date(asof),test_end=date(asof+horizon-1),**r))
            for j,d in enumerate(test):
                for r in metric_rows(final[j:j+1],truth[j:j+1],data.fixed_price):
                    all_daily.append(dict(fold_month=month,validation_days=val_days,date=date(d),day=int(d),**r))
            archive = OUT/f'cv_m{month:02}_val{val_days}.npz'
            np.savez_compressed(archive,origins=test*144,values=final,raw=raw[start:])
            seconds = perf_counter()-began
            audit = dict(fold_month=month,validation_days=val_days,training_days=len(train),
                train_start_day=int(train[0]),train_end_day=int(train[-1]),
                validation_start_day=int(validation[0]),validation_end_day=int(validation[-1]),
                train_label_stop_exclusive=cutoff*144,validation_label_stop_exclusive=asof*144,
                first_test_origin=asof*144,test_start=date(asof),test_end=date(asof+27),
                xtrain_sha256=array_hash(xtrain),xval_sha256=array_hash(xval),
                ytrain_sha256=array_hash(ytrain),yval_sha256=array_hash(yval),
                weight_sha256=array_hash(weights),forecast_sha256=digest(archive),
                models=model_rows,seconds=seconds,genuine_refit=True,
                training_validation_inputs_and_first_issue_features_unchanged_under_future_mutation=True,
                baseline_raw_max_difference_kw=raw_delta if val_days==7 else None,
                baseline_full_max_difference_kw=final_delta)
            audits.append(audit)
            if pilot_seconds is None:
                pilot_seconds = seconds
                write_json('pilot_timing.json',dict(fold_month=month,validation_days=val_days,
                    seconds=seconds,remaining_fits_estimate_seconds=seconds*8,
                    scope_retained='3 folds x3 validation lengths, 4 models per fit'))
            write_json('cv_training_audit.json',audits)
            print(json.dumps(dict(stage='cv',month=month,validation_days=val_days,seconds=seconds,
                                  first_fit_pilot_seconds=pilot_seconds)),flush=True)
    frame = pd.DataFrame(all_scores)
    key = ['fold_month','test_days','channel']
    ref = frame[frame.validation_days==7].set_index(key)
    for metric in ['rmse_kw','mae_kw','daily_energy_rmse_kwh','cumulative_error_rmse_kwh']:
        frame[metric+'_change_pct_vs_val7'] = [100*(r[metric]/ref.loc[tuple(r[k] for k in key),metric]-1) for r in all_scores]
    frame.to_csv(OUT/'cv_metrics.csv',index=False)
    pd.DataFrame(all_daily).to_csv(OUT/'cv_daily.csv',index=False)
    pooled = []
    for val_days in VALIDATIONS:
        for horizon in HORIZONS:
            packs = [np.load(OUT/f'cv_m{m:02}_val{val_days}.npz') for m in FOLDS]
            p = np.concatenate([z['values'][:horizon] for z in packs])
            o = np.concatenate([z['origins'][:horizon] for z in packs])
            y = data.actual[o[:,None]+np.arange(144),:2]
            for r in metric_rows(p,y,data.fixed_price):
                pooled.append(dict(validation_days=val_days,test_days_per_fold=horizon,folds=3,**r))
            for z in packs:z.close()
    pd.DataFrame(pooled).to_csv(OUT/'cv_pooled_metrics.csv',index=False)
    return all_scores,audits


def run():
    if OUT.exists():
        raise FileExistsError(f'Refusing to overwrite completed or partial analysis: {OUT}')
    OUT.mkdir(parents=True)
    began = perf_counter()
    data = Data()
    source_files = [Path(__file__),ROOT/'experiments/exp008/forecast_absolute_hgb.py',
        ROOT/'experiments/exp008/forecast_absolute_extra_trees.py',ROOT/'experiments/exp008/forecast_calibration.py',
        ROOT/'experiments/exp008/load_energy_memory.py',ROOT/'experiments/exp008/forecast_shape_diagnostic.py',
        ROOT/'experiments/exp008/forecast_net_hgb.py',ROOT/'experiments/problem2/exp004/data.py',
        ROOT/'experiments/problem2/exp003/data.py']
    for p in source_files:register(p)
    for name in data.hashes:register(ROOT/'data/raw'/name)
    for directory in ['forecast_absolute_hgb','forecast_absolute_extra_trees','forecast_hgb_extra_trees_half']:
        for filename in ['protocol.json','provenance.json','causality_verification.json','independent_verification.json']:
            p = register(OLD/directory/filename)
            if filename.endswith('verification.json'): assert json.loads(p.read_text())['passed']
    for directory,filename in [('forecast_absolute_hgb','direct_hgb_raw.npz'),('forecast_absolute_extra_trees','absolute_extra_trees_raw.npz')]:
        register(OLD/directory/filename)
    protocol = dict(purpose='已选定exp008的Q2最终预测：局部敏感性与时间顺序重训稳定性；不替换最终模型',
        candidates=candidates(),sensitivity_origin_range=['2025-02-01','2025-12-31'],
        raw_data_range=['2025-01-01','2025-12-31'],sensitivity_days=334,slots_per_day=144,
        raw_fixed_source='Signed monthly HGB and ExtraTrees issued archives, 31 historical/calendar features, seed42',
        parameter_definition=dict(weight='raw HGB share, ExtraTrees=1-weight; both channels same',
            window='joint Ridge history in prior issued days; half-life remains window/2 as original family definition',
            penalty_multiplier='multiply intercept penalty2 and other penalties20 equally',
            gain='nonrecursive previous underlying Ridge daily mean load residual coefficient; PV unchanged'),
        cv=dict(fold_months=FOLDS,validation_days=VALIDATIONS,test_prefix_days=HORIZONS,
            training_start='2025-01-08',training_end='day before validation window',
            repeated_fits=9,models_per_fit=4,model_parameters='exact frozen HGB and ExtraTrees parameters',
            validation_role='HGB explicit temporal early stopping; ExtraTrees diagnostic only, no model selection',
            fit_rule='genuinely reinstantiate and fit all4 models for everyfold/validation setting',
            issue_rule='00:00 next144slots; model held fixed within each28day block; features and postprocessing update after completed days',
            warm_start='shared frozen issued raw/Ridge prefix before fold start; new policy begins exactly at fold start; no retrospective in-sample history predictions',
            reference_check='val7 refits reproduce archived monthly raw and final forecasts within1e-8kW',
            horizon_role='7/14/28 are nested scoring windows of the genuinely retrained28day forecast, not independent test samples',
            no_random_shuffle=True,no_test_driven_parameter_selection=True),
        limitations=['2025全期已经参与开发，所有结果均为开发期敏感性/时间顺序回测，非未触碰独立测试。',
            '融合权重、后处理参数单因素变化；窗口变化同步保留原半衰期=window/2规则，不宣称隔离半衰期影响。',
            '重训部分3个固定起点，不能代表所有月份或多年度稳定性。',
            '7/14/28日窗口嵌套且相邻每日误差相关，不当作独立样本或显著性检验。',
            'CV前历史为共同已发布归档暖启动，比较在起点切换验证方案；并非各方案从1月独立全程训练。',
            '单seed42，无跨随机种子的方差结论；预测变化不直接代表调度费用变化。'],
        input_sha256=INPUTS.copy(),numpy_version=np.__version__,sklearn_version=sklearn.__version__)
    write_json('protocol.json',protocol)
    hgb_store = AbsoluteHGBStore('direct_hgb_raw')
    et_store = AbsoluteExtraTreesStore('absolute_extra_trees_raw')
    origins = hgb_store.origins
    np.testing.assert_array_equal(origins,et_store.origins)
    np.testing.assert_array_equal(origins,np.arange(31,365)*144)
    truth = data.actual[origins[:,None]+np.arange(144),:2]
    annual,ridge = sensitivity(data,hgb_store.values,et_store.values,origins,truth)
    cv,audits = temporal_cv(data,hgb_store.values,et_store.values,origins,ridge)
    assert all(digest(ROOT/p)==h for p,h in INPUTS.items())
    net = [r for r in annual if r['channel']=='net']
    baseline = next(r for r in net if r['name']=='baseline')
    summary = dict(complete=True,seconds=perf_counter()-began,sensitivity_configurations=len(candidates()),
        baseline_exact_array_reproduction=True,baseline_net=baseline,
        net_rmse_change_pct_range=[min(r['rmse_kw_change_pct'] for r in net),max(r['rmse_kw_change_pct'] for r in net)],
        sensitivity_all_labels_before_issue=True,sensitivity_future_mutation_checks=18*7,
        temporal_genuine_refits=len(audits),saved_models=sum(len(a['models']) for a in audits),
        temporal_cv_validation_days=VALIDATIONS,temporal_fold_months=FOLDS,
        baseline_refit_forecast_max_difference_kw=max(a['baseline_full_max_difference_kw'] or 0. for a in audits),
        all_inputs_unchanged=True,input_sha256=INPUTS,
        metric_independent_fsum_checks=True,development_not_independent_test=True,
        new_final_model_selection=False,limitations=protocol['limitations'])
    write_json('summary.json',summary)
    (OUT/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    write_json('output_manifest.json',dict(files_sha256={str(p.relative_to(OUT)):digest(p) for p in OUT.rglob('*') if p.is_file() and p.name!='output_manifest.json'}))
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__ == '__main__':
    run()
