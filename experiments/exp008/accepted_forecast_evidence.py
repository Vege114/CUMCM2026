"""Read-only report evidence for the user-accepted fixed blend candidate.

No fitting, dispatch, model selection, or optimization is performed here.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.problem2.exp003.data import Data, ROOT

OUT=ROOT/'data/results/exp008/accepted_forecast_evidence'
FORECAST=ROOT/'data/results/exp008/forecast_hgb_extra_trees_half'
HGB=ROOT/'data/results/exp008/forecast_absolute_hgb'
ET=ROOT/'data/results/exp008/forecast_absolute_extra_trees'
DISPATCH=ROOT/'data/results/exp008/mode_budget_hold1_physical/hgb_extra_trees_half_hold1_cap8_kappa0_334days'
PRIMARY='hgb_extra_trees_half_ridge28_memory'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def metrics(pred,truth,price):
    error=pred-truth
    n=error.size; sae=float(np.abs(error).sum());sse=float((error**2).sum())
    denominator=float(np.abs(truth).sum())
    high=price>=np.quantile(price,.75)
    energy=error.sum(axis=1)/6
    cumulative=error.cumsum(axis=1)/6
    return {'days':len(error),'samples':n,'mae_kw':sae/n,'rmse_kw':float(np.sqrt(sse/n)),
        'bias_kw':float(error.mean()),'wape_pct':100*sae/denominator if denominator else None,
        'absolute_error_sum_kw':sae,'squared_error_sum_kw2':sse,'actual_absolute_sum_kw':denominator,
        'price_weighted_rmse_kw':float(np.sqrt(np.sum(error**2*price)/np.sum(np.broadcast_to(price,error.shape)))),
        'high_price_rmse_kw':float(np.sqrt(np.mean(error[:,high]**2))),
        'daily_energy_rmse_kwh':float(np.sqrt(np.mean(energy**2))),
        'cumulative_error_rmse_kwh':float(np.sqrt(np.mean(cumulative**2)))}


def run():
    if OUT.exists():
        raise FileExistsError('Evidence extraction outputs already exist')
    OUT.mkdir(parents=True)
    source_paths=[FORECAST/name for name in ('protocol.json','provenance.json',
        'independent_verification.json','causality_verification.json','metrics.csv','monthly_metrics.csv',
        'hgb_extra_trees_half_raw.npz','hgb_extra_trees_half_ridge28.npz',PRIMARY+'.npz')]
    source_paths += [directory/name for directory in (HGB,ET) for name in
        ('protocol.json','training_audit.json','provenance.json','causality_verification.json','independent_verification.json')]
    source_paths += [DISPATCH/name for name in ('dispatch.npz','summary.json','independent_audit.json',
        'complete_provenance.json','runtime_source_consistency_before.json','runtime_source_consistency_after.json',
        'matched_HGB_hold1_effect_summary.json','evaluation_protocol.json')]
    signatures={str(p):sha(p) for p in source_paths}
    data=Data();dates=pd.date_range('2025-02-01','2025-12-31')
    stores={}
    for label,name in [('raw_blend','hgb_extra_trees_half_raw'),('ridge28','hgb_extra_trees_half_ridge28'),('final_memory',PRIMARY)]:
        with np.load(FORECAST/f'{name}.npz') as z:
            origins=z['origins'].copy();stores[label]=z['values'].copy()
            np.testing.assert_array_equal(origins,np.arange(31,365)*144)
    truth=data.actual[origins[:,None]+np.arange(144),:2]
    annual,monthly,daily,lead,daylight=[],[],[],[],[]
    for label,values in stores.items():
        for channel in ('load','pv','net'):
            index=int(channel=='pv')
            p=values[:,:,0]-values[:,:,1] if channel=='net' else values[:,:,index]
            y=truth[:,:,0]-truth[:,:,1] if channel=='net' else truth[:,:,index]
            annual.append({'stage':label,'channel':channel,**metrics(p,y,data.fixed_price)})
            for month in range(2,13):
                mask=dates.month==month
                monthly.append({'stage':label,'channel':channel,'month':month,**metrics(p[mask],y[mask],data.fixed_price)})
            if label=='final_memory':
                for i,date in enumerate(dates):
                    row={'date':str(date.date()),'day':i+31,'channel':channel,**metrics(p[i:i+1],y[i:i+1],data.fixed_price)}
                    row['signed_daily_energy_error_kwh']=float((p[i]-y[i]).sum()/6)
                    daily.append(row)
                for hour in range(24):
                    e=(p-y)[:,hour*6:(hour+1)*6]
                    lead.append({'channel':channel,'lead_start_exclusive_minutes':hour*60,
                        'lead_end_inclusive_minutes':(hour+1)*60,'samples':e.size,
                        'rmse_kw':float(np.sqrt(np.mean(e**2))),'mae_kw':float(np.mean(np.abs(e))),
                        'bias_kw':float(e.mean())})
        pv_error=values[:,:,1]-truth[:,:,1]
        for month in [None,*range(2,13)]:
            ids=np.ones(334,dtype=bool) if month is None else dates.month==month
            valid=truth[ids,:,1]>0
            e=pv_error[ids][valid];y=truth[ids,:,1][valid]
            daylight.append({'stage':label,'month':'all' if month is None else month,
                'condition':'actual_PV_kw>0 for retrospective scoring only','samples':len(e),
                'mae_kw':float(np.abs(e).mean()),'rmse_kw':float(np.sqrt(np.mean(e**2))),
                'bias_kw':float(e.mean()),'wape_pct':float(100*np.abs(e).sum()/np.abs(y).sum())})
    frames={'annual_three_stage':annual,'monthly_three_stage':monthly,
        'daily_final':daily,'hourly_lead_final':lead,'actual_PV_generation_periods':daylight}
    for filename,rows in frames.items():
        pd.DataFrame(rows).to_csv(OUT/f'{filename}.csv',index=False)
    # Check all already-published metrics instead of silently redefining them.
    published=pd.read_csv(FORECAST/'metrics.csv')
    mapping={'raw_blend':'hgb_extra_trees_half_raw','ridge28':'hgb_extra_trees_half_ridge28','final_memory':PRIMARY}
    for row in annual:
        old=published[(published.name==mapping[row['stage']])&(published.channel==row['channel'])].iloc[0]
        for key in ('mae_kw','rmse_kw','bias_kw','price_weighted_rmse_kw','high_price_rmse_kw',
                    'daily_energy_rmse_kwh','cumulative_error_rmse_kwh'):
            np.testing.assert_allclose(old[key],row[key],rtol=0,atol=1e-8)
        subset=[r for r in monthly if r['stage']==row['stage'] and r['channel']==row['channel']]
        np.testing.assert_allclose(sum(r['squared_error_sum_kw2'] for r in subset),row['squared_error_sum_kw2'],rtol=1e-14)
    training=[]
    for family,directory in [('HGB',HGB),('ExtraTrees',ET)]:
        for row in json.loads((directory/'training_audit.json').read_text()):
            start=row['asof_day']
            for model in row['models']:
                path=Path(model['model_path']);assert sha(path)==model['model_sha256']
                training.append({'family':family,'month':row['month'],'channel':model['channel'],
                    'train_first_day':min(row['training_days']),'train_last_day':max(row['training_days']),
                    'validation_first_day':min(row['validation_days']),'validation_last_day':max(row['validation_days']),
                    'first_formal_day':start,'train_samples':row['training_rows'],
                    'validation_samples':row['validation_rows'],'iterations':model.get('iterations'),
                    'trees':model.get('trees'),'training_seconds':model['training_seconds'],
                    'model_path':str(path),'model_sha256':model['model_sha256']})
    pd.DataFrame(training).to_csv(OUT/'training_models_and_boundaries.csv',index=False)
    final=[r for r in annual if r['stage']=='final_memory']
    daily_net=[r for r in daily if r['channel']=='net']
    worst_rmse=sorted(daily_net,key=lambda r:r['rmse_kw'],reverse=True)[:5]
    worst_energy=sorted(daily_net,key=lambda r:abs(r['signed_daily_energy_error_kwh']),reverse=True)[:5]
    config={family:json.loads((directory/'protocol.json').read_text())['model_configuration']
            for family,directory in [('HGB',HGB),('ExtraTrees',ET)]}
    payload={'accepted_Q2_dispatch':str(DISPATCH/'dispatch.npz'),'model_id':PRIMARY,
        'raw_forecast_model_families':config,'monthly_model_count_per_family':22,'total_raw_models':44,
        'fixed_raw_blend_weights':[.5,.5],'seed':42,'number_of_seeds':1,
        'seed_sample_standard_deviation':None,'seed_std_reason':'Only one formal seed; no multi-seed stability estimate.',
        'features':json.loads((HGB/'protocol.json').read_text())['features'],
        'final_annual_metrics':final,'worst_net_daily_RMSE':worst_rmse,'worst_net_daily_energy':worst_energy,
        'training_seconds_recorded_sum':{family:sum(r['training_seconds'] for r in training if r['family']==family)
                                         for family in ('HGB','ExtraTrees')},
        'timing_scope':'Sum of 22 recorded primary model fits per family; excludes feature construction, audit retraining, prediction, postprocessing, dispatch, export and report.',
        'unmeasured_stages_seconds':{'forecast_prediction':None,'ridge_and_memory':None,'report':None},
        'metric_definitions':{'error_sign':'prediction minus actual','evaluation':'334days,48096intervals,2025-02-01..2025-12-31',
            'RMSE':'sqrt(sum(error^2)/sample_count); never average monthly RMSE',
            'MAE':'sum(abs(error))/sample_count','WAPE':'100*sum(abs(error))/sum(abs(actual)); null if denominator zero',
            'daily_energy_RMSE':'sqrt(mean_days((sum_slots(error_kw)/6)^2))',
            'cumulative_error_RMSE':'sqrt(mean_days_slots((cumsum_slots(error_kw)/6)^2))',
            'high_price':'fixed_price >= its .75 empirical quantile',
            'PV_generation_mask':'actual PV>0 only for retrospective metric; never a forecasting input'},
        'fit_boundary':'Monthly training days7..month_start-8; prior7complete validation days; model frozen within formal month; daily features use strictly earlier actuals.',
        'ridge28':'Two residual regressions with cross-channel forecast features; preceding at most28 issued days, half-life14; diagonal penalties intercept2,others20; corrections clipped to max(100kW,3*historical residualRMSE).',
        'memory':'load_final[d,t]=max(0,load_ridge[d,t]+.5*mean_t(actual_load[d-1,t]-load_ridge[d-1,t])); PV unchanged; prior residual is pre-memory, hence nonrecursive.',
        'cold_start':'No earlier issued calibration output on Feb1: raw passes through; January planning errors retain explicitly labelled periodic forecast.',
        'development_not_independent_test':True,'does_not_change_user_accepted_model':True,
        'not_final_report_and_no_new_optimization':True}
    save(OUT/'forecast_method_report_payload.json',payload)
    assert all(sha(Path(path))==digest for path,digest in signatures.items())
    save(OUT/'evidence_manifest.json',{'complete':True,'inputs_sha256':signatures,'data_sha256':data.hashes,
        'source_sha256':sha(Path(__file__)),'published_metrics_recomputed_and_matched':True,
        'monthly_sums_reconcile_to_annual':True,'all44_raw_model_file_hashes_checked':True,
        'no_fit_or_dispatch_or_optimization_performed':True,
        'outputs_sha256':{str(p.relative_to(OUT)):sha(p) for p in OUT.iterdir() if p.is_file()}})
    (OUT/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({'final_metrics':final,'worst_RMSE_day':worst_rmse[0],
        'training_seconds':payload['training_seconds_recorded_sum']},ensure_ascii=False,indent=2))


if __name__=='__main__':
    run()
