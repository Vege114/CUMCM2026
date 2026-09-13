"""Read-only diagnostics of fixed additive versus proportional energy memory.

No forecast fitting, solver call, coefficient search or regional holiday label.
The only proportional comparator has the same fixed gain 0.5.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'data/results/exp008/load_memory_scale_diagnostic'


def corr(x,y):
    if len(x)<3 or min(np.std(x),np.std(y))==0:
        return None
    return float(np.corrcoef(x,y)[0,1])


def summarize(group,transition,frame):
    x=frame
    e=x.raw_daily_error_kw.to_numpy();previous=x.previous_error_kw.to_numpy()
    relative=x.relative_error.to_numpy();old_relative=x.previous_relative_error.to_numpy()
    additive=x.additive_daily_error_kw.to_numpy();proportional=x.proportional_daily_error_kw.to_numpy()
    a=.5*previous;positive=a*e>0
    raw=float(np.sum(e**2));add=float(np.sum(additive**2));prop=float(np.sum(proportional**2))
    return {'group':group,'transition':transition,'pairs':len(x),
            'mean_forecast_daily_load_kw':float(x.forecast_mean_kw.mean()),
            'forecast_scale_ratio_mean':float(x.forecast_scale_ratio.mean()),
            'forecast_scale_ratio_std':float(x.forecast_scale_ratio.std(ddof=0)),
            'median_absolute_scale_jump_pct':float(100*np.median(np.abs(x.forecast_scale_ratio-1))),
            'absolute_error_lag_correlation':corr(previous,e),
            'relative_error_lag_correlation':corr(old_relative,relative),
            'absolute_normalized_persistence_mse':float(np.mean((e-previous)**2)/np.mean(e**2)),
            'relative_normalized_persistence_mse':float(np.mean((relative-old_relative)**2)/np.mean(relative**2)),
            'raw_daily_error_sse_kw2':raw,'additive_half_daily_error_sse_kw2':add,
            'proportional_half_daily_error_sse_kw2':prop,
            'additive_half_sse_change_vs_raw_pct':100*(add/raw-1),
            'proportional_half_sse_change_vs_raw_pct':100*(prop/raw-1),
            'proportional_vs_additive_sse_change_pct':100*(prop/add-1),
            'raw_daily_energy_rmse_kwh':24*float(np.sqrt(np.mean(e**2))),
            'additive_daily_energy_rmse_kwh':24*float(np.sqrt(np.mean(additive**2))),
            'proportional_daily_energy_rmse_kwh':24*float(np.sqrt(np.mean(proportional**2))),
            'additive_wrong_sign_days':int(np.sum(a*e<0)),
            'additive_same_sign_overcorrection_days':int(np.sum(positive&(np.abs(a)>np.abs(e)))),
            'additive_same_sign_undercorrection_days':int(np.sum(positive&(np.abs(a)<np.abs(e)))),
            'raw_slot_load_sse_kw2':float(x.raw_slot_load_sse_kw2.sum()),
            'additive_slot_load_sse_kw2':float(x.additive_slot_load_sse_kw2.sum()),
            'proportional_slot_load_sse_kw2':float(x.proportional_slot_load_sse_kw2.sum()),
            'raw_slot_net_sse_kw2':float(x.raw_slot_net_sse_kw2.sum()),
            'additive_slot_net_sse_kw2':float(x.additive_slot_net_sse_kw2.sum()),
            'proportional_slot_net_sse_kw2':float(x.proportional_slot_net_sse_kw2.sum())}


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    forecast_path=ROOT/'data/results/exp008/neural_joint_calibration/joint_ridge28.npz'
    with np.load(forecast_path) as archive:
        origins=archive['origins'].copy();forecast=archive['values'].copy()
    np.testing.assert_array_equal(origins,np.arange(31,365)*144)
    source_paths=[ROOT/'data/raw'/name for name in ('附件2_小区负载.csv','附件2_光伏发电实际功率.csv')]
    actual=np.stack([pd.read_csv(path).iloc[:,1:].to_numpy(float) for path in source_paths],axis=-1)[31:]
    means=forecast[:,:,0].mean(axis=1)
    error=(actual[:,:,0]-forecast[:,:,0]).mean(axis=1)
    assert means.min()>0
    relative=error/means
    dates=pd.date_range('2025-02-01','2025-12-31')
    additive=forecast.copy();proportional=forecast.copy()
    additive[1:,:,0]=np.maximum(0,forecast[1:,:,0]+.5*error[:-1,None])
    factors=np.maximum(0,1+.5*relative[:-1])
    proportional[1:,:,0]=forecast[1:,:,0]*factors[:,None]
    with np.load(ROOT/'data/results/exp008/load_energy_memory/predictions.npz') as memory:
        np.testing.assert_array_equal(additive,memory['values'])
    rows=[]
    for i in range(1,334):
        previous='weekday' if dates[i-1].weekday()<5 else 'weekend'
        current='weekday' if dates[i].weekday()<5 else 'weekend'
        row={'day':i+31,'date':str(dates[i].date()),'month':int(dates[i].month),
             'previous_date':str(dates[i-1].date()),'transition':previous+'_to_'+current,
             'forecast_mean_kw':float(means[i]),'previous_forecast_mean_kw':float(means[i-1]),
             'forecast_scale_ratio':float(means[i]/means[i-1]),
             'raw_daily_error_kw':float(error[i]),'previous_error_kw':float(error[i-1]),
             'relative_error':float(relative[i]),'previous_relative_error':float(relative[i-1]),
             'additive_correction_kw':float(.5*error[i-1]),
             'proportional_daily_correction_kw':float(.5*relative[i-1]*means[i]),
             'proportional_factor':float(factors[i-1]),
             'previous_label_end_exclusive':int(origins[i-1]+144),
             'current_issue_origin':int(origins[i])}
        for name,value in [('raw',forecast),('additive',additive),('proportional',proportional)]:
            difference=actual[i]-value[i]
            row[name+'_daily_error_kw']=float(difference[:,0].mean())
            row[name+'_slot_load_sse_kw2']=float(np.sum(difference[:,0]**2))
            row[name+'_slot_net_sse_kw2']=float(np.sum((difference[:,0]-difference[:,1])**2))
        rows.append(row)
    daily=pd.DataFrame(rows)
    daily.to_csv(OUT/'daily_pairs.csv',index=False)
    groups={'all333_pairs':np.ones(len(daily),bool),'first30_evaluation_days':daily.day<=60,
            **{f'month_{month:02d}':daily.month==month for month in (2,6,7,9)}}
    summaries=[]
    for group,mask in groups.items():
        frame=daily[mask]
        summaries.append(summarize(group,'all',frame))
        for transition in ('weekday_to_weekday','weekday_to_weekend','weekend_to_weekday','weekend_to_weekend'):
            subset=frame[frame.transition==transition]
            if len(subset):
                summaries.append(summarize(group,transition,subset))
    table=pd.DataFrame(summaries)
    table.to_csv(OUT/'grouped_diagnostics.csv',index=False)
    files=source_paths+[forecast_path,Path(__file__)]
    protocol={'read_only_diagnostic':True,'solver_calls':0,'model_fits':0,'coefficient_search':False,
              'fixed_gain':.5,'only_weekday_definition':'Monday-Friday vs Saturday-Sunday; no regional holiday assumptions',
              'residual':'actual daily mean load minus same-day issued Joint+Ridge28 daily mean load',
              'relative_residual':'same daily mean error divided by that issued daily mean load',
              'additive_formula':'max(0,issued_load[t]+0.5*previous_daily_mean_error)',
              'proportional_comparator':'issued_load[t]*max(0,1+0.5*previous_relative_daily_mean_error)',
              'causal_comparator_labels':'only previous completed day; current actual appears only in ex post scoring',
              'first_formal_day':'unchanged and excluded from lag-pair diagnostics because no previous formal Joint issue',
              'proportional_archived_as_candidate':False,
              'additive_exactly_matches_existing_memory_archive':True,
              'negative_proportional_factor_clips':int(np.sum(1+.5*relative[:-1]<0)),
              'forecast_daily_mean_load_min_kw':float(means.min()),
              'sources_sha256':{str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
              'interpretation_limits':['No causal attribution of MIP/LP fee differences from forecast SSE alone.',
                'The LP result uses334 days while the physical pilot uses30; same-window scores are separately supplied.',
                'Transition month cells contain few pairs; correlations and SSE are descriptive, not significance claims.',
                'Absolute and relative residual variances have different units; compare normalized persistence errors and common-kW correction SSE.']}
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    (OUT/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    print(table[table.transition=='all'][['group','pairs','absolute_error_lag_correlation',
        'relative_error_lag_correlation','absolute_normalized_persistence_mse','relative_normalized_persistence_mse',
        'additive_half_sse_change_vs_raw_pct','proportional_vs_additive_sse_change_pct']].to_string(index=False))
    return table


if __name__=='__main__':
    main()
