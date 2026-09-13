import json,numpy as np,pandas as pd
from pathlib import Path
from experiments.exp008.forecast_absolute_hgb import AbsoluteHGBStore
from experiments.problem2.exp003.data import Data
out=Path('data/results/exp008/hgb_residual_state_diagnostic');d=Data();s=AbsoluteHGBStore();true=d.actual[s.origins[:,None]+np.arange(144)];e=true-s.values;net=(true[:,:,0]-true[:,:,1])/6
bp=Path('data/results/exp008/forecast_absolute_hgb/lp_bridge/direct_hgb_ridge28_memory_tree28_q08_buffer500_334days')
with np.load(bp/'supports.npz') as z:q=z['supports']
pred=(s.values[:,:,0]-s.values[:,:,1])/6
load=e[:,:,0];pv=e[:,:,1];allnet=load-pv
j={
'actual_minus_forecast_global_bias_kw':{'load':float(load.mean()),'pv':float(pv.mean()),'net':float(allnet.mean())},
'component_mse_fraction_from_daily_mean':{'load':float(np.mean(load.mean(1)**2)/np.mean(load**2)),'pv':float(np.mean(pv.mean(1)**2)/np.mean(pv**2))},
'net_energy_rmse_kwh':float(np.sqrt(np.mean((allnet.sum(1)/6)**2))),
'load_energy_rmse_kwh':float(np.sqrt(np.mean((load.sum(1)/6)**2))),
'pv_energy_rmse_kwh':float(np.sqrt(np.mean((pv.sum(1)/6)**2))),
'actual_energy_residual_load_pv_correlation':float(np.corrcoef(load.sum(1),pv.sum(1))[0,1]),
'Tree28':{'mean_shift_from_forecast_kw':float(6*(q.mean(2)-pred).mean()),'median_shift_from_forecast_kw':float(6*(q[:,:,4]-pred).mean()),'mean_distribution_minus_actual_bias_kw':float(6*(q.mean(2)-net).mean()),'median_distribution_minus_actual_bias_kw':float(6*(q[:,:,4]-net).mean()),'lp_q08_actual_coverage':float((net<=np.quantile(q,.8,axis=2)).mean())}}
frame=pd.read_csv(out/'monthly_proper_scores.csv')
monthly=[]
for month in sorted(frame.month.unique()):
 a=frame[(frame.month==month)&(frame.name=='state_weighted56')].iloc[0]
 row={'month':int(month)}
 for ref in ['frozen_uniform56','online_raw28']:
  b=frame[(frame.month==month)&(frame.name==ref)].iloc[0]
  for k in ['price_weighted_slot_crps','prefix_crps_kwh','daily_energy_crps_kwh']:
   row[ref+'__'+k+'_change']=float(a[k]-b[k])
 monthly.append(row)
j['paired_month_score_changes']=monthly
corr=pd.read_csv(out/'residual_serial_correlations.csv');j['annual_serial_correlations']=corr[corr.month==0].to_dict('records')
h=pd.read_csv(out/'hourly_concentration.csv');j['hour_concentration']={'18_to_21_fee':float(h.loc[h.hour.between(18,20),'emergency_cost'].sum()),'18_to_21_share':float(h.loc[h.hour.between(18,20),'annual_emergency_fee_share'].sum()),'08_to_11_fee':float(h.loc[h.hour.between(8,10),'emergency_cost'].sum()),'08_to_11_share':float(h.loc[h.hour.between(8,10),'annual_emergency_fee_share'].sum())}
(out/'additional_diagnostics.json').write_text(json.dumps(j,indent=2))
print(json.dumps(j,indent=2))
