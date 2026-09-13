"""Score frozen Q4 prices and reconstruct issued Q3/Q4-3 arrays from logged values.

This module imports no forecaster, learner or planner. Published load bias, PV
knot corrections and price coefficients are immutable audit values. Arithmetic
reconstruction must match every stored issued-prediction SHA256 before scoring.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'reports/experiments/exp008/evidence'
BASE=ROOT/'data/results/exp008'


def sha_array(a):
 return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def ref(p):
 p=Path(p)
 return {'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}


def metric(y,p):
 err=np.asarray(p)-np.asarray(y);den=float(np.abs(y).sum())
 return {'n':int(err.size),'mae':float(np.abs(err).mean()),'rmse':float(np.sqrt(np.square(err).mean())),
         'wape_pct':float(100*np.abs(err).sum()/den) if den else None,'bias':float(err.mean())}


def main():
 raw=ROOT/'data/raw'
 load=pd.read_csv(raw/'附件2_小区负载.csv').iloc[:,1:].to_numpy(float).ravel()
 pvactual=pd.read_csv(raw/'附件2_光伏发电实际功率.csv').iloc[:,1:].to_numpy(float).ravel()
 actualprice=pd.read_csv(raw/'附件4.csv').iloc[:,1:].to_numpy(float).ravel()
 fixed=pd.read_csv(raw/'附件1.csv').iloc[:,1].to_numpy(float)
 frame=pd.read_csv(raw/'附件3.csv');frame.iloc[:,0]=frame.iloc[:,0].ffill();official={}
 epoch=pd.Timestamp('2025-01-01')
 for row in frame.itertuples(index=False,name=None):
  day=(pd.Timestamp(row[0])-epoch).days;slot=int(str(row[1]).split(':')[0])*6
  official[day*144+slot]=np.asarray(row[2:],float)
 hgb_path=BASE/'forecast_absolute_hgb/direct_hgb_ridge28_memory_half.npz'
 with np.load(hgb_path,allow_pickle=False) as z:midnight=z['values'].copy();origins=z['origins'].copy()
 assert np.array_equal(origins,np.arange(31,365)*144)
 sources=[Path(__file__),hgb_path,*sorted(raw.glob('*.csv'))]
 rows=[];check_rows=[];forecasts={}
 for scenario in ('3','4-3'):
  ap=BASE/f'absolute_hgb_update_dispatch/full334/{scenario}/audit.json';sources.append(ap)
  audits=json.loads(ap.read_text());assert len(audits)==1336
  arrays=[]
  for row in audits:
   day,slot=row['day'],row['slot'];origin=day*144+slot;n=144-slot;f=row['forecast']
   assert f['information_cutoff_exclusive']==origin and f['max_observed_index']<origin
   lp=midnight[day-31,slot:,0].copy()
   if slot:lp=np.maximum(0,lp+f['load_correction_kw']*np.exp(-np.arange(n)/36))
   rawhourly=official[origin];correction=np.asarray(f['pv_correction_knot_kw'],float)
   corrected=np.where(rawhourly>0,np.maximum(0,rawhourly+correction),0)
   anchor=pvactual[origin-1]
   right=np.interp(np.arange(1,145),np.arange(0,145,6),np.r_[anchor,np.maximum(corrected,0)])
   pp=((np.r_[anchor,right[:-1]]+right)/2)[:n].copy()
   if scenario=='3':price=fixed[slot:].copy()
   else:
    target=np.arange(origin,(day+1)*144);week=np.where(target>=7*144,target-7*144,target-144)
    assert np.all(target-144<origin) and np.all(week<origin)
    phase=2*np.pi*(target%144)/144
    x=np.column_stack((np.ones(n),fixed[target%144],actualprice[target-144],actualprice[week],np.sin(phase),np.cos(phase),np.sin(2*phase),np.cos(2*phase)))
    price=np.maximum(1e-4,np.einsum('ni,i->n',x,np.asarray(f['price_coefficients']),optimize=False))
   predicted={'load_kw':lp,'pv_kw':pp,'price':price}
   for key,value in predicted.items():
    assert sha_array(value)==row['issued_prediction_sha256'][key],(scenario,day,slot,key,'issued hash mismatch; no scoring allowed')
   check_rows.append({'scenario':scenario,'day':day,'issue_slot':slot,'all_three_published_hashes_match':True})
   arrays.append({'day':day,'slot':slot,**predicted})
  forecasts[scenario]=arrays
  byday={r['day']:r for r in arrays if r['slot']==0}
  for horizon in ('next_6h','remaining_day'):
   for variant in ('updated_issued','hold_midnight_issued'):
    fragments=[]
    for r in arrays:
     day,slot=r['day'],r['slot'];n=36 if horizon=='next_6h' else 144-slot
     pred={k:r[k][:n] if variant=='updated_issued' else byday[day][k][slot:slot+n] for k in ('load_kw','pv_kw','price')}
     ids=day*144+slot+np.arange(n)
     fragments.append(pd.DataFrame({'day':day,'month':(epoch+pd.Timedelta(days=day)).month,'issue_hour':slot//6,'lead_slot':np.arange(1,n+1),
      'load_actual':load[ids],'pv_actual':pvactual[ids],'price_actual':actualprice[ids],
      'load_prediction':pred['load_kw'],'pv_prediction':pred['pv_kw'],'price_prediction':pred['price']}))
    data=pd.concat(fragments,ignore_index=True)
    data['net_load_actual']=data.load_actual-data.pv_actual
    data['net_load_prediction']=data.load_prediction-data.pv_prediction
    groups=[('annual','all',data),*[('monthly',str(m),d) for m,d in data.groupby('month')],*[('issue_hour',str(h),d) for h,d in data.groupby('issue_hour')]]
    if horizon=='remaining_day':groups.extend(('lead_slot',str(h),d) for h,d in data.groupby('lead_slot'))
    for granularity,group,d in groups:
     for target in ['load','pv','net_load']+(['price'] if scenario=='4-3' else []):
      for population in (['all','pv_generating'] if target=='pv' else ['all']):
       part=d if population=='all' else d[d.pv_actual>0]
       if not len(part):continue
       rows.append({'scenario':scenario,'model':variant,'target':target,'horizon':horizon,'population':population,'granularity':granularity,'group':group,'unit':'yuan/kWh' if target=='price' else 'kW',**metric(part[target+'_actual'].to_numpy(),part[target+'_prediction'].to_numpy())})
 pricepath=BASE/'hgb_linked_price_forecast/price_predictions.npz';sources.append(pricepath)
 with np.load(pricepath,allow_pickle=False) as z:prices={k:z[k].copy() for k in z.files}
 truth=actualprice[31*144:].reshape(334,144);assert np.array_equal(prices['actual'],truth)
 assert np.array_equal(prices['origins'],origins)
 month_ids=np.array([(epoch+pd.Timedelta(days=int(day))).month for day in range(31,365)])
 for key,name in [('baseline','original_causal_price'),('linked_hgb','linked_HGB_price')]:
  for granularity,group,mask in [('annual','all',np.ones(334,bool)),*[('monthly',str(m),month_ids==m) for m in range(2,13)]]:
   rows.append({'scenario':'4-2','model':name,'target':'price','horizon':'midnight_24h','population':'all','granularity':granularity,'group':group,'unit':'yuan/kWh',**metric(truth[mask],prices[key][mask])})
 output=pd.DataFrame(rows);csv=OUT/'other_question_forecasts.csv';output.to_csv(csv,index=False)
 annual=[r for r in rows if r['granularity']=='annual']
 def changes(s,h,t,pop='all'):
  oldname='original_causal_price' if s=='4-2' else 'hold_midnight_issued';newname='linked_HGB_price' if s=='4-2' else 'updated_issued'
  old=next(r for r in annual if r['scenario']==s and r['horizon']==h and r['target']==t and r['population']==pop and r['model']==oldname)
  new=next(r for r in annual if r['scenario']==s and r['horizon']==h and r['target']==t and r['population']==pop and r['model']==newname)
  return {'scenario':s,'horizon':h,'target':t,'population':pop,'previous_model':oldname,'current_model':newname,'n':new['n'],'previous':old,'current':new,'rmse_reduction_pct':100*(1-new['rmse']/old['rmse']),'wape_change_percentage_points':new['wape_pct']-old['wape_pct']}
 comparisons=[changes('4-2','midnight_24h','price')]
 for s in ('3','4-3'):
  for target in ['load','pv','net_load']+(['price'] if s=='4-3' else []):comparisons.append(changes(s,'next_6h',target))
 source_refs=[ref(p) for p in sources]
 result={'scope':'Frozen forecast-array scoring plus arithmetic reconstruction from previously recorded issuance corrections/coefficients; no forecaster call, fit, planning or dispatch.',
  'passed':True,'period':['2025-02-01','2025-12-31'],'days':334,'source_refs':source_refs,'csv':ref(csv),
  'Q3_Q4_3_reconstruction':{'issued_records':len(check_rows),'individual_channel_hash_checks':3*len(check_rows),'all_published_hashes_match':True,'checks':check_rows,'reconstruction':'load = frozen midnight HGB+logged decayed prefix bias; PV = frozen official knots+logged knot corrections integrated with observed anchor; price=logged coefficients*known lag/harmonic columns; every resulting array must exactly hash-match its original audit'},
  'horizon_definitions':{'midnight_24h':'Q4-2 one midnight issue x144slots x334days=48096','remaining_day':'Q3/Q4-3 four issues with remaining144/108/72/36slots;120240 forecast-target records, overlapping targets count once per issued prediction','next_6h':'only first36slots after each0/6/12/18issue;334*4*36=48096 nonoverlapping operation intervals'},
  'hold_midnight_definition':'Keep the legally issued0:00 prediction unchanged for all later blocks of same day, including its official0:00PV correction. This is forecast-only scoring, not a new or replayed dispatch policy.',
  'pv_generating_definition':'actualPV>0 evaluation-only subset; no future-known feature implication',
  'bias_sign':'prediction minus actual','wape_unit':'percent; difference in WAPE is percentage points',
  'annual_metrics':annual,'controlled_forecast_comparisons':comparisons,
  'limitations':['Forecast improvement does not quantify savings or prove every later issue is economically necessary; no same-current-model no-update dispatch ablation was run.',
   'Full remaining-day and next6h metrics have different target weighting and must not be combined or compared with192168 historical full24h release samples.',
   'Q3 fixed known tariff has no learned price error; Q4-3 price is original causal regression, not Q4-2 linked price.',
   '2025 is a repeatedly examined development year; posthoc evaluation subsets and comparisons are not new holdout validation.']}
 (OUT/'other_question_forecasts.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
 print(json.dumps({'rows':len(rows),'reconstruction_hash_checks':len(check_rows)*3,'comparisons':[{k:v for k,v in r.items() if k in ['scenario','target','n','rmse_reduction_pct','wape_change_percentage_points']} for r in comparisons]},ensure_ascii=False))


if __name__=='__main__':main()
