"""Assemble every predeclared exp009 result; never select by realized performance."""
from __future__ import annotations
import hashlib
import json
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from experiments.exp008.report_payload import (component_fees, daily_tables, workbook_data,
    FEE_KEYS, DATES, interval, history_payload)
from experiments.exp008.verify import source_arrays, verify_arrays
from experiments.exp009.q1_cost_model import audit as q1_audit

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'reports/experiments/exp009'
RESULTS=ROOT/'data/results/exp009'
SPECS={
 '2':('q2_cost_only/dispatch.npz','q2_cost_only/planning_audit.json','q2_cost_only/summary.json'),
 '3':('update_cost_only/full334/3/dispatch_3.npz','update_cost_only/full334/3/audit.json','update_cost_only/full334/3/independent_verification.json'),
 '4-2':('q4_2_cost_only/full334/dispatch_4-2.npz','q4_2_cost_only/full334/audit.json','q4_2_cost_only/independent_audit.json'),
 '4-3':('update_cost_only/full334/4-3/dispatch_4-3.npz','update_cost_only/full334/4-3/audit.json','update_cost_only/full334/4-3/independent_verification.json')}
def read(p):return json.loads(Path(p).read_text())
def save(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):
 p=Path(p);return {'path':str(p.relative_to(ROOT)), 'sha256':sha(p),'bytes':p.stat().st_size}

def annual(s,protocol):
 archive,audit_path,independent=(RESULTS/p for p in SPECS[s])
 audit=read(audit_path);ind=read(independent)
 assert ind.get('passed',ind.get('verification',{}).get('passed')) is True, str(independent)
 with np.load(archive,allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
 actual,price=source_arrays(s,31,334)
 check=verify_arrays(a,s,expected_days=334,start_day=31,initial_soc=protocol['fixed']['initial_soc_kwh'][s],
  initial_mode=1,initial_power_kw=protocol['fixed']['annual_initial_power_kw'],
  source_actual=actual,source_price=price,audit_records=audit)
 assert check['passed'],check['errors']
 check.pop('goal',None) # The exp008 cost/count stopping rule is not this control's acceptance rule.
 dates=pd.date_range('2025-02-01','2025-12-31').strftime('%Y-%m-%d').tolist()
 daily=[{'date':d,**component_fees(a['fees'][i]),
  **{k+'_kwh':float(a[v][i].sum()) for k,v in [('original_purchase','original'),('final_purchase','final'),('emergency','emergency'),('charge','charge'),('discharge','discharge')]},
  'soc_start_kwh':float(a['states'][i,0]),'soc_end_kwh':float(a['states'][i,-1])} for i,d in enumerate(dates)]
 sums=[*FEE_KEYS,'adjustment_cost_yuan','total_cost_yuan','original_purchase_kwh','final_purchase_kwh','emergency_kwh','charge_kwh','discharge_kwh']
 monthly=[{'month':m,**{k:sum(r[k] for r in daily if r['date'].startswith(m)) for k in sums}} for m in sorted({d[:7] for d in dates})]
 return {'scenario':s,'role':'predeclared_cost_only_control','archive':ref(archive),'issue_audit':ref(audit_path),
  'independent_audit':ref(independent),'verification':check,'fees':component_fees(a['fees']),
  'battery':check['battery_metrics'],'daily':daily,'monthly':monthly,
  'specified_dates':[daily_tables(a,dates.index(d),d) for d in DATES],
  'worst_days_by_bill':sorted(daily,key=lambda r:r['total_cost_yuan'],reverse=True)[:5],
  'workbook_data':workbook_data(a,dates,s)}

def q1():
 path=RESULTS/'q1/cost_only.json';r=read(path)
 assert len(r['stages'])==1 and r['stages'][0]['stage']==1
 q={k:np.array(v,float) for k,v in r['trajectory'].items()}
 actual,price=source_arrays('1',0,1)
 specialized=q1_audit(np.column_stack((price,actual[0])),dict(r,trajectory=q))
 g=q['g'][None];zero=np.zeros_like(g)
 a=dict(original=g,final=g,charge=q['c'][None]/6,discharge=q['d'][None]/6,
  states=q['E'][None],emergency=zero,surplus=q['w'][None],actual=actual,price=price[None],
  fees=np.stack((g*price,zero,zero,zero),axis=-1))
 check=verify_arrays(a,'1',expected_days=1,start_day=0,initial_soc=6000,initial_mode=0,initial_power_kw=0,source_actual=actual,source_price=price)
 assert check['passed'],check['errors']
 check.pop('goal',None)
 tables=daily_tables(a,0,'attachment1_given_day')
 return {'archive':ref(path),'verification':check,'single_cost_stage_verification':specialized,
  'stages':r['stages'],'config':r['config'],'fees':component_fees(a['fees']),'battery':check['battery_metrics'],
  'wall_seconds':r['wall_seconds'],'final_trajectory_stage':1,'specified_tables':tables,
  'workbook_data':{'target_filename':'result1.xlsx','plan_rows':[[interval(t),float(q['g'][t])] for t in range(144)],
   'battery_rows':[[b['interval'],b['charge_kwh'],b['discharge_kwh'],'0:00' if i==0 else '24:00' if i==1 else None,
    float(q['E'][0 if i==0 else -1]) if i<2 else None] for i,b in enumerate(tables['table2']['four_hour_blocks'])]}}

def build():
 protocol=read(ROOT/'experiments/exp009/protocol.json')
 old=read(ROOT/'reports/experiments/exp008/final_payload.json')
 cases={s:annual(s,protocol) for s in SPECS};one=q1()
 rows=[]
 for s,c in [('1',one),*cases.items()]:
  previous=old['q1']['stages'][-1]['cost'] if s=='1' else old['scenarios'][s]['fees']['total_cost_yuan']
  current=c['fees']['total_cost_yuan']
  rows.append({'scenario':s,'previous_experiment':'exp008','experiment':'exp009','role':'predeclared_cost_only_control',
   'days':1 if s=='1' else 334,'seed':None if s=='1' else 42,'previous':previous,'current':current,
   'absolute_change_yuan':current-previous,'relative_change_pct':100*(current-previous)/abs(previous),
   'comparable':True,**c['fees'],'emergency_kwh':0 if s=='1' else sum(d['emergency_kwh'] for d in c['daily'])})
 snapshots=[]
 for name in ('report.md','history-comparison.md','battery-power.md'):
  source=f"{protocol['template_main_commit']}:reports/templates/{name}"
  target=OUT/'evidence/templates'/name
  try:raw=subprocess.check_output(['git','show',source],cwd=ROOT,stderr=subprocess.DEVNULL)
  except subprocess.CalledProcessError:
   raw=target.read_bytes()
   expected=next(v['sha256'] for v in old['templates']['files'] if v['git_source'].endswith('/'+name))
   assert hashlib.sha256(raw).hexdigest()==expected
  target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
  snapshots.append({'source':source,**ref(target)})
 issues=[];history=history_payload(issues)
 assert not issues,issues
 history['experiments'].append({'experiment_id':'exp008','original':read(ROOT/'reports/registry/exp008.json'),
  'source':ref(ROOT/'reports/registry/exp008.json')})
 payload={'experiment_id':'exp009','mode':'final','status':'all predeclared complete results; no fee/count gate',
  'protocol':protocol,'protocol_sha256':sha(ROOT/'experiments/exp009/protocol.json'),
  'templates':snapshots,'q1':one,'scenarios':cases,'comparison':rows,'history':history,
  'exp008_registry':ref(ROOT/'reports/registry/exp008.json'),
  'forecast_evidence_reused':old['prediction_evidence'], 'metric_contract':old['metric_contract'],
  'acceptance':'one full predefined run each; independent physics, billing, forecast identity; regressions retained',
  'data_hashes':read(ROOT/'reports/registry/exp008.json')['data_hashes']}
 OUT.mkdir(parents=True,exist_ok=True)
 save(OUT/'final_payload.json',payload);save(OUT/'data_hashes.json',payload['data_hashes'])
 save(OUT/'evidence/exp008_comparison.json',rows)
 pd.DataFrame(rows).to_csv(OUT/'evidence/exp008_comparison.csv',index=False)
 for s,c in [('1',one),*cases.items()]:save(ROOT/f'.work/exp009/workbook-{s}.json',c['workbook_data'])
 print(json.dumps({'complete':True,'comparison':rows},ensure_ascii=False))
 return payload

if __name__=='__main__':build()
