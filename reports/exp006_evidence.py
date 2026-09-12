"""Reviewed exp006 evidence: read frozen artifacts only, never plan or execute."""
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/results/exp006'
REPORT = ROOT / 'reports/experiments/exp006'
EVIDENCE = REPORT / 'evidence'
OLD = ROOT / 'reports/experiments/exp004/evidence'
ETA = float(np.sqrt(.9))
SPECIFIED = ['2025-03-20', '2025-06-21', '2025-09-23', '2025-12-21']
LABELS = {
 'primary':'exp006 正式·树DP', 'seed_2026':'exp006 树DP·种子2026', 'seed_3407':'exp006 树DP·种子3407',
 'causal_42':'exp006 历史季节·树DP', 'causal_2026':'exp006 历史季节·2026', 'causal_3407':'exp006 历史季节·3407',
 'cost_only':'exp006 仅费用', 'no_tree':'exp006 无条件树', 'more_throughput_penalty':'exp006 较强吞吐惩罚',
 'zero_terminal':'exp006 无终值', 'greedy_execution':'exp006 旧执行·重新规划', 'no_deadband':'exp006 无死区',
 'primary_grid25':'exp006 正式·25kWh', 'causal_42_grid25':'exp006 历史季节·25kWh',
 'fixed_primary_greedy':'exp006 固定购电·旧执行'}
TARGETS = {'load':'负载','pv':'光伏','net_load':'净负载'}
ROLES = {k:('正式策略' if k=='primary' else '种子稳定性' if k in ('seed_2026','seed_3407','causal_2026','causal_3407') else '预测兼容对照' if k=='causal_42' else '网格精度核查' if 'grid25' in k else '执行控制对照' if k=='fixed_primary_greedy' else '消融') for k in LABELS}

def read(path): return json.loads(Path(path).read_text())
def sha256(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def battery(z, eta=ETA):
    c,d=np.asarray(z['charge']).ravel(),np.asarray(z['discharge']).ravel()
    modes=np.sign(c-d);nonidle=modes[np.abs(c-d)>1e-6]
    return {'charge_kwh':float(c.sum()),'discharge_kwh':float(d.sum()),'throughput_kwh':float((c+d).sum()),
      'equivalent_full_cycles':float((eta*c.sum()+d.sum()/eta)/24000),
      'direction_reversals':int(np.sum(nonidle[1:]*nonidle[:-1]<0)),
      'simultaneous_slots':int(np.sum((c>1e-6)&(d>1e-6))),
      'power_variation_kw':float(np.abs(np.diff((c-d)*6)).sum()),
      'initial_soc':float(np.asarray(z['states'])[0,0]),'final_soc':float(np.asarray(z['states'])[-1,-1])}

def build_evidence():
    EVIDENCE.mkdir(parents=True,exist_ok=True)
    summary=pd.read_csv(OUT/'summary.csv')
    if read(OUT/'run_manifest.json').get('complete') is not True:raise RuntimeError('Formal manifest not complete')
    if 'id' not in summary: summary['id']=summary['run_id'] if 'run_id' in summary else summary['name']
    expected=set(LABELS)
    if set(summary.id)!=expected: raise RuntimeError(f'Incomplete formal matrix: {expected-set(summary.id)}')
    for id in expected:
        v=read(OUT/id/'verification.json')
        if v.get('status')!='passed' and v.get('passed') is not True: raise RuntimeError(f'Unverified result: {id}')
        daily_cost=pd.read_csv(OUT/id/'daily.csv')
        ordered=np.sort(daily_cost.total_cost.to_numpy())[::-1];mass=.1*len(ordered);whole=int(mass)
        cvar=(ordered[:whole].sum()+(mass-whole)*ordered[whole])/mass
        audit=read(OUT/id/'planning_audit.json')['days'];values=[a['objective_surrogate_yuan'] for a in audit if a.get('objective_surrogate_yuan') is not None]
        summary.loc[summary.id==id,'daily_cvar90']=cvar
        summary.loc[summary.id==id,'worst_date']=str(daily_cost.loc[daily_cost.total_cost.idxmax(),'date'])
        summary.loc[summary.id==id,'worst_day_cost']=float(daily_cost.total_cost.max())
        summary.loc[summary.id==id,'sum_daily_surrogate_objective']=sum(values) if len(values)==334 and id!='fixed_primary_greedy' else np.nan
    summary['continuous_optimality_gap']=np.nan
    summary['label']=summary.id.map(LABELS);summary['role']=summary.id.map(ROLES)
    summary['name']=summary.id;summary['experiment']='exp006';summary['policy_id']='exp006/'+summary.id
    summary['physically_comparable']=True;summary['ranking_allowed']=True;summary['exploratory']=False
    summary['source']=summary.id.map(lambda x:f'data/results/exp006/{x}/dispatch_2.npz')
    summary['same_dispatch']=False;summary['status']='同物理口径；新规划及执行'
    summary['total_wan']=summary.total_cost/10000;summary['planned_wan']=summary.planned_cost/10000;summary['emergency_wan']=summary.emergency_cost/10000
    if 'solve_execute_seconds' not in summary:
        summary['solve_execute_seconds']=summary.get('planning_seconds',0)+summary.get('execution_seconds',0)+summary.get('risk_seconds',0)
    old=pd.read_csv(OLD/'cost_history.csv')
    old['role']=old.apply(lambda r:'原登记不可比' if not r.ranking_allowed and not r.exploratory else '含未来探索' if r.exploratory else '当轮正式策略' if r.policy_id in ('exp002/primary','exp003/primary','exp004/causal_season') else '同预测基线' if r.policy_id=='exp004/no_season' else '历史对照',axis=1)
    costs=pd.concat([old,summary],ignore_index=True)
    forecast=pd.read_csv(OLD/'forecast_history.csv')
    forecast['forecast_reused']=False
    new=[]
    for id,variant in [('primary','no_season'),('causal_42','causal_season')]:
        p=forecast[(forecast.experiment=='exp004')&(forecast.name==variant)].copy()
        p['label']=LABELS[id];p['name']=id;p['experiment']='exp006';p['forecast_reused']=True;p['role']=ROLES[id]
        new.append(p)
    forecast=pd.concat([forecast,*new],ignore_index=True)
    # The raw archives give serial switching metrics; never infer them from annual totals.
    battery_rows=[]
    archive={'exp001/original':'data/results/exp001/dispatch_2.npz','exp002/primary':'data/results/exp002/dispatch_2.npz','exp003/primary':'data/results/exp003/dispatch_2.npz',
      'exp004/no_season':'data/results/exp004/no_season/dispatch_2.npz','exp004/causal_season':'data/results/exp004/causal_season/dispatch_2.npz','exp004/oracle_season':'data/results/exp004/oracle_season/dispatch_2.npz'}
    for r in costs.to_dict('records'):
        b={k:r.get(k) for k in ['experiment','policy_id','label','role','physically_comparable','ranking_allowed','exploratory','source','charge_kwh','discharge_kwh','initial_soc','final_soc']}
        b['throughput_kwh']=r['charge_kwh']+r['discharge_kwh']
        b['equivalent_full_cycles']=(ETA*r['charge_kwh']+r['discharge_kwh']/ETA)/24000 if r['physically_comparable'] else None
        b.update(direction_reversals=None,simultaneous_slots=None,power_variation_kw=None)
        file=archive.get(r['policy_id'],r['source'] if r['experiment']=='exp006' else None)
        if file and (ROOT/file).exists():
            with np.load(ROOT/file) as z:
                # Original exp001 used a different battery model; do not assign v2 EFC.
                metrics=battery(z);metrics['equivalent_full_cycles']=metrics['equivalent_full_cycles'] if r['physically_comparable'] else None
                for key in ('charge_kwh','discharge_kwh'):
                    if not np.isclose(metrics[key],r[key],rtol=1e-8,atol=1e-4): raise RuntimeError(f'Historical archive mismatch {r["policy_id"]}/{key}')
                b.update(metrics);b['source']=file
        battery_rows.append(b)
    batteries=pd.DataFrame(battery_rows)
    comparisons=[]
    primary=summary[summary.id=='primary'].iloc[0]
    for r in old.to_dict('records'):
        for metric,unit in [('total_cost','元'),('planned_cost','元'),('emergency_cost','元'),('planned_kwh','kWh'),('emergency_kwh','kWh')]:
            allowed=bool(r['ranking_allowed']);pv=r[metric];cv=primary[metric]
            comparisons.append({'previous_experiment':r['experiment'],'previous_label':r['label'],'current_experiment':'exp006','current_label':primary.label,'metric':metric,'unit':unit,
                'previous':pv,'current':cv,'absolute_change':cv-pv if allowed else None,'relative_change_pct':100*(cv-pv)/abs(pv) if allowed and pv else None,
                'comparable':allowed,'previous_source':r['source'],'current_source':primary.source,'seed':42,'period':'2025-02-01/2025-12-31'})
    for target in TARGETS:
        for population in ['all','pv_generating']:
            p=forecast[(forecast.experiment=='exp006')&(forecast.name=='primary')&(forecast.target==target)&(forecast.population==population)]
            if p.empty: continue
            p=p.iloc[0]
            for r in forecast[(forecast.experiment!='exp006')&(forecast.target==target)&(forecast.population==population)].to_dict('records'):
                for metric in ['mae','rmse','wape_pct','bias']:
                    allowed=not r['exploratory'];pv=r[metric];cv=p[metric]
                    comparisons.append({'previous_experiment':r['experiment'],'previous_label':r['label'],'current_experiment':'exp006','current_label':p.label,'metric':target+'_'+metric,'unit':'%' if metric=='wape_pct' else 'kW','population':population,
                      'previous':pv,'current':cv,'absolute_change':cv-pv if allowed else None,'relative_change_pct':100*(cv-pv)/abs(pv) if allowed and pv else None,'absolute_change_unit':'百分点' if metric=='wape_pct' else 'kW',
                      'comparable':allowed,'previous_source':r['source'],'current_source':p.source,'seed':42,'period':'2025-02-01/2025-12-31'})
    daily=[];details=[];audits=[]
    dates=pd.date_range('2025-02-01','2025-12-31').strftime('%Y-%m-%d').tolist()
    forecast_npz=np.load(ROOT/'data/results/exp004/predictions.npz')
    for row in summary.to_dict('records'):
        id=row['id'];d=pd.read_csv(OUT/id/'daily.csv');d['name']=id;d['id']=id;d['label']=LABELS[id];d['role']=ROLES[id]
        if 'month' not in d:d['month']=pd.to_datetime(d.date).dt.month
        daily.append(d)
        v=read(OUT/id/'verification.json');audits.append({'id':id,'label':LABELS[id],**{k:(json.dumps(val,ensure_ascii=False) if isinstance(val,(dict,list)) else val) for k,val in v.items()}})
        if id not in ['primary','causal_42','cost_only','greedy_execution','fixed_primary_greedy']: continue
        variant=row.get('forecast','causal_season' if id.startswith('causal') else 'no_season');variant=variant if variant in ['no_season','causal_season'] else 'no_season'
        pred=forecast_npz[f'{variant}_seed_42'];worst=str(d.loc[d.total_cost.idxmax(),'date'])
        with np.load(OUT/id/'dispatch_2.npz') as z:
            for date in sorted(set(SPECIFIED+[worst])):
                di=dates.index(date);actual=z['actual'][di]
                for slot in range(144):
                    details.append({'name':id,'label':LABELS[id],'date':date,'hour':slot/6,'slot':slot,
                     'actual_load_kw':float(actual[slot,0]),'actual_pv_kw':float(actual[slot,1]),'forecast_load_kw':float(pred[di,slot,0]),'forecast_pv_kw':float(pred[di,slot,1]),
                     'actual_net_kwh':float((actual[slot,0]-actual[slot,1])/6),'forecast_net_kwh':float((pred[di,slot,0]-pred[di,slot,1])/6),
                     'planned_kwh':float(z['original'][di,slot]),'emergency_kwh':float(z['emergency'][di,slot]),'charge_kwh':float(z['charge'][di,slot]),'discharge_kwh':float(z['discharge'][di,slot]),'soc_kwh':float(z['states'][di,slot])})
    daily=pd.concat(daily,ignore_index=True)
    monthly=daily.groupby(['name','label','month'],as_index=False)[['total_cost','planned_cost','emergency_cost','planned_kwh','emergency_kwh','charge_kwh','discharge_kwh']].sum()
    monthly['date']=monthly.month.map(lambda m:f'2025-{int(m):02d}-01')
    h_daily=pd.read_csv(ROOT/'data/results/exp004/daily_metrics.csv');h_daily=h_daily[(h_daily.seed==42)&(h_daily.name=='no_season')]
    paired=daily[daily.id=='primary'][['date','total_cost']].merge(h_daily[['date','total_cost']],on='date',suffixes=('_current','_baseline'))
    paired['difference_yuan']=paired.total_cost_current-paired.total_cost_baseline
    seeds=summary[summary.id.isin(['primary','seed_2026','seed_3407','causal_42','causal_2026','causal_3407'])].copy()
    seeds['family']=seeds.id.map(lambda x:'历史季节预测' if x.startswith('causal') else '无季节预测')
    seeds['seed_label']=seeds.seed.map(lambda s:f'种子{int(s)}')
    seed_stats=[]
    for name,p in seeds.groupby('family',sort=False):
        for metric in ['total_cost','emergency_cost','emergency_kwh','charge_kwh','discharge_kwh']:
            seed_stats.append({'family':name,'metric':metric,'n':len(p),'mean':p[metric].mean(),'sample_std':p[metric].std(ddof=1)})
    timings=pd.read_csv(OLD/'timings.csv')
    if (OUT/'timings.csv').exists():
        t=pd.read_csv(OUT/'timings.csv')
        if 'stage' in t:
            t['experiment']='exp006'
            if 'label' not in t:t['label']=t['id'].map(LABELS)
            if 'groups' not in t:t['groups']=334
            if 'scope' not in t:t['scope']='单组334日累计；当前CPU与LP并行，非端到端加速比'
            t['source']='data/results/exp006/timings.csv';timings=pd.concat([timings,t],ignore_index=True)
        else:
            times=[]
            for r in summary.to_dict('records'):
                for key,stage in [('risk_seconds','条件分布准备'),('planning_seconds','动态规划'),('execution_seconds','实际执行'),('compute_seconds','调度执行'),('storage_seconds','存储'),('tree_fit_seconds','树拟合（含在条件准备）')]:
                    if key in r:
                        times.append({'experiment':'exp006','label':LABELS[r['id']],'name':r['id'],'stage':stage,'seconds':r[key],'groups':334,'scope':'单组334日累计；当前CPU与LP并行，阶段有包含关系，不相加所有图项','source':'data/results/exp006/summary.csv'})
            timings=pd.concat([timings,pd.DataFrame(times)],ignore_index=True)
    # Reusing forecasts consumes no new training: zero is measured scope, not unavailable history.
    timings=pd.concat([timings,pd.DataFrame([{'experiment':'exp006','label':'exp006 复用预测','stage':'训练','seconds':0,'groups':0,'scope':'冻结exp004预测，不重复训练；原训练费归exp004','source':'experiments/problem2/tree_planning/protocol.approved.json'}, {'experiment':'exp006','label':'exp006 复用预测','stage':'检查点预测','seconds':0,'groups':0,'scope':'直接读取冻结数组，不重新运行预测网络','source':'experiments/problem2/tree_planning/protocol.approved.json'}])],ignore_index=True)
    frames={'cost_history':costs,'forecast_history':forecast,'battery_history':batteries,'relative_comparison':pd.DataFrame(comparisons),'timings':timings,
      'all_runs':summary,'daily':daily,'monthly_cost':monthly,'paired_daily':paired,'detail':pd.DataFrame(details),'specified':daily[(daily.id=='primary')&daily.date.isin(SPECIFIED)],
      'seed_costs':seeds,'seed_summary':pd.DataFrame(seed_stats),'verification':pd.DataFrame(audits)}
    for name,frame in frames.items():frame.to_csv(EVIDENCE/f'{name}.csv',index=False,float_format='%.15g')
    for name in ['run_manifest.json','summary.csv','timings.csv','verification.json','supplemental_verification.json','frozen_inputs_before_formal.json']:
        if (OUT/name).exists():shutil.copy2(OUT/name,EVIDENCE/(name if name not in ('summary.csv','timings.csv') else 'raw_'+name))
    shutil.copy2(ROOT/'experiments/problem2/tree_planning/protocol.approved.json',EVIDENCE/'protocol.json')
    registries={p.stem:sha256(p) for p in sorted((ROOT/'reports/registry').glob('*.json')) if p.stem<'exp006'}
    write_json(EVIDENCE/'registry_coverage.json',{'experiments':list(registries),'sha256':registries,'comparison_included':['exp001','exp002','exp003','exp004'],'note':'exp005未纳入本轮冻结比较；exp001原登记不同物理口径，oracle含未来，均不排名。exp001/2原共享四目标早停边界保留。'})
    write_json(EVIDENCE/'evidence_manifest.json',{'status':'passed','rows':{k:len(v) for k,v in frames.items()},'files':{k+'.csv':sha256(EVIDENCE/(k+'.csv')) for k in frames},'generator':'reports/exp006_evidence.py','solver_invoked':False})
    return frames

if __name__=='__main__':build_evidence()
