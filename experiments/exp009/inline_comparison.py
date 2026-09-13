"""Rebuild conversation comparison from verified payload/evidence only."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
REPORT=ROOT/'reports/experiments/exp009';EV=REPORT/'evidence'
TEMPLATE=ROOT/'experiments/exp009/inline_comparison.template.html'
OUTPUT=REPORT/'comparison-inline.html'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run():
    sources={}
    def register(p):sources[str(p.relative_to(ROOT))]=sha(p);return p
    payload=json.loads(register(REPORT/'final_payload.json').read_text())
    forecast=pd.read_csv(register(EV/'history_forecast.csv'))
    history=pd.read_csv(register(EV/'history_cost.csv'))
    evidence=json.loads(register(EV/'battery/final_evidence.json').read_text())
    assert payload['mode']=='final' and evidence['complete'] and not evidence['pending']
    assert all(c['passed'] for c in evidence['cases'].values())
    costs=[]
    for r in payload['comparison']:
        assert r['days']==(1 if r['scenario']=='1' else 334)
        costs.append({k:r[k] for k in ('scenario','days','previous','current','relative_change_pct')})
        if r['scenario']!='1':
            for version,key in [('exp008','previous'),('exp009','current')]:
                row=history[(history.scenario.astype(str)==r['scenario'])&(history.experiment==version)]
                assert len(row)==1;np.testing.assert_allclose(row.iloc[0]['cost'],r[key],rtol=0,atol=1e-6)
    scores=[]
    for model,label in [('final_blend','Q2融合'),('single_hgb','Q3/4单HGB')]:
        for target,word in [('load','负载'),('pv','光伏'),('net_load','净负荷')]:
            rows=[]
            for version in ('exp008','exp009'):
                q=forecast[(forecast.experiment==version)&(forecast.model==model)&(forecast.target==target)&(forecast.population=='all')]
                assert len(q)==1;rows.append(q.iloc[0])
            for key in ('rmse_kw','wape_pct'):
                assert np.isfinite(rows[0][key]);assert rows[0][key]==rows[1][key]
            scores.append({'model':model,'family':label,'target':target,'label':word,
                'rmse_kw':float(rows[0].rmse_kw),'wape_pct':float(rows[0].wape_pct),'exp008_equals_exp009':True})
    dates=evidence['random_dates'];assert evidence['random_seed']==20260912 and len(dates)==4
    power=[]
    for date in dates:
        item={'date':date,'series':{}}
        for version in ('exp008','exp009'):
            path=EV/f'battery/q2/{version}/power_{date}.csv';f=pd.read_csv(register(path))
            assert len(f)==144 and pd.to_datetime(f.interval_start).dt.strftime('%Y-%m-%d').eq(date).all()
            np.testing.assert_allclose(f.net_battery_power_kw,f.charge_power_kw-f.discharge_power_kw,rtol=0,atol=1e-8)
            item['series'][version]=f.net_battery_power_kw.tolist()
        power.append(item)
    hist=[]
    for version in ('exp001','exp002','exp003','exp004','exp005','exp006','exp008','exp009'):
        r=history[(history.scenario.astype(str)=='2')&(history.experiment==version)]
        assert len(r)==1;r=r.iloc[0]
        hist.append({'experiment':version,'cost':float(r.cost),'comparable':bool(r.comparable)})
    assert next(x for x in hist if x['experiment']=='exp005')['comparable'] is False
    data={'costs':costs,'forecast_scores':scores,'power_days':power,'random_seed':20260912,
        'history_cost':hist,'units':{'cost_q1':'yuan/single given day','cost_annual':'yuan/334days',
        'power':'kW at each original10minute interval end','forecast':'00:00 issue, all slots; not mixed with intraday releases'},
        'sources_sha256':sources}
    literal=TEMPLATE.read_text();assert literal.count('<!-- EXP009_DATA -->')==1
    output=literal.replace('<!-- EXP009_DATA -->',json.dumps(data,ensure_ascii=False,separators=(',',':')))
    assert len(output.encode())<1_000_000 and '\\"' not in output and '\\n' not in output
    assert not any(x in output.lower() for x in ('<!doctype','<html','<head','<body','fetch(','xmlhttprequest','websocket'))
    OUTPUT.write_text(output)
    audit={'passed':True,'output':str(OUTPUT.relative_to(ROOT)),'output_sha256':sha(OUTPUT),
        'bytes':OUTPUT.stat().st_size,'sources_sha256':sources,'template_sha256':sha(TEMPLATE),
        'generator_sha256':sha(Path(__file__)),'five_cost_comparisons':len(costs),
        'forecast_metric_pairs_exactly_equal':len(scores)*2,'forecast_population':'all',
        'power_random_dates':dates,'power_points_per_version_per_day':144,
        'power_values_are_complete_original_CSV_rows':True,'q1_single_day_has_separate_axis':True,
        'exp005_physical_difference_flagged':True,'exp007_excluded':True,'no_training_optimization_or_network_data_access':True}
    (EV/'inline').mkdir(exist_ok=True)
    (EV/'inline/data_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'passed':True,'bytes':audit['bytes'],'forecast_pairs':12,'power_points':4*2*144}))
    return audit
if __name__=='__main__':run()
