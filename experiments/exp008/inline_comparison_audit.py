"""Audit the literal final-conversation fragment and preserve original runtime evidence."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

ROOT=Path(__file__).resolve().parents[2]
REPORT=ROOT/'reports/experiments/exp008'
EVIDENCE=REPORT/'evidence'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,data):
    Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def run():
    fragment=REPORT/'comparison.inline.html'
    text=fragment.read_text()
    soup=BeautifulSoup(text,'html.parser')
    assert not any(soup.find(t) for t in ('html','head','body')) and '<!doctype' not in text.lower()
    assert fragment.stat().st_size<1_000_000
    assert '\\"' not in text and '\\n' not in text
    assert 'document.currentScript' not in text
    assert not any(token in text for token in ('fetch(', 'XMLHttpRequest', 'WebSocket'))
    assert [s['src'] for s in soup.find_all('script',src=True)]==['https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js']
    data=json.loads(soup.find('script',id='exp008-comparison-data').string)
    inputs={}
    def read(path):
        inputs[str(path.relative_to(ROOT))]=sha(path)
        return json.loads(path.read_text())
    def csv(path):
        inputs[str(path.relative_to(ROOT))]=sha(path)
        return pd.read_csv(path)
    comparison=csv(EVIDENCE/'same_question_history_cost.csv')
    for q in data['questions']:
        old=comparison[comparison.scenario.astype(str)==q['id']].iloc[0]
        np.testing.assert_allclose([q['previous'],q['current'],q['delta_pct']],
            [old.previous,old.current,old.relative_change_pct],rtol=0,atol=1e-7)
        assert q['reference']==old.reference and old.physical_settlement_comparable
    baselines=read(ROOT/'data/results/exp008/baselines.json')
    for row in data['history_costs']:
        if row['id']=='exp008/final':
            expected=next(x['current'] for x in data['questions'] if x['id']=='2')
        else:
            baseline=next(x for x in baselines['q2_rows'] if x['policy_id']==row['id'])
            expected=baseline['total_cost']
            assert row['separate']==(not baseline['physically_comparable'])
        np.testing.assert_allclose(row['cost'],expected,rtol=0,atol=1e-8)
    forecast=csv(EVIDENCE/'forecast/history_midnight_comparable.csv')
    verified_forecast_fields=0
    for row in data['forecasts']:
        model='final_blend' if row['id']=='exp008' else row['id']
        for target in ('load','pv'):
            for i,(population,key) in enumerate((('all','rmse_kw'),('all','wape_pct'),('pv_generating','rmse_kw'),('pv_generating','wape_pct'))):
                prior=forecast[(forecast.model==model)&(forecast.target==target)&(forecast.population==population)].iloc[0]
                value=row[target][i]
                if pd.isna(prior[key]): assert value is None
                else: np.testing.assert_allclose(value,prior[key],rtol=0,atol=1e-10)
                verified_forecast_fields+=1
    registries={i:read(ROOT/f'reports/registry/exp{i:03}.json') for i in (1,2,3,4,6)}
    five_path=EVIDENCE/'forecast/source_registry_exp005.json'
    five=read(five_path)
    assert sha(five_path)=='5af3de5bdf08706696eea4ceb4a57d072b9a957bad61eef16d46bb0b6b52c43f'
    now=read(EVIDENCE/'runtime_by_stage.json')
    runtime=[]
    for i in (1,2,3,4):
        records=[r for r in registries[i]['metrics'] if str(r.get('scenario'))=='2']
        if i==1: records=[r for r in records if r.get('variant')=='selected']
        elif i in (2,3): records=[r for r in records if r.get('name')=='primary' and str(r.get('seed'))=='42']
        else: records=[r for r in records if r.get('name')=='no_season' and str(r.get('seed'))=='42']
        assert len(records)==1
        r=records[0]
        runtime.append({'id':f'exp{i:03}','seconds':r['solve_execute_seconds'],'field':'solve_execute_seconds',
            'stage_zh':'原登记求解+执行','source':f'reports/registry/exp{i:03}.json',
            'source_sha256':inputs[f'reports/registry/exp{i:03}.json'],'selected_record':r,
            'comparison_scope':'descriptive only; original stage, hardware and algorithms differ',
            'timeout_count':r.get('timeout_count'),'full_workflow_seconds_available':False})
    components=[r for r in five['overlap_summary'] if r['beta']==.1]
    assert len(components)==4
    runtime.append({'id':'exp005','seconds':sum(r['solver_seconds'] for r in components),
        'field':'sum(overlap_summary[beta=0.1].solver_seconds)',
        'stage_zh':'午夜层1/层2、执行层1/层2的LP求解时间之和',
        'source':'git:8108d0475ae4f2ea25565a806dfec5e836f62c0a:reports/registry/exp005.json',
        'retained_source':str(five_path.relative_to(ROOT)),'source_sha256':sha(five_path),
        'components':components,'comparison_scope':'descriptive only; original ramp/emergency-charging physical protocol',
        'full_workflow_seconds_available':False})
    six=next(r for r in registries[6]['metrics'] if r['run_id']=='primary')
    runtime.append({'id':'exp006','seconds':six['run_wall_seconds'],'field':'run_wall_seconds',
        'stage_zh':'原树DP正式策略的运行墙钟，不含预测模型训练',
        'source':'reports/registry/exp006.json','source_sha256':inputs['reports/registry/exp006.json'],
        'selected_record':six,'comparison_scope':'descriptive only; not the same timer boundary as solve_execute_seconds or current MIP-only sum',
        'full_workflow_seconds_available':False})
    current_lookup={r['id']:r for r in now['components']}
    for identifier,key,value in [
        ('exp008_mip','Q2',current_lookup['Q2']['solver_seconds']['sum']),
        ('exp008_refine','Q2',current_lookup['Q2']['refinement_seconds']['sum']),
        ('exp008_hgb','forecast_absolute_hgb',current_lookup['forecast_absolute_hgb']['fit_plus_validation_prediction_seconds']['sum']),
        ('exp008_et','forecast_absolute_extra_trees',current_lookup['forecast_absolute_extra_trees']['fit_plus_validation_prediction_seconds']['sum']),
        ('exp008_wall','Q2',current_lookup['Q2']['run_wall_seconds'])]:
        literal=next(r for r in data['runtime'] if r['id']==identifier)
        runtime.append({'id':identifier,'seconds':value,'field':literal['field'],'stage_zh':literal['label'],
            'source':'reports/experiments/exp008/evidence/runtime_by_stage.json',
            'source_sha256':inputs['reports/experiments/exp008/evidence/runtime_by_stage.json'],
            'upstream_source':current_lookup[key]['source'],
            'comparison_scope':'retain measured component only; different and potentially overlapping stages are not end-to-end wall time',
            'missing_reason':'No original whole-run wall timer' if value is None else None})
    for literal in data['runtime']:
        source=next(r for r in runtime if r['id']==literal['id'])
        if source['seconds'] is None: assert literal['seconds'] is None
        else: np.testing.assert_allclose(literal['seconds'],source['seconds'],rtol=0,atol=1e-10)
    save(EVIDENCE/'history_runtime_descriptive.json',{'unit':'seconds','scope':'Q2 historical original records and final exp008 recorded components; descriptive stage comparison only',
        'rows':runtime,'unknown':None,'speedup_ratio_allowed':False,'stage_sums_are_not_total_workflow':True,
        'exp001_time_is_original_physical_protocol_not_rebased_cost_run':True,
        'source_hashes':inputs,'producer':str(Path(__file__).relative_to(ROOT)),'producer_sha256':sha(__file__)})
    for name,digest in inputs.items(): assert sha(ROOT/name)==digest
    audit={'passed':True,'fragment':str(fragment.relative_to(ROOT)),'fragment_sha256':sha(fragment),'fragment_bytes':fragment.stat().st_size,
        'sources_sha256':inputs,'literal_markup_fragment_contract':True,'self_contained_data_no_network_data_calls':True,
        'only_external_resource':'version-pinned D3 on approved CDN','four_annual_cost_panels_verified':True,
        'Q2_history_rows_verified':len(data['history_costs']),'prediction_scalar_fields_verified':verified_forecast_fields,
        'all_requested_dimensions_present':True,'historical_exp005_missing_values_preserved':True,
        'runtime_rows_verified':len(runtime),'exp001_original_and_exp005_noncomparable_cost_rows_separate':True,
        'no_exp007_or_new_optimization':True,'producer_sha256':sha(__file__)}
    if (EVIDENCE/'inline_comparison_browser_qa.json').exists():
        browser=json.loads((EVIDENCE/'inline_comparison_browser_qa.json').read_text())
        for item in browser['observations']:
            assert not item['errors'] and not item['layout']['bodyOverflow']
            assert abs(item['layout']['rootWidth']-item['width'])<1e-6
            assert all(not v['conflicts'] and not v['outside'] for v in item['layout']['svg'])
            assert all(v['xTickCount']<=4 for v in item['layout']['svg'] if v['width']<=400)
        audit['browser_layout_four_width_theme_combinations_passed']=True
        audit['browser_qa_source_sha256']=sha(EVIDENCE/'inline_comparison_browser_qa.json')
    save(EVIDENCE/'inline_comparison_audit.json',audit)
    print(json.dumps(audit,ensure_ascii=False))


if __name__=='__main__':
    run()
