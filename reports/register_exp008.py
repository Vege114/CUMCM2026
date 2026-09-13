"""Register the user-accepted, physically verified exp008 without rewriting history."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

from experiments.exp008.report_payload import enforce_final

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'reports/experiments/exp008'


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def register(commit):
    if not re.fullmatch(r'[0-9a-f]{40}', commit):
        raise ValueError('A full source commit is required')
    p = read(REPORT/'final_payload.json')
    enforce_final(p)
    source_paths = ['reports/build_report_exp008.py', 'reports/register_exp008.py',
                    'experiments/exp008/report_payload.py', 'experiments/exp008/final_selection.json']
    hashes = {}
    for path in source_paths:
        content = subprocess.check_output(['git', 'show', f'{commit}:{path}'], cwd=ROOT)
        if content != (ROOT/path).read_bytes():
            raise ValueError(f'Source differs from recorded commit: {path}')
        hashes[path] = hashlib.sha256(content).hexdigest()
    if read(REPORT/'report_build.json')['code_commit'] != commit:
        raise ValueError('Rebuild report with this source commit first')
    for artifact in ['report.md', 'report.html', 'specified_dates.md',
                     'result1.xlsx', 'result2.xlsx', 'result3.xlsx', 'result4-2.xlsx', 'result4-3.xlsx']:
        if not (REPORT/artifact).is_file():
            raise ValueError(f'Missing artifact: {artifact}')
    validation = read(REPORT/'evidence/data_validation.json')
    data_hashes = {r['source']['path']: r['source']['sha256'] for r in validation['files']}
    forecast = pd.read_csv(REPORT/'evidence/forecast/annual_metrics.csv')
    forecast_rows = json.loads(forecast.to_json(orient='records', force_ascii=False, double_precision=15))
    for r in forecast_rows:
        r.update({'experiment':'exp008', 'scenario':2 if r['model']=='final_blend' else None,
                  'mae':r['mae_kw'], 'rmse':r['rmse_kw'], 'bias':r['bias_kw'],
                  'unit':'kW', 'period':'annual', 'source':'evidence/forecast/annual_metrics.csv'})
    metrics = [{'scenario':'1', 'role':'primary', 'days':1, 'total_cost':p['q1']['verification']['recomputed_total_cost'],
                'source':p['q1']['archive'], 'verified':p['q1']['verification']['passed']}]
    for scenario, case in p['scenarios'].items():
        metrics.append({'scenario':scenario, 'role':'primary', 'days':334, 'seed':42,
                        'total_cost':case['fees']['total_cost_yuan'], **case['fees'], **case['battery'],
                        'source':case['archive'], 'verified':case['verification']['passed']})
    record = {
        'experiment_id':'exp008', 'title':'费用与换向改善的多问题优化成果',
        'scope':['1','2','3','4-2','4-3'],
        'protocol':{
            'version':'exp008-accepted-2026-09-13',
            'period':{'start':'2025-02-01','end':'2025-12-31','days':334,'q1':'附件1给定单日'},
            'time_alignment':'10-minute interval ending00:10 means00:00–00:10;144slots/day',
            'billing':'actual original purchase bill +1.5p positive final adjustment +0.5p negative final adjustment +5p emergency; Q2/4-2 adjustment zero',
            'efficiency':'eta_charge=eta_discharge=sqrt(0.9)',
            'capacity_kwh':12000,'soc_bounds_kwh':[1200,10800],'power_limit_kw':5000,
            'selection':'development-year iterative selection; user accepts current complete results',
            'finalization_basis':p['finalization_basis'],
            'initial_states':{s:{'soc_kwh':c['battery']['initial_soc'],'power_kw':c['battery']['initial_power_kw']} for s,c in p['scenarios'].items()},
            'user_instruction':'行，我觉得现在这个结果比较满意了，就这样写报告然后提交吧'},
        'code_commit':commit,
        'code_commit_status':'report/selection source bytes verified; executed-run source snapshots remain in their provenance',
        'source_sha256':hashes, 'data_hashes':data_hashes,
        'environment':{'python':sys.version,'platform':platform.platform(),
                       'dependencies':{k:importlib.metadata.version(k) for k in ['numpy','scipy','scikit-learn','pandas','matplotlib']},
                       'numerical_threads':1, 'timing_scope':'current report environment; historical run metadata retained'},
        'seeds':[42], 'primary_seed':42, 'visual_sample_seed':20260912,
        'models':['Q1 two-stage MILP','Q2 HGB/ExtraTrees blend + mode MILP + fixed-mode purchase refinement',
                  'Q3 single HGB + issued official PV + rolling LP',
                  'Q4-2 single HGB + linked-price Ridge + hourly modes',
                  'Q4-3 single HGB + original causal price Ridge + issued official PV + rolling LP'],
        'model_configuration':read(REPORT/'evidence/forecast/model_configuration.json'),
        'metric_definitions':p['metric_contract'], 'forecast_metrics':forecast_rows, 'metrics':metrics,
        'technical_path':['past-only monthly tree fitting and fixed raw blend','Ridge28 calibration then previous-day load memory',
                          'question-specific issued inputs and historical residual paths',
                          'causal physical execution with each policy own actual SOC','actual tariff settlement',
                          'independent raw-data, battery, forecast and workbook verification'],
        'artifacts':{'report':'experiments/exp008/report.md','html':'experiments/exp008/report.html',
                     'results':'experiments/exp008/evidence','payload':'experiments/exp008/final_payload.json',
                     'workbooks':[f'experiments/exp008/result{s}.xlsx' for s in ['1','2','3','4-2','4-3']]},
        'history_note':'exp001–006 retained with original and rescored roles; exp007 explicitly excluded; exp005 different physics not ranked',
        'formal_run_executed':True, 'original_eight_percent_target_met':False,
        'accepted_result_is_untouched_test':False,
    }
    schema=read(ROOT/'reports/templates/experiment.schema.json')
    for key in schema['required']:
        if record.get(key) in (None, {}, [], ''):
            raise ValueError(f'Missing record field: {key}')
    for key in schema['properties']['protocol']['required']:
        if key not in record['protocol']:
            raise ValueError(f'Missing protocol field: {key}')
    destination=ROOT/'reports/registry/exp008.json'
    if destination.exists() and read(destination)!=record:
        raise ValueError('Refusing to overwrite a different registered exp008')
    write(destination,record);write(REPORT/'record.json',record);write(REPORT/'record.draft.json',record)
    write(REPORT/'data_hashes.json',data_hashes)
    print(f'Registered exp008 at source commit {commit}; earlier registries unchanged')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--code-commit',required=True)
    register(parser.parse_args().code_commit)
