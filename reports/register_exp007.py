"""Register the measured PPO run without rewriting heterogeneous old records.

The exp007 evidence adapter supplies explicit historical comparability. The
legacy generic registrar assumes homogeneous schemas and would select the last
same-scenario ablation as primary; this registrar keeps only the prespecified
primary in `metrics` and records all six runs separately.
"""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT/'reports/experiments/exp007'


def read(path):
    return json.loads(Path(path).read_text())


def main():
    data = read(REPORT/'evidence/report_data.json')
    manifest = read(ROOT/'data/results/exp007/run_manifest.json')
    independent = read(REPORT/'evidence/independent_qa.json')
    if not manifest['complete'] or not data['verification']['passed']:
        raise RuntimeError('Only completed, verified experiments can be registered')
    if not (independent.get('passed') or independent.get('status') == 'passed'):
        raise RuntimeError('Independent annual verification must pass')
    summary = next(r for r in data['tables']['all_runs'] if r['run_id'] == 'regularized_42')
    metrics = {k: v for k, v in summary.items() if k not in ('family', 'label')}
    metrics.update(scenario='2', role='primary', model='categorical_PPO', violations=0,
                   solve_execute_seconds=summary['planning_seconds']+summary['execution_seconds'])
    protocol = read(ROOT/'experiments/problem2/rl_planning/protocol.approved.json')
    forecasts = [dict(r, role='primary', scenario='2', variant='frozen_exp004_no_season')
                 for r in data['tables']['forecast_history']
                 if r['policy_id'] == 'exp007/regularized_42']
    record = {
        'experiment_id': 'exp007', 'title': '第二问 PPO 规划：电池动作减少但总费用上升',
        'scope': ['2'], 'protocol': {
            'version': 'q2-causal-ppo-v1', 'period': ['2025-02-01', '2025-12-31'],
            'time_alignment': 'interval-end actual labels; 144 ten-minute midnight forecast slots',
            'billing': 'sum(price*planned_kwh + 5*price*emergency_kwh); no residual credit',
            'efficiency': 'eta_charge=eta_discharge=sqrt(0.9)',
            'initial_soc_kwh': 1421.7991105135516, 'soc_bounds_kwh': [1200, 10800],
            'max_power_kw': 5000, 'execution': 'fixed midnight plan plus causal intended-action projection',
            'approved_snapshot': protocol, 'same_physics_comparisons':
            ['exp004/no_season', 'exp006/primary', 'exp006/greedy_execution'],
            'exp005_comparison': 'same numerical inputs and billing; different hard ramp/emergency charging; descriptive only',
        },
        'code_commit': manifest['code_commit'], 'data_hashes': manifest['evidence']['data_sha256'],
        'environment': read(ROOT/'data/results/exp007/pilot/environment.json'),
        'seeds': [42, 2026, 3407], 'forecast_seed': 42,
        'models': ['categorical_PPO'], 'model_configuration': protocol['algorithm'],
        'metric_definitions': data['definitions'], 'forecast_metrics': forecasts,
        'metrics': [metrics], 'all_runs': data['tables']['all_runs'],
        'seed_summary': data['tables']['seed_summary'],
        'technical_path': ['frozen exp004 midnight forecasts', 'past-only conditional residual tree',
                           '29 forecast/intended-state observations', '45 signed battery/purchase actions',
                           'causal chronological PPO training', 'lock all 144 midnight purchases',
                           'physical action projection and actual settlement',
                           'independent source/budget/physics/archive/export verification'],
        'artifacts': {'report': 'experiments/exp007/report.md', 'html': 'experiments/exp007/report.html',
                      'results': 'experiments/exp007/evidence',
                      'workbooks': ['experiments/exp007/result2.xlsx',
                                    'experiments/exp007/cost_only_42/result2.xlsx']},
        'measured_source_sha256': manifest['evidence']['source_sha256'],
        'measured_run_signature': manifest['signature'],
        'formal_run_executed': True, 'formal_run_wall_seconds': manifest['formal_wall_seconds'],
        'primary_role_retained': True,
        'report_template_remote_main_checked': 'bca8dbe3004b0a87cd7be71a06891a4af87a9eb8',
        'report_template_content_commit': '7ecd9b988665ad628dbf048ed295457166170ffc',
        'limitations': ['fixed-budget, discretized partially observed daily PPO; no optimality/convergence guarantee',
                        'daily terminal inventory shaping is absent from actual bills; actor has no final-date indicator',
                        'three seeds share the same year and forecasts, not independent annual generalization',
                        'cycling/intensity proxies are not calibrated battery lifetime estimates'],
    }
    record['environment'] = {k: v for k, v in record['environment'].items()
                             if k not in ('formal_run_executed', 'protocol_sha256', 'tests_sha256')}
    schema = read(ROOT/'reports/templates/experiment.schema.json')
    for key in schema['required']:
        if record.get(key) in (None, {}, [], ''):
            raise RuntimeError(f'Required registry field missing: {key}')
    for key in schema['properties']['protocol']['required']:
        if key not in record['protocol']:
            raise RuntimeError(f'Required protocol field missing: {key}')
    for name in ('report.md', 'report.html', 'result2.xlsx', 'cost_only_42/result2.xlsx'):
        if not (REPORT/name).is_file():
            raise RuntimeError(f'Missing deliverable {name}')
    encoded = json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False)+'\n'
    destination = ROOT/'reports/registry/exp007.json'
    if destination.exists() and destination.read_text() != encoded:
        raise RuntimeError('Refusing to replace a different registered result')
    destination.write_text(encoded)
    (REPORT/'record.json').write_text(encoded)
    (REPORT/'record.draft.json').write_text(encoded)
    latest = ROOT/'reports/latest.md'
    text = latest.read_text()
    if 'experiments/exp007/report.md' not in text:
        entry = ('- [exp007：PPO 规划，电池动作减少但费用上升](experiments/exp007/report.md) · '
                 '[离线交互报告](experiments/exp007/report.html) · '
                 '[主组工作簿](experiments/exp007/result2.xlsx) · '
                 '[历次对比证据](experiments/exp007/evidence/relative_comparison.csv)\n\n')
        text = text.replace('# 实验报告索引\n\n', '# 实验报告索引\n\n'+entry, 1)
        latest.write_text(text)
    (REPORT/'evidence/registration.json').write_text(json.dumps({
        'registered': True, 'registry_sha256': hashlib.sha256(destination.read_bytes()).hexdigest(),
        'schema_required_fields_checked': True, 'old_registry_files_modified': False,
        'history_coverage': [f'exp{i:03}' for i in range(1, 7)],
        'adapter': 'reports/exp007_evidence.py; explicit source and comparability mapping',
    }, indent=2)+'\n')
    print('Registered exp007; old registries unchanged')


if __name__ == '__main__':
    main()
