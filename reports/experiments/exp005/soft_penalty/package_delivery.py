"""Freeze the completed report record and package its reviewable deliverables."""
import hashlib
import json
import shutil
import subprocess
import zipfile
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
REPORT = ROOT / 'reports/experiments/exp005'
EVIDENCE = REPORT / 'soft_penalty/evidence'


def main():
    record = json.loads((REPORT / 'soft-penalty-record.json').read_text())
    assert record['complete']
    assert json.loads((REPORT / 'app/src/data.json').read_text())['buildStatus'] == 'complete'
    verified = json.loads((ROOT / 'data/results/exp005/workbook_verification.json').read_text())
    assert hashlib.sha256((REPORT / 'result2.xlsx').read_bytes()).hexdigest() == verified['sha256']
    (REPORT / 'workbook_verification.json').write_text(json.dumps(verified, ensure_ascii=False, indent=2))
    protocol = json.loads((ROOT / 'data/results/exp005/soft-penalty-beta-0.1/protocol.json').read_text())
    original = json.loads((ROOT / 'data/results/exp005/soft-penalty-beta-0.1/original_code_hashes_before.json').read_text())
    assert all(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == h for p, h in original.items())
    entry = {'experiment_id': 'exp005', 'title': '周转惩罚与充放电重叠检验', 'scope': ['2'],
        'variant': 'soft-penalty', 'primary_seed': 42, 'primary_beta': .1, 'sensitivity_beta': .01,
        'protocol': protocol, 'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'new_code_uncommitted': True, 'original_code_hashes': original,
        'new_code_hashes': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((REPORT / 'soft_penalty').glob('*.py'))},
        'metrics': [r for r in record['period'] if r['beta'] is not None],
        'overlap_summary': record['overlap_summary'], 'environment': record['environment'],
        'forecast_metrics': json.loads(pd.read_csv(EVIDENCE / 'forecast_breakdown.csv').query('month == 0').to_json(orient='records')),
        'artifacts': {'report': 'experiments/exp005/report.md', 'html': 'experiments/exp005/report.html', 'workbook': 'experiments/exp005/result2.xlsx'},
        'comparison_note': '与exp004同预测但控制器与爬坡协议不同，只作描述性费用比较；两β同输入初态连续执行，可作权重敏感性对照。'}
    registry = ROOT / 'reports/registry/exp005.json'
    content = json.dumps(entry, ensure_ascii=False, indent=2, allow_nan=False)
    if registry.exists():
        assert json.loads(registry.read_text()) == entry, 'Do not overwrite a different historical registry record'
    else:
        registry.write_text(content)
    shutil.copy2(registry, REPORT / 'experiment-record.json')
    index = ROOT / 'reports/latest.md'
    text = index.read_text()
    if 'experiments/exp005/report.md' not in text:
        index.write_text(text + '\n- [exp005：周转惩罚与充放电重叠检验](experiments/exp005/report.md) · [交互报告](experiments/exp005/report.html) · [主方案Excel](experiments/exp005/result2.xlsx)\n')
    paths = [REPORT / name for name in ['report.html', 'report.md', 'README.md', 'result2.xlsx', 'specified_dates.md',
             'soft-penalty-record.json', 'experiment-record.json', 'workbook_verification.json']]
    paths += sorted(REPORT.glob('section-*.md'))
    paths += sorted((REPORT / 'soft_penalty').glob('*.py'))
    paths += sorted(EVIDENCE.glob('*.csv'))
    paths += sorted((REPORT / 'soft_penalty/figures').glob('*'))
    paths += sorted((REPORT / 'workbook-previews').glob('2-*.png'))
    manifest = {str(p.relative_to(REPORT)): {'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths}
    (REPORT / 'delivery-manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    paths.append(REPORT / 'delivery-manifest.json')
    target = REPORT / 'exp005-delivery.zip'
    with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in paths:
            z.write(p, p.relative_to(REPORT))
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
    print(json.dumps({'file': str(target), 'bytes': target.stat().st_size, 'files': len(paths)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
