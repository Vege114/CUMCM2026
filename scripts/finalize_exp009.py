"""Collect runnable sources, bind paper copies, and index exp009 results.

Run after solver/export jobs finish. This command does not train or optimize.
"""
import argparse
import ast
import hashlib
import importlib.metadata
import json
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/results/exp009'
CODE = ROOT / 'Xelatex/code/final'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_closure():
    entries = list((ROOT/'experiments/exp009').glob('*.py'))
    entries += [ROOT/'Xelatex/code/q1_reproduce.py', ROOT/'experiments/exp008/q1.py', ROOT/'scripts/finalize_exp009.py', ROOT/'scripts/verify_exp009_delivery.py']
    entries.append(ROOT/'scripts/prepare_exp009_support.py')
    entries += [ROOT/'experiments/exp008'/name for name in
                ('forecast_absolute_extra_trees.py','forecast_hgb_extra_trees_half.py','audited_store_bridge.py')]
    seen, queue = set(), list(entries)
    while queue:
        path = queue.pop()
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                names = [x.name for x in node.names]
            elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
                names = [node.module]
            else:
                names = []
            for name in names:
                if name.startswith('experiments.'):
                    dependency = ROOT/Path(name.replace('.', '/')).with_suffix('.py')
                    if dependency.is_file():
                        queue.append(dependency)
    # The figure wrappers intentionally load these local style modules by path.
    for name in ['setup_style.py', 'export_figure.py', 'visual_qa.py']:
        seen.add(ROOT/'Xelatex/q4_run/vendor'/name)
    return sorted(seen)


def copy_sources():
    sources = source_closure()
    mapping = []
    for path in sources:
        relative = path.relative_to(ROOT)
        target = CODE/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        assert digest(path) == digest(target)
        mapping.append({'source':relative.as_posix(), 'paper_copy':target.relative_to(ROOT).as_posix(),
                        'sha256':digest(path), 'bytes':path.stat().st_size})
    # Only the executable model and its import dependencies are needed in the
    # printed appendix; figure/export/packaging utilities remain in the package.
    selected = []
    initial = ['Xelatex/code/q1_reproduce.py', 'experiments/exp009/q12_run.py',
               'experiments/exp009/q12_forecast.py', 'experiments/exp009/q34_run.py',
               'experiments/exp009/q34_planner.py', 'experiments/exp009/q34_forecast.py']
    for rel in initial:
        selected.append(ROOT/rel)
    selected += [p for p in sources if p not in selected and 'experiments/' in p.relative_to(ROOT).as_posix()
                 and '/exp009/' not in p.as_posix() and p.name != 'q1.py']
    text = r'''% !TeX root = ../数模通用模板.tex
\section{完整求解程序}\label{app:complete-code}
下列程序按支撑材料中的相对目录组织。问题一可直接运行；年度策略读取附件CSV与已发布预测文件，从1月1日开始执行完整调度。复现时保留目录结构，在支撑包根目录安装所列依赖后运行：
\begin{lstlisting}[language=bash,basicstyle=\ttfamily\small,numbers=none,breaklines=true]
python scripts/prepare_exp009_support.py
python Xelatex/code/q1_reproduce.py --quick --out q1_output
python -m experiments.exp009.q12_run --scenario 2 --days 365 --out data/results/exp009/reproduced/q2
python -m experiments.exp009.q12_run --scenario 4-2 --days 365 --out data/results/exp009/reproduced/q4_2
python -m experiments.exp009.q34_run --scenario 3 --days 365 --label reproduction
python -m experiments.exp009.q34_run --scenario 4-3 --days 365 --label reproduction
\end{lstlisting}
以下列出求解入口及所需本地模块的完整内容。绘图、工作簿导出及核验程序随支撑包提供；预测发布值与重新训练预测器分别说明，不将读取已保存预测的运行称为重新训练。
支撑包保留全部365日轨迹，为减少重复文件，准备命令从完整数组恢复一月与评价期切片，并逐文件核对原归档哈希，不重新计算调度。
\begingroup
\renewcommand{\baselinestretch}{1}\selectfont
\lstset{language=Python,basicstyle=\ttfamily\fontsize{8}{10}\selectfont,
numberstyle=\tiny\color{gray},frame=single,breaklines=true,
columns=fullflexible,keepspaces=true,showstringspaces=false,
keywordstyle=\color{blue!65!black},commentstyle=\color{green!35!black},
stringstyle=\color{red!45!black},xleftmargin=1.5em}
'''
    for path in selected:
        rel = path.relative_to(ROOT).as_posix()
        text += '\n\\subsection*{\\nolinkurl{' + rel + '}}\n'
        text += '\\lstinputlisting{code/final/' + rel + '}\n'
    text += '\\endgroup\n'
    (ROOT/'Xelatex/sections/14-附录问题一程序.tex').write_text(text)
    return mapping


def finalize():
    mapping = copy_sources()
    workbooks = ['result1.xlsx','result2.xlsx','result3.xlsx','result4-2.xlsx','result4-3.xlsx']
    missing = [name for name in workbooks if not (OUT/name).is_file()]
    if missing:
        raise RuntimeError('Export not complete: '+', '.join(missing))
    forecasts = [ROOT/'data/results/exp008/forecast_absolute_hgb/direct_hgb_ridge28_memory_half.npz',
                 ROOT/'data/results/exp008/forecast_hgb_extra_trees_half/hgb_extra_trees_half_ridge28_memory.npz']
    forecast_metadata = [ROOT/'data/results/exp008'/folder/name
                         for folder in ('forecast_absolute_hgb','forecast_absolute_extra_trees','forecast_hgb_extra_trees_half')
                         for name in ('protocol.json','provenance.json','training_audit.json','metrics.csv',
                                      'causality_verification.json','independent_verification.json')
                         if (ROOT/'data/results/exp008'/folder/name).is_file()]
    inputs = forecast_metadata + sorted((ROOT/'data/raw').glob('*.csv')) + sorted((ROOT/'data/templates').glob('*.xlsx')) + forecasts
    configuration = [ROOT/'experiments/common/neural_v2/protocol.json']
    configuration += [p for p in (ROOT/'experiments/problem2').glob('**/protocol.json') if p.is_file()]
    configuration += list((OUT/'provenance').glob('*.py'))
    # Retain the bounded historical sensitivity evidence cited in Section 9.
    evidence = ROOT/'reports/experiments/exp008/robustness/evidence'
    configuration += sorted((evidence/'q1').glob('*.json'))
    configuration += sorted((evidence/'q1').glob('*.csv'))
    configuration += [evidence/'execution'/name for name in
                      ('protocol.json','noise_summary.csv','summary.json')]
    if (OUT/'source_documentation_changes.json').is_file():
        configuration.append(OUT/'source_documentation_changes.json')
    libraries = ['numpy','scipy','pandas','scikit-learn','openpyxl','matplotlib','seaborn',
                 'Pillow','joblib','threadpoolctl']
    dependencies = {name:importlib.metadata.version(name) for name in libraries}
    (OUT/'requirements.txt').write_text('\n'.join(name+'=='+version for name,version in dependencies.items())+'\n')
    trajectories = []
    for folder in ['q1','q2','q3','q4_2','q4_3']:
        trajectories += sorted((OUT/folder).glob('*.npz'))
        trajectories += sorted((OUT/folder).glob('summary*.json'))
        if folder == 'q1':
            trajectories += [OUT/'q1/baseline.json', OUT/'q1/revised.json']
    ai = ROOT/'Xelatex/build/ai/AI工具使用详情.pdf'
    if ai.is_file():
        shutil.copy2(ai, OUT/'AI工具使用详情.pdf')
    slices = []
    for folder, scenario in [('q3','3'),('q4_3','4-3')]:
        for period, name in [('feb_dec',f'dispatch_{scenario}.npz'),('january','warmup.npz')]:
            slices.append({'source':f'data/results/exp009/{folder}/dispatch_365.npz',
                           'target':f'data/results/exp009/{folder}/{name}','period':period})
    manifest = {'base_experiment_commit':'459cdb28', 'experiment':'exp009_refund_coldstart',
        'seed':42, 'run_days':365, 'evaluation_days':334,
        'evaluation_dates':['2025-02-01','2025-12-31'],
        'cold_start':'Each accepted controller starts 2025-01-01 at 6000 kWh using its causal insufficient-history rules; no exp002 dispatch warmup.',
        'settlement':'p*(g + 1.5*max(q-g,0) - 0.5*max(g-q,0) + 5*e), final effective volume relative to midnight plan',
        'python':sys.version, 'dependencies':dependencies, 'source_bindings':mapping,
        'inputs':{p.relative_to(ROOT).as_posix():digest(p) for p in inputs+configuration},
        'trajectories':{p.relative_to(ROOT).as_posix():digest(p) for p in trajectories},
        'workbooks':{name:digest(OUT/name) for name in workbooks},
        'support_package':{'prepare_command':'python scripts/prepare_exp009_support.py',
                           'derived_slices':slices,
                           'scope':'Only redundant cuts are omitted from ZIP; full 365-day arrays and all original hashes are retained. Git keeps every original slice.'},
        'verification_command':'python scripts/verify_exp009_delivery.py',
        'reproduction_commands':['python Xelatex/code/q1_reproduce.py --quick --out q1_output',
            'python -m experiments.exp009.q12_run --scenario 2 --days 365 --out data/results/exp009/reproduced/q2',
            'python -m experiments.exp009.q12_run --scenario 4-2 --days 365 --out data/results/exp009/reproduced/q4_2',
            'python -m experiments.exp009.q34_run --scenario 3 --days 365 --label reproduction',
            'python -m experiments.exp009.q34_run --scenario 4-3 --days 365 --label reproduction'],
        'actual_run_commands':'See per-scenario protocol.json; original run sources remain hash-bound, with any documentation-only cleanup recorded separately.',
        'forecast_scope':'Accepted issued prediction caches are reused; these solver commands do not retrain forecasting models. Exp008 forecast sources and historical training audits identify the upstream forecasting pipeline.'}
    (OUT/'复现清单.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    archive_files = {ROOT/m[key] for m in mapping for key in ('source','paper_copy')}|set(inputs+configuration)
    for path in OUT.rglob('*'):
        if (path.is_file() and path.suffix not in {'.zip','.log'}
                and not any(part in {'daily_chunks','smoke','reproduced','__pycache__'} or part.startswith('q3_reproduction') or part.startswith('q4_3_reproduction') for part in path.relative_to(OUT).parts)):
            archive_files.add(path)
    for path in [ROOT/'scripts/verify_exp009_delivery.py', ROOT/'Xelatex/AI工具使用详情.tex']:
        if path.is_file(): archive_files.add(path)
    archive = OUT/'支撑材料.zip'
    omitted = {ROOT/item['target'] for item in slices}
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for path in sorted(archive_files):
            if path not in omitted:
                z.write(path,path.relative_to(ROOT).as_posix())
        for m in mapping:
            source = ROOT/m['source']
            for ancestor in source.parents:
                if ancestor == ROOT: break
                init=ancestor/'__init__.py'
                if init.is_file() and init not in archive_files:
                    z.write(init,init.relative_to(ROOT).as_posix());archive_files.add(init)
        z.writestr('README.md', '# 正式支撑材料\n\n在解压目录中依次运行：\n\n```bash\n'
                   'python -m pip install -r data/results/exp009/requirements.txt\n'
                   'python scripts/prepare_exp009_support.py\n'
                   'python scripts/verify_exp009_delivery.py\n```\n\n'
                   '准备命令只从完整365日数组恢复重复的31日和334日切片，并逐SHA-256检查与原归档完全一致。'
                   '全部工作簿及完整年度数据已在包内。独立目录重新求解的命令见 data/results/exp009/README.md。'
                   '绘图还使用原项目 Xelatex/YaHei.Consolas.1.11b.ttf 字体，求解与数据核验不依赖此字体。\n')
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
    print(json.dumps({'sources':len(mapping),'archive_mb':archive.stat().st_size/1024**2,
                      'manifest':'data/results/exp009/复现清单.json'},ensure_ascii=False))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources-only',action='store_true')
    args=parser.parse_args()
    if args.sources_only:
        print(json.dumps({'sources':len(copy_sources())}))
    else:
        finalize()
