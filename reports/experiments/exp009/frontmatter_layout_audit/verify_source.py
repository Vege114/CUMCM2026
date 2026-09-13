"""Read-only audit of the frontmatter compression and local pagination revision."""
from pathlib import Path
import difflib, hashlib, json, re, subprocess

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).parent
BASE = '88d6b1602e79111e7802e79abdcd3d453b82ee83'
ALLOWED = {'Xelatex/sections/02-问题分析.tex',
           'Xelatex/sections/08-问题四.tex',
           'Xelatex/sections/10-模型评价改进与推广.tex',
           'Xelatex/数模通用模板.tex'}
BUILD_ARTIFACTS = {'Xelatex/assumptions-check.pdf', 'Xelatex/q3-table-merge-check.pdf',
                   'Xelatex/q4-table-merge-check.pdf', 'Xelatex/section9-check.pdf',
                   'Xelatex/table-merge-check.pdf'}
checks, issues = [], []
sha = lambda data: hashlib.sha256(data).hexdigest()
def original(path):
    return subprocess.check_output(['git', 'show', f'{BASE}:{path}'], cwd=ROOT)
def check(name, ok, detail=None):
    checks.append({'name': name, 'pass': bool(ok), 'detail': detail})
    if not ok: issues.append(name)

protected, changed = [], []
entries = subprocess.check_output(['git', 'ls-tree', '-r', '-z', BASE], cwd=ROOT)
for entry in entries.split(b'\0'):
    if not entry: continue
    meta, name = entry.split(b'\t', 1)
    _, kind, blob = meta.decode().split(); path = name.decode()
    if kind != 'blob' or path in ALLOWED or path in BUILD_ARTIFACTS: continue
    if not path.startswith(('Xelatex/', 'experiments/', 'scripts/', 'data/', 'C题/',
                            'reports/experiments/exp008/robustness/evidence/')): continue
    p = ROOT / path
    if not p.exists(): changed.append(path); continue
    data = p.read_bytes()
    actual = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
    exact = actual == blob; ok = exact; checkout = None
    if not ok and path in {'scripts/run_ml.ps1', 'scripts/setup_ml.ps1'}:
        attrs = subprocess.check_output(['git', 'check-attr', 'text', 'eol', '--', path], cwd=ROOT, text=True)
        normalized = data.replace(b'\r\n', b'\n')
        normalized_blob = hashlib.sha1(b'blob ' + str(len(normalized)).encode() + b'\0' + normalized).hexdigest()
        ok = ': text: set' in attrs and ': eol: crlf' in attrs and normalized_blob == blob
        if ok: checkout = 'Existing text/eol=crlf checkout conversion only; no Git content change.'
    if not ok: changed.append(path)
    protected.append({'path': path, 'sha256': sha(data), 'baseline_git_blob': blob,
                      'raw_byte_identical': exact, 'pass': ok, 'checkout_note': checkout})
check('protected_manuscript_models_inputs_results_tables_and_figures', not changed,
      {'files': len(protected), 'raw_byte_identical': sum(x['raw_byte_identical'] for x in protected),
       'changed': changed})

main = 'Xelatex/数模通用模板.tex'
old = original(main).decode(); new = (ROOT / main).read_text()
expected = old
for chapter in ['09-模型分析与检验', '10-模型评价改进与推广']:
    pattern = '\\input{sections/' + chapter + '}\n\\newpage'
    assert expected.count(pattern) == 1
    expected = expected.replace(pattern, pattern.rsplit('\n', 1)[0], 1)
pattern = '\\input{sections/08-问题四}\n\\endgroup\n\\newpage'
assert expected.count(pattern) == 1
expected = expected.replace(pattern, pattern.rsplit('\n', 1)[0], 1)
check('main_only_removes_three_additional_hard_page_breaks', new == expected)
check('preamble_font_sizes_margins_class_and_local_group_settings_unchanged',
      new.split('\\input{sections/01-问题背景与问题重述}')[0] == old.split('\\input{sections/01-问题背景与问题重述}')[0])
p8 = 'Xelatex/sections/08-问题四.tex'
spacing_before = '\\renewcommand{\\arraystretch}{1.05}\n\\input{final_results/q4_4-2_specified.tex}'
spacing_after = spacing_before.replace('{1.05}', '{0.96}')
q4_before = original(p8).decode()
check('q4_only_changes_specified_table_group_row_spacing',
      q4_before.count(spacing_before) == 1 and (ROOT / p8).read_text() == q4_before.replace(spacing_before, spacing_after, 1))
check('latest_remote_sections01_and09_preserved_byte_identical',
      all(original(p) == (ROOT / p).read_bytes() for p in [
          'Xelatex/sections/01-问题背景与问题重述.tex',
          'Xelatex/sections/09-模型分析与检验.tex']))

def analysis_parts(s):
    matches = list(re.finditer(r'\\par\\medskip\\noindent\\textbf\{问题([1-4])分析。\}', s))
    result = {}
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else s.index('\\begin{figure}', m.end())
        result[m[1]] = s[m.end():end]
    return result
p2 = 'Xelatex/sections/02-问题分析.tex'
previous = analysis_parts(original(p2).decode()); current = analysis_parts((ROOT / p2).read_text())
check('analysis_question1_paragraph_byte_identical', previous['1'] == current['1'])
check('analysis_questions2_to4_each_one_paragraph',
      all('\n\n' not in current[k].strip() for k in ['2', '3', '4']))
analysis = (ROOT / p2).read_text()
check('obsolete_neural_network_and_dynamic_programming_claims_removed',
      not any(term in analysis for term in ['卷积神经网络', '神经网络模型', '条件误差树', '动态规划']))

def hanzi(s): return len(re.findall(r'[\u4e00-\u9fff]', s))
p10 = 'Xelatex/sections/10-模型评价改进与推广.tex'
evaluation = (ROOT / p10).read_text().split('\\section{模型的评价、改进与推广}', 1)[1]
evaluation_count = hanzi(evaluation)
check('evaluation_body_100_to_200_hanzi', 100 <= evaluation_count <= 200,
      {'body_hanzi': evaluation_count})
p1 = (ROOT / 'Xelatex/sections/01-问题背景与问题重述.tex').read_text()
check('four_questions_retained_in_restatement',
      all('\\textbf{问题' + q + '}' in p1 for q in '一二三四'))

def expand(path, baseline=False, seen=None):
    seen = set() if seen is None else seen
    seen.add(path)
    text = original(path).decode() if baseline else (ROOT / path).read_text()
    text = re.sub(r'(?<!\\)%[^\n]*', '', text)
    def include(m):
        p = Path('Xelatex') / m[1]
        if not p.suffix: p = p.with_suffix('.tex')
        return expand(str(p), baseline, seen)
    return re.sub(r'\\(?:input|include)\{([^}]+)\}', include, text)
active_before = expand(main, True); active = expand(main)
labels = re.findall(r'\\label\{([^}]+)\}', active)
refs = re.findall(r'\\(?:ref|eqref|pageref)\{([^}]+)\}', active)
missing = sorted(set(refs) - set(labels))
check('all_active_references_resolve', not missing, {'missing': missing})
duplicates = sorted({x for x in labels if labels.count(x) > 1})
check('no_duplicate_active_labels', not duplicates, {'duplicates': duplicates})
bib = set(re.findall(r'\\bibitem(?:\[[^]]*\])?\{([^}]+)\}', active))
cited = {k.strip() for group in re.findall(r'\\cite(?:\[[^]]*\])?\{([^}]+)\}', active) for k in group.split(',')}
check('citations_and_bibliography_resolve_both_ways', cited == bib,
      {'missing': sorted(cited - bib), 'uncited': sorted(bib - cited)})
eq = lambda s: re.findall(r'\\begin\{(equation\*?|align\*?|gather\*?)\}(.*?)\\end\{\1\}', s, re.S)
check('all_display_equation_contents_byte_identical', eq(active_before) == eq(active),
      {'display_environments': len(eq(active))})
tab = lambda s: re.findall(r'\\begin\{tabular\}.*?\\end\{tabular\}', s, re.S)
check('all_active_tabular_contents_byte_identical', tab(active_before) == tab(active),
      {'tabular_count': len(tab(active))})
fig = lambda s: re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}', s)
previous_figures = fig(active_before); current_figures = fig(active)
check('all_latest_remote_figure_references_retained',
      current_figures == previous_figures, {'figure_paths': current_figures})
required = []
for f in current_figures:
    p = ROOT / 'Xelatex' / f
    required.append({'path': str(p.relative_to(ROOT)), 'exists': p.exists()})
missing_figure_assets = [x['path'] for x in required if not x['exists']]

report = {'status': 'PASS' if not issues else 'FAIL', 'baseline_commit': BASE,
          'checks': checks, 'issues': issues, 'protected_files': protected,
          'edited_source_sha256': {p: sha((ROOT / p).read_bytes()) for p in sorted(ALLOWED)},
          'analysis_paragraph_hanzi': {k: hanzi(v) for k, v in current.items()},
          'evaluation_body_hanzi': evaluation_count,
          'build_artifact_tracking_exemptions': sorted(BUILD_ARTIFACTS),
          'missing_figure_assets': missing_figure_assets,
          'publication_status': 'BLOCKED_MISSING_REMOTE_FIGURES' if missing_figure_assets else 'AWAITING_PDF_W2',
          'scope': 'Read-only paper source, exact protected-file bytes, unchanged formulas/tables/figures, citations, and requested prose lengths. No solver or training command.'}
(OUT / 'source_verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'status': report['status'], 'checks': len(checks), 'protected_files': len(protected),
                  'evaluation_hanzi': evaluation_count, 'issues': issues}, ensure_ascii=False))
