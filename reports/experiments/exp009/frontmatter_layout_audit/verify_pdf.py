"""Read-only page and equation-placement audit; requires pdfplumber."""
from pathlib import Path
import hashlib, json, re, sys
import pdfplumber

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).parent
PDF = ROOT / (sys.argv[1] if len(sys.argv) > 1 else 'Xelatex/数模通用模板.pdf')
meta = json.loads((OUT / 'baseline.json').read_text())
reference = meta.get('early_reference', meta)
BASE = ROOT / reference['baseline_pdf']
missing = [x for x in meta.get('required_figures', []) if not (ROOT / x).exists()]
if missing:
    raise SystemExit('W2 pending: missing current remote figure assets: ' + ', '.join(missing))
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
issues, pages, found = [], [], []

def expand(path):
    s = re.sub(r'(?<!\\)%[^\n]*', '', path.read_text())
    def include(m):
        p = ROOT / 'Xelatex' / m[1]
        if not p.suffix: p = p.with_suffix('.tex')
        return expand(p)
    return re.sub(r'\\(?:input|include)\{([^}]+)\}', include, s)
active = expand(ROOT / 'Xelatex/数模通用模板.tex')
labels = []
for m in re.finditer(r'\\begin\{equation\}(.*?)\\end\{equation\}', active, re.S):
    lab = re.search(r'\\label\{([^}]+)\}', m[1]); labels.append(lab[1] if lab else None)
numlabels = {i: lab for i, lab in enumerate(labels, 1)
             if lab and lab.startswith(('eq:q2-', 'eq:q3-', 'eq:q4-'))}
assert len(numlabels) == 6, numlabels
def characters(p):
    return [(c['text'], round(c['x0'], 2), round(c['top'], 2),
             round(c['x1'], 2), round(c['bottom'], 2)) for c in p.chars]

titles = ['问题背景与问题重述', '问题分析', '模型假设', '符号说明',
          '问题一的模型建立与求解', '问题二的模型建立与求解',
          '问题三的模型建立与求解', '问题四的模型建立与求解',
          '模型分析与检验', '模型的评价、改进与推广', 'AI工具使用声明',
          '附录A文件列表']
alltext = []
with pdfplumber.open(PDF) as pdf:
    current_abstract = characters(pdf.pages[0])
    for pno, p in enumerate(pdf.pages, 1):
        lines = p.extract_text_lines(); text = p.extract_text() or ''
        outside = [{'text': c['text'], 'bbox': [c['x0'], c['top'], c['x1'], c['bottom']]}
                   for c in p.chars if c['x0'] < -.1 or c['top'] < -.1
                   or c['x1'] > p.width + .1 or c['bottom'] > p.height + .1]
        if outside: issues.append(f'page {pno}: outside-page characters')
        if not text.strip(): issues.append(f'page {pno}: blank page')
        if '\ufffd' in text: issues.append(f'page {pno}: replacement character')
        if '??' in text: issues.append(f'page {pno}: unresolved reference marker')
        entries = []
        for i, line in enumerate(lines):
            m = re.search(r'\((\d{1,2})\)\s*$', line['text'])
            if m and line['x1'] >= 500 and int(m[1]) in numlabels:
                entries.append((i, int(m[1])))
        for k, (idx, num) in enumerate(entries):
            end = entries[k + 1][0] if k + 1 < len(entries) else len(lines)
            explanation = any(x['text'].lstrip().startswith('式中') for x in lines[idx + 1:end])
            if not explanation: issues.append(numlabels[num] + ': explanation first line on another page')
            found.append({'number': num, 'label': numlabels[num], 'page': pno,
                          'explanation_same_page': explanation})
        compact = re.sub(r'\s+', '', text)
        pages.append({'page': pno, 'characters': len(p.chars), 'outside_characters': outside,
                      'headings': [t for t in titles if t in compact],
                      'captions': re.findall(r'^(?:表|图)\s*\d+[^\n]*', text, re.M),
                      'start': text[:160]})
        alltext.append(text); p.close()
    total = len(pdf.pages)
if sorted(x['number'] for x in found) != sorted(numlabels):
    issues.append('retained equation missing or duplicate')
if not BASE.exists(): raise FileNotFoundError('Recorded comparison baseline PDF is required: ' + str(BASE))
if sha(BASE) != reference['pdf_sha256']: issues.append('baseline SHA mismatch')
with pdfplumber.open(BASE) as baseline:
    abstract_same = characters(baseline.pages[0]) == current_abstract
    if not abstract_same: issues.append('unchanged abstract character geometry changed')
appendix = next(x['page'] for x in pages if '附录A文件列表' in x['headings'])
evaluation = (ROOT / 'Xelatex/sections/10-模型评价改进与推广.tex').read_text().split('\\section{模型的评价、改进与推广}', 1)[1]
only_hanzi = lambda s: ''.join(re.findall(r'[\u4e00-\u9fff]', s))
evaluation_found = only_hanzi(evaluation) in only_hanzi('\n'.join(alltext))
if not evaluation_found: issues.append('compressed evaluation paragraph not fully present in PDF')
report = {'status': 'PASS' if not issues else 'FAIL', 'pdf': str(PDF.relative_to(ROOT)),
          'pdf_sha256': sha(PDF), 'pdf_bytes': PDF.stat().st_size,
          'baseline_commit': meta['baseline_commit'], 'comparison_reference_pdf_sha256': reference['pdf_sha256'],
          'exact_latest_baseline_page_comparison': 'early_reference' not in meta,
          'early_reference_total_pages': reference['total_pages'], 'early_reference_body_pages': reference['body_pages'],
          'total_pages': total, 'body_pages': appendix - 2, 'appendix_start_page': appendix,
          'net_page_reduction': None if 'early_reference' in meta else meta['total_pages'] - total,
          'abstract_characters_and_geometry_unchanged': abstract_same,
          'evaluation_full_text_present': evaluation_found,
          'retained_equations': found, 'pages': pages, 'issues': issues}
(OUT / 'pdf_verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({k: report[k] for k in ['status', 'pdf_sha256', 'total_pages', 'body_pages',
                                      'appendix_start_page', 'net_page_reduction', 'issues']}, ensure_ascii=False))
