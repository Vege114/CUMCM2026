"""Read-only PDF pagination/boundary audit; run with bundled Python (pdfplumber)."""
import hashlib
import json
from pathlib import Path
import re
import pdfplumber

ROOT=Path(__file__).resolve().parents[4]
OUT=Path(__file__).parent
PDF=ROOT/'完整论文.pdf'
BASE=ROOT/'tmp/conflict-resolution/notation-layout/before.pdf'
SECTIONS=['06-问题二.tex','07-问题三.tex','08-问题四.tex']
LABELS=[]
for path in SECTIONS:
    text=(ROOT/'Xelatex/sections'/path).read_text()
    LABELS+=re.findall(r'\\begin\{equation\}\\label\{([^}]+)\}',text)
# Equations 1-9 precede Q2; 10-48 are the 39 unchanged Q2-Q4 displays.
assert len(LABELS)==39
numlabels=dict(zip(range(10,49),LABELS))
EXEMPT={'eq:q3-adjust-linear','eq:q3-feedback-discharge'}
issues=[]; pages=[]; eqs=[]; textout=[]; affected=[]
with pdfplumber.open(PDF) as pdf:
    for pno,p in enumerate(pdf.pages,1):
        lines=p.extract_text_lines()
        text=p.extract_text() or ''
        invalid=[{'char':c['text'],'box':[c['x0'],c['top'],c['x1'],c['bottom']]} for c in p.chars if c['x0']<-.1 or c['top']<-.1 or c['x1']>p.width+.1 or c['bottom']>p.height+.1]
        if invalid:issues.append(f'page {pno}: out-of-page characters')
        if not text.strip():issues.append(f'page {pno}: empty')
        if '\ufffd' in text:issues.append(f'page {pno}: replacement character')
        if '??' in text:issues.append(f'page {pno}: unresolved reference marker')
        entries=[]
        for i,line in enumerate(lines):
            m=re.search(r'\((\d{1,2})\)\s*$',line['text'])
            if m and line['x1']>=500 and int(m[1]) in numlabels:
                n=int(m[1]); entries.append((i,n,line))
        for k,(i,n,line) in enumerate(entries):
            label=numlabels[n]
            end=entries[k+1][0] if k+1<len(entries) else len(lines)
            following=[x for x in lines[i+1:end] if x['text'].lstrip().startswith('式中')]
            no_new=label in EXEMPT
            if not following and not no_new:issues.append(f'{label}: explanation first line on another page')
            eqs.append({'number':n,'label':label,'page':pno,'equation_number_top':round(line['top'],3),'no_new_symbols':no_new,'explanation_on_same_page':bool(following),'explanation_top':round(following[0]['top'],3) if following else None})
        info={'page':pno,'characters':len(p.chars),'outside_characters':invalid,'text_start':text[:140],'table_captions':re.findall(r'^表\s*\d+[^\n]*',text,re.M)}
        pages.append(info);textout.append(f'=== PAGE {pno} ===\n{text}')
        p.close()
    total=len(pdf.pages)
baseline=json.loads((OUT/'baseline.json').read_text())
beforepages=baseline['total_pages']
baseline_sha=baseline['pdf_sha256']
baseline_mode='fixed audited metadata; ignored baseline PDF is not present'
if BASE.exists():
    with pdfplumber.open(BASE) as pdf:actual_beforepages=len(pdf.pages)
    actual_before_sha=hashlib.sha256(BASE.read_bytes()).hexdigest()
    if actual_beforepages!=beforepages or actual_before_sha!=baseline_sha:issues.append('local baseline PDF does not match fixed audited metadata')
    baseline_mode='local baseline PDF independently read and SHA256 verified against fixed audited metadata'
nums=[e['number'] for e in eqs]
if sorted(nums)!=list(range(10,49)):issues.append('equation number sequence missing/duplicated')
report={'status':'PASS' if not issues else 'FAIL','pdf':str(PDF.relative_to(ROOT)),'sha256':hashlib.sha256(PDF.read_bytes()).hexdigest(),'baseline_pdf_sha256':baseline_sha,'baseline_evidence_mode':baseline_mode,'baseline_pages':beforepages,'current_pages':total,'page_change':total-beforepages,'scope':'All-page text/boundary scan plus 39 unchanged equation numbers and same-page first explanation; actual rendering recorded separately.','equations':eqs,'pages':pages,'issues':issues}
(OUT/'pdf_verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
tmp=ROOT/'tmp/conflict-resolution/notation-layout/audit'
tmp.mkdir(parents=True,exist_ok=True)
(tmp/'final_pdf.txt').write_text('\n\n'.join(textout))
print(json.dumps({'status':report['status'],'pages':total,'before':beforepages,'equations':len(eqs),'issues':issues,'sha256':report['sha256']},ensure_ascii=False))
