"""Read-only final PDF pagination and protected-prefix audit; requires pdfplumber."""
from pathlib import Path
import hashlib,json,re,sys
import pdfplumber
ROOT=Path(__file__).resolve().parents[4]
OUT=Path(__file__).parent
PDF=(ROOT/(sys.argv[1] if len(sys.argv)>1 else 'Xelatex/build/数模通用模板.pdf')).resolve()
meta=json.loads((OUT/'baseline.json').read_text());BASE=ROOT/meta['baseline_pdf']
issues=[];pages=[];found=[];textout=[]
# Expand active inputs to derive display equation numbers, rather than assume
# the preceding chapters or retained equation count stay fixed forever.
seen=set()
def expand(path):
 s=re.sub(r'(?<!\\)%[^\n]*','',path.read_text())
 def inc(m):
  p=ROOT/'Xelatex'/m[1]
  if not p.suffix:p=p.with_suffix('.tex')
  return expand(p)
 return re.sub(r'\\(?:input|include)\{([^}]+)\}',inc,s)
active=expand(ROOT/'Xelatex/数模通用模板.tex')
labels=[]
for m in re.finditer(r'\\begin\{equation\}(.*?)\\end\{equation\}',active,re.S):
 lab=re.search(r'\\label\{([^}]+)\}',m[1]);labels.append(lab[1] if lab else None)
numlabels={i:lab for i,lab in enumerate(labels,1) if lab and lab.startswith(('eq:q2-','eq:q3-','eq:q4-'))}
assert len(numlabels)==6,numlabels

def protected_prefix(pdf):
 prefix=[];q2=None
 for pno,p in enumerate(pdf.pages,1):
  lines=p.extract_text_lines()
  headings=[l for l in lines if '问题二的模型建立与求解' in re.sub(r'\s+','',l['text'])]
  cutoff=headings[0]['top']-.1 if headings else float('inf')
  chars=[(c['text'],round(c['x0'],2),round(c['top'],2),round(c['x1'],2),round(c['bottom'],2)) for c in p.chars if c['top']<cutoff]
  prefix.append(chars)
  if headings:q2={'page':pno,'heading_top':round(headings[0]['top'],2)};break
  p.close()
 return prefix,q2

with pdfplumber.open(PDF) as pdf:
 for pno,p in enumerate(pdf.pages,1):
  lines=p.extract_text_lines();text=p.extract_text() or ''
  bad=[{'char':c['text'],'bbox':[c['x0'],c['top'],c['x1'],c['bottom']]} for c in p.chars if c['x0']<-.1 or c['top']<-.1 or c['x1']>p.width+.1 or c['bottom']>p.height+.1]
  if bad:issues.append(f'page {pno}: out-of-page glyphs')
  if not text.strip():issues.append(f'page {pno}: blank')
  if '\ufffd' in text:issues.append(f'page {pno}: replacement marker')
  if '??' in text:issues.append(f'page {pno}: unresolved reference marker')
  entries=[]
  for i,l in enumerate(lines):
   m=re.search(r'\((\d{1,2})\)\s*$',l['text'])
   if m and l['x1']>=500 and int(m[1]) in numlabels:entries.append((i,int(m[1]),l))
  for k,(idx,num,l) in enumerate(entries):
   end=entries[k+1][0] if k+1<len(entries) else len(lines)
   explanation=next((n for n in lines[idx+1:end] if n['text'].lstrip().startswith('式中')),None)
   if explanation is None:issues.append(f'{numlabels[num]}: first explanation line separated')
   found.append({'number':num,'label':numlabels[num],'page':pno,'explanation_same_page':explanation is not None})
  compact=re.sub(r'\s+','',text)
  headings=[title for title in ['问题二的模型建立与求解','问题三的模型建立与求解','问题四的模型建立与求解','模型分析与检验','模型的评价、改进与推广','AI工具使用声明','附录A文件列表','附录B完整求解程序'] if title in compact]
  pages.append({'page':pno,'characters':len(p.chars),'outside_characters':bad,'headings':headings,'start':text[:180],'captions':re.findall(r'^(?:表|图)\s*\d+[^\n]*',text,re.M)})
  textout.append(f'=== PAGE {pno} ===\n{text}');p.close()
 total=len(pdf.pages)
if sorted(x['number'] for x in found)!=sorted(numlabels):issues.append('retained equation sequence missing or duplicate')
if not BASE.exists():raise FileNotFoundError('Comparable baseline PDF required for current protected-prefix audit: '+str(BASE))
if hashlib.sha256(BASE.read_bytes()).hexdigest()!=meta['pdf_sha256']:issues.append('baseline PDF SHA mismatch')
with pdfplumber.open(BASE) as old, pdfplumber.open(PDF) as new:
 op,oq=protected_prefix(old);np,nq=protected_prefix(new)
 prefix_pass=op==np and oq==nq
 if not prefix_pass:issues.append('Q1 and earlier rendered character geometry changed')
 baseline_count=len(old.pages)
appendix=next(x['page'] for x in pages if '附录A文件列表' in x['headings'])
report={'status':'PASS' if not issues else 'FAIL','pdf':str(PDF.relative_to(ROOT)),'pdf_sha256':hashlib.sha256(PDF.read_bytes()).hexdigest(),'pdf_bytes':PDF.stat().st_size,'baseline_commit':meta['baseline_commit'],'baseline_pdf_sha256':meta['pdf_sha256'],'baseline_pages':baseline_count,'total_pages':total,'net_page_reduction':baseline_count-total,'appendix_start_page':appendix,'body_pages':appendix-2,'protected_prefix':{'pass':prefix_pass,'baseline_q2_heading':oq,'current_q2_heading':nq,'pages_or_partial_pages_compared':len(op),'comparison':'Every preceding character and its rounded 0.01pt geometry; final shared page clipped above Q2 heading.'},'retained_equations':found,'pages':pages,'issues':issues}
(OUT/'pdf_verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
tmp=ROOT/'tmp/conflict-resolution/q234-compression/audit';tmp.mkdir(parents=True,exist_ok=True);(tmp/'current_pdf.txt').write_text('\n\n'.join(textout))
print(json.dumps({k:report[k] for k in ['status','total_pages','body_pages','appendix_start_page','net_page_reduction','protected_prefix','issues','pdf_sha256']},ensure_ascii=False))
