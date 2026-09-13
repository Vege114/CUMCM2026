"""Independent read-only validation of the two figures supplied by remote main."""
from pathlib import Path
import csv,hashlib,json,re,subprocess
import pdfplumber
from pypdf import PdfReader
ROOT=Path(__file__).resolve().parents[4];OUT=Path(__file__).parent
FIG=ROOT/'Xelatex/final_results/section9_visuals';EVID=ROOT/'reports/experiments/exp008/robustness/evidence'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
rows=lambda p:list(csv.DictReader(p.open(encoding='utf-8-sig')))
q1=rows(EVID/'q1/sensitivity.csv');q2=rows(EVID/'execution/noise_summary.csv')
base=next(r for r in q1 if r['name']=='center');nonbase=[r for r in q1 if r['name']!='center']
issues=[];fonts=[];pdfmeta={};maps=[];maps2=[]
manifest=json.loads((FIG/'sources.json').read_text());hashes=[]
for k,v in manifest['sources'].items():
 p=ROOT/k.replace('\\','/');good=sha(p)==v
 hashes.append({'path':str(p.relative_to(ROOT)),'sha256':sha(p),'matches_remote_manifest':good})
 if not good:issues.append('source_hash:'+k)
for name in ['q1_sensitivity','q2_execution_stress']:
 p=FIG/(name+'.pdf')
 with pdfplumber.open(p) as doc:
  page=doc.pages[0];text=page.extract_text();outside=[c['text'] for c in page.chars if c['x0']<-.1 or c['top']<-.1 or c['x1']>page.width+.1 or c['bottom']>page.height+.1]
  pdfmeta[name]={'pdf_sha256':sha(p),'width_pt':page.width,'height_pt':page.height,'pages':len(doc.pages),'raster_images':len(page.images),'outside_characters':outside,'text':text}
  if outside or len(doc.pages)!=1 or page.images:issues.append(name+':geometry')
  if name=='q1_sensitivity':
   vals=[re.match(r'(.+?)\s+([+-]?\d+\.\d{3}%)\s+([+-]?\d+\.\d{3}%)$',x) for x in text.splitlines()]
   vals=[x.groups() for x in vals if x]
   if len(vals)!=14:issues.append('q1_row_count')
   expected_names=['delta_0p0005','delta_0p0009','delta_0p0011','delta_0p0020','ramp_900','ramp_1100','up_2','up_4','down_1','down_3','eta_rt_0p855','eta_rt_0p945','capacity_0p9','capacity_1p1']
   if [r['name'] for r in nonbase]!=expected_names:issues.append('q1_labels_order')
   for r,(label,a,b) in zip(nonbase,vals):
    raw=[(float(r[k])/float(base[k])-1)*100 for k in ['cost','tv']]
    show=lambda v:f'{v:+.3f}%' if abs(v)>=.0005 else '0.000%'
    expected=[show(v) for v in raw];ok=[a,b]==expected
    if not ok:issues.append('q1_row:'+r['name'])
    maps.append({'name':r['name'],'label':label,'cost_change_raw_pct':raw[0],'tv_change_raw_pct':raw[1],'pdf_cost':a,'pdf_tv':b,'pass':ok})
   if f"{float(base['cost']):.2f}" not in text or f"{float(base['tv']):.2f}" not in text:issues.append('q1_baseline_annotation')
  else:
   vals=re.findall(r'(\d+\.\d{3})%\s*\[(\d+\.\d{3}),\s*(\d+\.\d{3})\]',text)
   if len(vals)!=4:issues.append('q2_row_count')
   for r,shown in zip(q2,vals):
    expect=tuple(f'{float(r[k]):.3f}' for k in ['mean_cost_change_pct','p05_cost_change_pct','p95_cost_change_pct'])
    ok=expect==shown and int(r['replicates'])==30
    if not ok:issues.append('q2_row:'+r['log_noise_std'])
    maps2.append({'noise_std':r['log_noise_std'],'replicates':int(r['replicates']),'pdf_mean':shown[0],'pdf_p05':shown[1],'pdf_p95':shown[2],'pass':ok})
 reader=PdfReader(p)
 for ref in reader.pages[0]['/Resources']['/Font'].values():
  f=ref.get_object();desc=f.get('/FontDescriptor')
  if '/DescendantFonts' in f:desc=f['/DescendantFonts'][0].get_object().get('/FontDescriptor')
  desc=desc.get_object() if desc else {};emb=any(k in desc for k in ['/FontFile','/FontFile2','/FontFile3'])
  fonts.append({'figure':name,'font':str(f.get('/BaseFont')),'embedded':emb,'to_unicode':'/ToUnicode' in f})
  if not emb:issues.append(name+':font_not_embedded')
paths=[FIG/'build_figures.py',FIG/'sources.json',*(FIG/(n+ext) for n in ['q1_sensitivity','q2_execution_stress'] for ext in ['.pdf','.png'])]
for p in paths:
 if p.read_bytes()!=subprocess.check_output(['git','show','88d6b1602e79111e7802e79abdcd3d453b82ee83:'+str(p.relative_to(ROOT))],cwd=ROOT):issues.append('not_remote_bytes:'+str(p.relative_to(ROOT)))
report={'status':'PASS' if not issues else 'FAIL','baseline_commit':'88d6b1602e79111e7802e79abdcd3d453b82ee83','independent_review':True,'files':{str(p.relative_to(ROOT)):sha(p) for p in paths},'source_hashes':hashes,'q1_mapping':maps,'q1_denominator':{'cost':float(base['cost']),'tv':float(base['tv']),'formula':'100*(scenario/center-1)','near_zero_annotation_rule':'absolute change below 0.0005% displayed as 0.000%'},'q2_mapping':maps2,'pdf_metadata':pdfmeta,'fonts':fonts,'method_review':'Generator read only: independent categorical cost/TV bar panels, raw signs retained and negative bars hatched; Q2 point is 30-run mean and segment is empirical p05-p95 relative to archived unperturbed plan. No new solver/training run. Remote09 limits conclusions to single-day one-factor reoptimization or fixed-plan execution replay.','issues':issues}
(OUT/'remote_visuals_verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'status':report['status'],'q1_percent_annotations':2*len(maps),'q2_mean_quantiles':3*len(maps2),'issues':issues},ensure_ascii=False))
