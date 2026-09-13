"""Final artifact integrity checks; never run a forecasting model or planner."""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from urllib.parse import unquote
import numpy as np
from experiments.exp009.report_payload import ROOT, OUT, read, save, sha

def run(require_committed_source=False):
 p=read(OUT/'final_payload.json');record=read(OUT/'record.json');errors=[]
 def require(value,description):
  if not value:errors.append(description)
 require(p['protocol_sha256']==sha(ROOT/'experiments/exp009/protocol.json'),'protocol byte identity')
 require(read(OUT/'evidence/battery/final_evidence.json')['complete'],'complete independent five-question evidence')
 for name in ('independent_cell_audit','saved_workbook_audit','visual_review'):
  audit=read(OUT/f'evidence/workbooks/{name}.json')
  require(audit['passed'],name+' workbook QA')
  require(audit.get('final_payload_sha256',audit.get('payload_sha256'))==sha(OUT/'final_payload.json'),name+' current payload binding')
 for s,c in [('1',p['q1']),*p['scenarios'].items()]:
  require(c['verification']['passed'],s+' physical/billing verification')
  require(sha(ROOT/c['archive']['path'])==c['archive']['sha256'],s+' frozen archive')
  require((OUT/f'result{s}.xlsx').is_file(),s+' workbook')
 for path,digest in p['data_hashes'].items():require(sha(ROOT/path)==digest,path+' data bytes')
 for path,digest in record['source_sha256'].items():
  require(sha(ROOT/path)==digest,path+' current source SHA')
  if require_committed_source:
   try:raw=subprocess.check_output(['git','show',record['code_commit']+':'+path],cwd=ROOT,stderr=subprocess.DEVNULL)
   except subprocess.CalledProcessError:errors.append(path+' missing at source commit');continue
   require(hashlib.sha256(raw).hexdigest()==digest,path+' source commit bytes')
 targets=[]
 for name in ('report.md','specified_dates.md','methods.md'):
  source=OUT/name;text=source.read_text()
  require('{{' not in text,name+' unresolved placeholder')
  for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',text):
   if target.startswith(('http','mailto:','#')):continue
   path=unquote(target.split('#')[0]).strip('<>');resolved=(source.parent/path).resolve()
   exists=resolved.exists() or resolved==OUT/'evidence/delivery_qa.json'
   targets.append({'source':name,'target':target,'exists':exists})
   require(exists,name+' missing link '+target)
 require(len(re.findall(r'(?m)^## \d+\.',(OUT/'report.md').read_text()))==8,'template eight sections')
 browser=read(OUT/'evidence/report_browser/browser_checks.json');require(browser['passed'],'report browser runtime/layout')
 for r in p['comparison']:
  require(abs(r['absolute_change_yuan']-(r['current']-r['previous']))<1e-8,'fee comparison delta '+r['scenario'])
  require(abs(r['relative_change_pct']-100*(r['current']-r['previous'])/abs(r['previous']))<1e-10,'fee comparison percentage '+r['scenario'])
 oldpaths=['data/results/exp008','experiments/exp008','reports/experiments/exp008','reports/registry/exp008.json']
 diff=subprocess.check_output(['git','diff',p['protocol']['base_commit'],'--',*oldpaths],cwd=ROOT)
 require(not diff,'exp008 tracked source and evidence unchanged')
 files=[f for f in OUT.rglob('*') if f.is_file() and f.name!='delivery_qa.json']
 result={'passed':not errors,'errors':errors,'source_commit_verified':require_committed_source,
  'code_commit':record['code_commit'],'checks':['all five archives and actual physics/billing','raw/source byte SHA256','all local Markdown links','eight main template sections','offline desktop/mobile browser','comparison arithmetic','exp008 tracked preservation','five workbooks independent cell/live formula/visual QA bound to current payload'],
  'local_links':targets,'file_count':len(files),'artifact_bytes':sum(f.stat().st_size for f in files),
  'source_sha256':sha(Path(__file__)),'report_sha256':sha(OUT/'report.md'),'html_sha256':sha(OUT/'report.html'),
  'workbooks':{s:sha(OUT/f'result{s}.xlsx') for s in ['1','2','3','4-2','4-3']}}
 save(OUT/'evidence/delivery_qa.json',result)
 print(json.dumps({'passed':result['passed'],'errors':errors,'files':len(files),'links':len(targets)},ensure_ascii=False))
 if errors:raise SystemExit(1)

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--require-committed-source',action='store_true');args=a.parse_args();run(args.require_committed_source)
