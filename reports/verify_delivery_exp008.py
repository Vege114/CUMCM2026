"""Check the final exp008 report package, frozen evidence and selected workbook bindings."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import unquote
from zipfile import ZipFile

from experiments.exp008.report_payload import enforce_final

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'reports/experiments/exp008'


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def verify(require_registry=True):
    errors=[];checks={}
    def check(name, ok, detail=None):
        checks[name]={'passed':bool(ok),'detail':detail}
        if not ok:errors.append(name)
    p=read(REPORT/'final_payload.json')
    enforce_final(p)
    check('accepted_gate_preserves_failed_original_target',p['finalization_basis']['user_accepted_current_verified_result'] and not p['finalization_basis']['original_eight_percent_target_met'])
    check('selected_q2_sha',p['scenarios']['2']['archive']['sha256']=='8b97181d2f82659774c609632d39a2bd812e69598fcd37696a2d793debcfe6d7')
    check('selected_q2_cost',abs(p['scenarios']['2']['fees']['total_cost_yuan']-13201981.94794217)<1e-6)
    check('selected_q2_reversals',p['scenarios']['2']['battery']['direction_reversals']==2533)
    for name, item in p['supporting_evidence'].items():
        check('support_'+name,sha(Path(item['path']))==item['sha256'])
    check('payload_builder_source',sha(ROOT/'experiments/exp008/report_payload.py')==p['source_code']['sha256'])
    for name in ['evidence/final_selection_audit.json','evidence/battery/independent_audit.json',
                 'evidence/forecast/verification.json','evidence/forecast/independent_report_audit.json',
                 'evidence/workbooks/saved_workbook_audit.json','evidence/workbooks/source_to_payload_audit.json',
                 'evidence/report_browser/browser_checks.json']:
        obj=read(REPORT/name)
        passed=obj.get('passed',obj.get('all_passed'))
        if name=='evidence/final_selection_audit.json':
            passed=all(c['verification']['passed'] for c in obj['scenarios'].values())
        check('audit_'+name,passed is True,{'top_level_keys':list(obj)})
    missing=[]
    for filename in ['report.md','specified_dates.md','README.md']:
        body=(REPORT/filename).read_text()
        for match in re.finditer(r'!?\[[^\]]*\]\(([^)]+)\)',body):
            dest=match[1].split('#')[0]
            if not dest or re.match(r'^[a-z]+:',dest):continue
            target=REPORT/unquote(dest)
            if not target.is_file() and not (not require_registry and target.name in {'record.json','exp008.json','data_hashes.json'}):
                missing.append({'source':filename,'target':dest})
    check('all_main_report_links',not missing,missing)
    body=(REPORT/'report.md').read_text()
    check('exactly_eight_template_sections',len(re.findall(r'(?m)^## ',body))==8)
    check('no_unfilled_placeholders','{{' not in body)
    check('all_history_experiments',all(f'exp{i:03d}' in body for i in range(1,7)))
    check('all_four_question_history_facets',(REPORT/'figures/report/all_questions_history_cost.svg').is_file())
    check('specified_daily_totals',(REPORT/'specified_dates.md').read_text().count('全天总购电量')==17)
    figures=[]
    for png in (REPORT/'figures').rglob('*.png'):
        svg=png.with_suffix('.svg');ok=svg.is_file()
        if ok:
            try:ET.parse(svg)
            except ET.ParseError:ok=False
        figures.append({'png':str(png.relative_to(REPORT)),'paired_svg_valid':ok})
    check('all_figure_png_svg_pairs',all(r['paired_svg_valid'] for r in figures),{'count':len(figures)})
    for s in ['1','2','3','4-2','4-3']:
        wb=REPORT/f'result{s}.xlsx'
        with ZipFile(wb) as z:
            check('xlsx_zip_'+s,z.testzip() is None)
    build=read(REPORT/'report_build.json')
    check('report_hash_bound',build['report_sha256']==sha(REPORT/'report.md'))
    check('html_hash_bound',build['html_sha256']==sha(REPORT/'report.html'))
    check('payload_hash_bound',build['payload_sha256']==sha(REPORT/'final_payload.json'))
    tests=(REPORT/'tests.txt').read_text()
    check('unit_tests_passed',bool(re.search(r'Ran \d+ tests? in ',tests)) and re.search(r'(?m)^OK$',tests) is not None)
    if require_registry:
        record=read(ROOT/'reports/registry/exp008.json')
        check('registry_and_report_commit',record['code_commit']==build['code_commit'])
        for file,expected in record['source_sha256'].items():
            actual=subprocess.check_output(['git','show',f'{record["code_commit"]}:{file}'],cwd=ROOT)
            check('committed_source_'+file,hashlib.sha256(actual).hexdigest()==expected)
    out={'passed':not errors,'errors':errors,'checks':checks,'figures':figures,
         'report_sha256':sha(REPORT/'report.md'),'html_sha256':sha(REPORT/'report.html'),
         'payload_sha256':sha(REPORT/'final_payload.json'),
         'artifacts':{f'result{s}.xlsx':sha(REPORT/f'result{s}.xlsx') for s in ['1','2','3','4-2','4-3']},
         'new_training_or_optimization_executed':False}
    (REPORT/'evidence/delivery_verification.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'passed':out['passed'],'errors':errors,'checks':len(checks),'figure_pairs':len(figures)},ensure_ascii=False,indent=2))
    if errors:raise SystemExit(1)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--pre-register',action='store_true')
    verify(not parser.parse_args().pre_register)
