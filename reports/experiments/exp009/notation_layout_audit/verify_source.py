"""Read-only audit of the notation/layout revision against c90ccb1a and frozen trajectories.

No solver, forecast fitting, or production-asset generator is called.
Only writes this audit's JSON report beside the script.
"""
import collections
import hashlib
import json
from pathlib import Path
import re
import subprocess
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
BASE = 'c90ccb1ad8384e22c792b0a9241c1d9e8901738f'
OUT = Path(__file__).parent
SECTIONS = [f'Xelatex/sections/{x}' for x in ('06-问题二.tex','07-问题三.tex','08-问题四.tex')]
TABLES = {'q2':'Xelatex/final_results/q2/tables.tex', 'q3':'Xelatex/final_results/q3_specified.tex', 'q4_4-2':'Xelatex/final_results/q4_4-2_specified.tex', 'q4_4-3':'Xelatex/final_results/q4_4-3_specified.tex'}
DATES = ['20250320','20250621','20250923','20251221']
DISPLAY = ['3月20日','6月21日','9月23日','12月21日']
DAYS = [78,171,265,354]
SLOTS = [60,72,84,96,108,120]
TIME = re.compile(r'^\d{2}:\d{2}--\d{2}:\d{2}$')
NUM = re.compile(r'-?\d+\.\d+')
checks = []
issues = []

def check(name, ok, detail=None):
    checks.append({'name':name,'pass':bool(ok), **({'detail':detail} if detail is not None else {})})
    if not ok: issues.append(name)

def prior(path): return subprocess.check_output(['git','show',f'{BASE}:{path}'],cwd=ROOT,text=True)
def now(path): return (ROOT/path).read_text()
def sha(s): return hashlib.sha256(s.encode()).hexdigest()

def tables(text):
    labels=list(re.finditer(r'\\label\{([^}]+)\}', text)); out={}
    for i,m in enumerate(labels):
        tail=text[m.end():labels[i+1].start() if i+1<len(labels) else len(text)]
        t=re.search(r'\\begin\{tabular\}\{([^}]+)\}(.*?)\\end\{tabular\}',tail,re.S)
        if t:
            captions=re.findall(r'\\caption\{([^{}]*)\}',text[:m.start()])
            out[m.group(1)]={'spec':t.group(1),'body':t.group(2),'caption':captions[-1] if captions else ''}
    return out

def rows(table):
    s=table['body']
    s=re.sub(r'\\(?:toprule|midrule|bottomrule)','',s)
    s=re.sub(r'\\multicolumn\{\d+\}\{[^}]+\}\{([^{}]*)\}',r'\1',s)
    return [[c.strip() for c in row.strip().split('&')] for row in s.split('\\\\') if row.strip()]

def canonical(text,kind):
    ts=tables(text); out={}
    def put(*args):
        *key,val=args; key='|'.join(str(k) for k in key)
        assert key not in out, key
        out[key]=val
    # Each stored value is the printed decimal string, so comparisons preserve rounding.
    if kind=='q2':
        if 'tab:q2-purchase-dates' in ts:
            rs=rows(ts['tab:q2-purchase-dates'])
            assert rs[0][1:]==DISPLAY
            for row in rs[1:]:
                metric=row[0]
                if metric.startswith('全天计划'): metric='total'
                elif metric.startswith('全天购电费'): metric='cost'
                for date,v in zip(DATES,row[1:]):put(date,'purchase',metric,v)
        else:
            for date in DATES:
                rs=rows(ts['tab:q2-purchase-'+date])
                for row in rs:
                    for i,c in enumerate(row):
                        if TIME.match(c):put(date,'purchase',c,row[i+1])
                        elif '全天计划购电量：' in c:put(date,'purchase','total',NUM.search(c).group())
                        elif '全天购电费：' in c:put(date,'purchase','cost',NUM.search(c).group())
        for date in DATES:
            rs=rows(ts['tab:q2-battery-'+date])
            for row in rs:
                for i,c in enumerate(row):
                    if TIME.match(c):
                        put(date,'battery',c,'charge',row[i+1]);put(date,'battery',c,'discharge',row[i+2])
                    elif c.startswith(('0:00储电量','24:00储电量')):
                        value=NUM.search(c).group() if NUM.search(c) else row[i+1]
                        put(date,'battery',c.split('：')[0],value)
    else:
        for ptype in ['original','final']:
            label=f'tab:{kind}-{ptype}'
            if label not in ts:continue
            rs=rows(ts[label]);assert rs[0][1:]==DISPLAY
            for row in rs[1:]:
                metric={'全天电量':'total','全天总费用/元':'cost'}.get(row[0],row[0])
                for date,v in zip(DATES,row[1:]):put(date,ptype,metric,v)
        for date,idx in zip(DATES,[47,140,234,323]):
            rs=rows(ts[f'tab:{kind}-storage-{idx}']);assert rs[0]==['时间段','充电量','放电量']
            for row in rs[1:]:
                if TIME.match(row[0]):
                    put(date,'battery',row[0],'charge',row[1]);put(date,'battery',row[0],'discharge',row[2])
                else:put(date,'battery',row[0],row[1])
    # Emergency tables may be one four-date table or two two-date tables.
    for label,t in ts.items():
        if 'emergency' not in label:continue
        rs=rows(t); dates=[DATES[DISPLAY.index(c)] for c in rs[0]]
        for row in rs[2:]:
            for i,date in enumerate(dates):
                interval,value=row[2*i:2*i+2]
                if interval=='--': assert value=='--';continue
                put(date,'emergency',interval,value)
    return out,ts

sources={}
for p in SECTIONS:
    a,b=prior(p),now(p)
    eq=lambda t:re.findall(r'\\begin\{equation\}(.*?)\\end\{equation\}',t,re.S)
    ea,eb=eq(a),eq(b)
    check(p+':equations_byte_identical',ea==eb,{'before_count':len(ea),'after_count':len(eb)})
    below=[]
    no_new_symbols={
        'eq:q3-adjust-linear':'q, g0 and a+/a- were defined below eq:q3-adjustment; omitted day/release indices are stated before eq:q3-piecewise.',
        'eq:q3-feedback-discharge':'All symbols and actual/scenario distinctions were defined immediately below eq:q3-feedback-charge.',
    }
    for m in re.finditer(r'\\begin\{equation\}(.*?)\\end\{equation\}',b,re.S):
        label=re.search(r'\\label\{([^}]+)\}',m.group(1)).group(1)
        after=b[m.end():].lstrip()
        below.append({'label':label,'immediate_explanation':after.startswith('式中'),'no_new_symbols':no_new_symbols.get(label)})
    check(p+':first_use_equations_have_immediate_explanation',all(x['immediate_explanation'] or x['no_new_symbols'] for x in below),below)
    sources[p]={'before_sha256':sha(a),'after_sha256':sha(b)}

canonical_data={}
for kind,p in TABLES.items():
    old,oldts=canonical(prior(p),kind);new,newts=canonical(now(p),kind)
    check(p+':semantic_value_mapping',old==new,{'printed_values':len(old),'missing':sorted(set(old)-set(new)),'extra':sorted(set(new)-set(old)),'changed':[k for k in old.keys()&new.keys() if old[k]!=new[k]]})
    if kind=='q2':
        for label in ['tab:q2-annual','tab:q2-prediction']:
            check(label+':unchanged_rows',rows(oldts[label])==rows(newts[label]))
    canonical_data[kind]=new
    captions=[]
    for label,t in newts.items():
        if 'battery-' in label:
            date=label.rsplit('-',1)[-1]
        elif 'storage-' in label:
            date=DATES[[47,140,234,323].index(int(label.rsplit('-',1)[-1]))]
        else:continue
        match=re.search(r'(\d{1,2})月(\d{1,2})日',t['caption'])
        ok=match is not None and (int(match[1]),int(match[2]))==(int(date[4:6]),int(date[6:8]))
        captions.append({'label':label,'caption':t['caption'],'pass':ok,'unit_kWh':'kWh' in t['caption']})
    check(kind+':storage_caption_dates_and_units',all(c['pass'] and c['unit_kWh'] for c in captions),captions)
    sources[p]={'before_sha256':sha(prior(p)),'after_sha256':sha(now(p))}

# Recompute only the displayed aggregates from accepted immutable archives.
archives={'q2':'q2/dispatch.npz','q3':'q3/dispatch_365.npz','q4_4-2':'q4_2/dispatch.npz','q4_4-3':'q4_3/dispatch_365.npz'}
for kind,p in archives.items():
    path=ROOT/'data/results/exp009'/p
    z=np.load(path)
    expected={}
    def put(*args):
        *key,value=args; expected['|'.join(str(k) for k in key)]=value
    for date,day in zip(DATES,DAYS):
        types=[('purchase','original')] if kind=='q2' else [('final','final')]+([('original','original')] if kind!='q4_4-2' else [])
        for pt,field in types:
            for slot in SLOTS:
                interval=f'{slot//6:02d}:{slot%6*10:02d}--{(slot+1)//6:02d}:{(slot+1)%6*10:02d}'
                put(date,pt,interval,f'{z[field][day,slot]:.4f}')
            put(date,pt,'total',f'{z[field][day].sum():.4f}')
            if pt!='original':put(date,pt,'cost',f'{z["fees"][day].sum():.2f}')
        for i in range(6):
            interval=f'{i*4:02d}:00--{(i+1)*4:02d}:00'
            for field in ['charge','discharge']:put(date,'battery',interval,field,f'{z[field][day,i*24:(i+1)*24].sum():.4f}')
        put(date,'battery','0:00储电量',f'{z["states"][day,0]:.4f}')
        put(date,'battery','24:00储电量',f'{z["states"][day,-1]:.4f}')
        vals=z['emergency'][day]; start=None
        for i in range(145):
            on=i<144 and vals[i]>1e-7
            if on and start is None:start=i
            if not on and start is not None:
                interval=f'{start//6:02d}:{start%6*10:02d}--{i//6:02d}:{i%6*10:02d}'
                put(date,'emergency',interval,f'{vals[start:i].sum():.4f}');start=None
        if kind!='q2':put(date,'emergency','合计',f'{vals.sum():.4f}')
    actual=canonical_data[kind]
    check(kind+':frozen_trajectory_aggregates',expected==actual,{'checked_values':len(expected),'mismatches':[{ 'key':k,'expected':expected.get(k),'actual':actual.get(k)} for k in expected.keys()|actual.keys() if expected.get(k)!=actual.get(k)],'archive_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})

# Traverse only active manuscript inputs. Listing source code is not another TeX input.
seen={}
def walk(path):
    path=path.resolve()
    if path in seen:return
    s=re.sub(r'(?<!\\)%[^\n]*','',path.read_text());seen[path]=s
    for inc in re.findall(r'\\(?:input|include)\{([^}]+)\}',s):
        ip=ROOT/'Xelatex'/inc
        if not ip.suffix:ip=ip.with_suffix('.tex')
        assert ip.exists(),ip
        walk(ip)
walk(ROOT/'Xelatex/数模通用模板.tex')
alltext='\n'.join(seen.values())
labels=re.findall(r'\\label\{([^}]+)\}',alltext)
refs=re.findall(r'\\(?:ref|eqref|autoref)\{([^}]+)\}',alltext)
check('active_manuscript:references_resolve',set(refs)<=set(labels),{'missing':sorted(set(refs)-set(labels)),'labels':len(labels),'references':len(refs)})
check('active_manuscript:no_duplicate_labels',len(labels)==len(set(labels)),{'duplicates':[k for k,v in collections.Counter(labels).items() if v>1]})
bib=set(re.findall(r'\\bibitem\{([^}]+)\}',alltext))
citations={x.strip() for c in re.findall(r'\\cite(?:\[[^]]*\])?\{([^}]+)\}',alltext) for x in c.split(',')}
check('active_manuscript:citations_resolve',citations<=bib,{'missing':sorted(citations-bib),'bibliography_items':len(bib)})
report={'baseline_commit':BASE,'scope':'Read-only equations, printed table semantics, frozen-trajectory aggregation, and active TeX references; no optimization or retraining.','status':'PASS' if not issues else 'FAIL','sources':sources,'checks':checks,'issues':issues,'canonical_table_values':canonical_data}
(OUT/'source_verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'status':report['status'],'checks':len(checks),'issues':issues},ensure_ascii=False))
