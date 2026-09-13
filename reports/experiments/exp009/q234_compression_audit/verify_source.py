"""Read-only paper compression audit. No solver, training, or result generator is called."""
from pathlib import Path
import collections, hashlib, json, re, subprocess
ROOT=Path(__file__).resolve().parents[4]
OUT=Path(__file__).parent
BASE='b24d106715a550c16f541f1bc7b34e275d1955db'
INTERMEDIATE_BASE='cfea86a8342d0feaa6d22e22d0bd78b2a5574a2e'
OLD_BASE='52be6e025a82b67991b0379fdbfa45623fe50cc1'
DATES=['20250320','20250621','20250923','20251221']
DISPLAY=['3月20日','6月21日','9月23日','12月21日']
TIME=re.compile(r'^\d{2}:\d{2}--\d{2}:\d{2}$')
NUM=re.compile(r'-?\d+\.\d+')
checks=[]; issues=[]
def check(name,ok,detail=None):
    checks.append({'name':name,'pass':bool(ok),'detail':detail})
    if not ok:issues.append(name)
def before(path,base=BASE):return subprocess.check_output(['git','show',f'{base}:{path}'],cwd=ROOT)
def text_before(path,base=BASE):return before(path,base).decode()
def now(path):return (ROOT/path).read_text()
def sha(data):return hashlib.sha256(data).hexdigest()
def tables(text):
    labels=list(re.finditer(r'\\label\{([^}]+)\}', text)); out={}
    for i,m in enumerate(labels):
        tail=text[m.end():labels[i+1].start() if i+1<len(labels) else len(text)]
        t=re.search(r'\\begin\{tabular\}',tail)
        if t:
            start=t.end()
            while tail[start].isspace():start+=1
            assert tail[start]=='{'
            depth=0;end=start
            for end in range(start,len(tail)):
                if tail[end]=='{':depth+=1
                elif tail[end]=='}':
                    depth-=1
                    if depth==0:break
            spec=tail[start+1:end]
            stop=tail.index('\\end{tabular}',end)
            body=tail[end+1:stop]
            captions=re.findall(r'\\caption\{([^{}]*)\}',text[:m.start()])
            out[m.group(1)]={'spec':spec,'body':body,'caption':captions[-1] if captions else ''}
    return out

def rows(table):
    s=table['body']
    s=re.sub(r'\\(?:toprule|midrule|bottomrule)','',s)
    s=re.sub(r'\\cmidrule(?:\([^)]*\))?\{[^}]+\}','',s)
    s=re.sub(r'\\multicolumn\{\d+\}\{[^}]+\}\{([^{}]*)\}',r'\1',s)
    return [[c.strip() for c in row.strip().split('&')] for row in s.split('\\\\') if row.strip()]

def canonical_original(text,kind):
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

def canonical(text,kind):
    out,ts=canonical_original(text,kind)
    if kind=='q3' and 'tab:q3-original' not in ts:
        candidates=[]
        for label,table in ts.items():
            rs=rows(table)
            if len(rs)>2 and '原计划' in table['body'] and '最终' in table['body'] and all(d in table['body'] for d in DISPLAY):
                candidates.append((label,rs))
        assert len(candidates)==1,[(x,len(y)) for x,y in candidates]
        label,rs=candidates[0]
        header_dates=re.findall(r'\d{1,2}月\d{1,2}日',' '.join(rs[0]))
        assert header_dates==DISPLAY,header_dates
        header_roles=[c for c in rs[1] if c]
        assert sum('原计划' in c for c in header_roles)==4 and sum('最终' in c for c in header_roles)==4,header_roles
        for row in rs[2:]:
            metric={'全天电量':'total','全天电量/kWh':'total','全天总费用/元':'cost','全天费用/元':'cost'}.get(row[0],row[0])
            assert len(row)==9,row
            for idx,date in enumerate(DATES):
                for typ,offset in [('original',1),('final',2)]:
                    value=row[idx*2+offset]
                    if metric=='cost' and typ=='original':
                        assert value in {'--','—','-'};continue
                    key='|'.join([date,typ,metric]);assert key not in out,key;out[key]=value
    return out,ts

# Exact byte protection relative to the synchronized current-main baseline.
allowed={
 'Xelatex/sections/06-问题二.tex','Xelatex/sections/07-问题三.tex','Xelatex/sections/08-问题四.tex',
 'Xelatex/final_results/q2/discussion.tex','Xelatex/final_results/q3_specified.tex','Xelatex/final_results/q4_narrative.tex','Xelatex/数模通用模板.tex'}
raw=subprocess.check_output(['git','ls-tree','-r','-z',BASE],cwd=ROOT)
protected=[];changes=[]
for entry in raw.split(b'\0'):
    if not entry:continue
    meta,name=entry.split(b'\t',1);mode,typ,obj=meta.decode().split();path=name.decode()
    if typ!='blob':continue
    relevant=path.startswith(('Xelatex/','experiments/','scripts/','data/'))
    if not relevant or path in allowed:continue
    fp=ROOT/path
    if not fp.exists():changes.append(path);continue
    data=fp.read_bytes();blob=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
    raw_identical=blob==obj
    ok=raw_identical
    checkout_note=None
    authorized_exception=None
    if path=='Xelatex/sections/05-问题一.tex':
        baseline_data=before(path)
        old_ref=b'\\ref{q1-compare}'
        new_ref=b'\\ref{tab:q1-compare}'
        ok=baseline_data.count(old_ref)==2 and data==baseline_data.replace(old_ref,new_ref)
        authorized_exception={'user_approval':'允许，仅修正这两处引用（推荐）','old':old_ref.decode(),'new':new_ref.decode(),'replacement_count':2,'exact_replacement_only':ok}
    if not ok and path in {'scripts/run_ml.ps1','scripts/setup_ml.ps1'}:
        attrs=subprocess.check_output(['git','check-attr','text','eol','--',path],cwd=ROOT,text=True)
        normalized=data.replace(b'\r\n',b'\n')
        normalized_blob=hashlib.sha1(b'blob '+str(len(normalized)).encode()+b'\0'+normalized).hexdigest()
        if ': text: set' in attrs and ': eol: crlf' in attrs and normalized_blob==obj:
            ok=True
            checkout_note='Declared text/eol=crlf checkout: worktree CRLF becomes the identical baseline LF Git blob; no textual or executable change.'
    if not ok:changes.append(path)
    protected.append({'path':path,'sha256':sha(data),'baseline_git_blob':obj,'matches_required_bytes':ok,'raw_byte_identical':raw_identical,'checkout_note':checkout_note,'authorized_exception':authorized_exception})
check('protected_bytes_unchanged_except_two_authorized_q1_refs',not changes,{'checked_files':len(protected),'raw_byte_identical_files':sum(x['raw_byte_identical'] for x in protected),'declared_checkout_crlf_files':[x['path'] for x in protected if x['checkout_note']],'authorized_q1_reference_fix':[x['authorized_exception'] for x in protected if x['authorized_exception']],'changed':changes})
secprotected=[x for x in protected if x['path'].startswith('Xelatex/sections/')]
check('00_to_05_and_09_to_14_protection_with_explicit_q1_exception',len(secprotected)==12 and all(x['matches_required_bytes'] for x in secprotected),{'count':len(secprotected),'raw_byte_identical_count':sum(x['raw_byte_identical'] for x in secprotected),'paths':[x['path'] for x in secprotected]})
# Main entry may remove two clearpages and patch table spacing inside the
# already-existing local Q2-Q4 group; the class and surrounding text stay intact.
main='Xelatex/数模通用模板.tex';old=text_before(main);new=now(main)
expected=old
for sec in ['06-问题二','07-问题三']:
    pattern='\\input{sections/'+sec+'}\n\\clearpage'
    assert expected.count(pattern)==1
    expected=expected.replace(pattern,'\\input{sections/'+sec+'}',1)
local_format='% 仅在问题二至四采用紧凑表格行距，覆盖类文件在每张表内强制的1.38倍设置。\n\\patchcmd{\\tabular}{\\renewcommand{\\arraystretch}{1.38}}{}{}{\\PackageError{paper}{Local table spacing patch failed}{Check the tabular definition in cumcmthesis.cls.}}\n\\renewcommand{\\arraystretch}{1.05}\n'
expected=expected.replace('\\input{sections/06-问题二}',local_format+'\\input{sections/06-问题二}',1)
check('main_only_removes_two_clearpages_and_adds_local_q234_spacing',new==expected,{'local_group_only':True,'patch_failure_is_explicit_error':True})
check('main_prefix_through_q1_byte_identical_except_q234_local_format',new.split('\\input{sections/06-问题二}')[0].replace(local_format,'',1)==old.split('\\input{sections/06-问题二}')[0])
check('main_after_q4_byte_identical',new.split('\\input{sections/08-问题四}',1)[1]==old.split('\\input{sections/08-问题四}',1)[1])

# Synchronization of front chapters did not alter the Q2-Q4 comparison baseline.
qpaths=['Xelatex/sections/06-问题二.tex','Xelatex/sections/07-问题三.tex','Xelatex/sections/08-问题四.tex','Xelatex/final_results/q2/tables.tex','Xelatex/final_results/q3_specified.tex','Xelatex/final_results/q4_4-2_specified.tex','Xelatex/final_results/q4_4-3_specified.tex']
check('q234_baselines_match_before_front_chapter_sync',all(before(p)==before(p,OLD_BASE)==before(p,INTERMEDIATE_BASE) for p in qpaths))

# Data keys include dates, time intervals, purchase versions, and column roles.
values={}
for kind,path in {'q2':'Xelatex/final_results/q2/tables.tex','q3':'Xelatex/final_results/q3_specified.tex','q4_4-2':'Xelatex/final_results/q4_4-2_specified.tex','q4_4-3':'Xelatex/final_results/q4_4-3_specified.tex'}.items():
    a,ats=canonical(text_before(path),kind);b,bts=canonical(now(path),kind)
    check(kind+':specified_result_semantic_mapping',a==b,{'numeric_values':len(a),'missing':sorted(set(a)-set(b)),'extra':sorted(set(b)-set(a)),'changed':[k for k in a.keys()&b.keys() if a[k]!=b[k]]})
    values[kind]=b
    if kind!='q3':check(kind+':result_table_file_byte_identical',before(path)==(ROOT/path).read_bytes())

# Every retained display objective is compared to its baseline algebra, allowing
# only deleted explanatory definitions or changed constraint references.
def equations(text):
    out={}
    for m in re.finditer(r'\\begin\{equation\}(.*?)\\end\{equation\}',text,re.S):
        lab=re.search(r'\\label\{([^}]+)\}',m[1]);assert lab
        out[lab[1]]=m[1]
    return out
norm=lambda s:re.sub(r'\s+','',s)
eq_report={}
for p in qpaths[:3]:
    a=equations(text_before(p));b=equations(now(p))
    eq_report[p]={'retained':sorted(a.keys()&b.keys()),'deleted':sorted(a.keys()-b.keys()),'new':sorted(b.keys()-a.keys())}
for label in ['eq:q2-mode-budget','eq:q2-plan-objective']:
    p=qpaths[0];a=equations(text_before(p));b=equations(now(p));check(label+':unchanged',label in b and a[label]==b[label])
p=qpaths[1];a=equations(text_before(p));b=equations(now(p))
for label in ['eq:q3-adjustment','eq:q3-bill']:
    check(label+':unchanged_algebra',label in b and norm(a[label])==norm(b[label]))
label='eq:q3-objective'
def objective(expr):
    expr=expr.split('\\text{s.t.}')[0]
    expr=expr.replace('\\begin{aligned}','').replace('\\end{aligned}','')
    expr=re.sub(r'\\label\{[^}]+\}','',expr)
    return norm(expr).rstrip('.,\\&')
check(label+':unchanged_objective_algebra',label in b and objective(a[label])==objective(b[label]))
p=qpaths[2];a=equations(text_before(p));b=equations(now(p))
label='eq:q4-price'
def price_objective(expr):
    return norm(expr.split('\\widehat p_{i\\mid o}')[0])
check(label+':unchanged_regression_objective',label in b and price_objective(a[label])==price_objective(b[label]))

# Resolve only active TeX inputs; never follow code-listing source as manuscript.
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
walk(ROOT/main);alltext='\n'.join(seen.values())
labels=re.findall(r'\\label\{([^}]+)\}',alltext);refs=re.findall(r'\\(?:ref|eqref|autoref)\{([^}]+)\}',alltext)
check('active_refs_resolve',set(refs)<=set(labels),{'missing':sorted(set(refs)-set(labels))})
check('no_duplicate_labels',len(labels)==len(set(labels)),{'duplicates':[k for k,v in collections.Counter(labels).items() if v>1]})
bib=set(re.findall(r'\\bibitem\{([^}]+)\}',alltext));cites={x.strip() for c in re.findall(r'\\cite(?:\[[^]]*\])?\{([^}]+)\}',alltext) for x in c.split(',')}
check('active_citations_resolve',cites<=bib,{'missing':sorted(cites-bib)})
# All result figure files continue to be included (Q3 grouping can change).
for p in qpaths[:3]:
    figs=lambda t:re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}',t)
    check(p+':all_result_figures_preserved',sorted(figs(text_before(p)))==sorted(figs(now(p))))
report={'status':'PASS' if not issues else 'FAIL','baseline_commit':BASE,'intermediate_baseline_commit':INTERMEDIATE_BASE,'previous_baseline_commit':OLD_BASE,'scope':'Read-only source-scope, protected-file bytes, printed result mappings, retained objective algebra, active references and figure preservation. Semantic prose audit is separate.','checks':checks,'issues':issues,'protected_files':protected,'source_sha256':{p:sha((ROOT/p).read_bytes()) for p in sorted(allowed)},'equations':eq_report,'canonical_result_values':values}
(OUT/'source_verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'status':report['status'],'checks':len(checks),'protected_files':len(protected),'issues':issues},ensure_ascii=False))
