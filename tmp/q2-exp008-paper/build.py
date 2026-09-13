from pathlib import Path
from datetime import date,timedelta
import json,hashlib,csv
import numpy as np
import openpyxl
root=Path(__file__).resolve().parents[2]; work=root/'tmp/q2-exp008-paper'
exp=root/'reports/experiments/exp008'
selection=json.loads((root/'experiments/exp008/final_selection.json').read_text(encoding='utf-8'))
archive=root/selection['scenarios']['2']['archive']
assert hashlib.sha256(archive.read_bytes()).hexdigest()==selection['finalization_acceptance']['accepted_q2_archive_sha256']
z=np.load(archive); p=json.loads((exp/'final_payload.json').read_text(encoding='utf-8'))['scenarios']['2']
g,c,d,e,s,price,actual,mode=[z[k] for k in ['original','charge','discharge','emergency','states','price','actual','allowed_charge']]
eta=np.sqrt(.9); n=(actual[:,:,0]-actual[:,:,1])/6
def same(a,b):assert np.max(np.abs(np.asarray(a)-np.asarray(b)))<1e-6
same(g,z['final']); same(g+d+e-c-z['surplus'],n); same(np.diff(s,axis=1),eta*c-d/eta); same(s[1:,0],s[:-1,-1])
assert s.min()>=1200-1e-6 and s.max()<=10800+1e-6 and max(c.max(),d.max())<=5000/6+1e-6
same(c*d,0); same(c*e,0)
B=g-n
same(c,mode*np.minimum.reduce([np.maximum(B,0),np.full_like(B,5000/6),(10800-s[:,:-1])/eta]))
same(d,(~mode)*np.minimum.reduce([np.maximum(-B,0),np.full_like(B,5000/6),(s[:,:-1]-1200)*eta]))
same(z['fees'][:,:,0],g*price);same(z['fees'][:,:,3],5*e*price);same(z['fees'][:,:,1:3],0)
same(z['fees'].sum(),p['fees']['total_cost_yuan'])
flat=mode.ravel(); flips=np.r_[flat[0]!=True,flat[1:]!=flat[:-1]].reshape(334,144); assert flips.sum(1).max()<=8
wbpath=exp/'result2.xlsx'; whash=hashlib.sha256(wbpath.read_bytes()).hexdigest()
wb=openpyxl.load_workbook(wbpath,read_only=True,data_only=True)
rows=list(wb.worksheets[0].values); assert len(rows)==335
for i,r in enumerate(rows[1:]):
    assert r[0].date()==date(2025,2,1)+timedelta(days=i)
    same(r[1:145],g[i]);same(r[145],g[i].sum());same(r[146],z['fees'][i].sum())
rows=list(wb.worksheets[1].values)[1:]; assert len(rows)==2004
for i in range(334):
    for j in range(6):
        r=rows[6*i+j];same(r[2],c[i,j*24:(j+1)*24].sum());same(r[3],d[i,j*24:(j+1)*24].sum())
    same(rows[6*i][5],s[i,0]);same(rows[6*i+1][5],s[i,-1])
rows=list(wb.worksheets[2].values)[1:]; count=0
for r in rows:
    if r[0] is not None: idx=(r[0].date()-date(2025,2,1)).days
    if not r[1] or '-' not in str(r[1]):
        same(e[idx].sum(),0);continue
    a,b=r[1].split('-');slot=lambda v:int(v.split(':')[0])*6+int(v.split(':')[1])//10
    same(r[2],e[idx,slot(a):slot(b)].sum());count+=1
same(sum(float(r[2] or 0) for r in rows),e.sum());wb.close()
prefix='\\begin{table}[H]\n\\centering\\small\n\\setlength{\\tabcolsep}{3pt}\n\\renewcommand{\\arraystretch}{1.13}\n'
end='\\bottomrule\\end{tabularx}\n\\end{table}\n'
def row(vals):return ' & '.join(vals)+r'\\'+'\n'
def f(v):return f'{v:.4f}'
def period(v):return v.replace('-','--')
tables=[]
for k,q in enumerate(p['specified_dates']):
    idx=(date.fromisoformat(q['date'])-date(2025,2,1)).days
    stamp=q['date'].replace('-','');dt=date.fromisoformat(q['date']);ds=f'{dt.year}年{dt.month}月{dt.day}日'
    t=('\\clearpage\n' if k==2 else '')+prefix+rf'\caption{{{ds}指定时段购电量及全天费用}}\label{{tab:q2-purchase-{stamp}}}'+'\n'+r'\begin{tabularx}{\textwidth}{CrCrCr}'+'\n'+r'\toprule'+'\n'+row(['时间段','购电量']*3)+r'\midrule'+'\n'
    for ids in ([0,1,2],[3,4,5]):
        vals=[]
        for j in ids:
            v=q['table1'][j];same(v['original_purchase_kwh'],g[idx,v['slot']]);vals.extend([period(v['interval']),f(v['original_purchase_kwh'])])
        t+=row(vals)
    same(q['fees']['total_cost_yuan'],z['fees'][idx].sum())
    t+=r'\midrule'+'\n'+row([r'\multicolumn{2}{c}{全天计划购电量}',f(g[idx].sum()),r'\multicolumn{2}{c}{全天购电费}',f(q['fees']['total_cost_yuan'])])+end
    t+=prefix+rf'\caption{{{ds}实际充放电量与储电量}}\label{{tab:q2-battery-{stamp}}}'+'\n'+r'\begin{tabularx}{\textwidth}{CrrCrr}'+'\n'+r'\toprule'+'\n'+row(['时间段','充电量','放电量']*2)+r'\midrule'+'\n'
    for ids in ([0,1],[2,3],[4,5]):
        vals=[]
        for j in ids:
            v=q['table2']['four_hour_blocks'][j];same(v['charge_kwh'],c[idx,j*24:(j+1)*24].sum());same(v['discharge_kwh'],d[idx,j*24:(j+1)*24].sum());vals.extend([period(v['interval']),f(v['charge_kwh']),f(v['discharge_kwh'])])
        t+=row(vals)
    same(q['table2']['soc_00_kwh'],s[idx,0]);same(q['table2']['soc_24_kwh'],s[idx,-1])
    t+=r'\midrule'+'\n'+row([r'\multicolumn{2}{c}{0:00储电量}',f(s[idx,0]),r'\multicolumn{2}{c}{24:00储电量}',f(s[idx,-1])])+end
    tables.append(t)
em=prefix+r'\caption{四个指定日期的全部紧急购电结果（kWh）}\label{tab:q2-emergency-all}'+'\n'+r'\begin{tabularx}{\textwidth}{CrCrCrCr}'+'\n'+r'\toprule'+'\n'+row([r'\multicolumn{2}{c}{'+q['date'].replace('-','.')+'}' for q in p['specified_dates']])+r'\midrule'+'\n'+row(['时间段','购电量']*4)
for j in range(2):
    vals=[]
    for q in p['specified_dates']:
        ev=q['table3']['events'];idx=(date.fromisoformat(q['date'])-date(2025,2,1)).days
        if j<len(ev):
            v=ev[j];same(v['energy_kwh'],e[idx,v['start_slot']:v['end_slot_exclusive']].sum());vals.extend([period(v['interval']),f(v['energy_kwh'])])
        else:vals.extend(['--','--'])
    em+=row(vals)
em+=r'\midrule'+'\n'+row([val for q in p['specified_dates'] for val in ['合计',f(q['table3']['total_emergency_kwh'])]])+end
hist=list(csv.DictReader((exp/'evidence/cost_history.csv').open(encoding='utf-8-sig')));base=next(r for r in hist if r['policy_id']=='exp006/primary')
b=p['battery']
comp=prefix+r'\caption{实验六与实验八的同口径评价结果}\label{tab:q2-comparison}'+'\n'+r'\begin{tabularx}{\textwidth}{Xrr}'+'\n'+r'\toprule'+'\n'+row(['指标','实验六正式主组','实验八最终方案'])+r'\midrule'+'\n'
for vals in [('计划费/万元',f(float(base['planned_cost'])/1e4),f(p['fees']['planned_cost_yuan']/1e4)),('紧急费/万元',f(float(base['emergency_cost'])/1e4),f(p['fees']['emergency_cost_yuan']/1e4)),('总费用/万元',f(float(base['total_cost'])/1e4),f(p['fees']['total_cost_yuan']/1e4)),('实际非空换向/次','2729',str(b['direction_reversals'])),('连续充放电段起点/个','3791',str(b['charge_starts']+b['discharge_starts'])),('活跃槽/个','23028',str(b['active_slots'])),('交流吞吐/万kWh',f(float(base['throughput_kwh'])/1e4),f(b['throughput_kwh']/1e4)),('等效满循环',f(float(base['equivalent_full_cycles'])),f(b['equivalent_full_cycles'])),('功率总变差/万kW','3045.1039',f(b['power_ramp_total_kw']/1e4))]:comp+=row(vals)
comp+=end
m=(work/'methods.tex').read_text(encoding='utf-8').replace('% SPECIFIED_TABLES','\n'.join(tables)).replace('% EMERGENCY_TABLE',em).replace('% COMPARISON_TABLE',comp)
section=root/'Xelatex/sections/06-问题二.tex';main=root/'Xelatex/数模通用模板.tex'
if not (work/'before-section.tex').exists():(work/'before-section.tex').write_bytes(section.read_bytes())
if not (work/'before-main.tex').exists():(work/'before-main.tex').write_bytes(main.read_bytes())
section.write_text(m,encoding='utf-8',newline='\r\n')
old=main.read_text(encoding='utf-8')
if r'\input{sections/06-问题二.tex}' not in old:
    a=old.index(r'\section{问题二的模型的建立和求解}');b=old.index(r'\section{问题三的模型的建立和求解}',a)
    main.write_text(old[:a]+'\\input{sections/06-问题二.tex}\n\n'+old[b:],encoding='utf-8',newline='\r\n')
assert whash==hashlib.sha256(wbpath.read_bytes()).hexdigest()
receipt={'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'workbook_sha256':whash,'days':334,'slots':48096,'specified_emergency_events':7,'workbook_emergency_events':count,'physics_and_mode_budget':'passed','workbook_readback':'passed','source_tables_match':'passed','total_cost':float(z['fees'].sum()),'experiments_rerun':False,'tables':m.count(r'\begin{table}')}
(work/'verification.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(receipt,ensure_ascii=False))
