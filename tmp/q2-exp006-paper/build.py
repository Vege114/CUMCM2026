from pathlib import Path
from datetime import date, timedelta
import csv, json, re, hashlib
import numpy as np
import openpyxl

root = Path(__file__).resolve().parents[2]
work = root / 'tmp/q2-exp006-paper'
exp = root / 'reports/experiments/exp006'
j = json.loads((exp/'specified_tables/specified_tables.json').read_text(encoding='utf-8'))
z = np.load(root/'data/results/exp006/primary/dispatch_2.npz')
days = [date(2025,2,1)+timedelta(days=i) for i in range(334)]
index = {d.isoformat():i for i,d in enumerate(days)}
g, c, dis, e, states, p = [z[k] for k in ['original','charge','discharge','emergency','states','price']]
eta=np.sqrt(.9)
def same(a,b,tol=1e-7):
    assert np.max(np.abs(np.asarray(a)-np.asarray(b))) < tol
assert g.shape==(334,144)
same(g,z['final'])
same(np.diff(states,axis=1),eta*c-dis/eta)
same(states[:-1,-1],states[1:,0])
same(g+e+dis,(z['actual'][:,:,0]-z['actual'][:,:,1])/6+c+z['surplus'])
assert states.min() >= 1200-1e-8 and states.max() <=10800+1e-8
assert max(c.max(),dis.max()) <=5000/6+1e-8
assert not np.any((c>1e-7)&(dis>1e-7))
b=g+(z['actual'][:,:,1]-z['actual'][:,:,0])/6
expected_c=np.where(b>=0,np.minimum.reduce([z['intended_charge'],np.maximum(b,0),np.full_like(g,5000/6),np.maximum((10800-states[:,:-1])/eta,0)]),0)
expected_d=np.where(b<0,np.minimum.reduce([z['intended_discharge'],np.maximum(-b,0),np.full_like(g,5000/6),np.maximum((states[:,:-1]-1200)*eta,0)]),0)
expected_c[expected_c<2]=0; expected_d[expected_d<2]=0
same(c,expected_c); same(dis,expected_d)
planfees=(g*p).sum(axis=1); emergencyfees=(5*e*p).sum(axis=1)
same(z['fees'][:,:,0],g*p); same(z['fees'][:,:,3],5*e*p)
same(planfees.sum()+emergencyfees.sum(),14066257.477256786)

for r in j['table1_summary']:
    i=index[r['date']]
    for name,val in {'planned_kwh':g[i].sum(),'planned_cost_yuan':planfees[i],'emergency_cost_yuan':emergencyfees[i],'total_cost_yuan':planfees[i]+emergencyfees[i],'emergency_kwh':e[i].sum(),'initial_soc_kwh':states[i,0],'final_soc_kwh':states[i,-1]}.items(): same(val,r[name])
def slot(s):
    hh,mm=map(int,s.split(':')); return hh*6+mm//10
for r in j['table1_intervals']:
    same(g[index[r['date']],slot(r['period'].split('-')[0])],r['planned_kwh'])
for r in j['table2_battery']:
    a,b0=map(slot,r['period'].split('-')); i=index[r['date']]
    same(c[i,a:b0].sum(),r['charge_kwh']); same(dis[i,a:b0].sum(),r['discharge_kwh'])
for r in j['table3_emergency']:
    same(e[index[r['date']],r['start_slot']:r['end_slot_exclusive']].sum(),r['emergency_kwh'])
for r in j['table1_summary']:
    same(sum(x['emergency_kwh'] for x in j['table3_emergency'] if x['date']==r['date']),r['emergency_kwh'])

# Read back the delivered workbook; retain its original bytes and template layout.
book=exp/'result2.xlsx'
before_hash=hashlib.sha256(book.read_bytes()).hexdigest()
w=openpyxl.load_workbook(book,data_only=True,read_only=True)
rows=list(w['计划购电量'].values)
assert len(rows)==335 and len(rows[0])==147
for i,r in enumerate(rows[1:]):
    assert r[0].date()==days[i]
    same(r[1:145],g[i]); same(r[145],g[i].sum()); same(r[146],planfees[i]+emergencyfees[i])
rows=list(w['充放电量'].values)[1:]
assert len(rows)==334*6
for i in range(334):
    chunk=rows[i*6:(i+1)*6]
    assert chunk[0][0].date()==days[i]
    for k,r in enumerate(chunk):
        same(r[2],c[i,24*k:24*(k+1)].sum()); same(r[3],dis[i,24*k:24*(k+1)].sum())
    same(chunk[0][5],states[i,0]); same(chunk[1][5],states[i,-1])
emergency_totals=np.zeros(334); emergency_rows=0; current=None
for r in list(w['紧急购电量'].values)[1:]:
    if r[0] is not None: current=index[r[0].date().isoformat()]
    if r[1]=='无': same(r[2],0); continue
    a,b0=map(slot,r[1].split('-'))
    same(r[2],e[current,a:b0].sum())
    emergency_totals[current]+=r[2]; emergency_rows+=1
same(emergency_totals,e.sum(axis=1)); w.close()
assert before_hash==hashlib.sha256(book.read_bytes()).hexdigest()

f=lambda x:f'{x:.4f}'
period=lambda s:s.replace('-','--')
table_prefix=r'''\begin{table}[H]
\centering\small
\setlength{\tabcolsep}{3pt}
\renewcommand{\arraystretch}{1.13}
'''
tables=[]
for n,s in enumerate(j['table1_summary']):
    dt=s['date']; ds=dt[5:].replace('-','月')+'日'; ds=ds.lstrip('0')
    stamp=dt.replace('-','')
    if n==2: tables.append(r'\clearpage')
    intervals=[r for r in j['table1_intervals'] if r['date']==dt]
    batteries=[r for r in j['table2_battery'] if r['date']==dt]
    tab=table_prefix+rf'\caption{{2025年{ds}指定时段购电量及全天费用}}\label{{tab:q2-purchase-{stamp}}}'+'\n'
    tab+=r'\begin{tabularx}{\textwidth}{CrCrCr}'+'\n'+r'\toprule 时间段 & 购电量 & 时间段 & 购电量 & 时间段 & 购电量\\'+'\n'+r'\midrule'+'\n'
    for ids in ([0,1,2],[3,4,5]):
        tab+=' & '.join(' & '.join([period(intervals[k]['period']),f(intervals[k]['planned_kwh'])]) for k in ids)+r'\\'+'\n'
    tab+=r'\midrule'+'\n'+rf'\multicolumn{{2}}{{c}}{{全天计划购电量}} & {f(s["planned_kwh"])} & \multicolumn{{2}}{{c}}{{全天购电费}} & {f(s["total_cost_yuan"])}\\'+'\n'
    tab+=r'\bottomrule\end{tabularx}'+'\n'+r'\end{table}'+'\n'
    tab+=table_prefix+rf'\caption{{2025年{ds}储能设备实际充放电量与储电量}}\label{{tab:q2-battery-{stamp}}}'+'\n'
    tab+=r'\begin{tabularx}{\textwidth}{CrrCrr}'+'\n'+r'\toprule 时间段 & 充电量 & 放电量 & 时间段 & 充电量 & 放电量\\'+'\n'+r'\midrule'+'\n'
    for ids in ([0,1],[2,3],[4,5]):
        tab+=' & '.join(' & '.join([period(batteries[k]['period']),f(batteries[k]['charge_kwh']),f(batteries[k]['discharge_kwh'])]) for k in ids)+r'\\'+'\n'
    tab+=r'\midrule'+'\n'+rf'\multicolumn{{2}}{{c}}{{0:00储电量}} & {f(s["initial_soc_kwh"])} & \multicolumn{{2}}{{c}}{{24:00储电量}} & {f(s["final_soc_kwh"])}\\'+'\n'
    tab+=r'\bottomrule\end{tabularx}'+'\n'+r'\end{table}'+'\n'
    tables.append(tab)

dates=[r['date'] for r in j['table1_summary']]
cols=[[x for x in j['table3_emergency'] if x['date']==d] for d in dates]
em=table_prefix+r'\caption{四个指定日期的全部紧急购电结果（kWh）}\label{tab:q2-emergency-all}'+'\n'
em+=r'\begin{tabularx}{\textwidth}{CrCrCrCr}'+'\n'+r'\toprule'+'\n'
em+=' & '.join(r'\multicolumn{2}{c}{'+d.replace('-','.')+'}' for d in dates)+r'\\'+'\n'
em+=r'\cmidrule(lr){1-2}\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}'+'\n'
em+=' & '.join(['时间段 & 购电量']*4)+r'\\'+'\n'+r'\midrule'+'\n'
for i in range(max(map(len,cols))):
    em+=' & '.join((period(col[i]['period'])+' & '+f(col[i]['emergency_kwh'])) if i<len(col) else r'-- & --' for col in cols)+r'\\'+'\n'
em+=r'\midrule'+'\n'+' & '.join('合计 & '+f(r['emergency_kwh']) for r in j['table1_summary'])+r'\\'+'\n'+r'\bottomrule\end{tabularx}'+'\n'+r'\end{table}'+'\n'

annual=table_prefix+r'\caption{评价期真实费用与电池运行指标对照}\label{tab:q2-annual}'+'\n'
annual+=r'\begin{tabularx}{\textwidth}{Xrrr}'+'\n'+r'\toprule 指标 & 同预测基础调度 & 正式主组 & 旧执行重规划对照\\'+'\n'+r'\midrule'+'\n'
annual+=r'''计划购电/万kWh & 2099.7148 & 2170.2244 & 2156.0477\\
总费用/万元 & 1361.1588 & 1406.6257 & 1353.4197\\
紧急购电/kWh & 160912.1654 & 197559.7826 & 74976.2978\\
非空方向反转/次 & 7715 & 2729 & 6755\\
电芯等效满循环 & 499.1216 & 477.7055 & 489.3580\\
功率总变差/万kW & 3343.0433 & 3045.1039 & 3266.6146\\
\bottomrule\end{tabularx}
\end{table}
'''
# Bind every annual number to saved evidence instead of manual approximation.
hist=list(csv.DictReader((exp/'evidence/cost_history.csv').open(encoding='utf-8-sig')))
allrows=list(csv.DictReader((exp/'evidence/all_runs.csv').open(encoding='utf-8-sig')))
base=next(r for r in hist if r['policy_id']=='exp004/no_season')
battery=list(csv.DictReader((exp/'evidence/battery_history.csv').open(encoding='utf-8-sig')))
base.update(next(r for r in battery if r['policy_id']=='exp004/no_season'))
base['power_ramp_total_kw']=base['power_variation_kw']
main=next(r for r in allrows if r['run_id']=='primary')
greedy=next(r for r in allrows if r['run_id']=='greedy_execution')
metrics=[('计划购电/万kWh','planned_kwh',1e4),('总费用/万元','total_cost',1e4),('紧急购电/kWh','emergency_kwh',1),('非空方向反转/次','direction_reversals',1),('电芯等效满循环','equivalent_full_cycles',1),('功率总变差/万kW','power_ramp_total_kw',1e4)]
lines=[]
for label,key,scale in metrics:
    vals=[float(row[key])/scale for row in [base,main,greedy]]
    lines.append(label+' & '+' & '.join(str(round(x)) if key=='direction_reversals' else f(x) for x in vals)+r'\\')
annual=annual[:annual.index('计划购电/万kWh')]+'\n'.join(lines)+'\n'+r'\bottomrule\end{tabularx}'+'\n'+r'\end{table}'+'\n'

methods=(work/'methods.tex').read_text(encoding='utf-8')
methods=methods.replace(r'\subsection{指定日期购电及储能结果}',r'\clearpage'+'\n'+r'\subsection{指定日期购电及储能结果}')
methods=methods.replace(r'\subsection{紧急购电与评价期绩效}',r'\clearpage'+'\n'+r'\subsection{紧急购电与评价期绩效}')
methods=methods.replace('% Q2_SPECIFIED_TABLES','\n'.join(tables)).replace('% Q2_EMERGENCY_TABLE',em).replace('% Q2_ANNUAL_TABLE',annual)
tex=root/'Xelatex/数模通用模板.tex'
old=tex.read_text(encoding='utf-8')
start=old.index(r'\section{问题二的模型的建立和求解}')
end=old.index(r'\section{问题三的模型的建立和求解}',start)
new=old[:start]+methods+'\n'+old[end:]
a=new.index(r'\subsection*{问题2分析}')
b0=new.index(r'\subsection*{问题3分析}',a)
new=new[:a]+r'''\subsection*{问题2分析}
日前购电量必须在实际供需揭晓前确定，预测缺口与剩余购电的费用并不对称。采用卷积残差预测生成负载与光伏曲线，再以条件误差决策树构造逐槽净需求支持，通过离散储能动态规划权衡计划费用、补购风险和操作强度。以第六次实验正式主组为结果依据，执行时按实际供需、储电量及计划意图裁剪电池动作，缺口按5倍电价补购。最后给出四个指定日期的购电、实际充放电与全部紧急购电区间，并以真实费用及运行指标检验方案。

'''+new[b0:]
def outside(s):
    s=re.sub(r'\\subsection\*\{问题2分析\}.*?(?=\\subsection\*\{问题3分析\})','<analysis>',s,flags=re.S)
    return re.sub(r'\\section\{问题二的模型(?:的建立和求解|建立与求解)\}.*?(?=\\section\{问题三的模型的建立和求解\})','<q2>',s,flags=re.S)
assert outside(old)==outside(new)
tex.write_text(new,encoding='utf-8',newline='\r\n')
summary={'source':'exp006/primary','source_workbook_sha256':before_hash,'checked_days':334,'checked_slots':48096,'checked_workbook_emergency_rows':emergency_rows,'specified_purchase_values':24,'specified_battery_blocks':24,'specified_soc_values':8,'specified_emergency_intervals':37,'tables_inserted':methods.count('\\begin{table}'),'actual_cost':float(planfees.sum()+emergencyfees.sum()),'physical_and_projected_execution_check':'passed','workbook_full_readback':'passed','other_sections_unchanged':True,'new_optimization_run':False}
(work/'verification.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False))
