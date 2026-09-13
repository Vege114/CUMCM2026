"""Rebuild Q4 tables/figures from the accepted, immutable exp008 trajectories.

This script neither retrains forecasting models nor changes purchase decisions.
Run: python Xelatex/q4_run/q4_assets.py
"""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'vendor'))
from setup_style import setup_style
from export_figure import export_figure
from visual_qa import render_preview, audit_layout

OUT = ROOT/'paper'
FIG = OUT/'figures'
OUT.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)
DATES = pd.date_range('2025-02-01', '2025-12-31')
COLORS = ['#0072B2', '#D55E00']
setup_style(journal='general', lang='zh', serif_for_zh=True, use_sciplots=False)
font_path = ROOT.parent/'YaHei.Consolas.1.11b.ttf'
fm.fontManager.addfont(str(font_path))
font_name = fm.FontProperties(fname=str(font_path)).get_name()
plt.rcParams.update({'font.family': [font_name, 'DejaVu Sans'], 'font.size': 9, 'figure.dpi': 100,
    'axes.labelsize': 9, 'axes.titlesize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'legend.fontsize': 8, 'axes.unicode_minus': False, 'pdf.fonttype': 42,
    'svg.fonttype': 'none', 'axes.spines.top': False, 'axes.spines.right': False})

manifest=json.loads((ROOT/'data/source_manifest.json').read_text())
for name,item in manifest.items():
    if 'sha256' in item:
        assert hashlib.sha256((ROOT/'data'/name).read_bytes()).hexdigest()==item['sha256'],name
archives = {s:dict(np.load(ROOT/'data'/f'dispatch_{s}.npz')) for s in ['4-2','4-3']}
summary={}
rows=[]
for scenario,a in archives.items():
    assert np.array_equal(a['days'],np.arange(31,365))
    assert all(np.isfinite(v).all() for v in a.values())
    p=a['price']; g=a['original']; q=a['final']; c=a['charge']; d=a['discharge']; e=a['emergency']; E=a['states']
    up=np.maximum(q-g,0);down=np.maximum(g-q,0)
    fees=p[...,None]*np.stack([g,1.5*up,-.5*down,5*e],axis=-1)
    balance=q+d+e-c-a['surplus']-(a['actual'][...,0]-a['actual'][...,1])/6
    checks={'balance_kwh':float(np.abs(balance).max()),
        'soc_kwh':float(np.abs(np.diff(E)-np.sqrt(.9)*c+d/np.sqrt(.9)).max()),
        'cross_day_soc_kwh':float(np.abs(E[1:,0]-E[:-1,-1]).max()),
        'billing_yuan':float(np.abs(fees-a['fees']).max()),
        'simultaneous_slots':int(((c>1e-6)&(d>1e-6)).sum()),
        'emergency_charging_slots':int(((c>1e-6)&(e>1e-6)).sum()),
        'down_adjustment_kwh':float(down.sum())}
    assert max(checks.values())<1e-6,checks
    assert E.min()>=1200-1e-6 and E.max()<=10800+1e-6
    assert min(c.min(),d.min(),q.min(),e.min(),a['surplus'].min())>=-1e-6
    assert max(c.max(),d.max())<=5000/6+1e-6
    signs=np.sign((c-d).ravel());signs=signs[signs!=0]
    # Include the observed warm-up boundary (previous nonidle direction = charge).
    reversals=int(np.count_nonzero(np.diff(np.r_[1,signs])))
    summary[scenario]={'total_cost':float(fees.sum()),'planned_cost':float(fees[...,0].sum()),
        'up_cost':float(fees[...,1].sum()),'emergency_cost':float(fees[...,3].sum()),
        'purchase_kwh':float(q.sum()),'emergency_kwh':float(e.sum()),
        'reversals':reversals,'efc':float((np.sqrt(.9)*c+d/np.sqrt(.9)).sum()/24000),
        'initial_soc':float(E[0,0]),'final_soc':float(E[-1,-1]),'checks':checks}
    for i,date in enumerate(DATES):
        rows.append(dict(scenario=scenario,date=date.strftime('%Y-%m-%d'),month=date.month,
            total_cost=fees[i].sum(),planned_cost=fees[i,:,0].sum(),up_cost=fees[i,:,1].sum(),
            emergency_cost=fees[i,:,3].sum(),purchase_kwh=q[i].sum(),emergency_kwh=e[i].sum(),
            charge_kwh=c[i].sum(),discharge_kwh=d[i].sum(),initial_soc=E[i,0],final_soc=E[i,-1],
            load_kwh=a['actual'][i,:,0].sum()/6))
daily=pd.DataFrame(rows)
daily.to_csv(OUT/'daily.csv',index=False,float_format='%.10f')
monthly=daily.groupby(['scenario','month'],as_index=False).sum(numeric_only=True)
monthly.to_csv(OUT/'monthly.csv',index=False,float_format='%.10f')

price=dict(np.load(ROOT/'data/price_predictions.npz'))
assert np.array_equal(price['actual'], archives['4-2']['price'])
assert np.array_equal(price['days'],np.arange(31,365))
price_rows=[]
for key in ['baseline','linked_hgb']:
    err=price[key]-price['actual']
    summary[key]={'rmse':float(np.sqrt(np.mean(err**2))),'mae':float(np.abs(err).mean())}
    for m in range(2,13):
        x=err[DATES.month==m]
        price_rows.append({'model':key,'month':m,'rmse':np.sqrt(np.mean(x*x)),'mae':np.abs(x).mean()})
pd.DataFrame(price_rows).to_csv(OUT/'price_monthly.csv',index=False,float_format='%.10f')
(OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))

# Tables are typeset directly from the same verified arrays.
def table(caption,label,spec,header,body):
    return ('\\begin{table}[H]\n\\centering\\small\n\\caption{'+caption+'}\\label{'+label+'}\n'
      '\\begin{tabular}{'+spec+'}\n\\toprule\n'+header+'\\\\\n\\midrule\n'
      +'\\\\\n'.join(body)+'\\\\\n\\bottomrule\n\\end{tabular}\n\\end{table}\n')
a,b=summary['4-2'],summary['4-3']
metrics=[('原计划费用/万元','planned_cost',1e4,2),('上调费用/万元','up_cost',1e4,2),
    ('紧急购电费用/万元','emergency_cost',1e4,2),('总费用/万元','total_cost',1e4,2),
    ('常规购电量/万kWh','purchase_kwh',1e4,2),('紧急购电量/kWh','emergency_kwh',1,2),
    ('非空方向反转/次','reversals',1,0),('电芯等效满循环/次','efc',1,2)]
body=[f'{name} & {a[key]/div:.{digits}f} & {b[key]/div:.{digits}f}' for name,key,div,digits in metrics]
(OUT/'annual_table.tex').write_text(table('波动电价下两类方案的评价期结果','tab:q4-annual','lrr','指标 & 方案4-2 & 方案4-3',body))
selected=['2025-03-20','2025-06-21','2025-09-23','2025-12-21']
body=[]
for dt in selected:
    x=daily[(daily.date==dt)&(daily.scenario=='4-2')].iloc[0]
    y=daily[(daily.date==dt)&(daily.scenario=='4-3')].iloc[0]
    name=f'{int(dt[5:7])}月{int(dt[8:10])}日'
    body.append(f'{name} & {x.total_cost:.2f} & {y.total_cost:.2f} & {x.emergency_kwh:.2f} & {y.emergency_kwh:.2f}')
(OUT/'selected_days_table.tex').write_text(table('四个指定日的费用与紧急购电量','tab:q4-days','lrrrr',
    '日期 & \\multicolumn{2}{c}{总费用/元} & \\multicolumn{2}{c}{紧急购电量/kWh}\\\\\n'
    '\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n & 方案4-2 & 方案4-3 & 方案4-2 & 方案4-3',body))

qa={}
def finish(fig,name,size):
    render_preview(fig,str(FIG/(name+'_preview.png')),dpi=150)
    issues=audit_layout(fig)
    qa[name]=issues
    assert not any(x[0] in ('FAIL','WARN') for x in issues),issues
    export_figure(fig,basename=str(FIG/name),formats=['pdf','png','svg'],size_inches=size,dpi=450,grayscale_preview=True,tight=False)
    plt.close(fig)

# Monthly error evolution + distribution across all paired slots.
fig,axes=plt.subplots(1,2,figsize=(6.25,2.5),layout='constrained',gridspec_kw={'width_ratios':[1.45,1]})
for k,color,marker,ls,name in [('baseline',COLORS[0],'o','-','周期价格特征'),('linked_hgb',COLORS[1],'s','--','加入供需预测')]:
    v=pd.DataFrame(price_rows).query('model==@k')
    axes[0].plot(v.month,v.rmse,color=color,marker=marker,ms=3,lw=1,ls=ls,label=name)
axes[0].set(xlabel='月份',ylabel='电价RMSE/(元/kWh)',xticks=np.arange(2,13,2),title='(a) 月度预测误差')
axes[0].legend(loc='upper left',bbox_to_anchor=(0,1.30),ncol=2,frameon=False)
vals=[np.abs(price[k]-price['actual']).ravel() for k in ['baseline','linked_hgb']]
bp=axes[1].boxplot(vals,tick_labels=['周期特征','供需特征'],patch_artist=True,showfliers=False,whis=(5,95),widths=.45)
for box,col in zip(bp['boxes'],COLORS):box.set_facecolor(col);box.set_alpha(.5)
axes[1].set(ylabel='绝对误差/(元/kWh)',title='(b) 逐槽误差分布',ylim=(0,.14))
finish(fig,'q4_price',(6.25,2.5))

fig,axes=plt.subplots(1,2,figsize=(6.25,2.5),layout='constrained',gridspec_kw={'width_ratios':[1.4,1]})
for i,(s,color) in enumerate(zip(['4-2','4-3'],COLORS)):
    v=monthly[monthly.scenario==s]
    axes[0].bar(v.month+(i-.5)*.36,v.total_cost/1e4,width=.36,color=color,label='方案'+s,hatch=None if i==0 else '//')
    frac=v.emergency_cost/v.total_cost*100
    axes[1].plot(v.month,frac,color=color,marker=['o','s'][i],ls=['-','--'][i],lw=1,ms=3,label='方案'+s)
axes[0].set(xlabel='月份',ylabel='月总费用/万元',title='(a) 月度费用',xticks=np.arange(2,13,2))
axes[0].legend(ncol=2,loc='upper left',bbox_to_anchor=(0,1.30),frameon=False)
axes[1].set(xlabel='月份',ylabel='紧急费占总费用/%',title='(b) 紧急费占比',xticks=np.arange(2,13,2))
axes[1].set_ylim(bottom=0)
finish(fig,'q4_monthly',(6.25,2.5))

fig,axes=plt.subplots(4,1,figsize=(6.25,4.9),sharex=True,layout='constrained')
i=list(DATES.strftime('%Y-%m-%d')).index('2025-09-23');hours=np.arange(144)/6;edges=np.arange(145)/6
axes[0].stairs(price['actual'][i],edges,baseline=None,color='#333333',lw=1,label='实际电价')
axes[0].stairs(price['linked_hgb'][i],edges,baseline=None,color=COLORS[0],ls='--',lw=1,label='4-2午夜预测')
axes[0].stairs(price['baseline'][i],edges,baseline=None,color=COLORS[1],ls=':',lw=1,label='4-3午夜预测')
axes[0].set(ylabel='电价\n元/kWh')
axes[0].legend(loc='lower center',bbox_to_anchor=(.5,1.03),ncol=3,frameon=False)
for j,(s,col) in enumerate(zip(['4-2','4-3'],COLORS)):
    a=archives[s];ls=['-','--'][j]
    axes[1].stairs(a['final'][i],edges,baseline=None,color=col,ls=ls,lw=.9,label='方案'+s)
    axes[2].stairs(6*(a['charge'][i]-a['discharge'][i])/1000,edges,baseline=None,color=col,ls=ls,lw=.9)
    axes[3].plot(np.arange(145)/6,a['states'][i]/1000,color=col,ls=ls,lw=1)
axes[1].set(ylabel='常规购电\nkWh');axes[1].legend(loc='upper center',ncol=2)
axes[2].set(ylabel='电池功率\nMW',ylim=(-5.5,5.5));axes[2].axhline(0,color='#999999',lw=.5)
axes[3].set(ylabel='储电量\nMWh',xlabel='时刻/h',ylim=(0,12),xlim=(0,24),xticks=np.arange(0,25,3))
for y in [1.2,10.8]:axes[3].axhline(y,color='#999999',ls=':',lw=.7)
for ax in axes:
    ax.grid(axis='y',alpha=.15)
    for h in [6,12,18]:ax.axvline(h,color='#999999',ls=':',lw=.6)
finish(fig,'q4_dispatch',(6.25,4.9))
(OUT/'figure_qa.json').write_text(json.dumps(qa,ensure_ascii=False,indent=2))
print(json.dumps(summary,ensure_ascii=False,indent=2))

# Complete problem-specified tables, using interval-start indices.
purchase_rows=[]
for slot in [60,72,84,96,108,120]:
    label=f'{slot//6:02d}:00--{slot//6:02d}:10'
    vals=[]
    for dt in selected:
        i=list(DATES.strftime('%Y-%m-%d')).index(dt)
        x=archives['4-2']['final'][i,slot]
        y=archives['4-3']['original'][i,slot]
        z=archives['4-3']['final'][i,slot]
        vals.extend([f'{x:.2f}',f'{y:.2f}/{z:.2f}'])
    purchase_rows.append(label+' & '+' & '.join(vals))
for label,key,scale in [('全天常规电量','final',1),('全天总费用','fees',1)]:
    vals=[]
    for dt in selected:
        i=list(DATES.strftime('%Y-%m-%d')).index(dt)
        x=archives['4-2'][key][i].sum()/scale
        if key=='final':
            y=archives['4-3']['original'][i].sum()/scale
            z=archives['4-3']['final'][i].sum()/scale
            vals.extend([f'{x:.0f}',f'{y:.0f}/{z:.0f}'])
        else: vals.extend([f'{x:.2f}',f"{archives['4-3'][key][i].sum()/scale:.2f}"])
    purchase_rows.append(label+' & '+' & '.join(vals))
# Split by scenario to retain legible numbers, especially original/final pairs.
purchase_tex=''
for scenario in ['4-2','4-3']:
    body=[]
    for slot in [60,72,84,96,108,120]:
        values=[]
        for dt in selected:
            i=list(DATES.strftime('%Y-%m-%d')).index(dt);a=archives[scenario]
            v=f"{a['final'][i,slot]:.2f}"
            if scenario=='4-3':v=f"{a['original'][i,slot]:.2f}/"+v
            values.append(v)
        body.append(f'{slot//6:02d}:00--{slot//6:02d}:10 & '+' & '.join(values))
    for key,label in [('original','全天原计划电量'),('final','全天常规购电量'),('fees','全天总费用/元')]:
        if scenario=='4-2' and key=='original':continue
        vals=[f"{archives[scenario][key][list(DATES.strftime('%Y-%m-%d')).index(dt)].sum():.2f}" for dt in selected]
        body.append(label+' & '+' & '.join(vals))
    caption=f'方案{scenario}指定时段与全天购电结果（电量单位：kWh）'
    purchase_tex+=table(caption,'tab:q4-purchase-'+scenario,'lrrrr','时间段 & 3月20日 & 6月21日 & 9月23日 & 12月21日',body)
    if scenario=='4-3':purchase_tex+='\\noindent\\small 表中十分钟槽的“原计划/最终量”分别表示午夜原计划和日内调整后的常规购电量；全天常规电量不含紧急购电。\\normalsize\n'
(OUT/'purchase_table.tex').write_text(purchase_tex)

storage_tex=''
for scenario in ['4-2','4-3']:
    body=[]
    for h in range(0,24,4):
        values=[]
        for dt in selected:
            i=list(DATES.strftime('%Y-%m-%d')).index(dt);a=archives[scenario]
            values.append(f"{a['charge'][i,h*6:(h+4)*6].sum():.2f}/{a['discharge'][i,h*6:(h+4)*6].sum():.2f}")
        body.append(f'{h:02d}:00--{h+4:02d}:00 & '+' & '.join(values))
    for pos,label in [(0,'0:00储电量'),(-1,'24:00储电量')]:
        body.append(label+' & '+' & '.join(f"{archives[scenario]['states'][list(DATES.strftime('%Y-%m-%d')).index(dt),pos]:.2f}" for dt in selected))
    storage_tex+=table(f'方案{scenario}指定日期的储能运行结果（单位：kWh）','tab:q4-storage-'+scenario,'lrrrr','时间段 & 3月20日 & 6月21日 & 9月23日 & 12月21日',body)
(OUT/'storage_tables.tex').write_text(storage_tex)

def clock(slot):return f'{slot//6:02d}:{slot%6*10:02d}'
all_events=[]
emergency_tex=''
for scenario in ['4-2','4-3']:
    events=[]
    for dt in selected:
        i=list(DATES.strftime('%Y-%m-%d')).index(dt);e=archives[scenario]['emergency'][i]
        spans=[];start=None
        for t in range(145):
            active=t<144 and e[t]>1e-8
            if active and start is None:start=t
            if not active and start is not None:
                amount=float(e[start:t].sum());span=f'{clock(start)}--{clock(t)}'
                spans.append((span,amount));all_events.append(dict(scenario=scenario,date=dt,start=clock(start),end=clock(t),kwh=amount));start=None
        assert abs(sum(v for _,v in spans)-e.sum())<1e-6
        events.append(spans)
    body=[]
    for row in range(max(map(len,events))):
        cells=[]
        for spans in events:
            cells.extend([spans[row][0],f'{spans[row][1]:.2f}'] if row<len(spans) else ['--','--'])
        body.append(' & '.join(cells))
    cells=[]
    for spans in events:cells.extend(['合计',f'{sum(v for _,v in spans):.2f}'])
    body.append(' & '.join(cells))
    head=' & '.join(f'\\multicolumn{{2}}{{c}}{{{name}}}' for name in ['3月20日','6月21日','9月23日','12月21日'])+'\\\\\n'+' & '.join(['时间段','电量']*4)
    tex=table(f'方案{scenario}指定日期的紧急购电区间（电量单位：kWh）','tab:q4-emergency-'+scenario,'lr'*4,head,body)
    tex=tex.replace('\\centering\\small','\\centering\\footnotesize\n\\setlength{\\tabcolsep}{5pt}')
    emergency_tex+=tex
(OUT/'emergency_tables.tex').write_text(emergency_tex)
pd.DataFrame(all_events).to_csv(OUT/'selected_emergency.csv',index=False,float_format='%.10f')
print('specified emergency groups:',pd.DataFrame(all_events).groupby(['scenario','date']).size().to_dict())
