"""Standalone publication figures, all using the report's reviewed dataframes."""
import os

os.environ.setdefault('MPLCONFIGDIR','/tmp/cumcm-exp004-mpl')
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

from reports.exp004_evidence import EVIDENCE, LABELS, REPORT, SPECIFIED, TARGETS

COLORS={'no_season':'#357c89','causal_season':'#aa6d37','oracle_season':'#8b7c9e'}


def build_figures(frames=None):
    f=frames or {p.stem:pd.read_csv(p) for p in EVIDENCE.glob('*.csv')}
    for p in [Path('/System/Library/Fonts/PingFang.ttc'),Path('/System/Library/Fonts/STHeiti Medium.ttc')]:
        if p.exists():
            font_manager.fontManager.addfont(p);plt.rcParams['font.family']=font_manager.FontProperties(fname=p).get_name();break
    plt.rcParams.update({'font.size':10,'axes.unicode_minus':False,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    out=REPORT/'figures';out.mkdir(exist_ok=True)
    def save(fig,name):
        for ext in ['png','svg']:fig.savefig(out/f'{name}.{ext}',dpi=170,bbox_inches='tight')
        vector=out/f'{name}.svg'
        vector.write_text('\n'.join(line.rstrip() for line in vector.read_text().splitlines())+'\n')
        plt.close(fig)
    def bars(ax,part,field,unit,scale=1):
        values=part[field].to_numpy()/scale;labels=part.label.to_list();positions=np.arange(len(part))
        colors=[COLORS.get(str(r.get('name')),'#9aa4ab') for r in part.to_dict('records')]
        ax.barh(positions,values,color=colors,height=.58)
        ax.set_yticks(positions,labels);ax.invert_yaxis();ax.set_xlabel(unit);ax.set_xlim(0,np.nanmax(values)*1.20);ax.grid(axis='x',alpha=.15)
        for i,v in enumerate(values):
            if np.isfinite(v):ax.text(v+np.nanmax(values)*.015,i,f'{v:,.2f}',va='center',fontsize=9)
    costs=f['cost_history'];causal=costs[costs.physically_comparable&~costs.exploratory]
    fig,axes=plt.subplots(2,1,figsize=(11,8),layout='constrained',gridspec_kw={'height_ratios':[3,1]})
    bars(axes[0],causal,'total_cost','总购电费（万元）',10000);axes[0].set_title('历次绝对费用 · 2025年2–12月334天 · 主种子42')
    excluded=costs[~costs.physically_comparable|costs.exploratory].copy()
    excluded.loc[excluded.exploratory,'label']='exp004 全年探索（含未来）'
    bars(axes[1],excluded,'total_cost','总购电费（万元）',10000);axes[1].set_title('单独保留原值：物理口径不同／未来信息，不作正式排名')
    save(fig,'history-costs')
    fig,axes=plt.subplots(2,2,figsize=(14,9),layout='constrained')
    for ax,field,title,scale,unit in zip(axes.flat,['planned_cost','emergency_cost','planned_kwh','emergency_kwh'],['计划购电费','紧急购电费','计划购电量','紧急购电量'],[10000,10000,10000,10000],['万元','万元','万kWh','万kWh']):
        bars(ax,causal,field,unit,scale);ax.set_title(title)
    save(fig,'absolute-performance')
    p=f['forecast_history'];p=p[(p.population=='all')&~p.exploratory]
    fig,axes=plt.subplots(2,3,figsize=(16,9),layout='constrained')
    for j,target in enumerate(TARGETS):
        part=p[p.target==target]
        for i,(field,unit) in enumerate([('rmse','kW'),('wape_pct','%')]):
            bars(axes[i,j],part,field,unit);axes[i,j].set_title(TARGETS[target]+('RMSE' if i==0 else 'WAPE'))
    fig.suptitle('午夜同48,096槽位 · 误差退步一并展示')
    save(fig,'history-errors')
    fig,axes=plt.subplots(1,3,figsize=(16,5),layout='constrained')
    for ax,stage in zip(axes,['训练','检查点预测','调度执行']):
        part=f['timings'];part=part[(part.stage==stage)&part.seconds.notna()].copy();bars(ax,part,'seconds','累计秒数');ax.set_title(stage)
    fig.suptitle('不同设备／任务组数，仅描述性比较；缺失阶段见原表')
    save(fig,'stage-timings')
    fig,axes=plt.subplots(2,1,figsize=(11,6),layout='constrained',sharex=True)
    for variant,part in f['seasonal_shifts'].groupby('variant',sort=False):
        for ax,field in zip(axes,['load_shift_kw','pv_shift_kw']):ax.plot(pd.to_datetime(part.date),part[field],label=LABELS[variant],color=COLORS[variant],lw=1.4)
    for ax,title in zip(axes,['负载年内修正（kW）','PV年内修正（kW）']):ax.set_ylabel(title);ax.axhline(0,color='#939393',lw=.7);ax.grid(alpha=.13)
    axes[0].legend(ncol=3);axes[0].set_title('目标日相对周期参考日的拟合季节修正 · 全年探索含未来')
    save(fig,'seasonal-shifts')
    fig,axes=plt.subplots(2,1,figsize=(11,7),layout='constrained',sharex=True)
    p=f['monthly_cost'];p=p[p.seed==42]
    for name,part in p.groupby('name',sort=False):
        axes[0].plot(part.month,part.total_cost/10000,marker='o',label=LABELS[name],color=COLORS[name])
        axes[1].plot(part.month,part.emergency_kwh/10000,marker='o',label=LABELS[name],color=COLORS[name])
    axes[0].set_ylabel('总费（万元）');axes[1].set_ylabel('紧急量（万kWh）');axes[1].set_xticks(range(2,13));axes[1].set_xlabel('2025月份');axes[0].legend(ncol=3)
    for ax in axes:ax.grid(alpha=.15)
    save(fig,'monthly-costs')
    fig,ax=plt.subplots(figsize=(10,5),layout='constrained');p=f['seed_costs']
    for i,name in enumerate(COLORS):
        part=p[p.name==name].set_index('seed').loc[[42,2026,3407]];v=part.total_cost/10000;x=np.arange(3)+(i-1)*.23
        ax.bar(x,v,width=.22,label=LABELS[name],color=COLORS[name])
        for a,b in zip(x,v):ax.text(a,b+10,f'{b:.2f}',ha='center',fontsize=9)
    ax.set_xticks(range(3),['42','2026','3407']);ax.set_xlabel('固定随机种子');ax.set_ylabel('总购电费（万元）');ax.set_ylim(0,1550);ax.legend(ncol=3,loc='upper center',bbox_to_anchor=(.5,1.10));ax.grid(axis='y',alpha=.15)
    save(fig,'seed-costs')
    fig,axes=plt.subplots(4,2,figsize=(13,11),layout='constrained',sharex=True)
    detail=f['detail']
    for i,date in enumerate(SPECIFIED):
        d=detail[detail.date==date]
        truth=d[d.name=='no_season']
        for j,target in enumerate(['load','pv']):
            ax=axes[i,j];ax.plot(truth.hour,truth['actual_'+target+'_kw'],color='#414141',label='实际',lw=1.5)
            for name,part in d.groupby('name',sort=False):ax.plot(part.hour,part['forecast_'+target+'_kw'],color=COLORS[name],label=LABELS[name],lw=1)
            ax.set_title(date+' '+TARGETS[target]);ax.set_ylabel('kW');ax.grid(alpha=.12);ax.set_xticks([0,6,12,18,24]);ax.set_xlim(0,24)
    axes[0,0].legend(fontsize=8,ncol=2);axes[-1,0].set_xlabel('小时');axes[-1,1].set_xlabel('小时')
    save(fig,'specified-forecasts')
    worst=p[(p.name=='causal_season')&(p.seed==42)].iloc[0].worst_date
    d=detail[(detail.date==worst)&(detail.name=='causal_season')]
    fig,axes=plt.subplots(2,1,figsize=(11,6),layout='constrained',sharex=True)
    for field,label,color in [('actual_net_kwh','实际净需求','#414141'),('forecast_net_kwh','预测净需求','#aa6d37'),('planned_kwh','计划购电','#357c89'),('emergency_kwh','紧急购电','#a75964')]:axes[0].plot(d.hour,d[field],label=label,color=color,lw=1.3)
    axes[0].set_title('历史季节组最贵日 '+worst);axes[0].set_ylabel('每十分钟kWh');axes[0].legend(ncol=4)
    axes[1].plot(d.hour,d.soc_kwh,color='#357c89');axes[1].set_ylabel('SOC（kWh）');axes[1].set_xlabel('小时');axes[1].set_ylim(0,12000)
    for value in [1200,10800]:axes[1].axhline(value,ls='--',color='#999999',lw=.8)
    for ax in axes:ax.grid(alpha=.15);ax.set_xticks([0,6,12,18,24]);ax.set_xlim(0,24)
    save(fig,'failure-case')


if __name__=='__main__':build_figures()
