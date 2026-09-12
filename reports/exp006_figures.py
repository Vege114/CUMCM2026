"""Standalone PNG/SVG publication figures from the same reviewed report rows."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/cumcm-exp006-mpl')
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from reports.exp006_evidence import REPORT,EVIDENCE,LABELS,SPECIFIED,TARGETS

BLUE='#357c89';GOLD='#aa6d37';GRAY='#737b80';PINK='#a75964';PURPLE='#8b7c9e'

def build_figures(frames=None):
    f=frames or {p.stem:pd.read_csv(p) for p in EVIDENCE.glob('*.csv')}
    for p in [Path('/System/Library/Fonts/PingFang.ttc'),Path('/System/Library/Fonts/STHeiti Medium.ttc')]:
        if p.exists():font_manager.fontManager.addfont(p);plt.rcParams['font.family']=font_manager.FontProperties(fname=p).get_name();break
    plt.rcParams.update({'font.size':10,'axes.unicode_minus':False,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    out=REPORT/'figures';out.mkdir(exist_ok=True)
    def save(fig,name):
        for ext in ['png','svg']:fig.savefig(out/f'{name}.{ext}',dpi=180,bbox_inches='tight')
        p=out/f'{name}.svg';p.write_text('\n'.join(x.rstrip() for x in p.read_text().splitlines())+'\n');plt.close(fig)
    def color(r):
        name=str(r.get('name',r.get('policy_id','')))
        return GOLD if name in ['primary','exp006/primary'] and r.get('experiment')=='exp006' else BLUE if 'greedy' in name else PURPLE if name.startswith('causal') else GRAY
    def bars(ax,p,key,unit,scale=1):
        p=p.copy();values=p[key].to_numpy(dtype=float)/scale;positions=np.arange(len(p))
        ax.barh(positions,values,color=[color(r) for r in p.to_dict('records')],height=.6)
        ax.set_yticks(positions,p.label);ax.invert_yaxis();ax.set_xlabel(unit);ax.grid(axis='x',alpha=.14)
        maximum=np.nanmax(values) if np.isfinite(values).any() else 1;ax.set_xlim(0,maximum*1.19 if maximum>0 else 1)
        for i,v in enumerate(values):
            if np.isfinite(v):ax.text(v+maximum*.012,i,f'{v:,.2f}',va='center',fontsize=9)
    costs=f['cost_history'];ids=['exp001/legacy_rebased','exp002/primary','exp002/new_deterministic','exp003/primary','exp004/no_season','exp004/causal_season','exp006/primary','exp006/greedy_execution','exp006/fixed_primary_greedy']
    p=costs[costs.policy_id.isin(ids)]
    fig,axes=plt.subplots(2,1,figsize=(12,9),layout='constrained',gridspec_kw={'height_ratios':[4,1]})
    bars(axes[0],p,'total_cost','总购电费（万元）',10000);axes[0].set_title('同物理结算口径 · 2025年2–12月334日 · 主种子42')
    excluded=costs[~costs.ranking_allowed];bars(axes[1],excluded,'total_cost','总购电费（万元）',10000);axes[1].set_title('单独保留原值：exp001原协议／oracle未来信息，不作排名')
    save(fig,'history-costs')
    fig,axes=plt.subplots(2,2,figsize=(14,10),layout='constrained')
    for ax,field,title in zip(axes.flat,['planned_cost','emergency_cost','planned_kwh','emergency_kwh'],['计划购电费（万元）','紧急购电费（万元）','计划购电量（万kWh）','紧急购电量（万kWh）']):bars(ax,p,field,title,10000);ax.set_title(title)
    save(fig,'absolute-performance')
    runs=f['all_runs'];p=runs[runs.id.isin(['primary','cost_only','no_tree','more_throughput_penalty','zero_terminal','greedy_execution','no_deadband','primary_grid25','fixed_primary_greedy'])]
    fig,ax=plt.subplots(figsize=(12,6),layout='constrained');bars(ax,p,'total_cost','真实购电费（万元）',10000);ax.set_title('预先指定的主组、消融与网格核查 · 不按全年结果改主组');save(fig,'ablation-costs')
    b=f['battery_history'];b=b[b.policy_id.isin(['exp002/primary','exp003/primary','exp004/no_season','exp004/causal_season','exp006/primary','exp006/greedy_execution','exp006/fixed_primary_greedy'])]
    fig,axes=plt.subplots(1,3,figsize=(17,6),layout='constrained')
    for ax,key,title,unit,scale in zip(axes,['throughput_kwh','direction_reversals','equivalent_full_cycles'],['交流侧吞吐','非空方向反转','电芯侧等效满循环'],['万kWh','次','EFC'],[10000,1,1]):bars(ax,b,key,unit,scale);ax.set_title(title)
    fig.suptitle('费用优先之下的辅助操作强度 · 同时充放电槽数均为0 · 非寿命预测');save(fig,'battery-comparison')
    p=f['monthly_cost'];fig,axes=plt.subplots(2,1,figsize=(12,7),sharex=True,layout='constrained')
    for name,col in [('primary',GOLD),('greedy_execution',BLUE),('causal_42',PURPLE),('fixed_primary_greedy',GRAY)]:
        d=p[p.name==name];axes[0].plot(d.month,d.total_cost/10000,'o-',color=col,label=LABELS[name],lw=1.3);axes[1].plot(d.month,d.emergency_kwh/10000,'o-',color=col,lw=1.3)
    axes[0].set_ylabel('总费（万元）');axes[1].set_ylabel('紧急购电（万kWh）');axes[1].set_xticks(range(2,13));axes[1].set_xlabel('2025月份');axes[0].legend(ncol=2)
    for ax in axes:ax.grid(alpha=.14)
    save(fig,'monthly-costs')
    p=f['seed_costs'];fig,ax=plt.subplots(figsize=(11,5),layout='constrained')
    for i,(family,col) in enumerate([('无季节预测',GOLD),('历史季节预测',PURPLE)]):
        d=p[p.family==family].sort_values('seed');x=np.arange(len(d))+(i-.5)*.30;y=d.total_cost/10000;ax.bar(x,y,width=.29,label=family,color=col)
        for xx,yy in zip(x,y):ax.text(xx,yy+6,f'{yy:.2f}',ha='center',fontsize=9)
    ax.set_xticks(range(3),['42','2026','3407']);ax.set_ylabel('真实总费（万元）');ax.set_xlabel('固定种子；42为正式，未择优');ax.legend(ncol=2);ax.set_ylim(0,p.total_wan.max()*1.13);ax.grid(axis='y',alpha=.14);save(fig,'seed-costs')
    forecast=f['forecast_history'];forecast=forecast[(forecast.population=='all')&~forecast.exploratory]
    fig,axes=plt.subplots(2,3,figsize=(19,11),layout='constrained')
    for j,target in enumerate(TARGETS):
        p=forecast[forecast.target==target]
        for i,(metric,unit) in enumerate([('rmse','kW'),('wape_pct','%')]):bars(axes[i,j],p,metric,unit);axes[i,j].set_title(TARGETS[target]+('RMSE' if i==0 else 'WAPE'))
    fig.suptitle('午夜同48096槽 · exp006完全复用exp004预测 · 保留误差退步');save(fig,'history-errors')
    timing=f['timings'];fig,axes=plt.subplots(1,3,figsize=(18,7),layout='constrained')
    for ax,stage in zip(axes,['训练','检查点预测','调度执行']):
        p=timing[(timing.stage==stage)&timing.seconds.notna()&((timing.experiment!='exp006')|timing.label.isin([LABELS['primary'],LABELS['greedy_execution'],'exp006 复用预测']))].copy();bars(ax,p,'seconds','累计秒数');ax.set_title(stage)
    fig.suptitle('描述性比较：设备、组数与计时范围不同；exp006新训练/推理为0');save(fig,'stage-timings')
    daily=f['daily'];worst=daily[daily.id=='primary'].sort_values('total_cost').iloc[-1].date;d=f['detail'];d=d[(d.name=='primary')&(d.date==worst)]
    fig,axes=plt.subplots(3,1,figsize=(12,9),sharex=True,layout='constrained')
    for key,label,col in [('actual_net_kwh','实际净需求',GRAY),('forecast_net_kwh','预测净需求',PURPLE),('planned_kwh','固定购电',BLUE),('emergency_kwh','紧急购电',PINK)]:axes[0].plot(d.hour,d[key],label=label,color=col,lw=1.2)
    axes[0].legend(ncol=4);axes[0].set_title('正式组最贵日 '+worst);axes[0].set_ylabel('每10分钟kWh')
    axes[1].plot(d.hour,d.charge_kwh,label='充电',color=BLUE);axes[1].plot(d.hour,d.discharge_kwh,label='放电',color=GOLD);axes[1].legend();axes[1].set_ylabel('每10分钟kWh')
    axes[2].plot(d.hour,d.soc_kwh,color=BLUE);axes[2].set_ylim(0,12000);axes[2].set_ylabel('SOC（kWh）');axes[2].set_xlabel('小时')
    for value in [1200,10800]:axes[2].axhline(value,color=GRAY,ls='--',lw=.7)
    for ax in axes:ax.set_xlim(0,24);ax.set_xticks([0,6,12,18,24]);ax.grid(alpha=.14)
    save(fig,'failure-case')
    fig,axes=plt.subplots(4,2,figsize=(14,12),layout='constrained',sharex=True)
    for i,date in enumerate(SPECIFIED):
        d=f['detail'];d=d[(d.date==date)&(d.name=='primary')]
        for key,label,col in [('actual_net_kwh','实际净需求',GRAY),('planned_kwh','固定购电',BLUE),('emergency_kwh','紧急购电',PINK)]:axes[i,0].plot(d.hour,d[key],label=label,color=col,lw=1.1)
        axes[i,0].set_title(date+' 供需');axes[i,0].set_ylabel('每10分钟kWh');axes[i,1].plot(d.hour,d.soc_kwh,color=BLUE);axes[i,1].set_title(date+' 实际SOC');axes[i,1].set_ylim(0,12000);axes[i,1].set_ylabel('kWh')
        for val in [1200,10800]:axes[i,1].axhline(val,color=GRAY,ls='--',lw=.6)
    axes[0,0].legend(ncol=3,fontsize=9)
    for ax in axes.flat:ax.set_xlim(0,24);ax.set_xticks([0,6,12,18,24]);ax.grid(alpha=.13)
    axes[-1,0].set_xlabel('小时');axes[-1,1].set_xlabel('小时');save(fig,'specified-days')
    fig,ax=plt.subplots(figsize=(12,6),layout='constrained');ax.axis('off')
    steps=[('冻结预测','exp004 144×2 kW\n无季节seed42正式'),('条件误差树','最近28个完整历史日\n5特征/深度5/叶48'),('边际支持','每槽9个等权分位点\n不代表联合日路径'),('SOC动态规划','50kWh网格+实际初SOC\n购电80%分位+次要惩罚'),('锁定计划','144槽原计划全日不变\n充放意图仅单方向'),('实际执行与结算','本槽观测后物理裁剪\n总费Σp(g+5e)')]
    for i,(head,body) in enumerate(steps):
        row=i//3;col=i%3 if row==0 else 2-i%3;x=.16+col*.335;y=.78-row*.52
        ax.text(x,y,head+'\n\n'+body,ha='center',va='center',fontsize=12,bbox={'boxstyle':'round,pad=.7','facecolor':'#f7f9f9','edgecolor':BLUE if row==0 else GOLD,'linewidth':1})
        if row==0 and col<2:ax.annotate('',xy=(x+.215,y),xytext=(x+.13,y),arrowprops={'arrowstyle':'->','color':GRAY})
        if row==1 and col>0:ax.annotate('',xy=(x-.215,y),xytext=(x-.13,y),arrowprops={'arrowstyle':'->','color':GRAY})
    ax.annotate('',xy=(.83,.42),xytext=(.83,.62),arrowprops={'arrowstyle':'->','color':GRAY});ax.set_title('规划代理与真实结算必须分开 · 实际末SOC连续进入下一天',pad=15);save(fig,'technical-route')

if __name__=='__main__':build_figures()
