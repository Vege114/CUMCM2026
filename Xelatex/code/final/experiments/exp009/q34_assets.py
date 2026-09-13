"""Rebuild paper tables and figures from the newly verified annual arrays."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.dates as mdates
from experiments.exp009.q34_run import ROOT, OUT, save, verify, digest
from experiments.exp009.q34_forecast import SelectedForecasts
from experiments.exp009.q12_forecast import YearForecasts

sys.path.insert(0,str(ROOT/'Xelatex/q4_run/vendor'))
from setup_style import setup_style
from export_figure import export_figure
from visual_qa import audit_layout

PAPER=ROOT/'Xelatex/final_results'
DATES=pd.date_range('2025-02-01','2025-12-31')
SELECTED=[78-31,171-31,265-31,354-31]
LABELS=['3月20日','6月21日','9月23日','12月21日']
COLORS=['#0072B2','#D55E00','#009E73']


def style():
    setup_style(journal='general',lang='zh',serif_for_zh=True,use_sciplots=False)
    fp=ROOT/'Xelatex/YaHei.Consolas.1.11b.ttf';fm.fontManager.addfont(str(fp))
    name=fm.FontProperties(fname=str(fp)).get_name()
    plt.rcParams.update({'font.family':[name,'DejaVu Sans'],'font.size':9,'axes.labelsize':9,
        'axes.titlesize':9,'legend.fontsize':8,'xtick.labelsize':8,'ytick.labelsize':8,
        'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False,
        'figure.constrained_layout.use':False,'figure.autolayout':False})


def finish(fig,name):
    fig.canvas.draw()
    qa=audit_layout(fig);save(PAPER/(name+'.layout.json'),qa)
    assert not any(level in ('WARN','FAIL') for level,_ in qa),qa
    export_figure(fig,str(PAPER/name),formats=['pdf','png','svg'],size_inches=tuple(fig.get_size_inches()),
        dpi=320,grayscale_preview=False,tight=False)
    # The inherited helper's grayscale branch re-saves the color PNG tightly,
    # changing its physical size. Convert the already exported exact-size PNG.
    from PIL import Image
    source=Image.open(PAPER/(name+'.png'))
    source.convert('L').save(PAPER/(name+'_grayscale.png'),dpi=source.info['dpi'])
    plt.close(fig)


def table(caption,label,spec,header,body):
    return ('\\begin{table}[H]\n\\centering\\small\n\\caption{'+caption+'}\\label{'+label+'}\n'
        '\\begin{tabular}{'+spec+'}\n\\toprule\n'+header+'\\\\\n\\midrule\n'+
        '\\\\\n'.join(body)+'\\\\\n\\bottomrule\n\\end{tabular}\n\\end{table}\n')


def clock(slot):return f'{slot//6:02d}:{slot%6*10:02d}'


def load(scenario):
    if scenario=='4-2':
        summary=json.loads((OUT/'q4_2/summary.json').read_text())
        assert summary['complete'] and summary['days']==365,'Q4-2 full year must finish first'
        path=OUT/'q4_2/dispatch.npz'
        with np.load(path) as z:a={k:z[k][31:].copy() for k in z.files}
    else:
        path=OUT/('q3' if scenario=='3' else 'q4_3')/f'dispatch_{scenario}.npz'
        with np.load(path) as z:a={k:z[k].copy() for k in z.files}
    assert np.array_equal(a['days'],np.arange(31,365))
    summary=verify(a,float(a['states'][0,0]));summary['source_sha256']=digest(path)
    return a,summary


def specified_tables(a,scenario,prefix):
    result=''
    for key,description in [('original','原计划'),('final','最终常规')]:
        if key=='original' and scenario=='4-2':continue
        body=[]
        for slot in (60,72,84,96,108,120):
            body.append(f'{clock(slot)}--{clock(slot+1)} & '+' & '.join(f'{a[key][i,slot]:.4f}' for i in SELECTED))
        body.append('全天电量 & '+' & '.join(f'{a[key][i].sum():.4f}' for i in SELECTED))
        if key=='final':body.append('全天总费用/元 & '+' & '.join(f'{a["fees"][i].sum():.2f}' for i in SELECTED))
        result+=table(f'问题{scenario}指定日期的{description}购电量（单位：kWh）',f'tab:{prefix}-{key}',
            'lrrrr','时间段 & '+' & '.join(LABELS),body)
    # Retain four-decimal numbers with a separate narrow table per date.
    for i,date in zip(SELECTED,LABELS):
        body=[]
        for start in range(0,144,24):
            body.append(f'{clock(start)}--{clock(start+24)} & {a["charge"][i,start:start+24].sum():.4f} & {a["discharge"][i,start:start+24].sum():.4f}')
        body.append(f'0:00储电量 & \\multicolumn{{2}}{{c}}{{{a["states"][i,0]:.4f}}}')
        body.append(f'24:00储电量 & \\multicolumn{{2}}{{c}}{{{a["states"][i,-1]:.4f}}}')
        result+=table(f'问题{scenario}在{date}的储能结果（单位：kWh）',f'tab:{prefix}-storage-{i}',
            'lrr','时间段 & 充电量 & 放电量',body)
    all_spans=[]
    for i in SELECTED:
        e=a['emergency'][i];edges=np.diff(np.r_[False,e>1e-6,False].astype(int))
        all_spans.append([(f'{clock(int(start))}--{clock(int(stop))}',float(e[start:stop].sum()))
            for start,stop in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1))])
    # Split paired dates to retain legibility at 4 decimal places.
    for block in (0,2):
        body=[]
        for j in range(max(len(all_spans[block]),len(all_spans[block+1]))):
            cells=[]
            for spans in all_spans[block:block+2]:
                cells.extend([spans[j][0],f'{spans[j][1]:.4f}'] if j<len(spans) else ['--','--'])
            body.append(' & '.join(cells))
        cells=[]
        for spans in all_spans[block:block+2]:cells.extend(['合计',f'{sum(v for _,v in spans):.4f}'])
        body.append(' & '.join(cells))
        result+=table(f'问题{scenario}指定日期紧急购电区间（单位：kWh）',f'tab:{prefix}-emergency-{block}',
            'lrlr',f'\\multicolumn{{2}}{{c}}{{{LABELS[block]}}} & \\multicolumn{{2}}{{c}}{{{LABELS[block+1]}}}\\\\\n时间段 & 电量 & 时间段 & 电量',body)
    (PAPER/f'{prefix}_specified.tex').write_text(result)


def q3():
    a,s=load('3');specified_tables(a,'3','q3')
    rows=[]
    for label,value,digits in [('原计划费/元',s['billing']['planned_cost'],2),('上调费/元',s['billing']['up_cost'],2),
        ('下调退款/元',s['billing']['down_refund'],2),('紧急购电费/元',s['billing']['emergency_cost'],2),
        ('总费用/元',s['total_cost'],2),('最终常规购电量/kWh',s['final_regular_kwh'],4),
        ('下调电量/kWh',s['down_kwh'],4),('紧急购电量/kWh',s['emergency_kwh'],4),
        ('非空方向反转/次',s['battery']['direction_reversals'],0)]:rows.append(f'{label} & {value:.{digits}f}')
    (PAPER/'q3_annual.tex').write_text(table('问题三评价期购电费用与运行结果','tab:q3-annual','lr','指标 & 数值',rows))
    metrics=pd.read_csv(OUT/'q34_forecast_metrics.csv');metrics=metrics[metrics.scenario.astype(str)=='3']
    body=[]
    for target,name in [('load','负载'),('pv','光伏'),('net','净需求')]:
        x=metrics[(metrics.target==target)&(metrics['mode']=='midnight')].iloc[0]
        y=metrics[(metrics.target==target)&(metrics['mode']=='successive_nonoverlap_6h')].iloc[0]
        body.append(f'{name} & {x.mae:.4f} & {y.mae:.4f} & {x.rmse:.4f} & {y.rmse:.4f}')
    (PAPER/'q3_forecast.tex').write_text(table('同一48096槽上午夜保持与日内更新的预测误差（单位：kW）',
        'tab:q3-prediction','lrrrr','变量 & 午夜MAE & 更新MAE & 午夜RMSE & 更新RMSE',body))
    for key,title,ylabel in [('purchase','购电','购电量/kWh'),('storage','储能','储电量/kWh')]:
        fig,axes=plt.subplots(2,2,figsize=(6.4,4.1),sharex=True)
        for ax,i,date in zip(axes.flat,SELECTED,LABELS):
            if key=='purchase':
                ax.stairs(a['original'][i],np.arange(145)/6,baseline=None,lw=.85,color=COLORS[0],label='午夜原计划')
                ax.stairs(a['final'][i],np.arange(145)/6,baseline=None,lw=.9,ls='--',color=COLORS[1],label='最终常规量')
            else:
                ax.plot(np.arange(145)/6,a['states'][i],lw=1,color=COLORS[0])
                for y in (1200,10800):ax.axhline(y,lw=.6,ls=':',color='#777777')
                ax.set_ylim(0,12000)
            for h in (6,12,18):ax.axvline(h,lw=.5,ls=':',color='#aaaaaa')
            ax.set(xlim=(0,24),xticks=(0,6,12,18,24),xlabel='时刻/h',ylabel=ylabel)
            ax.text(.5,1.04,date,transform=ax.transAxes,ha='center',va='bottom',fontsize=9)
        fig.subplots_adjust(left=.14,right=.98,bottom=.12,top=.87,hspace=.52,wspace=.48)
        for ax in axes[0]:ax.set_xlabel('')
        for ax in axes[:,1]:ax.set_ylabel('')
        if key=='purchase':fig.legend(*axes.flat[0].get_legend_handles_labels(),loc='upper center',ncol=2,frameon=False)
        finish(fig,f'q3_{key}')
    # One continuous signed field shows downward refunds alongside upward purchases.
    delta=a['final']-a['original'];limit=float(np.quantile(np.abs(delta),.995))
    fig,ax=plt.subplots(figsize=(6.4,3.35))
    fig.subplots_adjust(left=.11,right=.81,bottom=.18,top=.97)
    plot=ax.imshow(delta,aspect='auto',origin='lower',extent=(0,24,0,334),cmap='RdBu_r',vmin=-limit,vmax=limit)
    ticks=[int(np.flatnonzero(DATES.month==m)[0]) for m in (2,4,6,8,10,12)]
    ax.set(xlabel='时刻/h',ylabel='月份',xticks=(0,6,12,18,24),yticks=ticks,yticklabels=('2','4','6','8','10','12'))
    cax=fig.add_axes([.85,.20,.025,.72])
    fig.colorbar(plot,cax=cax,label='最终量减原计划/kWh',extend='both')
    finish(fig,'q3_adjustments')
    save(PAPER/'q3_summary.json',s)


def q4():
    a,sa=load('4-2');b,sb=load('4-3');archives={'4-2':a,'4-3':b}
    for scenario,values in archives.items():specified_tables(values,scenario,'q4_'+scenario)
    rows=[]
    for label,key,digits in [('原计划费/元','planned_cost',2),('上调费/元','up_cost',2),('下调退款/元','down_refund',2),('紧急购电费/元','emergency_cost',2)]:
        rows.append(f'{label} & {sa["billing"][key]:.{digits}f} & {sb["billing"][key]:.{digits}f}')
    for label,key,digits in [('总费用/元','total_cost',2),('最终常规购电量/kWh','final_regular_kwh',4),('紧急购电量/kWh','emergency_kwh',4)]:
        rows.append(f'{label} & {sa[key]:.{digits}f} & {sb[key]:.{digits}f}')
    for label,key,digits in [('非空方向反转/次','direction_reversals',0),('电芯等效满循环/次','equivalent_full_cycles',4)]:
        rows.append(f'{label} & {sa["battery"][key]:.{digits}f} & {sb["battery"][key]:.{digits}f}')
    (PAPER/'q4_annual.tex').write_text(table('波动电价下两类完整方案的评价期结果','tab:q4-annual','lrr','指标 & 方案4-2 & 方案4-3',rows))
    price42=YearForecasts('4-2');price43=SelectedForecasts()
    pa=np.stack([price42.get(day)['price'] for day in range(31,365)])
    pb=np.stack([price43.get(day,0,'4-3')['price'] for day in range(31,365)])
    actual=a['price'];price_results={}
    fig,axes=plt.subplots(1,2,figsize=(6.4,2.8),gridspec_kw={'width_ratios':[1.4,1]})
    fig.subplots_adjust(left=.13,right=.98,bottom=.23,top=.80,wspace=.56)
    for predicted,color,label,marker in [(pb,COLORS[1],'8维周期模型','s'),(pa,COLORS[0],'10维供需模型','o')]:
        err=predicted-actual
        monthly=[float(np.sqrt(np.mean(err[DATES.month==m]**2))) for m in range(2,13)]
        axes[0].plot(np.arange(2,13),monthly,color=color,lw=1,marker=marker,ms=3,label=label)
        price_results[label]={'mae':float(np.abs(err).mean()),'rmse':float(np.sqrt(np.mean(err**2)))}
    axes[0].set(xlabel='月份',ylabel='午夜电价RMSE/(元/kWh)',xticks=np.arange(2,13,2))
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',ncol=2,frameon=False)
    boxes=axes[1].boxplot([np.abs(pb-actual).ravel(),np.abs(pa-actual).ravel()],tick_labels=['8维周期','10维供需'],
        showfliers=False,whis=(5,95),patch_artist=True,widths=.5)
    for box,color in zip(boxes['boxes'],[COLORS[1],COLORS[0]]):box.set_facecolor(color);box.set_alpha(.5)
    axes[1].set(ylabel='绝对误差/(元/kWh)')
    finish(fig,'q4_price')
    fig,axes=plt.subplots(1,2,figsize=(6.4,2.7),gridspec_kw={'width_ratios':[1.5,1]})
    fig.subplots_adjust(left=.12,right=.98,bottom=.23,top=.93,wspace=.60)
    monthly={}
    for j,(scenario,values) in enumerate(archives.items()):
        cost=np.array([values['fees'][DATES.month==m].sum() for m in range(2,13)])
        emergency=np.array([values['fees'][DATES.month==m,:,3].sum() for m in range(2,13)])
        monthly[scenario]={'cost':cost.tolist(),'emergency_cost':emergency.tolist()}
        axes[0].bar(np.arange(2,13)+(j-.5)*.36,cost/1e4,width=.36,color=COLORS[j],label='方案'+scenario)
        axes[1].plot(np.arange(2,13),emergency/cost*100,color=COLORS[j],lw=1,ls=['-','--'][j],marker=['o','s'][j],ms=3)
    axes[0].set(xlabel='月份',ylabel='月总费用/万元',xticks=np.arange(2,13,2));axes[0].legend(frameon=False)
    axes[1].set(xlabel='月份',ylabel='紧急费占总费用/%',xticks=np.arange(2,13,2),ylim=(0,None))
    finish(fig,'q4_monthly')
    i=265-31;fig,axes=plt.subplots(4,1,figsize=(6.4,5.2),sharex=True);edges=np.arange(145)/6
    fig.subplots_adjust(left=.15,right=.98,bottom=.10,top=.91,hspace=.19)
    axes[0].stairs(actual[i],edges,baseline=None,color='#333333',lw=1,label='实际电价')
    axes[0].stairs(pa[i],edges,baseline=None,color=COLORS[0],lw=.9,ls='--',label='4-2午夜预测')
    axes[0].stairs(pb[i],edges,baseline=None,color=COLORS[1],lw=.9,ls=':',label='4-3午夜预测')
    axes[0].legend(loc='lower center',bbox_to_anchor=(.5,1.02),ncol=3,frameon=False);axes[0].set(ylabel='电价\n元/kWh')
    for j,(scenario,values) in enumerate(archives.items()):
        axes[1].stairs(values['final'][i],edges,baseline=None,color=COLORS[j],lw=.9,ls=['-','--'][j],label='方案'+scenario)
        axes[2].stairs(6*(values['charge'][i]-values['discharge'][i])/1000,edges,baseline=None,color=COLORS[j],lw=.8,ls=['-','--'][j])
        axes[3].plot(edges,values['states'][i]/1000,color=COLORS[j],lw=1,ls=['-','--'][j])
    axes[1].set(ylabel='常规购电\nkWh',ylim=(0,1.25*max(a['final'][i].max(),b['final'][i].max())))
    axes[1].legend(frameon=False,ncol=2,loc='upper right')
    axes[2].set(ylabel='电池功率\nMW',ylim=(-5.5,5.5));axes[2].axhline(0,color='#999999',lw=.5)
    axes[3].set(ylabel='储电量\nMWh',xlabel='时刻/h',ylim=(0,12),xlim=(0,24),xticks=(0,6,12,18,24))
    for y in (1.2,10.8):axes[3].axhline(y,color='#777777',lw=.6,ls=':')
    for ax in axes:
        for h in (6,12,18):ax.axvline(h,color='#aaaaaa',lw=.5,ls=':')
    finish(fig,'q4_dispatch')
    fig,ax=plt.subplots(figsize=(6.4,2.7))
    fig.subplots_adjust(left=.15,right=.98,bottom=.23,top=.95)
    difference=a['fees'].sum(axis=(1,2))-b['fees'].sum(axis=(1,2))
    ax.plot(DATES,difference,lw=.8,color=COLORS[0])
    ax.axhline(0,color='#777777',lw=.7,ls='--')
    ax.set(xlabel='日期',ylabel='4-2费用减4-3费用/元',xlim=(DATES[0],DATES[-1]))
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(2,4,6,8,10,12)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m月'))
    finish(fig,'q4_daily_difference')
    comparison={'4-2':sa,'4-3':sb,'cost_difference':sa['total_cost']-sb['total_cost'],
        'cost_difference_pct':100*(sa['total_cost']-sb['total_cost'])/sa['total_cost'],
        'emergency_reduction_pct':100*(sa['emergency_kwh']-sb['emergency_kwh'])/sa['emergency_kwh'],
        'prices':price_results,'monthly':monthly}
    save(PAPER/'q4_summary.json',comparison)
    report42=json.loads((OUT/'q4_2/summary.json').read_text())
    completion43=json.loads((OUT/'q4_3/completion.json').read_text())
    text=(
        f'两方案均从1月1日6000\\,kWh按各自正式策略连续运行；1月历史不足时采用周期预测，随后逐步积累同一预测流程的历史误差。'
        f'2月1日期初储电量分别为{a["states"][0,0]:.4f}和{b["states"][0,0]:.4f}\\,kWh，'
        f'评价期均为2月1日至12月31日334日、48096槽，末日实际储电量分别为{a["states"][-1,-1]:.4f}和{b["states"][-1,-1]:.4f}\\,kWh。'
        f'方案4-2的混合整数规划单次限时5\\,s、目标间隙0.2\\%，固定模式购电修正最多120次迭代。'
        f'其全年365次规划均取得可行解，混合整数规划与固定模式修正的累计耗时分别为{report42["mip_seconds_sum"]:.2f}、{report42["refinement_seconds_sum"]:.2f}\\,s，'
        f'实际最大相对间隙为{100*report42["mip_gap_max"]:.4f}\\%。这些间隙只度量单次代理规划，不构成全年真实费用的最优性界。'
        f'方案4-3采用HiGHS求解1460次线性规划，均获得可行解，全年运行与归档耗时{completion43["wall_seconds"]:.2f}\\,s。\n\n'
        f'图\\ref{{fig:q4-price}}按同一48096个午夜预测槽比较价格模型，8维模型RMSE为{price_results["8维周期模型"]["rmse"]:.5f}元/kWh，'
        f'10维模型为{price_results["10维供需模型"]["rmse"]:.5f}元/kWh。价格误差的下降反映预测精度改善，经济影响还需经过购电和储能决策。\n\n'
        f'评价期账单见表\\ref{{tab:q4-annual}}。方案4-2与4-3总费用分别为{sa["total_cost"]:.2f}元、{sb["total_cost"]:.2f}元，'
        f'差额为{comparison["cost_difference"]:.2f}元（{comparison["cost_difference_pct"]:.2f}\\%）。'
        f'方案4-3下调退款为{-sb["billing"]["down_refund"]:.2f}元，紧急购电量为{sb["emergency_kwh"]:.4f}\\,kWh；'
        f'与方案4-2相比紧急购电量减少{comparison["emergency_reduction_pct"]:.2f}\\%。'
        f'两者的非空方向反转分别为{sa["battery"]["direction_reversals"]}与{sb["battery"]["direction_reversals"]}次，'
        '这些指标仅计评价期内动作，是运行强度的描述，不能直接换算电池寿命。\n')
    (PAPER/'q4_narrative.tex').write_text(text)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--question',choices=('q3','q4','all'),default='all');args=parser.parse_args()
    PAPER.mkdir(parents=True,exist_ok=True);style()
    if args.question in ('q3','all'):q3()
    if args.question in ('q4','all'):q4()
