"""Tables and figures from frozen Q1 stage two / newly computed annual arrays.

This command never solves a dispatch model or trains a predictor.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'Xelatex/q4_run/vendor'))
from setup_style import setup_style
from export_figure import export_figure
from visual_qa import audit_layout
from experiments.exp008.q1 import audit, read_data


def style():
    setup_style(journal='general', lang='zh', serif_for_zh=True, use_sciplots=False)
    font = ROOT/'Xelatex/YaHei.Consolas.1.11b.ttf'
    fm.fontManager.addfont(str(font))
    name = fm.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({'font.family':[name, 'DejaVu Sans'], 'font.size':9, 'figure.dpi':100,
        'axes.labelsize':9, 'axes.titlesize':9, 'legend.fontsize':8,
        'xtick.labelsize':8, 'ytick.labelsize':8, 'axes.unicode_minus':False})


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def time(t):
    return f'{t//6:02d}:{t%6*10:02d}'


def figure(fig, name):
    # Freeze constrained-layout positions on the final Agg renderer before
    # checking text; SVG/PDF renderers otherwise leave stale text bounds.
    fig.canvas.draw()
    qa = audit_layout(fig)
    output = export_figure(fig, str(name), formats=['pdf','svg','png'],
        size_inches=tuple(fig.get_size_inches()), dpi=320, grayscale_preview=True)
    write(name.with_suffix('.layout.json'), qa)
    plt.close(fig)
    return output


def q1():
    out = ROOT/'Xelatex/q1_run'
    revised = json.loads((out/'revised.json').read_text())
    base = json.loads((out/'baseline.json').read_text())
    data=read_data()
    for result in (revised, base):
        result['trajectory']={k:np.asarray(v) for k,v in result['trajectory'].items()}
        checked=audit(data,result)
        assert checked['violations']==0
    q=revised['trajectory']; m=revised['metrics']; b=base['metrics']
    summary={'stage':2, 'source_sha256':hashlib.sha256((out/'revised.json').read_bytes()).hexdigest(),
        'metrics':m, 'baseline_metrics':b, 'stages':revised['stages'],
        'specified':[{'interval':f'{time(t)}--{time(t+1)}', 'purchase_kwh':float(q['g'][t])} for t in (60,72,84,96,108,120)],
        'storage_blocks':[{'interval':f'{4*j}:00--{4*(j+1)}:00',
            'charge_kwh':float(q['c'][24*j:24*(j+1)].sum()/6),
            'discharge_kwh':float(q['d'][24*j:24*(j+1)].sum()/6)} for j in range(6)],
        'initial_kwh':float(q['E'][0]), 'final_kwh':float(q['E'][-1]),
        'engineering_cost_increment':revised['stages'][0]['cost']-base['stages'][0]['cost'],
        'stage2_cost_increment':m['cost']-revised['stages'][0]['cost'],
        'tv_reduction_pct':100*(1-m['tv']/b['tv']),
        'total_cost_increment':m['cost']-b['cost'], 'total_cost_increment_pct':100*(m['cost']/b['cost']-1)}
    write(out/'summary.json',summary)
    write(ROOT/'data/results/exp009/q1/summary.json',summary)
    p=summary['specified']; z=summary['storage_blocks']
    lines=[r'\begin{table}[H]\centering\small',r'\caption{两层混合整数规划求解结果}\label{tab:q1-stages}',
        r'\begin{tabular}{lrrrr}\toprule 层次&购电费/元&购电量/kWh&$F_2$&MIP间隙\\\midrule']
    for row in revised['stages']:
        lines.append(f"第{row['stage']}层&{row['cost']:.2f}&{row['quantity']:.4f}&{row['smooth']:.9f}&{row['gap']:.0f}"+r'\\')
    lines += [r'\bottomrule\end{tabular}\end{table}',
        r'\begin{table}[H]\centering\small', r'\caption{指定时段购电量及全天购电结果}\label{tab:q1-purchase}',
        r'\begin{tabular}{lrlrlr}\toprule 时间段&购电量/kWh&时间段&购电量/kWh&时间段&购电量/kWh\\\midrule']
    for start in (0,3):
        lines.append('&'.join(f"{x['interval']}&{x['purchase_kwh']:.4f}" for x in p[start:start+3])+r'\\')
    lines += [r'\midrule',rf"\multicolumn{{3}}{{l}}{{全天购电量：{m['quantity']:.4f}\,kWh}}&\multicolumn{{3}}{{l}}{{全天购电费：{m['cost']:.2f}元}}\\",
        r'\bottomrule\end{tabular}\end{table}',r'\begin{table}[H]\centering\small',
        r'\caption{储能分时充放电量与起止储电量}\label{tab:q1-storage}',
        r'\begin{tabular}{lrrlrr}\toprule 时间段&充电量/kWh&放电量/kWh&时间段&充电量/kWh&放电量/kWh\\\midrule']
    for start in (0,2,4):
        lines.append('&'.join(f"{x['interval']}&{x['charge_kwh']:.4f}&{x['discharge_kwh']:.4f}" for x in z[start:start+2])+r'\\')
    lines += [r'\midrule\multicolumn{3}{l}{0:00储电量：6000.0000\,kWh}&\multicolumn{3}{l}{24:00储电量：6000.0000\,kWh}\\',
        r'\bottomrule\end{tabular}\end{table}',r'\begin{table}[H]\centering\small',
        r'\caption{储能调度方案的经济性与运行指标}\label{tab:q1-compare}',
        r'\begin{tabular}{lrr}\toprule 指标&经济基准&两层规划方案\\\midrule']
    for label,key,digits in [('购电费/元','cost',2),('购电量/kWh','quantity',4),('循环功率总变差/kW','tv',4),
                            ('最大相邻净功率变化/kW','max_step',4),('运行段启动次数/次','starts',0),('短于30分钟的运行段数/段','short_runs',0)]:
        lines.append(f'{label}&{b[key]:.{digits}f}&{m[key]:.{digits}f}'+r'\\')
    lines += [f"转换损耗/kWh&{b['charge']-b['discharge']:.4f}&{m['charge']-m['discharge']:.4f}"+r'\\',r'\bottomrule\end{tabular}\end{table}']
    (out/'tables.tex').write_text(('\n'.join(lines)+'\n').replace('-0.0000', '0.0000'))
    style(); edges=np.arange(145)/6
    fig, axes=plt.subplots(2,1,figsize=(7.0,4.8),sharex=True)
    fig.subplots_adjust(left=.13,right=.985,bottom=.12,top=.86,hspace=.11)
    for key,result,color,ls in [('经济基准',base,'#D55E00','--'),('两层规划方案',revised,'#0072B2','-')]:
        tr=result['trajectory']
        axes[0].stairs(tr['c']-tr['d'],edges,label=key,color=color,linestyle=ls,lw=1.1)
        axes[1].plot(edges,tr['E'],label=key,color=color,linestyle=ls,lw=1.1)
    axes[0].set_ylabel('净充电功率 / kW');axes[0].legend(loc='upper center',bbox_to_anchor=(.5,1.2),ncol=2,frameon=False)
    axes[1].set(xlabel='时刻 / h',ylabel='储电量 / kWh',xlim=(0,24),xticks=np.arange(0,25,2))
    for ax in axes:ax.grid(alpha=.18)
    figure(fig,out/'q1_dispatch')
    return summary


def q2():
    source=ROOT/'data/results/exp009/q2'
    status=json.loads((source/'summary.json').read_text())
    assert status['complete'] and status['days']==365
    with np.load(source/'dispatch.npz',allow_pickle=False) as z:
        a={k:z[k].copy() for k in z.files}
    assert np.array_equal(a['days'],np.arange(365))
    output=ROOT/'Xelatex/final_results/q2'
    output.mkdir(parents=True,exist_ok=True)
    dates=('2025-03-20','2025-06-21','2025-09-23','2025-12-21')
    specified, lines, emergencies = [], [], []
    for date in dates:
        day=(pd.Timestamp(date)-pd.Timestamp('2025-01-01')).days
        label=date.replace('-',''); zh=date.replace('-','年',1).replace('-','月')+'日'
        quantities=[float(a['original'][day,t]) for t in (60,72,84,96,108,120)]
        blocks=[{'start':24*j,'stop':24*(j+1),
            'charge_kwh':float(a['charge'][day,24*j:24*(j+1)].sum()),
            'discharge_kwh':float(a['discharge'][day,24*j:24*(j+1)].sum())} for j in range(6)]
        spans=[]; e=a['emergency'][day]; active=e>1e-6
        boundaries=np.diff(np.r_[False,active,False].astype(int))
        for start, stop in zip(np.flatnonzero(boundaries==1),np.flatnonzero(boundaries==-1)):
            spans.append({'start_slot':int(start),'stop_slot':int(stop),'interval':f'{time(start)}--{time(stop)}','kwh':float(e[start:stop].sum())})
        emergencies.append(spans)
        detail={'date':date,'day':day,'specified_purchase_kwh':quantities,'blocks':blocks,
            'planned_kwh':float(a['original'][day].sum()),'emergency_kwh':float(e.sum()),
            'total_cost':float(a['fees'][day].sum()),'initial_soc':float(a['states'][day,0]),
            'final_soc':float(a['states'][day,-1]),'emergency_intervals':spans}
        specified.append(detail)
        lines += [rf'{zh}的购电与储能结果见表\ref{{tab:q2-purchase-{label}}}和表\ref{{tab:q2-battery-{label}}}。',
            r'\begin{table}[H]\centering\small',
            rf'\caption{{{zh}指定时段计划购电量及全天费用}}\label{{tab:q2-purchase-{label}}}',
            r'\begin{tabular}{lrlrlr}\toprule 时间段&购电量&时间段&购电量&时间段&购电量\\\midrule']
        times=(60,72,84,96,108,120)
        for start in (0,3):
            lines.append('&'.join(f'{time(times[i])}--{time(times[i]+1)}&{quantities[i]:.4f}' for i in range(start,start+3))+r'\\')
        lines += [r'\midrule',rf"\multicolumn{{3}}{{l}}{{全天计划购电量：{detail['planned_kwh']:.4f}\,kWh}}&\multicolumn{{3}}{{l}}{{全天购电费：{detail['total_cost']:.2f}元}}\\",
            r'\bottomrule\end{tabular}\end{table}',r'\begin{table}[H]\centering\small',
            rf'\caption{{{zh}指定时段充放电量及起止储电量}}\label{{tab:q2-battery-{label}}}',
            r'\begin{tabular}{lrrlrr}\toprule 时间段&充电量&放电量&时间段&充电量&放电量\\\midrule']
        for start in (0,2,4):
            lines.append('&'.join(f"{time(x['start'])}--{time(x['stop'])}&{x['charge_kwh']:.4f}&{x['discharge_kwh']:.4f}" for x in blocks[start:start+2])+r'\\')
        lines += [r'\midrule',rf"\multicolumn{{3}}{{l}}{{0:00储电量：{detail['initial_soc']:.4f}\,kWh}}&\multicolumn{{3}}{{l}}{{24:00储电量：{detail['final_soc']:.4f}\,kWh}}\\",
            r'\bottomrule\end{tabular}\end{table}']
    lines += [r'\begin{table}[H]\centering\small',r'\caption{四个指定日期的连续紧急购电区间及电量}\label{tab:q2-emergency}',
        r'\setlength{\tabcolsep}{3pt}\begin{tabular}{lrlrlrlr}\toprule',
        r'\multicolumn{2}{c}{3月20日}&\multicolumn{2}{c}{6月21日}&\multicolumn{2}{c}{9月23日}&\multicolumn{2}{c}{12月21日}\\',
        r'时间段&电量&时间段&电量&时间段&电量&时间段&电量\\\midrule']
    for i in range(max(map(len,emergencies))):
        lines.append('&'.join(f"{spans[i]['interval']}&{spans[i]['kwh']:.4f}" if i<len(spans) else '--&--' for spans in emergencies)+r'\\')
    lines += [r'\bottomrule\end{tabular}\end{table}']
    metrics=[]
    for name,sl in [('1—12月',slice(0,365)),('2—12月',slice(31,365))]:
        fees=a['fees'][sl].sum((0,1))
        metrics.append({'period':name,'days':len(a['days'][sl]),'planned_kwh':float(a['original'][sl].sum()),
            'emergency_kwh':float(a['emergency'][sl].sum()),'total_grid_kwh':float((a['final'][sl]+a['emergency'][sl]).sum()),
            'planned_cost':float(fees[0]),'emergency_cost':float(fees[3]),'total_cost':float(fees.sum())})
    lines += [r'表\ref{tab:q2-annual}同时给出全年账单与附件5要求区间的账单，二者不混用。',
        r'\begin{table}[H]\centering\small',r'\caption{问题二不同统计区间的购电量与实际费用}\label{tab:q2-annual}',
        r'\begin{tabular}{lrr}\toprule 指标&1--12月（365日）&2--12月（334日）\\\midrule']
    for label,key,digits in [('计划购电量/kWh','planned_kwh',4),('紧急购电量/kWh','emergency_kwh',4),
        ('外网合计购电量/kWh','total_grid_kwh',4),('计划购电费/元','planned_cost',2),('紧急购电费/元','emergency_cost',2),('合计费用/元','total_cost',2)]:
        lines.append(f'{label}&{metrics[0][key]:.{digits}f}&{metrics[1][key]:.{digits}f}'+r'\\')
    lines += [r'\bottomrule\end{tabular}\end{table}']
    # Midnight-issued full-pipeline errors, scored on the same 48,096 slots.
    pred=pd.read_csv(ROOT/'data/results/exp008/forecast_hgb_extra_trees_half/metrics.csv')
    pred=pred[pred.name=='hgb_extra_trees_half_ridge28_memory']
    lines += [r'预测误差采用2--12月每日午夜最终输出，所有槽各计一次，样本数均为48096；结果见表\ref{tab:q2-prediction}。',
        r'\begin{table}[H]\centering\small',r'\caption{午夜最终供需预测的误差（2--12月，不重叠日窗口）}\label{tab:q2-prediction}',
        r'\begin{tabular}{lrrr}\toprule 变量&MAE/kW&RMSE/kW&偏差/kW\\\midrule']
    for _, row in pred.iterrows():
        label={'load':'负载','pv':'光伏','net':'净负载'}[row.channel]
        lines.append(f'{label}&{row.mae_kw:.4f}&{row.rmse_kw:.4f}&{row.bias_kw:.4f}'+r'\\')
    lines += [r'\bottomrule\end{tabular}\end{table}']
    (output/'tables.tex').write_text(('\n'.join(lines)+'\n').replace('-0.0000','0.0000'))
    checks=json.loads((source/'verification.json').read_text())
    audits=json.loads((source/'planning_audit.json').read_text())
    limit_days=sum(r['mip']['status']==1 for r in audits)
    success=sum(r['refinement']['success'] for r in audits)
    b=status['feb_dec_battery']; gap=status['mip_gap_max']*100
    text=(f"全年实际购电费为{metrics[0]['total_cost']:.2f}元，其中1月费用为{status['january_cost']:.2f}元，"
        f"2--12月费用为{metrics[1]['total_cost']:.2f}元。2月1日初始储电量为{status['feb_initial_soc']:.4f}\\,kWh，"
        "由一月同一策略的实际末状态连续承接，未重新设为6000\\,kWh。"
        f"四个指定日期共出现{sum(map(len,emergencies))}个连续紧急购电区间；购电不足由紧急补购补齐后，"
        "全部52560槽均满足供需平衡、功率与储电量约束。\n\n"
        f"2--12月实际非空方向反转为{b['direction_reversals']}次，活动槽为{b['active_slots']}个，"
        f"交流侧充放电吞吐为{b['throughput_kwh']:.4f}\\,kWh。反转先剔除空闲动作，再在评价区间内部连续计数，"
        "不计入一月与二月的连接边界。这些量分别反映运行强度的不同侧面，不将计划模式变化预算解释为所有实际动作指标都会下降。\n\n"
        f"365次模式规划均取得可行解，其中{limit_days}次达到求解时限，最大实际相对间隙为{gap:.4f}\\%；"
        f"固定模式修正中{success}次按求解器成功条件停止。模式规划累计耗时{status['mip_seconds_sum']:.2f}\\,s，"
        f"购电修正累计耗时{status['refinement_seconds_sum']:.2f}\\,s；"
        "这些是各阶段求解累计时间，不包含模型训练与制图，也不是多进程运行的总墙钟时间。"
        f"实际账单逐槽复算最大残差为{checks['max_slot_fee_error_yuan']:.2e}元。\n")
    (output/'discussion.tex').write_text(text)
    payload={'summary':status,'periods':metrics,'specified':specified,'verification':checks,
        'archive_sha256':hashlib.sha256((source/'dispatch.npz').read_bytes()).hexdigest(),
        'mip_limit_days':limit_days,'refinement_success_days':success}
    write(output/'summary.json',payload)
    write(source/'paper_summary.json',payload)
    daily=pd.read_csv(source/'daily.csv'); daily['month']=pd.to_datetime(daily.date).dt.month
    month=daily.groupby('month')[['planned_cost','emergency_cost']].sum()
    month.to_csv(output/'monthly.csv')
    style(); fig,ax=plt.subplots(figsize=(7.0,3.2))
    fig.subplots_adjust(left=.115,right=.985,bottom=.19,top=.81)
    x=month.index.to_numpy(); y=month.planned_cost.to_numpy()/1e4; e=month.emergency_cost.to_numpy()/1e4
    ax.bar(x,y,color='#0072B2',label='计划购电费',width=.62)
    ax.bar(x,e,bottom=y,color='#D55E00',label='紧急购电费',width=.62)
    ax.set(xlabel='月份',ylabel='月购电费 / 万元',xticks=x,ylim=(0,None))
    ax.legend(loc='upper center',bbox_to_anchor=(.5,1.19),ncol=2,frameon=False)
    ax.grid(axis='y',alpha=.16)
    figure(fig,output/'q2_monthly')
    return payload


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--q1',action='store_true')
    parser.add_argument('--q2',action='store_true')
    args=parser.parse_args()
    if args.q1: print(json.dumps(q1(),ensure_ascii=False))
    if args.q2: print(json.dumps(q2(),ensure_ascii=False))
