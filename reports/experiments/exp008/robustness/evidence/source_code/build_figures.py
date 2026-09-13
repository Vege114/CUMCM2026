"""Reproducible standalone, paper-ready figures from measured robustness tables."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'.cache/exp008-robustness-mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageDraw

DATA = ROOT/'data/results/exp008/robustness'
OUT = ROOT/'paper/figures/exp008_robustness'
BLUE, ORANGE, GOLD, OLIVE = '#27658A', '#C06A36', '#B48E2F', '#687A43'
COLORS = [BLUE, ORANGE, GOLD, OLIVE]
META=[]
BOOK=None
plt.rcParams.update({'font.family':'sans-serif', 'font.sans-serif':['Arial Unicode MS','DejaVu Sans'],
    'font.size':11, 'axes.titlesize':12, 'axes.labelsize':11,
    'axes.spines.top':False, 'axes.spines.right':False, 'axes.edgecolor':'#60656B',
    'axes.labelcolor':'#272C30', 'text.color':'#272C30', 'xtick.color':'#41474C',
    'ytick.color':'#41474C', 'axes.unicode_minus':False, 'grid.color':'#E0E3E5',
    'grid.linewidth':.65, 'lines.linewidth':1.8, 'lines.markersize':5,
    'savefig.facecolor':'white', 'figure.facecolor':'white', 'pdf.fonttype':42,
    'ps.fonttype':42, 'svg.fonttype':'path'})


def table(section, name):
    return pd.read_csv(DATA/section/name)


def axes_grid(nrows=1, ncols=1, width=8.6, height=4.5):
    fig, axes = plt.subplots(nrows, ncols, figsize=(width,height), squeeze=False)
    for ax in axes.ravel():
        ax.grid(axis='y', zorder=0)
        ax.set_axisbelow(True)
    return fig, axes


def finish(fig, name, title, caption, sources, note='', rect=(0, .07, 1, .94)):
    fig.suptitle(title, x=.5, y=.985, fontsize=14, fontweight='normal')
    if note:
        fig.text(.055,.018,note, fontsize=9, color='#5E666C', va='bottom')
    fig.tight_layout(rect=rect, pad=1.2)
    for ext in ('png','pdf','svg'):
        fig.savefig(OUT/f'{name}.{ext}', dpi=450 if ext=='png' else 150)
    if BOOK:
        BOOK.savefig(fig)
    META.append(dict(id=name,title=title,caption=caption,sources=sources,
        outputs={ext:str((OUT/f'{name}.{ext}').relative_to(ROOT)) for ext in ('png','pdf','svg')}))
    plt.close(fig)


def execution():
    x=table('execution','stress_metrics.csv')
    fig, axes=axes_grid(1,2,width=10,height=4.9)
    for ax,families in zip(axes[0], [('load_bias','pv_bias'),('combined_bias',)]):
        for j,family in enumerate(families):
            s=x[x.family==family]
            labels={'load_bias':'负荷单独变化','pv_bias':'光伏单独变化','combined_bias':'负荷 +b、光伏 −b'}
            ax.plot(100*s.level,s.cost_change_pct, marker=['o','s'][j],color=COLORS[j],label=labels[family])
        ax.axhline(0,color='#6B7075',lw=.8)
        ax.axvline(0,color='#6B7075',lw=.8,ls=':')
        ax.set(xlabel='实况持续偏移 b / %',ylabel='费用相对原方案变化 / %')
        ax.legend(frameon=False,loc='upper left')
    axes[0,0].set_title('(a) 单通道扰动')
    axes[0,1].set_title('(b) 联合扰动')
    finish(fig,'12_execution_bias','固定购电与模式计划下的持续偏差压力测试',
        '对334日实况施加持续乘性偏差，保持原购电量与模式序列，按扰动实况逐槽执行并连续传递自身SOC。正的联合偏移表示负荷增加、光伏减少。费用相对未扰动exp008计算，未与未扰动exp006比较。',
        ['execution/stress_metrics.csv'],note='334日条件回放；购电和模式不重新优化，纵轴范围因面板不同而不同。')

    fig,axes=axes_grid(1,3,width=11,height=4.7)
    settings=[('capacity','容量 / kWh',12000.),('power','功率上限 / kW',5000.),('efficiency','往返效率 / %',100.)]
    for ax,(family,label,scale) in zip(axes[0],settings):
        s=x[x.family==family]
        ax.plot(s.level*scale,s.cost_change_pct,'o-',color=BLUE)
        ax.axhline(0,color='#6B7075',lw=.8)
        center=12000 if family=='capacity' else 5000 if family=='power' else 90
        ax.axvline(center,color='#6B7075',ls=':',lw=1)
        ax.set(xlabel=label,ylabel='费用变化 / %')
        ax.ticklabel_format(axis='x',style='plain',useOffset=False)
    finish(fig,'13_hardware_sensitivity','电池参数变化对固定计划执行费用的影响',
        '容量±5%和±10%、功率上限±5%和±10%、往返效率85%—95%的单因素条件回放。改变容量时保持初始SOC比例，SOC上下限保持10%—90%；费用均相对冻结Q2未扰动方案。',
        ['execution/stress_metrics.csv'],note='虚线为原参数；每种设定连续执行334日，所有设定均通过物理约束核验。')

    fig,axes=axes_grid(width=8.2,height=4.8)
    ax=axes[0,0]
    noise=x[x.family=='correlated_noise']
    levels=sorted(noise.level.unique())
    data=[noise[noise.level==v].cost_change_pct.to_numpy() for v in levels]
    bp=ax.boxplot(data,positions=np.arange(4),widths=.45,patch_artist=True,showfliers=False,
                  medianprops={'color':BLUE,'linewidth':2},boxprops={'edgecolor':BLUE},
                  whiskerprops={'color':BLUE},capprops={'color':BLUE})
    for box in bp['boxes']:
        box.set_facecolor('#E7EFF4')
    jitter=np.random.default_rng(42)
    for i, values in enumerate(data):
        ax.scatter(i+jitter.uniform(-.14,.14,len(values)),values,s=12,color=BLUE,alpha=.55,zorder=3)
    ax.axhline(0,color='#6B7075',lw=.8)
    ax.set(xticks=np.arange(4),xticklabels=[f'{v*100:g}%' for v in levels],
           xlabel='乘性噪声的对数标准差',ylabel='费用相对原方案变化 / %')
    finish(fig,'14_correlated_noise','相关随机扰动下的费用分布',
        '采用乘性对数正态噪声，并将乘性因子均值校正为1；高斯对数扰动的槽间AR(1)系数为0.95，负荷与光伏两通道扰动的相关系数为−0.3。四种强度各30次，共用随机数。箱线图与散点描述假设压力分布，不是数据估计的置信区间；光伏夜间零值保留。',
        ['execution/stress_metrics.csv','execution/noise_summary.csv'],
        note='固定购电和模式的334日回放；每组30个假设情景，箱体为四分位距，散点为全部情景。')


def q1_figures():
    x=table('q1','sensitivity.csv')
    x=x[x.eligible_for_comparison==True].copy()
    baseline=x[x.name=='center'].iloc[0]
    s=x[x.parameter.isin(['center','delta'])].sort_values('delta')
    fig,axes=axes_grid(2,2,width=9.4,height=7.8)
    ax=axes[0,0]
    ax.plot(s.delta*100,s.cost,'o-',color=BLUE)
    ax.axhline(baseline.stage1_cost,color='#646A70',ls='--',lw=1,label='第一阶段最低费用')
    ax.set(xlabel='相对费用容差 δ / %',ylabel='第二阶段费用 / 元')
    ax.legend(frameon=False,fontsize=10)
    ax=axes[0,1]
    ax.plot(s.delta*100,s.tv,'s-',color=ORANGE)
    ax.set(xlabel='相对费用容差 δ / %',ylabel='功率总变差 / kW')
    ax=axes[1,0]
    ax.plot(s.delta*100,s.smooth_change_pct_from_frozen,'o-',color=BLUE)
    ax.axhline(0,color='#646A70',lw=.8)
    ax.set(xlabel='相对费用容差 δ / %',ylabel='综合平稳目标 S 变化 / %')
    ax=axes[1,1]
    ax.plot(s.delta*100,s.starts,'s-',color=ORANGE)
    ax.set(xlabel='相对费用容差 δ / %',ylabel='充放电启动次数 / 次',yticks=range(3,8))
    for ax in axes.ravel():
        ax.axvline(.1,color='#646A70',ls=':',lw=1)
    finish(fig,'04_q1_budget','第一问：费用容差与调度平稳性',
        '固定其余设定，仅改变第二阶段相对费用预算δ。每种设定均先求第一阶段最低费用，再在对应费用预算内求平稳轨迹。虚线标示最终δ=0.1%；分别报告费用、包含跨午夜差分的循环日净功率总变差、综合平稳目标S和充放电启动次数。',
        ['q1/sensitivity.csv'],note='附件1单日；中心解逐值复现最终成果，展示通过物理核验的求解结果。')

    fig,axes=axes_grid(2,3,width=11,height=7.4)
    for ax,(parameter,label,center,mult) in zip(axes.ravel(),[
        ('ramp','爬坡上限 / kW·(10min)⁻¹',1000.,1.),('up','最短开启时间 / min',3,10),
        ('down','同模式最短关闭 / min',2,10),('eta_rt','往返效率 / %',.9,100),
        ('capacity_factor','容量 / kWh',1.,12000.)]):
        g=x[x.parameter==parameter].copy()
        values=np.r_[g[parameter].to_numpy(),center]*mult
        costs=np.r_[g.cost_change_pct_from_frozen.to_numpy(),0.]
        costs[np.abs(costs)<1e-9]=0.
        order=np.argsort(values)
        ax.plot(values[order],costs[order],'o-',color=BLUE)
        ax.axhline(0,lw=.8,color='#646A70')
        ax.axvline(center*mult,lw=1,ls=':',color='#646A70')
        ax.set(xlabel=label,ylabel='费用相对原方案变化 / %')
        if np.max(np.abs(costs)) < 1e-8:
            ax.set_ylim(-.01,.01)
            ax.text(.5,.75,'变化小于数值容差',ha='center',transform=ax.transAxes,fontsize=10)
        ax.ticklabel_format(axis='x',style='plain',useOffset=False)
    ax=axes.ravel()[-1]
    ax.axis('off')
    ax.text(.04,.85,'每次仅改变一个参数\n\n两阶段同步改变物理约束\n\n容量变化保持 SOC 比例\n\n各面板独立纵轴',va='top',fontsize=12)
    finish(fig,'05_q1_parameters','第一问：运行约束与硬件参数敏感性',
        '爬坡上限、最短开启/关闭时间、往返效率和容量的单因素重优化。每组费用预算均相对该组自身第一阶段最低费用计算，不将不同约束下的预算混用。',
        ['q1/sensitivity.csv'],note='附件1单日的两阶段重新优化；不能将单日规律直接外推为全年规律。',rect=(0,.065,1,.95))


def forecast_figures():
    x=table('forecast','sensitivity_annual.csv')
    x=x[x.channel=='net'].copy()
    base=x[x.name=='baseline']
    fig,axes=axes_grid(2,2,width=10,height=7.7)
    for ax,(factor,label,center) in zip(axes.ravel(),[
        ('weight','原始 HGB 融合权重',.5),('window','校准窗口 / 日（半衰期为窗口一半）',28),
        ('penalty_multiplier','Ridge 正则强度倍率',1.),('gain','昨日负荷残差记忆系数',.5)]):
        s=pd.concat([base,x[x.factor==factor]]).sort_values(factor)
        for metric,color,marker,label2 in [('rmse_kw_change_pct',BLUE,'o','槽级净负荷 RMSE'),
            ('daily_energy_rmse_kwh_change_pct',ORANGE,'s','日累计净能量 RMSE')]:
            ax.plot(s[factor],s[metric],marker=marker,color=color,label=label2)
        ax.axhline(0,lw=.8,color='#646A70')
        ax.axvline(center,lw=1,ls=':',color='#646A70')
        ax.set(xlabel=label,ylabel='误差相对原参数变化 / %')
    axes[0,0].legend(frameon=False,fontsize=9)
    finish(fig,'01_forecast_sensitivity','第二问：融合与因果后处理参数敏感性',
        '在同一334日已发布原始HGB/ExtraTrees预测上逐日重算Ridge与非递归记忆层。每次只改变指定参数族，窗口变化按原规则同时改变半衰期。所有通道使用同一融合权重；中心参数逐元素复现冻结预测。正值为误差增加，负值为误差降低；不据此重新挑选最终参数。',
        ['forecast/sensitivity_annual.csv'],note='334日开发期敏感性；虚线为最终参数。校准与记忆仅使用各发布时刻之前的实况。',rect=(0,.065,1,.95))

    from matplotlib.colors import LinearSegmentedColormap
    cv=table('forecast','cv_metrics.csv')
    s=cv[(cv.channel=='net')&(cv.test_days==28)]
    fig,axes=axes_grid(1,2,width=10,height=4.9)
    fields=['rmse_kw','rmse_kw_change_pct_vs_val7']
    for ax,field in zip(axes[0],fields):
        matrix=s.pivot(index='fold_month',columns='validation_days',values=field)
        ax.grid(False)
        arr=matrix.to_numpy()
        if field=='rmse_kw':
            cmap=LinearSegmentedColormap.from_list('blue_paper',['#F3F6F8',BLUE])
            im=ax.imshow(arr,cmap=cmap,aspect='auto')
            title='(a) 净负荷 RMSE / kW'
        else:
            vmax=max(np.abs(arr).max(),.01)
            cmap=LinearSegmentedColormap.from_list('signed_paper',[BLUE,'#F6F5F3',ORANGE])
            im=ax.imshow(arr,cmap=cmap,aspect='auto',vmin=-vmax,vmax=vmax)
            title='(b) 相对 7 日验证方案变化 / %'
        for (i,j),value in np.ndenumerate(arr):
            color='white' if im.norm(value)>.73 or (field!='rmse_kw' and im.norm(value)<.23) else '#222B31'
            ax.text(j,i,f'{value:.2f}',ha='center',va='center',color=color,fontsize=12)
        ax.set(xticks=range(len(matrix.columns)),xticklabels=[str(v) for v in matrix.columns],
            yticks=range(len(matrix.index)),yticklabels=[f'{m}月1日起' for m in matrix.index],
            xlabel='验证窗口 / 日',title=title)
        fig.colorbar(im,ax=ax,shrink=.8,pad=.035)
    finish(fig,'02_temporal_cv','训练／验证划分改变后的真实重训回测',
        '以4月、7月和10月1日为3个起点，每次重训4个树模型，验证窗口取7/14/21个完整日，随后28日逐日发布预测并用已完成日期更新历史特征和后处理。训练、验证、测试按时间顺序隔离；7日验证配置重新训练后复现冻结月份预测。',
        ['forecast/cv_metrics.csv','forecast/cv_training_audit.json'],note='每格28日；验证窗加长也减少训练日数。28日内逐日发布并更新已完成的历史输入。')

    pooled=table('forecast','cv_pooled_metrics.csv')
    pooled=pooled[pooled.channel=='net']
    fig,axes=axes_grid(1,2,width=9.7,height=4.9)
    for val,color,marker in zip([7,14,21],COLORS,['o','s','^']):
        s=pooled[pooled.validation_days==val].sort_values('test_days_per_fold')
        axes[0,0].plot(s.test_days_per_fold,s.rmse_kw,marker=marker,color=color,label=f'验证 {val} 日')
        axes[0,1].plot(s.test_days_per_fold,s.daily_energy_rmse_kwh,marker=marker,color=color,label=f'验证 {val} 日')
    for ax,ylabel in zip(axes[0],['净负荷 RMSE / kW','日累计净能量 RMSE / kWh']):
        ax.set(xlabel='每个起点的测试窗口 / 日',ylabel=ylabel,xticks=[7,14,28])
        ax.legend(frameon=False,fontsize=10)
    finish(fig,'03_test_window','不同测试窗口长度下的预测误差',
        '对真实重训的逐日预测取前7/14/28日，分别合并3个起点的21/42/84个日期，从全部误差平方重算RMSE。窗口嵌套且日误差存在相关性，不将各窗口视为独立试验，也不对月度RMSE作算术平均。',
        ['forecast/cv_pooled_metrics.csv'],note='3个起点合并评分；每种长度的日数为21 / 42 / 84。嵌套测试窗，不是独立样本。')

    hyper=table('forecast','hyperparameter_metrics.csv')
    hyper=hyper[hyper.channel=='net']
    fig,axes=axes_grid(1,2,width=10,height=4.9)
    for ax,family,column,default in zip(axes[0],['hgb','extra_trees'],
        ['HGB 学习率','ExtraTrees 叶节点最少样本'],[.08,10]):
        for month,color,marker in zip([4,7,10],COLORS,['o','s','^']):
            s=hyper[(hyper.fold_month==month)&(hyper.family==family)]
            xv=np.r_[s.value,default]
            yv=np.r_[s.rmse_kw_change_pct,0.]
            order=np.argsort(xv)
            ax.plot(xv[order],yv[order],marker=marker,color=color,label=f'{month}月起点')
        ax.axhline(0,lw=.8,color='#646A70')
        ax.axvline(default,lw=1,ls=':',color='#646A70')
        ax.set(xlabel=column,ylabel='净负荷 RMSE 相对默认值变化 / %')
        ax.legend(frameon=False,fontsize=10)
    finish(fig,'15_tree_hyperparameters','核心树模型参数微调后的真实重训',
        'HGB学习率由0.08改变为0.064/0.096，ExtraTrees叶节点最少样本由10改变为8/12；仅重训被修改的模型族两个输出模型，另一族保留已发布预测。每个参数在4/7/10月各评28日，验证固定7日。共新增24个模型，图中原点对应各折已核验的默认配置。',
        ['forecast/hyperparameter_metrics.csv','forecast/hyperparameter_training_audit.json'],
        note='3个起点，每个起点28日；仅改变一个模型族参数，保持50/50融合与因果后处理。')


def temporal_figures():
    windows=table('temporal','window_metrics.csv')
    windows['scenario']=windows.scenario.astype(str)
    final=windows[windows.strategy=='exp008_final']
    q2=final[final.scenario=='2']
    s=q2[q2.window_type=='month'].sort_values('start')
    fig,axes=axes_grid(width=9,height=4.8)
    ax=axes[0,0]
    xx=np.arange(len(s))
    ax.bar(xx,s.cost_reduction_pct,width=.66,color=BLUE,zorder=3)
    full=q2[q2.window_type=='full'].iloc[0].cost_reduction_pct
    ax.axhline(full,lw=1.1,ls='--',color=ORANGE,label=f'334日合计 {full:.2f}%')
    for i,row in enumerate(s.itertuples()):
        ax.text(i,row.cost_reduction_pct+.16,f'{row.cost_reduction_pct:.2f}',ha='center',fontsize=10)
    ax.set(xticks=xx,xticklabels=[f'{d.month}月' for d in pd.to_datetime(s.start)],
        ylabel='相对 exp006 费用降低 / %',ylim=(0,s.cost_reduction_pct.max()*1.2))
    ax.legend(frameon=False)
    finish(fig,'06_monthly_saving','第二问：月度费用改善的稳定性',
        '逐月对齐exp008与exp006的真实费用，从当月总费用计算降费比例。全年比例由全年费用计算；不平均月度百分比。使用两种策略各自跨日连续的冻结SOC轨迹，窗口开始时不重置。',
        ['temporal/window_metrics.csv'],note='2025年2—12月共334日；同日配对比较，月度改善均为正。')

    fig,axes=axes_grid(1,2,width=10.5,height=5)
    ax=axes[0,0]
    for size,color,ls in zip([30,60,90],COLORS,['-','--',':']):
        s=q2[(q2.window_type=='rolling')&(q2.window_days==size)].sort_values('end')
        ax.plot(pd.to_datetime(s.end),s.cost_reduction_pct,color=color,ls=ls,label=f'{size} 日窗口')
    ax.set(ylabel='相对 exp006 费用降低 / %',xlabel='滚动窗口结束日期')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.set_xticks(pd.to_datetime(['2025-03-02','2025-06-01','2025-09-01','2025-12-31']))
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False,fontsize=10)
    ax=axes[0,1]
    s=q2[q2.window_type=='chronological_tail'].sort_values('requested_tail_fraction')
    xx=np.arange(len(s))
    ax.bar(xx,s.cost_reduction_pct,width=.55,color=BLUE,zorder=3)
    for i,row in enumerate(s.itertuples()):
        ax.text(i,row.cost_reduction_pct+.12,f'{row.cost_reduction_pct:.2f}%\n{row.n_days}日',ha='center',fontsize=10)
    ax.set(xticks=xx,xticklabels=[f'末尾 {v*100:.0f}%' for v in s.requested_tail_fraction],
           ylabel='相对 exp006 费用降低 / %',ylim=(0,6.5))
    finish(fig,'07_rolling_and_tail','第二问：滚动窗口与后段样本调整',
        '左：逐日移动30/60/90日窗口；右：选取334日序列的末尾20%/30%/40%重评分。这里只调整已冻结策略的评分日期，不宣称重新训练，也不把开发期后段称作未触碰测试集。',
        ['temporal/window_metrics.csv'],note='窗口内采用策略各自历史延续的SOC；滚动窗口相互重叠，不能当作独立重复实验。')

    s=table('temporal','bootstrap_summary.csv')
    fig,axes=axes_grid(width=9,height=4.7)
    ax=axes[0,0]
    for j,method in enumerate(['paired_moving_block','paired_quarter_stratified_moving_block']):
        group=s[s.method==method].sort_values('block_days')
        yy=np.arange(3)+(j-.5)*.18
        ax.errorbar(group.bootstrap_median_pct,yy,
            xerr=np.vstack([group.bootstrap_median_pct-group.percentile_95_lower_pct,
                            group.percentile_95_upper_pct-group.bootstrap_median_pct]),
            fmt=['o','s'][j],capsize=4,color=COLORS[j],label=['全期移动块','季度分层移动块'][j])
    ax.axvline(full,ls='--',color='#666C72',lw=1,label=f'原样本 {full:.2f}%')
    ax.set(yticks=range(3),yticklabels=['7 日块','14 日块','28 日块'],
        xlabel='相对 exp006 费用降低 / %',ylim=(-.55,2.55))
    ax.legend(frameon=False,loc='lower left',bbox_to_anchor=(0,1.01),fontsize=10,ncol=3)
    finish(fig,'08_block_bootstrap','第二问：保留短期相关性的区块重采样',
        '将exp008与exp006同一天的费用配对，按7/14/28日连续块重采样，每种方法4096次。季度分层版本保持季度样本占比。点为重采样中位数，线段为2.5%—97.5%百分位区间。这是固定策略及本年度数据条件下的描述区间，不包含训练、参数选择或跨年度不确定性。',
        ['temporal/bootstrap_summary.csv'],note='同日费用配对、每组4096次；线段为条件性95%重采样区间，非新年度绩效保证。',rect=(0,.07,1,.89))

    fig,axes=axes_grid(1,2,width=10.5,height=5)
    labels={'2':'问题二','3':'问题三','4-2':'四问·方案2','4-3':'四问·方案3'}
    for scenario,color,marker in zip(labels,COLORS,['o','s','^','D']):
        s=final[(final.scenario==scenario)&(final.window_type=='month')].sort_values('start')
        months=pd.to_datetime(s.start).dt.month
        axes[0,0].plot(months,s.mean_daily_cost_yuan/10000,marker=marker,color=color,label=labels[scenario])
        axes[0,1].plot(months,s.mean_daily_emergency_kwh,marker=marker,color=color,label=labels[scenario])
    for ax,ylabel in zip(axes[0],['月内日均费用 / 万元','月内日均紧急购电 / kWh']):
        ax.set(xlabel='月份',ylabel=ylabel,xticks=[2,4,6,8,10,12])
    axes[0,0].legend(frameon=False,ncol=2,fontsize=9)
    finish(fig,'09_all_scenarios_monthly','各问最终方案的跨月运行表现',
        '以日均费用与日均紧急购电量比较不同长度月份。四个方案分别采用各自冻结预测器、信息发布与计价规则；Q4实际电价不同于Q2/Q3，因此曲线用于观察时间变化，不能直接按费用高低判定算法优劣。',
        ['temporal/window_metrics.csv'],note='334日冻结轨迹重评分；各问信息边界和电价不同，不作跨问题算法排名。')

    stress=table('temporal','stress_group_metrics.csv')
    stress['scenario']=stress.scenario.astype(str)
    s=stress[(stress.scenario=='2')&(stress.strategy=='exp008_final')]
    fig,axes=axes_grid(width=8.6,height=4.8)
    ax=axes[0,0]
    for group,color,marker,label in zip(['high_load','low_pv','high_net_load'],COLORS,['o','s','^'],
                                       ['高负荷日','低光伏日','高净负荷日']):
        g=s[s.stress_group==group].sort_values('nominal_tail_fraction')
        ax.plot(g.nominal_tail_fraction*100,g.cost_reduction_pct,marker=marker,color=color,label=label)
    ax.axhline(full,color='#686E73',ls='--',lw=1,label=f'全部334日 {full:.2f}%')
    ymax=s[s.stress_group.isin(['high_load','low_pv','high_net_load'])].cost_reduction_pct.max()
    ax.set(xlabel='压力组所选尾部比例 / %',ylabel='相对 exp006 费用降低 / %',xticks=[10,20,30],ylim=(0,ymax*1.18))
    ax.legend(frameon=False,ncol=2,fontsize=10)
    finish(fig,'10_observed_stress_groups','第二问：压力日分组及阈值稳定性',
        '按真实日负荷、光伏和净负荷的10%/20%/30%尾部分位事后分组，分别对相同日期的exp008与exp006费用配对比较。分组仅用于事后解释，未来实况未用于当时调度。复合高负荷且低光伏20%仅1日，不用于主图外推。',
        ['temporal/stress_group_metrics.csv'],note='真实实况事后分组；10% / 20% / 30%组各34 / 67 / 100日，分组相互重叠。')

    billing=table('temporal','billing_sensitivity.csv')
    billing['scenario']=billing.scenario.astype(str)
    billing=billing[billing.strategy=='exp008_final']
    fig,axes=axes_grid(1,2,width=10,height=4.9)
    for ax,param,xlabel in zip(axes[0],['emergency_multiplier','up_adjustment_multiplier'],
        ['紧急购电计费倍率','上调购电计费倍率']):
        for scenario,color,marker in zip(labels,COLORS,['o','s','^','D']):
            s=billing[(billing.scenario==scenario)&(billing.parameter==param)].sort_values('value')
            if s.empty: continue
            ax.plot(s.value,s.cost_change_pct,marker=marker,color=color,label=labels[scenario])
        ax.axhline(0,lw=.8,color='#646A70')
        ax.set(xlabel=xlabel,ylabel='费用相对原计价变化 / %')
    axes[0,0].legend(frameon=False,ncol=2,fontsize=9)
    axes[0,1].legend(frameon=False,fontsize=9)
    finish(fig,'11_billing_sensitivity','计费参数变化下的费用敏感性',
        '固定已执行购电及电池动作，仅调整紧急购电倍率与上调倍率重新计费。原倍率分别为5和1.5。本图是成本结构的静态暴露分析，未重新优化；下调为零导致下调倍率敏感性为零，其完整结果保留在源表。',
        ['temporal/billing_sensitivity.csv'],note='按全部334日已执行动作重新结算；改变计费倍率不改变物理轨迹。')


def overview():
    paths=[OUT/f"{row['id']}.png" for row in META]
    thumb_w,thumb_h=480,340
    ncols=3
    nrows=int(np.ceil(len(paths)/ncols))
    canvas=Image.new('RGB',(ncols*thumb_w,nrows*thumb_h),'#FFFFFF')
    draw=ImageDraw.Draw(canvas)
    for i,p in enumerate(paths):
        with Image.open(p) as im:
            thumb=ImageOps.contain(im.convert('RGB'),(thumb_w-12,thumb_h-30))
        x=(i%ncols)*thumb_w+(thumb_w-thumb.width)//2
        y=(i//ncols)*thumb_h+24
        canvas.paste(thumb,(x,y))
        draw.text(((i%ncols)*thumb_w+12,(i//ncols)*thumb_h+4),p.stem,fill='#333333')
    canvas.save(OUT/'overview.png')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--partial',action='store_true')
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    global BOOK
    with PdfPages(OUT/'all_figures.pdf') as book:
        BOOK=book
        execution()
        if (DATA/'q1/sensitivity.csv').exists():
            q1_figures()
        if not args.partial:
            forecast_figures()
            temporal_figures()
    BOOK=None
    META.sort(key=lambda row: row['id'])
    (OUT/'figure_manifest.json').write_text(json.dumps(META,ensure_ascii=False,indent=2))
    overview()
    print(json.dumps({'figures':len(META),'directory':str(OUT)},ensure_ascii=False))


if __name__=='__main__':
    main()
