"""Publication figures from reviewed, unrounded experiment outputs."""

import numpy as np
import pandas as pd


def supplemental_figures(report, yearly, seed_stats, summary, selection, out, names, targets):
    import matplotlib.pyplot as plt

    directory = report/'figures'
    def save(fig,name):
        fig.savefig(directory/f'{name}.png',dpi=180,bbox_inches='tight')
        fig.savefig(directory/f'{name}.svg',bbox_inches='tight')
        plt.close(fig)

    fig,axes=plt.subplots(4,3,figsize=(15,15),layout='constrained')
    metrics={'mae':'平均绝对误差','rmse':'均方根误差','wape_pct':'加权绝对百分比误差（%）'}
    variants=['yesterday','weekly','mlp','gru','tcn']
    colors={'yesterday':'#8b9299','weekly':'#abb0b5','issued':'#8b9299','mlp':'#3c7aa5','gru':'#258675','tcn':'#bb7c37'}
    for i,(target,label) in enumerate(targets.items()):
        variants=['issued','mlp','gru','tcn'] if target=='pv_corrected' else ['yesterday','weekly','mlp','gru','tcn']
        f=yearly[(yearly.target==target)&(yearly.population=='all')&
                 yearly.seed.isin(['mean','baseline'])].set_index('variant').loc[variants]
        for j,(metric,title) in enumerate(metrics.items()):
            ax=axes[i,j]
            ax.barh([names[v] for v in variants],f[metric],color=[colors[v] for v in variants])
            ax.set(title=f'{label} · {title}',xlabel='%' if metric=='wape_pct' else '元/千瓦时' if target=='price' else '千瓦')
            ax.grid(axis='x',alpha=.15)
    save(fig,'annual-three-error-metrics')

    fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    for ax,(target,label) in zip(axes.ravel(),targets.items()):
        f=seed_stats[seed_stats.target==target]
        ax.errorbar(np.arange(len(f)),f.mae_mean,yerr=f.mae_std,fmt='o',capsize=5,color='#377eb8')
        ax.set(xticks=np.arange(len(f)),xticklabels=[names[v] for v in f.variant],
               title=label,xlabel='网络与特征消融',ylabel='平均绝对误差：三种子均值 ± 样本标准差')
        ax.tick_params(axis='x',rotation=20);ax.grid(axis='y',alpha=.15)
    save(fig,'seed-variation')

    fig,ax=plt.subplots(figsize=(11,6),layout='constrained')
    bottom=np.zeros(len(summary))
    for column,label,color in [('planned_cost','凌晨计划费','#3c7aa5'),('up_cost','最终上调费','#52a18c'),
                               ('down_cost','最终下调费','#a5bece'),('emergency_cost','紧急购电费','#d98b3d')]:
        values=summary[column].to_numpy()/10000
        ax.bar(summary.scenario_label,values,bottom=bottom,label=label,color=color)
        bottom+=values
    ax.set(ylabel='全年费用（万元）',title='正式策略的费用组成');ax.legend(ncol=4);ax.grid(axis='y',alpha=.15)
    save(fig,'cost-components')

    # A measured high-cost failure case, selected by its observed cost only for
    # diagnosis. It is not used for model choice or retrospective optimization.
    daily=pd.read_csv(out/'daily_metrics.csv',dtype={'scenario':str,'seed':str})
    candidates=daily[(daily.scenario=='4-3')&(daily.variant=='selected')&(~daily.known_price)&
                     daily.corrected&(daily.update_schedule=='0+6+12+18')]
    worst=candidates.loc[candidates.total_cost.idxmax()]
    index=int(worst.day)-31;month=int(worst.month);origin=int(worst.day)*144
    variant=selection[(selection.month==month)&(selection.scenario=='4-3')&selection.selected].iloc[0].variant
    with np.load(out/'dispatch_4-3.npz') as archive:
        detail={k:archive[k][index] for k in archive.files}
    with np.load(out/'ensemble_predictions.npz') as archive:
        origins=archive['origins']; predictions=archive[variant]
        first=predictions[np.flatnonzero(origins==origin)[0]]
        latest=np.empty_like(first)
        for start in (0,36,72,108):
            prediction=predictions[np.flatnonzero(origins==origin+start)[0]]
            latest[start:start+36]=prediction[:36]
    hours=np.arange(1,145)/6
    fig,axes=plt.subplots(5,1,figsize=(13,13),sharex=True,layout='constrained')
    for ax,channel,pred_channel,label in [(axes[0],0,0,'负载'),(axes[1],1,3,'光伏')]:
        ax.plot(hours,detail['actual'][:,channel],label='实际值',color='#263b4a')
        ax.plot(hours,first[:,pred_channel],label='凌晨预测',color='#3c7aa5',linestyle='--')
        ax.plot(hours,latest[:,pred_channel],label='当前已发布预测',color='#258675')
        ax.set(ylabel=f'{label}功率（千瓦）');ax.legend(ncol=3)
    axes[2].plot(hours,detail['price'],label='实际结算价',color='#263b4a')
    axes[2].plot(hours,latest[:,2],label='当前预测价',color='#258675')
    axes[2].set(ylabel='电价（元/千瓦时）');axes[2].legend(ncol=2)
    for field,label,color in [('original','凌晨计划','#3c7aa5'),('final','最终承诺','#258675'),('emergency','紧急购电','#d98b3d')]:
        axes[3].plot(hours,detail[field],label=label,color=color)
    axes[3].set(ylabel='区间电量（千瓦时）');axes[3].legend(ncol=3)
    axes[4].plot(np.arange(145)/6,detail['states'],color='#70579e')
    axes[4].axhline(1200,color='#888',linestyle='--');axes[4].axhline(10800,color='#888',linestyle='--')
    axes[4].set(ylabel='实际储电量（千瓦时）',xlabel='当天时刻（小时）',xticks=range(0,25,2),ylim=(500,11500))
    for ax in axes:
        for boundary in (6,12,18): ax.axvline(boundary,color='#c8ced4',linestyle=':',linewidth=.8)
        ax.grid(axis='y',alpha=.15)
    axes[0].set_title(f'问题 4-3 高费用案例：{worst.date} · 当月采用{names[variant]}')
    save(fig,'failure-case')
    case=pd.DataFrame({'date':worst.date,'hour':hours,'actual_load':detail['actual'][:,0],
                       'forecast_load':latest[:,0],'actual_pv':detail['actual'][:,1],
                       'forecast_pv':latest[:,3],'actual_price':detail['price'],
                       'forecast_price':latest[:,2],'original':detail['original'],
                       'final':detail['final'],'emergency':detail['emergency'],'soc':detail['states'][1:],
                       'charge':detail['charge'],'discharge':detail['discharge']})
    case.to_csv(report/'failure_case.csv',index=False)
    return case
