"""Build a source-backed Markdown/Data-app report and append-only experiment registry."""

import argparse
import datetime as dt
import json
import os
import re
import shutil
import subprocess
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd
from history import comparison_rows, differences, read_registry
from plots import supplemental_figures

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))


NAMES = {"mlp":"多层感知机", "gru":"门控循环网络", "tcn":"因果卷积网络",
         "gru_no_calendar":"移除日历特征", "gru_one_day":"一天历史窗口",
         "yesterday":"昨日同期", "weekly":"历史周同期", "selected":"按月验证选模", "issued":"附件原始光伏预报"}
TARGET_NAMES = {"load":"小区负载", "pv":"光伏历史预测", "price":"电价", "pv_corrected":"光伏预报修正"}
PALETTE = {'yesterday':'#8b9299','weekly':'#abb0b5','issued':'#8b9299',
           'mlp':'#3c7aa5','gru':'#258675','tcn':'#bb7c37'}
TITLES = ["结论与成绩", "题目指标及信息边界", "数据与时间验证", "逐步技术讲解",
          "实验设置", "结果及失败案例", "历次指标和技术路线对比", "复现说明"]


def records(frame):
    return json.loads(frame.to_json(orient="records", force_ascii=False, double_precision=12))


def presentation_blocks(markdown, section, queries, experiment):
    """Keep editable prose and reviewed Markdown tables on separate render paths."""
    lines=markdown.splitlines();blocks=[];prose=[];i=0;table_number=0
    heading=TITLES[section-1]
    def flush():
        if prose:
            blocks.append({'type':'prose','markdown':'\n'.join(prose).strip()});prose.clear()
    def cells(line):
        return [value.strip() for value in line.strip().strip('|').split('|')]
    while i<len(lines):
        if lines[i].startswith('#'):
            heading=lines[i].lstrip('#').strip()
        if (i+1<len(lines) and lines[i].strip().startswith('|')
                and re.fullmatch(r'\s*\|(?:\s*:?-+:?\s*\|)+\s*',lines[i+1])):
            flush();headers=cells(lines[i]);i+=2;rows=[];table_number+=1
            while i<len(lines) and lines[i].strip().startswith('|'):
                values=cells(lines[i]);assert len(values)==len(headers)
                rows.append({f'column_{j}':v for j,v in enumerate(values)});i+=1
            key=f'narrative_{section}_table_{table_number}'
            queries[key]={'rows':rows,'source':{'label':heading,
                'files':[f'reports/experiments/{experiment}/section-{section}.md'],
                'metricDefinitions':[{'label':'显示精度','definition':'与 Markdown 正文逐格一致的格式化显示值；原始精度保留在本报告的指标 CSV 中。'}]}}
            blocks.append({'type':'table','queryId':key,'title':heading,'rows':rows,
                           'columns':[[f'column_{j}',h] for j,h in enumerate(headers)]})
        else:
            prose.append(lines[i]);i+=1
    flush();return blocks


def markdown_table(frame, columns, digits=4):
    header = "| " + " | ".join(columns.values()) + " |"
    lines = [header, "|" + "|".join("---" for _ in columns) + "|"]
    for row in frame.to_dict("records"):
        values = []
        for key in columns:
            x = row[key]
            values.append(f"{x:,.{digits}f}" if isinstance(x,(float,np.floating)) else str(x))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def combine_forecasts(frame, grouping):
    x = frame.groupby(grouping, dropna=False).agg(n=("n","sum"),
        absolute_error_sum=("absolute_error_sum","sum"), squared_error_sum=("squared_error_sum","sum"),
        actual_abs_sum=("actual_abs_sum","sum")).reset_index()
    x['mae'] = x.absolute_error_sum / x.n
    x['rmse'] = np.sqrt(x.squared_error_sum / x.n)
    x['wape_pct'] = 100 * x.absolute_error_sum / x.actual_abs_sum.replace(0,np.nan)
    return x


def full_schedule(frame):
    adjustable=frame.scenario.isin(['3','4-3'])
    return ((~adjustable)&(frame.update_schedule=='0')) | (adjustable&(frame.update_schedule=='0+6+12+18'))


def figures(report, forecasts, costs, daily):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for candidate in ("/System/Library/Fonts/PingFang.ttc", "/System/Library/Fonts/STHeiti Medium.ttc",
                      "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"):
        if Path(candidate).exists():
            font_manager.fontManager.addfont(candidate)
            plt.rcParams['font.family'] = font_manager.FontProperties(fname=candidate).get_name()
            break
    plt.rcParams.update({'axes.unicode_minus':False,'font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'savefig.bbox':'tight'})
    directory = report/'figures'; directory.mkdir(exist_ok=True)
    def save(fig,name):
        fig.savefig(directory/f'{name}.png',dpi=180)
        fig.savefig(directory/f'{name}.svg')
        plt.close(fig)
    fig, axes = plt.subplots(2,2,figsize=(13,8),layout='constrained')
    for ax,(target,label) in zip(axes.ravel(),TARGET_NAMES.items()):
        for variant in (('issued','mlp','gru','tcn') if target=='pv_corrected' else ('yesterday','weekly','mlp','gru','tcn')):
            f=forecasts[(forecasts.target==target)&(forecasts.variant==variant)].sort_values('month')
            ax.plot(f.month,f.mae,label=NAMES[variant],color=PALETTE[variant],linestyle='--' if variant in ('yesterday','weekly','issued') else '-',marker='o',markersize=3)
        ax.set(title=label,xlabel='月份',ylabel='平均绝对误差（元/千瓦时）' if target=='price' else '平均绝对误差（千瓦）',xticks=range(2,13))
        ax.grid(axis='y',alpha=.18)
    handles,labels=axes[0,0].get_legend_handles_labels()
    issued_handle,issued_label=axes[1,1].get_legend_handles_labels()
    fig.legend(handles+[issued_handle[0]],labels+[issued_label[0]],loc='outside lower center',ncol=3)
    save(fig,'monthly-forecast-error')
    fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    for ax,scenario in zip(axes.ravel(),('2','3','4-2','4-3')):
        f=costs[costs.scenario==scenario].sort_values('total_cost')
        ax.barh(f.model_label,f.total_cost/10000,color='#377eb8')
        ax.set(title=f'问题 {scenario}',xlabel='全年总费用（万元）')
        ax.grid(axis='x',alpha=.18)
    save(fig,'annual-cost-comparison')
    fig,axes=plt.subplots(2,2,figsize=(13,7),layout='constrained')
    for ax,scenario in zip(axes.ravel(),('2','3','4-2','4-3')):
        f=daily[daily.scenario==scenario]
        ax.plot(pd.to_datetime(f.date),f.emergency_kwh,color='#d95f02',linewidth=.8)
        ax.set(title=f'问题 {scenario}',ylabel='紧急购电量（千瓦时）',xlabel='日期')
        ax.tick_params(axis='x',rotation=30);ax.grid(axis='y',alpha=.18)
    save(fig,'daily-emergency-energy')
    fig,ax=plt.subplots(figsize=(13,7));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    nodes=[('数据与时标','CSV，发布时刻，十分钟区间'),('历史特征','七天历史，昨日及上周同期'),
           ('三种历史编码','多层感知机／门控循环／因果卷积'),('144 步预测','目标特征，残差修正，恢复单位'),
           ('凌晨计划','能量平衡与储能边界的线性规划'),('日内修正','6、12、18 时更新剩余区间'),
           ('真实执行','供负载，充放电，紧急购电'),('结算与验证','费用分项，预测误差，物理约束')]
    positions=[(.14,.8),(.38,.8),(.62,.8),(.86,.8),(.86,.36),(.62,.36),(.38,.36),(.14,.36)]
    for (title,detail),(x,y) in zip(nodes,positions):
        ax.text(x,y,title,ha='center',va='center',fontsize=12,bbox={'boxstyle':'round,pad=.6','facecolor':'#edf3f7','edgecolor':'#4d677b'})
        ax.text(x,y-.12,detail.replace('，','\n'),ha='center',va='top',fontsize=9)
    for (x,y),(xx,yy) in pairwise(positions):
        ax.annotate('',xy=(xx-(.105 if xx>x else -.105) if y==yy else xx, yy if y==yy else yy+.07),
                    xytext=(x+(.105 if xx>x else -.105) if y==yy else x,y if y==yy else y-.22),
                    arrowprops={'arrowstyle':'->','color':'#536878','lw':1.6})
    ax.set_title('预测、调度与结算的完整技术路径',fontsize=16,pad=16)
    save(fig,'technical-route')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--experiment',default='exp001')
    p.add_argument('--code-commit')
    p.add_argument('--title')
    args=p.parse_args()
    out=ROOT/'data/results'/args.experiment
    measured_protocol = out / 'protocol.json'
    if measured_protocol.exists() and json.loads(measured_protocol.read_text()).get('architecture', {}).get('name') == 'periodic_residual_mlp':
        import build_report_v2
        build_report_v2.main()
        return
    report=ROOT/'reports/experiments'/args.experiment
    report.mkdir(parents=True,exist_ok=True)
    commit=args.code_commit or subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    protocol=json.loads((ROOT/'experiments/common/neural_v1/protocol.json').read_text())
    protocol['experiment_id']=args.experiment
    title=args.title or ('首轮神经网络预测与基础调度' if args.experiment=='exp001' else f'{args.experiment} 神经网络预测与基础调度')
    forecast=pd.read_csv(out/'forecast_metrics.csv',dtype={'seed':str})
    dispatch=pd.read_csv(out/'dispatch_metrics.csv',dtype={'scenario':str,'seed':str})
    daily=pd.read_csv(out/'daily_metrics.csv',dtype={'scenario':str,'seed':str})
    selection=pd.read_csv(out/'model_selection.csv',dtype={'scenario':str})
    training=json.loads((out/'training_metadata.json').read_text())
    verification=json.loads((out/'verification.json').read_text())
    data_hashes=json.loads((out/'data_hashes.json').read_text())
    assert len(training)==165 and all(r['save_reload_passed'] for r in training)
    assert all('GPU:0' in r['device'] for r in training)
    assert all(r['environment']==training[0]['environment'] for r in training)
    assert len(verification)==4 and all(v['violations']==0 for v in verification.values())
    assert len(selection)==11*4*3 and selection.selected.sum()==11*4
    primary_mask=(dispatch.variant=='selected')&(~dispatch.known_price)&dispatch.corrected&full_schedule(dispatch)
    summary=dispatch[primary_mask].copy()
    assert len(summary)==4
    summary['scenario_label']='问题 '+summary.scenario
    base=dispatch[dispatch.variant=='yesterday'].set_index('scenario').total_cost
    summary['improvement_pct']=[100*(base[s]-v)/base[s] for s,v in zip(summary.scenario,summary.total_cost)]
    weekly_base=dispatch[dispatch.variant=='weekly'].set_index('scenario').total_cost
    summary['weekly_improvement_pct']=[100*(weekly_base[s]-v)/weekly_base[s] for s,v in zip(summary.scenario,summary.total_cost)]
    primary_daily=daily[(daily.variant=='selected')&(~daily.known_price)&daily.corrected&full_schedule(daily)].copy()
    assert len(primary_daily)==4*334
    comparables=dispatch[(dispatch.seed.isin(['mean','baseline']))&(~dispatch.known_price)&dispatch.corrected&full_schedule(dispatch)].copy()
    comparables['model_label']=comparables.variant.map(NAMES)
    comparables=comparables.sort_values(['scenario','total_cost'])
    monthly=forecast[(forecast.seed.isin(['mean','baseline']))&(forecast.lead=='all')&
                     (forecast.population=='all')].copy()
    monthly['model_label']=monthly.variant.map(NAMES);monthly['month_label']=monthly.month.map(lambda m:f'2025-{m:02d}-01')
    monthly_full=forecast[(forecast.seed.isin(['mean','baseline']))&(forecast.lead=='all')].copy()
    monthly_full['model_label']=monthly_full.variant.map(NAMES)
    monthly_full['month_label']=monthly_full.month.map(lambda m:f'2025-{m:02d}-01')
    yearly=combine_forecasts(forecast[forecast.lead=='all'],['variant','seed','target','population'])
    yearly['model_label']=yearly.variant.map(NAMES);yearly['target_label']=yearly.target.map(TARGET_NAMES)
    yearly.to_csv(out/'annual_forecast_metrics.csv',index=False)
    yearly.to_csv(report/'annual_forecast_metrics.csv',index=False)
    mean_yearly=yearly[yearly.seed.isin(['mean','baseline']) & (yearly.population=='all')]
    selected_parts=[]
    for r in selection[selection.selected].to_dict('records'):
        target_list=['load','pv_corrected' if r['scenario'] in ('3','4-3') else 'pv']
        if r['scenario'].startswith('4'):target_list.append('price')
        part=forecast[(forecast.month==r['month'])&(forecast.variant==r['variant'])&
                      (forecast.seed=='mean')&(forecast.lead=='all')&forecast.target.isin(target_list)].copy()
        part['variant']='selected_'+r['scenario'];part['scenario']=r['scenario'];selected_parts.append(part)
    selected_forecasts=combine_forecasts(pd.concat(selected_parts),['variant','scenario','target','population'])
    selected_forecasts['model_label']='正式策略问题 '+selected_forecasts.scenario
    selected_forecasts['target_label']=selected_forecasts.target.map(TARGET_NAMES)
    selected_forecasts.to_csv(report/'selected_forecast_metrics.csv',index=False)
    lead=combine_forecasts(forecast[(forecast.seed.isin(['mean','baseline']))&(forecast.lead!='all')],
                          ['variant','target','lead','population'])
    lead['model_label']=lead.variant.map(NAMES)
    lead['lead_hours']=lead.lead.map({'0–6h':0,'6–12h':6,'12–18h':12,'18–24h':18})
    lead=lead.sort_values(['variant','target','population','lead_hours'])
    updates=dispatch[(dispatch.variant=='selected')&(~dispatch.known_price)&dispatch.scenario.isin(['3','4-3'])].copy()
    updates['adjustment_cost']=updates.up_cost+updates.down_cost
    price_compare=dispatch[(dispatch.variant=='selected')&dispatch.scenario.isin(['4-2','4-3'])&full_schedule(dispatch)]
    specified=pd.read_csv(out/'specified_dates.csv',dtype={'scenario':str})
    worst=primary_daily.sort_values('total_cost',ascending=False).groupby('scenario').head(1)
    seeds=yearly[yearly.seed.isin(['42','2026','3407']) & (yearly.population=='all')]
    seed_stats=seeds.groupby(['variant','target']).agg(mae_mean=('mae','mean'),mae_std=('mae','std'),
        rmse_mean=('rmse','mean'),rmse_std=('rmse','std'),wape_mean=('wape_pct','mean'),wape_std=('wape_pct','std')).reset_index()
    seed_stats.to_csv(report/'seed_statistics.csv',index=False)
    seed_stats['model_label']=seed_stats.variant.map(NAMES)
    seed_stats['target_label']=seed_stats.target.map(TARGET_NAMES)
    seed_costs=dispatch[dispatch.seed.isin(['42','2026','3407'])]
    seed_cost_stats=seed_costs.groupby(['scenario','variant']).agg(total_cost_mean=('total_cost','mean'),
        total_cost_std=('total_cost','std'),emergency_mean=('emergency_kwh','mean'),
        emergency_std=('emergency_kwh','std')).reset_index()
    seed_cost_stats['model_label']=seed_cost_stats.variant.map(NAMES)
    seed_cost_stats.to_csv(report/'seed_cost_statistics.csv',index=False)
    seconds=sum(r['seconds'] for r in training)
    capped=sum(r['epochs']==60 for r in training)
    figures(report,monthly,comparables,primary_daily)
    case=supplemental_figures(report,yearly,seed_stats,summary,selection,out,NAMES,TARGET_NAMES)
    tech=['区间终点对齐','因果历史窗口与已发布预报','三种独立分支神经网络残差预测',
          '三种子平均与月初验证费用选模','线性规划计划及日内重算','固定因果执行','逐区间独立结算']
    current={'experiment_id':args.experiment,'title':title,
             'protocol':protocol,'code_commit':commit,'data_hashes':data_hashes,
             'environment':training[0]['environment'],'seeds':[42,2026,3407],
             'models':list(NAMES)[:5],'metrics':records(summary),'technical_path':tech,
             'forecast_metrics':records(yearly[yearly.seed.isin(['mean','baseline'])])+records(selected_forecasts),
             'model_configuration':{k:protocol[k] for k in ('architecture','variants','batch_size','max_epochs','learning_rate','early_stopping_patience')},
             'metric_definitions':{'mae':'sum(abs(predicted-actual))/n',
                 'rmse':'sqrt(sum((predicted-actual)^2)/n)',
                 'wape_pct':'100*sum(abs(predicted-actual))/sum(abs(actual))',
                 'forecast_sample':'all four daily issues, 144 future intervals, observed labels only',
                 'total_cost':'sum(price*(original+1.5*max(final-original,0)+0.5*max(original-final,0)+5*emergency))'},
             'artifacts':{'report':f'experiments/{args.experiment}/report.md',
                          'html':f'experiments/{args.experiment}/report.html',
                          'results':f'../data/results/{args.experiment}'}}
    registry=ROOT/'reports/registry';registry.mkdir(exist_ok=True)
    prior=read_registry(registry,exclude=args.experiment)
    historical=[]
    for record in prior+[current]:
        reasons=differences(record,current)
        for metric in record['metrics']:
            historical.append({'experiment_id':record['experiment_id'],'scenario':metric['scenario'],
                               'total_cost':metric['total_cost'],'emergency_kwh':metric['emergency_kwh'],
                               'technical_path':' → '.join(record['technical_path']),
                               'comparison':'同口径' if not reasons else '不可直接排名：'+', '.join(reasons)})
    history=pd.DataFrame(historical)
    comparisons=pd.DataFrame(comparison_rows(prior,current),columns=[
        'previous_experiment','current_experiment','task','route','metric','previous','current',
        'relative_change_pct','comparison','previous_technical_path','current_technical_path'])
    comparisons.to_csv(report/'relative_comparison.csv',index=False)
    comparisons.to_csv(ROOT/'reports/relative_comparison.csv',index=False)
    summary_text=markdown_table(summary,{'scenario_label':'问题','total_cost':'总费用（元）',
        'emergency_kwh':'紧急购电（千瓦时）','improvement_pct':'相对昨日基线节省（%）',
        'weekly_improvement_pct':'相对周同期基线节省（%）'},2)
    sections=[
      f"## 1. 结论与成绩\n\n正式评估覆盖 2025 年 2 月 1 日至 12 月 31 日，共 334 天。三种网络和两个特征消融均完成 11 个月、三个随机种子的训练，共 165 组。正式策略按月依据此前验证期总费用选择网络，采用三个种子预测的平均值。\n\n{summary_text}\n\n节省率为负表示费用高于昨日基线。所有方案使用同一物理约束与结算规则；本轮不预设神经网络一定优于简单基线。",
      "## 2. 题目指标及信息边界\n\n问题 2 在凌晨预测负载和光伏，使用固定日内电价。问题 3 增加四次已发布光伏预报及三次可选日内调整。问题 4 主结果预测未来电价，已知当天价格只作为单独对照。\n\n每个自然日有 144 个十分钟区间。充放电效率各为 90%，储电范围为 1200—10800 千瓦时，最大充放电功率为 5000 千瓦。1 月 1 日从 6000 千瓦时开始，1 月使用因果历史基线预热。每日储电状态连续传递，没有每日重置或日初日末相等条件。\n\n总费用包括原计划费、最终调整相对原计划的上调费和下调费、紧急购电费。原计划不退款，上调价格为 1.5 倍，下调另收 0.5 倍，紧急购电价格为 5 倍。多次更新只对最终执行量相对原计划结算一次。预测误差以平均绝对误差、均方根误差和加权绝对百分比误差度量，精确定义见技术讲解。",
      "## 3. 数据与时间验证\n\n三个实测数据表各包含 365 天乘 144 个数值，光伏预报有 365 天乘四次发布乘 24 个提前小时。数值区无缺失。原始日期空白仅存在于预报表同一天的后续行，已按日期向下填充。\n\n数据中的 00:10 对应 00:00—00:10。结果模板原来的时段文字整体偏移十分钟，生成文件已改为自然日表头；原始模板字节不变。训练窗口、验证窗口和标签完成时刻分别检查，改变未来实测值及后来发布预报的测试不得改变此前特征或计划。年底超出已知真实值范围的预测不计入误差。\n\n输入文件的 SHA-256 校验值保存在随报告提供的 data_hashes.json。",
      "## 4. 逐步技术讲解\n\n"+(ROOT/'reports/templates/methods-neural-v1.md').read_text(),
      f"## 5. 实验设置\n\n多层感知机的历史编码为 64、32 个隐藏单元；门控循环网络为 32 个循环单元；因果卷积为四层、每层 32 通道、卷积核宽度 3、膨胀率 1、2、4、8。每个分支都连接 32 单元的逐目标隐藏层和线性残差输出层。\n\n批量大小 64，学习率 0.001，最多训练 60 轮，验证连续六轮不改善则停止，恢复验证损失最低轮次。每月从头拟合，不将验证周重新并入训练；随机种子为 42、2026、3407。两个特征消融分别移除日历周期和把小时历史缩短为一天；后者保留共同的昨日、上周同期目标特征。\n\nGPU 环境为 Apple M5、TensorFlow 2.18.1、Metal 1.2.0、单精度浮点数，关闭即时编译。每组训练均核查输出设备、参数更新和保存读取一致性。累计训练调用耗时为 {seconds/60:.2f} 分钟，不包括数据准备、保存、报告与调度。{capped} 组达到 60 轮上限，因此不能声称所有模型已经完全收敛。\n\n训练设备及每轮损失见 training_metadata.json。种子均值、标准差与每个种子的具体成绩均保存为 CSV；种子标准差不作为时间序列的独立样本置信区间。",
      "## 6. 结果及失败案例\n\n### 6.1 全年预测成绩\n\n"+markdown_table(mean_yearly,{'model_label':'模型','target_label':'目标','mae':'平均绝对误差','rmse':'均方根误差','wape_pct':'加权绝对百分比误差（%）'})+
      "\n\n负载与光伏误差单位为千瓦，电价误差单位为元每千瓦时。表中是三个种子平均预测的误差，不能等同于三个种子各自误差的平均。所有种子的逐项成绩和真实发电区间成绩见 annual_forecast_metrics.csv。\n\n### 6.2 预报时刻消融\n\n"+
      markdown_table(updates,{'scenario':'问题','update_schedule':'使用的预报时刻','total_cost':'总费用（元）','adjustment_cost':'调整费用（元）','emergency_kwh':'紧急购电量（千瓦时）'},2)+
      "\n\n这些对照保持月初所选网络不变，只改变预报使用时刻，因此可以检验额外预报在本轮执行规则下的实际价值。\n\n### 6.3 价格可见性对照\n\n"+
      markdown_table(price_compare,{'scenario':'问题','known_price':'当天电价提前已知','total_cost':'总费用（元）','emergency_kwh':'紧急购电（千瓦时）'},2)+
      "\n\n已知价格对照仍然使用负载和光伏预测，并非所有未来信息都已知的理论下界。\n\n### 6.4 高费用日期与限制\n\n"+
      markdown_table(worst,{'scenario':'问题','date':'费用最高日期','total_cost':'总费用（元）','planned_cost':'计划费（元）','emergency_cost':'紧急购电费（元）','emergency_kwh':'紧急购电量（千瓦时）'},2)+
      "\n\n费用最高的日期不一定是预测误差最大的日期，因为负载、电价、光伏和当时储电状态共同影响结果。当前模型没有气象输入、风险缓冲、储能寿命费用或日末储能价值项；均方误差也没有直接表示五倍紧急购电的非对称代价。单年数据不足以证明跨年份泛化。\n\n题目要求的四个日期及全部指定十分钟时段、四小时充放电、日初日末储电量和紧急购电明细见配套 specified_dates.md。",
      "## 7. 历次指标和技术路线对比\n\n"+(
          f"本次报告读取此前 {len(prior)} 次实验记录。" if prior else "这是首次有正式训练成绩的实验，当前没有此前训练实验可供比较；昨日同期与历史周同期是本次对照，不是历史实验。")+
      "新实验会读取 registry 中全部已有记录，同时比较数据、评价时期、计费、物理约束和技术路径。口径不同的实验列出差异，不混在同一排行榜。\n\n"+
      markdown_table(history,{'experiment_id':'实验','scenario':'问题','total_cost':'总费用（元）','comparison':'可比性'},2)+
      "\n\n本轮技术路径为："+' → '.join(tech)+"。每个历史实验的完整路径保存在历史对比 CSV 中。旧报告保留生成时的比较快照，最新索引更新为全部实验。",
      f"## 8. 复现说明\n\n代码提交：`{commit}`。数据校验值、环境版本、训练参数、选择结果和核验成绩随报告保留。[四个结果工作簿与逐区间档案](https://github.com/Vege114/CUMCM2026/tree/codex/neural-forecasting-v1/data/results/exp001)保存在实验分支；报告网页的数据已完整内嵌，可以独立浏览。下面的复现命令在 codex/neural-forecasting-v1 实验分支执行；上述提交固定训练与评估代码，报告生成器使用随本报告提供的版本。\n\n```bash\nuv sync --locked\nuv run --locked python scripts/check_environment.py\nuv run --locked python -m unittest discover -s tests -v\nuv run --locked python -m experiments.common.neural_v1.train\nuv run --locked python -m experiments.common.neural_v1.complete_validation\nuv run --locked python -m experiments.common.neural_v1.evaluate\nuv run --locked python -m experiments.common.neural_v1.export\nuv run --locked python reports/build_report.py --experiment {args.experiment}\n```\n\n权重保存在被忽略的 runs 目录；代码和输入签名相同才允许恢复。预测归档使用显式发布索引、目标区间索引和目标变量顺序，定义见 prediction_archive.json。正式工作簿为 result2.xlsx、result3.xlsx、result4-2.xlsx、result4-3.xlsx。\n\n所有工作簿的全天购电费表示包含计划、调整、紧急购电的总费；不同工作表的这个值不能再相加。verification.json 记录独立费用复算、能量平衡、连续储电和保存文件读取结果。\n\n技术资料：[Apple Metal 插件](https://developer.apple.com/metal/tensorflow-plugin/)、[TensorFlow 时间序列教程](https://www.tensorflow.org/tutorials/structured_data/time_series)、[门控循环层接口](https://keras.io/api/layers/recurrent_layers/gru/)、[因果卷积层接口](https://keras.io/api/layers/convolution_layers/convolution1d/)、[SciPy 线性规划接口](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html)。这些资料说明实现接口，实验结论来自本仓库实测结果。"
    ]
    generating=yearly[yearly.seed.isin(['mean','baseline']) & (yearly.population=='generating')]
    correction=dispatch[(dispatch.variant=='gru')&(dispatch.seed=='mean')&dispatch.scenario.isin(['3','4-3'])]
    additions='\n\n#### 实际发电区间的预测误差\n\n'+markdown_table(generating,{
        'model_label':'模型','target_label':'光伏目标','mae':'平均绝对误差（千瓦）',
        'rmse':'均方根误差（千瓦）','wape_pct':'加权绝对百分比误差（%）'})
    additions+='\n\n#### 正式策略的费用与约束\n\n'+markdown_table(summary,{
        'scenario':'问题','planned_cost':'计划费（元）','up_cost':'上调费（元）',
        'down_cost':'下调费（元）','emergency_cost':'紧急购电费（元）',
        'violations':'违反数','initial_soc':'评估起始储电量','final_soc':'评估末储电量',
        'solve_execute_seconds':'求解与执行合计秒数'},2)
    additions+='\n\n求解与执行耗时包含线性规划和固定规则回放，不包含模型训练。各方案训练调用的累计时间单独报告。\n\n#### 是否修正附件光伏预报\n\n'+markdown_table(correction,{
        'scenario':'问题','corrected':'使用门控循环网络修正','total_cost':'总费用（元）',
        'emergency_kwh':'紧急购电（千瓦时）'},2)
    additions+='\n\n这组对照保持门控循环网络负载与电价预测、日内更新时刻、计费及实际执行规则不变，只把光伏修正量设为零。\n'
    for scenario in ('3','4-3'):
        c=correction[correction.scenario==scenario]
        raw=c[~c.corrected].iloc[0].total_cost
        adjusted=c[c.corrected].iloc[0].total_cost
        additions+=f'\n\n问题 {scenario} 中，修正光伏预报后的总费用相对直接使用附件预报变化 {100*(adjusted/raw-1):+.2f}%。预测误差改善与决策费用改善应分别判断。'
    sections[5]=sections[5].replace('### 6.2 预报时刻消融',additions+'\n### 6.2 预报时刻消融')
    urgent=case.emergency>1e-6
    floor_count=int((urgent&(case.soc<=1200+1e-6)).sum())
    power_count=int((urgent&(case.discharge>=5000/6-1e-6)).sum())
    peak_index=int(case.emergency.idxmax());peak=case.loc[peak_index]
    peak_period=f'{peak_index//6:02d}:{peak_index%6*10:02d}–{(peak_index+1)//6:02d}:{(peak_index+1)%6*10:02d}'
    sections[5]+=(f'\n\n### 逐区间检查高费用案例\n\n问题 4-3 的最高费用日 {peak.date} 有 {int(urgent.sum())} 个紧急购电区间。其中，{floor_count} 个区间在放电后达到储电下限，{power_count} 个区间达到放电功率上限，两类可以重叠。'
        f'紧急购电量最大的区间为 {peak_period}，电量为 {peak.emergency:,.4f} 千瓦时。失败案例图把该日实际使用的最新预测、实际电价、原计划与最终购电、储能轨迹放在同一时间轴上，供逐项核对。')
    sections[5]+=('\n\n### 月初所选网络的统一预测评价\n\n'+markdown_table(selected_forecasts,{
        'scenario':'问题','target_label':'目标','population':'评价时段','mae':'平均绝对误差',
        'rmse':'均方根误差','wape_pct':'加权绝对百分比误差（%）'})+
        '\n\n为跨实验比较保留稳定的正式策略标识。这张表统一评价所选网络每天四次发布的完整 24 小时预测；问题 2 和问题 4-2 的实际购电计划仍只使用凌晨预测。all 表示全天，generating 表示真实光伏功率大于零的区间。')
    update_conclusions=[]
    for scenario in ('3','4-3'):
        values=updates[updates.scenario==scenario].sort_values('update_schedule')
        base_cost=values[values.update_schedule=='0'].iloc[0].total_cost
        full_cost=values[values.update_schedule=='0+6+12+18'].iloc[0].total_cost
        best=values.loc[values.total_cost.idxmin()]
        delta=100*(full_cost/base_cost-1)
        update_conclusions.append(f"问题 {scenario} 中，使用全部四次预报相对只使用凌晨预报的费用变化为 {delta:+.2f}%；本轮费用最低的预报时刻组合是 {best.update_schedule}，总费用为 {best.total_cost:,.2f} 元。")
    sections[5]=sections[5].replace('### 6.3 价格可见性对照','\n\n'+'\n\n'.join(update_conclusions)+
        '\n\n这些是统一规则下的事后对照结果；正式工作簿使用预先规定的四次预报方案。\n\n### 6.3 价格可见性对照')
    selected_table=selection[selection.selected].pivot(index='month',columns='scenario',values='variant').map(NAMES.get).reset_index()
    sections[4]+='\n\n### 每个月实际选用的网络\n\n'+markdown_table(selected_table,{'month':'月份','2':'问题 2','3':'问题 3','4-2':'问题 4-2','4-3':'问题 4-3'})
    wins=int((summary.improvement_pct>0).sum())
    weekly_wins=int((summary.weekly_improvement_pct>0).sum())
    sections[0]+=f'\n\n四个正式问题中，{wins} 个相对昨日同期基线减少了总费用，{weekly_wins} 个优于历史周同期基线。'
    if weekly_wins==0:
        sections[0]+='本轮正式神经网络策略均未超过历史周同期基线，因此不能据此宣布神经网络已是费用最优方案。'
    sections[0]+='各网络与两种历史基线的完整费用比较见第六部分及配套图表。'
    forecast_conclusions=[]
    for target,label in TARGET_NAMES.items():
        available=mean_yearly[mean_yearly.target==target]
        networks=available[available.variant.isin(['mlp','gru','tcn'])]
        baselines=available[available.variant.isin(['yesterday','weekly','issued'])]
        neural=networks.loc[networks.mae.idxmin()]
        baseline=baselines.loc[baselines.mae.idxmin()]
        unit='元/千瓦时' if target=='price' else '千瓦'
        forecast_conclusions.append(f'{label}中，三种主要网络里平均绝对误差最低的是{NAMES[neural.variant]}，为 {neural.mae:.4f} {unit}；'
            f'最好的简单对照是{NAMES[baseline.variant]}，为 {baseline.mae:.4f} {unit}，网络相对变化为 {100*(neural.mae/baseline.mae-1):+.2f}%。')
    sections[5]=sections[5].replace('### 6.2 预报时刻消融',
        '#### 预测成绩的具体含义\n\n'+'\n\n'.join(forecast_conclusions)+
        '\n\n这些全年排名只用于事后分析，不能替代每月的历史验证选模。多层感知机的弱项同样保留在表格和图中；本轮没有继续搜索其超参数，因此不能把一个固定配置的表现推广到所有多层感知机。\n\n### 6.2 预报时刻消融')
    sections[6]+='\n\n相对变化按（本次值－此前值）／此前值绝对值计算，负值表示数值下降；期末储电量不按越小越好解释。全部历史预测与调度指标的逐项对比保存在 relative_comparison.csv。'
    for i in range(8):
        (report/f'section-{i+1}.md').write_text(sections[i])
    full='# '+title+'实验报告\n\n'+'\n\n'.join(sections)
    full=full.replace('## 5. 实验设置','![技术路径](figures/technical-route.png)\n\n## 5. 实验设置')
    full=full.replace('### 6.2 预报时刻消融','![月度预测误差](figures/monthly-forecast-error.png)\n\n![全年费用对比](figures/annual-cost-comparison.png)\n\n### 6.2 预报时刻消融')
    full+='\n\n![每日紧急购电](figures/daily-emergency-energy.png)\n\n[指定日期完整表格](specified_dates.md)\n'
    full+='\n\n![三个核心预测指标](figures/annual-three-error-metrics.png)\n\n![随机种子波动](figures/seed-variation.png)\n\n![费用分项](figures/cost-components.png)\n\n![高费用失败案例](figures/failure-case.png)\n'
    (report/'report.md').write_text(full)
    for name in ('forecast_metrics.csv','dispatch_metrics.csv','model_selection.csv','specified_dates.csv',
                 'training_metadata.json','data_hashes.json','verification.json','prediction_archive.json','causality_checks.json'):
        shutil.copy2(out/name,report/name)
    primary_daily.to_csv(report/'daily_primary.csv',index=False)
    history.to_csv(report/'history_comparison.csv',index=False)
    (registry/f'{args.experiment}.json').write_text(json.dumps(current,ensure_ascii=False,indent=2,allow_nan=False))
    (report/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2))
    (ROOT/'reports/latest.md').write_text('# 实验报告索引\n\n'+ '\n'.join(
        f"- [{r['experiment_id']}：{r['title']}]({r['artifacts']['report']})" for r in prior+[current]))
    history.to_csv(ROOT/'reports/history_comparison.csv',index=False)
    app=report/'app'
    snapshot=json.loads((app/'src/data.json').read_text())
    snapshot.update(title=title+'实验报告',buildStatus='complete',status='reviewed',
                    generatedAt=dt.datetime.now(dt.UTC).isoformat(),report={'asOf':'2025-12-31'},
                    reportContent=[{'title':t,'markdown':s} for t,s in zip(TITLES,sections)])
    def query(frame,file,definition):
        source_path=f'reports/experiments/{args.experiment}/{file}' if (report/file).exists() else f'data/results/{args.experiment}/{file}'
        return {'rows':records(frame),'source':{'label':file,'files':[source_path],
                'filters':['2025-02-01 至 2025-12-31'],'metricDefinitions':[{'label':'口径','definition':definition}]}}
    queries=snapshot['queries']
    queries['summary']=query(summary,'dispatch_metrics.csv','三个种子平均预测、过去验证选模的全年正式费用。')
    queries['monthly_forecast']=query(monthly_full,'forecast_metrics.csv','按月汇总全部预报发布—目标区间组合的误差；光伏可选实际发电区间。')
    queries['lead_forecast']=query(lead,'forecast_metrics.csv','按提前量分组，误差总和除以样本数。')
    queries['annual_forecast']=query(yearly,'annual_forecast_metrics.csv','每个随机种子分别计算误差；mean 为先平均预测再计算误差。')
    queries['seed_statistics']=query(seed_stats,'annual_forecast_metrics.csv','三个种子误差的算术平均与样本标准差，标准差采用自由度 1。')
    queries['seed_cost_statistics']=query(seed_cost_stats,'dispatch_metrics.csv','三个种子全年费用与紧急购电量的均值和样本标准差。')
    queries['correction_comparison']=query(correction,'dispatch_metrics.csv','门控循环网络其余部分不变，仅比较是否修正已发布光伏预报。')
    queries['model_selection']=query(selected_table,'model_selection.csv','每月只依据此前七个完整日期的调度费用选择，表中为三个种子平均预测的实际选模结果。')
    queries['selected_forecasts']=query(selected_forecasts,'selected_forecast_metrics.csv','正式策略所选网络的统一四次发布预测评价；问题 2 与问题 4-2 的实际决策只使用凌晨预测。')
    queries['cost_comparison']=query(comparables,'dispatch_metrics.csv','相同计费、物理边界和更新时刻下的全年费用。')
    queries['failure_case']=query(case,'failure_case.csv','问题 4-3 全年最高费用日，预测采用每个区间之前最近发布的版本，仅用于事后诊断。')
    curves=[]
    for r in records(case):
        for variable in ('load','pv','price'):
            for column,label in [('actual','实际值'),('forecast','区间之前最近的预测')]:
                curves.append({'date':r['date'],'hour':r['hour'],'variable':variable,
                               'series':label,'value':r[f'{column}_{variable}']})
    queries['failure_curves']=query(pd.DataFrame(curves),'failure_case.csv','从失败案例数据投影，实际值与此前最新预测分别呈现。')
    queries['daily_primary']=query(primary_daily,'daily_primary.csv','正式策略的逐日费用、紧急购电与真实储电状态。')
    queries['update_comparison']=query(updates,'dispatch_metrics.csv','固定模型，改变可用的日内预报时刻。')
    queries['specified_days']=query(specified,'specified_dates.csv','题目指定日期，从逐区间结果直接汇总。')
    periods=[]
    for r in records(specified):
        for hour in (10,12,14,16,18,20):
            periods.append({'scenario':r['scenario'],'date':r['date'],'period':f'{hour}:00–{hour}:10',
                            'original':r[f'plan_{hour:02d}'],'final':r[f'grid_{hour:02d}']})
    queries['specified_intervals']=query(pd.DataFrame(periods),'specified_dates.csv','指定十分钟区间的凌晨计划与最终执行前承诺购电量，单位千瓦时。')
    for key,file in [('specified_battery','battery_blocks.csv'),('specified_emergency','emergency_periods.csv')]:
        values=pd.read_csv(out/file,dtype={'scenario':str})
        values=values[values.date.isin(specified.date.unique())]
        values.to_csv(report/file,index=False)
        queries[key]=query(values,file,'题目指定日期的实际执行结果，电量单位千瓦时。')
    queries['history']={'rows':records(history),'source':{'label':'实验注册表','files':['reports/registry/'],
        'metricDefinitions':[{'label':'可比性','definition':'评价时期、时间对齐、计费、物理约束、指标定义及输入哈希相同时标为同口径；模型结构允许改变。'}]}}
    queries['relative_history']={'rows':records(comparisons),'source':{'label':'全部历史实验逐项对比',
        'files':['reports/registry/'],'metricDefinitions':[{'label':'相对变化',
        'definition':'100 ×（本次值−此前值）／此前值绝对值；零分母或不可比时不计算。'}]}}
    queries['technical_path']={'rows':[{'step':i+1,'operation':t} for i,t in enumerate(tech)],
        'source':{'label':'本次实现与协议','files':[f'reports/experiments/{args.experiment}/protocol.json'],
                  'metricDefinitions':[{'label':'步骤','definition':'按数据处理与决策依赖顺序列出的实际实现步骤。'}]}}
    for i,section in enumerate(snapshot['reportContent'],1):
        section['blocks']=presentation_blocks(section['markdown'],i,queries,args.experiment)
    (app/'src/data.json').write_text(json.dumps(snapshot,ensure_ascii=False,indent=2,allow_nan=False))
    print(json.dumps({'report':str(report/'report.md'),'fits':len(training),'training_minutes':seconds/60,
                      'summary':records(summary)},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
