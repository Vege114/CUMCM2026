"""Build the accepted exp008 report from frozen, independently checked results.

This command reads selected archives and never trains or runs a planner.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
import re
import time
from pathlib import Path

import markdown
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np
import pandas as pd

from experiments.exp008.report_payload import BASELINE_COST, enforce_final
from experiments.exp008.frozen_sources import final_template

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'reports/experiments/exp008'
EVIDENCE=REPORT/'evidence'
FIGURES=REPORT/'figures/report'
SCENARIOS=('2','3','4-2','4-3')
LABELS={'2':'第二问','3':'第三问','4-2':'第四问方案2','4-3':'第四问方案3'}
MODELS={'2':'HGB/ExtraTrees融合；10分钟模式+购电精修', '3':'单HGB；6小时更新LP+物理贪心',
    '4-2':'单HGB+联动价格；小时模式+购电精修','4-3':'单HGB+原因果价格；6小时更新LP'}
plt.rcParams.update({'font.family':'Arial Unicode MS','font.size':10,'axes.spines.top':False,
    'axes.spines.right':False,'axes.unicode_minus':False,'svg.fonttype':'path','path.simplify':False})


def read(path): return json.loads(Path(path).read_text())
def save(path, value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def clean(v):
    if v is None or isinstance(v,float) and not np.isfinite(v): return None
    if isinstance(v,np.generic): return v.item()
    return v
def fmt(v, digits=2):
    v=clean(v)
    if v is None:return '未记录'
    if isinstance(v,bool):return '是' if v else '否'
    if isinstance(v,(float,int)):return f'{v:,.{digits}f}'
    return str(v).replace('|','\\|').replace('\n',' ')
def table(rows, columns, digits=2):
    return '\n'.join(['| '+' | '.join(label for _,label in columns)+' |', '|'+'---|'*len(columns),
        *['| '+' | '.join(fmt(row.get(key),digits) for key,_ in columns)+' |' for row in rows]])
def figure(path, caption):return f'![{caption}]({Path(path).relative_to(REPORT)})\n\n{caption}'
def pair(fig,name):
    fig.savefig(FIGURES/f'{name}.png',dpi=160,bbox_inches='tight')
    fig.savefig(FIGURES/f'{name}.svg',bbox_inches='tight')
    plt.close(fig)
    return FIGURES/f'{name}.png'


def history_evidence(payload):
    audited=read(ROOT/'data/results/exp008/baselines.json')
    rows=[]
    for r in audited['q2_rows']:
        rows.append({**r,'current':payload['scenarios']['2']['fees']['total_cost_yuan'],
            'previous':r['total_cost'], 'absolute_change':payload['scenarios']['2']['fees']['total_cost_yuan']-r['total_cost'] if r.get('ranking_allowed') else None,
            'relative_change_pct':100*(payload['scenarios']['2']['fees']['total_cost_yuan']-r['total_cost'])/abs(r['total_cost']) if r.get('ranking_allowed') and r['total_cost'] else None})
    pd.DataFrame(rows).to_csv(EVIDENCE/'cost_history.csv',index=False)
    selected=['exp001/legacy_rebased','exp002/primary','exp003/primary','exp004/causal_season','exp005/beta_0.1','exp006/primary']
    chart=[next(r for r in rows if r['policy_id']==k) for k in selected]
    chart.append({'experiment':'exp008','label':'exp008 用户接受版本','total_cost':payload['scenarios']['2']['fees']['total_cost_yuan'],'ranking_allowed':True})
    fig,ax=plt.subplots(figsize=(10,5.1)); y=np.arange(len(chart)); costs=np.array([r['total_cost']/1e4 for r in chart])
    bars=ax.barh(y,costs,color=['#bbc2c8' if not r.get('ranking_allowed') else '#27647b' if r['experiment']=='exp008' else '#82a6b4' for r in chart],height=.6)
    for bar,r,value in zip(bars,chart,costs):
        if not r.get('ranking_allowed'):bar.set_hatch('///')
        ax.text(value+8,bar.get_y()+bar.get_height()/2,f'{value:,.2f}',va='center')
    ax.set_yticks(y,[r['label'] for r in chart]);ax.invert_yaxis();ax.set_xlim(0,costs.max()*1.15)
    ax.set_xlabel('334日实际购电总费用（万元）');ax.set_title('第二问历史费用：exp001–006与当前版本')
    fig.text(.01,.015,'exp001为同物理重算；exp005带额外硬爬坡且允许应急充电，斜线单列、无改善率。种子42。',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,1)); hp=pair(fig,'q2_history_cost')
    index=pd.read_csv(ROOT/'data/results/exp008/development_index.csv')
    levels=[('单HGB+同基础LP','forecast_absolute_hgb/lp_bridge/direct_hgb_ridge28_memory_tree28_q08_buffer500_334days'),
        ('融合预测+同基础LP','forecast_hgb_extra_trees_half/lp_bridge/hgb_extra_trees_half_ridge28_memory_tree28_q08_buffer500_334days'),
        ('融合预测+最终模式规划','mode_budget_hold1_physical/hgb_extra_trees_half_hold1_cap8_kappa0_334days')]
    core=[]
    for label,run in levels:
        r=index[index.run==run].iloc[0].to_dict();core.append({'label':label,'run':run,**{k:clean(r[k]) for k in ['total_cost','direction_reversals','throughput_kwh','active_slots','archive_sha256']}})
    pd.DataFrame(core).to_csv(EVIDENCE/'three_layer_control_comparison.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for ax,key,unit in zip(axes,['total_cost','direction_reversals'],['总费用（万元）','非空方向反转（次）']):
        vals=np.array([r[key] for r in core])/(1e4 if key=='total_cost' else 1)
        ax.bar(np.arange(3),vals,color=['#b2c5ce','#78a2b3','#27647b'])
        ax.set_xticks(np.arange(3),['单HGB\n基础LP','融合\n基础LP','融合\n最终模式']);ax.set_ylabel(unit);ax.set_ylim(0,vals.max()*1.15)
        for i,v in enumerate(vals):ax.text(i,v+vals.max()*.025,fmt(v,2 if key=='total_cost' else 0),ha='center')
    fig.suptitle('同一基础调度下改预测，再加入低换向模式约束');fig.tight_layout();cp=pair(fig,'three_layer_control')
    return rows,chart,core,hp,cp


def flow_diagram():
    fig,ax=plt.subplots(figsize=(12,6.5));ax.set_xlim(0,12);ax.set_ylim(0,6.5);ax.axis('off')
    def box(x,y,w,h,text):
        ax.add_patch(Rectangle((x,y),w,h,facecolor='#f3f6f8',edgecolor='#6f858f',lw=1))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=10)
    def arrow(a,b):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=11,color='#5b6f78',lw=1.1))
    box(.1,5.25,2.3,.9,'附件1给定单日\n电价、负荷、光伏')
    box(3,5.25,3,.9,'第一问：先最小费用\n再在0.1%容差内平稳功率')
    box(8.1,5.25,3.4,.9,'144槽最终计划\n日初末SOC均6000kWh');arrow((2.4,5.7),(3,5.7));arrow((6,5.7),(8.1,5.7))
    box(.1,3.45,2.3,1.1,'过去已完成实况\n31个历史/日历特征\n按月训练、过去7日验证')
    box(3,3.45,3,1.1,'HGB + ExtraTrees\n原始预测固定等权融合\nRidge28 → 0.5×昨日日均残差')
    box(6.65,3.45,2.2,1.1,'第二问\n3场景物理模式MIP\n每日计划最多8次变化')
    box(9.4,3.45,2.45,1.1,'固定模式、全28路径\n按实际贪心精修购电\n当前实况反馈执行')
    for a,b in [((2.4,4),(3,4)),((6,4),(6.65,4)),((8.85,4),(9.4,4))]:arrow(a,b)
    box(.1,1.4,2.3,1.1,'单HGB已发布预测\n同样校准和日记忆\n各历史日使用自身发布值')
    box(3.3,1.9,4.3,.85,'第三问 / 4-3：官方PV更新\n7路径LP → 物理贪心执行')
    box(3.3,.8,4.3,.85,'第四问方案2：负荷/PV联动价格\n小时模式+购电精修')
    box(9.4,1.4,2.45,1.1,'每问自身实际SOC连续\n按实际价格独立结算\n费用、功率和边界复核')
    arrow((2.4,2.2),(3.3,2.3));arrow((2.4,1.65),(3.3,1.2));arrow((7.6,2.3),(9.4,2.2));arrow((7.6,1.2),(9.4,1.65))
    ax.text(.1,.25,'第三、4-3问保留单HGB和原价格回归；4-2采用新联动价格。各问模型差异保留，不冒称共享最终融合模型。',fontsize=10)
    ax.set_title('数据、预测、规划、执行与结算的实际交接关系',loc='left',fontsize=14)
    return pair(fig,'method_flow')


def monthly_figure(payload):
    rows=[{'scenario':s,**r} for s,c in payload['scenarios'].items() for r in c['monthly']]
    frame=pd.DataFrame(rows);frame.to_csv(EVIDENCE/'monthly_cost.csv',index=False)
    top=frame.total_cost_yuan.max()/1e4*1.13
    fig,axes=plt.subplots(2,2,figsize=(11,7),sharey=True,sharex=True)
    for ax,s in zip(axes.ravel(),SCENARIOS):
        f=frame[frame.scenario==s];x=np.arange(len(f));bottom=np.zeros(len(f))
        for key,label,color in [('planned_cost_yuan','计划费','#27647b'),('adjustment_cost_yuan','调整费','#8faeb9'),('emergency_cost_yuan','紧急费','#bf8752')]:
            v=f[key].to_numpy()/1e4;ax.bar(x,v,bottom=bottom,color=color,label=label);bottom+=v
        ax.set_title(LABELS[s]);ax.set_xticks(x,[m[-2:] for m in f.month]);ax.set_ylim(0,top);ax.set_xlabel('2025年月');ax.set_ylabel('费用（万元）')
    axes[0,0].legend(frameon=False,ncol=3,fontsize=9);fig.suptitle('四个年度方案的逐月费用分项（统一纵轴）');fig.tight_layout()
    return frame,pair(fig,'monthly_cost')


def multi_question_history(payload, history_chart):
    rows=read(EVIDENCE/'history_other_questions_comparability.json')['rows']
    out=[{'scenario':'2','previous':BASELINE_COST,'current':payload['scenarios']['2']['fees']['total_cost_yuan'],
          'reference':'exp006','physical_settlement_comparable':True}]
    out.extend({**r,'reference':'exp002'} for r in rows)
    for row in out:
        row['question']=LABELS[row['scenario']]
        row['absolute_change_yuan']=row['current']-row['previous']
        row['relative_change_pct']=100*row['absolute_change_yuan']/row['previous']
    pd.DataFrame(out).to_csv(EVIDENCE/'same_question_history_cost.csv',index=False)
    fig,axes=plt.subplots(2,2,figsize=(12,8),sharey=True)
    top=max(r['total_cost'] for r in history_chart)/1e4*1.17
    for ax,s in zip(axes.ravel(),SCENARIOS):
        if s=='2':
            vals=[r['total_cost']/1e4 for r in history_chart]
            labels=[r['experiment'].replace('exp','') for r in history_chart]
            colors=['#bbc2c8' if not r.get('ranking_allowed') else '#27647b' if r['experiment']=='exp008' else '#82a6b4' for r in history_chart]
        else:
            r=next(r for r in rows if r['scenario']==s)
            vals=[r['previous']/1e4,r['current']/1e4];labels=['002','008'];colors=['#82a6b4','#27647b']
        bars=ax.bar(np.arange(len(vals)),vals,color=colors,width=.58)
        if s=='2':bars[4].set_hatch('///')
        for i,v in enumerate(vals):ax.text(i,v+top*.015,fmt(v,1),ha='center',fontsize=9)
        ax.set_xticks(np.arange(len(vals)),labels);ax.set_ylim(0,top);ax.set_xlabel('实验编号 exp');ax.set_ylabel('实际总费用（万元）');ax.set_title(LABELS[s])
    fig.suptitle('各问历史费用与最终结果：同一费用纵轴')
    fig.text(.02,.018,'Q2列全历史正式角色（001同物理重算；005斜线为不同物理设置）。其他问可用年度历史为002；未记录不补零。',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,.97))
    return out,pair(fig,'all_questions_history_cost')


def specified_tables(payload):
    pieces=['# exp008 题目指定日期完整表格','所有电量单位kWh、费用单位元。区间左闭右开；原数据右端00:10对应00:00–00:10。调整购电量填最终绝对量，非增量。']
    for s,case in [('1',payload['q1']),*payload['scenarios'].items()]:
        days=[case['specified_tables']] if s=='1' else case['specified_dates']
        pieces += [f'## 问题{s}',f'[完整成果工作簿](result{s}.xlsx)']
        for row in days:
            if s=='1':
                original=final=sum(r[1] for r in case['workbook_data']['plan_rows'])
            else:
                daily=next(r for r in case['daily'] if r['date']==row['date'])
                original=daily['original_purchase_kwh'];final=daily['final_purchase_kwh']
            day_label='附件1给定单日' if s=='1' else row['date']
            pieces += [f'### {day_label}', '表1：指定六个十分钟槽位',table(row['table1'],[('interval','时间段'),('original_purchase_kwh','原计划购电'),('final_purchase_kwh','最终购电'),('adjustment_delta_kwh','调整增量')],4),
                f'全天总购电量：原计划{original:,.4f}kWh；最终计划{final:,.4f}kWh（紧急购电另列）。',
                '表2：六个四小时区间',table(row['table2']['four_hour_blocks'],[('interval','时间段'),('charge_kwh','实际充电量'),('discharge_kwh','实际放电量')],4),
                f'00:00储电量{row["table2"]["soc_00_kwh"]:,.4f}kWh；24:00储电量{row["table2"]["soc_24_kwh"]:,.4f}kWh。',
                '表3：全部连续紧急购电区间',table(row['table3']['events'],[('interval','时间段'),('minutes','持续分钟'),('energy_kwh','紧急购电量')],4) if row['table3']['events'] else '无紧急购电。',
                table([row['fees']],[('planned_cost_yuan','计划费'),('adjustment_cost_yuan','调整费'),('emergency_cost_yuan','紧急费'),('total_cost_yuan','总费用')],4)]
    (REPORT/'specified_dates.md').write_text('\n\n'.join(pieces)+'\n')


def timing_rows():
    records=[]
    def number(v):return v.get('sum') if isinstance(v,dict) else v
    for r in read(EVIDENCE/'runtime_by_stage.json')['components']:
        records.append({'item':r['id'],'models':r.get('model_count'), 'fit':number(r.get('fit_plus_validation_prediction_seconds')),
            'solver':number(r.get('solver_seconds',r.get('sum_solver_seconds'))),'refine':number(r.get('refinement_seconds')),
            'wall':r.get('run_wall_seconds'),'prediction':number(r.get('prediction_seconds')),'calibration':number(r.get('calibration_seconds'))})
    pd.DataFrame(records).to_csv(EVIDENCE/'runtime_summary.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(10,4.1))
    fits=[r for r in records if r['fit'] is not None];solves=[r for r in records if r['solver'] is not None]
    for ax,selected,key,title in [(axes[0],fits,'fit','树模型拟合+验证集预测累计'),(axes[1],solves,'solver','各问求解器累计')]:
        vals=[r[key] for r in selected];ax.barh(np.arange(len(vals)),vals,color='#527f90');ax.set_yticks(np.arange(len(vals)),[r['item'].replace('forecast_absolute_','') for r in selected]);ax.invert_yaxis();ax.set_xlabel('实测秒');ax.set_title(title)
        ax.set_xlim(0,max(vals)*1.25)
        for i,v in enumerate(vals):ax.text(v+max(vals)*.015,i,fmt(v,2),va='center')
    fig.tight_layout();return records,pair(fig,'recorded_runtime')


def historical_runtime():
    rows=read(EVIDENCE/'history_runtime_descriptive.json')['rows']
    measured=[r for r in rows if r['seconds'] is not None]
    fig,ax=plt.subplots(figsize=(11,6))
    values=[r['seconds'] for r in measured]
    ax.barh(np.arange(len(measured)),values,color='#82a6b4',height=.58)
    ax.set_yticks(np.arange(len(measured)),[r['id']+' · '+r['stage_zh'] for r in measured]);ax.invert_yaxis()
    ax.set_xlabel('原日志实测秒（各行计时范围不同）');ax.set_xlim(0,max(values)*1.23)
    for i,v in enumerate(values):ax.text(v+max(values)*.012,i,fmt(v,2),va='center')
    ax.set_title('历史计算代价：按原阶段描述，不能据此排名端到端速度')
    fig.tight_layout();return rows,pair(fig,'historical_runtime_descriptive')


def build(payload_path,code_commit):
    began=time.perf_counter();payload=read(payload_path);enforce_final(payload)
    FIGURES.mkdir(parents=True,exist_ok=True)
    cases=payload['scenarios'];history,history_chart,core,histfig,corefig=history_evidence(payload)
    monthly,monthfig=monthly_figure(payload);timings,timefig=timing_rows();flowfig=flow_diagram();specified_tables(payload)
    other_history,all_histfig=multi_question_history(payload,history_chart)
    history_timings,historical_timefig=historical_runtime()
    annual=pd.read_csv(EVIDENCE/'forecast/annual_metrics.csv')
    methods=(REPORT/'methods.md').read_text();nested=re.sub(r'(?m)^(#{1,4}) ',lambda m:'#'*(len(m[1])+2)+' ',methods)
    validation=read(EVIDENCE/'data_validation.json');solver=read(EVIDENCE/'planning_solver_diagnostics.json')
    primary=[{'question':'第一问','method':'main最新两层确定性规划','period':'附件1给定单日','planned':payload['q1']['verification']['recomputed_total_cost'], 'adjustment':0,'emergency':0,'total':payload['q1']['verification']['recomputed_total_cost'],'emergency_kwh':0,'reference':'费用容差0.1%的阶段2'}]
    for s,c in cases.items():
        f=c['fees'];primary.append({'question':LABELS[s],'method':MODELS[s],'period':'334日', 'planned':f['planned_cost_yuan'],'adjustment':f['adjustment_cost_yuan'],'emergency':f['emergency_cost_yuan'],'total':f['total_cost_yuan'],'emergency_kwh':sum(r['emergency_kwh'] for r in c['daily']),
            'reference':f'较exp006：{100*(f["total_cost_yuan"]/BASELINE_COST-1):.4f}%' if s=='2' else '与各问自身历史版本比较，见第7节'})
    pd.DataFrame(primary).to_csv(EVIDENCE/'primary_results.csv',index=False)
    battery_rows=[{'question':LABELS[s],**c['battery']} for s,c in cases.items()]
    other_forecasts=read(EVIDENCE/'other_question_forecasts.json')
    issue_rows=[r for r in other_forecasts['annual_metrics'] if r['horizon'] in ('midnight_24h','next_6h') and (r['model']=='updated_issued' or r['scenario']=='4-2')]
    issue_rows=[{**r,'question':LABELS[r['scenario']]} for r in issue_rows]
    issue_comparisons=[{'question':LABELS[r['scenario']], 'target':r['target'], 'unit':r['current']['unit'],
                       'previous':r['previous']['rmse'],'current':r['current']['rmse'],'change':-r['rmse_reduction_pct'],
                       'wape_delta':r['wape_change_percentage_points']} for r in other_forecasts['controlled_forecast_comparisons']]
    data_rows=[{'file':r['source']['path'],'shape':'×'.join(map(str,r['shape'])),'missing':r.get('numeric_missing_count'),'hash':r['source']['sha256']} for r in validation['files']]
    gap_rows=[]
    for s,r in solver['scenarios'].items():
        if s=='1':continue
        gap_rows.append({'question':LABELS[s],'count':r.get('solve_count'),'feasible':r.get('feasible_count'), 'gapmean':100*r['gap']['mean'] if r.get('gap',{}).get('mean') is not None else None,
            'gapmax':100*r['gap']['max'] if r.get('gap',{}).get('max') is not None else None})
    forecast_columns=[('model','预测版本'),('target','目标'),('population','样本'),('n','槽位数'),('mae_kw','MAE/kW'),('rmse_kw','RMSE/kW'),('wape_pct','WAPE/%'),('bias_kw','偏差/kW')]
    bad=[{'question':LABELS[s],**r} for s,c in cases.items() for r in c['worst_days_by_bill'][:1]]
    sections={}
    sections[1]='\n\n'.join([
        '**第二问实际总费用13,201,981.95元，较exp006减少864,275.53元（6.1443%），非空方向反转2729→2533（减少7.1821%）。** 用户接受当前已验证结果并要求定稿，原先8%费用目标仍未达到，未将未完成试验作为结果。',
        table(primary,[('question','问题'),('method','采用模型与调度'),('period','范围'),('planned','计划费/元'),('adjustment','调整费/元'),('emergency','紧急费/元'),('total','总费用/元'),('emergency_kwh','紧急购电/kWh'),('reference','基线说明')]),
        '全部最终实际轨迹满足SOC、效率、最大功率及充放电互斥，紧急购电不用于给电池充电。第二问吞吐量、功率总变差降低，但连续充放电段3791→3898、活动槽23028→37211、相邻直接反转730→1407增加，不能说所有行为指标都下降。',
        '各问保留各自完整验证版本：Q2为HGB/ExtraTrees融合，Q3与4-3为单HGB滚动更新，Q4-2为单HGB和新联动电价。各问不共享同一个最终预测器，Q3/4-3也未实施Q2的8次计划模式预算。'])
    sections[2]='\n\n'.join([
        '评价区间为2025-02-01至12-31，334日、48096个十分钟槽；一月为冷启动和预热，未计入主费用及功率变化。第一问另用附件1给定单日，不与年度费用相加排名。',
        table([{'metric':'MAE','definition':'Σ|预测−实测|/n','unit':'kW','aggregation':'从误差总和重算'}, {'metric':'RMSE','definition':'√(Σ误差²/n)','unit':'kW','aggregation':'不平均月度RMSE'}, {'metric':'WAPE','definition':'100Σ|误差|/Σ|实测|','unit':'%','aggregation':'零分母标缺失；差值单位百分点'}, {'metric':'总费用','definition':'原计划费+最终调整惩罚+5倍紧急购电费','unit':'元','aggregation':'逐槽真实结算，不含辅助罚和终值'}, {'metric':'非空换向','definition':'删除空闲动作后相邻方向翻转数','unit':'次','aggregation':'跨日串联，排除预热边界'}, {'metric':'功率总变差','definition':'Σ|Pₜ−Pₜ₋₁|，P=6(c−d)','unit':'kW','aggregation':'连续原始时点，不平滑'}],[('metric','指标'),('definition','精确定义'),('unit','单位'),('aggregation','汇总')]),
        '电池12000kWh、SOC1200–10800kWh、双向5000kW、ηc=ηd=√0.9。效率固定，未拟合温度、倍率和老化；EFC和次数只作为运行强度代理。费用优先，其他目标不作为实际收入或折扣。',
        'Q2午夜发布后购电锁定；Q3/4-3只在0/6/12/18点使用已发布官方预报。Q4真实未来价格仅用于事后结算。所有历史残差均使用对应历史时点自身已发布的预测。'])
    sections[3]='\n\n'.join([
        '原始字节SHA256、表格形状、重复列、缺失与时间轴核验如下；完整证据见[data_validation.json](evidence/data_validation.json)。区间右端00:10对应00:00–00:10，所有工作簿采用修正后的144个时段。',
        table(data_rows,[('file','数据'),('shape','形状'),('missing','数值缺失'),('hash','SHA256')],0),
        '月度训练使用发布之前的数据，最近7个完整日验证，训练样本采用90日半衰期；历史校准最多28日、14日半衰期。1月冷启动有明确周期基线标记，不能伪称真实历史HGB发布。未来实况和未发布预报扰动检查通过。',
        '**2025年已被反复用于开发分析，不是未触碰的独立测试集。** 因果输入审计证明每次运行的信息边界，不消除跨候选选择造成的开发集偏差。'])
    sections[4]=figure(flowfig,'图：实际数据交接流程；最终树模型的并行融合与各问调度分支。')+'\n\n'+nested
    sections[5]='\n\n'.join([
        table([{'item':'第一问','setting':'互斥；1000kW/10min爬坡；最短开3/关2槽；SOC起止6000；费用容差0.1%'}, {'item':'HGB','setting':'2通道×11月，种子42；最多100轮、15叶、min_leaf50、L2=2、learning_rate0.08'}, {'item':'ExtraTrees','setting':'2通道×11月，种子42；128树、min_leaf10、max_features1、bootstrap=False'}, {'item':'融合与后处理','setting':'raw等权0.5/0.5；统一Ridge28；昨日负荷日均残差×0.5'}, {'item':'Q2规划','setting':'3条历史路径初始化物理MIP；hold1；每日计划变化≤8；5秒、gap目标0.2%；全28路径精修≤120次'}, {'item':'Q3/4-3','setting':'同发布历史7条代表路径LP；下一更新时间后短缺权重代理1.5；实际充电死区20kWh'}, {'item':'Q4-2','setting':'自身已发布负荷/PV加入10列价格岭回归；小时模式switching=50、wear=.002；5秒/gap.002；精修120次；终值.45(末日0)、deadband0；真实电价计费'}],[('item','配置'),('setting','实际设定')]),
        '44个树模型是两个家族、两个通道和11个月份，不是44个独立随机种子。正式种子仅42；跨种子样本数1，样本标准差未记录，不能用月间差异替代种子稳定性。随机示例日种子20260912仅用于画图。',
        '场景规划的未来动作是乐观近似，真实执行才严格逐槽因果。MIP有固定预算、精修为局部非光滑数值法，记录可行性和gap，不声称全局最优。完整超参及训练边界见[模型配置](evidence/forecast/model_configuration.json)、[求解记录](evidence/planning_solver_diagnostics.json)。'])
    forecasts='\n\n'.join(figure(p,'预测诊断：'+p.stem.replace('_',' ')) for p in sorted((REPORT/'figures/forecast').glob('*.png')))
    battery_figures='\n\n'.join(figure(p,'实际电池功率：'+p.stem.replace('_',' ')) for p in sorted((REPORT/'figures/battery').glob('*.png')) if 'history' not in p.name)
    sections[6]='\n\n'.join([
        '以下主预测表评价午夜发布：final_blend为Q2最终融合，single_hgb为其他问基础负荷/PV预测。single_hgb不代表Q3/4-3后续官方PV更新后的最终预测；价格使用元/kWh另列，不能混入kW误差。',
        table(annual.to_dict('records'),forecast_columns,4),
        'all为全部48096槽；pv_generating为实际光伏大于0的26880槽，仅用于评价分组，不作为未来已知信息。load、pv、net分别表示负荷、光伏和净需求。逐月和全部144个提前槽保留在[monthly_metrics.csv](evidence/forecast/monthly_metrics.csv)与[lead_metrics.csv](evidence/forecast/lead_metrics.csv)。以下图与上述同一数据生成。',forecasts,
        '### 第三、第四问实际发布预测与电价误差',
        table(issue_rows,[('question','问题'),('model','发布版本'),('target','变量'),('unit','误差单位'),('population','样本'),('n','槽位'),('mae','MAE'),('rmse','RMSE'),('wape_pct','WAPE/%'),('bias','偏差')],6),
        'Q3/4-3按每次发布后六小时不重叠覆盖计分，共48096槽；剩余自然日全提前量另有120240槽，不能混作同一总体。2672次发布的三通道序列依据冻结校正量和原始输入算术重建，8016个发布哈希全部匹配，未重新拟合。逐月、发布时间和剩余提前槽见[完整误差CSV](evidence/other_question_forecasts.csv)及[来源审计](evidence/other_question_forecasts.json)。',
        table(issue_comparisons,[('question','问题'),('target','变量'),('unit','RMSE单位'),('previous','保持午夜/旧价格RMSE'),('current','实际更新/新价格RMSE'),('change','RMSE变化/%'),('wape_delta','WAPE变化/百分点')],6),
        '该表对Q3/4-3比较“保持当天午夜已发布预测”与“实际日内更新”，只衡量预测更新价值；4-2比较原因果价格与联动价格。负荷、PV、净需求与电价保持各自单位。上述预测误差变化不等同于调度节费，也不证明每次更新具有经济必要性。',
        figure(monthfig,'四个年度问题的逐月费用分解，统一纵轴。'),
        table(battery_rows,[('question','问题'),('direction_reversals','非空换向/次'),('active_slots','活动槽'),('throughput_kwh','交流吞吐/kWh'),('equivalent_full_cycles','EFC'),('simultaneous_slots','同时充放槽'),('power_ramp_total_kw','功率总变差/kW')]),
        table(gap_rows,[('question','问题'),('count','求解次数'),('feasible','可行数'),('gapmean','平均gap/%'),('gapmax','最大gap/%')],4),
        '### 电池充放电功率与波动核验\n\n四个统一随机日为2025-04-15、06-07、12-06、12-25，随机种子20260912；另检查题目指定03-20、06-21、09-23、12-21。每张年度图保留48096个实际时点；每张日图保留144槽，不平滑或降采样。正为充电、负为放电。第一问为单独144槽。',battery_figures,
        '[全部原始功率、差分和计数依据](evidence/battery/README.md)。Q2比较排除1月预热边界；全年方向和TV包括2–12月之间的日边界。四小时汇总内充放两列都为正不代表同一十分钟同时充放。',
        '### 最贵日期与失败边界',table(bad,[('question','问题'),('date','日期'),('planned_cost_yuan','计划费/元'),('adjustment_cost_yuan','调整费/元'),('emergency_cost_yuan','紧急费/元'),('total_cost_yuan','总费/元')]),
        '这些是各方案自身最贵日期，不将费用最大直接解释为预测最差。费用还受到电价、负荷规模、SOC和可用调整机会影响。历史全路径分位、更多求解场景、按周重训及日内误差状态等候选均有未改善案例，完整开发摘要保留在[种子与消融证据](evidence/seed_and_ablation_results.json)。用户停止时两个自适应DP年度候选尚未完整，禁止和334日正式费用比较。',
        '### 计算时间',table(timings,[('item','阶段/方案'),('models','模型数'),('fit','拟合+验证预测/秒'),('solver','求解器累计/秒'),('refine','精修累计/秒'),('wall','运行墙钟/秒'),('prediction','预测/秒'),('calibration','校准/秒')]),figure(timefig,'已记录的拟合+验证预测和求解时间；缺失不补0，累计阶段不可直接当端到端墙钟。'),
        '训练、预测、风险校准、实际回放、导出和报告的计时范围并不完整。未记录值明确保留；报告和导出自身新测耗时分别记录在report_build.json和工作簿QA中，不反推既有阶段。',
        '[指定日期完整表格](specified_dates.md)。五份附件格式成果分别为[result1.xlsx](result1.xlsx)、[result2.xlsx](result2.xlsx)、[result3.xlsx](result3.xlsx)、[result4-2.xlsx](result4-2.xlsx)、[result4-3.xlsx](result4-3.xlsx)。'])
    history_table=[r for r in history if r['policy_id'] in {x.get('policy_id') for x in history_chart}|{'exp001/original','exp004/no_season','exp006/greedy_execution'}]
    sections[7]='\n\n'.join([
        'exp001–006全部历史均保留，exp007按用户要求忽略。各实验原角色、原登记、同口径重算与开发消融分别标明；本轮候选不冒充新的历史实验。exp005注册表从其固定Git提交读取并核验哈希，不因当前工作树缺少文件而遗漏。',
        figure(histfig,'第二问历史费用原值。exp005物理设置不同，只描述性并列，不连线排名或计算改善率。'),
        figure(all_histfig,'各问自身历史费用比较，所有子图采用相同费用尺度；其他问未登记的实验不填零。'),
        table(other_history,[('question','问题'),('reference','历史基准'),('previous','历史费/元'),('current','本次费/元'),('absolute_change_yuan','本次−历史/元'),('relative_change_pct','变化/%')],4),
        table(history_table,[('experiment','实验'),('label','原角色/策略'),('total_cost','历史费/元'),('current','本次费/元'),('absolute_change','本次−历史/元'),('relative_change_pct','变化/%'),('ranking_allowed','同口径排名'),('comparison_note','边界')]),
        figure(corefig,'三层核心对照：先在同一基础LP下改预测，再增加低换向模式。'),table(core,[('label','对照'),('total_cost','总费/元'),('direction_reversals','非空换向/次'),('throughput_kwh','吞吐/kWh'),('active_slots','活动槽')]),
        '这里的“旧预测”是本轮之前的单HGB管线，与exp006的冻结CNN不是同一层对照。融合降低同基础LP费用；加入最终模式规划进一步降低换向，但费用相对无模式LP增加。不能把三层差额全部归因于某一个模型，也不能把LP费用和另一个控制器的低换向拼为同一个方案。',
        figure(REPORT/'figures/battery/q2_history_battery_intensity.png','历史电池强度：同时保留减少和增加的指标。'),
        figure(REPORT/'figures/forecast/history_midnight_comparable.png','历史预测在同一午夜发布口径下的RMSE与WAPE，保留各目标的改善和退步。'),
        '原登记192168个多次发布样本与48096个午夜重评分不得混用；单独标明实际发电时段。WAPE差值为百分点，不与相对变化百分比混淆。完整数据见[evidence/forecast](evidence/forecast/README.md)。',
        '第三、4-2、4-3问较exp002分别减少8.4170%、11.9918%、7.8389%。数据、评价期、效率和实际计费相同，允许展示历史费用数值变化；原exp002预测器通过四目标共同验证损失选择早停点，对Q3和4-2的严格同信息公平性存在限制。各问模型、预测及控制器均有变化，不能将差额全部归因于日内更新或某一单因素。详见[历史可比性审计](evidence/history_other_questions_comparability.json)。',
        figure(timefig,'本轮实际记录的分阶段时间。历史阶段与设备未统一，禁止从此图计算端到端加速率。'),
        figure(historical_timefig,'历史原登记计时与本轮对应实测阶段分列；不是同一工作量基准。'),
        table(history_timings,[('id','实验/阶段'),('stage_zh','计时范围'),('seconds','实测秒'),('field','原字段'),('source','来源')],4),
        '原始树模型拟合及随后的验证集预测累计约108.21秒；纯拟合时间未单独记录；未把模型复用次数重复计入训练。不同设备、并行负载、训练组数和阶段定义的历史计时只作描述，不能宣称端到端加速。所有原登记时间与缺失说明保留在对应历史注册表和计时证据中。'])
    sections[8]='\n\n'.join([
        f'实现源码提交：`{code_commit}`。报告模板取自main提交`{payload["templates"]["commit"]}`。最终选择及用户接受记录见[final_selection.json](../../../experiments/exp008/final_selection.json)，最终数据契约见[final_payload.json](final_payload.json)。',
        '```bash\nOMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest discover -s tests -p \'test_exp008*.py\' -v\nOMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.exp008.refresh_report_payload\n.venv/bin/python -m reports.build_report_exp008 --payload reports/experiments/exp008/final_payload.json --code-commit '+code_commit+'\n```',
        '换目录克隆后先运行refresh_report_payload，将来源路径绑定到当前工作树；它确认五份工作簿数据逐项不变才更新派生载荷。缺少main或exp005本地分支时使用明确的哈希锁定快照。重建报告只读取冻结结果，不训练或重新优化。全新算法复现实验应在独立输出位置运行：实验程序拒绝覆盖既有档案。不同机器在相同MIP秒数预算下可能产生不同可行解，未承诺逐位同值。工作簿使用捆绑Node与artifact-tool，具体命令和核验见工作簿交接文档。',
        '[相关测试记录](tests.txt)、[最终选择独立核验](evidence/final_selection_audit.json)、[电池原始轨迹核验](evidence/battery/independent_audit.json)、[预测数值和来源核验](evidence/forecast/verification.json)。正式交付检查另记录文件哈希、全部链接、工作簿、图和浏览器视觉结果。',
        '此前本地迭代遇到瓶颈时已完成一次定向论文检索；[下载来源、页码与适配记录](../../../data/results/exp008/literature/literature_notes.md)保留任务导向预测、机会价值控制及场景相关性等思路。未完成的价值反馈候选没有替代本版选定方案，论文中的市场收益不作为本题实测结论。',
        '原8%费用条件保持未通过。用户接受当前已验证档案后允许定稿，这一接受绑定Q2完整档案SHA256；未豁免物理、信息边界、完整334日、费用降低及换向减少要求。'])
    template_bytes, _=final_template('reports/templates/report.md',root=ROOT)
    template=template_bytes.decode()
    header=template.split('\n## ')[0]
    header=header.replace('data_hashes.json','[data_hashes.json](data_hashes.json)').replace('record.draft.json，登记后见 registry','[record.json](record.json)，[正式登记](../../registry/exp008.json)')
    for key,value in {'experiment_id':'exp008','title':'费用与换向改善的多问题优化成果','protocol_version':'exp008-accepted-2026-09-13','evaluation_period':'2025-02-01—12-31（334日），Q1为附件1单日','code_commit':code_commit}.items():header=header.replace('{{'+key+'}}',value)
    titles=re.findall(r'(?m)^## (\d+)\. (.+)$',template)
    body=header+'\n\n'+'\n\n'.join(f'## {i}. {title}\n\n{sections[int(i)]}' for i,title in titles)+'\n'
    assert len(titles)==8 and '{{' not in body
    (REPORT/'report.md').write_text(body)
    build_info={'status':'content_built_awaiting_delivery_QA','payload_sha256':sha(payload_path),'report_sha256':sha(REPORT/'report.md'),
        'code_commit':code_commit,'template_main_commit':payload['templates']['commit'],'original_eight_percent_target_met':False,
        'user_accepted_current_result':True,'sections':8,'build_seconds':time.perf_counter()-began,'new_optimization_executed':False,
        'figure_pairs':len(list((REPORT/'figures').rglob('*.png')))}
    save(REPORT/'report_build.json',build_info)
    render_html(REPORT/'report.md',REPORT/'report.html')
    render_html(REPORT/'specified_dates.md',REPORT/'specified_dates.html')
    build_info['html_sha256']=sha(REPORT/'report.html')
    build_info['build_seconds']=time.perf_counter()-began
    build_info['figure_pairs']=len(list((REPORT/'figures').rglob('*.png')))
    save(REPORT/'report_build.json',build_info)
    print(json.dumps(build_info,ensure_ascii=False,indent=2))


def render_html(source,destination):
    text=source.read_text()
    eq_dir=REPORT/'figures/equations';eq_dir.mkdir(exist_ok=True)
    counter=0
    def equation(match):
        nonlocal counter;counter+=1
        tex=match.group(1).strip();fig=plt.figure(figsize=(11,.8));fig.text(.02,.4,'$'+tex+'$',fontsize=16,va='center')
        output=eq_dir/f'{source.stem}-{counter:02d}.png';fig.savefig(output,dpi=180,bbox_inches='tight');fig.savefig(output.with_suffix('.svg'),bbox_inches='tight');plt.close(fig)
        return f'<div class="equation"><img alt="{html.escape(tex,quote=True)}" src="data:image/png;base64,{base64.b64encode(output.read_bytes()).decode()}" /></div>'
    text=re.sub(r'\$\$(.*?)\$\$',equation,text,flags=re.S)
    rendered=markdown.markdown(text,extensions=['tables','fenced_code','toc','sane_lists'])
    rendered=re.sub(r'<td>([+\-\d,.%]+)</td>',r'<td class="numeric">\1</td>',rendered)
    def embed(match):
        target=match.group(1)
        if target.startswith(('http','data:')):return match.group(0)
        path=source.parent/target
        if not path.is_file():return match.group(0)
        mime='image/svg+xml' if path.suffix=='.svg' else 'image/png'
        return f'src="data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"'
    rendered=re.sub(r'src="([^"]+)"',embed,rendered)
    rendered=rendered.replace('href="specified_dates.md"','href="specified_dates.html"')
    rendered=re.sub(r'<table>', '<div class="table-scroll"><table>',rendered).replace('</table>','</table></div>')
    nav=''.join(f'<a href="#{m[0]}">{html.escape(m[1])}</a>' for m in re.findall(r'<h2 id="([^"]+)">(.+?)</h2>',rendered))
    document='''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>exp008 实验报告</title><style>
    :root{color-scheme:light;--ink:#233440;--muted:#5c6b75;--rule:#dce3e7;--accent:#27647b}*{box-sizing:border-box}body{margin:0;background:#fff;color:var(--ink);font-family:-apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif;line-height:1.8;font-size:16px}header{border-bottom:1px solid var(--rule);padding:16px 24px}header span{color:var(--muted)}.layout{max-width:1440px;margin:auto;display:grid;grid-template-columns:230px minmax(0,1fr);gap:36px;padding:30px 32px}nav{align-self:start;position:sticky;top:20px;max-height:90vh;overflow:auto;font-size:14px}nav a{display:block;padding:5px 0;color:var(--muted);text-decoration:none}nav a:hover{color:var(--accent)}main{max-width:1100px;min-width:0}h1{font-size:30px;line-height:1.45;margin:0 0 28px}h2{font-size:25px;border-top:1px solid var(--rule);padding-top:30px;margin-top:44px}h3{font-size:20px;margin-top:32px}h4{font-size:17px}a{color:var(--accent)}p{margin:14px 0}strong{font-weight:600}img{display:block;max-width:100%;height:auto;margin:18px auto}.equation img{max-height:140px;width:auto;margin-left:0}.table-scroll{overflow-x:auto;margin:18px 0}table{border-collapse:collapse;width:100%;font-size:13px;line-height:1.6}th,td{padding:9px 11px;text-align:left;border-bottom:1px solid var(--rule);vertical-align:top}th{background:#edf3f6;font-weight:600;white-space:nowrap}td{overflow-wrap:anywhere}.numeric{white-space:nowrap;font-variant-numeric:tabular-nums}pre{background:#f3f6f8;padding:18px;overflow:auto;font-size:13px}code{font-family:ui-monospace,SFMono-Regular,monospace;font-size:.88em}blockquote{margin:16px 0;padding-left:16px;border-left:3px solid var(--rule)}button{font:inherit;font-size:14px;background:white;border:1px solid #b3c3cb;border-radius:4px;padding:6px 12px;color:var(--accent);cursor:pointer}footer{border-top:1px solid var(--rule);margin-top:40px;padding:25px;color:var(--muted)}@media(max-width:800px){.layout{display:block;padding:20px 16px}nav{position:static;max-height:none;display:flex;gap:12px;flex-wrap:wrap;margin-bottom:24px}nav a{padding:0}h1{font-size:25px}h2{font-size:22px}body{font-size:15px}th,td{padding:7px 9px}}@media print{nav,header button{display:none}.layout{display:block;padding:0}main{max-width:none}body{font-size:10pt}h2{break-before:page}img,table{break-inside:avoid}.table-scroll{overflow:visible}}
    </style></head><body><header><b>exp008</b> <span>数学建模实验记录</span> <button onclick="window.print()">打印</button></header><div class="layout"><nav aria-label="章节目录">'''+nav+'</nav><main>'+rendered+'</main></div><footer>费用按实际购电结算。2025年是开发评价年度；原8%目标未达到，按用户接受的完整结果定稿。</footer></body></html>'
    destination.write_text(document)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--payload',type=Path,default=REPORT/'final_payload.json');parser.add_argument('--code-commit',required=True)
    args=parser.parse_args();assert re.fullmatch('[0-9a-f]{40}',args.code_commit)
    build(args.payload,args.code_commit)
