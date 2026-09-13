"""Build the fixed cost-only control report without retraining or optimization."""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
import re
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from experiments.exp009.report_payload import ROOT, OUT, RESULTS, SPECS, read, save, sha, ref
import reports.build_report_exp008 as legacy

FIG=OUT/'figures/report';EV=OUT/'evidence'
SCENARIOS=tuple(SPECS)
LABEL={'1':'第一问','2':'第二问','3':'第三问','4-2':'第四问方案2','4-3':'第四问方案3'}
COLORS=['#789daa','#174c65']
table=legacy.table
def figure(p,caption):return f'![{caption}]({Path(p).relative_to(OUT)})\n\n{caption}'
def pair(fig,name):
 for ext in ('png','svg'):fig.savefig(FIG/f'{name}.{ext}',dpi=160,bbox_inches='tight')
 plt.close(fig);return FIG/f'{name}.png'
def history(payload,old):
 rows=[]
 chosen=['exp001/legacy_rebased','exp002/primary','exp003/primary','exp004/causal_season','exp005/beta_0.1','exp006/primary']
 baselines=read(ROOT/'data/results/exp008/baselines.json')
 for ident in chosen:
  b=next(r for r in baselines['q2_rows'] if r['policy_id']==ident)
  rows.append({'scenario':'2','experiment':b['experiment'],'role':b['label'],'cost':b['total_cost'],
   'comparable':bool(b.get('ranking_allowed')),'source':b.get('source'),'scope':b.get('comparison_note')})
 for b in read(ROOT/'reports/experiments/exp008/evidence/history_other_questions_comparability.json')['rows']:
  rows.append({'scenario':b['scenario'],'experiment':'exp002','role':'原正式策略','cost':b['previous'],
   'comparable':True,'source':'reports/registry/exp002.json','scope':'同物理结算；共同四目标早停的信息公平性限制保留'})
 for exp,p in [('exp008',old),('exp009',payload)]:
  for s,c in p['scenarios'].items():rows.append({'scenario':s,'experiment':exp,'role':'用户接受正式策略' if exp=='exp008' else '预先固定费用目标对照',
   'cost':c['fees']['total_cost_yuan'],'comparable':True,'source':c['archive']['path'],'scope':'334日同物理结算；exp008与009同预测'})
 for r in rows:
  r['current']=payload['scenarios'][r['scenario']]['fees']['total_cost_yuan'];r['previous']=r['cost']
  r['absolute_change_yuan']=r['current']-r['previous'] if r['comparable'] else None
  r['relative_change_pct']=100*r['absolute_change_yuan']/abs(r['previous']) if r['comparable'] and r['previous'] else None
 pd.DataFrame(rows).to_csv(EV/'history_cost.csv',index=False)
 fig,axes=plt.subplots(2,2,figsize=(12,8),sharey=True)
 top=max(r['cost'] for r in rows)/1e4*1.18
 for ax,s in zip(axes.ravel(),SCENARIOS):
  rr=[r for r in rows if r['scenario']==s];v=[r['cost']/1e4 for r in rr]
  bars=ax.bar(range(len(v)),v,color=[COLORS[1] if r['experiment']=='exp009' else '#a3b7bf' for r in rr])
  for i,(b,r) in enumerate(zip(bars,rr)):
   if not r['comparable']:b.set_hatch('///')
   ax.text(i,v[i]+top*.014,f'{v[i]:,.1f}',ha='center',fontsize=9)
  ax.set_xticks(range(len(rr)),[r['experiment'].replace('exp','') for r in rr]);ax.set_ylim(0,top)
  ax.set_title(LABEL[s]);ax.set_xlabel('实验编号 exp');ax.set_ylabel('334日实际费用（万元）')
 fig.suptitle('历次正式策略与费用目标对照：所有年度问统一费用尺度')
 fig.text(.02,.018,'001为同物理重算；005不同物理用斜线，不作排名或改善率。007忽略；其他问缺失历史不补零。',fontsize=9)
 fig.tight_layout(rect=(0,.05,1,.97));path=pair(fig,'history_cost')
 hist=pd.read_csv(ROOT/'reports/experiments/exp008/evidence/forecast/history_midnight_comparable.csv')
 new=hist[hist.experiment=='exp008'].copy();new['experiment']='exp009';new['label']='exp009 固定预测复用';new['forecast_reused']=True
 hist=pd.concat([hist,new],ignore_index=True);hist.to_csv(EV/'history_forecast.csv',index=False)
 fig,axes=plt.subplots(3,2,figsize=(12,10))
 for row,target in enumerate(['load','pv','net_load']):
  for ax,key,unit in zip(axes[row],['rmse_kw','wape_pct'],['RMSE（kW）','WAPE（%）']):
   selected=hist[(hist.target.isin([target,'net' if target=='net_load' else target]))&(hist.population=='all')&hist[key].notna()&(hist.model!='single_hgb')]
   # The frozen history has one declared formal forecast role per experiment.
   ax.bar(range(len(selected)),selected[key],color=[COLORS[1] if e=='exp009' else '#a3b7bf' for e in selected.experiment])
   ax.set_xticks(range(len(selected)),selected.experiment.str.replace('exp','',regex=False));ax.set_ylabel(unit);ax.set_title({'load':'负荷','pv':'光伏','net_load':'净需求'}[target])
   topv=selected[key].max()*1.2;ax.set_ylim(0,topv)
   for i,v in enumerate(selected[key]):ax.text(i,v+topv*.016,f'{v:.2f}',ha='center',fontsize=9)
 fig.suptitle('历史午夜预测同口径重评分；exp009与exp008完全相同')
 fig.tight_layout();fp=pair(fig,'history_forecast')
 return rows,path,fp

def control_charts(payload,old):
 rows=payload['comparison'];fig,axes=plt.subplots(1,2,figsize=(12,4.6),gridspec_kw={'width_ratios':[1,3]})
 for ax,rr in [(axes[0],rows[:1]),(axes[1],rows[1:])]:
  x=np.arange(len(rr))
  for j,key in enumerate(['previous','current']):ax.bar(x+(j-.5)*.34,[r[key]/1e4 for r in rr],.34,label=['exp008','exp009'][j],color=COLORS[j])
  ax.set_xticks(x,[LABEL[r['scenario']] for r in rr]);ax.set_ylabel('实际费用（万元）');ax.set_ylim(0,max(r['previous']/1e4 for r in rr)*1.2)
  for i,r in enumerate(rr):ax.text(i,max(r['previous'],r['current'])/1e4*1.02,f"{r['relative_change_pct']:+.3f}%",ha='center')
 axes[0].set_title('附件1单日');axes[1].set_title('334日，统一尺度');axes[1].legend(frameon=False,ncol=2);fig.tight_layout();cp=pair(fig,'exp008_exp009_cost')
 fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True,sharey=True);top=max(r['total_cost_yuan'] for p in [payload,old] for c in p['scenarios'].values() for r in c['monthly'])/1e4*1.12
 monthly=[]
 for ax,s in zip(axes.ravel(),SCENARIOS):
  for j,(e,p) in enumerate([('exp008',old),('exp009',payload)]):
   rr=p['scenarios'][s]['monthly'];x=np.arange(len(rr));ax.plot(x,[r['total_cost_yuan']/1e4 for r in rr],marker=['o','s'][j],color=COLORS[j],label=e)
   monthly.extend({'experiment':e,'scenario':s,**r} for r in rr)
  ax.set_xticks(x,[r['month'][-2:] for r in rr]);ax.set_title(LABEL[s]);ax.set_ylim(0,top);ax.set_xlabel('2025年月');ax.set_ylabel('月度费用（万元）')
 axes[0,0].legend(frameon=False);fig.tight_layout();mp=pair(fig,'monthly_cost')
 pd.DataFrame(monthly).to_csv(EV/'monthly_cost.csv',index=False)
 return cp,mp

def flow():
 fig,ax=plt.subplots(figsize=(12,7));ax.set_xlim(0,12);ax.set_ylim(0,7);ax.axis('off')
 def box(x,y,w,h,t):ax.add_patch(Rectangle((x,y),w,h,fc='#f4f7f8',ec='#708995'));ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=10)
 def arrow(a,b):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=12,color='#708995'))
 box(.1,5.4,3,1.1,'相同已发布预测\nQ2：HGB/ET等权+Ridge28+记忆\n其他问：单HGB+原发布更新')
 box(4,5.4,3.6,1.1,'exp008固定来源\n预测、历史残差、电价、初态\n逐项哈希 / 数值身份检查')
 box(8.5,5.4,3.3,1.1,'exp009\n同模型、同预算、同物理\n仅移除强度偏好');arrow((3.1,6),(4,6));arrow((7.6,6),(8.5,6))
 descriptions=[('第一问','最小费用→0.1%容差平稳化','只执行最小费用阶段'),('第二问','日变化≤8；吞吐罚0.002','取消变化上限；吞吐罚0'),('第三问 / 4-3','吞吐/变化罚；20kWh死区','两罚归零；死区归零'),('第四问方案2','小时模式；换向50/吞吐0.002','保持小时模式；两罚归零')]
 for i,(q,a,b) in enumerate(descriptions):
  y=4.3-i*.88;ax.text(.15,y+.28,q,fontsize=11);box(2.6,y,4,.6,a);box(7.4,y,4.3,.6,b);arrow((6.6,y+.3),(7.4,y+.3))
 ax.text(.1,.2,'每问自身SOC连续 → 当前实况物理执行 → 原购电 / 最终调整 / 紧急购电一次结算 → 全部结果同时报告',fontsize=11)
 ax.set_title('固定预测与模型方法，移除运行强度辅助偏好',loc='left');fig.tight_layout();return pair(fig,'method_control_flow')

def timing_and_gaps():
 rows=[];gaps=[]
 one=read(RESULTS/'q1/cost_only.json');previous_one=read(ROOT/'data/results/exp008/q1/revised.json')
 rows.extend([{'experiment':'exp008','scenario':'1','wall_seconds':None,'solver_seconds':sum(r['seconds'] for r in previous_one['stages']),'refinement_seconds':None,'new_model_fits':None},
  {'experiment':'exp009','scenario':'1','wall_seconds':one['wall_seconds'],'solver_seconds':one['stages'][0]['seconds'],'refinement_seconds':None,'new_model_fits':0}])
 gaps.append({'scenario':'1','calls':1,'feasible':1,'gap_mean_pct':100*one['stages'][0]['gap'],'gap_max_pct':100*one['stages'][0]['gap'],'timeout_count':0,'refinement_success':None,'refinement_maxiter':None,'note':'单日确定性MILP；不使用年度预测种子'})
 for s in SCENARIOS:
  if s in ('3','4-3'):
   d=read(RESULTS/f'update_cost_only/full334/{s}/independent_verification.json');t=d['timing']
   for e in ('exp008','exp009'):rows.append({'experiment':e,'scenario':s,'wall_seconds':t[f'{e}_run_wall_seconds'],'solver_seconds':t[f'{e}_lp_solver_seconds_sum'],'refinement_seconds':None,'new_model_fits':0 if e=='exp009' else None})
   gaps.append({'scenario':s,'calls':1336,'feasible':1336,'gap_mean_pct':None,'gap_max_pct':None,'timeout_count':0,'refinement_success':None,'refinement_maxiter':None,'max_constraint_residual':format(max(r['solver']['constraint_residual'] for r in read(RESULTS/f'update_cost_only/full334/{s}/audit.json')),'.4e'),'note':'1336 LP optimal；MIP相对gap不适用'})
  else:
   path=RESULTS/('q2_cost_only' if s=='2' else 'q4_2_cost_only/full334')
   a=read(path/('planning_audit.json' if s=='2' else 'audit.json'))
   m=[r['mip'] if 'mip' in r else r['planning'] for r in a]
   # Q4-2 metadata are nested under the method's own key; retain unknowns.
   values=[r.get('mip_gap') for r in m];finite=[v for v in values if v is not None]
   refinements=[r.get('greedy_refinement',r.get('refinement',{})) for r in a]
   gaps.append({'scenario':s,'calls':len(m),'feasible':len(m),'gap_mean_pct':100*np.mean(finite) if finite else None,'gap_max_pct':100*max(finite) if finite else None,
    'timeout_count':sum(r.get('status')==1 for r in m),'refinement_success':sum(bool(r.get('success')) for r in refinements),
    'refinement_maxiter':sum(r.get('iterations')==120 for r in refinements),'note':'固定5秒；精修未收敛返回原算法当前解，未追加迭代'})
   summary=read(path/'summary.json')
   completion=summary if s=='2' else read(RESULTS/'q4_2_cost_only/completion.json')
   rows.append({'experiment':'exp009','scenario':s,'wall_seconds':completion.get('wall_seconds'),
    'solver_seconds':sum(r.get('seconds',0) for r in m) if all('seconds' in r for r in m) else None,
    'refinement_seconds':sum(r['planning_seconds'] for r in refinements),'new_model_fits':0})
   previous=next(r for r in read(ROOT/'reports/experiments/exp008/evidence/runtime_by_stage.json')['components'] if r['id']=='Q'+s)
   rows.append({'experiment':'exp008','scenario':s,'wall_seconds':previous['run_wall_seconds'],
    'solver_seconds':previous['solver_seconds']['sum'],'refinement_seconds':previous['refinement_seconds']['sum'],'new_model_fits':None})
 rows.sort(key=lambda r:(['1',*SCENARIOS].index(r['scenario']),r['experiment']))
 pd.DataFrame(rows).to_csv(EV/'runtime.csv',index=False);save(EV/'solver_diagnostics.json',gaps)
 fig,axes=plt.subplots(1,2,figsize=(11,4.5))
 for ax,key,title in zip(axes,['solver_seconds','wall_seconds'],['求解器累计','运行墙钟']):
  rr=[r for r in rows if r[key] is not None];v=[r[key] for r in rr]
  ax.barh(range(len(rr)),v,color=[COLORS[1] if r['experiment']=='exp009' else COLORS[0] for r in rr]);ax.set_yticks(range(len(rr)),[r['experiment']+' / '+r['scenario'] for r in rr]);ax.invert_yaxis();ax.set_xlabel('实测秒');ax.set_title(title);ax.set_xlim(0,max(v)*1.24)
  for i,x in enumerate(v):ax.text(x+max(v)*.01,i,f'{x:.2f}',va='center')
 fig.suptitle('阶段嵌套且并发负载不同：只作描述，不计算加速率');fig.tight_layout();return rows,gaps,pair(fig,'runtime')

def build(code_commit):
 start=time.perf_counter();p=read(OUT/'final_payload.json');old=read(ROOT/'reports/experiments/exp008/final_payload.json')
 FIG.mkdir(parents=True,exist_ok=True);EV.mkdir(parents=True,exist_ok=True)
 assert all(c['verification']['passed'] for c in [p['q1'],*p['scenarios'].values()])
 cp,mp=control_charts(p,old);hist,hp,fp=history(p,old);flowfig=flow();times,gaps,tp=timing_and_gaps()
 annual=pd.read_csv(ROOT/'reports/experiments/exp008/evidence/forecast/annual_metrics.csv')
 other=pd.read_csv(ROOT/'reports/experiments/exp008/evidence/other_question_forecasts.csv')
 other=other[(other.granularity=='annual')&other.model.isin(['updated_issued','linked_HGB_price'])&(other.horizon.isin(['next_6h','midnight_24h']))]
 # Full unchanged sufficient statistics remain in the immutable exp008 evidence.
 methods=(OUT/'methods.md').read_text();nested=re.sub(r'(?m)^## ', '### ',methods.split('\n',1)[1]);nested=re.sub(r'(?m)^### (3\.[12])',r'#### \1',nested)
 battery=[{'question':LABEL[s],**c['battery']} for s,c in [('1',p['q1']),*p['scenarios'].items()]]
 battery[0]['power_ramp_total_kw']=read(RESULTS/'q1/cost_only.json')['metrics']['tv']
 changes=pd.read_csv(EV/'battery/exp009_vs_exp008_comparison.csv')
 shown_metrics=['mean_absolute_delta_kw','rms_delta_kw','p95_absolute_delta_kw','max_absolute_delta_kw','direct_reversals','power_limit_share','large_jump_share','direction_reversals','episodes','throughput_kwh','total_variation_kw']
 variation=changes[(changes.scenario.astype(str)!='1')&changes.metric.isin(shown_metrics)]
 fee_changes=[]
 for s,c in p['scenarios'].items():
  fee_changes.append({'scenario':s,**{k:c['fees'][k]-old['scenarios'][s]['fees'][k] for k in ['planned_cost_yuan','adjustment_cost_yuan','emergency_cost_yuan','total_cost_yuan']}})
 pd.DataFrame(fee_changes).to_csv(EV/'cost_component_changes.csv',index=False)
 bad=[{'question':LABEL[s],**c['worst_days_by_bill'][0]} for s,c in p['scenarios'].items()]
 for s,c in p['scenarios'].items():
  rr=np.array([d['total_cost_yuan'] for d in c['daily']]);threshold=float(np.quantile(rr,.9))
  c['tail']={'daily_p90_yuan':threshold,'empirical_cvar90_yuan':float(rr[rr>=threshold].mean()),'maximum_yuan':float(rr.max())}
 sections={}
 maincols=[('scenario','问题'),('planned_cost_yuan','计划费/元'),('adjustment_cost_yuan','调整费/元'),('emergency_cost_yuan','紧急费/元'),('total_cost_yuan','总费用/元'),('emergency_kwh','紧急量/kWh'),('relative_change_pct','较008/%')]
 sections[1]='\n\n'.join(['本实验是exp008固定预测与物理口径下的费用目标对照。各问均只运行事先固定的一个候选，费用与电池动作的改善、退步全部保留。第一问为附件1单日，其余问评价2025年2–12月334天；不同问题不是同时发生的费用，不能相加作为总账单。',table(p['comparison'],maincols,4),figure(cp,'exp008与exp009同问费用。第一问单日独立尺度，年度四问统一尺度。'),'目标移除运行强度偏好，实际费用仍由相同真实结算规则计算。结果不证明该控制器已经达到真实年度费用全局最小。各问对照与前史采用同一份核验后的JSON/CSV。'])
 sections[2]='\n\n'.join(['净需求为(负荷kW−光伏kW)/6，十分钟槽电量单位kWh。实际账单=Σp[g⁰+1.5(g−g⁰)₊+0.5(g⁰−g)₊+5e]；第二问及4-2没有调整项。原计划费保留，最终差额只计一次，终值和软罚不列为实际费用。',
 'MAE=Σ|误差|/n；RMSE=√(Σ误差²/n)；WAPE=100Σ|误差|/Σ|实况|。全年从充分统计量重算，不平均月度RMSE/WAPE。WAPE差为百分点；相对变化为100×(本次−基准)/|基准|。零分母或缺失不填零。',
 '容量12000kWh，SOC1200–10800kWh，双向5000kW，单向效率均√0.9；往返效率解释为团队假设。真实轨迹严格互斥、无紧急电充电，各问带自身连续SOC。Q1另有固定工程爬坡/最短持续时间；Q4-2保留小时模式网格。',
 '预测与计划仅使用当时已发布信息；实况只用于已到达槽的执行/结算。4-2、4-3真实价格不进入当期预测。非空换向是删除空闲后相邻方向反转；直接反转是原相邻槽正负变化，二者不混用。吞吐、EFC和功率变化仅作运行强度诊断，不推断真实电池寿命。'])
 sections[3]='\n\n'.join(['所有五个输入CSV按SHA-256锁定，字节与exp008相同；逐问发布预测、历史残差、电价和选中情景也保持一致。Q3/4-3每问1336次发布、每次5类哈希一致。Q2/4-2读取冻结外生路径；旧购电量、旧模式与后续SOC不作为输入。',table([{'file':k,'sha256':v} for k,v in p['data_hashes'].items()],[('file','原始数据'),('sha256','SHA-256')]),
 '原始00:10标记右端[00:00,00:10)，每天144槽；2025-01为冷启动/训练/预热依据，主统计排除一月。缺失不补零，完整性与有限值由独立审计检查。各问初始化严格相同，之后各自演化；终末SOC规则相同。',
 '既有树模型按月，以更早历史训练、最近已完成7日验证，种子42；官方PV按0/6/12/18点合法发布时间更新。所有2025日期已用于此前开发，不能把对照称为未触碰测试集。本轮不重新训练HGB/ExtraTrees、不选新参数、不使用费用筛选候选。Q3/4-3沿原代码重新计算既定在线PV/价格岭校准并缓存；输出与exp008逐发布哈希一致，在线校准计算并未被省略。'])
 sections[4]=figure(flowfig,'流程图同时显示保留内容、各问修改和实际交接。')+'\n\n'+nested
 sections[5]='\n\n'.join([table([{'question':LABEL[s],'configuration':json.dumps(v,ensure_ascii=False)} for s,v in p['protocol']['treatment'].items()],[('question','问题'),('configuration','预先冻结配置')]),
 '所有原始预测44个树模型复用，新增训练组数0。HGB为2通道×11月、最多100轮/15叶、最小叶样本50、L2=2、学习率0.08；ExtraTrees为2通道×11月、128树、最小叶样本10。单HGB分配给第三/四问，第二问为固定raw等权融合后统一Ridge28与0.5日记忆。没有新的网络层；树结构及残差递推见第四节。',
 f'本机：{platform.platform()}；Python {sys.version.split()[0]}，数值线程1。完整依赖与模型配置继承记录见[record.json](record.json)。单正式种子42（n=1），均值为该结果，样本标准差不可计算。绘图随机种子20260912与模型种子不同。',
 '场景保持整日时间相关性：Q2/4-2从历史28条中等距选3条初始化、全28条精修；Q3/4-3选7条。同场景未来动作仍是乐观近似，并非严格非预见性树。保留未来购电机会成本代理与原终值，末日为0；预算固定且无可行解则记失败停止。'])
 sections[6]='\n\n'.join(['本轮预测身份核验通过，因此所有预测误差与exp008逐项一致，差值严格为0；不是重新训练得到相同分数。午夜基础预测如下，PV发电时段用实际PV>0评价分组，不能作为未来已知特征。',
 table(annual.to_dict('records'),[('model','模型'),('target','变量'),('population','总体'),('n','样本'),('mae_kw','MAE/kW'),('rmse_kw','RMSE/kW'),('wape_pct','WAPE/%')],4),
 '\n\n'.join(f'![未变预测诊断：{name}](../exp008/figures/forecast/{name}.png)' for name in ['monthly_metrics','lead_metrics','daily_energy_metrics']),
 '日内发布及电价误差沿用冻结[完整其他问预测证据](../exp008/evidence/other_question_forecasts.csv)。Q3/4-3 next_6h按每次发布后六小时不重叠覆盖，48096槽；全剩余提前量120240槽另列，不能混合。',table(other.to_dict('records'),[('scenario','问题'),('model','发布'),('target','变量'),('population','总体'),('unit','单位'),('n','样本'),('mae','MAE'),('rmse','RMSE'),('wape_pct','WAPE/%')],5),
 '全年、分月、全部144提前槽及发电时段的充分统计量：[annual](../exp008/evidence/forecast/annual_metrics.csv)、[monthly](../exp008/evidence/forecast/monthly_metrics.csv)、[lead](../exp008/evidence/forecast/lead_metrics.csv)。基础预测与修正量、三层旧控制实验及模型结构保留在[exp008预测证据](../exp008/evidence/forecast/README.md)，本轮未把旧消融作为新候选。',
 figure(mp,'逐月实际费用同轴比较；完整分项见evidence/monthly_cost.csv。'),
 table(fee_changes,[('scenario','问题'),('planned_cost_yuan','计划费变化/元'),('adjustment_cost_yuan','调整费变化/元'),('emergency_cost_yuan','紧急费变化/元'),('total_cost_yuan','总费变化/元')],4),
 '第二问计划费减少2,695.0632元，紧急费增加16,304.6103元，合计费用上升13,609.5471元；这是实际账单的直接分项分解。现有局部规划及场景近似可能使更自由的计划在真实误差下承担更多短缺，不能仅凭此次结果认定某个机制是已证实的因果解释。',
 table(battery,[('question','问题'),('direction_reversals','非空换向/次'),('active_slots','活跃槽'),('throughput_kwh','吞吐/kWh'),('equivalent_full_cycles','EFC'),('simultaneous_slots','同时充放槽'),('power_ramp_total_kw','功率TV/kW')],3),
        table(gaps,[('scenario','问题'),('calls','求解次数'),('feasible','可行解数'),('gap_mean_pct','平均gap/%'),('gap_max_pct','最大gap/%'),('timeout_count','时限可行解'),('max_constraint_residual','LP最大约束残差'),('refinement_success','精修成功'),('refinement_maxiter','精修达120次'),('note','范围')],5),
 '### 电池功率与波动核验\n\n净功率P=6(c−d)，正充负放。正式48096时点全数保留，跨午夜差分计入；首点与预热边界单列，不混进主统计。共同随机日为04-15、06-07、12-06、12-25（seed20260912），指定日03-20、06-21、09-23、12-21，各日144原始点，均不平滑/降采样。',
 '\n\n'.join(figure(f,'实际轨迹：'+f.stem) for f in sorted((OUT/'figures/battery').glob('*.png'))),
 '第一问采用循环日TV：exp008的33,209.1661→exp009的50,943.2784kW；循环模式启动4→6次。全年问采用线性完整评价期差分，Q1不含首尾连接的49,943.2784kW仅在CSV另作线性诊断。',
 table(variation.to_dict('records'),[('scenario','问题'),('metric','指标'),('previous','008'),('current','009'),('absolute_change','差值'),('relative_change_pct','相对变化/%')],4),
 '[逐槽功率、SOC及完整219条指标对照](evidence/battery/final_evidence.json)。差分指标单位kW/相邻十分钟槽，direct_reversals为相邻直接反转次，power_limit_share及large_jump_share为比例[0,1]，大跳变阈值>1000kW/10min。幅度受限不等于功率平滑，Q1的工程爬坡以外没有新增硬爬坡。',
 '### 最贵日与尾部费用',table(bad,[('question','问题'),('date','日期'),('planned_cost_yuan','计划费'),('adjustment_cost_yuan','调整费'),('emergency_cost_yuan','紧急费'),('total_cost_yuan','总费用')],3),
 table([{'scenario':s,**c['tail']} for s,c in p['scenarios'].items()],[('scenario','问题'),('daily_p90_yuan','日费用P90/元'),('empirical_cvar90_yuan','≥P90日均费用/元'),('maximum_yuan','最贵日/元')],3),
 '最贵日由价格、需求规模、SOC与预测误差共同决定，不能直接称预测最差日。取消强度偏好可能增加换向、吞吐及弃电；有限预算MIP、局部精修和贪心执行使优化代理目标与真实年度结算有差异。未做多种子检验，不声称统计显著。',
 table(times,[('experiment','实验'),('scenario','问题'),('new_model_fits','新拟合组'),('solver_seconds','求解累计/秒'),('refinement_seconds','精修/秒'),('wall_seconds','墙钟/秒')],3),figure(tp,'求解与墙钟按阶段分列，不能相加或解释成受控加速率。'),
 '新增HGB/ExtraTrees拟合次数0；既有HGB/ET拟合+验证预测累计44.97185/63.23339秒仅作复用来源。纯拟合、单独预测、校准、回放等未独立记录的阶段保留缺失；报告构建和导出计时分别写report_build.json与工作簿QA。',
 '[指定日期的完整表格](specified_dates.md)。附件格式结果：[第一问](result1.xlsx)、[第二问](result2.xlsx)、[第三问](result3.xlsx)、[4-2](result4-2.xlsx)、[4-3](result4-3.xlsx)。四小时汇总两列为正不等于同时充放电。'])
 ht=read(ROOT/'reports/experiments/exp008/evidence/history_runtime_descriptive.json')
 sections[7]='\n\n'.join(['读取全部历史exp001–006登记及exp008；exp007按原要求忽略。原登记角色、数值及同物理重算来源保留在final_payload.history，不把新对照冒充历史正式优化策略。exp005采用原冻结登记快照；没有对应年度问的登记不补零。',
 figure(hp,'历史费用原值；005为额外硬爬坡且允许应急充电，只描述性并列。'),table(hist,[('scenario','问题'),('experiment','实验'),('role','角色'),('previous','历史费用/元'),('current','009费用/元'),('relative_change_pct','同口径变化/%'),('comparable','物理结算可比')],4),
 figure(fp,'午夜全时段RMSE与WAPE；发电时段及MAE同源CSV完整保留。'),
 'exp001/2原预测共同四目标早停存在信息公平性限制；原多发布192168样本与午夜重评分48096样本分开。预测同口径重算不静默覆盖原登记。005物理不同禁止费用排名/改善率；相同物理结算也不代表历史模型控制完全相同。',
 figure(flowfig,'本轮唯一新增技术对照是移除强度偏好；预测来源不变。'),
 '![exp008已冻结三层核心对照](../exp008/figures/report/three_layer_control.png)',
 '此前三层核心对照仍见[同基础LP预测对照与最终模式](../exp008/evidence/three_layer_control_comparison.csv)，本轮新增的是“相同预测+008强度偏好”与“相同预测+009费用目标”，不存在新的预测贡献。',
 figure(tp,'008/009存在同阶段实测值时并列，缺失不补零。'),'![历史原登记阶段耗时，原始范围单列](../exp008/figures/report/historical_runtime_descriptive.png)',table(ht['rows'],[('id','历史实验/阶段'),('stage_zh','计时范围'),('seconds','秒'),('source','来源')],3),
 '历史训练组数、完整模型与各阶段原记录保留在history及exp008 record；不同设备、并发负载和累计范围只作描述，不计算端到端加速率。'])
 sections[8]='\n\n'.join([f'实现与报告构建源码提交：`{code_commit}`。模板取origin/main的`{p["protocol"]["template_main_commit"]}`，三个模板原字节保存在[evidence/templates](evidence/templates/report.md)。源代码闭包、输入档案SHA与原证据before/after见每问run_protocol/protocol及独立审计。',
 '```bash\nOMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.exp009.run_q1\nOMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.exp009.run_q2\nOMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.exp009.run_update_cost_only\nOMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.exp009.q4_2_cost_only\n.venv/bin/python -m experiments.exp009.q4_2_cost_only_audit\n.venv/bin/python -m experiments.exp009.final_evidence --require-all\n.venv/bin/python -m experiments.exp009.report_payload\n.venv/bin/python -m reports.build_report_exp009 --code-commit '+code_commit+'\n```',
 '以上前四条为新运行，已存在冻结目录时应在新检出/独立输出目录执行，禁止覆盖已归档成果。后两条数据整理与报告构建仅读取冻结结果，不训练/优化。跨机器固定MIP秒数预算可能产生不同可行解，不能保证逐位同值。工作簿命令与独立单元格核验见[evidence/workbooks](evidence/workbooks/README.md)。',
 '[物理/费用/功率独立核验](evidence/battery/final_evidence.json)、[全部输入SHA](data_hashes.json)、[测试记录](tests.txt)、[交付QA](evidence/delivery_qa.json)。最终提交时检查全部本地链接、PNG/SVG与HTML视觉结果、工作簿全部填值和原始轨迹一致。',
 '本轮完成条件为预先固定候选的全期运行与独立核验；不要求次数下降，不继承上一轮8%停止阈值，不根据测试费用再挑选版本。'])
 template=(EV/'templates/report.md').read_text();header=template.split('\n## ')[0]
 for k,v in {'experiment_id':'exp009','title':'固定exp008预测与物理口径的费用目标对照','protocol_version':'exp009-cost-only-control/1','evaluation_period':'2025-02-01—12-31（334日）；第一问附件1单日','code_commit':code_commit}.items():header=header.replace('{{'+k+'}}',v)
 header=header.replace('data_hashes.json','[data_hashes.json](data_hashes.json)').replace('record.draft.json，登记后见 registry','[record.json](record.json)，[登记](../../registry/exp009.json)')
 titles=re.findall(r'(?m)^## (\d+)\. (.+)$',template)
 body=header+'\n\n'+'\n\n'.join(f'## {i}. {t}\n\n{sections[int(i)]}' for i,t in titles)+'\n';assert len(titles)==8 and '{{' not in body
 (OUT/'report.md').write_text(body)
 legacy.REPORT=OUT;legacy.specified_tables({'q1':p['q1'],'scenarios':p['scenarios']})
 spec=OUT/'specified_dates.md';spec.write_text(spec.read_text().replace('# exp008','# exp009'))
 for source in (OUT/'report.md',spec):
  legacy.render_html(source,source.with_suffix('.html'))
  target=source.with_suffix('.html');html=target.read_text().replace('<title>exp008 实验报告</title>','<title>exp009 费用目标对照</title>').replace('<b>exp008</b>','<b>exp009</b>')
  html=html.replace('费用按实际购电结算。2025年是开发评价年度；原8%目标未达到，按用户接受的完整结果定稿。','费用按实际购电结算。固定exp008预测的单次完整对照，2025年为开发评价年度。')
  target.write_text(html)
 oldrecord=read(ROOT/'reports/registry/exp008.json')
 record={k:oldrecord[k] for k in ['environment','model_configuration','data_hashes']}
 record.update(experiment_id='exp009',title=p['protocol']['title'],scope=['1',*SCENARIOS],code_commit=code_commit,protocol=p['protocol'],
  seeds=[42],primary_seed=42,new_forecast_fits=0,seed_sample_standard_deviation=None,metrics=p['comparison'],
  models=['Q1 original-feasible-set cost MILP','Q2 frozen blend + uncapped physical modes + cost refinement','Q3 frozen HGB/official PV + cost LP','Q4-2 frozen linked price + hourly cost modes','Q4-3 frozen causal price + cost LP'],
  forecast_metrics_unchanged_from='exp008',new_forecast_fits_scope='HGB/ExtraTrees only; inherited online Ridge calibrations still computed for Q3/4-3',forecast_evidence=p['forecast_evidence_reused'],source_sha256={str(f.relative_to(ROOT)):sha(f) for f in [Path(__file__),*sorted((ROOT/'experiments/exp009').glob('*.py')),ROOT/'experiments/exp009/protocol.json']},
  artifacts={'report':'experiments/exp009/report.md','html':'experiments/exp009/report.html','payload':'experiments/exp009/final_payload.json','workbooks':[f'experiments/exp009/result{s}.xlsx' for s in ['1',*SCENARIOS]]},
  formal_run_executed=True,selection_performed=False,accepted_result_is_untouched_test=False)
 record['model_configuration']['selection_context']='Inherited exp008 tree selection; exp009 has one predeclared treatment and no further selection.'
 record['model_configuration']['reuse_note']='All model fits inherited unchanged from exp008; no new fit or hyperparameter selection in exp009.'
 save(OUT/'record.json',record);save(ROOT/'reports/registry/exp009.json',record)
 save(OUT/'report_build.json',{'code_commit':code_commit,'template_main_commit':p['protocol']['template_main_commit'],'seconds':time.perf_counter()-start,'payload_sha256':sha(OUT/'final_payload.json'),'report_sha256':sha(OUT/'report.md'),'html_sha256':sha(OUT/'report.html'),'sections':8,'new_optimization_executed':False})
 print(json.dumps({'report':str(OUT/'report.html'),'build_seconds':time.perf_counter()-start,'comparison':p['comparison']},ensure_ascii=False))

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--code-commit',required=True);args=parser.parse_args()
 assert re.fullmatch('[0-9a-f]{40}',args.code_commit);build(args.code_commit)
