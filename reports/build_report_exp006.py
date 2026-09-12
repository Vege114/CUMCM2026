"""Build the eight-section exp006 report from independently verified archives."""
import argparse
import json
import re
import shutil
import subprocess
import time

import pandas as pd

from reports.exp006_evidence import ROOT,OUT,REPORT,EVIDENCE,LABELS,ROLES,SPECIFIED,TARGETS,build_evidence,read,write_json,sha256

NODE=ROOT.home()/'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node'
PLUGIN=ROOT.home()/'.codex/plugins/cache/openai-curated-remote/data-analytics/1.0.2'
TITLES=['结论与成绩','题目指标及信息边界','数据与时间验证','逐步技术讲解','实验设置','结果及失败案例','历次指标和技术路线对比','复现说明']
SHORT_LABELS={LABELS[k]:v for k,v in {'primary':'6主组树DP','causal_42':'6季节树DP','greedy_execution':'6贪心重规划','fixed_primary_greedy':'6固定购电','seed_2026':'6树DP2026','seed_3407':'6树DP3407','causal_2026':'6季节2026','causal_3407':'6季节3407','primary_grid25':'6主组25kWh','causal_42_grid25':'6季节25kWh','cost_only':'6仅费用','no_tree':'6无条件树','more_throughput_penalty':'6强吞吐罚','zero_terminal':'6无终值','no_deadband':'6无死区'}.items()}

def records(f):return json.loads(f.to_json(orient='records',double_precision=15)) if isinstance(f,pd.DataFrame) else f
def md_table(rows,columns):
    def fmt(v):
        if v is None or isinstance(v,float) and pd.isna(v):return '—'
        if isinstance(v,bool):return '是' if v else '否'
        if isinstance(v,(int,float)):return f'{v:,.4f}'
        return str(v).replace('|','\\|')
    return '\n'.join(['| '+' | '.join(label for _,label in columns)+' |','|'+'---|'*len(columns),*['| '+' | '.join(fmt(r.get(k)) for k,_ in columns)+' |' for r in rows]])

def build(complete=False,node=NODE,skip_figures=False):
    start=time.monotonic();f=build_evidence();app=REPORT/'app';snapshot=read(app/'src/data.json')
    costs=f['cost_history'];runs=f['all_runs'];primary=runs[runs.id=='primary'].iloc[0]
    base=costs[costs.policy_id=='exp004/no_season'].iloc[0];oldformal=costs[costs.policy_id=='exp004/causal_season'].iloc[0]
    delta=primary.total_cost-base.total_cost;pct=100*delta/base.total_cost
    greedy_cost=runs[runs.id=='greedy_execution'].iloc[0]
    bat=f['battery_history'];bp=bat[bat.policy_id=='exp006/primary'].iloc[0];bb=bat[bat.policy_id=='exp004/no_season'].iloc[0]
    title='exp006：树与动态规划降低电池周转，但总费用上升' if delta>0 and bp.throughput_kwh<bb.throughput_kwh else 'exp006：树与动态规划的实测费用和电池权衡'
    snapshot.update(title=title,surface='report',buildStatus='complete' if complete else 'creating',report={'asOf':'2025-12-31'},queries={},specifiedDates=SPECIFIED,detailPolicies=['primary','causal_42','cost_only','greedy_execution','fixed_primary_greedy'],policyLabels=LABELS)
    queries=snapshot['queries'];sections=[{'title':s,'blocks':[]} for s in TITLES];markdown=[[] for s in TITLES]
    definitions={
      'cost_history':'策略×主种子42，2025-02-01至12-31的334日。金额元，电量kWh；总费Σp(g+5e)，original=final，调整费0，期末不抵残值。exp001原登记不同物理，oracle含未来，两者无改善率或排名。',
      'forecast_history':'午夜48096槽；功率kW。MAE=Σ|误差|/n，RMSE=√(Σ误差²/n)，WAPE=100Σ|误差|/Σ|实际|，bias=mean(pred−actual)。PV发电时段单列。exp006直接复用exp004数组，不产生预测改进。',
      'battery_history':'334日完整串联。交流吞吐Σ(c+d)kWh；电芯EFC=(√0.9Σc+Σd/√0.9)/24000；非空方向切换及功率总变差不计一月预热边界，包含评价期跨日。静态效率/功率约束是工程近似，EFC不等于寿命。无档案历史非加和项保留缺失。',
      'relative_comparison':'相对变化=100(current−previous)/abs(previous)，负值费用改善。WAPE绝对差单位百分点。exp001原登记和oracle无正式改善率。',
      'timings':'秒。单组334日累计和整批墙钟区分；树拟合属于条件准备，条件准备+动态规划+实际执行=compute_seconds。不同机器、原训练组数及并行竞争仅描述性比较。缺失留空，复用预测的新训练/推理次数确为0。',
      'all_runs':'全部预先声明的15项结果，主组角色不随测试年费用改变。12项50kWh+2项25kWh+固定购电旧执行回放，原始数值元/kWh/秒。',
      'daily':'组×日期，全部334日逐槽结算加和。每组自身实际SOC连续跨日，日末不重置。',
      'monthly_cost':'组×月，逐日金额元、电量kWh求和；11个评价月。',
      'paired_daily':'同日exp006正式减exp004无季节，单位元；正值为新方案更贵。',
      'detail':'组×指定日/最贵日×10分钟。负载/PV功率kW；净需求/购电/电池动作kWh；SOC为槽位起点kWh。预测从冻结exp004读取，动作/实际从已核验exp006 NPZ读取。',
      'specified':'正式主组四个题目指定日，全天费用元、电量kWh及日初末SOC。完整题目格式见specified_dates.md与result2.xlsx。',
      'seed_costs':'两预测家族×3个事先固定种子，独立334日。42是正式角色，不按费用挑选或集成。',
      'seed_summary':'三种子算术均值和样本标准差ddof=1，不是置信区间；同家族同协议。',
      'verification':'独立读取已保存数组逐项验证，布尔passed与数值误差来自各组verification.json；嵌套详情保持JSON文本。'}
    def query(qid,frame,source=None,definition=None,files=None):
        source=source or qid;rows=records(frame)
        for r in rows:
            if 'label' in r:r['chart_label']=SHORT_LABELS.get(r['label'],str(r['label']).replace('exp00','').replace(' 同口径重算','重算').replace(' 午夜重评分','午夜').replace(' 正式风险','风险').replace(' 同确定性','确定').replace(' 未校准','未校准').replace(' 正式·','正式·'))
            if 'previous_label' in r:r['previous_short']=str(r['previous_label']).replace('exp00','').replace(' 同口径重算','重算').replace(' 正式风险','风险').replace(' 同确定性','确定').replace(' 原登记（不可比）','原登记')
        definition=definition or definitions[source]
        queries[qid]={'rows':rows,'source':{'label':qid,'files':files or [f'evidence/{source}.csv'],'metricDefinitions':[{'label':'范围与指标','definition':definition}],
          'evidenceFlow':[{'title':'冻结档案读取','detail':'reports/exp006_evidence.py只读取结果，独立核验通过后重汇总，无训练、规划、执行调用。来源路径保留在source字段。'}]},'methods':[{'language':'text','code':definition}]}
    for name,frame in f.items():query(name,frame)
    def prose(i,text,qs=()):
        id=f's{i+1}-text-{len(sections[i]["blocks"])}';sections[i]['blocks'].append({'type':'prose','id':id,'markdown':text,'queryIds':list(qs)});markdown[i].append(text)
    def table(i,id,title,qid,columns):
        sections[i]['blocks'].append({'type':'table','id':id,'title':title,'queryId':qid,'columns':columns});markdown[i].append('### '+title+'\n\n'+md_table(queries[qid]['rows'],columns))
    def chart(i,id,title,qid,y,x='label',kind='horizontalBar',series=None,fields=None,unit='',height=380,scope=None,controls=False):
        spec={'type':kind,'x':'chart_label' if x=='label' else x,'y':y,'stackable':False,'valueDecimals':2,'showValues':kind=='horizontalBar','colorBySign':False,'colors':{y:'#357c89'}}
        spec['xLabel' if kind=='horizontalBar' else 'yLabel']=unit
        if series:spec['series']=series
        if fields:spec['fields']=fields
        spec['legend']={'labels':{'actual_load_kw':'实际负载','forecast_load_kw':'预测负载','actual_pv_kw':'实际光伏','forecast_pv_kw':'预测光伏','actual_net_kwh':'实际净需求','forecast_net_kwh':'预测净需求','planned_kwh':'计划购电','emergency_kwh':'紧急购电','charge_kwh':'充电','discharge_kwh':'放电','soc_kwh':'SOC','planned_wan':'计划费','emergency_wan':'紧急费'}}
        if kind=='horizontalBar':
            values=[r.get(k) for r in queries[qid]['rows'] for k in (fields or [y]) if isinstance(r.get(k),(float,int))]
            low=min([0,*values]);high=max([0,*values]);span=high-low or 1
            spec['presentation']='plot'
            spec['barOptions']={'orientation':'horizontal','categoryWidth':180,'grid':True,'domain':[low-.06*span if low<0 else 0,high+.20*span],
              'labels':{'value':not bool(fields)},'style':{'color':'#357c89','radius':0,'thickness':22},'format':{'maximumFractionDigits':2}}
            if fields:spec['barOptions']['series']=[{'key':field,'label':spec['legend']['labels'].get(field,field),'color':['#357c89','#aa6d37','#8b7c9e'][j%3]} for j,field in enumerate(fields)]
        sections[i]['blocks'].append({'type':'slotBattery' if id=='specified-battery' else 'chart','id':id,'title':title,'queryId':qid,'spec':spec,'height':height,'scope':scope,'controls':controls})
    for i,s in enumerate(TITLES):prose(i,f'## {i+1}. {s}')
    manifest=read(OUT/'run_manifest.json');protocol=read(EVIDENCE/'protocol.json')
    verdict='费用上升，不能取代同预测基线作为最低费用方案。' if delta>0 else '本年真实费用下降；仍须区分单年实绩与连续优化最优性。'
    prose(0,f'### 主要结论\n\n**{verdict}**\n\n- 事先冻结的正式树DP组总费 **{primary.total_cost/10000:.2f}万元**，相对同预测的exp004无季节 **{delta/10000:+.2f}万元（{pct:+.2f}%）**。相对exp004当轮正式历史季节组为 **{(primary.total_cost-oldformal.total_cost)/10000:+.2f}万元**。\n- 电池交流侧吞吐 **{bp.throughput_kwh/10000:.2f}万kWh**，相对同预测基线 **{100*(bp.throughput_kwh/bb.throughput_kwh-1):+.2f}%**；非空方向反转 **{int(bp.direction_reversals):,}次**，基线 **{int(bb.direction_reversals):,}次**。两者同时充放电槽数均为0。\n- 正式单组条件准备、规划与执行合计 **{primary.compute_seconds:.2f}秒**；整批15项实验墙钟 **{manifest.get("wall_seconds",0):.2f}秒**。预先定义的主组、种子及全部消融保留原角色，不按全年最低费用替换主组。', ['cost_history','battery_history','timings','all_runs'])
    prose(0,f'**费用优先时，本轮更值得保留的是已测的“旧执行·重新规划”对照。** 它的真实总费为 **{greedy_cost.total_cost/10000:.2f}万元**，比同预测基线少 **{base.total_cost-greedy_cost.total_cost:,.2f}元（{100*(base.total_cost-greedy_cost.total_cost)/base.total_cost:.3f}%）**；反转 **{int(greedy_cost.direction_reversals):,}次**，比基线少 **{100*(1-greedy_cost.direction_reversals/bb.direction_reversals):.2f}%**。这是单种子已测对照，保留其消融角色，不把它事后改名为正式主组，也不据此宣称跨种子或跨年胜出。', ['all_runs','cost_history','battery_history'])
    prose(0,f'主组相对同预测基线的费用分解为：计划费 **{primary.planned_cost-base.planned_cost:+,.2f}元**，紧急费 **{primary.emergency_cost-base.emergency_cost:+,.2f}元**，合计 **{delta:+,.2f}元**。费用上升来自计划费增加超过紧急费节省。',['cost_history'])
    focal=costs[costs.policy_id.isin(['exp004/no_season','exp004/causal_season','exp006/primary','exp006/causal_42','exp006/greedy_execution','exp006/fixed_primary_greedy'])]
    query('focal',focal,'cost_history')
    table(0,'primary-results','主结果与同预测基线 · 334日','focal',[('label','方案'),('role','角色'),('planned_cost','计划费/元'),('up_cost','上调费/元'),('down_cost','下调费/元'),('emergency_cost','紧急费/元'),('total_cost','总费/元'),('emergency_kwh','紧急/kWh')])
    chart(0,'focal-cost','同预测基线与本轮真实总费用','focal','total_wan',unit='万元',height=430)
    prose(1,'本轮修改规划及电池执行，输入复用exp004冻结预测。每天零点一次发布144×2负载/PV功率预测（kW），10分钟电量=功率/6（kWh）。日前购电全天锁定，original=final。实际总费 **C=Σp(g+5e)**，其中g为全部原计划购电、e为实际缺口补购；无日内调整费、售电收入或年末残值抵扣。\n\n预测只使用发布前信息；条件误差最多来自最近28个已完整揭晓日。零点只取已发布预测及已知电价，执行第t槽只读本槽实际值，不看未来实际。树的9列是边际支持，不是联合日场景。',['cost_history','all_runs'])
    prose(1,'电池总容量12000kWh，SOC范围1200–10800kWh、功率上限5000kW、单向效率√0.9。一月1日6000kWh预热后，2月1日共同初始SOC为1421.7991105135516kWh，来自已冻结一月预热；之后各组延续自己的实际末状态。\n\n主目标仍是实际购电总费。0.002元/交流侧kWh吞吐与0.05元/反转仅为规划次要偏好，2kWh死区避免极小动作；它们未计入题目费用，也不是估算出的电池折旧成本。EFC与切换只能描述操作强度，缺少化学体系、温度及寿命数据，不能据此计算寿命延长。',['battery_history'])
    data_hashes=manifest['evidence']['data_sha256'];query('hashes',[{'file':k,'sha256':v} for k,v in data_hashes.items()],'cost_history','读取原题白名单输入的SHA256，由正式运行清单绑定。',files=['evidence/run_manifest.json'])
    table(2,'input-hashes','输入文件与校验值','hashes',[('file','文件'),('sha256','SHA256')])
    prose(2,'附件2按区间终点解释：00:10为当天第一段，0:00+1为当天24:00。正式范围2025-02-01至12-31，共334日、每组48096槽。负载和PV数组保持原始kW单位，无插值重采样；所有正式组共享同一预测档案版本与预热状态。exp004既有训练、早停与信息边界由其冻结核验记录继承，树更新只用此前完整实际日。\n\n首两日不足两个完整冻结发布日时，条件误差来源明确回退到因果周期基线，属于样本来源回退，不是求解失败。状态与费用核验覆盖15组所有721440槽；未来扰动测试和独立核验结果在复现证据中保留。',['verification'])
    methods=(REPORT/'methods.md').read_text().split('\n',1)[1];methods=re.sub(r'```mermaid.*?```','',methods,flags=re.DOTALL).replace('## ','### ')
    prose(3,methods,['all_runs','battery_history'])
    routes=[{'label':'条件误差树','steps':['冻结144×2预测','最近28个完整历史日','五项特征/深度5/叶48','每槽9个等权分位支持']},{'label':'动态规划','steps':['SOC网格+真实初SOC','保存上次非空方向','购电取80%分位闭式解','逆向递推144槽','得到固定购电和意图动作']},{'label':'实际执行','steps':['仅本槽观测','按余电/缺口选方向','容量/功率/死区裁剪','紧急补购或弃电','真实SOC跨日'] }]
    query('method-routes',routes,'all_runs','冻结实现的三层计算数据流，来自risk.py、model.py和正式protocol。',files=['evidence/protocol.json'])
    sections[3]['blocks'].append({'type':'routes','id':'technical-routes','title':'从预测到真实结算的三层数据流','queryId':'method-routes'})
    query('config',[{'name':k,'value':str(v)} for k,v in protocol['primary_parameters'].items()],'all_runs','正式运行前冻结的主组参数；不是从全年费用选择。',files=['evidence/protocol.json'])
    table(4,'primary-configuration','冻结主组参数','config',[('name','参数'),('value','取值')])
    table(4,'run-matrix','全部正式、兼容、种子、消融与精度检查','all_runs',[('id','ID'),('role','角色'),('forecast','预测'),('seed','种子'),('grid_kwh','SOC网格/kWh'),('conditioning','条件分布'),('executor','执行器'),('throughput_yuan_per_kwh','吞吐惩罚'),('reversal_yuan','反转惩罚'),('deadband_kwh','死区/kWh')])
    prose(4,f'总计15项：12项50kWh、2项25kWh全年精度检查、1项固定正式购电的旧执行回放。正式组及seed42在全年评价前冻结，2026和3407只检查稳定性，未取平均决策或择优。当前CPU单线程计算，Python {manifest.get("python","").split()[0]}、NumPy {manifest.get("numpy")}、scikit-learn {manifest.get("sklearn")}。任务与原LP并行，因此实测计时也受共享机器竞争影响。\n\n无需训练强化学习或调用LP。离散DP能给出网格代理的最优动作；原连续随机控制最优间隙没有证书，保留为未知。全部参数与运行角色见[冻结协议](evidence/protocol.json)。',['all_runs','timings'])
    table(5,'all-run-costs','全部15项原始成绩 · 未按成绩重选主组','all_runs',[('label','方案'),('role','角色'),('planned_cost','计划费/元'),('emergency_cost','紧急费/元'),('total_cost','总费/元'),('emergency_kwh','紧急/kWh'),('final_soc','期末SOC/kWh'),('compute_seconds','计算/s')])
    table(5,'surrogate-versus-real','代理目标、实际费用及尾部风险','all_runs',[('label','方案'),('sum_daily_surrogate_objective','每日代理目标之和/元'),('total_cost','真实总费/元'),('daily_cvar90','最贵10%日均费/元'),('worst_day_cost','最贵日/元'),('continuous_optimality_gap','连续问题最优间隙')])
    prose(5,'每日代理目标之和包含辅助惩罚及普通日终值，各日又以各自真实SOC重新开始规划；因此它既不是全年同一路径的规划费用，也不能与真实费用直接相减解释为遗憾值。固定购电旧执行组未求解新规划，代理目标记为缺失。连续问题最优间隙均未知，不填0。日费用CVaR90取最贵10%日概率质量的分数权重均值，334日对应33.4日。',['all_runs'])
    focus_ids=['primary','cost_only','no_tree','more_throughput_penalty','zero_terminal','greedy_execution','no_deadband','fixed_primary_greedy']
    query('ablations',runs[runs.id.isin(focus_ids)],'all_runs');chart(5,'ablation-costs','消融与控制对照的总费用','ablations','total_wan',unit='万元',height=530)
    query('grid',runs[runs.id.isin(['primary','causal_42','primary_grid25','causal_42_grid25'])],'all_runs');chart(5,'grid-cost','50kWh与25kWh网格的真实费用','grid','total_wan',unit='万元',height=350)
    grid=runs[runs.id=='primary_grid25'].iloc[0];greedy=runs[runs.id=='fixed_primary_greedy'].iloc[0];replan=runs[runs.id=='greedy_execution'].iloc[0]
    prose(5,f'25kWh主预测组相对50kWh正式组总费差 **{grid.total_cost-primary.total_cost:+,.2f}元**，计算时间 **{grid.compute_seconds:.2f}秒**。更细网格只优化更丰富的代理动作，实际裁剪后的费用不保证单调；结果不构成连续最优间隙。\n\n固定正式购电、只换旧执行器的总费 **{greedy.total_cost/10000:.2f}万元**，比主组 **{greedy.total_cost-primary.total_cost:+,.2f}元**。旧执行并逐日重新规划组为 **{replan.total_cost/10000:.2f}万元**，其后续购电会随自身SOC改变，故不能把两组差异均归为纯执行效应。',[ 'all_runs'])
    query('seed-chart',f['seed_costs'],'seed_costs');chart(5,'seed-costs','两预测家族的三种子真实费用','seed-chart','total_wan',x='seed_label',kind='bar',series='family',unit='万元',height=400)
    table(5,'seed-summary-table','三种子均值与样本标准差 · 非置信区间','seed_summary',[('family','预测家族'),('metric','指标'),('n','种子数'),('mean','均值'),('sample_std','样本标准差')])
    monthly=f['monthly_cost'];query('monthly-focus',monthly[monthly.name.isin(['primary','causal_42','greedy_execution','fixed_primary_greedy'])],'monthly_cost')
    for metric,title_m,unit in [('total_cost','逐月总购电费','元'),('emergency_kwh','逐月紧急购电量','kWh')]:chart(5,'monthly-'+metric,title_m,'monthly-focus',metric,x='date',kind='line',series='label',unit=unit)
    bfocus=bat[bat.policy_id.isin(['exp004/no_season','exp004/causal_season',*['exp006/'+x for x in focus_ids]])];query('battery-focus',bfocus,'battery_history')
    for metric,title_m,unit in [('throughput_kwh','电池交流侧吞吐','kWh'),('direction_reversals','电池非空方向反转','次')]:chart(5,'battery-'+metric,title_m+' · 正式评价区间','battery-focus',metric,unit=unit,height=590)
    table(5,'battery-metrics','电池操作强度与互斥核验','battery-focus',[('label','方案'),('charge_kwh','充电/kWh'),('discharge_kwh','放电/kWh'),('equivalent_full_cycles','电芯EFC'),('direction_reversals','反转/次'),('simultaneous_slots','同时槽'),('power_variation_kw','功率总变差/kW')])
    costonly=runs[runs.id=='cost_only'].iloc[0]
    prose(5,f'正式组方向反转减少 **{100*(1-bp.direction_reversals/bb.direction_reversals):.2f}%**，电芯EFC只减少 **{100*(1-bp.equivalent_full_cycles/bb.equivalent_full_cycles):.2f}%**，两者含义不同。去掉弱惩罚和死区的仅费用组反转 **{int(costonly.direction_reversals):,}次**，正式组 **{int(bp.direction_reversals):,}次**；因此不能把相对旧基线的大幅反转下降归功于弱惩罚本身。规划与意图执行的组合改变才是主要区分。\n\n期末SOC：同预测旧基线 **{bb.final_soc:.4f}kWh**，正式组 **{bp.final_soc:.4f}kWh**，旧执行重规划 **{greedy_cost.final_soc:.4f}kWh**。三者均按实际购电费结算，没有残值抵扣。',['all_runs','battery_history'])
    daily=f['daily'];pday=daily[daily.id=='primary'];worst=pday.loc[pday.total_cost.idxmax()];paired=f['paired_daily'];bad=paired.loc[paired.difference_yuan.idxmax()]
    prose(5,f'正式组最贵日为 **{worst.date}**，总费 **{worst.total_cost:,.2f}元**、紧急购电 **{worst.emergency_kwh:,.2f}kWh**。相对同预测基线最不利日为 **{bad.date}**，多花 **{bad.difference_yuan:,.2f}元**。\n\n代理把电池意图动作与购电一同优化，实际执行又把动作按本槽余缺裁剪；该不一致是需要检查的模型局限。固定购电旧执行对照直接显示执行规则的影响。以下曲线并列净需求、固定购电、紧急补购与实际SOC，避免只展示代理费用或有利日期。',['daily','paired_daily','all_runs'])
    chart(5,'paired-daily-difference','每天新正式减同预测旧基线总费差','paired_daily','difference_yuan',x='date',kind='line',unit='元；正值更贵')
    query('worst-detail',f['detail'][(f['detail'].name=='primary')&(f['detail'].date==worst.date)],'detail')
    chart(5,'worst-energy',f'正式组最贵日供需 · {worst.date}','worst-detail','actual_net_kwh',x='hour',kind='line',fields=['actual_net_kwh','forecast_net_kwh','planned_kwh','emergency_kwh'],unit='每10分钟kWh')
    chart(5,'worst-soc',f'正式组最贵日SOC · {worst.date}','worst-detail','soc_kwh',x='hour',kind='line',unit='kWh')
    table(5,'specified-results','题目指定四日的正式组全天结果','specified',[('date','日期'),('planned_kwh','计划/kWh'),('emergency_kwh','紧急/kWh'),('total_cost','总费/元'),('initial_soc','日初SOC/kWh'),('final_soc','日末SOC/kWh')])
    for id,title_m,fields,unit in [('load','负载预测与实际',['actual_load_kw','forecast_load_kw'],'kW'),('pv','光伏预测与实际',['actual_pv_kw','forecast_pv_kw'],'kW'),('energy','供需与购电',['actual_net_kwh','forecast_net_kwh','planned_kwh','emergency_kwh'],'每10分钟kWh'),('battery','逐槽充放电',['charge_kwh','discharge_kwh'],'每10分钟kWh'),('soc','储能SOC',['soc_kwh'],'kWh')]:chart(5,'specified-'+id,'指定日'+title_m,'detail',fields[0],x='hour',kind='bar' if id=='battery' else 'line',fields=fields,unit=unit,scope='specified',controls=id=='load')
    prose(5,'[原题模板全年工作簿](result2.xlsx)包含334天三张表；[指定四日完整表格](specified_dates.md)列出表1六个槽位、表2六个4小时段充放电及日初末SOC、表3全部紧急购电区间。[详细方法](methods.md)保留公式、接口和全部近似边界。',['specified'])
    prose(5,'费用较低的已测对照也提供独立[旧执行·重新规划工作簿](greedy_execution/result2.xlsx)及[对应指定日表](greedy_execution/specified_dates.md)，与冻结正式主组分开标识。固定主组购电只换旧执行的控制结果说明：执行裁剪及其状态轨迹确实改变费用；再次逐日规划后的额外变化还涉及后续购电，不能全部归因于树。',['all_runs'])
    historic_ids=['exp001/legacy_rebased','exp002/primary','exp002/new_deterministic','exp003/primary','exp003/uncalibrated','exp004/no_season','exp004/causal_season','exp006/primary','exp006/causal_42','exp006/greedy_execution','exp006/fixed_primary_greedy']
    query('comparable',costs[costs.policy_id.isin(historic_ids)],'cost_history')
    prose(6,'以下按实验顺序并列，费用单位、评价334日、效率/SOC、固定日前购电与5倍紧急购电口径相同。exp004无季节是同预测主要基线，历史季节保留其当轮正式角色。exp002正式风险改变规划，exp006也改变执行，不能把这些费用差归因于预测。exp001/2的共享四目标早停边界仍存在。\n\nexp001原登记物理协议不同，单独保留原值；exp004全年探索使用未来实际信息，单独列示且排除正式排名和改善率。并行运行的exp005尚不属于本轮冻结比较。',['cost_history'])
    for metric,title_m,unit in [('total_wan','总费用','万元'),('planned_kwh','计划购电量','kWh'),('emergency_kwh','紧急购电量','kWh')]:chart(6,'history-'+metric,'历次'+title_m+' · 同物理结算口径','comparable',metric,unit=unit,height=650)
    chart(6,'history-cost-components','历次计划费与紧急费','comparable','planned_wan',fields=['planned_wan','emergency_wan'],unit='万元',height=660)
    history_table=costs[(costs.experiment!='exp006')|costs.name.isin(['primary','causal_42','greedy_execution','fixed_primary_greedy'])];query('history-table',history_table,'cost_history')
    table(6,'history-absolute-values','历史原登记、重算、探索与本轮正式原值','history-table',[('label','策略'),('role','原角色'),('status','可比性'),('planned_cost','计划费/元'),('emergency_cost','紧急费/元'),('total_cost','总费/元'),('final_soc','期末SOC/kWh')])
    compare=f['relative_comparison'];query('relative-cost',compare[compare.metric=='total_cost'],'relative_comparison')
    table(6,'history-relative-values','正式组对各历史费用的原值与变化','relative-cost',[('previous_label','历史策略'),('previous','历史/元'),('current','本次/元'),('absolute_change','差值/元'),('relative_change_pct','变化/%'),('comparable','可比')])
    query('relative-cost-valid',compare[(compare.metric=='total_cost')&compare.comparable],'relative_comparison');chart(6,'history-relative-cost','正式主组对历史总费的相对变化（负值改善）','relative-cost-valid','relative_change_pct',x='previous_short',unit='%',height=520)
    forecast=f['forecast_history'];query('forecast-current',forecast[(forecast.experiment=='exp006')&(forecast.population=='all')],'forecast_history')
    table(6,'forecast-current-errors','复用预测的完整误差 · 主种子42','forecast-current',[('label','本次组'),('target','目标'),('mae','MAE/kW'),('rmse','RMSE/kW'),('wape_pct','WAPE/%'),('bias','偏差/kW')])
    prose(6,'exp006没有重训或修改预测；正式组与exp004无季节的每一项预测误差完全相同，历史季节兼容组同理。规划费用变化不能宣称预测精度提高。误差图仍展示历次RMSE与WAPE，保留exp004组合相对exp003的误差退步。',[ 'forecast_history'])
    for target in TARGETS:
        q='history-error-'+target;query(q,forecast[(forecast.target==target)&(forecast.population=='all')&~forecast.exploratory],'forecast_history')
        for metric,unit in [('rmse','kW'),('wape_pct','%')]:chart(6,q+'-'+metric,'历次'+TARGETS[target]+('RMSE' if metric=='rmse' else 'WAPE')+' · 同48096槽',q,metric,unit=unit,height=550)
    query('pv-generating',forecast[(forecast.target=='pv')&(forecast.population=='pv_generating')],'forecast_history');table(6,'pv-generating-errors','实际光伏发电时段误差 · 探索保留原值','pv-generating',[('label','策略'),('n','槽位'),('mae','MAE/kW'),('rmse','RMSE/kW'),('wape_pct','WAPE/%'),('bias','偏差/kW')])
    # Show archived monthly and lead metrics unchanged, explicitly labelled as reused forecasts.
    for name in ['forecast_monthly','forecast_lead']:
        file=ROOT/'data/results/exp004'/f'{name}.csv';p=pd.read_csv(file);p=p[(p.seed==42)&p.name.isin(['no_season','causal_season'])].copy();p['label']=p.name.map({'no_season':LABELS['primary'],'causal_season':LABELS['causal_42']});p.to_csv(EVIDENCE/f'{name}.csv',index=False)
        query(name,p,'forecast_history',files=[f'evidence/{name}.csv'])
    monthly_error=queries['forecast_monthly']['rows'];lead_error=queries['forecast_lead']['rows']
    for target in ['load','pv']:
        m=pd.DataFrame(monthly_error);m=m[(m.target==target)&(m.population=='all')];m['date']=m.month.map(lambda v:f'2025-{int(v):02d}-01');q='monthly-error-'+target;query(q,m,'forecast_history','exp004冻结预测逐月目标误差，RMSE/MAE单位kW，WAPE单位%；由误差分子分母重算，exp006不重训。',files=['evidence/forecast_monthly.csv']);chart(6,q,'复用预测逐月'+TARGETS[target]+'RMSE',q,'rmse',x='date',kind='line',series='label',unit='kW')
        p=pd.DataFrame(lead_error);p=p[p.target==target];p['lead']=p.apply(lambda r:f'{int(r.lead_start_hour)}–{int(r.lead_end_hour)}h',axis=1);q='lead-error-'+target;query(q,p,'forecast_history','334日午夜预测按六个4小时时段分组；每组8016槽，RMSE/MAE单位kW，WAPE单位%；冻结exp004预测复用。',files=['evidence/forecast_lead.csv']);chart(6,q,'复用预测提前量分段'+TARGETS[target]+'RMSE',q,'rmse',x='lead',kind='bar',series='label',unit='kW')
    for stage,id in [('训练','training'),('检查点预测','prediction'),('调度执行','dispatch')]:
        p=f['timings'];p=p[(p.stage==stage)&((p.experiment!='exp006')|p.label.isin([LABELS['primary'],LABELS['greedy_execution'],'exp006 复用预测']))];query('time-'+id,p,'timings');chart(6,'time-'+id,'历次'+stage+'累计实测耗时','time-'+id,'seconds',unit='秒；描述性比较',height=530)
    table(6,'all-timings','阶段时间与计时范围','timings',[('label','实验/组'),('stage','阶段'),('seconds','秒'),('groups','组/日数'),('scope','计时范围')])
    routes=[{'label':'exp001','steps':['MLP/CNN/LSTM候选','四目标共享早停','旧登记与v2物理重算']},{'label':'exp002','steps':['固定残差MLP','四目标共享早停','风险正式与确定性对照']},{'label':'exp003','steps':['Q2独立MLP','一月费用校准α=0','周期预测+确定性调度/贪心执行']},{'label':'exp004','steps':['168小时双分支CNN','非对称损失','无季节/历史季节/全年探索','沿用exp003规划及执行']},{'label':'exp006','steps':['冻结exp004预测','条件树边际误差','SOC动态规划','固定购电+意图裁剪执行','费用与电池强度核验']}]
    query('history-routes',routes,'cost_history','路线读取各实验协议及代码；角色按原登记保留。',files=['evidence/registry_coverage.json','evidence/protocol.json']);sections[6]['blocks'].append({'type':'routes','id':'history-routes','title':'历次技术路线并列','queryId':'history-routes'})
    prose(7,'在本独立工作树根目录进行日常复核。此工作树没有自己的`.venv`，可用原项目绝对Python路径`/Users/vegetarianwolf/Projects/CUMCM2026/.venv/bin/python`或同依赖环境替换下列`/path/to/python`：\n\n```sh\n/path/to/python -m unittest discover -s tests -p \'test_q2_tree*.py\' -v\n/path/to/python -m experiments.problem2.tree_planning.verify data/results/exp006/primary\n/path/to/python -m experiments.problem2.tree_planning.verify data/results/exp006/greedy_execution\n/path/to/python -m reports.build_report_exp006 --complete --node /path/to/codex/node\n```\n\n工作簿如需重新导出，另执行`/path/to/python -m experiments.problem2.tree_planning.export --node /path/to/codex/node`；较低费用对照再增加`--case greedy_execution`。以上复核无需重跑求解，保留原始计时。\n\n完整重新计算应在新的隔离副本中进行：保留exp001–004输入档案，将该副本已有的`data/results/exp006`另存备份后，在副本根目录运行`/path/to/python -m experiments.problem2.tree_planning.run`。运行器没有`--output`参数；不要在已交付目录中用重跑覆盖原timings。\n\n运行器有签名绑定的按组/分段缓存；签名由代码、冻结预测、输入、预热及参数共同决定。报告构建仅读取核验档案，不调用规划或训练。完整预测仍来自exp004 `predictions.npz`，工作簿从对应组`dispatch_2.npz`生成。')
    prose(7,f'已保存15组逐日与逐槽档案、条件树及代理目标审计、独立物理/费用检查、原题工作簿读回与预览。正式预测档案SHA256：`{manifest["evidence"]["predictions_sha256"]}`。\n\n报告页、Markdown、PNG/SVG、对话图均由同一evidence CSV生成。浏览器视觉检查另由页面验收记录支持；构建成功本身不代表视觉验收。报告构建与工作簿导出时间分开记入[构建记录](report_build.json)和工作簿审计。',['verification'])
    prose(7,'测速提交为`a112933`，仅标识获准前的小样本测速状态。当前正式实现由`run_manifest.json`中的逐文件source_sha256绑定，不能用测速提交替代正式源码版本。最终提交号以同包登记记录为准。原LP任务、旧报告及历史输出均由独立工作树隔离；本报告不触发任何求解或修改旧成绩。')
    snapshot['reportContent']=sections
    for qid,q in queries.items():
        ids=[]
        for s in sections:
            for b in s['blocks']:
                if b.get('queryId')==qid:ids.append(b['id'])
                if qid in b.get('queryIds',[]):ids.append(b['id']+'-sources')
        q['source']['metricDefinitions'][0]['componentIds']=ids
    write_json(app/'src/data.json',snapshot)
    if not skip_figures:
        from reports.exp006_figures import build_figures
        build_figures(f)
    for i,names in [(0,['history-costs']),(3,['technical-route']),(5,['ablation-costs','battery-comparison','monthly-costs','seed-costs','failure-case','specified-days']),(6,['history-errors','stage-timings'])]:markdown[i].extend(f'![{name}](figures/{name}.png)' for name in names)
    (REPORT/'report.md').write_text('# '+title+'\n\n'+'\n\n'.join('\n\n'.join(s) for s in markdown)+'\n')
    script=PLUGIN/'scripts/data-app.mjs';subprocess.run([str(node),str(script),'build','--project-dir',str(app),'--separate-data'],check=True)
    offline=app/'.data-app-offline/exports/report.html';subprocess.run([str(node),str(script),'export-offline','--project-dir',str(app),'--output',str(offline)],check=True);shutil.copy2(offline,REPORT/'report.html')
    for name in ['methods.md','specified_dates.md','result2.xlsx','README.md']:
        if (REPORT/name).exists():shutil.copy2(REPORT/name,app/'dist'/name)
    shutil.copytree(EVIDENCE,app/'dist/evidence',dirs_exist_ok=True)
    shutil.copytree(REPORT/'figures',app/'dist/figures',dirs_exist_ok=True)
    shutil.copy2(REPORT/'report.md',app/'dist/report.md')
    if (REPORT/'greedy_execution').exists():shutil.copytree(REPORT/'greedy_execution',app/'dist/greedy_execution',dirs_exist_ok=True)
    write_json(REPORT/'report_build.json',{'buildStatus':snapshot['buildStatus'],'seconds':time.monotonic()-start,'html_sha256':sha256(REPORT/'report.html'),'data_sha256':sha256(app/'src/data.json'),'queries':len(queries),'components':sum(len(s['blocks']) for s in sections),'no_solver_invocation':True,'evidence':{p.name:sha256(p) for p in EVIDENCE.iterdir() if p.is_file()}})
    shutil.copy2(REPORT/'report_build.json',app/'dist/report_build.json')
    print('REPORT_BUILT',REPORT/'report.html')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--complete',action='store_true');p.add_argument('--node',default=NODE);p.add_argument('--skip-figures',action='store_true');build(**vars(p.parse_args()))
