"""Eight-part exp004 report from independently verified, frozen result files."""
import argparse
import json
import re
import shutil
import subprocess
import time

import pandas as pd

from experiments.problem2.exp004.data import HERE, OUT, ROOT, protocol, sha256, write_json
from reports.exp004_evidence import (
    EVIDENCE,
    LABELS,
    REPORT,
    SPECIFIED,
    TARGETS,
    build_evidence,
    read,
)

NODE = ROOT.home() / '.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node'
PLUGIN = ROOT.home() / '.codex/plugins/cache/openai-curated-remote/data-analytics/1.0.2'
TITLES = ['结论与成绩','题目指标及信息边界','数据与时间验证','逐步技术讲解','实验设置','结果及失败案例','历次指标和技术路线对比','复现说明']


def records(frame):
    return json.loads(frame.to_json(orient='records', double_precision=15)) if isinstance(frame,pd.DataFrame) else frame


def md_table(rows, columns):
    def fmt(value):
        if value is None or isinstance(value,float) and pd.isna(value): return '—'
        if isinstance(value,(float,int)) and not isinstance(value,bool): return f'{value:,.4f}'
        return str(value).replace('|','\\|')
    return '\n'.join(['| '+' | '.join(v for _,v in columns)+' |','|'+'---|'*len(columns),
                      *['| '+' | '.join(fmt(row.get(k)) for k,_ in columns)+' |' for row in rows]])


def build(complete=False, node=NODE, skip_figures=False):
    began=time.monotonic()
    f=build_evidence()
    app=REPORT/'app'
    snapshot=read(app/'src/data.json')
    snapshot.update(title='exp004：预测组合降低费用，历史季节预判未增益', surface='report',
                    buildStatus='complete' if complete else 'creating',report={'asOf':'2025-12-31'},specifiedDates=SPECIFIED,queries={})
    queries=snapshot['queries']; sections=[{'title':t,'blocks':[]} for t in TITLES]; markdown=[[] for _ in TITLES]
    definitions={
      'cost_history':'每行一项主种子策略，2025年2–12月334日；费用元、电量kWh，C=Σp(g+5e)。exp001原登记物理不同；全年探索含未来信息，均不参与正式排名。exp002正式风险还改变了调度，不能单独归因预测。',
      'forecast_history':'午夜同48096槽位。MAE=Σ|e|/n，RMSE=√(Σe²/n)，WAPE=100Σ|e|/Σ|y|，bias=mean(pred−actual)。全时段和实际PV>0单独计分，原始功率单位kW；exp001/2共同四目标早停边界保留。',
      'relative_comparison':'保留previous/current/absolute_change/relative_change_pct。相对变化=100(current−previous)/abs(previous)；WAPE绝对差单位为百分点。原协议不兼容或全年探索不计算正式改善率。',
      'timings':'实测阶段秒数，训练/检查点预测为任务累计、调度为主种子334日累计。设备、训练组数、样本范围不同，仅描述性比较。缺失/不适用保留空值。',
      'monthly_cost':'组×种子×月份，逐日费用元和电量kWh求和。',
      'monthly_forecast':'组×种子×月份×目标×评价时段；误差从分子/分母重算，不平均月度百分比。',
      'seasonal_shifts':'每次午夜年周期拟合的目标日减基线参考日修正量，144槽位平均，单位kW。历史组只用过去完整日，全年组含未来。图为预测修正，非实测季节曲线。',
      'specified':'题目四日、主种子42，表1指定购电槽位及全天指标，金额元、电量kWh。表2/3完整储能/紧急购电明细见同包specified_dates.md和三个模板工作簿。',
      'detail':'组×指定日或最贵日×十分钟槽位；功率kW，能量/SOC为kWh。SOC为槽位起点。来自核验的预测与原exp003回放。',
      'paired_daily':'主种子42，同日历史季节总费减无季节总费，正值为季节组更贵。每组SOC自身连续。',
      'seed_costs':'三组各三个固定种子，独立334日绝对绩效，42事先指定。',
      'seed_summary':'三种子成绩均值与样本标准差ddof=1，不是置信区间。'}
    def query(name, frame, source=None, definition=None):
        source=source or name; definition=definition or definitions[source]
        rows=records(frame)
        for row in rows:
            if 'label' in row:
                row['chart_label']=str(row['label']).replace('exp00','').replace(' ','').replace('同口径重算','重算').replace('午夜重评分','午夜').replace('正式风险','风险').replace('同确定性','确定').replace('未校准','未校').replace('无季节','无季').replace('历史季节','历史').replace('全年探索','全年*')
        queries[name]={'rows':rows,'source':{'label':name,'files':[f'evidence/{source}.csv'],
          'metricDefinitions':[{'label':'指标与边界','definition':definition}],
          'evidenceFlow':[{'title':'可复现来源','detail':'reports/exp004_evidence.py读取已核验档案，不训练或求解。原始文件来源保留于source列及登记覆盖清单。'}]},
          'methods':[{'language':'text','code':definition}]}
    for name,frame in f.items(): query(name,frame)
    for row in queries['seed_summary']['rows']:
        row['name']=LABELS[row['name']]
        row['metric']={'total_cost':'总费用/元','planned_cost':'计划费/元','emergency_cost':'紧急费/元','emergency_kwh':'紧急电量/kWh'}[row['metric']]
    query('protocol',[{'name':k,'value':str(v)} for k,v in protocol().items()],'cost_history','实际冻结配置：结构、训练、季节模块、输入边界及不变的规划。')
    queries['protocol']['source']['files']=['evidence/protocol.json']

    def prose(i,text,qs=()):
        sections[i]['blocks'].append({'type':'prose','id':f's{i+1}-text-{len(sections[i]["blocks"])}','markdown':text,'queryIds':list(qs)})
        markdown[i].append(text)
    def table(i,id,title,qid,columns):
        sections[i]['blocks'].append({'type':'table','id':id,'title':title,'queryId':qid,'columns':columns})
        markdown[i].append('### '+title+'\n\n'+md_table(queries[qid]['rows'],columns))
    def chart(i,id,title,qid,y,x='label',kind='horizontalBar',series=None,fields=None,unit='',height=380,scope=None,controls=False):
        spec={'type':kind,'x':'chart_label' if x=='label' else x,'y':y,'stackable':False,'valueDecimals':2,'yLabel':unit,'showValues':kind=='horizontalBar'}
        if kind=='horizontalBar': spec['xLabel']=unit;spec['yLabel']=''
        if series: spec['series']=series
        if fields: spec['fields']=fields
        labels={'actual_load_kw':'实际负载','forecast_load_kw':'预测负载','actual_pv_kw':'实际PV','forecast_pv_kw':'预测PV','actual_net_kwh':'实际净需求','forecast_net_kwh':'预测净需求','planned_kwh':'计划购电','emergency_kwh':'紧急购电','planned_wan':'计划费','emergency_wan':'紧急费','soc_kwh':'SOC'}
        spec['legend']={'labels':labels}
        sections[i]['blocks'].append({'type':'chart','id':id,'title':title,'queryId':qid,'spec':spec,'height':height,'scope':scope,'controls':controls})
    for i,title in enumerate(TITLES): prose(i,f'## {i+1}. {title}')
    costs=f['cost_history']; current=costs[costs.experiment=='exp004']; query('current',current,'cost_history')
    no=current[current.name=='no_season'].iloc[0];ca=current[current.name=='causal_season'].iloc[0];oracle=current[current.name=='oracle_season'].iloc[0];old=costs[costs.policy_id=='exp003/primary'].iloc[0]
    delta=ca.total_cost-no.total_cost;saved=old.total_cost-no.total_cost
    prose(0,f'**在相同调度器下，本轮预测组合降低了实测费用，额外历史季节模块没有带来收益。**\n\n- 无季节组总购电费 **{no.total_cost/10000:.2f}万元**，较exp003少 **{saved/10000:.2f}万元（{saved/old.total_cost*100:.2f}%）**。\n- 历史季节组 **{ca.total_cost/10000:.2f}万元**，反而多花 **{delta:,.2f}元（{delta/no.total_cost*100:.3f}%）**，其他两个种子也更贵。\n- 全年季节探索组 **{oracle.total_cost/10000:.2f}万元**，使用未来实际信息，不能作为可实施成绩或严格最优上界。\n\n历史季节组和种子42是事先指定的正式方案，没有按全年结果改名或选种子。现有结果支持规划组先接入无季节预测。', ['cost_history','seed_costs'])
    table(0,'primary-absolute','三组绝对绩效 · 主种子42','current',[('label','方案'),('planned_kwh','计划购电/kWh'),('emergency_kwh','紧急购电/kWh'),('planned_cost','计划费/元'),('emergency_cost','紧急费/元'),('total_cost','总费/元')])
    chart(0,'current-cost','三组总购电费 · 全年探索含未来信息','current','total_wan',unit='万元')
    prose(1,'本次只修改预测。题目直接绩效是总购电费及其计划费、紧急费构成，另外报告计划/紧急电量、充放电量及SOC。每天零点发布144×2的负载/PV数组(kW)，原计划全天锁定。10分钟电量为功率除以6(kWh)。总费C=Σp(g+5e)，p为固定电价，g为原计划电量，e为紧急购电量；计划全部计费，没有调整费或售电收入。\n\n沿用单向效率√0.9、1200–10800 kWh SOC与5000 kW功率上限。一月从6000 kWh预热，2月以同一初始SOC开始，之后各组分别连续跨日。年末仅有物理边界，不额外计残值。',['cost_history'])
    prose(1,'预测辅助指标：MAE=Σ|预测−实际|/n；RMSE=√(Σ(预测−实际)²/n)，单位kW。WAPE=100Σ|预测−实际|/Σ|实际|，单位%；零分母缺失。bias=mean(预测−实际)，正值表示多预测。WAPE绝对差为百分点，相对变化仍为%。全天和实际PV>0的发电槽位分别评分。日费用CVaR90为最贵10%日费用的分数权重平均。',['forecast_history'])
    ver=read(OUT/'verification.json')
    hashes=[{'file':k,'sha256':v} for k,v in ver['data_hashes'].items()]
    query('hashes',hashes,'cost_history','Q2白名单原始输入SHA-256，Data读取及独立核验。')
    queries['hashes']['source']['files']=['evidence/verification.json']
    table(2,'data-hashes','原始输入校验值','hashes',[('file','文件'),('sha256','SHA-256')])
    prose(2,'附件2终点00:10对应第一段，0:00+1对应24:00。原始365天完整、非负且日期无重复；正式评价334天、48,096个午夜目标。每月训练标签在最后7日早停段之前结束，验证标签在当月发布前结束，标准化仅用训练前缀。\n\n未来数据扰动测试证明正式两组特征/季节修正不随尚未发生的观测改变；全年探索阳性对照会改变。训练样本的季节特征在各历史发布日截断。这里“因果”仅指信息时序合规，不是因果推断。单年资料不能独立验证跨年度季节性。',['forecast_history'])
    methods=(REPORT/'methods.md').read_text().split('\n',2)[2]
    methods=re.sub(r'```mermaid.*?```','',methods,flags=re.DOTALL)
    methods=re.sub(r'(?m)^\|.*\n?', '',methods)
    methods=methods.replace('## ','### ')
    prose(3,methods,['protocol','seasonal_shifts'])
    network=[{'label':'负载/PV各自分支','steps':['168小时序列','Conv1D8核5膨胀1','Conv1D8核5膨胀2','6小时池化','Dense8摘要','拼接8项上下文','Dense16与Dense1残差','144槽位kW预测']}]
    query('network',network,'cost_history','本次4930参数的两分支短序列CNN，结构来自冻结protocol和已运行train.py。')
    queries['network']['source']['files']=['evidence/protocol.json']
    sections[3]['blocks'].append({'type':'routes','id':'network','title':'网络结构与数据流','queryId':'network'})
    markdown[3].append('```mermaid\nflowchart LR\n A[168小时负载/PV独立序列] --> B[各自两层因果Conv1D]\n B --> C[6小时池化与Dense8]\n C --> D[重复144槽位并拼接上下文]\n D --> E[Dense16与Dense1残差]\n E --> F[基线加季节修正加残差]\n F --> G[非负与历史日照掩码]\n G --> H[144×2功率预测kW]\n```')

    for target in ['load','pv']:
        chart(3,'seasonal-'+target,'每次发布的'+TARGETS[target]+'年内修正量','seasonal_shifts',target+'_shift_kw',x='date',kind='line',series='label',unit='kW')
    routes=[{'label':'exp001','steps':['共享四目标选模/早停','MLP/CNN/LSTM候选','原登记及v2重算']},
      {'label':'exp002','steps':['固定残差MLP、四目标早停','正式风险策略','另有确定性对照']},
      {'label':'exp003','steps':['Q2独立MLP','一月费用选择α=0','周期预测、确定性调度']},
      {'label':'exp004','steps':['168小时双分支CNN','90日半衰期和5:1损失','无/历史/全年季节','原exp003规划和储能执行']}]
    query('routes',routes,'cost_history','路线来自各实验冻结的代码、协议和历史登记。')
    sections[6]['blocks'].append({'type':'routes','id':'routes','title':'历次技术路线','queryId':'routes'})
    metadata=read(OUT/'training_metadata.json');train=pd.DataFrame(metadata)
    summary=train.groupby('variant',as_index=False).agg(groups=('seed','size'),epochs_mean=('epochs','mean'),train_seconds=('training_seconds','sum'),prediction_seconds=('prediction_seconds','sum'))
    summary['variant']=summary.variant.map(LABELS)
    query('training',summary,'timings')
    table(4,'training-summary','三组训练实测配置','training',[('variant','组'),('groups','月度训练组数'),('epochs_mean','平均训练轮数'),('train_seconds','训练/s'),('prediction_seconds','检查点预测/s')])
    prose(4,'三组各11月×3种子，共99组；两分支4930参数，种子42事先指定，其他种子不参与选择或集成。Adam=0.001，batch=32，上限60轮、早停耐心6；以最近7历史日同一非对称损失恢复最佳权重，全年不回选。CPU两线程，TensorFlow 2.18.1、NumPy 2.0.2，开启确定性算子。各组保存/重载输出均核验。\n\n用户限定预测范围，Q2T006–010的场景规划、增加求解预算、滚动储能、跨日预看留给规划组。只保留历史联合误差读取接口，不决定场景权重或购电计划。',['training','timings'])
    monthly=f['monthly_cost'];query('current_monthly',monthly[monthly.seed==42],'monthly_cost')
    for metric,title,unit in [('total_cost','逐月总购电费','元'),('emergency_kwh','逐月紧急购电量','kWh')]:
        chart(5,'monthly-'+metric,title+' · 主种子42','current_monthly',metric,x='date',kind='line',series='label',unit=unit)
    for target in TARGETS:
        part=f['monthly_forecast'];qid='monthly-'+target
        query(qid,part[(part.seed==42)&(part.target==target)&(part.population=='all')],'monthly_forecast')
        chart(5,'monthly-error-'+target,'逐月'+TARGETS[target]+'RMSE',qid,'rmse',x='date',kind='line',series='label',unit='kW')
    lead=pd.read_csv(OUT/'forecast_lead.csv')
    lead['label']=lead.name.map(LABELS)
    lead['lead']=lead.apply(lambda r:f'{int(r.lead_start_hour)}–{int(r.lead_end_hour)}h',axis=1)
    lead.to_csv(EVIDENCE/'forecast_lead.csv',index=False,float_format='%.15g')
    for target in ['load','pv']:
        qid='lead-'+target
        query(qid,lead[(lead.seed==42)&(lead.target==target)],'forecast_lead','午夜发布后的六个4小时预测区间，334天各8016槽位；RMSE、MAE单位kW，WAPE单位%。PV夜间分母接近零，不以WAPE评价夜间质量。')
        chart(5,qid+'-rmse','预测步长分段'+TARGETS[target]+'RMSE',qid,'rmse',x='lead',kind='bar',series='label',unit='kW')
    prose(5,'预测步长按0–4、4–8、8–12、12–16、16–20、20–24小时分段。各组全部种子的MAE、RMSE、WAPE与偏差保留于[步长误差CSV](evidence/forecast_lead.csv)。夜间PV实际功率总和接近零，会放大WAPE，图中使用绝对RMSE。',['lead-load','lead-pv'])
    seeds=f['seed_costs'].copy();seeds['label']=seeds.name.map(LABELS);seeds['seed_label']=seeds.seed.map(lambda s:f'种子{s}');seeds['total_wan']=seeds.total_cost/10000
    query('seeds',seeds,'seed_costs')
    chart(5,'seed-cost','三种子费用 · 全年探索含未来信息','seeds','total_wan',x='seed_label',kind='bar',series='label',unit='万元')
    table(5,'all-seeds','全部组和种子的绝对绩效','seeds',[('label','组'),('seed','种子'),('planned_cost','计划费/元'),('emergency_cost','紧急费/元'),('total_cost','总费/元'),('emergency_kwh','紧急电量/kWh')])
    table(5,'seed-summary','三种子均值与样本标准差','seed_summary',[('name','组'),('metric','指标'),('mean','均值'),('sample_std','样本标准差')])
    paired=f['paired_daily'];bad=paired.loc[paired.difference_yuan.idxmax()];worst=seeds[(seeds.name=='causal_season')&(seeds.seed==42)].iloc[0]
    low,high=read(EVIDENCE/'seasonal_sensitivity.json')['seven_day_block_95pct_interval_yuan']
    prose(5,f'历史季节组的净负载RMSE略降，但费用仍增加{delta:,.2f}元，误差与储能后的费用并不等价。配对最不利日期{bad.date}多花{bad.difference_yuan:,.2f}元。历史季节组最贵日{worst.worst_date}为{worst.worst_day_cost:,.2f}元。最坏日曲线另存evidence/detail.csv。\n\n7日区块2000次重采样的年度费用差95%区间为[{low:,.0f}, {high:,.0f}]元，跨过0。它只描述单年内的敏感性，不能证明季节模块必然有害或已经跨年验证。9组均无超时/回退，432,864槽位物理和费用核验通过。',['paired_daily','seed_costs'])
    chart(5,'paired-daily','历史季节减无季节的每日总费差','paired_daily','difference_yuan',x='date',kind='line',unit='元；正值为季节组更贵')
    query('worst-detail',f['detail'][(f['detail'].name=='causal_season')&(f['detail'].date==worst.worst_date)],'detail')
    chart(5,'worst-energy',f'最贵日购电与净需求 · {worst.worst_date} · 历史季节','worst-detail','actual_net_kwh',x='hour',kind='line',fields=['actual_net_kwh','forecast_net_kwh','planned_kwh','emergency_kwh'],unit='每十分钟kWh')
    table(5,'specified-absolute','题目指定四日的全天指标','specified',[('label','组'),('date','日期'),('planned_kwh','计划/kWh'),('emergency_kwh','紧急/kWh'),('total_cost','总费/元'),('initial_soc','日初SOC/kWh'),('final_soc','日末SOC/kWh')])
    for id,title,fields,unit in [('load','负载预测与实际',['actual_load_kw','forecast_load_kw'],'kW'),('pv','PV预测与实际',['actual_pv_kw','forecast_pv_kw'],'kW'),('energy','购电与净需求',['actual_net_kwh','forecast_net_kwh','planned_kwh','emergency_kwh'],'每十分钟kWh'),('soc','储能轨迹（槽位起点）',['soc_kwh'],'kWh')]:
        chart(5,'specified-'+id,'指定日'+title,'detail',fields[0],x='hour',kind='line',fields=fields,unit=unit,scope='specified',controls=id=='load')
    prose(5,'完整表1六个购电时段、表2六个4小时充放电块与日初/末SOC、表3全部紧急购电明细见[指定四日完整表格](specified_dates.md)。[无季节工作簿](no_season/result2.xlsx)、[历史季节工作簿](causal_season/result2.xlsx)、[全年探索工作簿](oracle_season/result2.xlsx)均为原题模板334天结果。',['specified'])
    history=costs[costs.physically_comparable&~costs.exploratory];query('comparable',history,'cost_history')
    prose(6,'按实验顺序并列绝对成绩，不按表现排序。exp001同口径重算、exp002确定性对照、exp003和本轮使用同一物理/结算；exp002正式风险策略还改变决策器。exp001/2历史共享四目标早停，不能宣称严格Q2隔离。原exp001登记的效率/SOC/控制不同，保留原值但不算改善率。全年季节探索含未来信息，不进入正式比较图。',['cost_history','forecast_history'])
    for id,title,metric,unit in [('total','总购电费','total_wan','万元'),('planned-energy','计划购电量','planned_kwh','kWh'),('emergency-energy','紧急购电量','emergency_kwh','kWh')]:
        chart(6,'history-'+id,'历次'+title+' · 334天','comparable',metric,unit=unit,height=420)
    chart(6,'history-components','历次计划费、紧急费绝对值','comparable','planned_wan',fields=['planned_wan','emergency_wan'],unit='万元',height=480)
    table(6,'history-values','全部原登记、重算与本轮原始值','cost_history',[('label','方案'),('status','口径'),('planned_kwh','计划/kWh'),('emergency_kwh','紧急/kWh'),('planned_cost','计划费/元'),('emergency_cost','紧急费/元'),('total_cost','总费/元'),('final_soc','年末SOC/kWh')])
    prose(6,'**费用降低伴随着常规误差退步。** 无季节组负载/PV的RMSE为258.31/424.00 kW，高于exp003正式的244.29/375.17 kW；净负载RMSE为524.84 kW，高于448.21 kW。无季节组平均多预测负载101.16 kW、少预测PV135.60 kW，与5:1损失的保守方向一致。原调度器因此倾向多作日前购电，紧急费下降超过计划费增加。这个费用代理并不等同于直接优化储能后的实际总费，也不能声称预测精度整体改善。',['forecast_history','cost_history'])
    annual=f['forecast_history'];query('current-errors',annual[(annual.experiment=='exp004')&(annual.population=='all')],'forecast_history')
    table(6,'current-errors','本轮全天完整误差 · 主种子42','current-errors',[('label','组'),('target','目标'),('mae','MAE/kW'),('rmse','RMSE/kW'),('wape_pct','WAPE/%'),('bias','偏差/kW')])
    for target in TARGETS:
        p=f['forecast_history'];qid='history-'+target
        query(qid,p[(p.target==target)&(p.population=='all')&~p.exploratory],'forecast_history')
        for metric,unit in [('rmse','kW'),('wape_pct','%')]:
            chart(6,qid+'-'+metric,'历次'+TARGETS[target]+('WAPE' if metric=='wape_pct' else 'RMSE')+' · 全天',qid,metric,unit=unit)
    pv=f['forecast_history'];query('pv-generating',pv[(pv.target=='pv')&(pv.population=='pv_generating')],'forecast_history')
    table(6,'pv-generating','实际发电时段PV误差 · 含探索原值','pv-generating',[('label','模型'),('n','槽位数'),('mae','MAE/kW'),('rmse','RMSE/kW'),('wape_pct','WAPE/%'),('bias','偏差/kW')])
    r=f['relative_comparison'];part=r[(r.task=='dispatch')&(r.metric=='total_cost')&r.comparable].copy();part['pair']=part.previous_label+' → '+part.current_label
    part['previous_short']=part.previous_label.str.replace('exp00','').str.replace(' ','').str.replace('同口径重算','重算').str.replace('正式风险','风险').str.replace('同确定性','确定').str.replace('未校准','未校')
    query('relative-cost',part,'relative_comparison')
    chart(6,'relative-cost','总费用相对变化（负值改善）','relative-cost','relative_change_pct',x='previous_short',series='current_label',unit='%',height=520)
    table(6,'relative-values','总费的原值、绝对差和相对变化','relative-cost',[('previous_label','此前'),('current_label','本次'),('previous','此前/元'),('current','本次/元'),('absolute_change','绝对差/元'),('relative_change_pct','相对变化/%')])
    for stage,id in [('训练','training-time'),('检查点预测','prediction-time'),('调度执行','dispatch-time')]:
        query(id,f['timings'][f['timings'].stage==stage],'timings')
        chart(6,id,'历次'+stage+'累计实测耗时',id,'seconds',unit='秒；描述性比较',height=420)
    table(6,'timing-details','阶段耗时范围与缺失项','timings',[('label','实验/组'),('stage','阶段'),('seconds','秒'),('groups','组/日数'),('scope','范围')])
    ev=read(OUT/'evaluation_manifest.json');export=read(OUT/'workbook_export.json')
    prose(6,f'本次99组训练累计{sum(m["training_seconds"] for m in metadata):.2f}秒，检查点推理{sum(m["prediction_seconds"] for m in metadata):.3f}秒。9组回放并行墙钟{ev["seconds"]:.2f}秒，三份工作簿导出/读回{export["seconds"]:.2f}秒。特征准备按同月同组共享一次，记录在training_metadata.json；报告构建单独记时。\n\nexp001为165组GPU训练，exp002/003各33组，本次CPU每组33次；设备、样本与计时范围不同，不能把训练耗时下降解释为端到端加速。此前缺失阶段保留空值，本次费用校准标为不适用。\n\n总体节省的费用来自计划费增加与紧急费减少的净差。短序列、降权、非对称损失共同变化，不能把组合收益单独分配给某个模块。',['timings','cost_history'])
    prose(7,'仓库根目录执行：\n\n```sh\n.venv/bin/python -m unittest discover -s tests -p \'test_q2_*.py\' -v\n.venv/bin/python -m experiments.problem2.exp004.train\n.venv/bin/python -m experiments.problem2.exp004.evaluate --workers 3\n.venv/bin/python -m experiments.problem2.exp004.verify\n.venv/bin/python -m experiments.problem2.exp004.export --node /absolute/path/to/codex/node\n.venv/bin/python -m reports.build_report_exp004 --complete --node /absolute/path/to/codex/node\n```\n\n预测接口`ForecastStore("no_season", seed=42).get(origin)`返回144×2 float64 kW。origin为自2025-01-01起的十分钟索引，必须是2–12月午夜。`completed_error_paths(origin)`只返回此前完整揭晓的误差路径；全年探索需`allow_oracle=True`。\n\n权重与可恢复逐日缓存放在忽略的runs/；已提交predictions.npz与训练元数据能直接供规划组和报告使用，无须权重。缓存签名绑定源代码、参数、数据与初始SOC。见[预测交接说明](README-prediction.md)。')
    prose(7,f'独立核验通过：9组完整全年、99组训练边界/重载、41项Q2测试、三份工作簿334天读回。既有规划和历次结果共{ver["unchanged_prior_files"]}个文件保持不变。代码/数据清单与评价签名保存于evidence；页面检查记录与构建记录分开。',['seed_costs'])
    for name in ['verification.json','prediction_manifest.json','evaluation_manifest.json','evaluation_sources.json','workbook_export.json']:
        shutil.copy2(OUT/name,EVIDENCE/name)
    shutil.copy2(HERE/'protocol.json',EVIDENCE/'protocol.json')
    shutil.copy2(HERE/'decision-log.md',REPORT/'decision-log.md')
    handoff=(HERE/'README.md').read_text().replace('../../../reports/experiments/exp004/','').replace('(protocol.json)','(evidence/protocol.json)')
    (REPORT/'README-prediction.md').write_text(handoff)
    snapshot['reportContent']=sections
    # Exact consumer IDs scope each query's definitions to its actual components.
    for qid,q in queries.items():
        ids=[]
        for s in sections:
            for b in s['blocks']:
                if b.get('queryId')==qid: ids.append(b['id'])
                if qid in b.get('queryIds',[]): ids.append(b['id']+'-sources')
        q['source']['metricDefinitions'][0]['componentIds']=ids
    write_json(app/'src/data.json',snapshot)
    if not skip_figures:
        from reports.exp004_figures import build_figures
        build_figures(f)
    for i,names in [(0,['history-costs']),(3,['seasonal-shifts']),(5,['monthly-costs','failure-case','specified-forecasts']),(6,['absolute-performance','history-errors','stage-timings','seed-costs'])]:
        markdown[i].extend(f'![{name}](figures/{name}.png)' for name in names)
    (REPORT/'report.md').write_text('# exp004：预测组合降低费用，历史季节预判未增益\n\n'+'\n\n'.join('\n\n'.join(p) for p in markdown)+'\n')
    script=PLUGIN/'scripts/data-app.mjs'
    subprocess.run([str(node),str(script),'build','--project-dir',str(app),'--separate-data'],check=True)
    offline=app/'.data-app-offline/exports/report.html'
    subprocess.run([str(node),str(script),'export-offline','--project-dir',str(app),'--output',str(offline)],check=True)
    shutil.copy2(offline,REPORT/'report.html')
    for name in ['specified_dates.md','methods.md','README-prediction.md','decision-log.md']:
        if (REPORT/name).exists(): shutil.copy2(REPORT/name,app/'dist'/name)
    shutil.copytree(EVIDENCE,app/'dist/evidence',dirs_exist_ok=True)
    for variant in LABELS:
        target=app/'dist'/variant;target.mkdir(exist_ok=True)
        shutil.copy2(REPORT/variant/'result2.xlsx',target/'result2.xlsx')
    write_json(REPORT/'report_build.json',{'buildStatus':snapshot['buildStatus'],'seconds':time.monotonic()-began,
      'html_sha256':sha256(REPORT/'report.html'),'data_sha256':sha256(app/'src/data.json'),'queries':len(queries),
      'components':sum(len(s['blocks']) for s in sections),'evidence':{p.name:sha256(p) for p in EVIDENCE.iterdir() if p.is_file()}})
    print('REPORT_BUILT',REPORT/'report.html')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--complete',action='store_true');p.add_argument('--node',default=NODE);p.add_argument('--skip-figures',action='store_true')
    build(**vars(p.parse_args()))
