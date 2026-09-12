"""Render exp007 only from reviewed evidence; never train, plan, replay or register."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'reports/experiments/exp007'
EVIDENCE = REPORT / 'evidence'
APP = REPORT / 'app'
NODE = Path.home()/'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node'
PLUGIN = Path.home()/'.codex/plugins/cache/openai-curated-remote/data-analytics/1.0.2'
RUNS = ['regularized_42','regularized_2026','regularized_3407','cost_only_42','cost_only_2026','cost_only_3407']
LABELS = {r:('RL 轻惩罚' if r.startswith('regularized') else 'RL 纯费用')+' · '+r.rsplit('_',1)[1] for r in RUNS}
LABELS['regularized_42'] = 'RL 正式 · 42'
TITLES = ['结论与成绩','题目指标及信息边界','数据与时间验证','逐步技术讲解','实验设置','结果及失败案例','历次指标和技术路线对比','复现说明']
SPECIFIED = ['2025-03-20','2025-06-21','2025-09-23','2025-12-21']
COLORS = ['#357c89','#bc6d3e','#8075a9','#5d8c69','#b25268','#9b843f']
NAMES = {'planned_cost':'计划费','emergency_cost':'紧急费','total_cost':'总费用','planned_wan':'计划费','emergency_wan':'紧急费','total_wan':'总费用','actual_net_kwh':'实际净需求','forecast_net_kwh':'预测净需求','planned_kwh':'锁定购电','emergency_kwh':'紧急购电','soc_kwh':'实际SOC','charge_kwh':'充电','discharge_kwh':'放电','entropy':'策略熵','approx_kl':'近似KL','clip_fraction':'裁剪比例','value_loss':'价值损失','loss':'总优化损失'}


def read(path): return json.loads(Path(path).read_text())
def sha256(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write_json(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def safe_float(value):
    try:
        n=float(value)
        return n if math.isfinite(n) else None
    except (TypeError,ValueError): return None

def n(row,*keys,default=None):
    for key in keys:
        val=safe_float(row.get(key))
        if val is not None:return val
    return default

def fmt(value,decimals=2):
    if value is None:return '—'
    if isinstance(value,bool):return '是' if value else '否'
    if isinstance(value,(float,int)):return f'{value:,.{decimals}f}'
    return str(value).replace('|','\\|')
def pct(value,base):return None if base in (None,0) or value is None else 100*(value-base)/abs(base)
def signed(value,decimals=2):return '—' if value is None else f'{value:+,.{decimals}f}'
def md_table(rows,columns):
    return '\n'.join(['| '+' | '.join(label for _,label in columns)+' |','|'+'---|'*len(columns),*['| '+' | '.join(fmt(r.get(key),4) for key,_ in columns)+' |' for r in rows]])
def csv_rows(path):
    if not Path(path).exists():return []
    with Path(path).open() as f:return list(csv.DictReader(f))
def key_of(row):
    value=row.get('run_id') or row.get('id') or row.get('name')
    if not value and str(row.get('policy_id','')).startswith('exp007/'):
        value=row['policy_id'].split('/',1)[1]
    return str(value or '')
def true(value):return value is True or str(value).lower()=='true'

def source_for(name,data,definition=None):
    provided=data.get('sources',{}).get(name,{}) if isinstance(data.get('sources'),dict) else {}
    if not isinstance(provided,dict):provided={}
    return {**provided,'label':provided.get('label',name),'files':provided.get('files',[f'evidence/{name}.csv']),
      'metricDefinitions':[{'label':'指标与范围','definition':definition or provided.get('definition','读取独立审核后的冻结实验数据；保留原单位、方案角色、样本范围及缺失值。'),'componentIds':[]}],
      'evidenceFlow':provided.get('evidenceFlow',[{'title':'冻结证据读取','detail':'reports/exp007_evidence.py 从已完成档案提取和独立核对；本报告构建器只读取 evidence/report_data.json，不调用训练、规划或执行。'}])}


def normalize(rows):
    output=[]
    for source in rows:
        row=dict(source)
        rid=key_of(row)
        if rid in RUNS:
            row.update(run_id=rid,id=rid,name=rid,label=LABELS[rid],rl_seed=int(rid.rsplit('_',1)[1]),family='轻惩罚' if rid.startswith('regularized') else '纯费用',policy_id='exp007/'+rid)
        if n(row,'total_cost') is not None:row['total_wan']=n(row,'total_cost')/10000
        if n(row,'planned_cost') is not None:row['planned_wan']=n(row,'planned_cost')/10000
        if n(row,'emergency_cost') is not None:row['emergency_wan']=n(row,'emergency_cost')/10000
        row['chart_label']=str(row.get('label') or row.get('policy_id') or rid).replace('exp00','').replace('正式·轻惩罚·','RL主组 ').replace('无季节','无季节').replace('旧执行·重新规划','贪心重规划').replace('固定购电·旧执行','固定购电贪心')
        if 'training_transitions' in row:row['transitions']=row['training_transitions']
        if 'soc_end_kwh' in row:row['soc_kwh']=row['soc_end_kwh']
        if not row.get('date') and isinstance(row.get('month'),(int,float)) and 1<=row['month']<=12:row['date']=f"2025-{int(row['month']):02d}-01"
        if row.get('family') in ('regularized','cost_only'):row['family']={'regularized':'轻惩罚','cost_only':'纯费用'}[row['family']]
        for alias,original in {'mean_abs_delta_kw':'mean_absolute_change_kw','rms_delta_kw':'rms_change_kw','p95_abs_delta_kw':'p95_absolute_change_kw','max_abs_delta_kw':'max_absolute_change_kw','direct_reversals':'direct_adjacent_reversals','at_limit_pct':'power_limit_hit_pct','big_jump_pct':'large_jump_pct','warmup_boundary_delta_kw':'warmup_boundary_jump_kw'}.items():
            if original in row:row[alias]=row[original]
        output.append(row)
    return output


def load_power(data):
    """Convert reviewed battery source paths to complete arrays, never sample."""
    battery=data.get('battery',{})
    curves=battery.get('full_curves',data.get('powerSeries',[]))
    if isinstance(curves,dict):curves=[{'key':k,**v} if isinstance(v,dict) else {'key':k,'values':v} for k,v in curves.items()]
    result=[]
    for curve in curves:
        r=dict(curve);key=r.get('key',r.get('policy_id'));values=r.get('values',r.get('power_kw',r.get('net_power_kw')))
        if values is not None:
            values=[float(v) for v in values]
            if len(values)!=48096 or not all(math.isfinite(v) for v in values):raise ValueError(f'Incomplete power curve {key}')
            result.append({'key':key,'label':r.get('label',key),'values':values,'source':r.get('source')})
    if result:return result
    # Evidence builder may expose explicit frozen NPZ files instead of arrays.
    policies=battery.get('policies',[])
    if isinstance(policies,dict):policies=[{'key':k,**v} if isinstance(v,dict) else {'key':k,'source':v} for k,v in policies.items()]
    for policy in policies:
        if not isinstance(policy,dict):continue
        file=policy.get('archive',policy.get('source',policy.get('path',policy.get('npz'))))
        if not file:continue
        path=Path(file)
        if not path.is_absolute():path=ROOT/path if (ROOT/path).exists() else EVIDENCE/path
        if path.suffix!='.npz' or not path.exists():continue
        import numpy as np
        with np.load(path) as z:values=(6*(z['charge']-z['discharge'])).ravel().tolist()
        key=policy.get('key',policy.get('policy_id'))
        if len(values)!=48096:raise ValueError(f'Incomplete power archive {key}')
        result.append({'key':key,'label':policy.get('label',key),'values':values,'source':str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)})
    return result


def build(complete=False,node=NODE,data_plugin=PLUGIN):
    started=time.monotonic()
    input_path=EVIDENCE/'report_data.json'
    if not input_path.is_file():raise FileNotFoundError('Reviewed evidence/report_data.json is required; no demo data are used.')
    data=read(input_path);tables={k:normalize(v) for k,v in data['tables'].items() if isinstance(v,list)}
    runs=tables.get('all_runs',[])
    if {key_of(r) for r in runs}!=set(RUNS):raise ValueError('Expected six prespecified complete runs')
    primary=next(r for r in runs if key_of(r)=='regularized_42')
    costs=tables.get('cost_history',[])
    baseline=next(r for r in costs if r.get('policy_id')=='exp004/no_season')
    tree=next(r for r in costs if r.get('policy_id')=='exp006/primary')
    total=n(primary,'total_cost');base=n(baseline,'total_cost');delta=total-base
    protocol=data.get('protocol') or read(ROOT/'experiments/problem2/rl_planning/protocol.approved.json')
    title='exp007：RL 降低电池周转，但购电费用上升' if delta>0 else 'exp007：RL 规划的购电费用与电池运行权衡'
    old=read(APP/'src/data.json')
    snapshot={'id':old['id'],'title':title,'surface':'report','buildStatus':'complete' if complete else 'updating','report':{'asOf':'2025-12-31'},'queries':{},'specifiedDates':SPECIFIED,'runIds':['regularized_42','cost_only_42'],'policyLabels':LABELS}
    queries=snapshot['queries'];sections=[{'title':t,'blocks':[]} for t in TITLES];markdown=[[] for _ in TITLES]
    def query(qid,rows,origin=None,definition=None,files=None):
        queries[qid]={'rows':rows,'source':source_for(origin or qid,data,definition)}
        if files:queries[qid]['source']['files']=files
        return qid
    for name,rows in tables.items():query(name,rows)
    def prose(s,id,text,qs=None):
        sections[s]['blocks'].append({'type':'prose','id':id,'markdown':text,'queryIds':qs or []});markdown[s].append(text)
    def table(s,id,title,qid,columns,limit=None):
        rows=queries[qid]['rows']
        if limit and len(rows)>limit:
            qid=query(qid+'-display',rows[:limit],qid)
        sections[s]['blocks'].append({'type':'table','id':id,'title':title,'queryId':qid,'columns':columns});markdown[s].extend(['### '+title,md_table(queries[qid]['rows'],columns)])
    def chart(s,id,title,qid,y,x='chart_label',kind='horizontalBar',series=None,fields=None,unit='',height=400,scope=None,controls=False):
        rows=queries[qid]['rows'];spec={'type':kind,'x':x,'y':y,'stackable':False,'valueDecimals':3 if y in ('approx_kl','entropy','clip_fraction') else 2,'showValues':kind=='horizontalBar','colors':{field:COLORS[i%len(COLORS)] for i,field in enumerate(fields or [y])},'legend':{'labels':NAMES}}
        if series:spec['series']=series
        if fields:spec['fields']=fields
        spec['xLabel' if kind=='horizontalBar' else 'yLabel']=unit
        if kind=='horizontalBar':
            values=[n(r,f) for r in rows for f in fields or [y] if n(r,f) is not None]
            low=min([0,*values]);high=max([0,*values]);span=high-low or 1
            spec.update(presentation='plot',barOptions={'orientation':'horizontal','categoryWidth':180,'grid':True,'domain':[low-.08*span if low<0 else 0,high+.2*span],'labels':{'value':True},'style':{'color':COLORS[0],'radius':0,'thickness':21},'format':{'maximumFractionDigits':2}})
            if fields:spec['barOptions']['series']=[{'key':field,'label':NAMES.get(field,field),'color':COLORS[i%len(COLORS)]} for i,field in enumerate(fields)]
        sections[s]['blocks'].append({'type':'chart','id':id,'title':title,'queryId':qid,'spec':spec,'height':height,'scope':scope,'controls':controls})
    def routes(s,id,title,rows):
        qid=query(id,rows,'protocol','按冻结源码和协议重述算法数据流。',files=['evidence/protocol.json','methods.md'])
        sections[s]['blocks'].append({'type':'routes','id':id,'title':title,'queryId':qid})
        markdown[s].append('### '+title+'\n\n'+'\n\n'.join('**'+r['label']+'**：'+' → '.join(r['steps']) for r in rows))
    for i,title_s in enumerate(TITLES):prose(i,f'chapter-{i+1}-title',f'## {i+1}. {title_s}')
    # The frozen primary role never follows the cheapest measured seed.
    battery=tables.get('battery_history',[])
    pbat=next((r for r in battery if r.get('policy_id')=='exp007/regularized_42'),primary)
    bbat=next((r for r in battery if r.get('policy_id')=='exp004/no_season'),baseline)
    efc=n(pbat,'equivalent_full_cycles','cell_side_equivalent_full_cycles');befc=n(bbat,'equivalent_full_cycles','cell_side_equivalent_full_cycles')
    turns=n(pbat,'direction_reversals','non_idle_direction_reversals');bturns=n(bbat,'direction_reversals','non_idle_direction_reversals')
    verdict='本轮固定预算 RL 未降低总费用，不能替代同预测基线作为最低费用方案。' if delta>0 else '本轮固定预算 RL 在该年同口径回放中降低了总费用；泛化范围仍限于本次协议。'
    prose(0,'primary-verdict',f'**{verdict}**\n\n- 预声明主组 `regularized_42` 的 334 日真实总费为 **{total/10000:,.2f} 万元**，相对 exp004 无季节 **{signed(delta/10000)} 万元（{signed(pct(total,base))}%）**；相对 exp006 正式树DP **{signed((total-n(tree,"total_cost"))/10000)} 万元**。\n- 电芯等效全循环为 **{fmt(efc)} 次**，相对同预测基线 **{signed(pct(efc,befc))}%**；非空方向反转 **{fmt(turns,0)} 次**，基线 **{fmt(bturns,0)} 次**。这些描述运行强度，不代表已证实寿命收益。\n- 六组训练预算和角色提前冻结。正式组保持 seed42；其余种子和纯费用对照全部展示，不按全年最低账单换主组。', ['all_runs','cost_history','battery_history'])
    dplan=n(primary,'planned_cost')-n(baseline,'planned_cost');dem=n(primary,'emergency_cost')-n(baseline,'emergency_cost')
    prose(0,'cost-decomposition',f'主组相对同预测基线：计划费变化 **{signed(dplan/10000)} 万元**，紧急费变化 **{signed(dem/10000)} 万元**，合计 **{signed(delta/10000)} 万元**。\n\n当前结论来自真实执行账单，奖励中的软罚与库存估值不计入题目费用。PPO 固定预算的训练诊断不能证明收敛或最优。',['all_runs','cost_history'])
    anchors=['exp004/no_season','exp006/primary','exp006/greedy_execution','exp007/regularized_42','exp007/cost_only_42']
    anchor_rows=[r for r in costs if r.get('policy_id') in anchors]
    for run_id in ('regularized_42','cost_only_42'):
        if not any(r.get('policy_id')=='exp007/'+run_id for r in anchor_rows):anchor_rows.append(next(r for r in runs if key_of(r)==run_id))
    query('main-comparison',anchor_rows,'cost_history','334 日、同冻结预测主种子42；真实费用=计划费+五倍紧急费。exp006 对照保持原消融角色。')
    table(0,'main-results','主组与同预测历史锚点','main-comparison',[('label','方案'),('role','角色'),('planned_cost','计划费/元'),('emergency_cost','紧急费/元'),('total_cost','总费/元'),('emergency_kwh','紧急/kWh')])
    chart(0,'main-costs','同预测的真实费用分项','main-comparison','planned_wan',fields=['planned_wan','emergency_wan'],unit='万元',height=430)
    # Definitions and immutable information boundaries.
    prose(1,'billing-boundary','每天零点一次锁定全部 144 槽普通购电，日内不更改；电价与负载/光伏预测在规划时已知，实际供需仅在该槽执行时揭示。普通购电全部计费，实际账单为 **C=Σp(g+5e)**；没有售电收入、普通计划调整费或年末库存残值抵扣。\n\n功率为 kW，十分钟电量为功率/6 kWh。SOC 使用电芯侧能量；充放电量使用交流母线侧能量，两者通过单向效率 sqrt(0.9) 连接。最大容量 12000 kWh，SOC 1200–10800 kWh，单向功率最多 5000 kW。主组没有额外硬爬坡上限。',['all_runs'])
    metrics=[{'metric':'MAE','definition':'Σ|预测−实际|/n','unit':'kW','scope':'目标与总体分别核算'}, {'metric':'RMSE','definition':'sqrt(Σ(预测−实际)²/n)','unit':'kW','scope':'不能平均月度RMSE'}, {'metric':'WAPE','definition':'100×Σ|预测−实际|/Σ|实际|；零分母为空','unit':'%','scope':'绝对差用百分点'}, {'metric':'题目费用','definition':'Σp×计划购电+Σ5p×紧急购电','unit':'元','scope':'奖励罚项不计入'}, {'metric':'电芯EFC','definition':'(ηΣ充电+Σ放电/η)/(2×12000)','unit':'次','scope':'运行强度代理'}, {'metric':'净功率变化','definition':'ΔP=P_t−P_(t−1)，连续334日含跨午夜','unit':'kW/相邻10分钟','scope':'预热边界另报，不补首点零'}, {'metric':'非空方向反转','definition':'去除闲置后，相邻非空动作方向相反','unit':'次','scope':'与直接相邻槽反转分开'}]
    query('definitions',metrics,'protocol','指标定义来自冻结协议及 main 电池模板。',files=['report_contract.md','evidence/protocol.json'])
    table(1,'metric-definitions','费用、预测和电池指标','definitions',[('metric','指标'),('definition','定义'),('unit','单位'),('scope','范围')])
    prose(1,'objective-priority','互斥、SOC 和单向功率限额由有符号动作及物理投影保证；轻惩罚是费用之外的次要偏好。三项软罚为 0.002 元/交流侧kWh 吞吐、0.05 元/非空反转、0.0002 元/kW 相邻功率变化。纯费用消融将三项都置零，保留相同物理、动作、2 kWh 死区及终值规则。不能把这些未标定偏好称为实际电池折旧。',['definitions'])
    # Source hashes and causal schedule.
    hashes=protocol.get('data_sha256',{})
    query('input-hashes',[{'file':k,'sha256':v} for k,v in hashes.items()],'protocol','冻结输入文件SHA-256；详见实验协议和独立核验。',files=['evidence/protocol.json'])
    table(2,'source-hashes','输入校验值','input-hashes',[('file','文件'),('sha256','SHA-256')])
    prose(2,'time-validation','正式评价为 **2025-02-01 至 12-31，334 日、每组 48,096 槽**。附件2按区间终点解释，00:10 是当天第一段，24:00 是当天最后一段。初始状态承接真实一月预热，2 月 1 日 SOC 1421.7991105135516 kWh、上一槽净功率 172.76 kW；各组此后延续自己的实际状态。\n\n首次训练只用 1 月 9–31 日的因果周期预测回退，后续每 14 日只用已完成的最近至多90日继续训练。当天144槽计划锁定前不访问当天实况奖励。预报、训练 cutoff、checkpoint 与执行档案的哈希和因果核验单独保留。',['all_runs'])
    forecast_hash=protocol.get('forecast',{}).get('sha256','未提供')
    prose(2,'source-predictions',f'六组统一复用 exp004 `no_season/seed42` 冻结预测，SHA-256 `{forecast_hash}`；没有重训预测网络。RL 随机种子与预测种子分别记录。旧实验共享早停、回顾性评价和含未来探索的边界沿历史记录说明，不能因本轮逐槽因果就抹去历史研究的先验接触。',['forecast_history'])
    methods=(REPORT/'methods.md').read_text();parts=re.split(r'^## ',methods,flags=re.MULTILINE)
    for i,part in enumerate(parts[1:]):prose(3,f'method-detail-{i+1}','### '+part,['all_runs'])
    network=[{'layer':'观测输入','units':29,'activation':'固定尺度','branch':'shared','purpose':'午夜预测与计划状态'}, {'layer':'共享层 1','units':64,'activation':'tanh','branch':'shared','purpose':''}, {'layer':'共享层 2','units':64,'activation':'tanh','branch':'shared','purpose':''}, {'layer':'策略头','units':45,'activation':'线性 logits → softmax','branch':'policy','purpose':'训练抽样；正式 argmax'}, {'layer':'价值头','units':1,'activation':'线性','branch':'value','purpose':'估计训练日回报'}]
    query('network',network,'protocol','29→64→64共享网络分成45类策略头与1维价值头，来自ppo.py与environment.py。',files=['evidence/protocol.json','methods.md'])
    sections[3]['blocks'].insert(1,{'type':'network','id':'ppo-network','title':'实际 PPO 网络：共享表示与双输出头','queryId':'network'})
    markdown[3].insert(1,md_table(network,[('layer','网络层'),('units','维度'),('activation','变换'),('purpose','用途')]))
    routes(3,'planning-route','规划、隐藏奖励与真实执行的信息流',[{'label':'午夜规划','steps':['冻结144×2预测','历史误差九点支持','29维观测','PPO逐槽选45类动作','锁定全天购电与意图']},{'label':'训练更新','steps':['只抽已完成历史日','32个完整日rollout','计划完成后执行并计奖励','GAE优势估计','四轮PPO更新']},{'label':'年度执行','steps':['固定午夜计划','仅本槽实际供需','意图动作按物理裁剪','紧急补购或弃电','真实SOC跨日']}])
    # Configured six runs and fixed training budgets.
    configuration=[]
    for group in ('algorithm','training','battery'):
        for key,value in protocol.get(group,{}).items():configuration.append({'group':group,'parameter':key,'value':json.dumps(value,ensure_ascii=False)})
    query('configuration',configuration,'protocol','批准后的固定算法/训练/电池协议，未由全年成绩调整。',files=['evidence/protocol.json'])
    table(4,'frozen-configuration','冻结算法、训练与物理配置','configuration',[('group','类别'),('parameter','参数'),('value','值')])
    table(4,'six-run-matrix','六组角色与训练种子','all_runs',[('label','方案'),('role','角色'),('rl_seed','RL种子'),('days','评价日'),('transitions','训练转移'),('run_wall_seconds','任务墙钟/s')])
    prose(4,'seed-design','轻惩罚与纯费用各有 RL seed42、2026、3407，共六组。正式主组提前固定为 `regularized_42`；其他轻惩罚种子检验训练稳定性，纯费用三种子是配对消融。每种子固定 1230 次更新、5,667,840 个转移，无未来费用选模、无种子集成。\n\n网络和优化在 CPU 单线程运行，32 个批量环境同步构造日计划。任务墙钟包含准备、训练、预测规划、执行与存储，阶段累计有包含关系，不能与墙钟相加。报告和工作簿导出另计。',['all_runs','configuration'])
    # Actual results and failures.
    table(5,'all-six-results','全部六组真实费用与电池结果','all_runs',[('label','方案'),('planned_cost','计划费/元'),('emergency_cost','紧急费/元'),('total_cost','总费/元'),('emergency_kwh','紧急/kWh'),('equivalent_full_cycles','电芯EFC'),('direction_reversals','非空反转'),('simultaneous_slots','同时槽'),('final_soc','期末SOC/kWh')])
    chart(5,'six-costs','轻惩罚与纯费用 · 六组真实总费用','all_runs','total_wan',unit='万元',height=500)
    control=next(r for r in runs if key_of(r)=='cost_only_42')
    regular=[r for r in runs if key_of(r).startswith('regularized')];plain=[r for r in runs if key_of(r).startswith('cost_only')]
    rmean=sum(n(r,'total_cost') for r in regular)/3;cmean=sum(n(r,'total_cost') for r in plain)/3
    prose(5,'paired-penalty-result',f'六组真实费用都高于 exp006 正式策略与其旧执行重规划对照，固定预算下未达到降本目标。轻惩罚三种子总费均值 **{rmean/10000:,.2f} 万元**，纯费用三种子均值 **{cmean/10000:,.2f} 万元**。\n\n同为RL seed42时，轻惩罚比纯费用多 **{fmt(total-n(control,"total_cost"))} 元（{signed(pct(total,n(control,"total_cost")))}%）**。EFC 从 **{fmt(n(control,"equivalent_full_cycles"))}** 降到 **{fmt(efc)}**，但非空反转从 **{fmt(n(control,"direction_reversals"),0)}** 增到 **{fmt(turns,0)}**，直接相邻反转从 **{fmt(n(control,"direct_adjacent_reversals"),0)}** 增到 **{fmt(n(primary,"direct_adjacent_reversals"),0)}**。轻惩罚没有让每项电池指标都改善；只能解释为本次组合和训练轨迹的结果。', ['all_runs','seed_summary'])

    if tables.get('seed_summary'):
        for r in queries['seed_summary']['rows']:r['metric_alias']=r['metric'];r['metric']={'total_cost':'总费用/元','emergency_cost':'紧急费/元','emergency_kwh':'紧急购电/kWh','equivalent_full_cycles':'电芯EFC/次','throughput_kwh':'交流吞吐/kWh','direction_reversals':'非空反转/次','total_variation_kw':'功率总变差/kW','training_seconds':'策略训练/s'}.get(r['metric'],r['metric'])
        table(5,'seed-summary','三种子均值与样本标准差','seed_summary',[('family','组别'),('metric','指标'),('n','样本数'),('mean','均值'),('sample_std','样本标准差')])
    if tables.get('monthly_cost'):chart(5,'monthly-cost','逐月真实总费用','monthly_cost','total_cost',x='date',kind='line',series='label',unit='元',height=430)
    if tables.get('paired_daily'):chart(5,'daily-difference','正式 RL 减同预测基线 · 逐日费用差','paired_daily','difference_yuan',x='date',kind='line',unit='元；正值更贵',height=390)
    daily=tables.get('daily',[]);pdaily=[r for r in daily if key_of(r)=='regularized_42']
    if pdaily:
        worst=max(pdaily,key=lambda r:n(r,'total_cost',default=-math.inf))
        prose(5,'worst-day',f'正式主组最贵日为 **{worst.get("date")}**，日费 **{fmt(n(worst,"total_cost"))} 元**，紧急购电 **{fmt(n(worst,"emergency_kwh"))} kWh**。这里选择最不利日期说明局限，随机日电池展示则独立按固定种子抽样。',['daily'])
    # Whole raw trajectories. No aggregation or downsampling is allowed.
    powers=load_power(data)
    if not powers:raise ValueError('Full 48,096-point battery arrays required before report build')
    snapshot['powerSeries']=powers;snapshot['randomBatteryDates']=data.get('battery',{}).get('dates',[])
    if len(snapshot['randomBatteryDates'])<4:raise ValueError('At least four fixed-seed battery dates required')
    power_rows=[{'policy_id':s['key'],'label':s['label'],'points':len(s['values']),'start_date':'2025-02-01','end_date':'2025-12-31','power_kw_json':json.dumps(s['values'],separators=(',',':'))} for s in powers]
    query('battery-full',power_rows,'power_variation','每方案48096个原始十分钟实际净功率点，P=6(charge−discharge)，充正放负、kW；完整数据内嵌，无平滑降采样。',files=data.get('battery',{}).get('csv_files',['evidence/battery_power_full.csv']))
    queries['battery-full']['payloadColumns']=['power_kw_json']
    sections[5]['blocks'].extend([{'type':'batteryRandom','id':'battery-random-days','title':'固定随机四日 · 实际净充放电功率','queryId':'battery-full'},{'type':'battery','id':'battery-full-year','title':'完整 334 日 · 每方案全部 48,096 个原始点','queryId':'battery-full'}])
    prose(5,'battery-interpretation','随机种子 **20260912** 从完整评价日不放回抽四日，所有策略用相同日期与纵轴：'+ '、'.join(snapshot['randomBatteryDates'])+'。图中充电为正、放电为负，十分钟电量乘6得到kW。全年视图包含全部原始点；可选单日或日期范围放大，并用滑块查看每个原始槽。图中±5000 kW为功率边界，未混入1月预热。\n\n正式期相邻功率差包含跨午夜；首点没有差分，预热末点到正式首点的跳变单列。大跳变定义为 |ΔP|>1000 kW/10min，仅为描述阈值。幅值受限、SOC平滑和功率变化受限是不同事实，不能互相替代。',['battery-full'])
    power_metrics=tables.get('power_variation',[])
    if power_metrics:
        query('power-variation-focus',[r for r in power_metrics if r.get('policy_id') in {x['key'] for x in powers}], 'power_variation')
        table(5,'power-variation','完整时间轴的功率波动指标','power-variation-focus',[('label','方案'),('mean_abs_delta_kw','平均|ΔP|/kW'),('rms_delta_kw','变化RMS/kW'),('p95_abs_delta_kw','P95|ΔP|/kW'),('max_abs_delta_kw','最大|ΔP|/kW'),('total_variation_kw','总变差/kW'),('direct_reversals','相邻直接反转'),('at_limit_pct','贴限/%'),('big_jump_pct','大跳变/%'),('warmup_boundary_delta_kw','预热跳变/kW')])
    prose(5,'power-limit-finding',f'主组平均绝对功率变化为 **{fmt(n(primary,"mean_absolute_change_kw"))} kW/10min**，RMS **{fmt(n(primary,"rms_change_kw"))}**，但最大跳变仍达 **{fmt(n(primary,"max_absolute_change_kw"))} kW/10min**。相邻直接反转 **{fmt(n(primary,"direct_adjacent_reversals"),0)} 次**，大于1000 kW的变化占 **{fmt(n(primary,"large_jump_pct"))}%**。这些结果不构成硬爬坡保证。',['power_variation'])
    if battery:
        focus_battery=[r for r in battery if r.get('policy_id') in [*anchors,'exp002/primary','exp003/primary','exp004/causal_season','exp005/beta_0.1','exp005/beta_0.01','exp001/legacy_rebased']]
        query('battery-focus',focus_battery,'battery_history')
        table(5,'battery-strength','电池吞吐与非空方向反转','battery-focus',[('label','方案'),('charge_kwh','充电/kWh'),('discharge_kwh','放电/kWh'),('equivalent_full_cycles','电芯EFC'),('direction_reversals','非空反转'),('simultaneous_slots','同时槽'),('initial_soc','初SOC/kWh'),('final_soc','末SOC/kWh')])
        chart(5,'battery-efc','电芯等效全循环 · 正式评价期','battery-focus','equivalent_full_cycles',unit='次；运行强度代理',height=550)
    training=tables.get('training',[])
    if training:
        for r in training:r['training_step']=n(r,'cumulative_transitions','transitions','steps','update','iteration','iteration_total',default=None)
        if not any(r['training_step'] is not None for r in training):
            counters={}
            for r in training:
                rid=key_of(r);counters[rid]=counters.get(rid,0)+1;r['training_step']=counters[rid]
        query('training',training,'training','固定预算的训练历史回合日志；训练池每14日改变，跨截止均值不代表收敛或独立验证改善；缺失诊断不填零。')
        for metric,title_m,unit in [('entropy','策略熵','nats'),('approx_kl','近似 KL','无量纲'),('clip_fraction','PPO 裁剪比例','0–1'),('value_loss','价值损失','缩放回报平方')]:
            if any(n(r,metric) is not None for r in training):chart(5,'training-'+metric,title_m+' · 六组训练过程','training',metric,x='training_step',kind='line',series='label',unit=unit,height=340)
        prose(5,'training-limitations','训练曲线显示固定预算内的优化行为。价值损失随不同历史池、随机初始状态和奖励分布可改变，不能只用其绝对大小判断优劣；策略熵下降也可能伴随过早确定化。每14日历史训练池变化，跨截止点均值变化不能称为收敛或独立验证改善。末日关闭终值只影响奖励核算，actor没有年末特征，已学到的库存偏好不会自动消失。PPO 没有全局最优证书，本轮没有凭全年费用调整训练预算。',['training'])
    detail=tables.get('detail',tables.get('specified_detail',[]))
    if detail:
        query('specified-detail',detail,'detail','题目指定四日，各策略实际供需、锁定购电、执行电池和SOC；十分钟kWh/功率kW分别标注。')
        x='hour' if 'hour' in detail[0] else 'slot'
        chart(5,'specified-energy','指定日供需与购电','specified-detail','actual_net_kwh',x=x,kind='line',fields=['actual_net_kwh','forecast_net_kwh','planned_kwh','emergency_kwh'],unit='每10分钟kWh',scope='specified',controls=True)
        chart(5,'specified-soc','指定日实际电芯 SOC','specified-detail','soc_kwh',x=x,kind='line',unit='kWh',scope='specified')
    specified_path=REPORT/'specified_tables/specified_tables.json'
    if specified_path.exists():
        specified_tables=read(specified_path)
        for name,tt,columns in [
          ('table1_summary','题目四个指定日 · 正式组日汇总',[('date','日期'),('planned_kwh','计划/kWh'),('planned_cost_yuan','计划费/元'),('emergency_cost_yuan','紧急费/元'),('total_cost_yuan','总费/元'),('initial_soc_kwh','日初SOC/kWh'),('final_soc_kwh','日末SOC/kWh')]),
          ('table1_intervals','表1 · 六个指定十分钟区间',[('date','日期'),('period','区间'),('planned_kwh','计划/kWh')]),
          ('table2_battery','表2 · 六个四小时段电池电量',[('date','日期'),('period','四小时段'),('charge_kwh','充电/kWh'),('discharge_kwh','放电/kWh')]),
          ('table3_emergency','表3 · 指定日全部紧急购电区间',[('date','日期'),('period','连续区间'),('emergency_kwh','紧急/kWh')])]:
            qid=query('specified-'+name,specified_tables[name],'specified','主工作簿同源原始题目表，读回核验通过；时间区间按题意汇总。',files=[f'specified_tables/{name}.csv','evidence/workbook_qa.json'])
            table(5,'specified-'+name,tt,qid,columns)
    if pdaily and detail:
        wrows=[r for r in detail if r.get('date')==worst['date'] and key_of(r)=='regularized_42']
        if wrows:
            query('worst-detail',wrows,'detail','正式组最贵日原始执行数据，不选择有利日期。')
            chart(5,'worst-energy',f'最贵日供需与购电 · {worst["date"]}','worst-detail','actual_net_kwh',x='hour',kind='line',fields=['actual_net_kwh','forecast_net_kwh','planned_kwh','emergency_kwh'],unit='每10分钟kWh')
            chart(5,'worst-soc',f'最贵日实际SOC · {worst["date"]}','worst-detail','soc_kwh',x='hour',kind='line',unit='kWh')
    prose(5,'workbook-delivery','[原题全年工作簿](result2.xlsx)与[指定四日完整表格](specified_dates.md)按正式主组输出。工作簿保留 334 日三张题目表，涵盖144槽普通购电、六个四小时段充放电与日初日末SOC、全部紧急购电区间。原始表格及读回证据同包保存；四小时聚合充放电均为正不意味着同一个十分钟槽同时充放电。另保留[纯费用seed42工作簿](cost_only_42/result2.xlsx)及其[指定日表](cost_only_42/specified_dates.md)，仍为消融角色。')
    # Full historical coverage: incompatible rows remain visible, never ranked.
    historical_ids=['exp001/legacy_rebased','exp002/primary','exp002/new_deterministic','exp003/primary','exp003/uncalibrated','exp004/no_season','exp004/causal_season','exp006/primary','exp006/greedy_execution','exp006/fixed_primary_greedy','exp007/regularized_42','exp007/cost_only_42']
    history_focus=[r for r in costs if r.get('policy_id') in historical_ids and true(r.get('ranking_allowed',r.get('comparable',False)))]
    query('history-comparable',history_focus,'cost_history','按实验编号并列同物理计费历史。不同控制器可比较实际费用，但差异不能单独归因预测或某一机制。')
    chart(6,'history-costs','历次正式与重要对照 · 同物理计费费用','history-comparable','total_wan',unit='万元',height=730)
    descriptive=[r for r in costs if r.get('experiment')=='exp005' or r.get('policy_id') in ('exp001/original','exp004/oracle_season')]
    query('history-descriptive',descriptive,'cost_history','原协议、未来探索及额外硬爬坡/紧急充电协议不同，只展示原值，不排名或算改善率。')
    table(6,'history-incompatible','协议不同或含未来 · 保留原始成绩','history-descriptive',[('label','策略'),('total_cost','总费/元'),('days','日数'),('role','原角色'),('status','可比性说明')])
    prose(6,'history-protocols','exp001 原登记与同物理重算分别保留；exp001/2 早期共享四目标早停边界照实列示。exp004 全年探索含未来实际信息，不能作为可用策略排名。exp004 无季节是本轮同预测锚点，历史季节保留当轮正式角色。\n\nexp005 的供需与电价数值、时间轴已由证据代理单独核对，但其 **1000 kW/10min 硬爬坡** 与 **允许紧急购电服务充电** 的执行协议不同，费用仅作描述性对照。exp006 正式树DP、旧执行重规划、固定购电旧执行保持原角色；不能因为其中消融费用较低而改写它的正式策略。',['cost_history'])
    relative=tables.get('relative_comparison',[])
    relcost=[r for r in relative if r.get('metric')=='total_cost']
    if relcost:
        query('relative-cost',relcost,'relative_comparison')
        table(6,'relative-cost-values','费用比较原值、绝对差和相对变化','relative-cost',[('previous_label','历史策略'),('previous','历史/元'),('current','RL/元'),('absolute_change','差值/元'),('relative_change_pct','相对变化/%'),('comparable','可比')])
        good=[dict(r,chart_label=r.get('previous_label',r.get('previous_experiment'))) for r in relcost if true(r.get('comparable'))]
        query('relative-cost-valid',good,'relative_comparison');chart(6,'relative-cost-chart','相对历史费用变化 · 负值才是下降','relative-cost-valid','relative_change_pct',unit='%',height=600)
    forecasts=tables.get('forecast_history',[])
    for target,label_t in [('load','负载'),('pv','光伏'),('net_load','净负载')]:
        rows=[r for r in forecasts if r.get('target')==target and r.get('population')=='all' and not true(r.get('exploratory'))]
        if rows:
            qid=query('history-forecast-'+target,rows,'forecast_history','同334日48096个午夜预测槽；每个目标单独计算RMSE/WAPE。RL复用预测，无新增预测训练。')
            for metric,unit in [('rmse','kW'),('wape_pct','%')]:chart(6,'forecast-'+target+'-'+metric,'历次'+label_t+' '+('RMSE' if metric=='rmse' else 'WAPE'),qid,metric,unit=unit,height=max(430,len(rows)*46+90))
    generating=[r for r in forecasts if r.get('target')=='pv' and r.get('population')=='pv_generating' and not true(r.get('exploratory'))]
    if generating:
        query('pv-generating',generating,'forecast_history');table(6,'pv-generating-errors','实际光伏发电时段预测误差','pv-generating',[('label','预测来源'),('n','槽数'),('mae','MAE/kW'),('rmse','RMSE/kW'),('wape_pct','WAPE/%')])
    for target,target_label in [('load','负载'),('pv','光伏')]:
        mrows=[r for r in tables.get('forecast_monthly',[]) if r.get('target')==target and r.get('population')=='all']
        if mrows:
            qid=query('forecast-monthly-'+target,mrows,'forecast_monthly','同冻结预测逐月误差，由误差分子分母独立汇总，RL不重训。')
            chart(6,'forecast-monthly-'+target,'复用预测逐月 '+target_label+' RMSE',qid,'rmse',x='date',kind='line',unit='kW')
        lrows=[dict(r,lead=f'{int(r["lead_start_hour"])}–{int(r["lead_end_hour"])} h') for r in tables.get('forecast_lead',[]) if r.get('target')==target]
        if lrows:
            qid=query('forecast-lead-'+target,lrows,'forecast_lead','午夜预测按六个四小时提前量分组，每组8016槽，误差从原始分子分母重汇总。')
            chart(6,'forecast-lead-'+target,'复用预测提前量 '+target_label+' RMSE',qid,'rmse',x='lead',kind='bar',unit='kW')
    prose(6,'reused-error','exp007 六组复用同一 exp004 无季节 seed42 预测，预测误差完全相同，不能声称 RL 提高预测精度。WAPE 原值单位是%，绝对差为百分点；相对变化仍为%。不把不同变量的原始单位强行放在同一坐标。',['forecast_history'])
    timings=tables.get('timings',[])
    if timings:
        timings=[r for r in timings if r.get('experiment')!='exp006' or any(t in str(r.get('label','')) for t in ('正式·树DP','旧执行·重新规划','复用预测'))]
        query('timings',timings,'timings')
        # Keep stage names and scope; split into meaningful comparable plots.
        table(6,'timing-values','全部阶段时间及计时范围','timings',[('label','方案'),('stage','阶段'),('seconds','累计/s'),('groups','组数/日数'),('scope','计时范围')])
        stage_groups=[('training',lambda r:'训练' in str(r.get('stage','')) or 'train' in str(r.get('stage','')).lower(),'训练计算时间 · 预测网络与RL含义见来源'),('planning',lambda r:any(s in str(r.get('stage','')).lower() for s in ['调度执行','planning','推理','规划执行','计划推理']),'规划/调度与执行累计时间')]
        for ident,keep,tt in stage_groups:
            rows=[dict(r,chart_label=str(r.get('label',''))+' · '+str(r.get('stage',''))) for r in timings if keep(r) and n(r,'seconds') is not None]
            if rows:
                qid=query('timing-'+ident,rows,'timings');chart(6,'timing-'+ident,tt,qid,'seconds',unit='秒；设备/计时范围不同，仅描述',height=max(450,len(rows)*45+70))
    routes(6,'history-routes','历次技术路线如何改变',[{'label':'exp001','steps':['多网络候选','共享四目标早停','原协议与同物理重算']},{'label':'exp002','steps':['固定残差MLP','风险正式/确定性对照','固定午夜购电']},{'label':'exp003','steps':['问题二独立MLP','一月费用校准','确定性规划+旧因果执行']},{'label':'exp004','steps':['168小时双分支CNN','无季节/因果季节','沿用exp003规划执行']},{'label':'exp005','steps':['冻结exp004预测','整日场景树两层LP','硬爬坡与软吞吐罚','紧急购电可服务充电']},{'label':'exp006','steps':['冻结预测','条件误差树','SOC动态规划','固定计划+意图裁剪']},{'label':'exp007','steps':['同预测/条件误差','PPO有限联合动作','固定预算因果训练','午夜开环计划+真实执行奖励']}])
    # Reproduction and current verification files (never overwrite QA evidence).
    qa_rows=[]
    for filename in ['independent_qa.json','workbook_qa.json','verification.json','evidence_manifest.json','exp005_compatibility_audit.json']:
        file=EVIDENCE/filename
        if file.exists():qa_rows.append({'file':'evidence/'+filename,'sha256':sha256(file),'summary':json.dumps(read(file),ensure_ascii=False)[:900]})
    query('qa-files',qa_rows,'verification','已存在的独立核验及导出记录；状态以各JSON内容为准，网页构建不替代视觉验收。',files=[r['file'] for r in qa_rows] or ['evidence/report_data.json'])
    table(7,'verification-evidence','核验记录与文件校验值','qa-files',[('file','证据文件'),('sha256','SHA-256'),('summary','状态摘要')])
    prose(7,'reproduction','在独立 RL 工作树根目录重建报告，读取已审核证据，不会启动训练或规划：\n\n```sh\n/path/to/python -m reports.exp007_evidence\n/path/to/python -m reports.build_report_exp007 --complete --node /path/to/codex/node --data-plugin /path/to/data-analytics\n```\n\n独立核验和工作簿导出的完整命令以本包 README、冻结协议和对应导出脚本帮助为准。原始 checkpoint、逐日计划、实际执行、训练诊断及签名在 `data/results/exp007/` 中。重新训练使用新的隔离副本，不覆盖本次原始计时。\n\n报告正文、图表、离线HTML和工作簿从同源冻结结果产生；显示时四舍五入，CSV和数组保留核算精度。HTML内嵌全部交互数据；工作簿和证据下载依赖同目录附件，请保留整个报告包。')
    manifest=read(EVIDENCE/'raw_run_manifest.json')
    qa=read(EVIDENCE/'independent_qa.json') if (EVIDENCE/'independent_qa.json').exists() else {}
    query('execution-manifest',[{'formal_wall_seconds':manifest.get('formal_wall_seconds'),'code_commit':manifest.get('code_commit'),'python':manifest.get('python'),'platform':manifest.get('platform'),'cpu_threads':manifest.get('cpu_threads'),'completed_runs':len(manifest.get('completed_runs',[]))}],'protocol','正式运行manifest和独立源码绑定核验。',files=['evidence/raw_run_manifest.json','evidence/independent_qa.json'])
    prose(7,'formal-reproduction',f'完整六组正式任务墙钟为 **{fmt(manifest.get("formal_wall_seconds"))} 秒**；主组墙钟 **{fmt(n(primary,"run_wall_seconds"))} 秒**，其中策略训练 **{fmt(n(primary,"training_seconds"))} 秒**。执行源码提交为 `{manifest.get("code_commit")}`，独立核验已逐文件对照冻结Git源码。\n\n复现入口为：\n\n```sh\n/path/to/python -m experiments.problem2.rl_planning.run\n/path/to/python -m experiments.problem2.rl_planning.verify\n/path/to/python -m experiments.problem2.rl_planning.export --node /path/to/codex/node\n```\n\n执行前阅读各入口帮助及本包README；完整重训在新隔离副本进行，已有结果保持原始时间记录。独立核验覆盖6组、每组334日×144槽、固定训练预算、历史截止、checkpoint、锁定计划、真实物理执行与费用；`passed={qa.get("passed")}`，错误条数 `{len(qa.get("errors",[]))}`。', ['execution-manifest','qa-files'])
    prose(7,'template-version','正式构建前已核对远程 main 为 `bca8dbe3004b0a87cd7be71a06891a4af87a9eb8`，其 `reports/` 与本地模板提交 `7ecd9b988665ad628dbf048ed295457166170ffc` 无差异。报告保留八节结构、新增随机日电池原始曲线、完整334日原始曲线和波动诊断；模板逐文件SHA见[报告契约](report_contract.md)。')
    prose(7,'visual-qa-note','代码构建与实际浏览器视觉验收分开记录。独立物理/费用/因果及工作簿核验以同包证据为准；浏览器截图、桌面/窄屏布局与代表性交互检查由最终验收记录另行确认，不以构建成功代替。',['qa-files'])
    # Static figures supplied by the same evidence builder.
    figures=data.get('figures',[])
    if isinstance(figures,dict):figures=list(figures.values())
    figure_paths=[]
    for figure in figures:
        file=figure.get('png',figure.get('path')) if isinstance(figure,dict) else figure
        if not isinstance(file,str):continue
        path=Path(file)
        if path.is_absolute() and path.is_relative_to(REPORT):path=path.relative_to(REPORT)
        if str(path).endswith('.png'):figure_paths.append(str(path))
    if not figure_paths:figure_paths=[str(p.relative_to(REPORT)) for p in sorted((REPORT/'figures').glob('*.png'))]
    for file in figure_paths:
        name=Path(file).stem
        section=6 if any(s in name for s in ('history','forecast','timing')) else 3 if any(s in name for s in ('network','route','technical')) else 5
        markdown[section].append(f'![{name}]({file})')
    # Every query definition points only to actual consuming component IDs.
    for qid,q in queries.items():
        ids=[]
        for s in sections:
            for b in s['blocks']:
                if b.get('queryId')==qid:ids.append(b['id'])
                if qid in b.get('queryIds',[]):ids.append(b['id']+'-sources')
        q['source']['metricDefinitions'][0]['componentIds']=ids
    snapshot['reportContent']=sections
    # Bounded structural verification: reject empty/misbound charts and duplicate IDs.
    component_ids=[];structural_errors=[]
    for section in sections:
        for block in section['blocks']:
            component_ids.append(block['id'])
            for qid in block.get('queryIds',[]):
                if qid not in queries:structural_errors.append(f'missing_source:{block["id"]}/{qid}')
            if block.get('queryId'):
                rows=queries[block['queryId']]['rows']
                if not rows:structural_errors.append(f'empty_source:{block["id"]}')
                if block['type']=='chart':
                    for field in [block['spec']['x'],*(block['spec'].get('fields') or [block['spec']['y']])]:
                        if not any(row.get(field) is not None for row in rows):structural_errors.append(f'missing_field:{block["id"]}/{field}')
    if len(set(component_ids))!=len(component_ids):structural_errors.append('duplicate_component_ids')
    if len(sections)!=8:structural_errors.append('expected_eight_sections')
    if structural_errors:raise ValueError(structural_errors)
    write_json(APP/'src/data.json',snapshot)
    (REPORT/'report.md').write_text('# '+title+'\n\n'+'\n\n'.join('\n\n'.join(s) for s in markdown)+'\n')
    script=Path(data_plugin)/'scripts/data-app.mjs'
    subprocess.run([str(node),str(script),'build','--project-dir',str(APP),'--separate-data'],check=True)
    offline=APP/'.data-app-offline/exports/report.html'
    subprocess.run([str(node),str(script),'export-offline','--project-dir',str(APP),'--output',str(offline)],check=True)
    shutil.copy2(offline,REPORT/'report.html')
    # Copy supported attachments into dist for the same live preview URL.
    for file in ['report.md','methods.md','result2.xlsx','specified_dates.md','report_contract.md']:
        if (REPORT/file).is_file():shutil.copy2(REPORT/file,APP/'dist'/file)
    for dirname in ['evidence','figures','specified_tables','workbook-previews','cost_only_42']:
        if (REPORT/dirname).is_dir():shutil.copytree(REPORT/dirname,APP/'dist'/dirname,dirs_exist_ok=True)
    report_build={'buildStatus':snapshot['buildStatus'],'built_at_utc':datetime.now(UTC).isoformat(),'seconds':time.monotonic()-started,'source_data_sha256':sha256(input_path),'html_sha256':sha256(REPORT/'report.html'),'snapshot_sha256':sha256(APP/'src/data.json'),'main_checked_commit':'bca8dbe3004b0a87cd7be71a06891a4af87a9eb8','template_content_commit':'7ecd9b988665ad628dbf048ed295457166170ffc','query_count':len(queries),'component_count':sum(len(s['blocks']) for s in sections),'battery_curves':len(powers),'battery_points_per_curve':48096,'random_dates':snapshot['randomBatteryDates'],'training_invoked':False,'solver_invoked':False,'formal_primary':'regularized_42','static_validation':{'passed':True,'errors':structural_errors,'unique_component_ids':len(component_ids),'chart_field_bindings_checked':True,'all_original_power_points_checked':True},'browser_visual_check':'separate final evidence; build itself is not visual QA'}
    write_json(REPORT/'report_build.json',report_build);shutil.copy2(REPORT/'report_build.json',APP/'dist/report_build.json')
    print(json.dumps(report_build,ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--complete',action='store_true');p.add_argument('--node',type=Path,default=NODE);p.add_argument('--data-plugin',type=Path,default=PLUGIN);build(**vars(p.parse_args()))
