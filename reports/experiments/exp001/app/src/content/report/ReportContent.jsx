import React, { useState } from "react";
import { DataComponent, EvidenceChart, Dropdown, RichNarrative, ReportSection, useDataApp } from "../../data-app-public.jsx";
import "./report.css";
const names={mlp:"多层感知机",gru:"门控循环网络",tcn:"因果卷积网络",gru_no_calendar:"移除日历特征",gru_one_day:"一天历史窗口",yesterday:"昨日同期",weekly:"历史周同期",selected:"按月验证选模",issued:"附件原始光伏预报"};
const targets={load:"小区负载",pv:"光伏历史预测",price:"电价",pv_corrected:"光伏预报修正"};
const metricLabels={mae:"平均绝对误差",rmse:"均方根误差",wape_pct:"加权绝对百分比误差"};
const fmt=x=>typeof x==="number"?x.toLocaleString("zh-CN",{maximumFractionDigits:4}):typeof x==="boolean"?(x?"是":"否"):(x??"—");
function Table({rows,columns}){return <div className="experiment-table-scroll"><table className="experiment-table" data-reviewed-rows><thead><tr>{columns.map(c=><th key={c[0]}>{c[1]}</th>)}</tr></thead><tbody>{rows.map((r,i)=><tr key={i}>{columns.map(c=><td key={c[0]} className={["technical_path","comparison"].includes(c[0])?"experiment-wrap-cell":undefined}>{fmt(r[c[0]])}</td>)}</tr>)}</tbody></table></div>;}
function Route({rows}){
 const [network,setNetwork]=useState("gru");
 const layers={mlp:["展开七天小时历史","64 个隐藏单元","32 个隐藏单元"],gru:["逐小时读取七天历史","32 个门控循环单元","保留最后的历史表示"],tcn:["读取七天小时历史","四层因果卷积","32 通道，膨胀率 1、2、4、8"]};
 return <DataComponent id="technical-route" queryId="technical_path" kind="custom" title="从数据到购电费用的技术路径" sourceRows={rows} displayRows={rows} headerControls={<Dropdown label="网络结构" showLabel value={network} choices={["mlp","gru","tcn"]} choiceLabels={names} onChange={setNetwork}/>}><div className="experiment-route" aria-label={`${names[network]}技术流程`}>{["CSV 与预报发布时刻","仅使用已完成的历史",...layers[network],"拼接目标时刻、滞后值与已发布预报","共享逐时刻输出层与残差修正","未来 144 个十分钟预测","线性规划生成购电计划","日内更新与实际储能执行","原计划、调整、紧急购电结算"].map((t,i)=><div className={`experiment-route-node ${i%3===2?"flow-down":Math.floor(i/3)%2?"flow-left":"flow-right"}`} style={{"--node-row":Math.floor(i/3)+1,"--node-col":Math.floor(i/3)%2?3-i%3:i%3+1}} key={t}><span>{String(i+1).padStart(2,"0")}</span>{t}</div>)}</div></DataComponent>;
}
export function ReportContent(){
 const {snapshot,appTitle}=useDataApp();
 const [scenario,setScenario]=useState("2"),[target,setTarget]=useState("load"),[model,setModel]=useState("all"),[month,setMonth]=useState("all");
 const [metric,setMetric]=useState("mae"),[question,setQuestion]=useState("2"),[dailyMonth,setDailyMonth]=useState("all"),[chosenDay,setChosenDay]=useState("2025-03-20");
 const [caseVariable,setCaseVariable]=useState("load"),[population,setPopulation]=useState("all");
 const activePopulation=target.startsWith("pv")?population:"all";
 const rows=q=>snapshot.queries?.[q]?.rows??[];
 const sections=snapshot.reportContent??[];
 const monthly=rows("monthly_forecast").filter(r=>r.target===target&&r.population===activePopulation&&(model==="all"||r.variant===model)&&(month==="all"||String(r.month)===month));
 const leads=rows("lead_forecast").filter(r=>r.target===target&&r.population===activePopulation&&(model==="all"||r.variant===model));
 const costs=rows("cost_comparison").filter(r=>r.scenario===scenario);
 const dailyRows=rows("daily_primary").filter(r=>r.scenario===scenario&&(dailyMonth==="all"||String(r.month)===dailyMonth));
 const availableModels=[...new Set(rows("monthly_forecast").filter(r=>r.target===target).map(r=>r.variant))];
 const allowedTargets=["load",question==="2"||question==="4-2"?"pv":"pv_corrected",...(question.startsWith("4")?["price"]:[])];
 const axisLabel=metricLabels[metric]+(metric==="wape_pct"?"（%）":target==="price"?"（元/千瓦时）":"（千瓦）");
 const seedRows=rows("annual_forecast").filter(r=>r.target===target&&r.population===activePopulation&&(model==="all"||r.variant===model));
 const specifiedRows=q=>rows(q).filter(r=>r.scenario===scenario&&r.date===chosenDay);
 const caseRows=rows("failure_curves").filter(r=>r.variable===caseVariable);
 const prose=i=>sections[i-1]&&(sections[i-1].blocks??[{type:"prose",markdown:sections[i-1].markdown}]).map((block,j)=>block.type==="table"?<ReportSection key={block.queryId} id={block.queryId} queryId={block.queryId} title={block.title} sourceRows={block.rows} showHeading={false}><Table rows={block.rows} columns={block.columns}/></ReportSection>:<RichNarrative key={j} id={`experiment:section-${i}:prose-${j}`} value={block.markdown}/>);
 const table=(id,q,title,columns)=><DataComponent id={id} queryId={q} kind="table" title={title} sourceRows={rows(q)} displayRows={rows(q)}><Table rows={rows(q)} columns={columns}/></DataComponent>;
 const controls=<div className="experiment-controls"><Dropdown label="预测问题" showLabel value={question} choices={["2","3","4-2","4-3"]} onChange={q=>{setQuestion(q);setTarget("load");setModel("all");}}/><Dropdown label="预测变量" showLabel value={target} choices={allowedTargets} choiceLabels={targets} onChange={t=>{setTarget(t);setModel("all");}}/>{target.startsWith("pv")&&<Dropdown label="评价时段" showLabel value={population} choices={["all","generating"]} choiceLabels={{all:"全天",generating:"实际发电区间"}} onChange={setPopulation}/>}<Dropdown label="误差指标" showLabel value={metric} choices={Object.keys(metricLabels)} choiceLabels={metricLabels} onChange={setMetric}/><Dropdown label="模型" showLabel value={model} choices={["all",...availableModels]} choiceLabels={{all:"全部模型",...names}} onChange={setModel}/><Dropdown label="月份" showLabel value={month} choices={["all",...Array.from({length:11},(_,i)=>String(i+2))]} choiceLabels={{all:"全部月份"}} onChange={setMonth}/></div>;
 return <article className="report-content experiment-report" aria-label="神经网络预测与微网调度实验报告"><header className="report-hero"><h1 data-data-app-title>{appTitle}</h1></header>
 {!sections.length&&<RichNarrative id="experiment:preparation" value="## 数据已核验，正式训练正在进行\n\n三个实测数据表均覆盖 365 天、每天 144 个十分钟区间，数值区没有缺失。首轮 GPU 训练及模型保存读取验证已通过；全年成绩将在训练和调度回放完成后填入。"/>}
 {!!sections.length&&<nav className="experiment-nav" aria-label="报告章节">{sections.map((s,i)=><a key={s.title} href={`#chapter-${i+1}`}>{i+1}. {s.title}</a>)}</nav>}
 <section id="chapter-1">{prose(1)}{!!rows("summary").length&&<>{table("summary-table","summary","四个问题的正式结果",[["scenario_label","问题"],["total_cost","总费用（元）"],["emergency_kwh","紧急购电（千瓦时）"],["improvement_pct","较昨日基线节省（%）"],["weekly_improvement_pct","较周同期基线节省（%）"]])}<EvidenceChart id="primary-cost-overview" queryId="summary" title="正式策略的全年总费用" rows={rows("summary")} sourceRows={rows("summary")} height={280} spec={{type:"horizontalBar",stackable:false,x:"scenario_label",y:"total_cost",yLabel:"费用（元）",valueDecimals:2}}/></>}</section>
 <section id="chapter-2">{prose(2)}</section>
 <section id="chapter-3">{prose(3)}{table("data-audit","data_audit","数据覆盖与缺失检查",[["source","数据"],["days","天数"],["intervals","十分钟记录"],["issues","预报发布次数"],["missing","数值缺失"]])}</section>
 <section id="chapter-4">{prose(4)}{!!rows("technical_path").length&&<Route rows={rows("technical_path")}/>}</section>
 <section id="chapter-5">{prose(5)}</section>
 <section id="chapter-6">{!!rows("monthly_forecast").length&&<nav className="experiment-nav" aria-label="结果图表"><a href="#prediction-charts">预测误差图</a><a href="#dispatch-charts">调度费用图</a><a href="#specified-tables">指定日期明细</a></nav>}{prose(6)}{!!rows("monthly_forecast").length&&<>
 <div id="prediction-charts"/>
 <EvidenceChart id="forecast-monthly" queryId="monthly_forecast" title={`问题 ${question}：预测误差随月份的变化`} headerControls={controls} rows={monthly} sourceRows={monthly} height={340} spec={{type:"line",x:"month_label",y:metric,series:"model_label",stackable:false,xLabel:"评估月份",yLabel:axisLabel,valueDecimals:4}}/>
 <EvidenceChart id="forecast-lead" queryId="lead_forecast" title={`问题 ${question}：预测提前量与误差（全年）`} rows={leads} sourceRows={leads} height={320} spec={{type:"line",x:"lead",y:metric,series:"model_label",stackable:false,yLabel:axisLabel,valueDecimals:4}}/>
 <DataComponent id="seed-scores" queryId="annual_forecast" kind="table" title={`${targets[target]}：每个种子与平均预测的全年成绩`} sourceRows={seedRows} displayRows={seedRows}><Table rows={seedRows} columns={[["model_label","模型"],["seed","种子；mean 为平均预测"],["mae","平均绝对误差"],["rmse","均方根误差"],["wape_pct","加权绝对百分比误差（%）"]]}/></DataComponent>
 {table("seed-statistics","seed_statistics","三次训练成绩的均值与标准差",[["model_label","模型"],["target_label","预测目标"],["mae_mean","平均绝对误差均值"],["mae_std","平均绝对误差标准差"],["rmse_mean","均方根误差均值"],["rmse_std","均方根误差标准差"],["wape_mean","百分比误差均值"],["wape_std","百分比误差标准差"]])}
 <div id="dispatch-charts"/>
 <EvidenceChart id="cost-comparison" queryId="cost_comparison" title="同一调度规则下的费用对比" headerControls={<Dropdown label="问题" showLabel value={scenario} choices={["2","3","4-2","4-3"]} onChange={setScenario}/>} rows={costs} sourceRows={costs} height={390} spec={{type:"horizontalBar",stackable:false,x:"model_label",y:"total_cost",yLabel:"总费用（元）",valueDecimals:2}}/>
 <EvidenceChart id="daily-cost" queryId="daily_primary" title={`问题 ${scenario} 正式策略的每日费用${dailyMonth==="all"?"":`（${dailyMonth} 月）`}`} headerControls={<Dropdown label="费用月份" showLabel value={dailyMonth} choices={["all",...Array.from({length:11},(_,i)=>String(i+2))]} choiceLabels={{all:"全部月份"}} onChange={setDailyMonth}/>} rows={dailyRows} sourceRows={dailyRows} height={300} spec={{type:"line",x:"date",y:"total_cost",yLabel:"费用（元）",valueDecimals:2}}/>
 {table("update-comparison","update_comparison","增加日内预报是否划算",[["scenario","问题"],["update_schedule","预报时刻"],["total_cost","总费用（元）"],["emergency_kwh","紧急购电（千瓦时）"],["adjustment_cost","调整费（元）"]])}
 {!!caseRows.length&&<EvidenceChart id="failure-case" queryId="failure_curves" title={`高费用案例 ${caseRows[0].date}：预测与实际值`} headerControls={<Dropdown label="案例变量" showLabel value={caseVariable} choices={["load","pv","price"]} choiceLabels={targets} onChange={setCaseVariable}/>} rows={caseRows} sourceRows={caseRows} height={320} spec={{type:"line",x:"hour",y:"value",series:"series",stackable:false,xLabel:"当天时刻（小时）",yLabel:caseVariable==="price"?"元/千瓦时":"功率（千瓦）",valueDecimals:4}}/>}
 {table("correction-comparison","correction_comparison","是否修正附件光伏预报",[["scenario","问题"],["corrected","使用网络修正"],["total_cost","费用（元）"],["emergency_kwh","紧急购电（千瓦时）"]])}
 {table("seed-cost-statistics","seed_cost_statistics","三个种子的调度费用与波动",[["scenario","问题"],["model_label","模型"],["total_cost_mean","费用均值（元）"],["total_cost_std","费用标准差（元）"],["emergency_mean","紧急购电均值"],["emergency_std","紧急购电标准差"]])}
 <div id="specified-tables"/>
 {table("specified-days","specified_days","题目指定日期的结果",[["scenario","问题"],["date","日期"],["planned_kwh","计划购电（千瓦时）"],["final_kwh","最终购电（千瓦时）"],["total_cost","总费用（元）"],["emergency_kwh","紧急购电（千瓦时）"]])}
 <DataComponent id="specified-intervals" queryId="specified_intervals" kind="table" title={`问题 ${scenario}：指定十分钟区间购电`} headerControls={<Dropdown label="指定日期" showLabel value={chosenDay} choices={["2025-03-20","2025-06-21","2025-09-23","2025-12-21"]} onChange={setChosenDay}/>} sourceRows={specifiedRows("specified_intervals")} displayRows={specifiedRows("specified_intervals")}><Table rows={specifiedRows("specified_intervals")} columns={[["period","区间"],["original","凌晨计划（千瓦时）"],["final","最终购电（千瓦时）"]]}/></DataComponent>
 <DataComponent id="specified-battery" queryId="specified_battery" kind="table" title={`${chosenDay}：实际充放电与储电量`} sourceRows={specifiedRows("specified_battery")} displayRows={specifiedRows("specified_battery")}><Table rows={specifiedRows("specified_battery")} columns={[["period","四小时区间"],["charge_kwh","充电（千瓦时）"],["discharge_kwh","放电（千瓦时）"],["initial_soc","0:00 储电量"],["final_soc","24:00 储电量"]]}/></DataComponent>
 <DataComponent id="specified-emergency" queryId="specified_emergency" kind="table" title={`${chosenDay}：连续紧急购电区间`} sourceRows={specifiedRows("specified_emergency")} displayRows={specifiedRows("specified_emergency")}><Table rows={specifiedRows("specified_emergency")} columns={[["period","区间"],["emergency_kwh","紧急购电（千瓦时）"]]}/></DataComponent>
 </>}</section>
 <section id="chapter-7">{prose(7)}{!!rows("history").length&&table("history-comparison","history","历次实验核心指标与实现路径",[["experiment_id","实验"],["scenario","问题"],["total_cost","费用（元）"],["technical_path","技术实现"],["comparison","比较口径"]])}{!!rows("relative_history").length&&table("relative-history","relative_history","本次与此前全部实验的相对变化",[["previous_experiment","此前实验"],["task","问题或变量"],["route","技术路线"],["metric","指标"],["previous","此前值"],["current","本次值"],["relative_change_pct","变化（%）"],["comparison","可比性"]])}</section>
 <section id="chapter-8">{prose(8)}</section></article>;
}
