import React, { useState } from "react";
import { DataComponent, Dropdown, EvidenceChart, ReportSection, RichNarrative, useDataApp } from "../../data-app-public.jsx";
import "./report.css";
import { SoftPenaltyReport } from "./SoftPenaltyReport.jsx";

const fmt = v => typeof v === "number" ? (v !== 0 && Math.abs(v) < .00001 ? v.toExponential(3) : v.toLocaleString("zh-CN", { maximumFractionDigits: 6 })) : typeof v === "boolean" ? (v ? "通过" : "未通过") : v;
const plot = rows => rows.map(row => Object.fromEntries(Object.entries(row).map(([k,v]) => [k, typeof v === "number" ? Number(v.toFixed(4)) : v])));
function Table({ rows, columns }) {
 return <div className="failure-table-scroll"><table className="failure-table" data-reviewed-rows><thead><tr>{columns.map(([k,l]) => <th key={k}>{l}</th>)}</tr></thead><tbody>{rows.map((r,i) => <tr key={i}>{columns.map(([k]) => <td key={k} className={typeof r[k] === "number" ? "number" : undefined}>{fmt(r[k])}</td>)}</tr>)}</tbody></table></div>;
}
export function ReportContent() {
 const { snapshot } = useDataApp();
 return snapshot.variant === "soft-penalty" ? <SoftPenaltyReport/> : <StrictReportContent/>;
}
function StrictReportContent() {
 const { snapshot, appTitle, canEdit, mode, setAppTitle } = useDataApp();
 const q = id => snapshot.queries?.[id]?.rows ?? [];
 const [model, setModel] = useState("严格互斥MILP");
 const [date, setDate] = useState("2025-02-01");
 const [scope, setScope] = useState("局部窗口");
 const dates = q("daily").map(r => r.date);
 const activeDate = dates.includes(date) ? date : dates[0];
 const actual = q("actual").filter(r => r.date === activeDate);
 const win = q("window").filter(r => r.model === model && (scope === "完整窗口" || r.slot <= 99));
 const day = q("daily").filter(r => r.date === activeDate);
 const forecastLabels = ["exp001 午夜重评分", "exp002 午夜重评分", "exp003 正式", "exp004 无季节", "exp005 沿用同一冻结预测"];
 const controls = <div className="failure-controls"><Dropdown label="模型" showLabel value={model} choices={["严格互斥MILP","原连续LP"]} onChange={setModel}/><Dropdown label="规划范围" showLabel value={scope} choices={["局部窗口","完整窗口"]} onChange={setScope}/></div>;
 const dayControl = <Dropdown label="实际执行日期" showLabel value={activeDate} choices={dates} onChange={setDate}/>;
 const line = { type:"line", valueDecimals:4, stackable:false };
 const storage = { ...line,x:"hour",y:"charge_kwh",fields:["charge_kwh","discharge_kwh"],xLabel:"时刻 / 小时",yLabel:"kWh / 十分钟",colors:{charge_kwh:"#258a79",discharge_kwh:"#bf6548"},legend:{labels:{charge_kwh:"充电量",discharge_kwh:"放电量"}} };
 const table = (id,key,title,rs,cols) => <DataComponent id={id} queryId={key} kind="table" title={title} sourceRows={rs}><Table rows={rs} columns={cols}/></DataComponent>;
 return <article className="report-content failure-report" aria-label="exp005严格互斥重跑报告">
 <header className="report-hero"><h1 data-data-app-title contentEditable={canEdit && mode === "edit"} suppressContentEditableWarning onBlur={canEdit && mode === "edit" ? e => setAppTitle(e.currentTarget.textContent.trim() || appTitle) : undefined}>{appTitle}</h1></header>
 <nav className="failure-nav" aria-label="八节报告导航">{snapshot.sections.map((s,i) => <a key={i} href={`#failure-chapter-${i+1}`}>{i+1}. {s.title}</a>)}</nav>
 {snapshot.sections.map((s,i) => {
 const ids = s.queryIds ?? [];
 return <section id={`failure-chapter-${i+1}`} key={i}>
 <ReportSection id={`failure-section-${i+1}`} title={s.title} showHeading={false} queryId={ids[0]} queryIds={ids} sourceRowsByQuery={Object.fromEntries(ids.map(id=>[id,q(id)]))}><RichNarrative id={`failure-prose-${i+1}`} value={s.markdown}/></ReportSection>
 {i===0 && <>
 <EvidenceChart id="strict-period-comparison" queryId="period" title="同一已完成日期的实际总购电费" rows={plot(q("period"))} sourceRows={q("period")} spec={{type:"horizontalBar",x:"strategy",y:"total_wan",valueDecimals:4,xLabel:"万元",stackable:false}} height={260}/>
 {table("strict-period-costs","period","实际结算分项",q("period"),[["strategy","策略"],["days","天数"],["planned_cost","计划费 / 元"],["emergency_cost","紧急费 / 元"],["total_cost","总费 / 元"],["emergency_kwh","紧急电量 / kWh"]])}
 </>}
 {i===3 && <div className="strict-flow" aria-label="技术流程：冻结预测与历史误差进入信息树，午夜锁定普通购电计划，日内仅观测当前供需并严格互斥滚动求解，执行当前动作，更新状态，最终按实际量结算">
 {["冻结午夜预测", "已完成日误差构成信息树", "午夜两层互斥求解，锁定购电", "当前计量＋真实SOC／末段功率", "日内两层互斥求解，只执行当前动作", "SOC／功率跨段传递，按实际量结算"].map((text,index)=><div className="strict-flow-step" key={text}><span>{index+1}</span><p>{text}</p></div>)}
 </div>}
 {i===4 && <><EvidenceChart id="strict-solver-time" queryId="timing" title="每日已完成两层求解的累计耗时" rows={plot(q("timing"))} sourceRows={q("timing")} spec={{...line,x:"date",y:"solver_seconds",yLabel:"秒",xLabel:"日期"}} height={290}/>
 {table("strict-gap-table","gaps","最优性数值差距：同时查看绝对量与相对量",q("gaps"),[["stage","目标及单位"],["max_absolute_gap","最大绝对差"],["max_reported_relative_gap","最大相对gap"],["unknown_gap_count","未知条数"],["numerical_retries","数值复算次数"]])}</>}
 {i===5 && <>
 {q("stopped_stages").length > 0 && table("strict-stopped-stages","stopped_stages","本轮停止窗口：候选解未执行",q("stopped_stages"),[["date","日期"],["slot","段编号（从0起）"],["layer","层"],["status","求解状态码"],["seconds","秒"],["reported_relative_gap","相对gap"],["max_candidate_overlap_kwh","候选最大重叠 / kWh"]])}
 {q("stopped_prefix").length > 0 && <EvidenceChart id="strict-stopped-prefix" queryId="stopped_prefix" title="停止当日已经认证的执行前缀" rows={plot(q("stopped_prefix"))} sourceRows={q("stopped_prefix")} spec={storage} height={320}/>}
 {table("strict-same-input-table","same","原失败窗口：同输入对照",q("same"),[["model","模型"],["max_overlap_kwh","最大重叠 / kWh"],["tv_kw","净功率总变差 / kW"],["emergency_cost","预期紧急费 / 元"]])}
 <RichNarrative id="strict-precision-note" value="绘图数据取四位小数，来源记录保留原始精度。下图是相同输入的未来规划，随后日期筛选展示的是重新回放的实际动作。"/>
 <EvidenceChart id="strict-same-input-window" queryId="window" title={`${model}：原13:30窗口的第二层规划`} rows={plot(win)} sourceRows={win} spec={storage} height={340} headerControls={controls}/>
 {table("strict-selected-day-table","daily",`${activeDate}实际日费与状态`,day,[["date","日期"],["total_cost","总费 / 元"],["initial_soc","日初SOC / kWh"],["final_soc","日末SOC / kWh"],["tv_kw","实际总变差 / kW"]])}
 <EvidenceChart id="strict-actual-charge-discharge" queryId="actual" title={`${activeDate}实际充放电量`} rows={plot(actual)} sourceRows={actual} spec={storage} height={330} headerControls={dayControl}/>
 <EvidenceChart id="strict-actual-soc" queryId="actual" title={`${activeDate}实际区间末储电量`} rows={plot(actual.map(r=>({...r,end_hour:r.hour+1/6})))} sourceRows={actual} spec={{...line,x:"end_hour",y:"end_soc_kwh",xLabel:"区间结束时刻 / 小时",yLabel:"内部储电量 / kWh"}} height={290}/>
 {q("specified_soc").length > 0 && <>
 {table("strict-specified-plan","specified_plan","指定日期·表1：普通购电计划",q("specified_plan"),[["date","日期"],["interval","时间段"],["grid_kwh","购电量 / kWh"]])}
 {table("strict-specified-battery","specified_battery","指定日期·表2：四小时实际充放电",q("specified_battery"),[["date","日期"],["interval","时间段"],["charge_kwh","充电量 / kWh"],["discharge_kwh","放电量 / kWh"]])}
 {table("strict-specified-soc","specified_soc","指定日期：真实日初与日末储电量",q("specified_soc"),[["date","日期"],["initial_soc_kwh","0:00 SOC / kWh"],["final_soc_kwh","24:00 SOC / kWh"]])}
 {table("strict-specified-emergency","specified_emergency","指定日期·表3：实际紧急购电区间",q("specified_emergency"),[["date","日期"],["interval","时间段"],["emergency_kwh","紧急电量 / kWh"]])}
 </>}
 </>}
 {i===6 && <>
 {q("controls").length > 0 && table("strict-controls-table","controls","同一日期的规划与控制器对照",q("controls"),[["strategy","方案"],["days","共同天数"],["last_date","截至日期"],["planned_cost","计划费 / 元"],["emergency_cost","紧急费 / 元"],["total_cost","总费 / 元"]])}
 {table("strict-control-runs","control_runs","各模式真实完成范围",q("control_runs"),[["strategy","方案"],["completed_days","完整日"],["last_date","最后日期"],["status","运行状态"],["numerical_retries","数值复算次数"]])}
 <EvidenceChart id="strict-daily-costs" queryId="daily" title="exp004与本轮的同日期日费" rows={plot(q("daily"))} sourceRows={q("daily")} spec={{...line,x:"day_number",y:"total_cost",fields:["baseline_cost","total_cost"],xLabel:"从2月1日起的已完成日序号",yLabel:"元 / 日",legend:{labels:{baseline_cost:"exp004 无季节·原调度器",total_cost:"exp005 严格互斥·滚动规划"}}}} height={330}/>
 <EvidenceChart id="strict-cumulative-cost-difference" queryId="daily" title="累计费用差：正值表示本轮更贵" rows={plot(q("daily"))} sourceRows={q("daily")} spec={{...line,x:"day_number",y:"cumulative_difference",xLabel:"从2月1日起的已完成日序号",yLabel:"本轮减exp004 / 元"}} height={290}/>
 {table("strict-history-table","history","保留历史原值，逐项标明评价期",q("history"),[["label","实验"],["period","评价期"],["total_cost","总费 / 元"],["status","可比性说明"]])}
 {[['load','负载'],['pv','光伏'],['net_load','净负载']].map(([target,label])=>{
 const selected = q("forecast_history").filter(r=>r.target===target && forecastLabels.includes(r.label));
 return <EvidenceChart key={target} id={`strict-forecast-${target}`} queryId="forecast_history" title={`${label}：334日午夜预测平均绝对误差`} rows={plot(selected)} sourceRows={selected} spec={{type:"horizontalBar",x:"label",y:"mae",xLabel:"MAE / kW",valueDecimals:4,stackable:false}} height={370}/>;
 })}
 {table("strict-forecast-history-table","forecast_history","预测误差完整历史值：全部334日",q("forecast_history"),[["label","实验"],["target","目标"],["n","样本数"],["mae","MAE / kW"],["rmse","RMSE / kW"],["wape_pct","WAPE / %"]])}
 {table("strict-timing-history-table","timing_history","各阶段历史耗时：保留原计时范围",q("timing_history"),[["label","实验"],["stage","阶段"],["seconds","秒"],["groups","组／日数"],["scope","范围"]])}
 </>}
 {i===7 && <>{table("strict-audit-table","audit","独立物理、结算与原源码核验",q("audit"),[["check","核验项"],["value","实测值"],["limit","上限"],["passed","结果"]])}
 {table("strict-controls-audit","control_audit","三个模式各自已完成日期的物理核验",q("control_audit"),[["strategy","方案"],["completed_days","完整日"],["max_actual_overlap_kwh","实际最大重叠 / kWh"],["max_node_overlap_kwh","节点最大重叠 / kWh"],["max_ramp_kw","最大功率变化 / kW"],["passed","结果"]])}</>}
 </section>;
 })}
 </article>;
}
