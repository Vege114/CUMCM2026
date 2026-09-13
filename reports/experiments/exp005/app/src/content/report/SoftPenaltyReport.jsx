import React, { useState } from "react";
import { DataComponent, Dropdown, EvidenceChart, ReportSection, RichNarrative, useDataApp } from "../../data-app-public.jsx";

const fmt = v => typeof v === "number" ? (v !== 0 && Math.abs(v) < .00001 ? v.toExponential(3) : v.toLocaleString("zh-CN", { maximumFractionDigits: 6 })) : typeof v === "boolean" ? (v ? "通过" : "未通过") : v;
const plot = rows => rows.map(r=>Object.fromEntries(Object.entries(r).map(([k,v])=>[k,typeof v==="number"?Number(v.toFixed(6)):v])));
function Table({rows,columns}) {return <div className="failure-table-scroll"><table className="failure-table" data-reviewed-rows><thead><tr>{columns.map(([k,l])=><th key={k}>{l}</th>)}</tr></thead><tbody>{rows.map((r,i)=><tr key={i}>{columns.map(([k])=><td key={k} className={typeof r[k]==="number"?"number":undefined}>{fmt(r[k])}</td>)}</tr>)}</tbody></table></div>;}

export function SoftPenaltyReport() {
 const {snapshot,appTitle,canEdit,mode,setAppTitle}=useDataApp();
 const q=id=>snapshot.queries?.[id]?.rows ?? [];
 const [beta,setBeta]=useState("0.1"), [date,setDate]=useState("2025-02-01"), [caseBeta,setCaseBeta]=useState("0"), [target,setTarget]=useState("net_load");
 const betas=q("runs").map(r=>String(r.beta)), activeBeta=betas.includes(beta)?beta:betas[0];
 const chosen=id=>q(id).filter(r=>r.beta===Number(activeBeta));
 const dates=chosen("daily").map(r=>r.date), activeDate=dates.includes(date)?date:dates[0];
 const actual=chosen("actual").filter(r=>r.date===activeDate), day=chosen("daily").filter(r=>r.date===activeDate);
 const heat=chosen("overlap_heatmap"), curves=q("preflight_curves").filter(r=>r.beta===Number(caseBeta)&&r.hour<=16.5);
 const betaControl=<Dropdown label="周转权重" showLabel value={activeBeta} choices={betas} onChange={setBeta}/>;
 const dayControl=<div className="failure-controls">{betaControl}<Dropdown label="模拟执行日期" showLabel value={activeDate} choices={dates} onChange={setDate}/></div>;
 const targetControl=<Dropdown label="预测目标" showLabel value={target} choices={["load","pv","net_load"]} onChange={setTarget}/>;
 const line={type:"line",valueDecimals:6,stackable:false};
 const storage={...line,x:"hour",y:"charge_kwh",fields:["charge_kwh","discharge_kwh"],xLabel:"时刻 / 小时",yLabel:"kWh / 十分钟",colors:{charge_kwh:"#287f8e",discharge_kwh:"#b75c3c"},legend:{labels:{charge_kwh:"充电量",discharge_kwh:"放电量"}}};
 const table=(id,key,title,rs,cols)=><DataComponent id={id} queryId={key} kind="table" title={title} sourceRows={rs}><Table rows={rs} columns={cols}/></DataComponent>;
 const chart=(id,key,title,rs,spec,height=320,control=null)=><EvidenceChart id={id} queryId={key} title={title} rows={plot(rs)} sourceRows={rs} spec={spec} height={height} headerControls={control}/>;
 const forecastLabels=["exp001 午夜重评分","exp002 午夜重评分","exp003 正式","exp004 无季节","exp005 周转惩罚·同一冻结预测"];
 const forecast=q("forecast_history").filter(r=>r.target===target&&forecastLabels.includes(r.label));
 return <article className="report-content failure-report" aria-label="exp005周转惩罚与重叠检验报告">
 <header className="report-hero"><h1 data-data-app-title contentEditable={canEdit&&mode==="edit"} suppressContentEditableWarning onBlur={canEdit&&mode==="edit"?e=>setAppTitle(e.currentTarget.textContent.trim()||appTitle):undefined}>{appTitle}</h1></header>
 <nav className="failure-nav" aria-label="八节报告导航">{snapshot.sections.map((s,i)=><a key={i} href={`#failure-chapter-${i+1}`}>{i+1}. {s.title}</a>)}</nav>
 {snapshot.sections.map((s,i)=>{const ids=s.queryIds??[];return <section id={`failure-chapter-${i+1}`} key={i}>
 <ReportSection id={`failure-section-${i+1}`} title={s.title} showHeading={false} queryId={ids[0]} queryIds={ids} sourceRowsByQuery={Object.fromEntries(ids.map(id=>[id,q(id)]))}><RichNarrative id={`failure-prose-${i+1}`} value={s.markdown}/></ReportSection>
 {i===0&&<>
 {chart("soft-period-comparison","period","主方案与exp004：同一完成日期的模拟结算",q("period").filter(r=>r.beta===null||r.beta===.1),{type:"horizontalBar",x:"label",y:"total_wan",xLabel:"万元",valueDecimals:4,stackable:false},280)}
 {table("soft-period-table","period","费用分项及各组覆盖天数",q("period"),[["label","方案"],["days","完整日"],["planned_cost","计划费 / 元"],["emergency_cost","紧急费 / 元"],["adjustment_cost","调整费 / 元"],["total_cost","总费 / 元"],["emergency_kwh","紧急电量 / kWh"]])}
 </>}
 {i===3&&<div className="strict-flow" aria-label="冻结预测和前缀信息树，午夜两层LP，锁定计划，当前反馈下两层滚动LP，记录重叠，只执行当前动作并结算">{["固定exp004午夜预测","历史联合误差与前缀树","午夜费用层＋周转平稳层","锁定普通计划，观测当前供需","连续LP滚动求解，记录全部重叠","只执行当前动作，状态跨日并结算"].map((t,j)=><div className="strict-flow-step" key={t}><span>{j+1}</span><p>{t}</p></div>)}</div>}
 {i===4&&<>
 {table("soft-run-table","runs","两组回放范围与实测时间",q("runs"),[["beta","β"],["completed_days","完整日"],["last_date","最后日期"],["status","状态"],["wall_seconds","回放墙钟 / 秒"],["solver_seconds","求解器累计 / 秒"]])}
 {chart("soft-solver-time","daily",`β=${activeBeta}每日两层求解累计耗时`,chosen("daily"),{...line,x:"date",y:"solver_seconds",yLabel:"秒"},290,betaControl)}
 </>}
 {i===5&&<>
 {table("soft-preflight-table","preflight","同一失败输入：重叠、平稳性与周转量共同观察",q("preflight"),[["label","权重"],["max_overlap_kwh","最大重叠 / kWh"],["expected_tv_kw","总变差 / kW"],["expected_throughput_kwh","周转量 / kWh"],["second_cost","剩余预期紧急费 / 元"]])}
 {chart("soft-preflight-curves","preflight_curves",`原13:30窗口的第二层曲线：β=${caseBeta}`,curves,storage,330,<Dropdown label="同输入惩罚权重" showLabel value={caseBeta} choices={["0","0.01","0.1"]} onChange={setCaseBeta}/>)}
 {table("soft-overlap-scopes","overlap_summary","两层规划重叠统计：节点包含重复预测",q("overlap_summary"),[["beta","β"],["scope_label","范围"],["calls","窗口数"],["nodes","节点数"],["overlap_nodes","重叠节点"],["windows_with_overlap","重叠窗口"],["max_overlap_kwh","原始最大重叠 / kWh"]])}
 {chart("soft-overlap-timeline","overlap_timeline",`β=${activeBeta}每日规划节点最大重叠`,chosen("overlap_timeline"),{...line,x:"date",y:"max_overlap_kwh",series:"label",yLabel:"原始最大重叠 / kWh"},310,betaControl)}
 {chart("soft-actual-overlap","daily",`β=${activeBeta}实际模拟执行的重叠区间数`,chosen("daily"),{type:"bar",x:"date",y:"overlap_intervals",yLabel:"十分钟区间数",stackable:false,valueDecimals:0},280,betaControl)}
 {chart("soft-overlap-hours","overlap_heatmap",`β=${activeBeta}实际重叠频率：月份与时钟小时`,heat,{type:"heatmap",x:"hour_label",y:"overlapRate",series:"month",colorDomain:[0,Math.max(.01,...heat.map(r=>r.overlapRate))],showValues:false,missingValues:"gap",tooltipFields:[{field:"intervals",label:"已完成区间数"},{field:"overlap_intervals",label:"重叠区间数"}]},350,betaControl)}
 {table("soft-monthly-table","monthly","月度模拟结算及重叠计数",chosen("monthly"),[["month","月份"],["days","天数"],["total_cost","总费 / 元"],["emergency_cost","紧急费 / 元"],["overlap_intervals","重叠段数"],["throughput_kwh","周转量 / kWh"]])}
 {table("soft-selected-day-table","daily",`${activeDate}模拟日费与状态`,day,[["date","日期"],["total_cost","总费 / 元"],["initial_soc","日初SOC / kWh"],["final_soc","日末SOC / kWh"],["overlap_intervals","重叠段数"],["tv_kw","总变差 / kW"]])}
 {chart("soft-actual-charge-discharge","actual",`β=${activeBeta} · ${activeDate}实际模拟充放电`,actual,storage,330,dayControl)}
 {chart("soft-actual-soc","actual",`${activeDate}实际模拟区间末储电量`,actual,{...line,x:"hour",y:"end_soc_kwh",xLabel:"区间起始时刻 / 小时",yLabel:"区间末SOC / kWh"},280)}
 {table("soft-node-events","overlap_nodes","全部超过阈值的规划重叠事件",q("overlap_nodes"),[["beta","β"],["date","日期"],["scope","范围"],["replay_slot","当前段（从0起）"],["future_slot","重叠所在段"],["layer","层"],["probability","概率"],["charge_kwh","充电 / kWh"],["discharge_kwh","放电 / kWh"],["overlap_kwh","重叠 / kWh"]])}
 {q("control_residual_case").length>0&&chart("soft-control-residual","control_residual_case",`β=0.01 · ${q("control_residual_case")[0].date}：残留重叠的未来规划路径`,q("control_residual_case"),{...storage,fields:["charge_kwh","discharge_kwh","overlap_kwh"],legend:{labels:{charge_kwh:"规划充电",discharge_kwh:"规划放电",overlap_kwh:"重叠min(c,d)"}}},350)}
 {q("specified_soc").length>0&&<>
 {table("soft-specified-plan","specified_plan","指定日期·表1：普通购电计划",q("specified_plan"),[["date","日期"],["interval","时间段"],["grid_kwh","购电量 / kWh"]])}
 {table("soft-specified-battery","specified_battery","指定日期·表2：四小时充放电",q("specified_battery"),[["date","日期"],["interval","时间段"],["charge_kwh","充电量 / kWh"],["discharge_kwh","放电量 / kWh"]])}
 {table("soft-specified-soc","specified_soc","指定日期：日初与日末储电量",q("specified_soc"),[["date","日期"],["initial_soc_kwh","0:00 SOC / kWh"],["final_soc_kwh","24:00 SOC / kWh"]])}
 {table("soft-specified-emergency","specified_emergency","指定日期·表3：紧急购电",q("specified_emergency"),[["date","日期"],["interval","时间段"],["emergency_kwh","紧急电量 / kWh"]])}
 </>}
 </>}
 {i===6&&<>
 {table("soft-paired-beta","paired_beta","两组相同完成日期的敏感性对照",q("paired_beta"),[["label","方案"],["days","共同日数"],["total_cost","总费 / 元"],["emergency_cost","紧急费 / 元"],["throughput_kwh","周转量 / kWh"],["tv_kw","总变差 / kW"],["overlap_intervals","实际重叠段数"]])}
 {chart("soft-daily-costs","daily",`β=${activeBeta}与exp004同日期日费`,chosen("daily"),{...line,x:"date",y:"total_cost",fields:["baseline_cost","total_cost"],yLabel:"元 / 日",legend:{labels:{baseline_cost:"exp004原调度",total_cost:`本轮β=${activeBeta}`}}},320,betaControl)}
 {chart("soft-cumulative-difference","daily",`β=${activeBeta}累计费用差：正值表示本轮更贵`,chosen("daily"),{...line,x:"date",y:"cumulative_difference",yLabel:"本轮减exp004 / 元"},290)}
 {table("strict-history-table","history","历史费用原值：不同评价期不排名",q("history"),[["label","实验"],["period","评价期"],["total_cost","总费 / 元"],["status","可比性"]])}
 {chart("soft-history-cost","history","历史334日费用：保留各控制器口径",q("history").filter(r=>["exp001 同口径重算","exp002 正式风险","exp003 正式","exp004 无季节"].includes(r.label)||(r.label.startsWith("exp005 周转惩罚")&&r.period.includes("334天"))),{type:"horizontalBar",x:"label",y:"total_cost",xLabel:"元",valueDecimals:2,stackable:false},420)}
 {table("soft-technical-comparison","technical_comparison","预测相同，规划与执行路线的变化",q("technical_comparison"),[["scheme","方案"],["forecast","预测"],["midnight","午夜规划"],["execution","日内执行"],["ramp","爬坡"],["overlap","充放电重叠处理"]])}
 <DataComponent id="soft-route-comparison" queryId="technical_comparison" kind="table" title="exp004与本轮：从输入到执行的并列流程" sourceRows={q("technical_comparison").filter(r=>!r.scheme.includes("旧版"))}><div className="soft-route-comparison" data-reviewed-rows>{q("technical_comparison").filter(r=>!r.scheme.includes("旧版")).map(r=><div key={r.scheme}><h3>{r.scheme}</h3>{[r.forecast,r.midnight,r.execution,r.ramp,r.overlap].map((text,index)=><React.Fragment key={text}>{index>0&&<span aria-hidden="true">↓</span>}<p>{text}</p></React.Fragment>)}</div>)}</div></DataComponent>
 {chart("soft-forecast-wape","forecast_history",`${target}：334日午夜预测WAPE`,forecast,{type:"horizontalBar",x:"label",y:"wape_pct",xLabel:"WAPE / %",valueDecimals:4,stackable:false},360,targetControl)}
 {chart("soft-forecast-rmse","forecast_history",`${target}：334日午夜预测RMSE`,forecast,{type:"horizontalBar",x:"label",y:"rmse",xLabel:"RMSE / kW",valueDecimals:4,stackable:false},360,targetControl)}
 {table("strict-forecast-history-table","forecast_history","预测历史完整值",q("forecast_history"),[["label","实验"],["target","目标"],["n","样本数"],["mae","MAE / kW"],["rmse","RMSE / kW"],["wape_pct","WAPE / %"]])}
 {chart("soft-forecast-monthly","forecast_breakdown",`${target}分月预测WAPE`,q("forecast_breakdown").filter(r=>r.target===target&&r.population==="all"&&r.month>0),{...line,x:"month",y:"wape_pct",xLabel:"2025年月",yLabel:"WAPE / %"},290,targetControl)}
 {table("soft-forecast-breakdown","forecast_breakdown","分月与实际发电时段预测误差",q("forecast_breakdown"),[["target","目标"],["month","月份（0为全年）"],["population","样本范围"],["n","样本数"],["mae","MAE / kW"],["rmse","RMSE / kW"],["wape_pct","WAPE / %"]])}
 {table("strict-timing-history-table","timing_history","各阶段历史耗时：保留原统计范围",q("timing_history"),[["label","实验"],["stage","阶段"],["seconds","秒"],["groups","组／日数"],["scope","计时范围"]])}
 {chart("soft-history-timing","timing_plot","历史阶段耗时：设备、组数和范围不同",q("timing_plot"),{type:"horizontalBar",x:"item",y:"seconds",xLabel:"秒",valueDecimals:3,stackable:false},740)}
 </>}
 {i===7&&table("soft-audit-table","audit","约束与结算核验：互斥作为诊断另表统计",q("audit"),[["beta","β"],["check_label","核验项"],["value","实测值"],["limit","验收上限"],["passed","结果"]])}
 </section>;})}
 </article>;
}
