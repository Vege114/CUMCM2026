import React, {useState} from "react";
import {DataComponent, Dropdown, EvidenceChart, ReportSection, RichNarrative, useDataApp} from "../../data-app-public.jsx";
import "./report.css";
const targets={load:"负载",pv:"光伏",net_load:"净负载"};
const metrics={mae:"MAE",rmse:"RMSE",wape_pct:"WAPE",bias:"平均偏差"};
const format=value=>typeof value==="boolean"?(value?"是":"否"):typeof value==="number"?value.toLocaleString("zh-CN",{maximumFractionDigits:4}):value??"—";
function Table({rows,columns}){return <div className="q2-table-scroll"><table className="q2-table" data-reviewed-rows><thead><tr>{columns.map(([key,label])=><th key={key}>{label}</th>)}</tr></thead><tbody>{rows.map((row,i)=><tr key={i}>{columns.map(([key])=><td key={key} className={typeof row[key]==="number"?"q2-number":undefined}>{format(row[key])}</td>)}</tr>)}</tbody></table></div>;}
function aggregate(rows,keys,kind){
 const groups=new Map(),fields=kind==="forecast"?["n","absolute_error_sum","squared_error_sum","signed_error_sum","actual_abs_sum"]:["planned_cost","emergency_cost","total_cost","emergency_kwh","solve_execute_seconds"];
 for(const row of rows){const id=JSON.stringify(keys.map(key=>row[key]));if(!groups.has(id))groups.set(id,Object.fromEntries(keys.map(key=>[key,row[key]])));const group=groups.get(id);for(const field of fields)group[field]=(group[field]??0)+(row[field]??0);}
 return [...groups.values()].map(row=>kind==="forecast"?{...row,mae:row.n?row.absolute_error_sum/row.n:null,rmse:row.n?Math.sqrt(row.squared_error_sum/row.n):null,bias:row.n?row.signed_error_sum/row.n:null,wape_pct:row.actual_abs_sum?100*row.absolute_error_sum/row.actual_abs_sum:null}:{...row,planned_wan:row.planned_cost/10000,emergency_wan:row.emergency_cost/10000,total_wan:row.total_cost/10000});
}
export function ReportContent(){
 const {snapshot,appTitle,canEdit,mode,setAppTitle}=useDataApp();
 const [month,setMonth]=useState("all"),[target,setTarget]=useState("net_load"),[metric,setMetric]=useState("rmse"),[population,setPopulation]=useState("all"),[seed,setSeed]=useState("42"),[day,setDay]=useState("2025-03-20");
 const query=id=>snapshot.queries?.[id]?.rows??[];
 const months=[...new Set(query("forecast_monthly").map(row=>String(row.month)))].sort((a,b)=>Number(a)-Number(b));
 const seedChoices=[...new Set(query("forecast_monthly").filter(row=>row.experiment==="exp003").map(row=>String(row.seed)))].sort((a,b)=>Number(a)-Number(b));
 const inMonth=row=>month==="all"||String(row.month)===month;
 const inSeed=row=>row.experiment!=="exp003"||row.name!=="primary"||seed==="all"||String(row.seed)===seed;
 const forecasts=query("forecast_monthly").filter(row=>row.target===target&&row.population===population&&inMonth(row)&&inSeed(row));
 const forecastControls=<div className="q2-controls"><Dropdown label="预测目标" showLabel value={target} choices={Object.keys(targets)} choiceLabels={targets} onChange={setTarget}/><Dropdown label="误差指标" showLabel value={metric} choices={Object.keys(metrics)} choiceLabels={metrics} onChange={setMetric}/><Dropdown label="评价时段" showLabel value={population} choices={["all","pv_generating"]} choiceLabels={{all:"全天",pv_generating:"实际PV > 0"}} onChange={setPopulation}/>{seedChoices.length>1&&<Dropdown label="新模型种子" showLabel value={seed} choices={[...seedChoices,"all"]} choiceLabels={{all:"分别展示全部"}} onChange={setSeed}/>}</div>;
 function block(item){
  if(item.type==="routes")return <ReportSection key={item.id} id={item.id} title={item.title} queryId={item.queryId} sourceRows={query(item.queryId)}><div className="q2-routes">{query(item.queryId).map(row=><div className="q2-route" key={row.route}><strong>{row.route}</strong><div className="q2-route-steps">{row.steps.map((step,i)=><React.Fragment key={step}>{i>0&&<span aria-hidden="true" className="q2-route-arrow">→</span>}<span className="q2-route-step">{step}</span></React.Fragment>)}</div></div>)}</div></ReportSection>;
  if(item.type==="prose"){
   const prose=<RichNarrative id={item.id} value={item.markdown}/>;
   if(!item.queryIds?.length)return <React.Fragment key={item.id}>{prose}</React.Fragment>;
   const sources=Object.fromEntries(item.queryIds.map(id=>[id,query(id)]));
   return <ReportSection key={item.id} id={`${item.id}-evidence`} title={item.title??"说明依据"} queryId={item.queryIds[0]} queryIds={item.queryIds} sourceRowsByQuery={sources} showHeading={false}>{prose}</ReportSection>;
  }
  let rows=query(item.queryId),sourceRows=rows,spec={...item.spec},title=item.title,controls=null;
  if(item.scope==="cost-filtered"){
   sourceRows=rows.filter(row=>inMonth(row)&&row.seed===42);
   rows=aggregate(sourceRows,["policy_id","policy_label","order"],"cost").sort((a,b)=>a.order-b.order);
   title+=` · ${month==="all"?"2—12月":`${month}月`}`;
  }else if(item.scope==="monthly-cost"){
   sourceRows=rows.filter(row=>row.seed===42&&inMonth(row));
   rows=aggregate(sourceRows,["month","month_label","policy_id","policy_label","order"],"cost").sort((a,b)=>a.month-b.month||a.order-b.order);
  }else if(item.scope?.startsWith("forecast")){
   sourceRows=forecasts;
   rows=aggregate(sourceRows,item.scope==="forecast-monthly"?["month","month_label","policy_label","predictor_id","order"]:["policy_label","predictor_id","order"],"forecast").sort((a,b)=>(a.month??0)-(b.month??0)||a.order-b.order);
   spec.y=metric;spec.yLabel=`${metrics[metric]}（${metric==="wape_pct"?"%":"kW"}）`;
   title+=` · ${targets[target]} · ${population==="all"?"全天":"实际PV > 0"}`;
   if(item.scope==="forecast-monthly")controls=forecastControls;
  }else if(item.scope==="specified"){
   sourceRows=rows.filter(row=>row.date===day);rows=sourceRows;title+=` · ${day}`;
   if(item.id==="specified-grid")controls=<Dropdown label="指定日期" showLabel value={day} choices={snapshot.specifiedDates??[]} onChange={setDay}/>;
  }
  if(!rows.length)return null;
  if(item.type==="chart")return <EvidenceChart key={item.id} id={item.id} queryId={item.queryId} title={title} rows={rows} sourceRows={sourceRows} spec={spec} headerControls={controls} height={item.height??330}/>;
  const content=<DataComponent id={item.id} queryId={item.queryId} kind="table" title={title} sourceRows={sourceRows} displayRows={rows} headerControls={controls}><Table rows={rows} columns={item.columns}/></DataComponent>;
  return item.disclosure?<details key={item.id} className="q2-disclosure"><summary>{item.title}</summary>{content}</details>:<React.Fragment key={item.id}>{content}</React.Fragment>;
 }
 return <article className="report-content q2-report" aria-label="第二问优化实验报告">
  <header className="report-hero"><h1 data-data-app-title contentEditable={canEdit&&mode==="edit"} suppressContentEditableWarning onBlur={canEdit&&mode==="edit"?event=>setAppTitle(event.currentTarget.textContent.trim()||appTitle):undefined}>{appTitle}</h1></header>
  <nav className="q2-nav" aria-label="报告章节">{(snapshot.reportContent??[]).map((section,i)=><a key={section.title} href={`#chapter-${i+1}`}>{i+1}. {section.title}</a>)}</nav>
  <div className="q2-controls"><Dropdown label="结果月份" showLabel value={month} choices={["all",...months]} choiceLabels={Object.fromEntries([["all","全部2—12月"],...months.map(m=>[m,`${m}月`])])} onChange={setMonth}/><span className="q2-filter-note">筛选费用与预测图；结论及历史明细固定全年。</span></div>
  {(snapshot.reportContent??[]).map((section,i)=><section id={`chapter-${i+1}`} key={section.title}>{section.blocks.map(block)}</section>)}
 </article>;
}
