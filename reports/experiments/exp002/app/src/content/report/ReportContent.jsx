import React, {useState} from "react";
import {DataComponent, Dropdown, EvidenceChart, RichNarrative, useDataApp} from "../../data-app-public.jsx";
import "./report.css";

const labels={load:"小区负载",pv:"历史光伏预测",pv_corrected:"光伏预报修正",price:"电价",all:"全部"};
const metrics={mae:"平均绝对误差",rmse:"均方根误差",wape_pct:"加权绝对百分比误差"};
const format=v=>typeof v==="number"?v.toLocaleString("zh-CN",{maximumFractionDigits:4}):typeof v==="boolean"?(v?"是":"否"):v??"—";
function Table({rows,columns}) {return <div className="experiment-table-scroll"><table className="experiment-table" data-reviewed-rows><thead><tr>{columns.map(([key,label])=><th key={key}>{label}</th>)}</tr></thead><tbody>{rows.map((row,i)=><tr key={i}>{columns.map(([key])=><td key={key}>{format(row[key])}</td>)}</tr>)}</tbody></table></div>;}
function aggregate(rows,keys,kind){const map=new Map();for(const row of rows){const key=JSON.stringify(keys.map(k=>row[k]));if(!map.has(key))map.set(key,Object.fromEntries(keys.map(k=>[k,row[k]])));const value=map.get(key);for(const field of kind==="forecast"?["n","absolute_error_sum","squared_error_sum","actual_abs_sum"]:["total_cost","planned_cost","up_cost","down_cost","emergency_cost","emergency_kwh","solve_execute_seconds","fallback_count","timeout_count","solver_calls"]){value[field]=(value[field]??0)+(row[field]??0);}}return [...map.values()].map(row=>kind==="forecast"?{...row,mae:row.absolute_error_sum/row.n,rmse:Math.sqrt(row.squared_error_sum/row.n),wape_pct:row.actual_abs_sum?100*row.absolute_error_sum/row.actual_abs_sum:null}:row);}

function Route({rows}){const [step,setStep]=useState("1");const active=rows.find(r=>String(r.step)===step)??rows[0];return <DataComponent id="technical-route" queryId="technical_path" kind="custom" title="从历史数据到实际购电费用" sourceRows={rows} displayRows={rows} headerControls={<Dropdown label="逐步讲解" showLabel value={step} choices={rows.map(r=>String(r.step))} choiceLabels={Object.fromEntries(rows.map(r=>[String(r.step),`${r.step}. ${r.title}`]))} onChange={setStep}/>}><ol className="experiment-route">{rows.map(row=><li key={row.step} className={row===active?"active-step":""}><span>{row.step}</span>{row.title}</li>)}</ol>{active&&<div className="experiment-route-detail" aria-live="polite"><p><strong>输入：</strong>{active.input}</p><p><strong>计算：</strong>{active.operation}</p><p><strong>输出：</strong>{active.output}</p></div>}</DataComponent>;}

function ScenarioTree({rows}){const [reveal,setReveal]=useState("0");const shown=rows.filter(r=>r.hour<=Number(reveal));const members=r=>String(r.members).split(",").map(Number);const position=r=>[55+r.hour/6*215,30+r.layout_y*48];return <DataComponent id="scenario-tree" queryId="tree_nodes" kind="custom" title="问题 4-3 · 3 月 20 日：场景树逐步揭示信息" sourceRows={rows} displayRows={shown} headerControls={<Dropdown label="已揭示到" showLabel value={reveal} choices={["0","6","12","18"]} choiceLabels={{0:"凌晨：一份共同信息",6:"6 时：首次更新",12:"12 时：第二次更新",18:"18 时：末段仍有误差"}} onChange={setReveal}/>}><div className="experiment-tree-scroll"><svg viewBox="0 0 880 440" role="img" aria-label={`场景树：已揭示到 ${reveal} 时；未来节点隐藏`} data-reviewed-rows>{shown.map(node=>{const [x,y]=position(node),child=members(node);const parent=rows.find(r=>r.hour===node.hour-6&&child.every(m=>members(r).includes(m)));const a=parent?position(parent):null;return <g key={`${node.hour}-${node.node}`}>{a&&<path d={`M${a[0]},${a[1]} L${x},${y}`} fill="none" stroke="var(--border)" strokeWidth="2"/>}<circle cx={x} cy={y} r="7" fill="var(--chart-1)"/><text x={x+12} y={y+4} fill="var(--text)" fontSize="12">节点 {node.node} · {node.path_count} 条路径</text></g>;})}{[0,6,12,18].map(h=><text key={h} x={55+h/6*215} y="425" fill="var(--secondary)" fontSize="13">{h} 时</text>)}</svg></div><p>只有已实现的前缀和新发布预报能够触发分支；同节点共享购电决策。每个末段节点仍保留未揭示的条件误差。</p></DataComponent>;}

export function ReportContent(){
 const {snapshot,appTitle}=useDataApp();
 const [scenario,setScenario]=useState("2"),[month,setMonth]=useState("all"),[target,setTarget]=useState("load"),[seed,setSeed]=useState("42"),[metric,setMetric]=useState("mae"),[population,setPopulation]=useState("all"),[chosenDay,setChosenDay]=useState("2025-03-20");
 const query=id=>snapshot.queries?.[id]?.rows??[];
 const changeScenario=value=>{setScenario(value);if(target==="price"&&!value.startsWith("4"))setTarget("load");else if(target.startsWith("pv"))setTarget(value==="2"||value==="4-2"?"pv":"pv_corrected");};
 const targets=["load",scenario==="2"||scenario==="4-2"?"pv":"pv_corrected",...(scenario.startsWith("4")?["price"]:[])];
 const selectedMonths=row=>month==="all"||String(row.month)===month;
 const forecastRows=query("forecast").filter(row=>row.target===target&&row.population===(target.startsWith("pv")?population:"all")&&selectedMonths(row)&&(row.seed==="none"||seed==="all"||row.seed===seed));
 const costRows=query("cost_daily").filter(row=>row.scenario===scenario&&selectedMonths(row)&&row.seed===42);
 const controls=<div className="experiment-controls"><Dropdown label="问题" showLabel value={scenario} choices={["2","3","4-2","4-3"]} onChange={changeScenario}/><Dropdown allLabel="全部" label="月份" showLabel value={month} choices={["all",...Array.from({length:11},(_,i)=>String(i+2))]} choiceLabels={labels} onChange={setMonth}/></div>;
 const forecastControls=<div className="experiment-controls"><Dropdown label="预测变量" showLabel value={target} choices={targets} choiceLabels={labels} onChange={setTarget}/><Dropdown label="误差指标" showLabel value={metric} choices={Object.keys(metrics)} choiceLabels={metrics} onChange={setMetric}/><Dropdown allLabel="全部" label="网络种子" showLabel value={seed} choices={["42","2026","3407","all"]} choiceLabels={labels} onChange={setSeed}/>{target.startsWith("pv")&&<Dropdown allLabel="全天" label="评价时段" showLabel value={population} choices={["all","generating"]} choiceLabels={{all:"全天",generating:"实际发电区间"}} onChange={setPopulation}/>}</div>;
 function block(item){
  if(item.type==="prose")return <RichNarrative key={item.id} id={item.id} value={item.markdown}/>;
  if(item.type==="route")return <Route key={item.id} rows={query("technical_path")}/>;
  if(item.type==="tree")return <ScenarioTree key={item.id} rows={query("tree_nodes")}/>;
  let rows=query(item.queryId),sourceRows=rows,spec={...item.spec},headerControls=null,title=item.title;
  if(item.scope?.startsWith("forecast")){
   sourceRows=forecastRows.filter(row=>item.scope==="forecast_lead"?row.lead!=="all":row.lead==="all");
   rows=aggregate(sourceRows,item.scope==="forecast_month"?["month","month_label","model_label"]:item.scope==="forecast_lead"?["lead","model_label"]:["variant","seed","model_label"],"forecast");
   spec.y=metric;spec.yLabel=metrics[metric]+(metric==="wape_pct"?"（%）":target==="price"?"（元/千瓦时）":"（千瓦）");
   if(item.scope==="forecast_month")headerControls=forecastControls;
   title=`${item.title} · ${labels[target]} · ${month==="all"?"2—12月":`${month}月`}`;
  }else if(item.scope==="cost"||item.scope==="bridge"){
   sourceRows=costRows.filter(row=>item.scope!=="bridge"||["legacy_rebased","new_deterministic","primary"].includes(row.name));
   rows=aggregate(sourceRows,["name","model_label"],"cost");title=`问题 ${scenario} · ${item.title} · ${month==="all"?"2—12月":`${month}月`}`;
  }else if(item.scope==="daily"){
   rows=costRows.filter(row=>row.name==="primary");sourceRows=rows;
  }else if(item.scope==="specified"){
   rows=rows.filter(row=>row.scenario===scenario&&row.date===chosenDay);sourceRows=rows;
   if(item.id==="specified-intervals")headerControls=<Dropdown label="指定日期" showLabel value={chosenDay} choices={["2025-03-20","2025-06-21","2025-09-23","2025-12-21"]} onChange={setChosenDay}/>;
   title=`问题 ${scenario} · ${chosenDay} · ${item.title}`;
  }else if(item.scope==="case"){
   rows=rows.filter(row=>row.scenario===scenario);sourceRows=rows;
  }else if(item.scope==="history"){
   rows=rows.filter(row=>row.route==="正式调度"?row.task===scenario:row.route.startsWith("正式预测问题")?row.route.startsWith(`正式预测问题${scenario}/`):row.task===target);sourceRows=rows;title=`问题 ${scenario} · ${item.title}`;
  }else if(item.scope==="annual"){
   rows=rows.filter(row=>row.scenario===scenario&&row.seed===42);sourceRows=rows;
  }else if(item.scope==="decomposition"){
   rows=rows.filter(row=>row.target===target);sourceRows=rows;
   title=`${item.title} · ${labels[target]}`;spec.yLabel=target==="price"?"元/千瓦时":"千瓦";
  }
  if(!rows.length)return null;
  if(item.type==="chart")return <EvidenceChart key={item.id} id={item.id} queryId={item.queryId} title={title} rows={rows} sourceRows={sourceRows} headerControls={headerControls} spec={spec} height={item.height??340}/>;
  return <DataComponent key={item.id} id={item.id} queryId={item.queryId} kind="table" title={title} sourceRows={sourceRows} displayRows={rows} headerControls={headerControls}><Table rows={rows} columns={item.columns}/></DataComponent>;
 }
 return <article className="report-content experiment-report" aria-label="固定轻量网络与多阶段风险调度实验报告"><header className="report-hero"><h1 data-data-app-title>{appTitle}</h1></header><nav className="experiment-nav" aria-label="报告章节">{(snapshot.reportContent??[]).map((section,i)=><a key={section.title} href={`#chapter-${i+1}`}>{i+1}. {section.title}</a>)}</nav>{controls}{(snapshot.reportContent??[]).map((section,i)=><section id={`chapter-${i+1}`} key={section.title}>{section.blocks.map(block)}</section>)}</article>;
}
