import React, {useState} from "react";
import {DataComponent, Dropdown, EvidenceChart, ReportSection, RichNarrative, useDataApp} from "../../data-app-public.jsx";
import "./report.css";
const format=v=>v==null?"—":typeof v==="boolean"?(v?"是":"否"):typeof v==="number"?v.toLocaleString("zh-CN",{maximumFractionDigits:4}):v;
function Table({rows,columns}){return <div className="exp004-table-scroll"><table className="exp004-table" data-reviewed-rows><thead><tr>{columns.map(([k,l])=><th key={k}>{l}</th>)}</tr></thead><tbody>{rows.map((r,i)=><tr key={i}>{columns.map(([k])=><td key={k} className={typeof r[k]==="number"?"number":undefined}>{format(r[k])}</td>)}</tr>)}</tbody></table></div>;}
export function ReportContent(){
 const {snapshot,appTitle,canEdit,mode,setAppTitle}=useDataApp();
 const [day,setDay]=useState("2025-03-20"),[variant,setVariant]=useState("causal_season");
 const query=id=>snapshot.queries?.[id]?.rows??[];
 const controls=<div className="exp004-controls"><Dropdown label="指定日期" showLabel value={day} choices={snapshot.specifiedDates??[]} onChange={setDay}/><Dropdown label="预测组" showLabel value={variant} choices={["no_season","causal_season","oracle_season"]} choiceLabels={{no_season:"无季节",causal_season:"历史季节",oracle_season:"全年探索（含未来）"}} onChange={setVariant}/></div>;
 function block(item){
  if(item.type==="prose"){
   const prose=<RichNarrative id={item.id} value={item.markdown}/>;
   if(!item.queryIds?.length)return <React.Fragment key={item.id}>{prose}</React.Fragment>;
   return <ReportSection key={item.id} id={`${item.id}-sources`} title="说明依据" showHeading={false} queryId={item.queryIds[0]} queryIds={item.queryIds} sourceRowsByQuery={Object.fromEntries(item.queryIds.map(id=>[id,query(id)]))}>{prose}</ReportSection>;
  }
  let rows=query(item.queryId);if(item.scope==="specified")rows=rows.filter(r=>r.date===day&&r.name===variant);
  const title=item.title+(item.scope==="specified"?` · ${day} · ${variant==="oracle_season"?"全年探索（含未来）":variant==="causal_season"?"历史季节":"无季节"}`:"");
  if(item.type==="chart")return <EvidenceChart key={item.id} id={item.id} queryId={item.queryId} title={title} rows={rows} sourceRows={rows} spec={item.spec} height={item.height??360} headerControls={item.controls?controls:undefined}/>;
  if(item.type==="routes")return <DataComponent key={item.id} id={item.id} queryId={item.queryId} kind="custom" title={title} sourceRows={rows}><div className="exp004-routes" data-reviewed-rows>{rows.map(r=><div className="exp004-route" key={r.label}><strong>{r.label}</strong><ol>{r.steps.map((s,i)=><li key={i}>{s}</li>)}</ol></div>)}</div></DataComponent>;
  return <DataComponent key={item.id} id={item.id} queryId={item.queryId} kind="table" title={title} sourceRows={rows} displayRows={rows}><Table rows={rows} columns={item.columns}/></DataComponent>;
 }
 return <article className="report-content exp004-report" aria-label="第二问预测实验"><header className="report-hero"><h1 data-data-app-title contentEditable={canEdit&&mode==="edit"} suppressContentEditableWarning onBlur={canEdit&&mode==="edit"?e=>setAppTitle(e.currentTarget.textContent.trim()||appTitle):undefined}>{appTitle}</h1></header><nav className="exp004-nav" aria-label="报告章节">{(snapshot.reportContent??[]).map((s,i)=><a key={s.title} href={`#chapter-${i+1}`}>{i+1}. {s.title}</a>)}</nav>{(snapshot.reportContent??[]).map((s,i)=><section id={`chapter-${i+1}`} key={s.title}>{s.blocks.map(block)}</section>)}</article>;
}
