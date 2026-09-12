import React,{useEffect,useState} from 'react';
import {DataComponent,Dropdown,EvidenceChart,ReportSection,RichNarrative,useDataApp} from '../../data-app-public.jsx';
import {BatteryExplorer} from './BatteryExplorer.jsx';
import './report.css';
const format=v=>v==null?'—':typeof v==='boolean'?(v?'是':'否'):typeof v==='number'?v.toLocaleString('zh-CN',{maximumFractionDigits:4}):typeof v==='object'?JSON.stringify(v):String(v);
function Table({rows,columns}){return <div className="exp007-table-scroll"><table className="exp007-table" data-reviewed-rows><thead><tr>{columns.map(([k,l])=><th key={k}>{l}</th>)}</tr></thead><tbody>{rows.map((r,i)=><tr key={i}>{columns.map(([k])=><td key={k} className={typeof r[k]==='number'?'number':undefined}>{format(r[k])}</td>)}</tr>)}</tbody></table></div>;}
export function ReportContent(){
 const {snapshot,appTitle,canEdit,mode,setAppTitle,visible}=useDataApp();
 const [narrow,setNarrow]=useState(false),[day,setDay]=useState('2025-03-20'),[run,setRun]=useState('regularized_42');
 useEffect(()=>{const media=window.matchMedia('(max-width:600px)');const update=()=>setNarrow(media.matches);update();media.addEventListener('change',update);return()=>media.removeEventListener('change',update);},[]);
 const query=id=>snapshot.queries?.[id]?.rows??[];
 const controls=<div className="exp007-controls"><Dropdown label="指定日期" showLabel value={day} choices={snapshot.specifiedDates??[]} onChange={setDay}/><Dropdown label="RL 方案" showLabel value={run} choices={snapshot.runIds??[]} choiceLabels={snapshot.policyLabels??{}} onChange={setRun}/></div>;
 function block(item){
  if(item.type==='prose'){
   const prose=<RichNarrative id={item.id} value={item.markdown}/>;
   if(!item.queryIds?.length)return <React.Fragment key={item.id}>{prose}</React.Fragment>;
   return <ReportSection key={item.id} id={`${item.id}-sources`} title="说明依据" showHeading={false} queryId={item.queryIds[0]} queryIds={item.queryIds} sourceRowsByQuery={Object.fromEntries(item.queryIds.map(id=>[id,query(id)]))}>{prose}</ReportSection>;
  }
  if(!visible(item.id))return null;
  if(item.type==='battery'||item.type==='batteryRandom')return <BatteryExplorer key={item.id} item={item} random={item.type==='batteryRandom'}/>;
  let rows=query(item.queryId);if(item.scope==='specified')rows=rows.filter(r=>r.date===day&&(r.run_id??r.name??r.id)===run);
  const title=item.title+(item.scope==='specified'?` · ${day} · ${snapshot.policyLabels?.[run]??run}`:'');
  if(item.type==='chart')return <EvidenceChart key={item.id} id={item.id} queryId={item.queryId} title={title} rows={rows} sourceRows={rows} spec={narrow&&item.spec.presentation==='plot'?{...item.spec,barOptions:{...item.spec.barOptions,categoryWidth:122,labels:{value:false}}}:item.spec} height={item.height??360} headerControls={item.controls?controls:undefined} renderPlot={(plot,current)=><>{current.presentation==='plot'&&current.barOptions?.series?.length>0&&<div className="exp007-power-legend" data-reviewed-rows>{current.barOptions.series.map(s=><span key={s.key} style={{color:s.color}}>■ {s.label??s.key}</span>)}</div>}{plot}</>}/>;
  if(item.type==='routes')return <DataComponent key={item.id} id={item.id} queryId={item.queryId} kind="custom" title={title} sourceRows={rows} displayRows={rows}><div className="exp007-routes" data-reviewed-rows>{rows.map(r=><div className="exp007-route" key={r.label}><strong>{r.label}</strong><ol>{r.steps.map((s,i)=><li key={i}>{s}</li>)}</ol></div>)}</div></DataComponent>;
  if(item.type==='network')return <DataComponent key={item.id} id={item.id} queryId={item.queryId} kind="custom" title={title} sourceRows={rows} displayRows={rows}><div className="exp007-network" data-reviewed-rows><div className="exp007-network-trunk">{rows.filter(r=>r.branch==='shared').map(r=><div key={r.layer}><strong>{r.layer}</strong><span>{r.units} 维 · {r.activation}</span></div>)}</div><div className="exp007-network-heads">{rows.filter(r=>r.branch!=='shared').map(r=><div key={r.layer}><strong>{r.layer}</strong><span>{r.units} 维 · {r.activation}</span><span>{r.purpose}</span></div>)}</div></div></DataComponent>;
  return <DataComponent key={item.id} id={item.id} queryId={item.queryId} kind="table" title={title} sourceRows={rows} displayRows={rows}><Table rows={rows} columns={item.columns}/></DataComponent>;
 }
 return <article className="report-content exp007-report" aria-label="exp007 强化学习规划实验报告"><header className="report-hero"><h1 data-data-app-title contentEditable={canEdit&&mode==='edit'} suppressContentEditableWarning onBlur={canEdit&&mode==='edit'?e=>setAppTitle(e.currentTarget.textContent.trim()||appTitle):undefined} onKeyDown={canEdit&&mode==='edit'?e=>{if(e.key==='Enter'){e.preventDefault();e.currentTarget.blur();}}:undefined}>{appTitle}</h1></header><nav className="exp007-nav" aria-label="报告章节">{(snapshot.reportContent??[]).map((s,i)=><a key={s.title} href={`#chapter-${i+1}`}>{i+1}. {s.title}</a>)}</nav>{(snapshot.reportContent??[]).map((s,i)=><section className="exp007-chapter" id={`chapter-${i+1}`} key={s.title}>{s.blocks.map(block)}</section>)}</article>;
}
