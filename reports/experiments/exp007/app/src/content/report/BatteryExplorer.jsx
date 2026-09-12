import React, {useEffect,useMemo,useRef,useState} from 'react';
import {DataComponent,Dropdown,useDataApp} from '../../data-app-public.jsx';
const COLORS=['#357c89','#bc6d3e','#8075a9','#5d8c69'];
const START=Date.UTC(2025,1,1), DAY=86400000;
const dateFor=i=>new Date(START+Math.floor(i/144)*DAY).toISOString().slice(0,10);
const dateIndex=date=>Math.round((Date.parse(date+'T00:00:00Z')-START)/DAY);
const fmt=v=>Number.isFinite(v)?v.toLocaleString('zh-CN',{maximumFractionDigits:3}):'缺失';
function slotTime(i){const minutes=(i%144+1)*10;return `${String(Math.floor(minutes/60)).padStart(2,'0')}:${String(minutes%60).padStart(2,'0')}`;}
function evidenceRows(series,start,stop){return series.map(s=>({policy_id:s.key,label:s.label,start_date:dateFor(start),end_date:dateFor(stop-1),start_slot:start,points:stop-start,power_kw_json:JSON.stringify(s.values.slice(start,stop))}));}
function PowerCanvas({series,start=0,stop=48096,height=290,daily=false}){
 const canvas=useRef(null),host=useRef(null);const [width,setWidth]=useState(760),[point,setPoint]=useState(start);
 useEffect(()=>{setPoint(start);},[start,stop]);
 useEffect(()=>{const element=host.current;if(!element)return;const resize=new ResizeObserver(entries=>setWidth(Math.max(260,entries[0].contentRect.width)));resize.observe(element);return()=>resize.disconnect();},[]);
 const pointCount=stop-start;
 useEffect(()=>{
  const el=canvas.current;if(!el||!pointCount)return;const scale=window.devicePixelRatio||1;el.width=Math.round(width*scale);el.height=Math.round(height*scale);const ctx=el.getContext('2d');ctx.scale(scale,scale);
  const style=getComputedStyle(host.current),fg=style.getPropertyValue('--text').trim()||'#25313b',border=style.getPropertyValue('--border').trim()||'#d5d8d9';
  const left=width<450?55:68,right=18,top=20,bottom=height-54;const x=i=>left+(i-start)/Math.max(1,pointCount-1)*(width-left-right),y=v=>bottom-(v+5500)/11000*(bottom-top);
  ctx.clearRect(0,0,width,height);ctx.font='12px sans-serif';ctx.fillStyle=fg;ctx.strokeStyle=border;ctx.lineWidth=1;
  [-5000,0,5000].forEach(v=>{ctx.setLineDash(v===0?[]:[4,4]);ctx.beginPath();ctx.moveTo(left,y(v));ctx.lineTo(width-right,y(v));ctx.stroke();ctx.textAlign='right';ctx.fillText(v.toLocaleString('en-US'),left-7,y(v)+4);});ctx.setLineDash([]);
  ctx.textAlign='left';ctx.fillText('kW · 充电为正',left,12);ctx.beginPath();ctx.moveTo(left,top);ctx.lineTo(left,bottom);ctx.lineTo(width-right,bottom);ctx.stroke();
  const ticks=width<450?3:5;
  for(let j=0;j<=ticks;j++){const idx=start+Math.round(j*(pointCount-1)/ticks);ctx.textAlign=j===0?'left':j===ticks?'right':'center';ctx.fillText(daily?slotTime(idx):new Date(START+(idx+1)*600000).toISOString().slice(5,10),x(idx),bottom+21);}
  ctx.textAlign='center';ctx.fillText(daily?'区间终点时刻':'日期 · 每个原始十分钟点均绘制',(left+width-right)/2,height-8);
  series.forEach((s,n)=>{ctx.strokeStyle=COLORS[n%COLORS.length];ctx.lineWidth=1.05;ctx.globalAlpha=.86;ctx.beginPath();let connected=false;for(let i=start;i<stop;i++){const value=s.values[i];if(!Number.isFinite(value)){connected=false;continue;}if(connected)ctx.lineTo(x(i),y(value));else{ctx.moveTo(x(i),y(value));connected=true;}}ctx.stroke();});ctx.globalAlpha=1;
  if(point>=start&&point<stop){ctx.strokeStyle=fg;ctx.setLineDash([2,4]);ctx.beginPath();ctx.moveTo(x(point),top);ctx.lineTo(x(point),bottom);ctx.stroke();ctx.setLineDash([]);series.forEach((s,n)=>{if(Number.isFinite(s.values[point])){ctx.fillStyle=COLORS[n%COLORS.length];ctx.beginPath();ctx.arc(x(point),y(s.values[point]),3,0,2*Math.PI);ctx.fill();}});}
 },[series,start,stop,width,height,point,pointCount,daily]);
 const follow=e=>{const rect=e.currentTarget.getBoundingClientRect();const left=width<450?55:68;setPoint(Math.min(stop-1,Math.max(start,start+Math.round((e.clientX-rect.left-left)/(width-left-18)*(pointCount-1)))));};
 return <div ref={host} className="exp007-power-plot" data-reviewed-rows><canvas ref={canvas} style={{width:'100%',height}} role="img" aria-label={`${dateFor(start)}至${dateFor(stop-1)}，${pointCount}个原始净功率点每方案，kW，充电为正`} onPointerMove={follow} onPointerDown={follow}/><div className="exp007-power-legend">{series.map((s,i)=><span key={s.key} style={{color:COLORS[i%COLORS.length]}}>━ {s.label}</span>)}</div><label className="exp007-point-control">查看原始槽位 <input aria-label="原始功率槽位" type="range" min={start} max={stop-1} value={point} onChange={e=>setPoint(Number(e.target.value))}/></label><div className="exp007-power-value" aria-live="polite">{dateFor(point)} {slotTime(point)}（区间终点）{series.map(s=>` · ${s.label} ${fmt(s.values[point])} kW`).join('')}</div></div>;
}
export function BatteryExplorer({random=false,item}){
 const {snapshot}=useDataApp();const all=snapshot.powerSeries??[];
 const initial=all.find(s=>s.key==='exp007/regularized_42')?.key??all[0]?.key;
 const [policy,setPolicy]=useState(initial),[baseline,setBaseline]=useState(all.find(s=>s.key==='exp004/no_season')?.key??all[1]?.key??initial),[mode,setMode]=useState('all'),[first,setFirst]=useState('2025-02-01'),[last,setLast]=useState('2025-12-31');
 const series=useMemo(()=>[all.find(s=>s.key===policy),all.find(s=>s.key===baseline)].filter((s,i,a)=>s&&a.findIndex(x=>x?.key===s.key)===i),[all,policy,baseline]);
 const labels=Object.fromEntries(all.map(s=>[s.key,s.label]));
 const controls=<div className="exp007-controls"><Dropdown label="策略" showLabel value={policy} choices={all.map(s=>s.key)} choiceLabels={labels} onChange={setPolicy}/><Dropdown label="对比策略" showLabel value={baseline} choices={all.map(s=>s.key)} choiceLabels={labels} onChange={setBaseline}/>{!random&&<><Dropdown label="查看范围" showLabel value={mode} choices={['all','day','range']} choiceLabels={{all:'全部 334 日',day:'单日放大',range:'日期范围'}} onChange={setMode}/>{mode!=='all'&&<label>开始日期<input type="date" aria-label="电池曲线开始日期" min="2025-02-01" max="2025-12-31" value={first} onChange={e=>{const date=e.target.value||'2025-02-01';setFirst(date);if(date>last)setLast(date);}}/></label>}{mode==='range'&&<label>结束日期<input type="date" aria-label="电池曲线结束日期" min={first} max="2025-12-31" value={last} onChange={e=>setLast(e.target.value||first)}/></label>}</>}</div>;
 const start=mode==='all'?0:Math.max(0,Math.min(333,dateIndex(first)))*144;const stop=mode==='all'?48096:mode==='day'?start+144:Math.max(start+144,Math.min(48096,(dateIndex(last)+1)*144));
 const fullRows=useMemo(()=>evidenceRows(series,start,stop),[series,start,stop]);
 const randomDays=snapshot.randomBatteryDates??[];const randomRows=useMemo(()=>randomDays.flatMap(date=>{const start=dateIndex(date)*144;return evidenceRows(series,start,start+144);}),[series,randomDays]);
 if(!all.length)return <RichEmpty/>;
 return <DataComponent id={item.id} queryId={item.queryId} kind="custom" title={item.title} headerControls={controls} sourceRows={random?randomRows:fullRows} displayRows={random?randomRows:fullRows}>{random?<div className="exp007-random-days">{randomDays.map(date=><section key={date}><h4 data-reviewed-rows>{date}</h4><PowerCanvas series={series} start={dateIndex(date)*144} stop={(dateIndex(date)+1)*144} daily height={245}/></section>)}</div>:<PowerCanvas series={series} start={start} stop={stop} daily={stop-start===144}/>}</DataComponent>;
}
function RichEmpty(){return <p>完整电池曲线尚未生成。</p>;}
