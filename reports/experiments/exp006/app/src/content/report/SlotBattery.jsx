import React, {useLayoutEffect,useRef,useState} from "react";

// Interval geometry is intentional: a bar spans the entire reviewed ten-minute
// interval. This authored chart uses the enclosing DataComponent's source rows.
export function SlotBattery({rows}) {
 const [active,setActive]=useState(null);
 const frame=useRef(null),[width,setWidth]=useState(900);
 useLayoutEffect(()=>{
  const element=frame.current;if(!element)return;
  const measure=()=>setWidth(Math.max(280,element.clientWidth));measure();
  const observer=new ResizeObserver(measure);observer.observe(element);return()=>observer.disconnect();
 },[]);
 const sorted=[...rows].sort((a,b)=>a.slot-b.slot);
 const W=width,H=350,L=48,R=18,T=32,B=52;
 const plotW=W-L-R,plotH=H-T-B;
 const maximum=1000;
 const x=h=>L+h/24*plotW,y=v=>T+plotH-v/maximum*plotH;
 const time=s=>`${String(Math.floor(s/6)).padStart(2,"0")}:${String((s%6)*10).padStart(2,"0")}`;
 const entry=active==null?null:sorted[active];
 const keyboard=(event,index)=>{
  const step=event.key==="ArrowRight"?1:event.key==="ArrowLeft"?-1:0;
  if(!step)return;
  event.preventDefault();const next=Math.max(0,Math.min(sorted.length-1,index+step));
  event.currentTarget.parentElement.querySelector(`[data-slot-focus="${next}"]`)?.focus();
 };
 return <div className="exp006-slot-chart" ref={frame} data-reviewed-rows>
  <div className="exp006-slot-legend"><span style={{color:'#357c89'}}>■ 充电</span><span style={{color:'#aa6d37'}}>■ 放电</span></div>
  <div className="exp006-slot-scroll">
   <svg className="exp006-slot-svg" viewBox={`0 0 ${W} ${H}`} role="group" aria-label="逐个10分钟区间的电池充电与放电，单位kWh">
    <text x={L} y={18} fill="var(--secondary)" fontSize="13">每10分钟 kWh</text>
    {[0,250,500,750,1000].map(v=><g key={v}><line x1={L} x2={W-R} y1={y(v)} y2={y(v)} stroke="var(--border)" strokeWidth=".8"/><text x={L-10} y={y(v)+4} textAnchor="end" fill="var(--secondary)" fontSize="13">{v}</text></g>)}
    {[0,6,12,18,24].map(h=><g key={h}><line x1={x(h)} x2={x(h)} y1={T} y2={T+plotH} stroke="var(--border)" strokeWidth=".6"/><text x={x(h)} y={T+plotH+23} textAnchor="middle" fill="var(--secondary)" fontSize="13">{h}</text></g>)}
    <line x1={L} x2={W-R} y1={y(0)} y2={y(0)} stroke="var(--text)" strokeWidth="1"/>
    <text x={L+plotW/2} y={H-8} textAnchor="middle" fill="var(--secondary)" fontSize="13">当日小时</text>
    {sorted.map((r,index)=>{
     const c=r.charge_kwh,d=r.discharge_kwh,width=plotW/144;
     return <g key={r.slot}>
      {c>0&&<rect x={x(r.hour)+.25} y={y(c)} width={width-.5} height={y(0)-y(c)} fill="#357c89"/>}
      {d>0&&<rect x={x(r.hour)+.25} y={y(d)} width={width-.5} height={y(0)-y(d)} fill="#aa6d37"/>}
     </g>;
    })}
    {entry&&<rect x={x(entry.hour)} y={T} width={plotW/144} height={plotH} fill="var(--text)" opacity=".10" pointerEvents="none"/>}
    <g>{sorted.map((r,index)=><rect key={r.slot} data-slot-focus={index} tabIndex={index===0?0:-1} role="button" x={x(r.hour)} y={T} width={plotW/144} height={plotH} fill="transparent" aria-label={`${r.date} ${time(r.slot)}至${time(r.slot+1)}，充电${r.charge_kwh.toFixed(2)}kWh，放电${r.discharge_kwh.toFixed(2)}kWh`} onMouseEnter={()=>setActive(index)} onPointerDown={()=>setActive(index)} onMouseLeave={()=>setActive(null)} onFocus={()=>setActive(index)} onBlur={()=>setActive(null)} onKeyDown={event=>keyboard(event,index)}/>)}</g>
   </svg>
  </div>
  <div className="exp006-slot-tooltip" role="status" aria-live="polite">{entry?`${entry.date} ${time(entry.slot)}–${time(entry.slot+1)}　充电 ${entry.charge_kwh.toFixed(2)} kWh　放电 ${entry.discharge_kwh.toFixed(2)} kWh`:'悬停查看各槽实际值；键盘聚焦图后可用左右方向键切换。'}</div>
 </div>;
}
