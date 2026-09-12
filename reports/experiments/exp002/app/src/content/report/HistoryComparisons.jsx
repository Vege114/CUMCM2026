import React, {useState} from "react";
import {Dropdown, EvidenceChart} from "../../data-app-public.jsx";

const questions=["2","3","4-2","4-3"];
const policies=["legacy_rebased","new_deterministic","primary","periodic"];
const policyLabels={legacy_rebased:"exp001 重算",new_deterministic:"exp002 基础",primary:"exp002 风险",periodic:"周期基线"};
const targetLabels={load:"负载",pv:"历史光伏",pv_corrected:"预报光伏*",price:"电价"};

export function HistoryCosts({rows}) {
  const sourceRows=rows.filter(r=>r.seed===42&&policies.includes(r.name)).sort((a,b)=>
    questions.indexOf(a.scenario)-questions.indexOf(b.scenario)||policies.indexOf(a.name)-policies.indexOf(b.name));
  const displayRows=sourceRows.map(r=>({...r,question:r.scenario,strategy:policyLabels[r.name],cost_wan:r.total_cost/10000}));
  return <EvidenceChart id="history-cost-comparison" queryId="cost_annual"
    title="exp001 重算与 exp002 · 全年费用（万元）· 四个问题" rows={displayRows} sourceRows={sourceRows} height={460}
    spec={{type:"horizontalBar",x:"question",y:"cost_wan",series:"strategy",stackable:false,
      colors:{"exp001 重算":"#a4aeb6","exp002 基础":"#5d9aae","exp002 风险":"#247e77","周期基线":"#967349"},
      yLabel:"全年费用（万元）",valueDecimals:4}}/>;
}

export function HistoryForecast({rows}) {
  const [population,setPopulation]=useState("all");
  const sourceRows=rows.filter(r=>r.population===population&&["wape_pct","rmse"].includes(r.metric));
  const displayRows=sourceRows.map(r=>({...r,category:`${r.scenario} · ${targetLabels[r.target]}`,
    metricLabel:r.metric==="wape_pct"?"WAPE":"RMSE","relativeChange(%)":r.relative_change_pct,
    absolute_change:r.current-r.previous}));
  const headerControls=<Dropdown label="历史预测时段" showLabel value={population}
    choices={["all","generating"]} choiceLabels={{all:"全时段",generating:"实际发电时段"}} onChange={setPopulation}/>;
  return <EvidenceChart id="history-forecast-comparison" queryId="official_forecast_comparison"
    title={`exp001 → exp002 · WAPE 与 RMSE 相对变化 · ${population==="all"?"全时段":"实际发电时段"}`}
    rows={displayRows} sourceRows={sourceRows} headerControls={headerControls} height={population==="all"?590:350}
    spec={{type:"horizontalBar",x:"category",y:"relativeChange(%)",series:"metricLabel",stackable:false,
      colors:{WAPE:"#2b788b",RMSE:"#8666a3"},
      yLabel:"相对变化（%，负值下降）",valueDecimals:3}}/>;
}

export function HistoryTraining({rows}) {
  return <div className="experiment-history-training">
    <EvidenceChart id="history-training-count" queryId="timing" title="exp001 → exp002 · 训练组数"
      rows={rows} sourceRows={rows} height={280} spec={{type:"bar",x:"experiment",y:"groups",yLabel:"组"}}/>
    <EvidenceChart id="history-training-time" queryId="timing" title="exp001 → exp002 · 累计训练耗时（分钟）"
      rows={rows} sourceRows={rows} height={280} spec={{type:"bar",x:"experiment",y:"training_minutes",yLabel:"分钟",valueDecimals:4}}/>
  </div>;
}
