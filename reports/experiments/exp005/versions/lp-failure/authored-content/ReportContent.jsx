import React, { useState } from "react";
import { DataComponent, Dropdown, EvidenceChart, ReportSection, RichNarrative, useDataApp } from "../../data-app-public.jsx";
import "./report.css";

const fmt = value => typeof value === "number" ? (value !== 0 && Math.abs(value) < 0.00001 ? value.toExponential(3) : value.toLocaleString("zh-CN", { maximumFractionDigits: 6 })) : typeof value === "boolean" ? (value ? "通过" : "触发停止") : value;
const chartRows = rows => rows.map(row => Object.fromEntries(Object.entries(row).map(([key, value]) => [key, typeof value === "number" ? Number(value.toFixed(4)) : value])));
function Table({ rows, columns }) {
  return <div className="failure-table-scroll"><table className="failure-table" data-reviewed-rows><thead><tr>{columns.map(([key, label]) => <th key={key}>{label}</th>)}</tr></thead><tbody>{rows.map((row, i) => <tr key={i}>{columns.map(([key]) => <td key={key} className={typeof row[key] === "number" ? "number" : undefined}>{fmt(row[key])}</td>)}</tr>)}</tbody></table></div>;
}

export function ReportContent() {
  const { snapshot, appTitle, canEdit, mode, setAppTitle } = useDataApp();
  const [stage, setStage] = useState("第二层·平稳性");
  const [scope, setScope] = useState("局部窗口");
  const query = id => snapshot.queries?.[id]?.rows ?? [];
  const visibleWindow = query("window").filter(r => scope === "完整剩余窗口" || r.slot <= 99);
  const windowRows = visibleWindow.filter(r => r.stage === stage);
  const nodeRows = query("window").filter(r => r.stage === stage && r.slot === 85);
  const sourcesBySection = [["stages", "prefix", "audit"], ["audit", "settings"], ["audit", "settings"], ["settings"], ["settings"], ["stages", "node", "window", "prefix", "audit"], [], ["audit"]];
  const controls = <div className="failure-controls"><Dropdown label="求解层" showLabel value={stage} choices={["第二层·平稳性", "第一层·费用"]} onChange={setStage}/><Dropdown label="规划范围" showLabel value={scope} choices={["局部窗口", "完整剩余窗口"]} onChange={setScope}/></div>;
  const lineStyle = { type: "line", x: "hour", valueDecimals: 4, stackable: false, xLabel: "时刻 / 小时", legend: { labels: { charge_kwh: "充电量", discharge_kwh: "放电量" } } };
  return <article className="report-content failure-report" aria-label="第二问D4-A失败证据报告">
    <header className="report-hero"><h1 data-data-app-title contentEditable={canEdit && mode === "edit"} suppressContentEditableWarning onBlur={canEdit && mode === "edit" ? e => setAppTitle(e.currentTarget.textContent.trim() || appTitle) : undefined}>{appTitle}</h1></header>
    <nav className="failure-nav" aria-label="八节报告导航">{snapshot.sections.map((s, i) => <a key={i} href={`#failure-chapter-${i + 1}`}>{i + 1}. {s.title}</a>)}</nav>
    {snapshot.sections.map((section, i) => {
      const prose = <RichNarrative id={`failure-prose-${i + 1}`} value={section.markdown}/>;
      const ids = sourcesBySection[i];
      return <section id={`failure-chapter-${i + 1}`} key={i}>
        {ids.length ? <ReportSection id={`failure-section-${i + 1}`} title={section.title} showHeading={false} queryId={ids[0]} queryIds={ids} sourceRowsByQuery={Object.fromEntries(ids.map(id => [id, query(id)]))}>{prose}</ReportSection> : prose}
        {i === 0 && <EvidenceChart id="failure-overlap-layers" queryId="stages" title="同一规划窗口，第二层出现566.5938 kWh重叠" rows={query("stages")} sourceRows={query("stages")} spec={{ type: "horizontalBar", x: "stage", y: "max_overlap_kwh", valueDecimals: 4, xLabel: "最大重叠量 / kWh", stackable: false }} height={250}/>}
        {i === 4 && <DataComponent id="failure-settings" queryId="settings" kind="table" title="保持原模型的固定参数" sourceRows={query("settings")}><Table rows={query("settings")} columns={[["parameter", "参数"], ["value", "值"]]}/></DataComponent>}
        {i === 5 && <>
          <DataComponent id="failure-stage-values" queryId="stages" kind="table" title="13:30窗口的两层求解结果" sourceRows={query("stages")}><Table rows={query("stages")} columns={[["stage", "层"], ["emergency_cost", "剩余预期紧急费 / 元"], ["tv_kw", "功率总变差 / kW"], ["max_overlap_kwh", "最大重叠 / kWh"]]}/></DataComponent>
          <RichNarrative id="failure-chart-precision" value="绘图数据取四位小数，表格显示最多六位小数；来源数据与下载证据保留原始求解精度。横轴以小时表示，十分钟对应约0.1667小时。"/>
          <EvidenceChart id="failure-charge-discharge" queryId="window" title={`${stage}：13:30求解的未来充放电量`} rows={chartRows(windowRows)} sourceRows={windowRows} spec={{ ...lineStyle, y: "charge_kwh", fields: ["charge_kwh", "discharge_kwh"], yLabel: "kWh / 十分钟", colors: { charge_kwh: "#258a79", discharge_kwh: "#bf6548" } }} height={340} headerControls={controls}/>
          <DataComponent id="failure-selected-node" queryId="window" kind="table" title={`${stage}：14:10—14:20节点`} sourceRows={nodeRows}><Table rows={nodeRows} columns={[["interval", "未来区间"], ["charge_kwh", "充电 / kWh"], ["discharge_kwh", "放电 / kWh"], ["overlap_kwh", "重叠 / kWh"], ["end_soc_kwh", "结束SOC / kWh"]]}/></DataComponent>
          <EvidenceChart id="failure-window-power" queryId="window" title="两层规划净功率对比" rows={chartRows(visibleWindow)} sourceRows={visibleWindow} spec={{ ...lineStyle, y: "power_kw", series: "stage", yLabel: "净充电功率 / kW", legend: { labels: {} } }} height={310}/>
          <EvidenceChart id="failure-actual-soc" queryId="prefix" title="实际执行81段的末储电量，截止13:30" rows={chartRows(query("prefix").map(r => ({ ...r, end_hour: r.hour + 1 / 6 })))} sourceRows={query("prefix")} spec={{ type: "line", x: "end_hour", y: "end_soc_kwh", valueDecimals: 4, stackable: false, xLabel: "区间结束时刻 / 小时", yLabel: "内部储电量 / kWh" }} height={310}/>
          <EvidenceChart id="failure-actual-power" queryId="prefix" title="实际执行81段的净充电功率" rows={chartRows(query("prefix"))} sourceRows={query("prefix")} spec={{ ...lineStyle, y: "net_power_kw", yLabel: "净充电功率 / kW" }} height={310}/>
        </>}
        {i === 7 && <DataComponent id="failure-audit-table" queryId="audit" kind="table" title="独立核验：残差通过，重叠诊断触发停止" sourceRows={query("audit")}><Table rows={query("audit")} columns={[["check", "核验项"], ["value", "实测值"], ["limit", "上限"], ["unit", "单位"], ["passed", "结果"]]}/></DataComponent>}
      </section>;
    })}
  </article>;
}
