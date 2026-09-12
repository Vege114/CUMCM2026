"""Build the standalone Q2 discussion report/template from the bundled JSON snapshot.

No model code, source experiments, GPU runtime, or network data is required.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
TEMPLATE = ROOT / "reports/templates/q2-discussion"
DEFAULT_NODE = Path.home() / ".cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node"
DEFAULT_PLUGIN = Path.home() / ".codex/plugins/cache/openai-curated-remote/data-analytics/1.0.2"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command(args):
    result = subprocess.run([str(v) for v in args], capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(result.stdout + "\n" + result.stderr)
    if result.stdout.strip():
        print(result.stdout.strip())
    if result.stderr.strip():
        print(result.stderr.strip())
    return result


def query(rows, files, component_ids, definitions, methods):
    return {
        "rows": rows,
        "source": {
            "name": "第二问题目、原始附件及已完成方案的本地核验快照",
            "kind": "local-reviewed-files",
            "files": files,
            "period": "原始数据：2025-01-01—2025-12-31；正式费用：2025-02-01—2025-12-31",
            "metricDefinitions": [{"label": label, "definition": meaning, "componentIds": component_ids} for label, meaning in definitions],
            "evidenceFlow": [{"title": "保存可独立阅读的证据", "detail": "本报告随包 evidence.json 保存原始数据、费用分项与指定日输出；build_evidence.py 记录从仓库附件和已完成结果重建及核验的过程。报告重建只需随包 JSON。"}],
        },
        "methods": [{"language": "text", "code": methods}],
    }


def normalize_modules(document):
    """Accept the authored catalog, retaining the full choice explanations."""
    result = []
    for original in document.get("modules", []):
        module = {**original, "summary": original.get("task", ""),
                  "responsibility_markdown": f'**建议责任角色**：{original.get("owner_role", "待定")}\n\n**并行安排**：{original.get("parallel", "")}\n\n**现有实现**：{original.get("baseline", "")}',
                  "decisions": []}
        for original_decision in original.get("decisions", []):
            decision = {**original_decision, "context": original_decision.get("question", ""), "options": []}
            for option in original_decision["options"]:
                paragraphs = option.get("explanation", [])
                steps = "\n\n".join(f"{i+1}. {text}" for i, text in enumerate(paragraphs))
                markdown = steps + f'\n\n**能带来什么**：{option.get("benefit", "")}\n\n**需要付出什么**：{option.get("tradeoff", "")}\n\n**如何验收**：{option.get("acceptance", "")}'
                if option.get("depends_on"):
                    markdown += "\n\n**组合条件**：" + "；".join(option["depends_on"])
                decision["options"].append({**option, "label": option["title"], "markdown": markdown,
                    "status_label": "现有实现／已实验做法" if option["status"] == "current" else "待讨论候选，尚未验证"})
            module["decisions"].append(decision)
        result.append(module)
    return result


def snapshot(evidence, catalog, *, template=False, complete=False):
    modules = normalize_modules(catalog)
    queries = {}
    quantitative = not template and bool(evidence)
    files = ["reports/discussions/q2-improvement/evidence.json"]
    generic_costs = [("总购电费（元）", "计划购电费与紧急购电费相加。原计划即使未用完也付费；紧急电量按同一时段固定电价的5倍收费。"), ("计划购电费（元）", "逐10分钟原计划购电量乘固定电价后求和。"), ("紧急购电费（元）", "逐10分钟紧急购电量乘固定电价的5倍后求和。")]
    costs=[]; monthly=[]; raw=[]; rawmonthly=[]; tariffs=[]; intervals=[]; states=[]; summaries=[]; output={k:[] for k in ("output_table1","output_table2","output_table3")}
    if quantitative:
        times=evidence["tariff"]["interval_starts"]
        tariffs=[{"time":t,"price":p} for t,p in zip(times,evidence["tariff"]["price_yuan_per_kwh"],strict=True)]
        for date,load,pv in zip(evidence["raw"]["dates"],evidence["raw"]["load_kw"],evidence["raw"]["pv_kw"],strict=True):
            raw.extend({"date":date,"time":t,"load_kw":l,"pv_kw":p} for t,l,p in zip(times,load,pv,strict=True))
        rawmonthly=[{**r,"month_label":f'{r["month"]}月',"load_mwh":r["load_kwh"]/1000,"pv_mwh":r["pv_kwh"]/1000} for r in evidence["raw"]["monthly"]]
        for c in evidence["comparisons"]:
            costs.append({k:v for k,v in c.items() if k not in ("daily","monthly")})
            monthly.extend({"id":c["id"],"label":c["label"],**r,"month_label":f'{r["month"]}月'} for r in c["monthly"])
        for date,day in evidence["dispatch_days"].items():
            summaries.append({"date":date,**day["summary"]})
            source=day["intervals"]
            intervals.extend({"date":date,"time":t,**{f:source[f][i] for f in ("planned_kwh","emergency_kwh","charge_kwh","discharge_kwh","planned_cost","emergency_cost")}} for i,t in enumerate(times))
            states.extend({"date":date,"time":t,"soc_kwh":s,"min_kwh":1200,"max_kwh":10800} for t,s in zip([*times,"24:00"],source["states_kwh"],strict=True))
            for table in (1,2,3):
                output[f"output_table{table}"].extend({"date":date,**r} for r in day[f"table{table}"])
    definitions = {
        "tariff": (tariffs,["tariff-chart"],[("电价", "附件1固定分时电价，每个区间单位为元/kWh。")],"按附件1原始144个区间记录，时间标签由区间终点转为起点。"),
        "raw_monthly": (rawmonthly,["raw-monthly-chart"],[("负载与光伏月电量（MWh）", "原始平均功率（kW）乘1/6小时得到每段kWh；按自然月求和，再除1000。")],"按自然月汇总全部365天的144点功率×1/6；不删一月。"),
        "raw_daily": (raw,["raw-day-chart"],[("实际负载和光伏功率（kW）", "附件2当日144点原始功率，只是事后实际情况，不能当成同日午夜已知信息。")],"保留365日144点；视图按所选日期筛选，图和来源使用同一日期。"),
        "cost_annual": (costs,["cost-components","cost-exact-table"],generic_costs,"对各方案同一334天逐10分钟计划电费和紧急电费分别求和，来源快照含独立回算核验。费用图将元除以10000显示为万元；精确值表仍为元。"),
        "cost_monthly": (monthly,["cost-components","cost-exact-table","monthly-cost-chart"],generic_costs,"对每个方案正式334天按自然月求和。月份筛选只作用于比较条形图和精确值表；月趋势始终展示完整2—12月。费用图将元除以10000显示为万元；精确值表仍为元。"),
        "dispatch_summary": (summaries,["dispatch-day-cost","dispatch-day-quantities"],generic_costs + [("电量及首尾储电量（kWh）", "全天计划、紧急、充电、放电电量分别求和，首末状态为当日00:00与24:00的内部储电量。")],"从当前方案的五日原始调度分别汇总；按所选日期呈现。"),
        "dispatch_intervals": (intervals,["dispatch-energy-chart","dispatch-storage-chart"],[("电量（kWh）", "计划、紧急、充电与放电均为该10分钟区间的电量；充放电以储能设备外部交流侧为准。")],"读取四指定日与当前最贵日的144段执行结果，图与来源按同一日期筛选。"),
        "dispatch_states": (states,["dispatch-soc-chart"],[("储能剩余电量（kWh）", "设备内部储电量；初始节点加144个区间结束节点共145个。允许范围为1200—10800。")],"保留145个储电量节点，起点00:00到終点24:00。"),
        "output_table1": (output["output_table1"],["output_table1"],[("计划与紧急购电量（kWh）", "按题目要求给出10:00、12:00、14:00、16:00、18:00、20:00起六个10分钟区间的原计划购电与紧急购电量。")],"分别取下标60、72、84、96、108、120的10分钟执行结果；不是四小时汇总。"),
        "output_table2": (output["output_table2"],["output_table2"],[("充电与放电量（kWh）", "设备交流侧每4小时累计充电量、放电量。")],"每24个连续10分钟区间分别求和，形成六个4小时区间。"),
        "output_table3": (output["output_table3"],["output_table3"],[("紧急购电时段及电量（kWh）", "每段紧急购电发生的时间及对应电量；本日没有记录时应说明没有发生紧急购电。")],"按保存的实际紧急购电事件输出，不把缺失日期误写成零。"),
    }
    for name,(rows,ids,defs,methods) in definitions.items():
        queries[name]=query(rows,files,ids,defs,methods)
    contract = [{"item": f"{i+1:02d}", "meaning": text, "effect": ["数据与预测", "计划与执行", "储能规划与执行", "损耗与费用比较", "状态接续与评价期", "结算与交付"][i] if i < 6 else "共同确认"} if isinstance(text,str) else text for i,text in enumerate(catalog.get("frozen_contract", []))]
    queries["frozen_contract"]=query(contract,["reports/discussions/q2-improvement/modules.json"],["frozen-contract"],[],"将题意与当前已决定的效率解释逐项列出；新技术选择不能静默更改这些比较口径。")
    module_rows=[{"id":m["id"],"title":m["title"],"summary":m.get("summary",""),"depends_on":", ".join(m.get("depends_on",[]))} for m in modules]
    option_rows=[]
    for m in modules:
        for d in m.get("decisions",[]):
            for o in d["options"]:
                option_rows.append({"module_id":m["id"],"decision_id":d["id"],"decision_title":d.get("title",d.get("question","")),"option_id":o["id"],"label":o.get("label",o.get("title","")),"status":o.get("status_label",o.get("status","")),"explanation":o.get("markdown",o.get("explanation",""))})
    queries["handoffs"]=query(catalog.get("handoffs",[]),["reports/discussions/q2-improvement/modules.json"],["module-handoffs"],[],"从每个模块输入输出列出实际交接及接收前检查，避免把候选路线当成已完成实验。")
    queries["modules"]=query(module_rows,["reports/discussions/q2-improvement/modules.json"],["module-dependency-map"],[],"分解已有求解过程的输入、输出与依赖；不将候选方法视为已运行结果。")
    queries["module_options"]=query(option_rows,["reports/discussions/q2-improvement/modules.json","reports/discussions/q2-improvement/discussion-options-log.md"],[f'module-source-{m["id"]}' for m in modules],[],"读取完整候选目录，会议选择默认留空；候选在新实验前不能写成性能改善。")
    queries["historical_decisions"]=query([{**r,"choice":r.get("selection","")+" · "+r.get("meaning","").split("；")[0],"origin":r.get("meaning","").split("；")[-1]} for r in catalog.get("historical_decisions",[])],["reports/discussions/q2-improvement/modules.json"],["historical-decisions"],[],"原样摘录既有实验决定，区分用户明确选择与继续后采用的默认选项。")
    intro="这份报告供小组逐项讨论第二问：先统一购电费用怎么算，再把数据、预测、优化和执行拆开分工。所有新路线都是待讨论候选，会议选择不会改动模型。"
    if template:
        intro="这是第二问人工讨论的可复用空模板。模块、解释与会议记录区可以沿用；原始数据和实测费用尚未绑定，图表不显示示例数值。"
    else:
        current=next(c for c in costs if c["id"]=="exp003_primary")
        previous=next(c for c in costs if c["id"]=="v2_primary")
        saving=previous["total_cost"]-current["total_cost"]
        intro=f'现有方案在 2—12 月的总购电费为 **{current["total_cost"]/10000:.2f} 万元**，比 v2 少 **{saving/10000:.2f} 万元**。但费用校准最终没有采用神经网络的修正量，结果与周期预测基本相同。本次讨论要决定下一轮先改变哪一部分，并明确各部分如何衔接。'
    target="## 01 · 第二问到底要做什么\n\n每天 00:00，先为当天的 144 个时段制定购电计划。附件1的分时电价已知；当天真实负载和光伏尚未发生，需要根据过去的数据估计。白天实际运行时，光伏、已购电量和储能共同满足负载；仍然缺电时向外网紧急购电，其单价为当时正常电价的 5 倍。\n\n**目标是使总购电费用尽可能低。** 少买可能增加紧急购电，多买则要为未使用的计划电量付费。因此，预测更复杂并不自动意味着总费用更低，所有模块最后都必须放到同一套结算规则下比较。\n\n逐时段费用 = 正常电价 × 原计划购电量 + 5 × 正常电价 × 紧急购电量。把正式评价期内所有时段相加，得到本报告比较的总购电费。"
    return {
        "title":"第二问人工讨论 · 可复用空模板" if template else "第二问改进：从购电费用出发组织分工",
        "surface":"report","status":"reviewed","buildStatus":"complete" if complete else "creating","report":{"asOf":"2025-12-31"} if quantitative else {},"filters":[],"queries":queries,
        "discussion":{
            "template":template,"intro":intro,"targetMarkdown":target,"modules":modules,
            "rawDates":evidence.get("raw",{}).get("dates",[]) if quantitative else [],"rawDefaultDate":"2025-03-20",
            "dispatchDates":list(evidence.get("dispatch_days",{})) if quantitative else [],
            "dispatchDateLabels":{date:f'{date} · {"题目指定日" if date in evidence.get("meta",{}).get("specified_dates",[]) else "当前总费用最高日"}' for date in evidence.get("dispatch_days",{})} if quantitative else {},
            "rawMarkdown":"原始数据覆盖 **2025 年 365 天**；正式费用比较覆盖 **2 月 1 日至 12 月 31 日，共 334 天**。一月用于训练、共同调参和储能预热。日曲线是事后已知的实际数据，用于理解供需变化，不能直接交给同日午夜的规划器。",
            "costMarkdown":"## 03 · 用总购电费判断已运行方案\n\n绑定已运行方案的费用证据后，这里按相同固定电价、储能物理边界和结算规则比较。图表分别展示计划购电费和紧急购电费，两项之和是总购电费。目前尚未绑定费用结果。" if template else "## 03 · 用总购电费判断现有方案\n\n下面四种方案使用相同固定电价、储能物理边界和结算规则。堆叠条形图把计划购电费和紧急购电费分开，两项之和才是总购电费。月份筛选只改变条形图和精确值表；下方月度趋势保留完整正式评价期。",
            "costCaveat":"**比较边界。** 当前方案采用 α负载 = 0、α光伏 = 0，即没有使用网络给出的修正量。它与周期基线仅有数值舍入差异，不构成对周期预测的实质改善。v2 与 exp003 的训练输入边界不同，所以两者费用差不能全部归因于费用校准。这里是已经查看过结果的回顾性比较，新候选尚未运行。" if quantitative else "尚无实测费用：添加证据后再写比较结论。不要用候选方案或演示值代替已运行结果。",
            "parallelMarkdown":catalog.get("parallel_markdown","数据口径模块先交付统一时间、单位和可用历史。预测与求解器可按统一接口并行实现；费用选择依赖预测候选及同一执行回放。执行模块固定后，最终由回放交付模块按全年相同条件核算。"),
            "methodMarkdown":"报告源文件随附 evidence.json、modules.json 与 discussion-options-log.md。build_report.py 从这些快照重建 HTML，不需要训练模型。build_evidence.py 记录证据如何从原始附件和已保存执行结果形成；evidence-audit.json 保存独立核验结果。\n\n本报告是第二问的讨论材料，只比较题目直接要求的总购电费及其计划、紧急两项费用；电量和储电量用于阅读题目表1—表3。模块选项的完整文本先保存在讨论选项日志，再用于本页。会议草稿与既有实验决定分开保存。",
        },
    }


def build_app(node, plugin, app, data, *, template=False):
    staging=ROOT/".work/q2-discussion"
    staging.mkdir(parents=True,exist_ok=True)
    snapshot_path=staging/("template-snapshot.json" if template else "report-snapshot.json")
    write(snapshot_path,data)
    if not app.exists():
        command([node,plugin/"scripts/prepare-data-app.mjs","--surface","report","--output",app,"--snapshot",snapshot_path])
    existing=read(app/"src/data.json")
    data["id"]=existing["id"]
    data["generatedAt"]=datetime.now(UTC).isoformat()
    write(app/"src/data.json",data)
    content=app/"src/content/report"
    for name in ("ReportContent.jsx","report.css"):
        shutil.copyfile(TEMPLATE/name,content/name)
    command([node,plugin/"scripts/data-app.mjs","build","--project-dir",app,"--separate-data"])
    offline=app/".data-app-offline/exports"/("template.html" if template else "report.html")
    command([node,plugin/"scripts/data-app.mjs","export-offline","--project-dir",app,"--output",offline])
    target=TEMPLATE/"template.html" if template else HERE/"report.html"
    shutil.copyfile(offline,target)
    return {"artifact_id":data["id"],"html":str(target.relative_to(ROOT)),"sha256":sha(target),"bytes":target.stat().st_size}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node",type=Path,default=DEFAULT_NODE)
    parser.add_argument("--plugin-root",type=Path,default=DEFAULT_PLUGIN)
    parser.add_argument("--complete",action="store_true")
    parser.add_argument("--template",action="store_true",help="Also rebuild the empty template")
    args=parser.parse_args()
    evidence=read(HERE/"evidence.json")
    catalog=read(HERE/"modules.json")
    record={"report":build_app(args.node,args.plugin_root,HERE/"app",snapshot(evidence,catalog,complete=args.complete)),"inputs":{"evidence.json":sha(HERE/"evidence.json")},"buildStatus":"complete" if args.complete else "creating"}
    if (HERE/"modules.json").exists():record["inputs"]["modules.json"]=sha(HERE/"modules.json")
    if args.template:
        record["template"]=build_app(args.node,args.plugin_root,ROOT/".work/q2-discussion/template-app",snapshot({},catalog,template=True,complete=args.complete),template=True)
    write(HERE/"report-build.json",record)
    print(json.dumps(record,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
