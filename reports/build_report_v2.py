"""Eight-section report for fixed architectures, stochastic dispatch and historical replay."""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd
from build_report import TITLES, markdown_table, presentation_blocks, records
from history import comparison_rows, read_registry
from record_v2 import make_record

ROOT = Path(__file__).resolve().parents[1]
NAMES = {"primary": "本轮正式策略", "new_deterministic": "新预测＋基础调度",
         "legacy_rebased": "旧正式预测＋基础调度", "periodic": "周期基础预测",
         "yesterday": "昨日同期", "weekly": "上周同期", "risk_zero": "风险权重为零",
         "no_anticipation": "不预先考虑更新", "raw_forecast": "原始光伏预报",
         "updates_0": "只在凌晨发布", "updates_0_6": "0、6 时发布",
         "updates_0_6_12": "0、6、12 时发布", "known_price": "已知未来价格（反事实）",
         "mlp": "固定轻量网络", "issued": "附件原始预报"}
TARGETS = {"load": "负载", "pv": "历史光伏", "pv_corrected": "光伏预报修正", "price": "电价"}
ROUTE = [
    ("对齐数据", "区间终点记录、24 个小时预报点", "功率除以 6；小时点线性积分；只用已观测锚点", "因果历史与区间预测"),
    ("建立周期预测", "昨日、上周及已发布光伏预报", "负载和电价取上周同期，历史光伏取昨日同期", "基础值 b"),
    ("构造 16 项特征", "对应变量的可观测历史与时钟", "训练历史标准化，提取滞后、水平、波动与周期", "每时段 4×16 输入"),
    ("训练固定小网络", "33 组月度训练与七日验证集", "四分支 32→16→1；零初始化输出；Huber 与 Adam", "固定种子 42 的正式检查点"),
    ("缓存修正预测", "基础预测及网络修正量", "反标准化并相加；光伏小时点积分成区间均值", "各问题共用一次生成的预测"),
    ("构建相关场景树", "最近最多 56 个完整历史日期", "联合残差收缩，至多 16 路径，仅随已揭示信息分支", "带非预见性约束的场景"),
    ("求解剩余购电", "当前 SOC、原计划与场景树", "期望费用＋λ×CVaR90；精确因果控制器；两秒预算", "经核验的可行计划及差距记录"),
    ("因果执行", "当前区间真实供需与已定购电量", "盈余充电，缺口放电后紧急购电；跨日传递 SOC", "最终执行与逐区间状态"),
    ("结算与核验", "原计划、最终计划、应急量、已到达电价", "全额计划费＋最终净调整费＋5倍紧急购电费", "费用、工作簿、报告同源数据"),
]


def table(frame, columns):
    return markdown_table(frame, columns) if len(frame) else "该项完整回放仍在运行，尚无全年汇总值。"


def build(experiment="exp002", partial=False, code_commit=None):
    out = ROOT / "data/results" / experiment
    report = ROOT / "reports/experiments" / experiment
    report.mkdir(parents=True, exist_ok=True)
    data_file = report / "app/src/data.json"
    snapshot = json.loads(data_file.read_text())
    snapshot.update(title="固定轻量网络与多阶段风险调度实验", surface="report",
                    status="reviewed", buildStatus="creating" if partial else "complete")
    queries = {}

    def query(name, frame, files, definition):
        queries[name] = {"rows": records(frame) if isinstance(frame, pd.DataFrame) else frame,
                         "source": {"label": name, "files": files,
                                    "metricDefinitions": [{"label": "口径", "definition": definition}]}}

    training = json.loads((out / "training_metadata.json").read_text())
    train = pd.DataFrame([{k: r[k] for k in ("month", "seed", "epochs", "best_epoch",
                             "training_seconds", "prediction_seconds", "parameters", "device")}
                          for r in training])
    old = read_registry(ROOT / "reports/registry", exclude=experiment)
    old_train = json.loads((ROOT / "data/results/exp001/training_metadata.json").read_text())
    timing = pd.DataFrame([{"experiment": "exp001", "groups": len(old_train),
                           "training_minutes": sum(r["seconds"] for r in old_train) / 60},
                          {"experiment": experiment, "groups": len(train),
                           "training_minutes": float(train.training_seconds.sum() / 60)}])
    forecast = pd.read_csv(out / "forecast_metrics.csv", dtype={"seed": str})
    forecast["model_label"] = [NAMES[v] + (f" · {s}" if v == "mlp" else "")
                               for v, s in zip(forecast.variant, forecast.seed)]
    forecast["month_label"] = forecast.month.map(lambda m: f"{m}月")
    annual_f = pd.read_csv(out / "annual_forecast_metrics.csv", dtype={"seed": str})
    annual_f["target_label"] = annual_f.target.map(TARGETS)
    annual_f["model_label"] = annual_f.variant.map(NAMES)
    annual_f["population_label"] = annual_f.population.map({"all": "全部区间", "generating": "实际发电区间"})
    seed_forecasts = annual_f[annual_f.variant == "mlp"].copy()
    seed_statistics = pd.read_csv(out / "seed_forecast_statistics.csv")
    seed_statistics["target_label"] = seed_statistics.target.map(TARGETS)
    seed_statistics["population_label"] = seed_statistics.population.map({"all": "全部区间", "generating": "实际发电区间"})
    main_f = annual_f[(annual_f.seed == "42") & (annual_f.population == "all")]
    calibration = json.loads((out / "risk_calibration.json").read_text())
    cal = pd.DataFrame([{"scenario": c["scenario"], "weight": r["weight"],
                         "cost": r["cost"], "selected": r["weight"] == c["selected_weight"]}
                        for c in calibration for r in c["candidates"]])
    cal["selected_label"] = cal.selected.map({True: "是", False: "否"})
    query("training", train, ["training_metadata.json"], "每行一个月份与种子，真实 GPU 训练时间，不含预测与调度。")
    query("timing", timing, ["training_metadata.json", "../exp001/training_metadata.json"], "各实验全部正式训练组的时间之和；非总墙钟耗时。")
    query("forecast", forecast, ["forecast_metrics.csv"], "按误差总和重新汇总；提前量分组与 all 不能重复求和；真实发电区间为实际光伏功率大于零。")
    query("calibration", cal, ["risk_calibration.json"], "1 月 25—31 日七天实际回放总费用，选择在 2 月 1 日完成，此后全年固定。")
    query("technical_path", [{"step": i+1, "title": t, "input": a, "operation": b, "output": c}
                              for i, (t, a, b, c) in enumerate(ROUTE)], ["protocol.json", "methods.md"], "源代码与已冻结实验协议定义的九步流程。")
    for name in ("forecast_decomposition", "training_trace", "tree_nodes", "official_forecast_comparison", "specified_dates",
                 "specified_intervals", "battery_blocks", "emergency_periods", "failure_intervals", "failure_storage", "solver_summary",
                 "phase_timing", "seed_cost_results", "seed_forecast_results", "seed_forecast_statistics", "monthly_dispatch", "core_contributions"):
        path = out / f"{name}.csv"
        if path.exists():
            query(name, pd.read_csv(path, dtype={"scenario": str}), [path.name],
                  "由本轮冻结预测及独立核验后的逐区间回放直接导出；数值保留原始精度。")
    complete = (out / "dispatch_metrics.csv").exists()
    if not partial and not complete:
        raise RuntimeError("Full dispatch results required; use --partial for a clearly marked preview")
    costs = pd.read_csv(out / "dispatch_metrics.csv", dtype={"scenario": str}) if complete else pd.DataFrame()
    daily = pd.read_csv(out / "daily_metrics.csv", dtype={"scenario": str}) if complete else pd.DataFrame()
    main = costs[(costs.name == "primary") & (costs.seed == 42)].copy() if complete else pd.DataFrame()
    if complete:
        for frame in (costs, daily):
            frame["model_label"] = frame.name.map(NAMES)
        query("cost_daily", daily, ["daily_metrics.csv"], "每行一个方案、问题、种子、日期；费用元，电量千瓦时。只在同一问题和种子内求和。")
        query("cost_annual", costs, ["dispatch_metrics.csv"], "全年 334 天的费用及尾部费用，尾部按最后 10% 概率质量计算。")
        current = make_record(experiment, code_commit)
        relative = pd.DataFrame(comparison_rows(old, current))
        relative.to_csv(report / "relative_comparison.csv", index=False)
        query("relative_history", relative, ["relative_comparison.csv", "record.draft.json"], "按正式策略及问题匹配；预测与调度分别检查可比性；不兼容时不计算相对排名。")
    metrics_columns = {"scenario": "问题", "planned_cost": "原计划费/元", "up_cost": "上调费/元",
                       "down_cost": "下调费/元", "emergency_cost": "紧急购电费/元",
                       "total_cost": "总费用/元", "emergency_kwh": "紧急购电量/kWh"}
    seeds = (costs[costs.name == "primary"].groupby("scenario").agg(
        mean_cost=("total_cost", "mean"), std_cost=("total_cost", "std"),
        mean_tail=("daily_cvar90", "mean"), std_tail=("daily_cvar90", "std")).reset_index()
             if complete else pd.DataFrame())
    history_rows = []
    for record in old:
        for metric in record["metrics"]:
            history_rows.append({"experiment": record["experiment_id"], "scenario": str(metric["scenario"]),
                                 "total_cost": metric["total_cost"], "route": " → ".join(record["technical_path"]),
                                 "comparability": "原记录保留；效率、光伏积分、执行与预热协议不同，不直接排名"})
    query("history", history_rows, [f"../../registry/{r['experiment_id']}.json" for r in old], "读取全部历史登记；旧值与新物理口径下的历史重算分列。")
    sections = ["" for _ in TITLES]
    sections[0] = f"""## 1. 结论与成绩

本轮固定四分支轻量网络，种子 42 是正式策略；另外两个种子用于稳定性评价。第一问沿用现有成果。正式评价为 2025 年 2 月 1 日至 12 月 31 日，共 334 天，每日 144 个区间。

已完成 {len(train)} 组 GPU 训练，累计 {train.training_seconds.sum()/60:.4f} 分钟；旧实验为 {len(old_train)} 组、{timing.iloc[0].training_minutes:.4f} 分钟。训练组数减少 80%，这不代表完整风险调度也同样节省时间。{'全部回放已完成。' if complete else '**全年调度仍在运行；此预览只展示已完成的训练、预测与一月校准，不能据此宣称全年费用改善。**'}

{table(main, metrics_columns)}

固定正式种子的全年预测成绩：

{table(main_f, {'target_label':'预测目标','mae':'MAE','rmse':'RMSE','wape_pct':'WAPE/%'})}
"""
    sections[1] = """## 2. 题目指标及信息边界

平均绝对误差 MAE = Σ|ŷ−y|/N；均方根误差 RMSE = √(Σ(ŷ−y)²/N)；加权绝对百分比误差 WAPE = 100Σ|ŷ−y|/Σ|y|。负载与光伏单位为千瓦，电价为元/千瓦时。按误差和重新合并月份，不平均月度百分比。预测覆盖四个发布时刻的未来 24 小时；跨年无实测的目标从分母中排除。调度每区间只结算一次，故预测样本数与调度区间数不同。

四次发布的误差统计用于统一比较预测算法；问题 2、4-2 的实际调度始终只采用凌晨预测和全日计划。它们不会利用日内新预测修改购电量。

购电量及储能状态单位为千瓦时。费用 = 原计划全额费＋最终上调量×1.5×电价＋最终下调量×0.5×电价＋紧急购电量×5×电价。日内中间调整不重复计费。问题 4 的未来价格必须预测，实际价格到达后才用于结算。

充、放电效率均为 √0.9，储电量 1200—10800 千瓦时，交流侧功率上限 5000 千瓦。日末仅施加物理边界，从 1 月 1 日的 6000 千瓦时连续传递，不每日重置。问题 2、4-2 的特征、预测与场景不读取附件 3；问题 3、4-3 只读取已经发布的版本。

实测日费用的 CVaR90 取最贵 10% 概率质量，按边界日期的部分权重计算；它与每次优化中场景费用的 CVaR 是不同层级，分别标明。
"""
    sections[2] = """## 3. 数据与时间验证

原始数据为 365 个自然日、每天 144 个十分钟区间。CSV 包含一列日期及 144 列区间终点值（00:10 至次日 00:00）；分离日期列后按自然日展开，得到 52560 条实际功率/价格记录。正式回放覆盖 48096 个区间。数据校验值保存于 data_hashes.json。

每个月初回看最近七个完整日期作为验证集，其起点为训练截止点，标签跨越截止点的发布样本剔除。检查点只在月初冻结；当月预测只用该月检查点。场景历史必须包括该历史日期所有预报版本的完整目标窗口，不能仅因日期已过去就读到其尚未实现的晚间目标。

风险权重在 2 月 1 日利用 1 月 25—31 日留出集选择。该留出集同时供二月检查点早停，因此一月校准不是独立的泛化成绩；正式二月及以后才是样本外回放。风险残差仍使用一月当时可以产生的周期基础预测，避免把二月模型倒灌为一月已经可用的模型。

额外扰动检查把二月及以后的实测与预报放大 100 倍，四个问题共 70 个一月决策时刻的预测、路径、概率与信息树逐值一致，训练标签和验证特征也不变。完整证据见 calibration_causality_check.json；购电非预见性另由小型可穷举实例核验。固定墙钟预算的两次搜索可能返回不同可行解，因此因果性检查比较信息和模型输入，并把数值求解重现性单独说明。

固定电价问题的树分支还额外检查了价格信息隔离：任意改变可变电价历史，问题 3 的实际供需场景与信息节点都必须不变。检查发现并修正过一次分支信息错误；受影响的六组问题 3 风险回放与其一月权重校准已作废重算。未受影响的缓存只有在源代码差异和逐月场景等价性通过检查后才保留，迁移记录见 causal_tree_fix_migration.json。所有正式成绩使用修正后的协议，训练仍为 33 组。

补充边界检查发现，非负裁剪之前的隐含误差不能当作已经观测到的发电或负载。修正后，观测与预报修订完全相同的路径不会分支；负载、光伏和价格裁剪均有回归测试。逐项比较保留了 10146 个等价日期，作废了 1694 个受影响日期，并重新校准问题 3、4-3，记录见 clipped_information_fix_migration.json。已知未来价格的反事实同样排除价格信息分支，记录见 known_price_information_migration.json。作废计算的耗时计入本轮总运行段，不隐藏重算成本。
"""
    sections[3] = "## 4. 逐步技术讲解\n\n" + (ROOT / "reports/templates/methods-neural-v2.md").read_text()
    if (out / "january_baselines.csv").exists():
        january = pd.read_csv(out / "january_baselines.csv")
        january["target_label"] = january.target.map(TARGETS)
        january["model_label"] = january.variant.map(NAMES)
        sections[2] += "\n一月历史基线诊断使用 1 月 8 日至 1 月 31 日 0 时的 93 个发布窗口，每个窗口的完整 24 小时标签都在一月内。只比较历史同期，不训练网络，也不使用二月以后的成绩选择路线。负载和电价的周同期误差更小，历史光伏的昨日同期误差更小；这一诊断支持方案中预先固定的周期组合。\n\n" + table(
            january, {"target_label": "变量", "model_label": "历史基线", "mae": "一月 MAE", "rmse": "一月 RMSE", "unit": "单位", "wape_pct": "一月 WAPE/%"})
    if (out / "worked_example.json").exists():
        example = json.loads((out / "worked_example.json").read_text())
        sections[3] += f"""\n### 4.9 一个实际样本如何通过网络

3 月 20 日 0 时发布、目标为 10:00—10:10 的负载预测：基础值为 {example['base_kw']:.4f} 千瓦，训练历史标准差 σ 为 {example['training_scale_kw']:.4f} 千瓦，网络输出的标准化修正为 {example['normalized_residual']:.6f}。还原后修正量为 {example['correction_kw']:.4f} 千瓦，因此预测为 {example['predicted_kw']:.4f} 千瓦；事后观测为 {example['actual_kw']:.4f} 千瓦。预测发布时没有读取这个事后观测。

该检查点第 1 轮训练目标（含 L2 正则）为 {example['initial_training_objective']:.6f}；共训练 {example['epochs']} 轮，恢复第 {example['best_epoch']} 轮，最低验证目标为 {example['best_validation_objective']:.6f}。16 项实际输入和完整数字保存于 worked_example.json。损失下降记录与“参数已更新”检查共同说明网络实际训练过，不能用它代替样本外费用评价。
"""
    sections[4] = f"""## 5. 实验设置

网络固定为四个独立的 16→32→16→1 分支，共 4356 个参数。11 个月×3 个种子 = 33 个正式训练组。Adam 学习率 0.001、批量 64、Huber 损失、最多 60 轮、早停耐心 6。{int((train.epochs==60).sum())} 组达到 60 轮上限，不能据此声称全部收敛。所有组验证了 GPU 输出、参数更新、零初始化及保存重载。

三层核心对照是“旧正式预测＋新口径基础调度”“新预测＋同一基础调度”“新预测＋风险调度”。旧网络不重新训练，按原月度正式选择读取档案。还保留昨日、周同期、周期基础和原始附件预报；种子 42 完成零风险权重、不预先考虑更新、原始光伏预报和 0/0+6/0+6+12/0+6+12+18 四种发布组合。问题 4 的已知价格仅是反事实诊断。

{table(cal, {'scenario':'问题','weight':'风险权重','cost':'一月验证费用/元','selected_label':'是否选定'})}

求解目标：期望费用＋λ×CVaR90，最多 16 条联合路径、56 天历史、两秒求解器预算、1% 相对差距目标。固定控制器区域 LP 给出可核验候选；未知最优差距保持空值。CPU 回放最初使用四个工作进程，保存并核验 5053 个已完成日期后以八个工作进程恢复；预测缓存共享复用，各求解器单线程。

{table(timing, {'experiment':'实验','groups':'正式训练组数','training_minutes':'累计训练分钟'})}
"""
    sections[5] = f"""## 6. 结果及失败案例

网页可按问题、月份、预测目标、种子、误差指标与实际发电时段筛选。全年预测原始分母、绝对误差和及平方误差和保存在 forecast_metrics.csv；图表与表格从同一数据重算。

{table(annual_f[(annual_f.seed.isin(['42','none']))], {'model_label':'预测方法','target_label':'目标','population_label':'时段','mae':'MAE','rmse':'RMSE','wape_pct':'WAPE/%'})}

三个固定种子的各自全年预测误差。MAE、RMSE 的单位随目标分别为千瓦或元/千瓦时，WAPE 为百分比；这三组独立成绩不构成预测集成：

{table(seed_forecasts, {'seed':'预测种子','target_label':'目标','population_label':'时段','mae':'MAE','rmse':'RMSE','wape_pct':'WAPE/%'})}

以下均值与样本标准差（ddof=1）在三个种子的全年指标之间计算，不是先平均预测再评价。WAPE 的标准差单位为百分点：

{table(seed_statistics, {'target_label':'目标','population_label':'时段','mae_mean':'MAE 均值','mae_std':'MAE 标准差','rmse_mean':'RMSE 均值','rmse_std':'RMSE 标准差','wape_mean':'WAPE 均值/%','wape_std':'WAPE 标准差/百分点'})}

三种子费用稳定性，标准差为样本标准差（ddof=1），不用于挑选正式种子。费用波动同时包含预测差异和限时求解的运行差异；三个重复不足以宣称广泛的统计显著性：

{table(seeds, {'scenario':'问题','mean_cost':'费用均值/元','std_cost':'费用标准差/元','mean_tail':'日费用CVaR90均值/元','std_tail':'尾部费用标准差/元'})}

{table(main, {'scenario':'问题','worst_date':'最差日期','worst_day_cost':'最差日费用/元','daily_cvar90':'日费用CVaR90/元','final_soc':'期末储电量/kWh','solver_calls':'求解次数','timeout_count':'超时次数','fallback_count':'回退次数','incumbent_count':'区域LP候选采用次数','gap_certified_count':'差距≤1%次数','solve_execute_seconds':'组装求解执行秒数'})}

[指定四日的完整表格](specified_dates.md)。全年紧急购电时段见 emergency_periods.csv，四小时充放电量见 battery_blocks.csv。超时不等同于回退；只有没有任何通过独立核验的可行解时才回退。采用区域 LP 候选且无全局差距时，报告不作最优性承诺。
"""
    sections[6] = f"""## 7. 历次指标和技术路线对比

自动读取此前全部 {len(old)} 次登记实验，原记录保持不变。正式策略按其角色匹配，网络名称不必相同。调度比较检查效率、结算、预热、时标与执行规则；预测误差另检查评价期、目标语义与数据，不因为电池效率变化就把负载误差判为不可比。

{table(pd.DataFrame(history_rows), {'experiment':'实验','scenario':'问题','total_cost':'原登记费用/元','route':'技术路线','comparability':'可比性说明'})}

旧正式预测经新光伏积分、共同一月预热和新储能协议后重新回放，形成“旧正式预测＋基础调度”。它与“新预测＋基础调度”的差异用于辨认预测贡献；再与正式风险调度比较，辨认调度贡献。历史重算不是旧实验原成绩，不覆盖旧注册记录。发布预报积分转换单独记录在 prediction_archive.json 与本轮协议。
"""
    bridge_file = out / "official_forecast_comparison.csv"
    if bridge_file.exists():
        bridge = pd.read_csv(bridge_file, dtype={"scenario": str})
        bridge["target_label"] = bridge.target.map(TARGETS)
        sections[6] += "\n正式预测的全年 WAPE 对比（光伏按新积分口径重算；完整 MAE/RMSE 见 CSV）：\n\n" + table(
            bridge[(bridge.metric == "wape_pct") & (bridge.population == "all")],
            {"scenario": "问题", "target_label": "变量", "previous": "旧正式预测/%", "current": "新正式预测/%", "relative_change_pct": "相对变化/%", "comparison": "口径"})
        regressions = bridge[(bridge.relative_change_pct > 0) & (bridge.metric == "rmse")]
        if len(regressions):
            details = [f"问题 {r.scenario} 的{TARGETS[r.target]}在{'全部区间' if r.population == 'all' else '实际发电区间'}的 RMSE 增加 {r.relative_change_pct:.3f}%"
                       for r in regressions.itertuples()]
            sections[5] += "\n预测误差并非全面改善。与旧正式预测按同一口径比较，" + "；".join(details) + "。这表明全年 WAPE 的下降不能代替大误差和实际发电时段的检查。\n"
    if complete:
        conclusions = []
        for row in main.itertuples():
            peers = costs[(costs.scenario == row.scenario) & (costs.seed == 42)].set_index("name")
            parts = []
            for name, label in (("legacy_rebased", "历史重算"), ("new_deterministic", "新预测基础调度"), ("periodic", "周期基线")):
                prior = float(peers.loc[name, "total_cost"])
                change = 100 * (row.total_cost / prior - 1)
                parts.append(f"较{label}{'降低' if change < 0 else '增加'} {abs(change):.3f}%")
            conclusions.append(f"问题 {row.scenario} 正式总费用 {row.total_cost:,.4f} 元，" + "，".join(parts) + "。")
        sections[0] += "\n全年费用结论（均在本轮统一协议内比较）：\n\n" + "\n\n".join(conclusions)
        sections[5] += "\n全部种子 42 对照的全年成绩（权重为零与正式策略相同时复用同一回放）：\n\n" + table(
            costs[costs.seed == 42].sort_values(["scenario", "total_cost"]),
            {"scenario": "问题", "model_label": "方案", "total_cost": "全年费用/元", "daily_cvar90": "日费用CVaR90/元", "emergency_kwh": "紧急购电/kWh", "fallback_count": "回退次数"})
        sections[5] += "\n三个固定种子的各自全年成绩：\n\n" + table(
            costs[costs.name == "primary"].sort_values(["scenario", "seed"]),
            {"scenario": "问题", "seed": "种子", "total_cost": "全年费用/元", "daily_cvar90": "日费用CVaR90/元", "final_soc": "期末储电量/kWh"})
        calls = int(main.solver_calls.sum())
        sections[5] += f"\n正式种子四个问题共 {calls} 次风险求解，其中 {int(main.timeout_count.sum())} 次超时、{int(main.fallback_count.sum())} 次回退，{int(main.gap_certified_count.sum())} 次有不超过 1% 的全局差距证书。两秒预算限制的是求解器阶段；组装、可行初始计划及独立核验另计。费用结果可以比较，不能把全部可行解称为全局最优解。\n"
        sections[5] += "\n失败案例按各问题最贵的实际日期固定选择。费用同时受真实净需求、电价、预测误差及日前计划影响；下图用于定位误差和储能不足的时段，不能单凭最贵日期归因于网络。预测误差降低也不保证结算费用降低，因为上调、下调与紧急购电具有不同代价。区间电量画在区间中点，储能轨迹使用 00:00—24:00 的 145 个实际状态节点。\n"
        for row in main.itertuples():
            base = costs[(costs.scenario == row.scenario) & (costs.seed == 42) & (costs.name == "new_deterministic")].iloc[0]
            delta = row.total_cost-base.total_cost
            if delta > 0:
                sections[5] += f"\n问题 {row.scenario} 的风险调度较同预测基础调度多花 {delta:,.4f} 元。实测费用变化分别为：原计划 {row.planned_cost-base.planned_cost:+,.4f} 元，上调 {row.up_cost-base.up_cost:+,.4f} 元，下调 {row.down_cost-base.down_cost:+,.4f} 元，紧急购电 {row.emergency_cost-base.emergency_cost:+,.4f} 元。这是可核算的费用来源；仅凭本轮结果，不能把原因唯一归于场景近似、信息树或求解时限中的某一项。\n"
        sections[6] += "\n\n三层核心调度对照（相同新物理与结算协议，全部固定种子 42）：\n\n" + table(
            costs[(costs.seed == 42) & costs.name.isin(["legacy_rebased", "new_deterministic", "primary"])],
            {"scenario": "问题", "model_label": "方案", "total_cost": "总费用/元", "daily_cvar90": "日费用CVaR90/元", "emergency_kwh": "紧急购电/kWh"})
        if (out / "core_contributions.csv").exists():
            sections[6] += "\n两步变化的费用分项（本步减前一步，负值表示节省；预测变化保持基础调度相同，调度变化保持新预测相同）：\n\n" + table(
                pd.read_csv(out / "core_contributions.csv", dtype={"scenario": str}),
                {"scenario": "问题", "step": "步骤", "planned_cost": "原计划变化/元", "up_cost": "上调变化/元", "down_cost": "下调变化/元", "emergency_cost": "紧急购电变化/元", "total_cost": "总费变化/元"})
    sections[7] = """## 8. 复现说明

统一入口：`.venv/bin/python -m experiments.common.neural_v2.run STAGE`。STAGE 依次为 prepare、train、predict、calibrate、replay、export、report；也支持 all。训练强制 GPU，各阶段可独立恢复，配置或源代码签名变化会拒绝复用旧缓存。详细命令、导出依赖及缓存约定见 experiments/common/neural_v2/README.md。

正式结果为 result2.xlsx、result3.xlsx、result4-2.xlsx、result4-3.xlsx。购电表的全天总费用已包含原计划、最终净调整和紧急购电，多个工作表中的重复总费用不能相加。逐日 CSV、指定日期表格、论文 PNG/SVG、四个工作簿和离线 report.html 同包交付；体积较大的逐区间 NPZ 留在实验分支的 data/results/exp002。主分支报告和全部网页筛选可以独立使用。

复现边界：有上限的混合整数求解会受到机器速度影响；固定种子与数据并不能保证每次在同一秒数预算下找到同一可行解。已有缓存可精确恢复已完成回放，从头重算需重新报告求解状态与耗时。所有最终核验结果保存为 verification.json、unit_tests.txt、browser_checks.json 和 consistency_checks.json。
"""
    if complete:
        sections[7] += f"\n本次冻结实验代码提交：`{current['code_commit']}`。预测检查点、上游档案、数据与源代码的 SHA-256 见 prediction_archive.json、data_hashes.json、source_hashes.json。\n"
    if (out / "phase_timing.csv").exists():
        sections[4] += "\n分阶段实测耗时（累计任务时间与墙钟时间不可直接相加）：\n\n" + table(
            pd.read_csv(out / "phase_timing.csv"), {"stage": "阶段", "minutes": "分钟", "scope": "计时口径"})
    narrative_sections = list(sections)
    figure_groups = {3: ["fixed-network-architecture", "baseline-correction-decomposition", "worked-example-training-loss", "scenario-information-tree"],
                     4: ["training-count-and-time"],
                     5: ["annual-three-error-metrics", "monthly-forecast-error", "lead-and-generating-error", "cost-components", "failure-case-and-storage", "cost-versus-tail-risk", "solver-gap-and-fallback", "daily-emergency-and-monthly-cost"],
                     6: ["three-layer-cost-comparison"]}
    for index, names in figure_groups.items():
        for name in names:
            if (report / "figures" / f"{name}.png").exists():
                sections[index] += f"\n\n![{name}](figures/{name}.png)\n\n[论文 SVG](figures/{name}.svg)\n"
    blocks = []
    for index, body in enumerate(sections, 1):
        (report / f"section-{index}.md").write_text(body)
        items = presentation_blocks(narrative_sections[index-1], index, queries, experiment)
        for j, item in enumerate(items):
            item["id"] = f"section-{index}-block-{j+1}"
            if item["type"] == "table":
                headers = [c[1] for c in item["columns"]]
                names = {"预测目标": "正式种子全年预测误差", "原计划费/元": "正式策略全年结算",
                         "一月验证费用/元": "风险权重的七日验证", "累计训练分钟": "正式训练规模与耗时",
                         "预测方法": "全年预测方法对照", "费用均值/元": "三个种子的费用稳定性",
                         "预测种子": "三个种子的各自全年预测误差", "MAE 均值": "三个种子的预测稳定性",
                         "最差日期": "失败日期、储能与求解统计", "原登记费用/元": "历次正式策略的原始登记",
                         "计时口径": "各阶段实测时间", "旧正式预测/%": "新旧正式预测的全年 WAPE",
                         "一月 MAE": "正式评价之前的一月基线诊断",
                         "原计划变化/元": "预测与调度的费用贡献", "总费用/元": "三层核心调度对照"}
                item["title"] = next((label for key, label in names.items() if key in headers), item["title"])
                if "全年费用/元" in headers:
                    item["title"] = "三个种子的各自全年成绩" if "种子" in headers else "全部正式种子对照"
        blocks.append({"title": TITLES[index-1], "blocks": items})
    blocks[3]["blocks"] = [{"type": "prose", "id": "technical-heading", "markdown": "## 4. 逐步技术讲解"},
                             {"type": "route", "id": "technical-route"},
                             {"type": "prose", "id": "technical-methods", "markdown": narrative_sections[3].split("\n\n", 1)[1]}]
    if "tree_nodes" in queries:
        blocks[3]["blocks"].append({"type": "tree", "id": "scenario-tree"})
    if "training_trace" in queries:
        blocks[3]["blocks"].append({"type": "chart", "id": "worked-training-loss", "queryId": "training_trace",
                                   "title": "实际样本使用的 3 月检查点 · 种子 42 · 训练过程",
                                   "spec": {"type": "line", "x": "epoch", "y": "loss", "fields": ["loss", "val_loss"],
                                            "stackable": False, "xLabel": "训练轮次", "yLabel": "标准化训练目标（含 L2 正则）",
                                            "legend": {"labels": {"loss": "训练目标", "val_loss": "验证目标"}}}})
    if "forecast_decomposition" in queries:
        blocks[3]["blocks"] += [
            {"type": "chart", "id": "forecast-decomposition", "queryId": "forecast_decomposition", "title": "种子 42 · 3 月 20 日凌晨预测的基础与修正", "scope": "decomposition",
             "spec": {"type": "line", "x": "hour", "y": "base", "fields": ["base", "prediction", "actual"], "stackable": False,
                      "legend": {"labels": {"base": "基础预测", "prediction": "网络修正后", "actual": "真实值"}}, "xLabel": "小时"}},
            {"type": "chart", "id": "residual-output", "queryId": "forecast_decomposition", "title": "种子 42 · 修正量分解", "scope": "decomposition",
             "spec": {"type": "line", "x": "hour", "y": "correction", "xLabel": "小时"}}]
    blocks[4]["blocks"] += [
        {"type": "chart", "id": "training-groups", "queryId": "timing", "title": "正式训练组数",
         "spec": {"type": "bar", "x": "experiment", "y": "groups", "yLabel": "组"}},
        {"type": "chart", "id": "training-minutes", "queryId": "timing", "title": "累计实测 GPU 训练耗时",
         "spec": {"type": "bar", "x": "experiment", "y": "training_minutes", "yLabel": "分钟"}}]
    blocks[5]["blocks"][1:1] = [
        {"type": "chart", "id": "forecast-monthly", "queryId": "forecast", "title": "分月预测误差", "scope": "forecast_month",
         "spec": {"type": "line", "x": "month_label", "y": "mae", "series": "model_label", "stackable": False}},
        {"type": "chart", "id": "forecast-lead", "queryId": "forecast", "title": "提前量误差", "scope": "forecast_lead",
         "spec": {"type": "line", "x": "lead", "y": "mae", "series": "model_label", "stackable": False}}]
    if complete:
        blocks[0]["blocks"].append({"type": "chart", "id": "core-bridge", "queryId": "cost_daily", "title": "三层核心对照", "scope": "bridge",
                                    "spec": {"type": "horizontalBar", "x": "model_label", "y": "total_cost", "yLabel": "元"}})
        blocks[5]["blocks"] += [
            {"type": "chart", "id": "cost-breakdown", "queryId": "cost_daily", "title": "费用分项", "scope": "cost",
             "spec": {"type": "horizontalStackedBar", "x": "model_label", "y": "planned_cost", "fields": ["planned_cost", "up_cost", "down_cost", "emergency_cost"], "yLabel": "元",
                      "legend": {"labels": {"planned_cost": "原计划", "up_cost": "上调", "down_cost": "下调", "emergency_cost": "紧急购电"}}}},
            {"type": "chart", "id": "emergency-daily", "queryId": "cost_daily", "title": "正式策略逐日紧急购电", "scope": "daily",
             "spec": {"type": "line", "x": "date", "y": "emergency_kwh", "yLabel": "千瓦时"}}]
        blocks[5]["blocks"] += [
            {"type": "chart", "id": "cost-tail-scatter", "queryId": "cost_annual", "title": "全年费用与日费用尾部风险", "scope": "annual",
             "spec": {"type": "scatter", "x": "total_cost", "y": "daily_cvar90", "series": "model_label", "xLabel": "全年费用（元）", "yLabel": "日费用 CVaR90（元）"}}]
    if "official_forecast_comparison" in queries:
        blocks[6]["blocks"].append({"type": "table", "id": "official-history-forecast", "queryId": "official_forecast_comparison", "title": "按正式策略匹配的历史预测比较", "scope": "case",
                                    "columns": [["target", "目标"], ["population", "时段"], ["metric", "指标"], ["previous", "旧正式预测"], ["current", "新正式预测"], ["relative_change_pct", "变化/%"], ["comparison", "比较口径"]]})
    if "relative_history" in queries:
        blocks[6]["blocks"].append({"type": "table", "id": "all-history-comparability", "queryId": "relative_history", "title": "此前全部实验的指标可比性", "scope": "history",
                                    "columns": [["previous_experiment", "旧实验"], ["task", "任务"], ["route", "策略角色"], ["metric", "指标"], ["previous", "此前"], ["current", "本次"], ["relative_change_pct", "相对变化/%"], ["comparison", "差异说明"]]})
    if "specified_intervals" in queries:
        for name, title, columns in (
            ("specified_intervals", "指定时段购电", [["period", "时段"], ["original", "凌晨计划/kWh"], ["final", "最终购电/kWh"]]),
            ("specified_dates", "全天汇总", [["planned_kwh", "原计划/kWh"], ["final_kwh", "最终购电/kWh"], ["total_cost", "总费用/元"], ["initial_soc", "日初储电量/kWh"], ["final_soc", "日末储电量/kWh"]]),
            ("battery_blocks", "四小时储能充放电", [["period", "时段"], ["charge_kwh", "充电/kWh"], ["discharge_kwh", "放电/kWh"]]),
            ("emergency_periods", "紧急购电连续区间", [["period", "时段"], ["emergency_kwh", "紧急购电/kWh"]])):
            blocks[5]["blocks"].append({"type": "table", "id": name.replace("_", "-"), "queryId": name,
                                        "title": title, "scope": "specified", "columns": columns})
    if "failure_intervals" in queries:
        blocks[5]["blocks"] += [
            {"type": "chart", "id": "failure-energy", "queryId": "failure_intervals", "title": "最贵日期的供需与购电", "scope": "case",
             "spec": {"type": "line", "x": "hour", "y": "actual_net", "fields": ["actual_net", "predicted_net", "final", "emergency"], "stackable": False, "yLabel": "十分钟电量（kWh）",
                      "legend": {"labels": {"actual_net": "实际净需求", "predicted_net": "凌晨预测净需求", "final": "最终购电", "emergency": "紧急购电"}}}},
            {"type": "chart", "id": "failure-soc", "queryId": "failure_storage", "title": "最贵日期的实际储电量 · 145 个状态节点", "scope": "case",
             "spec": {"type": "line", "x": "hour", "y": "soc", "xLabel": "小时（状态时刻）", "yLabel": "千瓦时"}}]
    if "solver_summary" in queries:
        blocks[5]["blocks"].append({"type": "table", "id": "solver-summary", "queryId": "solver_summary", "title": "三个种子的求解与证书记录", "scope": "case",
                                    "columns": [["seed", "种子"], ["calls", "调用次数"], ["timed_out", "超时"], ["fallback", "回退"], ["gap_known", "有全局差距"], ["gap_met", "差距≤1%"], ["mean_solver_seconds", "平均秒数"]]})
    snapshot["queries"] = queries
    snapshot["reportContent"] = blocks
    encoded = json.dumps(snapshot, ensure_ascii=False, indent=2, allow_nan=False)
    data_file.write_text(encoded)
    (report / "reviewed.json").write_text(encoded)
    (report / "report.md").write_text("# exp002：固定轻量网络与多阶段风险调度\n\n" + "\n\n".join(sections))
    for path in out.glob("*"):
        if path.suffix in (".json", ".csv", ".xlsx"):
            shutil.copy2(path, report / path.name)
    shutil.copy2(ROOT / "experiments/common/neural_v2/protocol.json", report / "protocol.json")
    shutil.copy2(ROOT / "reports/templates/methods-neural-v2.md", report / "methods.md")
    if (out / "README.md").exists():
        shutil.copy2(out / "README.md", report / "data_dictionary.md")
    return snapshot


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", default="exp002")
    p.add_argument("--partial", action="store_true")
    p.add_argument("--node")
    p.add_argument("--data-plugin")
    p.add_argument("--snapshot-only", action="store_true")
    p.add_argument("--code-commit")
    p.add_argument("--title", help="Accepted for the common builder; exp002 retains its established title")
    args = p.parse_args()
    from v2_evidence import derive, figures
    derive(args.experiment)
    figures(args.experiment)
    build(args.experiment, args.partial, args.code_commit)
    if not args.snapshot_only:
        node = Path(args.node) if args.node else Path.home() / ".cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node"
        plugin = Path(args.data_plugin) if args.data_plugin else Path.home() / ".codex/plugins/cache/openai-curated-remote/data-analytics/1.0.2"
        if not node.is_file() or not (plugin / "scripts/data-app.mjs").is_file():
            p.error("请通过 --node 和 --data-plugin 指定已安装的 Codex 捆绑运行时；或只生成 --snapshot-only。")
        app = ROOT / "reports/experiments" / args.experiment / "app"
        command = [str(node), str(plugin / "scripts/data-app.mjs")]
        subprocess.run([*command, "build", "--project-dir", str(app), "--separate-data"], check=True)
        offline = app / ".data-app-offline/exports/report.html"
        subprocess.run([*command, "export-offline", "--project-dir", str(app), "--output", str(offline)], check=True)
        shutil.copy2(offline, app.parent / "report.html")


if __name__ == "__main__":
    main()
