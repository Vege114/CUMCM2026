"""Rebuild the independent exp003 Q2 report from frozen evidence; never train/replay."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/experiments/exp003"
OUT = ROOT / "data/results/exp003"
BASE = OUT / "baseline"
NODE = Path.home() / ".cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node"
PLUGIN = Path.home() / ".codex/plugins/cache/openai-curated-remote/data-analytics/1.0.2"
TITLES = [
    "结论与成绩",
    "题目指标及信息边界",
    "数据与时间验证",
    "逐步技术讲解",
    "实验设置",
    "结果及失败案例",
    "历次指标和技术路线对比",
    "复现说明",
]
TARGETS = {"load": "负载", "pv": "光伏", "net_load": "净负载"}
LABELS = {
    "exp001_legacy_rebased": "v1旧预测·v2重算",
    "exp002_new_deterministic": "v2网络·确定性",
    "exp002_primary": "v2正式·风险调度",
    "exp002_periodic": "周期基线",
    "exp003_periodic": "周期基线",
    "exp003_uncalibrated": "Q2独立·未校准",
    "exp003_primary": "Q2独立·费用校准",
}
ORDER = {key: i for i, key in enumerate(LABELS)}
COLORS = {
    "v1旧预测·v2重算": "#9ba5ad",
    "v2网络·确定性": "#7188a3",
    "v2正式·风险调度": "#ab8461",
    "周期基线": "#9b9455",
    "Q2独立·未校准": "#599cad",
    "Q2独立·费用校准": "#247e77",
}
SPECIFIED = ["2025-03-20", "2025-06-21", "2025-09-23", "2025-12-21"]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def rows(frame):
    return (
        json.loads(frame.to_json(orient="records", double_precision=15))
        if isinstance(frame, pd.DataFrame)
        else frame
    )


def table(frame, columns):
    def fmt(value):
        if pd.isna(value):
            return "—"
        if isinstance(value, (bool, np.bool_)):
            return "是" if value else "否"
        return (
            f"{value:,.4f}"
            if isinstance(value, (int, float, np.number))
            else str(value).replace("|", "\\|")
        )

    return "\n".join(
        [
            "| " + " | ".join(columns.values()) + " |",
            "|" + "---|" * len(columns),
            *[
                "| " + " | ".join(fmt(row[key]) for key in columns) + " |"
                for row in frame.to_dict("records")
            ],
        ]
    )


def cost_frame(frame, experiment):
    frame = frame.copy()
    frame["experiment"] = experiment
    frame["policy_id"] = experiment + "_" + frame.name
    frame["policy_label"] = frame.policy_id.map(LABELS)
    frame["order"] = frame.policy_id.map(ORDER)
    for column in ("planned", "emergency", "total"):
        frame[column + "_wan"] = frame[column + "_cost"] / 10000
    if "month" in frame:
        frame["month_label"] = frame.month.map(lambda value: f"{value}月")
    return frame


def prepare_data(partial):
    manifest_file = OUT / "evaluation_manifest.json"
    evaluation = read_json(manifest_file) if manifest_file.exists() else {}
    complete = evaluation.get("status") == "complete"
    if not complete and not partial:
        raise RuntimeError(
            "Complete evaluation_manifest.json required; use --partial for the labelled historical preview"
        )
    if complete:
        for filename, expected in evaluation.get("output_hashes", {}).items():
            if digest(OUT / filename) != expected:
                raise RuntimeError(f"Evaluation output hash mismatch: {filename}")
    historical = pd.read_csv(BASE / "exp002_q2_costs.csv", dtype={"scenario": str})
    old_daily = pd.read_csv(BASE / "exp002_q2_daily.csv", dtype={"scenario": str})
    names = ["legacy_rebased", "new_deterministic", "primary"] + ([] if complete else ["periodic"])
    costs = cost_frame(historical[(historical.seed == 42) & historical.name.isin(names)], "exp002")
    daily = cost_frame(old_daily[(old_daily.seed == 42) & old_daily.name.isin(names)], "exp002")
    for frame in (costs, daily):
        frame.loc[frame.name == "legacy_rebased", "policy_id"] = "exp001_legacy_rebased"
        frame["policy_label"] = frame.policy_id.map(LABELS)
        frame["order"] = frame.policy_id.map(ORDER)
    if complete:
        costs = pd.concat(
            [
                costs,
                cost_frame(
                    pd.read_csv(OUT / "dispatch_metrics.csv", dtype={"scenario": str}), "exp003"
                ),
            ],
            ignore_index=True,
        )
        daily = pd.concat(
            [
                daily,
                cost_frame(
                    pd.read_csv(OUT / "daily_metrics.csv", dtype={"scenario": str}), "exp003"
                ),
            ],
            ignore_index=True,
        )
    costs = costs.sort_values(["order", "seed"])
    forecasts = {}
    for level in ("annual", "monthly"):
        legacy = pd.read_csv(BASE / f"forecast_{level}.csv")
        choices = {
            "exp001_selected": ("v1旧预测·v2重算", "exp001", "selected", "ensemble", 0),
            "exp002_seed_42": ("v2网络·确定性", "exp002", "primary", "42", 1),
        }
        if not complete:
            choices["attachment2_periodic"] = ("周期基线", "baseline", "periodic", "none", 4)
        legacy = legacy[legacy.predictor_id.isin(choices)].copy()
        for key, (label, experiment, name, seed, order) in choices.items():
            for field, value in (
                ("policy_label", label),
                ("experiment", experiment),
                ("name", name),
                ("seed", seed),
                ("order", order),
            ):
                legacy.loc[legacy.predictor_id == key, field] = value
        if complete:
            new = pd.read_csv(OUT / f"forecast_{level}.csv", dtype={"seed": str})
            new["experiment"] = "exp003"
            new["policy_label"] = new.name.map(lambda name: LABELS["exp003_" + name])
            new["order"] = new.name.map(lambda name: ORDER["exp003_" + name])
            new.loc[(new.name == "primary") & (new.seed != "42"), "policy_label"] += (
                " · " + new.loc[(new.name == "primary") & (new.seed != "42"), "seed"]
            )
            legacy = pd.concat([legacy, new], ignore_index=True)
        legacy["target_label"] = legacy.target.map(TARGETS)
        legacy["month_label"] = legacy.month.map(lambda value: f"{value}月")
        forecasts[level] = legacy.sort_values(["order", "seed", "target", "population", "month"])
    calibration = read_json(OUT / "alpha_calibration.json") if complete else None
    training = (
        read_json(OUT / "training_metadata.json")
        if (OUT / "training_metadata.json").exists()
        else []
    )
    return complete, evaluation, costs, daily, forecasts, calibration, training


def detail_rows(costs, calibration):
    main = costs[
        (costs.experiment == "exp003") & (costs.name == "primary") & (costs.seed == 42)
    ].iloc[0]
    dates = SPECIFIED + [str(main.worst_date)]
    with np.load(OUT / "dispatch_2.npz") as saved:
        arrays = {key: saved[key].copy() for key in saved.files}
    with np.load(OUT / "predictions.npz") as saved:
        prediction = np.maximum(
            0,
            saved["base"].astype(float)
            + np.asarray(calibration["selected_alpha"]) * saved["residual_42"].astype(float),
        )
        prediction[:, :, 1] = np.where(saved["daylight_mask"], prediction[:, :, 1], 0)
    intervals, states = [], []
    for date in dict.fromkeys(dates):
        i = (pd.Timestamp(date) - pd.Timestamp("2025-02-01")).days
        for t in range(144):
            intervals.append(
                {
                    "date": date,
                    "hour": (t + 0.5) / 6,
                    "plan_kwh": float(arrays["original"][i, t]),
                    "emergency_kwh": float(arrays["emergency"][i, t]),
                    "actual_net_kwh": float(
                        (arrays["actual"][i, t, 0] - arrays["actual"][i, t, 1]) / 6
                    ),
                    "predicted_net_kwh": float((prediction[i, t, 0] - prediction[i, t, 1]) / 6),
                }
            )
        states.extend(
            {"date": date, "hour": t / 6, "soc_kwh": float(arrays["states"][i, t])}
            for t in range(145)
        )
    return pd.DataFrame(intervals), pd.DataFrame(states), str(main.worst_date)


def paper_figures(costs, daily, forecasts, calibration, intervals, states, worst):
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    font = next(
        (
            p
            for p in (
                Path("/System/Library/Fonts/PingFang.ttc"),
                Path("/System/Library/Fonts/STHeiti Medium.ttc"),
            )
            if p.exists()
        ),
        None,
    )
    if font:
        font_manager.fontManager.addfont(font)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=font).get_name()
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.unicode_minus": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
        }
    )
    directory = REPORT / "figures"
    directory.mkdir(exist_ok=True)
    generated = []

    def save(fig, name):
        for ext in ("png", "svg"):
            fig.savefig(directory / f"{name}.{ext}", dpi=190, bbox_inches="tight")
        generated.append(name)
        plt.close(fig)

    official = costs[costs.seed == 42]
    fig, ax = plt.subplots(figsize=(10, 4.5), layout="constrained")
    y = np.arange(len(official))
    ax.barh(y, official.planned_wan, label="原计划费用", color="#5d9aae")
    ax.barh(
        y, official.emergency_wan, left=official.planned_wan, label="紧急购电费用", color="#ba9056"
    )
    ax.set(
        yticks=y,
        yticklabels=official.policy_label,
        xlabel="2025年2—12月费用（万元）",
        xlim=(0, official.total_wan.max() * 1.13),
    )
    for pos, value in zip(y, official.total_wan):
        ax.text(value + 1, pos, f"{value:.2f}", va="center", fontsize=9)
    ax.invert_yaxis()
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1), ncol=2, frameon=False)
    ax.grid(axis="x", alpha=0.15)
    save(fig, "q2-cost-components")
    monthly = (
        daily[daily.seed == 42].groupby(["policy_label", "month"])["total_cost"].sum().reset_index()
    )
    fig, ax = plt.subplots(figsize=(10, 4.2), layout="constrained")
    for label, part in monthly.groupby("policy_label", sort=False):
        ax.plot(
            part.month,
            part.total_cost / 10000,
            marker="o",
            ms=3,
            lw=1.6,
            color=COLORS[label],
            label=label,
        )
    ax.set(xticks=range(2, 13), xlabel="2025年月", ylabel="月度费用（万元）")
    ax.legend(ncol=3, fontsize=8)
    ax.grid(alpha=0.15)
    save(fig, "q2-monthly-cost")
    f = forecasts["monthly"]
    f = f[(f.population == "all") & f.seed.isin(["42", "none", "ensemble"])]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), layout="constrained")
    for ax, target in zip(axes, TARGETS):
        for label, part in f[f.target == target].groupby("policy_label", sort=False):
            ax.plot(
                part.month, part.rmse, color=COLORS[label], lw=1.5, marker="o", ms=2.5, label=label
            )
        ax.set(
            title=f"{TARGETS[target]} · 凌晨预测", ylabel="RMSE（kW）", xticks=[2, 4, 6, 8, 10, 12]
        )
        ax.grid(alpha=0.15)
    axes[0].legend(fontsize=7, loc="upper left")
    save(fig, "q2-monthly-error")
    if calibration:
        matrix = np.empty((3, 3))
        values = [0, 0.5, 1]
        for candidate in calibration["candidates"]:
            a, b = candidate["alpha"]
            matrix[values.index(b), values.index(a)] = candidate["cost"]
        fig, ax = plt.subplots(figsize=(6.5, 4.8), layout="constrained")
        image = ax.imshow(matrix, origin="lower", cmap="YlGnBu")
        ax.set(
            xticks=range(3),
            yticks=range(3),
            xticklabels=values,
            yticklabels=values,
            xlabel="负载修正强度 αL",
            ylabel="光伏修正强度 αV",
            title="1月25—31日共同调参段费用（元）",
        )
        for i in range(3):
            for j in range(3):
                selected = [values[j], values[i]] == calibration["selected_alpha"]
                ax.text(
                    j,
                    i,
                    f"{matrix[i, j]:,.0f}" + ("\n选定" if selected else ""),
                    ha="center",
                    va="center",
                    fontsize=11,
                    color="white"
                    if matrix[i, j] > (matrix.min() + matrix.max()) / 2
                    else "#242424",
                )
        fig.colorbar(image, ax=ax, label="七日总费用（元）")
        save(fig, "q2-alpha-calibration")
    if len(intervals):
        fig, axes = plt.subplots(
            4, 2, figsize=(12, 10), sharex=True, sharey="col", layout="constrained"
        )
        for date, row in zip(SPECIFIED, axes):
            part = intervals[intervals.date == date]
            state = states[states.date == date]
            row[0].plot(part.hour, part.actual_net_kwh, color="#8e959c", lw=1, label="实际净需求")
            row[0].plot(part.hour, part.plan_kwh, color="#247e77", lw=1.2, label="计划购电")
            row[0].plot(part.hour, part.emergency_kwh, color="#ba9056", lw=1.1, label="紧急购电")
            row[1].plot(state.hour, state.soc_kwh, color="#247e77", lw=1.5)
            row[1].axhline(1200, ls="--", color="#929aa1", lw=0.8)
            row[1].axhline(10800, ls="--", color="#929aa1", lw=0.8)
            row[0].set(title=date, ylabel="十分钟电量（kWh）")
            row[1].set(title=date, ylabel="储电量（kWh）", ylim=(0, 12000))
            for ax in row:
                ax.set(xticks=[0, 6, 12, 18, 24], xlim=(0, 24))
                ax.grid(alpha=0.12)
        axes[0, 0].legend(fontsize=8)
        axes[-1, 0].set_xlabel("小时")
        axes[-1, 1].set_xlabel("小时")
        save(fig, "q2-specified-days")
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, layout="constrained")
        part = intervals[intervals.date == worst]
        state = states[states.date == worst]
        for field, label, color in (
            ("actual_net_kwh", "实际净需求", "#8e959c"),
            ("predicted_net_kwh", "预测净需求", "#599cad"),
            ("plan_kwh", "计划购电", "#247e77"),
            ("emergency_kwh", "紧急购电", "#ba9056"),
        ):
            axes[0].plot(part.hour, part[field], label=label, color=color, lw=1.3)
        axes[0].set(title=f"正式策略最贵日：{worst}", ylabel="十分钟电量（kWh）")
        axes[0].legend(ncol=4, fontsize=8)
        axes[1].plot(state.hour, state.soc_kwh, color="#247e77")
        axes[1].set(
            ylabel="储电量（kWh）",
            xlabel="小时",
            ylim=(0, 12000),
            xticks=[0, 6, 12, 18, 24],
            xlim=(0, 24),
        )
        for value in (1200, 10800):
            axes[1].axhline(value, ls="--", color="#929aa1", lw=0.8)
        for ax in axes:
            ax.grid(alpha=0.15)
        save(fig, "q2-worst-day")
    return generated


def build(partial=False, snapshot_only=False, code_commit=None, node=None):
    REPORT.mkdir(parents=True, exist_ok=True)
    complete, evaluation, costs, daily, forecasts, calibration, training = prepare_data(partial)
    if complete and not partial:
        if read_json(OUT / "full_year_audit.json").get("status") != "passed":
            raise RuntimeError("Independent full-year audit must pass before final report")
        workbook_check = read_json(OUT / "verification.json")["result2.xlsx"]
        if workbook_check.get("saved_workbook_readback") != "passed":
            raise RuntimeError("Workbook readback must pass before final report")
        if digest(REPORT / "result2.xlsx") != workbook_check["sha256"]:
            raise RuntimeError("Report workbook does not match independently verified export")
    protocol = read_json(ROOT / "experiments/problem2/exp003/protocol.json")
    intervals, states, worst = (
        detail_rows(costs, calibration) if complete else (pd.DataFrame(), pd.DataFrame(), "")
    )
    evidence = REPORT / "evidence"
    evidence.mkdir(exist_ok=True)
    queries = {}
    outputs = {
        "cost_annual": costs,
        "cost_daily": daily,
        "forecast_annual": forecasts["annual"],
        "forecast_monthly": forecasts["monthly"],
    }
    for name, frame in outputs.items():
        frame.to_csv(evidence / f"{name}.csv", index=False)
    original = pd.read_csv(BASE / "exp001_original_q2_costs.csv")
    original.to_csv(evidence / "exp001_original_q2_costs.csv", index=False)
    training_frame = pd.DataFrame(
        [
            {
                key: row.get(key)
                for key in (
                    "month",
                    "seed",
                    "epochs",
                    "best_epoch",
                    "training_seconds",
                    "prediction_seconds",
                    "parameters",
                    "device",
                )
            }
            for row in training
        ]
    )
    if len(training_frame):
        training_frame.to_csv(evidence / "training.csv", index=False)
    cal = (
        pd.DataFrame(
            [
                {
                    "alpha_load": row["alpha"][0],
                    "alpha_pv": row["alpha"][1],
                    "load_label": f"αL={row['alpha'][0]:g}",
                    "pv_label": f"αV={row['alpha'][1]:g}",
                    "cost": row["cost"],
                    "selected": row["alpha"] == calibration["selected_alpha"],
                }
                for row in calibration["candidates"]
            ]
        )
        if calibration
        else pd.DataFrame()
    )
    if len(cal):
        cal.to_csv(evidence / "alpha_calibration.csv", index=False)
    for name, value in (
        ("protocol", protocol),
        ("baseline_boundaries", read_json(BASE / "baseline_boundaries.json")),
        ("baseline_manifest", read_json(BASE / "manifest.json")),
        ("evaluation_manifest", evaluation),
    ):
        (evidence / f"{name}.json").write_text(json.dumps(value, ensure_ascii=False, indent=2))
    for name in (
        "training_metadata.json",
        "model_checks.json",
        "input_isolation_check.json",
        "alpha_calibration.json",
        "prediction_archive.json",
        "full_year_audit.json",
        "verification.json",
        "workbook_export.json",
    ):
        if (OUT / name).exists():
            shutil.copy2(OUT / name, evidence / name)
    shutil.copy2(ROOT / "experiments/problem2/exp003/decision-log.md", REPORT / "decision-log.md")

    def query(name, frame, files, definition):
        queries[name] = {
            "rows": rows(frame),
            "source": {
                "label": definition.split("；")[0],
                "files": files,
                "metricDefinitions": [{"label": "指标与边界", "definition": definition}],
                "evidenceFlow": [
                    {
                        "title": "可复现来源",
                        "detail": "运行 reports/build_report_q2.py，从已完成评价和 baseline 午夜档案提取；本构建不训练、不回放。",
                    }
                ],
            },
            "methods": [{"language": "text", "code": definition}],
        }

    query(
        "cost_annual",
        costs,
        [
            "evidence/cost_annual.csv",
            "evidence/evaluation_manifest.json",
            "evidence/baseline_manifest.json",
        ],
        "费用元、电量kWh，2025年2—12月334天；v1旧预测按v2物理口径重算，v2历史网络存在共同早停耦合。相同物理/预热/执行/结算，可比较费用；训练边界不同，不能单独归因网络结构。",
    )
    query(
        "cost_daily",
        daily,
        ["evidence/cost_daily.csv", "evidence/protocol.json"],
        "每日一次固定计划；总费用=固定电价×(全额计划+5×紧急购电)。按同一策略和种子合计，跨日SOC连续；图表万元值由元除以10000。",
    )
    for level in ("annual", "monthly"):
        query(
            "forecast_" + level,
            forecasts[level],
            [f"evidence/forecast_{level}.csv", "evidence/baseline_boundaries.json"],
            "只评价0时发布、144个目标区间；负载/PV/净负载单位kW。MAE=绝对误差和/n，RMSE=平方误差和/n开方，WAPE=100×绝对误差和/实际绝对值和，bias=有符号误差和/n。PV>0人群依据实际光伏；净负载保留负值。月份聚合重算误差和，不平均百分比；旧网络共同早停边界保留。",
        )
    query(
        "original_cost",
        original,
        ["evidence/exp001_original_q2_costs.csv"],
        "v1原登记费用使用不同充放电效率、预热与执行协议；仅原始记录展示，不作直接排名。",
    )
    query(
        "protocol",
        [
            {"item": key, "value": json.dumps(value, ensure_ascii=False)}
            for key, value in protocol.items()
        ],
        ["evidence/protocol.json"],
        "本轮预先冻结的Q2配置；物理协议版本与算法版本分别记录。",
    )
    query(
        "training",
        training_frame,
        ["evidence/training.csv", "evidence/training_metadata.json"],
        "每行一个月和一个种子的实测训练任务；新模型两分支2178参数。只报告本轮Q2时间；历史四分支累计训练时间不是Q2专属耗时。",
    )
    if (REPORT / "relative_comparison.csv").exists():
        relative = pd.read_csv(REPORT / "relative_comparison.csv", dtype={"scenario": str})
        if set(relative.scenario) != {"2"}:
            raise RuntimeError("Historical comparison must contain Q2 only")
        query(
            "relative_comparison",
            relative,
            ["relative_comparison.csv"],
            "仅第二问；绝对变化=本次−此前，相对变化=100×绝对变化/此前绝对值。"
            "原登记v1费用不可直接排名；预测统一午夜重评；价格/能量/误差单位按每行unit保留。",
        )
    if len(cal):
        query(
            "calibration",
            cal,
            ["evidence/alpha_calibration.csv", "evidence/alpha_calibration.json"],
            "1月25—31日共同早停及费用调参段，九个αL/αV组合；相同起始SOC、同一确定性调度器。费用最低，并列1分钱内先取alpha总和小者；不使用正式年费用选参数。",
        )
    if complete:
        intervals.to_csv(evidence / "intervals.csv", index=False)
        states.to_csv(evidence / "states.csv", index=False)
        for name, frame in (
            ("specified_intervals", intervals[intervals.date.isin(SPECIFIED)]),
            ("specified_states", states[states.date.isin(SPECIFIED)]),
            ("worst_intervals", intervals[intervals.date == worst]),
            ("worst_states", states[states.date == worst]),
        ):
            query(
                name,
                frame,
                [
                    "evidence/intervals.csv",
                    "evidence/states.csv",
                    "evidence/evaluation_manifest.json",
                ],
                "正式种子42的逐区间回放；计划/应急/净需求为每十分钟kWh，SOC为145个边界状态；失败日按全年实际费用最大值选取，仅作诊断。",
            )
    figures = paper_figures(costs, daily, forecasts, calibration, intervals, states, worst)
    sections = [{"title": title, "blocks": []} for title in TITLES]
    markdown = [[] for _ in TITLES]

    def prose(section, text, sources=()):
        identity = f"q2-s{section + 1}-p{sum(block['type'] == 'prose' for block in sections[section]['blocks'])}"
        sections[section]["blocks"].append(
            {"type": "prose", "id": identity, "markdown": text, "queryIds": list(sources)}
        )
        markdown[section].append(text)

    def chart(section, identity, title, query_id, spec, scope=None, figure=None, height=330):
        sections[section]["blocks"].append(
            {
                "type": "chart",
                "id": identity,
                "title": title,
                "queryId": query_id,
                "spec": spec,
                "scope": scope,
                "height": height,
            }
        )
        if figure:
            markdown[section].append(
                f"![{title}](figures/{figure}.png)\n\n[论文SVG](figures/{figure}.svg)"
            )

    def add_table(section, identity, title, query_id, columns, frame=None, disclosure=False):
        sections[section]["blocks"].append(
            {
                "type": "table",
                "id": identity,
                "title": title,
                "queryId": query_id,
                "columns": list(columns.items()),
                "disclosure": disclosure,
            }
        )
        if not disclosure:
            markdown[section].append(
                table(
                    frame if frame is not None else pd.DataFrame(queries[query_id]["rows"]), columns
                )
            )

    for i, title in enumerate(TITLES):
        prose(i, f"## {i + 1}. {title}")
    main = costs[(costs.experiment == "exp003") & (costs.name == "primary") & (costs.seed == 42)]
    official = costs[costs.seed == 42]
    if complete:
        result = main.iloc[0]
        peer = official.set_index("policy_id")
        v2 = float(peer.loc["exp002_primary", "total_cost"])
        periodic = float(peer.loc["exp003_periodic", "total_cost"])
        uncal = float(peer.loc["exp003_uncalibrated", "total_cost"])
        old_det = float(peer.loc["exp002_new_deterministic", "total_cost"])
        change = lambda now, before: (
            f"{'降低' if now < before else '增加'}{abs(100 * (now / before - 1)):.3f}%"
        )
        prose(
            0,
            f"正式策略全年费用 **{result.total_cost / 10000:.2f}万元**，较v2正式风险策略{change(result.total_cost, v2)}，较周期基线费用变化{result.total_cost - periodic:+,.4f}元。紧急购电{result.emergency_kwh / 10000:.2f}万kWh，日费用CVaR90为{result.daily_cvar90:,.2f}元，较v2正式风险策略增加{100 * (result.daily_cvar90 / peer.loc['exp002_primary', 'daily_cvar90'] - 1):.3f}%。",
            ["cost_annual"],
        )
        prose(
            0,
            f"本轮仅优化第二问：将负载/PV训练与其他题目隔离，再用一月实际费用固定修正强度αL={calibration['selected_alpha'][0]:g}、αV={calibration['selected_alpha'][1]:g}。独立未校准模型较v2确定性策略费用变化{uncal - old_det:+,.2f}元；随后费用校准再变化{result.total_cost - uncal:+,.2f}元。后一步才是本轮修正强度校准的净贡献。",
            ["cost_annual", "calibration"],
        )
        if calibration["selected_alpha"] == [0, 0]:
            prose(
                0,
                "**费用校准拒绝了两个网络分支的残差，正式策略退回周期预测。** 因而本轮费用成绩不能归功于神经网络修正；保留未校准对照，是为了展示费用目标如何否决看似有用的预测调整。",
                ["calibration", "cost_annual"],
            )
            prose(
                0,
                f"正式策略较原始周期基线多{result.total_cost - periodic:,.4f}元，来自缓存基础预测的float32舍入及随后的调度数值变化，不代表实质改善。α为零使三个种子的正式预测相同，费用回放按相同预测复用；三份相同成绩不构成随机稳定性的独立证据。",
                ["cost_annual", "calibration"],
            )
        query(
            "primary_result",
            main,
            ["evidence/cost_annual.csv"],
            "本轮正式种子42；费用元，紧急电量kWh；固定期末只有物理边界，不额外计储电残值。",
        )
        add_table(
            0,
            "primary-result",
            "正式第二问成绩",
            "primary_result",
            {
                "planned_cost": "原计划费/元",
                "emergency_cost": "应急费/元",
                "total_cost": "总费用/元",
                "final_soc": "期末SOC/kWh",
            },
        )
    else:
        prose(
            0,
            "**本轮训练与评价尚未完成。当前仅展示已经核验的历史第二问证据，不包含exp003正式费用或改善结论。**",
        )
        old_main = official[official.policy_id == "exp002_primary"].iloc[0]
        old_periodic = official[official.policy_id == "exp002_periodic"].iloc[0]
        prose(
            0,
            f"v2正式风险策略费用{old_main.total_cost / 10000:.2f}万元，周期基线{old_periodic.total_cost / 10000:.2f}万元。新实验将分别辨认输入与早停修正、费用校准两层变化。",
            ["cost_annual"],
        )
    prose(
        1,
        "只用附件1固定电价和附件2已观测负载/PV；每天0时生成未来144区间并固定全日购电，日内不调整计划。总费用 C=Σp(g+5e)，计划电量全部付费，无售电收入。功率kW除以6得到十分钟电量kWh。",
        ["protocol"],
    )
    prose(
        1,
        "沿用v2团队物理假设：充放电效率各√0.9，往返90%；SOC为1200—10800kWh，充放电各不超过5000kW且互斥。2025年1月1日初始6000kWh，经共同周期基线预热后连续运行，日末只施加物理边界。充放电按当前净盈余与SOC贪心执行；这些假设限定本次费用比较。",
        ["protocol"],
    )
    prose(
        2,
        "正式回溯区间为2025年2—12月334天、48,096个十分钟区间。预测只评分凌晨窗口，实际PV>0人群为26,880区间；负载、PV与净负载的MAE/RMSE/bias均为kW，WAPE为%。净负载=负载−PV，不裁剪负值；平均偏差为预测−实际。",
        ["forecast_annual"],
    )
    prose(
        2,
        "训练样本的整个24小时标签不得越过训练截止点，标准化只拟合训练段；月末最近七个完整日期的凌晨窗口用于早停，当月冻结检查点。1月25—31日同时用于二月早停与费用校准，属于共同调参段；2025年既有结果已用于本轮诊断，因此报告称冻结配置后的时间顺序回溯评价，不称全新未接触测试集。",
        ["protocol"],
    )
    prose(
        3,
        "周期预测b先取负载上周同期与PV昨日同期。两个独立参数分支各采用16→32→16→1残差MLP；输出零初始化，学习标准化Huber损失并加入L2正则。两个分支的共同早停仅依赖负载/PV；附件3和附件4不进入本轮训练、损失、早停、签名或调度。",
        ["protocol"],
    )
    prose(
        3,
        "送入调度的预测为 max(0,b+αr)，光伏再施加仅由历史观测确定的夜间掩码。αL、αV各取0、0.5、1，九个组合用一月七天实际费用选一次，并列时优先较小修正。α=0保留周期基线，允许网络修正不被采用。对称误差下降并不保证购电费用下降；储能耦合下，单时段最优分位数不能直接替代全日求解。",
        ["protocol"],
    )
    prose(
        3,
        "正式与未校准策略使用同一确定性购电模型、同一因果执行器和两秒求解预算。详细损失、特征、校准规则与储能方程见[方法附录](methods.md)。",
        ["protocol"],
    )
    if training:
        seconds = sum(row["training_seconds"] for row in training)
        epochs_hit = sum(row["epochs"] == protocol["training"]["max_epochs"] for row in training)
        prose(
            4,
            f"已完成{len(training)}组两分支网络训练，2178参数/检查点；累计GPU训练{seconds / 60:.4f}分钟，{epochs_hit}组达到60轮上限。三个固定种子42/2026/3407分别评价，42为正式种子，另外两种子保留作敏感性记录，本次α为零不提供独立稳定性证据。历史共享四分支训练耗时不作为Q2专属效率比较。",
            ["training"],
        )
    else:
        prose(
            4,
            "冻结训练设置：11个月×3个种子、Adam学习率0.001、batch64、最多60轮、早停耐心6；训练计时将随正式产物补齐。",
            ["protocol"],
        )
    if len(cal):
        chart(
            4,
            "alpha-calibration",
            "九组修正强度的一月费用（元）",
            "calibration",
            {
                "type": "heatmap",
                "x": "load_label",
                "y": "cost",
                "series": "pv_label",
                "showValues": True,
                "valueDecimals": 0,
                "xLabel": "负载修正",
                "yLabel": "PV修正",
            },
            figure="q2-alpha-calibration",
            height=350,
        )
        add_table(
            4,
            "alpha-details",
            "九个候选的完整费用",
            "calibration",
            {"alpha_load": "αL", "alpha_pv": "αV", "cost": "一月费用/元", "selected": "是否选中"},
            disclosure=True,
        )
    chart(
        5,
        "cost-components",
        "费用组成对照（万元）",
        "cost_daily",
        {
            "type": "horizontalStackedBar",
            "x": "policy_label",
            "y": "planned_wan",
            "fields": ["planned_wan", "emergency_wan"],
            "legend": {"labels": {"planned_wan": "原计划费", "emergency_wan": "应急费"}},
            "colors": {"planned_wan": "#5d9aae", "emergency_wan": "#ba9056"},
            "valueDecimals": 2,
        },
        "cost-filtered",
        "q2-cost-components",
        390,
    )
    chart(
        5,
        "monthly-cost",
        "逐月购电费用（万元）",
        "cost_daily",
        {
            "type": "line",
            "x": "month_label",
            "y": "total_wan",
            "series": "policy_label",
            "stackable": False,
            "colors": COLORS,
            "yLabel": "万元",
        },
        "monthly-cost",
        "q2-monthly-cost",
    )
    chart(
        5,
        "monthly-error",
        "逐月凌晨预测误差",
        "forecast_monthly",
        {
            "type": "line",
            "x": "month_label",
            "y": "rmse",
            "series": "policy_label",
            "stackable": False,
            "colors": COLORS,
        },
        "forecast-monthly",
        "q2-monthly-error",
    )
    chart(
        5,
        "forecast-comparison",
        "所选范围预测误差",
        "forecast_monthly",
        {
            "type": "horizontalBar",
            "x": "policy_label",
            "y": "rmse",
            "stackable": False,
            "valueDecimals": 3,
        },
        "forecast-summary",
    )
    if complete:
        grid_spec = {
            "type": "line",
            "x": "hour",
            "y": "actual_net_kwh",
            "fields": ["actual_net_kwh", "plan_kwh", "emergency_kwh"],
            "stackable": False,
            "yLabel": "十分钟电量（kWh）",
            "xLabel": "小时",
            "colors": {
                "actual_net_kwh": "#8e959c",
                "plan_kwh": "#247e77",
                "emergency_kwh": "#ba9056",
                "predicted_net_kwh": "#599cad",
            },
            "legend": {
                "labels": {
                    "actual_net_kwh": "实际净需求",
                    "plan_kwh": "计划购电",
                    "emergency_kwh": "紧急购电",
                    "predicted_net_kwh": "预测净需求",
                }
            },
        }
        chart(
            5,
            "specified-grid",
            "指定日购电与净需求",
            "specified_intervals",
            grid_spec,
            "specified",
            "q2-specified-days",
        )
        chart(
            5,
            "specified-soc",
            "指定日储能状态",
            "specified_states",
            {"type": "line", "x": "hour", "y": "soc_kwh", "yLabel": "SOC（kWh）", "xLabel": "小时"},
            "specified",
        )
        result = main.iloc[0]
        prose(
            5,
            f"正式策略最贵日为{worst}，费用{result.worst_day_cost:,.2f}元。以下供需、购电与SOC取自该日真实回放，作为事后失败诊断；该日不参与重新选参。费用尾部CVaR90按全年最贵10%概率质量计算，包括边界日的部分权重。",
            ["cost_annual"],
        )
        chart(
            5,
            "worst-grid",
            f"最贵日{worst}：净需求与购电",
            "worst_intervals",
            {
                **grid_spec,
                "fields": ["actual_net_kwh", "predicted_net_kwh", "plan_kwh", "emergency_kwh"],
            },
            figure="q2-worst-day",
        )
        chart(
            5,
            "worst-soc",
            f"最贵日{worst}：145个SOC状态",
            "worst_states",
            {"type": "line", "x": "hour", "y": "soc_kwh", "yLabel": "SOC（kWh）", "xLabel": "小时"},
        )
    prose(
        6,
        "费用比较包含两层变化：v2确定性→Q2独立未校准，同时包含训练输入、预测发布验证边界修正；Q2未校准→费用校准只改变修正强度。v2正式风险策略额外改变调度器，不能把其求解耗时差归为预测网络提速。各方案使用相同物理、预热与结算；旧网络仍有附件3/4通过共同早停间接影响Q2的边界差异。",
        ["cost_annual"],
    )
    query(
        "history_cost",
        official,
        ["evidence/cost_annual.csv"],
        "全年正式种子42与历史正式记录的同物理费用；v1旧预测为v2协议重算，非v1原登记值。",
    )
    add_table(
        6,
        "history-costs",
        "仅第二问的全年费用与尾部风险",
        "history_cost",
        {
            "policy_label": "方案",
            "total_cost": "总费用/元",
            "daily_cvar90": "CVaR90/元",
            "emergency_kwh": "应急电量/kWh",
            "solve_execute_seconds": "组装求解执行/s",
        },
    )
    if complete:
        contribution = pd.DataFrame(
            [
                {
                    "step": label,
                    "planned_cost": float(
                        peer.loc[b, "planned_cost"] - peer.loc[a, "planned_cost"]
                    ),
                    "emergency_cost": float(
                        peer.loc[b, "emergency_cost"] - peer.loc[a, "emergency_cost"]
                    ),
                    "total_cost": float(peer.loc[b, "total_cost"] - peer.loc[a, "total_cost"]),
                }
                for a, b, label in [
                    ("exp002_new_deterministic", "exp003_uncalibrated", "独立训练与午夜早停"),
                    ("exp003_uncalibrated", "exp003_primary", "费用校准"),
                ]
            ]
        )
        query(
            "contributions",
            contribution,
            ["evidence/cost_annual.csv"],
            "后者减前者；总费变化等于计划费变化加应急费变化。负值表示该费用下降，不等同于统计因果结论。",
        )
        add_table(
            6,
            "contributions",
            "两层变化的费用分解",
            "contributions",
            {
                "step": "变化",
                "planned_cost": "计划费变化/元",
                "emergency_cost": "应急费变化/元",
                "total_cost": "总费变化/元",
            },
        )
    prose(
        6,
        f"v1原登记第二问费用{original.iloc[0].total_cost:,.2f}元，其效率与执行协议不同，仅保存在[原始记录](evidence/exp001_original_q2_costs.csv)，不纳入上述排名。历史预测均从原档案筛出0时窗口重新评分；旧四次发布统计不再沿用。完整三种子误差、偏差、费用及求解证书见[结果附表](appendix.md)。",
        ["original_cost", "forecast_annual"],
    )
    if "relative_comparison" in queries:
        add_table(
            6,
            "relative-details",
            "全部第二问相对比较",
            "relative_comparison",
            {
                "previous_experiment": "此前方案",
                "metric": "指标",
                "population": "人群",
                "previous": "此前",
                "current": "本次",
                "unit": "单位",
                "relative_change_pct": "变化/%",
                "comparison": "比较口径",
            },
            disclosure=True,
        )
    prose(
        7,
        "复现命令：`.venv/bin/python -m experiments.problem2.exp003.train` → `predict` → `evaluate`，随后运行`.venv/bin/python reports/build_report_q2.py`。本报告构建只读已冻结结果，不触发训练或回放。完整命令与缓存签名规则见项目实验README；每轮全部技术选项与选择保存在[决策日志](decision-log.md)。",
    )
    prose(
        7,
        "交付包括本八节正文、离线HTML、PNG/SVG、[result2.xlsx](result2.xlsx)、[方法附录](methods.md)、[结果附表](appendix.md)、[全部第二问比较](relative_comparison.csv)及evidence/机器可读快照。工作簿的全年费用与正文使用同一正式种子42逐区间结算；附表或多个工作表中的重复费用不可相加。",
    )
    if not complete:
        prose(7, "当前是标示未完成的历史预览；正式工作簿和新方案成绩需待评价完成后交付。")
    # Scope metric definitions to the actual source-backed consumers.
    for section in sections:
        for block in section["blocks"]:
            ids = [block["queryId"]] if "queryId" in block else block.get("queryIds", [])
            for query_id in ids:
                queries[query_id]["source"]["metricDefinitions"][0].setdefault(
                    "componentIds", []
                ).append(block["id"] if block["type"] != "prose" else block["id"] + "-evidence")
    snapshot = read_json(REPORT / "app/src/data.json")
    snapshot.update(
        title="第二问：独立预测与费用校准",
        surface="report",
        status="reviewed",
        buildStatus="complete" if complete else "creating",
        report={"asOf": "2025-12-31"},
        filters=[],
        queries=queries,
        reportContent=sections,
        specifiedDates=SPECIFIED,
    )
    encoded = json.dumps(snapshot, ensure_ascii=False, indent=2, allow_nan=False)
    (REPORT / "app/src/data.json").write_text(encoded)
    (REPORT / "reviewed.json").write_text(encoded)
    (REPORT / "report.md").write_text(
        "# exp003：第二问独立预测与费用校准\n\n"
        + "\n\n".join("\n\n".join(items) for items in markdown)
        + "\n"
    )
    for i, items in enumerate(markdown, 1):
        (REPORT / f"section-{i}.md").write_text("\n\n".join(items) + "\n")
    write_appendices(protocol, training_frame, costs, forecasts, cal)
    if complete:
        write_record(protocol, main, forecasts["annual"], training, code_commit)
    check = {
        "evaluation_complete": complete,
        "sections": len(sections),
        "scope": ["2"],
        "app_id": snapshot["id"],
        "queries": len(queries),
        "figures": figures,
        "rows": {key: len(value["rows"]) for key, value in queries.items()},
        "browser_checks": "pending_root_agent",
        "source_hashes": {
            str(p.relative_to(REPORT)): digest(p) for p in evidence.iterdir() if p.is_file()
        },
    }
    (REPORT / "report_build_checks.json").write_text(
        json.dumps(check, ensure_ascii=False, indent=2)
    )
    if not snapshot_only:
        app = REPORT / "app"
        command = [str(node or NODE), str(PLUGIN / "scripts/data-app.mjs")]
        subprocess.run(
            [*command, "build", "--project-dir", str(app), "--separate-data"], check=True
        )
        offline = app / ".data-app-offline/exports/report.html"
        subprocess.run(
            [*command, "export-offline", "--project-dir", str(app), "--output", str(offline)],
            check=True,
        )
        shutil.copy2(offline, REPORT / "report.html")
        for name in (
            "methods.md",
            "appendix.md",
            "decision-log.md",
            "result2.xlsx",
            "relative_comparison.csv",
        ):
            if (REPORT / name).exists():
                shutil.copy2(REPORT / name, app / "dist" / name)
        shutil.copytree(evidence, app / "dist/evidence", dirs_exist_ok=True)
    print(json.dumps(check, ensure_ascii=False))
    return check


def write_appendices(protocol, training, costs, forecasts, cal):
    methods = """# 第二问方法附录

## 信息、样本与特征

每个十分钟区间以结束时刻记录。预测发布为自然日0时；训练可使用0/6/12/18时历史窗口，但全部144标签须在训练边界内。每月从头训练，早停只检查最近七个完整日期的午夜窗口，标准化不读验证集。

每个目标16项特征：日内正余弦、星期正余弦、发布时刻正余弦、提前量、昨日同期、上周同期、最近观测、近6/24/168小时均值、近24/168小时标准差、周期基线。负载周期基线为上周同期，PV为昨日同期。只有附件1固定电价和附件2供需历史参与本轮。

## 两分支残差学习

两个独立参数分支各16→32→16→1，隐藏层ReLU，共2178参数。输出层零初始化。标准化残差目标z=(y−b)/σ；Huber(e)=e²/2（|e|≤1），否则|e|−1/2；训练损失平均144×2个目标，并加入隐藏层L2=0.0001。Adam学习率0.001、batch64、最多60轮、patience6恢复验证损失最低权重。两分支共享早停，但共享目标只含负载和PV。

网络标准化输出为fθ(x)，原始残差r=σfθ(x)，单位kW。最终预测max(0,b+αr)，PV的夜间掩码只查看发布前最多28个完整日相同区间是否出现正发电。费用校准施加于未裁剪原始残差，不把已裁剪后的预测误当原始残差。

## 九候选费用校准

αL与αV各取0/0.5/1；在1月25—31日、共同起始SOC、共同确定性调度器下回放。按七日实际总费用选最小者，1分钱内并列先取α之和小，再按αL、αV依次小。二月及之后固定该组合，三个随机种子共用正式种子42选出的强度。共同早停/费用调参段不是独立泛化评估；二月检查点用于一月留出段的离线选参，不声称一月当天已可部署。

## 费用目标为何会拒绝误差更小的模型

无储能单时段的期望费用为J(g)=pg+5pE[(N−g)₊]，g≥0；连续分布下内点导数J′(g)=p−5pP(N>g)，故最优非负购电为max(0,Q₀.₈(N))。这说明少买与多买的费用不对称，对称Huber误差并不直接代表经济目标。储能把全日决策及跨日状态耦合，不能机械逐时采用0.8分位数。

本轮在固定调度与执行映射下，直接最小化九个修正组合的留出段费用Ĵ(αL,αV)=Σday Cday(αL,αV)，以有限候选降低调参自由度。α=0抑制全部网络残差，α=1保留全部残差；实际选出两者均为零，只能说明该共同调参段更支持周期基线，不保证未来月份均有同样次序。

## 储能与费用

η=√0.9，q=g+(V−L)/6。q≥0时c=min(q,5000/6,(10800−E)/η)，d=0；q<0时d=min(−q,5000/6,(E−1200)η)，c=0；剩余缺口为紧急购电e。E'=E+ηc−d/η。物理边界、功率与充放电互斥独立核验。

全日计划g在0时确定，费用C=Σp(g+5e)；无调整费与售电。计划器使用确定性预测，执行器只读当前实际供需与当前SOC。1月共同周期预热，全年状态连续，日末仅物理边界。本轮不增加人为终端目标，也不计期末储电残值。两秒求解器预算与组装核验总时间分别记录；可行不等同于全局最优。

## 指标与解释范围

MAE=Σ|ŷ−y|/n；RMSE=√(Σ(ŷ−y)²/n)；WAPE=100Σ|ŷ−y|/Σ|y|；bias=Σ(ŷ−y)/n。净负载L−V保留负数。PV>0使用实际发电掩码，只作事后评价切片。全年指标汇总误差和，不平均月百分比。日费用CVaR90为最贵10%概率质量的加权均值，包括边界日期的部分质量。

v2网络的共同四目标早停可能受到附件3/4影响；凌晨重评分不能消除这个历史边界差异。v1原费用不可与新协议直接排名，legacy_rebased保留其旧月度预测选择后用v2物理口径重算。不同调度器的耗时差不能归因预测网络提速。结果是预先冻结配置后的回溯评价，不是新数据上的独立前瞻试验。
"""
    (REPORT / "methods.md").write_text(methods)
    parts = [
        "# 第二问结果附表",
        "所有费用单位元，电量kWh，预测MAE/RMSE/bias为kW，WAPE为%。空值不代表零。",
        "## 三种子及历史全年费用",
        table(
            costs,
            {
                "policy_label": "方案",
                "seed": "种子",
                "total_cost": "总费用",
                "daily_cvar90": "CVaR90",
                "emergency_kwh": "应急电量",
                "final_soc": "期末SOC",
                "solve_execute_seconds": "组装求解执行/s",
            },
        ),
        "## 午夜预测完整成绩",
        table(
            forecasts["annual"],
            {
                "policy_label": "方案",
                "seed": "种子",
                "target_label": "目标",
                "population": "人群",
                "n": "样本数",
                "mae": "MAE",
                "rmse": "RMSE",
                "wape_pct": "WAPE/%",
                "bias": "bias",
            },
        ),
    ]
    if len(training):
        parts += [
            "## 实测训练任务",
            table(
                training,
                {
                    "month": "月份",
                    "seed": "种子",
                    "epochs": "实际轮数",
                    "best_epoch": "保存轮数",
                    "training_seconds": "训练/s",
                    "prediction_seconds": "预测/s",
                    "parameters": "参数数",
                },
            ),
        ]
    parts += [
        "## 求解证书与回退",
        table(
            costs,
            {
                "policy_label": "方案",
                "seed": "种子",
                "solver_calls": "调用次数",
                "timeout_count": "超时",
                "fallback_count": "回退",
                "gap_certified_count": "差距达标次数",
            },
        ),
    ]
    if len(cal):
        parts += [
            "## 九组费用校准",
            table(
                cal, {"alpha_load": "αL", "alpha_pv": "αV", "cost": "一月费用", "selected": "选中"}
            ),
        ]
    (REPORT / "appendix.md").write_text("\n\n".join(parts) + "\n")


def write_record(protocol, main, forecast, training, code_commit):
    selected = forecast[
        (forecast.experiment == "exp003") & (forecast.name == "primary") & (forecast.seed == "42")
    ].copy()
    selected["role"] = "primary"
    selected["scenario"] = "2"
    selected["variant"] = "q2_calibrated_mlp"
    commit = (
        code_commit
        or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    )
    record = {
        "experiment_id": "exp003",
        "title": "第二问独立预测与费用校准",
        "scope": ["2"],
        "protocol": protocol,
        "code_commit": commit,
        "data_hashes": {
            name: digest(ROOT / "data/raw" / name) for name in protocol["allowed_inputs"]
        },
        "environment": training[0]["environment"],
        "seeds": protocol["seed_list"],
        "models": ["q2_periodic_residual_mlp"],
        "primary_seed": 42,
        "model_configuration": {
            "architecture": protocol["architecture"],
            "training": protocol["training"],
            "calibration": protocol["calibration"],
            "formal_training_groups": len(training),
            "selected_alpha": read_json(OUT / "alpha_calibration.json")["selected_alpha"],
            "selection_outcome": "alpha_zero_rejects_network_residuals"
            if read_json(OUT / "alpha_calibration.json")["selected_alpha"] == [0, 0]
            else "calibrated_network_residuals",
        },
        "metric_definitions": {
            "mae": "sum(abs(predicted-actual))/n",
            "rmse": "sqrt(sum((predicted-actual)^2)/n)",
            "wape_pct": "100*sum(abs(predicted-actual))/sum(abs(actual))",
            "bias": "sum(predicted-actual)/n",
            "forecast_sample": "midnight issues only, 144 same-day intervals, observed labels only",
            "total_cost": "sum(fixed_price*(original+5*emergency)); final=original",
            "daily_cvar90": "weighted mean of highest 10 percent daily cost probability mass, including fractional boundary day",
        },
        "forecast_metrics": rows(selected),
        "metrics": rows(main),
        "technical_path": [
            "Q2独立数据边界",
            "周期基线与两分支残差MLP",
            "午夜早停",
            "一月九候选费用校准",
            "固定全日计划与因果储能执行",
            "同口径午夜与费用评价",
        ],
        "artifacts": {
            "report": "experiments/exp003/report.md",
            "html": "experiments/exp003/report.html",
            "results": "experiments/exp003",
            "workbook": "experiments/exp003/result2.xlsx",
        },
        "historical_boundary_note": "Archived v1/v2 shared four-target stopping remains explicitly disclosed; original v1 fees are not directly ranked.",
    }
    (REPORT / "record.draft.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=["exp003"], default="exp003")
    parser.add_argument("--node", type=Path, default=NODE)
    parser.add_argument("--partial", action="store_true")
    parser.add_argument("--snapshot-only", action="store_true")
    parser.add_argument("--code-commit")
    args = parser.parse_args()
    build(args.partial, args.snapshot_only, args.code_commit, args.node)
