"""Derived, inspectable evidence for exp002; never retrains or changes replay caches."""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))

from experiments.common.neural_v2.data import EPOCH, Data, month_origins  # noqa: E402
from experiments.common.neural_v2.predict import LegacyStore  # noqa: E402
from experiments.common.neural_v2.scenarios import ScenarioFactory  # noqa: E402

SCENARIOS = ("2", "3", "4-2", "4-3")
LABELS = {"load": "负载", "pv": "历史光伏", "pv_corrected": "光伏预报修正", "price": "电价"}


class ReportForecastArchive:
    """Read published predictions so rebuilding a report needs no local training cache."""

    def __init__(self, data, directory, seed=42):
        self.data = data
        with np.load(directory / "predictions.npz") as saved:
            self.origins = saved["origins"].copy()
            self.predictions = saved[f"seed_{seed}"].copy()
        self.lookup = {int(origin): i for i, origin in enumerate(self.origins)}

    def get(self, origin, issued=True, raw_pv=False):
        if origin < 31*144:
            return self.data.baseline(origin, issued=issued)
        result = self.predictions[self.lookup[int(origin)], :, :4 if issued else 3].copy().astype(float)
        if raw_pv and issued:
            result[:, 3] = self.data.integrate_points(self.data.issued_points(origin), origin)
        return result


def derive(run_id="exp002"):
    out = ROOT / "data/results" / run_id
    annual = pd.read_csv(out / "annual_forecast_metrics.csv", dtype={"seed": str})
    seed_forecasts = annual[annual.variant == "mlp"].copy()
    seed_forecasts.to_csv(out / "seed_forecast_results.csv", index=False)
    summary = seed_forecasts.groupby(["target", "population"]).agg(
        seeds=("seed", "nunique"), mae_mean=("mae", "mean"), mae_std=("mae", "std"),
        rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
        wape_mean=("wape_pct", "mean"), wape_std=("wape_pct", "std")).reset_index()
    assert (summary.seeds == 3).all()
    summary.to_csv(out / "seed_forecast_statistics.csv", index=False)
    training = json.loads((out / "training_metadata.json").read_text())
    calibration = json.loads((out / "risk_calibration.json").read_text())
    timing_rows = [
        {"stage": "训练", "minutes": sum(r["training_seconds"] for r in training)/60,
         "scope": "33 组 GPU 训练任务累计，不含特征准备、预测及调度"},
        {"stage": "月度预测", "minutes": sum(r["prediction_seconds"] for r in training)/60,
         "scope": "33 个检查点首次推理累计，正式预测缓存只生成一次"},
        {"stage": "一月风险校准", "minutes": sum(d["solve_execute_seconds"] for c in calibration for r in c["candidates"] for d in r["days"])/60,
         "scope": "4 个问题×3 权重×7 日期的组装、求解和回放任务累计，非并行墙钟时间"},
    ]
    migration_file = out / "causal_tree_fix_migration.json"
    if migration_file.exists():
        migration = json.loads(migration_file.read_text())
        timing_rows[2]["minutes"] += migration["retired_calibration_task_seconds"]/60
        timing_rows[2]["scope"] += "；包含已作废的一次问题 3 初步校准"
    clipping_file = out / "clipped_information_fix_migration.json"
    if clipping_file.exists():
        clipping = json.loads(clipping_file.read_text())
        timing_rows[2]["minutes"] += clipping["retired_calibration_task_seconds"]/60
        timing_rows[2]["scope"] += "；也包含裁剪信息修正前已作废的问题 3、4-3 校准"
    data = Data()
    january_origins = np.arange(7*144, 31*144-144+1, 36)
    january_ids = january_origins[:, None] + np.arange(144)
    assert january_ids.max() < 31*144
    january_truth = data.actual[january_ids]
    january = []
    for kind in ("yesterday", "weekly", "periodic"):
        baseline = np.stack([data.baseline(int(origin), kind, issued=False) for origin in january_origins])
        for channel, target in enumerate(("load", "pv", "price")):
            error = baseline[:, :, channel]-january_truth[:, :, channel]
            january.append({"variant": kind, "target": target, "n": error.size, "origins": len(january_origins),
                            "mae": float(np.abs(error).mean()), "rmse": float(np.sqrt(np.square(error).mean())),
                            "wape_pct": float(100*np.abs(error).sum()/np.abs(january_truth[:, :, channel]).sum()),
                            "unit": "元/kWh" if target == "price" else "kW"})
    pd.DataFrame(january).to_csv(out / "january_baselines.csv", index=False)
    store = ReportForecastArchive(data, out, 42)
    day = (pd.Timestamp("2025-03-20") - EPOCH).days
    origin = day * 144
    pred = store.get(origin)
    baseline = data.baseline(origin)
    decomposition = []
    for j, target in enumerate(("load", "pv", "price", "pv_corrected")):
        actual = data.actual[origin:origin+144, [0, 1, 2, 1][j]]
        for t in range(144):
            decomposition.append({"target": target, "hour": (t+.5)/6,
                "base": float(baseline[t, j]), "correction": float(pred[t, j]-baseline[t, j]),
                "prediction": float(pred[t, j]), "actual": float(actual[t])})
    pd.DataFrame(decomposition).to_csv(out / "forecast_decomposition.csv", index=False)
    metadata = next(row for row in training if row["month"] == 3 and row["seed"] == 42)
    pd.DataFrame({"epoch": np.arange(1, metadata["epochs"]+1), "loss": metadata["history"]["loss"],
                  "val_loss": metadata["history"]["val_loss"]}).to_csv(out / "training_trace.csv", index=False)
    x, _, _, scale = data.features([origin], int(metadata["train_cutoff"]))
    slot = 60
    (out / "worked_example.json").write_text(json.dumps({
        "date": "2025-03-20", "issue_hour": 0, "target_interval": "10:00-10:10", "target": "load",
        "feature_names": ["target_day_sin", "target_day_cos", "target_week_sin", "target_week_cos",
                          "issue_day_sin", "issue_day_cos", "lead_fraction_of_day",
                          "yesterday_standardized", "weekly_standardized", "last_observation_standardized",
                          "last_6h_mean_standardized", "last_24h_mean_standardized", "last_7d_mean_standardized",
                          "last_24h_std_scaled", "last_7d_std_scaled", "base_standardized"],
        "features": x[0, slot, 0].astype(float).tolist(),
        "base_kw": float(baseline[slot, 0]), "correction_kw": float(pred[slot, 0]-baseline[slot, 0]),
        "training_scale_kw": float(scale[0]), "normalized_residual": float((pred[slot, 0]-baseline[slot, 0])/scale[0]),
        "predicted_kw": float(pred[slot, 0]), "actual_kw": float(data.actual[origin+slot, 0]),
        "initial_training_objective": metadata["history"]["loss"][0],
        "best_validation_objective": min(metadata["history"]["val_loss"]),
        "best_epoch": metadata["best_epoch"], "epochs": metadata["epochs"]}, indent=2))
    tree = ScenarioFactory(data, store).build(origin, pred, "4-3")
    tree_rows = []
    for slot in (0, 36, 72, 108):
        for node in np.unique(tree["groups"][:, slot]):
            members = np.flatnonzero(tree["groups"][:, slot] == node)
            tree_rows.append({"hour": slot//6, "node": int(node), "path_count": len(members),
                              "probability": float(tree["probabilities"][members].sum()),
                              "members": members.tolist()})
    children = {}
    for edge in tree["metadata"]["tree"]:
        children.setdefault(edge["parent"], []).append(edge["node"])
    leaves = []

    def visit(node):
        if node not in children:
            leaves.append(node)
        else:
            for child in sorted(children[node]):
                visit(child)

    visit(0)
    terminal = {r["node"]: r for r in tree_rows if r["hour"] == 18}
    path_order = {member: i for i, leaf in enumerate(leaves) for member in terminal[leaf]["members"]}
    for row in tree_rows:
        row["layout_y"] = float(np.mean([path_order[m] for m in row["members"]]))
    (out / "tree_example.json").write_text(json.dumps({"metadata": tree["metadata"], "nodes": tree_rows}, indent=2))
    pd.DataFrame([{**r, "members": ",".join(map(str, r["members"]))} for r in tree_rows]).to_csv(out / "tree_nodes.csv", index=False)
    # Explicit bridge by historical official policy, not by an architecture name.
    legacy = LegacyStore(data)
    origins = np.concatenate([month_origins(m) for m in range(2, 13)])
    ids = origins[:, None] + np.arange(144)[None, :]
    valid = ids < len(data.actual)
    truth = data.actual[np.minimum(ids, len(data.actual)-1)]
    with np.load(out / "predictions.npz") as f:
        current = f["seed_42"].copy()
    comparisons = []
    for scenario in SCENARIOS:
        prior = np.stack([legacy.get(int(o), scenario) for o in origins])
        targets = [(0, 0, "load"), (3, 1, "pv_corrected")] if scenario in ("3", "4-3") else [(0, 0, "load"), (1, 1, "pv")]
        if scenario.startswith("4"):
            targets.append((2, 2, "price"))
        for j, yj, target in targets:
            for population in ("all", "generating") if target.startswith("pv") else ("all",):
                mask = valid & (truth[:, :, yj] > 0) if population == "generating" else valid
                a, b = prior[:, :, j][mask]-truth[:, :, yj][mask], current[:, :, j][mask]-truth[:, :, yj][mask]
                denom = np.abs(truth[:, :, yj][mask]).sum()
                for metric, av, bv in (("mae", np.abs(a).mean(), np.abs(b).mean()),
                                      ("rmse", np.sqrt(np.square(a).mean()), np.sqrt(np.square(b).mean())),
                                      ("wape_pct", 100*np.abs(a).sum()/denom, 100*np.abs(b).sum()/denom)):
                    comparisons.append({"scenario": scenario, "role": "primary", "target": target, "population": population,
                        "metric": metric, "previous": float(av), "current": float(bv), "relative_change_pct": float(100*(bv-av)/abs(av)),
                        "comparison": "同评价区间与单位的历史重算；旧光伏十分钟点积分" if target == "pv_corrected" else "同口径正式预测；保留旧月度选择"})
    pd.DataFrame(comparisons).to_csv(out / "official_forecast_comparison.csv", index=False)
    if (out / "dispatch_metrics.csv").exists():
        costs = pd.read_csv(out / "dispatch_metrics.csv", dtype={"scenario": str})
        daily = pd.read_csv(out / "daily_metrics.csv", dtype={"scenario": str})
        main = costs[(costs.name == "primary") & (costs.seed == 42)]
        contributions = []
        for scenario in SCENARIOS:
            peers = costs[(costs.scenario == scenario) & (costs.seed == 42)].set_index("name")
            for previous, current, label in (("legacy_rebased", "new_deterministic", "预测变化"),
                                             ("new_deterministic", "primary", "调度变化")):
                contributions.append({"scenario": scenario, "step": label, "previous": previous, "current": current,
                    **{key: float(peers.loc[current, key]-peers.loc[previous, key])
                       for key in ("planned_cost", "up_cost", "down_cost", "emergency_cost", "total_cost", "emergency_kwh")}})
        pd.DataFrame(contributions).to_csv(out / "core_contributions.csv", index=False)
        cases, storage = [], []
        for row in main.itertuples():
            date = row.worst_date
            day_index = (pd.Timestamp(date)-pd.Timestamp("2025-02-01")).days
            absolute_day = day_index+31
            with np.load(out / f"dispatch_{row.scenario}.npz") as f:
                d = {k: f[k][day_index].copy() for k in f.files}
            prediction = store.get(absolute_day*144, issued=row.scenario in ("3", "4-3"))
            pv_col = 3 if row.scenario in ("3", "4-3") else 1
            storage.extend({"scenario": row.scenario, "date": date, "hour": t/6,
                            "soc": float(d["states"][t])} for t in range(145))
            for t in range(144):
                cases.append({"scenario": row.scenario, "date": date, "hour": (t+.5)/6,
                    "original": float(d["original"][t]), "final": float(d["final"][t]), "emergency": float(d["emergency"][t]),
                    "soc": float(d["states"][t+1]), "actual_net": float((d["actual"][t, 0]-d["actual"][t, 1])/6),
                    "predicted_net": float((prediction[t, 0]-prediction[t, pv_col])/6), "price": float(d["price"][t]),
                    "interval_cost": float(d["fees"][t].sum())})
        pd.DataFrame(cases).to_csv(out / "failure_intervals.csv", index=False)
        pd.DataFrame(storage).to_csv(out / "failure_storage.csv", index=False)
        costs[costs.name == "primary"].to_csv(out / "seed_cost_results.csv", index=False)
        costs[costs.name == "primary"].groupby("scenario").agg(
            mean_cost=("total_cost", "mean"), std_cost=("total_cost", "std"),
            mean_cvar=("daily_cvar90", "mean"), std_cvar=("daily_cvar90", "std")).to_csv(out / "seed_cost_statistics.csv")
        run_directory = ROOT / "experiments/common/neural_v2/runs" / run_id
        elapsed_file = run_directory / "evaluation_walltime.json"
        if not elapsed_file.exists():
            elapsed_file = out / "evaluation_walltime.json"
        elapsed = json.loads(elapsed_file.read_text())
        if (run_directory / "resume_events.json").exists():
            resume = json.loads((run_directory / "resume_events.json").read_text())
            elapsed["first_segment_seconds"] = resume["first_segment_seconds"]
            elapsed["resumed_segment_seconds"] = elapsed["seconds"]
            elapsed["seconds"] += resume["first_segment_seconds"]
            elapsed["worker_schedule"] = [4, 8]
            (out / "resume_events.json").write_text(json.dumps(resume, indent=2))
        if (run_directory / "causal_fix_event.json").exists():
            fix = json.loads((run_directory / "causal_fix_event.json").read_text())
            elapsed["second_segment_before_causal_fix_seconds"] = fix["second_segment_seconds"]
            elapsed["seconds"] += fix["second_segment_seconds"]
            elapsed["worker_schedule"] = [4, 8, 8]
            elapsed["includes_discarded_preliminary_q3_runs"] = True
            (out / "causal_fix_event.json").write_text(json.dumps(fix, indent=2))
        if (run_directory / "clipped_information_fix_event.json").exists():
            fix = json.loads((run_directory / "clipped_information_fix_event.json").read_text())
            elapsed["third_segment_before_clipped_information_fix_seconds"] = fix["third_segment_seconds"]
            elapsed["seconds"] += fix["third_segment_seconds"]
            elapsed["worker_schedule"] = [4, 8, 8, 8]
            elapsed["includes_discarded_preliminary_q3_and_q43_runs"] = True
            (out / "clipped_information_fix_event.json").write_text(json.dumps(fix, indent=2))
        (out / "evaluation_walltime.json").write_text(json.dumps(elapsed, indent=2))
        timing_rows.append({"stage": "全年调度", "minutes": elapsed["seconds"]/60,
                            "scope": "全部核心、基线、消融及三种子；累计全部已记录运行段，包含作废的问题 3、4-3 初步回放，不含暂停间隔"})
        solver = pd.read_csv(out / "solver_metrics.csv", dtype={"scenario": str})
        solver["gap_known"] = solver.mip_gap.notna()
        solver["gap_met"] = solver.mip_gap <= .010001
        solver["timed_out"] = solver.status == 1
        groups = solver.groupby(["scenario", "seed"])
        groups.agg(calls=("status", "size"), timed_out=("timed_out", "sum"),
                   fallback=("fallback", "sum"), gap_known=("gap_known", "sum"),
                   gap_met=("gap_met", "sum"), mean_solver_seconds=("solver_seconds", "mean"),
                   max_solver_seconds=("solver_seconds", "max"),
                   max_constraint_residual=("constraint_residual", "max"),
                   max_controller_error=("controller_error", "max")).to_csv(out / "solver_summary.csv")
        daily[(daily.name == "primary") & (daily.seed == 42)].groupby(["scenario", "month"]).agg(
            total_cost=("total_cost", "sum"), emergency_kwh=("emergency_kwh", "sum"),
            final_soc=("final_soc", "last")).to_csv(out / "monthly_dispatch.csv")
    stage_file = ROOT / "experiments/common/neural_v2/runs" / run_id / "stage_timings.json"
    if not stage_file.exists():
        stage_file = out / "stage_timings.json"
    if stage_file.exists():
        stage_rows = json.loads(stage_file.read_text())
        (out / "stage_timings.json").write_text(json.dumps(stage_rows, indent=2))
        for row in stage_rows:
            if row["stage"] in ("export", "report"):
                timing_rows.append({"stage": "工作簿导出" if row["stage"] == "export" else "报告构建",
                    "minutes": row["seconds"]/60, "scope": "一次完整阶段的墙钟时间；不含此前预览或人工检查"})
    pd.DataFrame(timing_rows).to_csv(out / "phase_timing.csv", index=False)


def figures(run_id="exp002"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font = next(p for p in (Path("/System/Library/Fonts/PingFang.ttc"),
                            Path("/System/Library/Fonts/STHeiti Medium.ttc"),
                            Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf")) if p.exists())
    font_manager.fontManager.addfont(font)
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=font).get_name(),
                         "font.size": 10, "axes.unicode_minus": False,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.bbox": "tight"})
    out = ROOT / "data/results" / run_id
    report = ROOT / "reports/experiments" / run_id / "figures"
    report.mkdir(parents=True, exist_ok=True)

    def save(fig, name):
        fig.savefig(report / f"{name}.png", dpi=180)
        fig.savefig(report / f"{name}.svg")
        plt.close(fig)

    forecast = pd.read_csv(out / "forecast_metrics.csv", dtype={"seed": str})
    f = forecast[(forecast.lead == "all") & (forecast.population == "all")]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for ax, target in zip(axes.ravel(), LABELS):
        for variant, seed, label in (("mlp", "42", "固定轻量网络 · 42"), ("mlp", "2026", "固定轻量网络 · 2026"),
                                      ("mlp", "3407", "固定轻量网络 · 3407"), ("periodic", "none", "周期基础预测")):
            part = f[(f.target == target) & (f.variant == variant) & (f.seed == seed)]
            ax.plot(part.month, part.mae, label=label, marker="o", markersize=3, linewidth=1.2)
        ax.set(title=LABELS[target], xlabel="月份", ylabel="MAE（元/千瓦时）" if target == "price" else "MAE（千瓦）", xticks=range(2, 13))
        ax.grid(axis="y", alpha=.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4)
    save(fig, "monthly-forecast-error")
    annual = pd.read_csv(out / "annual_forecast_metrics.csv", dtype={"seed": str})
    fig, axes = plt.subplots(4, 3, figsize=(14, 12), layout="constrained")
    for row, target in zip(axes, LABELS):
        variants = ["issued", "mlp"] if target == "pv_corrected" else ["yesterday", "weekly", "periodic", "mlp"]
        names = {"issued": "附件原始预报", "mlp": "固定网络 · 42", "yesterday": "昨日同期", "weekly": "上周同期", "periodic": "周期基础"}
        part = annual[(annual.target == target) & (annual.population == "all") & annual.seed.isin(["42", "none"])].set_index("variant").loc[variants]
        for ax, metric in zip(row, ("mae", "rmse", "wape_pct")):
            ax.barh([names[v] for v in variants], part[metric], color=["#2b788b" if v == "mlp" else "#a3adb5" for v in variants])
            unit = "%" if metric == "wape_pct" else "元/千瓦时" if target == "price" else "千瓦"
            ax.set(title=f"{LABELS[target]} · {metric.upper()}", xlabel=unit)
            ax.grid(axis="x", alpha=.15)
    save(fig, "annual-three-error-metrics")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for ax, target in zip(axes.ravel(), LABELS):
        x = forecast[(forecast.target == target) & (forecast.lead != "all") & (forecast.seed == "42")]
        for population, label in (("all", "全部区间"), ("generating", "实际发电区间")):
            part = x[x.population == population].groupby("lead").agg(n=("n", "sum"), error=("absolute_error_sum", "sum"))
            if len(part):
                part = part.loc[sorted(part.index, key=lambda label: int(label.split("–")[0]))]
                ax.plot(part.index, part.error/part.n, marker="o", label=label)
        ax.set(title=LABELS[target], xlabel="提前量分组（小时）", ylabel="MAE（元/千瓦时）" if target == "price" else "MAE（千瓦）")
        ax.grid(axis="y", alpha=.2)
        ax.legend(fontsize=8)
    save(fig, "lead-and-generating-error")
    decomp = pd.read_csv(out / "forecast_decomposition.csv")
    fig, axes = plt.subplots(4, 2, figsize=(13, 12), layout="constrained")
    for row, target in zip(axes, LABELS):
        x = decomp[decomp.target == target]
        for key, label in (("actual", "真实值"), ("base", "基础预测"), ("prediction", "网络修正后")):
            row[0].plot(x.hour, x[key], label=label, linewidth=1.3)
        row[0].set(title=f"3 月 20 日凌晨预测 · {LABELS[target]}", ylabel="元/千瓦时" if target == "price" else "千瓦")
        row[0].legend(ncol=3, fontsize=8)
        row[1].plot(x.hour, x.correction, color="#ba6d39")
        row[1].axhline(0, color="#888888", linewidth=.7)
        row[1].set(title="修正量 = 最终预测 − 基础预测", ylabel="元/千瓦时" if target == "price" else "千瓦")
        for ax in row:
            ax.set(xlabel="小时", xlim=(0, 24), xticks=range(0, 25, 6))
            ax.grid(axis="y", alpha=.15)
    save(fig, "baseline-correction-decomposition")
    new = json.loads((out / "training_metadata.json").read_text())
    old = json.loads((ROOT / "data/results/exp001/training_metadata.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    axes[0].bar(["exp001", "exp002"], [len(old), len(new)], color=["#a4aeb6", "#2b788b"])
    axes[0].set(title="正式训练组数", ylabel="组")
    axes[1].bar(["exp001", "exp002"], [sum(r["seconds"] for r in old)/60, sum(r["training_seconds"] for r in new)/60], color=["#a4aeb6", "#2b788b"])
    axes[1].set(title="累计实测训练耗时（不含调度）", ylabel="分钟")
    save(fig, "training-count-and-time")
    tree = json.loads((out / "tree_example.json").read_text())
    fig, ax = plt.subplots(figsize=(11, 6), layout="constrained")
    positions = {}
    for node in tree["nodes"]:
        x, y = node["hour"], node["layout_y"]
        positions[(x, node["node"])] = (x, y)
        ax.scatter(x, y, s=160, color="#2b788b", zorder=3)
        ax.annotate(f"节点 {node['node']}\n{node['path_count']} 条路径", (x, y), xytext=(8, 0), textcoords="offset points", fontsize=8, va="center", bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})
    for record in tree["metadata"]["tree"]:
        hour = record["reveal_slot"]//6
        a = positions.get((hour-6, record["parent"]))
        b = positions.get((hour, record["node"]))
        if a and b:
            ax.plot([a[0], b[0]], [a[1], b[1]], color="#98a8b3", linewidth=1)
    ax.set(xticks=(0, 6, 12, 18), xticklabels=("0 时\n仅当前预报", "6 时\n揭示前缀＋新预报", "12 时\n再次揭示", "18 时\n末段仍有误差"), yticks=[], xlim=(-1, 22), title="问题 4-3 · 3 月 20 日 · 场景树按信息揭示分支（16 条路径）")
    for spine in ax.spines.values():
        spine.set_visible(False)
    save(fig, "scenario-information-tree")
    from matplotlib.patches import FancyArrowPatch, Rectangle
    fig, ax = plt.subplots(figsize=(12, 6), layout="constrained")
    for row, target in enumerate(("load", "pv", "price", "pv_corrected")):
        y = 3-row
        ax.text(-.35, y, LABELS[target], ha="right", va="center", fontsize=11)
        for x, text in ((0, "16 项输入"), (2, "32 单元\nReLU"), (4, "16 单元\nReLU"), (6, "1 个修正量\n输出层零初始化"), (8.3, "基础预测＋修正\n恢复原单位")):
            ax.add_patch(Rectangle((x, y-.3), 1.45, .6, facecolor="#f0f5f6", edgecolor="#2b788b", linewidth=1.1))
            ax.text(x+.725, y, text, ha="center", va="center", fontsize=9)
            if x < 8:
                ax.add_patch(FancyArrowPatch((x+1.48, y), (x+1.98 if x < 6 else 8.25, y), arrowstyle="->", mutation_scale=12, color="#71858e"))
    ax.set(xlim=(-1.7, 10), ylim=(-.7, 3.7), title="一个检查点 · 四个独立分支 · 合计 4356 个参数")
    ax.axis("off")
    save(fig, "fixed-network-architecture")
    trace = pd.read_csv(out / "training_trace.csv")
    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    ax.plot(trace.epoch, trace.loss, label="训练目标", color="#2b788b")
    ax.plot(trace.epoch, trace.val_loss, label="验证目标", color="#cc8a3d")
    ax.set(title="3 月检查点 · 种子 42 · 四分支平均 Huber 损失与 L2 正则", xlabel="训练轮次", ylabel="标准化训练目标")
    ax.grid(alpha=.15)
    ax.legend()
    save(fig, "worked-example-training-loss")
    if not (out / "dispatch_metrics.csv").exists():
        return
    costs = pd.read_csv(out / "dispatch_metrics.csv", dtype={"scenario": str})
    costs = costs[costs.seed == 42]
    names = {"legacy_rebased": "旧预测＋基础", "new_deterministic": "新预测＋基础", "primary": "新预测＋风险"}
    colors = ["#a8afb5", "#6397b1", "#267c70"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for ax, scenario in zip(axes.ravel(), SCENARIOS):
        x = costs[costs.scenario == scenario].set_index("name").loc[list(names)]
        ax.bar(list(names.values()), x.total_cost/10000, color=colors)
        ax.set(title=f"问题 {scenario}", ylabel="全年费用（万元）")
        ax.grid(axis="y", alpha=.15)
    save(fig, "three-layer-cost-comparison")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for ax, scenario in zip(axes.ravel(), SCENARIOS):
        x = costs[costs.scenario == scenario].set_index("name").loc[list(names)]
        bottom = np.zeros(3)
        for key, label in (("planned_cost", "原计划"), ("up_cost", "上调"), ("down_cost", "下调"), ("emergency_cost", "紧急购电")):
            ax.bar(list(names.values()), x[key]/10000, bottom=bottom, label=label)
            bottom += x[key].to_numpy()/10000
        ax.set(title=f"问题 {scenario}", ylabel="万元")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4)
    save(fig, "cost-components")
    failures = pd.read_csv(out / "failure_intervals.csv", dtype={"scenario": str})
    storage = pd.read_csv(out / "failure_storage.csv", dtype={"scenario": str})
    fig, axes = plt.subplots(4, 2, figsize=(13, 12), layout="constrained")
    for row, scenario in zip(axes, SCENARIOS):
        x = failures[failures.scenario == scenario]
        for key, label in (("actual_net", "实际净需求"), ("predicted_net", "凌晨预测净需求"), ("final", "最终购电"), ("emergency", "紧急购电")):
            row[0].plot(x.hour, x[key], label=label, linewidth=1.1)
        row[0].set(title=f"问题 {scenario} · 最贵日 {x.date.iloc[0]}", ylabel="十分钟电量（kWh）")
        row[0].legend(fontsize=7, ncol=2)
        states = storage[storage.scenario == scenario]
        row[1].plot(states.hour, states.soc, color="#267c70")
        row[1].axhline(1200, color="#777", ls="--", linewidth=.8)
        row[1].axhline(10800, color="#777", ls="--", linewidth=.8)
        row[1].set(title="实际储电量轨迹", ylabel="kWh", ylim=(500, 11500))
        for ax in row:
            ax.set(xlabel="小时", xticks=range(0, 25, 6))
            ax.grid(axis="y", alpha=.15)
    save(fig, "failure-case-and-storage")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for ax, scenario in zip(axes.ravel(), SCENARIOS):
        x = costs[(costs.scenario == scenario) & costs.name.isin([*names, "risk_zero", "no_anticipation", "raw_forecast"])]
        for row in x.itertuples():
            extra = {"risk_zero": "风险权重为零", "no_anticipation": "不预先考虑更新", "raw_forecast": "原始光伏预报"}
            ax.scatter(row.total_cost/10000, row.daily_cvar90/10000, s=45, label=names.get(row.name, extra.get(row.name, row.name)))
        ax.set(title=f"问题 {scenario}", xlabel="全年费用（万元）", ylabel="日费用 CVaR90（万元）")
        ax.grid(alpha=.15)
        ax.legend(fontsize=7)
    save(fig, "cost-versus-tail-risk")
    summary = pd.read_csv(out / "solver_summary.csv", dtype={"scenario": str})
    summary = summary[summary.seed == 42]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    for key, label in (("timed_out", "超时"), ("fallback", "无可行解回退"), ("gap_met", "有证书且差距≤1%")):
        axes[0].plot(summary.scenario, 100*summary[key]/summary.calls, marker="o", label=label)
    axes[0].set(ylabel="占求解次数比例（%）", title="超时、回退与最优性证书分别统计", ylim=(-3, 103))
    axes[0].legend(fontsize=8)
    axes[1].bar(summary.scenario, summary.mean_solver_seconds, color="#2b788b")
    axes[1].set(ylabel="秒", title="平均求解器耗时（不含组装与初始候选）")
    save(fig, "solver-gap-and-fallback")
    daily = pd.read_csv(out / "daily_metrics.csv", dtype={"scenario": str})
    daily = daily[(daily.name == "primary") & (daily.seed == 42)]
    fig, axes = plt.subplots(4, 2, figsize=(13, 12), layout="constrained")
    for row, scenario in zip(axes, SCENARIOS):
        x = daily[daily.scenario == scenario]
        row[0].plot(pd.to_datetime(x.date), x.emergency_kwh, linewidth=.9, color="#b77444")
        row[0].set(title=f"问题 {scenario} · 逐日紧急购电", ylabel="kWh")
        monthly = x.groupby("month").total_cost.sum()
        row[1].bar(monthly.index, monthly/10000, color="#2b788b")
        row[1].set(title="分月实际总费用", xlabel="月份", ylabel="万元", xticks=range(2, 13))
        for ax in row:
            ax.grid(axis="y", alpha=.15)
    save(fig, "daily-emergency-and-monthly-cost")


if __name__ == "__main__":
    derive()
    figures()
