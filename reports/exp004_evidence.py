"""One reviewed evidence layer for Markdown, Data app, figures and conversation."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.problem2.exp004.data import OUT, ROOT, VARIANTS, sha256, write_json

REPORT = ROOT / "reports/experiments/exp004"
EVIDENCE = REPORT / "evidence"
LABELS = {"no_season": "exp004 无季节", "causal_season": "exp004 历史季节",
          "oracle_season": "exp004 全年探索"}
TARGETS = {"load": "负载", "pv": "光伏", "net_load": "净负载"}
SPECIFIED = ["2025-03-20", "2025-06-21", "2025-09-23", "2025-12-21"]


def read(path):
    return json.loads(Path(path).read_text())


def build_evidence():
    if read(OUT / "verification.json")["status"] != "passed":
        raise RuntimeError("Independent verification must pass")
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    cost, forecast, timings = [], [], []
    registries = {p.stem: read(p) for p in sorted((ROOT / "reports/registry").glob("*.json"))
                  if p.stem != "exp004"}
    if set(registries) != {"exp001", "exp002", "exp003"}:
        raise RuntimeError("Update comparison coverage for changed historical registry")
    base = ROOT / "data/results/exp003/baseline"
    historical = pd.read_csv(base / "exp002_q2_costs.csv")
    selections = [("legacy_rebased", "exp001", "exp001 同口径重算", True),
                  ("primary", "exp002", "exp002 正式风险", False),
                  ("new_deterministic", "exp002", "exp002 同确定性", True)]
    for name, experiment, label, same_dispatch in selections:
        row = historical[(historical.name == name) & (historical.seed == 42)].iloc[0].to_dict()
        row.update(experiment=experiment, policy_id=f"{experiment}/{name}", label=label,
                   same_dispatch=same_dispatch, physically_comparable=True, ranking_allowed=True,
                   source="data/results/exp003/baseline/exp002_q2_costs.csv", exploratory=False)
        cost.append(row)
    old = pd.read_csv(ROOT / "data/results/exp003/dispatch_metrics.csv")
    for name, label in [("primary", "exp003 正式"), ("uncalibrated", "exp003 未校准")]:
        row = old[(old.name == name) & (old.seed == 42)].iloc[0].to_dict()
        row.update(experiment="exp003", policy_id=f"exp003/{name}", label=label,
                   same_dispatch=True, physically_comparable=True, ranking_allowed=True,
                   source="data/results/exp003/dispatch_metrics.csv", exploratory=False)
        cost.append(row)
    current = pd.read_csv(OUT / "dispatch_metrics.csv")
    for row in current[current.seed == 42].to_dict("records"):
        row.update(experiment="exp004", policy_id=f"exp004/{row['name']}", label=LABELS[row["name"]],
                   same_dispatch=True, physically_comparable=True,
                   ranking_allowed=row["name"] != "oracle_season", source="data/results/exp004/dispatch_metrics.csv")
        cost.append(row)
    original = pd.read_csv(base / "exp001_original_q2_costs.csv").iloc[0].to_dict()
    original.update(experiment="exp001", policy_id="exp001/original", label="exp001 原登记（不可比）",
                    same_dispatch=False, physically_comparable=False, ranking_allowed=False,
                    source="data/results/exp003/baseline/exp001_original_q2_costs.csv", exploratory=False)
    cost.append(original)
    costs = pd.DataFrame(cost)
    costs["total_wan"] = costs.total_cost / 10000
    costs["planned_wan"] = costs.planned_cost / 10000
    costs["emergency_wan"] = costs.emergency_cost / 10000
    costs["status"] = np.where(~costs.physically_comparable, "原协议不可直接比较",
                               np.where(costs.exploratory, "含未来信息，仅探索", "同物理口径"))
    hist = pd.read_csv(base / "forecast_annual.csv")
    for pid, experiment, label in [("exp001_selected", "exp001", "exp001 午夜重评分"),
                                    ("exp002_seed_42", "exp002", "exp002 午夜重评分")]:
        part = hist[hist.predictor_id == pid].copy()
        if len(part) != 6:
            raise RuntimeError(f"Missing historical forecast populations: {pid}")
        part["label"], part["experiment"], part["source"] = label, experiment, "data/results/exp003/baseline/forecast_annual.csv"
        part["seed"], part["exploratory"] = 42, False
        forecast.extend(part.to_dict("records"))
    hist = pd.read_csv(ROOT / "data/results/exp003/forecast_annual.csv")
    for name, label in [("primary", "exp003 正式"), ("uncalibrated", "exp003 未校准")]:
        part = hist[(hist.name == name) & (hist.seed == 42)].copy()
        part["label"], part["experiment"], part["source"] = label, "exp003", "data/results/exp003/forecast_annual.csv"
        part["exploratory"] = False
        forecast.extend(part.to_dict("records"))
    now = pd.read_csv(OUT / "forecast_annual.csv")
    for row in now[now.seed == 42].to_dict("records"):
        row.update(label=LABELS[row["name"]], experiment="exp004", source="data/results/exp004/forecast_annual.csv")
        forecast.append(row)
    forecasts = pd.DataFrame(forecast)
    if set(forecasts[forecasts.population == "all"].n) != {48096}:
        raise RuntimeError("Forecast populations do not align")
    comparisons = []

    def compare(previous, current_value, metric, unit, task, population="all"):
        comparable = bool(previous.get("ranking_allowed", True) and not current_value.get("exploratory", False))
        pv, cv = previous.get(metric), current_value.get(metric)
        available = pd.notna(pv) and pd.notna(cv)
        comparisons.append({"previous_experiment": previous["experiment"], "previous_label": previous["label"],
            "current_experiment": "exp004", "current_label": current_value["label"], "task": task,
            "metric": metric, "population": population, "previous": pv, "current": cv, "unit": unit,
            "absolute_change_unit": "百分点" if unit == "%" else unit,
            "absolute_change": cv - pv if comparable and available else None,
            "relative_change_pct": 100 * (cv - pv) / abs(pv) if comparable and available and pv != 0 else None,
            "comparable": comparable and available, "seed": 42,
            "same_dispatch": previous.get("same_dispatch"), "previous_source": previous["source"],
            "current_source": current_value["source"], "period": "2025-02-01/2025-12-31"})

    for row in costs[costs.experiment == "exp004"].to_dict("records"):
        for prev in costs[costs.experiment != "exp004"].to_dict("records"):
            for metric, unit in [("total_cost", "元"), ("planned_cost", "元"), ("emergency_cost", "元"),
                                 ("planned_kwh", "kWh"), ("emergency_kwh", "kWh"),
                                 ("charge_kwh", "kWh"), ("discharge_kwh", "kWh"),
                                 ("daily_cvar90", "元"), ("final_soc", "kWh")]:
                compare(prev, row, metric, unit, "dispatch")
    for row in forecasts[forecasts.experiment == "exp004"].to_dict("records"):
        matching = forecasts[(forecasts.experiment != "exp004") & (forecasts.target == row["target"])
                             & (forecasts.population == row["population"])]
        for prev in matching.to_dict("records"):
            if prev["n"] != row["n"]:
                raise RuntimeError("Matched forecast count differs")
            for metric in ("mae", "rmse", "wape_pct", "bias"):
                compare(prev, row, metric, "%" if metric == "wape_pct" else "kW", row["target"], row["population"])
    for experiment in ("exp001", "exp002", "exp003"):
        file = ROOT / f"data/results/{experiment}/training_metadata.json"
        if file.exists():
            metadata = read(file)
            for field, label in [("training_seconds", "训练"), ("prediction_seconds", "检查点预测")]:
                key = "seconds" if experiment == "exp001" and field == "training_seconds" else field
                values = [row[key] for row in metadata if key in row]
                timings.append({"experiment": experiment, "label": experiment, "stage": label,
                                "seconds": sum(values) if values else None, "groups": len(metadata),
                                "scope": "历史训练任务累计；设备/目标数/样本数不同，仅描述", "source": str(file.relative_to(ROOT))})
        else:
            for stage in ("训练", "检查点预测"):
                timings.append({"experiment": experiment, "label": experiment, "stage": stage,
                                "seconds": None, "groups": None, "scope": "未取得统一阶段记录", "source": f"reports/registry/{experiment}.json"})
    metadata = read(OUT / "training_metadata.json")
    for variant in VARIANTS:
        group = [m for m in metadata if m["variant"] == variant]
        for field, label in [("training_seconds", "训练"), ("prediction_seconds", "检查点预测")]:
            timings.append({"experiment": "exp004", "label": LABELS[variant], "stage": label,
                "seconds": sum(m[field] for m in group), "groups": len(group),
                "scope": "CPU两线程，33组累计；特征准备另计，不是端到端耗时", "source": "data/results/exp004/training_metadata.json"})
    for row in costs[costs.physically_comparable].to_dict("records"):
        timings.append({"experiment": row["experiment"], "label": row["label"], "stage": "调度执行",
                        "seconds": row.get("solve_execute_seconds"), "groups": 334,
                        "scope": "主种子334日组装求解执行累计，非并行墙钟", "source": row["source"]})
    for experiment in ("exp002", "exp003"):
        timing_path = ROOT / f"data/results/{experiment}/stage_timings.json"
        rows = read(timing_path) if timing_path.exists() else []
        match = [r for r in rows if r.get("stage") == "calibrate"]
        timings.append({"experiment": experiment, "label": experiment, "stage": "费用校准",
                        "seconds": match[0]["seconds"] if match else None, "groups": None,
                        "scope": "历史阶段记录；exp002含四问。缺失不填零", "source": str(timing_path.relative_to(ROOT))})
    timings.append({"experiment": "exp004", "label": "exp004", "stage": "费用校准", "seconds": None,
                    "groups": 0, "scope": "不适用：未实施费用回选", "source": "experiments/problem2/exp004/protocol.json"})
    daily = pd.read_csv(OUT / "daily_metrics.csv")
    monthly = daily.groupby(["name", "seed", "month"], as_index=False)[
        ["planned_cost", "emergency_cost", "total_cost", "planned_kwh", "emergency_kwh"]].sum()
    monthly["label"] = monthly.name.map(LABELS)
    monthly["date"] = monthly.month.map(lambda m: f"2025-{m:02d}-01")
    months = pd.read_csv(OUT / "forecast_monthly.csv")
    months["label"] = months.name.map(LABELS)
    months["date"] = months.month.map(lambda m: f"2025-{m:02d}-01")
    seasonal, specified, detail = [], [], []
    with np.load(OUT / "predictions.npz") as z:
        for variant in VARIANTS:
            shifts = z[f"{variant}_seasonal_shift"]
            for i, origin in enumerate(z["origins"]):
                seasonal.append({"date": str((pd.Timestamp("2025-01-01") + pd.Timedelta(days=int(origin) // 144)).date()),
                    "variant": variant, "label": LABELS[variant], "load_shift_kw": shifts[i, :, 0].mean(),
                    "pv_shift_kw": shifts[i, :, 1].mean(), "shift_rms_kw": float(np.sqrt(np.mean(shifts[i] ** 2)))})
            dp = OUT / f"dispatch_{variant}_seed_42.npz"
            worst = daily[(daily.name == variant) & (daily.seed == 42)].sort_values("total_cost").iloc[-1].date
            with np.load(dp) as d:
                for date in dict.fromkeys([*SPECIFIED, worst]):
                    i = (pd.Timestamp(date) - pd.Timestamp("2025-02-01")).days
                    for t in range(144):
                        p = z[f"{variant}_seed_42"][i, t]
                        a = d["actual"][i, t]
                        detail.append({"date": date, "name": variant, "label": LABELS[variant], "hour": t / 6,
                            "actual_load_kw": a[0], "actual_pv_kw": a[1], "forecast_load_kw": p[0], "forecast_pv_kw": p[1],
                            "actual_net_kwh": (a[0] - a[1]) / 6, "forecast_net_kwh": (p[0] - p[1]) / 6,
                            "planned_kwh": d["original"][i, t], "emergency_kwh": d["emergency"][i, t],
                            "soc_kwh": d["states"][i, t], "total_cost": d["fees"][i, t].sum()})
            p = OUT / variant / "specified_dates.csv"
            if p.exists():
                part = pd.read_csv(p)
                part["label"], part["name"] = LABELS[variant], variant
                specified.extend(part.to_dict("records"))
    no = daily[(daily.name == "no_season") & (daily.seed == 42)].sort_values("day")
    causal = daily[(daily.name == "causal_season") & (daily.seed == 42)].sort_values("day")
    paired = causal[["day", "date", "month", "total_cost"]].copy()
    paired["no_season_cost"] = no.total_cost.to_numpy()
    paired["difference_yuan"] = paired.total_cost - paired.no_season_cost
    # Paired 7-day moving blocks describe sampling sensitivity within this year,
    # not independent seasonal generalization or uncertainty for unobserved years.
    values = paired.difference_yuan.to_numpy()
    rng = np.random.default_rng(42)
    means = []
    for _ in range(2000):
        starts = rng.integers(0, len(values) - 7 + 1, size=int(np.ceil(len(values) / 7)))
        sample = np.concatenate([values[s:s + 7] for s in starts])[:len(values)]
        means.append(sample.mean() * len(values))
    sensitivity = {"observed_annual_difference_yuan": float(values.sum()),
                   "seven_day_block_95pct_interval_yuan": np.quantile(means, [.025, .975]).tolist(),
                   "replicates": 2000, "seed": 42,
                   "interpretation": "单年内七日区块重采样敏感性，不能代表跨年泛化置信区间"}
    frames = {"cost_history": costs, "forecast_history": forecasts,
              "relative_comparison": pd.DataFrame(comparisons), "timings": pd.DataFrame(timings),
              "monthly_cost": monthly, "monthly_forecast": months,
              "seasonal_shifts": pd.DataFrame(seasonal), "specified": pd.DataFrame(specified),
              "detail": pd.DataFrame(detail), "paired_daily": paired, "seed_costs": current,
              "seed_summary": pd.read_csv(OUT / "seed_summary.csv")}
    for name, frame in frames.items():
        frame.to_csv(EVIDENCE / f"{name}.csv", index=False)
    write_json(EVIDENCE / "seasonal_sensitivity.json", sensitivity)
    write_json(EVIDENCE / "registry_coverage.json", {"experiments": list(registries),
        "sha256": {name: sha256(ROOT / f"reports/registry/{name}.json") for name in registries},
        "forecast_note": "exp001/exp002历史共享四目标训练，午夜重评分不移除其共同早停边界"})
    return frames


if __name__ == "__main__":
    build_evidence()
