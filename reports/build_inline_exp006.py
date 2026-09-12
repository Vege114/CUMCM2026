"""Bind the verified report evidence to the conversation visualization template."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "reports/experiments/exp006/evidence"
DEFAULT_OUTPUT = Path(
    "/Users/vegetarianwolf/.codex/visualizations/2026/09/12/"
    "01a0962f-1eb3-7cd2-a50f-eddb31b4abcc/q2-tree-comparison.html"
)
POLICIES = (
    ("exp001/legacy_rebased", "exp001 同口径"),
    ("exp002/primary", "exp002 正式"),
    ("exp003/primary", "exp003 正式"),
    ("exp004/no_season", "exp004 无季节"),
    ("exp004/causal_season", "exp004 历史季节"),
    ("exp006/primary", "exp006 主方案"),
    ("exp006/greedy_execution", "exp006 贪心对照"),
    ("exp006/fixed_primary_greedy", "exp006 同购电贪心"),
)


def numeric(value):
    return float(value) if pd.notna(value) else None


def pick(frame, policy):
    selected = frame.loc[frame.policy_id.eq(policy)]
    if len(selected) > 1:
        selected = selected.loc[pd.to_numeric(selected.seed, errors="coerce").eq(42)]
    if len(selected) != 1:
        raise ValueError(f"Expected one reviewed row for {policy}, found {len(selected)}")
    return selected.iloc[0]


def build(output=DEFAULT_OUTPUT):
    cost = pd.read_csv(EVIDENCE / "cost_history.csv")
    battery = pd.read_csv(EVIDENCE / "battery_history.csv")
    forecast = pd.read_csv(EVIDENCE / "forecast_history.csv")
    panels = []
    cost_rows, time_rows = [], []
    for policy, label in POLICIES:
        row = pick(cost, policy)
        assert str(row.physically_comparable).lower() == "true"
        assert str(row.ranking_allowed).lower() == "true"
        total = float(row.total_cost)
        planned, emergency = float(row.planned_cost), float(row.emergency_cost)
        assert abs(total - planned - emergency) < .01
        cost_rows.append({"label": label, "planned": planned / 10000,
                          "emergency": emergency / 10000, "primary": policy == "exp006/primary",
                          "detail": f"{label}：总费 {total:,.2f} 元；计划费 {planned:,.2f} 元；紧急费 {emergency:,.2f} 元。334 日，同物理计费；主方案角色事先冻结。"})
        seconds = numeric(row.get("solve_execute_seconds"))
        if policy != "exp006/fixed_primary_greedy":
            time_rows.append({"label": label, "value": seconds,
                              "primary": policy == "exp006/primary",
                              "detail": f"{label}：全年调度和执行 {seconds} 秒；历史计时口径及设备条件有差异，仅描述性比较。"})
    panels.append({"title": "实际购电费", "unit": "万元", "axisTitle": "计划费 + 紧急费（万元）",
                   "series": [{"key": "planned", "label": "计划费"},
                              {"key": "emergency", "label": "紧急费"}], "rows": cost_rows})
    for metric, title, unit, explanation in (
        ("equivalent_full_cycles", "电池等效全循环", "次", "电芯侧吞吐除以两倍12000 kWh；是运行强度代理，不是寿命预测"),
        ("direction_reversals", "充放方向切换", "次", "串联334日并忽略闲置，仅计评价期内部方向变化"),
    ):
        rows = []
        for policy, label in POLICIES:
            found = battery.loc[battery.policy_id.eq(policy)]
            if found.empty:
                continue
            row = pick(battery, policy)
            value = numeric(row.get(metric))
            rows.append({"label": label, "value": value, "primary": policy == "exp006/primary",
                         "detail": f"{label}：{value if value is not None else '未记录'} {unit}；{explanation}。"})
        panels.append({"title": title, "unit": unit, "axisTitle": title + f"（{unit}）",
                       "series": [{"key": "value", "label": title}], "rows": rows})
    # Forecast changes are intentionally zero versus the frozen exp004 input.
    forecast_rows = forecast.loc[forecast.target.eq("net_load") & forecast.population.eq("all")]
    chosen = []
    for experiment, needle, label in (
        ("exp001", None, "exp001 重评分"), ("exp002", None, "exp002 重评分"),
        ("exp003", "primary", "exp003 正式"), ("exp004", "no_season", "exp004 无季节"),
        ("exp004", "causal_season", "exp004 历史季节"), ("exp006", "primary", "exp006 主方案"),
    ):
        found = forecast_rows.loc[forecast_rows.experiment.eq(experiment)]
        if needle:
            names = found.astype(str).agg(" ".join, axis=1)
            found = found.loc[names.str.contains(needle, regex=False)]
        if "seed" in found:
            found = found.loc[pd.to_numeric(found.seed, errors="coerce").eq(42)]
        if len(found) != 1:
            raise ValueError(f"Forecast binding ambiguous: {experiment}/{needle}, {len(found)} rows")
        chosen.append((experiment, label, found.iloc[0]))
    for metric, title, unit in (("wape_pct", "净负荷预测 WAPE", "%"),
                                ("rmse", "净负荷预测 RMSE", "kW")):
        rows = [{"label": label, "value": float(row[metric]), "primary": experiment == "exp006",
                 "detail": f"{label}：{float(row[metric]):.6f} {unit}；48,096 个午夜预测槽。exp006 复用 exp004 无季节预测，未重新训练。"}
                for experiment, label, row in chosen]
        panels.append({"title": title, "unit": unit, "axisTitle": title + f"（{unit}）",
                       "series": [{"key": "value", "label": title}], "rows": rows})
    panels.append({"title": "全年调度与执行时间", "unit": "秒", "axisTitle": "累计计算时间（秒；历史仅描述性比较）",
                   "series": [{"key": "value", "label": "秒"}], "rows": time_rows})
    payload = {"panels": panels}
    template = (ROOT / "reports/templates/exp006-inline.html").read_text()
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace("</", "<\\/")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(template.replace("@@DATA@@", encoded))
    assert output.stat().st_size < 1_000_000
    manifest = {"output": str(output), "panels": len(panels),
                "input_sha256": {name: hashlib.sha256((EVIDENCE / name).read_bytes()).hexdigest()
                                 for name in ("cost_history.csv", "battery_history.csv", "forecast_history.csv")},
                "fragment_sha256": hashlib.sha256(output.read_bytes()).hexdigest()}
    (EVIDENCE / "inline_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    build(parser.parse_args().output)
