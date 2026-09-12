"""Write Q2-only comparisons from frozen, midnight-aligned experiment evidence."""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/results/exp003"
REPORT = ROOT / "reports/experiments/exp003"


def build():
    manifest = json.loads((OUT / "evaluation_manifest.json").read_text())
    if manifest.get("complete") is not True:
        raise RuntimeError("Complete evaluation required")
    new_cost = pd.read_csv(OUT / "dispatch_metrics.csv")
    old_cost = pd.read_csv(OUT / "baseline/exp002_q2_costs.csv")
    new = new_cost[(new_cost.name == "primary") & (new_cost.seed == 42)].iloc[0]
    records = []

    def add(previous_id, current_id, task, metric, previous, current, unit, source,
            comparison, comparable=True, population="all", n=None):
        records.append({
            "previous_experiment": previous_id, "current_experiment": current_id,
            "task": task, "scenario": "2", "metric": metric, "population": population,
            "previous": float(previous), "current": float(current), "unit": unit,
            "absolute_change": float(current - previous) if comparable else None,
            "absolute_change_unit": "百分点" if unit == "%" else unit,
            "relative_change_pct": 100 * (current - previous) / abs(previous)
            if comparable and previous != 0 else None,
            "comparable": comparable, "comparison": comparison, "seed": 42,
            "period": "2025-02-01/2025-12-31", "n": n,
            "previous_source": source, "current_source": "data/results/exp003/",
        })

    costs = [(f"exp002/{row['name']}", row, "data/results/exp003/baseline/exp002_q2_costs.csv")
             for row in old_cost[old_cost.name.isin(
                 ["legacy_rebased", "primary", "new_deterministic", "periodic"])
                 & (old_cost.seed == 42)].to_dict("records")]
    costs += [("exp003/uncalibrated", new_cost[new_cost.name == "uncalibrated"].iloc[0],
               "data/results/exp003/dispatch_metrics.csv")]
    for previous_id, row, source in costs:
        if "legacy_rebased" in previous_id:
            previous_id = "exp001/legacy_rebased"
        for metric, unit in (("total_cost", "元"), ("planned_cost", "元"),
                             ("emergency_cost", "元"), ("emergency_kwh", "kWh"),
                             ("daily_cvar90", "元"), ("worst_day_cost", "元"),
                             ("final_soc", "kWh")):
            add(previous_id, "exp003/primary", "dispatch", metric, row[metric], new[metric],
                unit, source, "同v2物理/计费与334天；旧模型保留历史四任务训练边界", n=334)
    original = pd.read_csv(OUT / "baseline/exp001_original_q2_costs.csv").iloc[0]
    add("exp001/original", "exp003/primary", "dispatch", "total_cost", original.total_cost,
        new.total_cost, "元", "data/results/exp003/baseline/exp001_original_q2_costs.csv",
        "原v1储能效率、初始SOC和控制口径不同；保留原值，不直接排名", comparable=False, n=334)
    historical = pd.read_csv(OUT / "baseline/forecast_annual.csv")
    current = pd.read_csv(OUT / "forecast_annual.csv")
    for identity in ("exp001_selected", "exp002_seed_42"):
        for old in historical[historical.predictor_id == identity].itertuples():
            for name in ("uncalibrated", "primary"):
                row = current[(current.name == name) & (current.seed == 42)
                              & (current.target == old.target)
                              & (current.population == old.population)].iloc[0]
                if old.n != row.n:
                    raise RuntimeError("Forecast population mismatch")
                for metric in ("mae", "rmse", "wape_pct", "bias"):
                    add(identity, f"exp003/{name}", old.target, metric, getattr(old, metric),
                        row[metric], "%" if metric == "wape_pct" else "kW",
                        "data/results/exp003/baseline/forecast_annual.csv",
                        "相同午夜发布/实测标签；历史共享选模边界保留，全年为回顾性时序评估",
                        population=old.population, n=int(row.n))
    frame = pd.DataFrame(records)
    for destination in (OUT / "relative_comparison.csv", REPORT / "relative_comparison.csv"):
        frame.to_csv(destination, index=False)
    print(f"Wrote {len(frame)} Q2-only comparisons")


if __name__ == "__main__":
    build()
