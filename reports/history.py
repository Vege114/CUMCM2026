"""Reusable experiment comparison; importing this module never mutates old reports."""

import json
from pathlib import Path

COMPARABILITY_FIELDS = (
    "version", "period", "time_alignment", "billing", "efficiency", "initial_soc_kwh",
    "soc_bounds_kwh", "max_power_kw", "warmup", "terminal_condition",
)
CORE_METRICS = ("total_cost", "planned_cost", "up_cost", "down_cost", "emergency_cost",
                "emergency_kwh", "violations", "final_soc", "solve_execute_seconds")


def differences(previous, current):
    reasons = [key for key in COMPARABILITY_FIELDS
               if previous["protocol"].get(key) != current["protocol"].get(key)]
    if previous["data_hashes"] != current["data_hashes"]:
        reasons.append("data_hashes")
    if previous.get("metric_definitions") != current.get("metric_definitions"):
        reasons.append("metric_definitions")
    return reasons


def comparison_rows(previous_records, current):
    """Compare current with every prior experiment, retaining incompatible rows."""
    rows = []
    for old in previous_records:
        reasons = differences(old, current)
        old_dispatch = {str(m["scenario"]): m for m in old["metrics"]}
        pairs = [(str(m["scenario"]), "正式调度", old_dispatch[str(m["scenario"])], m, CORE_METRICS)
                 for m in current["metrics"] if str(m["scenario"]) in old_dispatch]
        old_forecasts = {(m["variant"], m["target"], m["population"]): m
                         for m in old.get("forecast_metrics", [])}
        for m in current.get("forecast_metrics", []):
            key = (m["variant"], m["target"], m["population"])
            if key in old_forecasts:
                pairs.append((m["target"], f'{m["variant"]}/{m["population"]}',
                              old_forecasts[key], m, ("mae", "rmse", "wape_pct")))
        for task, route, a, b, metrics in pairs:
            for metric in metrics:
                if metric not in a or metric not in b:
                    continue
                av, bv = a[metric], b[metric]
                rows.append({"previous_experiment": old["experiment_id"],
                             "current_experiment": current["experiment_id"], "task": task,
                             "route": route, "metric": metric, "previous": av, "current": bv,
                             "relative_change_pct": 100 * (bv - av) / abs(av)
                             if not reasons and av not in (None, 0) and bv is not None else None,
                             "comparison": ", ".join(reasons) if reasons else "同口径",
                             "previous_technical_path": " → ".join(old["technical_path"]),
                             "current_technical_path": " → ".join(current["technical_path"])})
    return rows


def read_registry(directory, exclude=None):
    return [json.loads(p.read_text()) for p in sorted(Path(directory).glob("*.json")) if p.stem != exclude]
