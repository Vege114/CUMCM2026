"""Reusable experiment comparison; importing this module never mutates old reports."""

import json
from pathlib import Path

COMPARABILITY_FIELDS = (
    "version", "period", "time_alignment", "billing", "efficiency", "initial_soc_kwh",
    "soc_bounds_kwh", "max_power_kw", "warmup", "terminal_condition", "pv_interpolation", "execution",
)
CORE_METRICS = ("total_cost", "planned_cost", "up_cost", "down_cost", "emergency_cost",
                "emergency_kwh", "violations", "final_soc", "solve_execute_seconds")


def forecast_key(metric):
    role = metric.get("role")
    # Read the legacy official-selection convention without rewriting its registry.
    if role is None and str(metric["variant"]).startswith("selected_") and "scenario" in metric:
        role = "primary"
    return (role or metric["variant"], str(metric.get("scenario", "")) if role == "primary" else "",
            metric["target"], metric["population"])


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
        old_forecasts = {forecast_key(m): m for m in old.get("forecast_metrics", [])}
        for m in current.get("forecast_metrics", []):
            key = forecast_key(m)
            if key in old_forecasts:
                route = f'正式预测问题{key[1]}/{m["population"]}' if key[0] == "primary" else f'{m["variant"]}/{m["population"]}'
                pairs.append((m["target"], route,
                              old_forecasts[key], m, ("mae", "rmse", "wape_pct")))
        for task, route, a, b, metrics in pairs:
            metric_reasons = reasons if route == "正式调度" else forecast_differences(old, current, task)
            for metric in metrics:
                if metric not in a or metric not in b:
                    continue
                av, bv = a[metric], b[metric]
                rows.append({"previous_experiment": old["experiment_id"],
                             "current_experiment": current["experiment_id"], "task": task,
                             "route": route, "metric": metric, "previous": av, "current": bv,
                             "relative_change_pct": 100 * (bv - av) / abs(av)
                             if not metric_reasons and av not in (None, 0) and bv is not None else None,
                             "comparison": ", ".join(metric_reasons) if metric_reasons else "同口径",
                             "previous_technical_path": " → ".join(old["technical_path"]),
                             "current_technical_path": " → ".join(current["technical_path"])})
    return rows


def forecast_differences(previous, current, target):
    """Battery physics does not change forecast error; PV integration can change its target."""
    fields = ["period", "time_alignment"]
    if target == "pv_corrected":
        fields.append("pv_interpolation")
    reasons = [key for key in fields if previous["protocol"].get(key) != current["protocol"].get(key)]
    if previous["data_hashes"] != current["data_hashes"]:
        reasons.append("data_hashes")
    for key in ("mae", "rmse", "wape_pct", "forecast_sample"):
        if previous.get("metric_definitions", {}).get(key) != current.get("metric_definitions", {}).get(key):
            reasons.append(f"metric_definitions.{key}")
    return reasons


def read_registry(directory, exclude=None):
    return [json.loads(p.read_text()) for p in sorted(Path(directory).glob("*.json")) if p.stem != exclude]
