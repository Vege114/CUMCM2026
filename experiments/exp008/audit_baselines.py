"""Freeze historical comparators and solve an explicitly future-informed bound."""

import argparse
import csv
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from .verify import (
    BASELINE_COST,
    BASELINE_REVERSALS,
    CAPACITY,
    ETA,
    INITIAL_SOC,
    MAX_SOC,
    MIN_SOC,
    POWER_ENERGY,
    ROOT,
    battery_metrics,
    source_arrays,
)

OUT = ROOT / "data/results/exp008"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def baselines():
    history_path = ROOT / "reports/experiments/exp006/evidence/cost_history.csv"
    battery_path = ROOT / "reports/experiments/exp006/evidence/battery_history.csv"
    batteries = {row["policy_id"]: row for row in csv.DictReader(battery_path.open())}
    rows = []
    numeric = ("total_cost", "planned_cost", "emergency_cost", "charge_kwh", "discharge_kwh",
               "throughput_kwh", "equivalent_full_cycles", "direction_reversals",
               "active_slots", "simultaneous_slots", "initial_soc", "final_soc")
    for row in csv.DictReader(history_path.open()):
        battery = batteries.get(row["policy_id"], {})
        # Keep every contemporaneous variant; report authors choose visibly by role.
        record = {key: row.get(key) for key in ("experiment", "policy_id", "label", "source", "role")}
        for key in numeric:
            value = battery.get(key) or row.get(key)
            record[key] = float(value) if value not in (None, "") else None
        record.update(scenario="2", days=334, start_date="2025-02-01", end_date="2025-12-31",
                      physically_comparable=row["physically_comparable"] == "True",
                      ranking_allowed=row["ranking_allowed"] == "True")
        if record["experiment"] in ("exp001", "exp002"):
            record["forecast_selection_limitation"] = "original shared four-target early stopping; later rescoring does not remove that boundary"
        if record["policy_id"] == "exp001/original":
            record["comparison_note"] = "incompatible single-way eta=.9 (round trip .81) and warmup initial SOC; descriptive only"
        elif "oracle" in record["policy_id"]:
            record["comparison_note"] = "future-informed seasonal decomposition; excluded from causal ranking"
        else:
            record["comparison_note"] = "same Q2 physical and settlement convention; forecast and/or controller may differ"
        rows.append(record)
    # exp005 is complete on its own branch although absent from exp006's snapshot.
    ref = "codex/q2-linear-planning"
    registry_path = "reports/registry/exp005.json"
    registry_bytes = subprocess.check_output(["git", "show", f"{ref}:{registry_path}"], cwd=ROOT)
    exp5 = json.loads(registry_bytes)
    revision = subprocess.check_output(["git", "rev-parse", ref], cwd=ROOT, text=True).strip()
    for metric in exp5["metrics"]:
        rows.append({
            **metric, "experiment": "exp005", "policy_id": f"exp005/beta_{metric['beta']}",
            "scenario": "2", "role": "primary" if metric["beta"] == .1 else "sensitivity",
            "start_date": "2025-02-01", "end_date": "2025-12-31",
            "initial_soc": INITIAL_SOC, "source": f"git:{revision}:{registry_path}",
            "source_sha256": hashlib.sha256(registry_bytes).hexdigest(),
            "physically_comparable": False, "ranking_allowed": False,
            "settlement_comparable": True, "simultaneous_slots": metric["overlap_intervals"],
            "direction_reversals": None, "active_slots": None,
            "comparison_note": "completed LP simulation; additional 1000 kW/10min hard ramp and emergency charging allowed; show separately as descriptive historical comparator",
        })
    # Newer branches exp003/4/6 update only Q2; don't invent Q3/4 outcomes.
    cross_question = []
    with (ROOT / "data/results/exp002/dispatch_metrics.csv").open() as stream:
        for row in csv.DictReader(stream):
            if row.get("name") == "primary" and row.get("seed") == "42" and row["scenario"] != "2":
                cross_question.append({"experiment": "exp002", "scenario": row["scenario"],
                                       "total_cost": float(row["total_cost"]),
                                       "source": "data/results/exp002/dispatch_metrics.csv",
                                       "comparison_note": "archived older forecast/controller, descriptive only until a same-current-forecast replay is supplied"})
    result = {
        "primary_comparator": "exp006/primary", "evaluation_days": 334,
        "warmup_source": "data/results/exp003/warmup_2.npz", "initial_soc": INITIAL_SOC,
        "initial_mode": 1, "initial_power_kw": 172.75999999999976,
        "eta_charge": ETA, "eta_discharge": ETA, "capacity_kwh": CAPACITY,
        "max_power_kw": 5000, "settlement": "sum(p*locked_purchase + 5*p*emergency); no artificial terminal credit or secondary penalty in bill",
        "goal_cost_8pct": BASELINE_COST * .92, "goal_cost_10pct": BASELINE_COST * .9,
        "goal_reversals_strictly_below": BASELINE_REVERSALS,
        "metric_definition_source": "experiments/problem2/tree_planning/verify.py:battery_metrics",
        "metric_definition": {
            "direction_reversals": "remove idle slots then count chronological sign flips; cross-day included; warmup boundary excluded",
            "active_slots": "10-minute intervals with charge or discharge greater than 1e-6 kWh",
            "equivalent_full_cycles": "(eta*sum(charge)+sum(discharge)/eta)/(2*12000)",
        },
        "q2_rows": sorted(rows, key=lambda row: row["experiment"]),
        "older_other_question_rows": cross_question,
        "question_coverage": {"exp001": ["2", "3", "4-2", "4-3"], "exp002": ["2", "3", "4-2", "4-3"],
                              "exp003": ["2"], "exp004": ["2"], "exp005": ["2"], "exp006": ["2"]},
        "ignored_experiments": ["exp007"],
        "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in (history_path, battery_path)},
    }
    write_json(OUT / "baselines.json", result)
    return result


def oracle_lower_bound():
    """One continuous 48,096-slot LP, with actual future net load known.

    The positive-price deterministic LP does not need emergency purchase:
    planned grid energy is cheaper.  Simultaneous c,d can be cancelled at
    constant SOC and cost by increasing free curtailment, so its optimum
    admits mutually exclusive physical actions.  This is a bound for every
    feasible causal Q2 policy, not a proposed strategy or a success claim.
    """
    began = time.perf_counter()
    actual, price144 = source_arrays("2")
    net = ((actual[..., 0] - actual[..., 1]) / 6).ravel()
    price = np.tile(price144, len(actual))
    n = len(net)
    t = np.arange(n)
    g, c, d, w, s = (t + block * n for block in range(5))
    row = np.r_[t, t, t, t, n + t, n + t, n + t, n + t[1:]]
    col = np.r_[g, c, d, w, s, c, d, s[:-1]]
    vals = np.r_[np.ones(n), -np.ones(n), np.ones(n), -np.ones(n),
                 np.ones(n), np.full(n, -ETA), np.full(n, 1 / ETA), -np.ones(n - 1)]
    matrix = coo_matrix((vals, (row, col)), shape=(2 * n, 5 * n)).tocsc()
    rhs = np.r_[net, INITIAL_SOC, np.zeros(n - 1)]
    objective = np.r_[price, np.zeros(4 * n)]
    lower = np.r_[np.zeros(4 * n), np.full(n, MIN_SOC)]
    upper = np.r_[np.full(n, np.inf), np.full(2 * n, POWER_ENERGY),
                  np.full(n, np.inf), np.full(n, MAX_SOC)]
    result = linprog(objective, A_eq=matrix, b_eq=rhs, bounds=np.column_stack((lower, upper)),
                     method="highs", options={"primal_feasibility_tolerance": 1e-8,
                                              "dual_feasibility_tolerance": 1e-8})
    if not result.success:
        raise RuntimeError(f"Ideal LP lower bound failed: {result.message}")
    x = result.x.copy()
    # Cancel overlap while preserving each SOC transition; release losses as surplus.
    cancel = np.minimum(x[c], x[d] / (ETA * ETA))
    x[c] -= cancel
    x[d] -= ETA * ETA * cancel
    x[w] += (1 - ETA * ETA) * cancel
    residual = float(np.max(np.abs(matrix @ x - rhs)))
    overlap = int(np.sum((x[c] > 1e-6) & (x[d] > 1e-6)))
    if residual > 1e-6 or overlap:
        raise RuntimeError("Oracle purification failed independent constraint check")
    # Dual objective, including finite bound contributions, certifies the optimum.
    finite_upper = np.isfinite(upper)
    dual = (float(np.sum(rhs * result.eqlin.marginals))
            + float(np.sum(lower * result.lower.marginals))
            + float(np.sum(upper[finite_upper] * result.upper.marginals[finite_upper])))
    bound = float(result.fun)
    state_trace = np.r_[INITIAL_SOC, x[s]]
    battery = battery_metrics({
        "charge": x[c].reshape(334, 144), "discharge": x[d].reshape(334, 144),
        "states": np.stack([state_trace[day * 144:(day + 1) * 144 + 1] for day in range(334)]),
    })
    summary = {
        "scenario": "2", "information_regime": "perfect foresight of every 2025-02-01..12-31 actual load/PV slot",
        "oracle_uses_future_actual": True, "eligible_as_causal_strategy": False,
        "use": "theoretical physical-economic lower bound only; excluded from strategy ranking and goal pass",
        "days": 334, "intervals": n, "initial_soc": INITIAL_SOC, "final_soc": float(x[s][-1]),
        "cross_day_soc": "one continuous annual state trajectory; no daily reset",
        "terminal_soc_constraint": "only [1200,10800] bounds; no terminal credit",
        "eta_charge": ETA, "eta_discharge": ETA, "max_power_kw": 5000,
        "lower_bound_yuan": bound, "dual_objective_yuan": dual, "primal_dual_gap_yuan": abs(bound - dual),
        "baseline_cost_yuan": BASELINE_COST, "maximum_possible_reduction_pct": 100 * (1 - bound / BASELINE_COST),
        "target_8pct_yuan": .92 * BASELINE_COST, "target_10pct_yuan": .9 * BASELINE_COST,
        "8pct_not_ruled_out_by_physics": bool(bound <= .92 * BASELINE_COST),
        "10pct_not_ruled_out_by_physics": bool(bound <= .9 * BASELINE_COST),
        "interpretation": "a lower bound below the target does not prove attainability under causal forecasts or reduced battery-action constraints",
        "max_constraint_residual_kwh": residual, "simultaneous_slots_after_purification": overlap,
        "charge_kwh": float(x[c].sum()), "discharge_kwh": float(x[d].sum()),
        "throughput_kwh": float(x[c].sum() + x[d].sum()),
        "battery_metrics": battery,
        "solver": "scipy.optimize.linprog HiGHS, continuous annual LP",
        "solver_status": int(result.status), "solver_message": str(result.message),
        "solve_and_verify_seconds": time.perf_counter() - began,
        "source_sha256": {name: hashlib.sha256((ROOT / "data/raw" / name).read_bytes()).hexdigest()
                          for name in ("附件1.csv", "附件2_小区负载.csv", "附件2_光伏发电实际功率.csv")},
    }
    write_json(OUT / "oracle_lower_bound.json", summary)
    return summary


def fixed_purchase_oracle(path=None):
    """Diagnostic: best future-informed battery execution for frozen Q.

    Purchases cannot change; only remaining paid energy can charge storage.
    No emergency charging or discharging into surplus is allowed. This is
    an optimistic bound without the candidate's action/deadband restrictions.
    """
    if path is None:
        candidates = []
        for source in (OUT / "controller_candidates").glob("*/summary.json"):
            record = json.loads(source.read_text())
            if record.get("days") == 334:
                candidates.append((float(record["total_cost"]), source.parent / "dispatch_2.npz"))
        if not candidates:
            raise RuntimeError("No completed 334-day candidate exists")
        path = min(candidates)[1]
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        purchase = archive["original"].ravel().copy()
        original_fees = archive["fees"].copy()
    actual, price144 = source_arrays("2")
    net = ((actual[..., 0] - actual[..., 1]) / 6).ravel()
    price = np.tile(price144, 334)
    n = len(net)
    if purchase.shape != (n,):
        raise ValueError("Fixed-purchase oracle requires the full 334-day Q2 archive")
    began = time.perf_counter()
    balance = purchase - net
    t = np.arange(n)
    c, d, e, w, s = (t + block * n for block in range(5))
    row = np.r_[t, t, t, t, n + t, n + t, n + t, n + t[1:]]
    col = np.r_[c, d, e, w, s, c, d, s[:-1]]
    values = np.r_[-np.ones(n), np.ones(n), np.ones(n), -np.ones(n),
                   np.ones(n), np.full(n, -ETA), np.full(n, 1 / ETA), -np.ones(n - 1)]
    matrix = coo_matrix((values, (row, col)), shape=(2 * n, 5 * n)).tocsc()
    rhs = np.r_[-balance, INITIAL_SOC, np.zeros(n - 1)]
    lower = np.r_[np.zeros(4 * n), np.full(n, MIN_SOC)]
    upper = np.r_[np.minimum(np.maximum(balance, 0), POWER_ENERGY),
                  np.minimum(np.maximum(-balance, 0), POWER_ENERGY),
                  np.full(2 * n, np.inf), np.full(n, MAX_SOC)]
    objective = np.r_[np.zeros(2 * n), 5 * price, np.zeros(2 * n)]
    solution = linprog(objective, A_eq=matrix, b_eq=rhs,
                       bounds=np.column_stack((lower, upper)), method="highs",
                       options={"primal_feasibility_tolerance": 1e-8,
                                "dual_feasibility_tolerance": 1e-8})
    if not solution.success:
        raise RuntimeError(f"Fixed-purchase oracle failed: {solution.message}")
    x = solution.x
    max_residual = float(np.max(np.abs(matrix @ x - rhs)))
    finite_upper = np.isfinite(upper)
    dual = (float(np.sum(rhs * solution.eqlin.marginals))
            + float(np.sum(lower * solution.lower.marginals))
            + float(np.sum(upper[finite_upper] * solution.upper.marginals[finite_upper])))
    trace = np.r_[INITIAL_SOC, x[s]]
    battery = battery_metrics({
        "charge": x[c].reshape(334, 144), "discharge": x[d].reshape(334, 144),
        "states": np.stack([trace[day * 144:(day + 1) * 144 + 1] for day in range(334)]),
    })
    planned = float(np.sum(purchase * price))
    total = planned + float(solution.fun)
    result = {
        "scenario": "2", "source_fixed_purchase": str(path),
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "information_regime": "fixed causal historical purchases; battery sees every future actual net-load slot",
        "oracle_uses_future_actual": True, "eligible_as_causal_strategy": False,
        "use": "execution-only attainable lower bound diagnostic, not a deployable strategy or goal achievement",
        "days": 334, "initial_soc": INITIAL_SOC, "cross_day_soc": "one continuous annual state trajectory",
        "emergency_charging_allowed": False, "discharge_into_surplus_allowed": False,
        "deadband_ramp_mode_constraints": "relaxed; bound is optimistic relative to low-action candidate",
        "frozen_planned_cost_yuan": planned, "minimum_emergency_cost_yuan": float(solution.fun),
        "minimum_total_cost_yuan": total, "emergency_dual_objective_yuan": dual,
        "primal_dual_gap_yuan": abs(float(solution.fun) - dual),
        "candidate_actual_total_cost_yuan": float(original_fees.sum()),
        "candidate_actual_emergency_cost_yuan": float(original_fees[..., 3].sum()),
        "maximum_saving_by_execution_only_yuan": float(original_fees.sum()) - total,
        "cost_reduction_vs_exp006_pct": 100 * (1 - total / BASELINE_COST),
        "execution_only_8pct_not_ruled_out": bool(total <= .92 * BASELINE_COST),
        "execution_only_10pct_not_ruled_out": bool(total <= .9 * BASELINE_COST),
        "gap_to_8pct_target_yuan": total - .92 * BASELINE_COST,
        "gap_to_10pct_target_yuan": total - .9 * BASELINE_COST,
        "battery_metrics": battery, "max_constraint_residual_kwh": max_residual,
        "solver_status": int(solution.status), "solve_and_verify_seconds": time.perf_counter() - began,
    }
    write_json(OUT / "fixed_purchase_oracle.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=("baselines", "oracle", "fixed-purchase"))
    parser.add_argument("--archive", type=Path, help="optional frozen Q2 purchases for the fixed-purchase bound")
    args = parser.parse_args()
    if args.only in (None, "baselines"):
        result = baselines()
        print(f"Wrote {len(result['q2_rows'])} historical rows: {OUT / 'baselines.json'}")
    if args.only in (None, "oracle"):
        print(json.dumps(oracle_lower_bound(), ensure_ascii=False, indent=2))
    if args.only == "fixed-purchase":
        print(json.dumps(fixed_purchase_oracle(args.archive), ensure_ascii=False, indent=2))
