"""Question 2 fixed midnight purchases and causal physical replay.

Forecast/actual columns are load and PV in kW. Purchase, battery, emergency,
surplus and SOC arrays are in kWh; prices are yuan/kWh and fees are yuan.
The frozen v2 numerical model supplies the solver, its unchanged two-second
budget, sqrt(0.9) charge/discharge efficiencies, and the greedy executor.
This module neither reads files nor selects prediction/calibration parameters.
"""

import time
from itertools import pairwise

import numpy as np
import pandas as pd

from experiments.common.neural_v2.physics import (
    MAX_SOC,
    MIN_SOC,
    deterministic,
    execute,
    settle,
    validate_detail,
)
from experiments.common.neural_v2.risk import weighted_cvar

from .data import DAYS, EPOCH, STEPS

INITIAL_SOC = 6000.0
FEE_COMPONENTS = ("planned_cost", "up_cost", "down_cost", "emergency_cost")
SUM_METRICS = (
    "planned_kwh", "final_kwh", "emergency_kwh", "emergency_minutes",
    "charge_kwh", "discharge_kwh", "surplus_kwh", *FEE_COMPONENTS,
    "total_cost", "violations", "solve_execute_seconds", "solver_calls",
    "timeout_count", "fallback_count", "incumbent_count", "gap_certified_count",
)


def _day(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError("day must be a zero-based integer day of 2025")
    if not 0 <= value < DAYS:
        raise ValueError("day must be between 0 and 364")
    return int(value)


def _initial_soc(value, day):
    value = float(value)
    if not np.isfinite(value) or not MIN_SOC <= value <= MAX_SOC:
        raise ValueError("initial_soc must be finite and within [1200, 10800] kWh")
    if day == 0 and not np.isclose(value, INITIAL_SOC, rtol=0, atol=1e-6):
        raise ValueError("January 1 warmup must start at 6000 kWh")
    return value


def _nonnegative(values, shape, name):
    values = np.asarray(values, dtype=float)
    if values.shape != shape or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError(f"{name} must have shape {shape} with finite nonnegative values")
    return values.copy()


def day_run(data, forecast, day, initial_soc):
    """Solve once at 00:00, then replay 144 realized intervals.

    ``data`` exposes ``fixed_price[144]`` and ``actual[days*144,2]`` only.
    ``forecast[144,2]`` must have been produced using information before the
    day's origin. Its provenance is the caller's responsibility. No actual
    observation from this day is accessed until the complete plan is fixed.

    Return ``(summary, detail, solver)``. Detail follows v2 names and shapes,
    except ``actual`` has exactly two columns. ``states`` has 145 boundary
    nodes. ``fees[144,4]`` retains planned/up/down/emergency order; up/down
    are identically zero because ``original`` and ``final`` are identical.
    """
    day = _day(day)
    initial_soc = _initial_soc(initial_soc, day)
    forecast = _nonnegative(forecast, (STEPS, 2), "forecast (kW)")
    price = _nonnegative(data.fixed_price, (STEPS,), "fixed_price (yuan/kWh)")
    origin = day * STEPS
    began = time.monotonic()
    plan, metadata = deterministic(forecast[:, 0], forecast[:, 1], price, initial_soc)
    original = _nonnegative(plan, (STEPS,), "midnight purchase plan (kWh)")
    final = original.copy()

    # Realizations enter only physical execution, after the midnight plan is locked.
    actual = _nonnegative(data.actual[origin:origin + STEPS], (STEPS, 2), "actual (kW)")
    charge, discharge, emergency, surplus, states = execute(
        final, actual[:, 0], actual[:, 1], initial_soc,
    )
    fees = settle(original, final, emergency, price)
    detail = {
        "original": original, "final": final, "charge": charge,
        "discharge": discharge, "emergency": emergency, "surplus": surplus,
        "states": states, "fees": fees, "price": price, "actual": actual,
    }
    balance_error, state_error = validate_detail(detail)
    solver = {
        **metadata, "origin": origin, "day": day, "scenario": "2",
        "information_cutoff": origin, "issued_forecast_allowed": False,
        "known_future_price": False, "fixed_tariff": True,
    }
    date = EPOCH + pd.Timedelta(days=day)
    summary = {
        "day": day, "date": str(date.date()), "month": int(date.month), "scenario": "2",
        "planned_kwh": float(original.sum()), "final_kwh": float(final.sum()),
        "emergency_kwh": float(emergency.sum()),
        "emergency_minutes": int((emergency > 1e-6).sum() * 10),
        "charge_kwh": float(charge.sum()), "discharge_kwh": float(discharge.sum()),
        "surplus_kwh": float(surplus.sum()),
        "initial_soc": float(states[0]), "final_soc": float(states[-1]),
        **dict(zip(FEE_COMPONENTS, map(float, fees.sum(axis=0)))),
        "total_cost": float(fees.sum()), "violations": 0,
        "max_balance_error": balance_error, "max_state_error": state_error,
        "solve_execute_seconds": time.monotonic() - began, "solver_calls": 1,
        "timeout_count": int(solver["status"] == 1),
        "fallback_count": int(solver["fallback"]), "incumbent_count": 0,
        "gap_certified_count": int(solver.get("mip_gap") is not None
                                   and solver["mip_gap"] <= .010001),
        "weight": 0.0, "updates": 0,
    }
    return summary, detail, solver


def replay_days(data, days, predict_callable, initial_soc):
    """Replay nonempty consecutive integer days, carrying realized SOC forward.

    ``predict_callable(origin)`` runs once at each midnight, before the
    optimizer and before that day's physical observations are read. The
    return value is ``(daily_summaries, stacked_detail, solver_logs)`` with
    the day axis first in every stacked detail array. No results are saved.
    """
    days = [_day(day) for day in days]
    if not days or np.any(np.diff(days) != 1):
        raise ValueError("days must be a nonempty consecutive increasing sequence")
    state = _initial_soc(initial_soc, days[0])
    summaries, details, solvers = [], [], []
    for day in days:
        forecast = predict_callable(day * STEPS)
        summary, detail, solver = day_run(data, forecast, day, state)
        summaries.append(summary)
        details.append(detail)
        solvers.append(solver)
        state = summary["final_soc"]
    stacked = {key: np.stack([detail[key] for detail in details]) for key in details[0]}
    np.testing.assert_allclose(stacked["states"][:-1, -1], stacked["states"][1:, 0],
                               rtol=0, atol=1e-6)
    return summaries, stacked, solvers


def aggregate(summaries):
    """Sum replay metrics and report empirical, equally weighted daily CVaR90.

    Works for calibration, warmup, and full-year evaluation without assuming
    334 observations. It does not compare or select candidate strategies.
    """
    summaries = list(summaries)
    days = [_day(row["day"]) for row in summaries]
    if not days or np.any(np.diff(days) != 1):
        raise ValueError("summaries must contain nonempty consecutive increasing days")
    if any(row["scenario"] != "2" for row in summaries):
        raise ValueError("only Question 2 summaries may be aggregated")
    for previous, current in pairwise(summaries):
        if not np.isclose(previous["final_soc"], current["initial_soc"], rtol=0, atol=1e-6):
            raise ValueError("summary SOC is not continuous across days")
    costs = np.asarray([row["total_cost"] for row in summaries], dtype=float)
    if not np.isfinite(costs).all() or (costs < 0).any():
        raise ValueError("daily total costs must be finite and nonnegative")
    worst = summaries[int(np.argmax(costs))]
    return {
        "scenario": "2", "days": len(summaries),
        "start_date": summaries[0]["date"], "end_date": summaries[-1]["date"],
        **{key: sum(row[key] for row in summaries) for key in SUM_METRICS},
        "initial_soc": summaries[0]["initial_soc"], "final_soc": summaries[-1]["final_soc"],
        "daily_cvar90": weighted_cvar(costs, np.full(len(costs), 1 / len(costs))),
        "worst_day_cost": worst["total_cost"], "worst_date": worst["date"],
        "max_balance_error": max(row["max_balance_error"] for row in summaries),
        "max_state_error": max(row["max_state_error"] for row in summaries),
    }
