"""Exploratory causal inventory DP conditioned on a three-bin error state.

Historical net-demand errors alone fit time-specific location, scale, emissions
and shrunken transitions.  The three bins are an approximate observed error
state, not identified weather regimes.  Purchases are fixed before execution.
"""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

ETA = float(np.sqrt(.9))
LOW, HIGH, LIMIT = 1200., 10800., 5000 / 6
MODES = (-1, 0, 1)
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/results/exp008/controller_candidates"


def fit_error_model(forecast_net_kwh, historical_errors_kwh, *, history_origins=None,
                    cutoff=None, shrinkage=12., independent=False):
    """Pure fit from 1..28 complete historical error paths (history,time).

    Zero paths raise: callers must supply explicitly labelled causal periodic
    historical errors, never treat quantile columns as independent days. One
    path is accepted with a 10-kWh scale floor and shrunken pooled transitions.
    ``cutoff`` and origins are exclusive interval indices, when supplied.
    """
    net = np.asarray(forecast_net_kwh, dtype=float)
    errors = np.asarray(historical_errors_kwh, dtype=float)
    if net.ndim != 1 or errors.ndim != 2 or errors.shape[1] != len(net) or not 1 <= len(errors) <= 28:
        raise ValueError("need a net trajectory and 1..28 equally long complete historical paths")
    if not np.isfinite(net).all() or not np.isfinite(errors).all() or shrinkage < 0:
        raise ValueError("nonfinite inputs or negative shrinkage")
    origins = None if history_origins is None else np.asarray(history_origins, dtype=int)
    if origins is not None:
        if origins.shape != (len(errors),) or np.any(np.diff(origins) <= 0):
            raise ValueError("history origins must be ordered, unique and match the paths")
        if cutoff is None or np.any(origins + len(net) > cutoff):
            raise ValueError("every historical horizon must complete before the exclusive cutoff")
    mu = errors.mean(axis=0)
    scale = np.maximum(errors.std(axis=0), 10.)
    edges = np.array([-.55, .55])
    bins = np.searchsorted(edges, (errors - mu) / scale, side="right")
    h, n = errors.shape
    emission = np.empty((n, 3))
    probabilities = np.empty((n, 3))
    for t in range(n):
        counts = np.bincount(bins[:, t], minlength=3)
        probabilities[t] = (counts + .5) / (h + 1.5)
        for state in range(3):
            selected = errors[bins[:, t] == state, t]
            emission[t, state] = selected.mean() if len(selected) else mu[t] + (state - 1) * scale[t]
    local_counts = np.zeros((n, 3, 3))
    for t in range(1, n):
        np.add.at(local_counts[t], (bins[:, t - 1], bins[:, t]), 1.)
    consecutive = np.ones(max(0, h - 1), dtype=bool) if origins is None else np.diff(origins) == n
    if h > 1:
        np.add.at(local_counts[0], (bins[:-1, -1][consecutive], bins[1:, 0][consecutive]), 1.)
    pooled = local_counts[1:].sum(axis=0)
    pooled_next = probabilities.mean(axis=0)
    pooled_transition = (pooled + 3 * pooled_next[None, :]) / (pooled.sum(axis=1, keepdims=True) + 3)
    transition = np.empty_like(local_counts)
    for t in range(n):
        count = local_counts[t]
        prior = np.broadcast_to(probabilities[t], (3, 3)) if t == 0 else pooled_transition
        numerator = count + shrinkage * prior
        denominator = count.sum(axis=1, keepdims=True) + shrinkage
        transition[t] = np.divide(numerator, denominator, out=prior.copy(), where=denominator > 0)
        if independent:
            transition[t] = probabilities[t][None, :]
    return {
        "forecast_net_kwh": net.copy(), "mean_kwh": mu, "scale_kwh": scale,
        "state_edges": edges, "error_support_kwh": emission,
        "transition": transition, "marginal_probabilities": probabilities,
        "initial_error_state": int(bins[-1, -1]),
        "metadata": {
            "history_paths": h, "history_origins": None if origins is None else origins.tolist(),
            "information_cutoff_exclusive": None if cutoff is None else int(cutoff),
            "maximum_historical_label": None if origins is None else int(origins[-1] + n - 1),
            "scale_floor_kwh": 10., "state_edges_standardized": edges.tolist(),
            "transition_shrinkage_pseudocount": shrinkage,
            "initial_error_source": "last completed historical path's terminal standardized residual",
            "midnight_transition_prior": "current first-slot marginal; across-complete-day counts only",
            "disturbance_assumption": "independent_marginals" if independent else "three_state_first_order_error_markov_approximation",
            "not_a_weather_regime_identification": True,
        },
    }


def value_functions(purchase, model, price, *, grid_kwh=200., wear=.002,
                    switching=100., terminal=.45):
    """Return V[t,previous_error_state,previous_mode+1,SOC_grid].

    V is before observing t's disturbance. The current error is revealed only
    in the stage expectation; continuation conditions on that current bin.
    The optimum is for this grid/interpolation approximation, not continuous
    storage or an identified true stochastic process.
    """
    began = perf_counter()
    q, price = np.asarray(purchase, float), np.asarray(price, float)
    net = np.asarray(model["forecast_net_kwh"], float)
    support = net[:, None] + np.asarray(model["error_support_kwh"], float)
    transition = np.asarray(model["transition"], float)
    n = len(q)
    if (q.shape != (n,) or net.shape != (n,) or price.shape != (n,)
            or support.shape != (n, 3) or transition.shape != (n, 3, 3)
            or np.any(q < 0) or np.any(price <= 0) or grid_kwh <= 0
            or grid_kwh > HIGH - LOW or min(wear, switching, terminal) < 0):
        raise ValueError("invalid trajectory, grid, transition, or auxiliary cost")
    if not all(np.isfinite(v).all() for v in (q, price, support, transition)):
        raise ValueError("nonfinite planning inputs")
    if np.any(transition < 0) or not np.allclose(transition.sum(axis=-1), 1):
        raise ValueError("transition rows must be probabilities")
    grid = np.r_[np.arange(LOW, HIGH, grid_kwh), HIGH]
    values = np.zeros((n + 1, 3, 3, len(grid)))
    values[-1] = -terminal * (grid - LOW)
    delta = grid[None, :] - grid[:, None]
    for t in range(n - 1, -1, -1):
        observed_value = np.empty((3, 3, len(grid)))
        for state in range(3):
            balance = q[t] - support[t, state]
            direction = 1 if balance >= 0 else -1
            future = values[t + 1, state]
            nextv = future[direction + 1]
            if balance >= 0:
                available = np.minimum(min(balance, LIMIT), (HIGH - grid) / ETA)
                endpoint = grid + ETA * available
                eligible = (delta > 1e-8) & (delta <= ETA * available[:, None] + 1e-8)
                costs = nextv[None, :] + wear * delta / ETA
                end_value = np.interp(endpoint, grid, nextv) + wear * available
            else:
                available = np.minimum(min(-balance, LIMIT), (grid - LOW) * ETA)
                endpoint = grid - available / ETA
                eligible = (delta < -1e-8) & (delta >= -available[:, None] / ETA - 1e-8)
                costs = nextv[None, :] + 5 * price[t] * (-balance + ETA * delta) - wear * ETA * delta
                end_value = np.interp(endpoint, grid, nextv) + 5 * price[t] * (-balance - available) + wear * available
            active = np.minimum(np.min(np.where(eligible, costs, np.inf), axis=1),
                                np.where(available > 1e-8, end_value, np.inf))
            for mi, mode in enumerate(MODES):
                idle = future[mi] + 5 * price[t] * max(-balance, 0)
                switched = active + (switching if mode * direction == -1 else 0.)
                observed_value[state, mi] = np.minimum(idle, switched)
        values[t] = np.einsum("ij,jms->ims", transition[t], observed_value, optimize=False)
    return grid, values, {
        "method": "inventory_value_conditioned_on_current_error_and_last_direction",
        "seconds": perf_counter() - began, "grid_kwh": grid_kwh,
        "error_states": 3, "direction_states": 3, "wear_yuan_per_kwh_auxiliary": wear,
        "switching_yuan_auxiliary": switching, "terminal_value_yuan_per_soc_kwh": terminal,
        "values_axes": ["time_before_observation", "previous_error_state", "last_nonidle_direction_index", "soc"],
        "calibration": model["metadata"], "continuous_global_optimality_certificate": False,
    }


def initial_value(grid, values, model, soc, mode=1):
    if mode not in MODES or not LOW <= soc <= HIGH:
        raise ValueError("invalid initial mode or SOC")
    return float(np.interp(soc, grid, values[0, model["initial_error_state"], mode + 1]))


def execute(purchase, actual_kw, price, initial_soc, grid, values, model, *,
            wear=.002, switching=100., initial_mode=1):
    """Observe current net-demand error, then optimize its feasible one-step action."""
    q, actual, price = (np.asarray(value, float) for value in (purchase, actual_kw, price))
    n = len(q)
    if actual.shape[0] != n or actual.ndim != 2 or actual.shape[1] < 2:
        raise ValueError("actual must contain load/PV kW for matching slots")
    if not LOW <= initial_soc <= HIGH or initial_mode not in MODES:
        raise ValueError("invalid physical initial state")
    c, d, e, w = (np.zeros(n) for _ in range(4))
    states = np.empty(n + 1)
    states[0] = initial_soc
    error_states = np.empty(n, dtype=np.int8)
    mode = initial_mode
    for t in range(n):
        observed_net = (actual[t, 0] - actual[t, 1]) / 6
        standardized = (observed_net - model["forecast_net_kwh"][t] - model["mean_kwh"][t]) / model["scale_kwh"][t]
        state = int(np.searchsorted(model["state_edges"], standardized, side="right"))
        error_states[t] = state
        balance = q[t] - observed_net
        direction = 1 if balance >= 0 else -1
        current = states[t]
        future = values[t + 1, state]
        idle = np.interp(current, grid, future[mode + 1]) + 5 * price[t] * max(-balance, 0)
        if balance >= 0:
            available = min(balance, LIMIT, max(0., (HIGH - current) / ETA))
            endpoint = current + ETA * available
            choices = np.r_[grid[(grid > current) & (grid <= endpoint)], endpoint]
            costs = np.interp(choices, grid, future[2]) + wear * (choices - current) / ETA
        else:
            available = min(-balance, LIMIT, max(0., (current - LOW) * ETA))
            endpoint = current - available / ETA
            choices = np.r_[grid[(grid >= endpoint) & (grid < current)], endpoint]
            released = (current - choices) * ETA
            costs = np.interp(choices, grid, future[0]) + 5 * price[t] * (-balance - released) + wear * released
        costs += switching if mode * direction == -1 else 0.
        best = int(np.argmin(costs))
        following = float(choices[best]) if available > 1e-8 and costs[best] < idle - 1e-8 else current
        c[t], d[t] = max(0., (following - current) / ETA), max(0., (current - following) * ETA)
        if c[t] + d[t] > 1e-8:
            mode = direction
        e[t], w[t] = max(0., -balance - d[t]), max(0., balance - c[t])
        states[t + 1] = following
    fees = np.stack((q * price, np.zeros(n), np.zeros(n), 5 * e * price), axis=-1)
    detail = {"original": q.copy(), "final": q.copy(), "charge": c, "discharge": d,
              "emergency": e, "surplus": w, "states": states, "fees": fees,
              "actual": actual.copy(), "price": price.copy(), "observed_error_state": error_states}
    return detail, mode


def model_for_day(data, store, day, independent=False):
    """Causal adapter; no current-day actual observations are accessed here."""
    from experiments.problem2.tree_planning.risk import periodic_baseline

    origin = day * 144
    ids = np.flatnonzero(store.origins + 144 <= origin)[-28:]
    errors, origins = [], []
    fallback = len(ids) < 2
    if fallback:
        for old_day in range(max(1, day - 28), day):
            issue = old_day * 144
            yesterday = data.actual[issue - 144:issue, :2]
            previous_week = data.actual[issue - 7 * 144:issue - 6 * 144, :2] if old_day >= 7 else None
            forecast = periodic_baseline(yesterday, previous_week)
            truth = data.actual[issue:issue + 144, :2]
            errors.append(((truth[:, 0] - truth[:, 1]) - (forecast[:, 0] - forecast[:, 1])) / 6)
            origins.append(issue)
    else:
        for index in ids:
            issue = int(store.origins[index])
            truth = data.actual[issue:issue + 144, :2]
            forecast = store.values[index]
            errors.append(((truth[:, 0] - truth[:, 1]) - (forecast[:, 0] - forecast[:, 1])) / 6)
            origins.append(issue)
    current = store.get(origin)
    model = fit_error_model((current[:, 0] - current[:, 1]) / 6, np.asarray(errors),
                            history_origins=origins, cutoff=origin, independent=independent)
    variant = getattr(store, "name", None)
    source = "frozen_exp004_forecast" if variant in (None, "base") else f"causally_calibrated_exp004_{variant}"
    model["metadata"].update(residual_source="periodic_baseline" if fallback else source,
                              fallback=fallback, calibration_name=variant)
    return model


def pilot(switching=100., days=30, method="markov"):
    """Each executor carries its own SOC into the next q=.8 purchase LP."""
    from experiments.exp008.controller_candidate import plan_inventory, risk_cache
    from experiments.exp008.impulse_feedback import execute as independent_execute
    from experiments.exp008.impulse_feedback import value_functions as independent_values
    from experiments.exp008.verify import INITIAL_SOC, battery_metrics, verify_arrays
    from experiments.problem2.exp003.data import Data
    from experiments.problem2.exp004.predict import ForecastStore

    if method not in ("markov", "independent3", "impulse"):
        raise ValueError("unknown feedback method")
    began = perf_counter()
    case_id = f"mk_{method}_q080_k{switching:g}_g200_d{days}"
    path = OUT / case_id
    if (path / "summary.json").exists():
        raise RuntimeError(f"Refusing to overwrite an exploratory pilot: {path}")
    data, store = Data(), ForecastStore("no_season", seed=42)
    supports, source_audits = risk_cache(data, store)
    soc, mode = INITIAL_SOC, 1
    parts, rows, audits = [], [], []
    for day in range(31, 31 + days):
        origin = day * 144
        plan = plan_inventory(supports[day - 31], data.fixed_price, soc, {"quantile": .8}, final=day == 364)
        terminal = 0. if day == 364 else data.fixed_price.min() / ETA
        if method == "impulse":
            grid, values, metadata = independent_values(plan["purchase"], supports[day - 31], data.fixed_price,
                                                        grid_kwh=200., switching=switching, terminal=terminal)
            detail, nextmode = independent_execute(plan["purchase"], data.actual[origin:origin + 144],
                data.fixed_price, soc, grid, values, switching=switching, initial_mode=mode)
        else:
            model = model_for_day(data, store, day, independent=method == "independent3")
            grid, values, metadata = value_functions(plan["purchase"], model, data.fixed_price,
                grid_kwh=200., switching=switching, terminal=terminal)
            metadata["initial_model_value_yuan"] = initial_value(grid, values, model, soc, mode)
            detail, nextmode = execute(plan["purchase"], data.actual[origin:origin + 144], data.fixed_price,
                                      soc, grid, values, model, switching=switching, initial_mode=mode)
        parts.append(detail)
        audits.append({**source_audits[day - 31], "forecast_origin": origin,
                       "execution_initial_soc": soc, "execution_initial_mode": mode,
                       "execution_final_soc": float(detail["states"][-1]), "execution_final_mode": nextmode,
                       "feedback_model": metadata, "purchase_locked_before_actual_read": True})
        rows.append({"day": day, "date": str((pd.Timestamp("2025-01-01") + pd.Timedelta(days=day)).date()),
                     "total_cost": float(detail["fees"].sum()), "planned_cost": float(detail["fees"][:, 0].sum()),
                     "emergency_cost": float(detail["fees"][:, 3].sum()), "initial_soc": soc,
                     "final_soc": float(detail["states"][-1])})
        soc, mode = float(detail["states"][-1]), nextmode
    arrays = {key: np.stack([part[key] for part in parts]) for key in parts[0]}
    verification = verify_arrays(arrays, expected_days=days, source_actual=data.actual[31 * 144:(31 + days) * 144].reshape(days, 144, 2),
                                 source_price=data.fixed_price, audit_records=audits)
    if not verification["passed"]:
        raise RuntimeError(f"Pilot validation failed: {verification['errors']}")
    summary = {"spec": {"id": case_id, "method": method, "switching": switching,
                        "grid_kwh": 200., "quantile": .8}, "days": days,
               "total_cost": float(arrays["fees"].sum()), "planned_cost": float(arrays["fees"][..., 0].sum()),
               "emergency_cost": float(arrays["fees"][..., 3].sum()), **battery_metrics(arrays),
               "target_eligible": days == 334, "exploratory": True,
               "comparison_semantics": "same forecast and purchase rule; each executor carries its own realized SOC",
               "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "elapsed_seconds": perf_counter() - began, "verified": True}
    path.mkdir(parents=True)
    np.savez_compressed(path / "dispatch_2.npz", **arrays)
    pd.DataFrame(rows).to_csv(path / "daily.csv", index=False)
    for filename, obj in (("summary.json", summary), ("verification.json", verification), ("planning_audit.json", {"days": audits})):
        (path / filename).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--switching", type=float, nargs="+", default=[100., 300.])
    parser.add_argument("--methods", nargs="+", default=["markov", "independent3", "impulse"])
    args = parser.parse_args()
    for method in args.methods:
        for switching in args.switching:
            pilot(switching, args.days, method)
