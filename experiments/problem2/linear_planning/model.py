"""Two sequential LP solves: cost, then battery power smoothness.

No training, integer variables, complementarity constraints, or third solve.
Public forecasts/observations are load/PV in kW; dispatch arrays are kWh.
The relaxed planned battery path is diagnostic; only grid is committed.
Real execution rejects simultaneous charging/discharging instead of hiding it.
"""

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, vstack


@dataclass(frozen=True)
class Config:
    dt: float = 1 / 6
    eta_rt: float = 0.9
    soc_min: float = 1200.0
    soc_max: float = 10800.0
    power_max: float = 5000.0
    ramp_kw: float = 1000.0
    cost_relaxation: float = 0.001
    cost_tolerance: float = 1e-4
    throughput_weight: float = 1e-3
    time_limit: float = 30.0
    audit_tolerance: float = 1e-6

    def __post_init__(self):
        if not all(np.isfinite(v) for v in vars(self).values()):
            raise ValueError("all configuration values must be finite")
        if not (self.dt > 0 and 0 < self.eta_rt <= 1
                and 0 <= self.soc_min < self.soc_max and self.power_max > 0
                and self.ramp_kw >= 0 and self.cost_relaxation >= 0
                and self.cost_tolerance >= 0 and self.throughput_weight >= 0
                and self.time_limit > 0 and self.audit_tolerance > 0):
            raise ValueError("invalid configuration")


@dataclass(frozen=True)
class State:
    soc: float = 6000.0
    previous_power_kw: float = 0.0  # AC net charging power, signed


class SolveError(RuntimeError):
    """No certified optimum: do not silently use an incumbent as C*."""


class NonphysicalSolution(RuntimeError):
    """Relaxation produced an overlapping action that cannot be executed."""


def _array(value, shape, name, nonnegative=True):
    value = np.asarray(value, dtype=float)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f"{name}: expected finite array {shape}")
    if nonnegative and (value < 0).any():
        raise ValueError(f"{name}: negative values are not allowed")
    return value.copy()


def _state(state, cfg):
    if not (np.isfinite(state.soc) and np.isfinite(state.previous_power_kw)
            and cfg.soc_min - cfg.audit_tolerance <= state.soc <= cfg.soc_max + cfg.audit_tolerance
            and abs(state.previous_power_kw) <= cfg.power_max + cfg.audit_tolerance):
        raise ValueError("initial SOC/power outside physical bounds")


class _Rows:
    def __init__(self, size):
        self.size = size
        self.i, self.j, self.v, self.rhs = [], [], [], []

    def add(self, terms, rhs):
        row = len(self.rhs)
        for column, value in terms:
            self.i.append(row)
            self.j.append(column)
            self.v.append(value)
        self.rhs.append(rhs)

    def matrix(self):
        return coo_matrix((self.v, (self.i, self.j)),
                          shape=(len(self.rhs), self.size)).tocsr()


def _solve(forecast, price, state, cfg, fixed_grid=None):
    n = len(price)
    if n < 1:
        raise ValueError("empty horizon")
    price = _array(price, (n,), "price")
    forecast = _array(forecast, (n, 2), "forecast")
    _state(state, cfg)
    eta = np.sqrt(cfg.eta_rt)
    g, c, d, e, w, energy, tv = np.arange(7 * n).reshape(7, n)
    size = 7 * n
    bounds = np.tile([0., np.inf], (size, 1))
    bounds[c, 1] = bounds[d, 1] = cfg.power_max * cfg.dt
    bounds[energy, 0], bounds[energy, 1] = cfg.soc_min, cfg.soc_max
    cost = np.zeros(size)
    if fixed_grid is None:
        bounds[e, 1] = 0  # point-forecast plan meets its predicted demand
        cost[g] = price
    else:
        fixed_grid = _array(fixed_grid, (n,), "fixed grid")
        bounds[g, 0] = bounds[g, 1] = fixed_grid
        cost[e] = 5 * price  # committed grid cost is sunk at execution time

    eq, ub = _Rows(size), _Rows(size)
    net = (forecast[:, 0] - forecast[:, 1]) * cfg.dt
    for t in range(n):
        eq.add([(g[t], 1), (d[t], 1), (e[t], 1),
                (c[t], -1), (w[t], -1)], net[t])
        terms = [(energy[t], 1), (c[t], -eta), (d[t], 1 / eta)]
        if t:
            terms.append((energy[t - 1], -1))
        eq.add(terms, state.soc if t == 0 else 0)
        difference = [(c[t], 1 / cfg.dt), (d[t], -1 / cfg.dt)]
        prior = state.previous_power_kw if t == 0 else 0.
        if t:
            difference += [(c[t - 1], -1 / cfg.dt), (d[t - 1], 1 / cfg.dt)]
        negative = [(i, -v) for i, v in difference]
        ub.add(difference, cfg.ramp_kw + prior)
        ub.add(negative, cfg.ramp_kw - prior)
        ub.add(difference + [(tv[t], -1)], prior)
        ub.add(negative + [(tv[t], -1)], -prior)

    ae, au = eq.matrix(), ub.matrix()
    be, bu = np.asarray(eq.rhs), np.asarray(ub.rhs)
    logs = []

    def solve(objective, a, b, name):
        began = perf_counter()
        res = linprog(objective, A_ub=a, b_ub=b, A_eq=ae, b_eq=be,
                      bounds=bounds, method="highs",
                      options={"time_limit": cfg.time_limit,
                               "primal_feasibility_tolerance": 1e-8,
                               "dual_feasibility_tolerance": 1e-8})
        record = {"stage": name, "status": int(res.status),
                  "message": res.message, "seconds": perf_counter() - began,
                  "objective": float(res.fun) if res.fun is not None else None,
                  "variables": size, "integer_variables": 0}
        if not res.success:
            raise SolveError(str(record))
        residual = max(np.max(np.abs(ae @ res.x - be)),
                       np.max(np.maximum(a @ res.x - b, 0)),
                       np.max(np.maximum(bounds[:, 0] - res.x, 0)),
                       np.max(np.maximum(res.x - bounds[:, 1], 0)))
        record["max_constraint_residual"] = float(residual)
        if residual > cfg.audit_tolerance:
            raise SolveError(f"LP residual too large: {record}")
        logs.append(record)
        return res.x

    first = solve(cost, au, bu, "cost")
    optimum = float(cost @ first)
    budget = (1 + cfg.cost_relaxation) * optimum + cfg.cost_tolerance
    # One second-layer objective, not a hidden third-stage solve. Throughput
    # regularization discourages cost-neutral cycles; it is NOT a proof of exclusivity.
    quality = np.zeros(size)
    quality[tv] = 1 / (2 * cfg.power_max * n)
    quality[c] = quality[d] = cfg.throughput_weight / (2 * cfg.power_max * cfg.dt * n)
    second = solve(quality, vstack([au, coo_matrix(cost.reshape(1, -1))]).tocsr(),
                   np.append(bu, budget), "smoothness")

    def trajectory(x):
        return {"grid": x[g].copy(), "charge": x[c].copy(),
                "discharge": x[d].copy(), "emergency": x[e].copy(),
                "surplus": x[w].copy(), "states": np.r_[state.soc, x[energy]],
                "net_power_kw": (x[c] - x[d]) / cfg.dt}

    result = trajectory(second)
    result["grid"].setflags(write=False)
    overlap = np.minimum(result["charge"], result["discharge"])
    result["metadata"] = {
        "stages": logs, "stage_count": 2, "first_cost": optimum,
        "final_cost": float(cost @ second), "cost_budget": budget,
        "first_total_variation_kw": float(np.abs(np.diff(np.r_[
            state.previous_power_kw, (first[c] - first[d]) / cfg.dt])).sum()),
        "total_variation_kw": float(np.abs(np.diff(np.r_[
            state.previous_power_kw, result["net_power_kw"]])).sum()),
        "max_overlap_kwh": float(overlap.max()),
        "overlap_intervals": int((overlap > cfg.audit_tolerance).sum()),
        "quality_objective": float(quality @ second),
        "cost_scope": "predicted_planned_cost" if fixed_grid is None else "remaining_emergency_cost",
    }
    return result


def plan_day(forecast, price, state=State(), cfg=Config()):
    """Uses only externally supplied forecasts, known tariffs and prior state.

    Accepts any nonempty horizon for small-instance verification; production
    callers supply 144 rows. No terminal equality and no artificial daily reset.
    """
    return _solve(forecast, price, state, cfg)


def settle(grid, emergency, price):
    """Independent Q2-only invoice, columns: planned cost, emergency cost."""
    n = len(price)
    p = _array(price, (n,), "price")
    g = _array(grid, (n,), "grid")
    e = _array(emergency, (n,), "emergency")
    return np.column_stack((p * g, 5 * p * e))


def replay_day(forecast, price, observe, state=State(), cfg=Config()):
    """Freeze midnight grid, then request ONLY the current observation.

    observe(t) -> [actual load kW, actual PV kW]. Future estimates remain the
    supplied midnight forecast. Caller must not return future data from observe.
    Returns (midnight_plan, actual_detail, end_state). Execution solves two LPs
    over each shrinking horizon; only the current action is implemented.
    """
    forecast = _array(forecast, (len(price), 2), "forecast")
    price = _array(price, (len(price),), "price")
    plan = plan_day(forecast, price, state, cfg)
    n = len(price)
    detail = {k: np.zeros(n) for k in ("charge", "discharge", "emergency", "surplus")}
    detail["actual"] = np.zeros((n, 2))
    detail["states"] = np.empty(n + 1)
    detail["states"][0] = state.soc
    detail["grid"] = plan["grid"].copy()
    detail["execution_logs"] = []
    current = state
    for t in range(n):
        observation = _array(observe(t), (2,), "current observation")
        estimate = forecast[t:].copy()
        estimate[0] = observation
        action = _solve(estimate, price[t:], current, cfg, plan["grid"][t:])
        overlap = min(action["charge"][0], action["discharge"][0])
        if overlap > cfg.audit_tolerance:
            raise NonphysicalSolution(
                f"interval {t}: overlapping charge/discharge {overlap:.9g} kWh; "
                "no trajectory or final invoice is certified; inspect relaxation")
        for k in ("charge", "discharge", "emergency", "surplus"):
            detail[k][t] = max(0., float(action[k][0]))
        detail["actual"][t] = observation
        next_soc = current.soc + np.sqrt(cfg.eta_rt) * detail["charge"][t] - detail["discharge"][t] / np.sqrt(cfg.eta_rt)
        # Keep exact physical recurrence; tolerate only LP roundoff at endpoints.
        if abs(next_soc - cfg.soc_min) < 1e-8:
            next_soc = cfg.soc_min
        if abs(next_soc - cfg.soc_max) < 1e-8:
            next_soc = cfg.soc_max
        current = State(next_soc, (detail["charge"][t] - detail["discharge"][t]) / cfg.dt)
        detail["states"][t + 1] = current.soc
        detail["execution_logs"].append(action["metadata"])
    detail["fees"] = settle(detail["grid"], detail["emergency"], price)
    detail["audit"] = audit(detail, state, cfg)
    return plan, detail, current


def audit(detail, initial, cfg=Config()):
    """Recompute constraints from actual actions, independently of LP rows."""
    c, d, e, w = (np.asarray(detail[k]) for k in ("charge", "discharge", "emergency", "surplus"))
    g, soc, actual = detail["grid"], detail["states"], detail["actual"]
    power = (c - d) / cfg.dt
    step = np.abs(np.diff(np.r_[initial.previous_power_kw, power]))
    balance = g + (actual[:, 1] - actual[:, 0]) * cfg.dt + d + e - c - w
    transition = np.diff(soc) - np.sqrt(cfg.eta_rt) * c + d / np.sqrt(cfg.eta_rt)
    metrics = {"balance_residual_kwh": float(np.abs(balance).max()),
               "state_residual_kwh": float(np.abs(transition).max()),
               "initial_state_residual_kwh": float(abs(soc[0] - initial.soc)),
               "max_overlap_kwh": float(np.minimum(c, d).max()),
               "max_power_step_kw": float(step.max()),
               "total_variation_kw": float(step.sum())}
    errors = [metrics[k] for k in ("balance_residual_kwh", "state_residual_kwh", "initial_state_residual_kwh", "max_overlap_kwh")]
    errors += [float(cfg.soc_min - soc.min()), float(soc.max() - cfg.soc_max),
               float(max(c.max(), d.max()) / cfg.dt - cfg.power_max),
               float(step.max() - cfg.ramp_kw), float(-min(g.min(), c.min(), d.min(), e.min(), w.min()))]
    if not np.isfinite(errors).all() or max(errors) > cfg.audit_tolerance:
        raise NonphysicalSolution(f"actual trajectory failed audit: {metrics}; errors={errors}")
    return metrics
