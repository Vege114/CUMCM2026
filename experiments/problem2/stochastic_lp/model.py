"""Exactly two LP objectives on a supplied information tree; no integer logic.

The caller supplies only known forecasts/scenarios to midnight planning.
All energies are AC-side kWh except internal SOC. Node probabilities are
unconditional and sum to one at each time. A node has one shared action.
"""

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, vstack

from experiments.problem2.linear_planning.model import SolveError, State, _Rows


@dataclass(frozen=True)
class Config:
    dt: float = 1 / 6
    eta_rt: float = 0.9
    soc_min: float = 1200.0
    soc_max: float = 10800.0
    power_max: float = 5000.0
    ramp_kw: float = 1000.0
    delta: float = 0.001
    delta_exec: float = 0.001
    epsilon_cost: float = 1e-4
    beta: float = 0.0
    time_limit: float = 60.0
    tolerance: float = 1e-6

    def __post_init__(self):
        if not all(np.isfinite(v) for v in vars(self).values()):
            raise ValueError("configuration values must be finite")
        if not (self.dt > 0 and 0 < self.eta_rt <= 1 and 0 <= self.soc_min < self.soc_max
                and self.power_max > 0 and self.ramp_kw >= 0 and self.delta >= 0
                and self.delta_exec >= 0 and self.epsilon_cost >= 0 and self.beta >= 0
                and self.time_limit > 0 and self.tolerance > 0):
            raise ValueError("invalid configuration")


@dataclass
class Tree:
    time: np.ndarray
    parent: np.ndarray
    probability: np.ndarray
    supply_kw: np.ndarray
    horizon: int

    def validate(self):
        m = len(self.time)
        if (self.parent.shape != (m,) or self.probability.shape != (m,)
                or self.supply_kw.shape != (m, 2) or m == 0):
            raise ValueError("invalid tree shape")
        if not np.isfinite(self.supply_kw).all() or (self.supply_kw < 0).any():
            raise ValueError("invalid representative supply")
        if not np.isfinite(self.probability).all() or (self.probability <= 0).any():
            raise ValueError("invalid probability")
        if self.time.dtype.kind not in "iu" or self.parent.dtype.kind not in "iu":
            raise ValueError("integer time and parent required")
        if (self.time < 0).any() or (self.time >= self.horizon).any():
            raise ValueError("invalid node time")
        for t in range(self.horizon):
            if not np.isclose(self.probability[self.time == t].sum(), 1, atol=1e-12, rtol=0):
                raise ValueError("probabilities must sum to one at every time")
        for n, (t, a) in enumerate(zip(self.time, self.parent)):
            if t == 0:
                if a != -1:
                    raise ValueError("first period parent must be root")
            elif a < 0 or a >= n or self.time[a] != t - 1:
                raise ValueError("invalid unique parent")
            if t < self.horizon - 1 and not np.isclose(
                    self.probability[self.parent == n].sum(), self.probability[n], atol=1e-12, rtol=0):
                raise ValueError("child probability conservation failed")

    @classmethod
    def deterministic(cls, forecast):
        values = np.asarray(forecast, dtype=float).copy()
        h = len(values)
        return cls(np.arange(h), np.arange(h) - 1, np.ones(h), values, h)


DEFAULT_STATE = State()
DEFAULT_CONFIG = Config()


def solve_tree(tree, price, state=DEFAULT_STATE, cfg=DEFAULT_CONFIG, fixed_grid=None,
               battery_enabled=True, forbid_midnight_emergency=False):
    """Eqs. 16–25 / 27–30, D5-A emergency source, D2-A beta defaults to zero.

Returned node trajectories are continuous-relaxation diagnostics, never
implicitly certified physical actions. No overlap is silently removed.
"""
    tree.validate()
    h, m = tree.horizon, len(tree.time)
    price = np.asarray(price, dtype=float)
    if price.shape != (h,) or not np.isfinite(price).all() or (price <= 0).any():
        raise ValueError("positive price for each horizon interval required")
    if not (cfg.soc_min - cfg.tolerance <= state.soc <= cfg.soc_max + cfg.tolerance
            and abs(state.previous_power_kw) <= cfg.power_max + cfg.tolerance):
        raise ValueError("invalid initial state")
    size = h + 6 * m
    g = np.arange(h)
    c, d, e, w, energy, xi = h + np.arange(6 * m).reshape(6, m)
    bounds = np.tile([0., np.inf], (size, 1))
    bounds[c, 1] = bounds[d, 1] = cfg.power_max * cfg.dt if battery_enabled else 0
    bounds[energy] = [cfg.soc_min, cfg.soc_max]
    cost = np.zeros(size)
    cost[e] = 5 * price[tree.time] * tree.probability
    if fixed_grid is None:
        cost[g] = price
        if forbid_midnight_emergency:
            bounds[e, 1] = 0
    else:
        fixed_grid = np.asarray(fixed_grid, dtype=float)
        if (fixed_grid.shape != (h,) or not np.isfinite(fixed_grid).all()
                or fixed_grid.min() < -cfg.tolerance):
            raise ValueError("invalid locked ordinary purchase")
        bounds[g, 0] = bounds[g, 1] = np.maximum(fixed_grid, 0)

    eta = np.sqrt(cfg.eta_rt)
    net = (tree.supply_kw[:, 0] - tree.supply_kw[:, 1]) * cfg.dt
    eq, ub = _Rows(size), _Rows(size)
    for n, (t, a) in enumerate(zip(tree.time, tree.parent)):
        eq.add([(g[t], 1), (d[n], 1), (e[n], 1), (c[n], -1), (w[n], -1)], net[n])
        soc_terms = [(energy[n], 1), (c[n], -eta), (d[n], 1 / eta)]
        difference = [(c[n], 1 / cfg.dt), (d[n], -1 / cfg.dt)]
        if a >= 0:
            soc_terms.append((energy[a], -1))
            difference += [(c[a], -1 / cfg.dt), (d[a], 1 / cfg.dt)]
        eq.add(soc_terms, state.soc if a == -1 else 0)
        prior = state.previous_power_kw if a == -1 else 0
        negative = [(i, -v) for i, v in difference]
        ub.add(difference, cfg.ramp_kw + prior)
        ub.add(negative, cfg.ramp_kw - prior)
        ub.add(difference + [(xi[n], -1)], prior)
        ub.add(negative + [(xi[n], -1)], -prior)
    ae, au = eq.matrix(), ub.matrix()
    be, bu = np.asarray(eq.rhs), np.asarray(ub.rhs)
    logs = []

    def solve(objective, a, b, stage):
        began = perf_counter()
        res = linprog(objective, A_ub=a, b_ub=b, A_eq=ae, b_eq=be,
                      bounds=bounds, method="highs", options={
                          "time_limit": cfg.time_limit,
                          "primal_feasibility_tolerance": 1e-8,
                          "dual_feasibility_tolerance": 1e-8})
        log = {"stage": stage, "status": int(res.status), "message": res.message,
               "seconds": perf_counter() - began, "variables": size,
               "nodes": m, "integer_variables": 0,
               "objective": float(res.fun) if res.fun is not None else None}
        if not res.success:
            raise SolveError(str(log))
        residual = max(float(np.abs(ae @ res.x - be).max()),
                       float(np.maximum(a @ res.x - b, 0).max()),
                       float(np.maximum(bounds[:, 0] - res.x, 0).max()),
                       float(np.maximum(res.x - bounds[:, 1], 0).max()))
        log["max_constraint_residual"] = residual
        log["max_overlap_kwh"] = float(np.minimum(res.x[c], res.x[d]).max())
        if residual > cfg.tolerance:
            raise SolveError(f"Residual exceeds tolerance: {log}")
        logs.append(log)
        return res.x

    first = solve(cost, au, bu, "cost")
    optimum = float(np.sum(cost * first))
    delta = cfg.delta if fixed_grid is None else cfg.delta_exec
    budget = (1 + delta) * optimum + cfg.epsilon_cost
    quality = np.zeros(size)
    quality[xi] = tree.probability / (2 * cfg.power_max * h)
    quality[c] = quality[d] = (cfg.beta * tree.probability
                              / (2 * cfg.power_max * cfg.dt * h))
    second = solve(quality, vstack([au, coo_matrix(cost[None, :])]).tocsr(),
                   np.r_[bu, budget], "smoothness")
    power = (second[c] - second[d]) / cfg.dt
    previous = np.asarray([state.previous_power_kw if a < 0 else power[a]
                           for a in tree.parent])
    overlap = np.minimum(second[c], second[d])
    result = {"grid": second[g].copy(), "charge": second[c], "discharge": second[d],
              "emergency": second[e], "surplus": second[w], "end_soc": second[energy],
              "net_power_kw": power}
    # Preserve both solved layers for diagnosis without another optimization.
    result.update(first_charge=first[c], first_discharge=first[d],
                  first_emergency=first[e], first_surplus=first[w],
                  first_end_soc=first[energy], first_net_power_kw=(first[c] - first[d]) / cfg.dt)
    result["grid"].setflags(write=False)
    result["metadata"] = {
        "stages": logs, "stage_count": 2, "first_cost": optimum,
        "second_cost": float(np.sum(cost * second)), "budget": budget,
        "expected_tv_kw": float(np.sum(tree.probability * np.abs(power - previous))),
        "quality_objective": float(np.sum(quality * second)),
        "max_overlap_kwh": float(overlap.max()),
        "overlap_nodes": int((overlap > cfg.tolerance).sum()),
        "certified_no_overlap": bool((overlap <= cfg.tolerance).all()),
        "cost_scope": "tree_expected_total" if fixed_grid is None else "remaining_expected_emergency",
    }
    first_power = result["first_net_power_kw"]
    first_previous = np.array([state.previous_power_kw if a < 0 else first_power[a] for a in tree.parent])
    result["metadata"]["first_expected_tv_kw"] = float(np.sum(tree.probability * np.abs(first_power - first_previous)))
    return result


def settle(grid, emergency, price):
    """Only actual locked purchases and actual emergency energy are billable."""
    return np.column_stack((np.asarray(grid) * price, 5 * np.asarray(emergency) * price))
