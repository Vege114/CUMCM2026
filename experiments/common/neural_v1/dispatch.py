"""Linear planning, causal execution, and independent settlement."""

import time
from functools import lru_cache
from itertools import pairwise

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, hstack, vstack

from .data import EPOCH

ETA = 0.9
MIN_SOC, MAX_SOC, POWER_ENERGY = 1200.0, 10800.0, 5000 / 6


@lru_cache(maxsize=4)
def matrix(n):
    # Variables are grid, charge, discharge, surplus and interval-end storage.
    rows, cols, values = [], [], []
    for t in range(n):
        for block, value in ((0, 1), (1, -1), (2, 1), (3, -1)):
            rows.append(t); cols.append(block * n + t); values.append(value)
        for block, value in ((1, -ETA), (2, 1 / ETA), (4, 1)):
            rows.append(n + t); cols.append(block * n + t); values.append(value)
        if t:
            rows.append(n + t); cols.append(4 * n + t - 1); values.append(-1)
    return coo_matrix((values, (rows, cols)), shape=(2 * n, 5 * n)).tocsr()


def plan(load, pv, price, soc, original=None):
    n = len(load)
    p = np.maximum(price, 1e-8)
    rhs = np.r_[(load - pv) / 6, soc, np.zeros(n - 1)]
    a = matrix(n)
    bounds = ([(0, None)] * n + [(0, POWER_ENERGY)] * (2 * n)
              + [(0, None)] * n + [(MIN_SOC, MAX_SOC)] * n)
    objective = np.r_[p, np.full(2 * n, 1e-8), np.zeros(2 * n)]
    if original is not None:
        objective[:n] = 0  # Original plan payment is sunk and never refunded.
        objective = np.r_[objective, 1.5 * p, 0.5 * p]
        a = hstack([a, coo_matrix((2 * n, 2 * n))]).tocsr()
        ids = np.arange(n)
        adj = coo_matrix((np.r_[np.ones(n), -np.ones(n), np.ones(n)],
                          (np.tile(ids, 3), np.r_[ids, 5 * n + ids, 6 * n + ids])),
                         shape=(n, 7 * n)).tocsr()
        a = vstack([a, adj]).tocsr()
        rhs = np.r_[rhs, original]
        bounds += [(0, None)] * (2 * n)
    result = linprog(objective, A_eq=a, b_eq=rhs, bounds=bounds, method="highs")
    if not result.success:
        raise RuntimeError(f"Planning failed: {result.message}")
    g, c, d, surplus, _states = result.x[:5 * n].reshape(5, n)
    simultaneous = np.minimum(c, d / ETA**2)
    c = c - simultaneous
    d = d - simultaneous * ETA**2
    surplus = surplus + simultaneous * (1 - ETA**2)
    assert np.max(np.abs(g + d - c - surplus - (load - pv) / 6)) < 1e-5
    assert not np.any((c > 1e-6) & (d > 1e-6))
    return np.maximum(g, 0)


def settle(original, final, emergency, price):
    base = original * price
    up = np.maximum(final - original, 0) * 1.5 * price
    down = np.maximum(original - final, 0) * 0.5 * price
    urgent = emergency * price * 5
    return np.stack((base, up, down, urgent), axis=-1)


def execute(grid, load, pv, initial):
    n = len(grid)
    c, d, e, surplus = (np.zeros(n) for _ in range(4))
    s = np.empty(n + 1); s[0] = initial
    for t in range(n):
        balance = grid[t] + (pv[t] - load[t]) / 6
        if balance >= 0:
            c[t] = min(balance, POWER_ENERGY, max(0, (MAX_SOC - s[t]) / ETA))
            surplus[t] = balance - c[t]
        else:
            d[t] = min(-balance, POWER_ENERGY, max(0, (s[t] - MIN_SOC) * ETA))
            e[t] = -balance - d[t]
        s[t + 1] = s[t] + ETA * c[t] - d[t] / ETA
    return c, d, e, surplus, s


def day_run(data, day, provider, scenario, initial=6000, update_hours=(6, 12, 18),
            known_price=False, corrected=True):
    origin = day * 144
    actual = data.actual[origin:origin + 144]
    variable = scenario.startswith("4")
    adjustable = scenario in ("3", "4-3")
    price = actual[:, 2] if variable else data.fixed_price
    forecast = provider(origin)
    pv_channel = 3 if adjustable and corrected else 1
    if adjustable and not corrected:
        forecast = forecast.copy(); forecast[:, 1] = data.issued_pv(origin)
    optimization_price = (price if known_price or not variable else forecast[:, 2])
    t0 = time.monotonic()
    original = plan(forecast[:, 0], forecast[:, pv_channel], optimization_price, initial)
    final = original.copy()
    charge, discharge, emergency, surplus = (np.zeros(144) for _ in range(4))
    states = np.empty(145); states[0] = initial
    adjustments = []
    boundaries = [0] + ([h * 6 for h in update_hours] if adjustable else []) + [144]
    for start, stop in pairwise(boundaries):
        if start:
            f = provider(origin + start)
            if not corrected:
                f = f.copy(); f[:, 1] = data.issued_pv(origin + start)
            remainder = 144 - start
            cost = price[start:] if known_price or not variable else f[:remainder, 2]
            final[start:] = plan(f[:remainder, 0], f[:remainder, pv_channel], cost,
                                 states[start], original[start:])
            adjustments.append({"hour": start // 6,
                                "remaining_grid_kwh": float(final[start:].sum())})
        c, d, e, w, s = execute(final[start:stop], actual[start:stop, 0],
                                 actual[start:stop, 1], states[start])
        charge[start:stop], discharge[start:stop] = c, d
        emergency[start:stop], surplus[start:stop] = e, w
        states[start:stop + 1] = s
    fees = settle(original, final, emergency, price)
    balance = final + actual[:, 1] / 6 + discharge + emergency - actual[:, 0] / 6 - charge - surplus
    transition = np.diff(states) - ETA * charge + discharge / ETA
    violations = int(np.sum((states < MIN_SOC - 1e-6) | (states > MAX_SOC + 1e-6)))
    violations += int(np.sum((charge > POWER_ENERGY + 1e-6) | (discharge > POWER_ENERGY + 1e-6)))
    violations += int(np.sum((charge > 1e-6) & (discharge > 1e-6)))
    violations += int(np.sum(np.abs(balance) > 1e-6) + np.sum(np.abs(transition) > 1e-6))
    if violations:
        raise AssertionError(f"Physical constraint failure on day {day}: {violations}")
    summary = {"day": day, "date": str((EPOCH + __import__('pandas').Timedelta(days=day)).date()),
               "month": int((EPOCH + __import__('pandas').Timedelta(days=day)).month),
               "scenario": scenario, "planned_kwh": float(original.sum()),
               "final_kwh": float(final.sum()), "emergency_kwh": float(emergency.sum()),
               "emergency_minutes": int(np.sum(emergency > 1e-6) * 10),
               "charge_kwh": float(charge.sum()), "discharge_kwh": float(discharge.sum()),
               "surplus_kwh": float(surplus.sum()), "initial_soc": float(states[0]),
               "final_soc": float(states[-1]), "planned_cost": float(fees[:, 0].sum()),
               "up_cost": float(fees[:, 1].sum()), "down_cost": float(fees[:, 2].sum()),
               "emergency_cost": float(fees[:, 3].sum()), "total_cost": float(fees.sum()),
               "violations": violations, "max_balance_error": float(np.abs(balance).max()),
               "max_state_error": float(np.abs(transition).max()),
               "solve_execute_seconds": time.monotonic() - t0,
               "updates": len(adjustments)}
    detail = {"original": original, "final": final, "charge": charge, "discharge": discharge,
              "emergency": emergency, "surplus": surplus, "states": states, "fees": fees,
              "price": price, "actual": actual}
    return summary, detail
