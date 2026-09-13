"""Discrete SOC planning surrogate and causal, physically projected execution.

The tree supplies marginal net-demand supports, not joint scenarios. The DP
exactly optimizes a discrete open-loop surrogate; the physical controller can
curtail actions, so this is not an exact stochastic recourse optimization.
Units: AC-side kWh, battery-state kWh, yuan/kWh, and yuan.
"""

import time
from dataclasses import dataclass

import numpy as np

ETA = float(np.sqrt(0.9))
MIN_SOC, MAX_SOC, POWER_ENERGY = 1200.0, 10800.0, 5000 / 6
MODES = np.array([-1, 0, 1])


@dataclass(frozen=True)
class Config:
    grid_kwh: float = 100.0
    throughput_yuan_per_kwh: float = 0.002
    reversal_yuan: float = 0.05
    deadband_kwh: float = 2.0
    terminal_value: bool = True


DEFAULT_CONFIG = Config()


def stage_cost(net_support, shifts, price):
    """Exact newsvendor minimizer for equally weighted, discrete supports.

    Conditional on a fixed battery action, q = max(0, empirical Q_0.8).
    This statement is about the open-loop surrogate, not realized recourse.
    """
    support = np.asarray(net_support, dtype=float)
    shifts = np.asarray(shifts, dtype=float)
    quantile = np.quantile(support, 0.8, axis=-1, method="inverted_cdf")
    purchase = np.maximum(0, quantile[:, None] + shifts[None, :])
    shortage = np.maximum(
        support[:, None, :] + shifts[None, :, None] - purchase[:, :, None], 0
    ).mean(axis=-1)
    return purchase, np.asarray(price)[:, None] * (purchase + 5 * shortage)


def plan_day(net_support, price, initial_soc, config=DEFAULT_CONFIG, initial_mode=0,
             final_evaluation_day=False):
    began = time.perf_counter()
    support, price = np.asarray(net_support, float), np.asarray(price, float)
    if (support.ndim != 2 or support.shape[0] != len(price)
            or support.shape[1] < 1 or not np.isfinite(support).all()
            or not np.isfinite(price).all() or np.any(price <= 0)):
        raise ValueError("Expected finite net supports and positive matching prices")
    if not MIN_SOC <= initial_soc <= MAX_SOC or initial_mode not in MODES:
        raise ValueError("Invalid initial battery state or direction")
    if (config.grid_kwh <= 0 or config.grid_kwh > 600
            or config.throughput_yuan_per_kwh < 0 or config.reversal_yuan < 0
            or config.deadband_kwh < 0):
        raise ValueError("Invalid grid or secondary penalty")
    n_steps = len(price)
    grid = np.unique(np.r_[np.arange(MIN_SOC, MAX_SOC, config.grid_kwh),
                           MAX_SOC, float(initial_soc)])
    # Keep the exact initial SOC instead of rounding or resetting stored energy.
    difference = grid[None, :] - grid[:, None]
    charge = np.maximum(difference, 0) / ETA
    discharge = np.maximum(-difference, 0) * ETA
    feasible = np.maximum(charge, discharge) <= POWER_ENERGY + 1e-8
    counts = feasible.sum(axis=1)
    width = int(counts.max())
    next_index = np.zeros((len(grid), width), dtype=int)
    valid = np.arange(width)[None, :] < counts[:, None]
    for state in range(len(grid)):
        eligible = np.flatnonzero(feasible[state])
        next_index[state, :len(eligible)] = eligible
    delta = grid[next_index] - grid[:, None]
    c, d = np.maximum(delta, 0) / ETA, np.maximum(-delta, 0) * ETA
    action_mode = np.sign(delta).astype(int)
    # Uniform-grid differences repeat. Evaluate each AC energy shift once.
    shifts, inverse = np.unique(np.round(c - d, 9), return_inverse=True)
    inverse = inverse.reshape(delta.shape)
    purchases, costs = stage_cost(support, shifts, price)
    throughput = config.throughput_yuan_per_kwh * (c + d)
    terminal_lambda = (float(price.min()) / ETA
                       if config.terminal_value and not final_evaluation_day else 0.0)
    value = np.tile(-terminal_lambda * (grid - MIN_SOC), (3, 1))
    choices = np.empty((n_steps, 3, len(grid)), dtype=np.int32)
    for slot in range(n_steps - 1, -1, -1):
        updated = np.empty_like(value)
        stage = costs[slot, inverse] + throughput
        for mode_index, previous_mode in enumerate(MODES):
            following_mode = np.where(action_mode == 0, previous_mode, action_mode) + 1
            future = value[following_mode, next_index]
            switching = config.reversal_yuan * (action_mode * previous_mode == -1)
            candidates = np.where(valid, stage + future + switching, np.inf)
            selected = candidates.argmin(axis=1)
            updated[mode_index] = candidates[np.arange(len(grid)), selected]
            choices[slot, mode_index] = selected
        value = updated
    state = int(np.flatnonzero(grid == initial_soc)[0])
    mode = int(initial_mode)
    objective = float(value[mode + 1, state])
    plan, intended_charge, intended_discharge = (np.zeros(n_steps) for _ in range(3))
    states = np.empty(n_steps + 1)
    states[0] = initial_soc
    for slot in range(n_steps):
        column = choices[slot, mode + 1, state]
        plan[slot] = purchases[slot, inverse[state, column]]
        intended_charge[slot], intended_discharge[slot] = c[state, column], d[state, column]
        action = action_mode[state, column]
        if action:
            mode = int(action)
        state = int(next_index[state, column])
        states[slot + 1] = grid[state]
    return {
        "purchase": plan, "charge": intended_charge, "discharge": intended_discharge,
        "states": states,
        "metadata": {"planning_seconds": time.perf_counter() - began,
                     "grid_states": len(grid), "max_actions_per_state": width,
                     "objective_surrogate_yuan": objective,
                     "terminal_value_yuan_per_soc_kwh": terminal_lambda,
                     "exact_for_discrete_surrogate_only": True},
    }


def execute_plan(plan, actual_kw, price, initial_soc, config=DEFAULT_CONFIG, initial_mode=0,
                 greedy=False):
    """Fixed purchases; observe current slot only; never emergency-charge.

    Positive actions are clipped by available excess electricity. Negative
    actions are clipped by the load deficit. Actual SOC and last non-idle
    direction carry forward, including across days when called in sequence.
    """
    purchase = np.asarray(plan["purchase"], dtype=float)
    actual, price = np.asarray(actual_kw, dtype=float), np.asarray(price, dtype=float)
    if (actual.shape != (len(purchase), 2) or not np.isfinite(actual).all()
            or np.any(actual < 0) or not MIN_SOC <= initial_soc <= MAX_SOC):
        raise ValueError("Invalid physical observations or starting SOC")
    n_steps = len(purchase)
    c, d, emergency, surplus = (np.zeros(n_steps) for _ in range(4))
    states = np.empty(n_steps + 1)
    states[0] = initial_soc
    previous_mode, reversals = initial_mode, 0
    for slot in range(n_steps):
        balance = purchase[slot] + (actual[slot, 1] - actual[slot, 0]) / 6
        if balance >= 0:
            request = POWER_ENERGY if greedy else plan["charge"][slot]
            c[slot] = min(request, balance, POWER_ENERGY,
                          max(0, (MAX_SOC - states[slot]) / ETA))
            if not greedy and c[slot] < config.deadband_kwh:
                c[slot] = 0
            surplus[slot] = balance - c[slot]
        else:
            request = POWER_ENERGY if greedy else plan["discharge"][slot]
            d[slot] = min(request, -balance, POWER_ENERGY,
                          max(0, (states[slot] - MIN_SOC) * ETA))
            if not greedy and d[slot] < config.deadband_kwh:
                d[slot] = 0
            emergency[slot] = -balance - d[slot]
        states[slot + 1] = states[slot] + ETA * c[slot] - d[slot] / ETA
        mode = int(np.sign(c[slot] - d[slot]))
        if mode:
            reversals += int(previous_mode * mode == -1)
            previous_mode = mode
    fees = np.column_stack((purchase * price, np.zeros((n_steps, 2)),
                            5 * emergency * price))
    detail = {"original": purchase.copy(), "final": purchase.copy(), "charge": c,
              "discharge": d, "emergency": emergency, "surplus": surplus,
              "states": states, "fees": fees, "actual": actual.copy(), "price": price.copy()}
    balance_error = purchase + (actual[:, 1] - actual[:, 0]) / 6 + d + emergency - c - surplus
    state_error = np.diff(states) - ETA * c + d / ETA
    simultaneous = int(((c > 1e-6) & (d > 1e-6)).sum())
    assert max(np.abs(balance_error).max(), np.abs(state_error).max()) < 1e-6
    assert MIN_SOC - 1e-6 <= states.min() <= states.max() <= MAX_SOC + 1e-6
    assert max(c.max(), d.max()) <= POWER_ENERGY + 1e-6
    assert simultaneous == 0 and not ((c > 1e-6) & (emergency > 1e-6)).any()
    metrics = {"total_cost": float(fees.sum()), "planned_cost": float(fees[:, 0].sum()),
               "emergency_cost": float(fees[:, 3].sum()), "simultaneous_slots": simultaneous,
               "throughput_kwh": float((c + d).sum()), "direction_reversals": reversals,
               "equivalent_full_cycles": float((ETA * c.sum() + d.sum() / ETA) / 24000),
               "initial_soc": float(states[0]), "final_soc": float(states[-1]),
               "final_mode": previous_mode,
               "max_balance_error": float(np.abs(balance_error).max()),
               "max_state_error": float(np.abs(state_error).max())}
    return detail, metrics
