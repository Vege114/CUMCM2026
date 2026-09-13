"""Causal storage DP with an exact budget of nonidle direction reversals.

The finite SOC grid and three-state disturbance model are approximations.
The hard switch budget is enforced in actual execution, including a first
action opposite to the preceding day's last nonidle direction. Idle does not
change direction or consume budget. Purchase is fixed before observations.
"""
from __future__ import annotations

from time import perf_counter

import numpy as np

from experiments.exp008.markov_tail_feedback import fit_error_model

ETA, LOW, HIGH, LIMIT = float(np.sqrt(.9)), 1200., 10800., 5000 / 6
MODES = (-1, 1)


def value_functions(purchase, model, price, *, grid_kwh=200., wear=.002,
                    terminal=.45, budget=8):
    """V[t,previous_error_bin,previous_nonidle_direction,budget,SOC].

The current error bin and its within-bin emission are observed before the
action. Future continuation knows only the observed bin, not an emission ID
or a future historical trajectory. The same budget transition is used at run
time, where observations can lie outside the finite emission support.
"""
    began = perf_counter()
    q, price = np.asarray(purchase, float), np.asarray(price, float)
    net = np.asarray(model['forecast_net_kwh'], float)
    support = net[:, None, None] + model['conditional_error_support_kwh']
    weights = np.asarray(model['conditional_emission_weights'], float)
    transition = np.asarray(model['transition'], float)
    n = len(q)
    if (q.shape != (n,) or price.shape != (n,) or net.shape != (n,)
            or support.shape[:2] != (n, 3) or weights.shape != support.shape
            or transition.shape != (n, 3, 3) or budget < 0 or int(budget) != budget
            or not 0 < grid_kwh <= HIGH - LOW or min(wear, terminal) < 0
            or np.any(q < 0) or np.any(price <= 0)):
        raise ValueError('Invalid trajectories, model, SOC grid or budget')
    for value in (q, price, support, weights, transition):
        if not np.isfinite(value).all():
            raise ValueError('Finite inputs required')
    assert np.all(weights >= 0) and np.all(transition >= 0)
    assert np.allclose(weights.sum(-1), 1.) and np.allclose(transition.sum(-1), 1.)
    grid = np.r_[np.arange(LOW, HIGH, grid_kwh), HIGH]
    values = np.zeros((n + 1, 3, 2, budget + 1, len(grid)))
    values[-1] = -terminal * (grid - LOW)
    delta = grid[None, :] - grid[:, None]
    for t in range(n - 1, -1, -1):
        observed = np.zeros((3, 2, budget + 1, len(grid)))
        for state in range(3):
            future = values[t + 1, state]
            for emission in range(support.shape[2]):
                b = q[t] - support[t, state, emission]
                direction = 1 if b >= 0 else -1
                di = int(direction > 0)
                if b >= 0:
                    available = np.minimum(min(b, LIMIT), (HIGH - grid) / ETA)
                    endpoint = grid + ETA * available
                    eligible = (delta > 1e-8) & (delta <= ETA * available[:, None] + 1e-8)
                    stage_grid = wear * delta / ETA
                    stage_endpoint = wear * available
                else:
                    available = np.minimum(min(-b, LIMIT), (grid - LOW) * ETA)
                    endpoint = grid - available / ETA
                    eligible = (delta < -1e-8) & (delta >= -available[:, None] / ETA - 1e-8)
                    stage_grid = 5 * price[t] * (-b + ETA * delta) - wear * ETA * delta
                    stage_endpoint = 5 * price[t] * (-b - available) + wear * available
                # Active[R, current_SOC] for an action whose resulting budget is R.
                grid_cost = future[di, :, None, :] + stage_grid[None, :, :]
                best_grid = np.min(np.where(eligible[None, :, :], grid_cost, np.inf), axis=2)
                endpoint_cost = np.stack([np.interp(endpoint, grid, row) for row in future[di]])
                endpoint_cost += stage_endpoint[None, :]
                active = np.minimum(best_grid, np.where(available[None, :] > 1e-8, endpoint_cost, np.inf))
                for mi, mode in enumerate(MODES):
                    idle = future[mi] + 5 * price[t] * max(-b, 0.)
                    if mode == direction:
                        allowed_active = active
                    else:
                        allowed_active = np.concatenate((np.full((1, len(grid)), np.inf), active[:-1]))
                    observed[state, mi] += weights[t, state, emission] * np.minimum(idle, allowed_active)
        values[t] = np.einsum('ij,jmrs->imrs', transition[t], observed, optimize=False)
    # With an optional additional switch, the same policy remains feasible.
    budget_monotonicity_error = float(np.max(np.diff(values, axis=3))) if budget else 0.
    if budget_monotonicity_error > 1e-7:
        raise AssertionError('A larger optional switch budget cannot raise optimal approximate value')
    return grid, values, dict(method='observed_error_conditional_remaining_switch_budget_DP',
        grid_kwh=grid_kwh, grid_points=len(grid), maximum_daily_switches=budget,
        modes=list(MODES), error_states=3, conditional_emission_points=support.shape[2],
        terminal=terminal, wear=wear, seconds=perf_counter() - began,
        maximum_budget_monotonicity_error=budget_monotonicity_error,
        values_axes=['before_observation_time', 'previous_error_bin', 'last_nonidle_direction', 'remaining_budget', 'SOC'],
        continuous_global_optimality_certificate=False)


def initial_value(grid, values, model, soc, mode=1, remaining=8):
    if mode not in MODES or not LOW <= soc <= HIGH or not 0 <= remaining < values.shape[3]:
        raise ValueError('Invalid initial physical state or budget')
    return float(np.interp(soc, grid, values[0, model['initial_error_state'], int(mode > 0), remaining]))


def execute_paths(purchase, net_paths, price, initial_soc, grid, values, model,
                  *, wear=.002, initial_mode=1, budget=8):
    """Vectorized independent paths; each action depends only on its own prefix.

    A path is not passed to any planner or looked up by scenario identity. This
    function observes one net-demand column at a time and applies the already
    fixed DP table to each path's physical state, nonidle mode and budget.
    """
    q, paths, price = np.asarray(purchase, float), np.asarray(net_paths, float), np.asarray(price, float)
    k, n = paths.shape
    if q.shape != (n,) or price.shape != (n,) or initial_mode not in MODES or values.shape[0] != n + 1:
        raise ValueError('Mismatched horizon or invalid mode')
    if not LOW <= initial_soc <= HIGH or budget < 0 or budget >= values.shape[3]:
        raise ValueError('Invalid SOC or budget')
    states = np.full((k, n + 1), initial_soc)
    modes = np.full((k, n + 1), initial_mode, dtype=np.int8)
    remaining = np.full((k, n + 1), budget, dtype=np.int8)
    error_bins = np.zeros((k, n), dtype=np.int8)
    c, d, e, w = [np.zeros((k, n)) for _ in range(4)]
    for t in range(n):
        standardized = (paths[:, t] - model['forecast_net_kwh'][t] - model['mean_kwh'][t]) / model['scale_kwh'][t]
        bins = np.searchsorted(model['state_edges'], standardized, side='right')
        error_bins[:, t] = bins
        for j in range(k):
            current, mode, left = states[j, t], int(modes[j, t]), int(remaining[j, t])
            b = q[t] - paths[j, t]
            direction = 1 if b >= 0 else -1
            switch = int(direction != mode)
            future = values[t + 1, bins[j]]
            idle = np.interp(current, grid, future[int(mode > 0), left]) + 5 * price[t] * max(-b, 0.)
            following = current
            if left >= switch:
                next_value = future[int(direction > 0), left - switch]
                if b >= 0:
                    available = min(b, LIMIT, max(0., (HIGH - current) / ETA))
                    endpoint = current + ETA * available
                    choices = np.r_[grid[(grid > current + 1e-8) & (grid <= endpoint + 1e-8)], endpoint]
                    costs = np.interp(choices, grid, next_value) + wear * (choices - current) / ETA
                else:
                    available = min(-b, LIMIT, max(0., (current - LOW) * ETA))
                    endpoint = current - available / ETA
                    choices = np.r_[grid[(grid >= endpoint - 1e-8) & (grid < current - 1e-8)], endpoint]
                    released = (current - choices) * ETA
                    costs = np.interp(choices, grid, next_value) + 5 * price[t] * (-b - released) + wear * released
                best = int(np.argmin(costs))
                if available > 1e-8 and costs[best] < idle - 1e-8:
                    following = float(choices[best])
            c[j, t], d[j, t] = max(0., (following - current) / ETA), max(0., (current - following) * ETA)
            active = c[j, t] + d[j, t] > 1e-8
            modes[j, t + 1] = direction if active else mode
            remaining[j, t + 1] = left - (switch if active else 0)
            e[j, t], w[j, t] = max(0., -b - d[j, t]), max(0., b - c[j, t])
            states[j, t + 1] = following
    fees = np.zeros((k, n, 4))
    fees[:, :, 0] = q[None, :] * price
    fees[:, :, 3] = 5 * e * price
    return dict(charge=c, discharge=d, emergency=e, surplus=w, states=states,
        modes=modes, remaining_budget=remaining, observed_error_bins=error_bins, fees=fees)


def historical_objective(flow, *, wear=.002, terminal=.45):
    return float(np.mean(np.sum(flow['fees'], axis=(1, 2))
        + wear * np.sum(flow['charge'] + flow['discharge'], axis=1)
        - terminal * (flow['states'][:, -1] - LOW)))


def checks():
    """Scalar accounting, hard midnight budget and prefix-causality checks."""
    q, price = np.array([0., 400., 0., 400., 0., 400.]), np.ones(6)
    net = np.full(6, 200.)
    errors = np.zeros((4, 6))
    model = fit_error_model(net, errors, points=3)
    grid, values, metadata = value_functions(q, model, price, budget=1, terminal=0., wear=0.)
    flow = execute_paths(q, net[None, :], price, 5000., grid, values, model,
        budget=1, initial_mode=1, wear=0.)
    assert np.sum(np.diff(flow['modes'], axis=1) != 0) <= 1
    assert np.all(flow['remaining_budget'] >= 0)
    changed = net[None, :].copy()
    changed[:, 3:] += [1000., -1000., 2000.]
    alternate = execute_paths(q, changed, price, 5000., grid, values, model,
        budget=1, initial_mode=1, wear=0.)
    for key in ('charge', 'discharge', 'emergency', 'surplus', 'observed_error_bins'):
        np.testing.assert_array_equal(flow[key][:, :3], alternate[key][:, :3])
    for key in ('states', 'modes', 'remaining_budget'):
        np.testing.assert_array_equal(flow[key][:, :4], alternate[key][:, :4])
    # With zero switches and last mode charge, deficits cannot be discharged.
    zero_grid, zero_values, _ = value_functions(q, model, price, budget=0, terminal=0., wear=0.)
    zero = execute_paths(q, net[None, :], price, 5000., zero_grid, zero_values, model,
        budget=0, initial_mode=1, wear=0.)
    assert np.all(zero['discharge'] == 0.) and np.all(zero['remaining_budget'] == 0)
    np.testing.assert_allclose(initial_value(zero_grid, zero_values, model, 5000., remaining=0),
        float(5 * np.maximum(net - q, 0.).sum()), atol=1e-6, rtol=0.)
    # One slot: midnight reversal immediately consumes its one allowed switch.
    single = fit_error_model(np.array([200.]), np.zeros((4, 1)), points=3)
    sg, sv, _ = value_functions(np.array([0.]), single, np.ones(1), budget=1, terminal=0., wear=0.)
    sf = execute_paths(np.array([0.]), np.array([[200.]]), np.ones(1), 5000., sg, sv, single,
        budget=1, initial_mode=1, wear=0.)
    assert sf['remaining_budget'][0, -1] == 0 and sf['modes'][0, -1] == -1
    np.testing.assert_allclose(sf['discharge'][0, 0], 200., atol=1e-8, rtol=0.)
    return dict(passed=True, zero_budget_forbids_reversal=True, midnight_reverse_consumes_budget=True,
        prefix_mutation_passed=True, budget_monotonicity_passed=True, diagnostic_metadata=metadata)


if __name__ == '__main__':
    import json
    print(json.dumps(checks(), indent=2))
