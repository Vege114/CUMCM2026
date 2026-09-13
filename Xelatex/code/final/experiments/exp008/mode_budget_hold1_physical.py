"""Ten-minute modes with a hard daily eight-switch budget and no switch penalty.

The common purchase and scenario storage/complementarity constraints match
the existing physical planner. Scenario recourse remains an optimistic
surrogate; only actual greedy execution is prefix-causal.
"""

from __future__ import annotations

import json

import numpy as np

from experiments.common.neural_v2.physics import LinearModel
from experiments.exp008.planner import ETA, HIGH, LIMIT, LOW


def mode_state(mask, previous_mode, previous_run_slots, minimum=1):
    mask = np.asarray(mask, bool)
    previous = previous_mode > 0
    flips = np.r_[mask[0] != previous, mask[1:] != mask[:-1]]
    history = np.r_[np.repeat(previous, min(previous_run_slots, minimum)), mask]
    changes = np.flatnonzero(history[1:] != history[:-1])+1
    lengths = np.diff(np.r_[0, changes, len(history)])
    if len(lengths) > 1 and np.any(lengths[:-1] < minimum):
        raise AssertionError('A completed planned mode run violates minimum hold')
    own_changes = np.flatnonzero(mask[1:] != mask[:-1])+1
    tail = len(mask)-int(own_changes[-1]) if len(own_changes) else len(mask)
    if not len(own_changes) and mask[0] == previous:
        tail += previous_run_slots
    return {'planned_changes_including_boundary': int(np.sum(flips)),
            'final_planned_mode': 1 if mask[-1] else -1,
            'final_planned_run_slots': tail,
            'next_day_required_hold_slots': max(0, minimum-tail),
            'completed_run_minimum_passed': True}, flips


def plan(net_paths, price, soc, previous_mode=1, previous_run_slots=1,
         *, hold_slots=1, switching=0., wear=.002, seconds=5., gap=.002, final=False):
    paths, price = np.asarray(net_paths, float), np.asarray(price, float)
    k, n = paths.shape
    if price.shape != (n,) or not np.all(price > 0) or not np.isfinite(paths).all():
        raise ValueError('finite scenario net kWh and matching positive price required')
    if not LOW <= soc <= HIGH or previous_mode not in (-1, 1) or previous_run_slots < 1 or hold_slots != 1:
        raise ValueError('valid SOC, explicit previous mode/run and fixed hold1 required')
    if switching != 0.:
        raise ValueError('This candidate fixes the soft switching penalty to zero')
    m = LinearModel()
    q_upper = np.maximum(paths.max(0), 0.)+LIMIT
    q = m.variables((n,), upper=q_upper, cost=price)
    mode = m.variables((n,), upper=1., integer=True)
    flip = m.variables((n,), upper=1., cost=switching, integer=True)
    m.constraint([(index, 1.) for index in flip], upper=8.)
    c = m.variables((k, n), upper=LIMIT, cost=wear/k)
    d = m.variables((k, n), upper=LIMIT, cost=wear/k)
    emergency_upper = np.maximum(paths, 0.)
    e = m.variables((k, n), upper=emergency_upper, cost=5*price[None, :]/k)
    w = m.variables((k, n))
    s = m.variables((k, n), lower=LOW, upper=HIGH)
    may_charge = m.variables((k, n), upper=1., integer=True)
    for idx in s[:, -1]:
        m.objective[int(idx)] = 0. if final else -float(price.min())/ETA/k
    old = float(previous_mode > 0)
    for t in range(n):
        difference = [(mode[t], 1.)]
        if t:
            difference.append((mode[t-1], -1.))
        rhs = old if t == 0 else 0.
        m.constraint(difference+[(flip[t], -1.)], upper=rhs)
        m.constraint([(i, -v) for i, v in difference]+[(flip[t], -1.)], upper=-rhs)
        # The upper inequalities make flip the exact XOR, independently of
        # objective tolerance and its participation in minimum-hold windows.
        first = [(flip[t], 1.), (mode[t], -1.)]
        second = [(flip[t], 1.), (mode[t], 1.)]
        if t:
            first.append((mode[t-1], -1.))
            second.append((mode[t-1], 1.))
        m.constraint(first, upper=old if t == 0 else 0.)
        m.constraint(second, upper=2.-old if t == 0 else 2.)
    for t in range(n):
        m.constraint([(flip[j], 1.) for j in range(t, min(n, t+hold_slots))], upper=1.)
    pending = min(n, max(0, hold_slots-previous_run_slots))
    for t in range(pending):
        m.constraint([(mode[t], 1.)], old, old)
    for j in range(k):
        for t in range(n):
            m.constraint([(q[t], 1), (c[j, t], -1), (d[j, t], 1),
                          (e[j, t], 1), (w[j, t], -1)], paths[j, t], paths[j, t])
            terms = [(s[j, t], 1), (c[j, t], -ETA), (d[j, t], 1/ETA)]
            if t:
                terms.append((s[j, t-1], -1.))
            rhs = soc if t == 0 else 0.
            m.constraint(terms, rhs, rhs)
            m.constraint([(c[j, t], 1), (mode[t], -LIMIT)], upper=0.)
            m.constraint([(d[j, t], 1), (mode[t], LIMIT)], upper=LIMIT)
            m.constraint([(c[j, t], 1), (may_charge[j, t], -LIMIT)], upper=0.)
            m.constraint([(e[j, t], 1), (may_charge[j, t], emergency_upper[j, t])],
                         upper=emergency_upper[j, t])
    x, metadata = m.solve(seconds=seconds, gap=gap)
    if x is None:
        raise RuntimeError(f'No feasible variable-mode hold1 incumbent: {json.dumps(metadata)}')
    mask = x[mode] > .5
    boundary, exact_flips = mode_state(mask, previous_mode, previous_run_slots, hold_slots)
    if boundary['planned_changes_including_boundary'] > 8:
        raise AssertionError('Daily planned switch budget exceeded')
    states = np.column_stack((np.full(k, soc), x[s]))
    balance = x[q][None, :]+x[d]+x[e]-x[c]-x[w]-paths
    state_error = np.diff(states, axis=1)-ETA*x[c]+x[d]/ETA
    check = {'maximum_balance_error_kwh': float(np.abs(balance).max()),
        'maximum_soc_equation_error_kwh': float(np.abs(state_error).max()),
        'emergency_charging_slots': int(np.sum((x[c] > 1e-6) & (x[e] > 1e-6))),
        'simultaneous_charge_discharge_slots': int(np.sum((x[c] > 1e-6) & (x[d] > 1e-6))),
        'minimum_soc_kwh': float(states.min()), 'maximum_soc_kwh': float(states.max()),
        'maximum_charge_or_discharge_kwh': float(max(x[c].max(), x[d].max())),
        'minimum_variable': float(x.min()),
        'mode_flip_XOR_error': float(np.max(np.abs(x[flip]-exact_flips)))}
    check['passed'] = (max(check['maximum_balance_error_kwh'], check['maximum_soc_equation_error_kwh'],
        check['mode_flip_XOR_error']) < 1e-6 and check['emergency_charging_slots'] == 0 and
        check['simultaneous_charge_discharge_slots'] == 0 and states.min() >= LOW-1e-6 and
        states.max() <= HIGH+1e-6 and check['maximum_charge_or_discharge_kwh'] <= LIMIT+1e-6 and
        check['minimum_variable'] >= -1e-6)
    if not check['passed']:
        raise AssertionError(check)
    metadata.update(method='variable_hold1_daily_cap8_kappa0_physical_scenario_MIP',
        daily_planned_switch_cap=8, switching_cost=0., exact_binary_XOR=True,
        scenario_count=k, hold_slots=hold_slots, previous_planned_mode=previous_mode,
        previous_planned_run_slots=previous_run_slots, inherited_locked_prefix_slots=pending,
        nonanticipative_recourse_certificate=False, scenario_physical_checks=check,
        planned_mode_changes=boundary['planned_changes_including_boundary'], **boundary)
    return {'purchase': np.maximum(x[q], 0.), 'allowed_charge': mask,
        'planned_flips': exact_flips, 'solved_flip_values': x[flip], 'scenario_charge': x[c], 'scenario_discharge': x[d],
        'scenario_emergency': x[e], 'scenario_surplus': x[w], 'scenario_states': states,
        'metadata': metadata}
