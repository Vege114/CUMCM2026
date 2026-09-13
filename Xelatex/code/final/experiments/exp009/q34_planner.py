"""Joint historical-path purchase planning; realized dispatch is verified separately.

The scenario recourse LP is an optimistic planning approximation: its battery
states are scenario dependent, not a certified nonanticipative scenario tree.
Only purchases are shared. No future realized observations enter this module.
"""
from dataclasses import dataclass
from time import perf_counter

import numpy as np

from experiments.common.neural_v2.physics import LinearModel

ETA = float(np.sqrt(.9))
LOW, HIGH, LIMIT = 1200., 10800., 5000 / 6


@dataclass(frozen=True)
class Settings:
    scenarios: int = 7
    residual_scale: float = 1.
    throughput_penalty: float = .005
    variation_penalty: float = .001
    terminal_value: float = 1.
    emergency_weight: float = 5.
    future_shortfall_weight: float | None = None


def plan(forecast, initial_soc, error_paths=None, original=None, settings=Settings(),
         final_day=False, charge_mask=None, next_update_slot=None,
         future_shortfall_weight=None):
    """Return shared purchases plus representative inventory/mode intentions.

    next_update_slot is relative to this remaining horizon: 36 means its first
    36 ten-minute intervals execute before the next legal information release.
    The optional future weight applies from that index onward, only when both
    a boundary and a weight are present. It is an opportunity-value proxy, not
    an alternative billing rule or a nonanticipative recourse certificate.
    """
    began = perf_counter()
    net = (np.asarray(forecast['load_kw']) - forecast['pv_kw']) / 6
    price = np.asarray(forecast['price'])
    n = len(net)
    shortfall_weights = np.broadcast_to(np.asarray(settings.emergency_weight, float), (n,)).copy()
    future_weight = (getattr(settings, 'future_shortfall_weight', None)
                     if future_shortfall_weight is None else future_shortfall_weight)
    if future_weight is not None and (not np.isfinite(future_weight) or future_weight <= 0):
        raise ValueError('future_shortfall_weight must be finite and positive')
    if next_update_slot is not None:
        if (not isinstance(next_update_slot, (int, np.integer))
                or not 1 <= next_update_slot <= n):
            raise ValueError('next_update_slot must be a relative horizon index in 1..n')
        if future_weight is not None:
            shortfall_weights[next_update_slot:] = future_weight
    paths = np.asarray(error_paths) if error_paths is not None else np.zeros((1, n))
    if not len(paths):
        paths = np.zeros((1, n))
    if paths.shape[1] == 144:
        paths = paths[:, -n:]
    if len(paths) > settings.scenarios:
        # Deterministic equally spaced complete historical days, retaining
        # within-day correlations. No sorting by future or realized cost.
        paths = paths[np.linspace(0, len(paths)-1, settings.scenarios).astype(int)]
    demands = net[None, :] + settings.residual_scale * paths
    k = len(demands)
    assert demands.shape == (k, n) and price.shape == (n,)
    assert np.isfinite(demands).all() and np.all(price > 0)
    assert LOW - 1e-6 <= initial_soc <= HIGH + 1e-6
    m = LinearModel()
    g = m.variables((n,), cost=price if original is None else 0., lower=0.)
    if original is not None:
        original = np.asarray(original, float)
        up = m.variables((n,), cost=1.5 * price)
        down = m.variables((n,), cost=-0.5 * price)
        for t in range(n):
            m.constraint([(g[t], 1), (up[t], -1), (down[t], 1)], original[t], original[t])
        # Since simultaneous up/down costs +p per kWh, the LP itself chooses
        # their positive parts. The fixed original bill is a dropped constant.
    charge_limit = LIMIT if charge_mask is None else LIMIT*np.asarray(charge_mask)[None, :]
    discharge_limit = LIMIT if charge_mask is None else LIMIT*(1-np.asarray(charge_mask))[None, :]
    c = m.variables((k, n), upper=charge_limit, cost=settings.throughput_penalty/k)
    d = m.variables((k, n), upper=discharge_limit, cost=settings.throughput_penalty/k)
    e = m.variables((k, n), cost=shortfall_weights*price[None, :]/k)
    w = m.variables((k, n))
    s = m.variables((k, n), lower=LOW, upper=HIGH)
    tv = m.variables((k, n), cost=settings.variation_penalty*6/k)
    terminal = 0 if final_day else settings.terminal_value*price.min()/ETA
    for index in s[:, -1]:
        m.objective[int(index)] = -terminal/k
    for j in range(k):
        for t in range(n):
            m.constraint([(g[t], 1), (d[j,t], 1), (e[j,t], 1),
                          (c[j,t], -1), (w[j,t], -1)], demands[j,t], demands[j,t])
            transition = [(s[j,t], 1), (c[j,t], -ETA), (d[j,t], 1/ETA)]
            rhs = initial_soc if t == 0 else 0
            if t:
                transition.append((s[j,t-1], -1))
            m.constraint(transition, rhs, rhs)
            if t:
                diff = [(c[j,t],1),(d[j,t],-1),(c[j,t-1],-1),(d[j,t-1],1)]
                m.constraint(diff + [(tv[j,t],-1)], upper=0)
                m.constraint([(ids,-v) for ids,v in diff] + [(tv[j,t],-1)], upper=0)
    x, meta = m.solve(seconds=20, gap=1e-7)
    if x is None:
        raise RuntimeError(f'Joint-path LP failed: {meta}')
    return {'purchase': np.maximum(x[g], 0), 'charge': np.maximum(x[c].mean(0), 0),
            'discharge': np.maximum(x[d].mean(0), 0),
            'states': np.r_[initial_soc, x[s].mean(0)],
            'metadata': {**meta, 'planning_seconds': perf_counter()-began,
                         'method': 'joint_path_lp_surrogate', 'scenarios': k,
                         'nonanticipative_recourse_certificate': False,
                         'next_update_slot': None if next_update_slot is None else int(next_update_slot),
                         'future_shortfall_weight': future_weight,
                         'planning_emergency_weights': shortfall_weights.tolist(),
                         'opportunity_proxy_active': bool(future_weight is not None
                             and next_update_slot is not None and next_update_slot < n),
                         'downward_revision_allowed': original is not None,
                         'settlement': 'original+1.5*up-0.5*down+5*emergency'}}

