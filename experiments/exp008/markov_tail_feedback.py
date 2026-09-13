"""Three-state causal inventory DP with conditional residual-tail quadrature.

The error-state transitions and classification exactly follow markov_feedback.
Within each state, three equal-probability empirical quantile-bin means retain
the conditional mean while representing dispersion. They do not introduce
future information, extra regimes, or a guaranteed accurate extreme tail.
"""

from time import perf_counter

import numpy as np

from experiments.exp008 import markov_feedback as base
from experiments.exp008.markov_feedback import ETA, HIGH, LIMIT, LOW, MODES

execute = base.execute
initial_value = base.initial_value


def quantile_bin_means(sample, points=3):
    """Integrate the empirical quantile function in equal-mass intervals.

    Fractional allocation at interval edges makes the arithmetic mean of the
    representatives exactly equal to the original sample mean, even with one
    or two historical observations in a state.
    """
    values = np.sort(np.asarray(sample, float))
    if values.ndim != 1 or not len(values) or points not in (2, 3):
        raise ValueError('need observations and two or three quadrature points')
    edges = np.arange(len(values) + 1) / len(values)
    result = np.empty(points)
    for j in range(points):
        mass = np.maximum(0., np.minimum(edges[1:], (j + 1) / points)
                          - np.maximum(edges[:-1], j / points))
        result[j] = points * np.dot(mass, values)
    return result


def add_conditional_support(model, errors, points=3):
    errors = np.asarray(errors, float)
    bins = np.searchsorted(model['state_edges'],
                          (errors - model['mean_kwh']) / model['scale_kwh'], side='right')
    n = errors.shape[1]
    support = np.empty((n, 3, points))
    counts = np.empty((n, 3), dtype=int)
    for t in range(n):
        for state in range(3):
            selected = errors[bins[:, t] == state, t]
            counts[t, state] = len(selected)
            support[t, state] = (quantile_bin_means(selected, points) if len(selected)
                                  else model['error_support_kwh'][t, state])
    if not np.allclose(support.mean(axis=2), model['error_support_kwh'], atol=1e-10):
        raise AssertionError('tail quadrature must preserve all conditional means')
    model['conditional_error_support_kwh'] = support
    model['conditional_emission_weights'] = np.full_like(support, 1 / points)
    model['metadata'].update(
        emission_quadrature='equal_probability_empirical_quantile_bin_means',
        conditional_emission_points=points,
        conditional_means_preserved=True,
        empty_state_behavior='repeat original shrunken/fallback state mean',
        empty_slot_states=int(np.sum(counts == 0)),
        single_observation_slot_states=int(np.sum(counts == 1)),
        expected_positive_part_remains_a_discretization_approximation=True,
    )
    return model


def fit_error_model(forecast_net_kwh, historical_errors_kwh, *, points=3, **kwargs):
    model = base.fit_error_model(forecast_net_kwh, historical_errors_kwh, **kwargs)
    return add_conditional_support(model, historical_errors_kwh, points)


def model_for_day(data, store, day, independent=False, points=3):
    """Same historical observations and periodic cold start as the base model."""
    from experiments.problem2.tree_planning.risk import periodic_baseline

    model = base.model_for_day(data, store, day, independent=independent)
    errors = []
    for origin in model['metadata']['history_origins']:
        old_day = origin // 144
        if model['metadata']['fallback']:
            yesterday = data.actual[origin - 144:origin, :2]
            previous_week = (data.actual[origin - 7 * 144:origin - 6 * 144, :2]
                             if old_day >= 7 else None)
            forecast = periodic_baseline(yesterday, previous_week)
        else:
            forecast = store.get(origin)
        actual = data.actual[origin:origin + 144, :2]
        errors.append(((actual[:, 0] - actual[:, 1]) -
                       (forecast[:, 0] - forecast[:, 1])) / 6)
    errors = np.asarray(errors)
    if not np.allclose(errors.mean(axis=0), model['mean_kwh'], atol=1e-10):
        raise AssertionError('tail and base models must use identical historical residuals')
    return add_conditional_support(model, errors, points)


def value_functions(purchase, model, price, *, grid_kwh=200., wear=.002,
                    switching=100., terminal=.45):
    """V[t,previous_error_state,previous_direction+1,SOC] before t is observed.

    First optimize the action after the current state and emission are
    observed, then average emissions within state and apply its transition.
    Future continuation only knows the current three-bin state; it does not
    know a persistent quadrature index or any future sample emissions.
    """
    began = perf_counter()
    q, price = np.asarray(purchase, float), np.asarray(price, float)
    net = np.asarray(model['forecast_net_kwh'], float)
    support = net[:, None, None] + np.asarray(model['conditional_error_support_kwh'], float)
    weights = np.asarray(model['conditional_emission_weights'], float)
    transition = np.asarray(model['transition'], float)
    n = len(q)
    if (q.shape != (n,) or price.shape != (n,) or net.shape != (n,)
            or support.shape[:2] != (n, 3) or weights.shape != support.shape
            or transition.shape != (n, 3, 3) or grid_kwh <= 0 or grid_kwh > HIGH - LOW
            or min(wear, switching, terminal) < 0 or np.any(q < 0) or np.any(price <= 0)):
        raise ValueError('invalid trajectories, weights, grid, or auxiliary costs')
    if not all(np.isfinite(a).all() for a in (q, price, support, weights, transition)):
        raise ValueError('nonfinite planning inputs')
    if (np.any(weights < 0) or np.any(transition < 0)
            or not np.allclose(weights.sum(axis=-1), 1)
            or not np.allclose(transition.sum(axis=-1), 1)):
        raise ValueError('emission and transition probabilities must sum to one')
    grid = np.r_[np.arange(LOW, HIGH, grid_kwh), HIGH]
    values = np.zeros((n + 1, 3, 3, len(grid)))
    values[-1] = -terminal * (grid - LOW)
    delta = grid[None, :] - grid[:, None]
    for t in range(n - 1, -1, -1):
        observed_value = np.zeros((3, 3, len(grid)))
        for state in range(3):
            future = values[t + 1, state]
            for emission in range(support.shape[2]):
                balance = q[t] - support[t, state, emission]
                direction = 1 if balance >= 0 else -1
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
                    observed_value[state, mi] += weights[t, state, emission] * np.minimum(idle, switched)
        values[t] = np.einsum('ij,jms->ims', transition[t], observed_value, optimize=False)
    return grid, values, {
        'method':'inventory_value_with_conditional_tail_quadrature',
        'seconds':perf_counter() - began, 'grid_kwh':grid_kwh,
        'error_states':3, 'direction_states':3,
        'conditional_emission_points':support.shape[2],
        'wear_yuan_per_kwh_auxiliary':wear, 'switching_yuan_auxiliary':switching,
        'terminal_value_yuan_per_soc_kwh':terminal, 'calibration':model['metadata'],
        'continuous_global_optimality_certificate':False,
    }


def check():
    """Mean conservation, scalar positive-part check and collapsed-model parity."""
    errors = np.array([-300., -150., -50., 0., 50., 150., 300.])[:, None]
    model = fit_error_model(np.zeros(1), errors, independent=True)
    q, price = np.array([200.]), np.ones(1)
    grid, values, _ = value_functions(q, model, price, wear=0., switching=0., terminal=0.)
    realized = initial_value(grid, values, model, LOW)
    support = model['conditional_error_support_kwh'][0]
    exact = float(np.sum(model['marginal_probabilities'][0, :, None]
                         * model['conditional_emission_weights'][0]
                         * 5 * np.maximum(support - q[0], 0)))
    assert abs(realized - exact) < 1e-9
    bg, bv, _ = base.value_functions(q, model, price, wear=0., switching=0., terminal=0.)
    mean_only = base.initial_value(bg, bv, model, LOW)
    assert realized > mean_only + 1e-6
    rng = np.random.default_rng(8016)
    model = fit_error_model(np.ones(6) * 500., rng.normal(0, 150, (20, 6)))
    model['conditional_error_support_kwh'][:] = model['error_support_kwh'][:, :, None]
    q, price = np.full(6, 450.), np.linspace(.4, 1.2, 6)
    tg, tv, _ = value_functions(q, model, price)
    bg, bv, _ = base.value_functions(q, model, price)
    parity = float(np.max(np.abs(tv - bv)))
    assert np.array_equal(tg, bg) and parity < 1e-8
    actual = np.stack((np.ones(6) * 3300, np.ones(6) * 100), axis=1)
    first, _ = execute(q, actual, price, 5000., tg, tv, model)
    changed = actual.copy()
    changed[3:, 0] += 1000
    second, _ = execute(q, changed, price, 5000., tg, tv, model)
    assert np.array_equal(first['charge'][:3], second['charge'][:3])
    assert np.array_equal(first['discharge'][:3], second['discharge'][:3])
    return {'passed':True, 'conditional_mean_conserved':True,
            'scalar_positive_part_expected':exact, 'scalar_positive_part_dp':realized,
            'scalar_mean_only_dp':mean_only, 'collapsed_model_max_error':parity,
            'future_actual_prefix_invariance':True}


if __name__ == '__main__':
    import json
    print(json.dumps(check(), indent=2))
