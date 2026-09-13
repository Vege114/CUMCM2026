"""One month-forward whole-error-path weighting diagnostic; no model fitting.

Current HGB/Ridge28/memory forecasts and dispatch are immutable inputs.
Historical paths retain their joint load/PV and time dependence. The single
fixed weight rule does not re-estimate conditional marginals or a copula.
"""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd

from experiments.exp008.forecast_absolute_hgb import AbsoluteHGBStore
from experiments.exp008.absolute_hgb_forecast_adapter import AbsoluteHGBForecasts
from experiments.exp008.risk_window import WindowResidualScenarios
from experiments.problem2.exp003.data import Data, ROOT

OUT = ROOT / 'data/results/exp008/hgb_residual_state_diagnostic'
HGB = ROOT / 'data/results/exp008/forecast_absolute_hgb'
BRIDGE = HGB / 'lp_bridge/direct_hgb_ridge28_memory_tree28_q08_buffer500_334days'
STATE_NAMES = ['previous_complete_load_mean_residual_kw',
    'current_load_mean_minus_previous_7_actual_mean_kw',
    'current_PV_energy_minus_previous_7_actual_mean_kwh',
    'previous_complete_PV_AM_minus_PM_shape_residual_kw']
DATES = pd.date_range('2025-02-01', '2025-12-31')
SHAPE = np.zeros(144)
SHAPE[36:72], SHAPE[72:108] = 1 / 72, -1 / 72


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ah(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def save(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def observed(actual, start, stop, issue):
    if not 0 <= start < stop <= issue:
        raise ValueError('Incomplete actual data requested at issue')
    return np.asarray(actual[start:stop])


def states(actual, store):
    result = []
    for i, origin in enumerate(store.origins):
        origin = int(origin)
        history = observed(actual, origin - 7 * 144, origin, origin).reshape(7, 144, 2)
        previous = observed(actual, origin - 144, origin, origin)
        previous_error = previous - store.values[i - 1] if i else np.zeros_like(previous)
        result.append([previous_error[:, 0].mean(),
            store.values[i, :, 0].mean() - history[:, :, 0].mean(),
            store.values[i, :, 1].sum() / 6 - history[:, :, 1].sum(axis=1).mean() / 6,
            previous_error[:, 1] @ SHAPE])
    return np.asarray(result)


def pool_for_month(index, state, store):
    month_start = int(np.flatnonzero(DATES.month == DATES[index].month)[0])
    # First issued day has no same-model previous issued residual for its state.
    pool = np.arange(max(1, month_start - 56), month_start)
    if len(pool) < 14:
        raise ValueError('No month-forward diagnostic before 14 complete same-model states')
    x = state[pool]
    scale = np.maximum(np.std(x, axis=0), 1.)
    return pool, scale, month_start


def path_weights(current, old, scale):
    distance_squared = np.mean(((old - current) / scale) ** 2, axis=1)
    kernel = np.exp(-.5 * (distance_squared - distance_squared.min()))
    return .5 / len(old) + .5 * kernel / kernel.sum()


def weighted_crps(samples, truth, weights):
    """Proper CRPS of the stated weighted discrete predictive CDF.

    O(n log n) weighted pairwise absolute-difference formula includes diagonal
    pairs; it is not the unbiased score of an unspecified parent distribution.
    """
    order = np.argsort(samples, axis=0)
    sorted_x = np.take_along_axis(samples, order, axis=0)
    sorted_w = np.take_along_axis(np.broadcast_to(weights[:, None], samples.shape), order, axis=0)
    before = np.cumsum(sorted_w, axis=0) - sorted_w
    half_pair = np.sum(sorted_w * sorted_x * (2 * before + sorted_w - 1), axis=0)
    return np.sum(weights[:, None] * np.abs(samples - truth), axis=0) - half_pair


def risk_for_day(index, actual, store, state, weighted):
    pool, scale, month_start = pool_for_month(index, state, store)
    cutoff = int(store.origins[index])
    label_stop = int(store.origins[pool[-1]]) + 144
    assert label_stop <= int(store.origins[month_start]) <= cutoff
    errors = np.stack([observed(actual, int(store.origins[j]), int(store.origins[j]) + 144, cutoff)
                       - store.values[j] for j in pool])
    weights = path_weights(state[index], state[pool], scale) if weighted else np.ones(len(pool)) / len(pool)
    error_net = (errors[:, :, 0] - errors[:, :, 1]) / 6
    prediction_net = (store.values[index, :, 0] - store.values[index, :, 1]) / 6
    return prediction_net[None] + error_net, weights, errors, {
        'origin': cutoff, 'month_start_origin': int(store.origins[month_start]),
        'historical_origins': store.origins[pool].tolist(), 'label_stop_exclusive': label_stop,
        'current_state_last_observed_exclusive': cutoff, 'normalization_std': scale.tolist(),
        'current_state': state[index].tolist(), 'weights': weights.tolist(),
        'effective_sample_size': float(1 / np.square(weights).sum())}


def score(paths, truth, weights, price):
    slot = weighted_crps(paths, truth, weights)
    cumulative = np.cumsum(paths, axis=1)
    target = np.cumsum(truth)
    prefix = weighted_crps(cumulative, target, weights)
    mean_error = weights @ paths - truth
    mean_prefix = np.cumsum(mean_error)
    return {'slot_crps_kwh': float(slot.mean()),
        'price_weighted_slot_crps': float(np.average(slot, weights=price)),
        'prefix_crps_kwh': float(prefix.mean()), 'daily_energy_crps_kwh': float(prefix[-1]),
        'point_net_mse_kw2': float(np.mean((6 * mean_error) ** 2)),
        'point_prefix_mse_kwh2': float(np.mean(mean_prefix ** 2)),
        'point_daily_energy_squared_error_kwh2': float(mean_prefix[-1] ** 2),
        'point_daily_energy_bias_kwh': float(mean_prefix[-1])}


def protocol(store):
    if OUT.exists():
        raise FileExistsError('Diagnostic is immutable; no repeated parameter search')
    OUT.mkdir(parents=True)
    inputs = [HGB / 'direct_hgb_ridge28_memory_half.npz', BRIDGE / 'dispatch_2.npz', BRIDGE / 'supports.npz']
    result = {'candidate_count': 1, 'new_point_model_training': False, 'strategy_execution': False,
        'candidate': 'whole historical paired load/PV error paths with mild state-similarity weights',
        'state_names': STATE_NAMES,
        'pool': 'last56 complete same-model days strictly before current month start; exclude first issued state',
        'scored_period': 'March1 to December31, 306 days; February descriptive only',
        'held_month_target_labels_used': False,
        'current_day_state': 'known current issued forecasts and actual history strictly before current midnight',
        'scaling': 'per-feature std from month-frozen historical state pool, floor1 in recorded physical units',
        'weights': '0.5 uniform + 0.5 normalized exp(-0.5 * mean(standardized squared distances))',
        'fitting_or_bandwidth_search': False, 'conditional_marginal_or_copula_transformation': False,
        'references': ['same monthly frozen pool with uniform weights', 'online unweighted raw past28 same-model paths',
                       'original HGB complete pipeline point forecast', 'stored Tree28 slot marginal CDF only'],
        'CRPS': 'proper score of weighted empirical CDF; no iid scenario assumption',
        'gate': 'weighted improves price-weighted slot, prefix and daily-energy CRPS vs both historical-path references; '
                'also improves cumulative and daily-energy mean prediction RMSE vs frozen uniform and original forecast',
        'development_not_independent_test': True, 'source_sha256': digest(__file__),
        'input_sha256': {str(p): digest(p) for p in inputs}, 'prediction_values_sha256': ah(store.values),
        'raw_data_sha256': Data().hashes}
    save(OUT / 'protocol.json', result)
    shutil.copy2(__file__, OUT / 'source_snapshot.py')
    return result


def descriptive(actual, store, state):
    observed_all = actual[store.origins[:, None] + np.arange(144)]
    errors = observed_all - store.values
    with np.load(BRIDGE / 'dispatch_2.npz') as z:
        fees, emergency, soc = z['fees'], z['emergency'], z['states']
        np.testing.assert_array_equal(observed_all, z['actual'])
    load_mean = errors[:, :, 0].mean(axis=1)
    pv_energy = errors[:, :, 1].sum(axis=1) / 6
    pv_shape = errors[:, :, 1] @ SHAPE
    load_shape = errors[:, :, 0] - load_mean[:, None]
    # Daily zero-mean PV shape is a diagnostic decomposition, not a daytime forecast.
    pv_zero_mean = errors[:, :, 1] - errors[:, :, 1].mean(axis=1)[:, None]
    daily = pd.DataFrame({'day': store.origins // 144, 'date': DATES.astype(str), 'month': DATES.month,
        'load_mean_residual_kw': load_mean, 'pv_daily_energy_residual_kwh': pv_energy,
        'pv_am_minus_pm_shape_residual_kw': pv_shape,
        'net_daily_energy_residual_kwh': (errors[:, :, 0] - errors[:, :, 1]).sum(axis=1) / 6,
        'load_shape_rmse_kw': np.sqrt(np.mean(load_shape ** 2, axis=1)),
        'pv_zero_mean_shape_rmse_kw': np.sqrt(np.mean(pv_zero_mean ** 2, axis=1)),
        'emergency_cost': fees[:, :, 3].sum(axis=1), 'emergency_kwh': emergency.sum(axis=1),
        'predicted_load_mean_kw': store.values[:, :, 0].mean(axis=1),
        'predicted_pv_energy_kwh': store.values[:, :, 1].sum(axis=1) / 6,
        'initial_soc_kwh': soc[:, 0], 'final_soc_kwh': soc[:, -1]})
    for k, name in enumerate(STATE_NAMES):
        daily[name] = state[:, k]
    daily.to_csv(OUT / 'daily_descriptive.csv', index=False)
    month = daily.groupby('month').agg(days=('day', 'count'), emergency_cost=('emergency_cost', 'sum'),
        emergency_kwh=('emergency_kwh', 'sum'), load_bias_kw=('load_mean_residual_kw', 'mean'),
        pv_energy_bias_kwh=('pv_daily_energy_residual_kwh', 'mean'),
        pv_shape_bias_kw=('pv_am_minus_pm_shape_residual_kw', 'mean'))
    month['annual_emergency_fee_share'] = month.emergency_cost / daily.emergency_cost.sum()
    month.to_csv(OUT / 'monthly_concentration.csv')
    hourly = pd.DataFrame({'hour': np.arange(24),
        'emergency_cost': fees[:, :, 3].sum(axis=0).reshape(24, 6).sum(axis=1),
        'emergency_kwh': emergency.sum(axis=0).reshape(24, 6).sum(axis=1)})
    hourly['annual_emergency_fee_share'] = hourly.emergency_cost / daily.emergency_cost.sum()
    hourly.to_csv(OUT / 'hourly_concentration.csv', index=False)
    group_rows = []
    # Regime thresholds/ranks use only previous complete day states, never full-year quantiles.
    for feature in [STATE_NAMES[0], 'predicted_load_mean_kw', 'predicted_pv_energy_kwh', STATE_NAMES[3]]:
        values = daily[feature].to_numpy()
        labels = []
        for i, value in enumerate(values):
            if i < 7:
                labels.append('insufficient_history')
            else:
                prior = values[max(1, i - 28):i]
                rank = np.mean(prior <= value)
                labels.append('low' if rank < 1 / 3 else 'high' if rank >= 2 / 3 else 'middle')
        for label in sorted(set(labels)):
            ids = np.asarray(labels) == label
            group_rows.append({'known_state': feature, 'causal_rank_group': label, 'days': int(ids.sum()),
                'emergency_cost': float(daily.loc[ids, 'emergency_cost'].sum()),
                'emergency_cost_per_day': float(daily.loc[ids, 'emergency_cost'].mean()),
                'mean_realized_net_energy_residual_kwh': float(daily.loc[ids, 'net_daily_energy_residual_kwh'].mean())})
    pd.DataFrame(group_rows).to_csv(OUT / 'known_state_concentration.csv', index=False)
    targets = ['load_mean_residual_kw', 'pv_daily_energy_residual_kwh', 'pv_am_minus_pm_shape_residual_kw']
    correlation_rows = []
    for column in targets:
        for lag in (1, 2, 7):
            for month_num in (0, *range(2, 13)):
                current = daily[column].copy()
                previous = current.shift(lag)
                ids = np.ones(len(daily), bool) if month_num == 0 else DATES.month == month_num
                correlation_rows.append({'component': column, 'lag_complete_days': lag, 'month': month_num,
                    'correlation': float(current[ids].corr(previous[ids]))})
    pd.DataFrame(correlation_rows).to_csv(OUT / 'residual_serial_correlations.csv', index=False)
    return daily


def run():
    data, store = Data(), AbsoluteHGBStore()
    config = protocol(store)
    state = states(data.actual, store)
    daily = descriptive(data.actual, store, state)
    # CRPS implementation checked against explicit pairwise formula.
    sample, truth, w = np.array([[1., 7.], [2., 3.], [9., 4.]]), np.array([2., 5.]), np.array([.1, .3, .6])
    expected = np.sum(w[:, None] * np.abs(sample - truth), axis=0) - .5 * np.sum(
        w[:, None, None] * w[None, :, None] * np.abs(sample[:, None] - sample[None, :]), axis=(0, 1))
    np.testing.assert_allclose(weighted_crps(sample, truth, w), expected, rtol=0, atol=1e-12)
    rows, audits, shifts = [], [], []
    with np.load(BRIDGE / 'supports.npz') as z:
        tree_supports = z['supports'].copy()
    forecast_adapter = AbsoluteHGBForecasts()
    adapter_rows = []
    for day in (59, 90, 151, 243, 364):
        paths = forecast_adapter.net_error_paths(day)
        old = paths['origins'] // 144 - 31
        expected = data.actual[paths['origins'][:, None] + np.arange(144)] - store.values[old]
        np.testing.assert_array_equal(paths['errors_kw'], expected)
        assert paths['audit']['model_id'] == forecast_adapter.store.name
        adapter_rows.append({'day': day, 'same_model_error_paths_exact': True, 'complete_days': len(old)})
    for i in range(28, 334):
        weighted, weight, errors, audit = risk_for_day(i, data.actual, store, state, True)
        frozen, uniform, _, _ = risk_for_day(i, data.actual, store, state, False)
        # Construct every distribution before current truth is used to score it.
        prior = np.arange(i - 28, i)
        old_error = np.stack([observed(data.actual, int(store.origins[j]), int(store.origins[j]) + 144,
                             int(store.origins[i])) - store.values[j] for j in prior])
        base = (store.values[i, :, 0] - store.values[i, :, 1]) / 6
        online = base[None] + (old_error[:, :, 0] - old_error[:, :, 1]) / 6
        target = data.actual[int(store.origins[i]):int(store.origins[i]) + 144]
        truth = (target[:, 0] - target[:, 1]) / 6
        for name, paths, weights in [('frozen_uniform56', frozen, uniform),
                                     ('state_weighted56', weighted, weight),
                                     ('online_raw28', online, np.ones(28) / 28),
                                     ('original_point', base[None], np.ones(1))]:
            rows.append({'day': i + 31, 'date': str(DATES[i].date()), 'month': int(DATES[i].month),
                         'name': name, **score(paths, truth, weights, data.fixed_price)})
        tree = tree_supports[i].T
        # Nine marginal atoms carry the original equal LP weights. No prefix
        # score: equal-index atoms across time are not observed joint paths.
        slot = weighted_crps(tree, truth, np.ones(tree.shape[0]) / tree.shape[0])
        rows.append({'day': i + 31, 'date': str(DATES[i].date()), 'month': int(DATES[i].month),
            'name': 'tree28_slot_only', 'slot_crps_kwh': float(slot.mean()),
            'price_weighted_slot_crps': float(np.average(slot, weights=data.fixed_price))})
        for name, weights in [('frozen_uniform56', uniform), ('state_weighted56', weight)]:
            delta = np.einsum('n,ntc->tc', weights, errors)
            shifts.append({'day': i + 31, 'name': name, 'load_daily_mean_correction_kw': float(delta[:, 0].mean()),
                'pv_daily_energy_correction_kwh': float(delta[:, 1].sum() / 6),
                'pv_AM_minus_PM_shape_correction_kw': float(delta[:, 1] @ SHAPE),
                'net_daily_energy_correction_kwh': float((delta[:, 0] - delta[:, 1]).sum() / 6)})
        audits.append(audit)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / 'daily_proper_scores.csv', index=False)
    means = frame.groupby('name').mean(numeric_only=True).drop(columns=['day', 'month'])
    for stem in ('net', 'prefix', 'daily_energy'):
        col = {'net': 'point_net_mse_kw2', 'prefix': 'point_prefix_mse_kwh2',
               'daily_energy': 'point_daily_energy_squared_error_kwh2'}[stem]
        means['point_' + stem + '_rmse'] = np.sqrt(means[col])
    means.to_csv(OUT / 'aggregate_proper_scores.csv')
    monthly = frame.groupby(['month', 'name']).mean(numeric_only=True).drop(columns='day')
    monthly.to_csv(OUT / 'monthly_proper_scores.csv')
    pd.DataFrame(shifts).to_csv(OUT / 'daily_correction_decomposition.csv', index=False)
    save(OUT / 'monthly_forward_audit.json', audits)
    future_rows = []
    for day in (59, 90, 151, 243, 364):
        i = day - 31
        changed_data = data.actual.copy()
        changed_data[day * 144:] += np.array([70000., 50000.])
        changed_store = copy.copy(store)
        changed_store.values = store.values.copy()
        changed_store.values[i + 1:] += 80000.
        changed_state = states(changed_data, changed_store)
        np.testing.assert_array_equal(state[:i + 1], changed_state[:i + 1])
        a, w0, e0, _ = risk_for_day(i, data.actual, store, state, True)
        b, w1, e1, _ = risk_for_day(i, changed_data, changed_store, changed_state, True)
        for first, second in ((a, b), (w0, w1), (e0, e1)):
            np.testing.assert_array_equal(first, second)
        future_rows.append({'day': day, 'future_actual_and_forecast_mutation_unchanged': True})
    risk = WindowResidualScenarios(store.origins, store.values, data.actual, data.fixed_price)
    for day in (59, 90, 151, 243, 364):
        predicted, audit = risk.for_day(day)
        np.testing.assert_array_equal(predicted, tree_supports[day - 31])
    required = ['price_weighted_slot_crps', 'prefix_crps_kwh', 'daily_energy_crps_kwh']
    proper_gate = all(means.loc['state_weighted56', key] < means.loc[base, key]
                      for key in required for base in ['frozen_uniform56', 'online_raw28'])
    point_gate = all(means.loc['state_weighted56', key] < means.loc[base, key]
                     for key in ['point_prefix_rmse', 'point_daily_energy_rmse']
                     for base in ['frozen_uniform56', 'original_point'])
    # Convert missing slot-only prefix fields to JSON null.
    metrics = json.loads(means.to_json(orient='index'))
    summary = {'scored_days': 306, 'scoring_period': ['2025-03-01', '2025-12-31'],
        'metrics': metrics, 'proper_distribution_gate': bool(proper_gate), 'point_energy_gate': bool(point_gate),
        'worth_one_followup_trial': bool(proper_gate and point_gate),
        'candidate_count': 1, 'new_predictor_trained': False, 'dispatch_rerun': False,
        'scoring_is_month_forward_development_not_independent_test': True,
        'future_mutation_checks': future_rows, 'same_model_history_adapter_checks': adapter_rows,
        'stored_tree_support_reconstruction_checks': 5,
        'monthly_pool_labels_stop_before_held_month': all(a['label_stop_exclusive'] <= a['month_start_origin'] for a in audits),
        'mean_effective_sample_size': float(np.mean([a['effective_sample_size'] for a in audits])),
        'source_and_inputs_unchanged': all(digest(path) == h for path, h in config['input_sha256'].items())
            and digest(__file__) == config['source_sha256']}
    save(OUT / 'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    run()
