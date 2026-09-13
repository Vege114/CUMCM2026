"""Bounded retrospective forecast diagnostics; no network or dispatch training.

Seven fixed historical summaries inspect local mean, weekday and PV shape
information. Every summary uses completed days before its own midnight.
Scores use the examined 2025 development period, not an independent test.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.exp008.neural_joint_calibration import JointStore
from experiments.exp008.forecast_calibration import CalibratedStore
from experiments.exp008.load_energy_memory import EnergyMemoryStore
from experiments.problem2.exp003.data import Data, ROOT
from experiments.problem2.exp004.predict import ForecastStore

OUT = ROOT / 'data/results/exp008/forecast_shape_diagnostic'
WINDOWS = (7, 14, 28)


def robust_mean(values, axis=0):
    """Fixed 20% winsorized mean, not a score-selected quantile."""
    low, high = np.quantile(values, [.2, .8], axis=axis, keepdims=True)
    return np.clip(values, low, high).mean(axis=axis)


def historical_forecast(actual, day, window, weekday_load=False, normalized_pv=False):
    if day < window or window not in WINDOWS:
        raise ValueError('requires a complete fixed historical window')
    stop = day * 144
    history = np.asarray(actual[stop-window*144:stop, :2]).reshape(window, 24, 6, 2).mean(2)
    value = robust_mean(history)
    if weekday_load:
        same = (np.arange(day-window, day)-day) % 7 == 0
        value[:, 0] = robust_mean(history[same, :, 0])
    if normalized_pv:
        if window != 28:
            raise ValueError('one predeclared shape window: 28 days')
        pv = history[:, :, 1]
        energy = pv.sum(axis=1)  # kWh: one-hour interval mean kW.
        valid = energy > 1e-9
        shape = robust_mean(pv[valid] / energy[valid, None]) if valid.any() else np.zeros(24)
        shape /= max(shape.sum(), 1e-9)
        value[:, 1] = shape * robust_mean(energy[-7:])
    return np.repeat(np.maximum(value, 0), 6, axis=0)


def summary(error, price):
    high = price >= np.quantile(price, .75)
    energy = error.sum(1) / 6
    cumulative = np.cumsum(error, axis=1) / 6
    day_level = error.mean(1)
    return {'rmse_kw': float(np.sqrt(np.mean(error**2))),
            'mae_kw': float(np.abs(error).mean()), 'bias_kw': float(error.mean()),
            'price_weighted_rmse_kw': float(np.sqrt(np.mean(error**2*price)/price.mean())),
            'high_price_rmse_kw': float(np.sqrt(np.mean(error[:, high]**2))),
            'high_price_bias_kw': float(error[:, high].mean()),
            'daily_energy_rmse_kwh': float(np.sqrt(np.mean(energy**2))),
            'daily_energy_mae_kwh': float(np.abs(energy).mean()),
            'cumulative_error_rmse_kwh': float(np.sqrt(np.mean(cumulative**2))),
            'daily_level_mse_kw2': float(np.mean(day_level**2)),
            'within_day_mse_kw2': float(np.mean((error-day_level[:, None])**2)),
            'high_price_underprediction_kwh': float(np.maximum(-error[:, high], 0).sum()/6)}


def run():
    OUT.mkdir(exist_ok=True, parents=True)
    data = Data()
    original = ForecastStore('no_season', seed=42)
    days = original.origins // 144
    dates = pd.date_range('2025-02-01', '2025-12-31')
    values = {'original_raw': original.values, 'joint_raw': JointStore().values,
              'original_ridge28': CalibratedStore('ridge_28').values,
              'joint_ridge28': JointStore(calibrated=True).values}
    values['joint_ridge28_load_energy_memory_half'] = EnergyMemoryStore(data=data).values
    for window in WINDOWS:
        for weekday in (False, True):
            name = f'{"weekday" if weekday else "recent"}_hourly_winsor{window}'
            values[name] = np.stack([historical_forecast(data.actual, int(day), window, weekday)
                                     for day in days])
    values['weekday28_pvshape28_energy7'] = np.stack([
        historical_forecast(data.actual, int(day), 28, True, True) for day in days])
    with np.load(ROOT/'data/results/exp008/forecast_net_hgb/predictions.npz') as hgb:
        hgb_net = hgb['net_kw'].copy()
        assert np.array_equal(hgb['origins'], original.origins)
    mutation = []
    for day in (31, 62, 243, 364):
        changed = data.actual.copy()
        changed[day*144:] += 50000
        for window in WINDOWS:
            for weekday in (False, True):
                a = historical_forecast(data.actual, day, window, weekday)
                b = historical_forecast(changed, day, window, weekday)
                error = float(np.max(np.abs(a-b)))
                assert error == 0
                mutation.append({'day': day, 'window': window, 'weekday': weekday, 'max_error': error})
        a = historical_forecast(data.actual, day, 28, True, True)
        b = historical_forecast(changed, day, 28, True, True)
        assert np.array_equal(a, b)
    # Labels below are exclusively retrospective scoring inputs.
    truth = data.actual[original.origins[:, None]+np.arange(144), :2]
    net_truth = truth[:, :, 0]-truth[:, :, 1]
    annual, monthly, hourly, daily, decomposition = [], [], [], [], []
    errors = {}
    for name, prediction in values.items():
        for j, channel in enumerate(('load', 'pv', 'net')):
            error = (prediction[:, :, j]-truth[:, :, j] if j < 2 else
                     prediction[:, :, 0]-prediction[:, :, 1]-net_truth)
            errors[name, channel] = error
    errors['net_hgb', 'net'] = hgb_net-net_truth
    for (name, channel), error in errors.items():
        annual.append({'name': name, 'channel': channel, **summary(error, data.fixed_price)})
        for month in range(2, 13):
            ids = dates.month == month
            monthly.append({'name': name, 'channel': channel, 'month': month,
                            **summary(error[ids], data.fixed_price)})
        for hour in range(24):
            sl = slice(hour*6, (hour+1)*6)
            e = error[:, sl]
            hourly.append({'name': name, 'channel': channel, 'hour_start': hour,
                           'mean_tariff': float(data.fixed_price[sl].mean()),
                           'rmse_kw': float(np.sqrt(np.mean(e**2))), 'mae_kw': float(np.abs(e).mean()),
                           'bias_kw': float(e.mean()), 'squared_error_sum_kw2': float(np.sum(e**2))})
        if channel == 'net':
            for i, date in enumerate(dates):
                daily.append({'name': name, 'day': int(days[i]), 'date': str(date.date()),
                              'month': date.month, 'day_of_month': date.day,
                              **summary(error[i:i+1], data.fixed_price)})
            for group, ids in [('month_first7', dates.day<=7), ('month_rest', dates.day>7)]:
                decomposition.append({'name': name, 'group': group, 'days': int(ids.sum()),
                                      **summary(error[ids], data.fixed_price)})
    for filename, rows in [('annual', annual), ('monthly', monthly), ('hourly', hourly),
                           ('daily', daily), ('month_age', decomposition)]:
        pd.DataFrame(rows).to_csv(OUT/f'{filename}.csv', index=False)
    np.savez_compressed(OUT/'historical_predictions.npz', origins=original.origins,
                        **{k: v for k, v in values.items() if 'winsor' in k or 'pvshape' in k})
    protocol = {'historical_baselines': 7, 'windows': WINDOWS,
                'single_fixed_followup_candidate': 'joint_ridge28_load_energy_memory_half',
                'robust_mean': '20% winsorized mean of completed daily hourly means',
                'weekday': 'load uses strictly same weekday in last window; PV uses all days',
                'normalized_pv': '28-day robust mean of unit-energy hourly PV shapes times 7-day robust energy mean',
                'hour_interpretation': 'six ten-minute interval means collapsed to hourly mean, then repeated',
                'forecast_inputs': 'strictly prior complete actual load/PV and known weekday only',
                'known_fixed_tariff': 'evaluation metrics only; threshold upper quartile of attachment1 tariff',
                'hgb_semantics': 'score net only; adapter channels are not component forecasts',
                'model_fitting_performed': False, 'dispatch_performed': False,
                'development_year_not_independent_validation': True,
                'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'mutation_checks': mutation, 'normalized_shape_mutation_passed': True}
    (OUT/'protocol.json').write_text(json.dumps(protocol, indent=2))
    print(pd.DataFrame(annual).query('channel == "net"').to_string(index=False), flush=True)


if __name__ == '__main__':
    run()
