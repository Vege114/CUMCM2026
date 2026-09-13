"""Bounded causal calendar-shape residual calibration of the frozen CNN.

Two predeclared rolling Ridge configurations add one annual sine/cosine pair
interacting with a fixed smooth intraday Fourier basis. Calendar phases are
known at issue time; no annual observed shape, capacity or future label enters.
The annual pair is centered at the current issue to express a local smooth
seasonal correction. Its isotropic prior shrinks unknown annual amplitudes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from experiments.exp008.forecast_calibration import CalibratedStore, _daylight, _features
from experiments.problem2.exp003.data import ROOT, Data
from experiments.problem2.exp004.predict import ForecastStore

OUT = ROOT / 'data/results/exp008/calendar_forecast'
CONFIGS = [{'name': f'calendar_ridge{window}', 'window': window,
                'half_life': window/2, 'annual_harmonics': 1, 'daily_harmonics': 3,
                'seasonal_penalty': 20.} for window in (56, 90)]


def phases(days):
    phase = 2 * np.pi * np.asarray(days) / 365.2425
    return np.stack((np.sin(phase), np.cos(phase)), axis=-1)


def shape_basis():
    phase = 2 * np.pi * np.arange(144) / 144
    columns = [np.ones(144)]
    for harmonic in range(1, 4):
        columns.extend((np.sin(harmonic * phase), np.cos(harmonic * phase)))
    return np.stack(columns, axis=-1)


def issue_prediction(actual, origins, cnn, index, config, feature_cache=None):
    """Fit only completed historical issue outputs; return the next 144 slots."""
    cutoff = int(origins[index])
    day = cutoff // 144
    ids = np.arange(max(0, index - config['window']), index)
    if not np.all(origins[ids] + 144 <= cutoff):
        raise ValueError('training targets must be fully observed before issue')
    audit = {'origin': cutoff, 'day': day, 'history_origins': origins[ids].tolist(),
                 'information_cutoff_exclusive': cutoff, 'maximum_feature_actual_index': cutoff-1,
                 'maximum_training_label': int(origins[ids[-1]] + 143) if len(ids) else None,
                 'future_labels_used': False, 'cold_start': len(ids)==0}
    if not len(ids):
        return cnn[index].copy(), audit
    observed = np.stack([actual[int(origins[h]):int(origins[h]) + 144, :2] for h in ids])
    residual = observed - cnn[ids]
    age = np.arange(len(ids) - 1, -1, -1)
    day_weights = 2 ** (-age / config['half_life'])
    seasonal = phases(origins[ids] / 144) - phases(day)
    # The shape is analytic and identical for every date, not a fitted annual
    # observed curve. Periodic phase differences provide local extrapolation.
    shape = shape_basis()
    additional = np.einsum('hi,tj->htij', seasonal, shape, optimize=False).reshape(-1, 14)
    seasonal_penalty = config['seasonal_penalty'] * np.tile([1., 1., 1., 4., 4., 9., 9.], 2)
    delta = np.zeros((144, 2))
    coefficients = []
    for channel in (0, 1):
        old = (np.concatenate([_features(actual, int(origins[h] // 144), cnn[h], channel)
                               for h in ids]) if feature_cache is None else
               feature_cache[ids, channel].reshape(-1, 10))
        x = np.column_stack((old, additional))
        target = residual[:, :, channel].ravel() / 1000
        weights = np.repeat(day_weights, 144)
        if channel == 1:
            active = ((cnn[ids, :, 1] > 1) | (observed[:, :, 1] > 1)).ravel()
            weights *= np.where(active, 1., .1)
        penalty = np.r_[[2.], np.full(9, 20.), seasonal_penalty]
        gram = np.einsum('ni,nj,n->ij', x, x, weights, optimize=False)
        rhs = np.einsum('ni,n,n->i', x, weights, target, optimize=False)
        coefficient = np.linalg.solve(gram + np.diag(penalty), rhs)
        current = (_features(actual, day, cnn[index], channel) if feature_cache is None
                   else feature_cache[index, channel])
        correction = 1000 * np.einsum('ni,i->n', current, coefficient[:10], optimize=False)
        bound = max(100., 3 * float(np.sqrt(np.mean(residual[:, :, channel] ** 2))))
        delta[:, channel] = np.clip(correction, -bound, bound)
        coefficients.append(coefficient.tolist())
    values = np.maximum(0., cnn[index] + delta)
    values[:, 1] *= _daylight(actual, day)
    audit.update(coefficients=coefficients,
                 calendar_center_day=day,
                 learned_annual_amplitudes_kw=(1000*np.asarray(coefficients)[:,10:]).tolist())
    return values, audit


def score(values, truth, price):
    error = values - truth
    net = error[..., 0] - error[..., 1]
    energy = net.sum(axis=1) / 6
    intraday = np.cumsum(net / 6, axis=1)
    return {'net_rmse_kw': float(np.sqrt(np.mean(net**2))),
                'price_weighted_net_rmse_kw': float(np.sqrt(np.mean(net**2 * price) / np.mean(price))),
                'load_rmse_kw': float(np.sqrt(np.mean(error[...,0]**2))),
                'pv_rmse_kw': float(np.sqrt(np.mean(error[...,1]**2))),
                'net_bias_predicted_minus_actual_kw': float(net.mean()),
                'daily_net_energy_mae_kwh': float(np.abs(energy).mean()),
                'daily_net_energy_rmse_kwh': float(np.sqrt(np.mean(energy**2))),
                'intraday_cumulative_net_error_rmse_kwh': float(np.sqrt(np.mean(intraday**2))),
                'daily_max_abs_cumulative_net_error_mean_kwh': float(np.max(np.abs(intraday),axis=1).mean()),
                'cumulative_net_energy_error_kwh': float(energy.sum())}


def check():
    rng = np.random.default_rng(8417)
    actual = np.maximum(0., rng.normal(1000, 150, (50*144, 2)))
    origins = np.arange(31, 50) * 144
    cnn = np.maximum(0., rng.normal(950, 100, (len(origins), 144, 2)))
    index = 10
    original, audit = issue_prediction(actual, origins, cnn, index, CONFIGS[0])
    changed = actual.copy()
    changed[origins[index]:] += 90000
    future_cnn = cnn.copy()
    future_cnn[index+1:] += 80000
    mutated, _ = issue_prediction(changed, origins, future_cnn, index, CONFIGS[0])
    error = float(np.max(np.abs(original-mutated)))
    assert error == 0
    changed_cache=np.stack([np.stack([_features(changed,int(origin//144),future_cnn[i],c)
                                     for c in (0,1)]) for i,origin in enumerate(origins)])
    cached,_=issue_prediction(changed,origins,future_cnn,index,CONFIGS[0],changed_cache)
    cache_error=float(np.max(np.abs(original-cached)))
    assert cache_error == 0
    assert all(h+144<=origins[index] for h in audit['history_origins'])
    assert audit['maximum_training_label'] < origins[index]
    assert np.isfinite(original).all() and np.min(original)>=0
    return {'passed': True,'future_truth_and_future_forecast_invariance_max_error_kw': error,
                'precomputed_feature_cache_future_invariance_max_error_kw':cache_error,
                'completed_historical_labels_only': True,'finite_nonnegative_components': True}


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = {'configs': CONFIGS, 'selected_before_scoring': True,
        'retained_base': 'exp004 no_season CNN seed42', 'unchanged_original_ridge_features': True,
        'basis': 'one annual sine/cosine pair times constant and first 3 fixed intraday Fourier harmonics',
        'prior': 'zero annual residual amplitudes, isotropic phase prior, quadratic daily-frequency penalty',
        'no_annual_actual_shape_or_capacity': True, 'training': 'last 56 or 90 fully completed CNN issue days',
        'cold_start': 'unmodified CNN when no complete issued training day exists',
        'development_year_not_independent_test': True,
        'continuation_gate': 'point net and tariff-weighted RMSE improvement over ridge28 before cost linkage',
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    (OUT/'causality_checks.json').write_text(json.dumps(check(),indent=2))
    data, cnn = Data(), ForecastStore('no_season', seed=42)
    features=np.stack([np.stack([_features(data.actual,int(origin//144),cnn.values[i],c)
                                for c in (0,1)]) for i,origin in enumerate(cnn.origins)])
    predictions={'cnn':cnn.values,'ridge28':CalibratedStore('ridge_28').values,
                 'ridge56':CalibratedStore('ridge_56').values}
    for config in CONFIGS:
        begin=perf_counter()
        results=[issue_prediction(data.actual,cnn.origins,cnn.values,i,config,features)
                 for i in range(len(cnn.origins))]
        values=np.stack([r[0] for r in results])
        predictions[config['name']]=values
        path=OUT/f'{config["name"]}.npz'
        np.savez_compressed(path,origins=cnn.origins,values=values,delta=values-cnn.values)
        (OUT/f'{config["name"]}_audit.json').write_text(json.dumps({
            'config': config,'days': [r[1] for r in results],'source_data_hashes': data.hashes,
            'cnn_sha256': hashlib.sha256(cnn.values.tobytes()).hexdigest(),
            'archive_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'seconds': perf_counter()-begin},indent=2))
        print(config['name'],'finished',round(perf_counter()-begin,2),flush=True)
    # Full-year labels are used here only after predictions have been frozen.
    truth=data.actual[cnn.origins[:,None]+np.arange(144)]
    dates=pd.date_range('2025-02-01','2025-12-31')
    annual=[];monthly=[];daily=[]
    for name,values in predictions.items():
        annual.append(dict(name=name,**score(values,truth,data.fixed_price)))
        for month in range(2,13):
            ids=dates.month==month
            monthly.append(dict(name=name,month=month,**score(values[ids],truth[ids],data.fixed_price)))
        for i,date in enumerate(dates):
            daily.append(dict(name=name,date=str(date.date()),**score(values[i:i+1],truth[i:i+1],data.fixed_price)))
    pd.DataFrame(annual).to_csv(OUT/'annual_metrics.csv',index=False)
    pd.DataFrame(monthly).to_csv(OUT/'monthly_metrics.csv',index=False)
    pd.DataFrame(daily).to_csv(OUT/'daily_metrics.csv',index=False)
    print(json.dumps(annual,indent=2))


class CalendarStore:
    """Signed causal calendar outputs with the ordinary two-channel semantics."""
    def __init__(self, name='calendar_ridge90', directory=OUT):
        if name not in {config['name'] for config in CONFIGS}:
            raise ValueError('unknown fixed calendar configuration')
        self.name=name
        path=Path(directory)/f'{name}.npz'
        metadata=json.loads((Path(directory)/f'{name}_audit.json').read_text())
        if hashlib.sha256(path.read_bytes()).hexdigest()!=metadata['archive_sha256']:
            raise ValueError('calendar output archive hash mismatch')
        with np.load(path) as archive:
            self.origins=archive['origins'].copy()
            self.values=archive['values'].copy()
        self.audit=metadata['days']
        self.lookup={int(origin):i for i,origin in enumerate(self.origins)}

    def get(self, origin):
        return self.values[self.lookup[int(origin)]].copy()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    print(json.dumps(check(),indent=2)) if args.check else run()
