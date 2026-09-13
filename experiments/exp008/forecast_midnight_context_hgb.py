"""Fixed load-only HGB extension with the latest known evening load context."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from experiments.exp008.forecast_absolute_hgb import (
    AbsoluteHGBStore, ArrayStore, OUT as BASE, array_hash, features_for_day as base_features)
from experiments.exp008.forecast_calibration import CalibratedStore, _calibrate, CONFIG
from experiments.exp008.load_energy_memory import EnergyMemoryStore, correct_day
from experiments.exp008.forecast_shape_diagnostic import summary as score
from experiments.problem2.exp003.data import Data
from experiments.problem2.exp004.data import split_days, age_weights

OUT = Path('data/results/exp008/forecast_midnight_context_hgb')


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def features(data, day):
    x, names = base_features(data, day)
    load = data.actual[(day-7)*144:day*144, 0].reshape(7, 144)
    context, added = [], []
    for slots in (6, 18, 36):
        end_mean = load[:, -slots:].mean(1)
        context.extend([float(end_mean[-1]), float(end_mean[-1]-end_mean[:-1].mean())])
        added.extend([f'latest_{slots}_slot_load_mean', f'latest_{slots}_slot_load_minus_prior6_matching_mean'])
    context.extend([float(load[-1, -1]), float(load[-1, -18:].std()),
                    float(load[-1, -6:].mean()-load[-1, -18:-12].mean())])
    added.extend(['last_observed_load', 'latest18_load_std', 'latest_hour_minus_three_hours_ago_load'])
    return np.column_stack((x, np.broadcast_to(context, (144, len(context))))), names+added


def fit(data, month, x, config):
    train, val, formal, cutoff = split_days(month)
    xt = x[train-7].reshape(-1, x.shape[-1])
    xv = x[val-7].reshape(-1, x.shape[-1])
    yt = data.actual[train[:, None]*144+np.arange(144), 0].ravel()
    yv = data.actual[val[:, None]*144+np.arange(144), 0].ravel()
    assert (train[-1]+1)*144 <= cutoff*144 and (val[-1]+1)*144 <= formal[0]*144
    model = HistGradientBoostingRegressor(**config)
    with threadpool_limits(limits=1):
        model.fit(xt, yt, sample_weight=np.repeat(age_weights(train, cutoff, 90), 144),
                  X_val=xv, y_val=yv)
    audit = {'month': month, 'asof_day': int(formal[0]), 'training_days': train.tolist(),
             'validation_days': val.tolist(), 'formal_days': formal.tolist(),
             'maximum_label_exclusive': int(formal[0]*144), 'iterations': int(model.n_iter_),
             'training_features_sha256': array_hash(xt), 'validation_features_sha256': array_hash(xv),
             'training_labels_sha256': array_hash(yt), 'validation_labels_sha256': array_hash(yv)}
    return model, audit


def main():
    if OUT.exists():
        raise FileExistsError('Development candidates are immutable')
    OUT.mkdir(parents=True)
    config = json.loads((BASE/'protocol.json').read_text())['model_configuration']
    source = Path(__file__).read_bytes()
    (OUT/'source_snapshot.py').write_bytes(source)
    save(OUT/'protocol.json', {'one_fixed_candidate': True, 'model_configuration': config,
        'feature_count': 40, 'change': '9 load-only recent evening level/change features added to original31',
        'raw_pv': 'unchanged original absoluteHGB archive',
        'load_model': 'same originalmonthly training, prior7 validation,90dayweights,max100/patience10',
        'postprocessing': 'same Ridge28 and nonrecursive .5 load memory, both refit to augmented raw outputs',
        'all_features_at_each_historical_issue': True, 'no_current_day_actual_features': True,
        'gate_before_lp': 'netRMSE,highpriceNetRMSE,dailyEnergyRMSE,prefixEnergyRMSE all improve originalHGB',
        'development_not_independent_test': True, 'source_sha256': hashlib.sha256(source).hexdigest()})
    data = Data(); original_raw = AbsoluteHGBStore('direct_hgb_raw')
    original_hash = array_hash(original_raw.values)
    xx = [features(data, day) for day in range(7, 365)]
    x = np.stack([a[0] for a in xx]); names = xx[0][1]
    assert len(names) == 40
    values = original_raw.values.copy(); models = {}; audits = []
    for month in range(2, 13):
        model, audit = fit(data, month, x, config)
        days = np.asarray(audit['formal_days'])
        with threadpool_limits(limits=1):
            predicted = np.maximum(0., model.predict(x[days-7].reshape(-1, x.shape[-1]))).reshape(-1, 144)
        values[days-31, :, 0] = predicted
        path = OUT/f'load_m{month:02}.joblib'
        joblib.dump(model, path)
        with threadpool_limits(limits=1):
            reloaded = np.maximum(0., joblib.load(path).predict(x[days-7].reshape(-1, x.shape[-1]))).reshape(-1, 144)
        np.testing.assert_array_equal(reloaded, predicted)
        audit['model_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        audits.append(audit); models[month] = model
        print('midnight load context', month, 'iterations', audit['iterations'], flush=True)
    np.testing.assert_array_equal(values[:, :, 1], original_raw.values[:, :, 1])
    raw = ArrayStore(values, original_raw.origins, 'midnight_context_hgb_raw')
    ridge = CalibratedStore('ridge_28', data=data, use_cache=False, _base_store=raw,
                            directory=OUT/'calibration')
    memory = EnergyMemoryStore(data=data, base_store=ridge)
    memory.name = 'midnight_context_hgb_ridge28_memory_half'
    for a in memory.audit:
        a['underlying_forecast'] = 'load_midnight_context_HGB_original_rawPV_then_Ridge28'
    for name, store in (('raw', raw), ('ridge', ridge), ('memory', memory)):
        np.savez_compressed(OUT/f'{name}.npz', values=store.values, origins=store.origins)
    np.savez_compressed(OUT/'features.npz', values=x, days=np.arange(7, 365))
    save(OUT/'feature_names.json', names); save(OUT/'training_audit.json', audits)
    save(OUT/'ridge_audit.json', ridge.audit); save(OUT/'memory_audit.json', memory.audit)
    mutations = []
    for day in (31, 32, 59, 90, 151, 243, 364):
        changed = copy.copy(data); changed.actual = data.actual.copy()
        changed.actual[day*144:] += [50000., 30000.]
        changed_x, _ = features(changed, day)
        np.testing.assert_array_equal(changed_x, x[day-7])
        month = int((pd.Timestamp('2025-01-01')+pd.Timedelta(days=day)).month)
        with threadpool_limits(limits=1):
            predicted = np.maximum(0., models[month].predict(changed_x))
        np.testing.assert_array_equal(predicted, values[day-31, :, 0])
        i = day-31
        later_raw = raw.values.copy(); later_raw[i+1:] += [60000., 20000.]
        np.testing.assert_array_equal(_calibrate(data.actual, raw.origins, raw.values, i, CONFIG['ridge_28'])[0],
            _calibrate(changed.actual, raw.origins, later_raw, i, CONFIG['ridge_28'])[0])
        later_ridge = ridge.values.copy(); later_ridge[i+1:] += [30000., 80000.]
        np.testing.assert_array_equal(correct_day(data.actual, raw.origins, ridge.values, i)[0],
                                      correct_day(changed.actual, raw.origins, later_ridge, i)[0])
        mutations.append({'day': day, 'features_raw_load_and_postprocessing_unchanged': True})
    retraining = []
    for month in (2, 9):
        train, val, formal, _ = split_days(month)
        changed = copy.copy(data); changed.actual = data.actual.copy()
        changed.actual[formal[0]*144:] += [40000., 90000.]
        changed_x = np.stack([features(changed, day)[0] for day in range(7, 365)])
        np.testing.assert_array_equal(changed_x[np.r_[train, val]-7], x[np.r_[train, val]-7])
        fresh, _ = fit(changed, month, changed_x, config)
        with threadpool_limits(limits=1):
            a = models[month].predict(x[formal[0]-7]); b = fresh.predict(changed_x[formal[0]-7])
        np.testing.assert_array_equal(a, b)
        retraining.append({'month': month, 'whole_training_and_first_formal_day_prediction_unchanged': True})
    save(OUT/'verification.json', {'passed': True, 'all11_model_reloads_identical': True,
        'future_mutations': mutations, 'retraining_mutations': retraining,
        'raw_pv_unchanged': True, 'source_original_raw_sha256': original_hash})
    truth = data.actual[raw.origins[:, None]+np.arange(144)]
    comparisons = {'original_HGB': AbsoluteHGBStore(), 'midnight_context': memory}
    records = []
    for name, store in comparisons.items():
        for channel, e in [('load', store.values[:, :, 0]-truth[:, :, 0]),
                           ('pv', store.values[:, :, 1]-truth[:, :, 1]),
                           ('net', np.diff(truth, axis=2)[:, :, 0]-np.diff(store.values, axis=2)[:, :, 0])]:
            records.append({'name': name, 'channel': channel, **score(e, data.fixed_price)})
    pd.DataFrame(records).to_csv(OUT/'metrics.csv', index=False)
    net = {r['name']: r for r in records if r['channel'] == 'net'}
    keys = ('rmse_kw', 'high_price_rmse_kw', 'daily_energy_rmse_kwh', 'cumulative_error_rmse_kwh')
    # score() uses stable named fields; explicit gates avoid comparing biases.
    base_score, candidate_score = net['original_HGB'], net['midnight_context']
    gates = {key: candidate_score[key] < base_score[key] for key in keys}
    result = {'complete': True, 'net_metrics': net, 'gate_components': gates,
              'lp_continuation_gate_passed': all(gates.values()), 'final_model_selection': False}
    save(OUT/'summary.json', result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    if all(gates.values()):
        from experiments.exp008.audited_store_bridge import run
        run(memory, OUT/'lp_bridge', [OUT/'protocol.json', OUT/'verification.json', OUT/'memory.npz'])


if __name__ == '__main__':
    main()
