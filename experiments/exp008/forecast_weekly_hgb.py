"""One weekly refit candidate for the existing absolute two-channel HGB.

The original monthly model and every existing archive remain unchanged.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from experiments.exp008.forecast_absolute_hgb import (
    AbsoluteHGBStore, ArrayStore, OUT as BASE, array_hash, features_for_day)
from experiments.exp008.forecast_calibration import CalibratedStore, _calibrate, _daylight, CONFIG
from experiments.exp008.forecast_shape_diagnostic import summary as score
from experiments.exp008.load_energy_memory import EnergyMemoryStore, correct_day
from experiments.problem2.exp003.data import Data
from experiments.problem2.exp004.data import age_weights

OUT = Path('data/results/exp008/forecast_weekly_hgb')
ASOFS = tuple(range(31, 365, 7))
KEYS = ('rmse_kw', 'high_price_rmse_kw', 'daily_energy_rmse_kwh', 'cumulative_error_rmse_kwh')


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def fit(data, asof, x, config):
    cutoff = asof-7
    train, val = np.arange(7, cutoff), np.arange(cutoff, asof)
    formal = np.arange(asof, min(asof+7, 365))
    xt, xv = [x[days-7].reshape(-1, x.shape[-1]) for days in (train, val)]
    yt, yv = [data.actual[days[:, None]*144+np.arange(144), :2].reshape(-1, 2)
              for days in (train, val)]
    assert train[-1]+1 <= cutoff and val[-1]+1 == asof
    weights = np.repeat(age_weights(train, cutoff, 90), 144)
    models, channels = [], []
    began = perf_counter()
    with threadpool_limits(limits=1):
        for channel in range(2):
            model = HistGradientBoostingRegressor(**config)
            model.fit(xt, yt[:, channel], sample_weight=weights, X_val=xv, y_val=yv[:, channel])
            models.append(model)
            channels.append({'channel': channel, 'iterations': int(model.n_iter_)})
    return models, {'asof_day': asof, 'training_cutoff_day': cutoff,
        'training_days': train.tolist(), 'validation_days': val.tolist(), 'formal_days': formal.tolist(),
        'training_features_sha256': array_hash(xt), 'validation_features_sha256': array_hash(xv),
        'training_labels_sha256': array_hash(yt), 'validation_labels_sha256': array_hash(yv),
        'training_weight_sha256': array_hash(weights), 'channels': channels,
        'model_label_stop_exclusive': asof*144, 'training_seconds': perf_counter()-began}


def predict(data, days, x, models):
    days = np.asarray(days, int)
    with threadpool_limits(limits=1):
        result = np.column_stack([m.predict(x[days-7].reshape(-1, x.shape[-1])) for m in models])
    result = np.maximum(result.reshape(len(days), 144, 2), 0.)
    for i, day in enumerate(days):
        result[i, :, 1] *= _daylight(data.actual, int(day))
    return result


def main():
    if OUT.exists():
        raise FileExistsError('Existing development results are immutable')
    OUT.mkdir(parents=True)
    data = Data()
    configuration = json.loads((BASE/'protocol.json').read_text())['model_configuration']
    sources = ('forecast_weekly_hgb.py', 'forecast_absolute_hgb.py', 'forecast_net_hgb.py',
               'forecast_calibration.py', 'load_energy_memory.py', 'audited_store_bridge.py')
    source_hashes = {}
    for name in sources:
        source = Path(__file__).parent/name
        archived = OUT/'source_archive'/name
        archived.parent.mkdir(exist_ok=True)
        archived.write_bytes(source.read_bytes())
        source_hashes[name] = hashlib.sha256(archived.read_bytes()).hexdigest()
    save(OUT/'protocol.json', {'one_fixed_candidate': True, 'refit_asofs': ASOFS,
        'number_of_blocks': len(ASOFS), 'number_of_models': 2*len(ASOFS),
        'model_configuration': configuration, 'same_original31_features': True,
        'split': 'train[7,asof-7),validation[asof-7,asof),formal[asof,min(asof+7,365))',
        'history_weight_half_life_days': 90, 'training_from_scratch': True,
        'postprocessing': 'same Ridge28 and nonrecursive .5 memory on own issued history',
        'only_refit_calendar_changed_from_original_monthly_HGB': True,
        'prior_weekly_Joint_CNN_failure_is_a_different_family': True,
        'gate_before_LP': 'all4 net metrics strictly improve originalHGB fullpipeline',
        'gate_metrics': KEYS, 'development_not_independent_test': True,
        'source_sha256': source_hashes, 'data_sha256': data.hashes})
    x = np.stack([features_for_day(data, day)[0] for day in range(7, 365)])
    values = np.empty((334, 144, 2)); models = {}; audits = []
    model_ids = np.empty(334, int)
    for asof in ASOFS:
        bundle, audit = fit(data, asof, x, configuration)
        days = np.asarray(audit['formal_days'])
        value = predict(data, days, x, bundle)
        values[days-31] = value; model_ids[days-31] = asof
        path = OUT/f'weekly_asof{asof:03}.joblib'
        joblib.dump(bundle, path)
        np.testing.assert_array_equal(value, predict(data, days, x, joblib.load(path)))
        audit['model_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        save(OUT/f'weekly_asof{asof:03}.json', audit)
        audits.append(audit); models[asof] = bundle
        print('weeklyHGB asof', asof, 'iterations', [c['iterations'] for c in audit['channels']], flush=True)
    raw = ArrayStore(values, np.arange(31, 365)*144, 'weekly_absolute_hgb_raw')
    ridge = CalibratedStore('ridge_28', data=data, use_cache=False, _base_store=raw,
                            directory=OUT/'calibration')
    memory = EnergyMemoryStore(data=data, base_store=ridge)
    memory.name = 'weekly_absolute_hgb_ridge28_memory_half'
    for row in memory.audit:
        row['underlying_forecast'] = 'weekly_absolute_HGB_then_Ridge28'
    for name, store in (('raw', raw), ('ridge', ridge), ('memory', memory)):
        np.savez_compressed(OUT/f'{name}.npz', origins=store.origins, values=store.values, model_asofs=model_ids)
    save(OUT/'training_audit.json', audits); save(OUT/'ridge_audit.json', ridge.audit)
    save(OUT/'memory_audit.json', memory.audit)
    np.savez_compressed(OUT/'features.npz', values=x, days=np.arange(7, 365))
    mutations = []
    for day in (31, 37, 38, 90, 151, 243, 364):
        changed = copy.copy(data); changed.actual = data.actual.copy()
        changed.actual[day*144:] += [70000., 30000.]
        fresh_x, _ = features_for_day(changed, day)
        np.testing.assert_array_equal(fresh_x, x[day-7])
        with threadpool_limits(limits=1):
            fresh = np.maximum(0., np.column_stack([m.predict(fresh_x) for m in models[model_ids[day-31]]]))
        fresh[:, 1] *= _daylight(changed.actual, day)
        np.testing.assert_array_equal(fresh, values[day-31])
        i = day-31
        later = values.copy(); later[i+1:] += [80000., 20000.]
        np.testing.assert_array_equal(_calibrate(data.actual, raw.origins, values, i, CONFIG['ridge_28'])[0],
            _calibrate(changed.actual, raw.origins, later, i, CONFIG['ridge_28'])[0])
        later = ridge.values.copy(); later[i+1:] += [30000., 90000.]
        np.testing.assert_array_equal(correct_day(data.actual, raw.origins, ridge.values, i)[0],
            correct_day(changed.actual, raw.origins, later, i)[0])
        mutations.append({'day': day, 'model_asof': int(model_ids[i]),
                          'features_raw_prediction_and_postprocessing_unchanged': True})
    retraining = []
    for asof in (31, 241):
        changed = copy.copy(data); changed.actual = data.actual.copy()
        changed.actual[asof*144:] += [60000., 40000.]
        xx = np.stack([features_for_day(changed, day)[0] for day in range(7, 365)])
        np.testing.assert_array_equal(xx[:asof-7], x[:asof-7])
        bundle, audit = fit(changed, asof, xx, configuration)
        np.testing.assert_array_equal(predict(changed, [asof], xx, bundle), predict(data, [asof], x, models[asof]))
        retraining.append({'asof': asof, 'full_retraining_first_prediction_unchanged': True})
    # Raw first seven forecasts must reproduce the identical Feb1 split.
    original_raw = AbsoluteHGBStore('direct_hgb_raw')
    np.testing.assert_array_equal(values[:7], original_raw.values[:7])
    save(OUT/'verification.json', {'passed': True, 'all48_bundle_reloads_identical': True,
        'future_mutations': mutations, 'full_retraining_mutations': retraining,
        'first7_raw_values_identical_to_original_HGB': True,
        'all_model_issues_no_later_than_forecast': bool(np.all(model_ids <= np.arange(31, 365)))})
    truth = data.actual[raw.origins[:, None]+np.arange(144), :2]
    records = []
    for name, store in (('original_HGB', AbsoluteHGBStore()), ('weekly_HGB', memory)):
        for channel in ('load', 'pv', 'net'):
            e = store.values-truth
            error = e[:, :, 0]-e[:, :, 1] if channel == 'net' else e[:, :, int(channel == 'pv')]
            records.append({'name': name, 'channel': channel, **score(error, data.fixed_price)})
    pd.DataFrame(records).to_csv(OUT/'metrics.csv', index=False)
    net = {r['name']: r for r in records if r['channel'] == 'net'}
    gates = {key: net['weekly_HGB'][key] < net['original_HGB'][key] for key in KEYS}
    result = {'complete': True, 'net_metrics': net, 'gate_components': gates,
              'LP_continuation_gate_passed': all(gates.values()), 'final_model_selection': False}
    save(OUT/'summary.json', result); print(json.dumps(result), flush=True)
    if all(gates.values()):
        from experiments.exp008.audited_store_bridge import run
        run(memory, OUT/'lp_bridge', [OUT/'protocol.json', OUT/'verification.json', OUT/'memory.npz'])


if __name__ == '__main__':
    main()
