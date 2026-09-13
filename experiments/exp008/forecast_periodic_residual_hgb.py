"""One fixed periodic-reference residual HGB label comparison.

The 31 features, complexity, held-out validation and physical postprocessing
match the absolute HGB candidate. Only labels and output reconstruction use
one invariant causal baseline: last-week load and yesterday PV, in every
training, validation and forecast month. Never a CNN reference.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import shutil
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from experiments.exp008.forecast_net_hgb import features_for_day as residual_features
from experiments.exp008.forecast_calibration import CalibratedStore, _calibrate, _daylight, CONFIG
from experiments.exp008.forecast_shape_diagnostic import summary as error_summary
from experiments.exp008.forecast_absolute_hgb import AbsoluteHGBStore
from experiments.exp008.load_energy_memory import EnergyMemoryStore, correct_day
from experiments.exp008.neural_joint_calibration import JointStore
from experiments.problem2.exp003.data import Data, ROOT
from experiments.problem2.exp004.data import split_days, age_weights

OUT = ROOT / 'data/results/exp008/forecast_periodic_residual_hgb'
PRIMARY = 'periodic_residual_hgb_ridge28_memory_half'
COMPARATOR = 'direct_hgb_ridge28_memory_half'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_hash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def save(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def features_for_day(data, day):
    # The supplied zeros are immediately removed, along with base_is_cnn.
    # Reuse the exact remaining historical formulas without modifying old HGB.
    matrix, names = residual_features(data, day, np.zeros((144, 2)))
    return matrix[:, 3:-1], names[3:-1]


def periodic_base(data, day):
    if not 7 <= day < 365:
        raise ValueError('Seven complete days required for the fixed periodic baseline')
    issue = int(day) * 144
    load_start, pv_start = issue - 7 * 144, issue - 144
    if load_start < 0 or load_start + 144 > issue or pv_start + 144 > issue:
        raise ValueError('Periodic baseline must be observed before its own issue')
    return np.column_stack((data.actual[load_start:load_start + 144, 0],
                            data.actual[pv_start:pv_start + 144, 1]))


def historical_labels(data, days, cutoff):
    days = np.asarray(days, int)
    if np.any((days + 1) * 144 > cutoff):
        raise ValueError('Label interval is not complete at its fitting cutoff')
    return np.stack([data.actual[day * 144:(day + 1) * 144, :2] for day in days])


def fit_month(data, month, features, protocol):
    train, validation, formal, cutoff = split_days(month)
    asof = int(formal[0])
    assert len(validation) == 7 and train[-1] < validation[0] and validation[-1] < asof
    xtrain = features[train - 7].reshape(-1, len(protocol['features']))
    xval = features[validation - 7].reshape(-1, len(protocol['features']))
    ytrain = historical_labels(data, train, cutoff * 144).reshape(-1, 2)
    yval = historical_labels(data, validation, asof * 144).reshape(-1, 2)
    train_bases = np.stack([periodic_base(data, int(day)) for day in train]).reshape(-1, 2)
    validation_bases = np.stack([periodic_base(data, int(day)) for day in validation]).reshape(-1, 2)
    ytrain = ytrain - train_bases
    yval = yval - validation_bases
    weights = np.repeat(age_weights(train, cutoff, 90), 144)
    models, records = [], []
    with threadpool_limits(limits=1):
        for channel, name in enumerate(('load', 'pv')):
            model = HistGradientBoostingRegressor(**protocol['model_configuration'])
            began = perf_counter()
            model.fit(xtrain, ytrain[:, channel], sample_weight=weights,
                      X_val=xval, y_val=yval[:, channel])
            models.append(model)
            predicted = model.predict(xval)
            records.append({'channel': name, 'iterations': int(model.n_iter_),
                'training_seconds': perf_counter() - began,
                'training_score_by_iteration': model.train_score_.tolist(),
                'validation_score_by_iteration': model.validation_score_.tolist(),
                'validation_rmse_kw_before_physical_postprocessing': float(np.sqrt(np.mean((predicted - yval[:, channel])**2)))})
    return models, {'month': month, 'asof_day': asof, 'training_days': train.tolist(),
        'validation_days': validation.tolist(), 'formal_days': formal.tolist(),
        'train_label_stop_exclusive': cutoff * 144, 'validation_label_stop_exclusive': asof * 144,
        'model_information_cutoff_exclusive': asof * 144,
        'feature_rule': 'each training/validation/formal feature uses actuals only before its own daily issue',
        'target_semantics': 'load minus last-week actual and PV minus yesterday actual; invariant across January and all formal months',
        'periodic_training_base_sha256': array_hash(train_bases),
        'periodic_validation_base_sha256': array_hash(validation_bases),
        'periodic_or_CNN_reference_columns': False, 'validation_rows': len(xval),
        'training_rows': len(xtrain), 'training_feature_hash': array_hash(xtrain),
        'validation_feature_hash': array_hash(xval), 'training_label_hash': array_hash(ytrain),
        'validation_label_hash': array_hash(yval), 'models': records}


def predict_day(data, day, models):
    x, _ = features_for_day(data, day)
    with threadpool_limits(limits=1):
        values = np.column_stack([model.predict(x) for model in models])
    values = np.maximum(values + periodic_base(data, day), 0.)
    values[:, 1] *= _daylight(data.actual, day)
    return values


class ArrayStore:
    def __init__(self, values, origins, name):
        self.values, self.origins = np.asarray(values).copy(), np.asarray(origins).copy()
        self.name = name
        self.lookup = {int(origin): i for i, origin in enumerate(self.origins)}

    def get(self, origin):
        return self.values[self.lookup[int(origin)]].copy()


class PeriodicResidualHGBStore(ArrayStore):
    def __init__(self, stage=PRIMARY, directory=OUT):
        directory = Path(directory)
        meta = json.loads((directory / 'provenance.json').read_text())
        path = directory / f'{stage}.npz'
        if not meta['complete'] or digest(path) != meta['archives'][stage]:
            raise ValueError('Periodic residual HGB forecast archive is incomplete or changed')
        with np.load(path) as z:
            super().__init__(z['values'], z['origins'], stage)


def postprocess(data, values, origins):
    raw = ArrayStore(values, origins, 'periodic_residual_hgb_raw')
    ridge = CalibratedStore('ridge_28', seed=42, data=data, use_cache=False,
                            directory=OUT / 'calibration_recomputed', _base_store=raw)
    memory = EnergyMemoryStore(data=data, base_store=ridge)
    ridge.name, memory.name = 'periodic_residual_hgb_ridge28', PRIMARY
    for audit in memory.audit:
        audit['underlying_forecast'] = 'periodic_load_pv_residual_HGB_with_same_causal_base_then_same_fixed_Ridge28'
    save(OUT / 'ridge28_audit.json', ridge.audit)
    save(OUT / 'memory_half_audit.json', memory.audit)
    return {raw.name: raw, ridge.name: ridge, memory.name: memory}


def causality_checks(data, models_by_month, raw, processed, features, protocol):
    future = []
    for day in (31, 32, 59, 90, 151, 243, 364):
        changed = copy.copy(data)
        changed.actual = data.actual.copy()
        changed.actual[day * 144:] += np.array([50000., 30000.])
        a, _ = features_for_day(data, day)
        b, _ = features_for_day(changed, day)
        np.testing.assert_array_equal(a, b)
        month = int((pd.Timestamp('2025-01-01') + pd.Timedelta(days=day)).month)
        before, after = predict_day(data, day, models_by_month[month]), predict_day(changed, day, models_by_month[month])
        np.testing.assert_array_equal(before, after)
        index = day - 31
        future_raw = raw.values.copy()
        future_raw[index + 1:] += 70000.
        ridge0, _ = _calibrate(data.actual, raw.origins, raw.values, index, CONFIG['ridge_28'])
        ridge1, _ = _calibrate(changed.actual, raw.origins, future_raw, index, CONFIG['ridge_28'])
        np.testing.assert_array_equal(ridge0, ridge1)
        future_ridge = processed['periodic_residual_hgb_ridge28'].values.copy()
        future_ridge[index + 1:] += 80000.
        memory0, _ = correct_day(data.actual, raw.origins, processed['periodic_residual_hgb_ridge28'].values, index)
        memory1, _ = correct_day(changed.actual, raw.origins, future_ridge, index)
        np.testing.assert_array_equal(memory0, memory1)
        future.append({'day': day, 'origin': day * 144, 'features_identical': True,
            'raw_model_output_with_daylight_mask_identical': True, 'ridge28_identical': True,
            'fixed_memory_identical': True, 'changed_current_and_future_actual_and_future_forecasts': True})
    retraining = []
    for month in (2, 9):
        train, validation, formal, cutoff = split_days(month)
        issue = int(formal[0]) * 144
        changed = copy.copy(data)
        changed.actual = data.actual.copy()
        changed.actual[issue:] += np.array([90000., 50000.])
        x = np.stack([features_for_day(changed, day)[0] for day in range(7, 365)])
        for selected in (train, validation):
            np.testing.assert_array_equal(features[selected - 7], x[selected - 7])
        fresh, audit = fit_month(changed, month, x, protocol)
        a = predict_day(data, int(formal[0]), models_by_month[month])
        b = predict_day(changed, int(formal[0]), fresh)
        np.testing.assert_array_equal(a, b)
        retraining.append({'month': month, 'future_actual_mutation_from': issue,
            'full_train_and_validation_features_unchanged': True,
            'retrained_model_first_day_predictions_identical': True,
            'iterations': [r['iterations'] for r in audit['models']]})
    result = {'passed': True, 'future_actual_and_future_prediction_checks': future,
        'monthly_full_retraining_checks': retraining,
        'official_PV_not_loaded_by_model': True, 'CNN_feature_not_used': True, 'periodic_baseline_only_in_labels_and_reconstruction': True,
        'all_postprocessing_labels_past': all(r['history_last_label'] is None or
            r['history_last_label'] < r['origin'] for r in processed['periodic_residual_hgb_ridge28'].audit)
            and all(r['prior_label_end_exclusive'] is None or r['prior_label_end_exclusive'] <= r['origin']
                    for r in processed[PRIMARY].audit)}
    assert result['all_postprocessing_labels_past']
    save(OUT / 'causality_verification.json', result)
    return result


def score_all(data, values, origins):
    truth = data.actual[origins[:, None] + np.arange(144), :2]
    dates = pd.date_range('2025-02-01', '2025-12-31')
    annual, monthly, daily = [], [], []
    for name, value in values.items():
        for channel, error in [('load', value[:, :, 0] - truth[:, :, 0]),
                               ('pv', value[:, :, 1] - truth[:, :, 1]),
                               ('net', (value[:, :, 0] - value[:, :, 1]) - (truth[:, :, 0] - truth[:, :, 1]))]:
            annual.append({'name': name, 'channel': channel, **error_summary(error, data.fixed_price)})
            for month in range(2, 13):
                ids = dates.month == month
                monthly.append({'name': name, 'channel': channel, 'month': month,
                                **error_summary(error[ids], data.fixed_price)})
            if channel == 'net':
                for i, date in enumerate(dates):
                    daily.append({'name': name, 'date': str(date.date()), 'day': i + 31,
                                  **error_summary(error[i:i + 1], data.fixed_price)})
    for name, rows in [('annual_metrics', annual), ('monthly_metrics', monthly), ('daily_metrics', daily)]:
        pd.DataFrame(rows).to_csv(OUT / f'{name}.csv', index=False)
    return annual


def bridge():
    from experiments.exp008.risk_window import replay, SPEC
    from experiments.exp008.verify import verify_npz
    store = PeriodicResidualHGBStore()
    data = Data()
    directory = OUT / 'lp_bridge'
    case = 'periodic_residual_hgb_ridge28_memory_tree28_q08_buffer500'
    result = replay(case, 28, 'tree', 334, data, store, directory)
    case_dir = directory / f'{case}_334days'
    audit = json.loads((case_dir / 'audit.json').read_text())
    for row in audit:
        row['forecast_calibration'] = store.name
        row['residual_source'] = 'periodic_baseline' if row['fallback'] else f'{store.name}_prequential_forecast'
    save(case_dir / 'audit.json', audit)
    check = verify_npz(case_dir / 'dispatch_2.npz', audit_path=case_dir / 'audit.json')
    assert check['passed'], check['errors']
    result.update(fixed_spec={**SPEC, 'calibration': store.name},
                  evaluation_role='fixed forecast-only LP bridge; not final action-constrained policy',
                  forecast_values_sha256=array_hash(store.values))
    save(case_dir / 'summary.json', result)
    save(case_dir / 'independent_verification.json', check)
    return result


def run():
    protocol = json.loads((OUT / 'protocol.json').read_text())
    if (OUT / 'provenance.json').exists():
        raise FileExistsError('This fixed candidate is already complete; no refit or parameter search')
    for name, expected in protocol['source_before_training'].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f'Protocol source changed before training: {name}')
    for name in [__file__, *[str(ROOT / name) for name in protocol['source_before_training']]]:
        target = OUT / 'source_archive' / Path(name).name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(name, target)
    started = perf_counter()
    data = Data()
    features = np.stack([features_for_day(data, day)[0] for day in range(7, 365)])
    assert features.shape == (358, 144, 31) and np.isfinite(features).all()
    absolute_meta = json.loads((ROOT / 'data/results/exp008/forecast_absolute_hgb/provenance.json').read_text())
    assert array_hash(features) == absolute_meta['feature_tensor_sha256']
    values = np.empty((334, 144, 2))
    models_by_month, training, predictions = {}, [], []
    for month in range(2, 13):
        models, audit = fit_month(data, month, features, protocol)
        models_by_month[month] = models
        for channel, model in enumerate(models):
            path = OUT / 'models' / f'month{month:02d}_{("load", "pv")[channel]}_seed42.joblib'
            path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, path)
            reloaded = joblib.load(path)
            x = features[audit['formal_days'][0] - 7]
            with threadpool_limits(limits=1):
                np.testing.assert_array_equal(model.predict(x), reloaded.predict(x))
            audit['models'][channel].update(model_path=str(path), model_sha256=digest(path), reload_identical=True)
        training.append(audit)
        for day in audit['formal_days']:
            values[day - 31] = predict_day(data, day, models)
            predictions.append({'day': day, 'origin': day * 144, 'month_model': month,
                'training_label_stop': audit['train_label_stop_exclusive'],
                'validation_label_stop': audit['validation_label_stop_exclusive'],
                'feature_last_actual_index': day * 144 - 1, 'daylight_last_actual_index': day * 144 - 1,
                'raw_prediction_sha256': array_hash(values[day - 31]),
                'no_CNN_official_PV_future_price_input': True, 'periodic_base_latest_actual_index': day * 144 - 1,
                'periodic_base_sha256': array_hash(periodic_base(data, day))})
        print(json.dumps({'month': month, 'training_days': len(audit['training_days']),
            'validation_days': len(audit['validation_days']), 'iterations': [r['iterations'] for r in audit['models']]}), flush=True)
    origins = np.arange(31, 365) * 144
    processed = postprocess(data, values, origins)
    raw = processed['periodic_residual_hgb_raw']
    verified = causality_checks(data, models_by_month, raw, processed, features, protocol)
    comparison = {'joint_raw': JointStore().values, 'joint_ridge28': JointStore(calibrated=True).values,
                  'joint_ridge28_load_energy_memory_half': EnergyMemoryStore(data=data).values}
    comparison.update({name: AbsoluteHGBStore(stage=name).values for name in
        ('direct_hgb_raw', 'direct_hgb_ridge28', 'direct_hgb_ridge28_memory_half')})
    comparison.update({name: store.values for name, store in processed.items()})
    metrics = score_all(data, comparison, origins)
    archives = {}
    truth = data.actual[origins[:, None] + np.arange(144), :2]
    for name, store in processed.items():
        path = OUT / f'{name}.npz'
        np.savez_compressed(path, values=store.values, origins=origins, errors_kw=truth - store.values)
        archives[name] = digest(path)
    save(OUT / 'training_audit.json', training)
    save(OUT / 'prediction_audit.json', predictions)
    prior = next(r for r in metrics if r['name'] == COMPARATOR and r['channel'] == 'net')
    candidate = next(r for r in metrics if r['name'] == PRIMARY and r['channel'] == 'net')
    gate_metrics = [key for group in ('net', 'high_price', 'daily_accumulation') for key in protocol['dispatch_gate'][group]]
    checks = {key: candidate[key] < prior[key] for key in gate_metrics}
    gate = all(checks.values()) and verified['passed']
    save(OUT / 'dispatch_gate.json', {'passed': gate, 'primary_candidate': PRIMARY,
        'primary_comparator': COMPARATOR, 'metric_checks': checks, 'candidate_metrics': candidate,
        'comparator_metrics': prior, 'model_or_postprocessing_selected_by_formal_scores': False})
    provenance = {'complete': True, 'seed': 42, 'days': 334, 'fitted_monthly_estimators': 22,
        'source_sha256': digest(__file__), 'protocol_sha256': digest(OUT / 'protocol.json'),
        'source_data_sha256': data.hashes, 'feature_names': protocol['features'],
        'feature_tensor_sha256': array_hash(features), 'archives': archives,
        'primary': PRIMARY, 'comparison': COMPARATOR, 'dispatch_gate_passed': gate,
        'sklearn_version': sklearn.__version__, 'numpy_version': np.__version__,
        'total_seconds_before_bridge': perf_counter() - started,
        'training_and_predictions_causally_verified': verified['passed'],
        'comparison_inputs': {str(RESULT): digest(RESULT) for RESULT in
            [ROOT / 'data/results/exp008/neural_joint/predictions.npz',
             ROOT / 'data/results/exp008/neural_joint_calibration/joint_ridge28.npz',
             ROOT / 'data/results/exp008/load_energy_memory/predictions.npz',
             ROOT / 'data/results/exp008/forecast_absolute_hgb/direct_hgb_ridge28_memory_half.npz']},
        'development_not_independent_test': True}
    save(OUT / 'provenance.json', provenance)
    linked = bridge() if gate else None
    save(OUT / 'summary.json', {'predictive_gate_passed': gate, 'bridge_performed': linked is not None,
        'primary_metrics': candidate, 'baseline_metrics': prior, 'metric_checks': checks,
        'bridge_total_cost': linked['total_cost'] if linked else None,
        'final_model_selected': False, 'final_report_generated': False})
    print(json.dumps({'gate': gate, 'candidate': candidate, 'baseline': prior, 'bridge_performed': linked is not None}), flush=True)


if __name__ == '__main__':
    run()
