"""One fixed ExtraTrees absolute load/PV family with causal monthly fitting.

Shares all 31 historical/calendar features and train/validation days with the
original absolute HGB. This is a different model family, not a tuning-isolation
claim. Last-seven-day validation remains held out; it selects no parameter.
"""
from __future__ import annotations

import copy
import gc
import json
from pathlib import Path
import shutil
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import ExtraTreesRegressor
from threadpoolctl import threadpool_limits

from experiments.exp008.forecast_absolute_hgb import (
    OUT as BASE, AbsoluteHGBStore, ArrayStore, features_for_day, historical_labels,
    digest, array_hash, save,
)
from experiments.exp008.forecast_calibration import CalibratedStore, _calibrate, _daylight, CONFIG
from experiments.exp008.load_energy_memory import EnergyMemoryStore, correct_day
from experiments.exp008.forecast_shape_diagnostic import summary as error_summary
from experiments.problem2.exp003.data import Data, ROOT
from experiments.problem2.exp004.data import split_days, age_weights

OUT = ROOT / 'data/results/exp008/forecast_absolute_extra_trees'
COMPOSITE = ROOT / 'data/results/exp008/forecast_channel_composite'
RAW = 'absolute_extra_trees_raw'
RIDGE = 'absolute_extra_trees_ridge28'
PRIMARY = 'absolute_extra_trees_ridge28_memory_half'
GATE_KEYS = ('rmse_kw', 'high_price_rmse_kw', 'daily_energy_rmse_kwh', 'cumulative_error_rmse_kwh')
NUMERIC_ATOL_KW = 1e-8
MODEL_CONFIG = dict(n_estimators=128, min_samples_leaf=10, max_features=1.,
                    bootstrap=False, random_state=42, n_jobs=2,
                    criterion='squared_error')


def prepare_protocol():
    if OUT.exists():
        raise FileExistsError('One fixed candidate: do not overwrite or refit existing evidence')
    baseline_protocol = json.loads((BASE / 'protocol.json').read_text())
    sources = ['experiments/exp008/forecast_absolute_extra_trees.py',
        'experiments/exp008/forecast_absolute_hgb.py', 'experiments/exp008/forecast_net_hgb.py',
        'experiments/exp008/forecast_calibration.py', 'experiments/exp008/load_energy_memory.py',
        'experiments/exp008/forecast_shape_diagnostic.py', 'experiments/problem2/exp004/data.py',
        'experiments/problem2/exp003/data.py', 'experiments/exp008/audited_store_bridge.py',
        'experiments/exp008/risk_window.py']
    inputs = [BASE / p for p in ('protocol.json', 'provenance.json', 'training_audit.json',
        'causality_verification.json', 'direct_hgb_raw.npz', 'direct_hgb_ridge28_memory_half.npz')]
    inputs += [COMPOSITE / p for p in ('protocol.json', 'summary.json', 'memory.npz')]
    protocol = {'candidate_count': 1, 'model_id': PRIMARY,
        'family': 'sklearn ExtraTreesRegressor; independently fitted absolute load and PV',
        'prior_search': {'pattern': 'ExtraTreesRegressor|extra[_ -]?trees|extratrees',
            'command': 'rg -n -i PATTERN experiments reports pyproject.toml',
            'performed_before_creating_this_module': True, 'matched_existing_experiments': False},
        'not_a_single_hyperparameter_or_tuning_isolation_claim': True,
        'features': baseline_protocol['features'], 'feature_count': 31,
        'feature_reference': 'exact original absolute HGB features_for_day tensor and feature order',
        'target': 'absolute actual load and PV kW, same semantics in January and every formal month',
        'model_configuration': MODEL_CONFIG, 'monthly_models': 22,
        'training': 'same split_days(month): all complete days from day7 through month_start-8; no current-month fitting',
        'validation': 'same preceding seven complete days, held out; metrics only, no early stopping, model selection or refit',
        'sample_weights': 'same age_weights(training_days, train_cutoff, half_life=90), repeated for all144 daily slots',
        'sample_weight_supported_by_fit': True,
        'features_do_not_include': ['CNN or periodic forecast reference', 'official PV forecast',
            'future actual load/PV/price', 'future-year shape', 'holiday region assumptions'],
        'raw_postprocessing': 'both channels nonnegative; PV masked by same past28day positive-PV union expanded20minutes each side',
        'calibration': 'same unmodified joint-channel Ridge28 fitted on this raw family, then same nonrecursive .5 underlying-load daily memory',
        'predictive_gate': {'keys': list(GATE_KEYS), 'primary_comparator': 'original absolute HGB complete pipeline',
            'secondary_comparator': 'best fixed raw-channel composite complete pipeline',
            'rule': 'all four complete-pipeline metrics strictly improve on original HGB and all audits pass'},
        'conditional_bridge': 'only after predictive gate: one full334 tree28/q.8/state_buffer500 using own issued residuals, no reused risk cache',
        'full_physical_planning': 'not authorized by this experiment; positive LP only informs parent',
        'checks': {'selected_daily_future_mutations': [31,32,59,90,151,243,364],
            'full_monthly_retraining_future_mutations': [2,9], 'reload_all22_models_all_formal_predictions': True,
            'floating_tolerance_kw': NUMERIC_ATOL_KW,
            'tolerance_reason': 'n_jobs2 parallel prediction accumulation can change floating summation order; tree structures and parameters must match exactly'},
        'development_on_examined2025_not_independent_test': True,
        'model_or_parameters_selected_by_formal_truth': False,
        'source_sha256': {p: digest(ROOT / p) for p in sources},
        'input_artifact_sha256': {str(p): digest(p) for p in inputs},
        'source_data_sha256': Data().hashes}
    OUT.mkdir(parents=True)
    save(OUT / 'protocol.json', protocol)
    for name in sources:
        target = OUT / 'source_archive' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    return protocol


def tree_hashes(model):
    # Contents, not joblib byte layout or parallel prediction addition order.
    # Structured dtype padding bytes are not model data and may differ after
    # pickle reload. Hash each named field, preserving all meaningful nodes.
    return [{'node_fields': {name: array_hash(tree.tree_.__getstate__()['nodes'][name])
                            for name in tree.tree_.__getstate__()['nodes'].dtype.names},
             'values': array_hash(tree.tree_.__getstate__()['values']),
             'random_state': int(tree.random_state)} for tree in model.estimators_]


def predict_models(models, features):
    with threadpool_limits(limits=1):
        return np.column_stack([m.predict(features) for m in models])


def physical_output(data, day, values):
    values = np.maximum(values, 0.)
    values[:, 1] *= _daylight(data.actual, day)
    return values


def predict_day(data, day, models):
    x, _ = features_for_day(data, day)
    return physical_output(data, day, predict_models(models, x))


def fit_month(data, month, features):
    train, validation, formal, cutoff = split_days(month)
    asof = int(formal[0])
    assert len(validation) == 7 and train[-1] < validation[0] and validation[-1] < asof
    xtrain = features[train - 7].reshape(-1, 31)
    xval = features[validation - 7].reshape(-1, 31)
    ytrain = historical_labels(data, train, cutoff * 144).reshape(-1, 2)
    yval = historical_labels(data, validation, asof * 144).reshape(-1, 2)
    weights = np.repeat(age_weights(train, cutoff, 90), 144)
    models, records = [], []
    with threadpool_limits(limits=1):
        for channel, name in enumerate(('load', 'pv')):
            model = ExtraTreesRegressor(**MODEL_CONFIG)
            began = perf_counter()
            model.fit(xtrain, ytrain[:, channel], sample_weight=weights)
            predicted = model.predict(xval)
            records.append({'channel': name, 'training_seconds': perf_counter() - began,
                'trees': len(model.estimators_), 'node_count': sum(t.tree_.node_count for t in model.estimators_),
                'validation_rmse_kw_before_physical_postprocessing': float(np.sqrt(np.mean((predicted-yval[:, channel])**2))),
                'feature_importance': model.feature_importances_.tolist(),
                'tree_hashes': tree_hashes(model), 'resolved_model_parameters': model.get_params()})
            models.append(model)
    audit = {'month': month, 'asof_day': asof, 'training_days': train.tolist(),
        'validation_days': validation.tolist(), 'formal_days': formal.tolist(),
        'train_label_stop_exclusive': cutoff * 144, 'validation_label_stop_exclusive': asof * 144,
        'model_information_cutoff_exclusive': asof * 144,
        'training_feature_hash': array_hash(xtrain), 'validation_feature_hash': array_hash(xval),
        'training_label_hash': array_hash(ytrain), 'validation_label_hash': array_hash(yval),
        'training_weights_hash': array_hash(weights), 'training_rows': len(xtrain),
        'validation_rows': len(xval), 'validation_labels_used_for_fitting_or_model_selection': False,
        'models': records}
    return models, audit


def postprocess(data, values, origins):
    raw = ArrayStore(values, origins, RAW)
    ridge = CalibratedStore('ridge_28', data=data, use_cache=False, _base_store=raw,
                            directory=OUT / 'unused_calibration_cache')
    memory = EnergyMemoryStore(data=data, base_store=ridge)
    ridge.name, memory.name = RIDGE, PRIMARY
    for audit in ridge.audit:
        audit['actual_base_model_id'] = RAW
    for audit in memory.audit:
        audit['underlying_forecast'] = 'absolute_load_PV_ExtraTrees_then_same_Ridge28'
    save(OUT / 'ridge28_audit.json', ridge.audit)
    save(OUT / 'memory_half_audit.json', memory.audit)
    return {RAW: raw, RIDGE: ridge, PRIMARY: memory}


class AbsoluteExtraTreesStore(ArrayStore):
    def __init__(self, stage=PRIMARY):
        meta = json.loads((OUT / 'provenance.json').read_text())
        path = OUT / f'{stage}.npz'
        if not meta['complete'] or digest(path) != meta['archives'][stage]:
            raise ValueError('ExtraTrees archive is incomplete or changed')
        with np.load(path) as z:
            super().__init__(z['values'], z['origins'], stage)


def load_models(month):
    return [joblib.load(OUT / 'models' / f'month{month:02d}_{channel}_seed42.joblib')
            for channel in ('load', 'pv')]


def causal_checks(data, raw, processed, features, protocol):
    daily = []
    for day in protocol['checks']['selected_daily_future_mutations']:
        changed = copy.copy(data)
        changed.actual = data.actual.copy()
        changed.actual[day * 144:] += np.array([70000., 50000.])
        x0, _ = features_for_day(data, day)
        x1, _ = features_for_day(changed, day)
        np.testing.assert_array_equal(x0, x1)
        month = int((pd.Timestamp('2025-01-01') + pd.Timedelta(days=day)).month)
        models = load_models(month)
        a, b = predict_day(data, day, models), predict_day(changed, day, models)
        np.testing.assert_allclose(a, b, rtol=0, atol=NUMERIC_ATOL_KW)
        i = day - 31
        future_raw = raw.values.copy()
        future_raw[i + 1:] += 80000.
        r0, _ = _calibrate(data.actual, raw.origins, raw.values, i, CONFIG['ridge_28'])
        r1, _ = _calibrate(changed.actual, raw.origins, future_raw, i, CONFIG['ridge_28'])
        np.testing.assert_array_equal(r0, r1)
        future_ridge = processed[RIDGE].values.copy()
        future_ridge[i + 1:] += 90000.
        m0, _ = correct_day(data.actual, raw.origins, processed[RIDGE].values, i)
        m1, _ = correct_day(changed.actual, raw.origins, future_ridge, i)
        np.testing.assert_array_equal(m0, m1)
        daily.append({'day': day, 'origin': day * 144, 'features_exactly_unchanged': True,
            'raw_prediction_max_change_kw': float(np.max(np.abs(a-b))),
            'daylight_mask_unchanged': bool(np.array_equal(_daylight(data.actual, day), _daylight(changed.actual, day))),
            'ridge_and_memory_exactly_unchanged': True})
        del models
    retraining = []
    for month in protocol['checks']['full_monthly_retraining_future_mutations']:
        train, validation, formal, cutoff = split_days(month)
        issue = int(formal[0]) * 144
        changed = copy.copy(data)
        changed.actual = data.actual.copy()
        changed.actual[issue:] += np.array([90000., 60000.])
        new_features = np.stack([features_for_day(changed, day)[0] for day in range(7, 365)])
        for days in (train, validation):
            np.testing.assert_array_equal(features[days - 7], new_features[days - 7])
        fresh, audit = fit_month(changed, month, new_features)
        saved = load_models(month)
        for a, b in zip(saved, fresh):
            assert a.get_params() == b.get_params()
            assert tree_hashes(a) == tree_hashes(b)
        a = predict_day(data, int(formal[0]), saved)
        b = predict_day(changed, int(formal[0]), fresh)
        np.testing.assert_allclose(a, b, rtol=0, atol=NUMERIC_ATOL_KW)
        retraining.append({'month': month, 'future_mutation_origin': issue,
            'train_and_validation_features_exactly_unchanged': True,
            'retrained_model_parameters_and_all_tree_structures_exactly_identical': True,
            'first_day_prediction_max_change_kw': float(np.max(np.abs(a-b))),
            'training_seconds': sum(m['training_seconds'] for m in audit['models'])})
        print(json.dumps({'causal_retraining_month': month, 'passed': True}), flush=True)
        del fresh, saved, new_features
        gc.collect()
    labels_past = all(r['history_last_label'] is None or r['history_last_label'] < r['origin']
                      for r in processed[RIDGE].audit)
    labels_past &= all(r['prior_label_end_exclusive'] is None or r['prior_label_end_exclusive'] <= r['origin']
                       for r in processed[PRIMARY].audit)
    assert labels_past
    result = {'passed': True, 'selected_daily_checks': daily, 'full_retraining_checks': retraining,
        'all334_postprocessing_labels_past': bool(labels_past), 'floating_tolerance_kw': NUMERIC_ATOL_KW,
        'reference_feature_or_official_PV_used': False}
    save(OUT / 'causality_verification.json', result)
    return result


def metrics(data, stores, origins):
    truth = data.actual[origins[:, None] + np.arange(144), :2]
    dates = pd.date_range('2025-02-01', '2025-12-31')
    annual, monthly, daily = [], [], []
    for name, values in stores.items():
        for channel, error in [('load', values[:, :, 0]-truth[:, :, 0]),
            ('pv', values[:, :, 1]-truth[:, :, 1]),
            ('net', (values[:, :, 0]-values[:, :, 1])-(truth[:, :, 0]-truth[:, :, 1]))]:
            annual.append({'name': name, 'channel': channel, **error_summary(error, data.fixed_price)})
            for month in range(2, 13):
                monthly.append({'name': name, 'channel': channel, 'month': month,
                    **error_summary(error[dates.month == month], data.fixed_price)})
            if channel == 'net':
                for i, date in enumerate(dates):
                    daily.append({'name': name, 'date': str(date.date()), 'day': i+31,
                        **error_summary(error[i:i+1], data.fixed_price)})
    for name, rows in [('annual_metrics', annual), ('monthly_metrics', monthly), ('daily_metrics', daily)]:
        pd.DataFrame(rows).to_csv(OUT / f'{name}.csv', index=False)
    return annual


def source_checks(protocol):
    assert all(digest(ROOT / p) == sha for p, sha in protocol['source_sha256'].items())
    assert all(digest(p) == sha for p, sha in protocol['input_artifact_sha256'].items())


def run():
    protocol = prepare_protocol()
    source_checks(protocol)
    started = perf_counter()
    data = Data()
    features = np.stack([features_for_day(data, day)[0] for day in range(7, 365)])
    base_meta = json.loads((BASE / 'provenance.json').read_text())
    assert features.shape == (358, 144, 31) and np.isfinite(features).all()
    assert array_hash(features) == base_meta['feature_tensor_sha256']
    old_training = {r['month']: r for r in json.loads((BASE / 'training_audit.json').read_text())}
    values = np.empty((334, 144, 2))
    training, predictions = [], []
    for month in range(2, 13):
        models, audit = fit_month(data, month, features)
        for key in ('training_days', 'validation_days', 'formal_days', 'training_feature_hash',
                    'validation_feature_hash', 'training_label_hash', 'validation_label_hash'):
            assert audit[key] == old_training[month][key], key
        audit['exact_historical_features_labels_and_splits_match_original_HGB'] = True
        xformal = features[np.asarray(audit['formal_days']) - 7].reshape(-1, 31)
        raw_prediction = predict_models(models, xformal).reshape(-1, 144, 2)
        for channel, model in enumerate(models):
            path = OUT / 'models' / f'month{month:02d}_{("load", "pv")[channel]}_seed42.joblib'
            path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, path, compress=3)
            reloaded = joblib.load(path)
            assert model.get_params() == reloaded.get_params()
            assert tree_hashes(model) == tree_hashes(reloaded)
            with threadpool_limits(limits=1):
                restored = reloaded.predict(xformal).reshape(-1, 144)
            difference = float(np.max(np.abs(restored-raw_prediction[:, :, channel])))
            np.testing.assert_allclose(restored, raw_prediction[:, :, channel], rtol=0, atol=NUMERIC_ATOL_KW)
            audit['models'][channel].update(model_path=str(path), model_sha256=digest(path),
                reload_tree_parameters_exact=True, reload_all_formal_prediction_max_difference_kw=difference)
            del reloaded
        for i, day in enumerate(audit['formal_days']):
            values[day-31] = physical_output(data, day, raw_prediction[i])
            predictions.append({'day': day, 'origin': day*144, 'monthly_model': month,
                'train_label_stop_exclusive': audit['train_label_stop_exclusive'],
                'validation_label_stop_exclusive': audit['validation_label_stop_exclusive'],
                'feature_last_actual_index': day*144-1, 'daylight_last_actual_index': day*144-1,
                'raw_prediction_sha256': array_hash(values[day-31]),
                'negative_raw_load_slots': int(np.sum(raw_prediction[i, :, 0] < 0)),
                'negative_raw_pv_slots': int(np.sum(raw_prediction[i, :, 1] < 0)),
                'positive_raw_pv_outside_causal_daylight_mask_slots': int(np.sum(
                    (raw_prediction[i, :, 1] > 0) & ~_daylight(data.actual, day)))})
        training.append(audit)
        save(OUT / 'training_audit.json', training)
        print(json.dumps({'month': month, 'days': len(audit['training_days']),
            'validation_days': 7, 'training_seconds': sum(m['training_seconds'] for m in audit['models']),
            'saved_models': 2*len(training)}), flush=True)
        del models, model
        gc.collect()
    origins = np.arange(31, 365)*144
    processed = postprocess(data, values, origins)
    causal = causal_checks(data, processed[RAW], processed, features, protocol)
    np.testing.assert_array_equal(origins, AbsoluteHGBStore().origins)
    with np.load(COMPOSITE / 'memory.npz') as z:
        np.testing.assert_array_equal(origins, z['origins'])
        composite_values = z['values'].copy()
    comparison = {'original_HGB_raw': AbsoluteHGBStore(stage='direct_hgb_raw').values,
        'original_HGB_full': AbsoluteHGBStore().values, 'best_fixed_channel_composite_full': composite_values,
        **{name: store.values for name, store in processed.items()}}
    scores = metrics(data, comparison, origins)
    candidate, baseline, composite = [next(r for r in scores if r['name'] == name and r['channel'] == 'net')
        for name in (PRIMARY, 'original_HGB_full', 'best_fixed_channel_composite_full')]
    checks = {k: candidate[k] < baseline[k] for k in GATE_KEYS}
    gate = all(checks.values()) and causal['passed']
    truth = data.actual[origins[:, None]+np.arange(144), :2]
    archives = {}
    for name, store in processed.items():
        path = OUT / f'{name}.npz'
        np.savez_compressed(path, values=store.values, origins=origins, errors_kw=truth-store.values)
        archives[name] = digest(path)
    save(OUT / 'prediction_audit.json', predictions)
    source_checks(protocol)
    save(OUT / 'dispatch_gate.json', {'passed': gate, 'metric_checks_vs_original_HGB': checks,
        'candidate': candidate, 'original_HGB': baseline, 'best_fixed_channel_composite': composite,
        'metric_checks_vs_fixed_composite': {k: candidate[k] < composite[k] for k in GATE_KEYS},
        'formal_scores_are_development_results_not_independent_test': True})
    provenance = {'complete': True, 'days': 334, 'primary': PRIMARY, 'archives': archives,
        'source_sha256': digest(__file__), 'protocol_sha256': digest(OUT / 'protocol.json'),
        'source_data_sha256': data.hashes, 'feature_tensor_sha256': array_hash(features),
        'exact_original_HGB_feature_tensor_match': True, 'models': 22,
        'source_and_input_hashes_unchanged': True, 'causality_verified': causal['passed'],
        'all_monthly_models_reloaded_all_formal_predictions_verified': True,
        'sklearn_version': sklearn.__version__, 'numpy_version': np.__version__,
        'predictive_gate_passed': gate, 'elapsed_seconds_before_bridge': perf_counter()-started}
    save(OUT / 'provenance.json', provenance)
    summary = {'complete': True, 'model_id': PRIMARY, 'predictive_gate_passed': gate,
        'metrics': scores, 'gate_checks': checks, 'bridge_performed': False,
        'full_physical_performed': False, 'final_model_selected': False,
        'source_hashes_verified': True, 'audit_passed': True}
    save(OUT / 'summary.json', summary)
    print(json.dumps({'gate': gate, 'candidate': candidate, 'original_HGB': baseline,
        'fixed_composite': composite}), flush=True)
    if gate:
        from experiments.exp008.audited_store_bridge import run as bridge
        bridge(AbsoluteExtraTreesStore(), OUT / 'lp_bridge',
            [OUT / 'protocol.json', OUT / 'provenance.json', OUT / f'{PRIMARY}.npz',
             OUT / 'causality_verification.json', OUT / 'dispatch_gate.json'])
        summary['bridge_performed'] = True
        save(OUT / 'summary.json', summary)


if __name__ == '__main__':
    run()
