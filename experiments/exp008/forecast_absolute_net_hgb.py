"""One absolute-net HGB with causal variance-weighted load/PV reconstruction.

Net labels and all 31 features keep the same meaning in January and every
formal month. The load/PV interface is an explicit physical reconstruction
using existing raw absolute load/PV predictions, not two newly fitted models.
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

from experiments.exp008.forecast_absolute_hgb import (
    OUT as BASE, AbsoluteHGBStore, ArrayStore, features_for_day, historical_labels,
    digest, array_hash, save,
)
from experiments.exp008.forecast_calibration import CalibratedStore, _calibrate, _daylight, CONFIG
from experiments.exp008.load_energy_memory import EnergyMemoryStore, correct_day
from experiments.exp008.forecast_shape_diagnostic import summary as error_summary
from experiments.problem2.exp003.data import Data, ROOT
from experiments.problem2.exp004.data import split_days, age_weights

OUT = ROOT / 'data/results/exp008/forecast_absolute_net_hgb'
RAW = 'absolute_net_hgb_projected_raw'
RIDGE = 'absolute_net_hgb_projected_ridge28'
PRIMARY = 'absolute_net_hgb_projected_ridge28_memory_half'
COMPARATOR = 'direct_hgb_ridge28_memory_half'
GATE_KEYS = ('rmse_kw', 'high_price_rmse_kw', 'daily_energy_rmse_kwh', 'cumulative_error_rmse_kwh')


def prepare_protocol():
    if OUT.exists():
        raise FileExistsError('This unique candidate must not overwrite or refit an existing experiment')
    OUT.mkdir(parents=True)
    baseline_protocol = json.loads((BASE / 'protocol.json').read_text())
    old_training = json.loads((BASE / 'training_audit.json').read_text())
    projection = {}
    for old in old_training:
        month = old['month']
        variances = {model['channel']: model['validation_rmse_kw_before_physical_postprocessing'] ** 2
                     for model in old['models']}
        assert variances['load'] > 0 and variances['pv'] > 0
        weight = variances['load'] / (variances['load'] + variances['pv'])
        _, validation, formal, _ = split_days(month)
        assert old['validation_days'] == validation.tolist()
        assert old['validation_label_stop_exclusive'] == int(formal[0]) * 144
        projection[str(month)] = {'load_variance_kw2': variances['load'], 'pv_variance_kw2': variances['pv'],
            'load_delta_weight': weight, 'validation_days': validation.tolist(),
            'validation_label_stop_exclusive': int(formal[0]) * 144,
            'source': 'original absolute load/PV HGB validation RMSE before nonnegativity/daylight postprocessing'}
    source_names = ['experiments/exp008/forecast_absolute_net_hgb.py',
        'experiments/exp008/forecast_absolute_hgb.py', 'experiments/exp008/forecast_net_hgb.py',
        'experiments/exp008/forecast_calibration.py', 'experiments/exp008/load_energy_memory.py',
        'experiments/exp008/forecast_shape_diagnostic.py', 'experiments/problem2/exp004/data.py',
        'experiments/exp008/audited_store_bridge.py']
    inputs = [BASE / name for name in ['protocol.json', 'provenance.json', 'training_audit.json',
        'causality_verification.json', 'direct_hgb_raw.npz', f'{COMPARATOR}.npz']]
    protocol = {'candidate_count': 1, 'model_id': PRIMARY, 'seed': 42,
        'target': 'absolute net demand N=actual_load-actual_PV in kW, unchanged across all months',
        'models_per_month': 1, 'months': list(range(2, 13)), 'features': baseline_protocol['features'],
        'feature_count': 31, 'model_configuration': baseline_protocol['model_configuration'],
        'training': baseline_protocol['training'], 'validation': baseline_protocol['validation'],
        'history_weight_half_life_days': 90, 'official_PV_or_future_actual_price': False,
        'CNN_or_periodic_forecast_reference_features_or_labels': False,
        'load_PV_interface_semantics': 'variance-weighted physically constrained reconstruction around signed existing raw absolute HGB L0/P0; not independently new load/PV estimators',
        'projection_variances_known_before_each_month': projection,
        'projection': 'delta=N-(L0-P0); w=varL/(varL+varP); daylight L=max(L0+w*delta,max(N,0)), P=L-N',
        'night_rule': 'same causal past28day union daylight mask expanded20min; night PV0 and load=max(N,0); count all N<0 night corrections',
        'fixed_postprocessing': ['constrained projection', 'same unmodified Ridge28', 'same nonrecursive underlying-load memory gain0.5'],
        'primary_candidate': PRIMARY, 'primary_comparator': COMPARATOR,
        'dispatch_gate': {'metrics': list(GATE_KEYS), 'rule': 'all four full-pipeline metrics strictly better than original absolute full pipeline and causal tests pass'},
        'raw_net_and_physical_projected_net_reported_separately': True,
        'price_weighted_rmse_is_additional_not_selection_metric': True,
        'conditional_bridge': 'one signed-store tree28/q.8/buffer500 full334 bridge, own historical residuals; only after fixed gate',
        'formal_month_truth_used_to_choose_projection_weight_or_Net_model': False,
        'development_on_examined2025_not_independent_test': True,
        'source_sha256': {name: digest(ROOT / name) for name in source_names},
        'input_artifact_sha256': {str(path): digest(path) for path in inputs}, 'raw_data_sha256': Data().hashes}
    save(OUT / 'protocol.json', protocol)
    for name in source_names:
        target = OUT / 'source_archive' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    for path in inputs:
        target = OUT / 'baseline_evidence' / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    return protocol


def net_labels(data, days, cutoff):
    absolute = historical_labels(data, days, cutoff)
    return absolute[:, :, 0] - absolute[:, :, 1]


def fit_month(data, month, features, protocol):
    train, validation, formal, cutoff = split_days(month)
    asof = int(formal[0])
    x_train = features[train - 7].reshape(-1, 31)
    x_validation = features[validation - 7].reshape(-1, 31)
    y_train = net_labels(data, train, cutoff * 144).ravel()
    y_validation = net_labels(data, validation, asof * 144).ravel()
    weights = np.repeat(age_weights(train, cutoff, 90), 144)
    model = HistGradientBoostingRegressor(**protocol['model_configuration'])
    began = perf_counter()
    with threadpool_limits(limits=1):
        model.fit(x_train, y_train, sample_weight=weights, X_val=x_validation, y_val=y_validation)
        predicted = model.predict(x_validation)
    audit = {'month': month, 'asof_day': asof, 'training_days': train.tolist(),
        'validation_days': validation.tolist(), 'formal_days': formal.tolist(),
        'train_label_stop_exclusive': cutoff * 144, 'validation_label_stop_exclusive': asof * 144,
        'training_feature_sha256': array_hash(x_train), 'validation_feature_sha256': array_hash(x_validation),
        'training_label_sha256': array_hash(y_train), 'validation_label_sha256': array_hash(y_validation),
        'label_semantics': 'absolute net kW throughout all training and validation dates',
        'iterations': int(model.n_iter_), 'training_score_by_iteration': model.train_score_.tolist(),
        'validation_score_by_iteration': model.validation_score_.tolist(),
        'validation_net_rmse_kw': float(np.sqrt(np.mean((predicted - y_validation) ** 2))),
        'training_seconds': perf_counter() - began,
        'projection': protocol['projection_variances_known_before_each_month'][str(month)]}
    return model, audit


def project(net, old_raw, mask, weight):
    net, old_raw, mask = np.asarray(net, float), np.asarray(old_raw, float), np.asarray(mask, bool)
    assert net.shape == (144,) and old_raw.shape == (144, 2) and 0 < weight < 1
    effective = net.copy()
    negative_night = (~mask) & (effective < 0)
    effective[~mask] = np.maximum(effective[~mask], 0.)
    delta = effective - (old_raw[:, 0] - old_raw[:, 1])
    unconstrained_load = old_raw[:, 0] + weight * delta
    lower = np.maximum(effective, 0.)
    load = np.maximum(unconstrained_load, lower)
    pv = load - effective
    load[~mask], pv[~mask] = effective[~mask], 0.
    value = np.column_stack((load, pv))
    assert np.isfinite(value).all() and value.min() >= -1e-10
    np.testing.assert_allclose(load - pv, effective, rtol=0, atol=1e-9)
    return value, {'load_delta_weight': float(weight),
        'night_negative_net_clipped_slots': int(negative_night.sum()),
        'night_negative_net_clipped_kwh': float(-net[negative_night].sum() / 6),
        'daylight_nonnegative_constraint_active_slots': int(np.sum(mask & (unconstrained_load < lower))),
        'daylight_slots': int(mask.sum()), 'net_equality_error_kw': float(np.abs(load - pv - effective).max()),
        'raw_net_sha256': array_hash(net), 'projected_values_sha256': array_hash(value)}


def predict_day(data, day, model, old_raw, weight):
    x, _ = features_for_day(data, day)
    with threadpool_limits(limits=1):
        net = model.predict(x)
    value, audit = project(net, old_raw, _daylight(data.actual, day), weight)
    return net, value, audit


def postprocess(data, values, origins):
    raw = ArrayStore(values, origins, RAW)
    ridge = CalibratedStore('ridge_28', data=data, use_cache=False, _base_store=raw,
                            directory=OUT / 'unused_ridge_cache')
    memory = EnergyMemoryStore(data=data, base_store=ridge)
    ridge.name, memory.name = RIDGE, PRIMARY
    for audit in ridge.audit:
        audit['actual_base_model_id'] = RAW
    for audit in memory.audit:
        audit['underlying_forecast'] = 'absolute_net_HGB_constrained_reconstruction_then_same_Ridge28'
    save(OUT / 'ridge28_audit.json', ridge.audit)
    save(OUT / 'memory_half_audit.json', memory.audit)
    return {RAW: raw, RIDGE: ridge, PRIMARY: memory}


class AbsoluteNetStore(ArrayStore):
    def __init__(self, stage=PRIMARY):
        meta = json.loads((OUT / 'provenance.json').read_text())
        path = OUT / f'{stage}.npz'
        if not meta['complete'] or digest(path) != meta['archives'][stage]:
            raise ValueError('Absolute Net HGB store is incomplete or changed')
        with np.load(path) as z:
            super().__init__(z['values'], z['origins'], stage)


def causal_checks(data, models, original_raw, raw, processed, features, protocol):
    checks = []
    for day in (31, 32, 59, 90, 151, 243, 364):
        changed = copy.copy(data)
        changed.actual = data.actual.copy()
        changed.actual[day * 144:] += np.array([70000., 50000.])
        x0, _ = features_for_day(data, day)
        x1, _ = features_for_day(changed, day)
        np.testing.assert_array_equal(x0, x1)
        month = int((pd.Timestamp('2025-01-01') + pd.Timedelta(days=day)).month)
        weight = protocol['projection_variances_known_before_each_month'][str(month)]['load_delta_weight']
        base = original_raw.get(day * 144)
        n0, p0, _ = predict_day(data, day, models[month], base, weight)
        n1, p1, _ = predict_day(changed, day, models[month], base, weight)
        np.testing.assert_array_equal(n0, n1)
        np.testing.assert_array_equal(p0, p1)
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
        checks.append({'day': day, 'current_future_actual_and_future_predictions_mutated': True,
            'features_Net_projection_Ridge_memory_all_unchanged': True,
            'projection_weight_validation_stop_exclusive': protocol['projection_variances_known_before_each_month'][str(month)]['validation_label_stop_exclusive']})
    retraining = []
    for month in (2, 9):
        train, val, formal, _ = split_days(month)
        day = int(formal[0])
        changed = copy.copy(data)
        changed.actual = data.actual.copy()
        changed.actual[day * 144:] += np.array([90000., 20000.])
        altered_features = np.stack([features_for_day(changed, d)[0] for d in range(7, 365)])
        for ids in (train, val):
            np.testing.assert_array_equal(features[ids - 7], altered_features[ids - 7])
        fresh, audit = fit_month(changed, month, altered_features, protocol)
        w = protocol['projection_variances_known_before_each_month'][str(month)]['load_delta_weight']
        n0, p0, _ = predict_day(data, day, models[month], original_raw.get(day * 144), w)
        n1, p1, _ = predict_day(changed, day, fresh, original_raw.get(day * 144), w)
        np.testing.assert_array_equal(n0, n1)
        np.testing.assert_array_equal(p0, p1)
        retraining.append({'month': month, 'future_mutation_origin': day * 144,
            'full_Net_model_retrained': True, 'first_day_raw_Net_and_projection_identical': True,
            'projection_validation_variances_unchanged': True, 'new_iterations': audit['iterations']})
    result = {'passed': True, 'future_mutation_checks': checks, 'full_retraining_future_mutation_checks': retraining,
        'all_Ridge_labels_past': all(a['history_last_label'] is None or a['history_last_label'] < a['origin'] for a in processed[RIDGE].audit),
        'all_memory_labels_past': all(a['prior_label_end_exclusive'] is None or a['prior_label_end_exclusive'] <= a['origin'] for a in processed[PRIMARY].audit),
        'upstream_original_load_PV_training_evidence': 'copied baseline_evidence/causality_verification.json; original models and their validation weights not refit here'}
    assert result['all_Ridge_labels_past'] and result['all_memory_labels_past']
    save(OUT / 'causality_verification.json', result)
    return result


def score_all(data, raw_net, processed, originals):
    origins = processed[RAW].origins
    truth = data.actual[origins[:, None] + np.arange(144)]
    dates = pd.date_range('2025-02-01', '2025-12-31')
    errors = {}
    for name, value in {**{k: v.values for k, v in processed.items()},
                        **{k: v.values for k, v in originals.items()}}.items():
        for j, channel in enumerate(('load', 'pv', 'net')):
            errors[name, channel] = value[:, :, j] - truth[:, :, j] if j < 2 else (
                value[:, :, 0] - value[:, :, 1]) - (truth[:, :, 0] - truth[:, :, 1])
    errors['absolute_net_HGB_before_projection', 'net'] = raw_net - (truth[:, :, 0] - truth[:, :, 1])
    annual, monthly, daily = [], [], []
    for (name, channel), error in errors.items():
        annual.append({'name': name, 'channel': channel, **error_summary(error, data.fixed_price)})
        for month in range(2, 13):
            ids = dates.month == month
            monthly.append({'name': name, 'channel': channel, 'month': month,
                            **error_summary(error[ids], data.fixed_price)})
        if channel == 'net':
            for i, day in enumerate(range(31, 365)):
                daily.append({'name': name, 'day': day, 'date': str(dates[i].date()),
                              **error_summary(error[i:i + 1], data.fixed_price)})
    for name, values in [('annual_metrics', annual), ('monthly_metrics', monthly), ('daily_metrics', daily)]:
        pd.DataFrame(values).to_csv(OUT / f'{name}.csv', index=False)
    return annual


def run():
    began = perf_counter()
    protocol = prepare_protocol()
    data = Data()
    original_raw = AbsoluteHGBStore('direct_hgb_raw')
    original_full = AbsoluteHGBStore()
    features = np.stack([features_for_day(data, day)[0] for day in range(7, 365)])
    original_meta = json.loads((BASE / 'provenance.json').read_text())
    assert array_hash(features) == original_meta['feature_tensor_sha256']
    models, training, prediction_audit = {}, [], []
    raw_net = np.empty((334, 144))
    values = np.empty((334, 144, 2))
    for month in range(2, 13):
        model, audit = fit_month(data, month, features, protocol)
        models[month] = model
        path = OUT / 'models' / f'month{month:02d}_absolute_net_seed42.joblib'
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, path)
        loaded = joblib.load(path)
        with threadpool_limits(limits=1):
            test_x = features[np.asarray(audit['formal_days']) - 7].reshape(-1, 31)
            np.testing.assert_array_equal(model.predict(test_x), loaded.predict(test_x))
        audit.update(model_path=str(path), model_sha256=digest(path), reload_all_formal_days_identical=True)
        training.append(audit)
        weight = audit['projection']['load_delta_weight']
        for day in audit['formal_days']:
            raw_net[day - 31], values[day - 31], info = predict_day(data, day, model, original_raw.get(day * 144), weight)
            prediction_audit.append({'day': day, 'origin': day * 144, 'month_model': month,
                'train_label_stop_exclusive': audit['train_label_stop_exclusive'],
                'validation_label_stop_exclusive': audit['validation_label_stop_exclusive'],
                'feature_and_daylight_last_actual_index': day * 144 - 1,
                'load_PV_are_constrained_net_reconstruction': True, **info})
        print(json.dumps({'month': month, 'iterations': audit['iterations'], 'projection_load_weight': weight,
                          'training_seconds': audit['training_seconds']}), flush=True)
    origins = original_raw.origins.copy()
    processed = postprocess(data, values, origins)
    verified = causal_checks(data, models, original_raw, processed[RAW], processed, features, protocol)
    metrics = score_all(data, raw_net, processed, {'original_absolute_raw': original_raw, COMPARATOR: original_full})
    save(OUT / 'training_audit.json', training)
    save(OUT / 'prediction_audit.json', prediction_audit)
    np.savez_compressed(OUT / 'raw_absolute_net.npz', origins=origins, net_kw=raw_net)
    truth = data.actual[origins[:, None] + np.arange(144)]
    archives = {}
    for name, store in processed.items():
        path = OUT / f'{name}.npz'
        np.savez_compressed(path, origins=origins, values=store.values, errors_kw=truth - store.values,
                            delta_from_original_raw=store.values - original_raw.values)
        archives[name] = digest(path)
    candidate = next(r for r in metrics if r['name'] == PRIMARY and r['channel'] == 'net')
    baseline = next(r for r in metrics if r['name'] == COMPARATOR and r['channel'] == 'net')
    checks = {key: candidate[key] < baseline[key] for key in GATE_KEYS}
    gate = bool(all(checks.values()) and verified['passed'])
    save(OUT / 'dispatch_gate.json', {'passed': gate, 'candidate': candidate, 'baseline': baseline,
        'checks': checks, 'no_formal_month_model_or_weight_selection': True})
    assert all(digest(ROOT / p) == h for p, h in protocol['source_sha256'].items())
    assert all(digest(p) == h for p, h in protocol['input_artifact_sha256'].items())
    provenance = {'complete': True, 'primary': PRIMARY, 'seed': 42, 'days': 334, 'monthly_models': 11,
        'source_sha256': digest(__file__), 'protocol_sha256': digest(OUT / 'protocol.json'),
        'archives': archives, 'raw_absolute_net_sha256': digest(OUT / 'raw_absolute_net.npz'),
        'source_data_sha256': data.hashes, 'feature_tensor_sha256': array_hash(features),
        'same31_feature_tensor_as_original_absolute_HGB': True,
        'all11_models_reloaded_all_formal_outputs_exact': True, 'causal_checks_passed': True,
        'all_source_and_input_hashes_unchanged': True,
        'prediction_semantics': 'absolute_Net_HGB_then_physical_loadPV_reconstruction_Ridge28_memory',
        'sklearn_version': sklearn.__version__, 'numpy_version': np.__version__,
        'dispatch_gate_passed': gate, 'development_not_independent_test': True}
    save(OUT / 'provenance.json', provenance)
    linked = None
    if gate:
        from experiments.exp008.audited_store_bridge import run as bridge
        evidence = [OUT / name for name in ['protocol.json', 'provenance.json', 'training_audit.json',
            'prediction_audit.json', 'causality_verification.json', f'{PRIMARY}.npz']]
        linked = bridge(AbsoluteNetStore(), OUT / 'lp_bridge', evidence)
    result = {'predictive_gate_passed': gate, 'candidate_metrics': candidate, 'baseline_metrics': baseline,
        'metric_checks': checks, 'bridge_performed': linked is not None,
        'bridge_cost_change_yuan': None if linked is None else linked['cost_change_yuan'],
        'night_negative_net_clipped_slots': sum(r['night_negative_net_clipped_slots'] for r in prediction_audit),
        'night_negative_net_clipped_kwh': sum(r['night_negative_net_clipped_kwh'] for r in prediction_audit),
        'daylight_nonnegative_projection_active_slots': sum(r['daylight_nonnegative_constraint_active_slots'] for r in prediction_audit),
        'elapsed_seconds': perf_counter() - began, 'final_model_selected': False, 'final_report_generated': False}
    save(OUT / 'summary.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    run()
