"""One fixed weekly-refit ablation of the existing 4850-parameter joint CNN.

Architecture, symmetric component/net loss, training hyperparameters and
causal postprocessing are unchanged. Only the refit/forecast block changes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import pandas as pd

from experiments.exp008 import neural_joint as joint
from experiments.exp008.forecast import Forecasts
from experiments.exp008.forecast_calibration import CONFIG as CALIBRATION_CONFIG, CalibratedStore
from experiments.exp008.load_energy_memory import EnergyMemoryStore
from experiments.exp008.neural_joint_calibration import JointStore
from experiments.problem2.exp003.data import Data, ROOT
from experiments.problem2.exp004.data import Features, age_weights, protocol, sha256, write_json

OUT = ROOT/'data/results/exp008/neural_joint_weekly'
ASOFS = tuple(range(31, 365, 7))
GATE_METRICS = ('net_rmse_kw', 'net_mae_kw', 'price_weighted_net_rmse_kw',
                'high_price_net_rmse_kw', 'high_price_net_mae_kw',
                'daily_net_energy_rmse_kwh', 'daily_net_energy_mae_kwh',
                'intraday_cumulative_error_rmse_kwh')


def recorded_protocol(data):
    sources = ('experiments/exp008/neural_joint_weekly.py',
               'experiments/exp008/neural_joint.py',
               'experiments/problem2/exp004/data.py',
               'experiments/problem2/exp004/protocol.json',
               'experiments/problem2/exp003/data.py',
               'experiments/exp008/forecast_calibration.py',
               'experiments/exp008/load_energy_memory.py')
    config = {'candidate_count': 1, 'frequency_grid': False,
        'refit_every_complete_days': 7, 'refit_asof_days': ASOFS,
        'forecast_blocks': [[day, min(day+7, 365)] for day in ASOFS],
        'models': len(ASOFS), 'final_block_days': 365-ASOFS[-1],
        'architecture_and_effective_loss': joint.CANDIDATE,
        'parameters': 4850, 'effective_loss_override': joint.CANDIDATE['loss'],
        'inherited_training_hyperparameters': protocol()['training'],
        'training_hyperparameter_note': 'inherited asymmetric loss fields are overridden by symmetric JointTariffHuber unchanged from neural_joint',
        'split': 'train days [7,asof-7), validation [asof-7,asof), forecast [asof,min(asof+7,365))',
        'scales': 'mean/component std/net std from actual[:(asof-7)*144], never validation labels',
        'training_from_scratch': True, 'seed': 42, 'no_cumulative_loss': True,
        'postprocessing': 'each archive independently uses unchanged fixed Ridge28 then unchanged fixed0.5 memory',
        'main_comparison': 'weekly->Ridge28->fixed0.5 memory vs monthly Joint->same processing',
        'LP_gate_all_metrics_strictly_lower': GATE_METRICS,
        'official_pv_and_future_price_used': False,
        'development_on_examined_2025_not_independent_test': True,
        'source_sha256': {name: sha256(ROOT/name) for name in sources},
        'source_data_sha256': data.hashes}
    signature = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    OUT.mkdir(exist_ok=True, parents=True)
    path = OUT/'protocol.json'
    payload = json.loads(json.dumps({'signature': signature, **config}))
    if path.exists() and json.loads(path.read_text()) != payload:
        raise RuntimeError('Weekly protocol changed; refuse to reuse a prior experiment')
    write_json(path, payload)
    directory = OUT/'runs'/signature[:16]
    directory.mkdir(exist_ok=True, parents=True)
    for name in sources:
        target = directory/'source_archive'/name
        target.parent.mkdir(exist_ok=True, parents=True)
        target.write_bytes((ROOT/name).read_bytes())
    return signature, directory


def train_block(asof, features, directory, signature):
    stem = directory/f'joint_asof{asof:03d}_s42'
    if stem.with_suffix('.json').exists():
        meta = json.loads(stem.with_suffix('.json').read_text())
        assert meta['signature'] == signature
        assert meta['archive_sha256'] == sha256(stem.with_suffix('.npz'))
        assert meta['weights_sha256'] == sha256(stem.with_suffix('.keras'))
        print('RESUME', asof, flush=True)
        return meta
    cfg = protocol()['training']
    cutoff = asof-7
    train, validation, formal = np.arange(7, cutoff), np.arange(cutoff, asof), np.arange(asof, min(asof+7, 365))
    days = np.r_[train, validation, formal]
    nt, nv = len(train), len(validation)
    pack = features.arrays(days, cutoff, 'no_season')
    actual = features.data.actual
    labels = actual[:asof*144].reshape(-1, 144, 2)[days[:nt+nv]]
    target = ((labels-pack['base'][:nt+nv])/pack['scale']).astype('float32')
    observed_training = actual[:cutoff*144]
    net_scale = max(1., float((observed_training[:, 0]-observed_training[:, 1]).std()))
    weights = age_weights(train, cutoff, cfg['history_weight_half_life_days'])
    joint.tf.keras.backend.clear_session()
    joint.tf.keras.utils.set_random_seed(42)
    model = joint.build_model(features.data.fixed_price, pack['scale'], net_scale)
    x = [pack['sequence'], pack['context']]
    initial = model([a[:1] for a in x], training=False).numpy()
    assert np.array_equal(initial, np.zeros_like(initial)) and model.count_params() == 4850
    began = perf_counter()
    fit = model.fit(joint.dataset(x[0][:nt], x[1][:nt], target[:nt], weights, shuffle=True),
        validation_data=joint.dataset(x[0][nt:nt+nv], x[1][nt:nt+nv], target[nt:]),
        epochs=cfg['max_epochs'], verbose=0, shuffle=False,
        callbacks=[joint.tf.keras.callbacks.EarlyStopping(monitor='val_loss',
            patience=cfg['patience'], restore_best_weights=True)])
    seconds = perf_counter()-began
    residual = model([a[nt+nv:] for a in x], training=False).numpy().astype(float)*pack['scale']
    prediction = np.maximum(0., pack['base'][nt+nv:]+residual)
    prediction[:, :, 1] *= pack['mask'][nt+nv:]
    assert np.isfinite(prediction).all() and np.any(residual != 0)
    model.save(stem.with_suffix('.keras'))
    restored = joint.tf.keras.models.load_model(stem.with_suffix('.keras'))
    before = model([a[nt+nv:] for a in x], training=False).numpy()
    after = restored([a[nt+nv:] for a in x], training=False).numpy()
    reload_error = float(np.max(np.abs(before-after)))
    assert reload_error <= 1e-6
    pollution = [joint.feature_pollution_check(features, int(day), cutoff, restored)
                 for day in (formal[0], formal[-1])]
    np.testing.assert_array_equal(pack['mean'], observed_training.mean(0))
    np.testing.assert_array_equal(pack['scale'], np.maximum(observed_training.std(0), 1.))
    np.savez_compressed(stem.with_suffix('.npz'), days=formal, predictions=prediction,
        base=pack['base'][nt+nv:], residual=residual, mask=pack['mask'][nt+nv:],
        mean=pack['mean'], scale=pack['scale'], net_scale=np.array(net_scale))
    meta = {'signature': signature, 'asof_day': asof, 'forecast_days': formal.tolist(),
        'train_days': train.tolist(), 'validation_days': validation.tolist(),
        'scaler_cutoff_day_exclusive': cutoff,
        'train_latest_label_exclusive': int((train[-1]+1)*144),
        'validation_latest_label_exclusive': asof*144,
        'model_available_at_origin': asof*144,
        'mean_kw': pack['mean'].tolist(), 'scale_kw': pack['scale'].tolist(),
        'net_scale_kw': net_scale, 'net_scale_label_cutoff_exclusive': cutoff*144,
        'sample_weight_min': float(weights.min()), 'sample_weight_max': float(weights.max()),
        'history_weight_half_life_days': 90, 'parameters': model.count_params(),
        'epochs': len(fit.history['loss']), 'best_epoch': int(np.argmin(fit.history['val_loss']))+1,
        'history': fit.history, 'training_seconds': seconds,
        'save_reload_max_error': reload_error, 'save_reload_passed': True,
        'first_and_last_day_future_mutations': pollution,
        'weights_sha256': sha256(stem.with_suffix('.keras')),
        'archive_sha256': sha256(stem.with_suffix('.npz'))}
    write_json(stem.with_suffix('.json'), meta)
    print('TRAINED', asof, 'days', len(formal), 'epochs', meta['epochs'], 'seconds', round(seconds, 2), flush=True)
    return meta


def score(values, truth, price):
    result = joint.score(values, truth, price)
    net = (values[:, :, 0]-values[:, :, 1])-(truth[:, :, 0]-truth[:, :, 1])
    result['net_mae_kw'] = float(np.abs(net).mean())
    return result


def postprocess_and_score(data, raw, metadata, directory, signature):
    origins = np.arange(31, 365)*144
    weekly = SimpleNamespace(origins=origins, values=raw)
    monthly = JointStore()
    monthly_ridge = CalibratedStore('ridge_28', data=data, use_cache=False, _base_store=monthly)
    weekly_ridge = CalibratedStore('ridge_28', data=data, use_cache=False, _base_store=weekly)
    np.testing.assert_array_equal(monthly_ridge.values, JointStore(calibrated=True).values)
    monthly_memory = EnergyMemoryStore(data=data, base_store=monthly_ridge)
    weekly_memory = EnergyMemoryStore(data=data, base_store=weekly_ridge)
    with np.load(ROOT/'data/results/exp008/load_energy_memory/predictions.npz') as saved:
        np.testing.assert_array_equal(monthly_memory.values, saved['values'])
    stages = {'raw': raw, 'ridge28': weekly_ridge.values, 'memory': weekly_memory.values}
    hashes = {}
    for name, values in stages.items():
        path = OUT/f'{name}_predictions.npz'
        np.savez_compressed(path, origins=origins, values=values)
        hashes[name] = sha256(path)
    write_json(OUT/'ridge28_audit.json', {'config': CALIBRATION_CONFIG['ridge_28'], 'days': weekly_ridge.audit})
    write_json(OUT/'memory_audit.json', weekly_memory.audit)
    for day in (31, 38, 62, 243, 364):
        index = day-31
        changed = SimpleNamespace(actual=data.actual.copy())
        changed.actual[day*144:] += 70000
        altered_base = SimpleNamespace(origins=origins, values=raw.copy())
        altered_base.values[index+1:] += 90000
        # Only re-evaluate this day's calibration; complete-prefix causality is
        # checked from its function and all-day boundary metadata below.
        from experiments.exp008.forecast_calibration import _calibrate
        a, _ = _calibrate(data.actual, origins, raw, index, CALIBRATION_CONFIG['ridge_28'])
        b, _ = _calibrate(changed.actual, origins, altered_base.values, index, CALIBRATION_CONFIG['ridge_28'])
        np.testing.assert_array_equal(a, b)
        changed_ridge = SimpleNamespace(origins=origins, values=weekly_ridge.values.copy())
        changed_ridge.values[index+1:] += 90000
        changed_memory = EnergyMemoryStore(data=changed, base_store=changed_ridge)
        np.testing.assert_array_equal(changed_memory.values[:index+1], weekly_memory.values[:index+1])
    # Full-year future labels first enter this post-generation scoring stage.
    truth = data.actual[origins[:, None]+np.arange(144)]
    dates = pd.date_range('2025-02-01', '2025-12-31')
    predictions = {'monthly_raw': monthly.values, 'weekly_raw': raw,
        'monthly_ridge28': monthly_ridge.values, 'weekly_ridge28': weekly_ridge.values,
        'monthly_ridge28_memory': monthly_memory.values, 'weekly_ridge28_memory': weekly_memory.values}
    annual, monthly_rows, daily = [], [], []
    for name, values in predictions.items():
        annual.append({'name': name, **score(values, truth, data.fixed_price)})
        for month in range(2, 13):
            ids = dates.month == month
            monthly_rows.append({'name': name, 'month': month,
                                 **score(values[ids], truth[ids], data.fixed_price)})
        for i, date in enumerate(dates):
            daily.append({'name': name, 'day': i+31, 'date': str(date.date()),
                          **score(values[i:i+1], truth[i:i+1], data.fixed_price)})
    pd.DataFrame(annual).to_csv(OUT/'annual_metrics.csv', index=False)
    pd.DataFrame(monthly_rows).to_csv(OUT/'monthly_metrics.csv', index=False)
    pd.DataFrame(daily).to_csv(OUT/'daily_metrics.csv', index=False)
    reference, candidate = annual[-2], annual[-1]
    improvements = {name: {'monthly': reference[name], 'weekly': candidate[name],
                          'passed': candidate[name] < reference[name]} for name in GATE_METRICS}
    gate = {'passed': all(row['passed'] for row in improvements.values()),
            'comparisons': improvements, 'failed_action': 'stop_without_LP_or_frequency_search'}
    write_json(OUT/'prediction_gate.json', gate)
    manifest = {'complete': True, 'days': 334, 'models': len(metadata), 'signature': signature,
        'run_directory': str(directory), 'stage_archive_sha256': hashes,
        'all_weights_reload_and_future_feature_checks_passed': all(
            m['save_reload_passed'] and all(p['passed'] for p in m['first_and_last_day_future_mutations']) for m in metadata),
        'all_scales_stop_before_validation': all(m['scaler_cutoff_day_exclusive'] == m['asof_day']-7 for m in metadata),
        'all_train_validation_labels_prior_to_asof': all(m['validation_latest_label_exclusive'] == m['asof_day']*144 for m in metadata),
        'postprocessing_future_mutation_days': [31, 38, 62, 243, 364],
        'monthly_same_postprocessing_archives_exact': True,
        'training_seconds_sum': sum(m['training_seconds'] for m in metadata),
        'prediction_gate': gate['passed'], 'LP_bridge_performed': False,
        'new_frequency_candidates': 1, 'metrics': annual,
        'independent_validation_claimed': False}
    write_json(OUT/'manifest.json', manifest)
    print(json.dumps({'metrics': annual, 'gate': gate, 'training_seconds': manifest['training_seconds_sum']}, indent=2), flush=True)
    return manifest


class WeeklyStore:
    def __init__(self, stage='memory'):
        if stage not in ('raw', 'ridge28', 'memory'):
            raise ValueError('unknown weekly forecast stage')
        meta = json.loads((OUT/'manifest.json').read_text())
        path = OUT/f'{stage}_predictions.npz'
        if not meta['complete'] or sha256(path) != meta['stage_archive_sha256'][stage]:
            raise ValueError('weekly archive incomplete or changed')
        with np.load(path) as saved:
            self.origins, self.values = saved['origins'].copy(), saved['values'].copy()
        self.lookup = {int(origin): i for i, origin in enumerate(self.origins)}
        self.name = f'weekly_joint_{stage}'

    def get(self, origin):
        return self.values[self.lookup[int(origin)]].copy()


class WeeklyForecasts(Forecasts):
    def __init__(self):
        super().__init__()
        self.store = WeeklyStore('memory')
        self.calibration = self.store.name

    def get(self, day, slot=0, scenario='2'):
        result = super().get(day, slot, scenario)
        result['audit'].update(
            base_forecast='exp008_weekly_joint_shared_cnn' if day >= 31 else 'periodic_cold_start',
            output_calibration='same_fixed_Ridge28_then_memory_half' if day >= 31 else None,
            base_architecture_unchanged=day < 31,
            model_asof_day=31+7*((day-31)//7) if day >= 31 else None)
        return result


def run():
    environment = joint.configure_cpu()
    features = Features()
    signature, directory = recorded_protocol(features.data)
    write_json(directory/'environment.json', environment)
    preserved = {name: sha256(ROOT/name) for name in (
        'data/results/exp008/neural_joint/predictions.npz',
        'data/results/exp008/neural_joint_calibration/joint_ridge28.npz',
        'data/results/exp008/load_energy_memory/predictions.npz')}
    metadata = [train_block(day, features, directory, signature) for day in ASOFS]
    days, values = [], []
    for asof in ASOFS:
        with np.load(directory/f'joint_asof{asof:03d}_s42.npz') as saved:
            days.extend(saved['days'].tolist())
            values.append(saved['predictions'])
    np.testing.assert_array_equal(days, np.arange(31, 365))
    raw = np.concatenate(values)
    for name, digest in preserved.items():
        assert sha256(ROOT/name) == digest
    write_json(OUT/'original_archives_unchanged.json', {'passed': True, 'sha256': preserved})
    return postprocess_and_score(features.data, raw, metadata, directory, signature)


if __name__ == '__main__':
    run()
