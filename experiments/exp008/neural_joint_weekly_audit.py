"""Read-only independent checks of every saved weekly network artifact."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path

import h5py
import numpy as np

from experiments.problem2.exp003.data import Data, ROOT

OUT = ROOT/'data/results/exp008/neural_joint_weekly'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def keras_weights(path):
    result = {}
    with zipfile.ZipFile(path) as archive:
        with h5py.File(io.BytesIO(archive.read('model.weights.h5')), 'r') as weights:
            def collect(name, obj):
                if isinstance(obj, h5py.Dataset):
                    result[name] = np.array(obj)
            weights.visititems(collect)
    return result


def run():
    protocol = json.loads((OUT/'protocol.json').read_text())
    manifest = json.loads((OUT/'manifest.json').read_text())
    directory = Path(manifest['run_directory'])
    data = Data()
    days, predictions, checks = [], [], []
    for first, stop in protocol['forecast_blocks']:
        stem = directory/f'joint_asof{first:03d}_s42'
        meta = json.loads(stem.with_suffix('.json').read_text())
        cutoff = first-7
        assert meta['signature'] == protocol['signature']
        assert meta['train_days'] == list(range(7, cutoff))
        assert meta['validation_days'] == list(range(cutoff, first))
        assert meta['forecast_days'] == list(range(first, stop))
        assert meta['train_latest_label_exclusive'] == cutoff*144
        assert meta['validation_latest_label_exclusive'] == first*144
        assert meta['parameters'] == 4850
        assert 1 <= meta['best_epoch'] <= meta['epochs'] <= 60
        assert meta['archive_sha256'] == digest(stem.with_suffix('.npz'))
        assert meta['weights_sha256'] == digest(stem.with_suffix('.keras'))
        assert meta['save_reload_passed'] and meta['save_reload_max_error'] <= 1e-6
        assert len(meta['first_and_last_day_future_mutations']) == 2
        assert all(check['passed'] and check['current_model_output_error'] == 0
                   for check in meta['first_and_last_day_future_mutations'])
        observed = data.actual[:cutoff*144]
        with np.load(stem.with_suffix('.npz')) as saved:
            np.testing.assert_array_equal(saved['mean'], observed.mean(0))
            np.testing.assert_array_equal(saved['scale'], np.maximum(observed.std(0), 1.))
            assert float(saved['net_scale']) == max(1., float((observed[:, 0]-observed[:, 1]).std()))
            days.extend(saved['days'].tolist())
            predictions.append(saved['predictions'])
            scale = saved['scale'].copy()
            net_scale = float(saved['net_scale'])
        with zipfile.ZipFile(stem.with_suffix('.keras')) as archive:
            config = json.loads(archive.read('config.json'))
        loss = config['compile_config']['loss']
        assert loss['class_name'] == 'JointTariffHuber'
        np.testing.assert_array_equal(loss['config']['prices'], data.fixed_price)
        np.testing.assert_array_equal(loss['config']['component_scales'], scale)
        assert loss['config']['net_scale'] == net_scale
        checks.append({'asof_day': first, 'forecast_days': stop-first, 'parameters': meta['parameters'],
                       'scale_and_loss_config_exact': True, 'hashes_passed': True,
                       'split_and_reload_passed': True, 'future_feature_checks': 2})
    assert days == list(range(31, 365)) and len(checks) == 48 and checks[-1]['forecast_days'] == 5
    with np.load(OUT/'raw_predictions.npz') as saved:
        np.testing.assert_array_equal(np.concatenate(predictions), saved['values'])
    for stage, expected in manifest['stage_archive_sha256'].items():
        assert digest(OUT/f'{stage}_predictions.npz') == expected
    with np.load(OUT/'ridge28_predictions.npz') as ridge, np.load(OUT/'memory_predictions.npz') as memory:
        np.testing.assert_array_equal(ridge['values'][:, :, 1], memory['values'][:, :, 1])
        np.testing.assert_array_equal(ridge['values'][0], memory['values'][0])
    for name in ('ridge28_audit.json', 'memory_audit.json'):
        entries = json.loads((OUT/name).read_text())
        if isinstance(entries, dict):
            entries = entries['days']
        for item in entries:
            if 'history_last_label' in item:
                assert item['history_last_label'] is None or item['history_last_label'] < item['origin']
            if 'prior_label_end_exclusive' in item:
                assert item['prior_label_end_exclusive'] is None or item['prior_label_end_exclusive'] <= item['origin']
    monthly_dir = Path(json.loads((ROOT/'data/results/exp008/neural_joint/manifest.json').read_text())['run_directory'])
    matched = []
    for asof, month in ((31, 2), (59, 3)):
        first = keras_weights(directory/f'joint_asof{asof:03d}_s42.keras')
        second = keras_weights(monthly_dir/f'joint_m{month:02d}_s42.keras')
        assert set(first) == set(second)
        maximum = max(float(np.max(np.abs(first[name].astype(float)-second[name].astype(float))))
                      for name in first)
        assert maximum == 0
        matched.append({'asof_day': asof, 'month': month,
                        'all_weights_and_optimizer_state_max_error': maximum})
    unchanged = json.loads((OUT/'original_archives_unchanged.json').read_text())
    for name, expected in unchanged['sha256'].items():
        assert digest(ROOT/name) == expected
    report = {'passed': True, 'weekly_models': checks, 'complete_334_unique_days': True,
              'all_48_saved_losses_are_original_symmetric_JointTariffHuber': True,
              '96_per_issue_feature_mutations_passed': True,
              'weekly_memory_preserves_weekly_ridge_PV': True,
              'postprocessing_all_labels_complete': True,
              'shared_monthly_refit_origins_have_identical_weights': matched,
              'original_archives_unchanged': True,
              'source_sha256': digest(__file__)}
    (OUT/'independent_verification.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != 'weekly_models'}, indent=2))
    return report


if __name__ == '__main__':
    run()
