"""Calibrate weekly HGB against its own untrained validation/post-fit days.

Historical model-evaluated values are calibration hindcasts, not forecasts
issued on those historical days. Only the newly archived outputs are issued.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from experiments.exp008.forecast_absolute_hgb import AbsoluteHGBStore, ArrayStore, array_hash, features_for_day
from experiments.exp008.forecast_calibration import _calibrate, CONFIG
from experiments.exp008.forecast_weekly_hgb import OUT as WEEKLY, ASOFS, KEYS, predict
from experiments.exp008.forecast_shape_diagnostic import summary as score
from experiments.exp008.load_energy_memory import EnergyMemoryStore, correct_day
from experiments.problem2.exp003.data import Data

OUT = Path('data/results/exp008/weekly_hgb_aligned_calibration')


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def calibrated_day(data, day, model, asof, x):
    # This weekly model was fit only before asof-7. Its validation days were
    # also used for early stopping; they are not an independent validation of
    # the subsequent calibrator. All such labels are known before this issue.
    days = np.arange(asof-7, day+1)
    raw = predict(data, days, x, model)
    result, audit = _calibrate(data.actual, days*144, raw, len(days)-1, CONFIG['ridge_28'])
    audit.update(model_asof_day=asof, model_training_label_stop_exclusive=(asof-7)*144,
        model_selection_label_stop_exclusive=asof*144,
        history_value_role='current_model_hindcasts_of_untrained_validation_and_completed_postfit_days',
        historical_hindcasts_were_not_issued_predictions=True,
        same_validation_week_also_used_for_early_stopping=True,
        no_unseen_day_used_for_calibration=True, recalculated_raw_values_sha256=array_hash(raw))
    return result, audit, raw


def main():
    if OUT.exists():
        raise FileExistsError('Existing development results are immutable')
    OUT.mkdir(parents=True)
    assert json.loads((WEEKLY/'verification.json').read_text())['passed']
    artifacts = [WEEKLY/'protocol.json', WEEKLY/'verification.json', WEEKLY/'raw.npz']
    artifacts += sorted(WEEKLY.glob('weekly_asof*.joblib'))
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in artifacts}
    source = Path(__file__).read_bytes()
    (OUT/'source_snapshot.py').write_bytes(source)
    save(OUT/'protocol.json', {'one_fixed_candidate': True,
        'weekly_model_retrained': False, 'asof_days': ASOFS,
        'calibration_history': 'currentmodel_validation_start(asof-7) through day-1;7..13days',
        'historical_base_predictions': 'current model applied to each historical own-issue feature; not historical issued forecasts',
        'calibration_labels_not_used_in_raw_gradient_fit': True,
        'validation_week_also_used_for_HGB_early_stopping': True,
        'calibration_is_not_independent_of_model_selection': True,
        'calibration_formula': 'existingRidge28 feature/penalty/14dayweight formula; shorter model-aligned history',
        'postprocessing': 'same nonrecursive.5 memory over actual archived calibrated issues',
        'causal_information_boundary': 'all model-selection and calibration labels end before current issue',
        'january_validation_hindcasts_not_used_as_formal_issued_risk_history': True,
        'gate_metrics': KEYS, 'gate_before_LP': 'all4 strictly improve originalHGB fullpipeline',
        'development_not_independent_test': True, 'weekly_artifact_sha256': hashes,
        'source_sha256': hashlib.sha256(source).hexdigest()})
    data = Data()
    x = np.stack([features_for_day(data, day)[0] for day in range(7, 365)])
    models = {asof: joblib.load(WEEKLY/f'weekly_asof{asof:03}.joblib') for asof in ASOFS}
    raw_weekly = np.load(WEEKLY/'raw.npz')
    values, audits, historic = [], [], []
    for day in range(31, 365):
        asof = 31+7*((day-31)//7)
        value, audit, raw = calibrated_day(data, day, models[asof], asof, x)
        np.testing.assert_array_equal(raw[-1], raw_weekly['values'][day-31])
        assert audit['history_last_label'] < day*144 and audit['model_selection_label_stop_exclusive'] <= day*144
        values.append(value); audits.append(audit)
        # Variable-length histories pad only for archival storage, never fit.
        padded = np.full((14, 144, 2), np.nan); padded[:len(raw)] = raw
        historic.append(padded)
    origins = np.arange(31, 365)*144
    ridge = ArrayStore(np.stack(values), origins, 'weekly_HGB_model_aligned_ridge')
    memory = EnergyMemoryStore(data=data, base_store=ridge)
    memory.name = 'weekly_HGB_model_aligned_ridge_memory_half'
    for row in memory.audit:
        row['underlying_forecast'] = 'weekly_HGB_model_aligned_ridge_actual_issued_archive'
    for name, store in (('ridge', ridge), ('memory', memory)):
        np.savez_compressed(OUT/f'{name}.npz', values=store.values, origins=origins)
    np.savez_compressed(OUT/'calibration_hindcasts.npz', values=np.stack(historic),
                        history_lengths=[a['history_count']+1 for a in audits], origins=origins)
    save(OUT/'calibration_audit.json', audits); save(OUT/'memory_audit.json', memory.audit)
    mutations = []
    for day in (31, 37, 38, 90, 151, 243, 364):
        changed = copy.copy(data); changed.actual = data.actual.copy()
        changed.actual[day*144:] += [70000., 20000.]
        # Rebuild only requested/past features; no full-feature future read.
        asof = 31+7*((day-31)//7)
        xx = x.copy()
        for old in range(asof-7, day+1):
            xx[old-7] = features_for_day(changed, old)[0]
        value, _, _ = calibrated_day(changed, day, models[asof], asof, xx)
        np.testing.assert_array_equal(value, ridge.values[day-31])
        later = ridge.values.copy(); later[day-30:] += [40000., 80000.]
        np.testing.assert_array_equal(correct_day(data.actual, origins, ridge.values, day-31)[0],
            correct_day(changed.actual, origins, later, day-31)[0])
        mutations.append({'day': day, 'raw_hindcasts_calibration_and_memory_unchanged': True})
    # Separate full replay of archived calibration bases and chronological
    # memory gives an independent check of the model-evaluation loop.
    for i, audit in enumerate(audits):
        day = i+31; length = audit['history_count']+1
        base = historic[i][:length]
        ds = np.arange(audit['model_asof_day']-7, day+1)
        np.testing.assert_array_equal(_calibrate(data.actual, ds*144, base, length-1, CONFIG['ridge_28'])[0], ridge.values[i])
        np.testing.assert_array_equal(correct_day(data.actual, origins, ridge.values, i)[0], memory.values[i])
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == v for p, v in hashes.items())
    save(OUT/'verification.json', {'passed': True, 'future_mutations': mutations,
        'all334_current_raw_outputs_identical_to_weekly_archive': True,
        'all334_calibration_hindcast_replays_identical': True,
        'all334_memory_replays_identical': True, 'all_weekly_artifact_hashes_unchanged': True})
    truth = data.actual[origins[:, None]+np.arange(144), :2]
    records = []
    weekly_memory = np.load(WEEKLY/'memory.npz')['values']
    for name, arr in (('original_HGB', AbsoluteHGBStore().values),
                      ('weekly_issued_history', weekly_memory), ('weekly_aligned', memory.values)):
        e = arr-truth
        for channel in ('load', 'pv', 'net'):
            error = e[:, :, 0]-e[:, :, 1] if channel == 'net' else e[:, :, int(channel == 'pv')]
            records.append({'name': name, 'channel': channel, **score(error, data.fixed_price)})
    pd.DataFrame(records).to_csv(OUT/'metrics.csv', index=False)
    net = {r['name']: r for r in records if r['channel'] == 'net'}
    gates = {key: net['weekly_aligned'][key] < net['original_HGB'][key] for key in KEYS}
    result = {'complete': True, 'net_metrics': net, 'gate_components': gates,
              'LP_continuation_gate_passed': all(gates.values()), 'final_model_selection': False}
    save(OUT/'summary.json', result); print(json.dumps(result), flush=True)
    if all(gates.values()):
        from experiments.exp008.audited_store_bridge import run
        run(memory, OUT/'lp_bridge', [OUT/'protocol.json', OUT/'verification.json', OUT/'memory.npz'])


if __name__ == '__main__':
    main()
