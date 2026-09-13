"""Rebuild selection from past validation, reload models, and mutate future."""

from __future__ import annotations

import copy
import json

import joblib
import numpy as np

from experiments.exp008.forecast_absolute_hgb import features_for_day, fit_month, array_hash, ArrayStore
from experiments.exp008.forecast_calibration import CalibratedStore
from experiments.exp008.hgb_capacity_selection import (
    OUT, FAMILIES, choose, digest, model_configuration, save)
from experiments.exp008.hgb_stage_selection import best_stage, outputs, predict_at_stage
from experiments.exp008.load_energy_memory import EnergyMemoryStore
from experiments.problem2.exp003.data import Data, ROOT
from experiments.problem2.exp004.data import age_weights, split_days


def audit(retrain=True):
    protocol = json.loads((OUT/'protocol.json').read_text())
    audits = json.loads((OUT/'training_audit.json').read_text())
    selections = json.loads((OUT/'selection_audit.json').read_text())
    assert len(audits) == 33 and len(selections) == 11
    for name, expected in protocol['source_sha256'].items():
        assert digest(OUT/'source_archive'/name) == expected
    for name, expected in protocol['reused_artifact_sha256'].items():
        assert digest(OUT/'stage15_archive'/name) == expected
    for name, expected in json.loads((OUT/'source_data_sha256.json').read_text()).items():
        assert digest(ROOT/'data/raw'/name) == expected
    old_check = json.loads((OUT/'stage15_archive/independent_retraining_verification.json').read_text())
    assert old_check['passed']
    data = Data()
    features = np.stack([features_for_day(data, day)[0] for day in range(7, 365)])
    with np.load(OUT/'raw.npz') as z:
        raw_values, origins = z['values'], z['origins']
    with np.load(OUT/'stage15_archive/raw.npz') as z:
        prior15_raw = z['values']
    families, month_checks = {}, []
    largest_score_error = 0.
    for month in range(2, 13):
        train, validation, formal, cutoff = split_days(month)
        xtrain = features[train-7].reshape(-1, 31)
        xval = features[validation-7].reshape(-1, 31)
        ytrain = np.stack([data.actual[day*144:(day+1)*144] for day in train]).reshape(-1, 2)
        yval = np.stack([data.actual[day*144:(day+1)*144] for day in validation]).reshape(-1, 2)
        mutated = copy.copy(data)
        mutated.actual = data.actual.copy()
        mutated.actual[int(formal[0])*144:] += np.array([30000., 70000.])
        for ds, x, y in ((train, xtrain, ytrain), (validation, xval, yval)):
            xx = np.concatenate([features_for_day(mutated, int(day))[0] for day in ds])
            yy = np.concatenate([mutated.actual[day*144:(day+1)*144] for day in ds])
            np.testing.assert_array_equal(xx, x)
            np.testing.assert_array_equal(yy, y)
        records, family_rows = {}, []
        mapper = None
        for leaf in FAMILIES:
            record = next(row for row in audits if row['month'] == month and row['leaf_nodes'] == leaf)
            records[leaf] = record
            assert record['training_days'] == train.tolist()
            assert record['validation_days'] == validation.tolist()
            assert record['train_label_stop_exclusive'] == cutoff*144
            assert record['validation_label_stop_exclusive'] == int(formal[0])*144
            for key, value in (('training_feature_hash', xtrain), ('validation_feature_hash', xval),
                               ('training_label_hash', ytrain), ('validation_label_hash', yval)):
                assert record[key] == array_hash(value)
            path = OUT/record['model_archive']
            assert digest(path) == record['model_sha256']
            family = joblib.load(path)
            families[month, leaf] = family
            for channel, name in enumerate(('load', 'pv')):
                model = family['models'][channel]
                stage = best_stage(model)
                assert stage == family['stages'][channel]
                assert all(model.get_params()[key] == value for key, value in
                           model_configuration(protocol, leaf)['model_configuration'].items())
                if mapper is None:
                    # HGB's only feature scaling analogue is its bin mapper.
                    # Refit it solely on this month's training rows, then
                    # compare every family/channel's fitted thresholds.
                    weights = np.repeat(age_weights(train, cutoff, 90), 144)
                    mapper = type(model._bin_mapper)(**model._bin_mapper.get_params()).fit(xtrain, sample_weight=weights)
                for expected, observed in zip(mapper.bin_thresholds_, model._bin_mapper.bin_thresholds_, strict=True):
                    np.testing.assert_array_equal(expected, observed)
                pred = predict_at_stage(model, xval, stage)
                independent_score = -.5*float(np.mean((pred-yval[:, channel])**2))
                error = abs(independent_score-float(model.validation_score_[stage]))
                assert error < 1e-6
                largest_score_error = max(largest_score_error, error)
                family_rows.append({'leaf_nodes': leaf, 'channel': name, 'selected_stage': stage,
                    'best_validation_score': independent_score})
            if leaf == 15:
                for day in (int(formal[0]), int(formal[-1])):
                    np.testing.assert_array_equal(prior15_raw[day-31], outputs(data, day, family['models'], family['stages']))
        selection = next(row for row in selections if row['month'] == month)
        assert not selection['formal_target_used_for_selection']
        assert selection['selection_latest_label_exclusive'] == int(formal[0])*144
        winners = [choose([row for row in family_rows if row['channel'] == name]) for name in ('load', 'pv')]
        for fresh, recorded in zip(winners, selection['winners'], strict=True):
            assert fresh['leaf_nodes'] == recorded['leaf_nodes'] and fresh['selected_stage'] == recorded['selected_stage']
        models = [families[month, row['leaf_nodes']]['models'][channel] for channel, row in enumerate(winners)]
        stages = [row['selected_stage'] for row in winners]
        for day in formal:
            np.testing.assert_array_equal(raw_values[day-31], outputs(data, int(day), models, stages))
        np.testing.assert_array_equal(raw_values[int(formal[0])-31], outputs(mutated, int(formal[0]), models, stages))
        month_checks.append({'month': month, 'all_train_validation_feature_label_hashes_passed': True,
            'all_family_binning_thresholds_rebuilt_from_training_prefix': True,
            'future_formal_actual_cannot_change_validation_selection_inputs': True,
            'selected_leaf_counts': [row['leaf_nodes'] for row in winners],
            'selected_stages': stages, 'all_formal_raw_predictions_reloaded_exactly': True})
    # Postprocessing fairness: reconstruct both fixed layers independently
    # from the selected raw archive, never from an older model's residuals.
    raw = ArrayStore(raw_values, origins, 'capacity_raw_independent_rebuild')
    ridge = CalibratedStore('ridge_28', data=data, use_cache=False, _base_store=raw, directory=OUT/'independent_calibration')
    memory = EnergyMemoryStore(data=data, base_store=ridge)
    for name, values in (('ridge', ridge.values), ('memory', memory.values)):
        with np.load(OUT/f'{name}.npz') as z:
            np.testing.assert_array_equal(z['values'], values)
    fresh_fits = []
    if retrain:
        for month in protocol['monthly_retraining_future_mutation_check_months']:
            _, _, formal, _ = split_days(month)
            mutated = copy.copy(data)
            mutated.actual = data.actual.copy()
            mutated.actual[int(formal[0])*144:] += np.array([30000., 70000.])
            changed_features = np.stack([features_for_day(mutated, day)[0] for day in range(7, 365)])
            altered_rows = []
            for leaf in FAMILIES:
                if leaf == 15:
                    family = families[month, leaf]
                    models = family['models']
                else:
                    models, fresh = fit_month(mutated, month, changed_features, model_configuration(protocol, leaf))
                    old = next(row for row in audits if row['month'] == month and row['leaf_nodes'] == leaf)
                    for key in ('training_feature_hash', 'validation_feature_hash', 'training_label_hash', 'validation_label_hash'):
                        assert fresh[key] == old[key]
                    prior = families[month, leaf]
                    stages = [best_stage(model) for model in models]
                    assert stages == prior['stages']
                    for a, b in zip(models, prior['models'], strict=True):
                        np.testing.assert_array_equal(a.validation_score_, b.validation_score_)
                    np.testing.assert_array_equal(outputs(mutated, int(formal[0]), models, stages),
                        outputs(data, int(formal[0]), prior['models'], prior['stages']))
                    fresh_fits.append({'month': month, 'leaf_nodes': leaf,
                        'full_model_retraining_validation_scores_stages_predictions_exact': True})
                for channel, name in enumerate(('load', 'pv')):
                    stage = best_stage(models[channel])
                    altered_rows.append({'leaf_nodes': leaf, 'channel': name, 'selected_stage': stage,
                        'best_validation_score': float(models[channel].validation_score_[stage])})
            original = next(row for row in selections if row['month'] == month)['winners']
            for name, old in zip(('load', 'pv'), original, strict=True):
                winner = choose([row for row in altered_rows if row['channel'] == name])
                assert winner['leaf_nodes'] == old['leaf_nodes'] and winner['selected_stage'] == old['selected_stage']
    future = json.loads((OUT/'future_mutation_checks.json').read_text())
    assert future['passed']
    result = {'passed': True, 'monthly_checks': month_checks,
        'maximum_independent_validation_score_error': largest_score_error,
        'all33_families_66_models_archived_and_reloaded': True,
        'all334_selected_raw_predictions_exact': True,
        'both_fixed_postprocessing_layers_recomputed_exactly': True,
        'future_mutated_full_retraining_checks': fresh_fits,
        'reused15_family_previous_retraining_audit_passed': True,
        'daily_future_mutation_checks': future['checks'],
        'no_formal_month_labels_used_for_selection': True,
        'all_sources_reused_models_and_raw_data_hashes_checked': True,
        'source_sha256': digest(__file__)}
    save(OUT/'independent_audit.json', result)
    print('CAPACITY_INDEPENDENT_AUDIT_PASSED', flush=True)
    return result


if __name__ == '__main__':
    audit()
