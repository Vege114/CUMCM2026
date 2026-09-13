"""One predeclared monthly/channel leaf-capacity selection on past validation."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import shutil

import joblib
import numpy as np
import pandas as pd

from experiments.exp008.forecast_absolute_hgb import (
    OUT as BASE, AbsoluteHGBStore, ArrayStore, features_for_day, fit_month, array_hash)
from experiments.exp008.forecast_calibration import CalibratedStore, CONFIG, _calibrate
from experiments.exp008.forecast_shape_diagnostic import summary as score
from experiments.exp008.hgb_stage_selection import (
    OUT as STAGE15, best_stage, predict_at_stage, outputs)
from experiments.exp008.load_energy_memory import EnergyMemoryStore, correct_day
from experiments.problem2.exp003.data import Data, ROOT
from experiments.problem2.exp004.data import split_days

OUT = ROOT/'data/results/exp008/hgb_capacity_selection'
FAMILIES = (15, 31, 63)
PRIMARY = 'past_validation_capacity_HGB_Ridge28_memory_half'
GATE_KEYS = ('rmse_kw', 'high_price_rmse_kw', 'daily_energy_rmse_kwh', 'cumulative_error_rmse_kwh')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def choose(rows):
    """One-channel records; exact score ties choose smaller capacity."""
    return max(rows, key=lambda row: (row['best_validation_score'], -row['leaf_nodes']))


def freeze_protocol():
    if OUT.exists():
        raise FileExistsError('Existing capacity selection evidence is immutable')
    OUT.mkdir(parents=True)
    old = json.loads((STAGE15/'protocol.json').read_text())
    protocol = {'model_configuration_except_leaf_nodes': old['model_configuration'],
        'features': old['features'], 'leaf_families': list(FAMILIES),
        'selection': 'each month and channel: highest previous7day validation_score_ over every fitted nonzero stage and leaf family',
        'stage_tie': 'earliest nonzero stage', 'capacity_score_tie': 'smaller leaf count',
        'score_semantics': 'sklearn scoring=loss; validation_score_ is negative loss, not named RMSE',
        'additional_reported_score': 'raw unmasked previous7day validation RMSE at chosen stage',
        'previous_15_leaf_family_reused': True, 'new_training_families': [31, 63],
        'max_iter': 400, 'n_iter_no_change': 10, 'seed': 42,
        'same_training_boundaries': 'train day7..monthstart-8; previous7days validation; age half-life90days',
        'no_external_scaler': True, 'HGB_histogram_binning_fit_uses_training_rows_only': True,
        'same_day_features': 'past actuals only before each individual issue, including within frozen monthly block',
        'same_postprocessing': 'nonnegative and causal daylight; same fixed Ridge28; same nonrecursive0.5load memory',
        'formal_month_labels_used_for_capacity_or_stage_selection': False,
        'gate': {'against': 'original complete direct_HGB_Ridge28_memory_half', 'all_strictly_improve': list(GATE_KEYS)},
        'secondary_comparison': 'existing capacity15 best-validation-stage complete pipeline',
        'gate_action': 'only if all4 improve: one fixed334day tree28/q.8/buffer500 LP bridge',
        'no_additional_capacity_or_frequency_grid': True, 'development_year_not_independent_test': True,
        'monthly_retraining_future_mutation_check_months': [2, 9],
        'daily_future_mutation_check_days': [31, 32, 59, 90, 151, 243, 364],
        'source_sha256': {}, 'reused_artifact_sha256': {}}
    dependencies = ['hgb_capacity_selection', 'hgb_capacity_selection_audit', 'hgb_stage_selection',
        'forecast_absolute_hgb', 'forecast_net_hgb', 'forecast_calibration', 'forecast_shape_diagnostic',
        'load_energy_memory', 'audited_store_bridge', 'risk_window', 'controller_candidate', 'verify']
    sources = [ROOT/f'experiments/exp008/{name}.py' for name in dependencies]
    sources += [ROOT/f'experiments/problem2/{name}' for name in ('exp003/data.py', 'exp004/data.py')]
    for source in sources:
        target = OUT/'source_archive'/source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        protocol['source_sha256'][str(source.relative_to(ROOT))] = digest(target)
    reused = [STAGE15/name for name in ('protocol.json', 'training_audit.json', 'summary.json',
               'raw.npz', 'ridge.npz', 'memory.npz', 'independent_retraining_verification.json')]
    reused += [STAGE15/f'm{month:02}.joblib' for month in range(2, 13)]
    for source in reused:
        target = OUT/'stage15_archive'/source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        protocol['reused_artifact_sha256'][source.name] = digest(target)
    for name in ('protocol.json', 'provenance.json', 'direct_hgb_ridge28_memory_half.npz'):
        target = OUT/'original_HGB_archive'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(BASE/name, target)
    save(OUT/'protocol.json', protocol)
    return protocol


def model_configuration(protocol, leaf):
    config = copy.deepcopy(protocol['model_configuration_except_leaf_nodes'])
    config['max_leaf_nodes'] = leaf
    return {'features': protocol['features'], 'model_configuration': config}


def family_records(data, month, features, protocol, family, audit):
    _, validation, _, _ = split_days(month)
    xval = features[validation-7].reshape(-1, len(protocol['features']))
    yval = np.stack([data.actual[day*144:(day+1)*144] for day in validation]).reshape(-1, 2)
    rows = []
    for channel, name in enumerate(('load', 'pv')):
        model = family['models'][channel]
        stage = best_stage(model)
        assert stage == family['stages'][channel]
        prediction = predict_at_stage(model, xval, stage)
        rows.append({'month': month, 'channel': name, 'leaf_nodes': model.max_leaf_nodes,
            'fitted_iterations': int(model.n_iter_), 'selected_stage': stage,
            'best_validation_score': float(model.validation_score_[stage]),
            'validation_RMSE_kw_raw_unmasked': float(np.sqrt(np.mean((prediction-yval[:, channel])**2))),
            'validation_prediction_sha256': array_hash(prediction),
            'train_label_stop_exclusive': audit['train_label_stop_exclusive'],
            'validation_label_stop_exclusive': audit['validation_label_stop_exclusive']})
    return rows


def main():
    from experiments.exp008.hgb_capacity_selection_audit import audit
    protocol = freeze_protocol()
    data, original = Data(), AbsoluteHGBStore()
    source_hashes = data.hashes
    save(OUT/'source_data_sha256.json', source_hashes)
    features = np.stack([features_for_day(data, day)[0] for day in range(7, 365)])
    old_audits = {row['month']: row for row in json.loads((OUT/'stage15_archive/training_audit.json').read_text())}
    families, month_selection, training_audits, score_rows = {}, [], [], []
    values = np.empty_like(original.values)
    for month in range(2, 13):
        families[month] = {}
        month_rows = []
        for leaf in FAMILIES:
            config = model_configuration(protocol, leaf)
            if leaf == 15:
                family = joblib.load(OUT/f'stage15_archive/m{month:02}.joblib')
                fitted_audit = copy.deepcopy(old_audits[month])
                reused = True
            else:
                models, fitted_audit = fit_month(data, month, features, config)
                family = {'models': models, 'stages': [best_stage(model) for model in models]}
                reused = False
            for model in family['models']:
                for key, value in config['model_configuration'].items():
                    assert model.get_params()[key] == value
            path = OUT/f'models/leaf{leaf}_m{month:02}.joblib'
            path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(family, path)
            loaded = joblib.load(path)
            first = int(fitted_audit['formal_days'][0])
            np.testing.assert_array_equal(outputs(data, first, family['models'], family['stages']),
                                          outputs(data, first, loaded['models'], loaded['stages']))
            fitted_audit.update(leaf_nodes=leaf, reused_stage15_family=reused,
                model_archive=str(path.relative_to(OUT)), model_sha256=digest(path),
                selected_stages=family['stages'], first_day_reload_exact=True)
            training_audits.append(fitted_audit)
            rows = family_records(data, month, features, protocol, family, fitted_audit)
            month_rows.extend(rows)
            families[month][leaf] = family
        winners = [choose([row for row in month_rows if row['channel'] == name]) for name in ('load', 'pv')]
        selected_models = [families[month][row['leaf_nodes']]['models'][channel] for channel, row in enumerate(winners)]
        stages = [row['selected_stage'] for row in winners]
        formal = split_days(month)[2]
        for day in formal:
            values[day-31] = outputs(data, int(day), selected_models, stages)
        selection = {'month': month, 'issue_day': int(formal[0]), 'issue_origin': int(formal[0])*144,
            'selection_latest_label_exclusive': winners[0]['validation_label_stop_exclusive'],
            'winners': winners, 'formal_target_used_for_selection': False}
        month_selection.append(selection)
        score_rows.extend(month_rows)
        save(OUT/'training_audit.json', training_audits)
        save(OUT/'selection_audit.json', month_selection)
        pd.DataFrame(score_rows).to_csv(OUT/'capacity_selection_scores.csv', index=False)
        print('MONTH', month, 'selected', [(row['channel'], row['leaf_nodes'], row['selected_stage']) for row in winners], flush=True)
    raw = ArrayStore(values, original.origins, 'past_validation_capacity_HGB_raw')
    ridge = CalibratedStore('ridge_28', data=data, use_cache=False, _base_store=raw, directory=OUT/'calibration')
    memory = EnergyMemoryStore(data=data, base_store=ridge)
    memory.name = PRIMARY
    for row in memory.audit:
        row['underlying_forecast'] = 'past_validation_capacity_absolute_HGB_then_fixed_Ridge28'
    for name, store in (('raw', raw), ('ridge', ridge), ('memory', memory)):
        np.savez_compressed(OUT/f'{name}.npz', values=store.values, origins=store.origins)
    save(OUT/'ridge_audit.json', ridge.audit)
    save(OUT/'memory_audit.json', memory.audit)
    checks = []
    for day in protocol['daily_future_mutation_check_days']:
        changed = copy.copy(data)
        changed.actual = data.actual.copy()
        changed.actual[day*144:] += np.array([30000., 70000.])
        month = int((pd.Timestamp('2025-01-01')+pd.Timedelta(days=day)).month)
        rows = next(row['winners'] for row in month_selection if row['month'] == month)
        models = [families[month][row['leaf_nodes']]['models'][channel] for channel, row in enumerate(rows)]
        stages = [row['selected_stage'] for row in rows]
        np.testing.assert_array_equal(values[day-31], outputs(changed, day, models, stages))
        i = day-31
        future_raw = values.copy(); future_raw[i+1:] += 50000.
        a, _ = _calibrate(data.actual, original.origins, values, i, CONFIG['ridge_28'])
        b, _ = _calibrate(changed.actual, original.origins, future_raw, i, CONFIG['ridge_28'])
        np.testing.assert_array_equal(a, b)
        future_ridge = ridge.values.copy(); future_ridge[i+1:] += 50000.
        a, _ = correct_day(data.actual, original.origins, ridge.values, i)
        b, _ = correct_day(changed.actual, original.origins, future_ridge, i)
        np.testing.assert_array_equal(a, b)
        checks.append({'day': day, 'feature_prediction_and_both_postprocessing_future_mutation_passed': True})
    save(OUT/'future_mutation_checks.json', {'passed': True, 'checks': checks})
    # All fitting/capacity/stage decisions and issued arrays are frozen above.
    # Current formal actuals only enter the following post-hoc score report.
    truth = data.actual[original.origins[:, None]+np.arange(144)]
    with np.load(OUT/'stage15_archive/memory.npz') as archive:
        stage15 = archive['values']
    metrics, monthly = [], []
    dates = pd.date_range('2025-02-01', '2025-12-31')
    compared = [('original_HGB', original.values), ('best_stage15_HGB', stage15),
                ('capacity_HGB_raw', raw.values), ('capacity_HGB_ridge', ridge.values),
                ('capacity_HGB_memory', memory.values)]
    for name, value in compared:
        for channel, error in (('load', value[:, :, 0]-truth[:, :, 0]), ('pv', value[:, :, 1]-truth[:, :, 1]),
            ('net', value[:, :, 0]-value[:, :, 1]-truth[:, :, 0]+truth[:, :, 1])):
            metrics.append({'name': name, 'channel': channel, **score(error, data.fixed_price)})
            for month in range(2, 13):
                monthly.append({'name': name, 'channel': channel, 'month': month,
                    **score(error[dates.month == month], data.fixed_price)})
    pd.DataFrame(metrics).to_csv(OUT/'metrics.csv', index=False)
    pd.DataFrame(monthly).to_csv(OUT/'monthly_metrics.csv', index=False)
    old = next(row for row in metrics if row['name'] == 'original_HGB' and row['channel'] == 'net')
    new = next(row for row in metrics if row['name'] == 'capacity_HGB_memory' and row['channel'] == 'net')
    gate = {key: new[key] < old[key] for key in GATE_KEYS}
    save(OUT/'summary.json', {'complete': True, 'metrics': metrics, 'gate_checks': gate,
        'continuation_gate_passed': all(gate.values()),
        'bridge_status_at_this_score_snapshot': 'pending_independent_audit' if all(gate.values()) else 'not_authorized_gate_failed',
        'final_bridge_status_file': 'bridge_completion.json',
        'source_hashes': source_hashes, 'selected_values_sha256': array_hash(memory.values),
        'stage15_values_sha256': array_hash(stage15), 'original_values_sha256': array_hash(original.values),
        'all_monthly_family_reloads_exact': True, 'all_future_checks_passed': True})
    audited = audit(retrain=True)
    if all(gate.values()):
        from experiments.exp008.audited_store_bridge import run
        result = run(memory, OUT/'lp_bridge', [OUT/name for name in ('protocol.json', 'summary.json',
            'memory.npz', 'training_audit.json', 'selection_audit.json', 'independent_audit.json')])
        save(OUT/'bridge_completion.json', {'performed': True, 'audited': audited['passed'],
            'cost_change_vs_original_yuan': result['cost_change_yuan']})
    print(json.dumps({'gate': gate, 'net_baseline': old, 'net_candidate': new}, indent=2), flush=True)


if __name__ == '__main__':
    main()
