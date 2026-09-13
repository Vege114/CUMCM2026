"""Extract report evidence from frozen forecasts; never fit or optimize a model.

Run with ``.venv/bin/python -m experiments.exp008.final_forecast_evidence``.
The selected Q2 forecast and the single HGB used by Q3/Q4 are separate rows.
All values here describe 00:00 issues, not the other issue times of Q3/Q4.
"""
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.problem2.exp003.data import Data, ROOT
from experiments.exp008.frozen_sources import exp005_registry

OUT = ROOT / 'reports/experiments/exp008/evidence/forecast'
FIG = ROOT / 'reports/experiments/exp008/figures/forecast'
RESULT = ROOT / 'data/results/exp008'
HGB = RESULT / 'forecast_absolute_hgb'
ET = RESULT / 'forecast_absolute_extra_trees'
BLEND = RESULT / 'forecast_hgb_extra_trees_half'
DISPATCH = RESULT / 'mode_budget_hold1_physical/hgb_extra_trees_half_hold1_cap8_kappa0_334days'
MODELS = {
    'final_blend': (BLEND, 'hgb_extra_trees_half_ridge28_memory', 'Q2 accepted final midnight forecast'),
    'single_hgb': (HGB, 'direct_hgb_ridge28_memory_half', 'Q3/Q4 model; midnight diagnostic only'),
}
TARGETS = ('load', 'pv', 'net_load')
POPULATIONS = ('all', 'pv_generating')
EXP005_SPEC = '8108d0475ae4f2ea25565a806dfec5e836f62c0a:reports/registry/exp005.json'
EXP005_SHA = '5af3de5bdf08706696eea4ceb4a57d072b9a957bad61eef16d46bb0b6b52c43f'
INPUTS = {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rel(path):
    return str(Path(path).relative_to(ROOT))


def register(path):
    path = Path(path)
    INPUTS[rel(path)] = sha(path)
    return path


def read_json(path):
    return json.loads(register(path).read_text())


def save_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def load_store(directory, name, truth):
    path = register(directory / (name + '.npz'))
    with np.load(path) as z:
        np.testing.assert_array_equal(z['origins'], np.arange(31, 365) * 144)
        values = z['values'].copy()
        assert values.shape == (334, 144, 2) and np.isfinite(values).all()
        np.testing.assert_allclose(z['errors_kw'], truth - values, rtol=0, atol=1e-9)
    return values


def channels(values):
    return np.stack([values[:, :, 0], values[:, :, 1], values[:, :, 0] - values[:, :, 1]], axis=2)


def metric(prediction, actual, mask):
    e = np.asarray(prediction - actual)[mask]
    y = np.asarray(actual)[mask]
    n = len(e)
    sae, sse, signed, denom = map(float, (np.abs(e).sum(), (e * e).sum(), e.sum(), np.abs(y).sum()))
    return {
        'n': n, 'mae_kw': sae / n if n else None,
        'rmse_kw': float(np.sqrt(sse / n)) if n else None,
        'wape_pct': 100 * sae / denom if denom else None,
        'bias_kw': signed / n if n else None,
        'absolute_error_sum_kw': sae, 'squared_error_sum_kw2': sse,
        'signed_error_sum_kw': signed, 'actual_absolute_sum_kw': denom,
        'missing_reason': ('No actual PV>0 samples at this lead' if not n else
                           'Actual absolute sum is zero; WAPE unavailable' if not denom else ''),
    }


def extract_current(truth, dates):
    y = channels(truth)
    masks = {'all': np.ones(truth.shape[:2], bool), 'pv_generating': truth[:, :, 1] > 0}
    records = {key: [] for key in ('annual', 'monthly', 'lead', 'daily', 'daily_energy')}
    values_by_model = {}
    for model, (directory, archive, role) in MODELS.items():
        values = load_store(directory, archive, truth)
        values_by_model[model] = values
        pred = channels(values)
        common = {'model': model, 'model_id': archive, 'role': role, 'seed': 42,
                  'issue_hour': 0, 'evaluation_start': '2025-02-01', 'evaluation_end': '2025-12-31',
                  'forecast_source': rel(directory / (archive + '.npz'))}
        for c, target in enumerate(TARGETS):
            for population, mask in masks.items():
                base = {**common, 'target': target, 'population': population}
                records['annual'].append({**base, 'month': 0, 'days': 334, **metric(pred[:, :, c], y[:, :, c], mask)})
                for month in range(2, 13):
                    rows = dates.month == month
                    records['monthly'].append({**base, 'month': month, 'days': int(rows.sum()),
                        **metric(pred[rows, :, c], y[rows, :, c], mask[rows])})
                for slot in range(144):
                    records['lead'].append({**base, 'lead_slot': slot + 1,
                        'lead_start_exclusive_minutes': slot * 10, 'lead_end_inclusive_minutes': (slot + 1) * 10,
                        **metric(pred[:, slot, c], y[:, slot, c], mask[:, slot])})
                for day, date in enumerate(dates):
                    records['daily'].append({**base, 'date': str(date.date()), 'day': day + 31,
                        **metric(pred[day, :, c], y[day, :, c], mask[day])})
            for day, date in enumerate(dates):
                e = pred[day, :, c] - y[day, :, c]
                records['daily_energy'].append({**common, 'target': target, 'date': str(date.date()),
                    'actual_kwh': float(y[day, :, c].sum() / 6),
                    'predicted_kwh': float(pred[day, :, c].sum() / 6),
                    'signed_energy_error_kwh': float(e.sum() / 6),
                    'within_day_cumulative_error_rmse_kwh': float(np.sqrt(np.mean((e.cumsum() / 6) ** 2)))})
    for key, rows in records.items():
        pd.DataFrame(rows).to_csv(OUT / (key + '_metrics.csv'), index=False)
    return records, values_by_model


def extract_stages(truth):
    y = channels(truth)
    rows = []
    archive_names = {
        'final_blend': ['hgb_extra_trees_half_raw', 'hgb_extra_trees_half_ridge28', 'hgb_extra_trees_half_ridge28_memory'],
        'single_hgb': ['direct_hgb_raw', 'direct_hgb_ridge28', 'direct_hgb_ridge28_memory_half'],
    }
    arrays = {}
    for model, names in archive_names.items():
        directory = MODELS[model][0]
        for stage, name in zip(('raw', 'ridge28', 'memory_half'), names):
            arrays[model, stage] = load_store(directory, name, truth)
            p = channels(arrays[model, stage])
            for c, target in enumerate(TARGETS):
                for population in POPULATIONS:
                    mask = np.ones(y.shape[:2], bool) if population == 'all' else truth[:, :, 1] > 0
                    rows.append({'model': model, 'stage': stage, 'model_id': name, 'target': target,
                        'population': population, **metric(p[:, :, c], y[:, :, c], mask)})
    et_raw = load_store(ET, 'absolute_extra_trees_raw', truth)
    np.testing.assert_array_equal(arrays['final_blend', 'raw'], .5 * arrays['single_hgb', 'raw'] + .5 * et_raw)
    pd.DataFrame(rows).to_csv(OUT / 'stage_metrics.csv', index=False)
    return rows


def historical_evidence(current):
    source = register(ROOT / 'reports/experiments/exp006/evidence/forecast_history.csv')
    history = pd.read_csv(source)
    labels = ['exp001 午夜重评分', 'exp002 午夜重评分', 'exp003 正式', 'exp004 无季节', 'exp006 正式·树DP']
    selected = history[history.label.isin(labels)].copy()
    assert len(selected) == 30 and set(selected.issue_hour) == {0} and set(selected.month) == {0}
    original, registries, provenance = [], {}, {}
    for number in range(1, 7):
        experiment = f'exp{number:03d}'
        if number == 5:
            raw, frozen = exp005_registry(ROOT)
            (OUT / 'source_registry_exp005.json').write_bytes(raw)
            registry = json.loads(raw)
            origin, digest = frozen['source'], frozen['sha256']
        else:
            path = register(ROOT / f'reports/registry/{experiment}.json')
            registry = json.loads(path.read_text())
            origin, digest = rel(path), sha(path)
        registries[experiment] = registry
        provenance[experiment] = {'source': origin, 'sha256': digest,
            'forecast_metric_records': len(registry.get('forecast_metrics', [])),
            'protocol': registry.get('protocol'), 'primary_seed': registry.get('primary_seed'),
            'comparison_note': registry.get('comparison_note')}
        if number == 5:
            provenance[experiment].update(frozen)
        for i, record in enumerate(registry.get('forecast_metrics', [])):
            original.append({**record, 'registry_experiment': experiment,
                'registry_record': i, 'registry_source': origin, 'registry_sha256': digest,
                'comparison_basis': 'original registry; preserve original protocol and population',
                'raw_record_json': json.dumps(record, ensure_ascii=False, allow_nan=False)})
    pd.DataFrame(original).to_csv(OUT / 'history_original_registry.csv', index=False)
    rows, verified = [], 0
    for _, record in selected.iterrows():
        path = register(ROOT / record.source)
        upstream = pd.read_csv(path)
        matched = upstream[(upstream.target == record.target) & (upstream.population == record.population)
                           & (upstream.issue_hour == 0) & (upstream.month == 0)]
        if 'predictor_id' in upstream:
            matched = matched[matched.predictor_id == record.predictor_id]
        else:
            matched = matched[matched.case_id == record.case_id]
        assert len(matched) == 1, (record.label, record.target, len(matched))
        old = matched.iloc[0]
        for k in ('n', 'absolute_error_sum', 'squared_error_sum', 'actual_abs_sum', 'mae', 'rmse', 'wape_pct'):
            np.testing.assert_allclose(old[k], record[k], rtol=1e-12, atol=1e-8)
        np.testing.assert_allclose(record.mae, record.absolute_error_sum / record.n, rtol=1e-12)
        np.testing.assert_allclose(record.rmse, np.sqrt(record.squared_error_sum / record.n), rtol=1e-12)
        np.testing.assert_allclose(record.wape_pct, 100 * record.absolute_error_sum / record.actual_abs_sum, rtol=1e-12)
        rows.append({'experiment': record.experiment, 'model': record.experiment, 'label': record.label,
            'target': record.target, 'population': record.population, 'issue_hour': 0,
            'n': int(record.n), 'mae_kw': record.mae, 'rmse_kw': record.rmse, 'wape_pct': record.wape_pct,
            'absolute_error_sum_kw': record.absolute_error_sum, 'squared_error_sum_kw2': record.squared_error_sum,
            'actual_absolute_sum_kw': record.actual_abs_sum, 'available': True,
            'value_basis': 'existing midnight rescore' if record.experiment in ('exp001', 'exp002') else 'existing midnight formal score',
            'source': record.source, 'source_sha256': sha(path), 'source_row_label': record.label,
            'forecast_reused': record.experiment == 'exp006', 'missing_reason': ''})
        verified += 1
    # exp005 has real rounded registry values, but no independent current-branch rescore.
    for target in TARGETS:
        for population in POPULATIONS:
            originals = [r for r in registries['exp005']['forecast_metrics'] if r['target'] == target and
                         r['population'] == ('actual_generation' if population == 'pv_generating' else 'all')]
            assert len(originals) <= 1
            old = originals[0] if originals else {}
            available = bool(old)
            if available:
                same = [r for r in rows if r['experiment'] == 'exp004' and r['target'] == target and r['population'] == population][0]
                for dst, src in (('mae_kw', 'mae'), ('rmse_kw', 'rmse'), ('wape_pct', 'wape_pct')):
                    np.testing.assert_allclose(same[dst], old[src], rtol=0, atol=1e-9)
            rows.append({'experiment': 'exp005', 'model': 'exp005', 'label': 'exp005 原登记·复用exp004预测',
                'target': target, 'population': population, 'issue_hour': 0,
                'n': old.get('n'), 'mae_kw': old.get('mae'), 'rmse_kw': old.get('rmse'),
                'wape_pct': old.get('wape_pct'), 'absolute_error_sum_kw': None,
                'squared_error_sum_kw2': None, 'actual_absolute_sum_kw': None, 'available': available,
                'value_basis': 'original registry rounded values; forecast reuse explicitly documented',
                'source': provenance['exp005']['source'], 'source_sha256': EXP005_SHA,
                'source_row_label': 'protocol.forecast=exp004/no_season/seed42', 'forecast_reused': True,
                'missing_reason': '' if available else 'This target/population is absent from the exp005 original registry; no value inferred from reuse.'})
    for r in current:
        rows.append({**r, 'experiment': 'exp008', 'label': 'exp008 Q2 融合' if r['model'] == 'final_blend' else 'exp008 单HGB',
            'available': True, 'value_basis': 'new read-only score of frozen issued forecast',
            'source': r['forecast_source'], 'source_sha256': INPUTS[r['forecast_source']], 'forecast_reused': False})
    frame = pd.DataFrame(rows).sort_values(['experiment', 'model', 'target', 'population'])
    frame.to_csv(OUT / 'history_midnight_comparable.csv', index=False)
    save_json(OUT / 'history_registry_provenance.json', provenance)
    return {'selected_midnight_rows_verified_against_upstream': verified,
        'exp005_original_sha256_verified': True, 'exp005_present_metric_rows': 4,
        'exp005_absent_metric_rows': 2, 'exp005_no_new_rescore_or_inferred_metrics': True,
        'original_registry_metric_rows_preserved': len(original), 'exp007_excluded': True,
        'different_training_protocols_and_examined_year_preclude_independent_test_claim': True}


def configuration():
    configs, training = {}, []
    for family, directory in [('HGB', HGB), ('ExtraTrees', ET)]:
        protocol = read_json(directory / 'protocol.json')
        provenance = read_json(directory / 'provenance.json')
        audit = read_json(directory / 'training_audit.json')
        configs[family] = {'estimator': 'sklearn.ensemble.' + ('HistGradientBoostingRegressor' if family == 'HGB' else 'ExtraTreesRegressor'),
            'fixed_configuration': protocol['model_configuration'], 'protocol_source': rel(directory / 'protocol.json'),
            'sklearn_version': provenance.get('sklearn_version'), 'numpy_version': provenance.get('numpy_version'),
            'features': protocol['features'], 'model_count': 22, 'targets': ['absolute load kW', 'absolute PV kW']}
        if family == 'ExtraTrees':
            configs[family]['resolved_model_parameters'] = audit[0]['models'][0]['resolved_model_parameters']
        for row in audit:
            for model in row['models']:
                path = register(Path(model['model_path']))
                assert sha(path) == model['model_sha256']
                training.append({'family': family, 'month': row['month'], 'channel': model['channel'],
                    'train_first_day': min(row['training_days']), 'train_last_day': max(row['training_days']),
                    'validation_first_day': min(row['validation_days']), 'validation_last_day': max(row['validation_days']),
                    'formal_first_day': min(row['formal_days']), 'formal_last_day': max(row['formal_days']),
                    'training_rows': row['training_rows'], 'validation_rows': row['validation_rows'],
                    'iterations': model.get('iterations'), 'trees': model.get('trees'),
                    'training_seconds': model['training_seconds'], 'model_source': rel(path), 'model_sha256': sha(path)})
    assert configs['HGB']['features'] == configs['ExtraTrees']['features'] and len(configs['HGB']['features']) == 31
    pd.DataFrame(training).to_csv(OUT / 'training_models_and_boundaries.csv', index=False)
    timings = [{'family': family, 'stage': '22 primary fit plus validation-predict blocks',
                'seconds': sum(r['training_seconds'] for r in training if r['family'] == family),
                'measured': True, 'source': rel(directory / 'training_audit.json'),
                'scope': 'Original timer starts before model.fit and ends after the following model.predict(X_val), including surrounding call overhead. Excludes feature construction, formal prediction, calibration, audit retraining, dispatch and report; not pure fit time.'}
               for family, directory in [('HGB', HGB), ('ExtraTrees', ET)]]
    for family, directory, key in [('HGB', HGB, 'total_seconds_before_bridge'), ('ExtraTrees', ET, 'elapsed_seconds_before_bridge')]:
        timings.append({'family': family, 'stage': 'original pre-bridge pipeline wall time',
            'seconds': read_json(directory / 'provenance.json')[key], 'measured': True,
            'source': rel(directory / 'provenance.json') + ':' + key,
            'scope': 'Whole original workflow before LP bridge, including audits; not pure inference time and not directly comparable to fit sums.'})
    for family in ('HGB', 'ExtraTrees'):
        timings.append({'family': family, 'stage': 'fit only', 'seconds': None, 'measured': False,
            'source': '', 'scope': 'Original training_seconds also includes the subsequent validation prediction; a pure fit duration cannot be recovered.'})
    for family in ('HGB', 'ExtraTrees', 'fixed_half_blend'):
        for stage in ('feature construction', 'prediction only', 'Ridge28 only', 'memory only'):
            timings.append({'family': family, 'stage': stage, 'seconds': None, 'measured': False,
                'source': '', 'scope': 'No separate original timer; omitted rather than measured as zero. No rerun performed to manufacture timings.'})
    pd.DataFrame(timings).to_csv(OUT / 'runtime_evidence.csv', index=False)
    model_config = {
        'selected_Q2_model': MODELS['final_blend'][1], 'Q3_Q4_single_HGB_model': MODELS['single_hgb'][1],
        'seed': 42, 'number_of_formal_seeds': 1, 'sample_seed_standard_deviation': None,
        'seed_std_missing_reason': 'Only one seed; no multi-seed stability estimate exists.',
        'families': configs, 'unique_raw_models_for_blend': 44,
        'architecture': 'Two separate target regressors per monthly family; 31 engineered causal features per ten-minute slot. No CNN in this final forecast.',
        'train_validation_schedule': 'For zero-based month-start day D: train day7..D-8, validate dayD-7..D-1, issue only within current month. Daily features use earlier actuals; monthly fitted model remains frozen.',
        'training_weights': '2**(-(train_cutoff_day-1-training_day)/90), normalized to mean1 and repeated144 times; train_cutoff_day=D-7.',
        'HGB_early_stopping': 'Prior7day explicit validation, patience10, tol1e-7; retain final fitted iteration. Not the later best-stage400 variant.',
        'ExtraTrees_validation': 'Same prior7day holdout, metrics only; no early stopping or model selection/refit on this holdout.',
        'raw_physical_postprocessing': 'Both channels nonnegative. PV daylight mask is past28complete days positive-PV union by slot, expanded20minutes each side.',
        'raw_blend': {'HGB_weight': .5, 'ExtraTrees_weight': .5, 'both_channels': True,
            'fixed_before_evaluation': True, 'grid_or_weight_tuning': False,
            'order': ['causal raw HGB and raw ExtraTrees outputs', 'fixed arithmetic mean of both raw channels',
                      'Ridge28 recalibrated on that blend history', 'nonrecursive half daily load memory']},
        'Ridge28': {'regressions': 'Separate load/PV residual ridge fits, each with cross-channel forecast feature',
            'prior_issued_day_window': 28, 'half_life_days': 14, 'intercept_penalty': 2, 'other_coefficient_penalty': 20,
            'feature_count_per_channel': 10,
            'features': ['1', 'raw_channel/1000', '(yesterday_channel-raw_channel)/1000',
                '(previous_week_channel-raw_channel)/1000', '(mean3_channel-raw_channel)/1000',
                'raw_other_channel/1000', 'sin(slot)', 'cos(slot)', 'sin(2slot)', 'cos(2slot)'],
            'target': '(actual_channel-raw_channel)/1000', 'PV_sample_weight': '1 if rawPV>1 or actualPV>1; otherwise0.1, multiplied by age weight',
            'correction_clip_kw': 'plus/minus max(100,3*prior residualRMSE)',
            'postprocess': 'Nonnegative channels and same causal PV mask'},
        'load_memory': {'coefficient': .5, 'formula': 'load_final[d,t]=max(0,load_ridge[d,t]+0.5*mean_t(actual_load[d-1,t]-load_ridge[d-1,t]))',
            'PV_unchanged': True, 'nonrecursive': True, 'residual_basis': 'Previous underlying Ridge28 forecast, not its memory-adjusted output'},
        'cold_start': 'Feb1 has no earlier issued output: calibration/memory pass through. January dispatch residual cold-start uses explicitly labelled periodic forecasts.',
        'timing_evidence': timings, 'development_not_independent_test': True,
        'selection_context': '2025 outputs informed prior development. Final fixed0.5 design was frozen before this candidate evaluation, but2025 is not an untouched independent test set.'}
    save_json(OUT / 'model_configuration.json', model_config)
    return model_config


def verify(records, stages, truth):
    annual = pd.DataFrame(records['annual'])
    for key in ('monthly', 'lead', 'daily'):
        granular = pd.DataFrame(records[key])
        for _, row in annual.iterrows():
            subset = granular[(granular.model == row.model) & (granular.target == row.target) & (granular.population == row.population)]
            for additive in ('n', 'absolute_error_sum_kw', 'squared_error_sum_kw2', 'signed_error_sum_kw', 'actual_absolute_sum_kw'):
                np.testing.assert_allclose(subset[additive].sum(), row[additive], rtol=2e-13, atol=1e-7)
    assert len(records['annual']) == 12 and len(records['monthly']) == 132 and len(records['lead']) == 1728
    assert set(annual[annual.population == 'all'].n) == {48096}
    assert set(annual[annual.population == 'pv_generating'].n) == {26880}
    for model, (directory, archive, _) in MODELS.items():
        path = directory / ('metrics.csv' if model == 'final_blend' else 'annual_metrics.csv')
        published = pd.read_csv(register(path))
        for r in [s for s in stages if s['model'] == model and s['population'] == 'all']:
            old = published[(published.name == r['model_id']) & (published.channel == ('net' if r['target'] == 'net_load' else r['target']))].iloc[0]
            for measure in ('mae_kw', 'rmse_kw', 'bias_kw'):
                np.testing.assert_allclose(old[measure], r[measure], rtol=0, atol=1e-8)
    # Separate arithmetic implementation for every population/target annual value.
    for _, r in annual.iterrows():
        values = load_store(*MODELS[r.model][:2], truth)
        c = TARGETS.index(r.target)
        p, y = channels(values)[:, :, c], channels(truth)[:, :, c]
        use = np.ones(p.shape, bool) if r.population == 'all' else truth[:, :, 1] > 0
        e, actual = (p - y)[use].ravel().tolist(), y[use].ravel().tolist()
        import math
        np.testing.assert_allclose(r.mae_kw, math.fsum(abs(v) for v in e) / len(e), rtol=1e-13)
        np.testing.assert_allclose(r.rmse_kw, math.sqrt(math.fsum(v*v for v in e) / len(e)), rtol=1e-13)
        np.testing.assert_allclose(r.wape_pct, 100 * math.fsum(abs(v) for v in e) / math.fsum(abs(v) for v in actual), rtol=1e-13)
    signed = {}
    for directory in (HGB, ET, BLEND):
        audit = read_json(directory / 'independent_verification.json')
        assert audit['passed']
        read_json(directory / 'causality_verification.json')
        signed[rel(directory)] = {'independent_verification': rel(directory / 'independent_verification.json'),
            'causality_verification': rel(directory / 'causality_verification.json'), 'previous_audit_passed': True}
    for name in ('summary.json', 'independent_audit.json', 'complete_provenance.json',
                 'runtime_source_consistency_before.json', 'runtime_source_consistency_after.json', 'evaluation_protocol.json'):
        register(DISPATCH / name)
    with np.load(register(DISPATCH / 'issued_forecasts.npz')) as saved:
        # Adapter archive layout is recorded as evidence; compare the values when present.
        saved_keys = saved.files
        compare_key = 'values' if 'values' in saved_keys else 'forecast_kw' if 'forecast_kw' in saved_keys else None
        if compare_key is not None:
            np.testing.assert_array_equal(saved[compare_key], load_store(BLEND, MODELS['final_blend'][1], truth))
    return {'passed': True, 'no_training_or_optimization_performed': True,
        'all_annual_metrics_independently_recomputed_with_math_fsum': True,
        'monthly_and_144_leads_and_daily_additive_sums_reconcile_to_annual': True,
        'published_blend_and_HGB_three_stage_metrics_matched': True,
        'raw_half_blend_exact_on_all334_days_both_channels': True,
        'archived_errors_verified_equal_truth_minus_prediction': True,
        'prediction_metric_sign': 'prediction minus truth',
        'all44_raw_model_hashes_verified': True,
        'prior_causality_audits': signed,
        'selected_dispatch_issued_archive_keys': saved_keys,
        'selected_dispatch_issued_forecasts_exact_array_match': compare_key is not None,
        'metric_rows': {k: len(v) for k, v in records.items()},
        'full_period_samples': 48096, 'actual_PV_positive_samples': 26880,
        'point_prediction_intervals_available': False, 'multi_seed_std_available': False,
        'development_not_independent_test': True}


def plot_figures():
    os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/exp008_final_forecast_matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_path = Path('/Library/Fonts/Arial Unicode.ttf')
    assert font_path.exists(), 'A Chinese-capable font is required for report export.'
    font_manager.fontManager.addfont(str(font_path))
    plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False, 'axes.unicode_minus': False,
        'font.size': 11, 'axes.titlesize': 12, 'axes.labelsize': 11, 'svg.fonttype': 'path',
        'axes.edgecolor': '#667085', 'text.color': '#202938', 'axes.labelcolor': '#202938'})
    FIG.mkdir(parents=True, exist_ok=True)
    names = {'final_blend': 'Q2 最终融合', 'single_hgb': '后续问单 HGB'}
    colors = {'final_blend': '#2166AC', 'single_hgb': '#D18F28'}
    target_names = {'load': '负载', 'pv': '光伏', 'net_load': '净负荷'}
    outputs = []
    def finish(fig, filename, title, note):
        fig.suptitle(title, x=.06, ha='left', fontsize=15)
        fig.text(.06, .015, note, ha='left', fontsize=9, color='#475467')
        fig.tight_layout(rect=[0, .07, 1, .93])
        for suffix in ('png', 'svg'):
            p = FIG / (filename + '.' + suffix)
            fig.savefig(p, dpi=180, facecolor='white')
            outputs.append({'path': rel(p), 'sha256': sha(p), 'source_csv': filename})
        plt.close(fig)
    annual = pd.read_csv(OUT / 'annual_metrics.csv')
    monthly = pd.read_csv(OUT / 'monthly_metrics.csv')
    leads = pd.read_csv(OUT / 'lead_metrics.csv')
    energy = pd.read_csv(OUT / 'daily_energy_metrics.csv')
    history = pd.read_csv(OUT / 'history_midnight_comparable.csv')
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.4))
    for ax, field, title in zip(axes, ('mae_kw', 'rmse_kw', 'wape_pct'), ('MAE (kW)', 'RMSE (kW)', 'WAPE (%)')):
        for i, model in enumerate(MODELS):
            rows = annual[(annual.model == model) & (annual.population == 'all')].set_index('target').loc[list(TARGETS)]
            bars = ax.bar(np.arange(3) + (i-.5)*.34, rows[field], .34, color=colors[model], label=names[model],
                          hatch=None if i == 0 else '//', edgecolor='white')
            ax.bar_label(bars, fmt='%.2f', fontsize=8, padding=3)
        ax.set_xticks(np.arange(3), [target_names[t] for t in TARGETS]);ax.set_title(title)
        ax.set_ylim(0, ax.get_ylim()[1]*1.14);ax.grid(axis='y', alpha=.17);ax.set_axisbelow(True)
    axes[0].legend(loc='upper left', fontsize=9)
    finish(fig, 'annual_metrics', '同午夜预测：全期误差比较', '2025-02-01 至12-31，334日、48,096时段；同一实际值评分。开发期结果，无多种子区间。')
    fig, axes = plt.subplots(2, 3, figsize=(12.8, 7), sharex=True)
    for c, target in enumerate(TARGETS):
        for r, field in enumerate(('rmse_kw', 'wape_pct')):
            ax = axes[r, c]
            for i, model in enumerate(MODELS):
                rows = monthly[(monthly.model == model) & (monthly.target == target) & (monthly.population == 'all')]
                ax.plot(rows.month, rows[field], '-o' if i == 0 else '--s', color=colors[model], label=names[model], markersize=4)
            ax.set_title(target_names[target] + (' RMSE (kW)' if r == 0 else ' WAPE (%)'))
            ax.set_xticks(range(2,13));ax.grid(alpha=.17)
            if r == 1: ax.set_xlabel('月份')
    axes[0,0].legend(fontsize=9)
    finish(fig, 'monthly_metrics', '分月误差：保留各月份的升降', '各月独立聚合原始误差；纵轴按不同目标分别设置。全期RMSE不等于各月RMSE均值。')
    fig, axes = plt.subplots(2, 3, figsize=(12.8, 7), sharex=True)
    for c, target in enumerate(TARGETS):
        for r, field in enumerate(('rmse_kw', 'wape_pct')):
            ax = axes[r,c]
            for i, model in enumerate(MODELS):
                rows = leads[(leads.model == model) & (leads.target == target) & (leads.population == 'all')]
                ax.plot(rows.lead_end_inclusive_minutes/60, rows[field], '-' if i == 0 else '--', color=colors[model], label=names[model])
            ax.set_title(target_names[target] + (' RMSE (kW)' if r == 0 else ' WAPE (%)'))
            if r == 1 and target == 'pv':
                ax.set_yscale('log')
                ax.set_title('光伏 WAPE (%) · 对数轴')
            ax.set_xticks([1/6,6,12,18,24], ['0:10','6:00','12:00','18:00','24:00']);ax.grid(alpha=.17)
            if r == 1: ax.set_xlabel('相对00:00发布时间的提前量')
    axes[0,0].legend(fontsize=9)
    finish(fig, 'lead_metrics', '144个提前槽：午夜未来24小时', '每点合并334日同一槽。光伏WAPE用对数轴保留极小实际分母造成的高峰；分母为0时留空，不填0%。')
    fig, axes = plt.subplots(3, 1, figsize=(12.8, 8.4), sharex=True)
    for ax, target in zip(axes,TARGETS):
        for i, model in enumerate(MODELS):
            rows = energy[(energy.model == model) & (energy.target == target)]
            dates = pd.to_datetime(rows.date)
            if i == 0: ax.plot(dates, rows.actual_kwh/1000, color='#333B48', lw=1.4, label='实际')
            ax.plot(dates, rows.predicted_kwh/1000, color=colors[model], lw=1.1, alpha=.9,
                    ls='-' if i == 0 else '--', label=names[model])
        ax.set_ylabel(target_names[target]+' (MWh/日)');ax.grid(alpha=.17)
    ticks = pd.to_datetime(['2025-02-01','2025-04-01','2025-06-01','2025-08-01','2025-10-01','2025-12-31'])
    axes[-1].set_xticks(ticks, ['02-01','04-01','06-01','08-01','10-01','12-31'])
    axes[0].legend(loc='upper right', ncol=3, fontsize=9)
    finish(fig, 'daily_energy_metrics', '全年预测与实际：逐日总电量', '功率逐日求和后除以6换算kWh；图展示334个完整日。不存在预测置信区间，曲线差异不代表显著性检验。')
    fig, axes = plt.subplots(2, 3, figsize=(13.6, 7.4))
    order = [f'exp{i:03}' for i in range(1,7)]+list(MODELS)
    labels = ['01\n重评分','02\n重评分','03','04','05\n原登记','06','08\n融合','08\nHGB']
    for c,target in enumerate(TARGETS):
        for r,field in enumerate(('rmse_kw','wape_pct')):
            ax = axes[r,c]
            rows = history[(history.target == target) & (history.population == 'all')].set_index('model').loc[order]
            values = rows[field].to_numpy()
            palette = ['#A4ADBA']*6 + [colors['final_blend'],colors['single_hgb']]
            bars=ax.bar(np.arange(8), values, color=palette)
            bars[4].set_hatch('//');bars[4].set_edgecolor('#596579')
            ax.bar_label(bars,fmt='%.1f',fontsize=8,padding=3)
            ax.set_xticks(np.arange(8), labels,fontsize=9);ax.set_ylim(0,max(values)*1.2);ax.grid(axis='y',alpha=.17);ax.set_axisbelow(True)
            ax.set_title(target_names[target]+(' RMSE (kW)' if r==0 else ' WAPE (%)'))
    finish(fig, 'history_midnight_comparable', 'exp001—006与exp008：同午夜预测误差', '01/02为已有午夜重评分；05为历史提交原登记的四舍五入值。04/05/06复用同预测；忽略07，非统一训练协议的独立测试。')
    save_json(OUT / 'figure_manifest.json', {'figures': outputs, 'static_export': True,
        'uncertainty_intervals': None, 'font': plt.rcParams['font.family'],
        'source_csvs': {name: sha(OUT / (name + '.csv')) for name in
            ('annual_metrics','monthly_metrics','lead_metrics','daily_energy_metrics','history_midnight_comparable')},
        'visual_inspection': 'Pending separate rendered inspection; source-based arithmetic checks already passed.'})


def run():
    if OUT.exists():
        raise FileExistsError('Report forecast evidence already exists; preserve it and explicitly review any regeneration.')
    OUT.mkdir(parents=True)
    register(Path(__file__))
    data = Data()
    for name, digest in data.hashes.items():
        assert sha(register(ROOT / 'data/raw' / name)) == digest
    truth = data.actual[31*144:365*144, :2].reshape(334,144,2)
    dates = pd.date_range('2025-02-01','2025-12-31')
    records, _ = extract_current(truth, dates)
    stages = extract_stages(truth)
    history_check = historical_evidence(records['annual'])
    config = configuration()
    verification = verify(records, stages, truth)
    verification['historical_comparison'] = history_check
    assert all(sha(ROOT/path) == digest for path,digest in INPUTS.items())
    verification['all_sources_unchanged_before_after_extraction'] = True
    save_json(OUT / 'verification.json', verification)
    save_json(OUT / 'provenance.json', {
        'complete': True, 'producer': rel(Path(__file__)), 'source_sha256': sha(Path(__file__)),
        'sources_sha256': INPUTS, 'raw_data_sha256': data.hashes,
        'scope': '334 midnight24h issues,2025-02-01..12-31; Q3/Q4 other issue times are not scored by this dataset.',
        'models': {k: {'model_id': v[1], 'source': rel(v[0]/(v[1]+'.npz')), 'role': v[2]} for k,v in MODELS.items()},
        'metric_definitions': {'error_sign':'prediction minus actual', 'MAE':'sum(abs(error))/n',
            'RMSE':'sqrt(sum(error**2)/n)', 'WAPE':'100*sum(abs(error))/sum(abs(actual)); null if zero denominator',
            'generation_population':'actual PV>0, retrospective scoring mask only; all3targets use the same mask',
            'lead_slot':'1..144 interval ends00:10..24:00 following00:00 issue; each lead scored across334days',
            'aggregate_rule':'Sum error numerators and sample counts before dividing; never average monthly RMSE/WAPE'},
        'historical_evidence':'Original registry rows kept separately from midnight comparable scores. exp005 uses the immutable Git object or, only if that commit is absent, its explicitly SHA-locked frozen snapshot; no missing values inferred.',
        'model_source_and_prior_audits':'All44models hashed. Existing signed causality audits cited; no fitting or full causality rerun in this reporting step.',
        'runtime_limitations':'Original fit-plus-validation-predict timers and whole pre-bridge wall timers retained. Pure-fit, formal-prediction and postprocessing stage times unmeasured, shown null; no fabricated zero.',
        'accepted_Q2_dispatch':rel(DISPATCH), 'accepted_cost_yuan':13201981.94794217,
        'accepted_reversals':2533, 'cost_reduction_vs_exp006_pct':6.14431756785,
        'user_accepted_result_despite_original_8pct_goal':True, 'new_optimization_performed':False,
        'development_not_independent_test':True,
        'forecast_question_linkage':'Q2 uses blend. Q3/Q4 completed archives use singleHGB; current evidence labels that discrepancy rather than claiming unified blend everywhere.'})
    shutil.copyfile(__file__, OUT/'source_snapshot.py')
    (OUT/'README.md').write_text('''# 最终报告预测证据

本目录只从已冻结的逐日预测重新统计指标，不拟合模型、不选择参数、不运行调度。

`annual_metrics.csv`、`monthly_metrics.csv`、`lead_metrics.csv` 是报告主数据。两种模型分别为 Q2 最终融合与后续问单 HGB；均在午夜发布的未来144槽、2025-02-01至12-31评分。它不代表第三、四问所有其他发布时间的评分。每份文件都有负载、光伏、净负荷，及全天 / 实际PV>0两种总体。`n=0`或实际绝对值分母为0时指标留空，不填零。

`history_midnight_comparable.csv` 区分已有重评分、午夜正式评分、exp005原登记四舍五入值与本次冻结预测的只读评分。exp001/002原登记的多次发布总体不能与午夜评分混用；全部原记录另存 `history_original_registry.csv`。exp005仅保留原登记中真实存在的4项，发电时段load/net为空；来源为固定git提交而非当前main。exp007已排除。

`model_configuration.json` 给出两类树模型、31个特征、超参数、seed42及先raw融合后Ridge28/非递归记忆的顺序。`runtime_evidence.csv` 保留22个原始主模型fit加紧随的验证集predict计时之和及原pre-bridge整体计时；该timer并非纯训练时间，纯fit、正式预测及后处理分阶段耗时均缺失，不把缺失当0。仅有一个种子，标准差不估算。`training_models_and_boundaries.csv`含44模型的原文件hash和逐月训练/验证边界。

所有2025正式期已被用于开发比较，不能称为未触碰的独立测试集。最终Q2费用实际下降6.1443%，原8%门槛已由用户接受当前结果的指令解除；本证据不宣称达到8%。后续问保留各自已完整核验的单HGB模型，不能把Q2融合收益直接外推至Q3/Q4。

复现：`.venv/bin/python -m experiments.exp008.final_forecast_evidence`。脚本拒绝覆盖已有目录；先审查已有证据再决定是否另存重生成。图为PNG/SVG，直接读取本目录CSV。
''')
    plot_figures()
    save_json(OUT/'output_manifest.json', {'files_sha256':{p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.name!='output_manifest.json'},
        'figures_manifest':'figure_manifest.json', 'all_inputs_unchanged':all(sha(ROOT/path)==digest for path,digest in INPUTS.items())})
    print(json.dumps({'output':str(OUT),'verified':verification['passed'],'annual':records['annual'],
        'recorded_fit_plus_validation_predict_seconds':{k:sum(r['seconds'] for r in config['timing_evidence'] if r['family']==k and r['stage']=='22 primary fit plus validation-predict blocks') for k in ('HGB','ExtraTrees')}},ensure_ascii=False))


if __name__ == '__main__':
    run()
