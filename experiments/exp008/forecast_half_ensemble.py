"""One signed equal-weight ensemble of two complete forecast pipelines.

Weights remain .5/.5. No fitting or post-blend recalibration. The single
LP bridge rebuilds tree28 history from the fused issued predictions.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from experiments.exp008.forecast import Forecasts
from experiments.exp008.forecast_absolute_hgb import AbsoluteHGBStore, ArrayStore
from experiments.exp008.forecast_calibration import CalibratedStore
from experiments.exp008.forecast_override_adapter import AuditedForecastOverride
from experiments.exp008.load_energy_memory import EnergyMemoryStore
from experiments.exp008.neural_joint_calibration import JointStore
from experiments.exp008.risk_window import WindowResidualScenarios, SPEC, replay
from experiments.exp008.verify import ETA, verify_npz
from experiments.problem2.exp003.data import Data, ROOT

OUT = ROOT / 'data/results/exp008/forecast_half_ensemble'
DIAGNOSTIC = ROOT / 'data/results/exp008/forecast_ensemble_diagnostic'
HGB = ROOT / 'data/results/exp008/forecast_absolute_hgb'
MEMORY = ROOT / 'data/results/exp008/load_energy_memory'
MODEL_ID = 'fixed_half_absolute_HGB_memory_plus_half_joint_CNN_memory'
BASELINE = HGB / 'lp_bridge/direct_hgb_ridge28_memory_tree28_q08_buffer500_334days'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_hash(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


class EnsembleStore:
    def __init__(self):
        protocol = json.loads((DIAGNOSTIC / 'protocol.json').read_text())
        diagnostic = json.loads((DIAGNOSTIC / 'summary.json').read_text())
        if protocol['fixed_weight_hgb'] != .5 or protocol['fixed_weight_joint'] != .5:
            raise ValueError('Only the previously fixed equal mean is authorized')
        hgb = AbsoluteHGBStore()
        path = MEMORY / 'predictions.npz'
        memory_protocol = json.loads((MEMORY / 'protocol.json').read_text())
        if digest(path) != memory_protocol['archive_sha256']:
            raise ValueError('Joint-memory input archive hash changed')
        with np.load(path) as z:
            joint_values, joint_origins = z['values'].copy(), z['origins'].copy()
        with np.load(DIAGNOSTIC / 'predictions.npz') as z:
            self.values, self.origins = z['values'].copy(), z['origins'].copy()
        np.testing.assert_array_equal(hgb.origins, joint_origins)
        np.testing.assert_array_equal(self.origins, hgb.origins)
        np.testing.assert_array_equal(self.origins, np.arange(31, 365) * 144)
        if array_hash(hgb.values) != diagnostic['input_value_hashes']['hgb']:
            raise ValueError('HGB values differ from the predeclared fusion input')
        if array_hash(joint_values) != diagnostic['input_value_hashes']['joint']:
            raise ValueError('Joint-memory values differ from the predeclared fusion input')
        expected = .5 * hgb.values + .5 * joint_values
        np.testing.assert_array_equal(self.values, expected)
        assert self.values.shape == (334, 144, 2) and np.isfinite(self.values).all() and np.min(self.values) >= 0
        self.lookup = {int(origin): i for i, origin in enumerate(self.origins)}
        self.name = MODEL_ID
        self.identity = {'model_id': MODEL_ID, 'weights': {'hgb': .5, 'joint': .5},
            'post_blend_calibration': None, 'prediction_values_sha256': array_hash(self.values),
            'prediction_origins_sha256': array_hash(self.origins),
            'input_value_hashes': diagnostic['input_value_hashes'],
            'source_archive_sha256': {str(p): digest(p) for p in
                [HGB / 'direct_hgb_ridge28_memory_half.npz', path, DIAGNOSTIC / 'predictions.npz']},
            'common_334_midnight_origins_verified': True, 'all_slot_mean_equalities_exact': True}

    def get(self, origin):
        return self.values[self.lookup[int(origin)]].copy()


class EnsembleForecasts(Forecasts):
    def __init__(self, data=None):
        super().__init__(data=data, seed=42)
        self.store = EnsembleStore()
        self.calibration = 'none_after_equal_mean_of_two_complete_ridge28_memory_pipelines'

    def get(self, day, slot=0, scenario='2'):
        result = super().get(day, slot, scenario)
        result['audit'].update(base_forecast=MODEL_ID if day >= 31 else 'periodic_cold_start',
            output_calibration=self.calibration if day >= 31 else None,
            selected_model_id=MODEL_ID, ensemble_weights={'hgb': .5, 'joint': .5},
            base_architecture_unchanged=day < 31, input_pipeline_value_hashes=self.store.identity['input_value_hashes'])
        return result

    def net_error_paths(self, day, scenario='2', limit=28):
        result = super().net_error_paths(day, scenario, limit)
        result['audit'].update(source='same_half_ensemble_issued_forecasts_with_labelled_January_periodic_cold_start',
                               model_id=MODEL_ID, historical_predictions_are_fused=True)
        return result


def audited_forecasts():
    return AuditedForecastOverride(EnsembleForecasts(), MODEL_ID,
        [DIAGNOSTIC / 'protocol.json', DIAGNOSTIC / 'summary.json', DIAGNOSTIC / 'predictions.npz',
         HGB / 'direct_hgb_ridge28_memory_half.npz', MEMORY / 'predictions.npz'])


def fixed_protocol(store):
    if OUT.exists():
        raise FileExistsError('Keep the fixed ensemble experiment immutable')
    OUT.mkdir(parents=True)
    sources = ['forecast_half_ensemble', 'forecast_absolute_hgb', 'load_energy_memory',
        'forecast_calibration', 'neural_joint_calibration', 'forecast', 'forecast_override_adapter',
        'risk_window', 'controller_candidate', 'verify']
    hashes = {}
    for name in sources:
        source = ROOT / f'experiments/exp008/{name}.py'
        target = OUT / 'sources' / source.name
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(source, target)
        hashes[str(source.relative_to(ROOT))] = digest(source)
    evidence = [DIAGNOSTIC / 'protocol.json', DIAGNOSTIC / 'summary.json',
        HGB / 'protocol.json', HGB / 'provenance.json', HGB / 'causality_verification.json',
        MEMORY / 'protocol.json', MEMORY / 'verification.json',
        ROOT / 'data/results/exp008/neural_joint_calibration/manifest.json']
    for source in evidence:
        target = OUT / 'input_evidence' / source.relative_to(ROOT / 'data/results/exp008')
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    result = {'model_id': MODEL_ID, 'candidate_count': 1, 'fit_or_weight_search': False,
        'weights': {'hgb': .5, 'joint': .5}, 'post_blend_calibration': None,
        'predeclared_diagnostic': str(DIAGNOSTIC), 'input_and_mean_verification': store.identity,
        'same_tree28_q08_buffer500_parameters': {**SPEC, 'calibration': MODEL_ID},
        'baseline': str(BASELINE), 'full_evaluation_days': 334,
        'each_policy_carries_own_realized_soc': True, 'initial_soc_kwh': 1421.7991105135516,
        'historical_risk_rebuilt_from_fused_prequential_values': True,
        'prior_risk_cache_used': False, 'source_sha256': hashes, 'raw_data_hashes': Data().hashes,
        'development_year_not_independent_test': True, 'not_final_model_selection': True}
    save(OUT / 'protocol.json', result)
    np.savez_compressed(OUT / 'issued_forecasts.npz', origins=store.origins, values=store.values)
    return result


def recompute_complete_pipeline(data, raw_store):
    ridge = CalibratedStore('ridge_28', data=data, use_cache=False, _base_store=raw_store,
                            directory=OUT / 'unused_cache')
    return EnergyMemoryStore(data=data, base_store=ridge).values


def causal_checks(store):
    data = Data()
    hgb_raw, joint_raw = AbsoluteHGBStore('direct_hgb_raw'), JointStore()
    # Verify whole components before mutating actuals or future raw forecasts.
    rebuilt = .5 * recompute_complete_pipeline(data, hgb_raw) + .5 * recompute_complete_pipeline(data, joint_raw)
    np.testing.assert_array_equal(rebuilt, store.values)
    risk = WindowResidualScenarios(store.origins, store.values, data.actual, data.fixed_price)
    rows = []
    for day in (31, 32, 90, 243, 364):
        cutoff = day * 144
        altered = copy.copy(data)
        altered.actual = data.actual.copy()
        altered.actual[cutoff:] += np.array([60000., 40000.])
        shifted = []
        for raw in (hgb_raw, joint_raw):
            values = raw.values.copy()
            values[raw.origins > cutoff] += 80000.
            shifted.append(ArrayStore(values, raw.origins, 'counterfactual_future_only'))
        changed = .5 * recompute_complete_pipeline(altered, shifted[0]) + .5 * recompute_complete_pipeline(altered, shifted[1])
        ids = store.origins <= cutoff
        np.testing.assert_array_equal(store.values[ids], changed[ids])
        changed_risk = WindowResidualScenarios(store.origins, changed, altered.actual, data.fixed_price)
        a, audit = risk.for_day(day)
        b, changed_audit = changed_risk.for_day(day)
        np.testing.assert_array_equal(a, b)
        assert audit['max_observed_index'] < cutoff
        assert all(origin + 144 <= cutoff for origin in audit['training_origins'])
        rows.append({'day': day, 'cutoff': cutoff, 'both_Ridge28_and_memory_pipelines_recomputed': True,
            'current_future_actual_and_future_raw_predictions_changed': True,
            'fused_forecast_prefix_max_error_kw': 0., 'tree_support_max_error_kwh': 0.,
            'labels_stop_exclusive_max': audit['max_observed_index'] + 1, 'periodic_fallback': audit['fallback']})
    # Separate fresh issue adapters ensure caches cannot hide intraday reads.
    issued = []
    for day, slot, scenario in ((31, 0, '2'), (90, 36, '3'), (243, 108, '4-3')):
        base, altered = EnsembleForecasts(), EnsembleForecasts()
        cutoff = day * 144 + slot
        altered.data = copy.copy(altered.data)
        altered.data.actual = altered.data.actual.copy()
        altered.data.actual[cutoff:] += np.array([60000., 40000., 1000.])
        altered.data._forecasts = {k: np.asarray(v).copy() + (70000. if k > cutoff else 0.)
                                  for k, v in altered.data.forecasts.items()}
        f0, f1 = base.get(day, slot, scenario), altered.get(day, slot, scenario)
        for key in ('load_kw', 'pv_kw', 'price'):
            np.testing.assert_array_equal(f0[key], f1[key])
        assert not f1['audit']['known_future_price']
        issued.append({'day': day, 'slot': slot, 'scenario': scenario,
            'actual_all_channels_and_unreleased_PV_mutation_output_unchanged': True})
    result = {'passed': True, 'complete_component_recomputation_matches_input_mean': True,
        'future_prefix_and_tree_support_checks': rows, 'fresh_issue_adapter_checks': issued,
        'upstream_raw_model_fitting_causality': 'use copied HGB and joint CNN independent training/feature perturbation evidence; no upstream model retraining in ensemble task'}
    save(OUT / 'causality_verification.json', result)
    return result


def bridge(store):
    data = Data()
    name = 'half_ensemble_tree28_q08_buffer500'
    result = replay(name, 28, 'tree', 334, data, store, OUT / 'lp_bridge')
    directory = OUT / 'lp_bridge' / f'{name}_334days'
    audit = json.loads((directory / 'audit.json').read_text())
    for row in audit:
        row['forecast_calibration'] = MODEL_ID
        row['residual_source'] = 'periodic_baseline' if row['fallback'] else 'half_ensemble_prequential_forecast'
        row['input_pipeline_value_hashes'] = store.identity['input_value_hashes']
        row['fused_forecast_values_sha256'] = store.identity['prediction_values_sha256']
    save(directory / 'audit.json', audit)
    checked = verify_npz(directory / 'dispatch_2.npz', audit_path=directory / 'audit.json')
    assert checked['passed'], checked['errors']
    save(directory / 'independent_verification.json', checked)
    result.update(fixed_spec={**SPEC, 'calibration': MODEL_ID},
        evaluation_role='single fixed mean forecast-only LP bridge; not a final action target certificate',
        forecast_values_sha256=store.identity['prediction_values_sha256'])
    save(directory / 'summary.json', result)
    baseline = verify_npz(BASELINE / 'dispatch_2.npz', audit_path=BASELINE / 'audit.json')
    assert baseline['passed'], baseline['errors']
    save(OUT / 'baseline_independent_verification.json', baseline)
    old_cost, cost = baseline['recomputed_total_cost'], checked['recomputed_total_cost']
    old_battery, battery = baseline['battery_metrics'], checked['battery_metrics']
    depleted = max(0., old_battery['final_soc'] - battery['final_soc'])
    terminal_adjustment = depleted * ETA * 5 * float(data.fixed_price.max())
    summary = {'days': 334, 'model_id': MODEL_ID,
        'baseline': {'billing': baseline['billing'], 'battery': old_battery},
        'ensemble': {'billing': checked['billing'], 'battery': battery},
        'cost_change_yuan': cost - old_cost, 'cost_change_pct': 100 * (cost / old_cost - 1),
        'fee_component_changes': {key: checked['billing'][key] - baseline['billing'][key] for key in baseline['billing']},
        'direction_reversal_change': battery['direction_reversals'] - old_battery['direction_reversals'],
        'active_slot_change': battery['active_slots'] - old_battery['active_slots'],
        'throughput_change_kwh': battery['throughput_kwh'] - old_battery['throughput_kwh'],
        'final_soc_change_kwh': battery['final_soc'] - old_battery['final_soc'],
        'conservative_lower_terminal_soc_cost_yuan': terminal_adjustment,
        'cost_saving_after_conservative_terminal_adjustment_yuan': old_cost - cost - terminal_adjustment,
        'both_independently_verified': True, 'no_weight_search_or_recalibration': True,
        'worth_followup_physical_planning_on_cost_and_intensity': bool(cost + terminal_adjustment < old_cost
            and battery['direction_reversals'] <= old_battery['direction_reversals']
            and battery['throughput_kwh'] <= old_battery['throughput_kwh']),
        'followup_decision_scope': 'evidence-based diagnostic recommendation only; no physical planner run in this task',
        'Q2_goal': checked['goal'], 'final_model_selected': False}
    save(OUT / 'summary.json', summary)
    pd.DataFrame([{'model': label, **check['billing'], **check['battery_metrics']}
                  for label, check in [('absolute_hgb_memory', baseline), ('half_ensemble', checked)]]).to_csv(OUT / 'comparison.csv', index=False)
    old_daily, new_daily = pd.read_csv(BASELINE / 'daily.csv'), pd.read_csv(directory / 'daily.csv')
    joined = old_daily[['day', 'date', 'total_cost', 'planned_cost', 'emergency_cost']].merge(
        new_daily[['day', 'total_cost', 'planned_cost', 'emergency_cost']], on='day', suffixes=('_hgb', '_ensemble'))
    for key in ('total_cost', 'planned_cost', 'emergency_cost'):
        joined[key + '_change'] = joined[key + '_ensemble'] - joined[key + '_hgb']
    joined.to_csv(OUT / 'daily_comparison.csv', index=False)
    joined['month'] = pd.to_datetime(joined['date']).dt.month
    joined.groupby('month').sum(numeric_only=True).drop(columns='day').to_csv(OUT / 'monthly_comparison.csv')
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main():
    store = EnsembleStore()
    protocol = fixed_protocol(store)
    causal_checks(store)
    bridge(store)
    assert all(digest(ROOT / path) == expected for path, expected in protocol['source_sha256'].items())
    assert all(digest(path) == expected for path, expected in store.identity['source_archive_sha256'].items())
    save(OUT / 'source_verification.json', {'passed': True, 'all_input_archives_and_source_files_unchanged': True})


if __name__ == '__main__':
    main()
