"""Bounded three-day Q3/Q4-3 forecast-interface regression and linkage pilot.

Uses the explicit pre-edit run.py snapshot to test exact default parity.
No annual run or data-dependent hyperparameter choice is performed.
"""
import copy
import hashlib
import json
from pathlib import Path
import types
import unittest

import numpy as np

from experiments.exp008 import run as current
from experiments.exp008.forecast_override_adapter import AuditedForecastOverride
from experiments.exp008.issued_residual_paths import issued_error_paths
from experiments.exp008.neural_joint_dispatch import JointForecasts
from experiments.exp008.planner import Settings
from experiments.exp008.unified_forecast import UnifiedForecasts
from experiments.exp008.verify import verify_npz

ROOT_OUT = current.OUT / 'forecast_override_pilot'
OUT = ROOT_OUT / 'v2'
CASE = 'forecast_override_pilot/v2'
MODEL_ID = 'exp008_joint_shared_cnn_seed42_then_fixed_ridge28'
SOURCE_PATHS = [current.ROOT / 'data/results/exp008/neural_joint_calibration/joint_ridge28.npz',
                current.ROOT / 'data/results/exp008/neural_joint_calibration/manifest.json',
                current.ROOT / 'data/results/exp008/neural_joint/manifest.json']


def joint_override():
    return AuditedForecastOverride(JointForecasts(), MODEL_ID, SOURCE_PATHS)


def load_legacy():
    source = (ROOT_OUT / 'run_before_override.py').read_bytes()
    saved = json.loads((ROOT_OUT / 'before_source.json').read_text())
    assert hashlib.sha256(source).hexdigest() == saved['sha256']
    module = types.ModuleType('experiments.exp008._run_before_forecast_override')
    module.__package__ = 'experiments.exp008'
    # Keep original path-based ROOT/import semantics, while executing the
    # immutable snapshot's source rather than importing modified run.py.
    module.__file__ = str(Path(current.__file__).resolve())
    exec(compile(source, str(ROOT_OUT / 'run_before_override.py'), 'exec'), module.__dict__)
    return module


def arrays(case, scenario):
    with np.load(current.OUT / case / scenario / f'dispatch_{scenario}.npz') as z:
        return {key: z[key].copy() for key in z.files}


class ForecastOverrideTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        legacy = load_legacy()
        cls.results, cls.parity, cls.forecast_checks, cls.pollution_checks = {}, [], [], []
        settings = Settings(future_shortfall_weight=1.5)
        for scenario in ('3', '4-3'):
            for name, module, override in (
                ('legacy_default', legacy, None), ('new_default', current, None),
                ('joint_override', current, joint_override())):
                case = f'{CASE}/{name}'
                kwargs = dict(method='joint', settings=settings, deadband=20.,
                              days=3, updates=True, calibration=None, issued_residuals=True)
                if override is not None:
                    kwargs['forecast_override'] = override
                completion = module.run_case(case, scenario, **kwargs)
                directory = current.OUT / case / scenario
                soc, power = current.initial_state(scenario)
                check = verify_npz(directory / f'dispatch_{scenario}.npz', scenario, expected_days=3,
                    initial_soc=soc, initial_power_kw=power, initial_mode=int(np.sign(power)),
                    audit_path=directory / 'audit.json')
                if not check['passed']:
                    raise AssertionError(check['errors'])
                (directory / 'verification.json').write_text(json.dumps(check, indent=2) + '\n')
                cls.results[(name, scenario)] = {'completion': completion, 'verification': check}

    def test_default_full_archive_strict_parity(self):
        for scenario in ('3', '4-3'):
            old, new = arrays(f'{CASE}/legacy_default', scenario), arrays(f'{CASE}/new_default', scenario)
            self.assertEqual(set(old), set(new))
            for key in old:
                np.testing.assert_array_equal(old[key], new[key], err_msg=f'{scenario}/{key}')
            self.parity.append({'scenario': scenario, 'days': 3, 'strictly_equal_arrays': sorted(old),
                                'max_absolute_difference': 0.})
            self.assertNotIn('forecast_override', self.results[('new_default', scenario)]['completion']['config'])

    def test_existing_update_value_case_prefix_parity(self):
        for scenario in ('3', '4-3'):
            current_values = arrays(f'{CASE}/new_default', scenario)
            frozen = current.OUT / 'update_value_diagnostic/future1.5_d334_issued' / scenario / f'dispatch_{scenario}.npz'
            with np.load(frozen) as archived:
                for key in current_values:
                    np.testing.assert_array_equal(current_values[key], archived[key][:3],
                        err_msg=f'existing update_value case {scenario}/{key}')

    def test_same_issue_predictions_and_history_switch_together(self):
        base, joint = UnifiedForecasts(), joint_override()
        for scenario in ('3', '4-3'):
            for day in (31, 32, 33):
                for slot in (0, 36, 72, 108):
                    f0, f1 = base.get(day, slot, scenario), joint.get(day, slot, scenario)
                    r0 = issued_error_paths(base, day, slot, scenario=scenario)
                    r1 = issued_error_paths(joint, day, slot, scenario=scenario)
                    self.assertGreater(float(np.max(np.abs(f0['load_kw'] - f1['load_kw']))), 0.)
                    # Official PV and causal price mechanisms stay unchanged.
                    np.testing.assert_array_equal(f0['pv_kw'], f1['pv_kw'])
                    np.testing.assert_array_equal(f0['price'], f1['price'])
                    np.testing.assert_array_equal(r0['origins'], r1['origins'])
                    self.assertTrue(np.all(r1['label_stops_exclusive'] <= day * 144))
                    reconstructed = []
                    for origin in r1['origins']:
                        old = int(origin // 144)
                        f = joint.get(old, slot, scenario)
                        truth = joint._observed(int(origin), (old + 1) * 144, day * 144 + slot)
                        reconstructed.append((truth[:, 0] - truth[:, 1] - f['load_kw'] + f['pv_kw']) / 6)
                    np.testing.assert_allclose(r1['errors_kwh'], reconstructed, atol=1e-12, rtol=0)
                    delta = float(np.max(np.abs(r0['errors_kwh'] - r1['errors_kwh'])))
                    if day == 31:
                        self.assertEqual(delta, 0.)  # all January periodic cold start
                    else:
                        self.assertGreater(delta, 0.)
                    self.forecast_checks.append({'scenario': scenario, 'day': day, 'slot': slot,
                        'load_forecast_max_change_kw': float(np.max(np.abs(f0['load_kw'] - f1['load_kw']))),
                        'historical_net_error_max_change_kwh': delta,
                        'historical_label_stop_max': int(r1['label_stops_exclusive'].max()),
                        'cutoff': day * 144 + slot, 'official_pv_and_price_unchanged': True,
                        'historical_residuals_reconstructed_from_same_model': True})

    def test_future_actual_and_unreleased_pv_mutation(self):
        # Fresh adapters prevent memoized values from hiding an illegal read.
        for scenario in ('3', '4-3'):
            for day, slot in ((31, 0), (31, 36), (32, 72), (33, 108)):
                original, changed = joint_override(), joint_override()
                cutoff = day * 144 + slot
                changed.forecasts.data = copy.copy(changed.forecasts.data)
                changed.forecasts.data.actual = changed.forecasts.data.actual.copy()
                changed.forecasts.data.actual[cutoff:] += np.array([100000., 50000., 1000.])
                changed.forecasts.data._forecasts = {k: np.asarray(v).copy() + (99999. if k > cutoff else 0.)
                                                  for k, v in changed.forecasts.data.forecasts.items()}
                f0, f1 = original.get(day, slot, scenario), changed.get(day, slot, scenario)
                r0 = issued_error_paths(original, day, slot, scenario=scenario)
                r1 = issued_error_paths(changed, day, slot, scenario=scenario)
                for key in ('load_kw', 'pv_kw', 'price'):
                    np.testing.assert_array_equal(f0[key], f1[key])
                for key in ('errors_kwh', 'errors_kw', 'price_errors', 'origins', 'label_stops_exclusive'):
                    np.testing.assert_array_equal(r0[key], r1[key])
                self.assertFalse(f1['audit']['known_future_price'])
                self.assertLess(f1['audit']['max_observed_index'], cutoff)
                self.assertLess(r1['audit']['max_observed_index'], cutoff)
                self.pollution_checks.append({'scenario': scenario, 'day': day, 'slot': slot,
                    'future_actual_all_channels_changed': True, 'unreleased_official_pv_changed': True,
                    'current_predictions_and_same_issue_history_strictly_unchanged': True})

    def test_identity_and_cache_isolation(self):
        a = joint_override()
        identity = current.forecast_identity(a)
        self.assertEqual(identity['model_id'], MODEL_ID)
        index = (0, 0, 0)
        before = float(a.store.values[index])
        a.store.values[index] += 1.
        with self.assertRaisesRegex(ValueError, 'prediction_values_sha256'):
            current.forecast_identity(a)
        a.store.values[index] = before
        with self.assertRaises(TypeError):
            current.forecast_identity(JointForecasts())
        different_id = AuditedForecastOverride(JointForecasts(), MODEL_ID + '_distinct_identity', SOURCE_PATHS)
        with self.assertRaisesRegex(RuntimeError, 'Refusing to overwrite changed experiment'):
            current.run_case(f'{CASE}/joint_override', '3', settings=Settings(future_shortfall_weight=1.5),
                days=3, issued_residuals=True, forecast_override=different_id)
        for scenario in ('3', '4-3'):
            directory = OUT / 'joint_override' / scenario
            completion = self.results[('joint_override', scenario)]['completion']
            self.assertEqual(completion['config']['forecast_override']['model_id'], MODEL_ID)
            audits = json.loads((directory / 'audit.json').read_text())
            self.assertEqual(len(audits), 12)
            for row in audits:
                self.assertEqual(row['residual_paths']['model_id'], MODEL_ID)
                self.assertNotIn('frozen_cnn', row['residual_paths']['source'])
                self.assertEqual(row['forecast']['selected_model_id'], MODEL_ID)
                self.assertEqual(set(row['issued_prediction_sha256']), {'load_kw', 'pv_kw', 'price'})
                self.assertIn('errors_kwh_sha256', row['residual_paths'])
                self.assertTrue(row['residual_paths']['historical_predictions_use_same_override'])


def save_summary(result):
    paired = []
    for scenario in ('3', '4-3'):
        baseline = ForecastOverrideTests.results[('new_default', scenario)]['verification']
        joint = ForecastOverrideTests.results[('joint_override', scenario)]['verification']
        paired.append({'scenario': scenario, 'days': 3,
            'baseline': {'billing': baseline['billing'], 'battery': baseline['battery_metrics']},
            'joint': {'billing': joint['billing'], 'battery': joint['battery_metrics']},
            'cost_change_yuan': joint['recomputed_total_cost'] - baseline['recomputed_total_cost'],
            'relative_cost_change_pct': 100 * (joint['recomputed_total_cost'] / baseline['recomputed_total_cost'] - 1),
            'both_independently_verified': baseline['passed'] and joint['passed']})
    summary = {'days': 3, 'scenarios': ['3', '4-3'], 'tests_run': result.testsRun,
        'all_tests_passed': result.wasSuccessful(), 'paired_results': paired,
        'default_old_vs_new_strict_parity': ForecastOverrideTests.parity,
        'same_issue_model_switch_checks': ForecastOverrideTests.forecast_checks,
        'future_mutation_checks': ForecastOverrideTests.pollution_checks,
        'configuration': {'issued_residuals': True, 'future_shortfall_weight': 1.5,
            'settings': current.asdict(Settings(future_shortfall_weight=1.5)), 'deadband': 20.,
            'updates': True, 'baseline_calibration': None, 'override': MODEL_ID},
        'same_hyperparameters_as_update_value_future1_5_issued': True,
        'baseline_all_arrays_match_existing_update_value_case_first_three_days': True,
        'full_year_run_performed': False, 'final_model_selected': False,
        'causality_scope': 'adapter future actual/unreleased-PV invariance and same-issue historical labels; monthly network training and Ridge28 causality use their separate source manifests'}
    (OUT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'passed': result.wasSuccessful(), 'paired': paired}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ForecastOverrideTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if ForecastOverrideTests.results:
        save_summary(result)
    raise SystemExit(0 if result.wasSuccessful() else 1)
