"""Independent archive replay and causal-boundary checks for fixed ExtraTrees."""
import json
import unittest

import joblib
import numpy as np
import pandas as pd

from experiments.exp008.forecast_absolute_extra_trees import (
    OUT, BASE, COMPOSITE, RAW, RIDGE, PRIMARY, MODEL_CONFIG, NUMERIC_ATOL_KW,
    features_for_day, array_hash, digest, tree_hashes, source_checks,
    AbsoluteExtraTreesStore,
)
from experiments.exp008.forecast_calibration import _calibrate, CONFIG
from experiments.problem2.exp003.data import Data
from experiments.problem2.exp004.data import age_weights


class ExtraTreesVerification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = Data()
        cls.protocol = json.loads((OUT / 'protocol.json').read_text())
        cls.training = json.loads((OUT / 'training_audit.json').read_text())
        cls.stores = {name: AbsoluteExtraTreesStore(name) for name in (RAW, RIDGE, PRIMARY)}
        cls.features = np.stack([features_for_day(cls.data, d)[0] for d in range(7, 365)])
        cls.evidence = {}

    def test_a_signed_sources_exact_features_and_fixed_parameters(self):
        source_checks(self.protocol)
        baseline = json.loads((BASE / 'provenance.json').read_text())
        self.assertEqual(array_hash(self.features), baseline['feature_tensor_sha256'])
        self.assertEqual(self.protocol['model_configuration'], MODEL_CONFIG)
        self.assertTrue(json.loads((OUT / 'causality_verification.json').read_text())['passed'])
        for store in self.stores.values():
            np.testing.assert_array_equal(store.origins, np.arange(31, 365)*144)
            self.assertEqual(store.values.shape, (334, 144, 2))
            self.assertTrue(np.isfinite(store.values).all())
            self.assertGreaterEqual(float(store.values.min()), 0)
        self.evidence['signed_sources_and_exact31_feature_tensor'] = True

    def test_b_independent_dates_labels_weights_and_all_month_model_reload(self):
        maximum_difference = 0.
        for row in self.training:
            month = row['month']
            first = (pd.Timestamp(2025, month, 1)-pd.Timestamp('2025-01-01')).days
            stop = (pd.Timestamp(2025, month, 1)+pd.offsets.MonthBegin(1)-pd.Timestamp('2025-01-01')).days
            train, val, formal = np.arange(7, first-7), np.arange(first-7, first), np.arange(first, stop)
            self.assertEqual(train.tolist(), row['training_days'])
            self.assertEqual(val.tolist(), row['validation_days'])
            self.assertEqual(formal.tolist(), row['formal_days'])
            self.assertEqual(row['train_label_stop_exclusive'], (first-7)*144)
            self.assertEqual(row['validation_label_stop_exclusive'], first*144)
            for prefix, days in [('training', train), ('validation', val)]:
                x = self.features[days-7].reshape(-1, 31)
                y = self.data.actual[days[:, None]*144+np.arange(144), :2].reshape(-1, 2)
                self.assertEqual(array_hash(x), row[prefix+'_feature_hash'])
                self.assertEqual(array_hash(y), row[prefix+'_label_hash'])
            weights = np.repeat(age_weights(train, first-7, 90), 144)
            self.assertEqual(array_hash(weights), row['training_weights_hash'])
            predicted = []
            for model_audit in row['models']:
                path = model_audit['model_path']
                self.assertEqual(digest(path), model_audit['model_sha256'])
                model = joblib.load(path)
                for key, value in MODEL_CONFIG.items():
                    self.assertEqual(model.get_params()[key], value)
                self.assertEqual(tree_hashes(model), model_audit['tree_hashes'])
                predicted.append(model.predict(self.features[formal-7].reshape(-1, 31)).reshape(-1,144))
            predicted = np.maximum(np.stack(predicted, axis=-1), 0.)
            # Independently implement historical union and two-slot expansion.
            for i, day in enumerate(formal):
                pv = self.data.actual[max(0, day-28)*144:day*144,1].reshape(-1,144)
                daymask = (pv>0).any(axis=0)
                mask = daymask.copy()
                for shift in (-2,-1,1,2):
                    if shift < 0:
                        mask[:shift] |= daymask[-shift:]
                    else:
                        mask[shift:] |= daymask[:-shift]
                predicted[i,:,1] *= mask
            expected = self.stores[RAW].values[formal-31]
            difference = float(np.max(np.abs(predicted-expected)))
            maximum_difference = max(maximum_difference, difference)
            np.testing.assert_allclose(predicted, expected, rtol=0, atol=NUMERIC_ATOL_KW)
        self.evidence['all22_model_hashes_parameters_structures_and334_raw_predictions'] = True
        self.evidence['independent_all_month_reload_max_difference_kw'] = maximum_difference

    def test_c_all334_calibration_memory_and_issued_errors(self):
        raw, ridge, memory = [self.stores[name] for name in (RAW, RIDGE, PRIMARY)]
        for i in range(334):
            calibrated, audit = _calibrate(self.data.actual, raw.origins, raw.values, i, CONFIG['ridge_28'])
            np.testing.assert_array_equal(calibrated, ridge.values[i])
            expected = ridge.values[i].copy()
            if i:
                previous = int(raw.origins[i-1])
                correction = .5*np.mean(self.data.actual[previous:previous+144,0]-ridge.values[i-1,:,0])
                expected[:,0] = np.maximum(expected[:,0]+correction,0.)
            np.testing.assert_array_equal(expected, memory.values[i])
            self.assertTrue(audit['history_last_label'] is None or audit['history_last_label'] < int(raw.origins[i]))
        for name, store in self.stores.items():
            truth = self.data.actual[store.origins[:,None]+np.arange(144),:2]
            with np.load(OUT/f'{name}.npz') as z:
                np.testing.assert_array_equal(z['errors_kw'], truth-store.values)
        self.evidence['all334_Ridge28_and_independent_nonrecursive_half_memory'] = True
        self.evidence['all_issued_prediction_errors_reconstruct_exactly'] = True

    def test_d_selected_future_mutations_and_independent_gate_metrics(self):
        for day in (31,32,59,90,151,243,364):
            changed = type('DataView', (), {})()
            changed.actual = self.data.actual.copy()
            changed.actual[day*144:] += [50000.,30000.]
            np.testing.assert_array_equal(features_for_day(self.data,day)[0],features_for_day(changed,day)[0])
        gate = json.loads((OUT/'dispatch_gate.json').read_text())
        comparison_paths = {'candidate': OUT/f'{PRIMARY}.npz',
            'original_HGB': BASE/'direct_hgb_ridge28_memory_half.npz',
            'best_fixed_channel_composite': COMPOSITE/'memory.npz'}
        for label,path in comparison_paths.items():
            with np.load(path) as z:
                truth=self.data.actual[z['origins'][:,None]+np.arange(144),:2]
                difference=z['values']-truth
                error=difference[:,:,0]-difference[:,:,1]
            actual = {'rmse_kw':float(np.sqrt(np.mean(error*error))),
                'high_price_rmse_kw':float(np.sqrt(np.mean(error[:,self.data.fixed_price>=np.quantile(self.data.fixed_price,.75)]**2))),
                'daily_energy_rmse_kwh':float(np.sqrt(np.mean((error.sum(axis=1)/6)**2))),
                'cumulative_error_rmse_kwh':float(np.sqrt(np.mean((error.cumsum(axis=1)/6)**2)))}
            for key,value in actual.items():
                self.assertAlmostEqual(value,gate[label][key],places=8)
        expected=all(gate['candidate'][key]<gate['original_HGB'][key] for key in gate['metric_checks_vs_original_HGB'])
        self.assertEqual(expected,gate['passed'])
        self.evidence['selected_future_features_unchanged_and_all_four_gate_metrics_independently_recomputed']=True


if __name__ == '__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(ExtraTreesVerification)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    record={'passed':result.wasSuccessful(),'tests':result.testsRun,
        'failures':len(result.failures),'errors':len(result.errors),
        'evidence':getattr(ExtraTreesVerification,'evidence',{}),
        'auditor_source_sha256':digest(__file__)}
    (OUT/'independent_verification.json').write_text(json.dumps(record,indent=2)+'\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
