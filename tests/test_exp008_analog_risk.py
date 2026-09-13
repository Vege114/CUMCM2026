"""Analogue-day causality, whole-path identity and fixed recency checks."""
import unittest

import numpy as np

from experiments.exp008.analog_risk import (
    AnalogResidualScenarios,
    MatureAnalogResidualScenarios,
    select_neighbors,
)
from experiments.exp008.forecast_calibration import CalibratedStore
from experiments.problem2.exp003.data import Data
from experiments.problem2.tree_planning.risk import QUANTILE_LEVELS, TreeResidualScenarios


class AnalogRiskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data, cls.store = Data(), CalibratedStore('ridge_28')

    def risk(self, actual=None, forecasts=None):
        return AnalogResidualScenarios(self.store.origins,
                                      self.store.values if forecasts is None else forecasts,
                                      self.data.actual if actual is None else actual,
                                      self.data.fixed_price)

    def test_own_issue_features_and_future_mutations(self):
        risk = self.risk()
        changed = self.data.actual.copy()
        changed[60*144:] += 1e7
        np.testing.assert_array_equal(risk.features_for_day(60),
                                      self.risk(actual=changed).features_for_day(60))
        for day in (31, 32, 45, 140):
            actual, forecasts = self.data.actual.copy(), self.store.values.copy()
            actual[day*144:] += 1e7
            forecasts[self.store.origins > day*144] += 2e7
            first = risk.path_bundle_for_day(day)
            second = self.risk(actual, forecasts).path_bundle_for_day(day)
            np.testing.assert_array_equal(first['support'], second['support'])
            np.testing.assert_array_equal(first['error_paths'], second['error_paths'])
            self.assertEqual(first['audit']['selected_days'], second['audit']['selected_days'])
            self.assertLess(first['audit']['max_observed_index'], day*144)

    def test_selected_paths_are_complete_issued_residuals_with_equal_weights(self):
        risk = self.risk()
        for day in (32, 45, 140):
            bundle = risk.path_bundle_for_day(day)
            selected = bundle['audit']['selected_days']
            self.assertEqual(len(selected), min(21, day-31))
            self.assertEqual(len(set(selected)), len(selected))
            self.assertTrue(all(max(31, day-90) <= old < day for old in selected))
            expected = []
            for old in selected:
                actual = self.data.actual[old*144:(old+1)*144]
                forecast = self.store.get(old*144)
                expected.append(((actual[:, 0]-actual[:, 1])-
                                 (forecast[:, 0]-forecast[:, 1]))/6)
            np.testing.assert_array_equal(bundle['error_paths'], expected)
            forecast = self.store.get(day*144)
            net = (forecast[:, 0]-forecast[:, 1])/6
            support = net[:, None]+np.quantile(expected, QUANTILE_LEVELS, axis=0).T
            np.testing.assert_array_equal(bundle['support'], support)
            np.testing.assert_allclose(bundle['audit']['path_weights'], 1/len(selected))

    def test_zero_issued_history_uses_explicit_unchanged_periodic_fallback(self):
        common = self.store.origins, self.store.values, self.data.actual, self.data.fixed_price
        bundle = self.risk().path_bundle_for_day(31)
        np.testing.assert_array_equal(bundle['support'], TreeResidualScenarios(*common).for_day(31)[0])
        self.assertTrue(bundle['audit']['analog_fallback'])
        self.assertEqual(bundle['error_paths'].shape, (0, 144))

    def test_equal_similarity_prefers_recent_days_with_42_day_half_life(self):
        x = np.zeros((50, 10))
        ages = np.arange(50, 0, -1)
        indices, audit = select_neighbors(x, np.ones(10), ages)
        np.testing.assert_array_equal(indices, np.arange(29, 50))
        self.assertAlmostEqual(audit['log_similarity'][49]-audit['log_similarity'][7], np.log(2.))
        np.testing.assert_array_equal(audit['history_feature_mean'], 0.)
        np.testing.assert_array_equal(audit['history_feature_scale'], 1.)

    def test_fixed_21_day_warmup_then_unchanged_mature_analogues(self):
        common = self.store.origins, self.store.values, self.data.actual, self.data.fixed_price
        mature, original, tree = (MatureAnalogResidualScenarios(*common),
                                  AnalogResidualScenarios(*common), TreeResidualScenarios(*common))
        for day in range(31, 52):
            bundle = mature.path_bundle_for_day(day)
            np.testing.assert_array_equal(bundle['support'], tree.for_day(day)[0])
            self.assertEqual(bundle['audit']['selected_count'], 0)
        for day in (52, 60, 90):
            expected, actual = original.path_bundle_for_day(day), mature.path_bundle_for_day(day)
            np.testing.assert_array_equal(actual['support'], expected['support'])
            np.testing.assert_array_equal(actual['error_paths'], expected['error_paths'])
            self.assertEqual(actual['audit']['selected_days'], expected['audit']['selected_days'])
            self.assertEqual(actual['audit']['selected_count'], 21)
        for day in (40, 70):
            changed = self.data.actual.copy()
            changed[day*144:] += 1e7
            poisoned = MatureAnalogResidualScenarios(self.store.origins, self.store.values,
                                                     changed, self.data.fixed_price)
            np.testing.assert_array_equal(mature.for_day(day)[0], poisoned.for_day(day)[0])


if __name__ == '__main__':
    unittest.main()
