"""Regression boundaries for the release-aligned planning risk adapter."""
import unittest

import numpy as np

from experiments.common.neural_v2.data import Data
from experiments.exp008.forecast import Forecasts
from experiments.exp008.issued_residual_paths import issued_error_paths


class IssuedResidualTests(unittest.TestCase):
    def test_midnight_preserves_existing_paths_and_early_fallback_labels(self):
        store = Forecasts()
        for day in (31, 80):
            for scenario in ('2', '3', '4-2', '4-3'):
                old = store.net_error_paths(day, scenario)
                new = issued_error_paths(store, day, scenario=scenario)
                for key in ('errors_kwh', 'errors_kw', 'price_errors', 'origins'):
                    np.testing.assert_array_equal(old[key], new[key])
                self.assertEqual(old['audit']['fallback_days'], new['audit']['fallback_days'])

    def test_same_release_definition_and_future_mutation(self):
        day, slot = 70, 72
        first, changed = Data(), Data()
        # Even the current-day prefix is irrelevant to historical risk paths.
        changed.actual[day*144:] += 987654
        for origin in changed.forecasts:
            if origin >= day*144:
                changed.forecasts[origin] += 123456
        a, b = Forecasts(first), Forecasts(changed)
        for scenario in ('3', '4-3'):
            x = issued_error_paths(a, day, slot, scenario)
            y = issued_error_paths(b, day, slot, scenario)
            self.assertEqual(x['errors_kwh'].shape, (28, 72))
            self.assertTrue(np.all(x['label_stops_exclusive'] <= day*144))
            np.testing.assert_array_equal(x['errors_kwh'], y['errors_kwh'])
            np.testing.assert_array_equal(x['price_errors'], y['price_errors'])
            old = day-1
            predicted = a.get(old, slot, scenario)
            truth = first.actual[old*144+slot:(old+1)*144]
            expected = ((truth[:, 0]-truth[:, 1])-
                        (predicted['load_kw']-predicted['pv_kw']))/6
            np.testing.assert_allclose(x['errors_kwh'][-1], expected, atol=1e-12)
            np.testing.assert_allclose(x['price_errors'][-1], truth[:, 2]-predicted['price'])
            midnight = a.net_error_paths(day, scenario)['errors_kwh'][:, slot:]
            self.assertGreater(float(np.max(np.abs(x['errors_kwh']-midnight))), 1.)

    def test_illegal_releases_and_no_history_fail(self):
        store = Forecasts()
        for day, slot, scenario in ((1, 0, '3'), (40, 1, '3'), (40, 36, '2')):
            with self.assertRaises(ValueError):
                issued_error_paths(store, day, slot, scenario)


if __name__ == '__main__':
    unittest.main()
