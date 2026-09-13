"""Information-boundary tests for the shared forecast adapter."""
import unittest

import numpy as np

from experiments.common.neural_v2.data import Data
from experiments.exp008.forecast import Forecasts


class Exp008ForecastTests(unittest.TestCase):
    def test_q2_midnight_is_exact_latest_cnn_for_entire_evaluation(self):
        store = Forecasts()
        for day in range(31, 365):
            prediction = store.get(day)
            expected = store.store.get(day * 144)
            np.testing.assert_array_equal(prediction["load_kw"], expected[:, 0])
            np.testing.assert_array_equal(prediction["pv_kw"], expected[:, 1])
            np.testing.assert_array_equal(prediction["price"], store.data.fixed_price)

    def test_future_actuals_and_unreleased_official_versions_do_not_change_forecast(self):
        for day, slot in ((31, 0), (120, 36), (220, 72), (364, 108)):
            cutoff = day * 144 + slot
            original, changed = Data(), Data()
            changed.actual[cutoff:] = changed.actual[cutoff:] * 13 + 99999
            # The currently released issue stays fixed; all future versions change.
            for origin in list(changed.forecasts):
                if origin > cutoff:
                    changed.forecasts[origin] = changed.forecasts[origin] * 17 + 123456
            first, second = Forecasts(original), Forecasts(changed)
            scenarios = ("2", "3", "4-2", "4-3") if slot == 0 else ("3", "4-3")
            for scenario in scenarios:
                a = first.get(day, slot, scenario)
                b = second.get(day, slot, scenario)
                for name in ("load_kw", "pv_kw", "price"):
                    np.testing.assert_array_equal(a[name], b[name], err_msg=f"{day}/{slot}/{scenario}/{name}")
                    self.assertEqual(len(a[name]), 144 - slot)
                self.assertLess(a["audit"]["max_observed_index"], cutoff)
                if scenario.startswith("4"):
                    self.assertLess(a["audit"]["price_last_label"], cutoff)

    def test_realized_prefix_can_change_intraday_load_but_not_midnight_forecast(self):
        day, slot = 100, 36
        original, changed = Data(), Data()
        changed.actual[day * 144:day * 144 + slot, 0] += 500
        a, b = Forecasts(original), Forecasts(changed)
        np.testing.assert_array_equal(a.get(day)["load_kw"], b.get(day)["load_kw"])
        self.assertGreater(np.max(np.abs(a.get(day, slot, "3")["load_kw"] -
                                          b.get(day, slot, "3")["load_kw"])), 100)

    def test_joint_error_paths_have_no_future_labels(self):
        day = 45
        original, changed = Data(), Data()
        changed.actual[day * 144:] += 999999
        a, b = Forecasts(original), Forecasts(changed)
        for scenario in ("2", "3", "4-2", "4-3"):
            first, second = a.net_error_paths(day, scenario), b.net_error_paths(day, scenario)
            self.assertEqual(first["errors_kwh"].shape, (28, 144))
            self.assertTrue(np.all(first["origins"] + 144 <= day * 144))
            np.testing.assert_array_equal(first["errors_kwh"], second["errors_kwh"])
            np.testing.assert_array_equal(first["price_errors"], second["price_errors"])
            np.testing.assert_allclose(first["errors_kwh"],
                (first["errors_kw"][:, :, 0] - first["errors_kw"][:, :, 1]) / 6)

    def test_midnight_only_scenarios_reject_intraday_replanning(self):
        forecast = Forecasts()
        for scenario in ("2", "4-2"):
            with self.assertRaises(ValueError):
                forecast.get(100, 36, scenario)


if __name__ == "__main__":
    unittest.main()
