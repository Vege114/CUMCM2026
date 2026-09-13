"""Output calibration is causal, reconstructible and forecast-store compatible."""
import unittest

import numpy as np

from experiments.exp008.forecast_calibration import (
    CANDIDATES, CONFIG, CalibratedStore, _calibrate, _online_select,
)
from experiments.problem2.exp003.data import Data
from experiments.problem2.exp004.predict import ForecastStore


class Exp008CalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = Data()
        cls.base = ForecastStore("no_season", seed=42)

    def test_current_and_future_actuals_cannot_change_any_candidate_today(self):
        for day in (31, 45, 180, 364):
            index = day - 31
            changed = self.data.actual.copy()
            changed[day * 144:] = changed[day * 144:] * 9 + 99999
            for config in CANDIDATES:
                if config["method"] == "online_select":
                    continue
                a, audit = _calibrate(self.data.actual, self.base.origins, self.base.values, index, config)
                b, _ = _calibrate(changed, self.base.origins, self.base.values, index, config)
                np.testing.assert_array_equal(a, b, err_msg=f"{config['name']}/{day}")
                if audit["history_last_label"] is not None:
                    self.assertLess(audit["history_last_label"], day * 144)

    def test_archive_delta_and_historical_errors_reconstruct(self):
        store = CalibratedStore("ridge_28")
        np.testing.assert_allclose(store.base_values + store.delta, store.values, atol=1e-9)
        truth = self.data.actual[store.origins[:, None] + np.arange(144)]
        np.testing.assert_allclose(store.values + store.errors_kw, truth, atol=1e-9)
        self.assertTrue(np.isfinite(store.values).all())
        self.assertGreaterEqual(store.values.min(), 0)
        result = store.completed_error_paths(180 * 144)
        self.assertEqual(result["errors_kw"].shape, (28, 144, 2))
        self.assertTrue((result["origins"] + 144 <= 180 * 144).all())
        np.testing.assert_array_equal(store.get(180 * 144), store.values[149])

    def test_cold_start_uses_exact_cnn_without_fitting_current_truth(self):
        for config in CANDIDATES:
            if config["method"] == "online_select":
                continue
            value, audit = _calibrate(self.data.actual, self.base.origins, self.base.values, 0, config)
            np.testing.assert_array_equal(value, self.base.values[0])
            self.assertEqual(audit["history_count"], 0)

    def test_online_selection_uses_prequential_errors_before_today(self):
        candidates = {name: CalibratedStore(name).values for name in ("base", "global_bias_7", "ridge_28")}
        before, audit = _online_select(self.data.actual, self.base.origins, candidates, CONFIG["online_select_28"])
        changed = self.data.actual.copy()
        cutoff_day = 180
        changed[cutoff_day * 144:] += 100000
        after, revised = _online_select(changed, self.base.origins, candidates, CONFIG["online_select_28"])
        np.testing.assert_array_equal(before[:cutoff_day - 31 + 1], after[:cutoff_day - 31 + 1])
        self.assertEqual(audit[cutoff_day - 31], revised[cutoff_day - 31])
        for row in audit:
            if row["history_last_label"] is not None:
                self.assertLess(row["history_last_label"], row["origin"])


if __name__ == "__main__":
    unittest.main()
