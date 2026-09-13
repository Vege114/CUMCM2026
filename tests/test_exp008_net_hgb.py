"""Information and representation checks for the bounded net HGB trial."""
import json
import unittest

import numpy as np
from threadpoolctl import threadpool_limits

from experiments.exp008.forecast_net_hgb import (
    OUT, NetHGBStore, bases_and_features, features_for_day, fit_month,
)
from experiments.problem2.exp003.data import Data
from experiments.problem2.exp004.predict import ForecastStore


class Exp008NetHGBTests(unittest.TestCase):
    def test_day_features_ignore_current_and_future_actuals(self):
        original, changed = Data(), Data()
        store = ForecastStore("no_season")
        for day in (31, 90, 180, 364):
            changed.actual[:] = original.actual
            changed.actual[day * 144:] = original.actual[day * 144:] * 19 + 90000
            base = store.get(day * 144)
            a, names = features_for_day(original, day, base)
            b, other = features_for_day(changed, day, base)
            self.assertEqual(names, other)
            self.assertEqual(a.shape, (144, 35))
            np.testing.assert_array_equal(a, b)

    def test_month_refit_ignores_labels_and_features_after_month_start(self):
        original, changed = Data(), Data()
        store = ForecastStore("no_season", seed=42)
        month, asof = 4, 90
        changed.actual[asof * 144:] = changed.actual[asof * 144:] * 11 + 123456
        days, bases, first, _ = bases_and_features(original, store)
        _, _, second, _ = bases_and_features(changed, store)
        a, audit = fit_month(original, month, days, bases, first)
        b, altered_audit = fit_month(changed, month, days, bases, second)
        self.assertEqual(audit["training_last_label"], asof * 144 - 1)
        self.assertEqual(audit["training_days"], altered_audit["training_days"])
        with threadpool_limits(limits=1):
            np.testing.assert_array_equal(a.predict(first[asof - 7]), b.predict(second[asof - 7]))

    def test_net_adapter_and_delta_are_exactly_reconstructible(self):
        store = NetHGBStore()
        self.assertTrue(np.isfinite(store.values).all())
        self.assertGreaterEqual(store.values.min(), 0)
        np.testing.assert_allclose(store.values[:, :, 0] - store.values[:, :, 1], store.net_kw, atol=1e-9)
        np.testing.assert_allclose(store.base_values[:, :, 0] - store.base_values[:, :, 1]
                                   + store.net_delta_kw, store.net_kw, atol=1e-9)
        truth = Data().actual[store.origins[:, None] + np.arange(144)]
        np.testing.assert_allclose(store.values + store.errors_kw, truth, atol=1e-9)

    def test_all_months_and_prediction_audits_have_strict_cutoffs(self):
        training = json.loads((OUT / "training_audit.json").read_text())
        prediction = json.loads((OUT / "prediction_audit.json").read_text())
        self.assertEqual(len(training), 11)
        self.assertEqual(len(prediction), 334)
        for row in training:
            self.assertLess(row["training_last_label"], row["information_cutoff_exclusive"])
            self.assertLess(max(row["training_days"]), row["asof_day"])
            self.assertEqual(row["iterations"], 100)
        for row in prediction:
            self.assertLess(row["feature_last_actual_index"], row["origin"])
            self.assertLess(row["model_training_last_label"], row["origin"])


if __name__ == "__main__":
    unittest.main()
