"""Information boundaries and cost-aware forecasting, independent of outcomes."""

import unittest

import numpy as np

from experiments.problem2.exp003.data import STEPS, Data
from experiments.problem2.exp004.data import Features, age_weights, split_days


class Exp004BoundaryTests(unittest.TestCase):
    def test_all_months_labels_stop_before_deployment(self):
        for month in range(2, 13):
            train, validation, formal, cutoff = split_days(month)
            self.assertEqual(train[-1] + 1, cutoff)
            self.assertEqual(validation[-1] + 1, formal[0])
            self.assertEqual(len(validation), 7)
            self.assertEqual(train[0], 7)

    def test_future_mutation_does_not_change_causal_features(self):
        original, changed = Data(), Data()
        day, cutoff = 140, 120
        changed.actual[day * STEPS:] = changed.actual[day * STEPS:] * 17 + 987
        for variant in ("no_season", "causal_season"):
            a = Features(original).arrays([day], cutoff, variant)
            b = Features(changed).arrays([day], cutoff, variant)
            for key in a:
                self.assertTrue(np.isfinite(a[key]).all())
                np.testing.assert_array_equal(a[key], b[key], err_msg=variant + key)
        a = Features(original).seasonal_shift(day, "oracle_season")
        b = Features(changed).seasonal_shift(day, "oracle_season")
        self.assertGreater(float(np.max(np.abs(a - b))), 1.)

    def test_historical_training_features_do_not_use_later_training_labels(self):
        original, changed = Data(), Data()
        day = 60
        changed.actual[day * STEPS:] *= 3
        # Scaler is intentionally checked separately: fit once at training cutoff.
        a = Features(original).base(day, "causal_season")
        b = Features(changed).base(day, "causal_season")
        np.testing.assert_array_equal(a, b)

    def test_weights_keep_history_and_decay_by_half_life(self):
        weights = age_weights([10, 100], 110)
        self.assertGreater(weights.min(), 0)
        self.assertAlmostEqual(float(weights[1] / weights[0]), 2)
        self.assertAlmostEqual(float(weights.mean()), 1)
        with self.assertRaises(ValueError):
            age_weights([110], 110)

    def test_shapes_and_units(self):
        f = Features()
        p = f.arrays([31], 24, "no_season")
        self.assertEqual(p["sequence"].shape, (1, 168, 2))
        self.assertEqual(p["context"].shape, (1, 144, 2, 8))
        np.testing.assert_array_equal(p["base"][0], f.data.baseline(31 * STEPS))
        np.testing.assert_array_equal(p["seasonal_shift"], 0)

    def test_loss_penalizes_short_load_and_excess_pv_fivefold(self):
        from experiments.problem2.exp004.train import AsymmetricHuber, tf

        loss = AsymmetricHuber([1.])
        truth = tf.zeros((1, 1, 2))
        adverse = loss.call(truth, tf.constant([[[-.5, .5]]])).numpy()
        favorable = loss.call(truth, tf.constant([[[.5, -.5]]])).numpy()
        np.testing.assert_allclose(adverse, favorable * 5, rtol=0, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
