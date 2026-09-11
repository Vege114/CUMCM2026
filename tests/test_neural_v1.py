import copy
import unittest

import numpy as np

from experiments.common.neural_v1.data import Data, month_origins, split_origins
from experiments.common.neural_v1.dispatch import MAX_SOC, MIN_SOC, day_run, execute, settle
from experiments.common.neural_v1.evaluate import choose_models, warmup_baseline


class CausalPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = Data()

    def test_future_values_do_not_change_features_or_past_plan(self):
        origin = int(month_origins(3)[0])
        changed = copy.deepcopy(self.data)
        changed.actual[origin:] *= 3
        for k in changed.forecasts:
            if k > origin:
                changed.forecasts[k] *= 4
        for x, y in zip(self.data.features([origin], origin - 7 * 144),
                        changed.features([origin], origin - 7 * 144)):
            np.testing.assert_array_equal(x, y)
        np.testing.assert_array_equal(self.data.baseline(origin), changed.baseline(origin))
        # Altering future observations cannot alter an earlier day's plan or settlement.
        a, ad = day_run(self.data, origin // 144 - 1, self.data.baseline, "4-3")
        b, bd = day_run(changed, origin // 144 - 1, changed.baseline, "4-3")
        np.testing.assert_array_equal(ad['original'], bd['original'])
        self.assertEqual(a['total_cost'], b['total_cost'])

    def test_every_month_split_respects_label_completion(self):
        for month in range(2, 13):
            train, val, cutoff = split_origins(month)
            asof = month_origins(month)[0]
            self.assertLessEqual(train.max() + 144, cutoff)
            self.assertGreaterEqual(val.min(), cutoff)
            self.assertLessEqual(val.max() + 144, asof)
            self.assertEqual(asof - cutoff, 7 * 144)

    def test_interpolation_anchor_and_year_end(self):
        for origin in (0, 36, 365 * 144 - 36):
            p = self.data.issued_pv(origin)
            self.assertEqual(len(p), 144)
            np.testing.assert_allclose(p[5::6], self.data.forecasts[origin])
            self.assertTrue(np.isfinite(p).all())
        self.assertEqual(sum(len(month_origins(m)) for m in range(2, 13)), 334 * 4)

    def test_storage_limits_and_emergency(self):
        g = np.array([50000., 0., 0.])
        load = np.array([0., 60000., 60000.])
        c, d, e, w, s = execute(g, load, np.zeros(3), MAX_SOC)
        self.assertTrue((s >= MIN_SOC - 1e-8).all())
        self.assertTrue((s <= MAX_SOC + 1e-8).all())
        self.assertGreater(e.sum(), 0)
        self.assertGreater(w.sum(), 0)
        np.testing.assert_allclose(g + d + e - c - w, load / 6)

    def test_settlement_uses_final_deviation_only(self):
        fees = settle(np.array([10., 10., 10.]), np.array([5., 15., 10.]),
                      np.array([1., 0., 0.]), np.ones(3))
        np.testing.assert_allclose(fees.sum(0), [30, 7.5, 2.5, 5])

    def test_all_scenarios_energy_and_cost(self):
        for scenario in ("2", "3", "4-2", "4-3"):
            r, d = day_run(self.data, 79, self.data.baseline, scenario)
            self.assertEqual(r['violations'], 0)
            self.assertAlmostEqual(r['total_cost'], r['planned_cost'] + r['up_cost']
                                   + r['down_cost'] + r['emergency_cost'], places=6)
            self.assertEqual(len(d['original']), 144)

    def test_first_day_reference_uses_target_clock(self):
        prediction = warmup_baseline(self.data, 36)
        np.testing.assert_array_equal(prediction[:108, :3], self.data.reference[36:, [1, 2, 0]])
        np.testing.assert_array_equal(prediction[108:, :3], self.data.actual[:36])

    def test_future_data_cannot_change_validation_choice(self):
        asof = int(month_origins(2)[0])
        changed = copy.deepcopy(self.data)
        changed.actual[asof:] *= 17
        for origin in changed.forecasts:
            if origin >= asof:
                changed.forecasts[origin] *= 19

        class Store:
            def __init__(self, data): self.data = data
            def get(self, origin, variant, month):
                return self.data.baseline(origin) * {'mlp': .95, 'gru': 1.0, 'tcn': 1.05}[variant]

        states = {s: [6000.] * 365 for s in ('2', '3', '4-2', '4-3')}
        first, rows = choose_models(self.data, Store(self.data), states, months=[2])
        second, changed_rows = choose_models(changed, Store(changed), states, months=[2])
        self.assertEqual(first, second)
        self.assertEqual(rows, changed_rows)


if __name__ == '__main__':
    unittest.main()
