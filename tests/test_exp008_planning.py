"""Finite-difference and physical invariants independent of optimizer success."""

import unittest

import numpy as np

from experiments.exp008.closed_loop import objective
from experiments.exp008.planner import ETA, HIGH, LIMIT, LOW, Settings, execute, plan


class TestClosedLoopGradient(unittest.TestCase):
    def test_gradient_matches_central_differences_across_storage_regimes(self):
        rng = np.random.default_rng(923)
        n, k = 17, 4
        purchase = rng.uniform(150, 700, n)
        paths = purchase[None, :] + rng.normal(0, 350, (k, n))
        prices = rng.uniform(.35, 1.3, n)
        mask = (np.arange(n) // 4) % 2 == 0
        for initial in (LOW + 70, 6200., HIGH - 90):
            for selected_mask in (None, mask):
                with self.subTest(initial=initial, mask=selected_mask is not None):
                    kwargs = {"charge_mask": selected_mask, "throughput": .019,
                              "variation": .003, "terminal": .41, "deadband": 13.}
                    _, gradient = objective(purchase, paths, prices, initial, **kwargs)
                    finite = np.empty(n)
                    eps = 1e-4
                    for t in range(n):
                        plus, minus = purchase.copy(), purchase.copy()
                        plus[t] += eps
                        minus[t] -= eps
                        fp = objective(plus, paths, prices, initial, **kwargs)[0]
                        fm = objective(minus, paths, prices, initial, **kwargs)[0]
                        finite[t] = (fp - fm) / (2 * eps)
                    np.testing.assert_allclose(gradient, finite, rtol=2e-5, atol=2e-5)

    def test_objective_matches_independent_causal_physical_replay(self):
        rng = np.random.default_rng(719)
        n, k = 20, 3
        purchase = rng.uniform(100, 400, n)
        paths = rng.uniform(-200, 1000, (k, n))
        prices = rng.uniform(.3, 1.8, n)
        initial, throughput, variation, terminal, deadband = 1700., .007, .005, .2, 23.
        expected = float(np.sum(purchase * prices))
        for demand in paths:
            actual = np.column_stack((6 * np.maximum(demand, 0), 6 * np.maximum(-demand, 0)))
            detail = execute(purchase, actual, prices, initial, charge_deadband=deadband)
            c, d, e = (detail[key] for key in ("charge", "discharge", "emergency"))
            expected += float((np.sum(5 * prices * e + throughput * (c + d))
                               + 6 * variation * np.abs(np.diff(c - d)).sum()
                               - terminal * (detail["states"][-1] - LOW)) / k)
        actual = objective(purchase, paths, prices, initial, throughput=throughput,
                           variation=variation, terminal=terminal, deadband=deadband)[0]
        self.assertAlmostEqual(actual, expected, places=8)


class TestPhysicalExecution(unittest.TestCase):
    def test_charge_mask_controls_both_directions_even_with_ramp_clipping(self):
        purchase = np.array([0., 2000., 0., 2000.])
        actual = np.column_stack(([6000., 0., 6000., 0.], np.zeros(4)))
        # Slots 0 and 3 prohibit the sign requested by current balance.
        mask = np.array([True, True, False, False])
        detail = execute(purchase, actual, np.ones(4), 6000., charge_mask=mask,
                         ramp_kw=1000., previous_power=5000., charge_deadband=0.)
        np.testing.assert_array_equal(detail["charge"][~mask], 0.)
        np.testing.assert_array_equal(detail["discharge"][mask], 0.)
        self.assertEqual(detail["emergency"][0], 1000.)
        self.assertEqual(detail["surplus"][3], 2000.)

    def test_extreme_flows_keep_balance_soc_power_and_mutual_exclusion(self):
        n = 144
        purchase = np.where(np.arange(n) % 2, 2000., 0.)
        actual = np.column_stack((np.where(np.arange(n) % 2, 0., 12000.), np.zeros(n)))
        detail = execute(purchase, actual, np.ones(n), LOW + 20,
                         charge_deadband=35., discharge_deadband=9., ramp_kw=1000.)
        c, d, e, w, s = (detail[key] for key in ("charge", "discharge", "emergency", "surplus", "states"))
        np.testing.assert_allclose(purchase - actual[:, 0] / 6 + d + e - c - w, 0, atol=1e-8)
        np.testing.assert_allclose(np.diff(s), ETA * c - d / ETA, atol=1e-8)
        self.assertGreaterEqual(s.min(), LOW - 1e-8)
        self.assertLessEqual(s.max(), HIGH + 1e-8)
        self.assertLessEqual(max(c.max(), d.max()), LIMIT + 1e-8)
        self.assertFalse(np.any((c > 1e-6) & (d > 1e-6)))
        self.assertFalse(np.any((c > 1e-6) & (e > 1e-6)))

    def test_changing_future_observations_does_not_change_executed_prefix(self):
        rng = np.random.default_rng(312)
        purchase = rng.uniform(0, 700, 60)
        actual = rng.uniform(0, 6000, (60, 2))
        changed = actual.copy()
        changed[19:] *= 8
        first = execute(purchase, actual, np.ones(60), 6700., charge_deadband=40.)
        second = execute(purchase, changed, np.ones(60), 6700., charge_deadband=40.)
        for key in ("charge", "discharge", "emergency", "surplus"):
            np.testing.assert_array_equal(first[key][:19], second[key][:19])
        np.testing.assert_array_equal(first["states"][:20], second["states"][:20])

    def test_masked_point_plan_matches_its_energy_equations(self):
        forecast = {"load_kw": np.array([600., 1100., 1800., 900.]),
                    "pv_kw": np.zeros(4), "price": np.array([.4, .6, 1.4, .9])}
        mask = np.array([1, 1, 0, 0])
        result = plan(forecast, 1400., settings=Settings(scenarios=1),
                      final_day=True, charge_mask=mask)
        self.assertFalse(result["metadata"]["nonanticipative_recourse_certificate"])
        np.testing.assert_allclose(np.diff(result["states"]),
                                   ETA * result["charge"] - result["discharge"] / ETA, atol=1e-6)
        self.assertLess(np.max(result["charge"][mask == 0]), 1e-6)
        self.assertLess(np.max(result["discharge"][mask == 1]), 1e-6)
        revised = plan(forecast, 1400., original=result["purchase"],
                       settings=Settings(scenarios=1), final_day=True)
        self.assertTrue(np.all(revised["purchase"] >= result["purchase"] - 1e-6))


if __name__ == "__main__":
    unittest.main()
