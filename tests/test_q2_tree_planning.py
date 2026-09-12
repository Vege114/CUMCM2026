"""Small independent checks for the new algorithm, not performance claims."""

import unittest

import numpy as np

from experiments.problem2.tree_planning.model import (
    ETA,
    MAX_SOC,
    MIN_SOC,
    POWER_ENERGY,
    Config,
    execute_plan,
    plan_day,
    stage_cost,
)
from experiments.problem2.tree_planning.risk import TreeResidualScenarios


class TreePlanningTests(unittest.TestCase):
    def test_newsvendor_against_all_breakpoints(self):
        support = np.array([[-100.0, 10, 300, 500, 900]])
        shifts = np.array([-500.0, 0, 70.0])
        purchase, costs = stage_cost(support, shifts, np.array([1.4]))
        for i, shift in enumerate(shifts):
            candidates = np.unique(np.maximum(0, np.r_[0, support.ravel() + shift]))
            direct = [1.4 * (q + 5 * np.maximum(support + shift - q, 0).mean())
                      for q in candidates]
            self.assertAlmostEqual(costs[0, i], min(direct))
            self.assertGreaterEqual(purchase[0, i], 0)

    def test_dp_against_two_step_exhaustive_search(self):
        support = np.array([[100, 300, 700], [600, 1000, 1800]], float)
        prices = np.array([.3, 1.2])
        config = Config(grid_kwh=600, throughput_yuan_per_kwh=.002, reversal_yuan=.05)
        initial = 6042.75  # off-grid energy must not disappear.
        result = plan_day(support, prices, initial, config, initial_mode=-1)
        grid = np.unique(np.r_[np.arange(MIN_SOC, MAX_SOC, 600), MAX_SOC, initial])
        minimum = np.inf
        for s1 in grid:
            for s2 in grid:
                delta = np.diff([initial, s1, s2])
                c, d = np.maximum(delta, 0) / ETA, np.maximum(-delta, 0) * ETA
                if np.maximum(c, d).max() > POWER_ENERGY + 1e-8:
                    continue
                fee = 0.0
                previous = -1
                for t in range(2):
                    _, cost = stage_cost(support[t:t + 1], [c[t] - d[t]], prices[t:t + 1])
                    mode = int(np.sign(delta[t]))
                    fee += cost[0, 0] + .002 * (c[t] + d[t]) + .05 * (mode * previous == -1)
                    previous = mode or previous
                fee -= prices.min() / ETA * (s2 - MIN_SOC)
                minimum = min(minimum, fee)
        self.assertAlmostEqual(result["metadata"]["objective_surrogate_yuan"], minimum, places=5)
        self.assertEqual(result["states"][0], initial)

    def test_physical_projection_extreme_realizations_and_prefix_causality(self):
        support = np.tile(np.linspace(-1000, 1500, 9), (144, 1))
        prices = np.r_[np.full(72, .3), np.full(72, 1.2)]
        plan = plan_day(support, prices, 6042.75)
        actual = np.zeros((144, 2))
        actual[::2, 0], actual[1::2, 1] = 90000, 90000
        detail, metrics = execute_plan(plan, actual, prices, 6042.75)
        changed = actual.copy()
        changed[70:] = 12345
        revised, _ = execute_plan(plan, changed, prices, 6042.75)
        for key in ("charge", "discharge", "emergency", "surplus"):
            np.testing.assert_array_equal(detail[key][:70], revised[key][:70])
        self.assertEqual(metrics["simultaneous_slots"], 0)
        self.assertFalse(((detail["charge"] > 1e-6) & (detail["emergency"] > 1e-6)).any())

    def test_tree_future_actuals_do_not_change_midnight_support(self):
        rng = np.random.default_rng(123)
        actual = rng.uniform(0, 3000, (365 * 144, 2))
        origins = np.arange(31, 365) * 144
        forecasts = rng.uniform(0, 3000, (334, 144, 2))
        prices = np.linspace(.3, 1.3, 144)
        for day in (31, 80):
            mutated = actual.copy()
            mutated[day * 144:] = 9999999
            first, info = TreeResidualScenarios(origins, forecasts, actual, prices).for_day(day)
            second, _ = TreeResidualScenarios(origins, forecasts, mutated, prices).for_day(day)
            np.testing.assert_array_equal(first, second)
            self.assertLess(info["max_observed_index"], day * 144)
            self.assertEqual(first.shape, (144, 9))
            self.assertTrue(np.isfinite(first).all())


if __name__ == "__main__":
    unittest.main()
