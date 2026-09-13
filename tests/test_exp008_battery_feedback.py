"""Small exhaustive and causal tests for fixed-purchase inventory feedback."""

import itertools
import unittest

import numpy as np

from experiments.exp008.battery_feedback import ETA, HIGH, LIMIT, LOW, execute, value_functions


def exhaustive_two_step(initial, purchase, supports, price, grid, wear, terminal):
    """Enumerate contingent actions after each observed disturbance.

    These fixtures have all power-limited endpoints on the 100-kWh grid;
    the independent enumeration consequently has no interpolation error.
    """
    def transitions(state, balance):
        if balance >= 0:
            delta = grid - state
            feasible = (delta >= -1e-8) & (delta <= ETA * min(balance, LIMIT) + 1e-8)
        else:
            delta = state - grid
            feasible = (delta >= -1e-8) & (delta * ETA <= min(-balance, LIMIT) + 1e-8)
        return grid[feasible]

    def cost(before, after, balance, tariff):
        charge = max(0., (after - before) / ETA)
        release = max(0., (before - after) * ETA)
        return wear * (charge + release) + 5 * tariff * max(0., -balance - release)

    first_costs = []
    for net0 in supports[0]:
        b0 = purchase[0] - net0
        options = []
        for middle in transitions(initial, b0):
            remaining = []
            for net1 in supports[1]:
                b1 = purchase[1] - net1
                remaining.append(min(cost(middle, end, b1, price[1]) - terminal * (end - LOW)
                                     for end in transitions(middle, b1)))
            options.append(cost(initial, middle, b0, price[0]) + np.mean(remaining))
        first_costs.append(min(options))
    return float(np.mean(first_costs))


class TestBatteryFeedback(unittest.TestCase):
    def test_off_grid_soc_considers_minimum_charge_deadband_endpoint(self):
        grid = np.arange(LOW, HIGH + 1, 100.)
        values = np.zeros((2, len(grid)))
        values[1, 0] = 100.
        actual = np.zeros((1, 2))
        detail = execute(np.array([200.]), actual, np.array([1.]), 1280., grid, values,
                         wear=.002, charge_deadband=31.)
        self.assertAlmostEqual(float(detail["charge"][0]), 31., places=8)

    def test_two_step_bellman_value_matches_contingent_exhaustive_search(self):
        fixtures = [
            (LOW + 300, np.zeros(2), ETA * np.array([[100., 200.], [200., 400.]]), np.array([.5, 2.])),
            (LOW, np.array([200 / ETA, 0.]),
             np.array([[0., 100 / ETA], [100 * ETA, 200 * ETA]]), np.array([.4, 1.5])),
        ]
        for initial, purchase, supports, prices in fixtures:
            with self.subTest(initial=initial):
                grid, values, _ = value_functions(purchase, supports, prices, grid_kwh=100., wear=.02, terminal=.3)
                expected = exhaustive_two_step(initial, purchase, supports, prices, grid, .02, .3)
                observed = float(values[0, np.flatnonzero(np.isclose(grid, initial))[0]])
                self.assertAlmostEqual(observed, expected, places=8)
                # Average over all four independent observation trajectories.
                realized = []
                for disturbance in itertools.product(*supports):
                    net = np.asarray(disturbance)
                    actual = np.column_stack((6 * np.maximum(net, 0), 6 * np.maximum(-net, 0)))
                    result = execute(purchase, actual, prices, initial, grid, values, wear=.02)
                    realized.append(float(np.sum(5 * prices * result["emergency"]
                                                  + .02 * (result["charge"] + result["discharge"]))
                                          - .3 * (result["states"][-1] - LOW)))
                self.assertAlmostEqual(float(np.mean(realized)), expected, places=8)

    def test_actual_future_mutation_does_not_change_execution_prefix(self):
        rng = np.random.default_rng(981)
        n = 20
        purchase = rng.uniform(100, 500, n)
        supports = rng.uniform(-300, 900, (n, 5))
        price = rng.uniform(.3, 1.8, n)
        grid, values, _ = value_functions(purchase, supports, price)
        actual = rng.uniform(0, 6000, (n, 2))
        altered = actual.copy()
        altered[8:] *= 30
        first = execute(purchase, actual, price, 6433.2, grid, values)
        second = execute(purchase, altered, price, 6433.2, grid, values)
        for key in ("charge", "discharge", "emergency", "surplus", "fees"):
            np.testing.assert_array_equal(first[key][:8], second[key][:8])
        np.testing.assert_array_equal(first["states"][:9], second["states"][:9])

    def test_physical_projection_bounds_and_deadband_with_mask(self):
        rng = np.random.default_rng(914)
        n = 144
        purchase = rng.uniform(0, 1500, n)
        supports = rng.uniform(-1500, 2500, (n, 3))
        prices = rng.uniform(.3, 2, n)
        mask = (np.arange(n) // 8) % 2 == 0
        grid, values, _ = value_functions(purchase, supports, prices, charge_mask=mask, charge_deadband=31.)
        actual = rng.uniform(0, 13000, (n, 2))
        detail = execute(purchase, actual, prices, 1755.38, grid, values,
                         charge_mask=mask, charge_deadband=31.)
        c, d, e, w, s = (detail[key] for key in ("charge", "discharge", "emergency", "surplus", "states"))
        np.testing.assert_allclose(purchase + (actual[:, 1] - actual[:, 0]) / 6 + d + e - c - w, 0, atol=1e-8)
        np.testing.assert_allclose(np.diff(s), ETA * c - d / ETA, atol=1e-8)
        self.assertGreaterEqual(s.min(), LOW - 1e-8)
        self.assertLessEqual(s.max(), HIGH + 1e-8)
        self.assertLessEqual(max(c.max(), d.max()), LIMIT + 1e-8)
        self.assertFalse(np.any((c > 1e-6) & (d > 1e-6)))
        self.assertFalse(np.any((c > 1e-6) & (e > 1e-6)))
        self.assertFalse(np.any((c > 1e-6) & (c < 31 - 1e-6)))
        np.testing.assert_array_equal(c[~mask], 0.)
        np.testing.assert_array_equal(d[mask], 0.)


if __name__ == "__main__":
    unittest.main()
