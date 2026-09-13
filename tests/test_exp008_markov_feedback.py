"""Independent finite-state Bellman, historical-cutoff and physical checks."""

import unittest

import numpy as np

from experiments.exp008.markov_feedback import (
    ETA,
    HIGH,
    LIMIT,
    LOW,
    execute,
    fit_error_model,
    initial_value,
    value_functions,
)


def tiny_model():
    emission = np.array([[-200 / ETA, 0., 200 * ETA], [-400 / ETA, 0., 400 * ETA]])
    transition = np.array([[[.7, .2, .1], [.2, .5, .3], [.1, .2, .7]],
                           [[.8, .15, .05], [.2, .6, .2], [.05, .15, .8]]])
    return {"forecast_net_kwh": np.array([400., 400.]), "mean_kwh": np.zeros(2),
            "scale_kwh": np.full(2, 200.), "state_edges": np.array([-.55, .55]),
            "error_support_kwh": emission, "transition": transition,
            "initial_error_state": 1, "metadata": {"role": "synthetic exact test"}}


def exhaustive(model, purchase, prices, grid, t, state, previous_error, mode, wear, switching, terminal):
    if t == len(prices):
        return -terminal * (state - LOW)
    expected = 0.
    for error in range(3):
        balance = purchase[t] - model["forecast_net_kwh"][t] - model["error_support_kwh"][t, error]
        direction = 1 if balance >= 0 else -1
        options = []
        for following in grid:
            charge = max(0., (following - state) / ETA)
            discharge = max(0., (state - following) * ETA)
            if ((balance >= 0 and (discharge > 1e-7 or charge > min(balance, LIMIT) + 1e-7))
                    or (balance < 0 and (charge > 1e-7 or discharge > min(-balance, LIMIT) + 1e-7))):
                continue
            active = charge + discharge > 1e-7
            nextmode = direction if active else mode
            stage = 5 * prices[t] * max(0., -balance - discharge) + wear * (charge + discharge)
            stage += switching if active and mode * direction == -1 else 0.
            options.append(stage + exhaustive(model, purchase, prices, grid, t + 1, following,
                                              error, nextmode, wear, switching, terminal))
        expected += model["transition"][t, previous_error, error] * min(options)
    return float(expected)


class TestMarkovFeedback(unittest.TestCase):
    def test_two_step_value_matches_nonanticipative_exhaustive_tree(self):
        model = tiny_model()
        purchase, prices = np.array([400., 400.]), np.array([.5, 2.])
        grid, values, _ = value_functions(purchase, model, prices, switching=100., terminal=.3)
        for mode in (-1, 0, 1):
            expected = exhaustive(model, purchase, prices, grid, 0, 1800., 1, mode, .002, 100., .3)
            self.assertAlmostEqual(initial_value(grid, values, model, 1800., mode), expected, places=7)

    def test_fitting_uses_complete_history_and_shrinks_sparse_rows(self):
        rng = np.random.default_rng(934)
        errors = rng.normal(0, 20, (3, 8))
        model = fit_error_model(np.zeros(8), errors, history_origins=[0, 8, 16], cutoff=24)
        np.testing.assert_allclose(model["transition"].sum(axis=-1), 1., atol=1e-12)
        self.assertTrue(np.all(model["transition"] >= 0))
        with self.assertRaises(ValueError):
            fit_error_model(np.zeros(8), errors, history_origins=[0, 8, 17], cutoff=24)
        with self.assertRaises(ValueError):
            fit_error_model(np.zeros(8), np.empty((0, 8)))
        single = fit_error_model(np.zeros(8), errors[:1])
        self.assertTrue(np.isfinite(single["transition"]).all())
        independent = fit_error_model(np.zeros(8), errors, independent=True)
        np.testing.assert_array_equal(independent["transition"][:, 0], independent["transition"][:, 2])

    def test_physics_and_future_actual_prefix_invariance(self):
        rng = np.random.default_rng(389)
        n = 144
        purchase = rng.uniform(0, 1400, n)
        forecast = rng.uniform(-200, 1100, n)
        model = fit_error_model(forecast, rng.normal(0, 90, (28, n)),
                                history_origins=np.arange(28) * n, cutoff=28 * n)
        price = rng.uniform(.3, 1.8, n)
        grid, values, _ = value_functions(purchase, model, price, switching=300.)
        actual = rng.uniform(0, 10000, (n, 2))
        altered = actual.copy()
        altered[17:] *= 40
        first, _ = execute(purchase, actual, price, 1789.3, grid, values, model, switching=300.)
        second, _ = execute(purchase, altered, price, 1789.3, grid, values, model, switching=300.)
        for key in ("charge", "discharge", "emergency", "surplus", "observed_error_state"):
            np.testing.assert_array_equal(first[key][:17], second[key][:17])
        np.testing.assert_array_equal(first["states"][:18], second["states"][:18])
        c, d, e, w, s = (first[key] for key in ("charge", "discharge", "emergency", "surplus", "states"))
        np.testing.assert_allclose(purchase + (actual[:, 1] - actual[:, 0]) / 6 + d + e - c - w, 0, atol=1e-8)
        np.testing.assert_allclose(np.diff(s), ETA * c - d / ETA, atol=1e-8)
        self.assertGreaterEqual(s.min(), LOW - 1e-8)
        self.assertLessEqual(s.max(), HIGH + 1e-8)
        self.assertLessEqual(max(c.max(), d.max()), LIMIT + 1e-8)
        self.assertFalse(np.any((c > 1e-6) & (d > 1e-6)))
        self.assertFalse(np.any((c > 1e-6) & (e > 1e-6)))
        self.assertAlmostEqual(float(first["fees"].sum()), float(np.sum(purchase * price + 5 * price * e)), places=8)


if __name__ == "__main__":
    unittest.main()
