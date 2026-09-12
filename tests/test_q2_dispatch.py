"""Fixed purchases, causal replay, exact settlement, and chronological SOC."""

import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from experiments.common.neural_v2.physics import ETA, MAX_SOC, MIN_SOC, POWER_ENERGY
from experiments.problem2.exp003 import dispatch


def setUpModule():
    # v1 may initialize HiGHS with another process-global thread count.
    from scipy.optimize._highspy._core import _Highs

    _Highs.resetGlobalScheduler(True)


def tearDownModule():
    setUpModule()


def fixture(days=34):
    return SimpleNamespace(actual=np.zeros((days * 144, 2)), fixed_price=np.ones(144))


class Q2DispatchTests(unittest.TestCase):
    def test_full_day_fixed_plan_and_fivefold_emergency_fee(self):
        data = fixture()
        data.actual[31 * 144:32 * 144, 0] = 1200
        forecast = np.column_stack((np.full(144, 600.), np.zeros(144)))
        summary, detail, solver = dispatch.day_run(data, forecast, 31, MIN_SOC)
        np.testing.assert_allclose(detail["original"], 100, atol=1e-6)
        np.testing.assert_array_equal(detail["original"], detail["final"])
        np.testing.assert_allclose(detail["emergency"], 100, atol=1e-6)
        np.testing.assert_allclose(detail["fees"].sum(axis=0), [14400, 0, 0, 72000])
        self.assertAlmostEqual(summary["total_cost"], 86400)
        self.assertEqual(summary["updates"], 0)
        self.assertEqual(summary["emergency_minutes"], 1440)
        self.assertEqual(solver["information_cutoff"], 31 * 144)
        self.assertFalse(solver["issued_forecast_allowed"])
        self.assertFalse(solver["fallback"])
        self.assertEqual(detail["actual"].shape, (144, 2))
        self.assertEqual(detail["fees"].shape, (144, 4))

    def test_current_actual_is_read_only_after_predict_and_solve(self):
        events = []
        stored_actual = np.zeros((144, 2))

        class GuardedData:
            fixed_price = np.ones(144)

            @property
            def actual(self):
                self_test.assertEqual(events, ["predict", "solve"])
                events.append("actual")
                return stored_actual

        self_test = self

        def predict(origin):
            self.assertEqual(origin, 0)
            self.assertEqual(events, [])
            events.append("predict")
            return np.zeros((144, 2))

        original_solver = dispatch.deterministic

        def solve(*args):
            self.assertEqual(events, ["predict"])
            events.append("solve")
            return original_solver(*args)

        with patch.object(dispatch, "deterministic", side_effect=solve):
            summaries, detail, logs = dispatch.replay_days(GuardedData(), [0], predict, 6000)
        self.assertEqual(events, ["predict", "solve", "actual"])
        self.assertEqual(summaries[0]["initial_soc"], 6000)
        self.assertEqual(detail["states"].shape, (1, 145))
        self.assertEqual(len(logs), 1)

    def test_future_actuals_cannot_change_midnight_plan_or_executed_prefix(self):
        data = fixture()
        data.actual[31 * 144:32 * 144, 0] = 1200
        changed = copy.deepcopy(data)
        changed.actual[31 * 144 + 72:, 0] *= 30
        forecast = np.column_stack((np.full(144, 600.), np.zeros(144)))
        _, before, _ = dispatch.day_run(data, forecast, 31, MIN_SOC)
        _, after, _ = dispatch.day_run(changed, forecast, 31, MIN_SOC)
        np.testing.assert_array_equal(before["original"], after["original"])
        for key in ("charge", "discharge", "emergency", "surplus", "fees"):
            np.testing.assert_array_equal(before[key][:72], after[key][:72])
        np.testing.assert_array_equal(before["states"][:73], after["states"][:73])
        self.assertGreater(after["emergency"].sum(), before["emergency"].sum())

    def test_two_days_preserve_soc_and_match_separate_day_execution(self):
        data = fixture(2)
        data.actual[:144, 0] = 600
        data.actual[144:, 1] = 1200
        origins = []

        def predict(origin):
            origins.append(origin)
            return np.zeros((144, 2))

        summaries, detail, solvers = dispatch.replay_days(data, [0, 1], predict, 6000)
        self.assertEqual(origins, [0, 144])
        self.assertEqual(len(solvers), 2)
        self.assertAlmostEqual(detail["states"][0, -1], MIN_SOC)
        self.assertEqual(detail["states"][0, -1], detail["states"][1, 0])
        self.assertAlmostEqual(detail["states"][1, -1], MAX_SOC)
        second, second_detail, _ = dispatch.day_run(data, np.zeros((144, 2)), 1, MIN_SOC)
        self.assertAlmostEqual(second["total_cost"], summaries[1]["total_cost"])
        for key in detail:
            np.testing.assert_allclose(detail[key][1], second_detail[key], atol=1e-6)

    def test_greedy_extremes_respect_ac_power_efficiency_and_soc(self):
        grid = np.array([2000., 0., 0., 0.])
        load = np.array([0., 60000., 60000., 60000.])
        charge, discharge, emergency, surplus, states = dispatch.execute(
            grid, load, np.zeros(4), MAX_SOC - 10,
        )
        self.assertAlmostEqual(charge[0], 10 / ETA)
        self.assertAlmostEqual(states[1], MAX_SOC)
        self.assertGreater(surplus[0], 0)
        np.testing.assert_allclose(discharge[1:], POWER_ENERGY)
        np.testing.assert_allclose(np.diff(states), ETA * charge - discharge / ETA)
        np.testing.assert_allclose(grid + discharge + emergency - charge - surplus, load / 6)
        self.assertFalse(((charge > 1e-6) & (discharge > 1e-6)).any())
        self.assertTrue((states >= MIN_SOC).all() and (states <= MAX_SOC).all())
        _, d, e, _, s = dispatch.execute(np.zeros(1), np.array([600.]), np.zeros(1), MIN_SOC)
        np.testing.assert_array_equal(d, [0])
        np.testing.assert_array_equal(e, [100])
        np.testing.assert_array_equal(s, [MIN_SOC, MIN_SOC])

    def test_aggregate_cost_cvar_and_worst_day_for_short_replay(self):
        data = fixture()
        data.actual[31 * 144:32 * 144, 0] = 600
        data.actual[32 * 144:33 * 144, 0] = 1200
        summaries, detail, _ = dispatch.replay_days(
            data, [31, 32], lambda origin: np.zeros((144, 2)), MIN_SOC,
        )
        result = dispatch.aggregate(summaries)
        self.assertEqual(result["days"], 2)
        self.assertAlmostEqual(result["total_cost"], 216000)
        self.assertAlmostEqual(result["total_cost"], detail["fees"].sum())
        self.assertAlmostEqual(result["daily_cvar90"], 144000)
        self.assertAlmostEqual(result["worst_day_cost"], 144000)
        self.assertEqual(result["worst_date"], "2025-02-02")
        self.assertEqual(result["final_soc"], MIN_SOC)
        summaries[1]["initial_soc"] += 1
        with self.assertRaisesRegex(ValueError, "SOC"):
            dispatch.aggregate(summaries)

    def test_solver_failure_uses_frozen_v2_fallback_and_still_balances(self):
        metadata = {"status": 1, "message": "test timeout", "seconds": 0,
                    "feasible": False, "mip_gap": None}
        data = fixture()
        data.actual[31 * 144:32 * 144, 0] = 600
        forecast = np.column_stack((np.full(144, 600.), np.zeros(144)))
        with patch("experiments.common.neural_v2.physics.LinearModel.solve", return_value=(None, metadata)):
            summary, detail, solver = dispatch.day_run(data, forecast, 31, MIN_SOC)
        self.assertEqual(summary["fallback_count"], 1)
        self.assertEqual(summary["timeout_count"], 1)
        self.assertTrue(solver["fallback"])
        self.assertAlmostEqual(summary["total_cost"], 14400)
        np.testing.assert_array_equal(detail["emergency"], np.zeros(144))

    def test_invalid_boundaries_forecasts_and_nonconsecutive_days(self):
        data = fixture()
        valid = np.zeros((144, 2))
        for day in (-1, 365):
            with self.subTest(day=day), self.assertRaises(ValueError):
                dispatch.day_run(data, valid, day, 6000)
        for day in (1.5, True):
            with self.subTest(day=day), self.assertRaises(TypeError):
                dispatch.day_run(data, valid, day, 6000)
        for soc in (MIN_SOC - 1, MAX_SOC + 1, np.nan):
            with self.subTest(soc=soc), self.assertRaises(ValueError):
                dispatch.day_run(data, valid, 31, soc)
        with self.assertRaisesRegex(ValueError, "6000"):
            dispatch.day_run(data, valid, 0, MIN_SOC)
        for invalid in (np.zeros((144, 3)), np.zeros((143, 2)), valid - 1, valid + np.nan):
            with self.assertRaisesRegex(ValueError, "forecast"):
                dispatch.day_run(data, invalid, 31, MIN_SOC)
        for days in ([], [31, 33], [31, 31], [32, 31]):
            with self.subTest(days=days), self.assertRaisesRegex(ValueError, "consecutive"):
                dispatch.replay_days(data, days, lambda origin: valid, MIN_SOC)
        with self.assertRaises(ValueError):
            dispatch.aggregate([])


if __name__ == "__main__":
    unittest.main()
