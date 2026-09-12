"""Analytical cases, LP contracts, causality and physical execution."""

import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np

from experiments.problem2.linear_planning import model as m


class LinearPlanningTests(unittest.TestCase):
    def setUp(self):
        self.cfg = m.Config(ramp_kw=10000, cost_relaxation=0, cost_tolerance=0)

    def test_no_energy_available_requires_direct_purchase(self):
        result = m.plan_day([[600, 0]], [1], m.State(1200), self.cfg)
        self.assertAlmostEqual(result["grid"][0], 100)
        self.assertEqual(result["metadata"]["stage_count"], 2)
        self.assertEqual(len(result["metadata"]["stages"]), 2)
        self.assertTrue(all(s["integer_variables"] == 0 for s in result["metadata"]["stages"]))

    def test_efficient_price_arbitrage_analytical_cost(self):
        # Buy 100/0.9 at p=1, later discharge 100 instead of buying at p=2.
        r = m.plan_day([[0, 0], [600, 0]], [1, 2], m.State(1200), self.cfg)
        self.assertAlmostEqual(r["metadata"]["first_cost"], 100 / .9, places=5)
        self.assertAlmostEqual(r["grid"][1], 0, places=5)
        self.assertAlmostEqual(r["states"][-1], 1200, places=5)

    def test_actual_invoice_fivefold_and_no_refund(self):
        np.testing.assert_allclose(m.settle([100, 100], [0, 20], [1, 2]),
                                   [[100, 0], [200, 200]])

    def test_excess_pv_is_surplus_at_full_battery(self):
        r = m.plan_day([[0, 600]], [1], m.State(10800), self.cfg)
        self.assertAlmostEqual(r["surplus"][0], 100)
        self.assertLessEqual(r["metadata"]["max_overlap_kwh"], 1e-6)

    def test_cost_alone_does_not_logically_exclude_overlap(self):
        # Both idle and c=d=10 have zero grid cost at zero demand. The latter
        # dissipates stored energy while satisfying the relaxed equations.
        eta, c, d = np.sqrt(.9), 10., 10.
        soc = 6000 + eta * c - d / eta
        self.assertEqual(0 + d - c, 0)
        self.assertTrue(1200 < soc < 10800)
        self.assertEqual(1 * 0, 0)  # global lower bound for nonnegative grid cost
        r = m.plan_day([[0, 0]], [1], m.State(), self.cfg)
        self.assertAlmostEqual(r["charge"][0] + r["discharge"][0], 0)

    def test_smoothing_preserves_cost_budget(self):
        f = np.column_stack((np.full(12, 1800.), np.zeros(12)))
        p = np.tile([1, 1.01, 3], 4)
        r = m.plan_day(f, p, m.State(1200), m.Config())
        self.assertLessEqual(r["metadata"]["final_cost"], r["metadata"]["cost_budget"] + 1e-6)
        self.assertLessEqual(np.abs(np.diff(np.r_[0, r["net_power_kw"]])).max(), 1000 + 1e-6)

    def test_only_two_linprog_calls_for_plan(self):
        with patch.object(m, "linprog", wraps=m.linprog) as solve:
            m.plan_day([[600, 0]], [1], m.State(1200), self.cfg)
        self.assertEqual(solve.call_count, 2)
        self.assertTrue(all("integrality" not in call.kwargs for call in solve.call_args_list))

    def test_no_daily_soc_reset(self):
        _, d1, s1 = m.replay_day([[0, 0]], [1], lambda t: [0, 0], m.State(7000), self.cfg)
        _, d2, _ = m.replay_day([[0, 0]], [1], lambda t: [0, 0], s1, self.cfg)
        self.assertEqual(s1.soc, 7000)
        self.assertEqual(d1["states"][-1], d2["states"][0])

    def test_observations_follow_plan_and_do_not_change_prefix(self):
        f = np.column_stack((np.full(4, 600.), np.zeros(4)))
        actual1, actual2 = f.copy(), f.copy()
        actual2[2:, 0] += 600
        events = []
        original = m.plan_day

        def plan(*a, **kw):
            events.append("plan")
            return original(*a, **kw)

        def observe(t):
            self.assertEqual(events, ["plan"] + list(range(t)))
            events.append(t)
            return actual1[t]

        with patch.object(m, "plan_day", side_effect=plan):
            p1, d1, _ = m.replay_day(f, np.ones(4), observe, m.State(1200), self.cfg)
        p2, d2, _ = m.replay_day(f, np.ones(4), lambda t: actual2[t], m.State(1200), self.cfg)
        np.testing.assert_allclose(p1["grid"], p2["grid"])
        for key in ("charge", "discharge", "emergency", "surplus", "fees"):
            np.testing.assert_allclose(d1[key][:2], d2[key][:2])
        self.assertGreater(d2["fees"].sum(), d1["fees"].sum())

    def test_state_and_power_pass_across_day_boundary(self):
        cfg = m.Config()
        state = m.State(6000, 500)
        _, detail, end = m.replay_day([[600, 0], [600, 0]], [1, 1],
                                    lambda t: [600, 0], state, cfg)
        self.assertLessEqual(detail["audit"]["max_power_step_kw"], 1000 + 1e-6)
        self.assertAlmostEqual(end.previous_power_kw,
                               (detail["charge"][-1] - detail["discharge"][-1]) / cfg.dt)

    def test_invalid_input(self):
        for f, p in [([[np.nan, 0]], [1]), ([[-1, 0]], [1]), ([[0, 0]], [-1])]:
            with self.assertRaises(ValueError):
                m.plan_day(f, p)
        with self.assertRaises(ValueError):
            m.Config(cost_relaxation=-1)

    def test_infeasible_ramp_is_reported_not_silently_relaxed(self):
        # Empty battery, continuing discharge forced by ramp=0 is infeasible.
        with self.assertRaises(m.SolveError):
            m.plan_day([[600, 0]], [1], m.State(1200, -1000), replace(self.cfg, ramp_kw=0))

    def test_overlap_guard_blocks_nonphysical_actual_action(self):
        original = m._solve

        def overlap(*args, **kwargs):
            result = original(*args, **kwargs)
            if len(args) > 4:
                result["charge"][0] = result["discharge"][0] = 10
            return result

        with patch.object(m, "_solve", side_effect=overlap):
            with self.assertRaises(m.NonphysicalSolution):
                m.replay_day([[0, 0]], [1], lambda t: [0, 0], cfg=self.cfg)

    def test_positive_cost_with_ramp_can_still_produce_overlap(self):
        # At full SOC, a prior +1100 kW and 1000 kW ramp cap force at least
        # +100 kW now. A relaxed LP can dissipate energy by c,d>0, whereas
        # a real battery cannot keep charging here. This is a real LP case.
        cfg = m.Config(cost_relaxation=0, cost_tolerance=0)
        state = m.State(10800, 1100)
        r = m.plan_day([[0, 0]], [1], state, cfg)
        self.assertAlmostEqual(r["metadata"]["first_cost"], 100 / 6, places=5)
        self.assertGreater(r["metadata"]["max_overlap_kwh"], 1)
        with self.assertRaises(m.NonphysicalSolution):
            m.replay_day([[0, 0]], [1], lambda t: [0, 0], state, cfg)

    def test_roundoff_on_inherited_power_boundary_is_accepted(self):
        r = m.plan_day([[0, 0]], [1], m.State(6000, 5000 + 1e-10), self.cfg)
        self.assertEqual(r["metadata"]["stages"][0]["status"], 0)

    def test_timeout_is_not_reported_as_optimal(self):
        from types import SimpleNamespace
        result = SimpleNamespace(status=1, message="time limit", success=False, fun=0)
        with patch.object(m, "linprog", return_value=result):
            with self.assertRaises(m.SolveError):
                m.plan_day([[0, 0]], [1])


if __name__ == "__main__":
    unittest.main()
