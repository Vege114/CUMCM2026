"""Independent physical and information-boundary checks; no neural training."""

import unittest
from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from experiments.problem2.rl_planning.environment import (
    CausalDayCache,
    PlanningBatch,
    RewardConfig,
    collect_rollout,
    plan_day,
    score_plans,
)


class DeterministicActor:
    """Exercise charging, idling and discharging without importing TensorFlow."""

    def act(self, observation, deterministic=False):
        slot = np.rint(observation[:, 2] * 144).astype(np.int64)
        actions = ((slot // 8) % 9) * 5 + (slot // 7) % 5
        return actions, np.zeros(len(actions)), np.zeros(len(actions))


class RLPlanningTests(unittest.TestCase):
    def setUp(self):
        self.price = np.r_[np.full(48, .3), np.full(48, 1.2), np.full(48, .6)]
        self.forecast = np.column_stack((np.full(144, 6000.), np.zeros(144)))
        self.supports = np.tile(np.linspace(850, 1150, 9), (144, 1))

    def make_plan(self, initial_soc=6000.):
        return plan_day(DeterministicActor(), self.forecast, self.supports,
                        self.price, initial_soc, initial_mode=-1, initial_power=-500.)

    def assert_physics(self, detail, plan):
        # Reconstruct equations from exported quantities, not executor helpers.
        eta, tolerance = float(np.sqrt(.9)), 1e-6
        q, c, d, e, w, states = (detail[key] for key in
                                ("final", "charge", "discharge", "emergency",
                                 "surplus", "states"))
        for values in (q, c, d, e, w, states):
            self.assertTrue(np.isfinite(values).all())
            self.assertGreaterEqual(values.min(), -tolerance)
        np.testing.assert_array_equal(detail["original"], plan["purchase"])
        np.testing.assert_array_equal(q, plan["purchase"])
        np.testing.assert_allclose(
            q + (detail["actual"][:, 1] - detail["actual"][:, 0]) / 6 + d + e,
            c + w, atol=tolerance, rtol=0,
        )
        np.testing.assert_allclose(np.diff(states), eta*c - d/eta,
                                   atol=tolerance, rtol=0)
        self.assertEqual(states[0], plan["states"][0])
        self.assertGreaterEqual(states.min(), 1200-tolerance)
        self.assertLessEqual(states.max(), 10800+tolerance)
        self.assertLessEqual(max(c.max(), d.max()), 5000/6+tolerance)
        self.assertFalse(((c > tolerance) & (d > tolerance)).any())
        self.assertFalse(((c > tolerance) & (e > tolerance)).any())
        expected = np.column_stack((self.price*q, np.zeros((144, 2)),
                                    5*self.price*e))
        np.testing.assert_allclose(detail["fees"], expected, atol=tolerance, rtol=0)

    def test_full_day_must_be_locked_before_execution(self):
        env = PlanningBatch(self.forecast[None], self.supports[None],
                            self.price, [6000.])
        with self.assertRaisesRegex(ValueError, "complete day"):
            env.plans()
        for slot in range(144):
            obs = env.observe()
            self.assertEqual(obs.shape, (1, 29))
            self.assertTrue(np.isfinite(obs).all())
            env.step(np.array([22 if slot % 2 else 42]))
        self.assertEqual(len(env.plans()), 1)
        with self.assertRaisesRegex(ValueError, "complete"):
            env.observe()

    def test_hidden_training_actuals_cannot_change_policy_observations_or_plans(self):
        example = {"day": 10, "forecast": self.forecast,
                   "supports": self.supports, "actual": self.forecast.copy()}
        changed = {**example, "actual": np.column_stack((np.full(144, 90000.),
                                                         np.zeros(144)))}
        first, detail_first = collect_rollout(
            DeterministicActor(), [example], self.price,
            np.random.default_rng(42), n_envs=3,
        )
        second, detail_second = collect_rollout(
            DeterministicActor(), [changed], self.price,
            np.random.default_rng(42), n_envs=3,
        )
        for key in ("obs", "actions", "old_logprob", "values"):
            np.testing.assert_array_equal(first[key], second[key])
        for before, after in zip(detail_first, detail_second, strict=True):
            np.testing.assert_array_equal(before["original"], after["original"])
        self.assertFalse(np.array_equal(first["rewards"], second["rewards"]))

    def test_real_archive_future_mutations_preserve_issued_features_and_training_cutoff(self):
        from experiments.problem2.exp003.data import Data
        from experiments.problem2.exp004.predict import ForecastStore

        data, store = Data(), ForecastStore("no_season", seed=42)
        for day in (31, 60):
            with self.subTest(day=day):
                mutated = data.actual.copy()
                mutated[day*144:, 0], mutated[day*144:, 1] = 999999., 0.
                changed = SimpleNamespace(actual=mutated, fixed_price=data.fixed_price)
                before, after = CausalDayCache(data, store), CausalDayCache(changed, store)
                f1, s1, a1 = before.issued(day)
                f2, s2, a2 = after.issued(day)
                np.testing.assert_array_equal(f1, f2)
                np.testing.assert_array_equal(s1, s2)
                self.assertEqual(a1["information_cutoff"], day*144)
                self.assertLess(a1["max_observed_index"], day*144)
                self.assertEqual(a1["training_origins"], a2["training_origins"])
                p1 = plan_day(DeterministicActor(), f1, s1, data.fixed_price, 1421.7991105135516)
                p2 = plan_day(DeterministicActor(), f2, s2, data.fixed_price, 1421.7991105135516)
                for key in p1:
                    np.testing.assert_array_equal(p1[key], p2[key])
                t1, t2 = before.training_days(day), after.training_days(day)
                self.assertEqual([x["day"] for x in t1], list(range(8, day)))
                self.assertEqual([x["day"] for x in t1], [x["day"] for x in t2])
                for x, y in zip(t1, t2, strict=True):
                    self.assertLess(x["day"], day)
                    self.assertLessEqual((x["day"]+1)*144, day*144)
                    audit = x["audit"]
                    self.assertEqual(audit["information_cutoff"], x["day"]*144)
                    self.assertLess(audit["max_observed_index"], x["day"]*144)
                    self.assertTrue(all(o+144 <= x["day"]*144
                                        for o in audit["training_origins"]))
                    for key in ("forecast", "supports", "actual"):
                        np.testing.assert_array_equal(x[key], y[key])
                    if x["day"] < 31:
                        self.assertEqual(audit["residual_source"],
                                         "january_periodic_baseline_per_slot")
                with self.assertRaises(ValueError):
                    before.training_days(8)

    def test_extreme_actuals_and_soc_boundaries_obey_independent_physics(self):
        extremes = [np.column_stack((np.full(144, 90000.), np.zeros(144))),
                    np.column_stack((np.zeros(144), np.full(144, 90000.))),
                    np.zeros((144, 2))]
        alternating = np.zeros((144, 2))
        alternating[::2, 0], alternating[1::2, 1] = 90000., 90000.
        extremes.append(alternating)
        for initial_soc in (1200., 10800., 6042.75):
            for case, actual in enumerate(extremes):
                with self.subTest(initial_soc=initial_soc, case=case):
                    plan = self.make_plan(initial_soc)
                    rewards, details = score_plans([plan], [actual], self.price)
                    self.assertTrue(np.isfinite(rewards).all())
                    np.testing.assert_allclose(
                        np.diff(plan["states"]),
                        np.sqrt(.9)*plan["charge"]-plan["discharge"]/np.sqrt(.9),
                        atol=1e-6, rtol=0,
                    )
                    self.assertFalse(((plan["charge"] > 1e-6)
                                      & (plan["discharge"] > 1e-6)).any())
                    self.assertGreaterEqual(plan["states"].min(), 1200-1e-6)
                    self.assertLessEqual(plan["states"].max(), 10800+1e-6)
                    self.assert_physics(details[0], plan)

    def test_future_execution_changes_do_not_change_executed_prefix(self):
        plan = self.make_plan()
        actual = self.forecast.copy()
        changed = actual.copy()
        changed[70:, 0], changed[70:, 1] = 90000., 0.
        rewards1, details1 = score_plans([plan], [actual], self.price)
        rewards2, details2 = score_plans([plan], [changed], self.price)
        for key in ("charge", "discharge", "emergency", "surplus", "fees"):
            np.testing.assert_array_equal(details1[0][key][:70], details2[0][key][:70])
        np.testing.assert_array_equal(details1[0]["states"][:71], details2[0]["states"][:71])
        np.testing.assert_array_equal(rewards1[:70], rewards2[:70])
        np.testing.assert_array_equal(details1[0]["original"], details2[0]["original"])
        self.assertNotEqual(details1[0]["fees"].sum(), details2[0]["fees"].sum())

    def test_terminal_shaping_is_only_last_reward_and_disabled_on_final_day(self):
        env = PlanningBatch(self.forecast[None], self.supports[None], self.price, [6000.])
        for _ in range(144):
            env.step(np.array([42]))
        plan = env.plans()[0]
        config = RewardConfig()
        shaped, shaped_detail = score_plans([plan], [self.forecast], self.price, config)
        final, final_detail = score_plans([plan], [self.forecast], self.price, config,
                                        final_days=[True])
        disabled, _ = score_plans([plan], [self.forecast], self.price,
                                  replace(config, terminal_value=False))
        np.testing.assert_array_equal(final, disabled)
        np.testing.assert_array_equal(shaped[:-1], final[:-1])
        states = shaped_detail[0]["states"]
        terminal = self.price.min()/np.sqrt(.9)*(states[-1]-states[0])/1000
        self.assertGreater(terminal, 0)
        self.assertAlmostEqual(float(shaped[-1, 0]-final[-1, 0]), terminal, places=5)
        np.testing.assert_array_equal(shaped_detail[0]["fees"], final_detail[0]["fees"])

    def test_secondary_penalties_change_reward_but_not_real_billed_cost(self):
        plan = self.make_plan()
        unpenalized = RewardConfig(throughput_yuan_per_kwh=0, reversal_yuan=0,
                                   ramp_yuan_per_kw=0, terminal_value=False)
        penalized = replace(unpenalized, throughput_yuan_per_kwh=.02,
                            reversal_yuan=2., ramp_yuan_per_kw=.003)
        reward0, detail0 = score_plans([plan], [self.forecast], self.price, unpenalized,
                                      initial_modes=[-1], initial_powers=[-500.])
        reward1, detail1 = score_plans([plan], [self.forecast], self.price, penalized,
                                      initial_modes=[-1], initial_powers=[-500.])
        for key in detail0[0]:
            np.testing.assert_array_equal(detail0[0][key], detail1[0][key])
        detail = detail1[0]
        self.assert_physics(detail, plan)
        np.testing.assert_allclose(reward0[:, 0], -detail["fees"].sum(axis=1)/1000,
                                   atol=2e-5, rtol=0)
        c, d = detail["charge"], detail["discharge"]
        nonidle = np.sign(c-d)[np.abs(c-d) > 1e-6]
        sequence = np.r_[-1, nonidle]
        reversals = np.sum(sequence[1:]*sequence[:-1] == -1)
        ramp = np.abs(np.diff(np.r_[-500., 6*(c-d)])).sum()
        penalty = .02*(c+d).sum() + 2*reversals + .003*ramp
        self.assertGreater(reversals, 0)
        self.assertGreater(penalty, 0)
        self.assertAlmostEqual(float(reward0.astype(float).sum()-reward1.astype(float).sum()),
                               penalty/1000, places=4)


if __name__ == "__main__":
    unittest.main()
