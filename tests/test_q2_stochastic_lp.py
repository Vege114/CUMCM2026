import unittest
from dataclasses import replace

import numpy as np

from experiments.problem2.stochastic_lp.model import Config, State, Tree, settle, solve_tree
from experiments.problem2.stochastic_lp.run import replay_day
from experiments.problem2.stochastic_lp.scenarios import information_tree, sample_paths


class StochasticLPTests(unittest.TestCase):
    def test_paper_one_interval_example(self):
        tree = Tree(np.array([0, 0]), np.array([-1, -1]), np.array([.7, .3]),
                    np.array([[80 * 6., 0], [140 * 6., 0]]), 1)
        result = solve_tree(tree, np.ones(1), cfg=replace(Config(), delta=0),
                            battery_enabled=False)
        self.assertAlmostEqual(result["metadata"]["first_cost"], 140)
        self.assertAlmostEqual(result["grid"][0], 140, places=3)
        self.assertEqual(result["metadata"]["stage_count"], 2)

    def test_shared_node_then_branch(self):
        tree = Tree(np.array([0, 1, 1]), np.array([-1, 0, 0]), np.array([1., .5, .5]),
                    np.array([[500, 0], [600, 0], [200, 1000]]), 2)
        r = solve_tree(tree, np.array([.3, 1.2]))
        # A single root action/state is physically inherited by both children.
        eta = np.sqrt(.9)
        for child in [1, 2]:
            self.assertAlmostEqual(r["end_soc"][child], r["end_soc"][0]
                                   + eta * r["charge"][child] - r["discharge"][child] / eta)
        self.assertLessEqual(r["metadata"]["second_cost"], r["metadata"]["budget"] + 1e-6)

    def test_full_battery_hard_ramp_exposes_overlap(self):
        r = solve_tree(Tree.deterministic([[0, 0]]), np.ones(1), State(10800, 1100))
        self.assertGreater(r["metadata"]["max_overlap_kwh"], 1)
        self.assertFalse(r["metadata"]["certified_no_overlap"])
        self.assertLessEqual(r["end_soc"].max(), 10800 + 1e-6)

    def test_d5a_emergency_can_charge(self):
        r = solve_tree(Tree.deterministic([[0, 0]]), np.ones(1), State(5000, 1100),
                       fixed_grid=np.zeros(1))
        self.assertGreater(r["charge"][0], 0)
        self.assertGreater(r["emergency"][0], 0)

    def test_bad_probability_and_parent(self):
        for tree in [Tree(np.array([0, 1]), np.array([-1, 0]), np.array([1., .5]),
                          np.zeros((2, 2)), 2),
                     Tree(np.array([0, 1]), np.array([-1, -1]), np.ones(2), np.zeros((2, 2)), 2)]:
            with self.assertRaises(ValueError):
                tree.validate()

    def test_invoice_no_tree_probability_or_auxiliary_fee(self):
        invoice = settle(np.array([10., 20]), np.array([2., 0]), np.array([.3, 1.2]))
        np.testing.assert_allclose(invoice, [[3, 3], [24, 0]])

    def test_scenario_donor_visibility_and_joint_residual(self):
        f = np.full((144, 2), 100.)
        history = np.stack([f, f])
        actual = history + np.array([[[5., -2.]], [[10., 3.]]])
        b = sample_paths(f, history, actual, np.array([0, 144]), 288, np.ones(144))
        np.testing.assert_allclose(b["paths_kw"], actual)
        np.testing.assert_allclose(b["weights"], [.5, .5])
        with self.assertRaises(ValueError):
            sample_paths(f, history, actual, np.array([0, 288]), 288, np.ones(144))

    def test_empty_history_is_labelled_point_fallback(self):
        f = np.full((144, 2), 10.)
        b = sample_paths(f, np.empty((0, 144, 2)), np.empty((0, 144, 2)),
                         np.array([], dtype=int), 4464, np.zeros(144))
        self.assertTrue(b["fallback_point_only"])
        self.assertEqual(len(b["donor_origins"]), 0)
        self.assertTrue((b["paths_kw"][:, :, 1] == 0).all())

    def test_tree_splits_by_prefix_never_future(self):
        paths = np.ones((8, 8, 2)) * 100
        paths[4:, :4, 0] = 300
        tree, memberships = information_tree(paths, np.ones(2), branches=(3, 6))
        changed = paths.copy()
        changed[:, 4:] = np.arange(8)[:, None, None] * 1000
        other, groups = information_tree(changed, np.ones(2), branches=(3, 6))
        ids = tree.time <= 3
        np.testing.assert_array_equal(tree.time[ids], other.time[other.time <= 3])
        np.testing.assert_allclose(tree.supply_kw[ids], other.supply_kw[other.time <= 3])
        self.assertEqual(memberships[:int(ids.sum())], groups[:int(ids.sum())])
        self.assertTrue(all(len(group) >= 2 for group in memberships))

    def test_replay_future_actual_cannot_change_prior_actions(self):
        f = np.tile([2000., 0.], (6, 1))
        price = np.linspace(.3, 1.2, 6)
        bundle = {"paths_kw": f[None], "scale_kw": np.ones(2)}
        original = f.copy()
        altered = f.copy()
        altered[4:, 0] = 7000
        observed = []
        def observation(t):
            observed.append(t)
            return original[t]
        p, d, _, _ = replay_day(f, bundle, price, observation, State(4000, 0), Config())
        q, e, _, _ = replay_day(f, bundle, price, lambda t: altered[t], State(4000, 0), Config())
        self.assertEqual(observed, list(range(6)))
        np.testing.assert_allclose(p["grid"], q["grid"])
        for key in ["charge", "discharge", "emergency", "net_power_kw"]:
            np.testing.assert_allclose(d[key][:4], e[key][:4])

    def test_fixed_grid_is_never_adjusted(self):
        grid = np.array([100., 200.])
        r = solve_tree(Tree.deterministic([[4000, 0], [3000, 0]]), np.array([.3, 1.2]),
                       State(1200, 0), fixed_grid=grid)
        np.testing.assert_array_equal(r["grid"], grid)


if __name__ == "__main__":
    unittest.main()
