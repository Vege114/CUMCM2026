import sys
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from strict_backend import StrictBackend  # noqa: E402

from experiments.problem2.stochastic_lp import model, run  # noqa: E402


class StrictBackendTests(unittest.TestCase):
    def test_one_interval_manual_cost(self):
        tree = model.Tree(np.array([0, 0]), np.array([-1, -1]), np.array([.7, .3]),
                          np.array([[480., 0], [840., 0]]), 1)
        r = StrictBackend().solve_tree(tree, np.ones(1), cfg=replace(model.Config(), delta=0), battery_enabled=False)
        self.assertAlmostEqual(r['metadata']['first_cost'], 140.)
        self.assertEqual(r['metadata']['stage_count'], 2)
        self.assertEqual(r['metadata']['stages'][0]['integer_variables'], 2)

    def test_forced_overlap_is_infeasible(self):
        tree = model.Tree.deterministic([[0., 0.]])
        state = model.State(10800., 1100.)
        self.assertGreater(model.solve_tree(tree, np.ones(1), state)['metadata']['max_overlap_kwh'], 1)
        backend = StrictBackend()
        with self.assertRaises(model.SolveError):
            backend.solve_tree(tree, np.ones(1), state)
        self.assertEqual(backend.last_result.status, 2)

    def test_branch_actions_exclusive_and_state_shared(self):
        tree = model.Tree(np.array([0, 1, 1]), np.array([-1, 0, 0]), np.array([1., .5, .5]),
                          np.array([[500., 0], [600., 0], [200., 1000.]]), 2)
        r = StrictBackend().solve_tree(tree, np.array([.3, 1.2]))
        for prefix in ['', 'first_']:
            self.assertLessEqual(float(np.minimum(r[prefix + 'charge'], r[prefix + 'discharge']).max()), 1e-6)
            for child in [1, 2]:
                self.assertAlmostEqual(r[prefix + 'end_soc'][child], r[prefix + 'end_soc'][0]
                                       + np.sqrt(.9) * r[prefix + 'charge'][child]
                                       - r[prefix + 'discharge'][child] / np.sqrt(.9))
        self.assertLessEqual(r['metadata']['second_cost'], r['metadata']['budget'] + 1e-6)

    def test_fixed_purchase_d5a_and_patch_restoration(self):
        before = model.linprog
        r = StrictBackend().solve_tree(model.Tree.deterministic([[0., 0.]]), np.ones(1),
                                       model.State(5000., 1100.), fixed_grid=np.zeros(1))
        self.assertIs(model.linprog, before)
        self.assertEqual(r['grid'][0], 0)
        self.assertGreater(r['emergency'][0], 0)
        self.assertGreater(r['charge'][0], 0)
        self.assertLessEqual(abs(r['discharge'][0]), 1e-6)

    def test_future_actual_does_not_change_prior_strict_actions(self):
        forecast = np.tile([2000., 0.], (6, 1))
        altered = forecast.copy()
        altered[4:, 0] = 7000.
        bundle = {'paths_kw': forecast[None], 'scale_kw': np.ones(2)}
        def replay(actual):
            with patch.object(run, 'solve_tree', StrictBackend().solve_tree):
                return run.replay_day(forecast, bundle, np.linspace(.3, 1.2, 6), lambda t: actual[t],
                                      model.State(4000., 0.), model.Config())
        p, a, _, _ = replay(forecast)
        q, b, _, _ = replay(altered)
        np.testing.assert_array_equal(p['grid'], q['grid'])
        for key in ['charge', 'discharge', 'emergency', 'net_power_kw']:
            np.testing.assert_allclose(a[key][:4], b[key][:4], atol=1e-6, rtol=0)


if __name__ == '__main__':
    unittest.main()
