"""Numerical regression for update opportunity weights, separate from billing."""
import unittest
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np

from experiments.exp008.planner import Settings, plan


class UpdatePlanningTests(unittest.TestCase):
    def setUp(self):
        self.forecast = {
            'load_kw': np.array([1700., 2100., 2700., 1500., 800., 3500., 2500., 1700.]),
            'pv_kw': np.zeros(8),
            'price': np.array([.4, .4, .7, .7, .7, 1.3, 1.3, .5]),
        }
        self.paths = np.array([[-80., 20., 100., -50., -70., 100., 90., 0.],
                               [120., 70., -60., -20., 160., 40., -80., 20.]])

    def test_weighted_plan_exactly_matches_the_independent_vector_formulation(self):
        vector = np.r_[np.full(3, 5.), np.full(5, 1.5)]
        settings = SimpleNamespace(**{**asdict(Settings()), 'emergency_weight': vector})
        expected = plan(self.forecast, 1400., self.paths, settings=settings)
        actual = plan(self.forecast, 1400., self.paths,
                      settings=Settings(future_shortfall_weight=1.5), next_update_slot=3)
        explicit = plan(self.forecast, 1400., self.paths,
                        next_update_slot=3, future_shortfall_weight=1.5)
        for key in ('purchase', 'charge', 'discharge', 'states'):
            np.testing.assert_array_equal(actual[key], expected[key])
            np.testing.assert_array_equal(explicit[key], expected[key])
        self.assertFalse(actual['metadata']['nonanticipative_recourse_certificate'])
        self.assertTrue(actual['metadata']['opportunity_proxy_active'])

    def test_no_next_release_and_final_block_retain_original_shortfall_weight(self):
        expected = plan(self.forecast, 1400., self.paths)
        for boundary in (None, 8):
            actual = plan(self.forecast, 1400., self.paths,
                          settings=Settings(future_shortfall_weight=1.5),
                          next_update_slot=boundary)
            for key in ('purchase', 'charge', 'discharge', 'states'):
                np.testing.assert_array_equal(actual[key], expected[key])
            np.testing.assert_array_equal(actual['metadata']['planning_emergency_weights'], 5.)
            self.assertFalse(actual['metadata']['opportunity_proxy_active'])

    def test_invalid_relative_release_boundary_or_weight_fails(self):
        for boundary in (0, -1, 9, 2.5):
            with self.assertRaises(ValueError):
                plan(self.forecast, 1400., next_update_slot=boundary, future_shortfall_weight=1.5)
        for weight in (0., -1., np.nan, np.inf):
            with self.assertRaises(ValueError):
                plan(self.forecast, 1400., next_update_slot=3, future_shortfall_weight=weight)


if __name__ == '__main__':
    unittest.main()
