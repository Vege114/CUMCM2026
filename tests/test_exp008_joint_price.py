"""Paired-price objective and physical initialization regression checks."""
import copy
import unittest

import numpy as np

from experiments.exp008.closed_loop import objective as scalar_objective
from experiments.exp008.joint_price_closed_loop import objective
from experiments.exp008.joint_price_physical import plan
from experiments.exp008.mode_planning_physical import plan as scalar_plan
from experiments.exp008.planner import HIGH, LOW, execute
from experiments.exp008.run_q4_joint_price import planning_inputs
from experiments.exp008.unified_forecast import UnifiedForecasts


class JointPriceTests(unittest.TestCase):
    def test_identical_price_rows_exactly_preserve_scalar_objective_and_gradient(self):
        rng = np.random.default_rng(582)
        q, paths = rng.uniform(50, 600, 19), rng.uniform(-100, 1300, (4, 19))
        price = rng.uniform(.3, 1.8, 19)
        for soc in (LOW+33., 6300., HIGH-10.):
            for mask in (None, np.arange(19) % 6 < 3):
                kwargs = {'charge_mask': mask, 'throughput': .017, 'variation': .003,
                          'terminal': .45, 'deadband': 11.}
                expected = scalar_objective(q, paths, price, soc, **kwargs)
                actual = objective(q, paths, np.broadcast_to(price, paths.shape), soc, **kwargs)
                self.assertEqual(actual[0], expected[0])
                np.testing.assert_array_equal(actual[1], expected[1])

    def test_matrix_price_objective_matches_physical_replay_and_finite_difference(self):
        rng = np.random.default_rng(956)
        q, paths = rng.uniform(100, 700, 17), rng.uniform(-200, 1100, (4, 17))
        prices = rng.uniform(.15, 1.8, paths.shape)
        mask = np.arange(17) % 5 < 3
        kwargs = {'charge_mask': mask, 'throughput': .009, 'variation': .002,
                  'terminal': .3, 'deadband': 13.}
        for soc in (LOW+50., 7300., HIGH-33.):
            value, gradient = objective(q, paths, prices, soc, **kwargs)
            expected = float(np.dot(q, prices.mean(0)))
            for j, net in enumerate(paths):
                actual = np.column_stack((6*np.maximum(net, 0), 6*np.maximum(-net, 0)))
                detail = execute(q, actual, prices[j], soc, charge_mask=mask, charge_deadband=13.)
                c, d, e = [detail[key] for key in ('charge', 'discharge', 'emergency')]
                expected += float((np.sum(5*prices[j]*e+.009*(c+d))
                                   + .002*6*np.abs(np.diff(c-d)).sum()
                                   - .3*(detail['states'][-1]-LOW))/len(paths))
            self.assertAlmostEqual(value, expected, places=9)
            epsilon = 1e-4
            finite = np.zeros(17)
            for t in range(17):
                plus, minus = q.copy(), q.copy()
                plus[t] += epsilon
                minus[t] -= epsilon
                finite[t] = (objective(plus, paths, prices, soc, **kwargs)[0]-
                             objective(minus, paths, prices, soc, **kwargs)[0])/(2*epsilon)
            np.testing.assert_allclose(gradient, finite, atol=2e-5, rtol=2e-5)

    def test_identical_price_rows_preserve_physical_plan_and_varying_prices_stay_physical(self):
        paths = np.array([[-600., 0., 800., 500.], [0., 500., 1400., -200.],
                          [400., -500., 300., 800.]])
        price = np.array([.4, .6, 1.3, .8])
        kwargs = {'block_slots': 2, 'switching': 20., 'seconds': 5., 'gap': 1e-8}
        expected = scalar_plan(paths, price, 2300., **kwargs)
        actual = plan(paths, price, 2300., scenario_prices=np.broadcast_to(price, paths.shape), **kwargs)
        for key in ('purchase', 'allowed_charge', 'scenario_charge', 'scenario_discharge',
                    'scenario_emergency', 'scenario_surplus', 'scenario_states'):
            np.testing.assert_array_equal(actual[key], expected[key])
        varying = plan(paths, price, 2300., scenario_prices=price[None, :]*np.array([[.7], [1.5], [1.2]]), **kwargs)
        self.assertTrue(varying['metadata']['scenario_physical_checks']['passed'])
        self.assertFalse(varying['metadata']['nonanticipative_recourse_certificate'])

    def test_paired_inputs_ignore_current_future_actual_price_and_load(self):
        original = UnifiedForecasts(calibration='ridge_28')
        for day in (31, 90):
            data = copy.copy(original.data)
            data.actual = original.data.actual.copy()
            data.actual[day*144:] = 1e7
            changed = UnifiedForecasts(data=data, calibration='ridge_28')
            expected, actual = planning_inputs(original, day), planning_inputs(changed, day)
            for key in ('net_paths', 'scenario_prices', 'selected_indices', 'predicted_price', 'history_origins'):
                np.testing.assert_array_equal(actual[key], expected[key])
            self.assertTrue(np.all(actual['scenario_prices'] >= .01))
            history = original.net_error_paths(day, '4-2')
            issue = original.get(day, scenario='4-2')
            np.testing.assert_array_equal(actual['scenario_prices'],
                                          np.maximum(.01, issue['price'][None, :]+history['price_errors']))


if __name__ == '__main__':
    unittest.main()
