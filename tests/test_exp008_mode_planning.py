import unittest

import numpy as np

from experiments.exp008.mode_planning import plan
from experiments.exp008.planner import ETA, LIMIT, LOW
from experiments.exp008.verify import goal_check


class SharedModeTests(unittest.TestCase):
    def test_two_period_arbitrage_matches_analytic_solution(self):
        result=plan(np.array([[0.,2000.]]),np.array([.1,1.]),LOW,
                    block_slots=1,switching=10.,wear=0.,final=True)
        np.testing.assert_allclose(result['purchase'],[LIMIT,2000-ETA**2*LIMIT],atol=1e-5)
        np.testing.assert_array_equal(result['allowed_charge'],[True,False])
        self.assertTrue(result['metadata']['feasible'])

    def test_expensive_reversal_keeps_energy_unused(self):
        result=plan(np.array([[0.,2000.]]),np.array([.1,1.]),LOW,
                    block_slots=1,switching=1000.,wear=0.,final=True)
        np.testing.assert_allclose(result['purchase'],[0.,2000.],atol=1e-5)
        self.assertFalse(result['metadata']['nonanticipative_recourse_certificate'])

    def test_partial_run_cannot_compare_to_full_year(self):
        battery={'direction_reversals':10,'active_slots':100,
                 'throughput_kwh':10000,'simultaneous_slots':0}
        status=goal_check(10000,battery,True,False)
        self.assertFalse(status['passed'])
        self.assertFalse(status['checks']['cost_reduced_at_least_8pct'])
        self.assertIsNone(status['cost_reduction_pct'])


if __name__ == '__main__':
    unittest.main()
