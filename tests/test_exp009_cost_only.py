import unittest
import json
from pathlib import Path
import numpy as np

from experiments.exp009.q2_cost_model import plan
from experiments.exp008.closed_loop import objective


class CostOnlyContrastTests(unittest.TestCase):
    def test_alternating_free_solar_can_exceed_old_eight_switch_cap(self):
        paths=np.tile(np.tile([-800.,720.],10),(3,1))
        result=plan(paths,np.ones(20),1200.,final=True,seconds=5.)
        self.assertLess(result['purchase'].sum(),1e-5)
        self.assertGreater(result['metadata']['planned_changes_including_boundary'],8)
        self.assertTrue(result['metadata']['scenario_physical_checks']['passed'])
        self.assertIsNone(result['metadata']['daily_planned_switch_cap'])

    def test_cost_objective_matches_physical_bill_without_intensity_terms(self):
        # First slot stores1kWh AC, later releases0.9kWh; only0.1kWh emergency remains.
        val,_=objective(np.array([1.,0.]),np.array([[0.,1.]]),np.array([.4,1.]),1200.,throughput=0.,variation=0.,terminal=0.)
        self.assertAlmostEqual(val,.4+5*.1,places=10)

    def test_q1_really_has_one_solver_stage_not_zero_budget_second_stage(self):
        from experiments.exp009 import q1_cost_model as model
        root=Path(__file__).resolve().parents[1]
        data=model.read_data(root/'data/raw/附件1.csv')
        result=model.solve(data,delta=0.,ramp=1000.,up=3,down=2,eta_rt=.9)
        previous=json.loads((root/'data/results/exp008/q1/revised.json').read_text())
        self.assertEqual(len(result['stages']),1)
        self.assertEqual(result['metrics']['violations'],0)
        self.assertAlmostEqual(result['metrics']['cost'],previous['stages'][0]['cost'],places=4)
        self.assertLess(result['metrics']['cost'],previous['stages'][1]['cost'])


if __name__=='__main__':unittest.main()
