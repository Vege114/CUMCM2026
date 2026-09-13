import unittest
import numpy as np

from experiments.exp008.mode_planning_physical import plan
from experiments.exp008.weighted_mode_planning import plan_weighted,reduce_paths


class WeightedPhysicalModesTests(unittest.TestCase):
    def test_uniform_is_original_on_physical_counterexample(self):
        paths=np.array([[-1000.,0.]]*2+[[0.,1000.]])
        price=np.array([.4,1.])
        kwargs={'block_slots':1,'switching':0.,'final':True,'seconds':10.,'gap':0.}
        old=plan(paths,price,1200.,**kwargs)
        new=plan_weighted(paths,price,1200.,**kwargs)
        for key in ('purchase','scenario_charge','scenario_discharge','scenario_emergency','scenario_states'):
            np.testing.assert_allclose(old[key],new[key],atol=1e-6)
        self.assertTrue(new['metadata']['scenario_physical_checks']['passed'])

    def test_rational_weights_equal_replicated_scenarios(self):
        price=np.array([.4,1.]);paths=np.array([[-1000.,0.],[0.,1000.]])
        kwargs={'block_slots':1,'switching':0.,'final':True,'seconds':10.,'gap':0.}
        weighted=plan_weighted(paths,price,1200.,weights=np.array([2/3,1/3]),**kwargs)
        repeated=plan(np.repeat(paths,[2,1],axis=0),price,1200.,**kwargs)
        np.testing.assert_allclose(weighted['purchase'],repeated['purchase'],atol=1e-6)
        cost_w=weighted['purchase']@price+np.sum(5*price*weighted['scenario_emergency']*np.array([2/3,1/3])[:,None])
        cost_r=repeated['purchase']@price+np.mean((5*price*repeated['scenario_emergency']).sum(axis=1))
        self.assertAlmostEqual(cost_w,cost_r,places=6)

    def test_reduction_preserves_population_and_whole_paths(self):
        paths=np.array([[0.,0.],[.1,0.],[5.,6.],[5.1,6.],[10.,11.],[10.1,11.]])
        ids,weights,audit=reduce_paths(paths,np.ones(2))
        self.assertEqual(len(ids),3)
        np.testing.assert_allclose(weights,np.ones(3)/3)
        self.assertEqual(len(np.unique(audit['cluster_labels'])),3)
        ids_again,weights_again,_=reduce_paths(paths,np.ones(2))
        np.testing.assert_array_equal(ids,ids_again)
        np.testing.assert_array_equal(weights,weights_again)

    def test_invalid_weights_rejected(self):
        with self.assertRaises(ValueError):
            plan_weighted(np.ones((2,2)),np.ones(2),1200.,weights=np.array([.2,.2]))
