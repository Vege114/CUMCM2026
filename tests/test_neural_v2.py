import copy
import unittest
from unittest.mock import patch

import numpy as np

from experiments.common.neural_v2.data import Data, month_origins, split_origins
from experiments.common.neural_v2.physics import ETA, MAX_SOC, MIN_SOC, execute, settle
from experiments.common.neural_v2.predict import ForecastStore
from experiments.common.neural_v2.risk import optimize, weighted_cvar
from experiments.common.neural_v2.scenarios import ScenarioFactory


def setUpModule():
    # SciPy/HiGHS owns a process-global thread pool. The frozen v1 tests use
    # its default size, while v2 explicitly uses one thread. Isolate the test
    # module just as the production seven-stage CLI uses separate processes.
    from scipy.optimize._highspy._core import _Highs
    _Highs.resetGlobalScheduler(True)


def tearDownModule():
    setUpModule()


class ForecastInformationTests(unittest.TestCase):
    def setUp(self):self.data=Data()

    def test_monthly_labels_and_features_are_causal(self):
        for month in range(2,13):
            train,val,cutoff=split_origins(month);asof=month_origins(month)[0]
            self.assertLessEqual(train[-1]+144,cutoff)
            self.assertLessEqual(val[-1]+144,asof)
            changed=copy.deepcopy(self.data);changed.actual[asof:]*=7
            changed._forecasts={o:p.copy()*(11 if o>asof else 1) for o,p in self.data.forecasts.items()}
            for a,b in zip(self.data.features([asof],cutoff),changed.features([asof],cutoff)):
                np.testing.assert_array_equal(a,b)

    def test_independent_numeric_branches(self):
        origin=31*144;changed=copy.deepcopy(self.data);changed.actual[:origin,2]*=3
        a=self.data.features([origin],24*144)[0];b=changed.features([origin],24*144)[0]
        np.testing.assert_array_equal(a[:,:,[0,1,3]],b[:,:,[0,1,3]])

    def test_january_tuning_labels_exclude_february(self):
        train, validation, cutoff = split_origins(2)
        changed = copy.deepcopy(self.data)
        changed.actual[31*144:] *= 100
        changed._forecasts = {o: values*(100 if o >= 31*144 else 1)
                              for o, values in self.data.forecasts.items()}
        np.testing.assert_array_equal(self.data.labels(np.r_[train, validation]),
                                      changed.labels(np.r_[train, validation]))
        for a, b in zip(self.data.features(validation, cutoff), changed.features(validation, cutoff)):
            np.testing.assert_array_equal(a, b)

    def test_no_attachment_three_for_question_two(self):
        class NoForecasts(Data):
            @property
            def forecasts(self):raise AssertionError("attachment 3 forbidden")
        data=NoForecasts();origin=31*144
        data.features([origin],24*144,issued=False)
        store=ForecastStore(data,kind="periodic")
        prediction=store.get(origin,issued=False)
        bundle=ScenarioFactory(data,store).build(origin,prediction,"4-2")
        self.assertFalse(bundle["metadata"]["issued_forecast_allowed"])
        self.assertTrue((bundle["groups"]==0).all())

    def test_hourly_linear_integral(self):
        origin=31*144;self.data.actual[origin-1,1]=0
        result=self.data.interpolate_hourly(np.arange(1,25)*60.,origin)
        np.testing.assert_allclose(result,(np.arange(144)+.5)*10)
        self.assertAlmostEqual(result.sum()/6,24*1440/2)

    def test_scenario_tree_ignores_future_actuals_and_issues(self):
        origin=60*144
        changed=copy.deepcopy(self.data);changed.actual[origin:]*=20
        changed._forecasts={o:p.copy()*(17 if o>origin else 1) for o,p in self.data.forecasts.items()}
        a_store=ForecastStore(self.data,kind="periodic");b_store=ForecastStore(changed,kind="periodic")
        a=ScenarioFactory(self.data,a_store).build(origin,a_store.get(origin),"4-3")
        b=ScenarioFactory(changed,b_store).build(origin,b_store.get(origin),"4-3")
        for key in ("paths","groups","probabilities"):np.testing.assert_array_equal(a[key],b[key])
        self.assertEqual(a["metadata"],b["metadata"])
        self.assertLessEqual(a["metadata"]["latest_source_target"],origin)
        self.assertTrue((a["groups"][:,:36]==0).all())
        self.assertAlmostEqual(a["probabilities"].sum(),1)
        # Same causal information gives the same small, fully solved purchase problem.
        small_a = {"paths": a["paths"][:2, :2], "probabilities": np.array([.5, .5]),
                   "groups": np.zeros((2, 2), int), "boundaries": []}
        small_b = {**small_a, "paths": b["paths"][:2, :2]}
        pa, _ = optimize(small_a, MIN_SOC, small_a["paths"].mean(0), adjustable=False)
        pb, _ = optimize(small_b, MIN_SOC, small_b["paths"].mean(0), adjustable=False)
        np.testing.assert_allclose(pa, pb, atol=1e-6)


class RiskPhysicsTests(unittest.TestCase):
    def test_two_stage_nonanticipative_recourse_matches_enumeration(self):
        paths = np.array([[[600., 0., 2.], [600., 0., 1.]],
                          [[600., 0., 2.], [1200., 0., 1.]]])
        bundle = {"paths": paths, "groups": np.array([[0, 1], [0, 2]]),
                  "probabilities": np.array([.5, .5]), "boundaries": [1]}
        # Buying extra at the first interval costs 2/kWh, exceeding later upward
        # adjustment's 1.5/kWh, so precharging cannot improve these enumerated costs.
        candidates = [200 + original + .75*(max(100-original, 0)+max(200-original, 0))
                      for original in range(301)]
        plan, multi = optimize(bundle, MIN_SOC, paths.mean(0), weight=0)
        fixed = {**bundle, "groups": np.zeros((2, 2), int), "boundaries": []}
        _, single = optimize(fixed, MIN_SOC, paths.mean(0), weight=0, adjustable=False)
        self.assertAlmostEqual(multi["expected_cost"], min(candidates), places=4)
        self.assertAlmostEqual(multi["expected_cost"], 375., places=4)
        self.assertAlmostEqual(single["expected_cost"], 400., places=4)
        np.testing.assert_allclose(plan, [100., 100.], atol=1e-5)

    def test_efficiency_limits_and_exact_greedy_actions(self):
        g=np.array([2000.,0.,0.,0.]);load=np.array([0.,60000.,60000.,60000.])
        c,d,e,w,s=execute(g,load,np.zeros(4),6000)
        np.testing.assert_allclose(np.diff(s),ETA*c-d/ETA)
        np.testing.assert_allclose(g+d+e-c-w,load/6)
        self.assertGreater(e.sum(),0);self.assertGreater(w.sum(),0)
        self.assertTrue((s>=MIN_SOC-1e-7).all() and (s<=MAX_SOC+1e-7).all())
        self.assertFalse(((c>0)&(d>0)).any())

    def test_final_adjustment_settlement(self):
        fees=settle(np.array([10.,10.]),np.array([5.,15.]),np.array([1.,0.]),np.ones(2))
        np.testing.assert_allclose(fees.sum(0),[20,7.5,2.5,5])

    def test_weighted_tail_average(self):
        self.assertAlmostEqual(weighted_cvar(np.array([0.,100.]),np.array([.95,.05])),50.)

    def test_small_enumerable_risk_problem(self):
        paths=np.array([[[600.,0.,1.]],[[1200.,0.,1.]]])
        bundle={"paths":paths,"groups":np.zeros((2,1),int),"probabilities":np.array([.5,.5]),"boundaries":[]}
        g,meta=optimize(bundle,MIN_SOC,np.array([[900.,0.,1.]]),weight=.3,adjustable=False)
        candidates=[]
        for grid in np.arange(0,301,1):
            costs=np.array([grid+5*max(100-grid,0),grid+5*max(200-grid,0)])
            candidates.append(costs.mean()+.3*weighted_cvar(costs,np.array([.5,.5])))
        self.assertFalse(meta["fallback"])
        self.assertAlmostEqual(g[0],200,places=5)
        self.assertAlmostEqual(meta["objective"],min(candidates),places=4)
        self.assertLess(meta["controller_error"],1e-5)

    def test_multiple_stage_controller_encoding(self):
        paths=np.array([[[0.,12000.,1.],[9000.,0.,2.],[1000.,0.,1.]],
                        [[500.,12000.,1.],[6000.,0.,2.],[9000.,0.,1.]]])
        bundle={"paths":paths,"groups":np.array([[0,1,1],[0,2,2]]),
                "probabilities":np.array([.5,.5]),"boundaries":[1]}
        _,meta=optimize(bundle,MAX_SOC-100,np.mean(paths,axis=0),weight=.1)
        self.assertFalse(meta["fallback"])
        self.assertLess(meta["controller_error"],1e-4)

    def test_timeout_uses_checked_deterministic_fallback(self):
        paths=np.array([[[600.,0.,1.]],[[1200.,0.,1.]]])
        bundle={"paths":paths,"groups":np.zeros((2,1),int),"probabilities":np.array([.5,.5]),"boundaries":[]}
        original=__import__("experiments.common.neural_v2.physics",fromlist=["LinearModel"]).LinearModel.solve
        def fail_large(model,*args,**kwargs):
            if len(model.lower)>10:
                return None,{"status":1,"message":"forced timeout","seconds":0,"feasible":False,
                             "dual_bound":None,"mip_gap":None,"constraint_residual":None}
            return original(model,*args,**kwargs)
        with patch("experiments.common.neural_v2.physics.LinearModel.solve",fail_large):
            g,meta=optimize(bundle,MIN_SOC,np.array([[900.,0.,1.]]),adjustable=False)
        self.assertTrue(meta["fallback"])
        self.assertTrue(np.isfinite(g).all() and (g>=0).all())


if __name__=="__main__":unittest.main()
