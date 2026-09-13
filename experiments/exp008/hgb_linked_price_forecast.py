"""Single predeclared Q4-2 price-model diagnostic using own issued HGB inputs."""
from __future__ import annotations

import copy
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.exp008.absolute_hgb_forecast_adapter import AbsoluteHGBForecasts
from experiments.exp008.forecast_absolute_hgb import OUT as HGB_OUT, PRIMARY

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data/results/exp008/hgb_linked_price_forecast'
FIXED_Q = ROOT / 'data/results/exp008/q4_absolute_hgb_physical/full334/dispatch_4-2.npz'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


class LinkedPriceForecasts(AbsoluteHGBForecasts):
    """Only price regression changes; common forecast adapter stays untouched."""

    def design(self, indices, cutoff):
        indices = np.asarray(indices, dtype=int)
        original = self._price_design(indices, cutoff)
        dayids, slots = indices//144, indices%144
        # Each historical row receives its own midnight-issued load/PV,
        # not the current model retroactively applied to previous days.
        issued = np.empty((len(indices),2))
        for day in np.unique(dayids):
            assert int(day)*144 <= cutoff
            selected = dayids == day
            issued[selected] = self._midnight(int(day))[slots[selected]] / 1000.
        assert np.all(indices-144 < dayids*144)
        return np.column_stack((original,issued))

    @lru_cache(maxsize=1500)
    def _price(self, origin, stop):
        if origin < 144:
            return super()._price(origin,stop)
        target = np.arange(origin,stop)
        train = np.arange(max(144,origin-28*144),origin)
        x, xt = self.design(train,origin), self.design(target,origin)
        y = self._observed(int(train[0]),origin,origin)[:,2]
        weights = 2**(-(origin-1-train)/(14*144))
        prior = np.array([0.,0.,.5,.5,0.,0.,0.,0.,0.,0.])
        penalty = np.array([1e-6,5.,5.,5.,5.,5.,5.,5.,5.,5.])
        coefficient = np.linalg.solve(
            np.einsum('ni,nj,n->ij',x,x,weights,optimize=False)+np.diag(penalty),
            np.einsum('ni,n,n->i',x,weights,y,optimize=False)+penalty*prior)
        prediction = np.maximum(1e-4,np.einsum('ni,i->n',xt,coefficient,optimize=False))
        return prediction, {'price_method':'Ridge28_daily_weekly_harmonics_plus_own_issued_HGB_load_PV',
            'price_training_start':int(train[0]), 'price_last_label':origin-1,
            'price_training_rows':len(train), 'price_coefficients':coefficient.tolist(),
            'price_feature_columns':10, 'price_new_feature_scale_kw':1000,
            'price_historical_feature_origins':(np.unique(train//144)*144).tolist(),
            'price_historical_feature_model':'own-midnight HGB issued pipeline; January labelled periodic cold start',
            'known_future_price':False}


def scores(prediction, actual, q):
    err = prediction-actual
    high = actual >= 1.
    return {'rmse_yuan_per_kwh':float(np.sqrt(np.mean(err**2))),
            'mae_yuan_per_kwh':float(np.mean(np.abs(err))),
            'bias_yuan_per_kwh':float(np.mean(err)),
            'high_realized_price_ge1_rmse':float(np.sqrt(np.mean(err[high]**2))),
            'high_realized_price_ge1_mae':float(np.mean(np.abs(err[high]))),
            'high_realized_price_slots':int(high.sum()),
            'realized_price_weighted_mae':float(np.sum(actual*np.abs(err))/np.sum(actual)),
            'fixed_purchase_absolute_price_exposure_proxy_yuan':float(np.sum(q*np.abs(err))),
            'fixed_purchase_signed_price_exposure_proxy_yuan':float(np.sum(q*err))}


def main():
    if OUT.exists():
        raise FileExistsError('No candidate overwrite; version any amended diagnostic')
    OUT.mkdir(parents=True)
    protected = [Path(__file__),ROOT/'experiments/exp008/forecast.py',
        ROOT/'experiments/exp008/absolute_hgb_forecast_adapter.py',
        HGB_OUT/f'{PRIMARY}.npz',HGB_OUT/'protocol.json',HGB_OUT/'provenance.json',FIXED_Q]
    hashes = {str(p.relative_to(ROOT)):digest(p) for p in protected}
    # This file is saved before forecasting/scoring or reading realized labels.
    write(OUT/'protocol.json',{'candidate_count':1,'diagnostic_only':True,
        'forecast_target':'Q4-2 midnight 144-slot price, full334 comparable days',
        'change':'append own-issued HGB load and PV /1000 to the original eight regressors',
        'historical_feature_policy':'own historical midnight issuance; January periodic cold start; no retroactive current-model features',
        'lookback_days':28,'label_weight_half_life_days':14,
        'original_penalty_preserved':True,'new_feature_penalties':[5.,5.],
        'new_feature_prior_coefficients':[0.,0.],
        'candidate_chosen_after_prior_development_not_untouched_2025_test':True,
        'high_price_definition':'realized price >= 1 yuan/kWh, evaluation only',
        'proxy_definition':'sum frozen Q4-2 original-purchase kWh times abs(predicted-realized price); no action reoptimization or actual savings claim',
        'additional_price_weighted_metric':'sum realized_price * abs(price_error) / sum realized_price',
        'no_dispatch_or_Q2_goal_test':True,'sources_sha256':hashes,
        'future_mutation_days':[31,32,60,151,243,364]})
    (OUT/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    baseline, candidate = AbsoluteHGBForecasts(), LinkedPriceForecasts()
    pred_old, pred_new, audit = [],[],[]
    for day in range(31,365):
        old = baseline.get(day,scenario='4-2')
        new = candidate.get(day,scenario='4-2')
        for key in ('load_kw','pv_kw'):
            np.testing.assert_array_equal(old[key],new[key])
        origin = day*144
        train = np.arange(max(144,origin-28*144),origin)
        x = candidate.design(train,origin)
        np.testing.assert_array_equal(x[:,:8],baseline._price_design(train,origin))
        assert new['audit']['price_last_label'] < origin
        assert max(new['audit']['price_historical_feature_origins']) < origin
        pred_old.append(old['price']); pred_new.append(new['price']); audit.append(new['audit'])
    pred_old,pred_new = np.asarray(pred_old),np.asarray(pred_new)
    checks = []
    for day in (31,32,60,151,243,364):
        origin = day*144
        for kind in ('price_only','load_PV_only','future_store_only','combined'):
            data = copy.copy(candidate.data); data.actual=candidate.data.actual.copy()
            if kind in ('price_only','combined'):
                data.actual[origin:,2] += 10000.
            if kind in ('load_PV_only','combined'):
                data.actual[origin:,:2] += [70000.,30000.]
            changed = LinkedPriceForecasts(data=data)
            future = changed.store.origins > origin
            if kind in ('future_store_only','combined'):
                changed.store.values[future] += [60000.,20000.]
            before = candidate.get(day,scenario='4-2')
            after = changed.get(day,scenario='4-2')
            for key in ('load_kw','pv_kw','price'):
                np.testing.assert_array_equal(before[key],after[key])
            np.testing.assert_array_equal(before['audit']['price_coefficients'],after['audit']['price_coefficients'])
            checks.append({'day':day,'kind':kind,'passed':True,
                'future_store_rows':int(future.sum()) if kind in ('future_store_only','combined') else 0,
                'store_test_nonvacuous':bool(future.any()) if kind in ('future_store_only','combined') else None})
    with np.load(FIXED_Q,allow_pickle=False) as pack:
        q=pack['original'].copy()
    actual = baseline.data.actual[31*144:,2].reshape(334,144)
    previous,updated=scores(pred_old,actual,q),scores(pred_new,actual,q)
    daily=[]
    for i,day in enumerate(range(31,365)):
        for name, pred in (('baseline',pred_old),('linked_hgb',pred_new)):
            daily.append({'day':day,'month':(pd.Timestamp('2025-01-01')+pd.Timedelta(days=day)).month,
                'method':name,**scores(pred[i],actual[i],q[i])})
    pd.DataFrame(daily).to_csv(OUT/'daily_scores.csv',index=False)
    np.savez_compressed(OUT/'price_predictions.npz',baseline=pred_old,linked_hgb=pred_new,
        actual=actual,days=np.arange(31,365),origins=np.arange(31,365)*144)
    write(OUT/'prediction_audit.json',audit)
    write(OUT/'causality_audit.json',{'passed':True,'checks':checks,
        'all334_original_eight_design_columns_unchanged':True,
        'all334_load_PV_forecasts_unchanged':True,
        'all334_training_labels_and_feature_issue_origins_strictly_before_target_origin':True,
        'future_store_day364_has_no_later_rows':True})
    assert all(digest(ROOT/p)==h for p,h in hashes.items())
    result={'days':334,'baseline':previous,'linked_hgb':updated,
        'delta_linked_minus_baseline':{k:updated[k]-previous[k] for k in previous},
        'all_sources_unchanged':True,'causality_checks_passed':True,
        'planning_and_billing_not_run':True,'this_is_not_a_Q2_goal_test':True,
        'all_four_forecast_gates_improved':all(updated[k]<previous[k] for k in (
            'rmse_yuan_per_kwh','mae_yuan_per_kwh','high_realized_price_ge1_rmse',
            'fixed_purchase_absolute_price_exposure_proxy_yuan'))}
    write(OUT/'summary.json',result)
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
