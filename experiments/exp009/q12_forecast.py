"""Causal year-start adapters for the accepted Q2 and Q4-2 predictors.

January's periodic fallback is a declared part of the same forecast pipeline.
From February the accepted issued predictions are reused byte-for-byte; no
dispatch warmup archive or current/future observations are used as features.
"""
from functools import lru_cache
import hashlib
from pathlib import Path

import numpy as np

from experiments.common.neural_v2.data import Data, ROOT
from experiments.exp008.forecast import Forecasts, _integration_basis
from experiments.exp008.hgb_linked_price_forecast import LinkedPriceForecasts


class FrozenStore:
    def __init__(self, path):
        self.path = Path(path)
        self.sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()
        with np.load(self.path, allow_pickle=False) as z:
            self.origins = z['origins'].copy()
            self.values = z['values'].copy()
        assert np.array_equal(self.origins, np.arange(31, 365)*144)
        assert self.values.shape == (334, 144, 2)
        self.lookup = {int(x): i for i, x in enumerate(self.origins)}

    def get(self, origin):
        return self.values[self.lookup[int(origin)]].copy()


class YearForecasts(Forecasts):
    def __init__(self, scenario, data=None):
        self.data = Data() if data is None else data
        self.seed = 42
        self.scenario = scenario
        path = ('forecast_hgb_extra_trees_half/hgb_extra_trees_half_ridge28_memory.npz'
                if scenario == '2' else
                'forecast_absolute_hgb/direct_hgb_ridge28_memory_half.npz')
        self.store = FrozenStore(ROOT/'data/results/exp008'/path)
        self._basis = _integration_basis()
        difference = np.diff(np.eye(24), n=2, axis=0)
        self._pv_penalty = 12*np.einsum('ni,nj->ij', difference, difference) + 2*np.eye(24)

    design = LinkedPriceForecasts.design

    @lru_cache(maxsize=400)
    def _price(self, origin, stop):
        if origin < 144:
            return Forecasts._price(self, origin, stop)
        if origin == 144:
            # No training labels have valid one-day-lag regressors yet. The
            # unchanged regression prior is .5 daily + .5 weekly (daily fallback).
            indices = np.arange(origin, stop)
            value = self._observed(0, origin, origin)[indices-144, 2].copy()
            return value, {'price_method': 'lag_regression_prior_without_valid_training_rows',
                'price_last_label': origin-1, 'price_training_rows': 0,
                'known_future_price': False, 'price_feature_columns': 10}
        return LinkedPriceForecasts._price(self, origin, stop)

    def get(self, day, slot=0, scenario=None):
        scenario = self.scenario if scenario is None else scenario
        out = Forecasts.get(self, day, slot, scenario)
        out['audit'].update(base_forecast=('raw_HGB_ExtraTrees_half_then_common_Ridge28_then_half_memory'
            if scenario == '2' else 'single_HGB_then_Ridge28_then_half_memory') if day >= 31
            else 'causal_periodic_cold_start', load_method='same_issued_midnight_pipeline',
            no_historical_dispatch_warmup=True, seed=42)
        return out

    def net_error_paths(self, day, scenario=None, limit=28):
        scenario = self.scenario if scenario is None else scenario
        history = np.arange(max(0, day-limit), day, dtype=int)
        if len(history):
            errors, prices = [], []
            for old in history:
                issue = self.get(int(old), scenario=scenario)
                observed = self._observed(int(old)*144, (int(old)+1)*144, int(day)*144)
                errors.append(observed[:, :2]-np.column_stack((issue['load_kw'], issue['pv_kw'])))
                prices.append(observed[:, 2]-issue['price'])
            errors = np.asarray(errors)
            net = (errors[:, :, 0]-errors[:, :, 1])/6
        else:
            errors, net, prices = np.zeros((1, 144, 2)), np.zeros((1, 144)), np.zeros((1, 144))
        return {'errors_kwh': net, 'errors_kw': errors, 'price_errors': np.asarray(prices),
            'origins': history*144,
            'audit': {'information_cutoff_exclusive': day*144,
                'max_observed_index': day*144-1 if day else None,
                'training_origins': (history*144).tolist(),
                'fallback_days': history[history<31].tolist(),
                'source': 'own_issued_pipeline_complete_daily_errors',
                'no_history_zero_path': day == 0,
                'history_includes_day_zero_once_complete': True}}
