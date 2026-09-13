"""Selected exp008 HGB pipeline, with its causal January start integrated.

Reads no exp002 state or exp004 prediction archive. January uses the selected
adapter's periodic fallback; February onward uses the unchanged accepted HGB
Ridge28/half-memory issued archive. PV and price release rules remain exp008.
"""
import hashlib
import json
from functools import lru_cache
from pathlib import Path
import numpy as np

from experiments.common.neural_v2.data import Data
from experiments.exp008.forecast import Forecasts, _integration_basis

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / 'data/results/exp008/forecast_absolute_hgb'
MODEL = 'direct_hgb_ridge28_memory_half'


class Store:
    def __init__(self):
        path = DIRECTORY / f'{MODEL}.npz'
        meta = json.loads((DIRECTORY / 'provenance.json').read_text())
        assert meta['complete'] and hashlib.sha256(path.read_bytes()).hexdigest() == meta['archives'][MODEL]
        with np.load(path) as z:
            self.values, self.origins = z['values'].copy(), z['origins'].copy()
        self.lookup = {int(o): i for i, o in enumerate(self.origins)}

    def get(self, origin):
        return self.values[self.lookup[int(origin)]].copy()


class SelectedForecasts(Forecasts):
    def __init__(self, data=None):
        self.data = Data() if data is None else data
        self.seed = 42
        self.store = Store()
        self.calibration = MODEL
        self._basis = _integration_basis()
        difference = np.diff(np.eye(24), n=2, axis=0)
        self._pv_penalty = 12 * np.einsum('ni,nj->ij', difference, difference, optimize=False) + 2 * np.eye(24)

    @lru_cache(maxsize=3000)
    def _price(self, origin, stop):
        if origin == 144:
            return self._observed(0, 144, origin)[:, 2].copy(), {
                'price_method': 'observed_previous_day_before_ridge_has_training_rows',
                'price_last_label': 143, 'known_future_price': False}
        return super()._price(origin, stop)

    @lru_cache(maxsize=3000)
    def get(self, day, slot=0, scenario='3'):
        result = super().get(day, slot, scenario)
        result['audit'].update(
            base_forecast=MODEL if day >= 31 else 'causal_periodic_cold_start',
            load_method='selected_HGB_with_causal_prefix_bias' if day >= 31 else 'periodic_with_causal_prefix_bias',
            selected_model_id=MODEL,
            january_rule='appendix1_day0_then_observed_weekly_load_daily_PV',
            no_exp002_dispatch_or_state=True)
        return result


def residual_paths(forecasts, day, slot, scenario, limit=28):
    """All completed prior target days; zero path when no day is available."""
    history = np.arange(max(0, day - limit), day, dtype=int)
    n = 144 - slot
    errors, prices = [], []
    for old in history:
        pred = forecasts.get(int(old), slot, scenario)
        truth = forecasts._observed(int(old)*144+slot, (int(old)+1)*144, day*144+slot)
        errors.append(((truth[:, 0]-pred['load_kw'])-(truth[:, 1]-pred['pv_kw']))/6)
        prices.append(truth[:, 2]-pred['price'])
    paths = np.asarray(errors) if len(history) else np.zeros((1, n))
    return paths, {
        'source': 'same_selected_algorithm_prior_completed_target_days',
        'model_id': MODEL, 'origins': (history*144+slot).tolist(),
        'label_stops_exclusive': ((history+1)*144).tolist(),
        'max_observed_index': int(day*144-1) if day else None,
        'zero_history_deterministic_path': not len(history),
        'january_cold_start_days': history[history < 31].tolist(),
        'price_errors_used_in_planning': False,
        'errors_kwh_sha256': hashlib.sha256(np.ascontiguousarray(paths).tobytes()).hexdigest()}
