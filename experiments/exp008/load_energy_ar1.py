"""One predeclared rolling AR(1) load-memory candidate, shrunk toward 0.5.

No annual coefficient fit or hyperparameter grid. The residual reference is
the underlying joint CNN + Ridge28 output, before any memory correction.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.exp008.forecast import Forecasts
from experiments.exp008.neural_joint_calibration import JointStore
from experiments.problem2.exp003.data import Data, ROOT

OUT = ROOT/'data/results/exp008/load_energy_ar1'
CONFIG = {'window_complete_issued_days': 28, 'maximum_adjacent_pairs': 27,
          'prior_gain': .5, 'prior_equivalent_pairs': 7.,
          'coefficient_min': 0., 'coefficient_max': 1.,
          'historical_x_mean_square_floor_kw2': 1., 'intercept': False}


def protocol():
    return {'candidate_count': 1, 'config': CONFIG,
            'residual': 'actual load daily mean minus underlying Joint+Ridge28 issued load daily mean, kW',
            'pairs': 'adjacent complete issued days, both inside last28 complete issued days',
            'scale': 's2=max(mean(x*x),1 kW squared), only observed historical x',
            'objective': 'sum((y-phi*x)^2/s2)+7*(phi-0.5)^2; phi in [0,1]',
            'solution': 'clip((sum(x*y)+7*s2*0.5)/(sum(x*x)+7*s2),0,1)',
            'current_correction': 'phi times yesterday underlying load mean residual',
            'cold_start': 'no pair: phi0.5; no prior issued day: unchanged underlying forecast',
            'pv_unchanged': True, 'recursive_corrected_residual': False,
            'grid_search': False, 'annual_coefficient_fit': False,
            'coefficient_updates': 'daily using only prior completed labels',
            'gate': 'strict improvement vs fixed0.5 in net daily-energy RMSE, intraday-cumulative RMSE and high-price RMSE',
            'gate_effect': 'only then authorize one fixed 334day tree28/q.8/buffer500 LP bridge',
            'development_on_examined_2025_not_independent_test': True,
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def record_protocol():
    OUT.mkdir(exist_ok=True, parents=True)
    path = OUT/'protocol.json'
    config = protocol()
    if path.exists() and json.loads(path.read_text()) != config:
        raise RuntimeError('Refusing to change a previously recorded AR1 protocol')
    path.write_text(json.dumps(config, indent=2))
    (OUT/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())


def ridge_gain(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.shape != y.shape or x.ndim != 1 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('AR pairs must be finite vectors with equal shape')
    if not len(x):
        return .5, {'sample_count': 0, 'scale_squared_kw2': 1., 'unclipped_gain': .5}
    scale2 = max(float(np.mean(x*x)), CONFIG['historical_x_mean_square_floor_kw2'])
    ridge = CONFIG['prior_equivalent_pairs']*scale2
    gain = float((np.sum(x*y)+ridge*.5)/(np.sum(x*x)+ridge))
    return float(np.clip(gain, 0., 1.)), {'sample_count': len(x),
        'scale_squared_kw2': scale2, 'ridge_kw2': ridge, 'unclipped_gain': gain,
        'sum_x_squared_kw2': float(np.sum(x*x)), 'sum_xy_kw2': float(np.sum(x*y))}


def correct_day(actual, origins, base_values, index):
    origin = int(origins[index])
    prior = np.flatnonzero(origins+144 <= origin)[-CONFIG['window_complete_issued_days']:]
    if np.any(prior >= index):
        raise ValueError('issued origins must be chronological')
    residual = np.array([np.mean(np.asarray(actual[int(origins[h]):int(origins[h])+144, 0])
                                   -base_values[h, :, 0]) for h in prior])
    adjacent = np.diff(origins[prior]) == 144
    gain, fit = ridge_gain(residual[:-1][adjacent], residual[1:][adjacent])
    output = base_values[index].copy()
    yesterday = len(prior) > 0 and origins[prior[-1]]+144 == origin
    correction = gain*float(residual[-1]) if yesterday else 0.
    output[:, 0] = np.maximum(0., output[:, 0]+correction)
    pairs = np.column_stack((origins[prior[:-1]][adjacent], origins[prior[1:]][adjacent]))
    audit = {'day': origin//144, 'origin': origin, 'information_cutoff_exclusive': origin,
             'history_origins': origins[prior].astype(int).tolist(),
             'pair_origins': pairs.astype(int).tolist(),
             'history_residuals_kw': residual.tolist(),
             'maximum_label_index': int(origins[prior[-1]]+143) if len(prior) else None,
             'gain': gain, 'fit': fit, 'load_correction_kw': correction,
             'underlying_forecast': 'joint_CNN_then_unchanged_Ridge28',
             'current_truth_used': False, 'recursive_corrected_output_residual': False,
             'pv_unchanged': True, 'has_previous_complete_issued_day': bool(yesterday)}
    return output, audit


class AR1MemoryStore:
    def __init__(self, data=None, base_store=None):
        self.base_store = JointStore(calibrated=True) if base_store is None else base_store
        self.origins = self.base_store.origins.copy()
        self.base_values = self.base_store.values.copy()
        self.lookup = {int(origin): i for i, origin in enumerate(self.origins)}
        self.name = 'joint_ridge28_load_energy_ar1_w28_prior7'
        actual = (Data() if data is None else data).actual
        rows = [correct_day(actual, self.origins, self.base_values, i)
                for i in range(len(self.origins))]
        self.values = np.stack([row[0] for row in rows])
        self.audit = [row[1] for row in rows]
        self.delta = self.values-self.base_values

    def get(self, origin):
        return self.values[self.lookup[int(origin)]].copy()


class AR1MemoryForecasts(Forecasts):
    def __init__(self, data=None):
        super().__init__(data=data)
        self.store = AR1MemoryStore(data=self.data)
        self.calibration = self.store.name

    def get(self, day, slot=0, scenario='2'):
        result = super().get(day, slot, scenario)
        result['audit'].update(
            base_forecast='exp008_joint_shared_cnn' if day >= 31 else 'periodic_cold_start',
            output_calibration='same_fixed_Ridge28_then_load_energy_ar1_w28_prior7' if day >= 31 else None,
            base_architecture_unchanged=day < 31)
        if day >= 31:
            result['audit']['load_energy_ar1'] = self.store.audit[self.store.lookup[day*144]]
        return result


if __name__ == '__main__':
    record_protocol()
