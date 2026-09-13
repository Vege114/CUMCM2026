"""One fixed causal load-energy memory on joint CNN plus unchanged Ridge28.

The gain is 0.5, chosen before this candidate's score. Yesterday's residual
is measured against yesterday's *underlying joint/Ridge28* issued forecast,
not against this memory layer recursively. PV is unchanged. No fit or search.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.exp008.forecast import Forecasts
from experiments.exp008.neural_joint_calibration import JointStore
from experiments.problem2.exp003.data import Data, ROOT

OUT = ROOT / 'data/results/exp008/load_energy_memory'
GAIN = .5


def correct_day(actual, origins, base_values, index):
    origin = int(origins[index])
    output = base_values[index].copy()
    prior = np.flatnonzero(origins+144 == origin)
    audit = {'origin': origin, 'day': origin//144, 'gain': GAIN,
             'information_cutoff_exclusive': origin,
             'underlying_forecast': 'joint_CNN_then_fixed_Ridge28',
             'residual_reference': 'previous_day_underlying_forecast_before_memory_layer',
             'recursive_corrected_output_residual': False,
             'prior_forecast_origin': None, 'prior_label_end_exclusive': None,
             'load_correction_kw': 0., 'pv_unchanged': True,
             'current_truth_used': False}
    if len(prior):
        previous = int(prior[-1])
        old_origin = int(origins[previous])
        if previous >= index or old_origin+144 > origin:
            raise ValueError('memory labels must be completely observed at issue')
        yesterday = np.asarray(actual[old_origin:old_origin+144, 0], float)
        residual_kw = float(np.mean(yesterday-base_values[previous, :, 0]))
        output[:, 0] = np.maximum(0., output[:, 0]+GAIN*residual_kw)
        audit.update(prior_forecast_origin=old_origin,
                     prior_label_end_exclusive=old_origin+144,
                     prior_max_observed_index=old_origin+143,
                     underlying_prior_daily_energy_residual_kwh=24*residual_kw,
                     load_correction_kw=GAIN*residual_kw)
    else:
        audit['cold_start'] = 'unchanged_underlying_output_no_previous_issued_forecast'
    return output, audit


class EnergyMemoryStore:
    """Read-only construction from the signed joint/Ridge28 base archive."""

    def __init__(self, data=None, base_store=None):
        self.base_store = JointStore(calibrated=True) if base_store is None else base_store
        self.origins = self.base_store.origins.copy()
        self.base_values = self.base_store.values.copy()
        self.lookup = {int(origin): i for i, origin in enumerate(self.origins)}
        self.name = 'joint_ridge28_load_energy_memory_half'
        actual = (Data() if data is None else data).actual
        generated = [correct_day(actual, self.origins, self.base_values, i)
                     for i in range(len(self.origins))]
        self.values = np.stack([row[0] for row in generated])
        self.audit = [row[1] for row in generated]
        self.delta = self.values-self.base_values

    def get(self, origin):
        return self.values[self.lookup[int(origin)]].copy()


class EnergyMemoryForecasts(Forecasts):
    """Planning adapter; inherited historical risk calls these same outputs."""

    def __init__(self, data=None):
        super().__init__(data=data)
        self.store = EnergyMemoryStore(data=self.data)
        self.calibration = self.store.name

    def get(self, day, slot=0, scenario='2'):
        result = super().get(day, slot, scenario)
        result['audit'].update(
            base_forecast='exp008_joint_shared_cnn' if day >= 31 else 'periodic_cold_start',
            output_calibration='same_fixed_Ridge28_then_load_energy_memory_half' if day >= 31 else None,
            base_architecture_unchanged=day < 31)
        if day >= 31:
            result['audit']['load_energy_memory'] = self.store.audit[self.store.lookup[day*144]]
        return result


def save():
    OUT.mkdir(parents=True, exist_ok=True)
    data = Data()
    store = EnergyMemoryStore(data=data)
    path = OUT/'predictions.npz'
    np.savez_compressed(path, origins=store.origins, values=store.values,
                        base_values=store.base_values, delta=store.delta)
    protocol = {'candidate_count': 1, 'gain': GAIN,
                'base': 'signed_joint_CNN_then_unchanged_Ridge28',
                'current_and_historical_pipeline': 'apply_same_memory_to_each_historical_issue',
                'model_fitted': False, 'hyperparameter_search': False,
                'inputs': 'previous_completed_actual_load_and_underlying_prior_issued_load_forecast',
                'pv_unchanged': True, 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'base_values_sha256': hashlib.sha256(store.base_values.tobytes()).hexdigest(),
                'source_data_sha256': data.hashes,
                'archive_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'development_on_examined_2025_not_independent_test': True}
    (OUT/'protocol.json').write_text(json.dumps(protocol, indent=2))
    (OUT/'prediction_audit.json').write_text(json.dumps(store.audit, indent=2))
    (OUT/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    return store


if __name__ == '__main__':
    save()
