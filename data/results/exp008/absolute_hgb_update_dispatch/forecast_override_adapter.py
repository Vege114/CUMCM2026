"""Identity wrapper for store-backed forecast overrides; no model retraining.

The same wrapped get method feeds current issues and issued_error_paths.
January remains its explicitly labelled periodic cold start. The wrapper
does not load an old residual cache or change any prediction numerically.
"""
import hashlib
import inspect
from pathlib import Path

import numpy as np


def array_sha256(values):
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


class AuditedForecastOverride:
    def __init__(self, forecasts, model_id, source_paths=()):
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError('An explicit model_id is required')
        self.forecasts = forecasts
        self.model_id = model_id
        self._values_hash = array_sha256(forecasts.store.values)
        self._origins_hash = array_sha256(forecasts.store.origins)
        self.source_paths = tuple(Path(p).resolve() for p in source_paths)
        self._source_hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in self.source_paths}

    def __getattr__(self, name):
        return getattr(self.forecasts, name)

    def forecast_identity(self):
        # Fingerprint the loaded prediction values, not only the .npz name.
        # Source artifacts must also retain the version loaded by this object.
        for path in self.source_paths:
            if hashlib.sha256(path.read_bytes()).hexdigest() != self._source_hashes[str(path)]:
                raise ValueError(f'Forecast provenance artifact changed: {path}')
        method_path = Path(inspect.getfile(type(self.forecasts))).resolve()
        return {'model_id': self.model_id,
            'prediction_values_sha256': self._values_hash,
            'prediction_origins_sha256': self._origins_hash,
            'prediction_values_shape': list(self.store.values.shape),
            'prediction_values_dtype': str(self.store.values.dtype),
            'prediction_origins_dtype': str(self.store.origins.dtype),
            'implementation_class': f'{type(self.forecasts).__module__}.{type(self.forecasts).__name__}',
            'implementation_source': str(method_path),
            'implementation_source_sha256': hashlib.sha256(method_path.read_bytes()).hexdigest(),
            'wrapper_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'source_artifact_sha256': self._source_hashes.copy(),
            'seed': int(self.seed), 'calibration': getattr(self.forecasts, 'calibration', None),
            'history_source': 'same wrapped get at each prior completed target day and same legal issue',
            'January_history': 'unchanged explicitly labelled periodic cold start, not joint CNN output'}

    def get(self, day, slot=0, scenario='2'):
        result = self.forecasts.get(day, slot=slot, scenario=scenario)
        result['audit'] = {**result['audit'], 'selected_model_id': self.model_id,
            'selected_prediction_values_sha256': self._values_hash}
        return result

    def net_error_paths(self, day, scenario='2', limit=28):
        result = self.forecasts.net_error_paths(day, scenario=scenario, limit=limit)
        result['audit'] = {**result['audit'], 'source': 'same_selected_override_adapter_with_labelled_January_periodic_cold_start',
            'model_id': self.model_id, 'prediction_values_sha256': self._values_hash,
            'historical_predictions_use_same_override': True}
        return result
