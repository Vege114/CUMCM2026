"""Issue-time adapter for the fixed direct HGB pipeline; no fitting or search."""
import json

from experiments.exp008.forecast import Forecasts
from experiments.exp008.forecast_absolute_hgb import AbsoluteHGBStore, OUT, PRIMARY
from experiments.exp008.forecast_override_adapter import AuditedForecastOverride


class AbsoluteHGBForecasts(Forecasts):
    def __init__(self, data=None):
        super().__init__(data=data, seed=42)
        self.store = AbsoluteHGBStore()
        self.calibration = PRIMARY
        self.training_audit = {row['month']: row for row in json.loads((OUT / 'training_audit.json').read_text())}
        self.day_audit = {row['day']: row for row in json.loads((OUT / 'prediction_audit.json').read_text())}
        self.memory_audit = json.loads((OUT / 'memory_half_audit.json').read_text())

    def get(self, day, slot=0, scenario='2'):
        result = super().get(day, slot, scenario)
        result['audit'].update(base_forecast='direct_absolute_load_pv_HGB' if day >= 31 else 'periodic_cold_start',
            output_calibration='same_fixed_Ridge28_then_load_energy_memory_half' if day >= 31 else None,
            selected_model_id=PRIMARY, base_architecture_unchanged=day < 31)
        if day >= 31:
            result['audit']['direct_hgb_prediction'] = self.day_audit[day]
            result['audit']['load_energy_memory'] = self.memory_audit[day - 31]
        return result

    def net_error_paths(self, day, scenario='2', limit=28):
        result = super().net_error_paths(day, scenario, limit)
        result['audit'].update(source='same_direct_HGB_Ridge28_memory_with_labelled_January_periodic_cold_start',
                               model_id=PRIMARY)
        return result


def audited_forecasts():
    return AuditedForecastOverride(AbsoluteHGBForecasts(), PRIMARY,
        [OUT / 'provenance.json', OUT / 'protocol.json', OUT / f'{PRIMARY}.npz',
         OUT / 'training_audit.json', OUT / 'ridge28_audit.json', OUT / 'memory_half_audit.json'])
