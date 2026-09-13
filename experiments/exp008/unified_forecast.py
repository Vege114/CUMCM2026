"""Optional causal output calibration shared by all four annual scenarios."""
from .forecast import Forecasts
from .forecast_calibration import CalibratedStore


class UnifiedForecasts(Forecasts):
    def __init__(self, data=None, seed=42, calibration=None):
        super().__init__(data=data,seed=seed)
        self.calibration=calibration
        if calibration:
            self.store=CalibratedStore(calibration,seed=seed)

    def get(self, day, slot=0, scenario='2'):
        result=super().get(day,slot,scenario)
        result['audit']['output_calibration']=self.calibration
        result['audit']['base_architecture_unchanged']=True
        if self.calibration:
            result['audit']['calibration_labels_end_exclusive']=day*144
        return result
