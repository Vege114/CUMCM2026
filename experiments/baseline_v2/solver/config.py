"""Validated algorithm settings; official physical constants cannot be changed."""
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

from experiments.baseline_v1.baseline.config import Config as V1Config


@dataclass(frozen=True)
class Config(V1Config):
    survey_ring_m: float = 1150.0
    max_local_measurements: int = 12
    optical_grid_m: float = 27.5
    real_reserve_s: float = 25.0
    triangle_side_m: float = 980.0
    coverage_cell_m: float = 150.0
    distinct_measurement_m: float = 8.0
    information_weight: float = 0.65
    miss_penalty: float = 1.25
    onward_weight: float = 0.15
    localize_detour_m: float = 700.0
    opportunistic_clear_radius_m: float = 45.0
    opportunistic_scan_fraction: float = 0.12
    visibility_enabled: bool = True
    route_optimization: bool = True
    adaptive_coverage: bool = True
    triangular_coverage: bool = True
    oriented_optical: bool = True

    def validate(self):
        base_names = V1Config.__dataclass_fields__
        V1Config(**{k: getattr(self, k) for k in base_names}).validate()
        for name, value in asdict(self).items():
            default = self.__dataclass_fields__[name].default
            if isinstance(default, bool):
                if type(value) is not bool:
                    raise ValueError(f"{name} must be a boolean")
            else:
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError(f"{name} must be finite numeric data")
                if value < 0 or (isinstance(default, int) and type(value) is not int):
                    raise ValueError(f"Invalid nonnegative/integer parameter: {name}")
        for name, expected in (("arena_radius_m", 1800), ("reception_min_m", 1000),
                               ("reception_max_m", 1500), ("clear_radius_m", 20)):
            if getattr(self, name) != expected:
                raise ValueError(f"{name} is a problem constant ({expected})")
        if not 1.005 <= self.bearing_error_deg <= 2:
            raise ValueError("Bearing margin must include the 1 degree error and quantization")
        if not 0 < self.triangle_side_m < self.reception_min_m:
            raise ValueError("Triangle side must be smaller than the minimum reception radius")
        if not 50 <= self.coverage_cell_m <= 300:
            raise ValueError("coverage_cell_m must be between 50 and 300")
        if not 0 < self.distinct_measurement_m <= 50:
            raise ValueError("distinct_measurement_m must be in (0, 50]")
        if not 0 < self.opportunistic_scan_fraction <= 1:
            raise ValueError("opportunistic_scan_fraction must be in (0, 1]")
        if not 0 < self.max_virtual_time_s < 360000 or self.max_actions < 1:
            raise ValueError("Action/virtual budgets must leave a legal stopping point")
        if self.max_local_measurements < 1 or self.http_timeout_s <= 0 or self.http_retries > 5:
            raise ValueError("Invalid measurement/HTTP budget")
        worst_request = self.http_timeout_s * (self.http_retries + 1)
        worst_request += 0.25 * (2**self.http_retries - 1)
        if not worst_request + 2 <= self.real_reserve_s < 1200:
            raise ValueError("real_reserve_s must cover one complete HTTP retry sequence plus 2 seconds")
        return self


def load_config(path=None):
    data = json.loads(Path(path).read_text(encoding="utf-8-sig")) if path else {}
    if not isinstance(data, dict):
        raise ValueError("Configuration must be a JSON object")
    return Config(**data).validate()
