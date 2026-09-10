"""All adjustable strategy parameters, independent of protocol and geometry."""
from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class Config:
    arena_radius_m: float = 1800.0
    reception_min_m: float = 1000.0
    reception_max_m: float = 1500.0
    bearing_error_deg: float = 1.01  # 1 degree + 0.005 degree rounding + margin
    clear_radius_m: float = 20.0
    survey_ring_m: float = 1400.0
    survey_ring_count: int = 6
    directional_grid_m: float = 700.0
    polygon_sides: int = 64
    max_local_measurements: int = 8
    clear_cover_margin_m: float = 0.1
    optical_grid_m: float = 25.0
    movement_weight: float = 0.04
    max_actions: int = 12000
    max_virtual_time_s: float = 350000.0
    real_reserve_s: float = 15.0
    http_timeout_s: float = 5.0
    http_retries: int = 2

    def validate(self):
        import math
        for name, value in asdict(self).items():
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid nonnegative finite parameter: {name}")
        if not 0 < self.bearing_error_deg < 45:
            raise ValueError("bearing_error_deg must be in (0,45)")
        if self.polygon_sides < 8 or self.survey_ring_count < 3:
            raise ValueError("Too few polygon/ring vertices")
        if not 0 < self.directional_grid_m * math.sqrt(2) < self.reception_min_m:
            raise ValueError("Directional coverage requires sqrt(2)*grid < reception_min")
        if not 0 < self.optical_grid_m / math.sqrt(2) < self.clear_radius_m:
            raise ValueError("Optical grid does not guarantee 20 m coverage")
        if self.clear_cover_margin_m >= self.clear_radius_m:
            raise ValueError("Clear margin exceeds radius")
        # Ring + origin certificate, including the full radial interval.
        gap = math.pi / self.survey_ring_count
        for r in (self.reception_min_m, self.arena_radius_m):
            worst = math.sqrt(r*r + self.survey_ring_m**2 - 2*r*self.survey_ring_m*math.cos(gap))
            if worst > self.reception_min_m:
                raise ValueError("Q3 ring parameters leave an uncovered region")
        return self


def load_config(path: str | None = None) -> Config:
    values = json.loads(Path(path).read_text(encoding="utf-8")) if path else {}
    return Config(**values).validate()
