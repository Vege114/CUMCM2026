"""Finite observation-only state machine with separately auditable guarantees."""
from dataclasses import dataclass, field
import math
import time

from experiments.baseline_v1.baseline.strategy import BudgetReached, Strategy as V1Strategy, Track as V1Track
from .coverage import CoverageLedger
from .geometry import distance, enclosing_circle, optical_cover, oriented_optical_cover, polygon_diameter
from .planning import select_measurement
from .routing import insertion_cost, open_route
from .visibility import VisibilityModel


@dataclass
class Track(V1Track):
    misses: list = field(default_factory=list)
    failed_clears: list = field(default_factory=list)


class Strategy(V1Strategy):
    def __init__(self, client, cfg, problem, decisions_path):
        if problem not in (3, 4):
            raise ValueError("Problem must be 3 or 4")
        super().__init__(client, cfg, problem, decisions_path)
        self.tracks = {ch: Track(ch) for ch in range(1, 21)}
        self.coverage = CoverageLedger(problem, cfg)
        self.absent = set()

    def guard(self, point):
        if len(point) != 2 or not all(math.isfinite(v) and abs(v) <= 2_000_000 for v in point):
            raise ValueError("Illegal physical position")
        # Include a complete retry sequence before beginning another action.
        request_budget = self.cfg.http_timeout_s*(self.cfg.http_retries+1)
        request_budget += 0.25*(2**self.cfg.http_retries-1)
        if time.monotonic()+request_budget >= self.deadline:
            raise BudgetReached("Real-time reserve reached")
        super().guard(point)

    def measure(self, point, channel):
        point = tuple(point)
        if point in self.coverage.scanned[channel]:
            raise ValueError("Refusing a repeated fixed-location channel measurement")
        response = super().measure(point, channel)
        self.coverage.observe(channel, point)
        track = self.tracks[channel]
        if response["measure_result"] == "near" and not track.cleared:
            raise ValueError("A near result must be clearable at the same position")
        if response["measure_result"] == "no_signal":
            track.misses.append(point)
            self.record("signal_miss", channel=channel, position=point,
                        preserved_polygon=bool(track.polygon))
        if not track.bearings and not track.cleared and self.coverage.complete(channel):
            self.absent.add(channel)
            self.record("absence_certificate", **self.coverage.certificate(channel))
        return response

    def clear(self, point, channel, reason):
        success = super().clear(tuple(point), channel, reason)
        if not success:
            self.tracks[channel].failed_clears.append(tuple(point))
        return success

    def unknown(self):
        return [ch for ch, t in self.tracks.items()
                if not t.cleared and not t.bearings and ch not in self.absent]

    def opportunistic_scan(self):
        if self.cleared_count == 16 or not self.cfg.adaptive_coverage or self.problem != 3:
            return
        point = self.position
        channels = [ch for ch in self.unknown() if point not in self.coverage.scanned[ch]
                    and self.coverage.new_fraction(ch, point) >= self.cfg.opportunistic_scan_fraction]
        channels.sort(key=lambda ch: (ch != self.channel, ch))
        if channels:
            self.record("opportunistic_survey", position=point, channels=channels)
        for ch in channels:
            if self.cleared_count == 16:
                break
            self.measure(point, ch)

    def localize(self, track, onward=None):
        for _ in range(self.cfg.max_local_measurements):
            if track.cleared:
                return
            circle = enclosing_circle(track.polygon)
            if circle.radius <= self.cfg.clear_radius_m-self.cfg.clear_cover_margin_m:
                if not self.clear(circle.center, track.channel, "guaranteed_enclosing_circle"):
                    raise ValueError("Certified enclosing-circle clear failed")
                return
            if (len(track.bearings) >= 2 and circle.radius <= self.cfg.opportunistic_clear_radius_m
                    and all(distance(circle.center, p) >= self.cfg.distinct_measurement_m
                            for p in track.failed_clears)):
                if self.clear(circle.center, track.channel, "opportunistic_estimate"):
                    return
            point, info = select_measurement(track, self.position, self.cfg, self.problem, onward)
            if point is None:
                self.record("local_candidates_exhausted", channel=track.channel, **info)
                break
            self.record("next_measurement", channel=track.channel, position=point, **info)
            self.measure(point, track.channel)
        if track.cleared:
            return
        cover = (oriented_optical_cover(track.polygon, self.cfg.optical_grid_m)
                 if self.cfg.oriented_optical else optical_cover(track.polygon, self.cfg.optical_grid_m))
        # A previous failed clear at exactly this center already covers its cell.
        cover = [p for p in cover if all(distance(p, q) > 1e-8 for q in track.failed_clears)]
        self.record("optical_fallback", channel=track.channel, cells=len(cover),
                    covering_radius_m=self.cfg.optical_grid_m/math.sqrt(2))
        # 2-opt is useful on local routes; cap its quadratic work for large covers.
        route = open_route(self.position, cover, self.cfg.route_optimization and len(cover) <= 250)
        for point in route:
            if self.clear(point, track.channel, "guaranteed_grid_cover"):
                return
        raise ValueError("Exhaustive optical cover failed; feasible region inconsistent")

    def scan_waypoint(self, point):
        channels = [ch for ch in self.unknown() if point not in self.coverage.scanned[ch]]
        # Localize known targets with cheap additional observations en route.
        for ch, track in self.tracks.items():
            if not track.bearings or track.cleared:
                continue
            if any(distance(point, old) < self.cfg.distinct_measurement_m for old in track.tried):
                continue
            if (enclosing_circle(track.polygon).radius > self.cfg.clear_radius_m
                    and VisibilityModel(track, self.cfg, self.problem).score(point) >= 0.75):
                channels.append(ch)
        channels.sort(key=lambda ch: (ch != self.channel, ch))
        for ch in channels:
            if self.cleared_count == 16:
                break
            if not self.tracks[ch].cleared:
                self.measure(point, ch)
        self.survey_visited.append(point)

    def run(self, remaining_real_s):
        if not math.isfinite(remaining_real_s) or not 0 <= remaining_real_s <= 1200:
            raise ValueError("Invalid remaining real-time budget")
        self.deadline = time.monotonic()+max(0, remaining_real_s-self.cfg.real_reserve_s)
        pending = set(self.coverage.reference)
        self.record("coverage_plan", problem=self.problem, points=sorted(pending),
                    kind="triangle" if self.problem == 4 and self.cfg.triangular_coverage else "v1_cover")
        # Start at the origin. It is a reference node for both cover constructions.
        while True:
            if self.cleared_count == 16:
                self.record("stop_certificate", reason="all_16_cleared", cleared=16)
                return "all_16_cleared"
            known = [t for t in self.tracks.values() if t.bearings and not t.cleared]
            unknown = self.unknown()
            if not known and not unknown:
                self.record("stop_certificate", reason="coverage_complete", cleared=self.cleared_count,
                            absent=sorted(self.absent))
                return "coverage_complete"
            if not unknown:
                pending.clear()
            route = open_route(self.position, pending, self.cfg.route_optimization)
            waypoint = route[0] if route else None
            target = None
            if known:
                def cost(track):
                    c = enclosing_circle(track.polygon)
                    return insertion_cost(self.position, c.center, waypoint)+0.25*c.radius
                target = min(known, key=lambda t: (cost(t), t.channel))
            if target is not None and (waypoint is None or cost(target) <= self.cfg.localize_detour_m):
                self.record("target_schedule", channel=target.channel, insertion_score=cost(target), onward=waypoint)
                self.localize(target, waypoint)
                self.opportunistic_scan()
                continue
            if waypoint is not None:
                pending.remove(waypoint)
                self.scan_waypoint(waypoint)
                continue
            # A complete reference sweep must have certified every unknown channel.
            raise ValueError("Reference cover exhausted without a complete absence certificate")
