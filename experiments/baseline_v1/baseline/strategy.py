"""Search/localize/clear state machine. Only protocol observations inform actions."""
from dataclasses import dataclass, field
import json
import time
from .geometry import (clip_bearing, distance, enclosing_circle, optical_cover,
                       outer_circle, polygon_diameter)
from .planning import select_measurement, survey_points


class BudgetReached(RuntimeError):
    pass


@dataclass
class Track:
    channel: int
    polygon: list = field(default_factory=list)
    bearings: list = field(default_factory=list)
    tried: list = field(default_factory=list)
    cleared: bool = False


class Strategy:
    def __init__(self, client, cfg, problem: int, decisions_path):
        self.client, self.cfg, self.problem = client, cfg, problem
        self.position, self.channel = (0.0, 0.0), 1
        self.tracks = {ch: Track(ch) for ch in range(1, 21)}
        self.actions = 0
        self.survey_visited = []
        self.deadline = float("inf")
        self.decisions = decisions_path.open("w", encoding="utf-8")

    def record(self, event, **values):
        self.decisions.write(json.dumps({"event": event, "virtual_time_s": self.client.virtual_time_s,
                                         **values}, ensure_ascii=False)+"\n")
        self.decisions.flush()

    def guard(self, point):
        prospective = self.client.virtual_time_s + distance(self.position, point)/5 + 6
        if (time.monotonic() >= self.deadline or self.actions >= self.cfg.max_actions
                or prospective >= self.cfg.max_virtual_time_s):
            raise BudgetReached("Stopped with reserve for /exit; coverage may be incomplete")

    def measure(self, point, channel):
        self.guard(point)
        response = self.client.call("/measure", point, channel)
        self.position, self.channel = point, channel
        self.actions += 1
        track = self.tracks[channel]
        track.tried.append(point)
        result = response["measure_result"]
        if result == "direction":
            angle = float(response["svd_deg"])
            polygon = track.polygon or outer_circle(self.cfg.arena_radius_m, self.cfg.polygon_sides)
            polygon = clip_bearing(polygon, point, angle, self.cfg.bearing_error_deg, self.cfg.reception_max_m)
            if not polygon:
                raise ValueError(f"Empty uncertainty region for channel {channel}; inspect model/rounding")
            track.polygon = polygon
            track.bearings.append((point, angle))
            circle = enclosing_circle(polygon)
            self.record("bearing_update", channel=channel, position=point, svd_deg=angle,
                        vertices=polygon, diameter_m=polygon_diameter(polygon)[0],
                        enclosing_radius_m=circle.radius)
        elif result == "near":
            self.clear(point, channel, reason="near")
        elif result != "no_signal":
            raise ValueError(f"Unknown measure result: {result}")
        return response

    def clear(self, point, channel, reason):
        self.guard(point)
        response = self.client.call("/clear", point, channel)
        self.position = point  # /clear MUST NOT change self.channel.
        self.actions += 1
        success = response["clear_result"] == "success"
        if success:
            self.tracks[channel].cleared = True
            print(f"cleared channel={channel:02d} count={self.cleared_count} virtual={self.client.virtual_time_s:.2f}s", flush=True)
        self.record("clear_attempt", channel=channel, position=point, reason=reason, success=success)
        return success

    @property
    def cleared_count(self):
        return sum(t.cleared for t in self.tracks.values())

    def localize(self, track):
        """Refine interval geometry; cover its cells optically if signal is lost."""
        initial_point, initial_angle = track.bearings[0]
        for _ in range(self.cfg.max_local_measurements):
            if track.cleared:
                return
            circle = enclosing_circle(track.polygon)
            if circle.radius <= self.cfg.clear_radius_m-self.cfg.clear_cover_margin_m:
                if not self.clear(circle.center, track.channel, "guaranteed_enclosing_circle"):
                    raise ValueError("Certified optical clear failed; inspect model or protocol")
                return
            # A cheap attempt at an estimate is distinct from a geometric certificate.
            if len(track.bearings) >= 2 and circle.radius <= 65:
                if self.clear(circle.center, track.channel, "opportunistic_estimate"):
                    return
            point, info = select_measurement(track.polygon, initial_point, initial_angle,
                                             self.position, track.tried, self.cfg)
            self.record("next_measurement", channel=track.channel, position=point, **info)
            self.measure(point, track.channel)
        if track.cleared:
            return
        cover = optical_cover(track.polygon, self.cfg.optical_grid_m)
        self.record("optical_fallback", channel=track.channel, cells=len(cover))
        while cover:
            point = min(cover, key=lambda p: distance(self.position, p))
            cover.remove(point)
            if self.clear(point, track.channel, "guaranteed_grid_cover"):
                return
        raise ValueError("Exhaustive optical cover failed; feasible region inconsistent")

    def run(self, remaining_real_s):
        self.deadline = time.monotonic() + max(0, remaining_real_s-self.cfg.real_reserve_s)
        pending = survey_points(self.problem, self.cfg)
        self.record("coverage_plan", problem=self.problem, points=pending)
        while pending:
            point = min(pending, key=lambda p: distance(self.position, p))
            pending.remove(point)
            channels = [ch for ch, t in self.tracks.items() if not t.cleared]
            channels.sort(key=lambda ch: (ch != self.channel, ch))
            for channel in channels:
                self.measure(point, channel)
            self.survey_visited.append(point)
            known = [t for t in self.tracks.values() if t.bearings and not t.cleared]
            while known:
                track = min(known, key=lambda t: distance(self.position, enclosing_circle(t.polygon).center))
                self.localize(track)
                known.remove(track)
            if self.cleared_count == 16:
                return "all_16_cleared"
        return "coverage_complete"

    def close(self):
        self.decisions.close()
