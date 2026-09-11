"""Circular orientation constraints for ranking only; no truth or region deletion."""
from dataclasses import dataclass

from .geometry import bearing, contains, convex_hull, distance, hypotheses


def arc(center, half_width=90):
    lo, hi = (center-half_width) % 360, (center+half_width) % 360
    return [(lo, hi)] if lo <= hi else [(0.0, hi), (lo, 360.0)]


def intersect(left, right):
    return [(max(a, c), min(b, d)) for a, b in left for c, d in right
            if max(a, c) <= min(b, d)]


def length(intervals):
    return sum(b-a for a, b in intervals)


@dataclass
class State:
    source: tuple
    orientations: list
    radius_low: float
    omni_radius_high: float
    omni_possible: bool


class VisibilityModel:
    def __init__(self, track, cfg, problem):
        self.cfg, self.problem = cfg, problem
        self.received_hull = convex_hull([p for p, _ in track.bearings])
        self.states = []
        for q in hypotheses(track.polygon):
            if distance(q, (0, 0)) > cfg.arena_radius_m+1e-6:
                continue
            low = max([cfg.reception_min_m]+[distance(q, p) for p, _ in track.bearings])
            if low > cfg.reception_max_m+1e-6:
                continue
            if any(distance(q, p) <= cfg.clear_radius_m for p in track.failed_clears):
                continue
            angles = [(0.0, 360.0)]
            for p, _ in track.bearings:
                if distance(p, q) > 1e-7:
                    angles = intersect(angles, arc(bearing(q, p)))
            omni_high = min([cfg.reception_max_m]+[distance(q, p) for p in track.misses])
            for p in track.misses:
                # R >= low is forced by received points and the problem's lower
                # bound. Only then may a miss constrain the directional angle.
                if distance(q, p) <= low and distance(q, p) > 1e-7:
                    angles = intersect(angles, arc(bearing(q, p)+180))
            omni = omni_high > low or (not track.misses and omni_high >= low)
            if omni or angles:
                self.states.append(State(q, angles, low, omni_high, omni))

    def guaranteed(self, point):
        # The receiving half-disc is convex, even with unknown R/orientation.
        return contains(self.received_hull, point, tolerance=1e-9)

    def score(self, point):
        if self.guaranteed(point):
            return 1.0
        if not self.states:
            return 0.5  # Sampling can miss feasible truth; never claim inconsistency.
        values = []
        def radial(d, lo, hi):
            if d <= lo:
                return 1.0
            return max(0.0, (hi-d)/max(1e-9, hi-lo))
        for state in self.states:
            d = distance(point, state.source)
            p_range = radial(d, state.radius_low, self.cfg.reception_max_m)
            if self.problem == 3:
                values.append(radial(d, state.radius_low, state.omni_radius_high))
                continue
            total = length(state.orientations)
            if d <= 1e-7:
                angular = 1.0
            elif total > 1e-9:
                angular = length(intersect(state.orientations, arc(bearing(state.source, point))))/total
            else:
                angular = 0.5  # Zero-width feasible orientation is not an impossibility proof.
            if state.omni_possible and state.orientations:
                # Equal model weights are a ranking convention, not a learned prior.
                value = (radial(d, state.radius_low, state.omni_radius_high)+angular*p_range)/2
            elif state.omni_possible:
                value = radial(d, state.radius_low, state.omni_radius_high)
            else:
                value = angular*p_range
            values.append(value)
        return sum(values)/len(values)
