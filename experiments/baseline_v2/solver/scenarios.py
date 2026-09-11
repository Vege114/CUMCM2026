"""Independent, explicit offline fixtures; never used by official strategy code."""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import random

from experiments.baseline_v1.baseline.mock import MockTransport, Source, generate_sources


@dataclass(frozen=True)
class Scenario:
    name: str
    problem: int
    seed: int
    layout: str = "legacy"
    count: int = 12
    directional_fraction: float = 0.6
    radius_mode: str = "mixed"
    error_mode: str = "hash"

    def validate(self):
        if self.problem not in (3, 4) or type(self.seed) is not int:
            raise ValueError("Invalid scenario problem/seed")
        if type(self.count) is not int or not 10 <= self.count <= 16:
            raise ValueError("Offline source count must be 10..16")
        if self.layout not in {"legacy", "random", "boundary", "outward", "tangent", "cluster", "coincident"}:
            raise ValueError("Unknown layout")
        if self.error_mode not in {"hash", "positive", "negative", "alternating"}:
            raise ValueError("Unknown fixed-location error rule")
        if self.radius_mode not in {"minimum", "maximum", "mixed"}:
            raise ValueError("Unknown reception radius rule")
        if not 0 <= self.directional_fraction <= 1:
            raise ValueError("Invalid directional fraction")
        return self


def create_sources(spec):
    spec.validate()
    if spec.layout == "legacy":
        return generate_sources(spec.seed, spec.problem)
    rng = random.Random(spec.seed)
    channels = rng.sample(range(1, 21), spec.count)
    sources = []
    for i, ch in enumerate(channels):
        a = rng.uniform(0, 2*math.pi)
        r = 1800*math.sqrt(rng.random())
        if spec.layout in {"boundary", "outward", "tangent"}:
            a, r = 2*math.pi*i/spec.count+0.07, 1800
        elif spec.layout == "cluster":
            r = 100*math.sqrt(rng.random())
        elif spec.layout == "coincident":
            r, a = 0, 0
        point = (r*math.cos(a), r*math.sin(a))
        if spec.layout == "cluster":
            point = (point[0]+1400, point[1]+100)
        reception = {"minimum": 1000, "maximum": 1500,
                     "mixed": rng.uniform(1000, 1500)}[spec.radius_mode]
        orientation = None
        if spec.problem == 4 and i < round(spec.count*spec.directional_fraction):
            orientation = rng.uniform(0, 360)
            if spec.layout == "outward":
                orientation = math.degrees(a) % 360
            elif spec.layout == "tangent":
                orientation = (math.degrees(a)+90) % 360
        sources.append(Source(ch, point, reception, orientation))
    return sources


def scenario_hash(spec):
    value = {"scenario": asdict(spec), "sources": [asdict(s) for s in create_sources(spec)]}
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class ScenarioTransport(MockTransport):
    def __init__(self, sources, spec):
        super().__init__(sources, spec.seed)
        self.error_mode = spec.error_mode

    def send(self, path, request):
        # At the source itself every closed radiation half-plane contains the
        # point. v1's atan2(0,0) angle convention can falsely suppress "near";
        # fix this degenerate fixture for BOTH algorithms in paired tests.
        source = self.sources.get(request.get("channel"))
        point = request.get("position", {})
        coincident = (path == "/measure" and source is not None
                      and point.get("x") == source.position[0] and point.get("y") == source.position[1])
        orientation = source.orientation_deg if source is not None else None
        if coincident:
            source.orientation_deg = None
        try:
            return super().send(path, request)
        finally:
            if coincident:
                source.orientation_deg = orientation

    def error_at(self, point, channel):
        if self.error_mode == "positive":
            return 1.0
        if self.error_mode == "negative":
            return -1.0
        value = super().error_at(point, channel)
        return (1.0 if value >= 0 else -1.0) if self.error_mode == "alternating" else value
