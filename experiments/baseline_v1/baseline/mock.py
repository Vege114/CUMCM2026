"""Independent offline test double; NOT the official simulator's random generator."""
from dataclasses import dataclass
import hashlib
import json
import math
import random
import time
from .geometry import bearing, distance


@dataclass
class Source:
    channel: int
    position: tuple
    reception_m: float
    orientation_deg: float | None = None
    cleared: bool = False


def generate_sources(seed: int, problem: int) -> list[Source]:
    rng = random.Random(seed)
    sources = []
    for channel in rng.sample(range(1, 21), rng.randint(10, 16)):
        r, angle = 1800*math.sqrt(rng.random()), rng.uniform(0, 2*math.pi)
        sources.append(Source(channel, (r*math.cos(angle), r*math.sin(angle)),
                              rng.uniform(1000, 1500),
                              rng.uniform(0, 360) if problem == 4 and rng.random() < 0.6 else None))
    return sources


class MockTransport:
    """Protocol-compatible deterministic environment. Strategy receives only send()."""
    def __init__(self, sources: list[Source], seed=0):
        self.sources = {s.channel: s for s in sources}
        self.seed, self.position, self.channel = seed, (0.0, 0.0), 1
        self.virtual = 0.0
        self.active = False
        self.cache = {}

    def error_at(self, point, channel):
        raw = f"{self.seed}:{channel}:{point[0]:.8f}:{point[1]:.8f}".encode()
        number = int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")
        return 2*(number/(2**64-1))-1

    def send(self, path, request):
        key = request["request_id"]
        canonical = path+json.dumps(request, sort_keys=True)
        if key in self.cache:
            old_request, response = self.cache[key]
            return (200, response.copy()) if canonical == old_request else (409, {"accepted": False, "virtual_time_s": 0})
        if path == "/enter":
            if self.active:
                return 200, {"accepted": False, "virtual_time_s": 0}
            self.active = True
            extra = {"remaining_real_duration_s": 1200, "max_real_duration_s": 1200,
                     "max_virtual_duration_s": 360000}
        elif not self.active:
            return 200, {"accepted": False, "virtual_time_s": 0}
        elif path == "/exit":
            self.active = False
            extra = {"exit_reason": "user_exit"}
        else:
            p = (request["position"]["x"], request["position"]["y"])
            ch = request["channel"]
            self.virtual += distance(self.position, p)/5
            self.position = p
            source = self.sources.get(ch)
            if path == "/clear":
                success = source is not None and not source.cleared and distance(p, source.position) <= 20
                if success:
                    source.cleared = True
                self.virtual += 5 if success else 3
                extra = {"clear_result": "success" if success else "no_target_in_range"}
            elif path == "/measure":
                self.virtual += 5 + int(ch != self.channel)
                self.channel = ch
                visible = source is not None and not source.cleared and distance(p, source.position) <= source.reception_m
                if visible and source.orientation_deg is not None:
                    delta = (bearing(source.position, p)-source.orientation_deg+180) % 360 - 180
                    visible = abs(delta) <= 90+1e-9
                extra = {"measure_result": "no_signal"}
                if visible:
                    if distance(p, source.position) <= 5:
                        extra = {"measure_result": "near"}
                    else:
                        angle = round((bearing(p, source.position)+self.error_at(p, ch)) % 360, 2) % 360
                        extra = {"measure_result": "direction", "svd_deg": angle}
            else:
                raise ValueError(path)
        response = {"accepted": True, "real_timestamp_ms": round(time.time()*1000),
                    "virtual_time_s": round(self.virtual, 6), **extra}
        self.cache[key] = (canonical, response.copy())
        return 200, response
