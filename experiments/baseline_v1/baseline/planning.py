"""Coverage points, second-observation candidates and information/motion tradeoff."""
import math
from .config import Config
from .geometry import (Point, Polygon, bearing, clip_bearing, direction, distance,
                       enclosing_circle, polygon_diameter, shifted)


def survey_points(problem: int, cfg: Config) -> list[Point]:
    if problem == 3:
        return [(0.0, 0.0)] + [shifted((0, 0), direction(360*i/cfg.survey_ring_count),
                                     cfg.survey_ring_m) for i in range(cfg.survey_ring_count)]
    # All vertices of every grid square intersecting the target disk, plus a halo.
    # For any source in a cell, its four vertices surround it and are within sqrt(2)*h.
    # Every closed half-plane through that source includes a receiving vertex.
    h = cfg.directional_grid_m
    n = math.ceil(cfg.arena_radius_m / h)
    points = set()
    for i in range(-n, n):
        for j in range(-n, n):
            x = max(i*h, min(0, (i+1)*h))
            y = max(j*h, min(0, (j+1)*h))
            if math.hypot(x, y) <= cfg.arena_radius_m:
                points.update(((i*h, j*h), ((i+1)*h, j*h), (i*h, (j+1)*h), ((i+1)*h, (j+1)*h)))
    return sorted(points, key=lambda p: (distance((0, 0), p), p))


def measurement_candidates(poly: Polygon, origin: Point, angle: float, cfg: Config) -> list[Point]:
    """Q2 candidate region discretized in the first bearing's local coordinates."""
    u = direction(angle)
    diameter, _ = polygon_diameter(poly)
    circle = enclosing_circle(poly)
    candidates = []
    if diameter > 300:
        for advance in (0.25, 0.5, 0.75):
            for lateral in (-450, -250, 250, 450):
                candidates.append(shifted(origin, u, advance*cfg.reception_max_m, lateral))
    baseline = max(40, min(250, diameter))
    for a in range(0, 360, 45):
        candidates.append(shifted(circle.center, direction(a), baseline))
    return candidates


def select_measurement(poly: Polygon, origin: Point, angle: float, current: Point,
                       tried: list[Point], cfg: Config) -> tuple[Point, dict]:
    """Minimax hypothetical diameter plus travel penalty; a heuristic, not optimum.

    Uses polygon vertices and center as possible target positions, not simulator
    truth. +/- full error at the future point covers adversarial measurement error.
    """
    center = enclosing_circle(poly).center
    hypotheses = poly + [center]
    candidates = [p for p in measurement_candidates(poly, origin, angle, cfg)
                  if all(distance(p, q) >= 10 for q in tried)]
    if not candidates:
        raise ValueError("No distinct measurement candidate remains")
    # Prefer points guaranteed within the minimum reception radius for Q3.
    reachable = [p for p in candidates if max(distance(p, q) for q in poly) <= cfg.reception_min_m]
    candidates = reachable or candidates
    ranked = []
    for p in candidates:
        worst = 0.0
        for q in hypotheses:
            if distance(p, q) <= 5:
                continue
            predicted = bearing(p, q)
            for bias in (-cfg.bearing_error_deg, 0, cfg.bearing_error_deg):
                post = clip_bearing(poly, p, predicted+bias, cfg.bearing_error_deg, cfg.reception_max_m)
                worst = max(worst, polygon_diameter(post)[0])
        travel = distance(current, p)
        ranked.append((worst + cfg.movement_weight*travel, p, worst, travel))
    score, p, worst, travel = min(ranked)
    return p, {"predicted_worst_diameter_m": worst, "travel_m": travel,
               "objective": score, "candidates_evaluated": len(candidates),
               "guaranteed_range": bool(reachable)}
