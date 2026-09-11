"""Q2/local measurement choices combine interval geometry, reception and motion."""
from experiments.baseline_v1.baseline.planning import measurement_candidates
from .geometry import bearing, clip_bearing, direction, distance, enclosing_circle, polygon_diameter, shifted
from .visibility import VisibilityModel


def candidates(track, cfg):
    origin, angle = track.bearings[0]
    circle = enclosing_circle(track.polygon)
    values = measurement_candidates(track.polygon, origin, angle, cfg)
    # Move toward the estimate while retaining the side from which signal was
    # received. Flank both ways; repeated fixed-location readings add no evidence.
    for received, _ in track.bearings[-3:]:
        theta = bearing(received, circle.center)
        d = distance(received, circle.center)
        for fraction in (0.4, 0.7, 0.9):
            for lateral in (0, -min(100, circle.radius/2), min(100, circle.radius/2)):
                values.append(shifted(received, direction(theta), fraction*d, lateral))
    # Every convex combination of received positions is guaranteed to receive.
    for a, _ in track.bearings[-4:]:
        for b, _ in track.bearings[-4:]:
            if a != b:
                values.append(((a[0]+b[0])/2, (a[1]+b[1])/2))
    if track.misses:
        missed = track.misses[-1]
        received = min((p for p, _ in track.bearings), key=lambda p: distance(p, missed))
        for t in (0.25, 0.5, 0.75):
            values.append((received[0]+t*(missed[0]-received[0]),
                           received[1]+t*(missed[1]-received[1])))
    return [p for p in dict.fromkeys(values)
            if all(distance(p, old) >= cfg.distinct_measurement_m for old in track.tried)]


def select_measurement(track, current, cfg, problem, onward=None):
    poly = track.polygon
    circle = enclosing_circle(poly)
    diameter = polygon_diameter(poly)[0]
    model = VisibilityModel(track, cfg, problem)
    samples = poly + [circle.center]
    ranked = []
    for p in candidates(track, cfg):
        worst = 0.0
        for q in samples:
            if distance(p, q) <= 5:
                continue
            for error in (-cfg.bearing_error_deg, 0, cfg.bearing_error_deg):
                post = clip_bearing(poly, p, bearing(p, q)+error,
                                    cfg.bearing_error_deg, cfg.reception_max_m)
                worst = max(worst, polygon_diameter(post)[0])
        within_range = max(distance(p, v) for v in poly) <= cfg.reception_min_m
        visibility = model.score(p) if cfg.visibility_enabled else 1.0
        if problem == 3 and within_range:
            visibility = 1.0
        expected = visibility*worst + (1-visibility)*diameter*cfg.miss_penalty
        travel = distance(current, p)
        score = cfg.information_weight*expected + cfg.movement_weight*travel
        score += cfg.onward_weight*distance(p, circle.center)
        if onward is not None:
            score += 0.02*distance(p, onward)
        info = {"objective": score, "predicted_worst_diameter_m": worst,
                "visibility_score": visibility, "ranking_hypotheses": len(model.states),
                "guaranteed_signal": model.guaranteed(p) or (problem == 3 and within_range),
                "guaranteed_range": within_range, "travel_m": travel}
        ranked.append((score, p, info))
    if not ranked:
        return None, {"reason": "distinct_candidates_exhausted"}
    _, p, info = min(ranked, key=lambda item: (item[0], item[1]))
    return p, {**info, "candidates_evaluated": len(ranked)}
