"""Deterministic open paths: nearest-neighbor initialization and 2-opt."""
from .geometry import distance


def path_length(start, points):
    return sum(distance(a, b) for a, b in zip([start]+points, points))


def open_route(start, points, optimize=True):
    pending = sorted(set(points))
    route, here = [], start
    while pending:
        p = min(pending, key=lambda q: (distance(here, q), q))
        route.append(p)
        pending.remove(p)
        here = p
    if not optimize:
        return route
    # Fixed start, free end: also consider reversals that change the endpoint.
    for _ in range(8):
        changed = False
        for i in range(len(route)-1):
            a = start if i == 0 else route[i-1]
            for j in range(i+1, len(route)):
                old = distance(a, route[i])
                new = distance(a, route[j])
                if j+1 < len(route):
                    old += distance(route[j], route[j+1])
                    new += distance(route[i], route[j+1])
                if new < old-1e-7:
                    route[i:j+1] = reversed(route[i:j+1])
                    changed = True
        if not changed:
            break
    return route


def insertion_cost(current, target, onward):
    return distance(current, target) if onward is None else (
        distance(current, target)+distance(target, onward)-distance(current, onward))
