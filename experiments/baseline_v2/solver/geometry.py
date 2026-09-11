"""Small additions to v1's unchanged conservative half-plane geometry."""
import math

from experiments.baseline_v1.baseline.geometry import (
    bearing, clip_bearing, clip_half_plane, direction, distance, enclosing_circle,
    optical_cover, outer_circle, polygon_diameter, shifted,
)


def cross(a, b, c):
    return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])


def segment_distance(p, a, b):
    length2 = (a[0]-b[0])**2 + (a[1]-b[1])**2
    t = 0 if length2 == 0 else max(0, min(1, ((p[0]-a[0])*(b[0]-a[0]) +
                                              (p[1]-a[1])*(b[1]-a[1]))/length2))
    return distance(p, (a[0]+t*(b[0]-a[0]), a[1]+t*(b[1]-a[1])))


def convex_hull(points):
    points = sorted(set(points))
    if len(points) <= 1:
        return points
    def half(items):
        result = []
        for p in items:
            while len(result) >= 2 and cross(result[-2], result[-1], p) <= 0:
                result.pop()
            result.append(p)
        return result
    return half(points)[:-1] + half(reversed(points))[:-1]


def contains(poly, point, tolerance=1e-7):
    """Also handle point/segment degeneracy (needed for received-point hulls)."""
    if len(poly) < 3:
        return bool(poly) and segment_distance(point, poly[0], poly[-1]) <= tolerance
    return all(cross(a, b, point) >= -tolerance for a, b in zip(poly, poly[1:]+poly[:1]))


def disk_intersects_polygon(poly, radius):
    return (contains(poly, (0, 0)) or any(distance(p, (0, 0)) <= radius+1e-7 for p in poly)
            or any(segment_distance((0, 0), a, b) <= radius+1e-7
                   for a, b in zip(poly, poly[1:]+poly[:1])))


def hypotheses(poly):
    """Deterministic ranking samples, never a replacement for the feasible region."""
    center = (sum(p[0] for p in poly)/len(poly), sum(p[1] for p in poly)/len(poly))
    values = [center] + poly + [((p[0]+q[0])/2, (p[1]+q[1])/2)
                               for p, q in zip(poly, poly[1:]+poly[:1])]
    values += [((p[0]+center[0])/2, (p[1]+center[1])/2) for p in poly]
    return list(dict.fromkeys(values))


def oriented_optical_cover(poly, spacing):
    """Rotate into a long-axis frame; keep every intersecting square's center.

    Rotation preserves the spacing/sqrt(2) covering certificate. Choose the
    smallest of axis-aligned and diameter-aligned covers; neither clips truth.
    """
    _, pair = polygon_diameter(poly)
    angle = bearing(*pair) if pair else 0
    origin = poly[0]
    u = direction(angle)
    def local(p):
        x, y = p[0]-origin[0], p[1]-origin[1]
        return (x*u[0]+y*u[1], -x*u[1]+y*u[0])
    rotated = optical_cover([local(p) for p in poly], spacing)
    rotated = [shifted(origin, u, p[0], p[1]) for p in rotated]
    plain = optical_cover(poly, spacing)
    return rotated if len(rotated) < len(plain) else plain
