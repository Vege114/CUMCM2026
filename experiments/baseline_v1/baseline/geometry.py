"""Convex half-plane geometry. All distances in metres; bearings in degrees."""
import itertools
import math
from dataclasses import dataclass

Point = tuple[float, float]
Polygon = list[Point]
EPS = 1e-8


def distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def bearing(a: Point, b: Point) -> float:
    return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])) % 360


def direction(angle_deg: float) -> Point:
    a = math.radians(angle_deg)
    return math.cos(a), math.sin(a)


def shifted(p: Point, along: Point, forward: float, lateral: float = 0) -> Point:
    return (p[0] + forward*along[0] - lateral*along[1],
            p[1] + forward*along[1] + lateral*along[0])


def outer_circle(radius: float, sides: int = 64) -> Polygon:
    """Circumscribed polygon: never exclude a possible target on the disk edge."""
    rv = radius / math.cos(math.pi / sides)
    return [(rv*math.cos(2*math.pi*i/sides), rv*math.sin(2*math.pi*i/sides))
            for i in range(sides)]


def clip_half_plane(poly: Polygon, a: float, b: float, c: float) -> Polygon:
    """Sutherland-Hodgman clipping with a*x + b*y <= c."""
    if not poly:
        return []
    result = []
    for start, end in zip(poly, poly[1:] + poly[:1]):
        fs, fe = a*start[0]+b*start[1]-c, a*end[0]+b*end[1]-c
        ins, ine = fs <= EPS, fe <= EPS
        if ins != ine:
            t = fs / (fs-fe)
            result.append((start[0]+t*(end[0]-start[0]), start[1]+t*(end[1]-start[1])))
        if ine:
            result.append(end)
    return [p for i, p in enumerate(result) if i == 0 or distance(p, result[i-1]) > EPS]


def clip_bearing(poly: Polygon, position: Point, angle: float, error: float,
                 reception_max: float = 1500) -> Polygon:
    lo, hi = direction(angle-error), direction(angle+error)
    for a, b in ((lo[1], -lo[0]), (-hi[1], hi[0])):
        poly = clip_half_plane(poly, a, b, a*position[0] + b*position[1])
    # A conservative range upper bound. Forward projection <= R contains the disk.
    ux, uy = direction(angle)
    return clip_half_plane(poly, ux, uy, ux*position[0] + uy*position[1] + reception_max)


def polygon_diameter(poly: Polygon) -> tuple[float, tuple[Point, Point] | None]:
    """Exact vertex-pair maximum for a bounded convex polygon, O(v^2)."""
    if not poly:
        return 0.0, None
    pair = max(itertools.combinations(poly, 2), key=lambda p: distance(*p), default=(poly[0], poly[0]))
    return distance(*pair), pair


@dataclass(frozen=True)
class Circle:
    center: Point
    radius: float

    def contains(self, point: Point) -> bool:
        return distance(self.center, point) <= self.radius + 1e-6


def circle_three(a: Point, b: Point, c: Point) -> Circle | None:
    # Translate for better conditioning near large coordinates.
    bx, by, cx, cy = b[0]-a[0], b[1]-a[1], c[0]-a[0], c[1]-a[1]
    det = 2*(bx*cy-by*cx)
    if abs(det) < 1e-10:
        return None
    bb, cc = bx*bx+by*by, cx*cx+cy*cy
    center = (a[0]+(cy*bb-by*cc)/det, a[1]+(bx*cc-cx*bb)/det)
    return Circle(center, distance(a, center))


def enclosing_circle(poly: Polygon) -> Circle:
    """Exact minimum enclosing circle by 1/2/3 supporting vertices.

    Local uncertainty polygons are small; exhaustive enumeration is intentionally
    simple and independently testable. Empty regions are an error, not confidence.
    """
    if not poly:
        raise ValueError("Empty feasible region")
    if len(poly) == 1:
        return Circle(poly[0], 0)
    best = Circle(poly[0], max(distance(poly[0], p) for p in poly))
    candidates = [Circle(((a[0]+b[0])/2, (a[1]+b[1])/2), distance(a, b)/2)
                  for a, b in itertools.combinations(poly, 2)]
    for a, b, c in itertools.combinations(poly, 3):
        circle = circle_three(a, b, c)
        if circle:
            candidates.append(circle)
    for circle in candidates:
        if circle.radius < best.radius and all(circle.contains(p) for p in poly):
            best = circle
    return best


def point_in_polygon(point: Point, poly: Polygon) -> bool:
    if len(poly) < 3:
        return bool(poly) and min(distance(point, p) for p in poly) < EPS
    return all((b[0]-a[0])*(point[1]-a[1]) - (b[1]-a[1])*(point[0]-a[0]) >= -EPS
               for a, b in zip(poly, poly[1:]+poly[:1]))


def optical_cover(poly: Polygon, spacing: float) -> list[Point]:
    """Centers of every square cell touching P, including cells near its edge.

    Expanded axis-aligned half-plane clipping checks cell intersection. Every point
    in the region is <= spacing/sqrt(2) from a retained cell center.
    """
    xs, ys = zip(*poly)
    points = []
    for i in range(math.floor(min(xs)/spacing), math.floor(max(xs)/spacing)+1):
        for j in range(math.floor(min(ys)/spacing), math.floor(max(ys)/spacing)+1):
            cell = poly
            for a, b, c in ((-1, 0, -i*spacing), (1, 0, (i+1)*spacing),
                            (0, -1, -j*spacing), (0, 1, (j+1)*spacing)):
                cell = clip_half_plane(cell, a, b, c)
            if cell:
                points.append(((i+0.5)*spacing, (j+0.5)*spacing))
    return points
