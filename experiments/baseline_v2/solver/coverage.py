"""Provable reference covers and conservative per-channel Q3 cell certificates."""
import math

from experiments.baseline_v1.baseline.planning import survey_points as v1_survey_points
from .geometry import disk_intersects_polygon, distance


def survey_points(problem, cfg):
    if problem == 3 or not cfg.triangular_coverage:
        return v1_survey_points(problem, cfg)
    side = cfg.triangle_side_m
    def vertex(i, j):
        return (side*(i+j/2), side*math.sqrt(3)*j/2)
    n = math.ceil(2*cfg.arena_radius_m/side)+3
    points = set()
    for i in range(-n, n):
        for j in range(-n, n):
            a, b, c, d = vertex(i, j), vertex(i+1, j), vertex(i, j+1), vertex(i+1, j+1)
            for triangle in ([a, b, c], [b, d, c]):
                if disk_intersects_polygon(triangle, cfg.arena_radius_m):
                    points.update(triangle)
    # Every intersecting equilateral cell is retained. Its vertices are <= side
    # from every source in it and surround it: every radiation half-plane hits one.
    return sorted(points)


class CoverageLedger:
    """Positive and negative scans both count as coverage witnesses.

    Unknown channels can be declared absent only by a completed reference cover,
    or (Q3) discs covering whole cells, including all their corners. Q4 never
    reuses the omnidirectional negative-evidence certificate.
    """
    def __init__(self, problem, cfg):
        self.problem, self.cfg = problem, cfg
        self.reference = frozenset(survey_points(problem, cfg))
        self.scanned = {ch: set() for ch in range(1, 21)}
        self.masks = {ch: 0 for ch in range(1, 21)}
        self.mask_cache = {}
        self.cells = []
        if problem == 3 and cfg.adaptive_coverage:
            h = cfg.coverage_cell_m
            n = math.ceil(cfg.arena_radius_m/h)
            for i in range(-n, n):
                for j in range(-n, n):
                    corners = [(i*h, j*h), ((i+1)*h, j*h), ((i+1)*h, (j+1)*h), (i*h, (j+1)*h)]
                    if disk_intersects_polygon(corners, cfg.arena_radius_m):
                        self.cells.append(corners)
        self.full_mask = (1 << len(self.cells))-1

    def point_mask(self, point):
        if point not in self.mask_cache:
            mask = 0
            for i, corners in enumerate(self.cells):
                if max(distance(point, v) for v in corners) <= self.cfg.reception_min_m-1e-7:
                    mask |= 1 << i
            self.mask_cache[point] = mask
        return self.mask_cache[point]

    def observe(self, channel, point):
        self.scanned[channel].add(point)
        if self.cells:
            self.masks[channel] |= self.point_mask(point)

    def complete(self, channel):
        return (self.reference.issubset(self.scanned[channel]) or
                bool(self.cells) and self.masks[channel] == self.full_mask)

    def new_fraction(self, channel, point):
        if not self.cells:
            return 0.0
        new = self.point_mask(point) & ~self.masks[channel]
        return new.bit_count()/len(self.cells)

    def certificate(self, channel):
        return {"channel": channel, "reference_points_scanned": len(self.reference & self.scanned[channel]),
                "reference_points_total": len(self.reference), "covered_cells": self.masks[channel].bit_count(),
                "cells_total": len(self.cells), "complete": self.complete(channel)}
