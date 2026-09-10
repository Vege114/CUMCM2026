"""Boundary, geometry, accounting and transport invariants; no real simulator calls."""
import math
from pathlib import Path
import random
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baseline.config import Config
from baseline.geometry import (bearing, clip_bearing, distance, enclosing_circle,
                               optical_cover, outer_circle, point_in_polygon, polygon_diameter)
from baseline.mock import MockTransport, Source, generate_sources
from baseline.planning import survey_points
from baseline.protocol import Client, ProtocolError
from baseline.strategy import Strategy


class GeometryTests(unittest.TestCase):
    def test_diameter_circle_does_not_cover_equilateral_triangle(self):
        triangle = [(0, 0), (2, 0), (1, math.sqrt(3))]
        diameter, _ = polygon_diameter(triangle)
        self.assertAlmostEqual(diameter, 2)
        circle = enclosing_circle(triangle)
        self.assertAlmostEqual(circle.radius, 2/math.sqrt(3))
        self.assertGreater(circle.radius, diameter/2)
        self.assertTrue(all(circle.contains(p) for p in triangle))

    def test_bearing_wrap_and_extreme_error_contain_truth(self):
        rng = random.Random(19)
        for _ in range(100):
            truth = (rng.uniform(-900, 900), rng.uniform(-900, 900))
            poly = outer_circle(1800)
            for p, error in [((0, 0), -1.005), ((-200, 120), 1.005), ((320, -70), 0)]:
                if distance(p, truth) <= 1500:
                    poly = clip_bearing(poly, p, (bearing(p, truth)+error) % 360, 1.01)
                    self.assertTrue(point_in_polygon(truth, poly))
        poly = clip_bearing(outer_circle(1800), (0, 0), 359.99, 1.01)
        self.assertTrue(point_in_polygon((1000, 0), poly))

    def test_target_disk_outer_approximation(self):
        poly = outer_circle(1800)
        for i in range(720):
            a = i*math.pi/360
            self.assertTrue(point_in_polygon((1800*math.cos(a), 1800*math.sin(a)), poly))

    def test_optical_cover_includes_cell_boundary(self):
        poly = [(0, 0), (72, 0), (72, 37), (0, 37)]
        centers = optical_cover(poly, 25)
        for x in range(73):
            for y in range(38):
                self.assertLessEqual(min(distance((x, y), p) for p in centers), 20)


class CoverageTests(unittest.TestCase):
    def test_omnidirectional_coverage(self):
        points = survey_points(3, Config())
        for r in range(0, 1801, 100):
            for a in range(360):
                p = (r*math.cos(math.radians(a)), r*math.sin(math.radians(a)))
                self.assertLessEqual(min(distance(p, q) for q in points), 1000)

    def test_directional_boundary_and_grid_vertices(self):
        points = survey_points(4, Config())
        truths = [(1800*math.cos(math.radians(a)), 1800*math.sin(math.radians(a))) for a in range(0, 360, 10)]
        truths += [(0, 0), (700, 700), (700, 0), (1400, 0)]
        for p in truths:
            nearby = [q for q in points if 0 < distance(p, q) <= 1000]
            for orientation in range(0, 360, 5):
                self.assertTrue(any(abs((bearing(p, q)-orientation+180) % 360-180) <= 90+1e-7 for q in nearby))


class ProtocolTests(unittest.TestCase):
    def test_official_accounting_example(self):
        with tempfile.TemporaryDirectory() as directory:
            client = Client(MockTransport([]), "test", Path(directory)/"events.jsonl")
            client.call("/enter")
            self.assertEqual(client.call("/measure", (300, 400), 1)["virtual_time_s"], 105)
            self.assertEqual(client.call("/measure", (300, 400), 2)["virtual_time_s"], 111)
            self.assertEqual(client.call("/clear", (300, 0), 3)["virtual_time_s"], 194)
            self.assertEqual(client.call("/measure", (300, 0), 2)["virtual_time_s"], 199)
            self.assertEqual(client.call("/exit")["virtual_time_s"], 199)
            client.close()

    def test_lost_response_retries_exact_same_action(self):
        engine = MockTransport([])
        class LostResponse:
            lost = False
            def send(self, path, payload):
                response = engine.send(path, payload)
                if path == "/measure" and not self.lost:
                    self.lost = True
                    raise OSError("response dropped after execution")
                return response
        with tempfile.TemporaryDirectory() as directory:
            client = Client(LostResponse(), "test", Path(directory)/"events.jsonl")
            client.call("/enter")
            response = client.call("/measure", (300, 400), 2)
            self.assertEqual(response["virtual_time_s"], 106)
            self.assertEqual(len(engine.cache), 2)
            client.close()

    def test_rejection_does_not_reset_virtual_clock(self):
        with tempfile.TemporaryDirectory() as directory:
            client = Client(MockTransport([]), "test", Path(directory)/"events.jsonl")
            client.call("/enter")
            client.call("/measure", (300, 400), 2)
            with self.assertRaises(ProtocolError):
                client.call("/enter")
            self.assertEqual(client.virtual_time_s, 106)
            client.close()


class EndToEndTests(unittest.TestCase):
    def execute(self, sources, problem):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            engine = MockTransport(sources, seed=81)
            client = Client(engine, "test", folder/"events.jsonl")
            strategy = Strategy(client, Config(), problem, folder/"decisions.jsonl")
            try:
                client.call("/enter")
                result = strategy.run(1200)
                client.call("/exit")
                self.assertIn(result, ("all_16_cleared", "coverage_complete"))
                self.assertEqual(strategy.cleared_count, len(sources))
            finally:
                client.close()
                strategy.close()

    def test_fixed_scenarios(self):
        for problem in (3, 4):
            for seed in (3, 17, 2026):
                with self.subTest(problem=problem, seed=seed):
                    self.execute(generate_sources(seed, problem), problem)

    def test_outward_sources_at_minimum_reception_radius(self):
        sources = []
        for ch in range(1, 13):
            a = math.radians(ch*30)
            sources.append(Source(ch, (1800*math.cos(a), 1800*math.sin(a)), 1000, ch*30))
        self.execute(sources, 4)


if __name__ == "__main__":
    unittest.main()
