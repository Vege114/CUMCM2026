"""Geometry certificates, negative evidence, boundaries and state-machine failures."""
from dataclasses import replace
import json
import math
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1]))
from experiments.baseline_v1.baseline.mock import MockTransport, Source
from experiments.baseline_v2.solver.artifacts import safe_name
from experiments.baseline_v2.solver import artifacts
from experiments.baseline_v2.solver.config import Config, load_config
from experiments.baseline_v2.solver.evaluation import evaluate_run
from experiments.baseline_v2.solver.coverage import CoverageLedger, survey_points
from experiments.baseline_v2.solver.geometry import (
    bearing, clip_bearing, contains, convex_hull, direction, distance, oriented_optical_cover, outer_circle,
)
from experiments.baseline_v2.solver.protocol import CheckedTransport, Client, TransportUnavailable
from experiments.baseline_v2.solver.routing import open_route, path_length
from experiments.baseline_v2.solver.runner import execute_run
from experiments.baseline_v2.solver.scenarios import Scenario, ScenarioTransport, create_sources
from experiments.baseline_v2.solver.strategy import Strategy, Track
from experiments.baseline_v2.solver.visibility import VisibilityModel, arc, intersect, length
from experiments.baseline_v2.verify_artifacts import audit


class GeometryTests(unittest.TestCase):
    def test_degenerate_received_hulls(self):
        self.assertTrue(contains(convex_hull([(0, 0), (10, 0)]), (4, 0)))
        self.assertFalse(contains(convex_hull([(0, 0), (10, 0)]), (4, 0.001)))
        self.assertTrue(contains([(1, 2)], (1, 2)))
        self.assertFalse(contains([], (1, 2)))

    def test_angular_wrap_and_disjoint(self):
        self.assertEqual(length(arc(0)), 180)
        self.assertAlmostEqual(length(intersect(arc(350), arc(10))), 160)
        self.assertEqual(length(intersect(arc(0), arc(180))), 0)

    def test_rotated_optical_cover_preserves_boundary_truth(self):
        rng = random.Random(412)
        for angle in (0, 19, 47, 89, 150):
            ux, uy = direction(angle)
            def rotate(x, y):
                return (155+x*ux-y*uy, -291+x*uy+y*ux)
            polygon = [rotate(0, 0), rotate(1500, 0), rotate(1500, 36), rotate(0, 36)]
            points = oriented_optical_cover(polygon, 27.5)
            samples = polygon + [rotate(rng.uniform(0, 1500), rng.uniform(0, 36)) for _ in range(150)]
            for q in samples:
                self.assertLessEqual(min(distance(q, p) for p in points), 20)

    def test_route_free_endpoint_and_no_worse_than_greedy(self):
        rng = random.Random(319)
        points = [(rng.uniform(-1000, 1000), rng.uniform(-1000, 1000)) for _ in range(40)]
        greedy = open_route((0, 0), points, False)
        improved = open_route((0, 0), points, True)
        self.assertEqual(set(improved), set(points))
        self.assertEqual(len(improved), len(points))
        self.assertLessEqual(path_length((0, 0), improved), path_length((0, 0), greedy)+1e-7)


class CoverageTests(unittest.TestCase):
    def test_q3_smaller_ring_continuous_bound(self):
        cfg = Config().validate()
        for r in (1000, 1800):
            bound = math.sqrt(r*r+cfg.survey_ring_m**2-2*r*cfg.survey_ring_m*math.cos(math.pi/6))
            self.assertLess(bound, 1000)

    def test_triangle_cover_all_orientations_at_boundaries(self):
        points = survey_points(4, Config())
        self.assertEqual(len(points), 31)
        truths = [(1800*math.cos(i/180*math.pi), 1800*math.sin(i/180*math.pi)) for i in range(0, 360, 3)]
        truths += [p for p in points if distance(p, (0, 0)) <= 1800]
        rng = random.Random(119)
        truths += [(rng.uniform(-1000, 1000), rng.uniform(-1000, 1000)) for _ in range(100)]
        for q in truths:
            received = [p for p in points if distance(p, q) <= 1000]
            for angle in range(0, 360, 3):
                ux, uy = direction(angle)
                self.assertTrue(any(ux*(p[0]-q[0])+uy*(p[1]-q[1]) >= -1e-7 for p in received))

    def test_reference_complete_requires_every_channel_point(self):
        ledger = CoverageLedger(4, Config())
        for p in list(ledger.reference)[:-1]:
            ledger.observe(1, p)
        self.assertFalse(ledger.complete(1))
        for p in ledger.reference:
            ledger.observe(1, p)
        self.assertTrue(ledger.complete(1))
        self.assertFalse(ledger.complete(2))

    def test_q3_adaptive_cells_are_wholly_covered(self):
        ledger = CoverageLedger(3, Config())
        for point in ((0, 0), (1300, 0), (-1300, 0)):
            mask = ledger.point_mask(point)
            for i, corners in enumerate(ledger.cells):
                if mask & (1 << i):
                    self.assertTrue(all(distance(point, q) <= 1000 for q in corners))
        # A dense grid independently certifies all cells without using ring nodes.
        for x in range(-1800, 1801, 600):
            for y in range(-1800, 1801, 600):
                ledger.observe(1, (x, y))
        self.assertTrue(ledger.complete(1))
        self.assertFalse(ledger.reference.issubset(ledger.scanned[1]))


class VisibilityTests(unittest.TestCase):
    def make_track(self):
        return Track(channel=1, polygon=[(-1, -1), (1, -1), (1, 1), (-1, 1)],
                     bearings=[((100, 0), 180)], tried=[(100, 0)])

    def test_positive_and_guaranteed_range_negative_constrain_orientation(self):
        track = self.make_track()
        track.misses = [(0, 100)]
        before = list(track.polygon)
        model = VisibilityModel(track, Config(), 4)
        self.assertGreater(model.score((100, -100)), model.score((-100, 100)))
        self.assertEqual(track.polygon, before)
        self.assertTrue(all(not s.omni_possible for s in model.states))

    def test_far_no_signal_is_not_treated_as_directional_blindness(self):
        track = self.make_track()
        base = VisibilityModel(track, Config(), 4)
        track.misses = [(10000, 10000)]
        other = VisibilityModel(track, Config(), 4)
        self.assertAlmostEqual(base.score((100, -100)), other.score((100, -100)))
        self.assertTrue(all(s.omni_possible for s in other.states))

    def test_received_convex_hull_guarantees_signal_with_unknown_radius(self):
        track = self.make_track()
        track.bearings.append(((0, 100), 270))
        model = VisibilityModel(track, Config(), 4)
        self.assertTrue(model.guaranteed((50, 50)))
        self.assertEqual(model.score((50, 50)), 1)
        self.assertFalse(model.guaranteed((-50, -50)))


class ConfigTests(unittest.TestCase):
    def test_invalid_constants_types_nan_and_retry_budget_rejected(self):
        for values in ({"arena_radius_m": 1700}, {"clear_radius_m": 21}, {"bearing_error_deg": 1.0},
                       {"triangle_side_m": 1000}, {"optical_grid_m": 29}, {"max_actions": 1.5},
                       {"polygon_sides": 8.1}, {"movement_weight": float("nan")},
                       {"visibility_enabled": 1}, {"http_retries": -1},
                       {"http_timeout_s": 15}, {"max_virtual_time_s": 360000}, {"max_actions": True}):
            with self.subTest(values=values), self.assertRaises((ValueError, TypeError)):
                replace(Config(), **values).validate()

    def test_json_bom_and_unknown_field(self):
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp)/"config.json"
            file.write_text('\ufeff{"movement_weight": 0.05}', encoding="utf-8")
            self.assertEqual(load_config(file).movement_weight, 0.05)
            file.write_text('{"unknown": 1}', encoding="utf-8")
            with self.assertRaises(TypeError):
                load_config(file)

    def test_windows_names_and_directory_escape(self):
        for name in ("../foo", "/tmp/bar", "C:\\foo", "CON", "aux.json", "test.", ""):
            with self.subTest(name=name), self.assertRaises(ValueError):
                safe_name(name)
        self.assertEqual(safe_name("practice_q4_01"), "practice_q4_01")

    def test_source_hash_normalizes_windows_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            root = repo/"experiments/baseline_v2"
            root.mkdir(parents=True)
            (repo/"Benchmark").mkdir()
            (repo/"Benchmark/evaluate.py").write_text("# fixture\n", encoding="utf-8")
            script = root/"practice.ps1"
            with patch.object(artifacts, "ROOT", root), patch.object(artifacts, "REPO", repo):
                script.write_bytes(b"param()\nexit 0\n")
                first = artifacts.source_manifest()["sha256"]
                script.write_bytes(b"param()\r\nexit 0\r\n")
                self.assertEqual(first, artifacts.source_manifest()["sha256"])
                script.write_bytes(b"param()\r\nexit 1\r\n")
                self.assertNotEqual(first, artifacts.source_manifest()["sha256"])


class IntegrationTests(unittest.TestCase):
    def run_fixture(self, spec, cfg=None):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/"run"
            sources = create_sources(spec)
            metadata, metrics = execute_run(out, problem=spec.problem, cfg=cfg or Config().validate(),
                                           transport=ScenarioTransport(sources, spec), scenario=spec,
                                           sources=sources, quiet=True)
            self.assertEqual(metadata["status"], "completed", metadata.get("error"))
            self.assertTrue(metrics["validation"]["experiment_passed"])
            self.assertTrue(audit(out)["passed"], audit(out))
            events = [json.loads(s) for s in (out/"events.jsonl").read_text().splitlines()]
            seen = set()
            for e in events:
                if e["path"] == "/measure":
                    request = e["request"]
                    key = (request["position"]["x"], request["position"]["y"], request["channel"])
                    self.assertNotIn(key, seen)
                    seen.add(key)
            return metadata, metrics

    def test_legacy_seeds_both_problems(self):
        for q in (3, 4):
            for seed in (20260910, 20260911, 20260912):
                with self.subTest(q=q, seed=seed):
                    self.run_fixture(Scenario(f"q{q}_{seed}", q, seed))

    def test_outward_tangent_coincident_extreme_errors(self):
        for layout in ("outward", "tangent", "coincident"):
            with self.subTest(layout=layout):
                self.run_fixture(Scenario(layout, 4, 91, layout, 16, 1, "minimum", "positive"))

    def test_optical_fallback_is_exercised_and_still_covers(self):
        _, metrics = self.run_fixture(Scenario("fallback", 4, 12), replace(Config(), max_local_measurements=1))
        self.assertGreater(metrics["v2_diagnostics"]["optical_fallback_count"], 0)

    def test_budget_stops_incomplete_and_archives_exit(self):
        spec = Scenario("limited", 3, 2)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/"run"
            sources = create_sources(spec)
            metadata, metrics = execute_run(out, problem=3, cfg=replace(Config(), max_actions=1),
                                           transport=ScenarioTransport(sources, spec), sources=sources, quiet=True)
            self.assertEqual(metadata["status"], "incomplete")
            self.assertFalse(metrics["validation"]["experiment_passed"])
            self.assertTrue(metrics["validation"]["complete_event_log"])

    def test_unknown_transport_outcome_never_sends_exit(self):
        engine = MockTransport([])
        paths = []
        class LostResponse:
            def send(self, path, request):
                paths.append(path)
                response = engine.send(path, request)
                if path == "/measure":
                    raise OSError("response lost after execution")
                return response
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/"run"
            metadata, _ = execute_run(out, problem=3, cfg=replace(Config(), http_retries=0),
                                      transport=LostResponse(), sources=[], quiet=True)
            self.assertTrue(metadata["outcome_uncertain"])
            self.assertNotIn("/exit", paths)
            self.assertEqual(metadata["status"], "failed")

    def test_schema_recovery_reuses_original_id(self):
        engine = MockTransport([])
        ids = []
        class TruncatedBody:
            def send(self, path, request):
                status, response = engine.send(path, request)
                if path == "/measure":
                    ids.append(request["request_id"])
                    if len(ids) == 1:
                        response.pop("measure_result")
                return status, response
        with tempfile.TemporaryDirectory() as tmp:
            client = Client(CheckedTransport(TruncatedBody()), "secret-team", Path(tmp)/"events.jsonl")
            try:
                client.call("/enter")
                result = client.call("/measure", (300, 400), 2)
                self.assertEqual(result["virtual_time_s"], 106)
                self.assertEqual(ids[0], ids[1])
            finally:
                client.close()
            self.assertNotIn("secret-team", (Path(tmp)/"events.jsonl").read_text())

    def test_zero_distance_directional_source_is_near(self):
        spec = Scenario("same_position", 4, 3, "coincident", 12, 1, "minimum")
        sources = create_sources(spec)
        engine = ScenarioTransport(sources, spec)
        engine.send("/enter", {"request_id": "enter"})
        for source in sources:
            response = engine.send("/measure", {"request_id": f"m{source.channel}", "channel": source.channel,
                                                "position": {"x": 0, "y": 0}})[1]
            self.assertEqual(response["measure_result"], "near")
            self.assertIsNotNone(source.orientation_deg)

    def test_inconsistent_near_never_turns_into_absence_success(self):
        engine = MockTransport([])
        class BadNear:
            def send(self, path, request):
                status, response = engine.send(path, request)
                if path == "/measure":
                    response["measure_result"] = "near"
                return status, response
        with tempfile.TemporaryDirectory() as tmp:
            metadata, metrics = execute_run(Path(tmp)/"run", problem=3, cfg=Config(),
                                           transport=BadNear(), mode="practice", quiet=True)
            self.assertEqual(metadata["status"], "failed")
            self.assertIn("near", metadata["error"])
            self.assertFalse(metrics["validation"]["experiment_passed"])

    def test_practice_ui_total_disproving_full_clear_fails_validation(self):
        spec = Scenario("practice_fixture", 3, 9, "coincident", 10)
        sources = create_sources(spec)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/"run"
            metadata, metrics = execute_run(out, problem=3, cfg=Config(), transport=ScenarioTransport(sources, spec),
                                           mode="practice", quiet=True)
            self.assertTrue(metrics["validation"]["experiment_passed"])
            self.assertIsNone(metrics["official_metrics"]["cleared_ratio"])
            metadata.update(total_jammers=12, total_source="simulator_ui_post_run")
            metrics = evaluate_run(out, metadata)
            self.assertFalse(metrics["validation"]["experiment_passed"])
            artifacts.write_json(out/"metadata.json", metadata)
            self.assertFalse(audit(out)["passed"])

    def test_non_utf8_executed_response_retries_same_id(self):
        engine = MockTransport([])
        ids = []
        class InvalidUTF8:
            def send(self, path, request):
                response = engine.send(path, request)
                if path == "/measure":
                    ids.append(request["request_id"])
                    if len(ids) == 1:
                        b"\xff".decode("utf-8")
                return response
        with tempfile.TemporaryDirectory() as tmp:
            client = Client(CheckedTransport(InvalidUTF8()), "test", Path(tmp)/"events.jsonl")
            try:
                client.call("/enter")
                client.call("/measure", (300, 400), 1)
                self.assertEqual(client.virtual_time_s, 105)
                self.assertEqual(len(set(ids)), 1)
            finally:
                client.close()


if __name__ == "__main__":
    unittest.main()
