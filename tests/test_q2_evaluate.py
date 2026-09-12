"""January-only selection, immutable blending choices, and replay cache integrity."""

import contextlib
import io
import tempfile
import unittest
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from experiments.problem2.exp003 import evaluate
from experiments.problem2.exp003.data import STEPS, midnight_origins
from experiments.problem2.exp003.dispatch import MIN_SOC


def setUpModule():
    from scipy.optimize._highspy._core import _Highs

    _Highs.resetGlobalScheduler(True)


def tearDownModule():
    setUpModule()


class Q2EvaluationTests(unittest.TestCase):
    def test_ties_use_global_minimum_tolerance_then_alpha_complexity(self):
        candidates = [{"alpha": [1., 1.], "cost": 100.},
                      {"alpha": [.5, 0.], "cost": 100.005},
                      {"alpha": [0., .5], "cost": 100.01},
                      {"alpha": [0., 0.], "cost": 100.011}]
        self.assertEqual(evaluate.select_alpha(candidates)["alpha"], [0., .5])
        self.assertEqual(evaluate.select_alpha(candidates, 0)["alpha"], [1., 1.])
        self.assertEqual(evaluate.select_alpha(candidates[::-1])["alpha"], [0., .5])
        with self.assertRaises(ValueError):
            evaluate.select_alpha([])

    def test_calibration_uses_only_january_labels_and_common_initial_soc(self):
        queried, starts, reads = [], [], []

        class JanuaryActual:
            def __getitem__(self, region):
                self_test.assertIsInstance(region, slice)
                self_test.assertGreaterEqual(region.start, 24 * STEPS)
                self_test.assertLessEqual(region.stop, 31 * STEPS)
                reads.append(region)
                return np.column_stack((np.full(region.stop - region.start, 600.),
                                        np.zeros(region.stop - region.start)))

        class Store:
            def get(self, origin, alpha, validation_month):
                queried.append((origin, alpha, validation_month))
                return np.column_stack((np.full(STEPS, alpha[0] * 600.),
                                        np.full(STEPS, alpha[1] * 60.)))

        self_test = self
        data = SimpleNamespace(actual=JanuaryActual(), fixed_price=np.ones(STEPS))

        def record_replay(data, days, predictor, initial):
            starts.append(initial)
            return evaluate.replay_days(data, days, predictor, initial)

        with contextlib.redirect_stdout(io.StringIO()):
            result, daily = evaluate.calibrate_candidates(data, Store(), 2000., replay=record_replay)
        self.assertEqual(starts, [2000.] * 9)
        self.assertEqual(len(reads), 9 * 7)
        self.assertEqual(len(queried), 9 * 7)
        self.assertTrue(all(origin < 31 * STEPS and month == 2 for origin, _, month in queried))
        self.assertEqual(result["selection_time"], 31 * STEPS)
        self.assertEqual(result["validation_days"], list(range(24, 31)))
        self.assertEqual(len(result["candidates"]), 9)
        for candidate in result["candidates"]:
            rows = [row for row in daily if row["candidate_id"] == candidate["candidate_id"]]
            self.assertEqual(rows[0]["initial_soc"], 2000.)
            self.assertAlmostEqual(sum(row["total_cost"] for row in rows), candidate["cost"])
            for previous, current in pairwise(rows):
                self.assertEqual(previous["final_soc"], current["initial_soc"])

    def test_formal_outcomes_cannot_enter_alpha_selection_or_change_seed_blend(self):
        cfg = evaluate.protocol()
        cfg["calibration"]["end"] = "2025-02-01"
        with self.assertRaisesRegex(ValueError, "January"):
            evaluate.calibrate_candidates(None, None, MIN_SOC, cfg)
        selected = [0., .5]
        jobs = evaluate.cases(selected)
        selected[0] = 1.
        primary = [job for job in jobs if job["name"] == "primary"]
        self.assertEqual([job["seed"] for job in primary], [42, 2026, 3407])
        self.assertTrue(all(job["alpha"] == [0., .5] for job in primary))
        primary[0]["alpha"][0] = .5
        self.assertEqual(primary[1]["alpha"], [0., .5])
        self.assertEqual(len(jobs), 5)
        self.assertEqual(jobs[0]["alpha"], [0., 0.])
        self.assertEqual(jobs[1]["alpha"], [1., 1.])

    def test_midnight_forecast_scores_have_exact_unique_sample_counts_and_cost_units(self):
        origins = midnight_origins()
        actual = np.column_stack((np.full(365 * STEPS, 600.),
                                  np.tile(np.r_[np.zeros(72), np.full(72, 100.)], 365)))
        truth = actual[origins[:, None] + np.arange(STEPS)]
        values = truth.copy()
        values[:, :, 0] += 60
        job = evaluate.cases([.5, 1])[2]
        rows = evaluate.forecast_scores(SimpleNamespace(actual=actual), origins, values, job)
        annual = [row for row in rows if row["period"] == "annual"]
        self.assertEqual(len(annual), 6)
        self.assertEqual(len(rows), 12 * 6)
        for row in annual:
            self.assertEqual(row["n"], 48096 if row["population"] == "all" else 24048)
            self.assertEqual(row["issue_hour"], 0)
            self.assertEqual(row["unit"], "kW")
            months = [month for month in rows if month["period"] == "monthly"
                      and month["target"] == row["target"] and month["population"] == row["population"]]
            self.assertEqual(sum(month["n"] for month in months), row["n"])
            self.assertAlmostEqual(sum(month["absolute_error_sum"] for month in months), row["absolute_error_sum"])
        load = next(row for row in annual if row["target"] == "load" and row["population"] == "all")
        self.assertAlmostEqual(load["bias"], 60)
        self.assertAlmostEqual(load["wape_pct"], 10)
        with self.assertRaisesRegex(ValueError, "334"):
            evaluate.forecast_scores(SimpleNamespace(actual=actual), origins[:-1], values[:-1], job)

    def test_replay_cache_reuses_identical_bytes_and_rejects_changed_sources_forecasts_or_outputs(self):
        data = SimpleNamespace(actual=np.zeros((32 * STEPS, 2)), fixed_price=np.ones(STEPS))
        prediction = np.zeros((STEPS, 2))
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            first = evaluate.cached_replay(data, [31], lambda origin: prediction, MIN_SOC, directory, "source-a")
            with patch.object(evaluate, "day_run", side_effect=AssertionError("cached day must not rerun")):
                second = evaluate.cached_replay(data, [31], lambda origin: prediction, MIN_SOC, directory, "source-a")
            self.assertEqual(first[0], second[0])
            np.testing.assert_array_equal(first[1]["states"], second[1]["states"])
            with self.assertRaisesRegex(RuntimeError, "stale"):
                evaluate.cached_replay(data, [31], lambda origin: prediction, MIN_SOC, directory, "source-b")
            with self.assertRaisesRegex(RuntimeError, "stale"):
                evaluate.cached_replay(data, [31], lambda origin: prediction + 1, MIN_SOC, directory, "source-a")
            arrays = Path(directory) / "d031.npz"
            arrays.write_bytes(arrays.read_bytes() + b"modified")
            with self.assertRaisesRegex(RuntimeError, "stale"):
                evaluate.cached_replay(data, [31], lambda origin: prediction, MIN_SOC, directory, "source-a")

    def test_source_signature_binds_actual_prediction_weights_code_and_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            here = root / "experiments/problem2/exp003"
            here.mkdir(parents=True)
            for name in ("data.py", "dispatch.py", "evaluate.py", "baseline_evidence.py", "train.py",
                         "predict.py", "protocol.json", "provenance.py"):
                (here / name).write_text("source")
            frozen = root / "experiments/common/neural_v2"
            frozen.mkdir(parents=True)
            for name in ("physics.py", "risk.py"):
                (frozen / name).write_text("frozen")
            models = here / "runs/check"
            models.mkdir(parents=True)
            for month in range(2, 13):
                for suffix in (".json", ".npz", ".keras"):
                    (models / f"m{month:02d}_mlp_42{suffix}").write_bytes(b"initial")
            data, cfg = SimpleNamespace(hashes={"attachment": "hash"}), {"seed_list": [42]}
            with patch.object(evaluate, "ROOT", root), patch.object(evaluate, "HERE", here):
                previous = evaluate.source_manifest(data, "check", cfg)["signature"]
                for path in (models / "m02_mlp_42.npz", models / "m02_mlp_42.keras", here / "dispatch.py"):
                    path.write_bytes(b"changed")
                    current = evaluate.source_manifest(data, "check", cfg)["signature"]
                    self.assertNotEqual(previous, current)
                    previous = current
                cfg["alpha"] = [.5, 1]
                self.assertNotEqual(previous, evaluate.source_manifest(data, "check", cfg)["signature"])


if __name__ == "__main__":
    unittest.main()
