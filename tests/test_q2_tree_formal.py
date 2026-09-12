"""Meaningful annual-run boundaries, no-tree ablation and resume safeguards."""

import ast
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.problem2.tree_planning.model import Config, execute_plan, plan_day
from experiments.problem2.tree_planning.risk import QUANTILE_LEVELS, TreeResidualScenarios
from experiments.problem2.tree_planning.run import read_completed
from experiments.problem2.tree_planning.verify import battery_metrics, verify_arrays


class FormalPlanningTests(unittest.TestCase):
    def test_per_slot_supports_same_history_without_tree(self):
        rng = np.random.default_rng(921)
        actual = rng.uniform(1, 5000, (50 * 144, 2))
        origins = np.arange(31, 50) * 144
        forecasts = rng.uniform(1, 5000, (19, 144, 2))
        prices = np.ones(144)
        source = TreeResidualScenarios(origins, forecasts, actual, prices, conditioning="per_slot")
        for day in (31, 32, 40):
            supports, info = source.for_day(day)
            _, tree_info = TreeResidualScenarios(origins, forecasts, actual, prices).for_day(day)
            self.assertEqual(info["training_origins"], tree_info["training_origins"])
            self.assertEqual(info["residual_source"], tree_info["residual_source"])
            self.assertEqual(info["tree_nodes"], 0)
            self.assertEqual(info["fit_seconds"], 0)
            future = actual.copy()
            future[day * 144:] = 1e9
            alternative, _ = TreeResidualScenarios(
                origins, forecasts, future, prices, conditioning="per_slot").for_day(day)
            np.testing.assert_array_equal(supports, alternative)
        day = 40
        historical_truth = actual[(origins[:9, None] + np.arange(144))]
        residuals = ((historical_truth[:, :, 0] - historical_truth[:, :, 1])
                     - (forecasts[:9, :, 0] - forecasts[:9, :, 1])) / 6
        expected = ((forecasts[9, :, 0] - forecasts[9, :, 1]) / 6)[:, None]
        expected = expected + np.quantile(residuals, QUANTILE_LEVELS, axis=0).T
        np.testing.assert_allclose(supports, expected)

    def test_two_day_soc_direction_continuity_and_independent_tamper_detection(self):
        rng = np.random.default_rng(45)
        prices = np.r_[np.full(72, .3), np.full(72, 1.2)]
        config = Config(grid_kwh=600)
        state, mode = 1421.7991105135516, 1
        details, rows, audit = [], [], []
        for day in (31, 32):
            plan = plan_day(np.tile(np.linspace(10, 600, 9), (144, 1)), prices,
                            state, config, initial_mode=mode, final_evaluation_day=day == 32)
            detail, metrics = execute_plan(plan, rng.uniform(0, 6000, (144, 2)),
                                           prices, state, config, initial_mode=mode)
            rows.append({"day": day, **metrics})
            audit.append({"day": day, "information_cutoff": day * 144,
                          "max_observed_index": day * 144 - 1,
                          "training_origins": [(day - 1) * 144], "forecast_origin": day * 144,
                          "execution_initial_soc": state, "execution_initial_mode": mode,
                          "execution_final_mode": metrics["final_mode"]})
            state, mode = metrics["final_soc"], metrics["final_mode"]
            details.append(detail)
        stacked = {key: np.stack([d[key] for d in details]) for key in details[0]}
        report = verify_arrays(stacked, rows, audit, expected_days=2)
        self.assertTrue(report["passed"], report["errors"])
        stacked["fees"][0, 0, 0] += 1
        self.assertIn("slot_settlement", verify_arrays(stacked, rows, audit, expected_days=2)["errors"])
        stacked["fees"][0, 0, 0] -= 1
        audit[1]["max_observed_index"] = 32 * 144
        self.assertIn("future_actual_used:32", verify_arrays(stacked, rows, audit, expected_days=2)["errors"])

    def test_metrics_separate_warmup_boundary_from_evaluation(self):
        detail = {"charge": np.array([[0., 0., 2., 0., 0.]]),
                  "discharge": np.array([[2., 0., 0., 0., 2.]]),
                  "states": np.array([[1500., 1500.]])}
        result = battery_metrics(detail, initial_mode=1, initial_power_kw=3.)
        self.assertEqual(result["direction_reversals"], 2)
        self.assertEqual(result["direction_reversals_including_warmup_boundary"], 3)
        self.assertEqual(result["power_ramp_total_kw"], 48)
        self.assertEqual(result["power_ramp_total_kw_including_warmup_boundary"], 63)

    def test_completed_signature_mismatch_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            content = json.dumps({"signature": "old", "complete": True})
            (path / "completion.json").write_text(content)
            with self.assertRaisesRegex(RuntimeError, "incompatible completed"):
                read_completed(path, "new")
            self.assertEqual((path / "completion.json").read_text(), content)

    def test_greedy_matches_frozen_executor_without_calling_solver(self):
        source = Path(__file__).resolve().parents[1] / "experiments/common/neural_v2/physics.py"
        tree = ast.parse(source.read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == "execute")
        namespace = {"np": np, "ETA": np.sqrt(.9), "MAX_SOC": 10800.,
                     "MIN_SOC": 1200., "POWER_ENERGY": 5000 / 6}
        # Execute only the frozen local pure executor, excluding solver imports.
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)  # noqa: S102
        rng = np.random.default_rng(412)
        purchase = rng.uniform(0, 1500, 144)
        actual = rng.uniform(0, 9000, (144, 2))
        plan = {"purchase": purchase, "charge": np.zeros(144), "discharge": np.zeros(144)}
        detail, _ = execute_plan(plan, actual, np.ones(144), 1421.7991105135516,
                                 greedy=True, initial_mode=1)
        expected = namespace["execute"](purchase, actual[:, 0], actual[:, 1], 1421.7991105135516)
        for key, values in zip(("charge", "discharge", "emergency", "surplus", "states"), expected):
            np.testing.assert_array_equal(detail[key], values)


if __name__ == "__main__":
    unittest.main()
