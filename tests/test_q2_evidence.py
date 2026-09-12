import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from experiments.problem2.exp003.baseline_evidence import (
    RAW_FILES,
    boundary,
    legacy_selection,
    load_midnight,
    metric_row,
    read_actual,
    score_forecast,
)


class MidnightEvidenceTests(unittest.TestCase):
    def test_metrics_keep_signed_net_load_and_zero_denominators(self):
        row = metric_row([-1, 3], [-2, 2])
        self.assertEqual(row["wape_pct"], 50)
        self.assertEqual(row["bias"], 1)
        self.assertEqual(row["rmse"], 1)
        self.assertIsNone(metric_row([2, 1], [0, 0])["wape_pct"])
        self.assertIsNone(metric_row([], [])["mae"])
        with self.assertRaises(ValueError):
            metric_row([1, 2], [1])

    def test_monthly_sums_equal_annual_and_use_actual_pv_mask(self):
        truth = np.array([[[1., 0.], [1., 3.]], [[4., 1.], [3., 0.]]])
        prediction = truth + np.array([[[1., 5.], [2., 1.]], [[3., 2.], [4., 7.]]])
        rows = score_forecast(prediction, truth, np.array([2, 3]))
        for target in ("load", "pv", "net_load"):
            annual = next(r for r in rows if r["month"] == 0 and r["target"] == target and r["population"] == "all")
            monthly = [r for r in rows if r["month"] and r["target"] == target and r["population"] == "all"]
            for field in ("n", "absolute_error_sum", "squared_error_sum", "signed_error_sum", "actual_abs_sum"):
                self.assertEqual(annual[field], sum(r[field] for r in monthly))
            generating = next(r for r in rows if r["month"] == 0 and r["target"] == target and r["population"] == "pv_generating")
            self.assertEqual(generating["n"], 2)
        net = next(r for r in rows if r["month"] == 0 and r["target"] == "net_load" and r["population"] == "all")
        self.assertEqual(net["actual_abs_sum"], 9)  # 1 + |-2| + 3 + 3.

    def test_archive_uses_origins_and_rejects_duplicate_or_missing_midnights(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "predictions.npz"
            values = np.stack([np.full((144, 4), i) for i in (2., 1., 99.)])
            np.savez(path, origins=[4608, 4464, 4500], seed_42=values)
            selected = load_midnight(path, np.array([4464, 4608]))["seed_42"]
            self.assertEqual(selected.shape, (2, 144, 2))
            np.testing.assert_array_equal(selected[:, 0, 0], [1, 2])
            with self.assertRaisesRegex(ValueError, "missing"):
                load_midnight(path, np.array([4752]))
            np.savez(path, origins=[4464, 4464, 4500], seed_42=values)
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                load_midnight(path, np.array([4464]))

    def test_legacy_selection_preserves_q2_and_month_identity(self):
        table = pd.DataFrame([
            {"scenario": "3", "month": 2, "variant": "b", "selected": True},
            {"scenario": "2", "month": 2, "variant": "a", "selected": True},
            {"scenario": "2", "month": 3, "variant": "b", "selected": True},
            {"scenario": "2", "month": 2, "variant": "b", "selected": False},
        ])
        arrays = {"a": np.ones((2, 144, 2)), "b": np.full((2, 144, 2), 2.)}
        values, choices = legacy_selection(table, arrays, np.array([2, 3]))
        self.assertEqual(choices, {2: "a", 3: "b"})
        np.testing.assert_array_equal(values[:, 0, 0], [1, 2])
        with self.assertRaisesRegex(ValueError, "exactly one"):
            legacy_selection(pd.concat([table, table.iloc[[1]]]), arrays, np.array([2, 3]))

    def test_ground_truth_reader_never_reads_attachment3_or_attachment4(self):
        frame = pd.DataFrame(np.ones((365, 145)))
        frame[0] = pd.date_range("2025-01-01", "2025-12-31")
        requested = []

        def read(path):
            requested.append(Path(path).name)
            self.assertIn(Path(path).name, RAW_FILES)
            return frame.copy()

        with patch("experiments.problem2.exp003.baseline_evidence.pd.read_csv", side_effect=read):
            self.assertEqual(read_actual(Path("/unused")).shape, (52560, 2))
        self.assertEqual(requested, list(RAW_FILES))

    def test_historical_boundaries_are_not_relabelled_as_isolated(self):
        neural = boundary("old", "exp002", neural=True)
        self.assertFalse(neural["strict_q2_input_isolation_verified"])
        self.assertTrue(neural["attachment_3_4_common_early_stopping_coupling"])
        periodic = boundary("old_periodic", "exp002")
        self.assertFalse(periodic["strict_q2_input_isolation_verified"])
        rebuilt = boundary("periodic", "attachment2", archived=False)
        self.assertTrue(rebuilt["strict_q2_input_isolation_verified"])


if __name__ == "__main__":
    unittest.main()
