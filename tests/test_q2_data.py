"""Input isolation, chronological boundaries, and frozen v2 feature parity."""

import copy
import shutil
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from experiments.problem2.exp003.data import ALLOWED_INPUTS, ROOT, STEPS, Data, midnight_origins


@contextmanager
def input_fixture():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        raw = root / "data/raw"
        raw.mkdir(parents=True)
        for name in ALLOWED_INPUTS:
            shutil.copyfile(ROOT / "data/raw" / name, raw / name)
        yield root


class Q2DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = Data()

    def test_only_allowed_inputs_are_read_and_hashed(self):
        original_read = Path.read_bytes
        accessed = []

        def allowed_read(path):
            self.assertIn(path.name, ALLOWED_INPUTS)
            accessed.append(path.name)
            return original_read(path)

        with input_fixture() as root:
            with patch.object(Path, "read_bytes", autospec=True, side_effect=allowed_read), \
                    patch.object(Path, "glob", side_effect=AssertionError("broad input scans forbidden")):
                data = Data(root)
            self.assertEqual(accessed, list(ALLOWED_INPUTS))
            self.assertEqual(set(data.hashes), set(ALLOWED_INPUTS))
            self.assertEqual(data.actual.shape, (52560, 2))
            self.assertEqual(data.reference.shape, (144, 3))
            self.assertEqual(data.fixed_price.shape, (144,))
            for name in ("附件3.csv", "附件4.csv"):
                (root / "data/raw" / name).write_text("unreadable unrelated input")
            unrelated_changed = Data(root)
            self.assertEqual(data.hashes, unrelated_changed.hashes)
            np.testing.assert_array_equal(data.actual, unrelated_changed.actual)

    def test_future_actuals_do_not_change_current_features_or_baselines(self):
        for origin in (1, 137, 7 * STEPS, 31 * STEPS + 17, 364 * STEPS):
            with self.subTest(origin=origin):
                changed = copy.deepcopy(self.data)
                changed.actual[origin:] *= 100
                cutoff = min(origin, 24 * STEPS)
                for left, right in zip(self.data.features([origin], cutoff), changed.features([origin], cutoff)):
                    np.testing.assert_array_equal(left, right)
                for kind in ("periodic", "yesterday", "weekly"):
                    np.testing.assert_array_equal(self.data.baseline(origin, kind), changed.baseline(origin, kind))

    def test_reference_initializes_first_day_without_actuals(self):
        changed = copy.deepcopy(self.data)
        changed.actual[:] *= 100
        np.testing.assert_array_equal(self.data.baseline(0), self.data.reference[:, 1:3])
        np.testing.assert_array_equal(changed.baseline(0), self.data.baseline(0))

    def test_features_match_frozen_v2_first_two_branches(self):
        from experiments.common.neural_v2.data import Data as V2Data

        # Exercise the frozen feature implementation without reading its
        # broader Q3/Q4 input set, even in this comparison-only test.
        previous = V2Data.__new__(V2Data)
        previous.actual = np.column_stack((self.data.actual, np.zeros(len(self.data.actual))))
        previous.reference = self.data.reference
        origins = [7 * STEPS, 31 * STEPS + 17, 100 * STEPS + 73, 364 * STEPS + 108]
        actual = self.data.features(origins, 7 * STEPS)
        expected = previous.features(origins, 7 * STEPS, issued=False)
        np.testing.assert_array_equal(actual[0], expected[0][:, :, :2])
        np.testing.assert_array_equal(actual[1], expected[1][:, :, :2])
        np.testing.assert_array_equal(actual[2], expected[2][:2])
        np.testing.assert_array_equal(actual[3], expected[3][:2])

    def test_labels_respect_exclusive_historical_cutoff_and_year_end(self):
        cutoff = 24 * STEPS
        origin = cutoff - STEPS
        np.testing.assert_array_equal(self.data.labels([origin], cutoff), self.data.actual[None, origin:cutoff].astype("float32"))
        for forbidden in (origin + 1, cutoff):
            with self.assertRaisesRegex(ValueError, "target window"):
                self.data.labels([forbidden], cutoff)
        changed = copy.deepcopy(self.data)
        changed.actual[cutoff:] *= 100
        np.testing.assert_array_equal(self.data.labels([origin], cutoff), changed.labels([origin], cutoff))
        self.assertEqual(self.data.labels([364 * STEPS]).shape, (1, STEPS, 2))
        with self.assertRaisesRegex(ValueError, "target window"):
            self.data.labels([364 * STEPS + 1])
        self.assertEqual(self.data.labels([]).shape, (0, STEPS, 2))

    def test_midnight_evaluation_contains_exactly_48096_unique_targets(self):
        origins = midnight_origins()
        targets = origins[:, None] + np.arange(STEPS)
        self.assertEqual(len(origins), 334)
        np.testing.assert_array_equal(origins % STEPS, 0)
        np.testing.assert_array_equal(targets.ravel(), np.arange(31 * STEPS, 365 * STEPS))
        self.assertEqual(self.data.labels(origins).shape, (334, 144, 2))
        self.assertEqual(targets.size, 48096)
        np.testing.assert_array_equal(midnight_origins(2), midnight_origins(start="2025-02-01", end="2025-02-28"))
        self.assertEqual(len(midnight_origins(1)), 31)

    def test_malformed_shape_dates_times_and_values_are_rejected(self):
        def mutate_shape(frame):
            return frame.iloc[:-1]

        def mutate_date(frame):
            frame.iloc[1, 0] = frame.iloc[0, 0]
            return frame

        def mutate_header_time(frame):
            names = list(frame.columns)
            names[1], names[2] = names[2], names[1]
            frame.columns = names
            return frame

        def mutate_reference_time(frame):
            frame.iloc[-1, 0] = "00:00"
            return frame

        mutations = [(ALLOWED_INPUTS[1], mutate_shape, "shape"),
                     (ALLOWED_INPUTS[1], mutate_date, "every 2025 date"),
                     (ALLOWED_INPUTS[1], mutate_header_time, "interval ends"),
                     (ALLOWED_INPUTS[0], mutate_reference_time, "interval ends")]
        for name in ALLOWED_INPUTS:
            for value in (-1., np.inf, np.nan):
                def bad_value(frame, value=value):
                    frame = frame.astype({frame.columns[1]: float})
                    frame.iloc[0, 1] = value
                    return frame
                mutations.append((name, bad_value, "finite and nonnegative"))
        for name, mutate, error in mutations:
            with self.subTest(file=name, mutation=mutate.__name__, error=error), input_fixture() as root:
                path = root / "data/raw" / name
                mutate(pd.read_csv(path)).to_csv(path, index=False)
                with self.assertRaisesRegex(ValueError, error):
                    Data(root)

    def test_invalid_public_inputs_fail_explicitly(self):
        for origins in ([], [0], [-1], [1.5], [True], [[144]], [52560]):
            with self.subTest(origins=origins), self.assertRaises(ValueError):
                self.data.features(origins, 1)
        for cutoff in (0, 145, 1.5, True):
            with self.subTest(cutoff=cutoff), self.assertRaises(ValueError):
                self.data.features([144], cutoff)
        with self.assertRaises(ValueError):
            self.data.baseline(144, "unknown")
        for arguments in ({"month": 0}, {"month": 2, "start": "2025-02-01"},
                          {"start": "2025-03-01", "end": "2025-02-01"},
                          {"start": "2024-12-31"}, {"start": "2025-02-01 00:10"},
                          {"start": "2025-02-01T00:00:00Z"}):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                midnight_origins(**arguments)


if __name__ == "__main__":
    unittest.main()
