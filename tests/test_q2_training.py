"""Q2 training boundaries, cost-mixing semantics and artifact integrity."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from experiments.problem2.exp003.data import STEPS, Data
from experiments.problem2.exp003.predict import (
    ForecastStore,
    apply_residual,
    content_hash,
    file_hash,
    historical_daylight,
    training_fingerprint,
)
from experiments.problem2.exp003.train import build_model, prepare_month, split_origins


class Q2TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = Data()

    def test_training_four_issues_and_midnight_validation_are_purged(self):
        for month in range(2, 13):
            train, validation, cutoff = split_origins(month)
            np.testing.assert_array_equal(np.unique(train % STEPS), [0, 36, 72, 108])
            np.testing.assert_array_equal(validation % STEPS, 0)
            self.assertEqual(len(validation), 7)
            self.assertEqual(train[0], 7 * STEPS)
            self.assertLessEqual(train[-1] + STEPS, cutoff)
            self.assertEqual(validation[0], cutoff)
            self.assertEqual(validation[-1] + STEPS, cutoff + 7 * STEPS)

    def test_formal_targets_cannot_change_fit_inputs(self):
        changed = copy.deepcopy(self.data)
        changed.actual[31 * STEPS:] *= 100
        left = prepare_month(self.data, 2)
        right = prepare_month(changed, 2)
        n = len(left["train"]) + len(left["validation"])
        np.testing.assert_array_equal(left["x"][:n], right["x"][:n])
        for key in ("target", "mean", "scale", "train", "validation"):
            np.testing.assert_array_equal(left[key], right[key])
        self.assertEqual(left["target"].shape, (72, 144, 2))
        with patch.object(self.data, "labels", wraps=self.data.labels) as labels:
            prepare_month(self.data, 2)
            self.assertEqual(labels.call_count, 2)
            for call in labels.call_args_list:
                self.assertLessEqual(max(call.args[0]) + STEPS, call.args[1])
                self.assertLessEqual(call.args[1], 31 * STEPS)

    def test_mixing_uses_unclipped_raw_residual_and_independent_coefficients(self):
        base = np.array([[2., 4.], [2., 4.]])
        residual = np.array([[-6., -6.], [4., 8.]])
        mask = np.array([True, False])
        np.testing.assert_array_equal(apply_residual(base, residual, mask, (0, 0)), [[2, 4], [2, 0]])
        np.testing.assert_array_equal(apply_residual(base, residual, mask, (.5, .5)), [[0, 1], [4, 0]])
        np.testing.assert_array_equal(apply_residual(base, residual, mask, (0, 1)), [[2, 0], [2, 0]])
        for alpha in ((-1, 1), (np.nan, 1), (1,), 1):
            with self.assertRaises(ValueError):
                apply_residual(base, residual, mask, alpha)

    def test_daylight_mask_uses_only_last_28_completed_days(self):
        data = copy.deepcopy(self.data)
        origin = 60 * STEPS
        data.actual[:, 1] = 0
        data.actual[origin - 28 * STEPS + 8, 1] = 1
        data.actual[origin - 29 * STEPS + 9, 1] = 1
        data.actual[origin + 10, 1] = 1
        mask = historical_daylight(data, [origin])
        self.assertEqual(np.flatnonzero(mask[0]).tolist(), [8])
        data.actual[origin:, 1] *= 1000
        np.testing.assert_array_equal(mask, historical_daylight(data, [origin]))

    def test_two_branch_model_has_zero_residual_and_protocol_parameter_count(self):
        model = build_model()
        self.assertEqual(model.count_params(), 2178)
        x = np.random.default_rng(42).normal(size=(2, 144, 2, 16)).astype("float32")
        np.testing.assert_array_equal(model(x, training=False).numpy(), np.zeros((2, 144, 2)))

    def test_january_store_defaults_to_historical_baseline(self):
        store = ForecastStore(self.data)
        with patch.object(store, "month", side_effect=AssertionError("January may not read a future checkpoint")):
            np.testing.assert_array_equal(store.get(24 * STEPS, alpha=(.5, .5)), self.data.baseline(24 * STEPS))

    def test_checkpoint_hashes_reject_mutations_and_config_changes(self):
        signature, sources = training_fingerprint(self.data)
        with tempfile.TemporaryDirectory() as directory:
            stem = Path(directory) / "m02_mlp_42"
            stem.with_suffix(".keras").write_bytes(b"test-only immutable weights")
            arrays = {"origins": np.array([31 * STEPS]), "base": np.ones((1, 144, 2), "float32"),
                      "raw_residual": np.zeros((1, 144, 2), "float32"), "daylight_mask": np.ones((1, 144), bool),
                      "predictions": np.ones((1, 144, 2), "float32"), "signature": np.asarray(signature)}
            np.savez_compressed(stem.with_suffix(".npz"), **arrays)
            meta = {"signature": signature, "asof": 31 * STEPS, "weights_sha256": file_hash(stem.with_suffix(".keras")),
                    "npz_sha256": file_hash(stem.with_suffix(".npz")), "prediction_content_sha256": content_hash(arrays)}
            stem.with_suffix(".json").write_text(json.dumps(meta))
            store = ForecastStore(self.data)
            store.directory = Path(directory)
            self.assertEqual(store.get(31 * STEPS).shape, (144, 2))
            for suffix in (".keras", ".npz"):
                path = stem.with_suffix(suffix)
                original = path.read_bytes()
                path.write_bytes(original + b"tampered")
                store.month.cache_clear()
                with self.assertRaisesRegex(RuntimeError, "content hash"):
                    store.month(2)
                path.write_bytes(original)
            store.signature = "changed protocol or code"
            store.month.cache_clear()
            with self.assertRaisesRegex(RuntimeError, "signature"):
                store.month(2)
        self.assertEqual(set(sources["data_hashes"]), set(self.data.hashes))


if __name__ == "__main__":
    unittest.main()
