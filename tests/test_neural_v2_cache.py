import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from experiments.common.neural_v2 import manifest
from experiments.common.neural_v2.evaluate import cases


class CacheIdentityTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.root = root
        self.engine = root / "engine"
        self.run_dir = self.engine / "runs/check"
        self.run_dir.mkdir(parents=True)
        legacy = root / "data/results/exp001"
        legacy.mkdir(parents=True)
        (legacy / "ensemble_predictions.npz").write_bytes(b"historical-archive")
        self.selection = legacy / "model_selection.csv"
        self.selection.write_text("month,model\n2,original\n")
        for month in range(2, 13):
            stem = self.run_dir / f"m{month:02d}_mlp_42"
            stem.with_suffix(".json").write_text(json.dumps({"signature": "train-test"}))
            stem.with_suffix(".keras").write_bytes(b"saved-weights")
            np.savez(stem.with_suffix(".npz"), predictions=np.array([1., 2.]))
        for name, value in (("ROOT", root), ("HERE", self.engine), ("SEEDS", (42,))):
            self.stack.enter_context(patch.object(manifest, name, value))
        self.stack.enter_context(patch.object(manifest, "Data", return_value=SimpleNamespace(hashes={})))
        self.stack.enter_context(patch.object(manifest, "signature", return_value="train-test"))
        manifest.validate_upstream("check")

    def test_changed_historical_selection_and_weights_are_rejected(self):
        self.selection.write_text("month,model\n2,changed\n")
        with self.assertRaisesRegex(RuntimeError, "Historical forecasts"):
            manifest.validate_upstream("check")
        self.selection.write_text("month,model\n2,original\n")
        (self.run_dir / "m02_mlp_42.keras").write_bytes(b"changed-weights")
        with self.assertRaisesRegex(RuntimeError, "Actual weights"):
            manifest.validate_upstream("check")

    def test_changed_risk_calibration_requires_new_replay(self):
        out = self.root / "data/results/check"
        out.mkdir()
        weights = {"2": .1, "3": 0, "4-2": 0, "4-3": 0}
        calibration = [{"scenario": key, "selected_weight": value} for key, value in weights.items()]
        (out / "risk_calibration.json").write_text(json.dumps(calibration))
        (out / "evaluation_manifest.json").write_text(json.dumps(
            [{"signature": "replay-test", "job": job} for job in cases(weights)]))
        with patch("experiments.common.neural_v2.evaluate.evaluation_signature", return_value="replay-test"):
            manifest.validate_upstream("check", replay=True)
            calibration[0]["selected_weight"] = .3
            (out / "risk_calibration.json").write_text(json.dumps(calibration))
            with self.assertRaisesRegex(RuntimeError, "Calibration choices"):
                manifest.validate_upstream("check", replay=True)


if __name__ == "__main__":
    unittest.main()
