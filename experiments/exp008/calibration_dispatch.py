"""Bounded, reproducible calibrated-CNN / controller development runs."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.exp008.controller_candidate import evaluate, OUT
from experiments.exp008.forecast_calibration import CalibratedStore
from experiments.problem2.exp003.data import Data
from experiments.problem2.tree_planning.risk import TreeResidualScenarios
from experiments.problem2.tree_planning.verify import verify_arrays


def inputs(calibration, data):
    store = CalibratedStore(calibration)
    digest = hashlib.sha256(store.values.tobytes()).hexdigest()
    risk = TreeResidualScenarios(store.origins, store.values, data.actual, data.fixed_price)
    supports, audits = [], []
    for day in range(31, 365):
        support, audit = risk.for_day(day)
        audit.update(forecast_calibration=calibration, calibration_values_sha256=digest,
                     residual_source=("calibrated_cnn_prequential_error" if not audit["fallback"]
                                      else audit["residual_source"]))
        supports.append(support)
        audits.append(audit)
    directory = OUT / "calibration_risk"
    directory.mkdir(exist_ok=True)
    np.savez_compressed(directory / f"{calibration}.npz", supports=np.stack(supports), origins=store.origins)
    (directory / f"{calibration}.json").write_text(json.dumps({"forecast_values_sha256": digest,
        "base_cache_used": False, "days": audits}, indent=2) + "\n")
    return np.stack(supports), audits


def verify_own(spec, data):
    directory = OUT / spec["id"]
    with np.load(directory / "dispatch_2.npz") as archive:
        arrays = {k: archive[k].copy() for k in archive.files}
    if spec.get("planner") == "joint":
        arrays = {k: v for k, v in arrays.items() if not k.startswith("intended_")}
    daily = pd.read_csv(directory / "daily.csv").to_dict("records")
    audits = json.loads((directory / "planning_audit.json").read_text())["days"]
    report = verify_arrays(arrays, daily, audits,
        source_actual=data.actual[31 * 144:].reshape(334, 144, 2), source_price=data.fixed_price)
    (directory / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise RuntimeError(report["errors"])


def run(calibration, batch="first"):
    data = Data()
    supports, audits = inputs(calibration, data)
    if batch == "first":
        specifications = [
            {"id": f"cal_{calibration}_greedy_q080", "calibration": calibration,
             "quantile": .8, "controller": "greedy"},
            {"id": f"cal_{calibration}_blocks_q085_s60", "calibration": calibration,
             "quantile": .85, "controller": "fixed_blocks", "solar_start": 60},
        ]
    else:
        specifications = json.loads(Path(batch).read_text())
        if any(s.get("calibration") != calibration or not s["id"].startswith("cal_")
               for s in specifications):
            raise ValueError("extra batch must use this calibration and cal_ ids")
    results = []
    for spec in specifications:
        result = evaluate(spec, data, supports, audits)
        verify_own(spec, data)
        results.append({"id": spec["id"], **{k: result[k] for k in
            ("total_cost", "improvement_vs_exp006_pct", "direction_reversals", "active_slots",
             "throughput_kwh", "emergency_cost", "planned_cost")}, "verified": True})
    print(json.dumps({"calibration": calibration, "results": results}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--batch", default="first")
    args = parser.parse_args()
    run(args.calibration, args.batch)
