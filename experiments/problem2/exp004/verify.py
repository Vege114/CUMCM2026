"""Independent physical, archive, history, and prediction boundary checks."""

import json

import numpy as np
import pandas as pd

from experiments.common.neural_v2.export import independent_check
from experiments.problem2.exp003.data import ROOT, STEPS, Data

from .data import OUT, VARIANTS, protocol, sha256, source_signature, write_json
from .evaluate import verify_frozen
from .predict import ForecastStore


def run():
    data, cfg = Data(), protocol()
    manifest = json.loads((OUT / "evaluation_manifest.json").read_text())
    if not manifest["complete"]:
        raise RuntimeError("Full evaluation required")
    for name, expected in manifest["output_hashes"].items():
        if sha256(OUT / name) != expected:
            raise RuntimeError(f"Changed result {name}")
    if sha256(OUT / "evaluation_sources.json") != manifest["source_sha256"]:
        raise RuntimeError("Changed evaluation source manifest")
    sources = json.loads((OUT / "evaluation_sources.json").read_text())
    for name, expected in sources["sources"].items():
        if sha256(ROOT / name) != expected:
            raise RuntimeError(f"Changed evaluated source: {name}")
    signature, _ = source_signature(data)
    prediction_manifest = json.loads((OUT / "prediction_manifest.json").read_text())
    assert signature == prediction_manifest["signature"]
    assert sha256(OUT / "predictions.npz") == prediction_manifest["archive_sha256"]
    assert sha256(OUT / "training_metadata.json") == prediction_manifest["metadata_sha256"]
    metadata = json.loads((OUT / "training_metadata.json").read_text())
    assert len(metadata) == 99
    for m in metadata:
        assert m["signature"] == signature
        assert m["train_latest_label_exclusive"] <= m["scaler_cutoff_day"] * STEPS
        assert m["validation_latest_label_exclusive"] <= m["asof_day"] * STEPS
        assert m["save_reload_passed"]
        assert m["uses_future_seasonal_information"] == (m["variant"] == "oracle_season")
        assert m["sample_weight_min"] > 0
    expected_actual = data.actual[31 * STEPS:].reshape(334, STEPS, 2)
    costs = pd.read_csv(OUT / "dispatch_metrics.csv")
    checks = []
    for variant in VARIANTS:
        for seed in cfg["seed_list"]:
            case = f"{variant}_seed_{seed}"
            with np.load(OUT / f"dispatch_{case}.npz") as z:
                detail = {k: z[k].copy() for k in z.files}
            np.testing.assert_array_equal(detail["actual"], expected_actual)
            np.testing.assert_array_equal(detail["original"], detail["final"])
            np.testing.assert_array_equal(detail["price"], np.tile(data.fixed_price, (334, 1)))
            np.testing.assert_allclose(detail["states"][0, 0], manifest["initial_soc"], atol=1e-6, rtol=0)
            check = independent_check(detail)
            row = costs[costs.case_id == case].iloc[0]
            np.testing.assert_allclose(check["total_cost"], row.total_cost, atol=1e-6, rtol=0)
            for field, metric in (("original", "planned_kwh"), ("emergency", "emergency_kwh"),
                                  ("charge", "charge_kwh"), ("discharge", "discharge_kwh")):
                np.testing.assert_allclose(detail[field].sum(), row[metric], atol=1e-6, rtol=0)
            check.update(case_id=case, same_plan_all_day=True, future_season=variant == "oracle_season")
            checks.append(check)
    store = ForecastStore()
    empty = store.completed_error_paths(31 * STEPS)
    assert empty["errors_kw"].shape == (0, 144, 2)
    paths = store.completed_error_paths(100 * STEPS)
    assert np.all(paths["origins"] + STEPS <= 100 * STEPS)
    assert len(paths["origins"]) == 28
    try:
        ForecastStore("oracle_season")
    except ValueError:
        pass
    else:
        raise AssertionError("Oracle must require explicit opt-in")
    write_json(OUT / "verification.json", {
        "status": "passed", "cases": checks, "intervals_checked": 9 * 334 * STEPS,
        "unchanged_prior_files": verify_frozen(), "training_groups": 99,
        "training_label_boundaries": "passed", "fixed_plan_and_greedy_execution": "passed",
        "forecast_interface_and_completed_error_cutoff": "passed",
        "prediction_sha256": prediction_manifest["archive_sha256"],
        "data_hashes": data.hashes,
    })
    print("VERIFIED 9 cases, 432864 intervals, 99 training groups", flush=True)


if __name__ == "__main__":
    run()
