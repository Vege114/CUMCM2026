"""Explicitly preserve only replay prefixes with identical corrected information trees."""

import importlib.util
import json
import shutil

import numpy as np

from .data import HERE, ROOT, Data, month_origins
from .evaluate import evaluation_signature
from .manifest import digest
from .predict import ForecastStore
from .scenarios import ScenarioFactory


def run():
    archive = ROOT / ".work/exp002/clipped-information-before"
    before_path = archive / "scenarios.py"
    before, after = before_path.read_text(), (HERE / "scenarios.py").read_text()
    old = "residual[indices,:boundary,:reveal_channels]"
    new = "paths[:,:boundary,:reveal_channels]"
    assert before.count(old) == 1 and before.replace(old, new) == after
    for path in archive.glob("*"):
        if path.is_file() and path.name != "scenarios.py":
            assert path.read_bytes() == (HERE / path.name).read_bytes()
    spec = importlib.util.spec_from_file_location("experiments.common.neural_v2.before_clip_fix", before_path)
    previous = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(previous)
    directory = HERE / "runs/exp002"
    configs = sorted((directory / "evaluation").glob("*/configuration.json"))
    old_signature = json.loads(configs[0].read_text())["signature"]
    data = Data()
    new_signature = evaluation_signature(data, "exp002")
    checked = 0

    def equal(left, right, store, origin, job, validation=False):
        nonlocal checked
        forecast = store.get(origin, issued=job["scenario"] in ("3", "4-3"),
                             raw_pv=job["raw_pv"], validation_month=2 if validation else None)
        args = (origin, forecast, job["scenario"], job["update_hours"], job["raw_pv"], job["anticipate"])
        a, b = left.build(*args), right.build(*args)
        checked += 1
        return all(np.array_equal(a[key], b[key]) for key in ("paths", "groups", "probabilities")) and a["metadata"] == b["metadata"]

    operations = []
    for path in configs:
        config = json.loads(path.read_text())
        assert config["signature"] == old_signature
        job = config["case"]
        files = sorted(path.parent.glob("d*.json"))
        state = None
        for i, file in enumerate(files):
            summary = json.loads(file.read_text())["summary"]
            assert summary["day"] == 31+i
            if state is not None:
                assert abs(summary["initial_soc"]-state) < 1e-6
            state = summary["final_soc"]
            with np.load(file.with_suffix(".npz")) as arrays:
                assert arrays["states"].shape == (145,)
        keep = len(files)
        if job["method"] == "risk":
            store = ForecastStore(data, "exp002", job["seed"], job["kind"])
            left, right = previous.ScenarioFactory(data, store), ScenarioFactory(data, store)
            affected = job["scenario"] in ("3", "4-3") and job["anticipate"] and bool(job["update_hours"])
            if affected:
                for i, file in enumerate(files):
                    day = int(file.stem[1:])
                    if not all(equal(left, right, store, day*144+hour*6, job) for hour in [0, *job["update_hours"]]):
                        keep = i
                        break
            else:
                for month in range(2, 13):
                    assert equal(left, right, store, int(month_origins(month)[0]), job)
        operations.append((path, config, files, keep))

    calibrations = []
    for path in (directory / "calibration").glob("*.json"):
        record = json.loads(path.read_text())
        assert record["signature"] == old_signature
        unchanged = True
        if record["scenario"] in ("3", "4-3"):
            store = ForecastStore(data, "exp002", 42)
            left, right = previous.ScenarioFactory(data, store), ScenarioFactory(data, store)
            job = {"scenario": record["scenario"], "update_hours": [6, 12, 18], "raw_pv": False, "anticipate": True}
            unchanged = all(equal(left, right, store, day*144+hour*6, job, validation=True)
                            for day in range(24, 31) for hour in (0, 6, 12, 18))
        calibrations.append((path, record, unchanged))

    # Mutations start only after every source, cache and claimed-equivalent prefix passed.
    retired = archive / "retired-evaluation"
    retired.mkdir()
    records = []
    for path, config, files, keep in operations:
        if keep < len(files):
            target = retired / path.parent.name
            target.mkdir()
            for file in files[keep:]:
                shutil.move(str(file), target / file.name)
                shutil.move(str(file.with_suffix(".npz")), target / file.with_suffix(".npz").name)
        config["signature"] = new_signature
        path.write_text(json.dumps(config, indent=2))
        records.append({"cache_key": path.parent.name, "case": config["case"],
                        "kept_equivalent_days": keep, "retired_days": len(files)-keep})
    for path in (directory / "warmup").glob("*.json"):
        record = json.loads(path.read_text())
        assert record["signature"] == old_signature
        record["signature"] = new_signature
        path.write_text(json.dumps(record, indent=2))
    retired_calibration_seconds = 0
    calibration_actions = {}
    for path, record, unchanged in calibrations:
        calibration_actions[record["scenario"]] = "verified_identical_reuse" if unchanged else "retired_and_recalibrate"
        if unchanged:
            record["signature"] = new_signature
            path.write_text(json.dumps(record, indent=2))
        else:
            retired_calibration_seconds += sum(d["solve_execute_seconds"] for c in record["candidates"] for d in c["days"])
            shutil.move(str(path), archive / f"retired-calibration-{record['scenario']}.json")
    out = ROOT / "data/results/exp002"
    shutil.move(str(out / "risk_calibration.json"), archive / "risk_calibration.json")
    result = {"old_signature": old_signature, "new_signature": new_signature,
              "old_source_sha256": digest(before_path), "new_source_sha256": digest(HERE / "scenarios.py"),
              "checked_source_change": "Observed tree prefixes use physically clipped paths rather than latent pre-clipping residuals",
              "identical_tree_checks_including_first_mismatch": checked,
              "all_saved_days_readable_and_continuous": True,
              "retired_calibration_task_seconds": retired_calibration_seconds,
              "calibration": calibration_actions, "cases": records}
    (out / "clipped_information_fix_migration.json").write_text(json.dumps(result, indent=2))
    print("KEPT", sum(r["kept_equivalent_days"] for r in records), "days; RETIRED", sum(r["retired_days"] for r in records), "days", flush=True)
    print("CALIBRATION", calibration_actions, flush=True)


if __name__ == "__main__":
    run()
