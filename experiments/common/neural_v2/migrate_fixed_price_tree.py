"""One-time, explicitly checked reuse of caches outside the fixed-price tree fix."""

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
    archive = ROOT / ".work/exp002/fixed-price-tree-before"
    before_path = archive / "scenarios.py"
    before, after = before_path.read_text(), (HERE / "scenarios.py").read_text()
    old = """            visible=np.concatenate((residual[indices,:boundary].reshape(count,-1),
                                    innovations.reshape(count,-1)),axis=1)"""
    new = """            reveal_channels=3 if variable else 2
            visible=np.concatenate((residual[indices,:boundary,:reveal_channels].reshape(count,-1),
                                    innovations[:,:,:reveal_channels].reshape(count,-1)),axis=1)"""
    assert before.count(old) == 1 and before.replace(old, new) == after
    data = Data()
    old_signature = json.loads((archive / "old_signature.json").read_text())["signature"]
    new_signature = evaluation_signature(data, "exp002")
    spec = importlib.util.spec_from_file_location("experiments.common.neural_v2.before_tree_fix", before_path)
    previous = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(previous)
    directory = HERE / "runs/exp002"
    configurations = []
    checked = 0
    for path in sorted((directory / "evaluation").glob("*/configuration.json")):
        config = json.loads(path.read_text())
        assert config["signature"] == old_signature
        job = config["case"]
        affected = job["scenario"] == "3" and job["method"] == "risk" and job["anticipate"] and bool(job["update_hours"])
        files = sorted(path.parent.glob("d*.json"))
        state = None
        for file in files:
            value = json.loads(file.read_text())["summary"]
            if state is not None:
                assert abs(value["initial_soc"]-state) < 1e-6
            state = value["final_soc"]
            with np.load(file.with_suffix(".npz")) as arrays:
                assert arrays["states"].shape == (145,)
        if not affected and job["method"] == "risk":
            store = ForecastStore(data, "exp002", job["seed"], job["kind"])
            left, right = previous.ScenarioFactory(data, store), ScenarioFactory(data, store)
            for month in range(2, 13):
                origin = int(month_origins(month)[0])
                prediction = store.get(origin, issued=job["scenario"] in ("3", "4-3"), raw_pv=job["raw_pv"])
                args = (origin, prediction, job["scenario"], job["update_hours"], job["raw_pv"], job["anticipate"])
                a, b = left.build(*args), right.build(*args)
                for key in ("paths", "groups", "probabilities"):
                    np.testing.assert_array_equal(a[key], b[key])
                assert a["metadata"] == b["metadata"]
                checked += 1
        configurations.append((path, config, affected, len(files)))
    # No cache is changed until source equivalence and all saved day files pass.
    retired = archive / "retired-evaluation"
    retired.mkdir()
    records = []
    for path, config, affected, count in configurations:
        records.append({"cache_key": path.parent.name, "case": config["case"],
                        "saved_days": count, "action": "retired_and_recompute" if affected else "verified_equivalent_reuse"})
        if affected:
            shutil.move(str(path.parent), retired / path.parent.name)
        else:
            config["signature"] = new_signature
            path.write_text(json.dumps(config, indent=2))
    for path in (directory / "warmup").glob("*.json"):
        record = json.loads(path.read_text())
        assert record["signature"] == old_signature
        record["signature"] = new_signature
        path.write_text(json.dumps(record, indent=2))
    retired_calibration_seconds = 0
    for path in (directory / "calibration").glob("*.json"):
        record = json.loads(path.read_text())
        assert record["signature"] == old_signature
        if record["scenario"] == "3":
            retired_calibration_seconds = sum(d["solve_execute_seconds"] for c in record["candidates"] for d in c["days"])
            shutil.move(str(path), archive / "retired-calibration-3.json")
        else:
            record["signature"] = new_signature
            path.write_text(json.dumps(record, indent=2))
    out = ROOT / "data/results/exp002"
    for name in ("risk_calibration.json", "dispatch_3.npz", "result3.xlsx"):
        if (out / name).exists():
            shutil.move(str(out / name), archive / name)
    report = {"old_signature": old_signature, "new_signature": new_signature,
        "old_source_sha256": digest(before_path), "new_source_sha256": digest(HERE / "scenarios.py"),
        "checked_source_change": "Only fixed-price branching excludes the price channel; Q2 has no branches and Q4 retains all three channels",
        "identical_unaffected_tree_checks": checked, "all_saved_days_readable_and_continuous": True,
        "calibration_3": "discarded; rerun all three January weights before replay", "formal_training_groups": 33,
        "retired_calibration_task_seconds": retired_calibration_seconds,
        "cases": records}
    (out / "causal_tree_fix_migration.json").write_text(json.dumps(report, indent=2))
    print("MIGRATED", sum(not r[2] for r in configurations), "unaffected cases; RETIRED", sum(r[2] for r in configurations))


if __name__ == "__main__":
    run()
