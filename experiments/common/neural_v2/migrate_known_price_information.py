"""Keep ordinary cases while correcting the optional known-price counterfactual tree."""

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
    archive = ROOT / ".work/exp002/known-price-information-before"
    old_scenario = (archive / "scenarios.py").read_text()
    expected = old_scenario.replace("raw_pv=False,anticipate=True):", "raw_pv=False,anticipate=True,known_price=False):")
    expected = expected.replace("reveal_channels=3 if variable else 2", "reveal_channels=3 if variable and not known_price else 2")
    assert expected == (HERE / "scenarios.py").read_text()
    old_call = "bundle=factory.build(o,forecast,scenario,update_hours,raw_pv,anticipate)"
    new_call = "bundle=factory.build(o,forecast,scenario,update_hours,raw_pv,anticipate,known_price=known_price)"
    assert (archive / "evaluate.py").read_text().replace(old_call, new_call) == (HERE / "evaluate.py").read_text()
    spec = importlib.util.spec_from_file_location("experiments.common.neural_v2.before_known_price_fix", archive / "scenarios.py")
    previous = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(previous)
    directory = HERE / "runs/exp002"
    configs = sorted((directory / "evaluation").glob("*/configuration.json"))
    old_signature = json.loads(configs[0].read_text())["signature"]
    data = Data()
    signature = evaluation_signature(data, "exp002")
    store = ForecastStore(data, "exp002", 42)
    left, right = previous.ScenarioFactory(data, store), ScenarioFactory(data, store)
    for scenario in ("2", "3", "4-2", "4-3"):
        for month in range(2, 13):
            origin = int(month_origins(month)[0])
            forecast = store.get(origin, issued=scenario in ("3", "4-3"))
            a, b = left.build(origin, forecast, scenario), right.build(origin, forecast, scenario)
            for key in ("paths", "groups", "probabilities"):
                np.testing.assert_array_equal(a[key], b[key])
            assert a["metadata"] == b["metadata"]
    retired = archive / "retired-evaluation"
    retired.mkdir()
    records = []
    for path in configs:
        config = json.loads(path.read_text())
        assert config["signature"] == old_signature
        job = config["case"]
        affected = job["known_price"] and job["scenario"] == "4-3" and job["method"] == "risk"
        if affected:
            shutil.move(str(path.parent), retired / path.parent.name)
        else:
            config["signature"] = signature
            path.write_text(json.dumps(config, indent=2))
        records.append({"cache_key": path.parent.name, "action": "retired" if affected else "verified_equivalent_reuse"})
    for folder in ("warmup", "calibration"):
        for path in (directory / folder).glob("*.json"):
            record = json.loads(path.read_text())
            assert record["signature"] == old_signature
            record["signature"] = signature
            path.write_text(json.dumps(record, indent=2))
    result = {"old_signature": old_signature, "new_signature": signature,
              "ordinary_case_monthly_equivalence_checks": 44,
              "old_source_sha256": {n: digest(archive / n) for n in ("evaluate.py", "scenarios.py")},
              "new_source_sha256": {n: digest(HERE / n) for n in ("evaluate.py", "scenarios.py")},
              "scope": "Only known-price counterfactual excludes price from information branching; no ordinary policy changes",
              "cases": records}
    (ROOT / "data/results/exp002/known_price_information_migration.json").write_text(json.dumps(result, indent=2))
    print("PRESERVED ordinary cases; corrected optional known-price information boundary", flush=True)


if __name__ == "__main__":
    run()
