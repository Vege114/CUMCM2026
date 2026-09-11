"""Independent full-year audit of every unique replay, not just four submitted workbooks."""

import json

import numpy as np
import pandas as pd

from .data import HERE, ROOT
from .export import independent_check, read_npz
from .manifest import digest
from .manifest import run as model_manifest


def run(run_id="exp002"):
    out = ROOT / "data/results" / run_id
    manifest = json.loads((out / "evaluation_manifest.json").read_text())
    costs = pd.read_csv(out / "dispatch_metrics.csv", dtype={"scenario": str})
    unique = {row["cache_key"]: row for row in manifest}
    reports = []
    for key, row in unique.items():
        directory = HERE / "runs" / run_id / "evaluation" / key
        files = sorted(directory.glob("d*.json"))
        assert len(files) == 334
        details, calls, summaries = [], 0, []
        for day, path in zip(range(31, 365), files):
            record = json.loads(path.read_text())
            summary = record["summary"]
            assert summary["day"] == day
            summaries.append(summary)
            details.append(read_npz(path.with_suffix(".npz")))
            for log in record["solvers"]:
                assert log["information_cutoff"] == log["origin"]
                assert day*144 <= log["origin"] < (day+1)*144
                assert log.get("source_latest_target", 0) <= log["origin"]
                assert log["issued_forecast_allowed"] == (row["job"]["scenario"] in ("3", "4-3"))
                calls += 1
        d = {name: np.stack([part[name] for part in details]) for name in details[0]}
        check = independent_check(d)
        warmup = read_npz(HERE / "runs" / run_id / "warmup" / f"{row['job']['scenario']}.npz")
        np.testing.assert_allclose(d["states"][0, 0], warmup["states"][-1, -1], atol=1e-6)
        if row["job"]["scenario"] in ("2", "4-2"):
            np.testing.assert_array_equal(d["original"], d["final"])
        np.testing.assert_allclose(sum(s["total_cost"] for s in summaries), check["total_cost"], atol=1e-5)
        registered = costs[costs.cache_key == key]
        np.testing.assert_allclose(registered.total_cost, check["total_cost"], atol=1e-5)
        reports.append({"cache_key": key, "scenario": row["job"]["scenario"], "seed": row["job"]["seed"],
                        "solver_calls": calls, **check})
    model_manifest(run_id)
    hashes = json.loads((out / "data_hashes.json").read_text())
    assert all(digest(ROOT / "data/raw" / name) == value for name, value in hashes.items())
    result = {"status": "passed", "unique_yearly_replays": len(unique), "named_comparisons": len(manifest),
              "total_independently_checked_intervals": len(unique)*334*144,
              "energy_efficiency_mutex_power_soc_continuity": "passed", "four_fee_components": "passed",
              "causal_controller": "passed", "saved_information_cutoffs": "passed",
              "raw_data_hashes": "unchanged", "known_future_price_counterfactual_excluded_from_causality_claim": True,
              "cases": reports}
    (out / "full_year_audit.json").write_text(json.dumps(result, indent=2))
    print("AUDITED", len(unique), "unique full-year replays", flush=True)


if __name__ == "__main__":
    run()
