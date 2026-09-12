"""Use byte-identical exp003 planning/execution to assess changed forecasts."""

import argparse
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from experiments.problem2.exp003.baseline_evidence import metric_row, score_forecast
from experiments.problem2.exp003.data import EPOCH, ROOT, STEPS, Data
from experiments.problem2.exp003.dispatch import aggregate
from experiments.problem2.exp003.evaluate import cached_replay

from .data import HERE, OUT, VARIANTS, protocol, sha256, write_json


def verify_frozen():
    frozen = json.loads((HERE / "frozen_prior_manifest.json").read_text())
    changed = [name for name, expected in frozen["files"].items()
               if sha256(ROOT / name) != expected]
    if changed:
        raise RuntimeError(f"Prior evidence or planning code changed: {changed}")
    return len(frozen["files"])


def one_case(arguments):
    variant, seed, signature, initial = arguments
    data = Data()
    name = f"{variant}_seed_{seed}"
    with np.load(OUT / "predictions.npz") as z:
        values = z[name].copy()
        origins = z["origins"].copy()
    lookup = {int(o): v for o, v in zip(origins, values)}
    began = time.monotonic()
    summaries, detail, solvers = cached_replay(
        data, range(31, 365), lookup.__getitem__, initial,
        HERE / "runs/evaluation" / signature[:16] / name, signature,
    )
    elapsed = time.monotonic() - began
    role = {"name": variant, "seed": seed, "case_id": name, "scenario": "2",
            "exploratory": variant == "oracle_season"}
    path = OUT / f"dispatch_{name}.npz"
    np.savez_compressed(path, **detail)
    daily = [{**row, **role} for row in summaries]
    annual = {**aggregate(summaries), **role, "replay_wall_seconds": elapsed}
    logs = [{**row, **role} for row in solvers]
    print("CASE_DONE", name, annual["total_cost"], flush=True)
    return annual, daily, logs


def run(workers=3):
    began = time.monotonic()
    frozen_count = verify_frozen()
    manifest = json.loads((OUT / "prediction_manifest.json").read_text())
    if not manifest["complete"] or sha256(OUT / "predictions.npz") != manifest["archive_sha256"]:
        raise RuntimeError("Complete unchanged predictions required")
    data, cfg = Data(), protocol()
    sources = {str(path.relative_to(ROOT)): sha256(path) for path in [
        HERE / "evaluate.py", HERE / "protocol.json", HERE / "frozen_prior_manifest.json",
        ROOT / "experiments/problem2/exp003/dispatch.py",
        ROOT / "experiments/problem2/exp003/evaluate.py",
        ROOT / "experiments/common/neural_v2/physics.py",
        ROOT / "data/results/exp003/warmup_2.npz",
    ]}
    evidence = {"prediction_sha256": manifest["archive_sha256"], "sources": sources,
                "data": data.hashes, "workers": workers}
    signature = hashlib.sha256(json.dumps(evidence, sort_keys=True).encode()).hexdigest()
    with np.load(ROOT / "data/results/exp003/warmup_2.npz") as warm:
        initial = float(warm["states"][-1, -1])
    write_json(OUT / "evaluation_manifest.json", {"complete": False, "signature": signature})
    arguments = [(variant, seed, signature, initial) for variant in VARIANTS for seed in cfg["seed_list"]]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(one_case, arguments))
    annual, daily, solvers = [], [], []
    for a, d, s in results:
        annual.append(a)
        daily.extend(d)
        solvers.extend(s)
    pd.DataFrame(annual).to_csv(OUT / "dispatch_metrics.csv", index=False)
    pd.DataFrame(daily).to_csv(OUT / "daily_metrics.csv", index=False)
    pd.DataFrame(solvers).to_csv(OUT / "solver_metrics.csv", index=False)
    scores, leads, seed_summaries = [], [], []
    with np.load(OUT / "predictions.npz") as z:
        origins = z["origins"]
        truth = data.actual[origins[:, None] + np.arange(STEPS)]
        months = np.asarray((EPOCH + pd.to_timedelta(origins * 10, unit="min")).month)
        for variant in VARIANTS:
            for seed in cfg["seed_list"]:
                role = {"name": variant, "seed": seed, "case_id": f"{variant}_seed_{seed}",
                        "exploratory": variant == "oracle_season"}
                p = z[role["case_id"]]
                scores.extend({**row, **role} for row in score_forecast(p, truth, months))
                for start in range(0, STEPS, 24):
                    for k, target in enumerate(("load", "pv")):
                        leads.append({**role, "target": target, "lead_start_hour": start / 6,
                                      "lead_end_hour": (start + 24) / 6,
                                      **metric_row(p[:, start:start + 24, k], truth[:, start:start + 24, k])})
    frame = pd.DataFrame(scores)
    frame[frame.period == "annual"].to_csv(OUT / "forecast_annual.csv", index=False)
    frame[frame.period == "monthly"].to_csv(OUT / "forecast_monthly.csv", index=False)
    pd.DataFrame(leads).to_csv(OUT / "forecast_lead.csv", index=False)
    for variant, group in pd.DataFrame(annual).groupby("name", sort=False):
        for metric in ("total_cost", "planned_cost", "emergency_cost", "emergency_kwh"):
            seed_summaries.append({"name": variant, "metric": metric, "n": len(group),
                                   "mean": group[metric].mean(), "sample_std": group[metric].std(ddof=1),
                                   "min": group[metric].min(), "max": group[metric].max()})
    pd.DataFrame(seed_summaries).to_csv(OUT / "seed_summary.csv", index=False)
    files = [p for p in OUT.glob("*.csv")] + list(OUT.glob("dispatch_*.npz"))
    write_json(OUT / "evaluation_sources.json", evidence)
    write_json(OUT / "evaluation_manifest.json", {
        "complete": True, "signature": signature, "cases": [a["case_id"] for a in annual],
        "days_per_case": 334, "intervals_per_case": 48096, "initial_soc": initial,
        "unchanged_prior_files": frozen_count, "seconds": time.monotonic() - began,
        "workers": workers, "output_hashes": {p.name: sha256(p) for p in files},
        "source_sha256": sha256(OUT / "evaluation_sources.json"),
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=3)
    run(**vars(parser.parse_args()))
