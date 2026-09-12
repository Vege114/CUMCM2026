"""Small, fixed-date speed pilot only; there is deliberately no full-run flag."""
# ruff: noqa: E402 -- intentionally time scientific-library imports separately.

import time

IMPORT_START = time.perf_counter()

import argparse
import hashlib
import json
import os
import platform
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from threadpoolctl import threadpool_info, threadpool_limits

from experiments.problem2.exp003.data import EPOCH, ROOT, Data
from experiments.problem2.exp004.predict import ForecastStore

from .model import Config, execute_plan, plan_day
from .risk import TreeResidualScenarios

IMPORT_SECONDS = time.perf_counter() - IMPORT_START
DATES = ("2025-02-01", "2025-02-15", "2025-03-20", "2025-06-21",
         "2025-08-01", "2025-09-23", "2025-11-01", "2025-12-21")
OUT = ROOT / "data/results/exp006/pilot"


def run(repeats=3):
    if not 1 <= repeats <= 3:
        raise ValueError("The approval-gated pilot supports only 1–3 repeats")
    began = time.perf_counter()
    try:
        os.nice(10)
        priority_info = {"nice_increment_applied": 10}
    except PermissionError:
        priority_info = {"nice_increment_applied": 0,
                         "reason": "sandbox denies changing process priority; one CPU thread retained"}
    OUT.mkdir(parents=True, exist_ok=True)
    loaded = time.perf_counter()
    data = Data()
    store = ForecastStore("no_season", seed=42)
    risk = TreeResidualScenarios(store.origins, store.values, data.actual, data.fixed_price)
    loading_seconds = time.perf_counter() - loaded
    rows = []
    risk_log = []
    with threadpool_limits(limits=1):
        pools = threadpool_info()
        for repeat in range(repeats):
            for date in DATES:
                day = int((pd.Timestamp(date) - EPOCH).days)
                risk_start = time.perf_counter()
                supports, info = risk.for_day(day)
                risk_seconds = time.perf_counter() - risk_start
                risk_log.append({"date": date, "repeat": repeat, **info})
                for grid in (100.0, 50.0, 25.0):
                    config = Config(grid_kwh=grid)
                    # Independent 6000-kWh days are for timing and physical checks only.
                    # They are NOT a continuous annual trajectory or a cost ranking.
                    plan = plan_day(supports, data.fixed_price, 6000.0, config)
                    replay_start = time.perf_counter()
                    detail, metrics = execute_plan(
                        plan, data.actual[day * 144:(day + 1) * 144],
                        data.fixed_price, 6000.0, config,
                    )
                    projected_seconds = time.perf_counter() - replay_start
                    greedy_start = time.perf_counter()
                    _, greedy_metrics = execute_plan(
                        plan, data.actual[day * 144:(day + 1) * 144],
                        data.fixed_price, 6000.0, config, greedy=True,
                    )
                    greedy_seconds = time.perf_counter() - greedy_start
                    rows.append({
                        "date": date, "repeat": repeat, "grid_kwh": grid, "off_grid_check": False,
                        "risk_seconds": risk_seconds, **plan["metadata"],
                        "projected_seconds": projected_seconds, "greedy_seconds": greedy_seconds,
                        "full_day_seconds": risk_seconds + plan["metadata"]["planning_seconds"]
                                            + projected_seconds,
                        "greedy_simultaneous_slots": greedy_metrics["simultaneous_slots"],
                        **metrics,
                    })
                    if repeat == 0:
                        np.savez_compressed(OUT / f"pilot_{date}_grid{int(grid)}.npz", **detail)
                    if repeat == 0 and date in ("2025-06-21", "2025-12-21"):
                        off_grid_plan = plan_day(supports, data.fixed_price, 6042.75, config)
                        replay_start = time.perf_counter()
                        _, off_metrics = execute_plan(
                            off_grid_plan, data.actual[day * 144:(day + 1) * 144],
                            data.fixed_price, 6042.75, config,
                        )
                        replay_seconds = time.perf_counter() - replay_start
                        greedy_start = time.perf_counter()
                        _, off_greedy = execute_plan(
                            off_grid_plan, data.actual[day * 144:(day + 1) * 144],
                            data.fixed_price, 6042.75, config, greedy=True,
                        )
                        rows.append({
                            "date": date, "repeat": repeat, "grid_kwh": grid,
                            "off_grid_check": True, "risk_seconds": risk_seconds,
                            **off_grid_plan["metadata"], "projected_seconds": replay_seconds,
                            "greedy_seconds": time.perf_counter() - greedy_start,
                            "full_day_seconds": risk_seconds + replay_seconds
                                                + off_grid_plan["metadata"]["planning_seconds"],
                            "greedy_simultaneous_slots": off_greedy["simultaneous_slots"],
                            **off_metrics,
                        })
            print(f"pilot repeat {repeat + 1}/{repeats} complete; {len(rows)} measured plans",
                  flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "timings.csv", index=False)
    estimates = []
    for grid, group in frame.groupby("grid_kwh"):
        elapsed = group.full_day_seconds.to_numpy()
        estimates.append({
            "grid_kwh": grid, "sample_plans": len(group),
            "median_day_seconds": float(np.median(elapsed)),
            "p95_day_seconds": float(np.quantile(elapsed, .95)),
            "max_day_seconds": float(elapsed.max()),
            "median_risk_seconds": float(group.risk_seconds.median()),
            "median_dp_seconds": float(group.planning_seconds.median()),
            "median_execution_seconds": float(group.projected_seconds.median()),
            "estimated_334_day_seconds_median": float(334 * np.median(elapsed)),
            "estimated_334_day_seconds_p95": float(334 * np.quantile(elapsed, .95)),
            "estimated_12_run_seconds_p95_with_3x_margin":
                float(12 * 365 * np.quantile(elapsed, .95) * 3),
        })
    signature = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in sorted(Path(__file__).parent.glob("*.py"))}
    payload = {
        "status": "pilot_complete_awaiting_user_confirmation",
        "created_utc": datetime.now(UTC).isoformat(),
        "base_branch": "codex/q2-discussion-seasonality",
        "base_commit": "b42168d5271097762953f38d472d0ef5fe1a908d",
        "branch": "codex/q2-tree-planning", "dates": list(DATES),
        "repeats": repeats, "config_except_grid": asdict(Config()),
        "initial_soc_kwh_regular_pilot": 6000.0,
        "off_grid_checks": {"dates": ["2025-06-21", "2025-12-21"], "initial_soc_kwh": 6042.75},
        "full_run_executed": False, "predictor_retrained": False,
        "lp_working_directory_modified": False, "cpu_threads": 1, "process_priority": priority_info,
        "python": sys.version, "platform": platform.platform(),
        "numpy_version": np.__version__, "sklearn_version": sklearn.__version__,
        "threadpools": pools, "data_hashes": data.hashes,
        "predictions_sha256": hashlib.sha256(
            (ROOT / "data/results/exp004/predictions.npz").read_bytes()).hexdigest(),
        "source_sha256": signature, "import_seconds": IMPORT_SECONDS,
        "data_loading_seconds": loading_seconds, "pilot_wall_seconds": time.perf_counter() - began,
        "estimates": estimates, "physical_checks": {
            "simultaneous_slots": int(frame.simultaneous_slots.sum()),
            "greedy_simultaneous_slots": int(frame.greedy_simultaneous_slots.sum()),
            "max_balance_error": float(frame.max_balance_error.max()),
            "max_state_error": float(frame.max_state_error.max()),
        },
        "limitations": [
            "Eight independent dates with three timing repeats; no annual performance conclusion.",
            "Discretized open-loop objective differs from physically projected recourse costs.",
            "Tree residual supports are marginal distributions, not joint daily scenarios.",
            "The 3x timing margin is a budget allowance, not a statistical confidence interval.",
            "Report authoring, debugging, workbook export and visual QA are additional human-agent time.",
        ],
    }
    (OUT / "benchmark.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    (OUT / "risk_audit.json").write_text(json.dumps(risk_log, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"pilot_wall_seconds": payload["pilot_wall_seconds"],
                      "estimates": estimates, "physical_checks": payload["physical_checks"]},
                     ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    run(parser.parse_args().repeats)
