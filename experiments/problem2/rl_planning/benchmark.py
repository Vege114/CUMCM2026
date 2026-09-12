"""Strictly bounded speed pilot; contains no annual/formal execution switch."""
# ruff: noqa: E402 -- record scientific and neural library startup explicitly.
import time

IMPORT_START = time.perf_counter()

import argparse
import hashlib
import json
import platform
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

from experiments.problem2.exp003.data import Data
from experiments.problem2.exp004.predict import ForecastStore

from .environment import (
    ACTION_DIM,
    OBS_DIM,
    CausalDayCache,
    RewardConfig,
    collect_rollout,
    plan_day,
    score_plans,
)
from .ppo import PPO, generalized_advantage

IMPORT_SECONDS = time.perf_counter()-IMPORT_START
ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "data/results/exp007/pilot"


def hash_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(repeats=3):
    if not 1 <= repeats <= 3:
        raise ValueError("Speed-only pilot allows 1..3 measured PPO updates per batch size")
    if (OUT / "benchmark.json").exists():
        raise RuntimeError("Pilot already archived; use a fresh worktree to repeat it")
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    data = Data()
    store = ForecastStore("no_season", seed=42)
    cache = CausalDayCache(data, store)
    setup_seconds = time.perf_counter()-started
    # Initial and mature causal windows, never fit on their issue day's actuals.
    cutoffs = [31, 243]
    builds, training_sets = [], {}
    for cutoff in cutoffs:
        began = time.perf_counter()
        training_sets[cutoff] = cache.training_days(cutoff)
        builds.append({"cutoff_day": cutoff, "days": len(training_sets[cutoff]),
                       "seconds": time.perf_counter()-began,
                       "max_observed_index": cutoff*144-1})
    timings, inference, audits, checks = [], [], [], []
    rng = np.random.default_rng(42)
    config = RewardConfig()
    with threadpool_limits(1):
        agent_start = time.perf_counter()
        agent = PPO(OBS_DIM, ACTION_DIM, seed=42)
        initialization_seconds = time.perf_counter()-agent_start
        for n_envs in (16, 32):
            for repeat in range(repeats+1):
                cutoff = cutoffs[min(repeat, 1)]
                began = time.perf_counter()
                rollout, details = collect_rollout(
                    agent, training_sets[cutoff], data.fixed_price, rng, n_envs=n_envs,
                )
                rollout_seconds = time.perf_counter()-began
                advantages, returns = generalized_advantage(
                    rollout["rewards"], rollout["values"], gamma=1.0, lam=.95,
                )
                batch = {"obs": rollout["obs"].reshape(-1, OBS_DIM),
                         "actions": rollout["actions"].ravel(),
                         "old_logprob": rollout["old_logprob"].ravel(),
                         "advantages": advantages.ravel(), "returns": returns.ravel()}
                began = time.perf_counter()
                metrics = agent.update(batch, epochs=4, minibatch_size=512)
                update_seconds = time.perf_counter()-began
                row = {"n_envs": n_envs, "repeat": repeat, "warmup": repeat == 0,
                       "cutoff_day": cutoff, "transitions": n_envs*144,
                       "rollout_seconds": rollout_seconds, "update_seconds": update_seconds,
                       "iteration_seconds": rollout_seconds+update_seconds,
                       "transitions_per_second": n_envs*144/(rollout_seconds+update_seconds),
                       **metrics}
                timings.append(row)
                checks.append({"simultaneous_slots": sum(int(np.sum(
                    (d["charge"] > 1e-6) & (d["discharge"] > 1e-6))) for d in details),
                    "finite_training_batch": all(np.isfinite(x).all() for x in batch.values()),
                    "finite_update": all(np.isfinite(x) for x in metrics.values())})
                print(json.dumps({"stage": "ppo_pilot", **row}), flush=True)
        # This speed-only model has seen completed data up to August 31. It is
        # NOT a causal February evaluation policy. No evaluation cost is ranked.
        for day in [31, 73, 171, 265, 354]:
            began = time.perf_counter()
            forecast, supports, audit = cache.issued(day)
            risk_seconds = time.perf_counter()-began
            audits.append(audit)
            for repeat in range(repeats):
                began = time.perf_counter()
                plan = plan_day(agent, forecast, supports, data.fixed_price, 6000)
                planning_seconds = time.perf_counter()-began
                began = time.perf_counter()
                _, details = score_plans([plan], [data.actual[day*144:(day+1)*144]],
                                         data.fixed_price)
                execution_seconds = time.perf_counter()-began
                detail = details[0]
                balance = (detail["original"] + (detail["actual"][:, 1]
                           - detail["actual"][:, 0])/6 + detail["discharge"]
                           + detail["emergency"] - detail["charge"] - detail["surplus"])
                inference.append({"day": day, "repeat": repeat, "risk_seconds": risk_seconds,
                                  "planning_seconds": planning_seconds,
                                  "execution_seconds": execution_seconds,
                                  "max_balance_error_kwh": float(np.abs(balance).max())})
                if repeat == 0:
                    np.savez_compressed(OUT / f"speed_only_day{day:03}.npz", **detail)
        pools = threadpool_info()
    estimates = []
    for n_envs in (16, 32):
        samples = np.array([r["iteration_seconds"] for r in timings
                            if r["n_envs"] == n_envs and not r["warmup"]])
        # 1M initial transitions, 23 x 200k subsequent 14-day updates.
        initial_iterations = int(np.ceil(1_000_000/(n_envs*144)))
        online_iterations = int(np.ceil(200_000/(n_envs*144)))
        total_iterations = initial_iterations + 23*online_iterations
        day_time = np.quantile([r["planning_seconds"]+r["execution_seconds"]
                                for r in inference], .95)
        estimates.append({"n_envs": n_envs,
                          "iteration_median_seconds": float(np.median(samples)),
                          "iteration_p95_seconds": float(np.quantile(samples, .95)),
                          "initial_iterations": initial_iterations,
                          "online_iterations_each": online_iterations,
                          "online_updates": 23, "total_iterations_per_run": total_iterations,
                          "transitions_per_run": total_iterations*n_envs*144,
                          "initial_training_seconds": float(initial_iterations*np.median(samples)),
                          "single_run_median_seconds": float(total_iterations*np.median(samples)
                                                             + 334*day_time),
                          "single_run_p95_extrapolation_seconds": float(
                              total_iterations*np.quantile(samples, .95)+334*day_time),
                          "six_runs_median_seconds": float(6*(total_iterations*np.median(samples)
                                                               +334*day_time)),
                          "six_runs_p95_with_1_5_budget_seconds": float(
                              1.5*6*(total_iterations*np.quantile(samples, .95)+334*day_time)),
                          "caveat": "sample extrapolation, not measured full run or confidence interval"})
    files = list(Path(__file__).parent.glob("*.py"))
    payload = {"status": "speed_pilot_only_awaiting_user_confirmation",
               "formal_run_executed": False,
               "created_at": datetime.now(UTC).isoformat(),
               "python": sys.version, "platform": platform.platform(),
               "device": "CPU only; TensorFlow intra/inter-op 1; BLAS 1",
               "import_seconds": IMPORT_SECONDS, "setup_seconds": setup_seconds,
               "network_initialization_seconds": initialization_seconds,
               "pilot_wall_seconds_excluding_import": time.perf_counter()-started,
               "causal_dataset_builds": builds, "timings": timings, "inference": inference,
               "reward": asdict(config), "checks": checks, "estimates": estimates,
               "threadpools": pools, "source_sha256": {str(p.relative_to(ROOT)): hash_file(p)
                                                        for p in files},
               "data_sha256": data.hashes,
               "forecast_sha256": hash_file(ROOT / "data/results/exp004/predictions.npz"),
               "pilot_training_transitions": sum(r["transitions"] for r in timings),
               "speed_model_is_not_formal_policy": True,
               "limitations": ["Tiny update count does not establish PPO convergence.",
                               "Mixed-cutoff speed model discarded; never rank its replay costs.",
                               "No hyperparameters selected using evaluation costs.",
                               "Report/export/QA time is an additional workflow estimate."]}
    (OUT / "benchmark.json").write_text(json.dumps(payload, indent=2, allow_nan=False)+"\n")
    (OUT / "issue_audit.json").write_text(json.dumps(audits, indent=2)+"\n")
    print(json.dumps({"pilot_complete": True, "estimates": estimates}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    main(parser.parse_args().repeats)
