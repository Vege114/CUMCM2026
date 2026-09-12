"""Approved causal PPO evaluation with atomic, signed 14-day checkpoints."""
# ruff: noqa: E402 -- time TensorFlow/scientific imports as a separate stage.
import time

IMPORT_START = time.perf_counter()

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_info, threadpool_limits

from experiments.problem2.exp003.data import EPOCH, ROOT, Data
from experiments.problem2.exp004.predict import ForecastStore
from experiments.problem2.tree_planning.verify import battery_metrics, verify_arrays

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
HERE = Path(__file__).resolve().parent
OUT = ROOT / "data/results/exp007"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_hash(values):
    return hashlib.sha256(np.asarray(values, dtype="<f8").tobytes()).hexdigest()


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def save_npz(path, detail):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **detail)
    temporary.replace(path)


def stack(details):
    return {key: np.stack([d[key] for d in details]) for key in details[0]}


def restore_chunks(path, run_signature, agent, rng, data, initial):
    details, rows, audits, training = [], [], [], []
    cumulative_wall = 0.0
    for directory in sorted((path / "checkpoints").glob("block_*")):
        meta = json.loads((directory / "manifest.json").read_text())
        if meta["signature"] != run_signature or meta["start_day"] != 31+len(rows):
            raise RuntimeError(f"Incompatible or non-contiguous checkpoint: {directory}")
        for name, expected in meta["sha256"].items():
            if sha256(directory/name) != expected:
                raise RuntimeError(f"Changed checkpoint: {directory/name}")
        with np.load(directory / "dispatch.npz") as z:
            for index in range(len(meta["daily"])):
                details.append({key: z[key][index].copy() for key in z.files})
        rows.extend(meta["daily"])
        audits.extend(meta["audit"])
        training.extend(meta["training"])
        cumulative_wall += meta["block_wall_seconds"]
        last_directory = directory
    if rows:
        checked = verify_arrays(stack(details), rows, audits, expected_days=len(rows),
                                initial_soc=initial["soc"], initial_mode=initial["mode"],
                                initial_power_kw=initial["power_kw"],
                                source_actual=data.actual[31*144:(31+len(rows))*144]
                                .reshape(len(rows), 144, 2), source_price=data.fixed_price)
        if not checked["passed"]:
            raise RuntimeError(f"Checkpoint physics invalid: {checked['errors']}")
        agent.load(last_directory / "policy")
        rng.bit_generator.state = meta["rollout_rng"]
    return details, rows, audits, training, cumulative_wall


def one_case(data, protocol, spec, evidence_signature, initial):
    path = OUT / spec["id"]
    run_signature = signature({"evidence": evidence_signature, "spec": spec})
    completion = path / "completion.json"
    if completion.exists():
        saved = json.loads(completion.read_text())
        if saved["signature"] != run_signature or not saved["complete"]:
            raise RuntimeError("Incompatible completion; never overwrite archived results")
        for name, value in saved["output_sha256"].items():
            if sha256(path/name) != value:
                raise RuntimeError(f"Changed completed evidence {name}")
        print(f"CACHE_DONE {spec['id']}", flush=True)
        return saved["summary"], saved["timing"]
    (path / "checkpoints").mkdir(parents=True, exist_ok=True)
    began = time.perf_counter()
    cfg = RewardConfig(**spec["reward"])
    algorithm, training_config = protocol["algorithm"], protocol["training"]
    agent = PPO(OBS_DIM, ACTION_DIM, seed=spec["rl_seed"],
                hidden=algorithm["hidden_layers"][0], learning_rate=algorithm["learning_rate"])
    rng = np.random.default_rng(spec["rl_seed"])
    cache = CausalDayCache(data, ForecastStore(protocol["forecast"]["variant"],
                                              seed=protocol["forecast"]["seed"]))
    details, rows, audits, training, prior_wall = restore_chunks(
        path, run_signature, agent, rng, data, initial,
    )
    resumed_days = len(rows)
    soc = float(details[-1]["states"][-1]) if rows else initial["soc"]
    mode = int(audits[-1]["execution_final_mode"]) if rows else initial["mode"]
    power = float(6*(details[-1]["charge"][-1]-details[-1]["discharge"][-1])) if rows else initial["power_kw"]
    training_seconds, dataset_seconds, checkpoint_seconds = 0.0, 0.0, 0.0
    for cutoff in range(31+len(rows), 365, training_config["continue_every_days"]):
        block_start = time.perf_counter()
        start_index, training_start = len(rows), len(training)
        feature_start = time.perf_counter()
        examples = cache.training_days(cutoff, training_config["history_days_max"])
        feature_seconds = time.perf_counter()-feature_start
        dataset_seconds += feature_seconds
        iterations = (training_config["initial_iterations"] if cutoff == 31
                      else training_config["online_iterations_each"])
        for iteration in range(iterations):
            step_start = time.perf_counter()
            rollout, train_details = collect_rollout(
                agent, examples, data.fixed_price, rng,
                n_envs=algorithm["parallel_envs"], config=cfg,
            )
            advantages, returns = generalized_advantage(
                rollout["rewards"], rollout["values"], gamma=algorithm["gamma"],
                lam=algorithm["gae_lambda"],
            )
            rollout_seconds = time.perf_counter()-step_start
            update_start = time.perf_counter()
            metrics = agent.update({"obs": rollout["obs"].reshape(-1, OBS_DIM),
                                    "actions": rollout["actions"].ravel(),
                                    "old_logprob": rollout["old_logprob"].ravel(),
                                    "advantages": advantages.ravel(), "returns": returns.ravel()},
                                   epochs=algorithm["epochs_per_update"],
                                   minibatch_size=algorithm["minibatch_size"])
            update_seconds = time.perf_counter()-update_start
            training_seconds += rollout_seconds+update_seconds
            row = {"run_id": spec["id"], "rl_seed": spec["rl_seed"],
                   "cutoff_day": cutoff, "iteration_at_cutoff": iteration,
                   "global_iteration": len(training), "transitions": int(returns.size),
                   "training_start_day": examples[0]["day"],
                   "training_end_day": examples[-1]["day"], "history_days": len(examples),
                   "max_observed_index": cutoff*144-1,
                   "rollout_seconds": rollout_seconds, "update_seconds": update_seconds,
                   "mean_episode_reward": float(rollout["rewards"].sum(axis=0).mean()),
                   "mean_episode_cost_yuan": float(np.mean([d["fees"].sum() for d in train_details])),
                   "mean_episode_throughput_kwh": float(np.mean([
                       (d["charge"]+d["discharge"]).sum() for d in train_details])),
                   **metrics}
            if not all(np.isfinite(v) for v in row.values() if isinstance(v, (float, int))):
                raise FloatingPointError("Non-finite training diagnostics")
            training.append(row)
        for day in range(cutoff, min(365, cutoff+training_config["continue_every_days"])):
            risk_start = time.perf_counter()
            forecast, supports, info = cache.issued(day)
            risk_seconds = time.perf_counter()-risk_start
            plan_start = time.perf_counter()
            plan = plan_day(agent, forecast, supports, data.fixed_price, soc, mode, power, cfg)
            plan_hash = array_hash(plan["purchase"])
            planning_seconds = time.perf_counter()-plan_start
            execution_start = time.perf_counter()
            # First read of CURRENT real day occurs only after the full plan/hash.
            reward, actual_details = score_plans(
                [plan], [data.actual[day*144:(day+1)*144]], data.fixed_price, cfg,
                initial_modes=[mode], initial_powers=[power], final_days=[day == 364],
            )
            detail = actual_details[0]
            if array_hash(detail["original"]) != plan_hash:
                raise RuntimeError("Midnight purchase changed during execution")
            detail.update(intended_charge=plan["charge"].copy(),
                          intended_discharge=plan["discharge"].copy(),
                          intended_states=plan["states"].copy())
            execution_seconds = time.perf_counter()-execution_start
            battery = battery_metrics({k: v[None] for k, v in detail.items()}, mode, power)
            date = EPOCH+pd.Timedelta(days=day)
            daily = {"run_id": spec["id"], "role": spec["role"], "scenario": "2", "day": day,
                     "date": str(date.date()), "month": date.month,
                     "forecast": protocol["forecast"]["variant"], "forecast_seed": 42,
                     "seed": spec["rl_seed"], "rl_seed": spec["rl_seed"],
                     "planned_cost": float(detail["fees"][:, 0].sum()),
                     "up_cost": 0.0, "down_cost": 0.0,
                     "emergency_cost": float(detail["fees"][:, 3].sum()),
                     "total_cost": float(detail["fees"].sum()),
                     "planned_kwh": float(detail["original"].sum()),
                     "final_kwh": float(detail["final"].sum()),
                     "emergency_kwh": float(detail["emergency"].sum()),
                     "emergency_minutes": int(np.sum(detail["emergency"] > 1e-6)*10),
                     "surplus_kwh": float(detail["surplus"].sum()), **battery,
                     "risk_seconds": risk_seconds, "planning_seconds": planning_seconds,
                     "execution_seconds": execution_seconds,
                     "compute_seconds": risk_seconds+planning_seconds+execution_seconds,
                     "shaped_episode_reward": float(reward.sum())}
            audit = {**info, "day": day, "date": daily["date"], "run_id": spec["id"],
                     "forecast_origin": day*144, "forecast_seed": 42, "rl_seed": spec["rl_seed"],
                     "planning_initial_soc": soc, "planning_initial_mode": mode,
                     "planning_initial_power_kw": power,
                     "execution_initial_soc": soc, "execution_initial_mode": mode,
                     "execution_final_mode": battery["final_mode"],
                     "policy_training_cutoff_day": cutoff,
                     "policy_training_max_observed_index": cutoff*144-1,
                     "policy_total_iterations": len(training),
                     "policy_training_origins": [e["day"]*144 for e in examples],
                     "purchase_locked_before_actual_read": True,
                     "midnight_purchase_sha256": plan_hash,
                     "forecast_value_sha256": array_hash(forecast),
                     "actuals_in_policy_observation": False, "last_evaluation_day": day == 364,
                     "terminal_value_yuan_per_soc_kwh": 0.0 if day == 364 else
                     float(data.fixed_price.min()/np.sqrt(.9)),
                     "terminal_value_scope": "reward_only_not_policy_input_or_billed_credit"}
            details.append(detail)
            rows.append(daily)
            audits.append(audit)
            soc, mode, power = battery["final_soc"], battery["final_mode"], battery["final_power_kw"]
        checkpoint_start = time.perf_counter()
        # A directory rename commits policy/RNG + executed block as one unit.
        target = path / "checkpoints" / f"block_{cutoff:03}"
        with tempfile.TemporaryDirectory(prefix=".pending-", dir=path/"checkpoints") as tmp:
            temp = Path(tmp)
            agent.save(temp / "policy")
            save_npz(temp / "dispatch.npz", stack(details[start_index:]))
            files = [p for p in temp.rglob("*") if p.is_file()]
            write_json(temp / "manifest.json", {
                "signature": run_signature, "start_day": cutoff, "end_day": rows[-1]["day"],
                "sha256": {str(p.relative_to(temp)): sha256(p) for p in files},
                "rollout_rng": rng.bit_generator.state,
                "daily": rows[start_index:], "audit": audits[start_index:],
                "training": training[training_start:], "dataset_seconds": feature_seconds,
                "block_wall_seconds": time.perf_counter()-block_start,
            })
            if target.exists():
                raise RuntimeError(f"Refusing to replace committed checkpoint {target}")
            temp.rename(target)
        checkpoint_seconds += time.perf_counter()-checkpoint_start
        print(f"BLOCK_DONE {spec['id']} days={len(rows)}/334 iterations={len(training)} "
              f"elapsed={time.perf_counter()-began:.2f}s", flush=True)
    arrays = stack(details)
    verification = verify_arrays(
        arrays, rows, audits, source_actual=data.actual[31*144:].reshape(334, 144, 2),
        source_price=data.fixed_price,
    )
    if not verification["passed"]:
        raise RuntimeError(f"Independent physical verification failed: {verification['errors']}")
    if (len(training) != training_config["total_iterations_per_run"]
            or sum(r["transitions"] for r in training) != training_config["transitions_per_run"]):
        raise RuntimeError("Training budget differs from approved protocol")
    totals = {k: float(sum(r[k] for r in rows)) for k in (
        "planned_cost", "emergency_cost", "total_cost", "planned_kwh", "final_kwh",
        "emergency_kwh", "emergency_minutes", "surplus_kwh", "up_cost", "down_cost",
        "risk_seconds", "planning_seconds", "execution_seconds", "compute_seconds")}
    summary = {"run_id": spec["id"], "role": spec["role"], "scenario": "2", "days": 334,
               "seed": spec["rl_seed"], "rl_seed": spec["rl_seed"], "forecast_seed": 42,
               "forecast": protocol["forecast"]["variant"], **totals,
               **verification["battery_metrics"], "training_transitions": sum(r["transitions"] for r in training),
               "training_iterations": len(training), "verified": True,
               "terminal_credit_in_reported_cost": 0.0,
               "mean_daily_cost": float(np.mean([r["total_cost"] for r in rows])),
               "max_daily_cost": float(max(r["total_cost"] for r in rows)),
               "training_seconds": float(sum(r["rollout_seconds"]+r["update_seconds"] for r in training))}
    save_npz(path / "dispatch_2.npz", arrays)
    pd.DataFrame(rows).to_csv(path / "daily.csv", index=False)
    pd.DataFrame(training).to_csv(path / "training.csv", index=False)
    write_json(path / "planning_audit.json", {"signature": run_signature, "spec": spec, "days": audits})
    write_json(path / "verification.json", verification)
    timing = {"run_id": spec["id"], "training_seconds": summary["training_seconds"],
              "training_seconds_this_invocation": training_seconds, "dataset_seconds_this_invocation": dataset_seconds,
              "checkpoint_seconds_this_invocation": checkpoint_seconds,
              "prior_completed_block_wall_seconds": prior_wall, "resumed_days": resumed_days,
              "wall_seconds_this_invocation": time.perf_counter()-began,
              **{k: totals[k] for k in ("risk_seconds", "planning_seconds", "execution_seconds")}}
    summary["run_wall_seconds"] = prior_wall+timing["wall_seconds_this_invocation"]
    files = ["dispatch_2.npz", "daily.csv", "training.csv", "planning_audit.json", "verification.json"]
    write_json(completion, {"complete": True, "signature": run_signature,
                            "created_utc": datetime.now(UTC).isoformat(),
                            "summary": summary, "timing": timing,
                            "output_sha256": {name: sha256(path/name) for name in files}})
    print(f"CASE_DONE {spec['id']} cost={summary['total_cost']:.6f} "
          f"wall={summary['run_wall_seconds']:.2f}s", flush=True)
    return summary, timing


def run():
    began = time.perf_counter()
    protocol_path = HERE / "protocol.approved.json"
    protocol = json.loads(protocol_path.read_text())
    if not protocol.get("formal_run_authorized"):
        raise RuntimeError("Explicit user approval is required")
    expected = [r["id"] for r in protocol["runs"]]
    data = Data()
    if data.hashes != protocol["data_sha256"]:
        raise RuntimeError("Raw data changed from approved pilot")
    if sha256(ROOT/protocol["forecast"]["archive"]) != protocol["forecast"]["sha256"]:
        raise RuntimeError("Frozen forecast changed")
    for name, value in protocol["source_sha256"].items():
        if sha256(ROOT/name) != value:
            raise RuntimeError(f"Approved core implementation changed: {name}")
    with np.load(ROOT / "data/results/exp003/warmup_2.npz") as warm:
        net = (warm["charge"]-warm["discharge"]).ravel()
        initial = {"soc": float(warm["states"][-1, -1]),
                   "mode": int(np.sign(net[np.abs(net) > 1e-6][-1])), "power_kw": float(net[-1]*6)}
    evaluation = protocol["evaluation"]
    if (abs(initial["soc"]-evaluation["initial_soc_kwh"]) > 1e-9
            or initial["mode"] != evaluation["initial_mode"]
            or abs(initial["power_kw"]-evaluation["initial_power_kw"]) > 1e-9):
        raise RuntimeError("Common warmup boundary changed")
    paths = [HERE/name for name in ("run.py", "environment.py", "ppo.py", "protocol.approved.json")]
    paths += [ROOT/name for name in ("experiments/problem2/exp003/data.py",
                                    "experiments/problem2/exp004/predict.py",
                                    "experiments/problem2/tree_planning/model.py",
                                    "experiments/problem2/tree_planning/risk.py",
                                    "experiments/problem2/tree_planning/verify.py")]
    evidence = {"source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in paths},
                "data_sha256": data.hashes, "predictions_sha256": protocol["forecast"]["sha256"],
                "warmup_sha256": sha256(ROOT / "data/results/exp003/warmup_2.npz")}
    evidence_signature = signature(evidence)
    manifest_path = OUT / "run_manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous["signature"] != evidence_signature:
            raise RuntimeError("Incompatible experiment manifest; preserve original evidence")
        if previous.get("complete"):
            print("Experiment is already complete; inspect archives without rerunning.", flush=True)
            return
    manifest = {"experiment_id": "exp007", "status": "running", "complete": False,
                "signature": evidence_signature, "evidence": evidence, "initial": initial,
                "started_utc": datetime.now(UTC).isoformat(), "expected_runs": expected,
                "completed_runs": [], "forecast_retrained": False, "cpu_threads": 1,
                "import_seconds": IMPORT_SECONDS, "python": sys.version,
                "platform": platform.platform(), "pid": os.getpid(),
                "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()}
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(manifest_path, manifest)
    summaries, timings = [], []
    try:
        with threadpool_limits(1):
            manifest["threadpools"] = threadpool_info()
            for spec in protocol["runs"]:
                summary, timing = one_case(data, protocol, spec, evidence_signature, initial)
                summaries.append(summary)
                timings.append(timing)
                pd.DataFrame(summaries).to_csv(OUT / "summary.csv", index=False)
                pd.DataFrame(timings).to_csv(OUT / "timings.csv", index=False)
                manifest["completed_runs"].append(spec["id"])
                write_json(manifest_path, manifest)
        manifest.update(status="complete", complete=True, finished_utc=datetime.now(UTC).isoformat(),
                        formal_wall_seconds=time.perf_counter()-began,
                        output_sha256={name: sha256(OUT/name) for name in ("summary.csv", "timings.csv")})
    except Exception as error:
        manifest.update(status="failed", error=repr(error), failed_utc=datetime.now(UTC).isoformat())
        write_json(manifest_path, manifest)
        raise
    write_json(manifest_path, manifest)
    print(json.dumps({"complete": True, "runs": len(summaries),
                      "formal_wall_seconds": manifest["formal_wall_seconds"]}), flush=True)


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    run()
