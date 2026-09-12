"""Independent exp007 archive verification without importing PPO or its runner.

Recompute physics/billing from arrays and check training, forecast, checkpoint,
and continuity evidence against the frozen protocol and original source tables.
Archive assertions describe their scope; they do not alone prove source-code
causality, which is covered by the separate future-mutation environment tests.
"""

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.problem2.tree_planning.verify import verify_arrays

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
TOL = 1e-6


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_hash(values):
    return hashlib.sha256(np.asarray(values, dtype="<f8").tobytes()).hexdigest()


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def read_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {name: z[name].copy() for name in z.files}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def check_training(training, protocol, spec):
    """Validate every update's complete-day cutoff, fixed budget, and diagnostics."""
    errors = []

    def check(condition, message):
        if not bool(condition):
            errors.append(message)

    config, algorithm = protocol["training"], protocol["algorithm"]
    cutoffs = list(range(config["first_cutoff_day"], 365, config["continue_every_days"]))
    expected_rows = sum(config["initial_iterations"] if i == 0 else config["online_iterations_each"]
                        for i in range(len(cutoffs)))
    check(len(cutoffs) == config["online_updates"] + 1, "online_update_count")
    check(len(training) == expected_rows == config["total_iterations_per_run"], "training_row_count")
    required = ("cutoff_day", "iteration_at_cutoff", "global_iteration", "transitions",
                "training_start_day", "training_end_day", "history_days", "max_observed_index",
                "rollout_seconds", "update_seconds", "mean_episode_reward", "mean_episode_cost_yuan",
                "mean_episode_throughput_kwh", "loss", "policy_loss", "value_loss", "entropy",
                "approx_kl", "clip_fraction", "gradient_norm", "rl_seed")
    cursor = 0
    for block, cutoff in enumerate(cutoffs):
        iterations = config["initial_iterations"] if block == 0 else config["online_iterations_each"]
        expected_start = max(config["january_first_training_day"], cutoff - config["history_days_max"])
        for iteration in range(iterations):
            if cursor >= len(training):
                break
            row = training[cursor]
            if not all(key in row for key in required):
                errors.append(f"training_missing_columns:{cursor}")
                cursor += 1
                continue
            check(all(np.isfinite(row[key]) for key in required), f"training_nonfinite:{cursor}")
            for key, expected in (("cutoff_day", cutoff), ("iteration_at_cutoff", iteration),
                                  ("global_iteration", cursor), ("rl_seed", spec["rl_seed"]),
                                  ("transitions", 144 * algorithm["parallel_envs"]),
                                  ("training_start_day", expected_start),
                                  ("training_end_day", cutoff - 1),
                                  ("history_days", cutoff - expected_start),
                                  ("max_observed_index", cutoff * 144 - 1)):
                check(row[key] == expected, f"training_{key}:{cursor}")
            check(row.get("run_id") == spec["id"], f"training_run_id:{cursor}")
            for key in ("rollout_seconds", "update_seconds", "mean_episode_cost_yuan",
                        "mean_episode_throughput_kwh", "value_loss", "entropy", "gradient_norm"):
                check(row[key] >= 0, f"training_negative_{key}:{cursor}")
            check(row["entropy"] <= np.log(algorithm["actions"]) + TOL, f"entropy_bound:{cursor}")
            check(0 <= row["clip_fraction"] <= 1, f"clip_fraction_bound:{cursor}")
            check(row["approx_kl"] >= -TOL, f"approx_kl_bound:{cursor}")
            cursor += 1
    transitions = sum(row.get("transitions", 0) for row in training)
    check(transitions == config["transitions_per_run"], "training_transition_budget")
    return {"passed": not errors, "errors": errors, "cutoff_days": cutoffs,
            "initial_iterations": config["initial_iterations"],
            "online_updates": len(cutoffs) - 1,
            "online_iterations_each": config["online_iterations_each"],
            "iterations": len(training), "transitions": int(transitions),
            "january_training_days": list(range(config["january_first_training_day"], cutoffs[0])),
            "training_seconds": float(sum(r.get("rollout_seconds", 0) + r.get("update_seconds", 0)
                                          for r in training)),
            "causality_scope": "every logged update uses complete historical days before its exclusive cutoff"}


def check_policy_audit(detail, audits, protocol, spec, forecasts, initial_power):
    """Check every midnight lock and policy cutoff against independent arrays."""
    errors = []

    def check(condition, message):
        if not bool(condition):
            errors.append(message)

    config = protocol["training"]
    check(len(audits) == 334, "policy_audit_count")
    power = float(initial_power)
    for index, info in enumerate(audits):
        day = index + 31
        block = (day - 31) // config["continue_every_days"]
        cutoff = 31 + block * config["continue_every_days"]
        iterations = config["initial_iterations"] + block * config["online_iterations_each"]
        start = max(config["january_first_training_day"], cutoff - config["history_days_max"])
        expected = {
            "day": day, "run_id": spec["id"], "rl_seed": spec["rl_seed"],
            "forecast_seed": protocol["forecast"]["seed"],
            "policy_training_cutoff_day": cutoff,
            "policy_training_max_observed_index": cutoff * 144 - 1,
            "policy_total_iterations": iterations,
            "policy_training_origins": list(range(start * 144, cutoff * 144, 144)),
            "midnight_purchase_sha256": array_hash(detail["original"][index]),
            "forecast_value_sha256": array_hash(forecasts[index]),
            "terminal_value_scope": "reward_only_not_policy_input_or_billed_credit",
        }
        for key, value in expected.items():
            check(info.get(key) == value, f"policy_{key}:{day}")
        check(info.get("purchase_locked_before_actual_read") is True, f"midnight_lock_declaration:{day}")
        check(info.get("actuals_in_policy_observation") is False, f"hidden_actual_declaration:{day}")
        check(info.get("last_evaluation_day") is (day == 364), f"last_day_flag:{day}")
        check(abs(info.get("planning_initial_power_kw", np.inf) - power) <= TOL,
              f"cross_day_actual_power_source:{day}")
        check(abs(info.get("planning_initial_soc", np.inf) - detail["states"][index, 0]) <= TOL,
              f"cross_day_actual_soc_source:{day}")
        expected_terminal = 0.0 if day == 364 else float(detail["price"][index].min()/np.sqrt(.9))
        check(abs(info.get("terminal_value_yuan_per_soc_kwh", np.inf)-expected_terminal) <= TOL,
              f"terminal_reward_scope:{day}")
        power = float(6 * (detail["charge"][index, -1] - detail["discharge"][index, -1]))
    return {"passed": not errors, "errors": errors, "days": len(audits),
            "midnight_purchase_hash_algorithm": "SHA-256 of little-endian float64 contiguous day bytes",
            "forecast_values_checked_against_frozen_archive": True,
            "actual_cross_day_soc_and_previous_power_checked": True,
            "causality_scope": "archive cutoff/hash/declaration checks plus separately tested environment information boundaries"}


def check_execution_projection(detail, deadband_kwh):
    """Recompute clipping from intended requests and current-slot real balance."""
    eta = np.sqrt(.9)
    net = detail["original"] + (detail["actual"][:, :, 1]-detail["actual"][:, :, 0])/6
    available_charge = np.maximum((10800-detail["states"][:, :-1])/eta, 0)
    available_discharge = np.maximum((detail["states"][:, :-1]-1200)*eta, 0)
    charge = np.minimum.reduce((detail["intended_charge"], np.maximum(net, 0),
                                np.full_like(net, 5000/6), available_charge))
    discharge = np.minimum.reduce((detail["intended_discharge"], np.maximum(-net, 0),
                                   np.full_like(net, 5000/6), available_discharge))
    charge[charge < deadband_kwh] = 0
    discharge[discharge < deadband_kwh] = 0
    charge_error = float(np.max(np.abs(charge-detail["charge"])))
    discharge_error = float(np.max(np.abs(discharge-detail["discharge"])))
    errors = []
    if charge_error > TOL:
        errors.append("intended_charge_projection")
    if discharge_error > TOL:
        errors.append("intended_discharge_projection")
    return {"passed": not errors, "errors": errors, "deadband_kwh": deadband_kwh,
            "max_charge_error_kwh": charge_error, "max_discharge_error_kwh": discharge_error,
            "method": "independent vectorized min(request, real balance, available SOC, power limit), then deadband"}


def check_checkpoints(path, detail, daily, audits, training, protocol, run_signature):
    errors, observed_days, observed_iterations = [], 0, 0
    config = protocol["training"]
    directories = sorted((path / "checkpoints").glob("block_*"))
    cutoffs = list(range(31, 365, config["continue_every_days"]))
    if [p.name for p in directories] != [f"block_{cutoff:03}" for cutoff in cutoffs]:
        errors.append("checkpoint_block_inventory")
    for directory, cutoff in zip(directories, cutoffs):
        manifest = read_json(directory / "manifest.json")
        count = min(config["continue_every_days"], 365-cutoff)
        if (manifest["signature"] != run_signature or manifest["start_day"] != cutoff
                or manifest["end_day"] != cutoff+count-1):
            errors.append(f"checkpoint_boundary_or_signature:{cutoff}")
        actual_files = {str(p.relative_to(directory)) for p in directory.rglob("*")
                        if p.is_file() and p.name != "manifest.json"}
        if actual_files != set(manifest["sha256"]):
            errors.append(f"checkpoint_file_inventory:{cutoff}")
        for name, expected in manifest["sha256"].items():
            if not (directory/name).is_file() or sha256(directory/name) != expected:
                errors.append(f"checkpoint_hash:{cutoff}:{name}")
        policy_config = read_json(directory / "policy/configuration.json")
        algorithm = protocol["algorithm"]
        seed = next(item["rl_seed"] for item in protocol["runs"] if item["id"] == path.name)
        for key, expected in (("obs_dim", algorithm["observations"]),
                               ("action_dim", algorithm["actions"]),
                               ("hidden", algorithm["hidden_layers"][0]),
                               ("learning_rate", algorithm["learning_rate"]), ("seed", seed)):
            if policy_config.get(key) != expected:
                errors.append(f"checkpoint_policy_config:{cutoff}:{key}")
        local = read_npz(directory / "dispatch.npz")
        for key, values in detail.items():
            if key not in local or not np.array_equal(local[key], values[observed_days:observed_days+count]):
                errors.append(f"checkpoint_dispatch:{cutoff}:{key}")
        if manifest["audit"] != audits[observed_days:observed_days+count]:
            errors.append(f"checkpoint_audit:{cutoff}")
        for section, source, offset in (("daily", daily, observed_days),
                                         ("training", training, observed_iterations)):
            for i, row in enumerate(manifest[section]):
                for key, value in row.items():
                    archived = source[offset+i].get(key)
                    if isinstance(value, (int, float)):
                        equal = archived is not None and np.isclose(archived, value, atol=1e-9, rtol=1e-12)
                    else:
                        equal = str(archived) == str(value)
                    if not equal:
                        errors.append(f"checkpoint_{section}_mismatch:{cutoff}:{i}:{key}")
        observed_days += count
        observed_iterations += len(manifest["training"])
    if observed_days != len(daily) or observed_iterations != len(training):
        errors.append("checkpoint_total_coverage")
    return {"passed": not errors, "errors": errors, "blocks": len(directories),
            "days": observed_days, "iterations": observed_iterations,
            "model_optimizer_and_rng_files_hashed": True,
            "checkpoint_arrays_equal_final_archives": True}


def verify_directory(path, protocol=None, manifest=None):
    path = Path(path)
    protocol = protocol or read_json(HERE / "protocol.approved.json")
    manifest = manifest or read_json(path.parent / "run_manifest.json")
    spec = next(item for item in protocol["runs"] if item["id"] == path.name)
    detail = read_npz(path / "dispatch_2.npz")
    daily = pd.read_csv(path / "daily.csv").to_dict("records")
    training = pd.read_csv(path / "training.csv").to_dict("records")
    audit_file = read_json(path / "planning_audit.json")
    audits = audit_file["days"]
    completion = read_json(path / "completion.json")
    errors = []

    def check(condition, message):
        if not bool(condition):
            errors.append(message)

    dates = pd.date_range("2025-02-01", "2025-12-31").strftime("%Y-%m-%d").tolist()
    check([row["date"] for row in daily] == dates, "evaluation_dates")
    raw = []
    for name in ("附件2_小区负载.csv", "附件2_光伏发电实际功率.csv"):
        source = pd.read_csv(ROOT / "data/raw" / name)
        check(source.shape == (365, 145), f"source_shape:{name}")
        check(pd.to_datetime(source.iloc[31:, 0]).dt.strftime("%Y-%m-%d").tolist() == dates,
              f"source_dates:{name}")
        raw.append(source.iloc[31:, 1:].to_numpy(float))
    tariff = pd.read_csv(ROOT / "data/raw/附件1.csv")["电价"].to_numpy(float)
    warm_path = ROOT / "data/results/exp003/warmup_2.npz"
    warm = read_npz(warm_path)
    warm_net = (warm["charge"]-warm["discharge"]).ravel()
    initial_soc = float(warm["states"][-1, -1])
    initial_mode = int(np.sign(warm_net[np.abs(warm_net)>TOL][-1]))
    initial_power = float(warm_net[-1]*6)
    check(abs(warm["states"][0, 0]-6000) <= TOL, "warmup_january_initial_soc")
    for key, value in (("initial_soc_kwh", initial_soc), ("initial_mode", initial_mode),
                       ("initial_power_kw", initial_power)):
        check(abs(protocol["evaluation"][key]-value) <= TOL, f"warmup_protocol_boundary:{key}")
    physical = verify_arrays(detail, daily, audits, initial_soc=initial_soc,
                             initial_mode=initial_mode, initial_power_kw=initial_power,
                             source_actual=np.stack(raw, -1), source_price=tariff)
    errors.extend(physical["errors"])
    with np.load(ROOT / protocol["forecast"]["archive"], allow_pickle=False) as archive:
        origins = archive["origins"]
        values = archive[f"{protocol['forecast']['variant']}_seed_{protocol['forecast']['seed']}"]
        lookup = {int(origin): index for index, origin in enumerate(origins)}
        forecasts = np.stack([values[lookup[day*144]] for day in range(31, 365)])
    policy = check_policy_audit(detail, audits, protocol, spec, forecasts, initial_power)
    projection = check_execution_projection(detail, spec["reward"]["deadband_kwh"])
    training_qa = check_training(training, protocol, spec)
    errors.extend(policy["errors"])
    errors.extend(projection["errors"])
    errors.extend(training_qa["errors"])
    run_signature = signature({"evidence": manifest["signature"], "spec": spec})
    check(completion.get("complete") is True, "completion_flag")
    check(completion["signature"] == audit_file["signature"] == run_signature, "run_signature")
    check(audit_file["spec"] == spec, "approved_run_spec")
    for name, expected in completion["output_sha256"].items():
        check(sha256(path/name) == expected, f"completion_hash:{name}")
    checkpoints = check_checkpoints(path, detail, daily, audits, training, protocol, run_signature)
    errors.extend(checkpoints["errors"])
    summary = completion["summary"]
    for key, value in (("total_cost", detail["fees"].sum()),
                       ("planned_cost", detail["fees"][:, :, 0].sum()),
                       ("emergency_cost", detail["fees"][:, :, 3].sum()),
                       ("training_iterations", len(training)),
                       ("training_transitions", training_qa["transitions"]),
                       ("rl_seed", spec["rl_seed"]), ("forecast_seed", protocol["forecast"]["seed"])):
        check(abs(summary[key]-value) <= TOL, f"completion_summary:{key}")
    check(summary["role"] == spec["role"], "primary_role_preserved")
    check(summary["terminal_credit_in_reported_cost"] == 0, "billed_terminal_credit")
    return {"run_id": spec["id"], "role": spec["role"], "rl_seed": spec["rl_seed"],
            "passed": not errors, "errors": errors, "physical_and_billing": physical,
            "training": training_qa, "policy_information": policy, "checkpoints": checkpoints,
            "execution_projection": projection,
            "source": str(path.relative_to(ROOT)),
            "source_sha256": {name: sha256(path/name) for name in
                              ("dispatch_2.npz", "daily.csv", "training.csv", "planning_audit.json", "completion.json")}}


def verify_all(output=None):
    protocol_path = HERE / "protocol.approved.json"
    protocol = read_json(protocol_path)
    directory = ROOT / "data/results/exp007"
    manifest = read_json(directory / "run_manifest.json")
    errors = []
    if manifest.get("complete") is not True:
        errors.append("experiment_not_complete")
    if not protocol.get("formal_run_authorized"):
        errors.append("approved_protocol_flag")
    if manifest["signature"] != signature(manifest["evidence"]):
        errors.append("manifest_evidence_signature")
    for name, expected in manifest["evidence"]["source_sha256"].items():
        if sha256(ROOT/name) != expected:
            errors.append(f"core_source_hash:{name}")
        try:
            committed = subprocess.check_output(["git", "show", manifest["code_commit"]+":"+name],
                                                cwd=ROOT, stderr=subprocess.PIPE)
            if hashlib.sha256(committed).hexdigest() != expected:
                errors.append(f"frozen_commit_source_hash:{name}")
        except subprocess.CalledProcessError:
            errors.append(f"frozen_commit_source_missing:{name}")
    for name, expected in protocol["data_sha256"].items():
        if sha256(ROOT / "data/raw" / name) != expected:
            errors.append(f"raw_source_hash:{name}")
    if sha256(ROOT / protocol["forecast"]["archive"]) != protocol["forecast"]["sha256"]:
        errors.append("frozen_forecast_hash")
    if sha256(ROOT / "data/results/exp003/warmup_2.npz") != manifest["evidence"]["warmup_sha256"]:
        errors.append("warmup_hash")
    for name, expected in manifest.get("output_sha256", {}).items():
        if sha256(directory/name) != expected:
            errors.append(f"experiment_output_hash:{name}")
    expected = [spec["id"] for spec in protocol["runs"]]
    if manifest.get("completed_runs") != expected:
        errors.append("six_run_inventory")
    runs = [verify_directory(directory/spec["id"], protocol, manifest) for spec in protocol["runs"]]
    for run in runs:
        errors.extend(f"{run['run_id']}:{error}" for error in run["errors"])
    result = {"passed": not errors, "status": "passed" if not errors else "failed", "errors": errors,
              "runs": runs, "run_count": len(runs), "days_per_run": 334,
              "intervals_per_run": 48096, "policy_training_iterations_per_run": 1230,
              "primary_run": protocol["primary_run"], "forecast_retrained": False,
              "protocol_sha256": sha256(protocol_path),
              "frozen_code_commit": manifest["code_commit"],
              "core_sources_match_frozen_git_commit": not any("commit_source" in error for error in errors),
              "run_manifest_sha256": sha256(directory / "run_manifest.json"),
              "verifier_sha256": sha256(__file__),
              "imports_runner_policy_or_executor": False,
              "checked_utc": datetime.now(UTC).isoformat()}
    if output:
        write_json(output, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "reports/experiments/exp007/evidence/independent_qa.json")
    result = verify_all(**vars(parser.parse_args()))
    print(json.dumps({"passed": result["passed"], "runs": result["run_count"],
                      "errors": result["errors"]}, ensure_ascii=False))
    raise SystemExit(0 if result["passed"] else 1)
