"""Approved single-thread, causal annual tree/DP evaluation with signed resume."""
# ruff: noqa: E402 -- scientific import time is recorded separately.

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
from .verify import battery_metrics, verify_arrays, verify_directory

IMPORT_SECONDS = time.perf_counter() - IMPORT_START
HERE = Path(__file__).resolve().parent
OUT = ROOT / "data/results/exp006"
INITIAL_SOC = 1421.7991105135516
INITIAL_MODE = 1
INITIAL_POWER = 172.75999999999976
CHECKPOINT_DAYS = 14


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def save_npz(path, detail):
    temp = path.with_name(path.name + ".tmp")
    with temp.open("wb") as stream:
        np.savez_compressed(stream, **detail)
    temp.replace(path)


def signature_for(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def read_completed(path, signature):
    """Return verified matching completion; reject incompatible evidence."""
    manifest_path = path / "completion.json"
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("signature") != signature:
        raise RuntimeError(f"Refusing to overwrite incompatible completed run: {path}")
    if not manifest.get("complete"):
        return None
    for filename, expected in manifest["output_sha256"].items():
        if sha256(path / filename) != expected:
            raise RuntimeError(f"Completed output changed: {path / filename}")
    verified = verify_directory(path)
    if not verified["passed"]:
        raise RuntimeError(f"Cached completion failed independent verification: {verified['errors']}")
    return manifest


def stack_details(details):
    return {key: np.stack([detail[key] for detail in details]) for key in details[0]}


def load_chunks(path, signature):
    """Only complete, contiguous and hash-verified checkpoint chunks are reused."""
    details, rows, audits = [], [], []
    for meta_path in sorted((path / "checkpoints").glob("chunk_*.json")):
        info = json.loads(meta_path.read_text())
        if info["signature"] != signature:
            raise RuntimeError(f"Refusing incompatible checkpoint: {meta_path}")
        if info["start_day"] != 31 + len(rows):
            raise RuntimeError(f"Non-contiguous checkpoint: {meta_path}")
        array_path = meta_path.with_suffix(".npz")
        if sha256(array_path) != info["sha256"]:
            raise RuntimeError(f"Changed checkpoint: {array_path}")
        with np.load(array_path) as archive:
            for index in range(len(info["daily"])):
                details.append({key: archive[key][index].copy() for key in archive.files})
        rows.extend(info["daily"])
        audits.extend(info["audit"])
    if rows:
        report = verify_arrays(stack_details(details), rows, audits, expected_days=len(rows))
        if not report["passed"]:
            raise RuntimeError(f"Invalid checkpoint physics: {report['errors']}")
    return details, rows, audits


def daily_row(detail, metrics, day, spec, risk_seconds, planning_seconds,
              execution_seconds, initial_mode, initial_power):
    date = EPOCH + pd.Timedelta(days=day)
    output = {
        "run_id": spec["id"], "day": day, "date": str(date.date()), "month": int(date.month),
        "scenario": "2", "forecast": spec["forecast"], "seed": spec["seed"],
        "planned_kwh": float(detail["original"].sum()),
        "final_kwh": float(detail["final"].sum()),
        "emergency_kwh": float(detail["emergency"].sum()),
        "emergency_minutes": int(np.sum(detail["emergency"] > 1e-6) * 10),
        "surplus_kwh": float(detail["surplus"].sum()),
        "up_cost": 0.0, "down_cost": 0.0,
        **metrics,
        "risk_seconds": risk_seconds, "planning_seconds": planning_seconds,
        "execution_seconds": execution_seconds,
        "compute_seconds": risk_seconds + planning_seconds + execution_seconds,
    }
    battery = battery_metrics({k: v[None] for k, v in detail.items()}, initial_mode, initial_power)
    output.update(battery)
    return output


def one_case(data, spec, evidence_signature, initial, selected_primary=None):
    path = OUT / spec["id"]
    signature = signature_for({"evidence": evidence_signature, "spec": spec})
    completed = read_completed(path, signature)
    if completed is not None:
        print(f"CACHE_DONE {spec['id']} {completed['summary']['total_cost']:.6f}", flush=True)
        return completed["summary"], completed["timing"], signature
    path.mkdir(parents=True, exist_ok=True)
    (path / "checkpoints").mkdir(exist_ok=True)
    case_start = time.perf_counter()
    details, rows, audits = load_chunks(path, signature)
    config = Config(**spec["config"])
    load_start = time.perf_counter()
    controlled = spec["id"] == "fixed_primary_greedy"
    if controlled:
        if selected_primary is None:
            raise RuntimeError("Controlled replay requires the completed archived primary purchases")
        with np.load(OUT / "primary/dispatch_2.npz") as z:
            primary_detail = {key: z[key].copy() for key in z.files}
        primary_audit = json.loads((OUT / "primary/planning_audit.json").read_text())["days"]
        risk = None
    else:
        store = ForecastStore(spec["forecast"], seed=spec["seed"])
        risk = TreeResidualScenarios(store.origins, store.values, data.actual, data.fixed_price,
                                     conditioning=spec["conditioning"])
    loading_seconds = time.perf_counter() - load_start
    soc = float(details[-1]["states"][-1]) if details else initial["soc"]
    mode = int(audits[-1]["execution_final_mode"]) if audits else initial["mode"]
    power = float(6 * (details[-1]["charge"][-1] - details[-1]["discharge"][-1])) if details else initial["power_kw"]
    resumed_days = len(rows)
    pending_start = len(rows)
    storage_seconds = 0.0
    for day in range(31 + len(rows), 365):
        risk_start = time.perf_counter()
        if controlled:
            index = day - 31
            info = {k: v for k, v in primary_audit[index].items()
                    if not k.startswith("execution_")}
            plan = {"purchase": primary_detail["original"][index].copy(),
                    "charge": primary_detail["intended_charge"][index].copy(),
                    "discharge": primary_detail["intended_discharge"][index].copy(),
                    "states": primary_detail["intended_states"][index].copy(),
                    "metadata": {"planning_seconds": 0.0, "archived_primary_purchase": True}}
            risk_seconds = 0.0
        else:
            supports, info = risk.for_day(day)
            risk_seconds = time.perf_counter() - risk_start
            plan = plan_day(supports, data.fixed_price, soc, config, initial_mode=mode,
                            final_evaluation_day=day == 364)
            info.update(planning_initial_soc=soc, planning_initial_mode=mode)
        planning_seconds = float(plan["metadata"]["planning_seconds"])
        execution_start = time.perf_counter()
        # The current day's actual values are read only after the midnight plan exists.
        detail, metrics = execute_plan(plan, data.actual[day * 144:(day + 1) * 144],
                                       data.fixed_price, soc, config, initial_mode=mode,
                                       greedy=spec["executor"] == "old_greedy")
        execution_seconds = time.perf_counter() - execution_start
        detail.update(intended_charge=plan["charge"].copy(),
                      intended_discharge=plan["discharge"].copy(),
                      intended_states=plan["states"].copy())
        row = daily_row(detail, metrics, day, spec, risk_seconds, planning_seconds,
                        execution_seconds, mode, power)
        audit = {**info, **plan["metadata"], "day": day, "date": row["date"],
                 "forecast_origin": day * 144, "run_id": spec["id"],
                 "forecast": spec["forecast"], "seed": spec["seed"],
                 "execution_initial_soc": soc, "execution_initial_mode": mode,
                 "execution_final_soc": metrics["final_soc"],
                 "execution_final_mode": int(metrics["final_mode"]),
                 "executor": spec["executor"], "purchase_locked_before_actual_read": True,
                 "risk_seconds": risk_seconds, "execution_seconds": execution_seconds,
                 "last_evaluation_day": day == 364,
                 "controlled_replay_of_primary": controlled}
        details.append(detail)
        rows.append(row)
        audits.append(audit)
        soc, mode = metrics["final_soc"], int(metrics["final_mode"])
        power = float(6 * (detail["charge"][-1] - detail["discharge"][-1]))
        if len(rows) - pending_start == CHECKPOINT_DAYS or day == 364:
            storage_start = time.perf_counter()
            chunk = path / "checkpoints" / f"chunk_{31 + pending_start:03d}.npz"
            save_npz(chunk, stack_details(details[pending_start:]))
            write_json(chunk.with_suffix(".json"), {
                "signature": signature, "start_day": 31 + pending_start,
                "sha256": sha256(chunk), "daily": rows[pending_start:], "audit": audits[pending_start:],
            })
            storage_seconds += time.perf_counter() - storage_start
            pending_start = len(rows)
        if len(rows) % 100 == 0:
            print(f"PROGRESS {spec['id']} {len(rows)}/334 elapsed={time.perf_counter()-case_start:.2f}s", flush=True)
    stacked = stack_details(details)
    verify_start = time.perf_counter()
    verification = verify_arrays(stacked, rows, audits,
                                 source_actual=data.actual[31 * 144:].reshape(334, 144, 2),
                                 source_price=data.fixed_price)
    if not verification["passed"]:
        raise RuntimeError(f"Verification failed for {spec['id']}: {verification['errors']}")
    verification_seconds = time.perf_counter() - verify_start
    compute_seconds = {key: float(sum(row[key] for row in rows)) for key in
                       ("risk_seconds", "planning_seconds", "execution_seconds", "compute_seconds")}
    summary = {
        "run_id": spec["id"], "role": spec["role"], "scenario": "2", "days": 334,
        "forecast": spec["forecast"], "seed": spec["seed"], "executor": spec["executor"],
        "conditioning": spec["conditioning"], **asdict(config),
        **{key: float(sum(row[key] for row in rows)) for key in
           ("planned_kwh", "final_kwh", "emergency_kwh", "emergency_minutes", "surplus_kwh",
            "planned_cost", "up_cost", "down_cost", "emergency_cost", "total_cost")},
        **verification["battery_metrics"], **compute_seconds,
        "max_balance_error": verification["max_balance_error_kwh"],
        "max_state_error": verification["max_soc_error_kwh"], "verified": True,
        "mean_daily_cost": float(np.mean([row["total_cost"] for row in rows])),
        "max_daily_cost": float(max(row["total_cost"] for row in rows)),
        "risk_fallback_days": int(sum(bool(info.get("fallback", False)) for info in audits)),
        "tree_fit_seconds": float(sum(info.get("fit_seconds", 0.0) for info in audits)) if not controlled else 0.0,
        "terminal_credit_in_reported_cost": 0.0,
    }
    storage_start = time.perf_counter()
    save_npz(path / "dispatch_2.npz", stacked)
    pd.DataFrame(rows).to_csv(path / "daily.csv", index=False)
    write_json(path / "planning_audit.json", {"signature": signature, "spec": spec, "days": audits})
    write_json(path / "verification.json", verification)
    storage_seconds += time.perf_counter() - storage_start
    timing = {"run_id": spec["id"], "import_seconds_shared": IMPORT_SECONDS,
              "loading_seconds": loading_seconds, **compute_seconds,
              "storage_seconds_this_invocation": storage_seconds,
              "verification_seconds": verification_seconds,
              "wall_seconds_this_invocation": time.perf_counter() - case_start,
              "resumed_days": resumed_days, "cpu_threads": 1}
    summary.update(storage_seconds=storage_seconds,
                   run_wall_seconds=timing["wall_seconds_this_invocation"])
    files = ("dispatch_2.npz", "daily.csv", "planning_audit.json", "verification.json")
    write_json(path / "completion.json", {
        "complete": True, "signature": signature, "created_utc": datetime.now(UTC).isoformat(),
        "summary": summary, "timing": timing,
        "output_sha256": {name: sha256(path / name) for name in files},
    })
    print(f"CASE_DONE {spec['id']} cost={summary['total_cost']:.6f} wall={timing['wall_seconds_this_invocation']:.2f}s", flush=True)
    return summary, timing, signature


def run(only=None):
    began = time.perf_counter()
    protocol_path = HERE / "protocol.approved.json"
    protocol = json.loads(protocol_path.read_text())
    if not protocol["formal_run_authorized"]:
        raise RuntimeError("Formal evaluation requires the user's recorded approval")
    matrix = protocol["approved_run_matrix"]
    if only is not None:
        names = set(only.split(","))
        if names - {spec["id"] for spec in matrix}:
            raise ValueError("Unknown run id")
        matrix = [spec for spec in matrix if spec["id"] in names]
        if "fixed_primary_greedy" in names and "primary" not in names:
            raise ValueError("Controlled replay selection must include primary")
    loading_start = time.perf_counter()
    data = Data()
    warm_path = ROOT / "data/results/exp003/warmup_2.npz"
    with np.load(warm_path) as warm:
        soc = float(warm["states"][-1, -1])
        actions = (warm["charge"] - warm["discharge"]).ravel()
        nonidle = actions[np.abs(actions) > 1e-6]
        mode = int(np.sign(nonidle[-1])) if len(nonidle) else 0
        power = float(actions[-1] * 6)
    if abs(soc - INITIAL_SOC) > 1e-9 or mode != INITIAL_MODE or abs(power - INITIAL_POWER) > 1e-9:
        raise RuntimeError("Frozen warmup initial SOC/direction/power mismatch")
    with np.load(ROOT / "data/results/exp004/dispatch_no_season_seed_42.npz") as baseline:
        if abs(float(baseline["states"][0, 0]) - soc) > 1e-9:
            raise RuntimeError("exp004 initial SOC disagrees with frozen warmup")
    source_paths = [HERE / filename for filename in ("model.py", "risk.py", "run.py", "verify.py", "protocol.approved.json")]
    source_paths.extend(ROOT / filename for filename in
                        ("experiments/problem2/exp003/data.py", "experiments/problem2/exp004/predict.py",
                         "experiments/problem2/exp004/data.py"))
    evidence = {"source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in source_paths},
                "data_sha256": data.hashes, "predictions_sha256": sha256(ROOT / "data/results/exp004/predictions.npz"),
                "warmup_sha256": sha256(warm_path)}
    evidence_signature = signature_for(evidence)
    initial = {"soc": soc, "mode": mode, "power_kw": power}
    loading_seconds = time.perf_counter() - loading_start
    manifest = {"status": "running", "complete": False, "experiment_id": "exp006",
                "started_utc": datetime.now(UTC).isoformat(), "signature": evidence_signature,
                "evidence": evidence, "initial": initial,
                "selection_policy": protocol["selection_policy"], "forecast_retrained": False,
                "lp_called": False, "cpu_threads": 1, "import_seconds": IMPORT_SECONDS,
                "data_loading_seconds": loading_seconds, "selected_runs": [spec["id"] for spec in matrix],
                "expected_runs": [spec["id"] for spec in protocol["approved_run_matrix"]],
                "completed_runs": [], "python": sys.version, "numpy": np.__version__,
                "sklearn": sklearn.__version__, "platform": platform.platform(), "pid": os.getpid()}
    OUT.mkdir(parents=True, exist_ok=True)
    old_manifest = OUT / "run_manifest.json"
    if old_manifest.exists():
        previous = json.loads(old_manifest.read_text())
        if previous.get("signature") != evidence_signature and previous.get("status") == "complete":
            raise RuntimeError("Refusing to overwrite a completed incompatible experiment")
    write_json(old_manifest, manifest)
    summaries, timings, signatures = [], [], {}
    try:
        with threadpool_limits(limits=1):
            manifest["threadpools"] = threadpool_info()
            for spec in matrix:
                summary, timing, signature = one_case(data, spec, evidence_signature, initial,
                                                      selected_primary=signatures.get("primary"))
                summaries.append(summary)
                timings.append(timing)
                signatures[spec["id"]] = signature
                pd.DataFrame(summaries).to_csv(OUT / "summary.csv", index=False)
                pd.DataFrame(timings).to_csv(OUT / "timings.csv", index=False)
                manifest["completed_runs"] = list(signatures)
                manifest["case_signatures"] = signatures
                write_json(old_manifest, manifest)
    except BaseException as error:
        manifest.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                        failure=repr(error), wall_seconds=time.perf_counter() - began)
        write_json(old_manifest, manifest)
        raise
    complete = manifest["completed_runs"] == manifest["expected_runs"]
    manifest.update(status="complete" if complete else "selected_runs_complete", complete=complete,
                    completed_utc=datetime.now(UTC).isoformat(), wall_seconds=time.perf_counter() - began,
                    summary_sha256=sha256(OUT / "summary.csv"), timings_sha256=sha256(OUT / "timings.csv"))
    write_json(old_manifest, manifest)
    print(json.dumps({"status": manifest["status"], "wall_seconds": manifest["wall_seconds"],
                      "completed_runs": manifest["completed_runs"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", help="optional comma-separated prespecified run IDs; default all approved runs")
    run(**vars(parser.parse_args()))
