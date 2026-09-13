"""Predeclared full334 Q4-2 linked-HGB price model transfer with unchanged hardware.

The first3 days are only a computational and physical feasibility gate.
No partial-period cost or operation-count gate stops the fixed annual run.
Actual price enters only after optimization and physical replay, for settlement.
"""
from __future__ import annotations

import argparse
import ast
import sys
import copy
import hashlib
import json
from pathlib import Path
import shutil
from time import perf_counter

import numpy as np
import pandas as pd

from experiments.exp008.closed_loop import optimize
from experiments.exp008.mode_planning_physical import plan
from experiments.exp008.planner import execute
from experiments.exp008.hgb_linked_price_forecast import LinkedPriceForecasts, OUT as PRICE_OUT
from experiments.exp008.forecast_absolute_hgb import OUT as HGB_OUT, PRIMARY
from experiments.exp008.verify import verify_arrays

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/"data/results/exp008/q4_hgb_linked_price_physical"
CONFIG = {"scenario": "4-2", "forecast_calibration": PRIMARY, "holiday": False,
    "initialization_scenarios": 3, "refinement_history_days": 28,
    "scenario_selection": "deterministic equally spaced historical origins",
    "block_slots": 6, "switching": 50., "wear": .002, "seconds": 5., "gap": .002,
    "refinement_maxiter": 120, "deadband": 0., "variation": 0.,
    "refinement_terminal_value": .45, "final_day_terminal_value": 0.,
    "price_information": "own-issued HGB-linked midnight causal forecast for optimization; actual price only for settlement",
    "feasibility_days": 3, "annual_days": 334,
    "gate": "first3 feasible MIP incumbents plus independent physics; partial cost is not a gate",
    "nonanticipative_scenario_recourse_certificate": False,
    "candidate_count": 1, "development_not_independent_test": True,
    "forecast_change_only": True, "change": "price model appends own-issued HGB load/PV; load/PV predictions unchanged", "price_model": "Ridge28_plus_own_issued_HGB_load_PV", "this_is_not_a_Q2_goal_test": True}



def local_dependency_closure():
    """Freeze executed local modules plus statically discoverable local imports."""
    files = {Path(__file__).resolve()}
    files.update(Path(module.__file__).resolve() for name,module in list(sys.modules.items())
        if name.startswith('experiments.') and getattr(module,'__file__',None)
        and str(module.__file__).endswith('.py'))
    pending=list(files)
    while pending:
        path=pending.pop()
        tree=ast.parse(path.read_text())
        names=[]
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):
                names.extend(alias.name for alias in node.names)
            elif isinstance(node,ast.ImportFrom) and node.module and not node.level:
                names.append(node.module)
                names.extend(node.module+'.'+alias.name for alias in node.names)
        for name in names:
            if not name.startswith('experiments.'):
                continue
            file=ROOT.joinpath(*name.split('.')).with_suffix('.py')
            if not file.exists():
                file=ROOT.joinpath(*name.split('.'),'__init__.py')
            if file.exists() and file not in files:
                files.add(file); pending.append(file)
    return sorted(files)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n")


def read_npz(path):
    with np.load(path, allow_pickle=False) as pack:
        return {key: pack[key].copy() for key in pack.files}


def price_causality_check(forecast):
    checks = []
    for day in (31, 90, 273):
        altered_data = copy.copy(forecast.data)
        altered_data.actual = forecast.data.actual.copy()
        altered_data.actual[day*144:] = 1e7
        altered = LinkedPriceForecasts(data=altered_data)
        before = forecast.get(day, scenario="4-2")
        after = altered.get(day, scenario="4-2")
        for key in ("load_kw", "pv_kw", "price"):
            np.testing.assert_array_equal(before[key], after[key])
        np.testing.assert_array_equal(forecast.net_error_paths(day, "4-2", limit=28)["errors_kwh"],
                                      altered.net_error_paths(day, "4-2", limit=28)["errors_kwh"])
        assert before["audit"]["price_last_label"] < day*144
        assert not before["audit"]["known_future_price"]
        checks.append(day)
    return {"passed": True, "all_current_future_actual_channels_perturbed": True,
            "unchanged_planning_input_days": checks, "actual_price_optimization_forbidden": True}


def finish(out, details, audits, baseline, data, initial_soc, initial_mode, initial_power):
    days = len(details)
    directory = out/("feasibility3" if days == 3 else "full334")
    directory.mkdir(exist_ok=True)
    arrays = {key: np.stack([d[key] for d in details]) for key in details[0]}
    arrays["days"] = np.arange(31, 31+days)
    actual = data.actual[31*144:(31+days)*144].reshape(days, 144, 3)
    args = dict(scenario="4-2", expected_days=days, initial_soc=initial_soc,
                initial_mode=initial_mode, initial_power_kw=initial_power,
                source_actual=actual, source_price=actual[..., 2])
    verification = verify_arrays(arrays, audit_records=audits, **args)
    baseline_verification = verify_arrays({key: value[:days] for key, value in baseline.items()}, **args)
    assert verification["passed"], verification["errors"]
    assert baseline_verification["passed"], baseline_verification["errors"]
    metric, reference = verification["battery_metrics"], baseline_verification["battery_metrics"]
    total, base_total = verification["billing"]["total_cost"], baseline_verification["billing"]["total_cost"]
    gate = bool(total < base_total and metric["direction_reversals"] < reference["direction_reversals"])
    mip = [row["mip"] for row in audits]
    gaps = [row["mip_gap"] for row in mip if row["mip_gap"] is not None]
    summary = {"days": days, "scenario": "4-2", **verification["billing"],
        "battery": metric, "baseline_billing": baseline_verification["billing"],
        "baseline_battery": reference, "total_delta": total-base_total,
        "cost_reduction_pct": 100*(1-total/base_total),
        "reversals_delta": metric["direction_reversals"]-reference["direction_reversals"],
        "active_slots_delta": metric["active_slots"]-reference["active_slots"],
        "throughput_kwh_delta": metric["throughput_kwh"]-reference["throughput_kwh"],
        "verified": True, "cost_and_reversals_improved": gate,
        "initial_soc": initial_soc, "price_used_for_planning": "causal midnight forecast",
        "mip_feasible_days": sum(bool(row["feasible"]) for row in mip),
        "mip_gap_max": max(gaps) if gaps else None,
        "mip_gap_mean": float(np.mean(gaps)) if gaps else None,
        "mip_time_limit_days": sum(row["status"] == 1 for row in mip),
        "refinement_success_days": sum(row["refinement"]["success"] for row in audits),
        "global_optimization_certificate": False,
        "nonanticipative_scenario_recourse_certificate": False,
        "comparison_scope": "complete absolute HGB+Ridge28+memory with identical physical modes algorithm vs original ridge28 physical planner; same Q4-2 causal price and real settlement"}
    rows = [{"day": 31+i, "date": str((pd.Timestamp("2025-02-01")+pd.Timedelta(days=i)).date()),
             "planned_cost": float(d["fees"][:, 0].sum()),
             "emergency_cost": float(d["fees"][:, 3].sum()),
             "total_cost": float(d["fees"].sum()), "initial_soc": float(d["states"][0]),
             "final_soc": float(d["states"][-1])} for i, d in enumerate(details)]
    np.savez_compressed(directory/"dispatch_4-2.npz", **arrays)
    pd.DataFrame(rows).to_csv(directory/"daily.csv", index=False)
    write_json(directory/"audit.json", audits)
    write_json(directory/"verification.json", verification)
    write_json(directory/"baseline_verification.json", baseline_verification)
    write_json(directory/"summary.json", summary)
    print("STAGE", json.dumps({k: summary[k] for k in ("days", "total_cost", "total_delta",
        "reversals_delta", "active_slots_delta", "throughput_kwh_delta", "cost_and_reversals_improved")}), flush=True)
    return summary


def run(out=OUT):
    out = Path(out)
    if (out/"full334/summary.json").exists():
        raise FileExistsError("Completed HGB Q4-2 evaluations are immutable")
    out.mkdir(parents=True, exist_ok=True)
    chunks = out/"daily_chunks"
    chunks.mkdir(exist_ok=True)
    forecast = LinkedPriceForecasts()
    data = forecast.data
    warmup_path = ROOT/"data/results/exp002/warmup_4-2.npz"
    warmup = read_npz(warmup_path)
    initial_soc = float(warmup["states"][-1, -1])
    initial_power = float(6*(warmup["charge"][-1, -1]-warmup["discharge"][-1, -1]))
    initial_mode = int(np.sign(initial_power))
    assert abs(initial_soc-1390.382746315672) < 1e-8
    baseline_path = ROOT/"data/results/exp008/q4_absolute_hgb_physical/full334/dispatch_4-2.npz"
    baseline = read_npz(baseline_path)
    files = local_dependency_closure()
    files += [p for p in (ROOT/'experiments/problem2/exp004/protocol.json',
        ROOT/'pyproject.toml', ROOT/'uv.lock') if p.exists()]
    provenance = {"config": CONFIG, "initial_soc": initial_soc, "initial_mode": initial_mode,
        "initial_power_kw": initial_power, "raw_source_hashes": data.hashes,
        "source_hashes": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
        "baseline_sha256": hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
        "warmup_sha256": hashlib.sha256(warmup_path.read_bytes()).hexdigest(),
        "complete_HGB_pipeline_values_sha256": hashlib.sha256(forecast.store.values.tobytes()).hexdigest(),
        "forecast_origins_sha256": hashlib.sha256(forecast.store.origins.tobytes()).hexdigest(),
        "forecast_model_id": PRIMARY,
        "forecast_artifact_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [HGB_OUT/f"{PRIMARY}.npz", HGB_OUT/"provenance.json", HGB_OUT/"protocol.json",
                         HGB_OUT/"training_audit.json", HGB_OUT/"causality_verification.json",
                         HGB_OUT/"ridge28_audit.json", HGB_OUT/"memory_half_audit.json",
                         PRICE_OUT/"protocol.json", PRICE_OUT/"summary.json",
                         PRICE_OUT/"price_predictions.npz", PRICE_OUT/"independent_audit.json",
                         PRICE_OUT/"causality_audit.json"]}}
    protocol_path = out/"protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != provenance:
            raise RuntimeError("Source/configuration changed; choose a new output directory")
    else:
        write_json(protocol_path, provenance)
        snapshots = out/"source_snapshots"
        snapshots.mkdir(exist_ok=True)
        for path in files:
            (snapshots/(str(path.relative_to(ROOT)).replace("/", "__"))).write_bytes(path.read_bytes())
        evidence = out/"forecast_evidence"
        evidence.mkdir(exist_ok=True)
        for name in provenance["forecast_artifact_sha256"]:
            path = Path(name)
            shutil.copy2(path, evidence/(path.parent.name+"__"+path.name))
        np.savez_compressed(out/"issued_forecasts.npz", origins=forecast.store.origins, values=forecast.store.values)
    write_json(out/"price_causality_verification.json", price_causality_check(forecast))
    soc, mode, details, audits = initial_soc, initial_mode, [], []
    began = perf_counter()
    limit = 334
    for day in range(31, 31+limit):
        archive_path, audit_path = chunks/f"day_{day}.npz", chunks/f"day_{day}.json"
        if archive_path.exists() and audit_path.exists():
            detail, audit = read_npz(archive_path), json.loads(audit_path.read_text())
            if abs(float(detail["states"][0])-soc) > 1e-6:
                raise AssertionError("cached day does not continue preceding SOC")
        else:
            issue = forecast.get(day, scenario="4-2")
            history = forecast.net_error_paths(day, "4-2", limit=28)
            paths = (issue["load_kw"]-issue["pv_kw"])[None, :]/6+history["errors_kwh"]
            selected = np.linspace(0, len(paths)-1, min(3, len(paths))).astype(int)
            predicted_price = issue["price"]
            try:
                initialized = plan(paths[selected], predicted_price, soc, mode,
                    block_slots=6, switching=50., wear=.002, seconds=5., gap=.002, final=day == 364)
            except (RuntimeError, AssertionError) as error:
                write_json(out/"failure.json", {"day": day, "completed_days": len(details), "error": str(error)})
                raise
            refined = optimize(initialized["purchase"], paths, predicted_price, soc,
                charge_mask=initialized["allowed_charge"], throughput=.002, variation=0.,
                terminal=0. if day == 364 else .45, maxiter=120, deadband=0.)
            q = refined["purchase"]
            observed_load_pv = data.actual[day*144:(day+1)*144, :2]
            detail = execute(q, observed_load_pv, predicted_price, soc, charge_deadband=0.,
                             charge_mask=initialized["allowed_charge"])
            # Actual price enters only after optimization and physical actions.
            actual = data.actual[day*144:(day+1)*144].copy()
            realized_price = actual[:, 2]
            detail.update(original=q, final=q.copy(), actual=actual, price=realized_price.copy(),
                allowed_charge=initialized["allowed_charge"].copy())
            detail["fees"] = np.stack((q*realized_price, np.zeros(144), np.zeros(144),
                                         5*detail["emergency"]*realized_price), axis=-1)
            audit = {"day": day, "information_cutoff": day*144, "forecast": issue["audit"],
                "history": history["audit"], "mip": initialized["metadata"],
                "refinement": refined["metadata"], "initial_soc": soc, "initial_mode": mode,
                "selected_history_origins": history["origins"][selected].tolist(),
                "purchase_locked_before_current_actual_read": True,
                "optimization_price_source": "midnight_forecast", "actual_price_used_only_in_settlement": True}
            np.savez_compressed(chunks/f"planning_day_{day}.npz",
                initial_purchase=initialized["purchase"], refined_purchase=q,
                allowed_charge=initialized["allowed_charge"], all_net_paths=paths,
                selected_scenario_indices=selected, predicted_price=predicted_price, all_price_error_paths=history["price_errors"],
                initial_soc=np.array(soc), initial_mode=np.array(mode),
                **{key: initialized[key] for key in ("scenario_charge", "scenario_discharge",
                    "scenario_emergency", "scenario_surplus", "scenario_states")})
            np.savez_compressed(archive_path, **detail)
            write_json(audit_path, audit)
        details.append(detail)
        audits.append(audit)
        soc = float(detail["states"][-1])
        signs = np.sign(detail["charge"]-detail["discharge"])
        if np.any(signs):
            mode = int(signs[signs != 0][-1])
        if len(details) % 5 == 0:
            print("PROGRESS", json.dumps({"days": len(details), "total_cost": sum(float(d["fees"].sum()) for d in details),
                "seconds": perf_counter()-began, "last_mip_gap": audit["mip"]["mip_gap"]}), flush=True)
        if len(details) == 3:
            checked = finish(out, details, audits, baseline, data, initial_soc, initial_mode, initial_power)
            feasible = checked["verified"] and checked["mip_feasible_days"] == 3
            write_json(out/"first3_feasibility_gate.json", {"passed": feasible,
                "partial_cost_or_counts_used_for_gate": False, "full334_predeclared": True,
                "exact_first3_arrays_preserved": True})
            if not feasible:
                raise AssertionError("First3 computational or physical gate failed")
    result = finish(out, details, audits, baseline, data, initial_soc, initial_mode, initial_power)
    first3 = read_npz(out/"feasibility3/dispatch_4-2.npz")
    full = read_npz(out/"full334/dispatch_4-2.npz")
    equal = {key: bool(np.array_equal(value, full[key][:3])) for key, value in first3.items()}
    assert all(equal.values()), equal
    write_json(out/"first3_prefix_equality.json", equal)
    source_unchanged = all(hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == expected
        for path, expected in provenance["source_hashes"].items())
    artifact_unchanged = all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected
        for path, expected in provenance["forecast_artifact_sha256"].items())
    baseline_unchanged = hashlib.sha256(baseline_path.read_bytes()).hexdigest() == provenance["baseline_sha256"]
    assert source_unchanged and artifact_unchanged and baseline_unchanged
    write_json(out/"source_verification.json", {"passed": True, "source_files_unchanged": source_unchanged,
        "forecast_artifacts_unchanged": artifact_unchanged, "baseline_unchanged": baseline_unchanged})
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    run(args.out)
