"""Chronological Q2 replay. D4-A: stop physical certification on overlap.

python -m experiments.problem2.stochastic_lp.run --days 2 --out ...
Only successful full 334-day output may be exported as final result2.xlsx.
"""

import argparse
import hashlib
import json
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path
from time import perf_counter

import numpy as np

from experiments.problem2.exp003.data import Data

from .model import Config, State, Tree, settle, solve_tree
from .scenarios import information_tree, sample_paths

ROOT = Path(__file__).resolve().parents[3]


def save_json(path, data):
    def default(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(type(value).__name__)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=default))


def failure_record(day, slot, scope, result, tree, state):
    overlap = np.minimum(result["charge"], result["discharge"])
    n = int(np.argmax(overlap))
    return {"date": str(date(2025, 1, 1) + timedelta(days=day)), "scope": scope,
            "replay_slot": slot, "node": n, "node_time": int(tree.time[n]),
            "absolute_slot": int(tree.time[n]) + (slot or 0),
            "charge_kwh": float(result["charge"][n]),
            "discharge_kwh": float(result["discharge"][n]),
            "overlap_kwh": float(overlap[n]), "initial_state": asdict(state),
            "node_end_soc": float(result["end_soc"][n]),
            "node_net_power_kw": float(result["net_power_kw"][n]),
            "reason": "D4-A: simultaneous charge/discharge; physical certification stopped",
            "metadata": result["metadata"]}


def replay_day(forecast, bundle, price, observe, state, cfg, mode="scenario", event=None):
    """Observe is called once per current slot; future actual has no other API."""
    if mode.startswith("point"):
        tree = Tree.deterministic(forecast)
        memberships = [[0]] * len(forecast)
    else:
        tree, memberships = information_tree(bundle["paths_kw"], bundle["scale_kw"])
    plan = solve_tree(tree, price, state, cfg, forbid_midnight_emergency=mode.startswith("point"))
    if event:
        event("midnight", None, plan, tree, state, memberships)
    h = len(forecast)
    detail = {k: np.zeros(h) for k in ("charge", "discharge", "emergency", "surplus")}
    detail.update(grid=plan["grid"].copy(), states=np.r_[state.soc, np.zeros(h)],
                  actual=np.zeros((h, 2)), net_power_kw=np.zeros(h))
    logs = []
    for t in range(h):
        observation = np.asarray(observe(t), dtype=float).copy()
        if mode == "point":
            estimate = forecast[t:].copy()
            estimate[0] = observation
            remaining = Tree.deterministic(estimate)
        else:
            remaining, _ = information_tree(bundle["paths_kw"], bundle["scale_kw"], t, observation)
        result = solve_tree(remaining, price[t:], state, cfg, plan["grid"][t:])
        # All predicted nodes are diagnostic; any overlap stops certification,
        # not just the first action. Never hide a future relaxed trajectory.
        if event:
            try:
                event("execution", t, result, remaining, state, None)
            except CertificationStopped as error:
                error.partial_detail = {key: value[:t + 1].copy() if key == "states"
                                        else value[:t].copy() for key, value in detail.items()}
                error.partial_detail["fees"] = settle(detail["grid"][:t], detail["emergency"][:t], price[:t])
                error.partial_logs = logs
                raise
        for key in ("charge", "discharge", "emergency", "surplus"):
            detail[key][t] = result[key][0]
        detail["actual"][t] = observation
        detail["states"][t + 1] = result["end_soc"][0]
        detail["net_power_kw"][t] = result["net_power_kw"][0]
        state = State(float(result["end_soc"][0]), float(result["net_power_kw"][0]))
        logs.append(result["metadata"])
    detail["fees"] = settle(detail["grid"], detail["emergency"], price)
    return plan, detail, state, logs


class CertificationStopped(RuntimeError):
    pass


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--days", type=int, default=334)
    parser.add_argument("--mode", choices=["scenario", "point", "point_scenario_control"], default="scenario")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.days <= 334 or args.out.exists():
        parser.error("days must be 1..334 and output directory must be new")
    data, cfg = Data(), Config()
    source = ROOT / "data/results/exp004"
    archive = source / "predictions.npz"
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    manifest = json.loads((source / "prediction_manifest.json").read_text())
    if digest != manifest["archive_sha256"] or not manifest["complete"]:
        raise ValueError("forecast hash/completeness mismatch")
    with np.load(archive) as pack:
        forecasts = pack["no_season_seed_42"].copy()
        origins = pack["origins"].copy()
        masks = pack["no_season_mask"].copy()
    with np.load(source / "no_season/warmup_2.npz") as warmup:
        state = State(float(warmup["states"][-1, -1]),
                      float((warmup["charge"][-1, -1] - warmup["discharge"][-1, -1]) * 6))
    args.out.mkdir(parents=True)
    protocol = {"config": asdict(cfg), "mode": args.mode, "forecast": "exp004/no_season/seed42",
                "archive_sha256": digest, "data_hashes": data.hashes, "initial_state": asdict(state),
                "scenario_window": 56, "scenario_maximum": 16, "seed": 42,
                "branch_slots": [36, 72, 108], "minimum_child": 2,
                "intraday_weight_update": False, "tree_parameters_user_confirmed": True,
                "january_residuals": "none archived; Feb1 point-only fallback",
                "evaluation": "retrospective chronological; 2025-02-01 through 2025-12-31",
                "D": {"D1": "A", "D2": "A", "D3": "A", "D4": "A", "D5": "A", "D7": "A", "D9": "A"}}
    save_json(args.out / "protocol.json", protocol)
    records, details = [], []
    began = perf_counter()
    try:
        for i in range(args.days):
            origin, day = int(origins[i]), int(origins[i] // 144)
            bundle = sample_paths(forecasts[i], forecasts[:i],
                                  data.actual[origins[:i, None] + np.arange(144)],
                                  origins[:i], origin, masks[i])
            day_out = args.out / str(date(2025, 1, 1) + timedelta(days=day))
            day_out.mkdir()
            np.savez_compressed(day_out / "scenarios.npz", **bundle)
            initial = state

            def event(scope, t, result, tree, event_state, memberships, day_out=day_out, day=day):
                overlap = result["metadata"]["max_overlap_kwh"]
                if scope == "midnight" or overlap > cfg.tolerance:
                    name = "midnight" if scope == "midnight" else f"execution-{t:03}"
                    np.savez_compressed(day_out / f"{name}.npz",
                                        time=tree.time, parent=tree.parent, probability=tree.probability,
                                        supply_kw=tree.supply_kw,
                                        **{k: v for k, v in result.items() if k != "metadata"})
                    save_json(day_out / f"{name}.json", result["metadata"])
                    if memberships is not None:
                        save_json(day_out / "memberships.json", memberships)
                if overlap > cfg.tolerance:
                    failure = failure_record(day, t, scope, result, tree, event_state)
                    failure["max_overlap_node"] = int(np.argmax(np.minimum(result["charge"], result["discharge"])))
                    failure["current_action_overlap_kwh"] = float(min(result["charge"][0], result["discharge"][0]))
                    save_json(args.out / "certification_failure.json", failure)
                    raise CertificationStopped(f"{day_out.name} {scope} slot={t} overlap={overlap:.6f} kWh")

            plan, detail, state, logs = replay_day(forecasts[i], bundle, data.fixed_price,
                                                  lambda t, origin=origin: data.actual[origin + t], state, cfg,
                                                  args.mode, event)
            np.savez_compressed(day_out / "actual.npz", **detail)
            save_json(day_out / "execution_logs.json", logs)
            fees = detail["fees"].sum(axis=0)
            row = {"date": day_out.name, "planned_cost": float(fees[0]), "emergency_cost": float(fees[1]),
                   "total_cost": float(fees.sum()), "planned_kwh": float(detail["grid"].sum()),
                   "emergency_kwh": float(detail["emergency"].sum()),
                   "initial_soc": initial.soc, "final_soc": state.soc,
                   "tv_kw": float(np.abs(np.diff(np.r_[initial.previous_power_kw,
                                                        detail["net_power_kw"]])).sum()),
                   "plan_expected_cost": plan["metadata"]["second_cost"]}
            records.append(row)
            details.append(detail)
            save_json(args.out / "daily.json", records)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    except CertificationStopped as error:
        if hasattr(error, "partial_detail"):
            np.savez_compressed(day_out / "executed_prefix.npz", **error.partial_detail)
            save_json(day_out / "executed_prefix_logs.json", error.partial_logs)
        save_json(args.out / "status.json", {"status": "physical_certification_stopped", "reason": str(error),
                                             "completed_days": len(records), "seconds": perf_counter() - began})
        print(str(error), flush=True)
        return
    np.savez_compressed(args.out / "dispatch.npz", **{k: np.stack([d[k] for d in details]) for k in details[0]})
    save_json(args.out / "status.json", {"status": "complete" if args.days == 334 else "smoke_passed",
                                         "completed_days": len(records), "seconds": perf_counter() - began})


if __name__ == "__main__":
    main()
