"""Independent archive checks: no planner, executor, or optimization imports."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ETA = float(np.sqrt(0.9))
TOL = 1e-6


def battery_metrics(detail, initial_mode=1, initial_power_kw=0.0):
    charge = np.asarray(detail["charge"]).ravel()
    discharge = np.asarray(detail["discharge"]).ravel()
    power = 6 * (charge - discharge)
    modes = np.sign(charge - discharge).astype(int)
    nonidle = modes[np.abs(charge - discharge) > TOL]
    direction_sequence = np.r_[initial_mode, nonidle]
    reversals_with_warmup = int(np.sum(direction_sequence[1:] * direction_sequence[:-1] == -1))
    reversals = int(np.sum(nonidle[1:] * nonidle[:-1] == -1))
    ramps = np.abs(np.diff(power))
    boundary_ramp = float(abs(power[0] - initial_power_kw))
    return {
        "charge_kwh": float(charge.sum()), "discharge_kwh": float(discharge.sum()),
        "throughput_kwh": float((charge + discharge).sum()),
        "equivalent_full_cycles": float((ETA * charge.sum() + discharge.sum() / ETA) / 24000),
        "direction_reversals": reversals,
        "direction_reversals_including_warmup_boundary": reversals_with_warmup,
        "simultaneous_slots": int(np.sum((charge > TOL) & (discharge > TOL))),
        "charging_slots": int(np.sum(charge > TOL)),
        "discharging_slots": int(np.sum(discharge > TOL)),
        "active_slots": int(np.sum((charge > TOL) | (discharge > TOL))),
        "power_changes_count": int(np.sum(ramps > TOL)),
        "power_ramp_total_kw": float(ramps.sum()),
        "power_ramp_total_kw_including_warmup_boundary": float(ramps.sum() + boundary_ramp),
        "power_ramp_max_kw_per_10min": float(ramps.max(initial=0)),
        "power_ramp_p95_kw_per_10min": float(np.quantile(ramps, .95)),
        "mean_absolute_power_kw": float(np.abs(power).mean()),
        "max_charge_kw": float(6 * charge.max(initial=0)),
        "max_discharge_kw": float(6 * discharge.max(initial=0)),
        "initial_soc": float(detail["states"][0, 0]),
        "final_soc": float(detail["states"][-1, -1]),
        "initial_mode": int(initial_mode),
        "final_mode": int(nonidle[-1]) if len(nonidle) else int(initial_mode),
        "initial_power_kw": float(initial_power_kw), "final_power_kw": float(power[-1]),
    }


def verify_arrays(detail, daily, audit, expected_days=334, start_day=31,
                  initial_soc=1421.7991105135516, initial_mode=1,
                  initial_power_kw=172.75999999999976, source_actual=None, source_price=None):
    """Recompute physical, settlement, chronology and mode checks from arrays."""
    errors = []

    def check(condition, message):
        if not bool(condition):
            errors.append(message)

    shapes = {key: (expected_days, 144) for key in
              ("original", "final", "charge", "discharge", "emergency", "surplus", "price")}
    shapes.update(states=(expected_days, 145), fees=(expected_days, 144, 4),
                  actual=(expected_days, 144, 2))
    for key, shape in shapes.items():
        check(key in detail and detail[key].shape == shape, f"shape:{key}")
    if errors:
        return {"passed": False, "errors": errors}
    for key, values in detail.items():
        check(np.isfinite(values).all(), f"nonfinite:{key}")
        check(np.min(values) >= -TOL, f"negative:{key}")
    q, c, d, e, w, s, p = (detail[key] for key in
                            ("final", "charge", "discharge", "emergency", "surplus", "states", "price"))
    check(np.array_equal(detail["original"], q), "midnight_purchase_changed")
    check(np.all(p > 0), "nonpositive_tariff")
    if source_actual is not None:
        check(np.array_equal(detail["actual"], source_actual), "source_actual_mismatch")
    if source_price is not None:
        check(np.array_equal(p, np.broadcast_to(source_price, p.shape)), "source_tariff_mismatch")
    if all(key in detail for key in ("intended_charge", "intended_discharge", "intended_states")):
        ic, ide, ist = (detail[key] for key in
                       ("intended_charge", "intended_discharge", "intended_states"))
        check(ic.shape == c.shape and ide.shape == d.shape and ist.shape == s.shape,
              "intended_shapes")
        check(np.max(np.abs(np.diff(ist, axis=1) - ETA * ic + ide / ETA)) <= TOL,
              "intended_soc_equation")
        check(np.min(ist) >= 1200 - TOL and np.max(ist) <= 10800 + TOL, "intended_soc_bounds")
        check(max(np.max(ic), np.max(ide)) <= 5000 / 6 + TOL, "intended_power_limit")
        check(not np.any((ic > TOL) & (ide > TOL)), "intended_simultaneous_actions")
    balance = q + (detail["actual"][:, :, 1] - detail["actual"][:, :, 0]) / 6 + d + e - c - w
    soc_error = np.diff(s, axis=1) - ETA * c + d / ETA
    continuity = float(np.max(np.abs(s[1:, 0] - s[:-1, -1]), initial=0))
    max_balance = float(np.max(np.abs(balance)))
    max_state = float(np.max(np.abs(soc_error)))
    check(max_balance <= TOL, "supply_demand_balance")
    check(max_state <= TOL, "battery_state_equation")
    check(continuity <= TOL, "cross_day_soc_discontinuity")
    check(abs(float(s[0, 0]) - initial_soc) <= TOL, "initial_soc")
    check(np.min(s) >= 1200 - TOL and np.max(s) <= 10800 + TOL, "soc_bounds")
    check(max(np.max(c), np.max(d)) <= 5000 / 6 + TOL, "battery_power_limit")
    check(not np.any((c > TOL) & (d > TOL)), "simultaneous_charge_discharge")
    check(not np.any((c > TOL) & (e > TOL)), "emergency_charging")
    expected_fees = np.stack((q * p, np.zeros_like(q), np.zeros_like(q), 5 * e * p), axis=-1)
    fee_error = float(np.max(np.abs(expected_fees - detail["fees"])))
    check(fee_error <= TOL, "slot_settlement")
    expected_day_ids = list(range(start_day, start_day + expected_days))
    check(len(daily) == expected_days, "daily_row_count")
    check(len(audit) == expected_days, "audit_row_count")
    if len(daily) == len(audit) == expected_days:
        check([int(row["day"]) for row in daily] == expected_day_ids, "daily_order")
        check([int(row["day"]) for row in audit] == expected_day_ids, "audit_order")
        mode = initial_mode
        daily_cost_error = 0.0
        for index, (row, info) in enumerate(zip(daily, audit)):
            day = start_day + index
            check(int(info["information_cutoff"]) == day * 144, f"midnight_cutoff:{day}")
            check(int(info["max_observed_index"]) < day * 144, f"future_actual_used:{day}")
            check(all(int(o) + 144 <= day * 144 for o in info["training_origins"]),
                  f"incomplete_training_day:{day}")
            check(int(info["forecast_origin"]) == day * 144, f"forecast_origin:{day}")
            check(abs(float(info["execution_initial_soc"]) - s[index, 0]) <= TOL,
                  f"execution_initial_soc:{day}")
            check(int(info["execution_initial_mode"]) == mode, f"cross_day_initial_mode:{day}")
            if not info.get("controlled_replay_of_primary", False) and "planning_initial_soc" in info:
                check(abs(float(info["planning_initial_soc"]) - s[index, 0]) <= TOL,
                      f"planning_initial_soc:{day}")
                check(int(info["planning_initial_mode"]) == mode, f"planning_initial_mode:{day}")
            if day == 364 and "terminal_value_yuan_per_soc_kwh" in info:
                check(info["terminal_value_yuan_per_soc_kwh"] == 0, "last_day_terminal_value")
            observed_modes = np.sign(c[index] - d[index]).astype(int)
            nonidle = observed_modes[np.abs(c[index] - d[index]) > TOL]
            mode = int(nonidle[-1]) if len(nonidle) else mode
            check(int(info["execution_final_mode"]) == mode, f"cross_day_final_mode:{day}")
            daily_cost_error = max(daily_cost_error,
                                   abs(float(row["total_cost"]) - float(expected_fees[index].sum())))
            check(abs(float(row["initial_soc"]) - s[index, 0]) <= TOL, f"daily_initial_soc:{day}")
            check(abs(float(row["final_soc"]) - s[index, -1]) <= TOL, f"daily_final_soc:{day}")
        check(daily_cost_error <= TOL, "daily_settlement")
    else:
        daily_cost_error = None
    return {
        "passed": not errors, "errors": errors, "days": expected_days,
        "intervals": expected_days * 144, "max_balance_error_kwh": max_balance,
        "max_soc_error_kwh": max_state, "max_cross_day_soc_error_kwh": continuity,
        "max_slot_fee_error_yuan": fee_error, "max_daily_cost_error_yuan": daily_cost_error,
        "recomputed_total_cost": float(expected_fees.sum()),
        "reported_cost_excludes_penalties_and_terminal_value": True,
        "source_actual_and_tariff_checked": source_actual is not None and source_price is not None,
        "battery_metrics": battery_metrics(detail, initial_mode, initial_power_kw),
        "causality_scope": "archive cutoff audit plus separately tested future-mutation invariance",
    }


def verify_directory(path):
    from experiments.problem2.exp003.data import Data

    path = Path(path)
    with np.load(path / "dispatch_2.npz") as z:
        detail = {key: z[key] for key in z.files}
    daily = pd.read_csv(path / "daily.csv").to_dict("records")
    audit = json.loads((path / "planning_audit.json").read_text())["days"]
    data = Data()
    return verify_arrays(detail, daily, audit, source_actual=data.actual[31 * 144:].reshape(334, 144, 2),
                         source_price=data.fixed_price)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    result = verify_directory(parser.parse_args().directory)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["passed"] else 1)
