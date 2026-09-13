"""Independent physical, billing, chronology and goal checks for exp008.

No optimizer, forecast or controller modules are imported.  Archive checks
cannot prove forecast causality; optional issue audit records are checked
separately and their exact scope is reported.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ETA = float(np.sqrt(0.9))
MIN_SOC, MAX_SOC, CAPACITY = 1200.0, 10800.0, 12000.0
POWER_ENERGY, TOL = 5000 / 6, 1e-6
INITIAL_SOC = 1421.7991105135516
INITIAL_MODE, INITIAL_POWER = 1, 172.75999999999976
BASELINE_COST = 14066257.477256786
BASELINE_REVERSALS, BASELINE_ACTIVE = 2729, 23028
BASELINE_THROUGHPUT = 11480847.039823763
GOAL_COST = 12940956.8791


def battery_metrics(detail, initial_mode=INITIAL_MODE, initial_power_kw=INITIAL_POWER):
    """Count actions on the chronological 10-minute trace, never block sums."""
    c, d = (np.asarray(detail[key], float).ravel() for key in ("charge", "discharge"))
    states = np.asarray(detail["states"], float)
    power = 6 * (c - d)
    modes = np.where(np.abs(c - d) > TOL, np.sign(c - d), 0).astype(int)
    nonidle = modes[modes != 0]
    sequence = np.r_[initial_mode, nonidle]
    ramps = np.abs(np.diff(power))
    return {
        "charge_kwh": float(c.sum()), "discharge_kwh": float(d.sum()),
        "throughput_kwh": float((c + d).sum()),
        "equivalent_full_cycles": float((ETA * c.sum() + d.sum() / ETA) / (2 * CAPACITY)),
        "direction_reversals": int(np.sum(nonidle[1:] * nonidle[:-1] == -1)),
        "direction_reversals_including_warmup_boundary": int(np.sum(sequence[1:] * sequence[:-1] == -1)),
        "charging_slots": int(np.sum(c > TOL)), "discharging_slots": int(np.sum(d > TOL)),
        "active_slots": int(np.sum((c > TOL) | (d > TOL))),
        "simultaneous_slots": int(np.sum((c > TOL) & (d > TOL))),
        "max_overlap_kwh": float(np.minimum(c, d).max(initial=0)),
        "charge_starts": int(np.sum((modes == 1) & (np.r_[0, modes[:-1]] != 1))),
        "discharge_starts": int(np.sum((modes == -1) & (np.r_[0, modes[:-1]] != -1))),
        "power_changes_count": int(np.sum(ramps > TOL)),
        "power_ramp_total_kw": float(ramps.sum()),
        "power_ramp_total_kw_including_warmup_boundary": float(ramps.sum() + abs(power[0] - initial_power_kw)),
        "power_ramp_max_kw_per_10min": float(ramps.max(initial=0)),
        "power_ramp_p95_kw_per_10min": float(np.quantile(ramps, .95)) if ramps.size else 0.0,
        "mean_absolute_power_kw": float(np.abs(power).mean()),
        "max_charge_kw": float(6 * c.max(initial=0)), "max_discharge_kw": float(6 * d.max(initial=0)),
        "initial_soc": float(states[0, 0]), "final_soc": float(states[-1, -1]),
        "minimum_soc": float(states.min()), "maximum_soc": float(states.max()),
        "initial_mode": int(initial_mode), "final_mode": int(nonidle[-1]) if nonidle.size else int(initial_mode),
        "initial_power_kw": float(initial_power_kw), "final_power_kw": float(power[-1]),
        "degradation_interpretation": "operating intensity proxies; not calibrated battery ageing or lifetime",
    }


def goal_check(total_cost, battery, physics_passed=True, full_evaluation=True):
    """The fixed exp006 Q2 comparator is never reselected after iteration."""
    checks = {
        "cost_reduced_at_least_8pct": bool(full_evaluation and total_cost <= GOAL_COST + TOL),
        "direction_reversals_reduced": bool(full_evaluation and battery["direction_reversals"] < BASELINE_REVERSALS),
        "no_simultaneous_charge_discharge": bool(battery["simultaneous_slots"] == 0),
        "physical_checks_passed": bool(physics_passed),
        "full_334_day_evaluation": bool(full_evaluation),
    }
    return {
        "passed": all(checks.values()), "checks": checks,
        "count_definition": "exp006 non-idle direction reversals; active ten-minute slots are a duration diagnostic",
        "stricter_all_operation_metrics_passed": bool(all(checks.values())
            and battery["active_slots"] < BASELINE_ACTIVE
            and battery["throughput_kwh"] < BASELINE_THROUGHPUT),
        "active_slots_also_reduced": bool(full_evaluation and battery["active_slots"] < BASELINE_ACTIVE),
        "baseline": {"run_id": "exp006/primary", "total_cost": BASELINE_COST,
                     "direction_reversals": BASELINE_REVERSALS, "active_slots": BASELINE_ACTIVE,
                     "throughput_kwh": BASELINE_THROUGHPUT},
        "cost_threshold_yuan": GOAL_COST,
        "cost_reduction_pct": float(100 * (1 - total_cost / BASELINE_COST)) if full_evaluation else None,
        "direction_reversal_reduction_pct": float(100 * (1 - battery["direction_reversals"] / BASELINE_REVERSALS)) if full_evaluation else None,
        "active_slot_reduction_pct": float(100 * (1 - battery["active_slots"] / BASELINE_ACTIVE)) if full_evaluation else None,
        "throughput_reduction_pct": float(100 * (1 - battery["throughput_kwh"] / BASELINE_THROUGHPUT)) if full_evaluation else None,
        "throughput_also_reduced": bool(full_evaluation and battery["throughput_kwh"] < BASELINE_THROUGHPUT),
        "cost_reduced_at_least_10pct": bool(full_evaluation and total_cost <= .9 * BASELINE_COST + TOL),
        "comparison_scope": "full comparable 334 days" if full_evaluation else "partial run: no comparison to full-year totals",
    }


def verify_arrays(detail, scenario="2", *, expected_days=334, start_day=31,
                  initial_soc=INITIAL_SOC, initial_mode=INITIAL_MODE,
                  initial_power_kw=INITIAL_POWER, source_actual=None, source_price=None,
                  audit_records=None, ramp_limit_kw=None):
    """Verify arrays with shape (days,144), SOC (days,145), fees (...,4).

    Q2/4-2 lock purchases at midnight. Q3/4-3 use the frozen original-plan
    settlement: p*g0 + 1.5*p*(g-g0)+ + .5*p*(g0-g)+ + 5*p*emergency.
    Actual may retain additional channels; first two are load/PV in kW.
    """
    scenario = str(scenario)
    if scenario not in ("1", "2", "3", "4-2", "4-3"):
        raise ValueError("scenario must be 1, 2, 3, 4-2 or 4-3")
    errors = []

    def check(condition, message):
        if not bool(condition):
            errors.append(message)

    required = ("original", "final", "charge", "discharge", "emergency", "surplus", "price", "states", "fees", "actual")
    for key in required:
        check(key in detail, f"missing:{key}")
    if errors:
        return {"passed": False, "errors": errors, "scenario": scenario}
    a = {key: np.asarray(detail[key], dtype=float) for key in required}
    for key in required:
        expected = ((expected_days, 145) if key == "states" else
                    (expected_days, 144, 4) if key == "fees" else (expected_days, 144))
        if key == "actual":
            check(a[key].ndim == 3 and a[key].shape[:2] == (expected_days, 144)
                  and a[key].shape[-1] >= 2, "shape:actual")
        else:
            check(a[key].shape == expected, f"shape:{key}")
        check(np.isfinite(a[key]).all(), f"nonfinite:{key}")
        check(np.min(a[key], initial=0) >= -TOL, f"negative:{key}")
    if errors:
        return {"passed": False, "errors": errors, "scenario": scenario}
    original, q, c, d, e, w, s, p = (a[key] for key in
                                     ("original", "final", "charge", "discharge", "emergency", "surplus", "states", "price"))
    check(np.all(p > 0), "nonpositive_tariff")
    if scenario in ("1", "2", "4-2"):
        check(np.max(np.abs(original - q)) <= TOL, "midnight_purchase_changed")
    balance = q + (a["actual"][..., 1] - a["actual"][..., 0]) / 6 + d + e - c - w
    state_error = np.diff(s, axis=1) - ETA * c + d / ETA
    continuity = float(np.max(np.abs(s[1:, 0] - s[:-1, -1]), initial=0))
    max_balance, max_state = float(np.abs(balance).max()), float(np.abs(state_error).max())
    check(max_balance <= TOL, "supply_demand_balance")
    check(max_state <= TOL, "battery_state_equation")
    check(continuity <= TOL, "cross_day_soc_discontinuity")
    check(abs(float(s[0, 0]) - initial_soc) <= TOL, "initial_soc")
    check(s.min() >= MIN_SOC - TOL and s.max() <= MAX_SOC + TOL, "soc_bounds")
    check(max(c.max(), d.max()) <= POWER_ENERGY + TOL, "battery_power_limit")
    check(not np.any((c > TOL) & (d > TOL)), "simultaneous_charge_discharge")
    check(not np.any((c > TOL) & (e > TOL)), "emergency_charging")
    if scenario == "1":
        check(np.max(e) <= TOL, "q1_emergency_purchase")
        check(np.max(np.abs(s[:, -1] - s[:, 0])) <= TOL, "q1_daily_soc_equality")
    expected_fees = np.stack((original * p, 1.5 * p * np.maximum(q - original, 0),
                              .5 * p * np.maximum(original - q, 0), 5 * p * e), axis=-1)
    fee_error = float(np.max(np.abs(expected_fees - a["fees"])))
    check(fee_error <= TOL, "slot_settlement")
    if source_actual is not None:
        source = np.asarray(source_actual, float)
        check(source.shape[:2] == a["actual"].shape[:2] and source.shape[-1] >= 2
              and np.array_equal(a["actual"][..., :2], source[..., :2]), "source_actual_mismatch")
    if source_price is not None:
        check(np.array_equal(p, np.broadcast_to(source_price, p.shape)), "source_tariff_mismatch")
    # Explicit identifiers, when available, are compared rather than inferred.
    if "days" in detail:
        check(np.array_equal(detail["days"], np.arange(start_day, start_day + expected_days)), "day_identifiers")
    if "origins" in detail:
        check(np.array_equal(detail["origins"], np.arange(start_day, start_day + expected_days) * 144), "midnight_origins")
    audit_checks = 0
    if audit_records is not None:
        expanded = []
        outer_origins = []
        for row in audit_records:
            outer = row.get("information_cutoff", row.get("origin"))
            if outer is not None:
                outer_origins.append(int(outer))
            expanded.append(row)
            if isinstance(row.get("forecast"), dict):
                forecast_audit = row["forecast"]
                expanded.append(forecast_audit)
                if outer is not None and forecast_audit.get("origin") is not None:
                    check(int(outer) == int(forecast_audit["origin"]), "forecast_and_execution_issue_mismatch")
                    audit_checks += 1
        if outer_origins:
            check(outer_origins == sorted(outer_origins), "audit_chronological_order")
            check({value // 144 for value in outer_origins}
                  == set(range(start_day, start_day + expected_days)), "audit_day_coverage")
        for row in expanded:
            origin = row.get("information_cutoff", row.get("information_cutoff_exclusive",
                             row.get("origin", row.get("forecast_origin"))))
            if origin is None:
                continue
            origin = int(origin)
            check(origin % 144 in ((0, 36, 72, 108) if scenario in ("3", "4-3") else (0,)), "invalid_issue_time")
            if row.get("max_observed_index") is not None:
                check(int(row["max_observed_index"]) < origin, "future_actual_used")
                audit_checks += 1
            if "forecast_origin" in row:
                check(int(row["forecast_origin"]) <= origin, "future_forecast_issue_used")
                audit_checks += 1
            if "training_origins" in row:
                check(all(int(value) + 144 <= origin for value in row["training_origins"]), "incomplete_training_day")
                audit_checks += 1
            if "pv_calibration_origins" in row:
                check(all(int(value) + 144 <= origin for value in row["pv_calibration_origins"]), "incomplete_pv_calibration_horizon")
                audit_checks += 1
            for key in ("pv_calibration_last_label", "load_correction_last_label", "price_last_label"):
                if row.get(key) is not None:
                    check(int(row[key]) < origin, f"future_label:{key}")
                    audit_checks += 1
            if row.get("pv_issue_origin") is not None:
                check(int(row["pv_issue_origin"]) <= origin, "unreleased_official_pv_forecast")
                audit_checks += 1
            if "known_future_price" in row:
                check(not bool(row["known_future_price"]), "future_actual_price_used")
                audit_checks += 1
    battery = battery_metrics(a, initial_mode, initial_power_kw)
    if ramp_limit_kw is not None:
        power = 6 * (c - d).ravel()
        check(np.abs(np.diff(np.r_[initial_power_kw, power])).max(initial=0) <= ramp_limit_kw + TOL, "power_ramp_limit")
    billing = dict(zip(("planned_cost", "up_cost", "down_cost", "emergency_cost"),
                       map(float, expected_fees.sum(axis=(0, 1)))))
    billing["total_cost"] = float(expected_fees.sum())
    result = {
        "passed": not errors, "errors": errors, "scenario": scenario,
        "days": expected_days, "intervals": expected_days * 144, "start_day": start_day,
        "max_balance_error_kwh": max_balance, "max_soc_error_kwh": max_state,
        "max_cross_day_soc_error_kwh": continuity, "max_slot_fee_error_yuan": fee_error,
        "source_actual_and_tariff_checked": source_actual is not None and source_price is not None,
        "billing": billing, "recomputed_total_cost": billing["total_cost"],
        "reported_cost_excludes_penalties_and_terminal_value": True,
        "battery_metrics": battery,
        "causality_audit": {"provided": audit_records is not None, "checked_fields": audit_checks,
                            "scope": "provided issue/cutoff consistency only; forecast causality needs independent future-mutation tests"},
    }
    if scenario == "2":
        result["goal"] = goal_check(billing["total_cost"], battery, not errors,
                                     expected_days == 334 and start_day == 31)
    return result


def source_arrays(scenario, start_day=31, days=334, root=ROOT):
    raw = Path(root) / "data/raw"
    if str(scenario) == "1":
        reference = pd.read_csv(raw / "附件1.csv").iloc[:, 1:].to_numpy(float)
        return np.broadcast_to(reference[:, 1:3], (days, 144, 2)), reference[:, 0]
    arrays = [pd.read_csv(raw / name).iloc[:, 1:].to_numpy(float)
              for name in ("附件2_小区负载.csv", "附件2_光伏发电实际功率.csv")]
    actual = np.stack(arrays, axis=-1)[start_day:start_day + days]
    if str(scenario).startswith("4"):
        price = pd.read_csv(raw / "附件4.csv").iloc[:, 1:].to_numpy(float)[start_day:start_day + days]
    else:
        price = pd.read_csv(raw / "附件1.csv").iloc[:, 1].to_numpy(float)
    return actual, price


def verify_npz(path, scenario="2", *, expected_days=334, start_day=31,
               check_sources=True, audit_path=None, **kwargs):
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        detail = {key: archive[key].copy() for key in archive.files}
    if audit_path is not None:
        audit = json.loads(Path(audit_path).read_text())
        kwargs["audit_records"] = audit.get("days", audit.get("records", [])) if isinstance(audit, dict) else audit
    if check_sources:
        actual, price = source_arrays(scenario, start_day, expected_days)
        kwargs.update(source_actual=actual, source_price=price)
    result = verify_arrays(detail, scenario, expected_days=expected_days, start_day=start_day, **kwargs)
    result.update(source=str(path), source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--scenario", default="2")
    parser.add_argument("--days", type=int, default=334)
    parser.add_argument("--start-day", type=int, default=31)
    parser.add_argument("--initial-soc", type=float, default=INITIAL_SOC)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify_npz(args.archive, args.scenario, expected_days=args.days, start_day=args.start_day,
                        initial_soc=args.initial_soc, audit_path=args.audit)
    encoded = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded)
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
