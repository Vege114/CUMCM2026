"""Build inspectable report/workbook DATA from an explicit archive manifest.

This module never selects a model or writes HTML, Word, figures or Excel.
``prepare`` may contain missing evidence; ``final`` refuses an incomplete or
non-passing selection. In particular, Q2 must pass the fixed exp006 gate.
Run: python -m experiments.exp008.report_payload MANIFEST --output DATA.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.exp008.verify import (BASELINE_COST, BASELINE_REVERSALS,
    INITIAL_MODE, INITIAL_POWER, INITIAL_SOC, TOL, source_arrays, verify_arrays)
from experiments.exp008.frozen_sources import exp005_registry

ROOT = Path(__file__).resolve().parents[2]
SCENARIOS = ("2", "3", "4-2", "4-3")
DATES = ("2025-03-20", "2025-06-21", "2025-09-23", "2025-12-21")
TABLE1_SLOTS = (60, 72, 84, 96, 108, 120)
FEE_KEYS = ("planned_cost_yuan", "up_adjustment_cost_yuan",
            "down_adjustment_cost_yuan", "emergency_cost_yuan")
TEMPLATES = ("reports/templates/report.md", "reports/templates/battery-power.md",
             "reports/templates/history-comparison.md")


class FinalPayloadRejected(ValueError):
    """No final payload is written when the evidence or goal gate fails."""


def clock(slot):
    return f"{slot // 6:02d}:{slot % 6 * 10:02d}"


def interval(slot, end=None):
    return f"{clock(slot)}-{clock(slot + 1 if end is None else end)}"


def resolve(path):
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(resolve(path).read_text())


def reference(path):
    p = resolve(path)
    return {"path": str(p), "sha256": sha(p), "bytes": p.stat().st_size}


def issue(issues, key, reason, blocking=True):
    issues.append({"key": key, "reason": reason, "blocks_final_payload": blocking})


def evidence(path, key, issues, required=False):
    if not path or not resolve(path).is_file():
        if required:
            issue(issues, key, "explicit existing evidence path is required")
        return None
    return reference(path)


def component_fees(fees):
    value = np.asarray(fees).sum(axis=tuple(range(np.asarray(fees).ndim - 1)))
    result = dict(zip(FEE_KEYS, map(float, value)))
    result["adjustment_cost_yuan"] = float(value[1] + value[2])
    result["total_cost_yuan"] = float(value.sum())
    return result


def emergency_events(values):
    """Consecutive positive slots, including an event ending at 24:00."""
    values = np.asarray(values)
    active = np.flatnonzero(values > TOL)
    groups = np.split(active, np.flatnonzero(np.diff(active) > 1) + 1) if len(active) else []
    return [{"start_slot": int(g[0]), "end_slot_exclusive": int(g[-1] + 1),
             "interval": interval(int(g[0]), int(g[-1] + 1)),
             "energy_kwh": float(values[g].sum()), "minutes": int(len(g) * 10)}
            for g in groups]


def daily_tables(a, day, date):
    blocks = [{"interval": interval(b * 24, (b + 1) * 24),
               "charge_kwh": float(a["charge"][day, b * 24:(b + 1) * 24].sum()),
               "discharge_kwh": float(a["discharge"][day, b * 24:(b + 1) * 24].sum())}
              for b in range(6)]
    return {"date": date,
        "table1": [{"slot": t, "interval": interval(t),
                    "original_purchase_kwh": float(a["original"][day, t]),
                    "final_purchase_kwh": float(a["final"][day, t]),
                    "adjustment_delta_kwh": float(a["final"][day, t] - a["original"][day, t])}
                   for t in TABLE1_SLOTS],
        "table2": {"four_hour_blocks": blocks, "soc_00_kwh": float(a["states"][day, 0]),
                   "soc_24_kwh": float(a["states"][day, -1])},
        "table3": {"events": emergency_events(a["emergency"][day]),
                   "no_event_label": "无", "total_emergency_kwh": float(a["emergency"][day].sum())},
        "fees": component_fees(a["fees"][day])}


def workbook_data(a, dates, scenario):
    battery, emergency = [], []
    for day, date in enumerate(dates):
        table = daily_tables(a, day, date)
        for block, row in enumerate(table["table2"]["four_hour_blocks"]):
            label = "0:00" if block == 0 else "24:00" if block == 1 else None
            soc = float(a["states"][day, 0 if block == 0 else -1]) if block < 2 else None
            battery.append([date if block == 0 else None, row["interval"],
                            row["charge_kwh"], row["discharge_kwh"], label, soc])
        events = table["table3"]["events"]
        emergency.extend([[date if i == 0 else None, row["interval"], row["energy_kwh"]]
                          for i, row in enumerate(events)] if events else [[date, "无", 0.0]])
    return {"target_filename": f"result{scenario}.xlsx", "dates": dates,
        "interval_headers": [interval(t) for t in range(144)],
        "original": a["original"].tolist(), "final": a["final"].tolist(),
        "fees": a["fees"].sum(axis=(1, 2)).tolist(), "battery": battery, "emergency": emergency,
        "daily_fee_components": [component_fees(row) for row in a["fees"]],
        "fee_column_semantics": "daily total settled bill, repeated on both purchase sheets by legacy exporter; never add those repeated cells together",
        "final_semantics": "absolute implemented final purchase, NOT adjustment delta",
        "adapter": "reports/export_workbooks_v2.mjs input; extra named metadata may be ignored",
        "rows_without_header": {"purchase": len(dates), "battery": 6 * len(dates), "emergency": len(emergency)}}


def template_snapshot():
    from experiments.exp008.frozen_sources import final_template, TEMPLATE_MAIN_COMMIT
    commit = TEMPLATE_MAIN_COMMIT
    files = []
    for path in TEMPLATES:
        raw, provenance = final_template(path, root=ROOT)
        files.append({"git_source": f"{commit}:{path}", "sha256": hashlib.sha256(raw).hexdigest(), **provenance})
    return {"branch": "main", "commit": commit, "files": files}


def history_payload(issues):
    # Include all registry roles, rather than silently reducing each experiment
    # to its lowest-cost row. exp005 is preserved at its audited git revision.
    records = []
    baseline_path = ROOT / "data/results/exp008/baselines.json"
    audited = read_json(baseline_path) if baseline_path.exists() else {}
    for number in range(1, 7):
        exp = f"exp{number:03d}"
        path = ROOT / f"reports/registry/{exp}.json"
        original, source = None, None
        if path.exists():
            original, source = read_json(path), reference(path)
        else:
            candidates = [r for r in audited.get("q2_rows", []) if r.get("experiment") == exp]
            candidate = next((r for r in candidates if str(r.get("source", "")).startswith("git:")), None)
            if candidate:
                git_path = candidate["source"][4:]
                if exp == 'exp005':
                    raw, frozen = exp005_registry(ROOT)
                    if frozen['git_source_requested'] != git_path:
                        raise ValueError('The exp005 baseline no longer identifies the locked historical revision')
                else:
                    raw = subprocess.check_output(["git", "show", git_path], cwd=ROOT)
                    frozen = {}
                digest = hashlib.sha256(raw).hexdigest()
                if digest != candidate.get("source_sha256"):
                    raise ValueError(f"Historical source hash mismatch: {exp}")
                original, source = json.loads(raw), {"git_source": git_path, "sha256": digest, **frozen}
        if original is None:
            issue(issues, f"history.{exp}", "historical registry is unavailable; do not omit or fabricate its metrics")
            records.append({"experiment_id": exp, "available": False})
            continue
        rows = [r for r in audited.get("q2_rows", []) if r.get("experiment") == exp]
        records.append({"experiment_id": exp, "available": True, "source": source,
            "title": original.get("title"), "question_scope": original.get("scope", ["2", "3", "4-2", "4-3"]),
            "original_registry_metrics": original.get("metrics"),
            "original_forecast_metrics": original.get("forecast_metrics"),
            "original_models": original.get("models"), "original_protocol": original.get("protocol"),
            "original_model_configuration": original.get("model_configuration"),
            "original_environment": original.get("environment"),
            "audited_q2_rows": rows,
            "prediction_comparability": "evaluate information boundaries, target, horizon, training/update and selection rules separately; original exp001/2 early stopping used four targets",
            "dispatch_comparability": "use each audited row's physically_comparable/ranking_allowed and notes; missing flag means not established",
            "runtime_comparability": "descriptive unless hardware, stage scope and reuse/retraining are matched",
            "roles_preserved": True})
    if not audited:
        issue(issues, "history.comparability", "missing exp008/baselines.json comparability audit")
    return {"experiments": records, "comparability_source": reference(baseline_path) if audited else None,
        "older_other_question_rows": audited.get("older_other_question_rows", []),
        "ignored": [{"experiment_id": "exp007", "reason": "explicit user instruction"}],
        "ranking_rule": "only comparable rows; oracle/future-informed results are diagnostic bounds, never causal-strategy rankings"}


def power_reference(path, archive_hash, scenario, issues):
    manifest_ref = evidence(path, f"{scenario}.power_manifest", issues, required=True)
    if not manifest_ref:
        return None
    manifest = read_json(path)
    if manifest.get("source_sha256") != archive_hash or str(manifest.get("scenario")) != scenario:
        raise ValueError(f"Power evidence does not match selected archive: {scenario}")
    folder = resolve(path).parent
    metrics = read_json(folder / "power_metrics.json")
    if manifest.get("smoothing") is not False or manifest.get("annual_downsampling") is not False:
        issue(issues, f"{scenario}.power_sampling", "annual power must retain every original interval")
    if len(manifest.get("random_dates", [])) < 4:
        issue(issues, f"{scenario}.random_days", "at least four fixed-seed random dates required")
    return {"manifest": manifest_ref, "metadata": manifest, "metrics": metrics,
        "annual_csv": evidence(folder / "power_all_intervals.csv", f"{scenario}.annual_csv", issues, True),
        "daily_csv": [evidence(folder / f"power_{d}.csv", f"{scenario}.power_{d}", issues, True)
                      for d in manifest.get("random_dates", [])],
        "plot_contract": {"standalone_png_and_svg": True, "annual_all_points": True,
            "random_days_same_across_cases": True, "y_unit": "kW", "series": "6*(charge_kwh-discharge_kwh)",
            "warmup_boundary_excluded_from_formal_delta": True}}


def annual_case(scenario, spec, issues):
    path = resolve(spec["archive"])
    with np.load(path, allow_pickle=False) as stored:
        a = {key: stored[key].copy() for key in stored.files}
    days, start = len(a["charge"]), int(spec.get("start_day", 31))
    if "initial_soc_kwh" not in spec:
        raise ValueError(f"{scenario}: explicitly state audited initial_soc_kwh, never infer it from the candidate")
    audit = read_json(spec["audit"]) if spec.get("audit") else None
    if isinstance(audit, dict):
        audit = audit.get("days", audit.get("records", []))
    actual, price = source_arrays(scenario, start, days)
    check = verify_arrays(a, scenario, expected_days=days, start_day=start,
        initial_soc=float(spec["initial_soc_kwh"]), initial_mode=int(spec.get("initial_mode", INITIAL_MODE)),
        initial_power_kw=float(spec.get("initial_power_kw", INITIAL_POWER)),
        source_actual=actual, source_price=price, audit_records=audit)
    if not check["passed"]:
        raise ValueError(f"Invalid selected archive {scenario}: {check['errors']}")
    if days != 334 or start != 31:
        issue(issues, f"{scenario}.coverage", "final annual scenarios require all 334 days, 2025-02-01 to 2025-12-31")
    if scenario == "2" and abs(float(spec["initial_soc_kwh"]) - INITIAL_SOC) > TOL:
        issue(issues, "2.comparator_soc", "Q2 must use the frozen exp006 initial SOC")
    audit_ref = evidence(spec.get("audit"), f"{scenario}.issue_audit", issues, True)
    if not check["causality_audit"]["checked_fields"]:
        issue(issues, f"{scenario}.issue_fields", "no issue/cutoff fields were available to verify")
    causal_ref = evidence(spec.get("causality_verification"), f"{scenario}.causality_verification", issues, True)
    if not spec.get("role"):
        issue(issues, f"{scenario}.role", "explicit primary/development/control/ablation role is required")
    if spec.get("eligible_as_causal_strategy") is not True:
        issue(issues, f"{scenario}.causal_role", "selected case must explicitly attest causal strategy eligibility; bounds cannot be final strategies")
    if not spec.get("forecast_sources"):
        issue(issues, f"{scenario}.forecast_sources", "declare base model, correction, seed, information cutoff and artifact hashes")
    dates = pd.date_range(pd.Timestamp("2025-01-01") + pd.Timedelta(days=start), periods=days).strftime("%Y-%m-%d").tolist()
    daily = [{"date": date, **component_fees(a["fees"][i]),
              "original_purchase_kwh": float(a["original"][i].sum()),
              "final_purchase_kwh": float(a["final"][i].sum()),
              "emergency_kwh": float(a["emergency"][i].sum()),
              "charge_kwh": float(a["charge"][i].sum()), "discharge_kwh": float(a["discharge"][i].sum()),
              "soc_start_kwh": float(a["states"][i, 0]), "soc_end_kwh": float(a["states"][i, -1])}
             for i, date in enumerate(dates)]
    sums = [*FEE_KEYS, "adjustment_cost_yuan", "total_cost_yuan", "original_purchase_kwh",
            "final_purchase_kwh", "emergency_kwh", "charge_kwh", "discharge_kwh"]
    monthly = [{"month": month, **{k: float(sum(r[k] for r in daily if r["date"].startswith(month))) for k in sums}}
               for month in sorted({d[:7] for d in dates})]
    ref = reference(path)
    return {"scenario": scenario, "selection": spec, "archive": ref, "verification": check,
        "issue_audit": audit_ref, "causality_verification": causal_ref,
        "forecast_sources": spec.get("forecast_sources"), "fees": component_fees(a["fees"]),
        "battery": check["battery_metrics"], "daily": daily, "monthly": monthly,
        "specified_dates": [daily_tables(a, dates.index(date), date) for date in DATES if date in dates],
        "worst_days_by_bill": sorted(daily, key=lambda r: r["total_cost_yuan"], reverse=True)[:5],
        "workbook_data": workbook_data(a, dates, scenario),
        "full_trajectory_arrays": {key: {"archive": ref["path"], "key": key, "shape": list(a[key].shape),
            "unit": "channels 0/1: load/PV kW; extra channels retain source schema" if key == "actual" else "yuan/kWh" if key == "price" else "yuan" if key == "fees" else "kWh"}
            for key in ("original", "final", "charge", "discharge", "states", "emergency", "surplus", "price", "fees", "actual")},
        "power": power_reference(spec.get("power_manifest"), ref["sha256"], scenario, issues)}


def q1_case(spec, issues):
    from experiments.exp008.q1 import audit as q1_audit
    result = read_json(spec["archive"])
    if len(result["stages"]) != 2 or [r["stage"] for r in result["stages"]] != [1, 2]:
        raise ValueError("Q1 must contain the main two-layer solution, with stage 2 as final")
    q = {key: np.asarray(value, float) for key, value in result["trajectory"].items()}
    actual, price = source_arrays("1", 0, 1)
    data = np.column_stack((price, actual[0]))
    audited_result = dict(result, trajectory=q)
    specialized = q1_audit(data, audited_result)
    g = q["g"][None]
    a = dict(original=g, final=g, charge=q["c"][None] / 6, discharge=q["d"][None] / 6,
             emergency=np.zeros((1, 144)), surplus=q["w"][None], states=q["E"][None],
             actual=actual, price=price[None], fees=np.stack((g * price, np.zeros_like(g),
                 np.zeros_like(g), np.zeros_like(g)), axis=-1))
    check = verify_arrays(a, "1", expected_days=1, start_day=0, initial_soc=6000,
                          initial_mode=0, initial_power_kw=0, source_actual=actual, source_price=price)
    if not check["passed"]:
        raise ValueError(f"Q1 verification failed: {check['errors']}")
    rows = daily_tables(a, 0, "attachment1_given_day")
    return {"selection": spec, "archive": reference(spec["archive"]), "verification": check,
        "two_layer_verification": specialized, "stages": result["stages"], "config": result["config"],
        "cost_budget_yuan": result["budget"], "final_trajectory_stage": 2,
        "trajectory_source_semantics": "attachment 1 deterministic given day; stage-2 final trajectory, not measured future demand",
        "trajectory": {"intervals": [interval(t) for t in range(144)],
            "purchase_kwh": q["g"].tolist(), "charge_kw": q["c"].tolist(), "discharge_kw": q["d"].tolist(),
            "net_battery_power_kw": (q["c"] - q["d"]).tolist(), "charge_kwh": a["charge"][0].tolist(),
            "discharge_kwh": a["discharge"][0].tolist(), "soc_kwh": q["E"].tolist(), "surplus_kwh": q["w"].tolist()},
        "specified_tables": rows, "workbook_data": {"target_filename": "result1.xlsx",
            "plan_rows": [[interval(t), float(q["g"][t])] for t in range(144)],
            "battery_rows": [[r["interval"], r["charge_kwh"], r["discharge_kwh"],
                "0:00" if b == 0 else "24:00" if b == 1 else None,
                float(q["E"][0 if b == 0 else -1]) if b < 2 else None]
                for b, r in enumerate(rows["table2"]["four_hour_blocks"])],
            "adapter_status": "export_q1_workbook.mjs prepared and temporarily verified; official export requires a passing final payload"}}


def enforce_final(payload):
    issues = [row for row in payload["missing_items"] if row["blocks_final_payload"]]
    q2 = payload["scenarios"].get("2")
    acceptance = payload.get("manifest", {}).get("finalization_acceptance", {})
    accepted = bool(q2 and acceptance.get("basis") == "user_accepts_current_verified_result"
        and acceptance.get("user_message") == "行，我觉得现在这个结果比较满意了，就这样写报告然后提交吧"
        and acceptance.get("accepted_q2_archive_sha256") == q2["archive"]["sha256"])
    if q2 is None or (not q2["verification"].get("goal", {}).get("passed") and not accepted):
        issues.append({"key": "Q2.goal", "reason": "must have full 334-day independently verified cost <= 0.92*exp006 and non-idle reversals < 2729, with zero simultaneous operation"})
    # Keep the exact comparator arithmetic, independent of rounded verifier constants.
    if q2 and q2["fees"]["total_cost_yuan"] > .92 * BASELINE_COST + TOL and not accepted:
        issues.append({"key": "Q2.exact_cost_gate", "reason": "exact 8% cost threshold failed"})
    # The user's later acceptance waives the 8% stopping threshold, not physics,
    # full-period verification, a lower actual fee, or fewer reversals.
    if accepted and (not q2["verification"]["passed"]
            or q2["battery"]["simultaneous_slots"] != 0
            or q2["battery"]["direction_reversals"] >= BASELINE_REVERSALS
            or q2["fees"]["total_cost_yuan"] >= BASELINE_COST):
        issues.append({"key": "Q2.accepted_result_constraints", "reason": "accepted archive must retain verified physics, lower cost and fewer reversals"})
    if issues:
        raise FinalPayloadRejected(json.dumps({"final_payload_refused": True, "reasons": issues}, ensure_ascii=False))
    payload["finalization_basis"] = {"user_accepted_current_verified_result": accepted,
        "original_eight_percent_target_met": bool(q2["verification"].get("goal", {}).get("passed")),
        "q2_archive_sha256": q2["archive"]["sha256"],
        "note": "User accepted the current verified result and instructed report/commit; the original 8% target remains unmet." if accepted else "Original optimization gate passed."}


def build_payload(manifest, mode="prepare"):
    if mode not in ("prepare", "final"):
        raise ValueError("mode must be prepare or final")
    issues, cases = [], {}
    for scenario in SCENARIOS:
        spec = manifest.get("scenarios", {}).get(scenario)
        if not spec or not spec.get("archive"):
            issue(issues, f"{scenario}.selection", "no explicitly selected archive; no model is chosen automatically")
            continue
        cases[scenario] = annual_case(scenario, spec, issues)
    q1 = q1_case(manifest["q1"], issues) if manifest.get("q1", {}).get("archive") else None
    if q1 is None:
        issue(issues, "q1.selection", "explicit Q1 two-layer final-trajectory archive is required")
    forecasts = {}
    for name in ("annual_metrics", "monthly_metrics", "lead_metrics", "provenance", "model_configuration", "verification"):
        forecasts[name] = evidence(manifest.get("prediction_evidence", {}).get(name), f"prediction.{name}", issues, True)
    supporting = {}
    for name in ("runtime_by_stage", "seed_and_ablation_results", "planning_solver_diagnostics", "data_validation", "reproduction"):
        supporting[name] = evidence(manifest.get("supporting_evidence", {}).get(name), f"supporting.{name}", issues, True)
    dates = [v["power"]["metadata"].get("random_dates") for v in cases.values() if v.get("power")]
    if dates and any(d != dates[0] for d in dates):
        issue(issues, "power.common_random_dates", "random-day comparison dates differ between scenarios")
    payload = {"schema_version": "exp008-report-data/1", "mode": mode,
        "status": "technical preparation only; not a report or a claim of completed optimization",
        "manifest": manifest, "templates": template_snapshot(), "source_code": reference(__file__),
        "q1": q1, "scenarios": cases, "prediction_evidence": forecasts,
        "supporting_evidence": supporting, "history": history_payload(issues),
        "metric_contract": {"energy": "AC kWh per ten-minute slot; do not multiply by 1/6 again",
            "power": "6*(actual_charge_kwh-actual_discharge_kwh), kW",
            "billing": "p*g0 + 1.5*p*max(g-g0,0) + 0.5*p*max(g0-g,0) + 5*p*emergency",
            "penalty_and_terminal_value_in_reported_bill": False,
            "formal_period": "2025-02-01 through 2025-12-31; 334 days; 48096 slots",
            "time_labels": "raw right endpoint 00:10 represents [00:00,00:10)",
            "nonidle_reversals": "drop zero actions then count consecutive sign flips; exclude warmup boundary for exp006 goal",
            "direct_reversals": "adjacent original ten-minute power signs flip; idle slots interrupt this count",
            "MAE": "sum(abs(error))/n", "RMSE": "sqrt(sum(error**2)/n)",
            "WAPE_pct": "100*sum(abs(error))/sum(abs(actual)); zero denominator -> null",
            "aggregation": "recompute from error sufficient statistics; never average subgroup RMSE/WAPE",
            "seed_standard_deviation": "sample standard deviation; null for n<2",
            "unknown_solver_gap": None, "relative_change": "(current-previous)/abs(previous); zero baseline -> null",
            "WAPE_absolute_change_unit": "percentage points"},
        "primary_results_rows": [{"question": s, "role": v["selection"].get("role"), **v["fees"],
            "emergency_kwh": float(sum(r["emergency_kwh"] for r in v["daily"])),
            "goal_comparison": v["verification"].get("goal")} for s, v in cases.items()],
        "downstream_figure_contracts": ["method and network diagrams from model_configuration",
            "prediction annual/monthly/lead plots from prediction_evidence with separate PV active-period metrics",
            "annual full original power and common seeded random-day plots from existing power CSVs",
            "historical cost, battery, prediction and runtime figures retain role and comparability flags",
            "Q1 full stage-2 power/SOC and two-layer objective/budget values; no third optimization layer",
            "all produced figures need separate PNG/SVG plus visual QA; none are produced here"],
        "integration_notes": ["Do not run neural_v2.export.independent_check on mode-constrained exp008 execution; it asserts unrestricted greedy actions",
            "Use verified actual charge/discharge, not planned c/d or SOC differences, in reports and workbooks",
            "export_q1_workbook.mjs supports result1.xlsx and fixes the template one-slot label offset; official export requires final mode",
            "template battery/emergency demonstration rows must be replaced/expanded, never mistaken for complete annual coverage",
            "365-day claims require a separately verified matching warmup trace; the default export contains only 334 formal days"],
        "missing_items": issues}
    if mode == "final":
        enforce_final(payload)
        payload["status"] = "verified technical input eligible for final report generation; report/workbooks not yet generated"
    return payload


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("manifest", type=Path)
    p.add_argument("--mode", choices=("prepare", "final"), default="prepare")
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    payload = build_payload(read_json(args.manifest), args.mode)
    if args.output.exists():
        raise FileExistsError("Do not overwrite a prepared selection; choose a new output path")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "mode": args.mode,
                      "selected_scenarios": list(payload["scenarios"]),
                      "missing_items": payload["missing_items"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
