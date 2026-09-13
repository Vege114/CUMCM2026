"""OAT sensitivity of the accepted exp008 two-stage day-ahead MILP.

Run from the repository root::
    python experiments/exp008/robustness/q1_analysis.py --cases center
    python experiments/exp008/robustness/q1_analysis.py --resume

The original solver and independent audit are reused without editing frozen files.
Every changed physical constraint is applied to BOTH objective stages. The cost
budget is recalculated against that case's own stage-one economic optimum.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.exp008 import q1  # noqa: E402

OUT = ROOT / "data/results/exp008/robustness/q1"
FROZEN = ROOT / "data/results/exp008/q1/revised.json"
SELECTION = ROOT / "experiments/exp008/final_selection.json"
BASE = dict(delta=.001, ramp=1000., up=3, down=2, eta_rt=.9)
CASES = [
    ("center", "center", None),
    ("delta_0p0005", "delta", .0005),
    ("delta_0p0009", "delta", .0009),
    ("delta_0p0011", "delta", .0011),
    ("delta_0p0020", "delta", .002),
    ("ramp_900", "ramp", 900.),
    ("ramp_1100", "ramp", 1100.),
    ("up_2", "up", 2),
    ("up_4", "up", 4),
    ("down_1", "down", 1),
    ("down_3", "down", 3),
    ("eta_rt_0p855", "eta_rt", .855),
    ("eta_rt_0p945", "eta_rt", .945),
    ("capacity_0p9", "capacity_factor", .9),
    ("capacity_1p1", "capacity_factor", 1.1),
]


def serial(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(type(obj).__name__)


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=serial,
                               allow_nan=False) + "\n", encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite(value):
    return float(value) if value is not None and np.isfinite(value) else None


@contextmanager
def instrument_solver(time_limit, capacity_factor, records, incumbents):
    """Record HiGHS outcomes before q1.solve enforces proven optimality."""
    original_milp = q1.milp
    original_bounds = q1.EMIN, q1.EMAX, q1.EINIT

    def wrapped(*args, **kwargs):
        options = dict(kwargs.get("options", {}))
        options["time_limit"] = time_limit
        kwargs["options"] = options
        began = time.perf_counter()
        result = original_milp(*args, **kwargs)
        records.append(dict(
            stage=len(records) + 1, status=int(result.status),
            success=bool(result.success), message=str(result.message),
            objective=finite(result.fun),
            mip_gap=finite(getattr(result, "mip_gap", None)),
            mip_dual_bound=finite(getattr(result, "mip_dual_bound", None)),
            mip_node_count=finite(getattr(result, "mip_node_count", None)),
            seconds=time.perf_counter() - began,
            has_incumbent=result.x is not None,
        ))
        if result.x is not None:
            incumbents[f"stage_{len(records)}_x"] = result.x.copy()
        return result

    try:
        q1.milp = wrapped
        q1.EMIN, q1.EMAX, q1.EINIT = (v * capacity_factor for v in original_bounds)
        yield
    finally:
        q1.milp = original_milp
        q1.EMIN, q1.EMAX, q1.EINIT = original_bounds


def run_case(case, data, time_limit):
    name, parameter, value = case
    config = BASE.copy()
    capacity_factor = 1.
    if parameter == "capacity_factor":
        capacity_factor = value
    elif parameter != "center":
        config[parameter] = value
    records, incumbents = [], {}
    began = time.perf_counter()
    result = dict(name=name, parameter=parameter, parameter_value=value,
                  config=config, capacity_factor=capacity_factor,
                  nominal_capacity_kwh=12000 * capacity_factor,
                  emin_kwh=q1.EMIN * capacity_factor,
                  emax_kwh=q1.EMAX * capacity_factor,
                  initial_terminal_kwh=q1.EINIT * capacity_factor,
                  solver_records=records, eligible_for_comparison=False)
    with instrument_solver(time_limit, capacity_factor, records, incumbents):
        try:
            solved = q1.solve(data, **config)
            result["solution"] = solved
            result["eligible_for_comparison"] = solved["metrics"]["violations"] == 0
            result["outcome"] = "optimal_and_physical_audit_passed"
        except (RuntimeError, ValueError) as exc:
            result["outcome"] = "solver_or_audit_failure"
            result["error"] = str(exc)
    result["elapsed_seconds"] = time.perf_counter() - began
    result["input_sha256"] = hashlib.sha256(data.tobytes()).hexdigest()
    result["source_sha256"] = sha(Path(q1.__file__))
    result["frozen_sha256"] = sha(FROZEN)
    # Failed incumbents are retained as diagnostics, never as proven-optimal data.
    if not result["eligible_for_comparison"] and incumbents:
        path = OUT / f"{name}_unaccepted_incumbents.npz"
        np.savez_compressed(path, **incumbents)
        result["unaccepted_incumbent_file"] = str(path.relative_to(ROOT))
    dump(OUT / f"{name}.json", result)
    return result


def make_summary(results, frozen):
    center = next((r for r in results if r["name"] == "center"), None)
    baseline = frozen["metrics"]
    comparison = dict(available=center is not None, passed=False)
    if center and center["eligible_for_comparison"]:
        metric = center["solution"]["metrics"]
        differences = {k: float(metric[k] - baseline[k]) for k in
                       ("cost", "smooth", "tv", "quantity", "starts")}
        stage1_diff = center["solution"]["stages"][0]["cost"] - frozen["stages"][0]["cost"]
        max_trajectory_diff = max(float(np.max(np.abs(
            np.asarray(center["solution"]["trajectory"][k]) - np.asarray(v))))
            for k, v in frozen["trajectory"].items())
        comparison.update(
            passed=abs(differences["cost"]) <= 1e-4
            and abs(differences["smooth"]) <= 1e-7
            and abs(stage1_diff) <= 1e-4 and metric["violations"] == 0,
            metric_differences=differences,
            stage1_cost_difference_yuan=stage1_diff,
            max_trajectory_absolute_difference=max_trajectory_diff,
            criterion="stage-one and stage-two cost within 1e-4 yuan; smooth within 1e-7; zero physical violations",
            trajectory_note="MILP may have multiple optimal trajectories; objective and constraint agreement, not bitwise trajectory equality, is required.",
        )
    rows = []
    for item in results:
        sol = item.get("solution", {})
        metric = sol.get("metrics", {})
        recs = item["solver_records"]
        row = dict(name=item["name"], parameter=item["parameter"],
                   parameter_value=item["parameter_value"], **item["config"],
                   capacity_factor=item["capacity_factor"],
                   nominal_capacity_kwh=item["nominal_capacity_kwh"],
                   emin_kwh=item["emin_kwh"], emax_kwh=item["emax_kwh"],
                   initial_terminal_kwh=item["initial_terminal_kwh"],
                   eligible_for_comparison=item["eligible_for_comparison"],
                   outcome=item["outcome"], elapsed_seconds=item["elapsed_seconds"],
                   stage1_cost=sol.get("stages", [{}])[0].get("cost"),
                   budget=sol.get("budget"), **metric)
        for stage in (1, 2):
            rec = next((r for r in recs if r["stage"] == stage), {})
            for key in ("status", "success", "mip_gap", "mip_dual_bound", "seconds", "message"):
                row[f"stage{stage}_{key}"] = rec.get(key)
        if item["eligible_for_comparison"]:
            for key in ("cost", "smooth", "tv", "quantity"):
                row[f"{key}_change_pct_from_frozen"] = 100 * (metric[key] / baseline[key] - 1)
        rows.append(row)
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with (OUT / "sensitivity.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    valid = [r for r in rows if r["eligible_for_comparison"]]
    summary = dict(
        total_cases=len(rows), planned_cases=len(CASES),
        accepted_cases=len(valid), failures=len(rows) - len(valid),
        center_reproduction=comparison, frozen_metrics=baseline,
        all_accepted_cases_physical_violations=sum(r.get("violations", 0) for r in valid),
        cost_change_pct_range=[min(r["cost_change_pct_from_frozen"] for r in valid),
                               max(r["cost_change_pct_from_frozen"] for r in valid)] if valid else None,
        smooth_change_pct_range=[min(r["smooth_change_pct_from_frozen"] for r in valid),
                                 max(r["smooth_change_pct_from_frozen"] for r in valid)] if valid else None,
        rows=rows,
        limitations=[
            "Single attachment-1 typical day; not evidence of annual controller performance or out-of-sample forecasting.",
            "Each setting is reoptimized; this measures model/parameter sensitivity, not resilience of an unchanged dispatch to unknown disturbances.",
            "One-at-a-time grid does not estimate interaction effects or a statistical confidence interval.",
            "Minimum off-time applies separately to charge and discharge modes, matching the accepted q1 model.",
            "Capacity cases preserve SOC fractions (10%/90% limits, 50% initial and terminal SOC) and fixed 5000-kW inverter rating.",
            "Smoothness keeps the accepted 30-minute short-run penalty definition even when the minimum-run constraint is perturbed.",
        ],
    )
    dump(OUT / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="*", help="Names of cases; omitted means all 15.")
    parser.add_argument("--resume", action="store_true", help="Reuse existing case JSON only after provenance check.")
    parser.add_argument("--time-limit", type=float, default=180., help="Seconds per objective stage; original default 180.")
    args = parser.parse_args()
    if args.time_limit <= 0:
        parser.error("--time-limit must be positive")
    unknown = set(args.cases or []) - {c[0] for c in CASES}
    if unknown:
        parser.error(f"unknown cases: {sorted(unknown)}")
    OUT.mkdir(parents=True, exist_ok=True)
    frozen_hash, selection_hash, source_hash = sha(FROZEN), sha(SELECTION), sha(Path(q1.__file__))
    frozen = json.loads(FROZEN.read_text())
    if frozen["config"] != BASE:
        raise ValueError("Frozen q1 configuration differs from the protocol center.")
    data = q1.read_data()
    input_hash = hashlib.sha256(data.tobytes()).hexdigest()
    protocol = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        design="One-at-a-time deterministic MILP reoptimization; same final two-stage model; no final selection changes",
        center=BASE, cases=[dict(name=n, parameter=p, value=v) for n, p, v in CASES],
        solver=dict(backend="scipy.optimize.milp / HiGHS", mip_rel_gap=1e-9,
                    time_limit_seconds_per_stage=args.time_limit,
                    acceptance="Both stages must return success and pass q1.audit; timeout incumbents excluded"),
        objectives=dict(stage1="minimize electricity purchasing cost",
                        stage2="minimize original normalized total variation + mode starts + short-run deficit",
                        cost_budget="(1 + delta) * each case's own stage1 optimum + 1e-4 yuan"),
        input=dict(source="experiments/exp008/q1.py::EMBEDDED_CSV, 144 right-ended 10-minute intervals", array_sha256=input_hash),
        protected_hashes={str(FROZEN.relative_to(ROOT)): frozen_hash,
                          str(SELECTION.relative_to(ROOT)): selection_hash},
        source_hashes={"experiments/exp008/q1.py": source_hash,
                       str(Path(__file__).relative_to(ROOT)): sha(Path(__file__))},
        environment=dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__,
                         platform=platform.platform()),
        units=dict(delta="fraction", ramp="kW per 10 min", up="10-minute steps", down="10-minute steps",
                   eta_rt="round-trip efficiency fraction; one-way efficiency sqrt(eta_rt)",
                   capacity_factor="nominal 12000-kWh capacity multiplier; 5000-kW power unchanged"),
    )
    dump(OUT / "protocol.json", protocol)
    selected = [c for c in CASES if not args.cases or c[0] in args.cases]
    for case in selected:
        path = OUT / f"{case[0]}.json"
        if args.resume and path.exists():
            old = json.loads(path.read_text())
            if (old.get("input_sha256"), old.get("source_sha256"), old.get("frozen_sha256")) != (input_hash, source_hash, frozen_hash):
                raise ValueError(f"Resume provenance mismatch: {path}")
            print(f"reuse {case[0]}: {old['outcome']}", flush=True)
            continue
        print(f"solve {case[0]}", flush=True)
        result = run_case(case, data, args.time_limit)
        print(json.dumps(dict(name=case[0], outcome=result["outcome"],
                              seconds=result["elapsed_seconds"],
                              metrics=result.get("solution", {}).get("metrics", {})), ensure_ascii=False), flush=True)
    results = [json.loads((OUT / f"{c[0]}.json").read_text()) for c in CASES
               if (OUT / f"{c[0]}.json").exists()]
    summary = make_summary(results, frozen)
    unchanged = sha(FROZEN) == frozen_hash and sha(SELECTION) == selection_hash and sha(Path(q1.__file__)) == source_hash
    summary["protected_files_unchanged"] = unchanged
    dump(OUT / "summary.json", summary)
    if not unchanged:
        raise RuntimeError("Protected source or frozen files changed during run.")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("rows", "frozen_metrics")}, ensure_ascii=False), flush=True)
    if not summary["center_reproduction"]["passed"]:
        raise SystemExit("Center reproduction check did not pass; inspect summary before plotting.")


if __name__ == "__main__":
    main()
