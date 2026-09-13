"""Four preselected frozen-purchase perfect-information execution diagnostics.

No oracle action is exported or made available to a causal controller. The
annual LP follows audit_baselines.fixed_purchase_oracle, with explicit
emergency/spill sign bounds and independent source/physical/billing checks.
"""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from experiments.exp008.verify import (
    BASELINE_COST, ETA, INITIAL_SOC, MAX_SOC, MIN_SOC, POWER_ENERGY,
    source_arrays, verify_arrays,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/results/exp008/fixed_purchase_bounds"
QUANTILES = (.5, .6, .7, .8)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n")


def solve_fixed_purchase(path, out, quantile):
    path, out = Path(path), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with np.load(path, allow_pickle=False) as archive:
        source = {key: archive[key].copy() for key in archive.files}
    actual, price144 = source_arrays("2")
    source_check = verify_arrays(source, source_actual=actual, source_price=price144)
    source_check.pop("goal", None)
    if not source_check["passed"]:
        raise AssertionError(source_check["errors"])
    purchase = source["original"].ravel()
    net = ((actual[..., 0]-actual[..., 1])/6).ravel()
    price = np.tile(price144, 334)
    n = len(net)
    assert n == 334*144 and purchase.shape == (n,)
    began = perf_counter()
    balance = purchase-net
    t = np.arange(n)
    c, d, e, w, s = (t+block*n for block in range(5))
    row = np.r_[t, t, t, t, n+t, n+t, n+t, n+t[1:]]
    col = np.r_[c, d, e, w, s, c, d, s[:-1]]
    coefficient = np.r_[-np.ones(n), np.ones(n), np.ones(n), -np.ones(n),
        np.ones(n), np.full(n, -ETA), np.full(n, 1/ETA), -np.ones(n-1)]
    matrix = coo_matrix((coefficient, (row, col)), shape=(2*n, 5*n)).tocsc()
    rhs = np.r_[-balance, INITIAL_SOC, np.zeros(n-1)]
    lower = np.r_[np.zeros(4*n), np.full(n, MIN_SOC)]
    # Frozen purchase fixes the surplus/deficit sign. The bounds structurally
    # prohibit emergency charging, surplus discharge and simultaneous flows.
    surplus, deficit = np.maximum(balance, 0), np.maximum(-balance, 0)
    upper = np.r_[np.minimum(surplus, POWER_ENERGY), np.minimum(deficit, POWER_ENERGY),
                  deficit, surplus, np.full(n, MAX_SOC)]
    objective = np.r_[np.zeros(2*n), 5*price, np.zeros(2*n)]
    solution = linprog(objective, A_eq=matrix, b_eq=rhs,
        bounds=np.column_stack((lower, upper)), method="highs",
        options={"primal_feasibility_tolerance": 1e-8, "dual_feasibility_tolerance": 1e-8})
    if not solution.success:
        raise RuntimeError(solution.message)
    x = solution.x
    dual = float(np.sum(rhs*solution.eqlin.marginals)
                 + np.sum(lower*solution.lower.marginals)
                 + np.sum(upper*solution.upper.marginals))
    stationarity = float(np.abs(objective-matrix.T@solution.eqlin.marginals
                    - solution.lower.marginals-solution.upper.marginals).max())
    trace = np.r_[INITIAL_SOC, x[s]]
    original = purchase.reshape(334, 144)
    prices = price.reshape(334, 144)
    detail = {"original": original, "final": original.copy(),
        "charge": x[c].reshape(334, 144), "discharge": x[d].reshape(334, 144),
        "emergency": x[e].reshape(334, 144), "surplus": x[w].reshape(334, 144),
        "states": np.stack([trace[day*144:(day+1)*144+1] for day in range(334)]),
        "fees": np.stack([original*prices, np.zeros_like(original), np.zeros_like(original),
                           5*prices*x[e].reshape(334, 144)], axis=-1),
        "price": prices, "actual": actual.copy(), "days": np.arange(31, 365)}
    verification = verify_arrays(detail, source_actual=actual, source_price=price144)
    verification.pop("goal", None)
    assert verification["passed"], verification["errors"]
    assert not np.any((x[c] > 1e-6) & (balance < -1e-6))
    assert not np.any((x[d] > 1e-6) & (balance > 1e-6))
    planned = float((purchase*price).sum())
    total = planned+float(solution.fun)
    report = {
        "quantile": quantile, "source_fixed_purchase": str(path.relative_to(ROOT)),
        "source_sha256": sha256(path), "source_verified": True,
        "information_regime": "frozen causal historical purchases; battery sees all future actual net demand",
        "oracle_uses_future_actual": True, "eligible_as_causal_strategy": False,
        "use": "diagnostic lower bound only; no oracle actions exported or reused for policy selection",
        "days": 334, "initial_soc": INITIAL_SOC,
        "cross_day_soc": "single continuous annual trajectory",
        "terminal_soc": "physical lower bound only, consistent with annual candidate billing",
        "emergency_charging_allowed": False, "discharge_into_surplus_allowed": False,
        "simultaneous_charge_discharge_allowed": False,
        "deadband_ramp_mode_constraints": "relaxed; optimistic for any low-switch causal controller",
        "frozen_planned_cost_yuan": planned,
        "minimum_emergency_cost_yuan": float(solution.fun),
        "minimum_total_cost_yuan": total,
        "emergency_dual_objective_yuan": dual,
        "total_dual_bound_yuan": planned+dual,
        "primal_dual_gap_yuan": abs(float(solution.fun)-dual),
        "dual_stationarity_max_error": stationarity,
        "max_constraint_residual_kwh": float(np.abs(matrix@x-rhs).max()),
        "candidate_actual_total_cost_yuan": source_check["billing"]["total_cost"],
        "candidate_actual_emergency_cost_yuan": source_check["billing"]["emergency_cost"],
        "maximum_saving_by_execution_only_yuan": source_check["billing"]["total_cost"]-total,
        "cost_reduction_vs_exp006_pct": 100*(1-total/BASELINE_COST),
        "execution_only_8pct_not_ruled_out": bool(planned+dual <= .92*BASELINE_COST),
        "execution_only_10pct_not_ruled_out": bool(planned+dual <= .9*BASELINE_COST),
        "gap_to_8pct_target_yuan": total-.92*BASELINE_COST,
        "gap_to_10pct_target_yuan": total-.9*BASELINE_COST,
        "battery_metrics": verification["battery_metrics"],
        "solver": "scipy linprog HiGHS continuous annual LP",
        "solver_status": int(solution.status), "solver_message": str(solution.message),
        "solve_and_verify_seconds": perf_counter()-began,
    }
    write_json(out/"source_verification.json", source_check)
    write_json(out/"oracle_verification.json", verification)
    write_json(out/"bound.json", report)
    print(json.dumps({k: report[k] for k in ("quantile", "frozen_planned_cost_yuan",
          "minimum_emergency_cost_yuan", "minimum_total_cost_yuan", "primal_dual_gap_yuan",
          "gap_to_8pct_target_yuan")}), flush=True)
    return report


def run(out=OUT):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    protected = ROOT/"data/results/exp008/fixed_purchase_oracle.json"
    sources = [ROOT/f"data/results/exp008/controller_candidates/lp_q{q:.2f}_greedy/dispatch_2.npz"
               for q in QUANTILES]
    protocol = {"quantiles": list(QUANTILES),
        "selection_rule": "preselected named lp_q{q:.2f}_greedy archives; no selection on future oracle outcome",
        "source_files": [str(path.relative_to(ROOT)) for path in sources],
        "source_hashes": [sha256(path) for path in sources],
        "script_sha256": sha256(__file__),
        "protected_original_audit": str(protected.relative_to(ROOT)),
        "protected_original_audit_sha256": sha256(protected),
        "oracle_not_a_strategy": True, "maximum_configurations": 4}
    if (out/"protocol.json").exists():
        raise RuntimeError("Refusing to overwrite an existing oracle diagnostic; choose a fresh output")
    write_json(out/"protocol.json", protocol)
    reports = []
    for quantile, path in zip(QUANTILES, sources):
        reports.append(solve_fixed_purchase(path, out/f"q{quantile:.2f}", quantile))
    assert sha256(protected) == protocol["protected_original_audit_sha256"]
    scalar_rows = [{k: v for k, v in row.items() if isinstance(v, (str, int, float, bool))}
                   for row in reports]
    pd.DataFrame(scalar_rows).to_csv(out/"comparison.csv", index=False)
    write_json(out/"summary.json", {"complete": True, "protected_original_audit_unchanged": True,
        "oracle_action_traces_exported": False, "bounds": reports})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    run(args.out)
