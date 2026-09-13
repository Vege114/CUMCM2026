"""Bounded causal residual-window ablation; forecast, LP and execution fixed.

The only new settings are tree7, tree14 and per_slot14.  The tree28 control
is replayed because the existing ridge28 q=.8 archive has no state buffer.
This is development on 2025, not an independent validation sample.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

from experiments.exp008.controller_candidate import (
    INITIAL_SOC, execute_inventory, plan_inventory,
)
from experiments.exp008.forecast_calibration import CalibratedStore
from experiments.exp008.verify import verify_arrays
from experiments.problem2.exp003.data import Data
from experiments.problem2.tree_planning import risk as original_risk
from experiments.problem2.tree_planning.risk import (
    DT_HOURS, FEATURE_NAMES, QUANTILE_LEVELS, STEPS, TreeResidualScenarios,
    periodic_baseline, tree_features,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/results/exp008/risk_window"
CONFIGURATIONS = (("tree7", 7, "tree"), ("tree14", 14, "tree"),
                  ("per_slot14", 14, "per_slot"))
SPEC = {"calibration": "ridge_28", "quantile": .8, "controller": "greedy",
        "state_buffer": 500.}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


class WindowResidualScenarios(TreeResidualScenarios):
    """Original tree algorithm with a constructor-level history-day limit.

    ``for_day`` copies the original implementation rather than modifying a
    shared module global.  Tree hyperparameters, quantile interpolation and
    periodic fallback are identical.  Every actual read uses inherited
    ``_completed_day`` and its exclusive issue cutoff.
    """

    def __init__(self, *args, history_days=28, **kwargs):
        super().__init__(*args, **kwargs)
        if history_days not in (7, 14, 28):
            raise ValueError("This ablation only permits 7, 14 or control 28 days")
        self.history_days = int(history_days)

    def for_day(self, day):
        if isinstance(day, (bool, np.bool_)) or not isinstance(day, (int, np.integer)):
            raise TypeError("day must be a zero-based integer day index")
        day = int(day)
        cutoff = day * STEPS
        if cutoff not in self._lookup:
            raise ValueError("requested day has no frozen forecast")
        historical_ids = np.flatnonzero(self.origins + STEPS <= cutoff)[-self.history_days:]
        fallback = len(historical_ids) < 2
        features, residuals, training_days = [], [], []
        if fallback:
            for historical_day in range(max(1, day-self.history_days), day):
                issue_cutoff = historical_day * STEPS
                previous = self._completed_day(historical_day-1, issue_cutoff)
                weekly = (self._completed_day(historical_day-7, issue_cutoff)
                          if historical_day >= 7 else None)
                predicted = periodic_baseline(previous, weekly)
                observed = self._completed_day(historical_day, cutoff)
                features.append(tree_features(predicted, self.price144))
                residuals.append(((observed[:, 0]-observed[:, 1])
                                  - (predicted[:, 0]-predicted[:, 1])) * DT_HOURS)
                training_days.append(historical_day)
        else:
            for index in historical_ids:
                historical_day = int(self.origins[index] // STEPS)
                predicted = self.values[index]
                observed = self._completed_day(historical_day, cutoff)
                features.append(tree_features(predicted, self.price144))
                residuals.append(((observed[:, 0]-observed[:, 1])
                                  - (predicted[:, 0]-predicted[:, 1])) * DT_HOURS)
                training_days.append(historical_day)
        if not training_days:
            raise ValueError("no completed calibration day exists")
        x_train, y_train = np.concatenate(features), np.concatenate(residuals)
        current_forecast = self.values[self._lookup[cutoff]]
        x_current = tree_features(current_forecast, self.price144)
        tree, fit_seconds = None, 0.
        if self.conditioning == "tree":
            tree = DecisionTreeRegressor(max_depth=5, min_samples_leaf=48, random_state=42)
            began = perf_counter()
            tree.fit(x_train, y_train)
            fit_seconds = perf_counter()-began
            train_leaves = tree.apply(x_train)
            leaf_supports = {
                int(leaf): np.quantile(y_train[train_leaves == leaf], QUANTILE_LEVELS,
                                       method="linear") for leaf in np.unique(train_leaves)
            }
            errors = np.vstack([leaf_supports[int(leaf)] for leaf in tree.apply(x_current)])
        else:
            errors = np.quantile(np.stack(residuals), QUANTILE_LEVELS,
                                 axis=0, method="linear").T
        support = x_current[:, 2, None] + errors
        info = {
            "day": day, "cutoff": cutoff, "information_cutoff": cutoff,
            "cutoff_is_exclusive": True, "training_start_day": min(training_days),
            "training_end_day": max(training_days), "training_days": len(training_days),
            "training_origins": [int(value*STEPS) for value in training_days],
            "sample_count": len(y_train), "issued_history_days": len(historical_ids),
            "max_observed_index": int((max(training_days)+1)*STEPS-1),
            "fit_seconds": float(fit_seconds), "tree_nodes": int(tree.tree_.node_count) if tree else 0,
            "tree_leaves": int(tree.tree_.n_leaves) if tree else 0,
            "conditioning": self.conditioning, "history_days_limit": self.history_days,
            "fallback": bool(fallback),
            "residual_source": "periodic_baseline" if fallback else "ridge28_prequential_forecast",
            "forecast_calibration": "ridge_28", "base_risk_cache_used": False,
            "feature_names": list(FEATURE_NAMES), "residual_units": "kWh_per_10min",
            "scenario_units": "kWh_per_10min",
            "scenario_semantics": "conditional_slot_marginals_not_joint_daily_paths",
            "quantile_levels": QUANTILE_LEVELS.tolist(), "weights": self.weights.tolist(),
            "max_depth": 5, "min_samples_leaf": 48, "random_state": 42,
        }
        return support, info


def implementation_check(data, store):
    common = (store.origins, store.values, data.actual, data.fixed_price)
    control = WindowResidualScenarios(*common)
    reference = TreeResidualScenarios(*common)
    days = (31, 32, 33, 45, 90, 180, 364)
    for day in days:
        np.testing.assert_array_equal(control.for_day(day)[0], reference.for_day(day)[0])
    for name, window, conditioning in CONFIGURATIONS:
        risk = WindowResidualScenarios(*common, history_days=window, conditioning=conditioning)
        for day in (31, 45, 180):
            support, audit = risk.for_day(day)
            changed_actual = data.actual.copy()
            changed_actual[day*144:] = 1e7
            changed_forecast = store.values.copy()
            changed_forecast[store.origins > day*144] = 2e7
            changed = WindowResidualScenarios(store.origins, changed_forecast, changed_actual,
                data.fixed_price, history_days=window, conditioning=conditioning)
            np.testing.assert_array_equal(support, changed.for_day(day)[0])
            assert audit["training_days"] <= window
            assert audit["max_observed_index"] < day*144
            assert all(origin+144 <= day*144 for origin in audit["training_origins"])
    return {"passed": True, "tree28_exact_original_days": list(days),
            "future_actual_and_future_forecast_mutation_days": [31, 45, 180],
            "mutated_configurations": [item[0] for item in CONFIGURATIONS]}


def distribution_metrics(supports, truth, price):
    """Post-release diagnostics only; never used to form a day's support."""
    levels = QUANTILE_LEVELS
    errors = truth[..., None]-supports
    pinball = np.maximum(levels*errors, (levels-1)*errors)
    threshold = np.quantile(supports, .8, axis=2)
    peak = np.broadcast_to(price >= np.quantile(price, .75), truth.shape)
    coverage = (truth[..., None] <= supports).mean(axis=(0, 1))
    return {
        "support_quantile_levels": levels.tolist(), "empirical_coverage": coverage.tolist(),
        "mean_absolute_coverage_error": float(np.abs(coverage-levels).mean()),
        "mean_pinball_kwh": float(pinball.mean()),
        "price_weighted_pinball_yuan": float((pinball*price[None, :, None]).mean()),
        "central_support_range_coverage": float(((truth >= supports[..., 0])
                                                 & (truth <= supports[..., -1])).mean()),
        "central_support_range_nominal": float(levels[-1]-levels[0]),
        "central_support_range_mean_width_kwh": float((supports[..., -1]-supports[..., 0]).mean()),
        "lp_q080_coverage": float((truth <= threshold).mean()),
        "lp_q080_high_tariff_coverage": float((truth <= threshold)[peak].mean()),
        "lp_q080_mean_predicted_minus_actual_kwh": float((threshold-truth).mean()),
        "lp_q080_interpolated_underlying_level": float(np.quantile(levels, .8)),
        "lp_q080_note": "LP takes numpy quantile .8 over nine equally weighted support atoms; preserved exactly",
    }


def replay(name, window, conditioning, days, data, store, out):
    directory = out / f"{name}_{days}days"
    completed = directory / "summary.json"
    if completed.exists():
        return json.loads(completed.read_text())
    directory.mkdir(parents=True, exist_ok=True)
    risk = WindowResidualScenarios(store.origins, store.values, data.actual, data.fixed_price,
                                   history_days=window, conditioning=conditioning)
    soc, mode = INITIAL_SOC, 1
    parts, rows, audits, all_supports = [], [], [], []
    began = perf_counter()
    for day in range(31, 31+days):
        support, audit = risk.for_day(day)
        plan = plan_inventory(support, data.fixed_price, soc, SPEC, final=day == 364)
        observed = data.actual[day*144:(day+1)*144]
        detail, mode = execute_inventory(plan, observed, data.fixed_price, soc, mode, SPEC)
        audit.update(forecast_origin=day*144, purchase_locked_before_actual_read=True)
        parts.append(detail)
        all_supports.append(support)
        audits.append(audit)
        net = (observed[:, 0]-observed[:, 1])/6
        coverage = (net <= np.quantile(support, .8, axis=1)).mean()
        rows.append({"day": day, "date": str((pd.Timestamp("2025-01-01")+pd.Timedelta(days=day)).date()),
            "total_cost": float(detail["fees"].sum()), "planned_cost": float(detail["fees"][:, 0].sum()),
            "emergency_cost": float(detail["fees"][:, 3].sum()),
            "emergency_kwh": float(detail["emergency"].sum()), "lp_q080_coverage": float(coverage),
            "initial_soc": soc, "final_soc": float(detail["states"][-1])})
        soc = rows[-1]["final_soc"]
    arrays = {key: np.stack([part[key] for part in parts]) for key in parts[0]}
    arrays["days"] = np.arange(31, 31+days)
    supports = np.stack(all_supports)
    truth = (arrays["actual"][..., 0]-arrays["actual"][..., 1])/6
    verification = verify_arrays(arrays, expected_days=days,
        source_actual=data.actual[31*144:(31+days)*144].reshape(days, 144, 2),
        source_price=data.fixed_price, audit_records=audits)
    if not verification["passed"]:
        raise AssertionError(verification["errors"])
    summary = {"name": name, "history_days": window, "conditioning": conditioning,
        "fixed_spec": SPEC, "days": days, "start_day": 31,
        **verification["billing"], "battery": verification["battery_metrics"],
        "distribution": distribution_metrics(supports, truth, data.fixed_price),
        "verified": True, "elapsed_seconds": perf_counter()-began,
        "evaluation_role": "2025 development; fixed first 30 days gate before any annual extension"}
    np.savez_compressed(directory/"dispatch_2.npz", **arrays)
    np.savez_compressed(directory/"supports.npz", supports=supports, days=arrays["days"])
    pd.DataFrame(rows).to_csv(directory/"daily.csv", index=False)
    write_json(directory/"audit.json", audits)
    write_json(directory/"verification.json", verification)
    write_json(completed, summary)
    print(json.dumps({"name": name, "days": days, "cost": summary["total_cost"],
                      "emergency_cost": summary["emergency_cost"],
                      "lp_q080_coverage": summary["distribution"]["lp_q080_coverage"]}), flush=True)
    return summary


def run(out=OUT, extend=False):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    data, store = Data(), CalibratedStore("ridge_28")
    protocol = {"fixed_spec": SPEC, "control": ["tree28", 28, "tree"],
        "new_configurations": [list(item) for item in CONFIGURATIONS],
        "pilot": {"first_day": "2025-02-01", "last_day": "2025-03-02", "days": 30},
        "annual_gate": "at least 0.5% lower real pilot bill after adding an upper-bound cost for any lower final SOC; physics must pass",
        "final_soc_charge": "max(0, control_final_soc - candidate_final_soc) * sqrt(.9) * 5 * max(fixed_price)",
        "no_current_day_truth_window_selection": True,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "original_risk_sha256": hashlib.sha256(Path(original_risk.__file__).read_bytes()).hexdigest(),
        "controller_sha256": hashlib.sha256((ROOT/"experiments/exp008/controller_candidate.py").read_bytes()).hexdigest(),
        "forecast_values_sha256": hashlib.sha256(store.values.tobytes()).hexdigest(),
        "fixed_controls_note": "ridge28 q=.8 buffer500 control replayed; existing ridge28 q=.8 annual archive has no state buffer"}
    if (out/"protocol.json").exists():
        if json.loads((out/"protocol.json").read_text()) != protocol:
            raise RuntimeError("Refusing to overwrite changed risk-window protocol")
    else:
        write_json(out/"protocol.json", protocol)
    write_json(out/"implementation_verification.json", implementation_check(data, store))
    baseline = replay("tree28", 28, "tree", 30, data, store, out)
    pilots, eligible = [], []
    for name, window, conditioning in CONFIGURATIONS:
        result = replay(name, window, conditioning, 30, data, store, out)
        soc_cost = max(0, baseline["battery"]["final_soc"]-result["battery"]["final_soc"])*np.sqrt(.9)*5*data.fixed_price.max()
        adjusted_saving = baseline["total_cost"]-result["total_cost"]-soc_cost
        gate = bool(result["verified"] and adjusted_saving >= .005*baseline["total_cost"])
        pilots.append({"name": name, "total_cost": result["total_cost"],
            "delta_cost_vs_control": result["total_cost"]-baseline["total_cost"],
            "emergency_cost": result["emergency_cost"], "final_soc_depletion_upper_bound_cost": float(soc_cost),
            "conservatively_adjusted_saving": float(adjusted_saving), "annual_gate": gate,
            "distribution": result["distribution"], "battery": result["battery"]})
        if gate:
            eligible.append((name, window, conditioning))
    annual = []
    if extend and eligible:
        annual.append(replay("tree28", 28, "tree", 334, data, store, out))
        for args in eligible:
            annual.append(replay(*args, 334, data, store, out))
    summary = {"control": baseline, "pilot_candidates": pilots,
               "annual_eligible": [item[0] for item in eligible],
               "annual_completed": [row["name"] for row in annual]}
    write_json(out/"summary.json", summary)
    print(json.dumps({"annual_eligible": summary["annual_eligible"],
                      "annual_completed": summary["annual_completed"]}), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--extend-if-improved", action="store_true")
    arguments = parser.parse_args()
    run(arguments.out, arguments.extend_if_improved)
