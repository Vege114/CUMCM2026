"""One bounded monthly, decision-focused four-block purchase policy pilot.

Historical realized net paths are supervised training labels, available only
after the day has completed. Deployment never uses today's future actuals.
The inner objective and deployment both use the same causal greedy battery.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from experiments.exp008.closed_loop import objective
from experiments.exp008.controller_candidate import plan_inventory, execute_inventory, INITIAL_SOC
from experiments.problem2.exp003.data import Data, ROOT, EPOCH
from experiments.problem2.exp004.predict import ForecastStore
from experiments.problem2.tree_planning.risk import TreeResidualScenarios
from experiments.problem2.tree_planning.verify import battery_metrics, verify_arrays

OUT = ROOT / "data/results/exp008/policy_calibration"
HISTORICAL = ROOT / "data/results/exp008/controller_candidates/lp_q0.80_statebuffer500/dispatch_2.npz"
CONFIG = {"blocks": 4, "history_days": 28, "minimum_training_days": 14,
          "l2_yuan_per_squared_kwh": .5, "maxiter": 300,
          "coefficient_bound_kwh": 150., "quantile": .8, "seed": 42}
FEATURE_NAMES = ("intercept", "cnn_mean_net_div5000kw", "cnn_mean_pv_div5000kw",
    "past7_completed_cnn_net_error_div500kw", "yesterday_mean_load_div5000kw",
    "weekday_sin", "weekday_cos", "current_soc_center6000_div4800kwh")
BLOCK_INDEX = np.arange(144) // 36


def day_feature(data, store, day, initial_soc):
    """Eight dimensionless features; actual reads are before this issue."""
    issue = day * 144
    cnn = store.get(issue)
    ids = np.flatnonzero((store.origins + 144 <= issue) & (store.origins >= issue - 7 * 144))
    if len(ids):
        labels = data.actual[store.origins[ids, None] + np.arange(144), :2]
        difference = labels - store.values[ids]
        recent_error = float(np.mean(difference[:, :, 0] - difference[:, :, 1]))
    else:
        recent_error = 0.
    yesterday = data.actual[issue - 144:issue, 0]
    weekday = (day + 2) % 7
    value = np.array([1., np.mean(cnn[:, 0] - cnn[:, 1]) / 5000,
        np.mean(cnn[:, 1]) / 5000, recent_error / 500, np.mean(yesterday) / 5000,
        np.sin(2 * np.pi * weekday / 7), np.cos(2 * np.pi * weekday / 7),
        (initial_soc - 6000) / 4800])
    return value, {"day": day, "origin": issue, "max_feature_actual_index": issue - 1,
        "completed_error_origins": store.origins[ids].astype(int).tolist(),
        "feature": value.tolist(), "initial_soc": initial_soc}


def corrected_purchase(g0, weights, feature):
    delta_blocks = np.einsum("ij,j->i", weights, feature, optimize=False)
    raw = np.asarray(g0) + delta_blocks[BLOCK_INDEX]
    return np.maximum(0, raw), raw > 0, delta_blocks


def training_objective(flat_weights, x, g0, net_paths, prices, initial_soc, *, l2=.5):
    weights = np.asarray(flat_weights).reshape(4, 8)
    gradient = np.zeros_like(weights)
    bills = 0.
    for i in range(len(x)):
        purchase, active, _ = corrected_purchase(g0[i], weights, x[i])
        # No terminal credit, wear penalty or TV preference in the true bill.
        bill, grad_g = objective(purchase, net_paths[i:i+1], prices, initial_soc[i],
                                throughput=0., variation=0., terminal=0., deadband=0.)
        bills += bill
        grad_blocks = (grad_g * active).reshape(4, 36).sum(1)
        gradient += grad_blocks[:, None] * x[i][None, :]
    mean_bill = bills / len(x)
    penalty = l2 * float(np.square(weights).sum())
    gradient = gradient / len(x) + 2 * l2 * weights
    return mean_bill + penalty, gradient.ravel()


def train_month(month, data, store, historical):
    asof = int((pd.Timestamp(2025, month, 1) - EPOCH).days)
    train_days = np.arange(max(31, asof - CONFIG["history_days"]), asof)
    audit = {"month": month, "asof_day": asof, "information_cutoff_exclusive": asof * 144,
        "training_days": train_days.tolist(), "training_source": str(HISTORICAL.relative_to(ROOT)),
        "label_role": "previously_completed_realized_net_load_supervision",
        "historical_initial_states": "archived_reference_policy_actual_day_start",
        "training_controller": "greedy; historical statebuffer fees are not used as targets"}
    if len(train_days) < CONFIG["minimum_training_days"]:
        return np.zeros((4, 8)), {**audit, "frozen_zero": True,
            "reason": "fewer_than_14_completed_formal_days; no future or January pseudo-labels added"}
    indices = train_days - 31
    g0 = historical["original"][indices]
    initial_soc = historical["states"][indices, 0]
    x, feature_audits = [], []
    for day, soc in zip(train_days, initial_soc):
        feature, info = day_feature(data, store, int(day), float(soc))
        x.append(feature)
        feature_audits.append(info)
    x = np.stack(x)
    # Every historical target day ends before this model is trained.
    assert np.all((train_days + 1) * 144 <= asof * 144)
    actual = data.actual[train_days[:, None] * 144 + np.arange(144), :2]
    net = (actual[:, :, 0] - actual[:, :, 1]) / 6
    kwargs = dict(x=x, g0=g0, net_paths=net, prices=data.fixed_price, initial_soc=initial_soc,
                  l2=CONFIG["l2_yuan_per_squared_kwh"])
    started = time.perf_counter()
    initial_loss, _ = training_objective(np.zeros(32), **kwargs)
    result = minimize(lambda w: training_objective(w, **kwargs), np.zeros(32), jac=True,
        method="L-BFGS-B", bounds=[(-CONFIG["coefficient_bound_kwh"], CONFIG["coefficient_bound_kwh"])] * 32,
        options={"maxiter": CONFIG["maxiter"], "maxls": 20, "ftol": 1e-8, "gtol": 1e-5})
    if result.x is None or not np.isfinite(result.x).all():
        raise RuntimeError("policy training produced no finite coefficients")
    weights = result.x.reshape(4, 8)
    final_loss, _ = training_objective(result.x, **kwargs)
    penalty = CONFIG["l2_yuan_per_squared_kwh"] * float(np.square(weights).sum())
    audit.update(frozen_zero=False, training_last_label=asof * 144 - 1,
        feature_audits=feature_audits, weights_kwh=weights.tolist(),
        initial_mean_true_bill=initial_loss, final_mean_true_bill=final_loss - penalty,
        final_regularization_yuan_per_day=penalty, final_objective=final_loss,
        l2_units="yuan per (kWh-per-10min coefficient)^2 in mean-daily objective",
        training_seconds=time.perf_counter() - started, optimizer_success=bool(result.success),
        optimizer_message=str(result.message), iterations=int(result.nit), evaluations=int(result.nfev),
        global_optimality_certificate=False)
    return weights, audit


def gradient_and_feature_check(data, store):
    rng = np.random.default_rng(409)
    x = rng.normal(0, .3, (3, 8)); x[:, 0] = 1
    g0 = rng.uniform(100, 900, (3, 144))
    paths = rng.uniform(-300, 1100, (3, 144))
    soc = np.array([2300., 5600., 8900.])
    w = rng.normal(0, 2, 32)
    args = dict(x=x, g0=g0, net_paths=paths, prices=data.fixed_price, initial_soc=soc)
    _, analytic = training_objective(w, **args)
    checks = []
    for index in (0, 3, 7, 12, 20, 27):
        plus, minus = w.copy(), w.copy()
        step = 1e-4
        plus[index] += step; minus[index] -= step
        finite = (training_objective(plus, **args)[0] - training_objective(minus, **args)[0]) / (2 * step)
        checks.append({"index": index, "analytic": float(analytic[index]), "finite_difference": float(finite),
                       "absolute_error": float(abs(finite - analytic[index]))})
    if max(c["absolute_error"] for c in checks) > 2e-3:
        raise RuntimeError(f"block policy gradient check failed: {checks}")
    changed = Data()
    day = 75
    changed.actual[day * 144:] = changed.actual[day * 144:] * 17 + 999999
    first, _ = day_feature(data, store, day, 6200)
    second, _ = day_feature(changed, store, day, 6200)
    np.testing.assert_array_equal(first, second)
    return {"passed": True, "gradient_checks": checks, "feature_future_perturbation_passed": True}


def run(days=60, out=OUT):
    if days not in (30, 60):
        raise ValueError("bounded pilot supports 30 or 60 days")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "summary.json").exists():
        raise RuntimeError("pilot output already completed; use a fresh --out for new work")
    started = time.perf_counter()
    data, store = Data(), ForecastStore("no_season", seed=42)
    with np.load(HISTORICAL) as z:
        historical = {k: z[k].copy() for k in ("original", "states", "actual")}
    np.testing.assert_allclose(historical["actual"], data.actual[31 * 144:].reshape(334, 144, 2))
    protocol = {"config": CONFIG, "pilot_days": days, "features": FEATURE_NAMES,
        "blocks": ["00:00-06:00", "06:00-12:00", "12:00-18:00", "18:00-24:00"],
        "W_units": "kWh per ten-minute slot per dimensionless feature",
        "policy": "max(0, daily fresh q0.8 LP g0 + repeated block W*x)",
        "historical_archive_sha256": hashlib.sha256(HISTORICAL.read_bytes()).hexdigest(),
        "training_labels": "only whole days before monthly training issue",
        "deployment_controller": "same causal greedy battery for calibrated and zero-W baseline",
        "evaluation_role": "development pilot, not full-year target test",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    checks = gradient_and_feature_check(data, store)
    (out / "gradient_and_causality.json").write_text(json.dumps(checks, indent=2) + "\n")
    risk = TreeResidualScenarios(store.origins, store.values, data.actual, data.fixed_price)
    trained, training, records = {}, [], []
    details = {name: [] for name in ("zero_correction", "learned_blocks")}
    states = {name: INITIAL_SOC for name in details}
    modes = {name: 1 for name in details}
    audits = {name: [] for name in details}
    spec = {"quantile": .8, "controller": "greedy"}
    for day in range(31, 31 + days):
        date = EPOCH + pd.Timedelta(days=day)
        month = int(date.month)
        if month not in trained:
            weights, info = train_month(month, data, store, historical)
            trained[month] = weights
            training.append(info)
            (out / f"training_m{month:02d}.json").write_text(json.dumps(info, indent=2) + "\n")
            print(json.dumps({"month": month, "frozen_zero": info["frozen_zero"],
                "mean_training_bill_before": info.get("initial_mean_true_bill"),
                "mean_training_bill_after": info.get("final_mean_true_bill"),
                "training_seconds": info.get("training_seconds"),
                "iterations": info.get("iterations")}), flush=True)
        supports, risk_info = risk.for_day(day)
        for name in details:
            initial, old_mode = states[name], modes[name]
            plan = plan_inventory(supports, data.fixed_price, initial, spec, final=False)
            feature, feature_info = day_feature(data, store, day, initial)
            weights = np.zeros((4, 8)) if name == "zero_correction" else trained[month]
            g0 = plan["purchase"].copy()
            purchase, _, blocks = corrected_purchase(g0, weights, feature)
            plan["purchase"] = purchase
            actual = data.actual[day * 144:(day + 1) * 144]
            detail, new_mode = execute_inventory(plan, actual, data.fixed_price, initial, old_mode, spec)
            # Baseline LP intentions do not describe the corrected purchase;
            # only actual execution is a physical dispatch claim.
            detail = {k: v for k, v in detail.items() if not k.startswith("intended_")}
            detail["base_purchase"] = g0
            detail["purchase_correction"] = purchase - g0
            details[name].append(detail)
            states[name], modes[name] = float(detail["states"][-1]), new_mode
            row = {"policy": name, "day": day, "date": str(date.date()),
                "month": month, "initial_soc": initial, "final_soc": states[name],
                "total_cost": float(detail["fees"].sum()),
                "planned_cost": float(detail["fees"][:, 0].sum()),
                "emergency_cost": float(detail["fees"][:, 3].sum())}
            records.append(row)
            audits[name].append({**risk_info, **feature_info, "forecast_origin": day * 144,
                "execution_initial_soc": initial, "execution_initial_mode": old_mode,
                "execution_final_soc": states[name], "execution_final_mode": new_mode,
                "purchase_locked_before_actual_read": True, "training_month": month,
                "correction_blocks_kwh": blocks.tolist()})
        if day in (31, 58, 89, 90):
            print(json.dumps({"day": day, "date": str(date.date()), "daily": records[-2:]}), flush=True)
    summaries = []
    for name, parts in details.items():
        directory = out / name
        directory.mkdir(exist_ok=True)
        arrays = {k: np.stack([p[k] for p in parts]) for k in parts[0]}
        daily = [r for r in records if r["policy"] == name]
        # The generic verifier treats every extra array as a nonnegative
        # energy quantity. A procurement correction is signed, so verify its
        # reconstruction separately and pass only physical fields below.
        np.testing.assert_allclose(arrays["base_purchase"] + arrays["purchase_correction"],
                                   arrays["original"], atol=1e-9)
        assert np.isfinite(arrays["purchase_correction"]).all()
        physical = {k: v for k, v in arrays.items() if k not in ("base_purchase", "purchase_correction")}
        validation = verify_arrays(physical, daily, audits[name], expected_days=days,
            source_actual=data.actual[31 * 144:(31 + days) * 144].reshape(days, 144, 2), source_price=data.fixed_price)
        validation["signed_purchase_correction_reconstruction"] = True
        if not validation["passed"]:
            raise RuntimeError(validation["errors"])
        np.savez_compressed(directory / "dispatch_2.npz", **arrays)
        pd.DataFrame(daily).to_csv(directory / "daily.csv", index=False)
        (directory / "audit.json").write_text(json.dumps(audits[name], indent=2) + "\n")
        (directory / "verification.json").write_text(json.dumps(validation, indent=2) + "\n")
        summaries.append({"policy": name, "days": days, "total_cost": sum(r["total_cost"] for r in daily),
            "post_frozen_feb_cost": sum(r["total_cost"] for r in daily if r["month"] >= 3),
            **battery_metrics(arrays, 1, 172.75999999999976), "verified": True})
    pd.DataFrame(records).to_csv(out / "paired_daily.csv", index=False)
    summary = {"complete": True, "pilot_days": days, "eligible_for_annual_target": False,
        "policies": summaries, "training": [{k: v for k, v in t.items() if k != "feature_audits"} for t in training],
        "paired_cost_change_yuan": summaries[1]["total_cost"] - summaries[0]["total_cost"],
        "continue_gate": summaries[1]["total_cost"] < summaries[0]["total_cost"],
        "elapsed_seconds": time.perf_counter() - started}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    run(args.days, args.out)
