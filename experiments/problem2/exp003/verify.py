"""Independently reconcile Q2 archives, physical identities and reported scores."""

import argparse
import json
from itertools import product

import numpy as np
import pandas as pd

from experiments.common.neural_v2.export import independent_check, read_npz

from .baseline_evidence import metric_row
from .data import ALLOWED_INPUTS, EPOCH, HERE, ROOT, Data, midnight_origins
from .evaluate import source_manifest
from .predict import ForecastStore, file_hash, protocol
from .provenance import verify as verify_frozen


def _score_keys(frame, case_ids, months, period):
    keys = ["case_id", "month", "target", "population"]
    expected = set(product(case_ids, months, ("load", "pv", "net_load"),
                           ("all", "pv_generating")))
    assert not frame.duplicated(keys).any(), f"duplicate {period} forecast score"
    assert set(frame[keys].itertuples(index=False, name=None)) == expected
    assert frame.period.eq(period).all() and frame.issue_hour.eq(0).all()
    assert frame.scenario.eq("2").all() and frame.unit.eq("kW").all()


def _forecast_check(values, truth, monthly, annual, months):
    additive = ("n", "absolute_error_sum", "squared_error_sum", "signed_error_sum",
                "actual_abs_sum")
    for score in pd.concat([annual, monthly], ignore_index=True).itertuples():
        if score.target == "net_load":
            predicted = values[..., 0] - values[..., 1]
            actual = truth[..., 0] - truth[..., 1]
        else:
            column = {"load": 0, "pv": 1}[score.target]
            predicted, actual = values[..., column], truth[..., column]
        mask = np.ones(actual.shape, bool)
        if score.month:
            mask &= (months == score.month)[:, None]
        if score.population == "pv_generating":
            mask &= truth[..., 1] > 0
        calculated = metric_row(predicted[mask], actual[mask])
        assert score.n == calculated["n"]
        for key in (*additive[1:], "mae", "rmse", "wape_pct", "bias"):
            expected = np.nan if calculated[key] is None else calculated[key]
            np.testing.assert_allclose(getattr(score, key), expected, rtol=1e-12,
                                       atol=1e-7, equal_nan=True, err_msg=key)
    for score in annual.itertuples():
        rows = monthly[(monthly.target == score.target)
                       & (monthly.population == score.population)]
        pooled = rows[list(additive)].sum()
        assert score.n == pooled["n"]
        for key in additive[1:]:
            np.testing.assert_allclose(getattr(score, key), pooled[key], rtol=1e-12,
                                       atol=1e-7, err_msg=f"monthly additivity: {key}")
        metrics = {
            "mae": pooled.absolute_error_sum / pooled.n,
            "rmse": np.sqrt(pooled.squared_error_sum / pooled.n),
            "bias": pooled.signed_error_sum / pooled.n,
            "wape_pct": (100 * pooled.absolute_error_sum / pooled.actual_abs_sum
                         if pooled.actual_abs_sum else np.nan),
        }
        for key, value in metrics.items():
            np.testing.assert_allclose(getattr(score, key), value, rtol=1e-12,
                                       atol=1e-8, equal_nan=True, err_msg=key)


def _dispatch_summary_check(detail, days, annual, solvers):
    derived = {
        "planned_kwh": detail["original"].sum(axis=1),
        "final_kwh": detail["final"].sum(axis=1),
        "emergency_kwh": detail["emergency"].sum(axis=1),
        "emergency_minutes": (detail["emergency"] > 1e-6).sum(axis=1) * 10,
        "charge_kwh": detail["charge"].sum(axis=1),
        "discharge_kwh": detail["discharge"].sum(axis=1),
        "surplus_kwh": detail["surplus"].sum(axis=1),
        "planned_cost": (detail["original"] * detail["price"]).sum(axis=1),
        "up_cost": np.zeros(len(days)),
        "down_cost": np.zeros(len(days)),
        "emergency_cost": (5 * detail["emergency"] * detail["price"]).sum(axis=1),
    }
    derived["total_cost"] = derived["planned_cost"] + derived["emergency_cost"]
    for key, value in derived.items():
        np.testing.assert_allclose(days[key], value, rtol=0, atol=1e-6, err_msg=key)
        np.testing.assert_allclose(annual[key], value.sum(), rtol=0, atol=1e-6,
                                   err_msg=f"annual {key}")
    costs = derived["total_cost"]
    tail_mass = len(costs) * .1
    weights = np.clip(tail_mass - np.arange(len(costs)), 0, 1)
    cvar = float(np.dot(np.sort(costs)[::-1], weights) / tail_mass)
    np.testing.assert_allclose(annual.daily_cvar90, cvar, rtol=0, atol=1e-6)
    worst = int(np.argmax(costs))
    np.testing.assert_allclose(annual.worst_day_cost, costs[worst], rtol=0, atol=1e-6)
    assert annual.worst_date == days.iloc[worst].date
    assert annual.days == len(days)
    assert annual.start_date == days.iloc[0].date and annual.end_date == days.iloc[-1].date
    np.testing.assert_allclose(annual.initial_soc, detail["states"][0, 0], rtol=0, atol=1e-6)
    np.testing.assert_allclose(annual.final_soc, detail["states"][-1, -1], rtol=0, atol=1e-6)
    np.testing.assert_array_equal(solvers.day, days.day)
    np.testing.assert_array_equal(solvers.origin, days.day.to_numpy() * 144)
    np.testing.assert_array_equal(solvers.information_cutoff, solvers.origin)
    assert solvers.fixed_tariff.eq(True).all()
    assert solvers.issued_forecast_allowed.eq(False).all()
    assert solvers.known_future_price.eq(False).all()
    assert solvers.solver_method.eq("deterministic_milp").all()
    assert np.isfinite(solvers.seconds).all() and solvers.seconds.ge(0).all()
    counters = {
        "solver_calls": np.ones(len(days), dtype=int),
        "timeout_count": solvers.status.eq(1).to_numpy(dtype=int),
        "fallback_count": solvers.fallback.to_numpy(dtype=int),
        "incumbent_count": np.zeros(len(days), dtype=int),
        "gap_certified_count": (solvers.mip_gap.notna()
                                & solvers.mip_gap.le(.010001)).to_numpy(dtype=int),
        "violations": np.zeros(len(days), dtype=int),
    }
    for key, values in counters.items():
        np.testing.assert_array_equal(days[key], values, err_msg=key)
        assert annual[key] == values.sum(), key
    assert np.isfinite(days.solve_execute_seconds).all()
    assert (days.solve_execute_seconds.to_numpy() + 1e-6 >= solvers.seconds.to_numpy()).all()
    np.testing.assert_allclose(annual.solve_execute_seconds, days.solve_execute_seconds.sum(),
                               rtol=0, atol=1e-6)


def run(run_id="exp003"):
    out = ROOT / "data/results" / run_id
    data, cfg = Data(), protocol()
    manifest = json.loads((out / "evaluation_manifest.json").read_text())
    assert manifest.get("complete") is True
    assert source_manifest(data, run_id, cfg)["signature"] == manifest["signature"]
    for name, digest in manifest["output_hashes"].items():
        assert file_hash(out / name) == digest, name
    assert set(data.hashes) == set(ALLOWED_INPUTS)
    origins = midnight_origins()
    np.testing.assert_array_equal(origins, np.arange(31, 365) * 144)
    warm = read_npz(out / "warmup_2.npz")
    warm_check = independent_check(warm, 31)
    np.testing.assert_allclose(warm["actual"], data.actual[:31*144].reshape(31, 144, 2), rtol=0, atol=0)
    assert warm["states"][0, 0] == 6000
    predictions = read_npz(out / "evaluation_predictions.npz")
    np.testing.assert_array_equal(predictions["origins"], origins)
    daily = pd.read_csv(out / "daily_metrics.csv", dtype={"scenario": str})
    annual = pd.read_csv(out / "dispatch_metrics.csv", dtype={"scenario": str})
    scores = pd.read_csv(out / "forecast_annual.csv", dtype={"scenario": str})
    monthly = pd.read_csv(out / "forecast_monthly.csv", dtype={"scenario": str})
    solvers = pd.read_csv(out / "solver_metrics.csv", dtype={"scenario": str})
    case_ids = {case["case_id"] for case in manifest["cases"]}
    expected_cases = {"periodic_seed_42", "uncalibrated_seed_42",
                      *(f"primary_seed_{seed}" for seed in cfg["seed_list"])}
    assert case_ids == expected_cases and len(manifest["cases"]) == len(case_ids)
    assert set(predictions) == {"origins", *case_ids}
    for frame, keys in ((daily, ["case_id", "day"]),
                        (annual, ["case_id"]), (solvers, ["case_id", "day"])):
        assert set(frame.case_id) == case_ids and not frame.duplicated(keys).any()
        assert frame.scenario.eq("2").all()
    _score_keys(scores, case_ids, [0], "annual")
    _score_keys(monthly, case_ids, range(2, 13), "monthly")
    months = np.asarray((EPOCH + pd.to_timedelta(origins * 10, unit="min")).month)
    truth = data.actual[31*144:].reshape(334, 144, 2)
    cases = []
    for case in manifest["cases"]:
        name = case["case_id"]
        assert name == f"{case['name']}_seed_{case['seed']}" and case["scenario"] == "2"
        if case["name"] == "periodic":
            assert case["kind"] == "periodic" and case["alpha"] == [0, 0]
        elif case["name"] == "uncalibrated":
            assert case["kind"] == "mlp" and case["alpha"] == [1, 1]
        else:
            assert case["name"] == "primary" and case["kind"] == "mlp"
        assert file_hash(out / case["dispatch_file"]) == case["dispatch_sha256"]
        detail = read_npz(out / case["dispatch_file"])
        check = independent_check(detail, 334)
        np.testing.assert_array_equal(detail["original"], detail["final"])
        np.testing.assert_array_equal(detail["actual"], truth)
        np.testing.assert_array_equal(detail["price"], np.tile(data.fixed_price, (334, 1)))
        np.testing.assert_allclose(detail["states"][0, 0], warm["states"][-1, -1], rtol=0, atol=1e-6)
        day_rows = daily[daily.case_id == name].sort_values("day")
        np.testing.assert_array_equal(day_rows.day, np.arange(31, 365))
        np.testing.assert_allclose(day_rows.total_cost, detail["fees"].sum((1, 2)), rtol=0, atol=1e-6)
        np.testing.assert_allclose(day_rows.initial_soc, detail["states"][:, 0], rtol=0, atol=1e-6)
        np.testing.assert_allclose(day_rows.final_soc, detail["states"][:, -1], rtol=0, atol=1e-6)
        row = annual[annual.case_id == name]
        assert len(row) == 1
        np.testing.assert_allclose(row.iloc[0].total_cost, check["total_cost"], rtol=0, atol=1e-6)
        _dispatch_summary_check(detail, day_rows, row.iloc[0],
                                solvers[solvers.case_id == name].sort_values("day"))
        values = predictions[name]
        assert values.shape == truth.shape and np.isfinite(values).all() and (values >= 0).all()
        if case["kind"] == "periodic":
            reconstructed = np.stack([data.baseline(int(origin)) for origin in origins])
        else:
            store = ForecastStore(data, case["seed"], run_id)
            reconstructed = np.stack([store.get(int(origin), alpha=case["alpha"])
                                      for origin in origins])
        np.testing.assert_array_equal(values, reconstructed)
        _forecast_check(values, truth, monthly[monthly.case_id == name],
                        scores[scores.case_id == name], months)
        cases.append({"case_id": name, **check})

    calibration = json.loads((out / "alpha_calibration.json").read_text())
    assert calibration["validation_days"] == list(range(24, 31))
    assert calibration["selection_time"] == 31*144 and calibration["seed"] == 42
    assert len(calibration["candidates"]) == 9
    candidate_days = pd.read_csv(out / "candidate_daily.csv")
    minimum = min(candidate["cost"] for candidate in calibration["candidates"])
    eligible = [candidate for candidate in calibration["candidates"] if candidate["cost"] <= minimum + .01]
    expected = min(eligible, key=lambda candidate: (sum(candidate["alpha"]), *candidate["alpha"]))
    assert calibration["selected_alpha"] == expected["alpha"] == manifest["selected_alpha"]
    for candidate in calibration["candidates"]:
        days = candidate_days[candidate_days.candidate_id == candidate["candidate_id"]].sort_values("day")
        np.testing.assert_array_equal(days.day, np.arange(24, 31))
        np.testing.assert_allclose(days.total_cost.sum(), candidate["cost"], rtol=0, atol=1e-6)
        np.testing.assert_allclose(days.iloc[0].initial_soc, warm["states"][24, 0], rtol=0, atol=1e-6)
    for case in manifest["cases"]:
        if case["name"] == "primary":
            assert case["alpha"] == calibration["selected_alpha"]

    training = []
    for seed in cfg["seed_list"]:
        store = ForecastStore(data, seed, run_id)
        for month in range(2, 13):
            store.month(month)  # verifies source, weights and prediction content fingerprints
            meta = json.loads((HERE / "runs" / run_id / f"m{month:02d}_mlp_{seed}.json").read_text())
            assert meta["train_latest_target_exclusive"] <= meta["train_cutoff"]
            assert meta["validation_latest_target_exclusive"] <= meta["asof"]
            assert meta["validation_issue_hours"] == [0]
            assert set(meta["data_hashes"]) == set(ALLOWED_INPUTS)
            assert "GPU:0" in meta["device"] and "GPU:0" in meta["reload_device"]
            assert meta["parameters"] == 2178
            assert meta["zero_initial_output"] and meta["weights_updated"] and meta["save_reload_passed"]
            training.append(meta)
    result = {
        "status": "passed", "scope": ["2"], "days": 334, "intervals": 48096,
        "warmup": warm_check, "case_checks": cases,
        "selected_alpha": calibration["selected_alpha"], "calibration_candidates": 9,
        "training_groups": len(training), "gpu_and_reload_all": True,
        "training_label_cutoffs_all": True, "q2_inputs_only": True,
        "forecast_scores_recomputed": True, "fees_and_physics_recomputed": True,
        "forecast_archives_match_monthly_checkpoints": True,
        "forecast_annual_monthly_keys_complete_unique": True,
        "forecast_monthly_additivity_verified": True,
        "fee_components_energy_tail_worst_day_recomputed": True,
        "solver_counters_reconciled": True,
        "source_fingerprints_current": True, "frozen_source": verify_frozen(),
    }
    (out / "full_year_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "intervals", "training_groups", "selected_alpha")}, ensure_ascii=False))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="exp003")
    run(**vars(parser.parse_args()))
