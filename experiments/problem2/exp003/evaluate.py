"""January-only cost calibration and resumable Question 2 evaluation.

Run after all 33 verified prediction checkpoints are available:
``python -m experiments.problem2.exp003.evaluate``. No model is trained here.
"""

import argparse
import hashlib
import json
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.common.neural_v2.physics import validate_detail

from .baseline_evidence import score_forecast
from .data import EPOCH, HERE, ROOT, STEPS, Data, midnight_origins
from .dispatch import aggregate, day_run, replay_days
from .provenance import sha256


def protocol():
    return json.loads((HERE / "protocol.json").read_text())


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def array_fingerprint(value):
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256(str((array.dtype.str, array.shape)).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def _npz(path, arrays):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def select_alpha(candidates, tolerance=.01):
    """Among costs within tolerance of the minimum, choose the simpler blend."""
    if not candidates or not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("finite nonnegative tolerance and at least one candidate are required")
    costs = np.asarray([candidate["cost"] for candidate in candidates], dtype=float)
    if not np.isfinite(costs).all() or (costs < 0).any():
        raise ValueError("candidate costs must be finite and nonnegative")
    eligible = [candidate for candidate in candidates if candidate["cost"] <= costs.min() + tolerance]
    return min(eligible, key=lambda candidate: (sum(candidate["alpha"]), *candidate["alpha"]))


def calibrate_candidates(data, store, initial_soc, cfg=None, replay=None):
    """Use the February checkpoint only on January 25–31; return no formal scores.

    Every blend starts at the same January 25 SOC. Its seven daily executions
    are continuous. Early stopping and this cost tuning share these labels,
    as explicitly recorded in the protocol; this is not nested validation.
    """
    cfg = protocol() if cfg is None else cfg
    calibration = cfg["calibration"]
    dates = pd.date_range(calibration["start"], calibration["end"])
    days = ((dates - EPOCH).days).to_numpy(dtype=int)
    if not np.array_equal(days, np.arange(24, 31)):
        raise ValueError("cost calibration is frozen to January 25–31")
    if calibration["seed"] != 42:
        raise ValueError("alpha selection must use seed 42 only")
    replay = replay_days if replay is None else replay
    candidates, daily = [], []
    for load_alpha, pv_alpha in product(calibration["alpha_load_candidates"],
                                       calibration["alpha_pv_candidates"]):
        alpha = (float(load_alpha), float(pv_alpha))
        candidate_id = f"load_{alpha[0]:g}_pv_{alpha[1]:g}"

        def predict(origin, alpha=alpha):
            if not 24 * STEPS <= origin < 31 * STEPS or origin % STEPS:
                raise ValueError("calibration predictions must be January validation midnights")
            return store.get(origin, alpha=alpha, validation_month=2)

        summaries, _, _ = replay(data, days, predict, initial_soc)
        if [row["day"] for row in summaries] != days.tolist():
            raise ValueError("calibration replay returned unexpected target days")
        result = aggregate(summaries)
        candidate = {"candidate_id": candidate_id, "alpha": list(alpha),
                     "cost": result["total_cost"], "initial_soc": initial_soc,
                     "final_soc": result["final_soc"], "daily_cvar90": result["daily_cvar90"]}
        candidates.append(candidate)
        daily.extend({**row, "candidate_id": candidate_id,
                      "alpha_load": alpha[0], "alpha_pv": alpha[1]} for row in summaries)
        print("CALIBRATION", candidate_id, result["total_cost"], flush=True)
    selected = select_alpha(candidates, calibration["tie_tolerance_yuan"])
    result = {
        "scenario": "2", "seed": 42, "validation_days": days.tolist(),
        "initial_soc": float(initial_soc), "selection_time": 31 * STEPS,
        "selected_alpha": selected["alpha"], "selected_cost": selected["cost"],
        "minimum_candidate_cost": min(row["cost"] for row in candidates),
        "tie_tolerance_yuan": calibration["tie_tolerance_yuan"],
        "tie_break": calibration["tie_break"], "candidates": candidates,
        "selection_note": "Shared January early-stopping/cost-tuning labels; seed 42 only; "
                          "alpha frozen before formal February–December evaluation for every seed",
    }
    return result, daily


def cases(selected_alpha, seeds=(42, 2026, 3407)):
    """Five named roles; one January-selected blend is shared by every seed."""
    alpha = [float(value) for value in selected_alpha]
    if len(alpha) != 2 or not np.isfinite(alpha).all() or not all(0 <= value <= 1 for value in alpha):
        raise ValueError("selected_alpha must contain two finite weights in [0, 1]")
    jobs = [{"name": "periodic", "seed": 42, "kind": "periodic", "alpha": [0., 0.]},
            {"name": "uncalibrated", "seed": 42, "kind": "mlp", "alpha": [1., 1.]}]
    jobs.extend({"name": "primary", "seed": int(seed), "kind": "mlp", "alpha": alpha.copy()}
                for seed in seeds)
    for job in jobs:
        job.update(scenario="2", case_id=f"{job['name']}_seed_{job['seed']}",
                   alpha_load=job["alpha"][0], alpha_pv=job["alpha"][1],
                   method="deterministic", weight=0., updates=0)
    return jobs


def source_manifest(data, run_id, cfg):
    """Hash source, config, inputs and actual model/prediction bytes, not timestamps."""
    source_names = ("data.py", "dispatch.py", "evaluate.py", "baseline_evidence.py",
                    "train.py", "predict.py", "protocol.json", "provenance.py")
    sources = {str((HERE / name).relative_to(ROOT)): sha256(HERE / name) for name in source_names}
    for name in ("physics.py", "risk.py"):
        path = ROOT / "experiments/common/neural_v2" / name
        sources[str(path.relative_to(ROOT))] = sha256(path)
    models = {}
    directory = HERE / "runs" / run_id
    for month, seed, suffix in product(range(2, 13), cfg["seed_list"], (".json", ".npz", ".keras")):
        path = directory / f"m{month:02d}_mlp_{seed}{suffix}"
        if not path.is_file():
            raise FileNotFoundError(f"complete verified training/prediction archives are required: {path}")
        models[str(path.relative_to(ROOT))] = sha256(path)
    manifest = {"code": sources, "data": data.hashes, "configuration": cfg, "models": models}
    return {**manifest, "signature": fingerprint(manifest)}


def cached_replay(data, days, predictor, initial_soc, directory, signature):
    """Resume only days whose inputs, physical state and saved output bytes agree."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    summaries, details, solvers = [], [], []
    state = float(initial_soc)
    for day in days:
        day = int(day)
        forecast = predictor(day * STEPS)
        forecast_hash = array_fingerprint(forecast)
        path = directory / f"d{day:03d}.json"
        arrays_path = path.with_suffix(".npz")
        if path.exists():
            saved = json.loads(path.read_text())
            if (saved["signature"] != signature or saved["forecast_sha256"] != forecast_hash
                    or abs(saved["summary"]["initial_soc"] - state) > 1e-6
                    or not arrays_path.is_file() or sha256(arrays_path) != saved["output_sha256"]):
                raise RuntimeError(f"stale or changed replay cache: {path}")
            summary, solver = saved["summary"], saved["solver"]
            with np.load(arrays_path, allow_pickle=False) as archive:
                detail = {key: archive[key].copy() for key in archive.files}
            validate_detail(detail)
            if summary["day"] != day or abs(detail["fees"].sum() - summary["total_cost"]) > 1e-5:
                raise RuntimeError(f"inconsistent replay summary: {path}")
            if (abs(detail["states"][0] - state) > 1e-6
                    or abs(detail["states"][-1] - summary["final_soc"]) > 1e-6):
                raise RuntimeError(f"inconsistent cached SOC: {path}")
            if not np.array_equal(detail["original"], detail["final"]):
                raise RuntimeError("Question 2 cache contains intraday plan adjustments")
        else:
            summary, detail, solver = day_run(data, forecast, day, state)
            _npz(arrays_path, detail)
            _json(path, {"signature": signature, "forecast_sha256": forecast_hash,
                         "summary": summary, "solver": solver, "output_sha256": sha256(arrays_path)})
        summaries.append(summary)
        details.append(detail)
        solvers.append(solver)
        state = summary["final_soc"]
        if day in (0, 30, 31, 58, 89, 119, 150, 180, 211, 242, 272, 303, 333, 364):
            print("REPLAY", directory.name, summary["date"], flush=True)
    aggregate(summaries)
    return summaries, {key: np.stack([row[key] for row in details]) for key in details[0]}, solvers


def forecast_scores(data, origins, values, job):
    origins = np.asarray(origins)
    if not np.array_equal(origins, midnight_origins()):
        raise ValueError("formal prediction scores require exactly 334 unique midnights")
    truth = data.actual[origins[:, None] + np.arange(STEPS)]
    values = np.asarray(values, dtype=float)
    if values.shape != truth.shape or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("formal predictions must be finite nonnegative arrays of shape (334, 144, 2)")
    months = np.asarray((EPOCH + pd.to_timedelta(origins * 10, unit="min")).month)
    role = {key: job[key] for key in ("name", "seed", "case_id", "alpha_load", "alpha_pv")}
    return [{**row, **role, "predictor_id": job["case_id"]}
            for row in score_forecast(values, truth, months)]


def run(run_id="exp003"):
    from .predict import ForecastStore

    if Path(run_id).name != run_id or run_id in (".", ".."):
        raise ValueError("run_id must be a single directory name")
    began = time.monotonic()
    data, cfg = Data(), protocol()
    sources = source_manifest(data, run_id, cfg)
    signature = sources["signature"]
    out = ROOT / "data/results" / run_id
    out.mkdir(parents=True, exist_ok=True)
    cache = HERE / "runs" / run_id / "evaluation" / signature[:16]
    _json(out / "evaluation_manifest.json", {"status": "running", "complete": False,
                                            "signature": signature})
    _json(out / "evaluation_sources.json", sources)
    warm_summaries, warm, _ = cached_replay(
        data, range(31), data.baseline, cfg["initial_soc_kwh"], cache / "warmup", signature,
    )
    _npz(out / "warmup_2.npz", warm)
    _json(out / "warmup_summary.json", {"signature": signature, "days": warm_summaries,
                                       "summary": aggregate(warm_summaries)})
    calibration_path = cache / "alpha_calibration.json"
    calibration_days = cache / "candidate_daily.csv"
    if calibration_path.exists():
        calibration = json.loads(calibration_path.read_text())
        expected = {key: value for key, value in calibration.items() if key != "calibration_signature"}
        if (calibration["signature"] != signature or fingerprint(expected) != calibration["calibration_signature"]
                or not calibration_days.is_file()
                or sha256(calibration_days) != calibration["candidate_daily_sha256"]):
            raise RuntimeError("calibration inputs or candidate records changed; refusing reuse")
    else:
        store = ForecastStore(data, seed=42, run_id=run_id)
        calibration, rows = calibrate_candidates(data, store, float(warm["states"][24, 0]), cfg)
        pd.DataFrame(rows).to_csv(calibration_days, index=False)
        calibration.update(signature=signature, candidate_daily_sha256=sha256(calibration_days))
        calibration["calibration_signature"] = fingerprint(calibration)
        _json(calibration_path, calibration)
    _json(out / "alpha_calibration.json", calibration)
    (out / "candidate_daily.csv").write_bytes(calibration_days.read_bytes())

    # Alpha is now fixed and archived. Only now create and score formal cases.
    jobs = cases(calibration["selected_alpha"], cfg["seed_list"])
    origins = midnight_origins()
    days = origins // STEPS
    initial = float(warm["states"][-1, -1])
    replay_signature = fingerprint({"inputs": signature,
                                    "calibration": calibration["calibration_signature"]})
    reused, daily, annual, solver_rows, scores, manifest_cases = {}, [], [], [], [], []
    forecasts = {"origins": origins}
    for job in jobs:
        store = ForecastStore(data, seed=job["seed"], run_id=run_id) if job["kind"] == "mlp" else None
        values = np.stack([data.baseline(int(origin)) if store is None
                           else store.get(int(origin), alpha=job["alpha"]) for origin in origins])
        forecasts[job["case_id"]] = values
        prediction_hash = array_fingerprint(values)
        key = fingerprint({"signature": replay_signature, "predictions": prediction_hash,
                           "initial_soc": initial})[:20]
        if key not in reused:
            lookup = {int(origin): value for origin, value in zip(origins, values)}
            reused[key] = cached_replay(data, days, lookup.__getitem__, initial,
                                        cache / key, replay_signature)
        summaries, detail, logs = reused[key]
        role = {key: value for key, value in job.items() if key != "alpha"}
        daily.extend({**row, **role, "cache_key": key} for row in summaries)
        annual.append({**aggregate(summaries), **role, "cache_key": key})
        solver_rows.extend({**row, **role, "solver_method": row["method"], "cache_key": key} for row in logs)
        scores.extend(forecast_scores(data, origins, values, job))
        path = out / f"dispatch_{job['case_id']}.npz"
        _npz(path, detail)
        if job["name"] == "primary" and job["seed"] == cfg["primary_seed"]:
            _npz(out / "dispatch_2.npz", detail)
        manifest_cases.append({**job, "cache_key": key, "prediction_content_sha256": prediction_hash,
                               "dispatch_file": path.name, "dispatch_sha256": sha256(path),
                               "days": len(summaries), "total_cost": annual[-1]["total_cost"]})
        print("CASE_DONE", job["case_id"], annual[-1]["total_cost"], flush=True)
    pd.DataFrame(daily).to_csv(out / "daily_metrics.csv", index=False)
    pd.DataFrame(annual).to_csv(out / "dispatch_metrics.csv", index=False)
    pd.DataFrame(solver_rows).to_csv(out / "solver_metrics.csv", index=False)
    frame = pd.DataFrame(scores)
    frame[frame.period == "annual"].to_csv(out / "forecast_annual.csv", index=False)
    frame[frame.period == "monthly"].to_csv(out / "forecast_monthly.csv", index=False)
    _npz(out / "evaluation_predictions.npz", forecasts)
    products = ("daily_metrics.csv", "dispatch_metrics.csv", "solver_metrics.csv", "forecast_annual.csv",
                "forecast_monthly.csv", "warmup_2.npz", "dispatch_2.npz", "alpha_calibration.json",
                "candidate_daily.csv", "evaluation_predictions.npz", "evaluation_sources.json")
    result = {"status": "complete", "complete": True, "signature": signature,
              "replay_signature": replay_signature,
              "calibration_signature": calibration["calibration_signature"],
              "selected_alpha": calibration["selected_alpha"], "cases": manifest_cases,
              "days_per_case": len(days), "intervals_per_case": len(days) * STEPS,
              "named_case_count": len(jobs), "unique_case_count": len(reused),
              "initial_soc": initial, "seconds": time.monotonic() - began,
              "output_hashes": {name: sha256(out / name) for name in products}}
    _json(out / "evaluation_manifest.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="exp003")
    arguments = parser.parse_args()
    run(arguments.run_id)
