"""Compare all seeds, choose models using past validation, and replay causally."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import EPOCH, ROOT, TARGETS, Data, month_origins
from .dispatch import day_run

VARIANTS = ("mlp", "gru", "tcn", "gru_no_calendar", "gru_one_day")
SEEDS = (42, 2026, 3407)
SCENARIOS = ("2", "3", "4-2", "4-3")


class ForecastStore:
    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)
        self.cache = {}

    def pack(self, month, variant, seed=None):
        key = month, variant, seed
        if key not in self.cache:
            if seed is None:
                parts = [self.pack(month, variant, s) for s in SEEDS]
                p = {**parts[0], "predictions": np.mean([d["predictions"] for d in parts], 0)}
            else:
                with np.load(self.run_dir / f"m{month:02d}_{variant}_{seed}.npz") as f:
                    p = {k: f[k].copy() for k in f.files}
                meta = json.loads((self.run_dir / f"m{month:02d}_{variant}_{seed}.json").read_text())
                p["training_signature"] = meta["signature"]
                assert meta["asof"] == int(month_origins(month)[0])
                nv = int(p["validation_count"])
                assert np.all(p["origins"][:nv] + 144 <= meta["asof"])
                assert meta["train_cutoff"] <= int(p["origins"][0])
            p["lookup"] = {int(o): i for i, o in enumerate(p["origins"])}
            self.cache[key] = p
        return self.cache[key]

    def get(self, origin, variant, seed=None, month=None):
        month = month or int((EPOCH + pd.Timedelta(minutes=int(origin) * 10)).month)
        p = self.pack(month, variant, seed)
        if origin in p["lookup"]:
            return p["predictions"][p["lookup"][origin]]
        # The last validation day's intraday forecasts have incomplete 24-hour
        # labels at month start. They are inference-only, never early-stop labels.
        if seed is None:
            return np.mean([self.get(origin, variant, s, month) for s in SEEDS], 0)
        tail = self.run_dir / f"m{month:02d}_{variant}_{seed}.validation-tail.npz"
        with np.load(tail) as f:
            assert f["training_signature"].item() == p["training_signature"]
            position = np.flatnonzero(f["origins"] == origin)
            assert len(position) == 1
            return f["predictions"][position[0]]


def warmup_baseline(data, origin, days=1):
    prediction = data.baseline(origin, days)
    if origin < 144:
        target = origin + np.arange(144)
        unknown = target - days * 144 < 0
        reference = data.reference[target % 144][:, [1, 2, 0]]
        prediction[unknown, :3] = reference[unknown]
    return prediction


def scores(y, prediction):
    delta = prediction.astype(float) - y
    absolute = np.abs(delta)
    denominator = float(np.abs(y).sum())
    return {"n": len(y), "mae": float(absolute.mean()),
            "rmse": float(np.sqrt(np.mean(delta**2))),
            "wape_pct": float(100 * absolute.sum() / denominator) if denominator else None,
            "bias": float(delta.mean()), "absolute_error_sum": float(absolute.sum()),
            "squared_error_sum": float(np.square(delta).sum()),
            "actual_abs_sum": denominator}


def forecast_scores(data, origins, predictions, variant, seed, month):
    ids = origins[:, None] + np.arange(144)[None, :]
    valid = ids < len(data.actual)
    truth = data.actual[np.minimum(ids, len(data.actual) - 1)]
    truth = np.concatenate((truth, truth[:, :, 1:2]), -1)
    rows = []
    for channel, target in enumerate(TARGETS):
        if target == "pv_corrected" and variant == "weekly":
            continue  # This channel is the same issued forecast in both baselines.
        label_variant = "issued" if target == "pv_corrected" and variant == "yesterday" else variant
        for lead, left, right in (("all", 0, 144), ("0–6h", 0, 36), ("6–12h", 36, 72),
                                  ("12–18h", 72, 108), ("18–24h", 108, 144)):
            mask = valid.copy(); mask[:, :left] = False; mask[:, right:] = False
            for population in (("all", "generating") if target.startswith("pv") else ("all",)):
                take = mask & (truth[:, :, channel] > 0) if population == "generating" else mask
                if not take.any():
                    continue
                r = scores(truth[:, :, channel][take], predictions[:, :, channel][take])
                r.update(variant=label_variant, seed=seed, month=month, target=target,
                         lead=lead, population=population, unit="元/千瓦时" if target == "price" else "千瓦")
                rows.append(r)
    return rows


def baseline_replay(data, scenario, days=1):
    state = 6000.0
    summaries, details, initial = [], [], []
    provider = lambda o: warmup_baseline(data, o, days)
    for day in range(365):
        initial.append(state)
        s, d = day_run(data, day, provider, scenario, initial=state)
        state = s["final_soc"]
        summaries.append(s); details.append(d)
    return summaries, details, initial


def choose_models(data, store, baseline_states, months=range(2, 13)):
    rows = []
    choices = {}
    for month in months:
        first_day = int(month_origins(month)[0] // 144)
        for scenario in SCENARIOS:
            candidates = []
            for variant in ("mlp", "gru", "tcn"):
                state = baseline_states[scenario][first_day - 7]
                total = 0
                provider = lambda o, v=variant, m=month: store.get(o, v, month=m)
                for day in range(first_day - 7, first_day):
                    r, _ = day_run(data, day, provider, scenario, initial=state)
                    state = r["final_soc"]
                    total += r["total_cost"]
                candidates.append((total, variant))
            best = min(candidates)[1]
            choices[(month, scenario)] = best
            for cost, variant in candidates:
                rows.append({"month": month, "scenario": scenario, "variant": variant,
                             "validation_cost": cost, "selected": variant == best,
                             "validation_start": str((EPOCH + pd.Timedelta(days=first_day - 7)).date()),
                             "decision_time": str(EPOCH + pd.Timedelta(days=first_day)),
                             "initial_soc": baseline_states[scenario][first_day - 7]})
        print(f"SELECTION month={month} {[choices[(month, s)] for s in SCENARIOS]}", flush=True)
    return choices, rows


def replay(data, store, choices, scenario, variant="selected", seed=None,
           update_hours=(6, 12, 18), known_price=False, corrected=True):
    state = 6000.0
    summaries, details = [], []
    for day in range(365):
        month = int((EPOCH + pd.Timedelta(days=day)).month)
        model = choices[(month, scenario)] if variant == "selected" and month > 1 else variant
        provider = (lambda o: warmup_baseline(data, o)) if month == 1 else (
            lambda o, v=model, m=month: store.get(o, v, seed, month=m))
        r, d = day_run(data, day, provider, scenario, initial=state, update_hours=update_hours,
                       known_price=known_price, corrected=corrected)
        state = r["final_soc"]
        if day >= 31:
            r.update(variant=variant, seed="mean" if seed is None else str(seed), model_used=model,
                     known_price=known_price, corrected=corrected,
                     update_schedule="0" + "".join(f"+{h}" for h in update_hours)
                     if scenario in ("3", "4-3") else "0")
            summaries.append(r); details.append(d)
    return summaries, details


def save_details(path, details):
    np.savez_compressed(path, **{k: np.stack([d[k] for d in details]) for k in details[0]})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="exp001")
    args = parser.parse_args()
    data = Data()
    run_dir = ROOT / "experiments/common/neural_v1/runs" / args.run_id
    out = ROOT / "data/results" / args.run_id
    out.mkdir(parents=True, exist_ok=True)
    cache_dir = run_dir / "evaluation"; cache_dir.mkdir(exist_ok=True)
    cache_signature = hashlib.sha256(
        Path(__file__).read_bytes() + (Path(__file__).parent/'dispatch.py').read_bytes()
        + json.dumps(data.hashes, sort_keys=True).encode()
        + b''.join(p.read_bytes() for p in sorted(run_dir.glob('m*.npz')))
    ).hexdigest()
    manifest = cache_dir/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())['signature'] != cache_signature:
        raise RuntimeError('Evaluation inputs changed. Remove only this generated evaluation cache and rerun.')
    manifest.write_text(json.dumps({'signature':cache_signature}))
    store = ForecastStore(run_dir)
    # Fail before expensive evaluation if any of the 165 fits is unavailable.
    for month in range(2, 13):
        for variant in VARIANTS:
            store.pack(month, variant)
    forecast_rows = []
    for month in range(2, 13):
        origins = month_origins(month)
        for variant in VARIANTS:
            for seed in (*SEEDS, None):
                p = store.pack(month, variant, seed)
                prediction = p["predictions"][int(p["validation_count"]):]
                forecast_rows.extend(forecast_scores(data, origins, prediction, variant,
                                                     "mean" if seed is None else str(seed), month))
        for variant, days in (("yesterday", 1), ("weekly", 7)):
            prediction = np.stack([data.baseline(int(o), days) for o in origins])
            forecast_rows.extend(forecast_scores(data, origins, prediction, variant, "baseline", month))
    pd.DataFrame(forecast_rows).to_csv(out / "forecast_metrics.csv", index=False)
    (out / "data_hashes.json").write_text(json.dumps(data.hashes, ensure_ascii=False, indent=2))
    all_daily = []
    baseline_states = {}
    for scenario in SCENARIOS:
        for variant, lag in (("yesterday", 1), ("weekly", 7)):
            summaries, details, states = baseline_replay(data, scenario, lag)
            if lag == 1:
                baseline_states[scenario] = states
            for s in summaries[31:]:
                s.update(variant=variant, seed="baseline", model_used=variant, known_price=False,
                         corrected=True, update_schedule="0+6+12+18" if scenario in ("3", "4-3") else "0")
            all_daily.extend(summaries[31:])
    choices, selection_rows = choose_models(data, store, baseline_states)
    pd.DataFrame(selection_rows).to_csv(out / "model_selection.csv", index=False)
    tasks = []
    for scenario in SCENARIOS:
        tasks.append((scenario, "selected", None, (6, 12, 18), False, True))
        for variant in VARIANTS:
            for seed in (*SEEDS, None):
                tasks.append((scenario, variant, seed, (6, 12, 18), False, True))
    for scenario in ("3", "4-3"):
        for hours in ((), (6,), (6, 12)):
            tasks.append((scenario, "selected", None, hours, False, True))
        tasks.append((scenario, "gru", None, (6, 12, 18), False, False))
    for scenario in ("4-2", "4-3"):
        tasks.append((scenario, "selected", None, (6, 12, 18), True, True))
    for i, task in enumerate(tasks):
        scenario, variant, seed, hours, known, corrected = task
        key = f"{scenario}_{variant}_{seed}_{'-'.join(map(str, hours))}_{known}_{corrected}"
        cached = cache_dir / f"{key}.json"
        primary = variant == "selected" and hours == (6, 12, 18) and not known and corrected
        if cached.exists():
            summaries = json.loads(cached.read_text())
            if primary and not (out / f"dispatch_{scenario}.npz").exists():
                raise RuntimeError("Missing primary detail archive; remove its evaluation cache entry.")
        else:
            summaries, details = replay(data, store, choices, *task[:2], seed=seed,
                                        update_hours=hours, known_price=known, corrected=corrected)
            cached.write_text(json.dumps(summaries, ensure_ascii=False))
            if primary:
                save_details(out / f"dispatch_{scenario}.npz", details)
        all_daily.extend(summaries)
        print(f"REPLAY {i + 1}/{len(tasks)} {key} cost={sum(r['total_cost'] for r in summaries):.2f}", flush=True)
    frame = pd.DataFrame(all_daily)
    frame.to_csv(out / "daily_metrics.csv", index=False)
    group = ["scenario", "variant", "seed", "known_price", "corrected", "update_schedule"]
    sums = ["planned_kwh", "final_kwh", "emergency_kwh", "emergency_minutes", "charge_kwh",
            "discharge_kwh", "surplus_kwh", "planned_cost", "up_cost", "down_cost", "emergency_cost",
            "total_cost", "violations", "solve_execute_seconds"]
    aggregate = frame.groupby(group, dropna=False).agg(
        **{c: (c, "sum") for c in sums}, initial_soc=("initial_soc", "first"),
        final_soc=("final_soc", "last"), days=("day", "count")).reset_index()
    assert (aggregate.days == 334).all()
    aggregate.to_csv(out / "dispatch_metrics.csv", index=False)
    # Compact, lossless arrays retain every ensemble prediction and its explicit origin index.
    all_origins = np.concatenate([month_origins(m) for m in range(2, 13)])
    arrays = {"origins": all_origins}
    for variant in VARIANTS:
        arrays[variant] = np.concatenate([store.pack(m, variant)["predictions"][int(
            store.pack(m, variant)["validation_count"]):] for m in range(2, 13)])
    np.savez_compressed(out / "ensemble_predictions.npz", **arrays)
    (out / "prediction_archive.json").write_text(json.dumps({
        "epoch": "2025-01-01T00:00:00", "origin_unit_minutes": 10,
        "target_order": TARGETS, "array_dimensions": ["origin", "future_interval", "target"],
        "interval_start": "epoch + (origin + future_interval) * 10 minutes",
        "interval_end": "interval_start + 10 minutes", "issue_time": "epoch + origin * 10 minutes",
        "model": "array key", "value": "nonnegative prediction in kW or yuan/kWh",
        "year_end": "future intervals beyond 2025 are forecast but excluded from scores"}, indent=2))
    metadata = [json.loads(p.read_text()) for p in sorted(run_dir.glob("m*.json"))]
    (out / "training_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(f"DONE {out}", flush=True)


if __name__ == "__main__":
    main()
