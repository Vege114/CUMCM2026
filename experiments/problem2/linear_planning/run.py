"""Small real-data smoke replay; optional supplied forecasts, never trains.

Run from repo root with python -m experiments.problem2.linear_planning.run.
All days start January 1 and carry state; Jan is separately classified as warmup.
Default two-day run is NOT a formal evaluation or a final result2 workbook.
"""

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from experiments.problem2.exp003.data import Data
from .model import Config, State, replay_day


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--days", type=int, default=2)
    parser.add_argument("--forecast-npz", type=Path,
                        help="forecast[days,144,2] in kW, issued at each midnight")
    parser.add_argument("--out", type=Path, default=Path("experiments/problem2/linear_planning/runs/smoke"))
    args = parser.parse_args()
    if not 1 <= args.days <= 365:
        parser.error("--days must be 1..365")
    if args.out.exists():
        parser.error("output directory already exists; choose a new --out")
    data, cfg, state = Data(), Config(), State()
    supplied = None
    if args.forecast_npz:
        with np.load(args.forecast_npz, allow_pickle=False) as pack:
            supplied = pack["forecast"].copy()
        if supplied.shape != (args.days, 144, 2):
            parser.error("forecast must have shape (days,144,2)")
    args.out.mkdir(parents=True)
    records, all_details, logs = [], [], []
    for day in range(args.days):
        origin = day * 144
        forecast = supplied[day] if supplied is not None else data.baseline(origin)
        plan, detail, end = replay_day(forecast, data.fixed_price,
                                      lambda t: data.actual[origin + t], state, cfg)
        records.append({"day": day, "period": "warmup" if day < 31 else "formal",
                        "planned_cost": float(detail["fees"][:, 0].sum()),
                        "emergency_cost": float(detail["fees"][:, 1].sum()),
                        "total_cost": float(detail["fees"].sum()),
                        "planned_kwh": float(detail["grid"].sum()),
                        "emergency_kwh": float(detail["emergency"].sum()),
                        "initial_soc": state.soc, "final_soc": end.soc,
                        "plan_max_overlap_kwh": plan["metadata"]["max_overlap_kwh"],
                        **detail["audit"]})
        logs.append({"day": day, "plan": plan["metadata"], "execution": detail["execution_logs"]})
        all_details.append(detail)
        state = end
        print(json.dumps(records[-1], ensure_ascii=False), flush=True)
    with (args.out / "daily.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    arrays = {k: np.stack([d[k] for d in all_details]) for k in
              ("grid", "charge", "discharge", "emergency", "surplus", "states", "actual", "fees")}
    np.savez_compressed(args.out / "dispatch.npz", **arrays)
    with (args.out / "intervals.csv").open("w", encoding="utf-8-sig", newline="") as f:
        fields = ["day", "interval", "grid_kwh", "charge_kwh", "discharge_kwh", "emergency_kwh",
                  "surplus_kwh", "start_soc_kwh", "end_soc_kwh", "planned_cost", "emergency_cost"]
        writer = csv.writer(f)
        writer.writerow(fields)
        for day, d in enumerate(all_details):
            for t in range(144):
                writer.writerow([day, t, *[d[k][t] for k in
                    ("grid", "charge", "discharge", "emergency", "surplus")],
                    d["states"][t], d["states"][t + 1], *d["fees"][t]])
    meta = {"config": asdict(cfg), "data_hashes": data.hashes,
            "forecast": str(args.forecast_npz) if supplied is not None else "existing_periodic_baseline_smoke_only",
            "days": args.days, "warmup_days": min(31, args.days),
            "formal_days": max(0, args.days - 31),
            "formal_total_cost": sum(r["total_cost"] for r in records if r["period"] == "formal"),
            "logs": logs}
    (args.out / "solver.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
