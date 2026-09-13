"""One fixed planning link for the predeclared holiday forecast candidate."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.exp008.controller_candidate import INITIAL_SOC, execute_inventory, plan_inventory
from experiments.exp008.holiday_forecast import HolidayStore, OUT, ROOT, write_json
from experiments.exp008.verify import verify_arrays
from experiments.problem2.exp003.data import Data
from experiments.problem2.tree_planning.risk import TreeResidualScenarios


def run():
    point = json.loads((OUT/"summary.json").read_text())
    if not point["forecast_advantage_gate"]:
        raise RuntimeError("No planning link without the fixed point-forecast gate")
    out = OUT/"dispatch_linkage"
    if out.exists():
        raise RuntimeError("Refusing to overwrite this single fixed planning candidate")
    out.mkdir()
    spec = {"quantile": .8, "controller": "greedy", "state_buffer": 500.}
    baseline_dir = ROOT/"data/results/exp008/risk_window/tree28_334days"
    baseline = json.loads((baseline_dir/"summary.json").read_text())
    write_json(out/"protocol.json", {"spec": spec, "source_forecast": "holiday_forecast",
        "baseline": str(baseline_dir.relative_to(ROOT)), "candidate_count": 1,
        "risk": "fresh tree28 from holiday prequential residuals, never old base risk cache",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "forecast_source_sha256": hashlib.sha256((ROOT/"experiments/exp008/holiday_forecast.py").read_bytes()).hexdigest(),
        "calendar_local_applicability": "hypothesis only"})
    data, store = Data(), HolidayStore()
    risk = TreeResidualScenarios(store.origins, store.values, data.actual, data.fixed_price)
    soc, mode, parts, rows, audits, supports = INITIAL_SOC, 1, [], [], [], []
    for day in range(31, 365):
        support, audit = risk.for_day(day)
        plan = plan_inventory(support, data.fixed_price, soc, spec, final=day == 364)
        detail, mode = execute_inventory(plan, data.actual[day*144:(day+1)*144],
                                        data.fixed_price, soc, mode, spec)
        audit.update(residual_source="periodic_baseline" if audit["fallback"] else "causal_holiday_adjusted_ridge28",
            forecast_origin=day*144, purchase_locked_before_actual_read=True,
            forecast_audit=store.audit[day-31])
        parts.append(detail)
        supports.append(support)
        audits.append(audit)
        rows.append({"day": day, "date": str((pd.Timestamp("2025-01-01")+pd.Timedelta(days=day)).date()),
            "total_cost": float(detail["fees"].sum()), "planned_cost": float(detail["fees"][:, 0].sum()),
            "emergency_cost": float(detail["fees"][:, 3].sum()),
            "initial_soc": soc, "final_soc": float(detail["states"][-1])})
        soc = rows[-1]["final_soc"]
    arrays = {key: np.stack([d[key] for d in parts]) for key in parts[0]}
    arrays["days"] = np.arange(31, 365)
    verification = verify_arrays(arrays, source_actual=data.actual[31*144:].reshape(334, 144, 2),
                                 source_price=data.fixed_price, audit_records=audits)
    assert verification["passed"], verification["errors"]
    summary = {"spec": spec, "days": 334, **verification["billing"],
        "battery": verification["battery_metrics"], "verified": True,
        "baseline_cost": baseline["total_cost"],
        "delta_total_cost": verification["billing"]["total_cost"]-baseline["total_cost"],
        "delta_emergency_cost": verification["billing"]["emergency_cost"]-baseline["emergency_cost"],
        "delta_reversals": verification["battery_metrics"]["direction_reversals"]-baseline["battery"]["direction_reversals"],
        "candidate_count": 1, "calendar_local_mechanism_claimed": False}
    pd.DataFrame(rows).to_csv(out/"daily.csv", index=False)
    np.savez_compressed(out/"dispatch_2.npz", **arrays)
    np.savez_compressed(out/"supports.npz", supports=np.stack(supports), origins=store.origins)
    write_json(out/"audit.json", audits)
    write_json(out/"verification.json", verification)
    write_json(out/"summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    run()
