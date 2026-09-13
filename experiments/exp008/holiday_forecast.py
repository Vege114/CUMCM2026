"""One causal holiday/workday load-residual diagnostic on ridge28 forecasts.

The State Council calendar was published before 2025. Its applicability to
the unidentified community is an explicit hypothesis, not a known local
mechanism. PV is unchanged; no holiday effect on irradiance is presumed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.exp008.forecast_calibration import CalibratedStore
from experiments.problem2.exp003.data import Data

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/"data/results/exp008/holiday_forecast"
SOURCE_URL = "https://big5.www.gov.cn/gate/big5/www.gov.cn/zhengce/zhengceku/202411/content_6986383.htm"
HOLIDAYS = (("2025-01-01", "2025-01-01", "new_year"),
            ("2025-01-28", "2025-02-04", "spring_festival"),
            ("2025-04-04", "2025-04-06", "qingming"),
            ("2025-05-01", "2025-05-05", "labour_day"),
            ("2025-05-31", "2025-06-02", "dragon_boat"),
            ("2025-10-01", "2025-10-08", "national_midautumn"))
MAKEUP = ("2025-01-26", "2025-02-08", "2025-04-27", "2025-09-28", "2025-10-11")
SETTINGS = {"minimum_history_days": 14, "half_life_days": 56.,
            "ridge_equivalent_days": 4., "blocks": 4, "max_abs_correction_kw": 500.}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n")


def calendar():
    dates = pd.date_range("2025-01-01", "2025-12-31")
    holiday_map = {date: name for start, stop, name in HOLIDAYS
                   for date in pd.date_range(start, stop)}
    makeup = set(pd.to_datetime(MAKEUP))
    rows = []
    for day, date in enumerate(dates):
        holiday, extra_work = date in holiday_map, date in makeup
        group = ("holiday" if holiday else "makeup_workday" if extra_work
                 else "weekend" if date.dayofweek >= 5 else "weekday")
        rows.append({"day": day, "date": str(date.date()), "weekday": date.dayofweek,
            "holiday": holiday, "makeup_workday": extra_work, "group": group,
            "official_workday": extra_work or (date.dayofweek < 5 and not holiday),
            "festival": holiday_map.get(date, "")})
    return pd.DataFrame(rows)


def design(day_rows):
    x = np.zeros((len(day_rows), 9))
    x[np.arange(len(day_rows)), day_rows.weekday.to_numpy(int)] = 1.
    x[:, 7] = day_rows.holiday.to_numpy(float)
    x[:, 8] = day_rows.makeup_workday.to_numpy(float)
    return x


def correct_day(actual, origins, base, index, calendar_rows):
    origin = int(origins[index])
    day = origin//144
    current = calendar_rows.iloc[[day]]
    historical_ids = np.flatnonzero(origins+144 <= origin)
    audit = {"day": day, "origin": origin, "information_cutoff": origin,
        "training_origins": origins[historical_ids].tolist(),
        "history_count": len(historical_ids), "current_truth_used": False,
        "calendar_known_since": "2024-11-12", "local_calendar_applicability": "hypothesis",
        "group": current.group.iloc[0], "delta_block_kw": [0.]*4,
        "max_observed_index": int(origins[historical_ids[-1]]+143) if len(historical_ids) else None,
        "calendar_increment_only": True}
    output = base[index].copy()
    if current.group.iloc[0] not in ("holiday", "makeup_workday"):
        audit["reason"] = "ordinary_day_unchanged"
        return output, audit
    if len(historical_ids) < SETTINGS["minimum_history_days"]:
        audit["reason"] = "not_enough_completed_formal_ridge_days"
        return output, audit
    observed = []
    for j in historical_ids:
        start, stop = int(origins[j]), int(origins[j])+144
        if stop > origin:
            raise ValueError("incomplete historical label")
        observed.append(actual[start:stop, 0])
    target = (np.stack(observed)-base[historical_ids, :, 0]).reshape(-1, 4, 36).mean(axis=2)
    history_days = origins[historical_ids]//144
    x = design(calendar_rows.iloc[history_days])
    weights = 2.**(-(day-1-history_days)/SETTINGS["half_life_days"])
    gram = np.einsum("ni,nj,n->ij", x, x, weights, optimize=False)
    rhs = np.einsum("ni,nb,n->ib", x, target, weights, optimize=False)
    coefficients = np.linalg.solve(gram+SETTINGS["ridge_equivalent_days"]*np.eye(9), rhs)
    # Weekday coefficients are nuisance controls, not another baseline change.
    column = 7 if bool(current.holiday.iloc[0]) else 8
    block_delta = np.clip(coefficients[column], -SETTINGS["max_abs_correction_kw"],
                          SETTINGS["max_abs_correction_kw"])
    output[:, 0] = np.maximum(0., output[:, 0]+np.repeat(block_delta, 36))
    audit.update(reason="past_only_calendar_increment", delta_block_kw=block_delta.tolist(),
        coefficients_kw=coefficients.tolist(),
        feature_names=[f"weekday_{d}" for d in range(7)]+["holiday", "makeup_workday"],
        training_target="ridge28 load residual averaged within six-hour block, kW",
        training_special_days=int(calendar_rows.iloc[history_days].group.isin(["holiday", "makeup_workday"]).sum()))
    return output, audit


class HolidayStore:
    """Compatible midnight forecast store; base ridge28 plus load-only delta."""
    def __init__(self, data=None, base_store=None):
        self.base_store = base_store or CalibratedStore("ridge_28")
        self.origins = self.base_store.origins.copy()
        self.base_values = self.base_store.values.copy()
        self.lookup = {int(origin): i for i, origin in enumerate(self.origins)}
        self.calendar = calendar()
        actual = (data or Data()).actual
        generated = [correct_day(actual, self.origins, self.base_values, i, self.calendar)
                     for i in range(len(self.origins))]
        self.values = np.stack([row[0] for row in generated])
        self.audit = [row[1] for row in generated]
        self.delta = self.values-self.base_values
        # Scoring labels are formed after every issue-time prediction exists.
        self.errors_kw = actual[self.origins[:, None]+np.arange(144), :2]-self.values

    def get(self, origin):
        return self.values[self.lookup[int(origin)]].copy()


def score(predicted, actual, price):
    error = predicted-actual
    net = error[..., 0]-error[..., 1]
    cumulative = np.cumsum(net, axis=1)/6
    high = price >= np.quantile(price, .75)
    return {"days": len(net), "net_rmse_kw": float(np.sqrt(np.square(net).mean())),
        "net_mae_kw": float(np.abs(net).mean()), "net_bias_kw": float(net.mean()),
        "load_rmse_kw": float(np.sqrt(np.square(error[..., 0]).mean())),
        "load_bias_kw": float(error[..., 0].mean()), "pv_bias_kw": float(error[..., 1].mean()),
        "high_tariff_net_rmse_kw": float(np.sqrt(np.square(net[:, high]).mean())),
        "high_tariff_net_bias_kw": float(net[:, high].mean()),
        "mean_daily_signed_net_energy_error_kwh": float(cumulative[:, -1].mean()),
        "mean_daily_absolute_net_energy_error_kwh": float(np.abs(cumulative[:, -1]).mean()),
        "mean_daily_max_abs_cumulative_net_error_kwh": float(np.abs(cumulative).max(axis=1).mean())}


def run(out=OUT):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if (out/"protocol.json").exists():
        raise RuntimeError("Refusing to overwrite an existing holiday experiment")
    write_json(out/"protocol.json", {"official_source": SOURCE_URL,
        "document": "国办发明电〔2024〕12号", "publication_date": "2024-11-12",
        "source_access": "official gov.cn traditional Chinese mirror verified; simplified endpoint returned 403",
        "settings": SETTINGS, "candidate_count": 1,
        "scope_assumption": "2025 mainland-China public calendar is a hypothetical feature; community location and actual observance unknown",
        "base_forecast": "ridge_28", "pv_unchanged": True,
        "calendar_used_before_year": True, "future_load_used_for_fit": False,
        "point_forecast_gate": "strictly lower overall net RMSE, special-day net RMSE, and special-day high-tariff net RMSE than ridge28",
        "selection_role": "2025 development gate only; no current-day result chooses coefficients or correction sign",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    data = Data()
    store = HolidayStore(data)
    cal = store.calendar.iloc[store.origins//144].reset_index(drop=True)
    truth = data.actual[store.origins[:, None]+np.arange(144)]
    rows = []
    predictions = {"cnn": store.base_store.base_values, "ridge28": store.base_values,
                   "holiday_increment": store.values}
    for name, values in predictions.items():
        for group in ("all", "special", "holiday", "makeup_workday", "weekday", "weekend"):
            mask = (np.ones(len(cal), bool) if group == "all" else
                    cal.group.isin(["holiday", "makeup_workday"]).to_numpy() if group == "special"
                    else (cal.group == group).to_numpy())
            rows.append({"model": name, "group": group, **score(values[mask], truth[mask], data.fixed_price)})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out/"metrics.csv", index=False)
    daily_rows = []
    for i, row in cal.iterrows():
        for name, values in predictions.items():
            daily_rows.append({**row.to_dict(), "model": name,
                **score(values[i:i+1], truth[i:i+1], data.fixed_price)})
    pd.DataFrame(daily_rows).to_csv(out/"daily_metrics.csv", index=False)
    cal.to_csv(out/"calendar_formal_days.csv", index=False)
    store.calendar.to_csv(out/"calendar_2025.csv", index=False)
    # Perturb all current/future labels and future forecasts for three eligible
    # calendar issues. The exact current correction must remain unchanged.
    checks = []
    for date in ("2025-04-04", "2025-09-28", "2025-10-03"):
        day = int((pd.Timestamp(date)-pd.Timestamp("2025-01-01")).days)
        origin = day*144
        index = store.lookup[origin]
        altered_actual = data.actual.copy()
        altered_actual[origin:] = 1e7
        altered_base = store.base_values.copy()
        altered_base[store.origins > origin] = 2e7
        prediction, audit = correct_day(altered_actual, store.origins, altered_base, index, store.calendar)
        np.testing.assert_array_equal(prediction, store.values[index])
        assert all(value+144 <= origin for value in audit["training_origins"])
        checks.append(date)
    assert np.array_equal(store.values[..., 1], store.base_values[..., 1])
    ordinary = ~cal.group.isin(["holiday", "makeup_workday"]).to_numpy()
    assert np.array_equal(store.values[ordinary], store.base_values[ordinary])
    assert np.isfinite(store.values).all() and (store.values >= 0).all()
    lookup = metrics.set_index(["model", "group"])
    gate_checks = {f"{group}_{metric}": bool(lookup.loc[("holiday_increment", group), metric]
                    < lookup.loc[("ridge28", group), metric])
                   for group, metric in (("all", "net_rmse_kw"), ("special", "net_rmse_kw"),
                                         ("special", "high_tariff_net_rmse_kw"))}
    summary = {"complete": True, "forecast_advantage_gate": all(gate_checks.values()),
        "gate_checks": gate_checks, "special_day_counts": cal.group.value_counts().to_dict(),
        "training_uses_prior_complete_days_only": True, "pv_unchanged": True,
        "ordinary_days_exact_base": True, "future_perturbation_dates_passed": checks,
        "source_url": SOURCE_URL, "local_mechanism_claimed": False,
        "new_model_count": 1, "metrics": rows}
    write_json(out/"audit.json", store.audit)
    write_json(out/"summary.json", summary)
    np.savez_compressed(out/"forecasts.npz", origins=store.origins,
        base_values=store.base_values, delta=store.delta, values=store.values, errors_kw=store.errors_kw)
    print(json.dumps({"gate": summary["forecast_advantage_gate"], "checks": gate_checks,
        "metrics": [r for r in rows if r["group"] in ("all", "special")]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    run(args.out)
