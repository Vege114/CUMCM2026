"""Post-selection temporal and accounting sensitivity of frozen exp008 traces.

Run: .venv/bin/python -m experiments.exp008.robustness.temporal_analysis
No forecaster, optimizer, or controller is imported or rerun.  Every monetary
amount is independently settled against the original task CSVs.  Chronological
subsets retain their archived carried battery state; they are not fresh tests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "data/results/exp008/robustness/temporal"
SELECTION = ROOT / "experiments/exp008/final_selection.json"
BASELINE = ROOT / "data/results/exp006/primary/dispatch_2.npz"
SEED = 20260913
N_BOOTSTRAP = 4096
ETA = float(np.sqrt(0.9))
CAPACITY = 12000.0
TOL = 1e-6
METHOD_LABEL = "frozen_trajectory_rescoring"
SUM_FIELDS = [
    "planned_cost_yuan", "up_cost_yuan", "down_cost_yuan", "emergency_cost_yuan",
    "total_cost_yuan", "original_purchase_kwh", "final_purchase_kwh", "up_adjustment_kwh",
    "down_adjustment_kwh", "emergency_kwh", "surplus_kwh", "charge_kwh", "discharge_kwh",
    "throughput_kwh", "equivalent_full_cycles", "active_slots", "emergency_slots",
    "nonidle_reversals", "charge_episodes", "discharge_episodes", "power_tv_kw",
    "load_kwh", "pv_kwh", "net_load_kwh",
]


def reference(path: Path) -> dict:
    return {"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size}


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, float_format="%.12g")


def raw_data() -> tuple[pd.DatetimeIndex, dict, list]:
    files = {"load": ROOT / "data/raw/附件2_小区负载.csv",
             "pv": ROOT / "data/raw/附件2_光伏发电实际功率.csv",
             "dynamic_price": ROOT / "data/raw/附件4.csv",
             "fixed_price": ROOT / "data/raw/附件1.csv"}
    data, dates = {}, None
    for key, path in files.items():
        frame = pd.read_csv(path)
        if key == "fixed_price":
            data[key] = frame.iloc[:, 1].to_numpy(float)
            assert data[key].shape == (144,)
        else:
            current = pd.DatetimeIndex(pd.to_datetime(frame.iloc[:, 0]))
            assert current.equals(pd.date_range("2025-01-01", "2025-12-31"))
            if dates is not None:
                assert current.equals(dates)
            dates = current
            data[key] = frame.iloc[:, 1:].to_numpy(float)
            assert data[key].shape == (365, 144)
        assert np.isfinite(data[key]).all() and (data[key] >= 0).all()
    return dates, data, [reference(p) for p in files.values()]


def extract_daily(scenario: str, strategy: str, path: Path, spec: dict,
                  dates: pd.DatetimeIndex, raw: dict) -> tuple[pd.DataFrame, dict]:
    with np.load(path) as archive:
        a = {key: archive[key].copy() for key in archive.files}
    start = int(spec["start_day"])
    n, slots = a["original"].shape
    assert slots == 144 and n == 334 and start == 31
    expected_days = np.arange(start, start + n)
    required = ["original", "final", "charge", "discharge", "emergency", "surplus", "price"]
    for key in required:
        assert a[key].shape == (n, 144), (scenario, key)
    assert a["states"].shape == (n, 145) and a["fees"].shape == (n, 144, 4)
    assert a["actual"].shape[:2] == (n, 144)
    numeric = required + ["states", "fees", "actual"]
    assert all(np.isfinite(a[key]).all() and a[key].min() >= -TOL for key in numeric)
    if "days" in a:
        assert np.array_equal(a["days"], expected_days)
    if "origins" in a:
        assert np.array_equal(a["origins"], expected_days * 144)
    load, pv = raw["load"][start:start + n], raw["pv"][start:start + n]
    price = (raw["dynamic_price"][start:start + n] if scenario.startswith("4")
             else np.broadcast_to(raw["fixed_price"], (n, 144)))
    original, final, c, d, e, w, states = (a[k] for k in
        ["original", "final", "charge", "discharge", "emergency", "surplus", "states"])
    up, down = np.maximum(final - original, 0), np.maximum(original - final, 0)
    # Original purchase remains fully charged; down adjustment is an additional
    # 50% penalty, not a negative refund. Only final net adjustment is settled.
    fee_parts = np.stack((price * original, 1.5 * price * up,
                          .5 * price * down, 5 * price * e), axis=-1)
    balance = final + e + d + pv / 6 - load / 6 - c - w
    soc_residual = states[:, 1:] - states[:, :-1] - ETA * c + d / ETA
    errors = {
        "source_load_max_abs_kw": float(np.max(abs(a["actual"][..., 0] - load))),
        "source_pv_max_abs_kw": float(np.max(abs(a["actual"][..., 1] - pv))),
        "source_price_max_abs_yuan_per_kwh": float(np.max(abs(a["price"] - price))),
        "balance_max_abs_kwh": float(np.max(abs(balance))),
        "soc_equation_max_abs_kwh": float(np.max(abs(soc_residual))),
        "midnight_continuity_max_abs_kwh": float(np.max(abs(states[1:, 0] - states[:-1, -1]))),
        "initial_soc_abs_error_kwh": abs(float(states[0, 0]) - float(spec["initial_soc_kwh"])),
        "archived_slot_fee_max_abs_yuan": float(np.max(abs(a["fees"] - fee_parts))),
    }
    assert all(value <= TOL for value in errors.values()), (scenario, strategy, errors)
    assert states.min() >= 1200 - TOL and states.max() <= 10800 + TOL
    assert max(c.max(), d.max()) * 6 <= 5000 + TOL
    assert not np.any((c > TOL) & (d > TOL))
    assert not np.any((c > TOL) & (e > TOL))
    if scenario in ["2", "4-2"]:
        assert np.max(abs(final - original)) <= TOL
    flat_c, flat_d = c.ravel(), d.ravel()
    mode = np.where(flat_c > TOL, 1, np.where(flat_d > TOL, -1, 0))
    prev = np.r_[0, mode[:-1]]
    active_indices = np.flatnonzero(mode)
    reversal_flags = np.zeros_like(mode)
    reversal_flags[active_indices[1:]] = (mode[active_indices[1:]] != mode[active_indices[:-1]])
    # Attribute reversals/episodes to their occurrence date using the unbroken
    # original trace. First formal active mode has no earlier formal reversal.
    charge_episode_flags = (mode == 1) & (prev != 1)
    discharge_episode_flags = (mode == -1) & (prev != -1)
    power = 6 * (flat_c - flat_d)
    tv = np.r_[0.0, abs(np.diff(power))].reshape(n, 144)
    daily = pd.DataFrame({"scenario": scenario, "strategy": strategy,
        "analysis_label": METHOD_LABEL, "date": dates[start:start + n].strftime("%Y-%m-%d"),
        "day_index": expected_days, "month": dates[start:start + n].month,
        "quarter": dates[start:start + n].quarter,
        "planned_cost_yuan": fee_parts[..., 0].sum(axis=1),
        "up_cost_yuan": fee_parts[..., 1].sum(axis=1),
        "down_cost_yuan": fee_parts[..., 2].sum(axis=1),
        "emergency_cost_yuan": fee_parts[..., 3].sum(axis=1),
        "total_cost_yuan": fee_parts.sum(axis=(1, 2)),
        "original_purchase_kwh": original.sum(axis=1), "final_purchase_kwh": final.sum(axis=1),
        "up_adjustment_kwh": up.sum(axis=1), "down_adjustment_kwh": down.sum(axis=1),
        "emergency_kwh": e.sum(axis=1), "surplus_kwh": w.sum(axis=1),
        "charge_kwh": c.sum(axis=1), "discharge_kwh": d.sum(axis=1),
        "throughput_kwh": (c + d).sum(axis=1),
        "equivalent_full_cycles": (ETA * c + d / ETA).sum(axis=1) / (2 * CAPACITY),
        "active_slots": ((c > TOL) | (d > TOL)).sum(axis=1),
        "emergency_slots": (e > TOL).sum(axis=1),
        "nonidle_reversals": reversal_flags.reshape(n, 144).sum(axis=1),
        "charge_episodes": charge_episode_flags.reshape(n, 144).sum(axis=1),
        "discharge_episodes": discharge_episode_flags.reshape(n, 144).sum(axis=1),
        "power_tv_kw": tv.sum(axis=1), "soc_start_kwh": states[:, 0], "soc_end_kwh": states[:, -1],
        "soc_min_kwh": states.min(axis=1), "soc_max_kwh": states.max(axis=1),
        "load_kwh": load.sum(axis=1) / 6, "pv_kwh": pv.sum(axis=1) / 6,
        "net_load_kwh": (load - pv).sum(axis=1) / 6,
        "load_peak_kw": load.max(axis=1), "net_load_peak_kw": (load - pv).max(axis=1),
        "mean_settlement_price_yuan_per_kwh": price.mean(axis=1),
        "mean_external_dynamic_price_yuan_per_kwh": raw["dynamic_price"][start:start+n].mean(axis=1),
        "balance_max_abs_kwh": abs(balance).max(axis=1),
        "soc_equation_max_abs_kwh": abs(soc_residual).max(axis=1),
    })
    # Alternative flattened dot-products validate aggregation without relying
    # on the stacked slot-fee layout used above.
    alternative_total = float(np.dot(price.ravel(), (original + 1.5 * up + .5 * down + 5 * e).ravel()))
    assert abs(alternative_total - daily.total_cost_yuan.sum()) < 1e-6
    assert int(daily.nonidle_reversals.sum()) == int(np.sum(np.diff(mode[mode != 0]) != 0))
    audit = {"scenario": scenario, "strategy": strategy, "source": reference(path),
        "days": n, "slots": n * slots, "start": daily.date.iloc[0], "end": daily.date.iloc[-1],
        "passed": True, "errors": errors, "finite_nonnegative_arrays": True,
        "soc_and_power_limits_passed": True, "no_simultaneous_charge_discharge": True,
        "no_emergency_charging": True, "day_identifiers_checked_if_present": True,
        "initial_soc_kwh": float(states[0, 0]), "final_soc_kwh": float(states[-1, -1]),
        "recomputed_total_cost_yuan": float(daily.total_cost_yuan.sum()),
        "alternative_flat_dot_total_yuan": alternative_total,
        "alternative_total_abs_difference_yuan": abs(alternative_total - daily.total_cost_yuan.sum()),
        "nonidle_reversals": int(daily.nonidle_reversals.sum()),
        "no_causality_reproof": "This script verifies archived physics, billing and source alignment; causality evidence remains the final-selection audits."}
    return daily, audit


def aggregate(frame: pd.DataFrame) -> dict:
    row = {key: float(frame[key].sum()) for key in SUM_FIELDS}
    n = len(frame)
    row.update({"n_days": n, "mean_daily_cost_yuan": row["total_cost_yuan"] / n,
                "mean_daily_emergency_kwh": row["emergency_kwh"] / n,
                "mean_daily_throughput_kwh": row["throughput_kwh"] / n,
                "soc_min_kwh": float(frame.soc_min_kwh.min()),
                "soc_max_kwh": float(frame.soc_max_kwh.max()),
                "soc_start_kwh": float(frame.soc_start_kwh.iloc[0]),
                "soc_end_kwh": float(frame.soc_end_kwh.iloc[-1]),
                "max_balance_abs_kwh": float(frame.balance_max_abs_kwh.max()),
                "max_soc_equation_abs_kwh": float(frame.soc_equation_max_abs_kwh.max()),
                "emergency_energy_fraction_pct": 100 * row["emergency_kwh"] / row["load_kwh"],
                "cost_per_load_kwh": row["total_cost_yuan"] / row["load_kwh"]})
    return row


def paired_fields(frame: pd.DataFrame, baseline: pd.DataFrame) -> dict:
    base = baseline.set_index("date").loc[frame.date]
    b, c = float(base.total_cost_yuan.sum()), float(frame.total_cost_yuan.sum())
    diff = base.total_cost_yuan.to_numpy() - frame.total_cost_yuan.to_numpy()
    return {"baseline_strategy": "exp006_primary", "baseline_cost_yuan": b,
            "cost_saving_yuan": b - c, "cost_reduction_pct": 100 * (b - c) / b,
            "paired_day_win_fraction": float(np.mean(diff > 1e-6)),
            "paired_day_cost_saving_sd_yuan": float(diff.std(ddof=1)) if len(diff) > 1 else 0.,
            "baseline_emergency_kwh": float(base.emergency_kwh.sum()),
            "baseline_throughput_kwh": float(base.throughput_kwh.sum())}


def window_analyses(daily: pd.DataFrame) -> pd.DataFrame:
    baseline = daily[daily.strategy == "exp006_primary"]
    rows = []
    for (scenario, strategy), frame in daily.groupby(["scenario", "strategy"], sort=False):
        frame = frame.reset_index(drop=True)
        windows = [("full", "all_334_days", np.arange(len(frame)), None, None)]
        for key in ["month", "quarter"]:
            for value, ids in frame.groupby(key).groups.items():
                windows.append((key, str(value), np.array(list(ids)), None, None))
        for length in [30, 60, 90]:
            for end in range(length, len(frame) + 1):
                ids = np.arange(end - length, end)
                windows.append(("rolling", f"{length}d_{frame.date.iloc[end-1]}", ids, length, None))
        for fraction in [.2, .3, .4]:
            tail_n = int(np.ceil(len(frame) * fraction))
            split = len(frame) - tail_n
            windows.extend([
                ("chronological_head", f"before_last_{round(fraction*100)}pct", np.arange(split), None, fraction),
                ("chronological_tail", f"last_{round(fraction*100)}pct", np.arange(split, len(frame)), None, fraction)])
        for typ, label, ids, length, fraction in windows:
            subset = frame.iloc[ids]
            row = {"scenario": scenario, "strategy": strategy, "analysis_label": METHOD_LABEL,
                "window_type": typ, "window_label": label, "window_days": length,
                "requested_tail_fraction": fraction, "realized_sample_fraction": len(ids) / len(frame),
                "start": subset.date.iloc[0], "end": subset.date.iloc[-1], **aggregate(subset)}
            if scenario == "2" and strategy == "exp008_final":
                row.update(paired_fields(subset, baseline))
            rows.append(row)
    return pd.DataFrame(rows)


def stress_analyses(daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline = daily[daily.strategy == "exp006_primary"]
    rows, memberships = [], []
    for (scenario, strategy), frame in daily.groupby(["scenario", "strategy"], sort=False):
        groups = [("all_days", 1.0, "all", None, np.ones(len(frame), bool))]
        for fraction in [.1, .2, .3]:
            for name, field, upper in [("high_load", "load_kwh", True), ("low_pv", "pv_kwh", False),
                                       ("high_net_load", "net_load_kwh", True)]:
                threshold = float(frame[field].quantile(1 - fraction if upper else fraction))
                mask = (frame[field] >= threshold if upper else frame[field] <= threshold).to_numpy()
                groups.append((name, fraction, field, threshold, mask))
            if scenario.startswith("4"):
                field = "mean_settlement_price_yuan_per_kwh"
                threshold = float(frame[field].quantile(1 - fraction))
                groups.append(("high_price", fraction, field, threshold, (frame[field] >= threshold).to_numpy()))
        high = frame.load_kwh >= frame.load_kwh.quantile(.8)
        low = frame.pv_kwh <= frame.pv_kwh.quantile(.2)
        groups.append(("high_load_and_low_pv", .2, "load_kwh>=q80 AND pv_kwh<=q20", None, (high & low).to_numpy()))
        for name, fraction, field, threshold, mask in groups:
            subset = frame.loc[mask]
            if subset.empty:
                continue
            row = {"scenario": scenario, "strategy": strategy, "analysis_label": "posthoc_observed_stress_group",
                   "stress_group": name, "nominal_tail_fraction": fraction, "threshold_field": field,
                   "threshold_value": threshold, "first_selected_date": subset.date.iloc[0],
                   "last_selected_date": subset.date.iloc[-1], "small_group_lt10_days": len(subset) < 10,
                   "threshold_load_q80_kwh": float(frame.load_kwh.quantile(.8)) if name == "high_load_and_low_pv" else None,
                   "threshold_pv_q20_kwh": float(frame.pv_kwh.quantile(.2)) if name == "high_load_and_low_pv" else None,
                   **aggregate(subset)}
            if scenario == "2" and strategy == "exp008_final":
                row.update(paired_fields(subset, baseline))
            rows.append(row)
            for date in subset.date:
                memberships.append({"scenario": scenario, "strategy": strategy, "date": date,
                                    "stress_group": name, "nominal_tail_fraction": fraction})
    return pd.DataFrame(rows), pd.DataFrame(memberships)


def block_indices(n: int, block: int, reps: int, rng: np.random.Generator) -> np.ndarray:
    assert block <= n
    count = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(reps, count))
    return (starts[..., None] + np.arange(block)).reshape(reps, -1)[:, :n]


def bootstrap_analyses(daily: pd.DataFrame, reps: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    final = daily[(daily.scenario == "2") & (daily.strategy == "exp008_final")].reset_index(drop=True)
    baseline = daily[daily.strategy == "exp006_primary"].set_index("date").loc[final.date]
    b = baseline.total_cost_yuan.to_numpy()
    f = final.total_cost_yuan.to_numpy()
    diff = b - f
    point = 100 * (b.sum() - f.sum()) / b.sum()
    summaries, draws = [], []
    for method_idx, method in enumerate(["paired_moving_block", "paired_quarter_stratified_moving_block"]):
        for block in [7, 14, 28]:
            # Stable independent streams: changing the number/order of other
            # configurations cannot silently change this configuration's draw.
            rng = np.random.default_rng(np.random.SeedSequence([seed, method_idx, block]))
            if method_idx == 0:
                ids = block_indices(len(final), block, reps, rng)
            else:
                parts = []
                for ids_q in final.groupby("quarter", sort=True).groups.values():
                    ids_q = np.array(list(ids_q))
                    parts.append(ids_q[block_indices(len(ids_q), block, reps, rng)])
                ids = np.concatenate(parts, axis=1)
            assert ids.shape == (reps, len(final))
            assert ids.min() >= 0 and ids.max() < len(final)
            bstar, fstar = b[ids].sum(axis=1), f[ids].sum(axis=1)
            saving = bstar - fstar
            pct = 100 * saving / bstar
            lower, median, upper = np.quantile(pct, [.025, .5, .975])
            summaries.append({"scenario": "2", "method": method, "block_days": block,
                "replicates": reps, "seed": seed, "n_original_days": len(final),
                "point_cost_reduction_pct": point, "bootstrap_median_pct": median,
                "percentile_95_lower_pct": lower, "percentile_95_upper_pct": upper,
                "bootstrap_sd_pct": float(pct.std(ddof=1)),
                "bootstrap_fraction_positive": float(np.mean(pct > 0)),
                "point_mean_daily_saving_yuan": float(diff.mean()),
                "mean_daily_saving_95_lower_yuan": float(np.quantile(saving / len(final), .025)),
                "mean_daily_saving_95_upper_yuan": float(np.quantile(saving / len(final), .975)),
                "interval_label": "conditional_descriptive_percentile_resampling_interval"})
            draws.extend({"method": method, "block_days": block, "replicate": idx,
                          "baseline_cost_yuan": float(bs), "final_cost_yuan": float(fs),
                          "cost_saving_yuan": float(sv), "cost_reduction_pct": float(pc)}
                         for idx, (bs, fs, sv, pc) in enumerate(zip(bstar, fstar, saving, pct)))
    acf_rows = []
    for lag in range(1, 61):
        acf_rows.append({"lag_days": lag,
                        "paired_daily_saving_correlation": float(np.corrcoef(diff[:-lag], diff[lag:])[0, 1]),
                        "exp006_daily_cost_correlation": float(np.corrcoef(b[:-lag], b[lag:])[0, 1]),
                        "exp008_daily_cost_correlation": float(np.corrcoef(f[:-lag], f[lag:])[0, 1])})
    return pd.DataFrame(summaries), pd.DataFrame(draws), pd.DataFrame(acf_rows)


def billing_sensitivity(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    base = daily[daily.strategy == "exp006_primary"]
    base_planned = float(base.planned_cost_yuan.sum())
    base_emergency_at_unit = float(base.emergency_cost_yuan.sum() / 5)
    for (scenario, strategy), frame in daily.groupby(["scenario", "strategy"], sort=False):
        p, u, d, e = (float(frame[k].sum()) for k in
                      ["planned_cost_yuan", "up_cost_yuan", "down_cost_yuan", "emergency_cost_yuan"])
        nominal_total = p + u + d + e
        configs = [("emergency_multiplier", value, 5.) for value in [4., 4.5, 5., 5.5, 6.]]
        if scenario in ["3", "4-3"]:
            configs += [("up_adjustment_multiplier", value, 1.5) for value in [1.2, 1.35, 1.5, 1.65, 1.8]]
            configs += [("down_adjustment_multiplier", value, .5) for value in [.4, .45, .5, .55, .6]]
        for parameter, value, nominal in configs:
            em = value if parameter == "emergency_multiplier" else 5.
            um = value if parameter == "up_adjustment_multiplier" else 1.5
            dm = value if parameter == "down_adjustment_multiplier" else .5
            total = p + u / 1.5 * um + d / .5 * dm + e / 5 * em
            row = {"scenario": scenario, "strategy": strategy,
                   "analysis_label": "fixed_action_billing_rescoring_not_reoptimization",
                   "parameter": parameter, "value": value, "nominal_value": nominal,
                   "parameter_change_pct": 100 * (value / nominal - 1),
                   "emergency_multiplier": em, "up_adjustment_multiplier": um, "down_adjustment_multiplier": dm,
                   "total_cost_yuan": total, "nominal_total_cost_yuan": nominal_total,
                   "cost_change_pct": 100 * (total / nominal_total - 1),
                   "cost_change_yuan": total - nominal_total,
                   "emergency_kwh": float(frame.emergency_kwh.sum()),
                   "throughput_kwh": float(frame.throughput_kwh.sum())}
            if scenario == "2" and strategy == "exp008_final":
                b = base_planned + base_emergency_at_unit * em
                row.update({"baseline_cost_yuan": b, "cost_saving_yuan": b - total,
                            "cost_reduction_pct": 100 * (b - total) / b})
            if value == nominal:
                assert abs(total - nominal_total) < 1e-6
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-replicates", type=int, default=N_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    assert args.bootstrap_replicates >= 2000
    OUT.mkdir(parents=True, exist_ok=True)
    selection = json.loads(SELECTION.read_text())
    dates, raw, sources = raw_data()
    source_paths = [SELECTION, BASELINE, Path(__file__),
                    *(ROOT / source["path"] for source in sources),
                    *(ROOT / spec["archive"] for spec in selection["scenarios"].values())]
    source_hashes_before = {str(p.relative_to(ROOT)): reference(p)["sha256"] for p in source_paths}
    frames, audits = [], []
    for scenario, spec in selection["scenarios"].items():
        assert spec["role"] == "accepted_final_strategy" and spec["eligible_as_causal_strategy"]
        frame, audit = extract_daily(scenario, "exp008_final", ROOT / spec["archive"], spec, dates, raw)
        frames.append(frame)
        audits.append(audit)
    base_spec = selection["scenarios"]["2"]
    frame, audit = extract_daily("2", "exp006_primary", BASELINE, base_spec, dates, raw)
    frames.append(frame)
    audits.append(audit)
    daily = pd.concat(frames, ignore_index=True)
    windows = window_analyses(daily)
    stress, membership = stress_analyses(daily)
    boots, draws, acf = bootstrap_analyses(daily, args.bootstrap_replicates, args.seed)
    billing = billing_sensitivity(daily)
    outputs = {"daily_metrics.csv": daily, "window_metrics.csv": windows,
               "stress_group_metrics.csv": stress, "stress_group_membership.csv": membership,
               "bootstrap_summary.csv": boots, "bootstrap_replicates.csv": draws,
               "paired_autocorrelation.csv": acf, "billing_sensitivity.csv": billing}
    for name, frame in outputs.items():
        write_csv(OUT / name, frame)
    # Verify exported partitions survive CSV serialization and remain additive.
    read_daily = pd.read_csv(OUT / "daily_metrics.csv", dtype={"scenario": str})
    read_windows = pd.read_csv(OUT / "window_metrics.csv", dtype={"scenario": str})
    for (scenario, strategy), part in read_daily.groupby(["scenario", "strategy"]):
        scored = read_windows[(read_windows.scenario == scenario) & (read_windows.strategy == strategy)]
        for field in ["total_cost_yuan", "emergency_kwh", "throughput_kwh", "nonidle_reversals"]:
            assert abs(part[field].sum() - scored[scored.window_type == "month"][field].sum()) < .001
        for fraction in [.2, .3, .4]:
            halves = scored[scored.window_type.isin(["chronological_head", "chronological_tail"])
                            & (scored.requested_tail_fraction == fraction)]
            assert int(halves.n_days.sum()) == 334
            assert abs(halves.total_cost_yuan.sum() - part.total_cost_yuan.sum()) < .001
    assert all(reference(ROOT / p)["sha256"] == digest for p, digest in source_hashes_before.items())
    q2 = windows[(windows.scenario == "2") & (windows.strategy == "exp008_final")]
    full = q2[q2.window_type == "full"].iloc[0]
    monthly = q2[q2.window_type == "month"]
    rolling = q2[q2.window_type == "rolling"]
    tail = q2[q2.window_type == "chronological_tail"]
    stress_q2 = stress[(stress.scenario == "2") & (stress.strategy == "exp008_final")]
    summary = {
        "analysis_label": METHOD_LABEL, "all_independent_checks_passed": all(a["passed"] for a in audits),
        "source_hashes_unchanged": True, "csv_readback_additive_checks_passed": True,
        "recomputed_annual_cost_yuan": {f'{r.scenario}/{r.strategy}': float(r.total_cost_yuan)
                                      for r in windows[windows.window_type == "full"].itertuples()},
        "q2_paired": {"baseline_cost_yuan": float(full.baseline_cost_yuan),
                      "final_cost_yuan": float(full.total_cost_yuan),
                      "cost_reduction_pct": float(full.cost_reduction_pct),
                      "saving_yuan": float(full.cost_saving_yuan),
                      "daily_win_fraction": float(full.paired_day_win_fraction),
                      "months_positive": int((monthly.cost_reduction_pct > 0).sum()),
                      "months_total": len(monthly),
                      "monthly_min_reduction_pct": float(monthly.cost_reduction_pct.min()),
                      "monthly_max_reduction_pct": float(monthly.cost_reduction_pct.max()),
                      "rolling": {str(length): {"n_windows": len(part),
                         "positive_fraction": float((part.cost_reduction_pct > 0).mean()),
                         "min_cost_reduction_pct": float(part.cost_reduction_pct.min()),
                         "max_cost_reduction_pct": float(part.cost_reduction_pct.max())}
                         for length, part in rolling.groupby("window_days")},
                      "chronological_tail_windows": tail[["window_label", "start", "end", "n_days", "cost_reduction_pct"]].to_dict("records"),
                      "bootstrap": boots.to_dict("records"),
                      "stress_groups_20pct": stress_q2[stress_q2.nominal_tail_fraction == .2][
                          ["stress_group", "n_days", "cost_reduction_pct", "mean_daily_emergency_kwh", "small_group_lt10_days"]].to_dict("records")},
        "independent_audits": audits,
    }
    write_json(OUT / "summary.json", summary)
    protocol = {
        "purpose": "Sensitivity and robustness evidence for the already accepted exp008 final selection; no reselection.",
        "script": reference(Path(__file__)), "final_selection": reference(SELECTION),
        "raw_sources": sources, "archive_sources": [a["source"] for a in audits],
        "definition_source": reference(ROOT / "C题/C题.md"),
        "sample": {"start": "2025-02-01", "end": "2025-12-31", "days": 334,
                   "intervals_per_day": 144, "source_calendar": "2025 local dates, no timezone conversion",
                   "january": "Warmup excluded; each final scenario inherits its own accepted initial battery state."},
        "analysis_labels": {"windows": METHOD_LABEL,
            "stress": "posthoc_observed_stress_group", "bootstrap": "conditional_descriptive_percentile_resampling_interval",
            "billing": "fixed_action_billing_rescoring_not_reoptimization"},
        "windows": {"calendar_months": 11, "quarters": "Q1 contains only February/March; Q2/Q3/Q4 complete",
                    "rolling_days": [30, 60, 90], "rolling_stride_days": 1,
                    "chronological_tail_fractions": [.2, .3, .4], "tail_count_rule": "ceil(334 * fraction)",
                    "state_rule": "Every subset retains recorded inherited SOC; no boundary reset or independent rerun.",
                    "operation_count_rule": "Date attribution on original continuous trace; incoming formal-boundary operation attributed to occurrence date. Warmup transition excluded. Not restarted per subset.",
                    "test_set_warning": "Head/tail windows are retrospective scoring subsets, neither retraining splits nor untouched holdouts."},
        "bootstrap": {"replicates": args.bootstrap_replicates, "seed": args.seed, "block_days": [7, 14, 28],
            "methods": ["paired_moving_block", "paired_quarter_stratified_moving_block"],
            "pairing": "Exactly the same resampled date indices index exp006 and exp008 daily costs.",
            "block_construction": "Uniform overlapping noncircular contiguous blocks; concatenate ceil(n/L) blocks and truncate to n days.",
            "quarter_stratification": "Independently sample blocks within each calendar quarter, preserving each quarter's original day count; Q1 has 59 days.",
            "statistic": "100 * (sum baseline cost - sum final cost) / sum baseline cost; ratio of sums, not mean daily percentages.",
            "interval": "2.5 and 97.5 percentiles; descriptive conditional resampling interval, not a formal post-selection inferential guarantee.",
            "serial_dependence": "Contiguous blocks preserve within-block serial dependence. Correlation beyond block length and across artificial joins is not preserved.",
            "seasonality": "Whole-period sampling assumes approximate block exchangeability despite observed seasonality; quarter stratification controls quarter composition only and does not establish stationarity.",
            "edge_effect": "Noncircular MBB samples end dates less frequently than interior dates; percentile centers may differ from point estimate.",
            "sign_fraction": "Fraction of bootstrap replicates with positive saving; not a p value or probability of future improvement."},
        "stress_groups": {"quantile_method": "pandas linear empirical quantile; >= upper and <= lower threshold; ties retained",
            "tail_fractions": [.1, .2, .3], "high_load": "daily actual load energy", "low_pv": "daily actual PV energy",
            "high_net_load": "daily actual load energy minus PV energy", "high_price": "actual daily mean settlement tariff; Q4 scenarios only",
            "fixed_tariff_scenarios": "Q2/Q3 share the same daily tariff profile; no distinct high-price-day group is defined.",
            "joint_group": "Actual daily load >= full-period q80 and PV <= q20; tiny group explicitly flagged.",
            "selection_rule": "All thresholds from full 334-day observed sample, solely for posthoc descriptive grouping; never fed to a forecasting or dispatch decision.",
            "small_group_rule": "n < 10 flagged; single-day composite extreme is descriptive and cannot establish robust generalization.",
            "noncontiguous_soc_warning": "First/last selected SOC fields do not form a connected trajectory for noncontiguous stress groups."},
        "accounting": {"formula": "sum p*g0 + alpha_up*p*max(g-g0,0) + alpha_down*p*max(g0-g,0) + alpha_emergency*p*e",
            "nominal": {"alpha_up": 1.5, "alpha_down": .5, "alpha_emergency": 5.},
            "down_adjustment": "Additional breach penalty; original scheduled quantity remains fully paid; no refund is subtracted.",
            "emergency_values": [4., 4.5, 5., 5.5, 6.], "up_values": [1.2, 1.35, 1.5, 1.65, 1.8],
            "down_values": [.4, .45, .5, .55, .6], "one_factor_at_a_time": True,
            "actions_fixed": True, "physical_metric_response": "No response by construction; accepted actions and SOC remain identical.",
            "purpose": "Accounting exposure only; altered prices are hypothetical and do not replace the original problem's tariffs."},
        "physical_units": {"raw_load_pv": "kW; divide by 6 for interval kWh", "charge_discharge": "AC kWh per 10 minutes",
            "soc": "kWh", "capacity_kwh": CAPACITY, "one_way_efficiency": ETA,
            "efficiency_convention": "accepted exp008 convention: round-trip 0.9, both one-way sqrt(0.9)",
            "efc": "sum(eta*charge + discharge/eta)/(2*12000)",
            "power_tv": "sum abs adjacent signed battery power difference; formal warmup boundary excluded"},
        "limitations": [
            "The 334 days have repeatedly informed development and model selection; they are not a new independent test set.",
            "All costs condition on one accepted fitted strategy per scenario; no refitting, policy adaptation or seed replication occurs here.",
            "These temporal checks and block intervals do not correct model-selection bias or prove future-year performance.",
            "Rolling windows overlap substantially and must not be counted as independent experiments.",
            "Only Q2 has the established same-physics, same-settlement exp006 primary comparator; Q3/Q4 absolute metrics are descriptive.",
            "Bootstrap operates on recorded daily costs, not a physically simulated battery trace across resampled block joins.",
            "Stress groups use actual outcomes post hoc; membership cannot be known ex ante and must not be described as forecast inputs.",
            "Changing billing multipliers with unchanged actions measures accounting sensitivity, not optimized policy response.",
            "Battery throughput and EFC are operational proxies, not calibrated lifetime or degradation estimates.",
        ],
        "output_tables": {name: {"rows": len(frame), "columns": list(frame.columns),
                                  **reference(OUT / name)} for name, frame in outputs.items()},
        "summary": reference(OUT / "summary.json"),
    }
    write_json(OUT / "protocol.json", protocol)
    print(json.dumps({"output": str(OUT.relative_to(ROOT)), "all_independent_checks_passed": True,
        "q2_cost_reduction_pct": summary["q2_paired"]["cost_reduction_pct"],
        "q2_months_positive": summary["q2_paired"]["months_positive"],
        "bootstrap_rows": len(draws), "tables": {k: len(v) for k, v in outputs.items()}}, indent=2))


if __name__ == "__main__":
    main()
