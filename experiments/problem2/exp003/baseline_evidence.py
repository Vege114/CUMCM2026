"""Archive Q2 midnight baseline evidence without importing training or replay code.

Run: .venv/bin/python -m experiments.problem2.exp003.baseline_evidence
All generated files remain in data/results/exp003/baseline/.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
EPOCH = pd.Timestamp("2025-01-01")
STEPS = 144
FORMAL_ORIGINS = np.arange(31, 365, dtype=np.int64) * STEPS
RAW_FILES = ("附件2_小区负载.csv", "附件2_光伏发电实际功率.csv")
NEURAL_BOUNDARY = (
    "Historical four-target checkpoint: load/PV parameters have separate branches, "
    "but a common validation loss chooses the saved epoch. Attachment 3 forecast "
    "and attachment 4 price targets can therefore indirectly influence Q2 predictions. "
    "Midnight filtering does not remove this coupling; do not claim strict Q2 input isolation."
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_actual(root):
    """Only attachment 2 is read to calculate forecast scores."""
    frames = [pd.read_csv(Path(root) / "data/raw" / name) for name in RAW_FILES]
    dates = pd.date_range("2025-01-01", "2025-12-31")
    for frame in frames:
        if frame.shape != (365, 145):
            raise ValueError("Attachment 2 must contain 365 dates and 144 intervals")
        if not pd.DatetimeIndex(pd.to_datetime(frame.iloc[:, 0])).equals(dates):
            raise ValueError("Attachment 2 dates are not the complete ordered 2025 calendar")
    actual = np.stack([frame.iloc[:, 1:].to_numpy(dtype=float).ravel() for frame in frames], -1)
    if not np.isfinite(actual).all() or (actual < 0).any():
        raise ValueError("Load and PV observations must be finite and nonnegative")
    return actual


def load_midnight(path, expected_origins=FORMAL_ORIGINS):
    """Select by recorded origin, never by positional assumptions about archive rows."""
    with np.load(path, allow_pickle=False) as archive:
        origins = archive["origins"]
        if origins.ndim != 1 or not np.issubdtype(origins.dtype, np.integer):
            raise ValueError("Archive origins must be a one-dimensional integer array")
        if len(np.unique(origins)) != len(origins):
            raise ValueError("Duplicate archive origins")
        lookup = {int(origin): i for i, origin in enumerate(origins)}
        if any(int(origin) % STEPS for origin in expected_origins):
            raise ValueError("Requested origins must be midnight")
        if any(int(origin) not in lookup for origin in expected_origins):
            raise ValueError("Archive is missing a formal midnight origin")
        indices = [lookup[int(origin)] for origin in expected_origins]
        predictions = {}
        for name in archive.files:
            if name == "origins":
                continue
            values = archive[name]
            if values.ndim != 3 or values.shape[:2] != (len(origins), STEPS) or values.shape[2] < 2:
                raise ValueError(f"Invalid prediction shape for {name}")
            # Convert before subtraction: net-load errors must not accumulate float32 rounding.
            predictions[name] = values[indices, :, :2].astype(np.float64)
            if not np.isfinite(predictions[name]).all() or (predictions[name] < 0).any():
                raise ValueError(f"Invalid load/PV predictions for {name}")
    return predictions


def legacy_selection(frame, arrays, months):
    selected = frame.selected.astype(str).str.lower().isin(("true", "1"))
    q2 = frame[(frame.scenario.astype(str) == "2") & selected]
    choices = {}
    for month in sorted(set(months)):
        rows = q2[q2.month == month]
        if len(rows) != 1 or rows.iloc[0].variant not in arrays:
            raise ValueError(f"Expected exactly one archived Q2 selection for month {month}")
        choices[int(month)] = str(rows.iloc[0].variant)
    return np.stack([arrays[choices[int(month)]][i] for i, month in enumerate(months)]), choices


def metric_row(predicted, actual):
    """WAPE uses sum(abs(actual)); bias is mean(predicted - actual)."""
    predicted, actual = np.asarray(predicted, float), np.asarray(actual, float)
    if predicted.shape != actual.shape or not np.isfinite(predicted).all() or not np.isfinite(actual).all():
        raise ValueError("Forecast and actual arrays must have equal shapes and finite values")
    error = predicted - actual
    n = error.size
    absolute = float(np.abs(error).sum())
    squared = float(np.square(error).sum())
    signed = float(error.sum())
    denominator = float(np.abs(actual).sum())
    return {"n": n, "absolute_error_sum": absolute, "squared_error_sum": squared,
            "signed_error_sum": signed, "actual_abs_sum": denominator,
            "mae": absolute / n if n else None,
            "rmse": float(np.sqrt(squared / n)) if n else None,
            "wape_pct": 100 * absolute / denominator if denominator else None,
            "bias": signed / n if n else None}


def score_forecast(values, truth, months):
    """Return monthly and directly pooled annual rows for the same midnight labels."""
    rows = []
    for month in [0, *sorted(set(months))]:
        selected = np.ones(len(months), bool) if month == 0 else months == month
        p, y = values[selected], truth[selected]
        generating = y[:, :, 1] > 0
        for target, prediction, actual in (
            ("load", p[:, :, 0], y[:, :, 0]),
            ("pv", p[:, :, 1], y[:, :, 1]),
            ("net_load", p[:, :, 0] - p[:, :, 1], y[:, :, 0] - y[:, :, 1]),
        ):
            for population, mask in (("all", np.ones(generating.shape, bool)),
                                     ("pv_generating", generating)):
                rows.append({"scenario": "2", "issue_hour": 0, "month": int(month),
                             "period": "annual" if month == 0 else "monthly", "target": target,
                             "population": population, "unit": "kW",
                             **metric_row(prediction[mask], actual[mask])})
    return rows


def boundary(predictor_id, experiment, archived=True, neural=False):
    return {"predictor_id": predictor_id, "source_experiment": experiment,
            "issue_hours_scored": [0], "targets_scored": ["load", "pv", "net_load"],
            "attachment_3_4_common_early_stopping_coupling": bool(neural),
            "strict_q2_input_isolation_verified": not archived,
            "historical_four_issue_scores_reused": False,
            "boundary_note": NEURAL_BOUNDARY if neural else (
                "Archived Q2 execution had no intraday purchase updates; its original full-run "
                "loader/signatures were not isolated to Q2 attachments. No strict-run isolation claim."
                if archived else "Reconstructed yesterday/weekly/periodic forecasts read only the two "
                "attachment 2 files; all lag indices precede midnight. No training or dispatch replay.")}


def archive_evidence(root=ROOT):
    root = Path(root).resolve()
    out = root / "data/results/exp003/baseline"
    sources = {}

    def source(relative, purpose):
        path = root / relative
        sources[relative] = {"sha256": sha256(path), "bytes": path.stat().st_size, "purpose": purpose}
        return path

    for name in RAW_FILES:
        source(f"data/raw/{name}", "Ground truth and causal historical lag forecasts")
    actual = read_actual(root)
    origins = FORMAL_ORIGINS.copy()
    ids = origins[:, None] + np.arange(STEPS)
    truth = actual[ids]
    months = np.asarray((EPOCH + pd.to_timedelta(origins * 10, unit="min")).month)
    forecasts, boundaries = {}, []
    old = load_midnight(source("data/results/exp001/ensemble_predictions.npz", "Saved three-seed ensemble forecasts"))
    selected = pd.read_csv(source("data/results/exp001/model_selection.csv", "Original monthly official Q2 selection"), dtype={"scenario": str})
    official, choices = legacy_selection(selected, old, months)
    for name, values in {**old, "selected": official}.items():
        key = f"exp001_{name}"
        forecasts[key] = values
        boundaries.append(boundary(key, "exp001", neural=True))
    current = load_midnight(source("data/results/exp002/predictions.npz", "Saved independent seed forecasts"))
    for name, values in current.items():
        key = f"exp002_{name}"
        forecasts[key] = values
        boundaries.append(boundary(key, "exp002", neural=True))
    yesterday, weekly = actual[ids - STEPS], actual[ids - 7 * STEPS]
    if not ((ids - STEPS) < origins[:, None]).all():
        raise ValueError("Historical baseline reads beyond the issue cutoff")
    periodic = weekly.copy()
    periodic[:, :, 1] = yesterday[:, :, 1]
    for name, values in (("yesterday", yesterday), ("weekly", weekly), ("periodic", periodic)):
        key = f"attachment2_{name}"
        forecasts[key] = values
        boundaries.append(boundary(key, "reconstructed_attachment2", archived=False))
    scores = [{"predictor_id": name, **row} for name, values in forecasts.items()
              for row in score_forecast(values, truth, months)]
    old_record = json.loads(source("reports/registry/exp001.json", "Immutable original protocol and registered Q2 cost").read_text())
    current_record = json.loads(source("reports/registry/exp002.json", "Immutable v2 evaluation protocol").read_text())
    costs = pd.read_csv(source("data/results/exp002/dispatch_metrics.csv", "Already replayed Q2 v2 costs"), dtype={"scenario": str})
    daily = pd.read_csv(source("data/results/exp002/daily_metrics.csv", "Already replayed Q2 daily states and fees"), dtype={"scenario": str})
    costs, daily = costs[costs.scenario == "2"].copy(), daily[daily.scenario == "2"].copy()
    if costs.empty or costs.duplicated(["name", "seed"]).any():
        raise ValueError("Missing or duplicate archived Q2 dispatch cases")
    for row in costs.itertuples():
        part = daily[(daily.name == row.name) & (daily.seed == row.seed)].sort_values("date")
        expected = pd.date_range("2025-02-01", "2025-12-31").strftime("%Y-%m-%d").tolist()
        if part.date.tolist() != expected or (part.updates != 0).any():
            raise ValueError("Q2 daily coverage or midnight-only execution mismatch")
        for name in ("total_cost", "planned_cost", "up_cost", "down_cost", "emergency_cost", "emergency_kwh"):
            np.testing.assert_allclose(part[name].sum(), getattr(row, name), rtol=1e-11, atol=1e-6)
        np.testing.assert_allclose(part.initial_soc.to_numpy()[1:], part.final_soc.to_numpy()[:-1], atol=1e-6)
        b = boundary(f"exp002_dispatch_{row.name}_seed_{row.seed}", "exp002", neural=row.kind in ("mlp", "legacy"))
        b.update(record_type="archived_dispatch", name=row.name, seed=int(row.seed),
                 cost_comparison="Conditional on preserving the v2 period, physics, billing, warmup and executor")
        boundaries.append(b)
    costs["source_experiment"] = "exp002"
    costs["comparison_status"] = "archived_v2_protocol_only"
    costs["strict_q2_input_isolation_verified"] = False
    original_costs = [{**row, "source_experiment": "exp001", "comparison_status": "not_directly_rankable_against_v2",
                       "strict_q2_input_isolation_verified": False}
                      for row in old_record["metrics"] if str(row["scenario"]) == "2"]
    boundaries.append({**boundary("exp001_registered_q2_cost", "exp001", neural=True),
                       "record_type": "original_registered_cost", "comparison_status": "not_directly_rankable_against_v2"})
    for relative in ("experiments/common/neural_v1/train.py", "experiments/common/neural_v2/train.py"):
        source(relative, "Evidence for common validation loss and saved-epoch coupling")
    source("experiments/problem2/exp003/baseline_evidence.py", "Evidence extraction implementation")
    manifest = {
        "schema_version": 1, "experiment_id": "exp003", "scope": ["2"],
        "purpose": "Frozen historical evidence; not a new training or dispatch experiment",
        "period": ["2025-02-01", "2025-12-31"], "issue_hours": [0],
        "days": len(origins), "intervals_per_day": STEPS, "intervals": int(truth.shape[0] * truth.shape[1]),
        "pv_generating_intervals": int((truth[:, :, 1] > 0).sum()),
        "forecast_targets": ["load", "pv", "net_load"], "array_channels": ["load", "pv"],
        "definitions": {"origin": "Ten-minute intervals since 2025-01-01 00:00; midnight only",
                        "month_0": "Direct annual aggregation, not mean of monthly metrics",
                        "net_load": "load - pv, without clipping; negative values retained",
                        "population_pv_generating": "Same actual PV > 0 mask for all three targets",
                        "bias": "mean(predicted - actual), kW",
                        "wape_pct": "100 * sum(abs(predicted - actual)) / sum(abs(actual)); null if denominator zero",
                        "cost_unit": "yuan", "energy_unit": "kWh"},
        "original_protocols": {"exp001": old_record["protocol"], "exp002": current_record["protocol"]},
        "legacy_q2_monthly_selection": choices, "sources": sources,
        "boundary_warning": NEURAL_BOUNDARY,
        "comparability": {"forecast": "All arrays rescored over the same midnight observations; historical boundary differences remain labelled",
                          "exp001_original_cost": "Not directly rankable: different efficiency, warmup and executor protocol",
                          "exp002_cost": "Archive only; re-evaluate under a new protocol if physical/billing/warmup/executor assumptions change",
                          "training_time": "Historical shared four-target training time is not Q2-specific runtime",
                          "generalization": "Previously inspected 2025 results are retrospective evidence, not a fresh untouched test set"},
        "training_calls": 0, "dispatch_replays": 0,
    }
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "midnight_predictions.npz", origins=origins, actual=truth, **forecasts)
    frame = pd.DataFrame(scores)
    frame[frame.period == "monthly"].to_csv(out / "forecast_monthly.csv", index=False)
    frame[frame.period == "annual"].to_csv(out / "forecast_annual.csv", index=False)
    costs.to_csv(out / "exp002_q2_costs.csv", index=False)
    daily.to_csv(out / "exp002_q2_daily.csv", index=False)
    pd.DataFrame(original_costs).to_csv(out / "exp001_original_q2_costs.csv", index=False)
    for name, value in (("manifest", manifest), ("baseline_boundaries", boundaries)):
        (out / f"{name}.json").write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    (out / "README.md").write_text(
        "# Q2 历史基线独立证据\n\n"
        "仅归档现有预测与费用，不训练、不回放。复现：\n\n"
        "`.venv/bin/python -m experiments.problem2.exp003.baseline_evidence`\n\n"
        "预测统一使用每日0时的144区间，2—12月共48,096区间。"
        "负载、光伏、净负载均保留全天与实际光伏>0两种人群；净负载不裁剪。"
        "MAE/RMSE/bias单位kW，WAPE单位%；bias=预测−实际。月度值通过误差和汇总，全年直接评分。\n\n"
        "旧v1/v2网络用四目标共同验证损失选择保存轮次，附件3/4可能间接影响Q2；"
        "午夜筛选不会消除这一耦合，不能宣称历史网络严格输入隔离。"
        "旧v2周期等费用也仅保留原执行证据，不追认原运行输入隔离。"
        "attachment2_*预测由本脚本仅用附件2历史重建，与原费用分列。\n\n"
        "exp001_original_q2_costs.csv为旧协议原登记值，不参与v2费用排名；"
        "exp002_q2_costs.csv中的legacy_rebased才是v2物理/预热/执行口径下的历史重算。"
        "若exp003改变这些口径，所有历史费用都需在新协议重算。\n\n"
        "manifest.json保留两代原协议、原月度选择、所有读取源的SHA256和口径；"
        "baseline_boundaries.json为每个预测器及费用方案保存边界标识。"
        "大数组只保留负载/PV；净负载由两列相减。CSV中的空指标表示零分母或空人群。\n"
    )
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    output = archive_evidence()
    print(f"Archived Q2 midnight evidence: {output}")


if __name__ == "__main__":
    main()
