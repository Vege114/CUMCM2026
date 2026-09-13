"""One fixed monthly HGB net-demand residual model, with issue-time features.

The existing CNN stays a baseline and input. This is a bounded development
experiment, not an independent test or a replacement of historical archives.
The returned two-column values are a NET DEMAND ADAPTER, not separate improved
load/PV point forecasts: the learned net correction is assigned to load.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from experiments.problem2.exp003.data import Data, ROOT, EPOCH
from experiments.problem2.exp004.predict import ForecastStore

OUT = ROOT / "data/results/exp008/forecast_net_hgb"
MODEL_CONFIG = dict(loss="squared_error", max_leaf_nodes=15, max_iter=100,
                    min_samples_leaf=50, l2_regularization=2., learning_rate=.08,
                    early_stopping=False, random_state=42)
HISTORY_HALF_LIFE_DAYS = 90


def _before(data, first, last, issue):
    if not 0 <= first <= last <= issue:
        raise ValueError("feature actuals must be strictly before their historical issue")
    return np.asarray(data.actual[first:last, :2], dtype=float)


def _triple(power):
    return np.column_stack((power, power[:, 0] - power[:, 1]))


def features_for_day(data, day, cnn_or_periodic):
    """All historical feature windows are anchored at this day's own issue."""
    if not 7 <= day < 365:
        raise ValueError("seven complete days required")
    issue = int(day) * 144
    history = _before(data, issue - 7 * 144, issue, issue).reshape(7, 144, 2)
    blocks = [_triple(cnn_or_periodic), _triple(history[-1]), _triple(history[0]),
              _triple(history[-3:].mean(0)), _triple(history.mean(0)),
              _triple(history.std(0))]
    names = [f"{group}_{channel}" for group in
             ("base", "yesterday", "previous_week", "mean3", "mean7", "std7")
             for channel in ("load", "pv", "net")]
    # std7_net above is a difference of component standard deviations, rather
    # than std(net); replace it with the actual historical net standard deviation.
    blocks[-1][:, 2] = (history[:, :, 0] - history[:, :, 1]).std(0)
    phase = 2 * np.pi * np.arange(144) / 144
    weekday = (day + 2) % 7  # 2025-01-01 was Wednesday, Monday=0.
    calendar = np.column_stack((np.sin(phase), np.cos(phase), np.sin(2 * phase),
        np.cos(2 * phase), np.full(144, np.sin(2 * np.pi * weekday / 7)),
        np.full(144, np.cos(2 * np.pi * weekday / 7))))
    names += ["slot_sin", "slot_cos", "slot_sin2", "slot_cos2", "weekday_sin", "weekday_cos"]
    yesterday = history[-1]
    statistics = np.r_[yesterday.mean(0), yesterday.std(0), yesterday.min(0), yesterday.max(0),
                       (yesterday[:, 0] - yesterday[:, 1]).mean(),
                       (yesterday[:, 0] - yesterday[:, 1]).std()]
    names += [f"yesterday_{stat}_{channel}" for stat in ("mean", "std", "min", "max")
              for channel in ("load", "pv")]
    names += ["yesterday_mean_net", "yesterday_std_net", "base_is_cnn"]
    features = np.column_stack((*blocks, calendar,
        np.broadcast_to(statistics, (144, len(statistics))), np.full(144, day >= 31)))
    return features, names


def bases_and_features(data, store):
    days = np.arange(7, 365)
    bases, arrays, feature_names = [], [], None
    for day in days:
        base = store.get(int(day) * 144) if day >= 31 else data.baseline(int(day) * 144)
        features, feature_names = features_for_day(data, int(day), base)
        bases.append(base)
        arrays.append(features)
    return days, np.stack(bases), np.stack(arrays), feature_names


def fit_month(data, month, days, bases, features, seed=42):
    asof = int((pd.Timestamp(2025, month, 1) - EPOCH).days)
    ids = np.flatnonzero(days < asof)
    train_days = days[ids]
    truth = np.stack([_before(data, int(day) * 144, (int(day) + 1) * 144, asof * 144)
                      for day in train_days])
    target = ((truth[:, :, 0] - truth[:, :, 1]) - (bases[ids, :, 0] - bases[ids, :, 1])).ravel()
    weights = np.repeat(2 ** (-(asof - 1 - train_days) / HISTORY_HALF_LIFE_DAYS), 144)
    weights /= weights.mean()
    config = {**MODEL_CONFIG, "random_state": int(seed)}
    model = HistGradientBoostingRegressor(**config)
    started = time.perf_counter()
    with threadpool_limits(limits=1):
        model.fit(features[ids].reshape(-1, features.shape[-1]), target, sample_weight=weights)
    audit = {"month": month, "asof_day": asof, "information_cutoff_exclusive": asof * 144,
        "training_days": train_days.astype(int).tolist(), "training_rows": len(target),
        "training_last_label": asof * 144 - 1,
        "january_periodic_rows": int(np.sum(train_days < 31) * 144),
        "historical_frozen_cnn_rows": int(np.sum(train_days >= 31) * 144),
        "iterations": int(model.n_iter_), "training_seconds": time.perf_counter() - started,
        "config": config, "early_stopping": False}
    return model, audit


def score(net, truth):
    e = net - truth
    return {"target": "net_load", "unit": "kW", "n": e.size,
        "mae": float(np.abs(e).mean()), "rmse": float(np.sqrt(np.square(e).mean())),
        "bias_predicted_minus_actual": float(e.mean()),
        "absolute_error_sum": float(np.abs(e).sum()), "squared_error_sum": float(np.square(e).sum())}


def run(seed=42, out=OUT):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    models_dir = out / "models"
    models_dir.mkdir(exist_ok=True)
    began = time.perf_counter()
    data, cnn = Data(), ForecastStore("no_season", seed=seed)
    protocol = {"model": {**MODEL_CONFIG, "random_state": seed}, "cnn_retrained": False,
        "one_fixed_candidate": True, "no_annual_actual_feature": True,
        "training": "monthly_refit_all_prior_completed_days; historical features obey historical issue",
        "history_weight_half_life_days": HISTORY_HALF_LIFE_DAYS,
        "january": "causal_weekly_load_yesterday_PV_cold_start_days7_to30",
        "official_pv_allowed": False, "future_realized_price_allowed": False,
        "development_year": "2025_already_examined_not_independent_test",
        "continuation_gate": "only continue mechanism if net RMSE meaningfully improves on ridge about388kW"}
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    days, bases, features, names = bases_and_features(data, cnn)
    net_delta = np.empty((334, 144))
    training, audits = [], []
    for month in range(2, 13):
        model, audit = fit_month(data, month, days, bases, features, seed)
        model_path = models_dir / f"m{month:02d}_seed{seed}.joblib"
        joblib.dump(model, model_path)
        audit["model_sha256"] = hashlib.sha256(model_path.read_bytes()).hexdigest()
        training.append(audit)
        begin = audit["asof_day"]
        stop = int((pd.Timestamp(2025, month, 1) + pd.offsets.MonthBegin(1) - EPOCH).days)
        test_days = np.arange(begin, stop)
        with threadpool_limits(limits=1):
            prediction = model.predict(features[test_days - 7].reshape(-1, len(names))).reshape(-1, 144)
        net_delta[test_days - 31] = prediction
        for day in test_days:
            audits.append({"day": int(day), "origin": int(day * 144), "month_model": month,
                "model_information_cutoff_exclusive": begin * 144,
                "model_training_last_label": begin * 144 - 1,
                "feature_last_actual_index": int(day * 144 - 1),
                "base_source": "frozen_exp004_no_season", "current_truth_used": False})
        print(json.dumps({"month": month, "training_seconds": audit["training_seconds"],
                          "training_rows": audit["training_rows"]}), flush=True)
    base = cnn.values.copy()
    net = base[:, :, 0] - base[:, :, 1] + net_delta
    # Preserve the predicted NET exactly while keeping adapter channels nonnegative.
    adapter = base.copy()
    adapter[:, :, 0] += net_delta
    below_zero = np.minimum(adapter[:, :, 0], 0)
    adapter[:, :, 0] -= below_zero
    adapter[:, :, 1] -= below_zero
    truth_power = data.actual[cnn.origins[:, None] + np.arange(144)]
    truth_net = truth_power[:, :, 0] - truth_power[:, :, 1]
    errors = truth_power - adapter
    path = out / "predictions.npz"
    np.savez_compressed(path, origins=cnn.origins, values=adapter, base_values=base,
        net_kw=net, net_delta_kw=net_delta, delta=adapter - base, errors_kw=errors,
        net_errors_kw=truth_net - net)
    dates = pd.date_range("2025-02-01", "2025-12-31")
    annual = [{"name": "cnn_base", **score(base[:, :, 0] - base[:, :, 1], truth_net)},
              {"name": "cnn_net_hgb", **score(net, truth_net)}]
    pd.DataFrame(annual).to_csv(out / "annual_metrics.csv", index=False)
    pd.DataFrame({"date": str(date.date()), **score(net[i:i+1], truth_net[i:i+1])}
                 for i, date in enumerate(dates)).to_csv(out / "daily_metrics.csv", index=False)
    pd.DataFrame({"month": month, **score(net[dates.month == month], truth_net[dates.month == month])}
                 for month in range(2, 13)).to_csv(out / "monthly_metrics.csv", index=False)
    (out / "training_audit.json").write_text(json.dumps(training, indent=2) + "\n")
    (out / "prediction_audit.json").write_text(json.dumps(audits, indent=2) + "\n")
    metadata = {"complete": True, "period": ["2025-02-01", "2025-12-31"], "seed": seed,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "base_sha256": hashlib.sha256(cnn.values.tobytes()).hexdigest(),
        "archive_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "source_data_sha256": data.hashes,
        "features": names, "feature_count": len(names), "training_models": 11,
        "seconds": time.perf_counter() - began, "metrics": annual,
        "semantics": "values are nonnegative net-demand adapter channels; do not score load/PV separately as improved component forecasts",
        "net_rmse_below_ridge388": annual[-1]["rmse"] < 388,
        "development_exploration_not_independent_test": True}
    (out / "provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, ensure_ascii=False), flush=True)
    return metadata


class NetHGBStore:
    """Signed archive adapter compatible with ForecastStore.get(origin)."""
    def __init__(self, seed=42, directory=OUT):
        directory = Path(directory)
        meta = json.loads((directory / "provenance.json").read_text())
        path = directory / "predictions.npz"
        if (not meta["complete"] or meta["seed"] != seed
                or meta["archive_sha256"] != hashlib.sha256(path.read_bytes()).hexdigest()):
            raise RuntimeError("HGB forecast archive is incomplete, changed or a different seed")
        with np.load(path) as z:
            for name in ("values", "origins", "base_values", "net_kw", "net_delta_kw", "errors_kw"):
                setattr(self, name, z[name].copy())
        self.lookup = {int(origin): i for i, origin in enumerate(self.origins)}
        self.seed = seed

    def get(self, origin):
        if origin not in self.lookup:
            raise ValueError("formal midnight origin required")
        return self.values[self.lookup[int(origin)]].copy()

    def completed_error_paths(self, origin, limit=28):
        self.get(origin)
        ids = np.flatnonzero(self.origins + 144 <= origin)[-limit:]
        return {"origins": self.origins[ids].copy(), "errors_kw": self.errors_kw[ids].copy(),
            "information_cutoff": int(origin), "variant": "cnn_net_hgb",
            "net_adapter_not_component_forecasts": True, "development_on_evaluation_year": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    run(args.seed, args.out)
