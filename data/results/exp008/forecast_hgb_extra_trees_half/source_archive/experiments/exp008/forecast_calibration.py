"""Causal output calibration of the unchanged exp004 midnight CNN.

These candidates are development experiments on the already examined 2025
year, not an independent test. Each day's adjustment is fitted exclusively
to completed earlier days and never selected using the current day's truth.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.problem2.exp003.data import Data, ROOT
from experiments.problem2.exp004.predict import ForecastStore

OUT = ROOT / "data/results/exp008/forecast_calibration"
CANDIDATES = (
    {"name": "base", "method": "none", "window": 0},
    *({"name": f"global_bias_{w}", "method": "global_bias", "window": w} for w in (7, 14, 28, 56)),
    *({"name": f"hour_bias_{w}", "method": "hour_bias", "window": w} for w in (14, 28)),
    *({"name": f"ridge_{w}", "method": "ridge", "window": w} for w in (28, 56)),
    {"name": "online_select_28", "method": "online_select", "window": 28},
)
CONFIG = {c["name"]: dict(c) for c in CANDIDATES}


def _history(actual, start, stop, cutoff):
    if not 0 <= start <= stop <= cutoff:
        raise ValueError("calibration cannot access an unobserved actual")
    return np.asarray(actual[start:stop, :2], dtype=float)


def _daylight(actual, day):
    origin = day * 144
    past = _history(actual, max(0, origin - 28 * 144), origin, origin)
    daylight = (past[:, 1].reshape(-1, 144) > 0).any(axis=0)
    return np.convolve(daylight.astype(int), np.ones(5), mode="same") > 0


def _features(actual, day, base, channel):
    """Features for a historical issue use only information at that issue."""
    origin = day * 144
    yesterday = _history(actual, origin - 144, origin, origin)[:, channel]
    weekly = _history(actual, origin - 7 * 144, origin - 6 * 144, origin)[:, channel]
    last3 = _history(actual, origin - 3 * 144, origin, origin).reshape(3, 144, 2)[:, :, channel].mean(0)
    phase = 2 * np.pi * np.arange(144) / 144
    predicted = base[:, channel]
    return np.column_stack((np.ones(144), predicted / 1000,
        (yesterday - predicted) / 1000, (weekly - predicted) / 1000,
        (last3 - predicted) / 1000, base[:, 1 - channel] / 1000,
        np.sin(phase), np.cos(phase), np.sin(2 * phase), np.cos(2 * phase)))


def _calibrate(actual, origins, base, index, config):
    day = int(origins[index] // 144)
    cutoff = day * 144
    method, window = config["method"], config["window"]
    history = np.arange(max(0, index - window), index, dtype=int)
    info = {"day": day, "origin": cutoff, "method": method,
        "window_days": window, "history_count": len(history),
        "history_origins": origins[history].astype(int).tolist(),
        "max_actual_index": cutoff - 1,
        "history_last_label": int(origins[history[-1]] + 143) if len(history) else None,
        "current_truth_used": False}
    if method == "none" or not len(history):
        info["cold_start"] = not len(history)
        return base[index].copy(), info
    observed = np.stack([_history(actual, int(origins[h]), int(origins[h]) + 144, cutoff)
                         for h in history])
    residual = observed - base[history]
    # Half-life is fixed at half of the chosen history window.
    weights = 2 ** (-np.arange(len(history) - 1, -1, -1) / max(1, window / 2))
    delta = np.zeros((144, 2))
    if method in ("global_bias", "hour_bias"):
        for channel in range(2):
            valid = np.ones_like(residual[:, :, channel], dtype=bool)
            if channel == 1:
                # Nighttime zeros must not dilute a daylight-only PV correction.
                valid = (base[history, :, 1] > 1) | (observed[:, :, 1] > 1)
            numerator = np.sum(weights[:, None] * residual[:, :, channel] * valid)
            denominator = np.sum(weights[:, None] * valid)
            global_bias = float(numerator / max(denominator, 1))
            if method == "global_bias":
                delta[:, channel] = global_bias
            else:
                for hour in range(24):
                    sl = slice(hour * 6, (hour + 1) * 6)
                    w = weights[:, None] * valid[:, sl]
                    # Equivalent to three days of shrinkage toward global bias.
                    prior_weight = 3 * 6 * weights.mean()
                    estimate = (np.sum(w * residual[:, sl, channel]) + prior_weight * global_bias)
                    delta[sl, channel] = estimate / (w.sum() + prior_weight)
        delta *= len(history) / (len(history) + 3)
    elif method == "ridge":
        coefficients = []
        for channel in range(2):
            x = np.concatenate([_features(actual, int(origins[h] // 144), base[h], channel)
                                for h in history])
            target = residual[:, :, channel].ravel() / 1000
            w = np.repeat(weights, 144)
            if channel == 1:
                active = ((base[history, :, 1] > 1) | (observed[:, :, 1] > 1)).ravel()
                w *= np.where(active, 1., .1)
            penalty = np.full(x.shape[1], 20.)
            penalty[0] = 2.
            gram = np.einsum("ni,nj,n->ij", x, x, w, optimize=False)
            rhs = np.einsum("ni,n,n->i", x, w, target, optimize=False)
            coefficient = np.linalg.solve(gram + np.diag(penalty), rhs)
            xt = _features(actual, day, base[index], channel)
            correction = 1000 * np.einsum("ni,i->n", xt, coefficient, optimize=False)
            bound = max(100., 3 * float(np.sqrt(np.mean(residual[:, :, channel] ** 2))))
            delta[:, channel] = np.clip(correction, -bound, bound)
            coefficients.append(coefficient.tolist())
        info["coefficients"] = coefficients
    else:
        raise ValueError(f"unknown calibration method: {method}")
    output = np.maximum(0., base[index] + delta)
    output[:, 1] *= _daylight(actual, day)
    info.update(cold_start=False, mean_output_delta_kw=(output - base[index]).mean(0).tolist())
    return output, info


class CalibratedStore:
    """Compatible with exp004 ForecastStore: ``get(origin)``, values, origins.

    Passing a data object always recomputes rather than reading saved output,
    permitting independent future-perturbation checks. Saved caches are signed.
    """
    def __init__(self, name="hour_bias_28", seed=42, data=None, directory=OUT,
                 use_cache=True, _base_store=None, _candidate_values=None):
        if name not in CONFIG:
            raise ValueError(f"unknown candidate {name}; choose {tuple(CONFIG)}")
        self.name, self.seed = name, int(seed)
        self.base_store = _base_store or ForecastStore("no_season", seed=self.seed)
        self.origins = self.base_store.origins.copy()
        self.base_values = self.base_store.values.copy()
        self.lookup = {int(origin): i for i, origin in enumerate(self.origins)}
        self.config = CONFIG[name]
        cache = Path(directory) / f"{name}_seed_{self.seed}.npz"
        meta_path = cache.with_suffix(".json")
        if data is None and use_cache and cache.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if meta["source_sha256"] == source_hash() and meta["base_sha256"] == base_hash(self.base_values):
                if meta["archive_sha256"] != hashlib.sha256(cache.read_bytes()).hexdigest():
                    raise RuntimeError("calibration archive hash mismatch")
                with np.load(cache) as pack:
                    self.values = pack["values"].copy()
                    self.delta = pack["delta"].copy()
                    self.errors_kw = pack["errors_kw"].copy()
                self.audit = meta["days"]
                return
        actual = (Data() if data is None else data).actual
        if self.config["method"] == "online_select":
            candidates = _candidate_values
            if candidates is None:
                candidates = {c["name"]: CalibratedStore(c["name"], self.seed, data,
                    directory, use_cache, _base_store=self.base_store).values
                    for c in CANDIDATES if c["method"] != "online_select"}
            self.values, self.audit = _online_select(actual, self.origins, candidates, self.config)
        else:
            outputs = [_calibrate(actual, self.origins, self.base_values, index, self.config)
                       for index in range(len(self.origins))]
            self.values = np.stack([r[0] for r in outputs])
            self.audit = [r[1] for r in outputs]
        self.delta = self.values - self.base_values
        # These arrays are post-hoc scoring labels, never calibration inputs.
        truth = actual[self.origins[:, None] + np.arange(144)][..., :2]
        self.errors_kw = truth - self.values

    def get(self, origin):
        if origin not in self.lookup:
            raise ValueError("requires a formal February–December midnight")
        return self.values[self.lookup[int(origin)]].copy()

    def completed_error_paths(self, origin, limit=28):
        self.get(origin)
        ids = np.flatnonzero(self.origins + 144 <= origin)[-limit:]
        return {"origins": self.origins[ids].copy(), "errors_kw": self.errors_kw[ids].copy(),
            "information_cutoff": int(origin), "variant": self.name,
            "exploratory": False, "development_on_evaluation_year": True}


def _online_select(actual, origins, candidates, config):
    """Select today's calibrator by prior complete-day prequential net MSE."""
    names = list(candidates)
    values, audits = [], []
    for i, origin in enumerate(origins):
        indices = np.arange(max(0, i - config["window"]), i)
        scores = {}
        if len(indices) >= 3:
            truth = np.stack([_history(actual, int(origins[h]), int(origins[h]) + 144, int(origin))
                              for h in indices])
            net_truth = truth[:, :, 0] - truth[:, :, 1]
            for name, array in candidates.items():
                net = array[indices, :, 0] - array[indices, :, 1]
                scores[name] = float(np.mean((net - net_truth) ** 2))
            selected = min(names, key=lambda name: scores[name])
        else:
            selected = "base"
        values.append(candidates[selected][i])
        audits.append({"day": int(origin // 144), "origin": int(origin),
            "selected": selected, "historical_net_mse": scores,
            "selection_label_origins": origins[indices].astype(int).tolist(),
            "current_truth_used": False,
            "history_last_label": int(origins[indices[-1]] + 143) if len(indices) else None})
    return np.stack(values), audits


def source_hash():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def base_hash(values):
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def metrics(prediction, truth):
    output = []
    for target in ("load", "pv", "net_load"):
        p = prediction[:, :, 0] - prediction[:, :, 1] if target == "net_load" else prediction[:, :, int(target == "pv")]
        y = truth[:, :, 0] - truth[:, :, 1] if target == "net_load" else truth[:, :, int(target == "pv")]
        e = p - y
        output.append({"target": target, "n": e.size, "unit": "kW",
            "mae": float(np.abs(e).mean()), "rmse": float(np.sqrt(np.square(e).mean())),
            "bias": float(e.mean()), "wape_pct": float(100 * np.abs(e).sum() / np.abs(y).sum()),
            "absolute_error_sum": float(np.abs(e).sum()), "squared_error_sum": float(np.square(e).sum())})
    return output


def run(seed=42, out=OUT):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    began = time.perf_counter()
    data, base_store = Data(), ForecastStore("no_season", seed=seed)
    # This list is written before evaluating any candidate.
    (out / "candidate_protocol.json").write_text(json.dumps({"seed": seed,
        "candidates": CANDIDATES, "forecast_structure": "unchanged_frozen_no_season_CNN",
        "evaluation_role": "2025_development_exploration_not_independent_test",
        "selection_boundary": "all_fit_or_online_selection_labels_are_prior_complete_days",
        "source_sha256": source_hash()}, indent=2) + "\n")
    all_rows, daily_rows, monthly_rows, candidate_values = [], [], [], {}
    truth = data.actual[base_store.origins[:, None] + np.arange(144)][..., :2]
    dates = pd.date_range("2025-02-01", "2025-12-31")
    for config in CANDIDATES:
        t = time.perf_counter()
        name = config["name"]
        store = CalibratedStore(name, seed, data, out, False, base_store, candidate_values)
        seconds = time.perf_counter() - t
        candidate_values[name] = store.values
        path = out / f"{name}_seed_{seed}.npz"
        np.savez_compressed(path, origins=store.origins, values=store.values,
            delta=store.delta, base_values=store.base_values, errors_kw=store.errors_kw)
        path.with_suffix(".json").write_text(json.dumps({"config": config, "seed": seed,
            "source_sha256": source_hash(), "base_sha256": base_hash(store.base_values),
            "archive_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "source_data_sha256": data.hashes, "days": store.audit,
            "delta_definition": "final_output_minus_frozen_CNN_including_nonnegative_and_daylight_projection",
            "error_definition": "actual_minus_calibrated_prediction; labels_for_scoring_after_release",
            "generation_seconds": seconds,
            "evaluation_role": "development_year_exploration_not_independent_test"}, indent=2) + "\n")
        for row in metrics(store.values, truth):
            all_rows.append({"name": name, "seconds": seconds, **row})
        for i, date in enumerate(dates):
            for row in metrics(store.values[i:i+1], truth[i:i+1]):
                daily_rows.append({"name": name, "date": str(date.date()), **row})
        for month in range(2, 13):
            ids = dates.month == month
            for row in metrics(store.values[ids], truth[ids]):
                monthly_rows.append({"name": name, "month": month, **row})
        print(json.dumps({"name": name, "seconds": seconds, "net": all_rows[-1]}, ensure_ascii=False), flush=True)
    pd.DataFrame(all_rows).to_csv(out / "annual_metrics.csv", index=False)
    pd.DataFrame(daily_rows).to_csv(out / "daily_metrics.csv", index=False)
    pd.DataFrame(monthly_rows).to_csv(out / "monthly_metrics.csv", index=False)
    summary = {"complete": True, "seconds": time.perf_counter() - began,
        "candidate_count": len(CANDIDATES), "period": ["2025-02-01", "2025-12-31"],
        "source_sha256": source_hash(), "role": "development_exploration",
        "independent_test": False, "cnn_retrained": False}
    (out / "completion.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    run(args.seed, args.out)
